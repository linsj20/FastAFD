#!/usr/bin/env python3
"""Focused CUDA numeric smoke for fused Qwen3.5 GDN decode preparation."""

from __future__ import annotations

import argparse

import torch

from minisgl.kernel.fp8_quant import per_token_cast_to_fp8, per_token_cast_to_fp8_torch
from minisgl.kernel.qwen35_gdn import (
    qwen35_causal_conv1d_decode,
    qwen35_causal_conv1d_decode_packed,
    qwen35_gdn_decode,
    qwen35_gdn_gates,
    qwen35_rmsnorm_gated,
    qwen35_rmsnorm_gated_fp8,
)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _bench_ms(function, iterations: int, peer_stream=None) -> float:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(10):
            function()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup_stream):
        if peer_stream is not None:
            peer_stream.wait_stream(warmup_stream)
        captured_output = function()
        if peer_stream is not None:
            warmup_stream.wait_stream(peer_stream)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _check_projection_streams(config_path: str, rows: int, iterations: int) -> None:
    """Compare complete GDN decode with owned versus caller-supplied side streams."""
    import json
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    from minisgl.core import get_global_ctx
    from minisgl.distributed.info import set_tp_info, try_get_tp_info
    from minisgl.kvcache import create_hybrid_gdn_state_pool
    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen35_moe import GdnProjectionSync, Qwen3_5LinearAttention
    from minisgl.utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    config = ModelConfig.from_hf(json.loads(Path(config_path).read_text()))
    config = replace(config, num_layers=1, layer_types=("linear_attention",))
    device = torch.device("cuda")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Qwen3_5LinearAttention(config, 0)
    weights = {}
    for key, value in layer.state_dict().items():
        if key.endswith("weight_scale"):
            tensor = torch.full(value.shape, 0.01, device=device)
        elif key == "norm.weight":
            tensor = torch.ones(value.shape, device=device)
        elif key in ("A_log", "dt_bias"):
            tensor = torch.zeros(value.shape, device=device)
        else:
            tensor = torch.randn(value.shape, device=device) * 0.02
        weights[key] = tensor.to(value.dtype)
    layer.load_state_dict(weights)
    ctx = get_global_ctx()
    pool = ctx.gdn_state_pool = create_hybrid_gdn_state_pool(
        config, max_slots=rows, dtype=torch.bfloat16, device=device
    )
    pool.conv_state.uniform_(-0.01, 0.01)
    pool.recurrent_state.uniform_(-0.01, 0.01)
    initial_conv, initial_state = pool.conv_state.clone(), pool.recurrent_state.clone()
    hidden = torch.randn(rows, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    batch = SimpleNamespace(
        is_decode=True, state_indices=torch.arange(rows, device=device, dtype=torch.int64),
        positions=torch.ones(rows, device=device, dtype=torch.int32),
    )
    peer = torch.cuda.Stream()
    timeout = torch.zeros(2, dtype=torch.int64, device=device)
    sync = GdnProjectionSync(device, timeout, 30000, 256)
    with ctx.forward_batch(batch):
        expected = layer.forward(hidden)
        expected_conv, expected_state = pool.conv_state.clone(), pool.recurrent_state.clone()
        pool.conv_state.copy_(initial_conv)
        pool.recurrent_state.copy_(initial_state)
        actual = layer.forward(hidden, projection_stream=peer)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(pool.conv_state, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(pool.recurrent_state, expected_state, rtol=0, atol=0)
        pool.conv_state.copy_(initial_conv)
        pool.recurrent_state.copy_(initial_state)
        actual = layer.forward(hidden, projection_stream=peer, projection_sync=sync)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(pool.conv_state, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(pool.recurrent_state, expected_state, rtol=0, atol=0)

        # Exercise both ownership directions and signal reuse across local
        # layers and graph replays. Compare every output plus both mutable states.
        from minisgl.cuda_graph_utils import capture_cuda_graph

        owner = torch.cuda.Stream()
        peer_sync = GdnProjectionSync(device, timeout, 30000, 256)
        handoff = GdnProjectionSync(device, timeout, 30000, 256)
        expected_outputs = []
        pool.conv_state.copy_(initial_conv)
        pool.recurrent_state.copy_(initial_state)
        for _ in range(6):
            expected_outputs.append(layer.forward(hidden).clone())
        expected_conv, expected_state = pool.conv_state.clone(), pool.recurrent_state.clone()
        captured_outputs = []

        def body():
            pool.conv_state.copy_(initial_conv)
            pool.recurrent_state.copy_(initial_state)
            for _ in range(3):
                captured_outputs.append(layer.forward(
                    hidden, projection_stream=peer, projection_sync=sync
                ))
            handoff.release(handoff.ready)
            with torch.cuda.stream(peer):
                handoff.wait(handoff.ready)
                for _ in range(3):
                    captured_outputs.append(layer.forward(
                        hidden, projection_stream=owner, projection_sync=peer_sync
                    ))

        # Compile/warm each ownership direction before capture.
        owner.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(owner):
            body()
        torch.cuda.synchronize()
        captured_outputs.clear()
        graph, _ = capture_cuda_graph(
            device=device, engine_stream=owner, comm_stream=peer,
            overlap_comm=True, pool=None, fn=body,
        )
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            for actual, expected in zip(captured_outputs, expected_outputs, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(pool.conv_state, expected_conv, rtol=0, atol=0)
            torch.testing.assert_close(pool.recurrent_state, expected_state, rtol=0, atol=0)
        assert not timeout.any().item()
        for signals in (sync, peer_sync, handoff):
            assert signals.ready.item() == signals.done.item() == 0

        timings = []
        for name, signals in (("events", None), ("device", sync), ("device", sync), ("events", None)):
            timings.append((name, _bench_ms(
                lambda: layer.forward(hidden, projection_stream=peer, projection_sync=signals),
                iterations, peer_stream=peer,
            )))
    print(f"PROJECTION_STREAMS rows={rows} exact_output_and_state=PASS bidirectional_replays=8 abba_ms={timings}", flush=True)


def _check_fusions(rows: int, seed: int, bench_iters: int) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    slots = rows + 1
    key_heads = 16
    value_heads = 64
    head_dim = 128
    key_dim = key_heads * head_dim
    value_dim = value_heads * head_dim
    channels = 2 * key_dim + value_dim

    x = (torch.randn(rows, channels, device=device) * 0.1).to(dtype)
    weight = (torch.randn(channels, 1, 4, device=device) * 0.1).to(dtype)
    conv_initial = (torch.randn(slots, channels, 3, device=device) * 0.1).to(dtype)
    state_indices = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.arange(1, rows + 1, dtype=torch.int32, device=device)

    conv_control_state = conv_initial.clone()
    mixed = qwen35_causal_conv1d_decode(
        x, weight, conv_control_state, state_indices, positions
    )
    q_control, k_control, v_control = (
        part.contiguous()
        for part in mixed.split((key_dim, key_dim, value_dim), dim=-1)
    )

    conv_candidate_state = conv_initial.clone()
    q, k, v, q_scale, k_scale = qwen35_causal_conv1d_decode_packed(
        x,
        weight,
        conv_candidate_state,
        state_indices,
        positions,
        key_dim=key_dim,
        value_dim=value_dim,
        key_head_dim=head_dim,
    )
    torch.testing.assert_close(q, q_control, rtol=0, atol=0)
    torch.testing.assert_close(k, k_control, rtol=0, atol=0)
    torch.testing.assert_close(v, v_control, rtol=0, atol=0)
    torch.testing.assert_close(
        conv_candidate_state, conv_control_state, rtol=0, atol=0
    )

    a = (torch.randn(rows, value_heads, device=device) * 0.1).to(dtype)
    b = (torch.randn(rows, value_heads, device=device) * 0.1).to(dtype)
    A_log = torch.randn(value_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(value_heads, dtype=torch.float32, device=device) * 0.1
    g, beta = qwen35_gdn_gates(a, b, A_log, dt_bias)
    recurrent_initial = (
        torch.randn(slots, value_heads, head_dim, head_dim, device=device) * 0.01
    ).float()
    recurrent_control = recurrent_initial.clone()
    recurrent_candidate = recurrent_initial.clone()

    control = qwen35_gdn_decode(
        q_control.view(rows, key_heads, head_dim),
        k_control.view(rows, key_heads, head_dim),
        v_control.view(rows, value_heads, head_dim),
        a,
        b,
        A_log,
        dt_bias,
        recurrent_control,
        state_indices,
        positions,
    )
    candidate = qwen35_gdn_decode(
        q.view(rows, key_heads, head_dim),
        k.view(rows, key_heads, head_dim),
        v.view(rows, value_heads, head_dim),
        a,
        b,
        A_log,
        dt_bias,
        recurrent_candidate,
        state_indices,
        positions,
        q_scale,
        k_scale,
        g,
        beta,
    )
    torch.testing.assert_close(candidate, control, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        recurrent_candidate, recurrent_control, rtol=2e-4, atol=2e-4
    )

    z_storage = (torch.randn(rows, value_dim + 256, device=device) * 0.1).to(dtype)
    z = z_storage[:, :value_dim].view(rows, value_heads, head_dim)
    norm_weight = torch.randn(head_dim, device=device).to(dtype)
    gated = qwen35_rmsnorm_gated(candidate, z, norm_weight, 1.0e-6)
    fp8_control, scale_control = per_token_cast_to_fp8_torch(
        gated.flatten(1),
        use_ue8m0=True,
        gran_k=128,
        use_packed_ue8m0=True,
    )
    fp8_candidate, scale_candidate = qwen35_rmsnorm_gated_fp8(
        candidate, z, norm_weight, 1.0e-6
    )
    if not torch.equal(scale_candidate.contiguous(), scale_control):
        raise AssertionError("fused gated-RMSNorm produced different UE8M0 scales")
    torch.testing.assert_close(
        fp8_candidate.flatten(1).float(),
        fp8_control.float(),
        rtol=0,
        atol=0,
    )

    print(
        "PASS "
        f"rows={rows} conv_max={_max_abs(q, q_control):.3e} "
        f"gdn_max={_max_abs(candidate, control):.3e} "
        f"state_max={_max_abs(recurrent_candidate, recurrent_control):.3e} "
        f"fp8_max={_max_abs(fp8_candidate.flatten(1), fp8_control):.3e}"
    )

    if bench_iters:
        def control_conv():
            mixed_output = qwen35_causal_conv1d_decode(
                x, weight, conv_control_state, state_indices, positions
            )
            return tuple(
                part.contiguous()
                for part in mixed_output.split((key_dim, key_dim, value_dim), dim=-1)
            )

        def candidate_conv():
            return qwen35_causal_conv1d_decode_packed(
                x,
                weight,
                conv_candidate_state,
                state_indices,
                positions,
                key_dim=key_dim,
                value_dim=value_dim,
                key_head_dim=head_dim,
            )

        def control_recurrent():
            return qwen35_gdn_decode(
                q_control.view(rows, key_heads, head_dim),
                k_control.view(rows, key_heads, head_dim),
                v_control.view(rows, value_heads, head_dim),
                a,
                b,
                A_log,
                dt_bias,
                recurrent_control,
                state_indices,
                positions,
            )

        def candidate_recurrent():
            return qwen35_gdn_decode(
                q.view(rows, key_heads, head_dim),
                k.view(rows, key_heads, head_dim),
                v.view(rows, value_heads, head_dim),
                a,
                b,
                A_log,
                dt_bias,
                recurrent_candidate,
                state_indices,
                positions,
                q_scale,
                k_scale,
                g,
                beta,
            )

        def control_norm_quant():
            norm_output = qwen35_rmsnorm_gated(
                candidate, z, norm_weight, 1.0e-6
            )
            return per_token_cast_to_fp8(
                norm_output.flatten(1),
                use_ue8m0=True,
                gran_k=128,
                use_packed_ue8m0=True,
            )

        timings = {
            "control_conv_materialize_ms": _bench_ms(control_conv, bench_iters),
            "candidate_conv_scale_pack_ms": _bench_ms(
                candidate_conv, bench_iters
            ),
            "candidate_gate_ms": _bench_ms(
                lambda: qwen35_gdn_gates(a, b, A_log, dt_bias), bench_iters
            ),
            "control_recurrent_ms": _bench_ms(
                control_recurrent, bench_iters
            ),
            "candidate_recurrent_ms": _bench_ms(
                candidate_recurrent, bench_iters
            ),
            "control_norm_quant_ms": _bench_ms(
                control_norm_quant, bench_iters
            ),
            "candidate_norm_quant_ms": _bench_ms(
                lambda: qwen35_rmsnorm_gated_fp8(
                    candidate, z, norm_weight, 1.0e-6
                ),
                bench_iters,
            ),
        }
        timings["conv_delta_us"] = 1000.0 * (
            timings["candidate_conv_scale_pack_ms"]
            - timings["control_conv_materialize_ms"]
        )
        timings["recurrent_plus_gate_delta_us"] = 1000.0 * (
            timings["candidate_recurrent_ms"]
            + timings["candidate_gate_ms"]
            - timings["control_recurrent_ms"]
        )
        timings["norm_quant_delta_us"] = 1000.0 * (
            timings["candidate_norm_quant_ms"]
            - timings["control_norm_quant_ms"]
        )
        print("BENCH " + " ".join(f"{key}={value:.6f}" for key, value in timings.items()))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[3])
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--model-config", help="Optional HF config for complete synthetic-weight GDN checks")
    parser.add_argument("--projection-rows", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        raise ValueError("--rows must be positive")
    if args.bench_iters < 0:
        raise ValueError("--bench-iters must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 GDN fusion validation requires CUDA")

    if args.projection_rows and (not args.model_config or args.bench_iters <= 0):
        raise ValueError("projection checks require --model-config and --bench-iters > 0")
    if any(rows <= 0 for rows in args.projection_rows):
        raise ValueError("projection rows must be positive")

    for rows in args.rows:
        _check_fusions(rows, args.seed, args.bench_iters)

    if args.projection_rows:
        from minisgl.core import Context, set_global_ctx
        set_global_ctx(Context(64))
    for projection_rows in args.projection_rows:
        _check_projection_streams(args.model_config, projection_rows, args.bench_iters)


if __name__ == "__main__":
    main()
