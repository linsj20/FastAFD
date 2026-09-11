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


def _bench_ms(function, iterations: int) -> float:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(10):
            function()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bench-iters", type=int, default=0)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.bench_iters < 0:
        raise ValueError("--bench-iters must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 GDN fusion validation requires CUDA")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = args.rows
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

    if args.bench_iters:
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
            "control_conv_materialize_ms": _bench_ms(control_conv, args.bench_iters),
            "candidate_conv_scale_pack_ms": _bench_ms(
                candidate_conv, args.bench_iters
            ),
            "candidate_gate_ms": _bench_ms(
                lambda: qwen35_gdn_gates(a, b, A_log, dt_bias), args.bench_iters
            ),
            "control_recurrent_ms": _bench_ms(
                control_recurrent, args.bench_iters
            ),
            "candidate_recurrent_ms": _bench_ms(
                candidate_recurrent, args.bench_iters
            ),
            "control_norm_quant_ms": _bench_ms(
                control_norm_quant, args.bench_iters
            ),
            "candidate_norm_quant_ms": _bench_ms(
                lambda: qwen35_rmsnorm_gated_fp8(
                    candidate, z, norm_weight, 1.0e-6
                ),
                args.bench_iters,
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


if __name__ == "__main__":
    main()
