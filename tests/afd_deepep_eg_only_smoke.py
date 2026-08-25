from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from minisgl.core import Context, set_global_ctx
from minisgl.distributed import set_ep_info, set_tp_info
from minisgl.kernel import deepgemm as deep_gemm
from minisgl.kernel.deepep_moe import DeepEPMoeElasticBuffer
from minisgl.layers.moe import MoELayer
from minisgl.models.config import QuantConfig
from minisgl.utils import torch_dtype


def _reference(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    top_k: int,
    intermediate_size: int,
) -> torch.Tensor:
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    weights, experts = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    expert_class = (experts % 8).float().unsqueeze(-1)
    gate_scale = 0.125 * (expert_class + 1.0)
    up_scale = 0.5 + 0.0625 * expert_class
    down_scale = 0.75 + 0.03125 * expert_class
    x = hidden[:, None, :intermediate_size].float()
    activated = torch.nn.functional.silu(x * gate_scale) * (x * up_scale)
    output[:, :intermediate_size] = (
        weights.unsqueeze(-1) * activated * down_scale
    ).sum(dim=1)
    return output


def _fill_bf16_weights(layer: MoELayer, rank: int, experts_per_rank: int) -> None:
    intermediate_size = int(layer.intermediate_size)
    hidden_size = int(layer.hidden_size)
    identity_in = torch.eye(
        intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=layer.gate_up_proj.device,
    )
    identity_out = torch.eye(
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=layer.down_proj.device,
    )
    layer.gate_up_proj.zero_()
    layer.down_proj.zero_()
    for local_expert in range(experts_per_rank):
        expert = rank * experts_per_rank + local_expert
        expert_class = expert % 8
        layer.gate_up_proj[local_expert, :intermediate_size].copy_(
            identity_in * (0.125 * (expert_class + 1))
        )
        layer.gate_up_proj[local_expert, intermediate_size:].copy_(
            identity_in * (0.5 + 0.0625 * expert_class)
        )
        layer.down_proj[local_expert].copy_(
            identity_out * (0.75 + 0.03125 * expert_class)
        )


def _fill_fp8_weights(layer: MoELayer, rank: int, experts_per_rank: int) -> None:
    intermediate_size = int(layer.intermediate_size)
    hidden_size = int(layer.hidden_size)
    identity_in = torch.eye(
        intermediate_size,
        hidden_size,
        dtype=torch.float32,
        device=layer.gate_up_proj.device,
    )
    identity_out = torch.eye(
        hidden_size,
        intermediate_size,
        dtype=torch.float32,
        device=layer.down_proj.device,
    )
    layer.gate_up_proj.zero_()
    layer.down_proj.zero_()
    layer.gate_up_proj_scale.fill_(1.0)
    layer.down_proj_scale.fill_(1.0)
    for local_expert in range(experts_per_rank):
        expert = rank * experts_per_rank + local_expert
        expert_class = expert % 8
        layer.gate_up_proj[local_expert, :intermediate_size].copy_(
            (identity_in * (0.125 * (expert_class + 1))).to(torch.float8_e4m3fn)
        )
        layer.gate_up_proj[local_expert, intermediate_size:].copy_(
            (identity_in * (0.5 + 0.0625 * expert_class)).to(torch.float8_e4m3fn)
        )
        layer.down_proj[local_expert].copy_(
            (identity_out * (0.75 + 0.03125 * expert_class)).to(torch.float8_e4m3fn)
        )


def _check(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_limit: float,
    max_rel_limit: float,
) -> tuple[float, float]:
    max_abs = (actual.float() - expected).abs().max()
    scale = expected.abs().clamp_min(0.125)
    max_rel = ((actual.float() - expected).abs() / scale).max()
    metrics = torch.stack((max_abs, max_rel))
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    abs_value = float(metrics[0])
    rel_value = float(metrics[1])
    if abs_value > max_abs_limit or rel_value > max_rel_limit:
        raise AssertionError(
            f"EG-only DeepEP {label} reference mismatch: "
            f"max_abs={abs_value} max_rel={rel_value}"
        )
    return abs_value, rel_value


def _run_staged(
    layer: MoELayer,
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    prepared = layer.prepare_deepep(hidden, router_logits)
    dispatched = layer.dispatch_deepep(prepared)
    expert_output = layer.run_deepep_experts(dispatched)
    return layer.combine_deepep(expert_output)


def _run_staged_from_gate(
    layer: MoELayer,
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    prepared = layer.prepare_deepep_from_gate(hidden, gate_weight)
    dispatched = layer.dispatch_deepep(prepared)
    expert_output = layer.run_deepep_experts(dispatched)
    return layer.combine_deepep(expert_output)


def _benchmark_staged(
    layer: MoELayer,
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    iterations: int,
) -> float:
    if iterations <= 0:
        return 0.0
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _run_staged(layer, hidden, router_logits)
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        [start.elapsed_time(end) / iterations],
        dtype=torch.float64,
        device=hidden.device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed[0])


def _benchmark_staged_from_gate(
    layer: MoELayer,
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    iterations: int,
) -> float:
    if iterations <= 0:
        return 0.0
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _run_staged_from_gate(layer, hidden, gate_weight)
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        [start.elapsed_time(end) / iterations],
        dtype=torch.float64,
        device=hidden.device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed[0])


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 4:
        raise RuntimeError(f"EG-only DeepEP smoke requires four ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )
    set_tp_info(rank=0, size=1)
    set_ep_info(rank=rank, size=world_size)

    rows = int(os.environ.get("AFD_MOE_SMOKE_ROWS", "7"))
    hidden_size = int(os.environ.get("AFD_MOE_SMOKE_HIDDEN", "256"))
    intermediate_size = int(os.environ.get("AFD_MOE_SMOKE_INTERMEDIATE", "128"))
    experts_per_rank = int(os.environ.get("AFD_MOE_SMOKE_EXPERTS_PER_RANK", "2"))
    num_experts = experts_per_rank * world_size
    top_k = int(os.environ.get("AFD_MOE_SMOKE_TOP_K", "2"))
    benchmark_iterations = int(os.environ.get("AFD_MOE_SMOKE_ITERS", "0"))
    if rows > hidden_size:
        raise RuntimeError(
            "EG-only DeepEP smoke requires rows <= hidden size for its exact "
            f"fused-router construction, got rows={rows} hidden={hidden_size}"
        )
    deepgemm_sms_raw = os.environ.get("AFD_MOE_SMOKE_DEEPGEMM_SMS")
    deepep_sms_raw = os.environ.get("AFD_MOE_SMOKE_DEEPEP_SMS")
    if (deepgemm_sms_raw is None) != (deepep_sms_raw is None):
        raise RuntimeError(
            "AFD_MOE_SMOKE_DEEPGEMM_SMS and AFD_MOE_SMOKE_DEEPEP_SMS "
            "must be set together"
        )
    total_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    if deepgemm_sms_raw is None:
        deepgemm_sms = int(deep_gemm.get_num_sms())
        deepep_sms = int(DeepEPMoeElasticBuffer.OVERLAP_NUM_SMS)
    else:
        try:
            deepgemm_sms = int(deepgemm_sms_raw)
            deepep_sms = int(deepep_sms_raw)
        except ValueError as error:
            raise RuntimeError(
                "AFD_MOE_SMOKE_DEEPGEMM_SMS and AFD_MOE_SMOKE_DEEPEP_SMS "
                f"must be integers, got {deepgemm_sms_raw!r} and {deepep_sms_raw!r}"
            ) from error
        if deepgemm_sms + deepep_sms != total_sms:
            raise RuntimeError(
                "MoE smoke compute and communication SMs must partition the device: "
                f"compute={deepgemm_sms} communication={deepep_sms} total={total_sms}"
            )
        if deepep_sms not in DeepEPMoeElasticBuffer.SUPPORTED_OVERLAP_NUM_SMS:
            raise RuntimeError(
                "AFD_MOE_SMOKE_DEEPEP_SMS must be a supported complete-cluster "
                f"count {DeepEPMoeElasticBuffer.SUPPORTED_OVERLAP_NUM_SMS}, "
                f"got {deepep_sms}"
            )
        deep_gemm.set_num_sms(deepgemm_sms)
        actual_deepgemm_sms = int(deep_gemm.get_num_sms())
        if actual_deepgemm_sms != deepgemm_sms:
            raise RuntimeError(
                "DeepGEMM rejected the smoke compute-SM budget: "
                f"requested={deepgemm_sms} actual={actual_deepgemm_sms}"
            )
    ctx = Context(page_size=1)
    ctx.replica_id = rank
    ctx.server_args = SimpleNamespace(
        afd_moe_a2a_backend="deepep",
        afd_moe_runner_backend="deep_gemm",
    )
    set_global_ctx(ctx)
    ctx.mlp_deepep_buffer = DeepEPMoeElasticBuffer(
        group=dist.group.WORLD,
        num_max_dispatch_tokens_per_rank=rows,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        overlap_num_sms=deepep_sms,
    )
    ctx.moe_num_token_non_padded = rows
    ctx.moe_deepep_dispatch_max_tokens_per_rank = rows

    with torch.device(device), torch_dtype(torch.bfloat16):
        layer = MoELayer(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=True,
            quant=None,
            a2a_backend="deepep",
            runner_backend="deep_gemm",
        )
    _fill_bf16_weights(layer, rank, experts_per_rank)

    # Give every configured row an exact independent feature so the fused
    # router construction remains nonsingular for production-sized smokes.
    hidden_f32 = torch.zeros((rows, hidden_size), dtype=torch.float32, device=device)
    hidden_f32[:, :rows] = torch.eye(rows, dtype=torch.float32, device=device)
    hidden_f32.add_(rank * 0.015625)
    hidden = hidden_f32.to(torch.bfloat16)
    router_logits = torch.full(
        (rows, num_experts), -7.0, dtype=torch.bfloat16, device=device
    )
    row_indices = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    route_indices = torch.arange(top_k, device=device, dtype=torch.int64)[None, :]
    selected_experts = (
        rank * experts_per_rank + row_indices * top_k + route_indices * 17
    ) % num_experts
    selected_scores = 2.0 - 0.125 * route_indices.to(torch.float32)
    router_logits.scatter_(
        1,
        selected_experts,
        selected_scores.expand(rows, top_k).to(torch.bfloat16),
    )

    expected = _reference(
        hidden,
        router_logits,
        top_k=top_k,
        intermediate_size=intermediate_size,
    )
    bf16_abs, bf16_rel = _check(
        "BF16",
        _run_staged(layer, hidden, router_logits),
        expected,
        max_abs_limit=0.04,
        max_rel_limit=0.04,
    )
    bf16_ms = _benchmark_staged(
        layer,
        hidden,
        router_logits,
        iterations=benchmark_iterations,
    )

    with torch.device(device), torch_dtype(torch.bfloat16):
        fp8_layer = MoELayer(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=True,
            quant=QuantConfig(
                method="fp8",
                activation_scheme="dynamic",
                weight_block_size=(128, 128),
            ),
            a2a_backend="deepep",
            runner_backend="deep_gemm",
        )
    _fill_fp8_weights(fp8_layer, rank, experts_per_rank)
    fp8_abs, fp8_rel = _check(
        "FP8",
        _run_staged(fp8_layer, hidden, router_logits),
        expected,
        max_abs_limit=0.08,
        max_rel_limit=0.15,
    )

    # Solve for a BF16 gate with the same wide routing margins as the
    # hand-authored logits. This exercises the production tensor-core router
    # without making the smoke depend on random near-ties at the top-k edge.
    hidden_f32 = hidden.float()
    gate_weight = (
        router_logits.float().T
        @ torch.linalg.solve(hidden_f32 @ hidden_f32.T, hidden_f32)
    ).to(torch.bfloat16).contiguous()
    fused_logits = torch.nn.functional.linear(hidden, gate_weight)
    fused_expected = _reference(
        hidden,
        fused_logits,
        top_k=top_k,
        intermediate_size=intermediate_size,
    )
    fused_abs, fused_rel = _check(
        "FP8 fused router",
        _run_staged_from_gate(fp8_layer, hidden, gate_weight),
        fused_expected,
        max_abs_limit=0.08,
        max_rel_limit=0.15,
    )
    fp8_ms = _benchmark_staged(
        fp8_layer,
        hidden,
        fused_logits,
        iterations=benchmark_iterations,
    )
    fp8_fused_router_ms = _benchmark_staged_from_gate(
        fp8_layer,
        hidden,
        gate_weight,
        iterations=benchmark_iterations,
    )

    dist.barrier()
    ctx.mlp_deepep_buffer.destroy()
    if rank == 0:
        print(
            "AFD_EG_ONLY_DEEPEP_SMOKE_OK "
            f"ranks={world_size} rows={rows} experts={num_experts} top_k={top_k} "
            f"bf16_max_abs={bf16_abs:.8f} bf16_max_rel={bf16_rel:.8f} "
            f"fp8_max_abs={fp8_abs:.8f} fp8_max_rel={fp8_rel:.8f} "
            f"fused_router_max_abs={fused_abs:.8f} "
            f"fused_router_max_rel={fused_rel:.8f} "
            f"benchmark_iterations={benchmark_iterations} "
            f"deepgemm_sms={deepgemm_sms} deepep_sms={deepep_sms} "
            f"bf16_staged_ms={bf16_ms:.6f} fp8_staged_ms={fp8_ms:.6f} "
            f"fp8_fused_router_staged_ms={fp8_fused_router_ms:.6f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
