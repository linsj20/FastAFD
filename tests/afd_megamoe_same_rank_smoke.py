"""Four-GPU correctness and exact-Qwen-shape smoke for same-rank MegaMoE."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from minisgl.kernel import deepgemm
from minisgl.kernel import megamoe_mega
from minisgl.kernel import megamoe_m2n_mega as weight_utils
from minisgl.kernel.fp8_quant import per_token_cast_to_fp8
from minisgl.kernel.moe_topk import gate_topk


def _positive_int_list(name: str, default: str) -> list[int]:
    values = [int(value) for value in os.environ.get(name, default).split(",")]
    if not values or any(value <= 0 for value in values):
        raise RuntimeError(f"{name} must be a comma-separated list of positive integers")
    return values


def _time_graph(
    run_stage,
    output: torch.Tensor,
    *,
    warmups: int,
    iterations: int,
) -> tuple[float, float]:
    run_stage()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_stage()
    dist.barrier()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        [start.elapsed_time(end) / iterations], dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if not torch.isfinite(output).all():
        raise RuntimeError("exact Qwen MegaMoE output is non-finite")
    return float(elapsed[0]), float(output.abs().max())


def _time_graph_with_resident_cta(
    run_stage,
    output: torch.Tensor,
    *,
    sleep_cycles: int,
    warmups: int,
    iterations: int,
) -> tuple[float, float]:
    run_stage()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_stage()
    resident_stream = torch.cuda.Stream(priority=-1)
    for _ in range(warmups):
        with torch.cuda.stream(resident_stream):
            torch.cuda._sleep(sleep_cycles)
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        with torch.cuda.stream(resident_stream):
            torch.cuda._sleep(sleep_cycles)
        graph.replay()
    end.record()
    end.synchronize()
    resident_stream.synchronize()
    elapsed = torch.tensor(
        [start.elapsed_time(end) / iterations], dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if not torch.isfinite(output).all():
        raise RuntimeError("contended exact Qwen MegaMoE output is non-finite")
    return float(elapsed[0]), float(output.abs().max())


def _max_active_experts(
    ids: torch.Tensor, local_experts: int, world_size: int
) -> int:
    gathered = torch.empty(
        (world_size * ids.numel(),), dtype=ids.dtype, device=ids.device
    )
    dist.all_gather_into_tensor(gathered, ids.flatten())
    return max(
        int(
            torch.unique(
                gathered[
                    (gathered >= owner * local_experts)
                    & (gathered < (owner + 1) * local_experts)
                ]
            ).numel()
        )
        for owner in range(world_size)
    )


def _make_weights(
    local_experts: int,
    hidden: int,
    intermediate: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
]:
    l1 = (torch.randn(
        (local_experts, 2 * intermediate, hidden), device="cuda"
    ) * 0.125).to(torch.float8_e4m3fn)
    l2 = (torch.randn(
        (local_experts, hidden, intermediate), device="cuda"
    ) * 0.125).to(torch.float8_e4m3fn)

    def scales(n: int, k: int) -> torch.Tensor:
        unpacked = torch.ones(
            (local_experts, n, k // 32), dtype=torch.float32, device="cuda"
        )
        return deepgemm.transform_sf_into_required_layout(
            unpacked, n, k, (1, 32), local_experts
        )

    def block_scales(n: int, k: int) -> torch.Tensor:
        return torch.ones(
            (local_experts, (n + 127) // 128, (k + 127) // 128),
            dtype=torch.float32,
            device="cuda",
        )

    l1_reference = (l1, scales(2 * intermediate, hidden))
    l2_reference = (l2, scales(hidden, intermediate))
    if os.environ.get("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp8") == "fp4":
        l1_kernel = weight_utils.requant_qwen_fp8_weights_to_fp4(
            l1, block_scales(2 * intermediate, hidden)
        )
        l2_kernel = weight_utils.requant_qwen_fp8_weights_to_fp4(
            l2, block_scales(hidden, intermediate)
        )
    else:
        l1_kernel, l2_kernel = l1_reference, l2_reference
    transformed = weight_utils.transform_weights_for_mega_moe(
        l1_kernel, l2_kernel
    )
    return l1_reference, l2_reference, transformed


def _stage(
    buffer: megamoe_mega.MegaMoESymmBuffer,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_q, x_sf = per_token_cast_to_fp8(
        hidden,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=False,
        backend="torch",
    )
    rows = int(hidden.shape[0])
    buffer.x[:rows].copy_(x_q)
    buffer.x_sf[:rows].copy_(deepgemm.pack_ue8m0_to_int(x_sf))
    buffer.topk_idx[:rows].copy_(ids)
    buffer.topk_weights[:rows].copy_(weights)
    return x_q, x_sf


def _numeric_case(rank: int, world_size: int) -> dict[str, float]:
    rows, hidden, intermediate = 3, 512, 512
    local_experts, top_k = 2, 2
    num_experts = local_experts * world_size
    l1, l2, transformed = _make_weights(local_experts, hidden, intermediate)
    buffer = megamoe_mega.MegaMoESymmBuffer(
        dist.group.WORLD,
        num_experts=num_experts,
        num_max_tokens_per_rank=rows,
        num_topk=top_k,
        hidden=hidden,
        intermediate_hidden=intermediate,
    )
    hidden_states = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
    ids = (
        torch.arange(rows * top_k, dtype=torch.int64, device="cuda").view(rows, top_k)
        + rank * rows
    ) % num_experts
    weights = torch.softmax(
        torch.randn((rows, top_k), dtype=torch.float32, device="cuda"), dim=-1
    )
    x_q, x_sf = _stage(buffer, hidden_states, ids, weights)
    output = buffer.y[:rows]
    megamoe_mega.fp8_fp8_mega_moe(output, transformed[0], transformed[1], buffer)

    x_dequant = x_q.float() * torch.repeat_interleave(x_sf.float(), 32, dim=1)
    all_x = torch.empty(
        (world_size * rows, hidden), dtype=torch.float32, device="cuda"
    )
    all_ids = torch.empty(
        (world_size * rows, top_k), dtype=torch.int64, device="cuda"
    )
    all_weights = torch.empty(
        (world_size * rows, top_k), dtype=torch.float32, device="cuda"
    )
    dist.all_gather_into_tensor(all_x, x_dequant)
    dist.all_gather_into_tensor(all_ids, ids)
    dist.all_gather_into_tensor(all_weights, weights)
    all_x = all_x.view(world_size, rows, hidden)
    all_ids = all_ids.view(world_size, rows, top_k)
    all_weights = all_weights.view(world_size, rows, top_k)

    reference = torch.zeros(
        (world_size, rows, hidden), dtype=torch.float32, device="cuda"
    )
    for source in range(world_size):
        for token in range(rows):
            for route in range(top_k):
                expert = int(all_ids[source, token, route])
                if expert // local_experts != rank:
                    continue
                local_expert = expert % local_experts
                gate_up = F.linear(all_x[source, token], l1[0][local_expert].float())
                gate, up = gate_up.chunk(2)
                activated = F.silu(gate) * up * all_weights[source, token, route]
                act_q, act_sf = per_token_cast_to_fp8(
                    activated[None],
                    use_ue8m0=True,
                    gran_k=32,
                    use_packed_ue8m0=False,
                    backend="torch",
                )
                act_dequant = act_q[0].float() * torch.repeat_interleave(
                    act_sf[0].float(), 32
                )
                reference[source, token].add_(
                    F.linear(act_dequant, l2[0][local_expert].float())
                )
    dist.all_reduce(reference)
    expected = reference[rank]
    actual = output.float()
    max_abs = float((actual - expected).abs().max())
    cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    min_cosine = (
        0.96
        if os.environ.get("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp8") == "fp4"
        else 0.98
    )
    if not torch.isfinite(actual).all() or cosine < min_cosine:
        raise RuntimeError(
            f"same-rank MegaMoE numeric mismatch rank={rank} "
            f"max_abs={max_abs:.6f} cosine={cosine:.6f}"
        )
    return {"max_abs": max_abs, "cosine": cosine}


def _exact_qwen_case(
    rank: int, world_size: int
) -> dict[str, dict[str, float | int]]:
    row_counts = _positive_int_list("MEGAMOE_SMOKE_ROWS", "3")
    sm_counts = _positive_int_list(
        "MEGAMOE_SMOKE_SMS",
        str(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
    )
    warmups = int(os.environ.get("MEGAMOE_SMOKE_WARMUPS", "10"))
    iterations = int(os.environ.get("MEGAMOE_SMOKE_ITERATIONS", "100"))
    bucket = int(os.environ.get("MEGAMOE_SMOKE_BUCKET", str(max(row_counts))))
    contend_cycles = int(os.environ.get("MEGAMOE_SMOKE_CONTEND_CYCLES", "0"))
    if warmups < 0 or iterations <= 0:
        raise RuntimeError("MegaMoE smoke warmups must be nonnegative and iterations positive")
    if contend_cycles < 0:
        raise RuntimeError("MEGAMOE_SMOKE_CONTEND_CYCLES must be nonnegative")
    if bucket < max(row_counts):
        raise RuntimeError(
            f"MEGAMOE_SMOKE_BUCKET={bucket} is smaller than rows={max(row_counts)}"
        )

    hidden, intermediate = 4096, 1536
    local_experts, top_k = 32, 8
    num_experts = local_experts * world_size
    _l1, _l2, transformed = _make_weights(local_experts, hidden, intermediate)
    del _l1, _l2
    buffer = megamoe_mega.MegaMoESymmBuffer(
        dist.group.WORLD,
        num_experts=num_experts,
        num_max_tokens_per_rank=bucket,
        num_topk=top_k,
        hidden=hidden,
        intermediate_hidden=intermediate,
    )
    hidden_states = torch.randn(
        (max(row_counts), hidden), dtype=torch.bfloat16, device="cuda"
    )
    gate_weight = torch.randn(
        (num_experts, hidden), dtype=torch.bfloat16, device="cuda"
    )
    expert_map = torch.arange(num_experts, dtype=torch.int64, device="cuda")
    results: dict[str, dict[str, float | int]] = {}

    for rows in row_counts:
        hidden_view = hidden_states[:rows]
        ids = buffer.topk_idx[:rows]
        weights = buffer.topk_weights[:rows]
        output = buffer.y[:rows]

        def run_stage() -> None:
            gate_topk(
                hidden_view,
                gate_weight,
                top_k,
                renormalize=True,
                expert_map=expert_map,
                topk_idx_dtype=torch.int64,
                quant_out=(buffer.x[:rows], buffer.x_sf[:rows], None),
                out=(ids, weights),
            )
            megamoe_mega.fp8_fp8_mega_moe(
                output, transformed[0], transformed[1], buffer
            )

        run_stage()
        torch.cuda.synchronize()
        max_active = _max_active_experts(ids, local_experts, world_size)
        for num_sms in sm_counts:
            deepgemm.set_num_sms(num_sms)
            elapsed, output_max = _time_graph(
                run_stage, output, warmups=warmups, iterations=iterations
            )
            results[f"bucket{bucket}_rows{rows}_sms{num_sms}"] = {
                "milliseconds": elapsed,
                "output_max": output_max,
                "max_active_experts": max_active,
            }
            dist.barrier()

    fixed_rows = min(row_counts)
    if fixed_rows * top_k != 24:
        return results
    fixed_hidden = hidden_states[:fixed_rows]
    fixed_weights = torch.full(
        (fixed_rows, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    for pattern in ("balanced24", "one_rank32"):
        flat = torch.arange(fixed_rows * top_k, dtype=torch.int64, device="cuda")
        if pattern == "balanced24":
            fixed_ids = (flat % world_size) * local_experts + rank * 6 + flat // 4
        else:
            fixed_ids = (flat + rank * fixed_rows * top_k) % local_experts
        fixed_ids = fixed_ids.view(fixed_rows, top_k)
        _stage(buffer, fixed_hidden, fixed_ids, fixed_weights)
        ids = buffer.topk_idx[:fixed_rows]
        output = buffer.y[:fixed_rows]
        max_active = _max_active_experts(ids, local_experts, world_size)

        def run_fixed() -> None:
            megamoe_mega.fp8_fp8_mega_moe(
                output, transformed[0], transformed[1], buffer
            )

        for num_sms in sm_counts:
            deepgemm.set_num_sms(num_sms)
            elapsed, output_max = _time_graph(
                run_fixed, output, warmups=warmups, iterations=iterations
            )
            results[f"bucket{bucket}_{pattern}_sms{num_sms}"] = {
                "milliseconds": elapsed,
                "output_max": output_max,
                "max_active_experts": max_active,
            }
            if contend_cycles:
                elapsed, output_max = _time_graph_with_resident_cta(
                    run_fixed,
                    output,
                    sleep_cycles=contend_cycles,
                    warmups=warmups,
                    iterations=iterations,
                )
                results[
                    f"bucket{bucket}_{pattern}_sms{num_sms}_"
                    f"resident{contend_cycles}"
                ] = {
                    "milliseconds": elapsed,
                    "output_max": output_max,
                    "max_active_experts": max_active,
                }
            dist.barrier()
    return results


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 4:
        raise RuntimeError(f"same-rank MegaMoE smoke requires four ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    expert_weight_dtype = os.environ.get(
        "MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp8"
    )
    if expert_weight_dtype not in ("fp8", "fp4"):
        raise RuntimeError("MegaMoE smoke expert weight dtype must be fp8 or fp4")
    if expert_weight_dtype == "fp4":
        megamoe_mega.fp8_fp8_mega_moe = megamoe_mega.fp8_fp4_mega_moe
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )
    deepgemm.set_num_sms(
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    )
    torch.manual_seed(20260825 + rank)
    numeric = _numeric_case(rank, world_size)
    dist.barrier()
    exact = _exact_qwen_case(rank, world_size)
    if rank == 0:
        print(json.dumps({
            "status": "ok",
            "expert_weight_dtype": expert_weight_dtype,
            "numeric": numeric,
            "exact_qwen": exact,
        }, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
