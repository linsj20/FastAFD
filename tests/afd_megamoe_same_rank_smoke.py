"""Correctness and exact-Qwen-shape smoke for same-rank MegaMoE."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from minisgl.kernel import deepgemm
from minisgl.kernel import megamoe_mega
from minisgl.kernel import megamoe_m2n_mega as weight_utils
from minisgl.kernel.fp8_quant import per_token_cast_to_fp8
from minisgl.kernel.moe_topk import gate_topk


def _check_source_results(case: str, rank: int, **values: Any) -> None:
    """Compare raw tensor bytes across source trees, outside timed regions."""
    root = os.environ.get("MEGAMOE_RESULT_REFERENCE_DIR", "")
    mode = os.environ.get("MEGAMOE_RESULT_MODE", "")
    if not root and not mode:
        return
    if not root or mode not in ("record", "compare"):
        raise RuntimeError("result comparison requires a directory and record/compare mode")
    tensors: dict[str, torch.Tensor] = {}

    def collect(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu().contiguous()
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                collect(f"{name}.{index}", item)
        else:
            raise TypeError(f"unsupported result snapshot value: {name}")

    for name, value in values.items():
        collect(name, value)
    path = Path(root) / f"{case}_rank{rank}.pt"
    if mode == "record":
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently replace the control with candidate data.
        with path.open("xb") as stream:
            torch.save(tensors, stream)
    else:
        reference = torch.load(path, map_location="cpu", weights_only=True)
        if reference.keys() != tensors.keys():
            raise RuntimeError(f"result snapshot keys differ: {path}")
        for name, actual in tensors.items():
            expected = reference[name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f"result metadata differs: {path} tensor={name}")
            different = actual.view(torch.uint8) != expected.view(torch.uint8)
            if different.any():
                raise RuntimeError(
                    f"result bytes differ: {path} tensor={name} "
                    f"mismatched_bytes={int(different.sum())}"
                )
    print(f"MEGAMOE_RESULT_{mode.upper()}_OK case={case} rank={rank}", flush=True)


_MEGAMOE_TIMING_SLOTS = (
    "dispatch_pre_pull",
    "dispatch_pull",
    "dispatch_cleanup",
    "tma_a_loop",
    "tma_a_first_l1_arrival_wait",
    "tma_a_l1_arrival_wait",
    "tma_a_l2_arrival_wait",
    "tma_a_empty_wait",
    "tma_b_loop",
    "tma_b_empty_wait",
    "mma_loop",
    "mma_first_full_wait",
    "mma_tmem_empty_wait",
    "mma_full_wait",
    "epilogue_loop",
    "epilogue_tmem_full_wait",
    "l2_remote_writeback",
    "tag2_barrier",
    "combine",
    "tma_a_first_l1_ready_from_kernel_start",
    "mma_first_full_ready_from_kernel_start",
)


def _positive_int_list(name: str, default: str) -> list[int]:
    values = [int(value) for value in os.environ.get(name, default).split(",")]
    if not values or any(value <= 0 for value in values):
        raise RuntimeError(f"{name} must be a comma-separated list of positive integers")
    return values


def _nonnegative_int_list(name: str, default: str) -> list[int]:
    values = [int(value) for value in os.environ.get(name, default).split(",")]
    if not values or any(value < 0 for value in values):
        raise RuntimeError(f"{name} must be a comma-separated list of nonnegative integers")
    return values


def _time_graph(
    run_stage,
    output: torch.Tensor,
    *,
    warmups: int,
    iterations: int,
) -> tuple[float, float]:
    graph_batch = int(os.environ.get("MEGAMOE_SMOKE_GRAPH_BATCH", "1"))
    if graph_batch < 1 or iterations % graph_batch:
        raise ValueError("graph batch must be positive and divide timing iterations")
    run_stage()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_batch):
            run_stage()
    dist.barrier()
    for _ in range((warmups + graph_batch - 1) // graph_batch):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    profiler_range = int(os.environ.get("MEGAMOE_SMOKE_CUDA_PROFILER_RANGE", "0"))
    if profiler_range not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_CUDA_PROFILER_RANGE must be 0 or 1")
    if profiler_range:
        torch.cuda.profiler.start()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations // graph_batch):
        graph.replay()
    end.record()
    end.synchronize()
    if profiler_range:
        torch.cuda.profiler.stop()
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
    *,
    build_reference: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor] | None,
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

    is_fp4 = os.environ.get("MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp8") == "fp4"
    l1_reference = l2_reference = None
    if build_reference or not is_fp4:
        l1_reference = (l1, scales(2 * intermediate, hidden))
        l2_reference = (l2, scales(hidden, intermediate))
    if is_fp4:
        l1_kernel = weight_utils.requant_qwen_fp8_weights_to_fp4(
            l1, block_scales(2 * intermediate, hidden)
        )
        l2_kernel = weight_utils.requant_qwen_fp8_weights_to_fp4(
            l2, block_scales(hidden, intermediate)
        )

        def dequantized_fp4_reference(weight: torch.Tensor) -> torch.Tensor:
            experts = []
            for expert in range(local_experts):
                packed, sf = weight_utils.per_token_cast_to_fp4(
                    weight[expert].float(), use_ue8m0=True, gran_k=32
                )
                experts.append(weight_utils.dequant_fp4(packed, sf, gran_k=32))
            return torch.stack(experts)

        if build_reference:
            # Numeric tests need these; fixed-perfect profiling discards them.
            l1_reference = (dequantized_fp4_reference(l1), l1_reference[1])
            l2_reference = (dequantized_fp4_reference(l2), l2_reference[1])
    else:
        l1_kernel, l2_kernel = l1_reference, l2_reference
    transformed = weight_utils.transform_weights_for_mega_moe(
        l1_kernel, l2_kernel
    )
    return l1_reference, l2_reference, transformed


def _make_or_load_transformed_weights(
    local_experts: int,
    hidden: int,
    intermediate: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    weight_cache_input = os.environ.get("MEGAMOE_SMOKE_WEIGHT_CACHE_INPUT", "")
    weight_cache_output = os.environ.get("MEGAMOE_SMOKE_WEIGHT_CACHE_OUTPUT", "")
    if weight_cache_input and weight_cache_output:
        raise RuntimeError("MegaMoE smoke weight cache input and output are exclusive")
    if weight_cache_input:
        transformed = torch.load(
            weight_cache_input, map_location="cuda", weights_only=True
        )
        if (
            not isinstance(transformed, tuple)
            or len(transformed) != 2
            or any(not isinstance(stage, tuple) or len(stage) != 2 for stage in transformed)
            or any(not tensor.is_cuda for stage in transformed for tensor in stage)
        ):
            raise RuntimeError("invalid MegaMoE smoke transformed-weight cache")
        return transformed

    _, _, transformed = _make_weights(
        local_experts, hidden, intermediate, build_reference=False
    )
    if weight_cache_output:
        torch.save(transformed, weight_cache_output)
    return transformed


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


def _check_fused_quant_coverage() -> None:
    """Check complete producer ownership across small and full expert grids."""
    hidden = torch.randn((3, 1024), dtype=torch.bfloat16, device="cuda")
    expected_x, expected_sf = per_token_cast_to_fp8(
        hidden, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True, backend="torch"
    )
    for experts in (8, 16, 24, 64, 128, 512):
        gate = torch.randn((experts, 1024), dtype=torch.bfloat16, device="cuda")
        x = torch.empty_like(expected_x)
        sf = torch.full_like(expected_sf, -1)
        x.view(torch.uint8).fill_(0xFF)
        gate_topk(hidden, gate, 2, renormalize=True, quant_out=(x, sf, None))
        if (not torch.equal(x.view(torch.uint8), expected_x.view(torch.uint8))
                or not torch.equal(sf, expected_sf)):
            raise RuntimeError(f"fused quantization coverage mismatch: experts={experts}")


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
    if not math.isfinite(cosine) or not torch.isfinite(actual).all() or cosine < min_cosine:
        raise RuntimeError(
            f"same-rank MegaMoE numeric mismatch rank={rank} "
            f"max_abs={max_abs:.6f} cosine={cosine:.6f}"
        )
    _check_source_results(
        "numeric", rank, hidden=hidden_states, transformed=transformed,
        x=buffer.x[:rows], sf=buffer.x_sf[:rows], ids=ids,
        weights=weights, output=output,
    )
    return {"max_abs": max_abs, "cosine": cosine}


def _numeric_row_sweep(rank: int, world_size: int) -> dict[str, dict[str, float]]:
    """Reference-check A1/A2 row counts across graph-stable buffer buckets."""
    row_counts = sorted(set(_positive_int_list("MEGAMOE_NUMERIC_ROWS", "3,6")))
    bucket_counts = sorted(
        set(_positive_int_list("MEGAMOE_NUMERIC_BUCKETS", "518,1036"))
    )
    if min(bucket_counts) < max(row_counts):
        raise RuntimeError(
            "MEGAMOE_NUMERIC_BUCKETS must all cover the largest numeric row count"
        )

    hidden = int(os.environ.get("MEGAMOE_NUMERIC_HIDDEN", "1024"))
    intermediate = int(os.environ.get("MEGAMOE_NUMERIC_INTERMEDIATE", "512"))
    num_experts = int(os.environ.get(
        "MEGAMOE_NUMERIC_NUM_EXPERTS", str(2 * world_size)
    ))
    top_k = int(os.environ.get("MEGAMOE_NUMERIC_TOP_K", "2"))
    if hidden % 512 or intermediate % 512:
        raise RuntimeError(
            "MegaMoE numeric dimensions must be divisible by 512: "
            f"hidden={hidden} intermediate={intermediate}"
        )
    if num_experts <= 0 or num_experts % world_size:
        raise RuntimeError(
            "MEGAMOE_NUMERIC_NUM_EXPERTS must be positive and divisible by "
            f"world_size={world_size}, got {num_experts}"
        )
    if not 0 < top_k <= num_experts:
        raise RuntimeError(
            f"MEGAMOE_NUMERIC_TOP_K must be in [1, {num_experts}], got {top_k}"
        )
    local_experts = num_experts // world_size
    l1, l2, transformed = _make_weights(local_experts, hidden, intermediate)
    hidden_states = torch.randn(
        (max(row_counts), hidden), dtype=torch.bfloat16, device="cuda"
    )
    gate_weight = torch.randn(
        (num_experts, hidden), dtype=torch.bfloat16, device="cuda"
    )
    expert_map = torch.arange(
        num_experts, dtype=torch.int64, device="cuda"
    )
    _check_source_results(
        "numeric_rows_inputs", rank, hidden=hidden_states,
        gate=gate_weight, transformed=transformed,
    )

    def reference_for(
        buffer: megamoe_mega.MegaMoESymmBuffer,
        rows: int,
    ) -> torch.Tensor:
        # Consume the fused gate's registered FP8 values and packed UE8M0
        # scales so the reference starts from the exact kernel inputs.
        x_q = buffer.x[:rows]
        x_sf_exponents = (
            buffer.x_sf[:rows]
            .view(torch.uint8)
            .view(rows, hidden // 32)
            .to(torch.int32)
        )
        x_sf = torch.exp2(x_sf_exponents.float() - 127.0)
        x_dequant = x_q.float() * torch.repeat_interleave(x_sf, 32, dim=1)
        ids = buffer.topk_idx[:rows]
        weights = buffer.topk_weights[:rows]
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
                    gate_up = F.linear(
                        all_x[source, token], l1[0][local_expert].float()
                    )
                    gate, up = gate_up.chunk(2)
                    activated = (
                        F.silu(gate)
                        * up
                        * all_weights[source, token, route]
                    )
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
        return reference[rank]

    def run_case(bucket: int, rows: int, prepared: bool) -> dict[str, float]:
        # A1 and A2 are separate processes with separately initialized buffers.
        # Keep this scope self-contained so its rendezvous allocation is released
        # before the next shape is created.
        buffer = megamoe_mega.MegaMoESymmBuffer(
            dist.group.WORLD,
            num_experts=num_experts,
            num_max_tokens_per_rank=bucket,
            num_topk=top_k,
            hidden=hidden,
            intermediate_hidden=intermediate,
        )
        ids = buffer.topk_idx[:rows]
        weights = buffer.topk_weights[:rows]
        output = buffer.y[:rows]
        output.fill_(float("nan"))
        gate_topk(
            hidden_states[:rows],
            gate_weight,
            top_k,
            renormalize=True,
            expert_map=expert_map,
            num_token_non_padded=rows,
            topk_idx_dtype=torch.int64,
            quant_out=(buffer.x[:rows], buffer.x_sf[:rows], None),
            out=(ids, weights),
            route_prepare=(
                buffer.route_prepare_args(rank_ready=True) if prepared else None
            ),
        )
        megamoe_mega.fp8_fp8_mega_moe(
            output,
            transformed[0],
            transformed[1],
            buffer,
            routes_prepared=prepared,
            rank_gated_combine=prepared,
            route_ready_dispatch=prepared,
            rank_ready_route_publish=prepared,
        )
        torch.cuda.synchronize()
        # Independently check producer coverage, not only MoE math against
        # whatever data happened to reach the registered input buffer.
        expected_x, expected_sf = per_token_cast_to_fp8(
            hidden_states[:rows], use_ue8m0=True, gran_k=32,
            use_packed_ue8m0=True, backend="torch",
        )
        if not torch.equal(
            buffer.x[:rows].view(torch.uint8), expected_x.view(torch.uint8)
        ) or not torch.equal(
            buffer.x_sf[:rows].view(torch.uint8), expected_sf.view(torch.uint8)
        ):
            raise RuntimeError(
                f"fused producer quantization mismatch rank={rank} "
                f"rows={rows} prepared={int(prepared)}"
            )
        expected = reference_for(buffer, rows)
        actual = output.float()
        max_abs = float((actual - expected).abs().max())
        cosine = float(
            F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        )
        if not math.isfinite(cosine) or not torch.isfinite(actual).all() or cosine < 0.96:
            nonfinite_by_row = (~torch.isfinite(actual)).sum(dim=1).tolist()
            reference_nonfinite = int((~torch.isfinite(expected)).sum())
            input_nonfinite = int((~torch.isfinite(buffer.x[:rows].float())).sum())
            weight_nonfinite = int((~torch.isfinite(weights)).sum())
            sf_nan_codes = int((buffer.x_sf[:rows].view(torch.uint8) == 255).sum())
            raise RuntimeError(
                "same-rank MegaMoE row sweep mismatch "
                f"rank={rank} bucket={bucket} rows={rows} "
                f"prepared={int(prepared)} max_abs={max_abs:.6f} "
                f"cosine={cosine:.6f} nonfinite_by_row={nonfinite_by_row} "
                f"reference_nonfinite={reference_nonfinite} "
                f"input_nonfinite={input_nonfinite} weight_nonfinite={weight_nonfinite} "
                f"sf_nan_codes={sf_nan_codes}"
            )
        _check_source_results(
            f"numeric_bucket{bucket}_rows{rows}_prepared{int(prepared)}", rank,
            x=buffer.x[:rows], sf=buffer.x_sf[:rows], ids=ids,
            weights=weights, output=output,
        )
        return {"max_abs": max_abs, "cosine": cosine}

    results: dict[str, dict[str, float]] = {}
    for bucket in bucket_counts:
        for rows in row_counts:
            for prepared in (False, True):
                result = run_case(bucket, rows, prepared)
                key = f"bucket{bucket}_rows{rows}_prepared{int(prepared)}"
                results[key] = result
                if rank == 0:
                    print(
                        "MEGAMOE_NUMERIC_ROW_OK "
                        f"bucket={bucket} rows={rows} "
                        f"prepared={int(prepared)} "
                        f"cosine={result['cosine']:.9f} "
                        f"max_abs={result['max_abs']:.6f}",
                        flush=True,
                    )
                torch.cuda.synchronize()
                dist.barrier()
    return results


def _exact_qwen_fixed_perfect_case(
    rank: int, world_size: int, *, rows: int | None = None,
) -> dict[str, dict[str, Any]]:
    row_counts = _positive_int_list("MEGAMOE_SMOKE_ROWS", "3")
    sm_counts = _positive_int_list(
        "MEGAMOE_SMOKE_SMS",
        str(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
    )
    if len(sm_counts) != 1 or (rows is None and len(row_counts) != 1):
        raise RuntimeError("fixed-perfect MegaMoE case requires one row and SM count")
    rows, num_sms = row_counts[0] if rows is None else rows, sm_counts[0]
    if rows not in row_counts:
        raise RuntimeError("fixed-perfect row is absent from the requested cases")
    warmups = int(os.environ.get("MEGAMOE_SMOKE_WARMUPS", "10"))
    iterations = int(os.environ.get("MEGAMOE_SMOKE_ITERATIONS", "100"))
    bucket = int(os.environ.get("MEGAMOE_SMOKE_BUCKET", str(rows)))
    hidden = int(os.environ.get("MEGAMOE_SMOKE_HIDDEN", "4096"))
    intermediate = int(os.environ.get("MEGAMOE_SMOKE_INTERMEDIATE", "1536"))
    num_experts = int(os.environ.get("MEGAMOE_SMOKE_NUM_EXPERTS", "128"))
    top_k = int(os.environ.get("MEGAMOE_SMOKE_TOP_K", "8"))
    production_routes = int(
        os.environ.get("MEGAMOE_SMOKE_PRODUCTION_ROUTES", "0")
    )
    if warmups < 0 or iterations <= 0 or bucket < rows:
        raise RuntimeError("invalid fixed-perfect MegaMoE timing dimensions")
    if int(os.environ.get("MEGAMOE_SMOKE_DEBUG_TIMINGS", "0")) != 0:
        raise RuntimeError("fixed-perfect MegaMoE profiling forbids kernel timing")
    if num_experts <= 0 or num_experts % world_size:
        raise RuntimeError("fixed-perfect experts must be divisible by EP world size")
    if not 0 < top_k <= num_experts:
        raise RuntimeError("fixed-perfect top-k is out of range")
    if production_routes not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_PRODUCTION_ROUTES must be 0 or 1")

    local_experts = num_experts // world_size
    global_tokens = int(os.environ.get("MEGAMOE_SMOKE_GLOBAL_TOKENS", str(rows * world_size)))
    if global_tokens <= 0 or (global_tokens + world_size - 1) // world_size != rows:
        raise RuntimeError("global token count must match the balanced padded row shape")
    base_rows, extra_rows = divmod(global_tokens, world_size)
    valid_rows = base_rows + int(rank < extra_rows)
    token_offset = rank * base_rows + min(rank, extra_rows)
    valid_count = (torch.tensor([valid_rows], dtype=torch.int64, device="cuda")
                   if extra_rows else None)
    hidden_states = torch.randn(
        (rows, hidden), dtype=torch.bfloat16, device="cuda"
    )
    transformed = _make_or_load_transformed_weights(
        local_experts, hidden, intermediate
    )
    buffer = megamoe_mega.MegaMoESymmBuffer(
        dist.group.WORLD,
        num_experts=num_experts,
        num_max_tokens_per_rank=bucket,
        num_topk=top_k,
        hidden=hidden,
        intermediate_hidden=intermediate,
    )
    flat = torch.arange(rows * top_k, dtype=torch.int64, device="cuda")
    fixed_ids = ((flat + token_offset * top_k) % num_experts).view(rows, top_k)
    balance_ranks = int(os.environ.get("MEGAMOE_SMOKE_BALANCE_RANKS", "0"))
    if balance_ranks not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_BALANCE_RANKS must be 0 or 1")
    if balance_ranks:
        fixed_ids = (fixed_ids % world_size) * local_experts + fixed_ids // world_size
    fixed_ids[valid_rows:] = -1
    ids = buffer.topk_idx[:rows]
    weights = buffer.topk_weights[:rows]
    if production_routes:
        gate_weight = torch.zeros(
            (num_experts, hidden), dtype=torch.bfloat16, device="cuda"
        )
        expert_map = torch.arange(
            num_experts, dtype=torch.int64, device="cuda"
        )
        # The production perfect router starts at rank*valid_rows. Rotate its
        # existing map to the prefix of the balanced, possibly unequal counts.
        expert_map = (expert_map + (token_offset - rank * valid_rows) * top_k) % num_experts
        if balance_ranks:
            expert_map = (expert_map % world_size) * local_experts + expert_map // world_size

        def prepare_input() -> None:
            gate_topk(
                hidden_states,
                gate_weight,
                top_k,
                renormalize=True,
                expert_map=expert_map,
                num_token_non_padded=valid_count,
                topk_idx_dtype=torch.int64,
                quant_out=(buffer.x[:rows], buffer.x_sf[:rows], None),
                out=(ids, weights),
                route_prepare=buffer.route_prepare_args(rank_ready=True),
                perfect_balance=True,
            )
    else:
        fixed_weights = torch.full(
            (rows, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
        )
        _stage(buffer, hidden_states, fixed_ids, fixed_weights)

        def prepare_input() -> None:
            pass

    output = buffer.y[:rows]
    deepgemm.set_num_sms(num_sms)
    rank_gated_combine = True

    def run_megamoe() -> None:
        prepare_input()
        megamoe_mega.fp8_fp8_mega_moe(
            output,
            transformed[0],
            transformed[1],
            buffer,
            routes_prepared=bool(production_routes),
            rank_gated_combine=rank_gated_combine,
            route_ready_dispatch=bool(production_routes),
            rank_ready_route_publish=bool(production_routes),
        )

    dist.barrier()
    output.fill_(float("nan"))
    run_megamoe()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise RuntimeError("fixed-perfect MegaMoE output is non-finite")
    if not torch.equal(ids, fixed_ids):
        raise RuntimeError("fixed-perfect route producer emitted unexpected IDs")
    compare_combine = int(os.environ.get("MEGAMOE_SMOKE_COMPARE_COMBINE", "0"))
    if compare_combine not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_COMPARE_COMBINE must be 0 or 1")
    if compare_combine:
        expected_output = output.clone()
        # The full-barrier reference uses the conventional combine arena.
        # Keep its allocation and generations independent from the selected path.
        selected = buffer, output, ids, weights
        selected_combine = os.environ.get("DG_MEGAMOE_LAMPORT_COMBINE")
        try:
            os.environ["DG_MEGAMOE_LAMPORT_COMBINE"] = "0"
            buffer = megamoe_mega.MegaMoESymmBuffer(
                dist.group.WORLD, num_experts=num_experts,
                num_max_tokens_per_rank=bucket, num_topk=top_k,
                hidden=hidden, intermediate_hidden=intermediate,
            )
            output, ids, weights = buffer.y[:rows], buffer.topk_idx[:rows], buffer.topk_weights[:rows]
            if not production_routes:
                _stage(buffer, hidden_states, fixed_ids, fixed_weights)
            rank_gated_combine = False
            run_megamoe()
            torch.cuda.synchronize()
            if not torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8)):
                raise RuntimeError("rank-gated combine differs from full-barrier output")
        finally:
            buffer, output, ids, weights = selected
            rank_gated_combine = True
            if selected_combine is None:
                os.environ.pop("DG_MEGAMOE_LAMPORT_COMBINE", None)
            else:
                os.environ["DG_MEGAMOE_LAMPORT_COMBINE"] = selected_combine
        dist.barrier()
        print(f"MEGAMOE_COMBINE_EXACT_OK rank={rank} global_tokens={global_tokens}", flush=True)
    _check_source_results(
        f"fixed_bucket{bucket}_rows{rows}_sms{num_sms}", rank,
        hidden=hidden_states, transformed=transformed,
        x=buffer.x[:rows], sf=buffer.x_sf[:rows], ids=ids,
        weights=weights, output=output,
    )
    max_active = _max_active_experts(fixed_ids, local_experts, world_size)
    elapsed, output_max = _time_graph(
        run_megamoe, output, warmups=warmups, iterations=iterations
    )
    return {
        f"bucket{bucket}_rows{rows}_sms{num_sms}": {
            "ep_size": world_size,
            "model_dimensions": {"hidden": hidden, "intermediate": intermediate,
                                 "experts": num_experts, "top_k": top_k},
            "rows_per_rank": rows,
            "global_tokens": global_tokens,
            "valid_rows_per_rank": [base_rows + int(r < extra_rows) for r in range(world_size)],
            "combine_exact_checked": bool(compare_combine),
            "payload_mode": getattr(buffer, "payload_mode", None),
            "timing_warmups": warmups,
            "timing_iterations": iterations,
            "timing_graph_batch": int(os.environ.get("MEGAMOE_SMOKE_GRAPH_BATCH", "1")),
            "timing_rank_reduction": "max",
            "timing_cuda_profiler_range": int(os.environ.get("MEGAMOE_SMOKE_CUDA_PROFILER_RANGE", "0")),
            "milliseconds": elapsed,
            "output_max": output_max,
            "max_active_experts": max_active,
            "timed_scope": (
                "route_producer_and_megamoe"
                if production_routes
                else "megamoe_kernel_only"
            ),
            "routing": (
                "fixed_global_sequence_rank_interleaved_balance"
                if balance_ranks else "fixed_global_sequence_perfect_balance"
            ),
            "routes_prepared": bool(production_routes),
        }
    }


def _exact_qwen_case(
    rank: int, world_size: int
) -> dict[str, dict[str, Any]]:
    row_counts = _positive_int_list("MEGAMOE_SMOKE_ROWS", "3")
    sm_counts = _positive_int_list(
        "MEGAMOE_SMOKE_SMS",
        str(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
    )
    warmups = int(os.environ.get("MEGAMOE_SMOKE_WARMUPS", "10"))
    iterations = int(os.environ.get("MEGAMOE_SMOKE_ITERATIONS", "100"))
    bucket = int(os.environ.get("MEGAMOE_SMOKE_BUCKET", str(max(row_counts))))
    contend_cycles = int(os.environ.get("MEGAMOE_SMOKE_CONTEND_CYCLES", "0"))
    debug_timings = int(os.environ.get("MEGAMOE_SMOKE_DEBUG_TIMINGS", "0"))
    route_prepare_modes = _nonnegative_int_list(
        "MEGAMOE_SMOKE_ROUTE_PREPARE_MODES", "0"
    )
    combine_modes = _nonnegative_int_list(
        "MEGAMOE_SMOKE_COMBINE_MODES", "0"
    )
    route_ready_dispatch_modes = _nonnegative_int_list(
        "MEGAMOE_SMOKE_ROUTE_READY_DISPATCH_MODES", "0"
    )
    paired_modes_text = os.environ.get(
        "MEGAMOE_SMOKE_SPECIALIZATION_MODES", ""
    ).strip()
    if warmups < 0 or iterations <= 0:
        raise RuntimeError("MegaMoE smoke warmups must be nonnegative and iterations positive")
    if contend_cycles < 0:
        raise RuntimeError("MEGAMOE_SMOKE_CONTEND_CYCLES must be nonnegative")
    if debug_timings not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_DEBUG_TIMINGS must be 0 or 1")
    if any(mode not in (0, 1) for mode in route_prepare_modes):
        raise RuntimeError("MEGAMOE_SMOKE_ROUTE_PREPARE_MODES only supports 0 and 1")
    if any(mode not in (0, 1) for mode in combine_modes):
        raise RuntimeError("MEGAMOE_SMOKE_COMBINE_MODES only supports 0 and 1")
    if any(mode not in (0, 1) for mode in route_ready_dispatch_modes):
        raise RuntimeError(
            "MEGAMOE_SMOKE_ROUTE_READY_DISPATCH_MODES only supports 0 and 1"
        )
    if paired_modes_text:
        specialization_modes: list[tuple[int, int, int, int]] = []
        for entry in paired_modes_text.split(","):
            fields = entry.split(":")
            if len(fields) not in (2, 3, 4):
                raise RuntimeError(
                    "MEGAMOE_SMOKE_SPECIALIZATION_MODES entries must be "
                    "route:combine[:dispatch[:rank-ready]] tuples"
                )
            mode = (
                int(fields[0]),
                int(fields[1]),
                int(fields[2]) if len(fields) >= 3 else 0,
                int(fields[3]) if len(fields) == 4 else 0,
            )
            if any(value not in (0, 1) for value in mode):
                raise RuntimeError(
                    "MEGAMOE_SMOKE_SPECIALIZATION_MODES only supports 0/1 "
                    "route, combine, dispatch, and rank-ready fields"
                )
            if mode[2] and not mode[0]:
                raise RuntimeError(
                    "route-ready dispatch requires prepared routes"
                )
            if mode[3] and (not mode[0] or not mode[2]):
                raise RuntimeError(
                    "rank-ready publication requires prepared, route-ready dispatch"
                )
            specialization_modes.append(mode)
    else:
        specialization_modes = [
            (route_mode, combine_mode, dispatch_mode, 0)
            for route_mode in route_prepare_modes
            for combine_mode in combine_modes
            for dispatch_mode in route_ready_dispatch_modes
            if not dispatch_mode or route_mode
        ]
    if bucket < max(row_counts):
        raise RuntimeError(
            f"MEGAMOE_SMOKE_BUCKET={bucket} is smaller than rows={max(row_counts)}"
        )
    if debug_timings and (len(row_counts) != 1 or len(sm_counts) != 1):
        raise RuntimeError(
            "MEGAMOE_SMOKE_DEBUG_TIMINGS requires exactly one row and SM configuration"
        )

    hidden = int(os.environ.get("MEGAMOE_SMOKE_HIDDEN", "4096"))
    intermediate = int(os.environ.get("MEGAMOE_SMOKE_INTERMEDIATE", "1536"))
    num_experts = int(os.environ.get("MEGAMOE_SMOKE_NUM_EXPERTS", "128"))
    top_k = int(os.environ.get("MEGAMOE_SMOKE_TOP_K", "8"))
    perfect_routing = int(os.environ.get("MEGAMOE_SMOKE_PERFECT_ROUTING", "0"))
    if perfect_routing not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_PERFECT_ROUTING must be 0 or 1")
    if perfect_routing and any(not mode[0] for mode in specialization_modes):
        raise RuntimeError("perfect routing requires route-prepared specialization modes")
    if num_experts <= 0 or num_experts % world_size:
        raise RuntimeError(
            "MEGAMOE_SMOKE_NUM_EXPERTS must be positive and divisible by "
            f"world_size={world_size}, got {num_experts}"
        )
    if not 0 < top_k <= num_experts:
        raise RuntimeError(
            f"MEGAMOE_SMOKE_TOP_K must be in [1, {num_experts}], got {top_k}"
        )
    probe_hidden = torch.ones((1, hidden), dtype=torch.bfloat16, device="cuda")
    probe_gate = (
        torch.exp2(
            torch.arange(num_experts, dtype=torch.float32, device="cuda") / 64
        )
        .to(torch.bfloat16)[:, None]
        .expand(num_experts, hidden)
        .contiguous()
    )
    probe_ids, _ = gate_topk(probe_hidden, probe_gate, top_k, renormalize=True)
    expected_probe_ids = torch.arange(
        num_experts - 1,
        num_experts - top_k - 1,
        -1,
        dtype=probe_ids.dtype,
        device=probe_ids.device,
    )[None]
    if not torch.equal(probe_ids, expected_probe_ids):
        raise RuntimeError(
            "normal fused router did not select the controlled highest experts: "
            f"got={probe_ids.tolist()} expected={expected_probe_ids.tolist()}"
        )
    del probe_hidden, probe_gate, probe_ids, expected_probe_ids
    local_experts = num_experts // world_size
    transformed = _make_or_load_transformed_weights(
        local_experts, hidden, intermediate
    )
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
    _check_source_results(
        "exact_inputs", rank, hidden=hidden_states,
        gate=gate_weight, transformed=transformed,
    )
    results: dict[str, dict[str, Any]] = {}

    for rows in row_counts:
        hidden_views = (
            hidden_states[:rows],
            -hidden_states[:rows],
        )
        hidden_view = hidden_views[0]
        ids = buffer.topk_idx[:rows]
        weights = buffer.topk_weights[:rows]
        output = buffer.y[:rows]

        route_prepare_mode = 0
        rank_gated_combine_mode = 0
        route_ready_dispatch_mode = 0

        rank_ready_route_publish_mode = 0

        validity = None

        def prepare_input() -> None:
            gate_topk(
                hidden_view,
                gate_weight,
                top_k,
                renormalize=True,
                num_token_non_padded=validity,
                expert_map=expert_map,
                topk_idx_dtype=torch.int64,
                quant_out=(buffer.x[:rows], buffer.x_sf[:rows], None),
                out=(ids, weights),
                route_prepare=(
                    buffer.route_prepare_args(
                        rank_ready=bool(rank_ready_route_publish_mode)
                    ) if route_prepare_mode else None
                ),
                perfect_balance=bool(perfect_routing),
            )

        def run_stage() -> None:
            prepare_input()
            megamoe_mega.fp8_fp8_mega_moe(
                output,
                transformed[0],
                transformed[1],
                buffer,
                routes_prepared=bool(route_prepare_mode),
                rank_gated_combine=bool(rank_gated_combine_mode),
                route_ready_dispatch=bool(route_ready_dispatch_mode),
                rank_ready_route_publish=bool(
                    rank_ready_route_publish_mode
                ),
            )

        reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        reference_alt: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for num_sms in sm_counts:
            deepgemm.set_num_sms(num_sms)
            for specialization_trial, (
                route_prepare_mode,
                rank_gated_combine_mode,
                route_ready_dispatch_mode,
                rank_ready_route_publish_mode,
            ) in enumerate(specialization_modes):
                output.fill_(float("nan"))
                run_stage()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        "MegaMoE specialization did not overwrite every output "
                        f"for rows={rows} sms={num_sms}"
                    )
                current = (ids.clone(), weights.clone(), output.clone())
                result_case = (
                    f"exact_bucket{bucket}_rows{rows}_sms{num_sms}"
                    f"_trial{specialization_trial}"
                )
                _check_source_results(
                    result_case, rank, x=buffer.x[:rows],
                    sf=buffer.x_sf[:rows], result=current,
                )
                if perfect_routing:
                    expected_ids = (
                        torch.arange(
                            rows * top_k, dtype=ids.dtype, device=ids.device
                        ).view(rows, top_k)
                        + rank * rows * top_k
                    ) % num_experts
                    if not torch.equal(current[0], expected_ids):
                        raise RuntimeError(
                            "perfect-routing IDs do not match the deterministic "
                            f"global sequence on rank={rank}"
                        )
                if reference is None:
                    reference = current
                elif not all(
                    torch.equal(expected, actual)
                    for expected, actual in zip(reference, current)
                ):
                    raise RuntimeError(
                        "MegaMoE specialization is not bitwise equal to control "
                        f"for rows={rows} sms={num_sms} "
                        f"route_mode={route_prepare_mode} "
                        f"combine_mode={rank_gated_combine_mode} "
                        f"dispatch_mode={route_ready_dispatch_mode}"
                        f" rank_ready_mode={rank_ready_route_publish_mode}"
                    )
                hidden_view = hidden_views[1]
                output.fill_(float("nan"))
                run_stage()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        "MegaMoE specialization did not overwrite every output "
                        f"for alternate input rows={rows} sms={num_sms}"
                    )
                current_alt = (ids.clone(), weights.clone(), output.clone())
                _check_source_results(
                    result_case + "_alternate", rank, x=buffer.x[:rows],
                    sf=buffer.x_sf[:rows], result=current_alt,
                )
                if reference_alt is None:
                    reference_alt = current_alt
                elif not all(
                    torch.equal(expected, actual)
                    for expected, actual in zip(reference_alt, current_alt)
                ):
                    raise RuntimeError(
                        "MegaMoE specialization is not bitwise equal to control "
                        f"for alternate input rows={rows} sms={num_sms} "
                        f"route_mode={route_prepare_mode} "
                        f"combine_mode={rank_gated_combine_mode} "
                        f"dispatch_mode={route_ready_dispatch_mode}"
                        f" rank_ready_mode={rank_ready_route_publish_mode}"
                    )
                # Exercise arbitrary and empty-destination routing even when
                # the timed case deliberately uses perfect balancing.
                saved_perfect_routing = perfect_routing
                perfect_routing = 0
                for phase, hidden_view in enumerate(
                    (*hidden_views, torch.zeros_like(hidden_views[0]))
                ):
                    output.fill_(float("nan"))
                    run_stage()
                    torch.cuda.synchronize()
                    if (not torch.isfinite(output).all()
                            or (phase == 2 and torch.count_nonzero(output))):
                        raise RuntimeError(
                            f"normal/zero route output mismatch: rows={rows} phase={phase}"
                        )
                    _check_source_results(
                        result_case + f"_normal{phase}", rank, x=buffer.x[:rows],
                        sf=buffer.x_sf[:rows], ids=ids, weights=weights, output=output,
                    )
                hidden_view = hidden_views[0]
                prefix = rows // 2 if rank % 2 else 0
                row_map = []
                compact = 0
                for row in range(rows):
                    real = (row + rank) % 3 != 0
                    row_map.append(compact if real else -1)
                    compact += int(real)
                for phase, metadata in enumerate(([prefix], [compact, *row_map])):
                    validity = torch.tensor(metadata, dtype=torch.int64, device="cuda")
                    output.fill_(float("nan"))
                    run_stage()
                    torch.cuda.synchronize()
                    padding = (torch.arange(rows, device="cuda") >= prefix if phase == 0
                               else validity[1:] < 0)
                    if (not torch.isfinite(output).all() or torch.count_nonzero(output[padding])
                            or (ids[padding] >= 0).any() or torch.count_nonzero(weights[padding])):
                        raise RuntimeError(f"padding/empty-rank output mismatch: rows={rows} phase={phase}")
                    _check_source_results(
                        result_case + f"_padding{phase}", rank, ids=ids, weights=weights, output=output,
                    )
                validity = None
                perfect_routing = saved_perfect_routing
                hidden_view = hidden_views[0]
                max_active = _max_active_experts(current[0], local_experts, world_size)
                elapsed, output_max = _time_graph(
                    run_stage, output, warmups=warmups, iterations=iterations
                )
                key = f"bucket{bucket}_rows{rows}_sms{num_sms}"
                if len(specialization_modes) > 1:
                    key += (
                        f"_routes{route_prepare_mode}"
                        f"_combine{rank_gated_combine_mode}"
                        f"_dispatch{route_ready_dispatch_mode}"
                        f"_trial{specialization_trial}"
                        f"_rankready{rank_ready_route_publish_mode}"
                    )
                results[key] = {
                    "milliseconds": elapsed,
                    "output_max": output_max,
                    "max_active_experts": max_active,
                    "timed_scope": "router_and_megamoe",
                    "calls_per_graph": int(os.environ.get("MEGAMOE_SMOKE_GRAPH_BATCH", "1")),
                }
                if debug_timings:
                    timing_tensor = torch.zeros(
                        len(_MEGAMOE_TIMING_SLOTS), dtype=torch.int64, device="cuda"
                    )
                    # Compile the timing specialization before the synchronized
                    # measurement so one rank's JIT cannot become another rank's
                    # in-kernel communication wait.
                    prepare_input()
                    megamoe_mega.fp8_fp8_mega_moe(
                        output,
                        transformed[0],
                        transformed[1],
                        buffer,
                        debug_timings=timing_tensor,
                        routes_prepared=bool(route_prepare_mode),
                        rank_gated_combine=bool(rank_gated_combine_mode),
                        route_ready_dispatch=bool(route_ready_dispatch_mode),
                        rank_ready_route_publish=bool(
                            rank_ready_route_publish_mode
                        ),
                    )
                    torch.cuda.synchronize()
                    dist.barrier()
                    timing_tensor.zero_()
                    torch.cuda.synchronize()
                    dist.barrier()
                    prepare_input()
                    megamoe_mega.fp8_fp8_mega_moe(
                        output,
                        transformed[0],
                        transformed[1],
                        buffer,
                        debug_timings=timing_tensor,
                        routes_prepared=bool(route_prepare_mode),
                        rank_gated_combine=bool(rank_gated_combine_mode),
                        route_ready_dispatch=bool(route_ready_dispatch_mode),
                        rank_ready_route_publish=bool(
                            rank_ready_route_publish_mode
                        ),
                    )
                    torch.cuda.synchronize()
                    dist.barrier()
                    local_timings = dict(
                        zip(
                            _MEGAMOE_TIMING_SLOTS,
                            (int(value) for value in timing_tensor.cpu()),
                        )
                    )
                    timings_by_rank: list[dict[str, int] | None] = [None] * world_size
                    dist.all_gather_object(timings_by_rank, local_timings)
                    results[key]["debug_timing_cycles_by_rank"] = timings_by_rank
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
            routes_per_owner = (fixed_rows * top_k) // world_size
            fixed_ids = (flat % world_size) * local_experts + (
                rank * routes_per_owner + flat // world_size
            ) % local_experts
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
    if world_size not in (2, 4, 8, 16, 32, 64):
        raise RuntimeError(
            "same-rank MegaMoE smoke supports EP2/4/8/16/32/64 topology "
            f"checks, got {world_size}"
        )
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
    profile_only = int(os.environ.get("MEGAMOE_SMOKE_PROFILE_ONLY", "0"))
    if profile_only not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_PROFILE_ONLY must be 0 or 1")
    fixed_perfect_only = int(
        os.environ.get("MEGAMOE_SMOKE_FIXED_PERFECT_ONLY", "0")
    )
    if fixed_perfect_only not in (0, 1):
        raise RuntimeError("MEGAMOE_SMOKE_FIXED_PERFECT_ONLY must be 0 or 1")
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
    numeric = None
    numeric_rows = None
    if not profile_only:
        _check_fused_quant_coverage()
        numeric = _numeric_case(rank, world_size)
        dist.barrier()
        numeric_rows = _numeric_row_sweep(rank, world_size)
        dist.barrier()
    if fixed_perfect_only:
        row_counts = _positive_int_list("MEGAMOE_SMOKE_ROWS", "3")
        if len(set(row_counts)) != len(row_counts):
            raise RuntimeError("fixed-perfect cases must have distinct row counts")
        if len(row_counts) > 1 and "MEGAMOE_SMOKE_GLOBAL_TOKENS" in os.environ:
            raise RuntimeError("multi-row fixed cases derive global tokens from rows and EP")
        exact = {}
        for rows in row_counts:
            # Match standalone case inputs regardless of case order or prior RNG use.
            if len(row_counts) > 1:
                torch.manual_seed(20260825 + rank)
            exact.update(_exact_qwen_fixed_perfect_case(rank, world_size, rows=rows))
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                print(f"MEGAMOE_FIXED_CASE_OK rows={rows}", flush=True)
    else:
        exact = _exact_qwen_case(rank, world_size)
    if rank == 0:
        payload = {
            "status": "ok",
            "expert_weight_dtype": expert_weight_dtype,
            "numeric": numeric,
            "numeric_rows": numeric_rows,
            "exact_qwen": exact,
        }
        output_path = os.environ.get("MEGAMOE_SMOKE_OUTPUT")
        if output_path:
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
        print(json.dumps(payload, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
