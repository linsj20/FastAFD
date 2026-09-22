"""Distributed adapter checks: padding, skew and double-buffered graph replay.

Run with the maintained same-rank sbatch launcher and FP4 weights. The reference
is a pull invocation with identical inputs and transformed weights.
"""

import os
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.distributed as dist

from afd_megamoe_same_rank_smoke import _make_or_load_transformed_weights
from minisgl.cuda_graph_utils import capture_cuda_graph
from minisgl.kernel import deepgemm
from minisgl.moe.megamoe_afd import MegaMoEAfdAdapter


def benchmark_chain(adapters, gate, norm, engine, peer, *, iterations):
    """Time actual recycling and norm in an ordinary, two-lane graph."""
    from flashinfer import fused_add_rmsnorm

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codex_scripts"))
    from collect_cuda_rank_timings import collect_rank_timings, make_timing_event_pairs

    rows = int(os.environ.get("MEGAMOE_ADAPTER_BENCH_ROWS", "6"))
    warmups = int(os.environ.get("MEGAMOE_SMOKE_WARMUPS", "10"))
    if not 0 < rows <= adapters[1].all_push_reset_rows or warmups < 0:
        raise ValueError("Benchmark rows must fit the validated reset prefix")
    input_seed = 20260920 + dist.get_rank()
    generator = torch.Generator(device=gate.device).manual_seed(input_seed)
    x = torch.randn(rows, gate.size(1), dtype=gate.dtype, device=gate.device,
                    generator=generator) * 0.1
    residuals = [torch.full_like(x, 0.125) for _ in range(4)]
    outputs = [None] * 4
    pairs = make_timing_event_pairs(iterations, device=x.device)
    # Reset only the norm's input fixture outside timing. Buffer maintenance is
    # always part of the measured graph, including every replay and bank reuse.
    with torch.cuda.stream(engine):
        preparation = torch.cuda.CUDAGraph()
        with torch.cuda.graph(preparation, stream=engine):
            for residual in residuals:
                residual.fill_(0.125)
    reference = None
    profile = bool(os.environ.get("MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX"))
    profile_graphs = []
    for trial, mode in enumerate(("pull", "all_push", "all_push", "pull")):
        adapter = adapters[0 if mode == "pull" else 1]
        adapter.perfect_routing = True
        adapter.all_push_reset_rows = rows

        def body():
            previous = None
            for layer in range(2):
                for lane, stream in enumerate((engine, peer)):
                    if previous is not None:
                        stream.wait_stream(previous)
                    with torch.cuda.stream(stream):
                        index = 2 * layer + lane
                        output = adapter.forward(
                            layer, x, gate, lane=lane, num_token_non_padded=None)
                        if mode == "all_push":
                            assert adapter.normalize_output(norm, output, residuals[index])
                        else:
                            fused_add_rmsnorm(output, residuals[index], norm.weight, norm.eps)
                        outputs[index] = output
                    previous = stream

        with torch.cuda.stream(engine):
            preparation.replay()
            body()  # Materialize device specializations before capture.
        torch.cuda.synchronize()
        dist.barrier()
        graph, _ = capture_cuda_graph(
            device=x.device, engine_stream=engine, comm_stream=peer,
            overlap_comm=True, pool=None, fn=body,
        )
        with torch.cuda.stream(engine):
            for _ in range(warmups):
                preparation.replay()
                graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.stream(engine):
            for start, end in pairs:
                preparation.replay()
                start.record(engine)
                graph.replay()
                end.record(engine)
        pairs[-1][1].synchronize()
        actual = [value.clone() for value in (*outputs, *residuals)]
        if reference is None:
            reference = actual
        if any(not torch.isfinite(a).all() or not torch.equal(a, b)
               for a, b in zip(actual, reference)):
            raise AssertionError(f"Benchmark replay differs from pull: trial={trial} mode={mode}")
        collect_rank_timings(
            [start.elapsed_time(end) for start, end in pairs],
            Path(os.environ["OUTPUT_ROOT"]) / "chain_timings" / f"{trial}-{mode}.json",
            scope="four router->MegaMoE->norm calls, two lanes/two layers; reset included",
            metadata={"mode": mode, "trial": trial, "rows": rows, "calls_per_graph": 4,
                      "perfect_routing": True, "reset_rows": rows,
                      "input_seed": input_seed,
                      "allocated_capacity": adapter.buffers[0].num_max_tokens_per_rank},
        )
        if profile and trial < 2:
            profile_graphs.append(graph)
        else:
            del graph
    if profile:
        # A separate capture keeps profiler overhead out of ordinary samples.
        # These are the already-validated pull and all-push graphs, in that order.
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()
        with torch.cuda.stream(engine):
            for graph in profile_graphs:
                for _ in range(10):
                    preparation.replay()
                    graph.replay()
        engine.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def main():
    rank, ranks = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo", rank=rank, world_size=ranks,
                            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    hidden = int(os.environ.get("MEGAMOE_SMOKE_HIDDEN", "4096"))
    intermediate = int(os.environ.get("MEGAMOE_SMOKE_INTERMEDIATE", "1536"))
    experts = int(os.environ.get("MEGAMOE_SMOKE_NUM_EXPERTS", "128"))
    topk = int(os.environ.get("MEGAMOE_SMOKE_TOP_K", "8"))
    deepgemm.set_num_sms(int(os.environ.get("MEGAMOE_SMOKE_SMS", "150")))
    torch.manual_seed(20260919 + rank)
    transformed = _make_or_load_transformed_weights(experts // ranks, hidden, intermediate)
    reset_rows = int(os.environ.get("MEGAMOE_ADAPTER_RESET_ROWS", "33"))
    adapters = []
    for mode in ("pull", "all_push"):
        os.environ["MINISGL_MEGAMOE_PAYLOAD_MODE"] = mode
        adapter = MegaMoEAfdAdapter(
            group=dist.group.WORLD, num_experts=experts, num_local_experts=experts // ranks,
            hidden_size=hidden, intermediate_size=intermediate, num_layers=2,
            top_k=topk, num_max_tokens_per_rank=int(os.environ.get("MEGAMOE_SMOKE_BUCKET", "96")), num_lanes=2,
            num_max_decode_tokens_per_rank=reset_rows,
            tp_size=1, tp_rank=0, gate_renormalize=True,
        )
        adapter.weights = {0: transformed, 1: transformed}
        adapter.allocate_buffers()
        adapters.append(adapter)
    control, candidate = adapters
    engine = torch.cuda.Stream()
    peer = torch.cuda.Stream()
    lanes = (engine, peer)
    gate = torch.randn(experts, hidden, dtype=torch.bfloat16, device=device) * 0.01
    # All ranks use the same gate; the skew fixture targets rank zero's experts.
    dist.broadcast(gate, src=0)
    norm = SimpleNamespace(weight=torch.randn(hidden, dtype=torch.bfloat16, device=device), eps=1e-6)
    from flashinfer import fused_add_rmsnorm
    diagnostic = False
    trace_buffers = {}
    checks = 0
    fixtures = (("normal", 6), ("empty_rank", 6), ("row_map", 7),
                ("skew", 8), ("shape_reuse", 33), ("small_after_large", 6))
    selected = os.environ.get("MEGAMOE_ADAPTER_CASES", "").split(",")
    if selected == [""]:
        selected = [name for name, _ in fixtures]
    if not set(selected) <= {name for name, _ in fixtures} or len(set(selected)) != len(selected):
        raise ValueError("Unknown or duplicate adapter test case")
    for label, rows in fixtures:
        if label not in selected:
            continue
        x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device) * 0.1
        if label == "empty_rank":
            validity = torch.tensor([rows if rank % 2 == 0 else 0], dtype=torch.int64, device=device)
        elif label == "row_map":
            valid = (torch.arange(rows, device=device) + rank) % 3 != 0
            indices = valid.to(torch.int64).cumsum(0) - 1
            validity = torch.cat((valid.sum().reshape(1), torch.where(valid, indices, -1)))
        else:
            validity = None
        case_gate = gate
        if label == "skew":
            x.fill_(0.125)
            case_gate = torch.zeros_like(gate)
            case_gate[:topk].fill_(0.01)

        def reference():
            engine.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(engine):
                result = control.forward(0, x, case_gate, lane=0,
                                         num_token_non_padded=validity).clone()
            with torch.cuda.stream(engine):
                raw = result.clone()
                norm_residual = torch.full_like(result, 0.125)
                fused_add_rmsnorm(result, norm_residual, norm.weight, norm.eps)
            torch.cuda.synchronize()
            dist.barrier()
            return raw, result, norm_residual

        outputs = {}
        residuals = {(lane, step): torch.full_like(x, 0.125)
                     for lane in range(2) for step in range(3)}

        def body():
            previous = None
            for step, layer in enumerate((0, 1, 0)):
                for lane, stream in enumerate(lanes):
                    # Match the model's FFN compute handoff. Independent
                    # persistent kernels can occupy peers in opposite orders.
                    if previous is not None:
                        stream.wait_stream(previous)
                    with torch.cuda.stream(stream):
                        residuals[lane, step].fill_(0.125)
                        result = candidate.forward(
                            layer, x, case_gate, lane=lane,
                            num_token_non_padded=validity,
                            debug_timings=trace_buffers[lane, step] if diagnostic else None,
                        )
                        # The real next consumer must overlap reset. Copying y
                        # first would insert a full dependency and mask the race.
                        assert candidate.normalize_output(
                            norm, result, residuals[lane, step],
                            debug_timings=trace_buffers[lane, step] if diagnostic else None,
                        )
                        outputs[lane, step] = result.clone()
                    previous = stream

        def check(expected, phase):
            nonlocal checks
            torch.cuda.synchronize()
            for key, value in outputs.items():
                for actual, target in ((value, expected[1]), (residuals[key], expected[2])):
                    if not torch.isfinite(actual).all() or not torch.equal(actual, target):
                        delta = (actual.float() - target.float()).abs()
                        raise AssertionError((rank, label, phase, key, delta.max().item(),
                                              int((actual != target).sum().item())))
                    checks += 1
            dist.barrier()
            print(f"MEGAMOE_ADAPTER_EXACT_OK rank={rank} case={label} phase={phase}", flush=True)

        expected = reference()
        with torch.cuda.stream(engine):
            raw = candidate.forward(0, x, case_gate, lane=0,
                                    num_token_non_padded=validity).clone()
        torch.cuda.synchronize()
        assert torch.equal(raw, expected[0]), (rank, label, "raw output")
        checks += 1
        print(f"MEGAMOE_ADAPTER_EXACT_OK rank={rank} case={label} phase=raw", flush=True)
        with torch.cuda.stream(engine):
            body()
        check(expected, "eager")
        graph, _ = capture_cuda_graph(
            device=device, engine_stream=engine, comm_stream=peer,
            overlap_comm=True, pool=None, fn=body,
        )
        for replay in range(3):
            # Fresh values at the same captured addresses expose stale-bank reads.
            if replay:
                x.mul_(0.75)
                expected = reference()
            profile = label == "normal" and replay == 1 and not int(
                os.environ.get("MEGAMOE_ADAPTER_BENCH_ITERATIONS", "0")) and bool(
                os.environ.get("MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX"))
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.stream(engine):
                graph.replay()
            check(expected, f"replay{replay}")
            if profile:
                torch.cuda.cudart().cudaProfilerStop()
        del graph
    # One separately instrumented graph measures useful norm work directly.
    # Nsight kernel end times may include dependency retirement after that work.
    diagnostic = True
    trace_buffers = {(lane, step): torch.zeros(24 + 3 * rows + int(os.environ.get("MEGAMOE_SMOKE_SMS", "150")) + 16, dtype=torch.int64, device=device)
                     for lane in range(2) for step in range(3)}
    with torch.cuda.stream(engine):
        body()
    check(expected, "diagnostic_eager")
    graph, _ = capture_cuda_graph(
        device=device, engine_stream=engine, comm_stream=peer,
        overlap_comm=True, pool=None, fn=body,
    )
    with torch.cuda.stream(engine):
        graph.replay()
    check(expected, "diagnostic_replay")
    stamps = {f"{lane}:{step}": {"norm_rows": rows, "hidden": hidden, "stamps": values[21:-16].tolist(), "reset_worker_stamps": values[-16:].tolist()}
              for (lane, step), values in trace_buffers.items()}
    if candidate.recycle_ahead:
        for (lane, step), values in trace_buffers.items():
            stamps[f"{lane}:{step}"]["reset_ahead"] = values[:21].tolist()
    timing_root = Path(os.environ["OUTPUT_ROOT"]) / "reset_timings"
    timing_root.mkdir(exist_ok=True)
    timing_path = timing_root / f"rank-{rank}.json"
    timing_path.write_text(json.dumps({"rank": rank, "calls": stamps}) + "\n")
    print(f"MEGAMOE_RECYCLE_TIMING_FILE rank={rank} path={timing_path}", flush=True)
    del graph
    print(f"MEGAMOE_ADAPTER_PASS rank={rank} comparisons={checks} "
          f"requested_capacity={candidate.all_push_capacity} "
          f"allocated_capacity={candidate.buffers[0].num_max_tokens_per_rank}", flush=True)
    iterations = int(os.environ.get("MEGAMOE_ADAPTER_BENCH_ITERATIONS", "0"))
    if iterations < 0:
        raise ValueError("Benchmark iterations must be nonnegative")
    if iterations:
        benchmark_chain(adapters, gate, norm, engine, peer, iterations=iterations)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
