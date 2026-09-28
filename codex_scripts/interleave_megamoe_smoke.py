#!/usr/bin/env python3
"""Run the existing MegaMoE smoke with two interleaved prepared CUDA graphs.

Kernel sources and smoke correctness cases stay in FASTAFD_SOURCE_REPO. Only
the prepared timing function changes. Variants must share the buffer layout.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys


def timing_schedule(labels, iterations):
    if len(labels) != 2 or len(set(labels)) != 2 or iterations <= 0 or iterations % 2:
        raise ValueError("Require two distinct arms and an even positive call count")
    a, b = labels
    return [a, b, b, a] * (iterations // 2)


def time_interleaved(run_stage, prepare_stage, output, *, warmups, iterations):
    import torch
    import torch.distributed as dist
    from collect_cuda_rank_timings import aggregate_rank_timings, make_timing_event_pairs

    variants = json.loads(os.environ["MEGAMOE_INTERLEAVED_VARIANTS"])
    entry_protocol = os.environ.get("MEGAMOE_INTERLEAVED_ENTRY", "synchronized")
    if entry_protocol not in {"synchronized", "queued"}:
        raise ValueError("Interleaved entry must be synchronized or queued")
    if (not isinstance(variants, dict) or len(variants) != 2
            or any(not isinstance(v, dict) or not v for v in variants.values())):
        raise ValueError("Variants must be two nonempty environment dictionaries")
    labels = list(variants)
    schedule = timing_schedule(labels, iterations)
    keys = set(variants[labels[0]])
    if (any(set(v) != keys for v in variants.values())
            or any(not k.startswith("DG_MEGAMOE_ALL_PUSH_") for k in keys)
            or any(not isinstance(value, str) for v in variants.values() for value in v.values())):
        raise ValueError("Both arms require the same DG_MEGAMOE_ALL_PUSH_ keys and string values")
    if (os.environ.get("MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT") != "joined"
            or os.environ.get("MEGAMOE_SMOKE_GRAPH_BATCH", "1") != "1"
            or os.environ.get("MEGAMOE_SMOKE_CUDA_PROFILER_RANGE", "0") != "0"
            or os.environ.get("MINISGL_MEGAMOE_PAYLOAD_MODE") != "all_push"
            or warmups <= 0):
        raise ValueError("Interleaving requires ordinary joined all-push graphs, batch=1 and warmups")
    configs = [None] * dist.get_world_size()
    dist.all_gather_object(configs, (list(variants.items()), iterations, warmups, entry_protocol))
    if any(config != configs[0] for config in configs):
        raise ValueError("Ranks disagree on variants or replay schedule")

    saved = {key: os.environ.get(key) for key in keys}
    graphs, reference = {}, None
    try:
        for label, env in variants.items():
            os.environ.update(env)
            prepare_stage()
            run_stage()
            torch.cuda.synchronize()
            if reference is None:
                reference = output.detach().clone()
            elif not torch.equal(output.view(torch.uint8), reference.view(torch.uint8)):
                raise RuntimeError(f"Interleaved eager output differs: {label}")
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                prepare_stage()
                run_stage()
            graphs[label] = graph
    finally:
        for key, value in saved.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value

    def check_graphs():
        for label, graph in graphs.items():
            graph.replay()
            torch.cuda.synchronize()
            if (not torch.isfinite(output).all()
                    or not torch.equal(output.view(torch.uint8), reference.view(torch.uint8))):
                raise RuntimeError(f"Interleaved graph output differs: {label}")

    check_graphs()
    pairs = make_timing_event_pairs(len(schedule), device=output.device)
    if entry_protocol == "queued":
        dist.barrier()
    for _ in range(warmups):
        for graph in graphs.values():
            graph.replay()
    if entry_protocol == "synchronized":
        torch.cuda.synchronize()
        dist.barrier()
    for label, (start, end) in zip(schedule, pairs):
        start.record()
        graphs[label].replay()
        end.record()
    pairs[-1][1].synchronize()
    durations = [start.elapsed_time(end) for start, end in pairs]
    check_graphs()
    print(f"MEGAMOE_INTERLEAVED_EXACT_OK rank={dist.get_rank()} calls_per_arm={iterations}", flush=True)
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, dict(rank=dist.get_rank(), durations_ms=durations))
    scope = "Interleaved joined graph: preparation barrier/reset, producer, selector and MegaMoE"
    arms = {label: aggregate_rank_timings([
        dict(rank=r["rank"], scope=scope,
             durations_ms=[v for arm, v in zip(schedule, r["durations_ms"]) if arm == label],
             metadata=dict(variant=variants[label])) for r in records]) for label in labels}
    if dist.get_rank() == 0:
        path = Path(os.environ["MEGAMOE_INTERLEAVED_OUTPUT"])
        with path.open("x") as stream:
            json.dump(dict(status="PASS_INTERLEAVED_EXACT", variants=variants, schedule=schedule,
                           calls_per_arm=iterations, warmups_per_arm=warmups,
                           entry_protocol=entry_protocol, scalar_result_arm=labels[0], arms=arms), stream)
            stream.write("\n")
    return arms[labels[0]]["maximum_rank_mean_ms"], float(output.abs().max())


def main():
    source = Path(os.environ["FASTAFD_SOURCE_REPO"])
    sys.path.insert(0, str(source / "codex_scripts"))
    spec = importlib.util.spec_from_file_location("megamoe_smoke", source / "tests/afd_megamoe_same_rank_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    smoke._time_prepared_graph = time_interleaved
    smoke.main()


if __name__ == "__main__":
    main()
