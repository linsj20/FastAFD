#!/usr/bin/env python3
"""Bound reset overlap using one clock-offset interval for a same-device capture.

Every device marker must lie inside its owning Nsight kernel. Intersect all such
constraints without fitting an offset or selecting samples. Kernel execution
overlap does not establish instruction-level overlap or absence of contention.
"""

import argparse
import json
from pathlib import Path

from analyze_afd_attention_gap import graphs
from summarize_afd_kernel_families import dependent_overlap


def audit(captured, markers, rank):
    indexed = {(r["step"], r["call"]): r for r in markers["records"] if r["rank"] == rank}
    count = markers["layers"] * markers["lanes"]
    if len(indexed) != count * len(markers["measured_steps"]):
        raise ValueError("Incomplete marker rank coverage")
    calls, lower, upper = [], None, None
    for step in markers["measured_steps"]:
        graph = captured[step - 1]
        pairs = sorted(dependent_overlap(graph, "sm100_fp8_fp4_mega_moe_impl",
                                        "megamoe_output_add_rmsnorm"),
                       key=lambda r: r["primary_start_ns"])
        if len(pairs) != count:
            raise ValueError(f"Unmatched measured graph {step}")
        chains = {}
        for stream in {r["stream"] for r in pairs}:
            groups = [[r for r in graph if r["streamId"] == stream and name in r["name"]]
                      for name in ("gate_topk_fused_tc_kernel", "gate_topk_select",
                                   "sm100_fp8_fp4_mega_moe_impl")]
            if len({len(group) for group in groups}) != 1:
                raise ValueError("Unmatched gate/selector/main chain")
            for gate, selector, main in zip(*groups):
                if not gate["start"] <= selector["start"] <= main["start"]:
                    raise ValueError("Unexpected chain ordering")
                chains[main["start"]] = (gate, selector)
        previous_combine = 0
        for ordinal, pair in enumerate(pairs):
            call = f"{ordinal // markers['lanes']}:{ordinal % markers['lanes']}"
            record = indexed[step, call]
            combine = [s for s in record["combine_store_end_ns"] if s]
            if max(combine) <= previous_combine:
                raise ValueError("Marker call order differs from kernel call order")
            previous_combine = max(combine)
            norm = (record["stamps_ns"][3:3 + 2 * record["norm_rows"]]
                    + record["residual_preload_end_ns"])
            gate, selector = chains[pair["primary_start_ns"]]
            gate_stamps = [s for worker in record.get("gate_worker_intervals_ns", []) for s in worker]
            if "gate_ready_ns" in record:
                gate_stamps.append(record["gate_ready_ns"])
            current_workers = record["reset_worker_intervals_ns"]
            workers = list(current_workers) if record.get("reset_current_performed", True) else []
            reset = [s for worker in current_workers for s in worker] + record["stamps_ns"][1:3]
            reset += record.get("gate_consumer_ready_ns", [])
            if "ahead_reset_performed" in record:
                ahead_workers = record["ahead_reset_worker_intervals_ns"]
                owner = record.get("ahead_reset_kernel", "selector")
                if owner not in ("norm", "selector", "gate"):
                    raise ValueError(f"Unknown ahead reset kernel: {owner}")
                stamps = {"norm": norm, "selector": reset, "gate": gate_stamps}[owner]
                stamps += [s for worker in ahead_workers for s in worker]
                stamps += record["ahead_reset_observation_ns"]
                if record["ahead_reset_performed"]:
                    workers += ahead_workers
            # globaltimer = Nsight + offset; no average or best-fit alignment.
            bounds = [
                (combine, pair["primary_start_ns"], pair["primary_end_ns"]),
                (norm, pair["consumer_start_ns"], pair["consumer_end_ns"]),
                (reset, selector["start"], selector["end"]),
            ]
            if gate_stamps:
                bounds.append((gate_stamps, gate["start"], gate["end"]))
            for stamps, start, end in bounds:
                lo, hi = max(stamps) - end, min(stamps) - start
                lower = lo if lower is None else max(lower, lo)
                upper = hi if upper is None else min(upper, hi)
            calls.append({"step": step, "call": call, "workers": workers,
                          "gate_ns": [gate["start"], gate["end"]],
                          "main_ns": [pair["primary_start_ns"], pair["primary_end_ns"]]})
    if lower is None or lower > upper:
        raise ValueError(f"No common clock offset: [{lower}, {upper}]; check rank/call mapping")
    for call in calls:
        for name in ("gate", "main"):
            start, end = call[name + "_ns"]
            overlap = [min(max(0, min(b, end + offset) - max(a, start + offset))
                           for offset in (lower, upper)) for a, b in call["workers"]]
            call[name + "_workers_guaranteed_overlap"] = sum(v > 0 for v in overlap)
            call[name + "_workers_guaranteed_inside"] = sum(
                a >= start + upper and b <= end + lower for a, b in call["workers"])
            call[name + "_minimum_worker_overlap_ns"] = min(overlap) if overlap else None
        call["last_clear_before_main_start_ns"] = [
            call["main_ns"][0] + offset - max(b for _, b in call["workers"])
            for offset in (lower, upper)] if call["workers"] else None
    return {"rank": rank, "calls": len(calls), "clock_offset_ns": [lower, upper],
            "clock_offset_width_ns": upper - lower,
            "calls_all_workers_inside_gate": sum(
                bool(c["workers"]) and c["gate_workers_guaranteed_inside"] == len(c["workers"]) for c in calls),
            "calls_any_worker_overlaps_gate": sum(c["gate_workers_guaranteed_overlap"] > 0 for c in calls),
            "calls_any_worker_overlaps_main": sum(c["main_workers_guaranteed_overlap"] > 0 for c in calls),
            "calls_without_clear": sum(not c["workers"] for c in calls),
            "calls_clear_before_main": sum(c["last_clear_before_main_start_ns"] is not None and
                                           c["last_clear_before_main_start_ns"][0] >= 0 for c in calls),
            "records": calls, "interpretation": __doc__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--markers", type=Path, required=True,
                        help="Model output from summarize_megamoe_recycling.py")
    parser.add_argument("--ffn-rank", type=int, required=True,
                        help="Device rank, not the profiler filename's worker tag")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(graphs(args.sqlite), json.loads(args.markers.read_text()), args.ffn_rank)
    result.update(sqlite=str(args.sqlite), markers=str(args.markers))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
