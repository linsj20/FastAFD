#!/usr/bin/env python3
"""Summarize selected kernel families in complete captured AFD graph replays.

Kernel sums can overlap and are not an additive decomposition of step latency.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_afd_attention_gap import graphs, span, union_duration


def stats(values):
    return {"count": len(values), "mean_us": mean(values),
            "median_us": median(values), "min_us": min(values),
            "max_us": max(values), "sum_us": sum(values)}


def router_chains(graph):
    """Pair each stream's producer, selector and MegaMoE, retaining PDL overlap."""
    patterns = ("gate_topk_fused_tc_kernel", "gate_topk_select",
                "sm100_fp8_fp4_mega_moe_impl")
    streams = sorted({row["streamId"] for row in graph if patterns[2] in row["name"]})
    records = []
    for stream in streams:
        groups = [[row for row in graph if row["streamId"] == stream and pattern in row["name"]]
                  for pattern in patterns]
        if len({len(group) for group in groups}) != 1:
            raise ValueError(f"Unmatched router chain on stream {stream}")
        for producer, selector, megamoe in zip(*groups):
            if not producer["start"] <= selector["start"] <= megamoe["start"]:
                raise ValueError(f"Unexpected router launch order on stream {stream}")
            records.append({
                "producer_to_megamoe_end_us": (megamoe["end"] - producer["start"]) / 1000,
                "megamoe_launch_delay_us": (megamoe["start"] - producer["start"]) / 1000,
                "selector_megamoe_overlap_us": max(0, min(selector["end"], megamoe["end"])
                                                  - megamoe["start"]) / 1000,
            })
    if not records:
        raise ValueError("No router chains captured")
    return {key: [record[key] for record in records] for key in records[0]}


def attention_unions(graph):
    groups = {name: [r for r in graph if pattern in r["name"]] for name, pattern in
              (("fmha", "fmhaSm"), ("publish", "publish_o"), ("wait", "wait_ready_kernel"))}
    if any(not rows for rows in groups.values()) or sum(map(len, groups.values())) != len(graph):
        raise ValueError("Attention graph must contain only FMHA, publication and ready-wait kernels")
    active = union_duration(graph)
    useful = union_duration(groups["fmha"] + groups["publish"])
    fmha, publish, wait = (union_duration(groups[name]) for name in ("fmha", "publish", "wait"))
    start, end = span(graph)
    return {key: value / 1000 for key, value in {
        "graph_span_us": end - start, "fmha_union_us": fmha,
        "publish_union_us": publish, "useful_union_us": useful,
        "fmha_publish_overlap_us": fmha + publish - useful,
        "wait_union_us": wait, "wait_useful_overlap_us": wait + useful - active,
        "wait_only_us": active - useful, "no_graph_kernel_us": end - start - active,
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", action="append", required=True,
                        help="LABEL=literal kernel-name substring")
    parser.add_argument("--expected-graphs", type=int, default=16)
    parser.add_argument("--warmup-graphs", type=int, default=1)
    parser.add_argument("--router-chains", action="store_true",
                        help="Measure per-stream producer-to-MegaMoE intervals and selector overlap")
    parser.add_argument("--attention-unions", action="store_true",
                        help="Measure useful attention work, overlapping waits and uncovered intervals")
    args = parser.parse_args()
    families = {}
    for item in args.family:
        label, pattern = item.split("=", 1)
        if not label or not pattern or label in families:
            raise ValueError("Family labels must be unique and nonempty")
        families[label] = pattern
    captured = graphs(args.sqlite)
    if len(captured) != args.expected_graphs:
        raise ValueError(f"Expected {args.expected_graphs} graphs, got {len(captured)}")
    if not 0 <= args.warmup_graphs < len(captured):
        raise ValueError("Invalid warmup graph count")
    samples = []
    values = {label: [] for label in families}
    names = {label: set() for label in families}
    chain_values = {}
    attention_values = {}
    for ordinal, graph in enumerate(captured[args.warmup_graphs:], args.warmup_graphs + 1):
        start, end = span(graph)
        sample = {"graph_ordinal": ordinal, "graph_nodes": len(graph),
                  "graph_span_us": (end - start) / 1000, "families": {}}
        for label, pattern in families.items():
            selected = [row for row in graph if pattern in row["name"]]
            if not selected:
                raise ValueError(f"Missing {label} in graph {ordinal}")
            durations = [(row["end"] - row["start"]) / 1000 for row in selected]
            if min(durations) <= 0:
                raise ValueError(f"Nonpositive kernel duration: {label}, graph {ordinal}")
            sample["families"][label] = stats(durations)
            values[label].extend(durations)
            names[label].update(row["name"] for row in selected)
        if args.router_chains:
            chains = router_chains(graph)
            sample["router_chains"] = {key: stats(value) for key, value in chains.items()}
            for key, value in chains.items():
                chain_values.setdefault(key, []).extend(value)
        if args.attention_unions:
            timeline = attention_unions(graph)
            sample["attention_timeline"] = timeline
            for key, value in timeline.items():
                attention_values.setdefault(key, []).append(value)
        samples.append(sample)
    result = {"sqlite": str(args.sqlite), "captured_graphs": len(captured),
              "warmup_graphs_excluded": args.warmup_graphs,
              "measured_graphs": len(samples), "samples": samples,
              "families": {label: {**stats(values[label]),
                                     "kernel_names": sorted(names[label])}
                           for label in families},
              "interpretation": "All measured graphs retained. Kernel sums may overlap; "
                                "they must not be added to infer step latency."}
    if args.router_chains:
        result["router_chains"] = {key: stats(value) for key, value in chain_values.items()}
    if args.attention_unions:
        result["attention_timeline"] = {key: stats(value) for key, value in attention_values.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: stats(value) for label, value in values.items()}))


if __name__ == "__main__":
    main()
