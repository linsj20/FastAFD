#!/usr/bin/env python3
"""Pool repeated rank-timing captures without dropping samples or aligning clocks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from collect_cuda_rank_timings import aggregate_rank_timings


def compare(inputs):
    groups = defaultdict(list)
    contract = None
    seen = set()
    for label, path in inputs:
        if path.resolve() in seen:
            raise ValueError(f"Duplicate capture: {path}")
        seen.add(path.resolve())
        data = json.loads(path.read_text())
        ranks = sorted(data["ranks"], key=lambda r: r["rank"])
        audited = aggregate_rank_timings(ranks)
        meta = ranks[0]["metadata"]
        calls = meta["calls_per_graph"]
        if type(calls) is not int or calls <= 0:
            raise ValueError(f"Invalid calls_per_graph in {path}")
        shape = (audited["scope"], audited["world_size"], calls,
                 meta["rows"], meta["perfect_routing"])
        if any((r["metadata"]["calls_per_graph"], r["metadata"]["rows"],
                r["metadata"]["perfect_routing"]) != shape[2:] for r in ranks):
            raise ValueError(f"Rank workload contracts differ in {path}")
        if contract is not None and shape != contract:
            raise ValueError(f"Workload contract differs in {path}")
        contract = shape
        groups[label].append((path, ranks, audited))
    if contract is None:
        raise ValueError("Require at least one capture")
    factor = 1000 / contract[2]
    summaries = {}
    for label, captures in groups.items():
        first = captures[0][1][0]["metadata"]
        if any(any(r["metadata"][key] != first[key]
                   for key in ("mode", "allocated_capacity", "reset_rows"))
               for _, records, _ in captures for r in records):
            raise ValueError(f"Cannot pool different backend/buffer configurations: {label}")
        pooled = [dict(captures[0][1][rank], durations_ms=[
            value for _, records, _ in captures for value in records[rank]["durations_ms"]
        ]) for rank in range(contract[1])]
        result = aggregate_rank_timings(pooled)
        summaries[label] = {
            "graphs_per_rank": result["iterations"],
            "maximum_rank_mean_us_per_call": result["maximum_rank_mean_ms"] * factor,
            "mean_per_graph_maximum_us_per_call": result["mean_per_call_maximum_ms"] * factor,
            "captures": [{"path": str(path), "graphs_per_rank": audited["iterations"],
                          "maximum_rank_mean_us_per_call": audited["maximum_rank_mean_ms"] * factor}
                         for path, _, audited in captures],
        }
    return {
        "scope": contract[0], "ranks": contract[1], "calls_per_graph": contract[2],
        "rows": contract[3], "perfect_routing": contract[4], "arms": summaries,
        "interpretation": "All rank-local CUDA event durations retained. Per-call values "
                          "divide complete graph duration by calls per graph. Rank clocks "
                          "are not aligned; these are neither model E2E nor global latest-finish times.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", metavar="NAME=PATH",
                        help="Repeat NAME to pool multiple captures of the same arm")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = []
    for item in args.captures:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("Captures must be NAME=PATH")
        inputs.append((name, Path(path)))
    result = compare(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
