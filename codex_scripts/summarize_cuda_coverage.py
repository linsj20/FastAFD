#!/usr/bin/env python3
"""Partition captured graph time into selected successor spans and their complement.

Outside selected spans, buckets retain every active kernel family (or idle).
This is disjoint timeline accounting, not attribution of a performance cause.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

from analyze_afd_attention_gap import graphs


def partition(graph, regions, families):
    events = defaultdict(Counter)
    for row in graph:
        matches = [label for label, pattern in families.items() if pattern in row["name"]]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous family: {row['name']}")
        label = matches[0] if matches else row["name"]
        events[row["start"]][label] += 1
        events[row["end"]][label] -= 1
    for start, end in regions:
        if not min(events) <= start < end <= max(events):
            raise ValueError("Selected region lies outside its graph")
        events[start]["__region__"] += 1
        events[end]["__region__"] -= 1
    active, duration = Counter(), Counter()
    previous = min(events)
    for stamp in sorted(events):
        labels = sorted(k for k, v in active.items() if v and k != "__region__")
        bucket = "selected_regions" if active["__region__"] else " & ".join(labels) or "idle"
        duration[bucket] += stamp - previous
        active.update(events[stamp])
        if any(v < 0 for v in active.values()):
            raise ValueError("Unbalanced interval endpoints")
        previous = stamp
    if any(active.values()) or sum(duration.values()) != max(events) - min(events):
        raise ValueError("Timeline partition does not cover its graph")
    return {k: v / 1000 for k, v in duration.items() if v}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--successors", type=Path, required=True,
                        help="Output from summarize_afd_kernel_families.py --successors")
    parser.add_argument("--consumer", action="append", required=True)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.successors.read_text())
    if Path(report["sqlite"]).resolve() != args.sqlite.resolve():
        raise ValueError("Successor records came from another trace")
    families = dict(item.split("=", 1) for item in args.family)
    if len(families) != len(args.family) or any(
            not k or not v or k in ("__region__", "selected_regions", "idle")
            for k, v in families.items()):
        raise ValueError("Require unique, nonempty family labels and patterns")
    captured, samples = graphs(args.sqlite), []
    for sample in report["samples"]:
        graph = captured[sample["graph_ordinal"] - 1]
        if len(graph) != sample["graph_nodes"]:
            raise ValueError("Graph coverage changed")
        regions = [(r["current_start_ns"], r["successor_end_ns"])
                   for r in sample["successors"]
                   if any(p in r["successor"] for p in args.consumer)]
        if not regions:
            raise ValueError("No selected successor regions")
        samples.append({"graph_ordinal": sample["graph_ordinal"],
                        "selected_regions": len(regions),
                        "bucket_us": partition(graph, regions, families)})
    keys = sorted({k for s in samples for k in s["bucket_us"]})
    result = {"sqlite": str(args.sqlite), "samples": samples,
              "mean_bucket_us": {k: mean(s["bucket_us"].get(k, 0) for s in samples)
                                 for k in keys},
              "interpretation": __doc__}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean_bucket_us"]))


if __name__ == "__main__":
    main()
