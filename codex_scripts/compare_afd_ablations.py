#!/usr/bin/env python3
"""Compare named, paired AFD arms, retaining every measured sample."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


CONTRACT = (
    "case_id", "metric_version", "active_gpus", "allocated_gpus",
    "num_microbatches", "target_batch_per_attention_dp_lane", "latency_basis",
    "profiled_roles", "decode_step_ids", "warmup_decode_step_id",
)


def compare(paths, comparisons=None):
    if comparisons is None:
        comparisons = {
            "megamoe_path_saving_ms": ("pull", "no_tail_pdl"),
            "pdl_saving_ms": ("no_tail_pdl", "early_pdl"),
            "combined_saving_ms": ("pull", "early_pdl"),
        }
    if not comparisons or any(len(pair) != 2 or any(arm not in paths for arm in pair)
                              for pair in comparisons.values()):
        raise ValueError("Every comparison must name two existing arms")
    counts = {len(group) for group in paths.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("Require the same positive number of paired runs per arm")
    contract, arms = None, {}
    for arm, files in paths.items():
        arms[arm] = []
        for path in files:
            record = json.loads(path.read_text())
            current = {key: record[key] for key in CONTRACT}
            current["profiled_roles"] = sorted(record["profiled_roles"])
            if contract is None:
                contract = current
            if current != contract:
                raise ValueError(f"Unmatched measurement contract: {path}")
            steps = record["decode_step_ids"]
            values = record["per_step_cuda_execution_critical_ms"]
            if (not steps or steps != sorted(set(steps))
                    or record["warmup_decode_step_id"] in steps
                    or steps != record["latency_decode_step_ids"]
                    or len(values) != len(steps)
                    or len(values) != record["sample_count"]
                    or record["excluded_decode_step_ids"]
                    or record["outlier_count"] != 0
                    or any(not math.isfinite(x) or x <= 0 for x in values)):
                raise ValueError(f"Incomplete or filtered measurement: {path}")
            arms[arm].append({"path": str(path), "mean_ms": mean(values),
                              "samples_ms": values, "step_ids": steps})

    def gains(means):
        return {name: means[base] - means[candidate]
                for name, (base, candidate) in comparisons.items()}

    pooled = {arm: mean(x for run in runs for x in run["samples_ms"])
              for arm, runs in arms.items()}
    orders = {str(i): gains({arm: runs[i]["mean_ms"] for arm, runs in arms.items()})
              for i in range(next(iter(counts)))}
    return {"contract": contract, "arms": arms, "comparisons": comparisons, "pooled_ms": pooled,
            "samples": sum(len(run["samples_ms"]) for runs in arms.values() for run in runs),
            **gains(pooled), "orders": orders}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append")
    parser.add_argument("--without-pdl", type=Path, action="append")
    parser.add_argument("--with-pdl", type=Path, action="append")
    parser.add_argument("--arm", action="append", metavar="NAME=FILE",
                        help="Repeat each name for paired runs, in matching order")
    parser.add_argument("--difference", action="append", metavar="NAME=BASE:CANDIDATE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.arm:
        if any((args.baseline, args.without_pdl, args.with_pdl)) or not args.difference:
            parser.error("Named arms require --difference and cannot mix with legacy arm options")
        paths, comparisons = {}, {}
        for value in args.arm:
            name, separator, path = value.partition("=")
            if not name or not separator or not path:
                parser.error("--arm requires NAME=FILE")
            paths.setdefault(name, []).append(Path(path))
        for value in args.difference:
            name, separator, pair = value.partition("=")
            arms = pair.split(":")
            if not name or not separator or name in comparisons or len(arms) != 2:
                parser.error("--difference requires a unique NAME=BASE:CANDIDATE")
            comparisons[name] = arms
        result = compare(paths, comparisons)
    else:
        if args.difference or not all((args.baseline, args.without_pdl, args.with_pdl)):
            parser.error("Supply named arms and differences, or all three legacy arm options")
        result = compare({"pull": args.baseline, "no_tail_pdl": args.without_pdl,
                          "early_pdl": args.with_pdl})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}))


if __name__ == "__main__":
    main()
