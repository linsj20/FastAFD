#!/usr/bin/env python3
"""Compare paired AFD baseline/path/PDL arms, retaining every measured sample."""

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


def compare(paths):
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
        return {
            "megamoe_path_saving_ms": means["pull"] - means["no_tail_pdl"],
            "pdl_saving_ms": means["no_tail_pdl"] - means["early_pdl"],
            "combined_saving_ms": means["pull"] - means["early_pdl"],
        }

    pooled = {arm: mean(x for run in runs for x in run["samples_ms"])
              for arm, runs in arms.items()}
    orders = {str(i): gains({arm: runs[i]["mean_ms"] for arm, runs in arms.items()})
              for i in range(next(iter(counts)))}
    return {"contract": contract, "arms": arms, "pooled_ms": pooled,
            "samples": sum(len(run["samples_ms"]) for runs in arms.values() for run in runs),
            **gains(pooled), "orders": orders}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--without-pdl", type=Path, action="append", required=True)
    parser.add_argument("--with-pdl", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare({"pull": args.baseline, "no_tail_pdl": args.without_pdl,
                      "early_pdl": args.with_pdl})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}))


if __name__ == "__main__":
    main()
