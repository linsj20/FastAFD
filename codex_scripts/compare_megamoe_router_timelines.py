#!/usr/bin/env python3
"""Compare paired sweep-arm producer-to-MegaMoE timelines, retaining every record."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import mean


METRICS = ("selector_start_us", "selector_end_us", "megamoe_start_us",
           "megamoe_end_us", "kernel_us")


def gain(before, after):
    return 100 * (before - after) / before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", action="append", required=True,
                        help="Arm prefix; both PREFIX and PREFIX_repeat are required")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.arm)) != len(args.arm) or len(args.arm) < 2:
        parser.error("Provide at least two distinct arm prefixes")
    if args.samples <= 0:
        parser.error("Sample count must be positive")
    timelines = {}
    for prefix in args.arm:
        if Path(prefix).name != prefix or prefix in {".", ".."}:
            parser.error("Each arm prefix must be a directory name")
        for arm in (prefix, prefix + "_repeat"):
            if arm in timelines:
                parser.error("Arm prefixes overlap with repeat names")
            timelines[arm] = json.loads(
                (args.root / arm / "router_timeline.json").read_text())
    cases = set(next(iter(timelines.values())))
    if not cases or any(set(t) != cases for t in timelines.values()):
        raise ValueError("All arms must contain the same nonempty case set")
    results = []
    for case in sorted(cases):
        arm_means = {}
        for arm, timeline in timelines.items():
            entry = timeline[case]
            records = entry["records"]
            if entry["samples"] != args.samples or len(records) != args.samples:
                raise ValueError(f"Unexpected sample count: {arm}/{case}")
            if any(not isfinite(r[k]) for r in records for k in METRICS):
                raise ValueError(f"Nonfinite timeline: {arm}/{case}")
            arm_means[arm] = {k: mean(r[k] for r in records) for k in METRICS}
            if arm_means[arm]["megamoe_end_us"] <= 0:
                raise ValueError(f"Nonpositive full-chain latency: {arm}/{case}")
        pooled = {p: {k: mean(arm_means[a][k] for a in (p, p + "_repeat"))
                      for k in METRICS} for p in args.arm}
        fullchain = {a: v["megamoe_end_us"] for a, v in arm_means.items()}
        comparisons = {}
        for candidate in args.arm:
            comparisons[candidate] = {}
            for control in args.arm:
                if candidate == control:
                    continue
                comparisons[candidate][control] = {
                    "pooled_gain_pct": gain(pooled[control]["megamoe_end_us"],
                                            pooled[candidate]["megamoe_end_us"]),
                    "worst_repeat_gain_pct": gain(
                        min(fullchain[control], fullchain[control + "_repeat"]),
                        max(fullchain[candidate], fullchain[candidate + "_repeat"])),
                    "launch_delta_us": pooled[candidate]["megamoe_start_us"]
                                       - pooled[control]["megamoe_start_us"],
                    "kernel_delta_us": pooled[candidate]["kernel_us"]
                                       - pooled[control]["kernel_us"],
                }
        results.append({"case_id": case, "pooled_means_us": pooled,
                        "arm_fullchain_us": fullchain, "comparisons": comparisons})
    report = {
        "scope": "Matched rank0 producer-to-MegaMoE means; all records retained, "
                 "equal-count repeats. Not distributed request latency or a "
                 "statistical-independence claim. Routing follows the input captures.",
        "samples_per_arm_case": args.samples,
        "arms": args.arm,
        "cases": results,
    }
    output = args.output or args.root / "fullchain_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cases": len(cases),
                      "records": len(cases) * len(timelines) * args.samples}))


if __name__ == "__main__":
    main()
