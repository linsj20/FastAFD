#!/usr/bin/env python3
"""Align fixed-call Nsight router/MegaMoE timelines to each producer's start."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decomposition", type=Path)
    parser.add_argument("job", nargs="?")
    parser.add_argument("--flat-trials", action="store_true",
                        help="Read ROOT/TRIAL/nsys rather than decomposition/ARM/JOB/ranks/rank-0/nsys")
    parser.add_argument("--sweep", action="store_true",
                        help="Read all collected cases from ROOT/decomposition.csv, using each baseline's sample count")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Also write per-arm interval distributions")
    parser.add_argument("--arms", nargs="+", default=["control", "reference", "alternate", "candidate"])
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.sweep and (args.flat_trials or args.job is not None):
        parser.error("--sweep cannot be combined with a job or --flat-trials")
    if not args.sweep and not args.flat_trials and args.job is None:
        parser.error("job is required for decomposition captures")
    root = args.decomposition.resolve()
    results = {}
    captures = []
    if args.sweep:
        with (root / "decomposition.csv").open() as stream:
            sweep_rows = list(csv.DictReader(stream))
        if not sweep_rows or len({row["case_id"] for row in sweep_rows}) != len(sweep_rows):
            raise ValueError("Require nonempty, unique collected sweep cases")
        for row in sweep_rows:
            run = (root / row["artifact"]).parent
            baseline = json.loads((run / "baseline.json").read_text())
            timings = list(baseline["exact_qwen"].values())
            if len(timings) != 1:
                raise ValueError(f"Expected one baseline timing record: {run}")
            captures.append((row["case_id"], run / "ranks/rank-0/nsys/megamoe-rank0.sqlite",
                             timings[0]["timing_iterations"]))
    else:
        for arm in args.arms:
            database = (root / arm / "nsys/megamoe-rank0.sqlite" if args.flat_trials else
                        root / arm / args.job / "ranks/rank-0/nsys/megamoe-rank0.sqlite")
            captures.append((arm, database, args.samples))
    for arm, database, samples in captures:
        if samples < 1:
            raise ValueError(f"Invalid timing sample count: {database}")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT k.start,k.end,s.value,k.gridX,k.blockX,k.streamId "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
                "ORDER BY k.start").fetchall()
        patterns = {"producer": "gate_topk_fused_tc_kernel", "selector": "gate_topk_select",
                    "megamoe": "sm100_fp8_fp4_mega_moe_impl"}
        groups = {key: [row for row in rows if pattern in row[2]] for key, pattern in patterns.items()}
        if any(len(group) != samples for group in groups.values()):
            raise ValueError(f"{database}: expected one producer/selector/MegaMoE per call")
        records = []
        for producer, selector, megamoe in zip(groups["producer"], groups["selector"], groups["megamoe"]):
            if not (producer[5] == selector[5] == megamoe[5]
                    and producer[0] <= selector[0] <= megamoe[0]
                    and producer[1] <= megamoe[1] and selector[1] <= megamoe[1]):
                raise ValueError(f"{database}: router/MegaMoE stream or ordering mismatch")
            records.append({
                "selector_start_us": (selector[0] - producer[0]) / 1000,
                "selector_end_us": (selector[1] - producer[0]) / 1000,
                "megamoe_start_us": (megamoe[0] - producer[0]) / 1000,
                "megamoe_end_us": (megamoe[1] - producer[0]) / 1000,
                "kernel_us": (megamoe[1] - megamoe[0]) / 1000,
            })
        result = {"database": str(database), "samples": samples,
                  "selector_grid_block": list(groups["selector"][0][3:5]),
                  "means": {key: mean(row[key] for row in records) for key in records[0]},
                  "records": records}
        gaps = [(producer[0] - previous[1]) / 1000 for producer, previous in
                zip(groups["producer"][1:], groups["megamoe"][:-1])]
        result["inter_call_gaps_us"] = gaps
        result["mean_inter_call_gap_us"] = mean(gaps) if gaps else None
        cycles = [row["megamoe_end_us"] + gap for row, gap in zip(records, gaps)]
        result["producer_cycle_samples_us"] = cycles
        result["mean_producer_cycle_us"] = mean(cycles) if cycles else None
        results[arm] = result
        if not args.sweep:
            print(json.dumps({"arm": arm, "means": result["means"],
                              "mean_inter_call_gap_us": result["mean_inter_call_gap_us"],
                              "mean_producer_cycle_us": result["mean_producer_cycle_us"]}), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if args.sweep:
        print(json.dumps({"root": str(root), "captured_cases": len(results),
                          "samples": sum(result["samples"] for result in results.values()),
                          "scope": "Collected cases only; full-sweep coverage requires the sweep audit."}), flush=True)
    if args.csv:
        summaries = []
        for arm, result in results.items():
            records = result["records"]
            intervals = {
                "router_to_megamoe_end": [r["megamoe_end_us"] for r in records],
                "megamoe_kernel": [r["kernel_us"] for r in records],
                "selector": [r["selector_end_us"] - r["selector_start_us"] for r in records],
                "megamoe_launch_delay": [r["megamoe_start_us"] for r in records],
            }
            row = {"case_id": arm, "samples": len(records)}
            for name, values in intervals.items():
                for label, function in (("mean", mean), ("median", median), ("min", min), ("max", max)):
                    row[f"{name}_{label}_us"] = function(values)
            summaries.append(row)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
