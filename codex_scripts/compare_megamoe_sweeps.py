#!/usr/bin/env python3
"""Compare complete audited MegaMoE sweeps, retaining every case and history root.

Historical best is selected independently per case from recorded phase medians.
These are unpaired measurements; sample policies and combine implementations may
differ. This report measures exposed dispatch, not end-to-end model latency.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


DISPATCH = "pre_compute_dispatch_exposure_us"
CONTRACT_KEYS = (
    "model_shape", "ep_size", "rows_per_ep_rank_per_microbatch",
    "global_tokens_per_launch", "hidden", "intermediate", "experts", "top_k",
    "activation_dtype", "weight_dtype", "microbatches", "num_sms",
    "input_route_distribution", "kernel_assumes_perfect_routing",
    "kernel_debug_timing", "routes_prepared", "route_ready_dispatch",
    "rank_ready_route_publish", "rank_interleaved_expert_mapping",
    "rank_gated_combine", "iket_mode", "iket_profile_rank",
)


def read(path):
    return json.loads(path.read_text())


def gain(before, after):
    if before <= 0:
        raise ValueError("Percentage reduction requires a positive reference")
    return 100 * (before - after) / before


def load(root, expected_phase_samples=None, ep=None):
    audit = read(root / "audit_summary.json")
    rows = list(csv.DictReader((root / "decomposition.csv").open()))
    if not rows or audit["artifact_contracts_checked"] != len(rows):
        raise ValueError(f"Sweep records are not audited: {root}")
    if ep is None:
        if not audit["complete"] or audit["missing"] or audit["expected_cases"] != len(rows):
            raise ValueError(f"Sweep is not complete/audited: {root}")
        audited_samples = audit["samples_checked"]
    else:
        manifest = read(root / "manifest.json")
        expected = {case["case_id"] for case in manifest["cases"] if case["ep"] == ep}
        completion = read(root / f"ep{ep}" / "completion.json")
        rows = [row for row in rows if int(row["ep"]) == ep]
        if (not expected or completion["status"] != "COMPLETE_AUDITED" or
                completion["cases"] != len(expected) or len(rows) != len(expected) or
                {row["case_id"] for row in rows} != expected or
                expected.intersection(audit["missing"])):
            raise ValueError(f"EP group is not complete/audited: {root}/ep{ep}")
        audited_samples = completion["phase_samples"]
    records = {}
    samples = 0
    for row in rows:
        case = row["case_id"]
        if case in records:
            raise ValueError(f"Duplicate case: {root}/{case}")
        artifact = read(root / row["artifact"])
        baseline = read((root / row["artifact"]).parent / "baseline.json")
        timing = next(iter(baseline["exact_qwen"].values()))
        values = [s[DISPATCH] for s in artifact["samples"]]
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid dispatch samples: {root}/{case}")
        if expected_phase_samples is not None and len(values) != expected_phase_samples:
            raise ValueError(f"Unexpected phase sample count: {root}/{case}")
        dispatch = median(values)
        if not math.isclose(dispatch, float(row[DISPATCH + "_median"]), abs_tol=1e-8):
            raise ValueError(f"CSV/artifact dispatch mismatch: {root}/{case}")
        kernel = artifact["nsys_kernel_us"]["mean"]
        if not math.isfinite(kernel) or kernel <= 0:
            raise ValueError(f"Invalid ordinary kernel mean: {root}/{case}")
        contract = artifact["contract"]
        if (int(row["ep"]), int(row["global_tokens"])) != (
                contract["ep_size"], contract["global_tokens_per_launch"]):
            raise ValueError(f"Case shape mismatch: {root}/{case}")
        records[case] = {"dispatch_us": dispatch, "kernel_mean_us": kernel,
                         "phase_samples": len(values), "contract": contract,
                         "ordinary_samples": timing["timing_iterations"],
                         "ordinary_warmups": timing["timing_warmups"],
                         "artifact": str(root / row["artifact"])}
        samples += len(values)
    if samples != audited_samples:
        raise ValueError(f"Audit sample count mismatch: {root}")
    return records


def summarize(rows, threshold):
    gains = [r["dispatch_gain_pct"] for r in rows]
    return {
        "cases": len(rows),
        "dispatch_gain_mean_pct": mean(gains),
        "dispatch_gain_median_pct": median(gains),
        "dispatch_gain_min_pct": min(gains),
        "dispatch_gain_max_pct": max(gains),
        "reduction_of_mean_dispatch_us_pct": gain(
            mean(r["historical_dispatch_us"] for r in rows),
            mean(r["candidate_dispatch_us"] for r in rows)),
        "cases_above_threshold": sum(g > threshold for g in gains),
        "cases_improved": sum(g > 0 for g in gains),
        "cases_regressed": sum(g < 0 for g in gains),
        "all_cases_above_threshold": all(g > threshold for g in gains),
    }


def summarize_kernel(rows):
    gains = [r["kernel_gain_pct"] for r in rows]
    return {
        "mean_per_case_gain_pct": mean(gains),
        "median_per_case_gain_pct": median(gains),
        "reduction_of_mean_kernel_us_pct": gain(
            mean(r["historical_kernel_mean_us"] for r in rows),
            mean(r["candidate_kernel_mean_us"] for r in rows)),
        "cases_regressed": sum(g < 0 for g in gains),
        "scope": "Ordinary MegaMoE kernel means against the same historical rows; "
                 "not producer-to-MegaMoE or full-model latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--phase-samples", type=int, required=True)
    parser.add_argument("--ep", type=int,
                        help="Compare one completed audited EP group; explicitly not a full-sweep result")
    parser.add_argument("--threshold", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    if args.phase_samples < 1 or not math.isfinite(args.threshold):
        parser.error("Require a positive sample count and finite threshold")
    candidate = load(args.root, args.phase_samples, args.ep)
    histories = [(root, load(root)) for root in args.history]
    comparisons = []
    by_history = {str(root): [] for root, _ in histories}
    for case, current in candidate.items():
        references = []
        for root, records in histories:
            if case not in records:
                continue
            reference = records[case]
            mismatched = [key for key in CONTRACT_KEYS
                          if current["contract"][key] != reference["contract"][key]]
            if mismatched:
                raise ValueError(f"Incompatible contract {root}/{case}: {mismatched}")
            row = {"case_id": case, "ep": current["contract"]["ep_size"],
                   "global_tokens": current["contract"]["global_tokens_per_launch"],
                   "historical_root": str(root),
                   "historical_dispatch_us": reference["dispatch_us"],
                   "candidate_dispatch_us": current["dispatch_us"],
                   "dispatch_gain_pct": gain(reference["dispatch_us"], current["dispatch_us"]),
                   "historical_kernel_mean_us": reference["kernel_mean_us"],
                   "candidate_kernel_mean_us": current["kernel_mean_us"],
                   "kernel_gain_pct": gain(reference["kernel_mean_us"], current["kernel_mean_us"]),
                   "historical_phase_samples": reference["phase_samples"],
                   "candidate_phase_samples": current["phase_samples"],
                   "historical_ordinary_samples": reference["ordinary_samples"],
                   "candidate_ordinary_samples": current["ordinary_samples"],
                   "historical_ordinary_warmups": reference["ordinary_warmups"],
                   "candidate_ordinary_warmups": current["ordinary_warmups"],
                   "historical_artifact": reference["artifact"],
                   "candidate_artifact": current["artifact"]}
            references.append(row)
            by_history[str(root)].append(row)
        if not references:
            raise ValueError(f"No historical coverage for {case}")
        comparisons.append(min(references, key=lambda r: r["historical_dispatch_us"]))
    if any(not rows for rows in by_history.values()):
        raise ValueError("A supplied history has no comparable cases")
    comparisons.sort(key=lambda r: (r["ep"], r["global_tokens"]))
    summary = summarize(comparisons, args.threshold)
    report = {"candidate_root": str(args.root), "required_gain_pct": args.threshold,
              "coverage_scope": "FULL_SWEEP" if args.ep is None else "COMPLETE_EP_GROUP_NOT_FULL_SWEEP",
              "ep_filter": args.ep,
              "contract_fields_checked": CONTRACT_KEYS,
              "summary_vs_historical_best": summary,
              "ordinary_kernel_vs_dispatch_selected_history": summarize_kernel(comparisons),
              "by_ep_vs_historical_best": {
                  str(ep): summarize([r for r in comparisons if r["ep"] == ep], args.threshold)
                  for ep in sorted({r["ep"] for r in comparisons})},
              "by_history": {root: {"summary": summarize(rows, args.threshold),
                                     "ordinary_kernel": summarize_kernel(rows),
                                     "missing_candidate_cases": sorted(set(candidate) - {r["case_id"] for r in rows}),
                                     "cases": rows} for root, rows in by_history.items()},
              "cases_vs_historical_best": comparisons,
              "scope": __doc__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    title = "Full-sweep" if args.ep is None else f"Completed EP{args.ep} group (not full sweep)"
    lines = [f"# {title} exposed dispatch comparison", "", __doc__.strip(), "",
             f"Compared all {len(comparisons)} candidate cases; threshold >{args.threshold:g}%.",
             f"Mean per-case reduction: {summary['dispatch_gain_mean_pct']:.3f}%; "
             f"reduction of mean dispatch time: {summary['reduction_of_mean_dispatch_us_pct']:.3f}%.",
             f"Above threshold: {summary['cases_above_threshold']}/{len(comparisons)}; "
             f"regressed: {summary['cases_regressed']}.", "",
             f"Ordinary MegaMoE kernel mean per-case reduction: "
             f"{report['ordinary_kernel_vs_dispatch_selected_history']['mean_per_case_gain_pct']:.3f}%. "
             "Uses the same per-case historical rows selected for dispatch; "
             "this is not a full-model latency measurement.", "",
             "| EP | Cases | Mean gain | Median gain | >Threshold | Regressed |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for ep, values in report["by_ep_vs_historical_best"].items():
        lines.append(f"| {ep} | {values['cases']} | {values['dispatch_gain_mean_pct']:.3f}% | "
                     f"{values['dispatch_gain_median_pct']:.3f}% | "
                     f"{values['cases_above_threshold']} | {values['cases_regressed']} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
