#!/usr/bin/env python3
"""Compare an audited ordinary full-path sweep with explicit historical sources."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from compare_megamoe_sweeps import CONTRACT_KEYS, gain, load
from compare_megamoe_router_timelines import METRICS


def read(path):
    return json.loads(path.read_text())


def audit_fullpath(root, candidate, control, candidate_source, control_source,
                   expected_cases, job, samples=30, history=None):
    """Recompute the E2E gate from ordinary records from one matched timing job."""
    if candidate == control or samples <= 0 or not expected_cases:
        raise ValueError("Require distinct full-path arms and nonempty sample coverage")
    means, component_means = {}, {}
    for prefix, source in ((candidate, candidate_source), (control, control_source)):
        if Path(prefix).name != prefix or prefix in {".", ".."}:
            raise ValueError("Full-path arm must be a directory name")
        for arm in (prefix, prefix + "_repeat"):
            if arm in means:
                raise ValueError("Full-path arm names overlap")
            env = read(root / arm / "env.json")
            if Path(env["FASTAFD_SOURCE_REPO"]).name != source:
                raise ValueError(f"Full-path source mismatch: {arm}")
            timeline = read(root / arm / "router_timeline.json")
            if set(timeline) != set(expected_cases):
                raise ValueError(f"Full-path case coverage mismatch: {arm}")
            means[arm] = {}
            component_means[arm] = {}
            for case, entry in timeline.items():
                records = entry["records"]
                if (entry["samples"] != samples or len(records) != samples
                        or str(job) not in Path(entry["database"]).parts):
                    raise ValueError(f"Full-path job/sample mismatch: {arm}/{case}")
                if history is not None:
                    before = read(Path(history[case]["best"]["artifact"]))["contract"]
                    current = read(root / arm / case / str(job) / "decomposition.json")["contract"]
                    if any(before[k] != current[k] for k in CONTRACT_KEYS):
                        raise ValueError(f"Historical full-path input mismatch: {arm}/{case}")
                for r in records:
                    start, end, kernel = (r[k] for k in
                        ("megamoe_start_us", "megamoe_end_us", "kernel_us"))
                    if (not all(math.isfinite(v) for v in (start, end, kernel))
                            or start < 0 or kernel <= 0 or end <= start
                            or not math.isclose(end - start, kernel, abs_tol=1e-6)):
                        raise ValueError(f"Invalid full-path record: {arm}/{case}")
                means[arm][case] = mean(r["megamoe_end_us"] for r in records)
                producer = []
                for r in records:
                    start, end = r["selector_start_us"], r["selector_end_us"]
                    if (not all(math.isfinite(v) for v in (start, end))
                            or start < 0 or end <= start):
                        raise ValueError(f"Invalid producer record: {arm}/{case}")
                    producer.append(end)
                artifact = read(root / arm / case / str(job) / "decomposition.json")
                phases = artifact["samples"]
                expected_phases = int(env["MEGAMOE_IKET_ITERATIONS"])
                if expected_phases <= 0 or len(phases) != expected_phases:
                    raise ValueError(f"Component phase coverage mismatch: {arm}/{case}")
                components = {
                    "producer_completion_us": mean(producer),
                    "megamoe_start_us": mean(r["megamoe_start_us"] for r in records),
                }
                for metric in ("compute_span_us", "combine_span_us"):
                    values = [r[metric] for r in phases]
                    if any(not math.isfinite(v) or v <= 0 for v in values):
                        raise ValueError(f"Invalid component phase: {arm}/{case}/{metric}")
                    components[metric] = median(values)
                component_means[arm][case] = components
    cases = []
    for case in sorted(expected_cases):
        before = [means[a][case] for a in (control, control + "_repeat")]
        after = [means[a][case] for a in (candidate, candidate + "_repeat")]
        pooled = 100 * (mean(before) - mean(after)) / mean(before)
        worst = 100 * (min(before) - max(after)) / min(before)
        cases.append({"case_id": case, "pooled_gain_pct": pooled,
                      "worst_repeat_gain_pct": worst,
                      "passed": pooled > 0 and worst > 0})
        if history is not None:
            best = history[case]["best"]["fullpath_us"]
            if not math.isfinite(best) or best <= 0:
                raise ValueError(f"Invalid historical full-path mean: {case}")
            historical_gain = 100 * (best - max(after)) / best
            cases[-1].update({"historical_best_us": best,
                             "history_worst_repeat_gain_pct": historical_gain})
            cases[-1]["passed"] &= historical_gain > 0
        components = {}
        for metric in component_means[candidate][case]:
            before_component = [component_means[a][case][metric]
                                for a in (control, control + "_repeat")]
            after_component = [component_means[a][case][metric]
                               for a in (candidate, candidate + "_repeat")]
            delta = mean(after_component) - mean(before_component)
            components[metric] = {
                "control_repeats_us": before_component,
                "candidate_repeats_us": after_component,
                "pooled_delta_us": delta,
                "worst_repeat_delta_us": max(after_component) - min(before_component),
                "passed": delta <= 0,
            }
        cases[-1]["components"] = components
        cases[-1]["fullpath_only_passed"] = cases[-1]["passed"]
        cases[-1]["other_parts_nonregressing"] = all(c["passed"] for c in components.values())
        cases[-1]["passed"] &= cases[-1]["other_parts_nonregressing"]
    return {"all_pass": all(c["passed"] for c in cases), "cases": cases,
            "job": str(job), "candidate_arm": candidate, "control_arm": control,
            "scope": "Ordinary rank0 producer-start to MegaMoE-end; every record "
                     "and both repeats retained. Not slowest-rank completion or "
                     "request latency.", "historical_gate_checked": history is not None,
            "component_policy": "Require nonincreasing pooled producer completion, MegaMoE "
                                "start, compute span and combine span versus the matched control. "
                                "Ordinary producer means and per-repeat phase medians retain all "
                                "records. Worst-repeat component deltas are reported separately. "
                                "Phase spans include stalls/overlap; they are conservative "
                                "regression guards, not isolated arithmetic or network efficiency."}


def timeline_means(entry, record):
    samples = entry["records"]
    if entry["samples"] != record["ordinary_samples"] or len(samples) != entry["samples"]:
        raise ValueError(f"Timeline sample count differs from ordinary capture: {record['artifact']}")
    artifact = Path(record["artifact"])
    database = Path(entry["database"])
    if database.parents[3].name != artifact.parent.name or database.parents[4].name != artifact.parent.parent.name:
        raise ValueError(f"Timeline case/job differs from audited artifact: {artifact}")
    for row in samples:
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in METRICS):
            raise ValueError(f"Invalid timeline value: {artifact}")
        if (row["selector_end_us"] < row["selector_start_us"]
                or row["megamoe_start_us"] < row["selector_start_us"]
                or row["megamoe_end_us"] < row["selector_end_us"]
                or row["kernel_us"] <= 0
                or not math.isclose(row["megamoe_end_us"] - row["megamoe_start_us"],
                                    row["kernel_us"], abs_tol=1e-6)):
            raise ValueError(f"Invalid producer/consumer timing order: {artifact}")
    return {k: mean(r[k] for r in samples) for k in METRICS}


def summarize(rows):
    gains = [r["gain_pct"] for r in rows]
    return {
        "cases": len(rows),
        "candidate_records": sum(r["candidate_samples"] for r in rows),
        "reference_records": sum(r["reference_samples"] for r in rows),
        "mean_per_case_chain_gain_pct": mean(gains),
        "reduction_of_mean_chain_pct": gain(mean(r["reference_chain_us"] for r in rows),
                                            mean(r["candidate_chain_us"] for r in rows)),
        "min_gain_pct": min(gains), "max_gain_pct": max(gains),
        "improved": sum(v > 0 for v in gains), "regressed": sum(v < 0 for v in gains),
        "mean_launch_delta_us": mean(r["launch_delta_us"] for r in rows),
        "mean_kernel_delta_us": mean(r["kernel_delta_us"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reference", type=Path, action="append", required=True,
                        help="Historical sweep; repeat to use the lowest mean per case")
    parser.add_argument("--phase-samples", type=int, default=5)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase_samples < 1:
        parser.error("Phase sample count must be positive")
    if len(set(args.reference)) != len(args.reference) or args.root in args.reference:
        parser.error("Reference roots must be distinct from each other and the candidate")
    candidate = load(args.root, args.phase_samples, args.ep)
    references = [(root, load(root), json.loads((root / "router_timeline.json").read_text()))
                  for root in args.reference]
    current_timeline = json.loads((args.root / "router_timeline.json").read_text())
    rows = []
    for case, record in candidate.items():
        observations = []
        for root, reference, timeline in references:
            before = reference[case]
            mismatched = [k for k in CONTRACT_KEYS if record["contract"][k] != before["contract"][k]]
            if mismatched:
                raise ValueError(f"Incompatible input contract: {root}/{case}/{mismatched}")
            observations.append((before, timeline_means(timeline[case], before)))
        before, prior = min(observations, key=lambda entry: entry[1]["megamoe_end_us"])
        current = timeline_means(current_timeline[case], record)
        rows.append({
            "case_id": case, "ep": record["contract"]["ep_size"],
            "global_tokens": record["contract"]["global_tokens_per_launch"],
            "candidate_chain_us": current["megamoe_end_us"],
            "reference_chain_us": prior["megamoe_end_us"],
            "gain_pct": gain(prior["megamoe_end_us"], current["megamoe_end_us"]),
            "launch_delta_us": current["megamoe_start_us"] - prior["megamoe_start_us"],
            "kernel_delta_us": current["kernel_us"] - prior["kernel_us"],
            "candidate_samples": record["ordinary_samples"],
            "reference_samples": before["ordinary_samples"],
            "candidate_artifact": record["artifact"], "reference_artifact": before["artifact"],
        })
    rows.sort(key=lambda r: (r["ep"], r["global_tokens"]))
    report = {
        "scope": "FULL_SWEEP" if args.ep is None else "COMPLETE_EP_GROUP_NOT_FULL_SWEEP",
        "measurement": "Unpaired rank0 source-router-start to MegaMoE-end means. "
                       "All ordinary records retained; no IKET phases added. "
                       "Not distributed request latency or a statistical-independence claim.",
        "candidate_root": str(args.root),
        "reference_root": str(args.reference[0]) if len(args.reference) == 1 else None,
        "reference_roots": [str(root) for root in args.reference],
        "reference_selection": "Lowest per-case mean over the explicitly listed histories; all calls retained",
        "summary": summarize(rows),
        "by_ep": {str(ep): summarize([r for r in rows if r["ep"] == ep])
                  for ep in sorted({r["ep"] for r in rows})},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
