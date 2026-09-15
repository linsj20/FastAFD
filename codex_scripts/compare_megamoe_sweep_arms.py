#!/usr/bin/env python3
"""Compare audited same-allocation sweep ABBA arms with matching historical cases."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


ARMS = ("control", "candidate", "candidate_repeat", "control_repeat")
PHASES = ("pre_compute_dispatch_exposure_us", "compute_span_us",
          "post_compute_combine_exposure_us", "post_combine_tail_us")


def read(path):
    return json.loads(path.read_text())


def gain(before, after):
    return 100 * (before - after) / before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--dispatch-gain", type=float, default=5)
    parser.add_argument("--candidate-arm", default="candidate",
                        help="Candidate prefix with a PREFIX_repeat arm")
    parser.add_argument("--control-arm", default="control",
                        help="Primary control prefix with a PREFIX_repeat arm")
    parser.add_argument("--output", type=Path,
                        help="Report path; defaults to dispatch_comparison.json under root")
    parser.add_argument("--additional-control", action="append", default=[],
                        help="Additional control prefix with a PREFIX_repeat arm; include in the gate")
    args = parser.parse_args()
    candidate_arm = args.candidate_arm
    control_arm = args.control_arm
    for arm in (candidate_arm, control_arm, *args.additional_control):
        if Path(arm).name != arm or arm in {".", ".."}:
            parser.error("arm prefixes must be directory names")
    references = (control_arm, *args.additional_control)
    arms = tuple(a for ref in (*references, candidate_arm) for a in (ref, ref + "_repeat"))
    assert len(set(arms)) == len(arms), "Control prefixes must name distinct arm pairs"
    selected = read(args.root / control_arm / "env.json")["MEGAMOE_SWEEP_CASES"].split(",")
    historical = [{r["case_id"]: r for r in csv.DictReader(
        (root / "decomposition.csv").read_text().splitlines())} for root in args.history]
    results = []
    for case in selected:
        records = {}
        contracts = []
        for arm in arms:
            root = args.root / arm
            env = read(root / "env.json")
            assert env["MEGAMOE_SWEEP_CASES"].split(",") == selected
            assert env["DG_MEGAMOE_LAMPORT_DISPATCH"] == "0"
            assert env["DG_MEGAMOE_LAMPORT_COMBINE"] == "1"
            assert case not in read(root / "audit_summary.json")["missing"]
            paths = list((root / case).glob("*/decomposition.json"))
            assert len(paths) == 1, (arm, case, paths)
            records[arm] = read(paths[0])
            contracts.append(records[arm]["contract"])
        assert all(c == contracts[0] for c in contracts), case
        combined = {}
        for variant in (*references, candidate_arm):
            pair = [records[variant], records[variant + "_repeat"]]
            combined[variant] = {
                phase: median(s[phase] for r in pair for s in r["samples"])
                for phase in PHASES}
            combined[variant]["kernel_mean_us"] = mean(
                r["nsys_kernel_us"]["mean"] for r in pair)
            combined[variant]["phase_samples"] = sum(len(r["samples"]) for r in pair)
        key = PHASES[0]
        best = min(float(history[case][key + "_median"]) for history in historical)
        control, candidate = combined[control_arm], combined[candidate_arm]
        history_gain = gain(best, candidate[key])
        matched_gain = gain(control[key], candidate[key])
        arm_dispatch = {a: median(s[key] for s in r["samples"]) for a, r in records.items()}
        repeat_gain = min(gain(best, arm_dispatch[a]) for a in (candidate_arm, candidate_arm + "_repeat"))
        matched_repeat_gain = min(
            gain(arm_dispatch[r], arm_dispatch[c])
            for r in (control_arm, control_arm + "_repeat")
            for c in (candidate_arm, candidate_arm + "_repeat"))
        results.append({"case_id": case, "historical_best_dispatch_us": best,
                        "arms_dispatch_us": arm_dispatch, **combined,
                        "historical_dispatch_gain_pct": history_gain,
                        "matched_dispatch_gain_pct": matched_gain,
                        "worst_candidate_repeat_history_gain_pct": repeat_gain,
                        "worst_candidate_repeat_matched_gain_pct": matched_repeat_gain,
                        "kernel_gain_pct": gain(control["kernel_mean_us"], candidate["kernel_mean_us"]),
                        "dispatch_gate": min(history_gain, matched_gain, repeat_gain,
                                             matched_repeat_gain) > args.dispatch_gain})
        if args.additional_control:
            additional = {}
            for ref in args.additional_control:
                additional[ref] = {
                    "pooled_dispatch_gain_pct": gain(combined[ref][key], candidate[key]),
                    "worst_repeat_dispatch_gain_pct": min(
                        gain(arm_dispatch[r], arm_dispatch[c])
                        for r in (ref, ref + "_repeat")
                        for c in (candidate_arm, candidate_arm + "_repeat")),
                    "kernel_gain_pct": gain(combined[ref]["kernel_mean_us"], candidate["kernel_mean_us"])}
            results[-1]["additional_control_comparisons"] = additional
            results[-1]["dispatch_gate"] &= all(
                min(r["pooled_dispatch_gain_pct"], r["worst_repeat_dispatch_gain_pct"]) > args.dispatch_gain
                for r in additional.values())
    report = {"cases": results, "all_dispatch_gates_pass": all(r["dispatch_gate"] for r in results),
              "candidate_arm": candidate_arm,
              "control_arm": control_arm,
              "required_dispatch_gain_pct": args.dispatch_gain,
              "history_roots": list(map(str, args.history)),
              "scope": "All phase samples retained; pooled phase medians and equal-count ordinary means. Historical runs are unpaired."}
    (args.output or args.root / "dispatch_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
