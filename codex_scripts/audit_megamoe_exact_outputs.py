#!/usr/bin/env python3
"""Audit a completed source-comparison job against explicit exact-check coverage.

The expectation file maps each comparison record to its case names and ranks,
and can require explicit runtime configuration tuples in the emitted log.
Record names, source paths, scheduler success and every unique comparison must
match. Existing logs and results are never filtered or rewritten.
"""

import argparse
import json
from pathlib import Path
import re

from summarize_megamoe_abba import parse_records


def read(path):
    return json.loads(path.read_text())


def audit(root, expectation):
    env, job = read(root / "env.json"), read(root / "job.json")
    state = read(root / "scheduler.json")
    if (str(state["job"]) != str(job["job"]) or state["state"] != "COMPLETED"
            or state["exit_code"] != "0:0"):
        raise ValueError("Exact check requires the recorded job to complete successfully")
    log = root / f"compare-{job['job']}.out"
    labels = env["MEGAMOE_ABBA_ORDER"].split()
    records = parse_records(log)
    if [label for label, _ in records] != labels or len(set(labels)) != len(labels):
        raise ValueError("Completed exact records differ from the submitted order")
    if any(record["status"] != "ok" for _, record in records):
        raise ValueError("Exact primary record did not pass")
    compare_labels = {label for label in labels if not label.startswith("control_")}
    if set(expectation) != compare_labels or not compare_labels:
        raise ValueError("Explicit coverage required for every comparison record")
    parts = re.split(r"CASE=(\S+) SOURCE=(\S+)\n", log.read_text())
    if parts[1::3] != labels:
        raise ValueError("Source log blocks differ from the expected records")
    results = {}
    for index in range(1, len(parts), 3):
        label, source, body = parts[index:index + 3]
        source_key = "FASTAFD_" + label.split("_", 1)[0].upper() + "_SOURCE"
        if source != env[source_key]:
            raise ValueError(f"Actual exact-check source mismatch: {label}")
        actual = re.findall(r"MEGAMOE_RESULT_COMPARE_OK case=(\S+) rank=(\d+)", body)
        if label not in compare_labels:
            if actual:
                raise ValueError(f"Unexpected comparison checks in recording arm: {label}")
            continue
        spec = expectation[label]
        cases, ranks = spec["cases"], list(map(str, spec["ranks"]))
        if (not cases or not ranks or len(set(cases)) != len(cases)
                or len(set(ranks)) != len(ranks)):
            raise ValueError(f"Invalid expected case/rank coverage: {label}")
        expected = {(case, rank) for case in cases for rank in ranks}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"Exact-check coverage mismatch: {label}; "
                             f"missing={sorted(expected - set(actual))}; "
                             f"unexpected={sorted(set(actual) - expected)}; "
                             f"duplicates={len(actual) - len(set(actual))}")
        required_configs = spec.get("configs", [])
        observed_configs = [dict((key, int(value)) for key, value in
                                 re.findall(r"\b(\w+)=(-?\d+)\b", header + "," + detail))
                            for header, detail in re.findall(
                                r"MegaMoEConfig\(([^)\n]*)\): MegaMoEConfig\(([^)\n]*)\)", body)]
        for config in required_configs:
            if (not isinstance(config, dict) or not config or
                    any(not isinstance(key, str) or not key.isidentifier() or
                        type(value) is not int for key, value in config.items())):
                raise ValueError(f"Invalid expected runtime configuration: {label}: {config}")
            if not any(all(actual.get(key) == value for key, value in config.items())
                       for actual in observed_configs):
                raise ValueError(f"Missing actual runtime configuration: {label}: {config}")
        results[label] = {"source": source, "cases": cases,
                          "ranks": spec["ranks"], "comparisons": len(actual)}
        if required_configs:
            results[label]["matched_runtime_configs"] = required_configs
    return {"status": "PASS", "job": str(job["job"]), "elapsed": state["elapsed"],
            "records": len(records), "comparisons": sum(r["comparisons"] for r in results.values()),
            "by_comparison_record": results,
            "scope": "Exact input/output checks for the explicitly listed cases and ranks; "
                     "not serving alignment or performance evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, read(args.expected))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "by_comparison_record"}))


if __name__ == "__main__":
    main()
