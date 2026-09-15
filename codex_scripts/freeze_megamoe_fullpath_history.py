#!/usr/bin/env python3
"""Freeze per-case historical full-path means from explicitly listed raw timelines.

The plan lists sources with model, root, timeline, and full_sweep fields and
expected_cases per model. Partial sources must have a matching sampled-arm env.
All ordinary records contribute to their capture mean. No phase minima are used.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

from compare_megamoe_fullchain import timeline_means
from compare_megamoe_sweeps import CONTRACT_KEYS, load


def read(path):
    return json.loads(path.read_text())


def source_records(spec):
    root = Path(spec["root"])
    if spec["full_sweep"]:
        return load(root)
    rows = list(csv.DictReader((root / "decomposition.csv").open()))
    audit = read(root / "audit_summary.json")
    expected = read(root / "env.json")["MEGAMOE_SWEEP_CASES"].split(",")
    if (not rows or len(rows) != len(expected)
            or {r["case_id"] for r in rows} != set(expected)
            or audit["artifact_contracts_checked"] != len(rows)
            or set(expected).intersection(audit["missing"])):
        raise ValueError(f"Unaudited sampled history: {root}")
    result = {}
    for row in rows:
        path = root / row["artifact"]
        artifact = read(path)
        timings = list(read(path.parent / "baseline.json")["exact_qwen"].values())
        if len(timings) != 1:
            raise ValueError(f"Ambiguous ordinary timing: {path}")
        result[row["case_id"]] = {
            "artifact": str(path), "contract": artifact["contract"],
            "ordinary_samples": timings[0]["timing_iterations"],
        }
    return result


def freeze(plan):
    models, contracts, seen = {}, {}, set()
    for spec in plan["sources"]:
        identity = (spec["model"], spec["root"], spec["timeline"])
        if identity in seen:
            raise ValueError(f"Duplicate historical source: {identity}")
        seen.add(identity)
        records = source_records(spec)
        timeline = read(Path(spec["timeline"]))
        if set(timeline) != set(records):
            raise ValueError(f"Timeline coverage differs from source: {identity}")
        model = models.setdefault(spec["model"], {})
        for case, record in records.items():
            key = (spec["model"], case)
            contract = record["contract"]
            prior = contracts.setdefault(key, contract)
            if any(prior[k] != contract[k] for k in CONTRACT_KEYS):
                raise ValueError(f"Incompatible historical input contracts: {key}")
            means = timeline_means(timeline[case], record)
            model.setdefault(case, []).append({
                "fullpath_us": means["megamoe_end_us"],
                "kernel_us": means["kernel_us"],
                "launch_us": means["megamoe_start_us"],
                "samples": record["ordinary_samples"],
                "root": spec["root"], "timeline": spec["timeline"],
                "database": timeline[case]["database"],
                "artifact": record["artifact"],
            })
    if set(models) != set(plan["expected_cases"]):
        raise ValueError("Historical model coverage mismatch")
    result = {}
    for model, cases in models.items():
        if set(cases) != set(plan["expected_cases"][model]):
            raise ValueError(f"Historical case coverage mismatch: {model}")
        result[model] = {
            case: {"best": min(options, key=lambda r: r["fullpath_us"]),
                   "observations": options}
            for case, options in sorted(cases.items())}
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "historical_cutoff": plan["historical_cutoff"],
        "scope": "Minimum capture mean per case over the explicitly listed historical "
                 "sources; every raw ordinary record retained. Unpaired rank0 "
                 "producer-start to MegaMoE-end. Sample counts can differ; this "
                 "does not replace matched repeated controls or establish "
                 "slowest-rank completion. No E2E value is inferred from IKET phases.",
        "sources": plan["sources"], "models": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze(read(args.plan))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({m: {"cases": len(cases),
                         "observations": sum(len(c["observations"]) for c in cases.values())}
                      for m, cases in report["models"].items()}))


if __name__ == "__main__":
    main()
