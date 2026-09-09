#!/usr/bin/env python3
"""Summarize MegaMoE ABBA Nsight kernel timings or legacy harness log timings."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


def summarize_nsys(root: Path) -> dict:
    records = {}
    contracts = set()
    for label in ("control_trial0", "candidate_trial1", "candidate_trial2", "control_trial3"):
        paths = list((root / label / "nsys").glob("*.cuda_gpu_kern_sum.csv"))
        if len(paths) != 1:
            raise ValueError(f"{root / label}: expected exactly one Nsight kernel CSV")
        raw = paths[0].read_text()
        header = raw.index("Time (%),")
        rows = [row for row in csv.DictReader(io.StringIO(raw[header:]))
                if "sm100_fp8_fp4_mega_moe_impl" in row["Name"]]
        if len(rows) != 1 or int(rows[0]["Instances"]) <= 0:
            raise ValueError(f"{paths[0]}: expected one measured MegaMoE specialization")
        row = rows[0]
        contracts.add((row["Name"], int(row["Instances"])))
        records[label] = {
            key: float(row[column]) / 1000
            for key, column in (("median_us", "Med (ns)"), ("mean_us", "Avg (ns)"),
                                ("min_us", "Min (ns)"), ("max_us", "Max (ns)"))
        }
    if len(contracts) != 1:
        raise ValueError(f"{root}: kernel specialization or sample counts differ")
    control = (records["control_trial0"]["median_us"] + records["control_trial3"]["median_us"]) / 2
    candidate = (records["candidate_trial1"]["median_us"] + records["candidate_trial2"]["median_us"]) / 2
    return {
        "timing_source": "Nsight GPU kernel duration",
        "instances_per_capture": next(iter(contracts))[1],
        "records": records, "control_us": control, "candidate_us": candidate,
        "delta_us": candidate - control,
        "delta_percent": (candidate / control - 1) * 100,
        "control_drift_us": records["control_trial3"]["median_us"] - records["control_trial0"]["median_us"],
    }


def parse_records(log_path: Path) -> list[tuple[str, dict[str, object]]]:
    records: list[tuple[str, dict[str, object]]] = []
    current_case: str | None = None
    for raw_line in log_path.read_text().splitlines():
        if raw_line.startswith("CASE="):
            current_case = raw_line.split(maxsplit=1)[0].removeprefix("CASE=")
        elif current_case is not None and raw_line.startswith("{"):
            payload = json.loads(raw_line)
            if payload.get("status") != "ok":
                raise RuntimeError(f"case {current_case} did not report status=ok")
            records.append((current_case, payload))
            current_case = None
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument("--nsys-root", nargs="+", type=Path,
                        help="ABBA artifact roots containing trial directories")
    args = parser.parse_args()
    if bool(args.logs) == bool(args.nsys_root):
        parser.error("provide either Slurm logs or --nsys-root directories")
    if args.nsys_root:
        print(json.dumps({str(root): summarize_nsys(root) for root in args.nsys_root}, indent=2))
        return

    expected = ["control_trial0", "candidate_trial1", "candidate_trial2", "control_trial3"]
    records: list[tuple[str, dict[str, object]]] = []
    for log_path in args.logs:
        log_records = parse_records(log_path)
        labels = [label for label, _ in log_records]
        if labels != expected:
            raise RuntimeError(
                f"{log_path}: expected completed ABBA order {expected}, got {labels}"
            )
        records.extend(log_records)

    dtypes = {str(payload["expert_weight_dtype"]) for _, payload in records}
    if len(dtypes) != 1:
        raise RuntimeError(f"inconsistent expert weight dtypes: {sorted(dtypes)}")

    metric_names = set.intersection(
        *(set(payload["exact_qwen"]) for _, payload in records)
    )
    if not metric_names:
        raise RuntimeError("ABBA records have no common exact_qwen metric")

    summary = {
        "expert_weight_dtype": dtypes.pop(),
        "logs": [str(path) for path in args.logs],
        "metrics": {},
    }
    for metric_name in sorted(metric_names):
        control = [
            float(payload["exact_qwen"][metric_name]["milliseconds"])
            for label, payload in records
            if label.startswith("control_")
        ]
        candidate = [
            float(payload["exact_qwen"][metric_name]["milliseconds"])
            for label, payload in records
            if label.startswith("candidate_")
        ]
        control_mean = sum(control) / len(control)
        candidate_mean = sum(candidate) / len(candidate)
        summary["metrics"][metric_name] = {
            "control_samples_us": [value * 1000 for value in control],
            "candidate_samples_us": [value * 1000 for value in candidate],
            "control_mean_us": control_mean * 1000,
            "candidate_mean_us": candidate_mean * 1000,
            "delta_us": (candidate_mean - control_mean) * 1000,
            "delta_percent": (candidate_mean / control_mean - 1.0) * 100,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
