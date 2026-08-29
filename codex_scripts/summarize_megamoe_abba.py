#!/usr/bin/env python3
"""Summarize control/candidate MegaMoE ABBA JSON records from a Slurm log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

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
