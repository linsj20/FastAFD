#!/usr/bin/env python3
"""Summarize MegaMoE ABBA Nsight kernel timings or legacy harness log timings."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median


def summarize_profiles(root: Path) -> dict:
    """Summarize flattened rank-0 captures; preserve missing arms and all samples."""
    env = json.loads((root / "env.json").read_text())
    result = {}
    for arm in env["MEGAMOE_ABBA_PROFILE_VARIANTS"].split():
        phase_path = root / arm / "decomposition.json"
        kernel_path = root / arm / "megamoe-rank0.cuda_gpu_kern_sum.csv"
        if not phase_path.exists() or not kernel_path.exists():
            result[arm] = {"status": "missing_capture"}
            continue
        samples = json.loads(phase_path.read_text())["samples"]
        if len(samples) != int(env["MEGAMOE_IKET_ITERATIONS"]):
            raise ValueError(f"{phase_path}: unexpected phase sample count")
        keys = (
            "pre_compute_dispatch_exposure_us",
            "post_compute_combine_exposure_us",
            "compute_span_us",
        )
        exposure = [sample[keys[0]] + sample[keys[1]] for sample in samples]
        dispatch = [s["compute_start_us"] - s["dispatch_start_us"] for s in samples]
        combine = [s["combine_end_us"] - s["compute_end_us"] for s in samples]
        direct_exposure = [d + c for d, c in zip(dispatch, combine)]
        raw = kernel_path.read_text()
        rows = [
            row for row in csv.DictReader(io.StringIO(raw[raw.index("Time (%),"):]))
            if "sm100_fp8_fp4_mega_moe_impl" in row["Name"]
        ]
        if (len(rows) != 1
                or int(rows[0]["Instances"]) != int(env["MEGAMOE_BASELINE_ITERATIONS"])):
            raise ValueError(f"{kernel_path}: unexpected kernel or sample count")
        row = rows[0]
        result[arm] = {
            **{key: median(sample[key] for sample in samples) for key in keys},
            "exposed_sum_us": median(exposure),
            "exposed_sum_samples_us": exposure,
            "phase_samples_retained": len(samples),
            "direct_dispatch_exposure_us": median(dispatch),
            "direct_combine_exposure_us": median(combine),
            "direct_exposed_sum_us": median(direct_exposure),
            "direct_exposed_sum_samples_us": direct_exposure,
            "post_combine_tail_us": median(s["post_combine_tail_us"] for s in samples),
            "kernel": {
                "scope": "full rank-0 MegaMoE kernel, including post-CombineEnd tail",
                "samples_retained": int(row["Instances"]),
                **{
                    key: float(row[column]) / 1000
                    for key, column in (
                        ("mean_us", "Avg (ns)"), ("median_us", "Med (ns)"),
                        ("min_us", "Min (ns)"), ("max_us", "Max (ns)"),
                    )
                },
            },
        }
    return result


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
    parser.add_argument("--profiles-root", nargs="+", type=Path,
                        help="Case roots with env.json and flattened per-arm rank-0 captures")
    parser.add_argument("--order", nargs="+", default=[
        "control_trial0", "candidate_trial1", "candidate_trial2", "control_trial3"
    ], help="Expected ordered labels for harness log comparisons")
    args = parser.parse_args()
    if sum(map(bool, (args.logs, args.nsys_root, args.profiles_root))) != 1:
        parser.error("provide Slurm logs, --nsys-root, or --profiles-root")
    if args.profiles_root:
        print(json.dumps({str(root): summarize_profiles(root) for root in args.profiles_root}, indent=2))
        return
    if args.nsys_root:
        print(json.dumps({str(root): summarize_nsys(root) for root in args.nsys_root}, indent=2))
        return

    expected = args.order
    if (len(set(expected)) != len(expected)
            or any(not label.startswith(("control_", "candidate_", "reference_", "alternate_")) for label in expected)
            or not any(label.startswith("control_") for label in expected)
            or not any(label.startswith("candidate_") for label in expected)):
        raise ValueError("order requires unique control/candidate/reference/alternate labels and both control and candidate")
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
            "control_first_to_last_drift_percent": (control[-1] / control[0] - 1.0) * 100,
            "control_range_percent": (max(control) / min(control) - 1.0) * 100,
        }
        reference = [
            float(payload["exact_qwen"][metric_name]["milliseconds"])
            for label, payload in records if label.startswith("reference_")
        ]
        if reference:
            reference_mean = sum(reference) / len(reference)
            summary["metrics"][metric_name].update({
                "reference_samples_us": [value * 1000 for value in reference],
                "reference_mean_us": reference_mean * 1000,
                "reference_range_percent": (max(reference) / min(reference) - 1.0) * 100,
                "control_vs_reference_percent": (control_mean / reference_mean - 1.0) * 100,
                "candidate_vs_reference_percent": (candidate_mean / reference_mean - 1.0) * 100,
            })
        alternate = [
            float(payload["exact_qwen"][metric_name]["milliseconds"])
            for label, payload in records if label.startswith("alternate_")
        ]
        if alternate:
            alternate_mean = sum(alternate) / len(alternate)
            summary["metrics"][metric_name].update({
                "alternate_samples_us": [value * 1000 for value in alternate],
                "alternate_mean_us": alternate_mean * 1000,
                "alternate_range_percent": (max(alternate) / min(alternate) - 1.0) * 100,
                "alternate_vs_control_percent": (alternate_mean / control_mean - 1.0) * 100,
            })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
