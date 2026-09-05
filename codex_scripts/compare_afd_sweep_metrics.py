#!/usr/bin/env python3
"""Compare a partial or complete AFD metric sweep with a paired baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


CASE_RE = re.compile(r"-fep(?P<ep>\d+)-r(?P<ratio>\d+)-atp\d+-b(?P<batch>\d+)-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-cases", type=int)
    return parser.parse_args()


def load_metrics(root: Path) -> dict[str, dict]:
    if not root.is_dir():
        raise RuntimeError(f"metric directory does not exist: {root}")
    metrics: dict[str, dict] = {}
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id != path.stem:
            raise RuntimeError(f"case_id/path mismatch: {path}")
        if case_id in metrics:
            raise RuntimeError(f"duplicate case_id: {case_id}")
        for field in ("cuda_wall_time_mean_ms", "tps_per_active_gpu"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"invalid {field} for {case_id}: {value!r}")
        metrics[case_id] = record
    return metrics


def geometric_delta(factors: list[float]) -> float:
    return math.expm1(sum(math.log(value) for value in factors) / len(factors)) * 100.0


def topology(record: dict) -> str:
    allocated_gpus = int(record["allocated_gpus"])
    if allocated_gpus % 4:
        raise RuntimeError(f"allocated_gpus is not a whole tray count: {allocated_gpus}")
    return f"{allocated_gpus // 4}t"


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)
    if args.expected_cases is not None and len(baseline) != args.expected_cases:
        raise RuntimeError(
            f"baseline has {len(baseline)} cases, expected {args.expected_cases}"
        )
    unpaired = sorted(candidate.keys() - baseline.keys())
    if unpaired:
        raise RuntimeError(f"candidate cases absent from baseline: {', '.join(unpaired)}")

    rows: list[dict] = []
    by_topology: dict[str, list[dict]] = defaultdict(list)
    for case_id in sorted(candidate):
        old = baseline[case_id]
        new = candidate[case_id]
        match = CASE_RE.search(case_id)
        if match is None:
            raise RuntimeError(f"unrecognized case_id: {case_id}")
        row = {
            "case_id": case_id,
            "topology": topology(new),
            "ep": int(match.group("ep")),
            "ratio": int(match.group("ratio")),
            "batch": int(match.group("batch")),
            "old_latency": float(old["cuda_wall_time_mean_ms"]),
            "new_latency": float(new["cuda_wall_time_mean_ms"]),
            "latency_factor": float(new["cuda_wall_time_mean_ms"])
            / float(old["cuda_wall_time_mean_ms"]),
            "old_tps": float(old["tps_per_active_gpu"]),
            "new_tps": float(new["tps_per_active_gpu"]),
            "tps_factor": float(new["tps_per_active_gpu"])
            / float(old["tps_per_active_gpu"]),
            "samples": int(new["sample_count"]),
            "outliers": int(new["outlier_count"]),
        }
        rows.append(row)
        by_topology[row["topology"]].append(row)

    print(
        "case_id\ttopology\tep\tratio\tbatch\told_ms\tnew_ms\tlatency_delta_pct"
        "\told_tps_gpu\tnew_tps_gpu\ttps_delta_pct\tsamples\toutliers"
    )
    for row in rows:
        print(
            f"{row['case_id']}\t{row['topology']}\t{row['ep']}\t{row['ratio']}"
            f"\t{row['batch']}\t{row['old_latency']:.9f}\t{row['new_latency']:.9f}"
            f"\t{(row['latency_factor'] - 1.0) * 100.0:+.6f}"
            f"\t{row['old_tps']:.9f}\t{row['new_tps']:.9f}"
            f"\t{(row['tps_factor'] - 1.0) * 100.0:+.6f}"
            f"\t{row['samples']}\t{row['outliers']}"
        )

    print(
        f"summary paired={len(rows)} baseline={len(baseline)} "
        f"remaining={len(baseline) - len(rows)}"
    )
    if not rows:
        return
    print(
        "overall "
        f"latency_geomean_delta_pct={geometric_delta([r['latency_factor'] for r in rows]):+.6f} "
        f"tps_geomean_delta_pct={geometric_delta([r['tps_factor'] for r in rows]):+.6f} "
        f"latency_faster={sum(r['latency_factor'] < 1.0 for r in rows)} "
        f"latency_slower={sum(r['latency_factor'] > 1.0 for r in rows)} "
        f"worst_latency_case={max(rows, key=lambda r: r['latency_factor'])['case_id']} "
        f"worst_latency_delta_pct={(max(r['latency_factor'] for r in rows) - 1.0) * 100.0:+.6f}"
    )
    for name in sorted(by_topology, key=lambda value: int(value[:-1])):
        group = by_topology[name]
        print(
            f"topology={name} paired={len(group)} "
            f"latency_geomean_delta_pct={geometric_delta([r['latency_factor'] for r in group]):+.6f} "
            f"tps_geomean_delta_pct={geometric_delta([r['tps_factor'] for r in group]):+.6f}"
        )


if __name__ == "__main__":
    main()
