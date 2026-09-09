#!/usr/bin/env python3
"""Fit fixed-global-work EP scaling from MegaMoE decomposition records."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def fit(ep_sizes: list[int], times: list[float]) -> dict:
    x = [1.0 / ep for ep in ep_sizes]
    xm, ym = statistics.mean(x), statistics.mean(times)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, times)) / sum(
        (a - xm) ** 2 for a in x
    )
    intercept = ym - slope * xm
    residuals = [y - (slope * a + intercept) for a, y in zip(x, times)]
    total = sum((y - ym) ** 2 for y in times)
    return {
        "model": "T(EP) = A / EP + C",
        "A_us_times_ep": slope,
        "C_us": intercept,
        "r_squared": 1 - sum(r * r for r in residuals) / total if total else None,
        "residuals_us": dict(zip(ep_sizes, residuals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = sorted(
        [(path, json.loads(path.read_text())) for path in args.records],
        key=lambda pair: pair[1]["contract"]["ep_size"],
    )
    ep_sizes = [record["contract"]["ep_size"] for _, record in records]
    if len(records) < 3 or len(set(ep_sizes)) != len(ep_sizes):
        raise SystemExit("need at least three distinct EP degrees")
    fixed_fields = (
        "global_tokens_per_launch", "routed_slots_per_launch", "routes_per_expert",
        "hidden", "intermediate", "experts", "top_k", "activation_dtype",
        "weight_dtype", "num_sms", "routes_prepared", "rank_gated_combine",
        "route_ready_dispatch", "rank_ready_route_publish", "input_route_distribution",
    )
    contract = {key: records[0][1]["contract"][key] for key in fixed_fields}
    for path, record in records:
        if {key: record["contract"][key] for key in fixed_fields} != contract:
            raise SystemExit(f"fixed-global-work contract differs: {path}")

    rows = []
    for path, record in records:
        row = {
            "ep": record["contract"]["ep_size"],
            "record": str(path),
            "rows_per_rank": record["contract"]["rows_per_ep_rank_per_microbatch"],
            "kernel_us": record["nsys_kernel_us"]["median"],
            "kernel_range_us": [record["nsys_kernel_us"][k] for k in ("min", "max")],
        }
        for name in (
            "compute_span_us", "dispatch_span_us", "combine_span_us",
            "post_compute_combine_exposure_us", "post_combine_tail_us",
        ):
            row[name] = record["summary_us"][name]["median"]
            row[name + "_range"] = [record["summary_us"][name][k] for k in ("min", "max")]
        rows.append(row)

    fits = {name: fit(ep_sizes, [row[name] for row in rows])
            for name in ("kernel_us", "compute_span_us")}
    doubling = []
    by_ep = {row["ep"]: row for row in rows}
    for ep, lower in by_ep.items():
        if 2 * ep not in by_ep:
            continue
        upper = by_ep[2 * ep]
        doubling.append({
            "ep_from": ep, "ep_to": 2 * ep,
            "kernel_time_ratio": upper["kernel_us"] / lower["kernel_us"],
            "compute_time_ratio": upper["compute_span_us"] / lower["compute_span_us"],
            "pairwise_C_us": 2 * upper["kernel_us"] - lower["kernel_us"],
        })
    result = {
        "fixed_contract": contract, "measurements": rows,
        "fits": fits, "doubling": doubling,
        "interpretation": (
            "Nsight measures uninstrumented kernel wall time; IKET phase spans are "
            "measured separately and overlap. C estimates non-scaling exposed cost "
            "under this model, including setup and synchronization. It is not pure "
            "NVLink latency or proof of an infinite-EP floor. Inspect residuals, "
            "sample ranges, and pairwise intercepts before accepting the model."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
