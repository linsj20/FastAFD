#!/usr/bin/env python3
"""Audit and compare repeated all-rank producer-to-MegaMoE trace summaries."""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean


METRICS = ("producer_end_us", "selector_start_us", "selector_end_us",
           "megamoe_start_us", "megamoe_end_us", "kernel_us")
IDENTITY_FIELDS = ["deviceId", "contextId", "globalPid", "graphId", "correlationId"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    require(isfinite(actual) and isclose(actual, expected, rel_tol=0, abs_tol=1e-8),
            f"Timestamp reconstruction/aggregation mismatch: {context}")


def audit_arm(arm, ranks, samples, label):
    require(arm["rank_count"] == ranks and arm["ranks"] == list(range(ranks))
            and set(arm["per_rank"]) == {str(r) for r in range(ranks)}
            and arm["samples_per_rank"] == samples, f"Rank/call coverage: {label}")
    rank_means, first_calls = {}, []
    for rank in range(ranks):
        trace = arm["per_rank"][str(rank)]
        rows, timestamps = trace["records"], trace["trace_timestamps_ns"]
        identity = trace["graph_identity_audit"]
        calls = identity["calls"]
        require(trace["rank"] == rank and trace["samples"] == samples
                and len(rows) == len(timestamps) == len(calls) == samples,
                f"Rank/call coverage: {label}/{rank}")
        require(identity["fields"] == IDENTITY_FIELDS
                and all(len(c) == 5 and all(type(v) is int for v in c) for c in calls)
                and len({tuple(c) for c in calls}) == samples
                and len({c[0] for c in calls}) == 1,
                f"Missing, duplicate or inconsistent graph identity: {label}/{rank}")
        previous_start = -1
        for ordinal, (row, ts) in enumerate(zip(rows, timestamps)):
            context = f"{label}/{rank}/{ordinal}"
            require(ts["ordinal"] == ordinal, f"Call ordinal: {context}")
            for kind in ("producer", "selector", "megamoe"):
                require(all(type(ts[kind][k]) is int for k in ("start", "end"))
                        and 0 <= ts[kind]["start"] < ts[kind]["end"],
                        f"Invalid kernel timestamps: {context}/{kind}")
            origin = ts["producer"]["start"]
            require(previous_start < origin <= ts["selector"]["start"]
                    <= ts["megamoe"]["start"]
                    and max(ts["producer"]["end"], ts["selector"]["end"])
                    <= ts["megamoe"]["end"], f"Invalid timeline order: {context}")
            previous_start = origin
            expected = {f"{kind}_{edge}_us": (ts[kind][edge] - origin) / 1000
                        for kind, edges in (("producer", ("end",)),
                                            ("selector", ("start", "end")),
                                            ("megamoe", ("start", "end")))
                        for edge in edges}
            expected["kernel_us"] = (ts["megamoe"]["end"] - ts["megamoe"]["start"]) / 1000
            for key in METRICS:
                equal(row[key], expected[key], context + "/" + key)
        rank_means[str(rank)] = {k: mean(row[k] for row in rows) for k in METRICS}
        for key in METRICS:
            equal(trace["means"][key], rank_means[str(rank)][key], f"{label}/{rank}/{key}")
        first_calls.append(rows[0]["megamoe_end_us"])
    require(len(arm["records"]) == samples, f"Aggregate call count: {label}")
    for ordinal, row in enumerate(arm["records"]):
        for key in METRICS:
            equal(row[key], mean(arm["per_rank"][str(r)]["records"][ordinal][key]
                                 for r in range(ranks)), f"{label}/aggregate/{ordinal}/{key}")
    means = {k: mean(v[k] for v in rank_means.values()) for k in METRICS}
    for key in METRICS:
        equal(arm["means"][key], means[key], f"{label}/mean/{key}")
    return {"means": means, "rank_means": rank_means,
            "first_call_mean_us": mean(first_calls)}


def compare(data, control, candidates, context, ranks, samples):
    prefixes = [control, *candidates]
    labels = [a for p in prefixes for a in (p, p + "_repeat")] + context
    require(ranks > 0 and samples > 1 and candidates and len(set(labels)) == len(labels),
            "Require positive ranks, at least two calls and distinct comparison labels")
    require(set(data) == set(labels), "Input arms must exactly match paired and context arms")
    audited = {a: audit_arm(data[a], ranks, samples, a) for a in labels}
    pooled = {p: {k: mean(audited[a]["means"][k] for a in (p, p + "_repeat"))
                  for k in METRICS} for p in prefixes}
    comparisons = {}
    for candidate in candidates:
        delta = {k: pooled[candidate][k] - pooled[control][k] for k in METRICS}
        rank_delta = {str(r): mean(audited[a]["rank_means"][str(r)]["megamoe_end_us"]
                                  for a in (candidate, candidate + "_repeat"))
                      - mean(audited[a]["rank_means"][str(r)]["megamoe_end_us"]
                             for a in (control, control + "_repeat")) for r in range(ranks)}
        first = (mean(audited[a]["first_call_mean_us"] for a in (candidate, candidate + "_repeat"))
                 - mean(audited[a]["first_call_mean_us"] for a in (control, control + "_repeat"))) / samples
        comparisons[candidate] = {
            "delta_us": delta,
            "e2e_regression_percent": 100 * delta["megamoe_end_us"] / pooled[control]["megamoe_end_us"],
            "rank_deltas_us": rank_delta,
            "ranks_improved": sum(v < 0 for v in rank_delta.values()),
            "order_deltas_us": [audited[candidate + suffix]["means"]["megamoe_end_us"]
                                - audited[control + suffix]["means"]["megamoe_end_us"]
                                for suffix in ("", "_repeat")],
            "first_call_contribution_us": first,
            "remaining_calls_diagnostic_delta_us": (delta["megamoe_end_us"] - first) * samples / (samples - 1),
        }
    return {"status": "PASS_TIMESTAMP_AND_AGGREGATION_AUDIT",
            "scope": "Mean rank-local producer-to-kernel-end spans; all calls retained. "
                     "No cross-rank clock alignment. Context arms are unpaired. "
                     "First-call accounting is diagnostic, not sample filtering. "
                     "Scheduler, correctness, capture configuration and profiles require separate audits.",
            "ordinary_graphs": len(labels) * ranks * samples,
            "rank_count": ranks, "samples_per_rank_pass": samples,
            "per_arm_means": {a: v["means"] for a, v in audited.items()},
            "paired_means": pooled, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--context", action="append", default=[])
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.input.read_text()), args.control, args.candidate,
                     args.context, args.ranks, args.samples)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "ordinary_graphs", "comparisons")}))


if __name__ == "__main__":
    main()
