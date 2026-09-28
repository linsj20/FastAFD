#!/usr/bin/env python3
"""Align fixed-call Nsight router/MegaMoE timelines to each producer's start."""

import argparse
import csv
import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path
from statistics import mean, median


def audit_joined_preparation(rows, producer, selector, *, barrier_only=False):
    """Validate the explicit collective/reset/collective graph prefix."""
    rows = sorted(rows)
    if (len(rows) != (2 if barrier_only else 4)
            or any("decomposition_collective_barrier(" not in rows[i][2]
                   for i in (0, 1 if barrier_only else 3))
            or (not barrier_only and any("at::native::FillFunctor<int>" not in rows[i][2] for i in (1, 2)))
            or any(row[5] != producer[5] or row[0] >= row[1]
                   or row[1] > selector[0] for row in rows)
            or rows[-1][0] > producer[0]):
        raise ValueError("Invalid joined collective preparation before metadata")
    # Nsight may report adjacent graph-node envelopes overlapping slightly.
    # The final barrier must finish before the selector consumes route metadata.
    return rows


def audit_event_creation(database, first_producer_ns, last_kernel_end_ns):
    """Reject lazy event allocation overlapping the measured replay window."""
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        if not connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]:
            raise ValueError(f"{database}: event-pool audit requires captured CUDA runtime calls")
        allocations = connection.execute(
            "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaEventCreate%' "
            "AND r.start < ? AND r.end > ? ORDER BY r.start",
            (last_kernel_end_ns, first_producer_ns)).fetchall()
    if allocations:
        raise ValueError(f"{database}: {len(allocations)} CUDA event creations overlap "
                         "the measured replay window; initialize the pool before warmups")
    return {"status": "PASS", "event_creations_overlapping_window": 0,
            "first_producer_ns": first_producer_ns, "last_kernel_end_ns": last_kernel_end_ns,
            "scope": "Same-report runtime calls and GPU replay window; no cross-rank alignment"}


def average_rank_timelines(by_rank):
    """Average local intervals, never raw timestamps from different ranks."""
    if not by_rank or sorted(by_rank) != list(range(len(by_rank))):
        raise ValueError("Require every unique rank from zero through world_size-1")
    arms = list(by_rank[0])
    if not arms or any(set(data) != set(arms) for data in by_rank.values()):
        raise ValueError("Rank traces must cover identical nonempty arm sets")
    results = {}
    for arm in arms:
        traces = [by_rank[rank][arm] for rank in sorted(by_rank)]
        samples = traces[0]["samples"]
        if samples < 1 or not traces[0]["records"]:
            raise ValueError(f"Require nonempty rank-local samples: {arm}")
        keys = set(traces[0]["records"][0])
        for rank, trace in enumerate(traces):
            if (trace["rank"] != rank or trace["samples"] != samples
                    or len(trace["records"]) != samples or samples < 1
                    or any(set(record) != keys or
                           any(not math.isfinite(value) for value in record.values())
                           for record in trace["records"])):
                raise ValueError(f"Incomplete or inconsistent rank/call coverage: {arm}, rank {rank}")
        records = [{key: mean(trace["records"][i][key] for trace in traces)
                    for key in traces[0]["records"][0]} for i in range(samples)]
        rank_e2e = [mean(row["megamoe_end_us"] for row in trace["records"])
                    for trace in traces]
        results[arm] = {
            "aggregation": "Arithmetic mean of rank-local producer-relative intervals per call; "
                           "buffer preparation excluded; raw cross-rank timestamps are not averaged",
            "rank_count": len(traces), "ranks": list(sorted(by_rank)),
            "samples_per_rank": samples, "records": records,
            "means": {key: mean(row[key] for row in records) for key in records[0]},
            "rank_e2e_means_us": rank_e2e,
            "rank_e2e_mean_range_us": {"min": min(rank_e2e), "max": max(rank_e2e)},
            "per_rank": {rank: trace for rank, trace in enumerate(traces)},
        }
    return results


def summarize_rank_mean(args):
    common = [str(args.decomposition)]
    if args.job is not None:
        common.append(args.job)
    for enabled, flag in ((args.flat_trials, "--flat-trials"), (args.sweep, "--sweep"),
                          (args.require_graph_identity, "--require-graph-identity"),
                          (args.require_preinitialized_events, "--require-preinitialized-events"),
                          (args.joined_collective_preparation, "--joined-collective-preparation"),
                          (args.joined_collective_barrier_only, "--joined-collective-barrier-only"),
                          (args.retain_timestamps, "--retain-timestamps")):
        if enabled:
            common.append(flag)
    common += ["--arms", *args.arms, "--samples", str(args.samples)]
    if args.prefix_kernel:
        common += ["--prefix-kernel", args.prefix_kernel]
    by_rank = {}
    for rank in range(args.rank_mean):
        output = args.output.with_name(f"{args.output.stem}_rank{rank}.json")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *common,
                        "--rank", str(rank), "--output", str(output)],
                       check=True, stdout=subprocess.DEVNULL)
        by_rank[rank] = json.loads(output.read_text())
    results = average_rank_timelines(by_rank)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    rows = []
    for arm, result in results.items():
        row = {"case_id": arm, "rank_count": result["rank_count"],
               "samples_per_rank": result["samples_per_rank"],
               "e2e_mean_us": result["means"]["megamoe_end_us"],
               "kernel_mean_us": result["means"]["kernel_us"],
               "rank_e2e_mean_min_us": result["rank_e2e_mean_range_us"]["min"],
               "rank_e2e_mean_max_us": result["rank_e2e_mean_range_us"]["max"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.csv:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decomposition", type=Path)
    parser.add_argument("job", nargs="?")
    parser.add_argument("--flat-trials", action="store_true",
                        help="Read ROOT/TRIAL/nsys rather than decomposition/ARM/JOB/ranks/rank-RANK/nsys")
    parser.add_argument("--sweep", action="store_true",
                        help="Read all collected cases from ROOT/decomposition.csv, using each baseline's sample count")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Also write per-arm interval distributions")
    parser.add_argument("--arms", nargs="+", default=["control", "reference", "alternate", "candidate"])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank-local trace to read; clocks are not aligned across ranks")
    parser.add_argument("--rank-mean", type=int, metavar="WORLD_SIZE",
                        help="Average all ranks' local intervals, retaining every rank/call trace")
    parser.add_argument("--require-graph-identity", action="store_true",
                        help="Require matching device/context/process/graph/correlation IDs per workload call")
    parser.add_argument("--retain-timestamps", action="store_true",
                        help="Retain raw database timestamps; this does not establish cross-rank clock alignment")
    parser.add_argument("--require-preinitialized-events", action="store_true",
                        help="Reject CUDA event creation inside the measured replay window")
    parser.add_argument("--joined-collective-preparation", action="store_true",
                        help="Require collective/two resets/collective in every workload graph")
    parser.add_argument("--joined-collective-barrier-only", action="store_true",
                        help="Require two preparation collectives without payload resets")
    parser.add_argument("--prefix-kernel",
                        help="Also measure a required kernel before each producer, matched by name substring")
    args = parser.parse_args()
    if args.rank < 0:
        parser.error("--rank must be nonnegative")
    if ((args.joined_collective_preparation or args.joined_collective_barrier_only)
            and not args.require_graph_identity):
        parser.error("--joined-collective-preparation requires --require-graph-identity")
    if args.joined_collective_preparation and args.joined_collective_barrier_only:
        parser.error("Choose reset or barrier-only preparation")
    if args.sweep and (args.flat_trials or args.job is not None):
        parser.error("--sweep cannot be combined with a job or --flat-trials")
    if not args.sweep and not args.flat_trials and args.job is None:
        parser.error("job is required for decomposition captures")
    if args.rank_mean is not None:
        if args.rank_mean < 1 or args.rank != 0:
            parser.error("--rank-mean requires a positive world size and no nonzero --rank")
        summarize_rank_mean(args)
        return
    root = args.decomposition.resolve()
    results = {}
    captures = []
    rank_database = Path(f"ranks/rank-{args.rank}/nsys/megamoe-rank{args.rank}.sqlite")
    if args.sweep:
        with (root / "decomposition.csv").open() as stream:
            sweep_rows = list(csv.DictReader(stream))
        if not sweep_rows or len({row["case_id"] for row in sweep_rows}) != len(sweep_rows):
            raise ValueError("Require nonempty, unique collected sweep cases")
        for row in sweep_rows:
            run = (root / row["artifact"]).parent
            baseline = json.loads((run / "baseline.json").read_text())
            timings = list(baseline["exact_qwen"].values())
            if len(timings) != 1:
                raise ValueError(f"Expected one baseline timing record: {run}")
            captures.append((row["case_id"], run / rank_database,
                             timings[0]["timing_iterations"]))
    else:
        for arm in args.arms:
            database = (root / arm / f"nsys/megamoe-rank{args.rank}.sqlite" if args.flat_trials else
                        root / arm / args.job / rank_database)
            captures.append((arm, database, args.samples))
    for arm, database, samples in captures:
        if samples < 1:
            raise ValueError(f"Invalid timing sample count: {database}")
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            identity_columns = (",k.deviceId,k.contextId,k.globalPid,k.graphId,k.correlationId"
                                if args.require_graph_identity else "")
            rows = connection.execute(
                "SELECT k.start,k.end,s.value,k.gridX,k.blockX,k.streamId" + identity_columns + " "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
                "ORDER BY k.start").fetchall()
        patterns = {"producer": "gate_topk_fused_tc_kernel", "selector": "gate_topk_select",
                    "megamoe": "sm100_fp8_fp4_mega_moe_impl"}
        groups = {key: [row for row in rows if pattern in row[2]] for key, pattern in patterns.items()}
        dispatches = [row for row in rows if "mega_moe_dispatch<" in row[2]]
        if dispatches and len(dispatches) != samples:
            raise ValueError(f"{database}: expected one generation-specialized dispatch per call")
        if args.prefix_kernel:
            groups["prefix"] = [row for row in rows if args.prefix_kernel in row[2]]
        if any(len(group) != samples for group in groups.values()):
            raise ValueError(f"{database}: expected one of each requested kernel per call")
        graph_calls = []
        if args.require_graph_identity:
            rows_by_identity = {}
            for row in rows:
                rows_by_identity.setdefault(row[6:], []).append(row)
            for ordinal, (producer, selector, megamoe) in enumerate(zip(
                groups["producer"], groups["selector"], groups["megamoe"]
            )):
                identity = producer[6:]
                expected = [producer, selector, megamoe]
                if dispatches:
                    expected.append(dispatches[ordinal])
                if args.prefix_kernel and groups["prefix"][ordinal][6:] == identity:
                    expected.append(groups["prefix"][ordinal])
                if args.joined_collective_preparation or args.joined_collective_barrier_only:
                    preparation = [row for row in rows_by_identity[identity] if row not in expected]
                    expected.extend(audit_joined_preparation(preparation, producer, selector,
                        barrier_only=args.joined_collective_barrier_only))
                if (any(value is None for value in identity) or identity[-2] <= 0
                        or identity[-1] <= 0 or identity in graph_calls
                        or any(row[6:] != identity for row in expected)
                        or sorted(rows_by_identity[identity]) != sorted(expected)):
                    raise ValueError(f"{database}: workload graph identity mismatch at call {ordinal}")
                graph_calls.append(identity)
        records = []
        for producer, selector, megamoe in zip(groups["producer"], groups["selector"], groups["megamoe"]):
            if not (producer[5] == selector[5] == megamoe[5]
                    and producer[0] <= selector[0] <= megamoe[0]
                    and producer[1] <= megamoe[1] and selector[1] <= megamoe[1]):
                raise ValueError(f"{database}: router/MegaMoE stream or ordering mismatch")
            records.append({
                "producer_end_us": (producer[1] - producer[0]) / 1000,
                "selector_start_us": (selector[0] - producer[0]) / 1000,
                "selector_end_us": (selector[1] - producer[0]) / 1000,
                "megamoe_start_us": (megamoe[0] - producer[0]) / 1000,
                "megamoe_end_us": (megamoe[1] - producer[0]) / 1000,
                "kernel_us": (megamoe[1] - megamoe[0]) / 1000,
            })
        for record, dispatch, producer, selector, megamoe in zip(
            records, dispatches, groups["producer"], groups["selector"], groups["megamoe"]
        ):
            if not (dispatch[5] == megamoe[5] and selector[0] <= dispatch[0] <= megamoe[0]
                    and dispatch[1] <= megamoe[1]):
                raise ValueError(f"{database}: dispatch stream or ordering mismatch")
            record.update(
                dispatch_start_us=(dispatch[0] - producer[0]) / 1000,
                dispatch_end_us=(dispatch[1] - producer[0]) / 1000,
                dispatch_kernel_us=(dispatch[1] - dispatch[0]) / 1000,
            )
        if args.prefix_kernel:
            for record, prefix, producer, megamoe in zip(
                records, groups["prefix"], groups["producer"], groups["megamoe"]
            ):
                if prefix[5] != producer[5] or prefix[1] > producer[0]:
                    raise ValueError(f"{database}: prefix kernel stream or ordering mismatch")
                record.update(
                    prefix_kernel_us=(prefix[1] - prefix[0]) / 1000,
                    prefix_to_producer_us=(producer[0] - prefix[0]) / 1000,
                    prefix_to_megamoe_end_us=(megamoe[1] - prefix[0]) / 1000,
                )
        result = {"database": str(database), "samples": samples, "rank": args.rank,
                  "genphase_dispatch": bool(dispatches),
                  "selector_grid_block": list(groups["selector"][0][3:5]),
                  "means": {key: mean(row[key] for row in records) for key in records[0]},
                  "records": records}
        if args.require_preinitialized_events:
            result["timing_event_pool_audit"] = audit_event_creation(
                database, groups["producer"][0][0], groups["megamoe"][-1][1])
        if args.retain_timestamps:
            result["timestamp_scope"] = (
                "Raw SQLite timestamp domain in ns; cross-report/GPU/host alignment is unspecified")
            result["trace_timestamps_ns"] = [
                {"ordinal": ordinal, **{
                    kind: {"start": group[ordinal][0], "end": group[ordinal][1]}
                    for kind, group in groups.items()}}
                for ordinal in range(samples)
            ]
            if dispatches:
                for record, dispatch in zip(result["trace_timestamps_ns"], dispatches):
                    record["dispatch"] = {"start": dispatch[0], "end": dispatch[1]}
        if args.require_graph_identity:
            result["graph_identity_audit"] = {
                "fields": ["deviceId", "contextId", "globalPid", "graphId", "correlationId"],
                "calls": graph_calls,
                "scope": "Exactly the requested workload and explicitly selected preparation kernels per graph call",
            }
            if args.joined_collective_preparation:
                result["joined_preparation_audit"] = {
                    "status": "PASS", "calls": samples, "preparation_nodes_per_call": 4,
                    "scope": "Same graph/stream; collective, two int resets, collective before metadata",
                }
        if args.prefix_kernel:
            result["prefix_kernel_pattern"] = args.prefix_kernel
        gaps = [(producer[0] - previous[1]) / 1000 for producer, previous in
                zip(groups["producer"][1:], groups["megamoe"][:-1])]
        result["inter_call_gaps_us"] = gaps
        result["mean_inter_call_gap_us"] = mean(gaps) if gaps else None
        cycles = [row["megamoe_end_us"] + gap for row, gap in zip(records, gaps)]
        result["producer_cycle_samples_us"] = cycles
        result["mean_producer_cycle_us"] = mean(cycles) if cycles else None
        results[arm] = result
        if not args.sweep:
            print(json.dumps({"arm": arm, "means": result["means"],
                              "mean_inter_call_gap_us": result["mean_inter_call_gap_us"],
                              "mean_producer_cycle_us": result["mean_producer_cycle_us"]}), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if args.sweep:
        print(json.dumps({"root": str(root), "captured_cases": len(results),
                          "samples": sum(result["samples"] for result in results.values()),
                          "scope": "Collected cases only; full-sweep coverage requires the sweep audit."}), flush=True)
    if args.csv:
        summaries = []
        for arm, result in results.items():
            records = result["records"]
            intervals = {
                "router_to_megamoe_end": [r["megamoe_end_us"] for r in records],
                "megamoe_kernel": [r["kernel_us"] for r in records],
                "selector": [r["selector_end_us"] - r["selector_start_us"] for r in records],
                "megamoe_launch_delay": [r["megamoe_start_us"] for r in records],
            }
            row = {"case_id": arm, "samples": len(records)}
            for name, values in intervals.items():
                for label, function in (("mean", mean), ("median", median), ("min", min), ("max", max)):
                    row[f"{name}_{label}_us"] = function(values)
            summaries.append(row)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
