#!/usr/bin/env python3
"""Audit adapter or model output, reset and consumer-work timestamps."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


def summarize(log, ranks, calls_per_rank, *, independent_readiness=False, hardware_output_dependency=False):
    records = []
    seen = set()
    for match in re.finditer(r"^MEGAMOE_RECYCLE_TIMING rank=(\d+) (.+)$", log, re.M):
        rank = int(match[1])
        if rank in seen:
            raise ValueError(f"Duplicate timing rank {rank}")
        seen.add(rank)
        calls = json.loads(match[2])
        if len(calls) != calls_per_rank:
            raise ValueError(f"Rank {rank}: expected {calls_per_rank} calls")
        for call, payload in calls.items():
            detail = {}
            if isinstance(payload, dict):
                count = payload["norm_rows"]
                stamps = payload["stamps"]
                if count <= 0 or len(stamps) <= 3 + 3 * count:
                    raise ValueError(f"Invalid detailed timestamp layout: {rank}/{call}")
                norm_stamps = stamps[3:3 + 2 * count]
                prefetch = stamps[3 + 2 * count:3 + 3 * count]
                combine = stamps[3 + 3 * count:]
                active = [i for i, stamp in enumerate(combine) if stamp]
                chunks = (payload["hidden"] * 2 + 4095) // 4096
                expected = min(len(combine), count * chunks)
                if active != list(range(len(combine) - expected, len(combine))):
                    raise ValueError(f"Missing combine writers: {rank}/{call}")
                if any(not 0 < p <= ready for p, ready in zip(prefetch, norm_stamps[::2])):
                    raise ValueError(f"Invalid residual preload ordering: {rank}/{call}")
                if hardware_output_dependency:
                    if stamps[0] != 0:
                        raise ValueError(f"Unexpected software output publication: {rank}/{call}")
                elif max(combine) > stamps[0]:
                    raise ValueError(f"Combine writer after output release: {rank}/{call}")
                detail = {
                    "residual_preload_end_ns": prefetch,
                    "combine_store_end_ns": combine,
                    "norm_rows_preloaded_before_last_combine_store": sum(p < max(combine) for p in prefetch),
                    "last_norm_preload_before_last_combine_store_us": (max(combine) - max(prefetch)) / 1000,
                }
            else:
                stamps = payload
                if len(stamps) < 5 or (len(stamps) - 3) % 2:
                    raise ValueError(f"Invalid timestamp layout: {rank}/{call}")
                norm_stamps = stamps[3:]
            output, reset_start, reset_end = stamps[:3]
            if hardware_output_dependency:
                if not isinstance(payload, dict):
                    raise ValueError("Hardware output dependency requires combine-writer stamps")
                if "reset_worker_stamps" not in payload:
                    raise ValueError("Hardware output dependency requires all reset-worker stamps")
                output = max(combine)
                detail["output_reference"] = "last_combine_store"
            rows = list(zip(norm_stamps[::2], norm_stamps[1::2]))
            if not (output > 0 and 0 < reset_start <= reset_end):
                raise ValueError(f"Invalid output/reset ordering: {rank}/{call}")
            if not independent_readiness and output > reset_end:
                raise ValueError(f"Readiness preceded output in joined-publication mode: {rank}/{call}")
            if any(not output <= start <= end for start, end in rows):
                raise ValueError(f"Consumer ran before output publication: {rank}/{call}")
            if isinstance(payload, dict) and "reset_worker_stamps" in payload:
                worker_stamps = payload["reset_worker_stamps"]
                if len(worker_stamps) != 16:
                    raise ValueError(f"Expected eight reset worker intervals: {rank}/{call}")
                workers = list(zip(worker_stamps[::2], worker_stamps[1::2]))
                if any(not 0 < start <= end <= reset_end for start, end in workers):
                    raise ValueError(f"Invalid reset worker ordering: {rank}/{call}")
                detail.update({
                    "reset_worker_intervals_ns": workers,
                    "reset_clear_span_us": (max(e for _, e in workers) - min(s for s, _ in workers)) / 1000,
                    "reset_clear_end_minus_norm_work_end_us": (max(e for _, e in workers) - max(e for _, e in rows)) / 1000,
                    "reset_clear_end_to_ready_us": (reset_end - max(e for _, e in workers)) / 1000,
                    "reset_workers_starting_before_last_combine_store": sum(s < max(combine) for s, _ in workers),
                })
            intervals = sorted((max(start, reset_start), min(end, reset_end))
                               for start, end in rows if end > reset_start and start < reset_end)
            union, previous = 0, 0
            for start, end in intervals:
                union += max(0, end - max(start, previous))
                previous = max(previous, end)
            records.append({
                "rank": rank, "call": call, "stamps_ns": stamps,
                "reset_issuer_to_publication_us": (reset_end - reset_start) / 1000,
                "output_to_reset_publication_us": (reset_end - output) / 1000,
                "norm_useful_overlap_us": union / 1000,
                "norm_last_work_before_reset_publication_us": (reset_end - max(e for _, e in rows)) / 1000,
                "norm_rows": len(rows),
                "norm_rows_finished_before_reset": sum(end < reset_end for _, end in rows),
                **detail,
            })
    if seen != set(range(ranks)):
        raise ValueError(f"Expected ranks 0..{ranks - 1}, found {sorted(seen)}")
    return dict(summarize_records(records, ranks), independent_readiness=independent_readiness,
                hardware_output_dependency=hardware_output_dependency,
                output_reference="last_combine_store" if hardware_output_dependency else "output_publication")


def summarize_records(records, ranks):
    fields = ["reset_issuer_to_publication_us", "output_to_reset_publication_us",
              "norm_useful_overlap_us", "norm_last_work_before_reset_publication_us"]
    if any("reset_clear_span_us" in r for r in records):
        if not all("reset_clear_span_us" in r for r in records):
            raise ValueError("Mixed trace layouts: missing reset-worker intervals")
        fields += ["reset_clear_span_us", "reset_clear_end_minus_norm_work_end_us",
                   "reset_clear_end_to_ready_us"]
    return {
        "ranks": ranks, "calls": len(records), "records": records,
        "calls_with_useful_overlap": sum(r["norm_useful_overlap_us"] > 0 for r in records),
        "calls_with_all_norm_work_before_reset": sum(
            r["norm_rows_finished_before_reset"] == r["norm_rows"] for r in records),
        "calls_with_residual_preload_before_last_combine_store": sum(
            r.get("norm_rows_preloaded_before_last_combine_store", 0) > 0 for r in records),
        "calls_with_readiness_after_norm_work": sum(
            r["norm_last_work_before_reset_publication_us"] > 0 for r in records),
        "stats": {key: {"mean": mean(r[key] for r in records),
                        "median": median(r[key] for r in records),
                        "min": min(r[key] for r in records),
                        "max": max(r[key] for r in records)} for key in fields},
        "interpretation": "Separate instrumented globaltimer spans, all calls retained. "
                          "Legacy reset start is one worker's issuer marker; others can start earlier. "
                          "Explicit worker intervals bound the clear stores and CTA synchronization, not physical memory drain. "
                          "Consumer work ends after its CTA's output stores, distinct from reported kernel exit. "
                          "These measurements do not establish ordinary model latency.",
    }


def read_rank_traces(root):
    sources = [json.loads(p.read_text()) for p in sorted(root.glob("rank-*.json"))]
    return "\n".join(f"MEGAMOE_RECYCLE_TIMING rank={r['rank']} " + json.dumps(r["calls"])
                     for r in sources), sources


def summarize_model(root, metadata, ranks, layers, lanes, *, independent_readiness=False, hardware_output_dependency=False):
    steps = metadata["decode_step_ids"]
    if not steps or steps != sorted(set(steps)) or metadata["warmup_decode_step_id"] in steps:
        raise ValueError("Capture must declare unique ordered measured steps excluding warmup")
    if len(steps) != metadata["capture_decode_steps"]:
        raise ValueError("Capture step count disagrees with measured steps")
    expected = {f"{layer}:{lane}" for layer in range(layers) for lane in range(lanes)}
    records = []
    for step in steps:
        log, sources = read_rank_traces(root / f"step-{step}")
        for source in sources:
            if set(source["calls"]) != expected:
                raise ValueError(f"Incomplete layer/lane coverage at step {step}, rank {source['rank']}")
        result = summarize(log, ranks, len(expected), independent_readiness=independent_readiness,
                           hardware_output_dependency=hardware_output_dependency)
        records.extend(dict(r, step=step) for r in result["records"])
    return dict(summarize_records(records, ranks), measured_steps=steps,
                layers=layers, lanes=lanes, independent_readiness=independent_readiness,
                hardware_output_dependency=hardware_output_dependency,
                output_reference="last_combine_store" if hardware_output_dependency else "output_publication")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--calls-per-rank", type=int)
    parser.add_argument("--capture-metadata", type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--independent-readiness", action="store_true",
                        help="Allow payload readiness before output publication; still require "
                             "all clears before readiness and output publication before output use")
    parser.add_argument("--hardware-output-dependency", action="store_true",
                        help="Require unused software-output stamp and all combine stores before "
                             "each consumer-work start; output-relative spans use last combine store")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hardware_output_dependency and not args.independent_readiness:
        parser.error("Hardware output dependency requires independent reset readiness")
    if args.capture_metadata:
        if not args.layers or not args.lanes or args.calls_per_rank is not None:
            parser.error("Model traces require --layers and --lanes, without --calls-per-rank")
        result = summarize_model(args.log, json.loads(args.capture_metadata.read_text()),
                                 args.ranks, args.layers, args.lanes,
                                 independent_readiness=args.independent_readiness,
                                 hardware_output_dependency=args.hardware_output_dependency)
    else:
        if not args.calls_per_rank:
            parser.error("Adapter traces require --calls-per-rank")
        log = read_rank_traces(args.log)[0] if args.log.is_dir() else args.log.read_text()
        result = summarize(log, args.ranks, args.calls_per_rank,
                           independent_readiness=args.independent_readiness,
                           hardware_output_dependency=args.hardware_output_dependency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
