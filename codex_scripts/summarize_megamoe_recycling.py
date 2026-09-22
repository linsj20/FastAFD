#!/usr/bin/env python3
"""Audit adapter or model output, reset and consumer-work timestamps."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


def interval_union_ns(intervals):
    total, previous = 0, 0
    for start, end in sorted((a, b) for a, b in intervals if a < b):
        total += max(0, end - max(start, previous))
        previous = max(previous, end)
    return total


def summarize(
    log, ranks, calls_per_rank, *, independent_readiness=False,
    hardware_output_dependency=False, reset_workers=8,
):
    if reset_workers not in (1, 4, 8):
        raise ValueError("Expected one, four or eight reset workers")
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
            ahead = payload.get("reset_ahead") if isinstance(payload, dict) else None
            if isinstance(payload, dict) and payload.get("reset_ahead_split_flags"):
                if ahead is None or len(ahead) != 21:
                    raise ValueError(f"Missing split reset actions: {rank}/{call}")
                current_action, ahead_action = ahead[20] & 0xffffffff, ahead[20] >> 32
                if current_action not in (0, 1) or ahead_action not in (0, 1):
                    raise ValueError(f"Invalid split reset actions: {rank}/{call}")
                ahead = [*ahead[:20], current_action | (ahead_action << 1)]
            if ahead is not None and (len(ahead) != 21 or ahead[20] not in range(4)):
                raise ValueError(f"Invalid ahead reset layout: {rank}/{call}")
            current_clear = ahead is None or bool(ahead[20] & 1)
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
                if len(worker_stamps) != 16 or any(worker_stamps[2 * reset_workers:]):
                    raise ValueError(f"Unexpected reset worker coverage: {rank}/{call}")
                workers = list(zip(worker_stamps[:2 * reset_workers:2],
                                   worker_stamps[1:2 * reset_workers:2]))
                if any(not 0 < start <= end or (current_clear and end > reset_end)
                       for start, end in workers):
                    raise ValueError(f"Invalid reset worker ordering: {rank}/{call}")
                detail.update({
                    "reset_worker_intervals_ns": workers,
                    "reset_clear_span_us": (max(e for _, e in workers) - min(s for s, _ in workers)) / 1000,
                    "reset_clear_end_minus_norm_work_end_us": (max(e for _, e in workers) - max(e for _, e in rows)) / 1000,
                    "reset_clear_end_to_ready_us": (reset_end - max(e for _, e in workers)) / 1000,
                    "reset_workers_starting_before_last_combine_store": sum(s < max(combine) for s, _ in workers),
                })
                if not current_clear:
                    for key in ("reset_clear_span_us", "reset_clear_end_minus_norm_work_end_us",
                                "reset_clear_end_to_ready_us"):
                        detail[key] = None
            if ahead is not None:
                ahead_clear = bool(ahead[20] & 2)
                ahead_workers_count = payload.get("ahead_reset_workers", reset_workers)
                if ahead_workers_count not in (1, 4, 8):
                    raise ValueError(f"Invalid ahead worker count: {rank}/{call}")
                if (not 0 < ahead[0] <= ahead[1]
                        or any(ahead[2 + 2 * ahead_workers_count:18])):
                    raise ValueError(f"Invalid ahead reset coverage: {rank}/{call}")
                ahead_workers = list(zip(ahead[2:2 + 2 * ahead_workers_count:2],
                                         ahead[3:2 + 2 * ahead_workers_count:2]))
                if any(not 0 < a <= b or (ahead_clear and b > ahead[1]) for a, b in ahead_workers):
                    raise ValueError(f"Invalid ahead reset ordering: {rank}/{call}")
                detail.update({
                    "reset_current_performed": current_clear,
                    "reset_target_generation": ahead[18],
                    "ahead_reset_performed": ahead_clear,
                    "ahead_target_generation": ahead[19],
                    "ahead_reset_observation_ns": ahead[:2],
                    "ahead_reset_worker_intervals_ns": ahead_workers,
                    "ahead_reset_kernel": payload.get("ahead_reset_kernel",
                        "norm" if payload.get("reset_ahead_split_flags") else "selector"),
                    "ahead_norm_useful_overlap_us": (interval_union_ns(
                        (max(a, c), min(b, d)) for a, b in ahead_workers for c, d in rows
                    ) / 1000 if ahead_clear else None),
                    "ahead_clear_span_us": ((max(b for _, b in ahead_workers) -
                                             min(a for a, _ in ahead_workers)) / 1000
                                            if ahead_clear else None),
                    "ahead_clear_end_to_ready_us": ((ahead[1] - max(b for _, b in ahead_workers)) / 1000
                                                    if ahead_clear else None),
                })
                if "gate_worker_stamps" in payload:
                    gate_stamps = payload["gate_worker_stamps"]
                    if len(gate_stamps) != 2 * payload["gate_blocks"] or not gate_stamps:
                        raise ValueError(f"Invalid gate worker coverage: {rank}/{call}")
                    gate_workers = list(zip(gate_stamps[::2], gate_stamps[1::2]))
                    if any(not 0 < a <= b for a, b in gate_workers):
                        raise ValueError(f"Invalid gate worker ordering: {rank}/{call}")
                    detail["gate_worker_intervals_ns"] = gate_workers
                    detail["ahead_gate_useful_overlap_us"] = (interval_union_ns(
                        (max(a, c), min(b, d)) for a, b in ahead_workers for c, d in gate_workers
                    ) / 1000 if ahead_clear else None)
                    if "gate_consumer_stamps" in payload or "gate_ready_stamp" in payload:
                        acquires = payload["gate_consumer_stamps"]
                        ready = payload.get("gate_ready_stamp", max(b for _, b in gate_workers))
                        if (len(acquires) != count or max(b for _, b in gate_workers) > ready
                                or any(a < ready for a in acquires)):
                            raise ValueError(f"Gate publication/acquisition ordering failed: {rank}/{call}")
                        if "gate_ready_stamp" in payload:
                            detail["gate_ready_ns"] = ready
                        detail["gate_consumer_ready_ns"] = acquires
            intervals = sorted((max(start, reset_start), min(end, reset_end))
                               for start, end in rows if end > reset_start and start < reset_end)
            union = interval_union_ns(intervals)
            records.append({
                "rank": rank, "call": call, "stamps_ns": stamps,
                "reset_issuer_to_publication_us": (reset_end - reset_start) / 1000 if current_clear else None,
                "output_to_reset_publication_us": (reset_end - output) / 1000 if current_clear else None,
                "norm_useful_overlap_us": union / 1000 if current_clear else None,
                "norm_last_work_before_reset_publication_us": (reset_end - max(e for _, e in rows)) / 1000 if current_clear else None,
                "norm_rows": len(rows),
                "norm_rows_finished_before_reset": sum(end < reset_end for _, end in rows) if current_clear else 0,
                **detail,
            })
    if seen != set(range(ranks)):
        raise ValueError(f"Expected ranks 0..{ranks - 1}, found {sorted(seen)}")
    return dict(summarize_records(records, ranks), reset_workers=reset_workers,
                independent_readiness=independent_readiness,
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
    ahead_records = [r for r in records if "ahead_reset_performed" in r]
    if ahead_records:
        if len(ahead_records) != len(records):
            raise ValueError("Mixed current-only and ahead reset trace layouts")
        fields += ["ahead_clear_span_us", "ahead_clear_end_to_ready_us", "ahead_norm_useful_overlap_us"]
    if any("ahead_gate_useful_overlap_us" in r for r in records):
        if not all("ahead_gate_useful_overlap_us" in r for r in records):
            raise ValueError("Mixed gate worker coverage")
        fields += ["ahead_gate_useful_overlap_us"]

    def stats(key):
        values = [r[key] for r in records if r[key] is not None]
        return ({"mean": mean(values), "median": median(values), "min": min(values),
                 "max": max(values)} if values else
                {"mean": None, "median": None, "min": None, "max": None})

    return {
        "ranks": ranks, "calls": len(records), "records": records,
        "calls_with_useful_overlap": sum((r["norm_useful_overlap_us"] or 0) > 0 for r in records),
        "calls_with_all_norm_work_before_reset": sum(
            r["norm_rows_finished_before_reset"] == r["norm_rows"] for r in records),
        "calls_with_residual_preload_before_last_combine_store": sum(
            r.get("norm_rows_preloaded_before_last_combine_store", 0) > 0 for r in records),
        "calls_with_readiness_after_norm_work": sum(
            (r["norm_last_work_before_reset_publication_us"] or 0) > 0 for r in records),
        "stats": {key: stats(key) for key in fields},
        **({"current_clears": sum(r["reset_current_performed"] for r in records),
            "ahead_clears": sum(r["ahead_reset_performed"] for r in records),
            "calls_with_ahead_norm_overlap": sum((r["ahead_norm_useful_overlap_us"] or 0) > 0
                                                 for r in records),
            "skipped_clear_semantics": "Null current/ahead clear statistics denote an already-ready bank; "
                                       "its event timestamps record observations, not new publications."}
           if ahead_records else {}),
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


def summarize_model(root, metadata, ranks, layers, lanes, *, independent_readiness=False,
                    hardware_output_dependency=False, reset_workers=8):
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
                           hardware_output_dependency=hardware_output_dependency,
                           reset_workers=reset_workers)
        records.extend(dict(r, step=step) for r in result["records"])
    return dict(summarize_records(records, ranks), measured_steps=steps, reset_workers=reset_workers,
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
    parser.add_argument("--reset-workers", type=int, choices=(1, 4, 8), default=8)
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
                                 hardware_output_dependency=args.hardware_output_dependency,
                                 reset_workers=args.reset_workers)
    else:
        if not args.calls_per_rank:
            parser.error("Adapter traces require --calls-per-rank")
        log = read_rank_traces(args.log)[0] if args.log.is_dir() else args.log.read_text()
        result = summarize(log, args.ranks, args.calls_per_rank,
                           independent_readiness=args.independent_readiness,
                           hardware_output_dependency=args.hardware_output_dependency,
                           reset_workers=args.reset_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
