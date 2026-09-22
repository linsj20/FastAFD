"""Collect untimed, per-invocation CUDA-event durations from every rank.

These are local durations, not cross-device timestamps. Their per-call maximum
does not establish the latest finish relative to a common start.
"""

import json
import math
from pathlib import Path
from statistics import mean


def make_timing_event_pairs(iterations, *, device):
    """Materialize CUDA handles before warmups, profiling, and timed replays.

    Constructing a torch.cuda.Event only creates its Python wrapper. Recording
    each event here keeps lazy driver allocation out of the replay launch loop.
    Later records overwrite these preparation timestamps on the same device.
    """
    import torch

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("Timing event pool requires a positive iteration count")
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        pairs = [(torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
        for start, end in pairs:
            start.record(stream)
            end.record(stream)
        pairs[-1][1].synchronize()
    return pairs


def aggregate_rank_timings(records):
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("Require one timing record per rank")
    world_size = len(records)
    ranks = [record["rank"] for record in records]
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(world_size)):
        raise ValueError("Missing, duplicate, or invalid rank identity")
    records = sorted(records, key=lambda record: record["rank"])
    scope = records[0]["scope"]
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("Require explicit timing scope")
    count = len(records[0]["durations_ms"])
    if not count:
        raise ValueError("Require nonempty invocation timings")
    for record in records:
        values = record["durations_ms"]
        if record["scope"] != scope or len(values) != count:
            raise ValueError("Rank timing scopes or invocation counts disagree")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in values):
            raise ValueError("Require finite positive durations")
    samples = []
    for ordinal in range(count):
        values = [record["durations_ms"][ordinal] for record in records]
        largest = max(values)
        samples.append({"ordinal": ordinal, "maximum_duration_ms": largest,
                        "ranks_with_maximum_duration": [
                            rank for rank, value in enumerate(values) if value == largest]})
    rank_means = [mean(record["durations_ms"]) for record in records]
    return {
        "scope": scope, "world_size": world_size, "iterations": count,
        "clock_alignment": "unmeasured; rank-local durations only",
        "distributed_latest_end_established": False,
        "mean_rank_mean_ms": mean(rank_means),
        "per_call_rank_mean_ms": [mean(record["durations_ms"][i] for record in records)
                                  for i in range(count)],
        "maximum_rank_mean_ms": max(rank_means),
        "mean_per_call_maximum_ms": mean(sample["maximum_duration_ms"] for sample in samples),
        "samples": samples, "ranks": records,
    }


def collect_rank_timings(durations_ms, output_path, *, scope, metadata):
    """Call collectively after timing ends; write once on rank zero."""
    import torch.distributed as dist

    record = {"rank": dist.get_rank(), "scope": scope,
              "durations_ms": durations_ms, "metadata": metadata}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    result = aggregate_rank_timings(records)
    if dist.get_rank() == 0:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(f"CUDA_RANK_TIMINGS={path}", flush=True)
