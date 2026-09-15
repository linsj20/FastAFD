#!/usr/bin/env python3
"""Check queued AFD graph replays with changing, interleaved source padding."""

import argparse
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from minisgl.afd_fmha_runtime import AfdFmhaRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-cycles", type=int, default=200_000_000)
    args = parser.parse_args()
    if args.delay_cycles <= 0:
        parser.error("delay cycles must be positive")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    runtime = object.__new__(AfdFmhaRuntime)
    runtime.device = device
    runtime.role = "mlp"
    runtime.remote_attention_layers = (0, 1)
    runtime.num_lanes = runtime.num_mb = 2
    runtime.config = SimpleNamespace(max_running_req=16)
    for name in ("input_ids", "positions", "out_loc", "state_indices",
                 "last_indices", "wb_table", "wb_position", "wb_batch"):
        setattr(runtime, "_graph_" + name, torch.zeros(16, dtype=torch.int64, device=device))
    batch = SimpleNamespace(
        **{name: torch.zeros(16, dtype=torch.int64, device=device)
           for name in ("input_ids", "positions", "out_loc", "state_indices")},
        afd_last_indices=torch.arange(14, device=device),
        afd_writeback={name: torch.zeros(14, dtype=torch.int64, device=device)
                       for name in ("table_indices", "write_positions", "batch_indices")},
        afd_mb_subbatches=[SimpleNamespace(afd_fmha_source_token_offsets=(0, 4, 8))
                          for _ in range(2)],
        afd_microbatch_token_offsets=(0, 8, 16),
    )
    entry = SimpleNamespace(
        batch=batch,
        valid_token_counts=tuple(torch.zeros(9, dtype=torch.int64, device=device) for _ in range(2)),
        valid_token_iota_host=np.arange(8, dtype=np.int64),
    )
    cases = (((3, 4), (3, 4)), ((1, 3), (3, 4)), ((0, 0), (2, 4)))
    plans, expected = [], []
    for counts in cases:
        plans.append(SimpleNamespace(
            microbatch_real_token_counts=tuple(map(sum, zip(*counts))),
            microbatch_token_offsets=(0, 8, 16),
            source_plans=tuple(SimpleNamespace(microbatch_real_token_counts=c) for c in counts),
        ))
        rows = []
        for mb in range(2):
            row, prefix = [sum(c[mb] for c in counts)], 0
            for source in counts:
                row.extend(range(prefix, prefix + source[mb]))
                row.extend([-1] * (4 - source[mb]))
                prefix += source[mb]
            rows.append(row)
        expected.append(rows)
    stream = torch.cuda.Stream()
    consumer = torch.empty((2, 9), dtype=torch.int64, device=device)
    with torch.cuda.stream(stream):
        runtime._prepare_graph_replay(entry, batch, plans[0], 16)
        torch.stack(entry.valid_token_counts, out=consumer)
        torch.cuda._sleep(1)  # Initialize the delay kernel before the timed gate.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.stack(entry.valid_token_counts, out=consumer)
    torch.cuda.synchronize()
    snapshots, retained_batches = [], []
    delay_done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        # A short deliberate GPU delay guarantees that all host mutations are
        # queued before the first H2D. This models finite command lead without
        # relying on incidental model runtime or loading model weights.
        torch.cuda._sleep(args.delay_cycles)
        delay_done.record(stream)
        for plan in plans:
            step_batch = copy(batch)
            retained_batches.append(step_batch)
            runtime._prepare_graph_replay(entry, step_batch, plan, 16)
            graph.replay()
            snapshots.append(consumer.clone())
    gate_was_pending = not delay_done.query()
    stream.synchronize()
    observed = [snapshot.cpu().tolist() for snapshot in snapshots]
    report = {"gate_was_pending_after_enqueue": gate_was_pending,
              "expected": expected, "observed": observed,
              "all_replays_correct": observed == expected,
              "scope": "Actual runtime metadata preparation, pinned H2D and graph consumers; three queued steps, two sources/microbatches, changing padding including an empty source."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not gate_was_pending:
        raise RuntimeError("Delay ended before host submissions; concurrency test is inconclusive")
    if observed != expected:
        raise AssertionError("Queued graph consumer observed another step's validity metadata")


if __name__ == "__main__":
    main()
