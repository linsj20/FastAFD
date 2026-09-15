#!/usr/bin/env python3
"""Summarize routing from saved exact-check tensors, without another GPU run.

Counts describe logical routes and unique destination ranks, not measured link
traffic. Supply the snapshots corresponding to the timed input, not padding or
alternate-input correctness snapshots.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def summarize(root, case, ep, experts):
    loads = [0] * ep
    expert_loads = [0] * experts
    fanout = Counter()
    valid_tokens = assignments = remote_frames = 0
    source_rows = []
    for rank in range(ep):
        snapshot = torch.load(root / f"{case}_rank{rank}.pt",
                              map_location="cpu", weights_only=True)
        ids = snapshot["result.0"] if "result.0" in snapshot else snapshot["ids"]
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{case}: expected a token by top-k integer tensor")
        source_rows.append(ids.shape[0])
        for row in ids.tolist():
            selected = [expert for expert in row if expert >= 0]
            if any(expert >= experts for expert in selected):
                raise ValueError(f"{case}: expert index exceeds declared geometry")
            if not selected:
                continue
            destinations = {expert // (experts // ep) for expert in selected}
            valid_tokens += 1
            assignments += len(selected)
            fanout[len(destinations)] += 1
            remote_frames += len(destinations - {rank})
            for expert in selected:
                expert_loads[expert] += 1
                loads[expert // (experts // ep)] += 1
    if not valid_tokens:
        raise ValueError(f"{case}: no valid routes")
    frames = sum(count * tokens for count, tokens in fanout.items())
    return {
        "source_rows": source_rows,
        "valid_tokens": valid_tokens,
        "expert_assignments": assignments,
        "unique_destination_frames": frames,
        "remote_destination_frames": remote_frames,
        "destination_fanout_histogram": dict(sorted(fanout.items())),
        "mean_destination_ranks_per_token": frames / valid_tokens,
        "copies_saved_vs_one_per_expert_pct": 100 * (1 - frames / assignments),
        "expert_assignments_per_destination_rank": loads,
        "destination_max_to_mean_load": max(loads) / (assignments / ep),
        "active_experts_per_destination_rank": [
            sum(n > 0 for n in expert_loads[r * (experts // ep):(r + 1) * (experts // ep)])
            for r in range(ep)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ep < 1 or args.experts < 1 or args.experts % args.ep:
        parser.error("experts must be positive and divisible by EP")
    if len(set(args.case)) != len(args.case) or any(Path(c).name != c for c in args.case):
        parser.error("case names must be unique filename prefixes")
    result = {
        "scope": "Saved input routes across all EP ranks; logical copy counts, not measured link traffic.",
        "ep": args.ep,
        "experts": args.experts,
        "cases": {c: summarize(args.root, c, args.ep, args.experts) for c in args.case},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
