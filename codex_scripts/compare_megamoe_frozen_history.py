#!/usr/bin/env python3
"""Compare an audited sweep with frozen per-case dispatch minima and provenance."""

import argparse
import csv
import json
import math
from pathlib import Path

from compare_megamoe_sweeps import DISPATCH, gain, load, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--phase-samples", type=int, default=5)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--threshold", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase_samples < 1 or not math.isfinite(args.threshold):
        parser.error("Require positive sample count and finite threshold")
    candidate = load(args.root, args.phase_samples, args.ep)
    with args.history_csv.open() as stream:
        history_rows = list(csv.DictReader(stream))
    history = {r["case_id"]: r for r in history_rows}
    if len(history) != len(history_rows):
        raise ValueError("Frozen history contains duplicate case IDs")
    rows = []
    for case, current in candidate.items():
        reference = history[case]
        before = float(reference[DISPATCH + "_median"])
        if not math.isfinite(before) or before <= 0 or not reference["source"]:
            raise ValueError(f"Invalid frozen history or provenance: {case}")
        rows.append({
            "case_id": case,
            "ep": current["contract"]["ep_size"],
            "global_tokens": current["contract"]["global_tokens_per_launch"],
            "historical_dispatch_us": before,
            "candidate_dispatch_us": current["dispatch_us"],
            "dispatch_gain_pct": gain(before, current["dispatch_us"]),
            "historical_source": reference["source"],
            "candidate_artifact": current["artifact"],
        })
    rows.sort(key=lambda r: (r["ep"], r["global_tokens"]))
    report = {
        "scope": "FULL_SWEEP" if args.ep is None else "COMPLETE_EP_GROUP_NOT_FULL_SWEEP",
        "measurement": "Exposed dispatch only; historical measurements are unpaired. "
                       "Frozen CSV includes all pre-campaign minima. No ordinary-kernel "
                       "or full-chain values are inferred from phase-only history.",
        "candidate_root": str(args.root),
        "history_csv": str(args.history_csv),
        "required_gain_pct": args.threshold,
        "summary": summarize(rows, args.threshold),
        "by_ep": {str(ep): summarize([r for r in rows if r["ep"] == ep], args.threshold)
                  for ep in sorted({r["ep"] for r in rows})},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
