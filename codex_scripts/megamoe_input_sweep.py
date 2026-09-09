#!/usr/bin/env python3
"""Plan deduplicated AFD-equivalent inputs and collect six-point profiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def plan(output: Path) -> None:
    cases = []
    limits = {}
    for ep in (2, 4, 8, 16, 32, 64):
        max_ratio = 72 // ep - 1
        limits[str(ep)] = max_ratio
        equivalent = {}
        for ratio in range(1, max_ratio + 1):
            for batch in range(1, 8):
                # AFD captures both MB2 lanes at ceil(batch/2) rows per
                # attention GPU. Real counts and padding are recorded below.
                tokens_per_attention_gpu = (batch + 1) // 2
                rows = ratio * tokens_per_attention_gpu
                equivalent.setdefault(rows, []).append({
                    "attention_to_ffn_ratio": f"{ratio}:1",
                    "attention_gpus": ep * ratio,
                    "batch_per_attention_gpu": batch,
                    "real_microbatch_sizes": [batch // 2, (batch + 1) // 2],
                    "tokens_per_attention_gpu_per_microbatch": tokens_per_attention_gpu,
                    "active_gpus": ep * (ratio + 1),
                })
        for rows, inputs in sorted(equivalent.items()):
            cases.append({
                "case_id": f"ep{ep}_rows{rows}", "ep": ep,
                "rows_per_rank": rows, "global_tokens": ep * rows,
                "routes_per_expert_min": ep * rows * 8 // 128,
                "routes_per_expert_max": (ep * rows * 8 + 127) // 128,
                "equivalent_inputs": inputs,
            })
    # EP64 has no A>=1 integer ratio inside NVL72. Record that boundary and
    # study the largest physically fitting attention side explicitly as1:8.
    for tokens_per_gpu in range(1, 5):
        global_tokens = 8 * tokens_per_gpu
        cases.append({
            "case_id": f"ep64_tokens{global_tokens}", "ep": 64,
            "rows_per_rank": 1, "global_tokens": global_tokens,
            "valid_rows_per_rank": [int(rank < global_tokens) for rank in range(64)],
            "routes_per_expert_min": global_tokens * 8 // 128,
            "routes_per_expert_max": (global_tokens * 8 + 127) // 128,
            "equivalent_inputs": [{
                "attention_to_ffn_ratio": "1:8", "attention_gpus": 8,
                "batch_per_attention_gpu": batch,
                "real_microbatch_sizes": [batch // 2, (batch + 1) // 2],
                "tokens_per_attention_gpu_per_microbatch": tokens_per_gpu,
                "active_gpus": 72,
            } for batch in range(1, 8) if (batch + 1) // 2 == tokens_per_gpu],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "contract": {"model": "Qwen3-235B-A22B", "experts": 128,
                     "hidden": 4096, "intermediate": 1536, "top_k": 8,
                     "activation_dtype": "fp8", "weight_dtype": "fp4",
                     "microbatches": 2, "batch_min": 1, "batch_max": 7,
                     "tokens_per_attention_gpu": [1, 2, 3, 4],
                     "attention_tp": 1, "gpu_budget": 72,
                     "num_sms": 150, "bucket": 3840,
                     "routing": "integer-balanced rank-interleaved expert sequence"},
        "max_attention_to_ffn_ratio": limits,
        "ep64_policy": "A1:F1 infeasible (128 GPUs); measure A1:F8 with 8 attention GPUs",
        "cases": cases,
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "unique_cases": len(cases),
                      "per_ep": {str(ep): sum(c["ep"] == ep for c in cases)
                                 for ep in (2, 4, 8, 16, 32, 64)}}))


def collect(manifest_path: Path, root: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    records = []
    missing = []
    for case in manifest["cases"]:
        paths = sorted((root / case["case_id"]).glob("*/decomposition.json"))
        if len(paths) > 1:
            raise RuntimeError(f"multiple completed captures for {case['case_id']}: {paths}")
        if not paths:
            missing.append(case["case_id"])
            continue
        if not paths[0].with_suffix(".csv").is_file():
            missing.append(case["case_id"])
            continue
        result = json.loads(paths[0].read_text())
        contract = result["contract"]
        if (contract["ep_size"], contract["global_tokens_per_launch"]) != (
            case["ep"], case["global_tokens"]
        ):
            raise RuntimeError(f"input mismatch: {paths[0]}")
        record = {"case_id": case["case_id"], "ep": case["ep"],
                  "rows_per_rank": case["rows_per_rank"],
                  "global_tokens": case["global_tokens"],
                  "nsys_kernel_median_us": result["nsys_kernel_us"]["median"],
                  "equivalent_input_count": len(case["equivalent_inputs"]),
                  "artifact": str(paths[0].relative_to(root))}
        for metric in ("dispatch_span_us", "compute_span_us", "combine_span_us",
                       "pre_compute_dispatch_exposure_us",
                       "post_compute_combine_exposure_us", "post_combine_tail_us",
                       "kernel_envelope_us"):
            for stat in ("min", "median", "max"):
                record[f"{metric}_{stat}"] = result["summary_us"][metric][stat]
        records.append(record)
    if records:
        with (root / "decomposition.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    # Compare only equal global work. Keep measured communication exposure
    # separate from compute; do not infer a pure wire latency from an intercept.
    by_tokens = {}
    for record in records:
        by_tokens.setdefault(record["global_tokens"], []).append(record)
    scaling = []
    for tokens, group in sorted(by_tokens.items()):
        if len(group) < 2:
            continue
        group.sort(key=lambda record: record["ep"])
        base = group[0]
        for record in group:
            scaling.append({
                "global_tokens": tokens, "ep": record["ep"],
                "case_id": record["case_id"], "baseline_ep": base["ep"],
                "ep_ratio": record["ep"] / base["ep"],
                **{key: record[key] for key in (
                    "nsys_kernel_median_us", "compute_span_us_median",
                    "pre_compute_dispatch_exposure_us_median",
                    "post_compute_combine_exposure_us_median",
                    "post_combine_tail_us_median")},
                "kernel_speedup": base["nsys_kernel_median_us"] / record["nsys_kernel_median_us"],
                "compute_speedup": base["compute_span_us_median"] / record["compute_span_us_median"],
                "artifact": record["artifact"],
            })
    if scaling:
        with (root / "ep_scaling.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scaling[0]))
            writer.writeheader()
            writer.writerows(scaling)
    aliases = []
    for case in manifest["cases"]:
        for equivalent in case["equivalent_inputs"]:
            aliases.append({
                "case_id": case["case_id"], "ep": case["ep"],
                "global_tokens": case["global_tokens"],
                **equivalent,
                "real_microbatch_sizes": "+".join(map(str, equivalent["real_microbatch_sizes"])),
            })
    with (root / "equivalent_inputs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aliases[0]))
        writer.writeheader()
        writer.writerows(aliases)
    summary = {"completed": len(records), "expected": len(manifest["cases"]),
               "missing": missing, "ep64_policy": manifest["ep64_policy"]}
    (root / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": len(records), "expected": len(manifest["cases"]),
                      "missing": len(missing), "equivalent_inputs": len(aliases)}))


def audit(manifest_path: Path, root: Path) -> None:
    """Check capture completeness and contracts without another GPU run."""
    manifest = json.loads(manifest_path.read_text())
    expected = {(ep, f"{ratio}:1", batch)
                for ep in (2, 4, 8, 16, 32)
                for ratio in range(1, 72 // ep)
                for batch in range(1, 8)}
    expected.update((64, "1:8", batch) for batch in range(1, 8))
    aliases = []
    unique_inputs = set()
    missing = []
    checked = 0
    markers = {f"Mega{phase}{bound}" for phase in ("Dispatch", "Compute", "Combine")
               for bound in ("Start", "End")}
    for case in manifest["cases"]:
        ep, tokens = case["ep"], case["global_tokens"]
        key = (ep, tokens)
        assert key not in unique_inputs, (case["case_id"], "duplicate input")
        unique_inputs.add(key)
        for alias in case["equivalent_inputs"]:
            batch = alias["batch_per_attention_gpu"]
            aliases.append((ep, alias["attention_to_ffn_ratio"], batch))
            assert alias["active_gpus"] == alias["attention_gpus"] + ep <= 72
            assert alias["attention_gpus"] * ((batch + 1) // 2) == tokens
            assert alias["real_microbatch_sizes"] == [batch // 2, (batch + 1) // 2]
        paths = list((root / case["case_id"]).glob("*/decomposition.json"))
        assert len(paths) <= 1, (case["case_id"], "duplicate capture")
        if not paths:
            missing.append(case["case_id"])
            continue
        path = paths[0]
        result = json.loads(path.read_text())
        contract = result["contract"]
        assert (contract["ep_size"], contract["global_tokens_per_launch"]) == key
        assert tuple(contract[k] for k in ("hidden", "intermediate", "experts", "top_k", "num_sms")) == (4096, 1536, 128, 8, 150)
        assert (contract["activation_dtype"], contract["weight_dtype"], contract["microbatches"]) == ("fp8", "fp4", 2)
        assert contract["iket_mode"] == "source-markers"
        assert contract["rank_interleaved_expert_mapping"] and contract["routes_prepared"]
        assert contract["input_route_distribution"] == "perfect_balance"
        # The first eight captures predate the variable-source-count schema.
        if path.parent.name == "7020835":
            assert ep == 2 and 1 <= case["rows_per_rank"] <= 8
            assert tokens == ep * contract["rows_per_ep_rank_per_microbatch"]
        else:
            counts = contract["valid_rows_per_rank"]
            assert len(counts) == ep and sum(counts) == tokens
            assert max(counts) - min(counts) <= 1
        assert contract["routes_per_expert_min"] == tokens * 8 // 128
        assert contract["routes_per_expert_max"] == (tokens * 8 + 127) // 128
        with path.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(result["samples"]) == 3
        for sample, row in zip(result["samples"], rows):
            assert set(sample["marker_counts"]) == markers
            assert min(sample["marker_counts"].values()) > 0
            assert int(row["rank"]) == sample["rank"] == 0
            components = [float(row[k]) for k in (
                "initial_interval_us", "pre_compute_dispatch_exposure_us",
                "compute_span_us", "post_compute_combine_exposure_us", "post_combine_tail_us")]
            # Derive gaps from boundaries, including the older CSV schema.
            components.extend((max(0, sample["compute_start_us"] - sample["dispatch_end_us"]),
                               max(0, sample["combine_start_us"] - sample["compute_end_us"])))
            assert all(math.isfinite(value) and value >= 0 for value in components)
            assert abs(sum(components) - float(row["kernel_envelope_us"])) < 1e-6
        assert math.isfinite(result["nsys_kernel_us"]["median"]) and result["nsys_kernel_us"]["median"] > 0
        for name in ("baseline.json", "profiled.json"):
            report = json.loads((path.parent / name).read_text())
            assert report["status"] == "ok"
            info = next(iter(report["exact_qwen"].values()))
            assert (info["ep_size"], info["global_tokens"], info["rows_per_rank"]) == (ep, tokens, case["rows_per_rank"])
            assert info["routing"] == "fixed_global_sequence_rank_interleaved_balance"
            if ep == 64 and name == "baseline.json":
                assert info["combine_exact_checked"]
        checked += 1
    assert set(aliases) == expected and len(aliases) == len(expected)
    report = {"complete": not missing, "artifact_contracts_checked": checked,
              "samples_checked": checked * 3, "expected_cases": len(manifest["cases"]),
              "equivalent_inputs_checked": len(aliases), "missing": missing,
              "checks": ["scope and deduplication", "input and balance contracts",
                         "three samples and six markers", "per-sample phase closure",
                         "normal/profiled success", "EP64 exact-combine results"]}
    (root / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("output", type=Path)
    p = sub.add_parser("collect")
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    p = sub.add_parser("audit")
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.output)
    elif args.command == "collect":
        collect(args.manifest, args.root)
    else:
        audit(args.manifest, args.root)
