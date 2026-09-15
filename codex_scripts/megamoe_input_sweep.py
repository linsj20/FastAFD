#!/usr/bin/env python3
"""Plan deduplicated AFD-equivalent inputs and collect six-point profiles."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shlex
import subprocess
from pathlib import Path


def plan(output: Path, model: str = "qwen3", batch_max: int = 7,
         capacity: Path | None = None) -> None:
    qwen35 = model == "qwen35"
    experts, intermediate, top_k = (512, 1024, 10) if qwen35 else (128, 1536, 8)
    eps = (4, 8, 16, 32, 64) if qwen35 else (2, 4, 8, 16, 32, 64)
    capacity_basis = None
    if capacity:
        capacity_basis = json.loads(capacity.read_text())
        context = capacity_basis["context_tokens"]
        page = capacity_basis["page_size"]
        output_tokens = capacity_basis["output_tokens"]
        tokens = capacity_basis["capacity_tokens"]
        resident = (context + page - 1) // page * page
        fresh = context + output_tokens
        reservation = output_tokens - 1 + page - 1
        batch_max = max(0, (tokens - fresh) // (resident + reservation) + 1)
        capacity_basis["calculated_scheduler_max_batch"] = batch_max
    if batch_max < 1:
        raise ValueError("batch maximum must be positive")
    cases, limits = [], {}
    for ep in eps:
        max_ratio = 72 // ep - 1
        limits[str(ep)] = max_ratio
        equivalent = {}
        ratios = [(ep * ratio, f"{ratio}:1")
                  for ratio in range(1, max_ratio + 1)]
        if ep == 64:
            ratios = [(8, "1:8")]
        for attention_gpus, ratio_label in ratios:
            for batch in range(1, batch_max + 1):
                per_attention = (batch + 1) // 2
                tokens = attention_gpus * per_attention
                equivalent.setdefault(tokens, []).append({
                    "attention_to_ffn_ratio": ratio_label,
                    "attention_gpus": attention_gpus,
                    "batch_per_attention_gpu": batch,
                    "real_microbatch_sizes": [batch // 2, per_attention],
                    "tokens_per_attention_gpu_per_microbatch": per_attention,
                    "active_gpus": attention_gpus + ep,
                })
        for tokens, inputs in sorted(equivalent.items()):
            rows = (tokens + ep - 1) // ep
            case = {
                "case_id": f"ep64_tokens{tokens}" if ep == 64 else f"ep{ep}_rows{rows}",
                "ep": ep, "rows_per_rank": rows, "global_tokens": tokens,
                "routes_per_expert_min": tokens * top_k // experts,
                "routes_per_expert_max": (tokens * top_k + experts - 1) // experts,
                "equivalent_inputs": inputs,
            }
            if ep == 64:
                case["valid_rows_per_rank"] = [tokens // ep + int(r < tokens % ep)
                                               for r in range(ep)]
            cases.append(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "contract": {"model": "Qwen3.5-397B-A17B" if qwen35 else "Qwen3-235B-A22B",
                     "experts": experts, "hidden": 4096,
                     "intermediate": intermediate, "top_k": top_k,
                     "activation_dtype": "fp8", "weight_dtype": "fp4",
                     "microbatches": 2, "batch_min": 1, "batch_max": batch_max,
                     "tokens_per_attention_gpu": list(range(1, (batch_max + 1) // 2 + 1)),
                     "attention_tp": 1, "gpu_budget": 72,
                     "num_sms": 150, "bucket": 3840, "eps": list(eps),
                     "routing": "integer-balanced rank-interleaved expert sequence"},
        "capacity_basis": capacity_basis,
        "max_attention_to_ffn_ratio": limits,
        "ep64_policy": "A1:F1 infeasible (128 GPUs); measure A1:F8 with 8 attention GPUs",
        "cases": cases,
    }, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "unique_cases": len(cases),
                      "batch_max": batch_max,
                      "equivalent_inputs": sum(len(c["equivalent_inputs"]) for c in cases),
                      "per_ep": {str(ep): sum(c["ep"] == ep for c in cases) for ep in eps}}))


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


def artifact_files(manifest_path: Path, root: Path) -> None:
    """Print an rsync files-from list without traversing caches or other ranks."""
    manifest = json.loads(manifest_path.read_text())
    paths = {manifest_path.relative_to(root)}
    for name in ("coverage.json", "audit_summary.json", "decomposition.csv",
                 "ep_scaling.csv", "equivalent_inputs.csv"):
        if (root / name).is_file():
            paths.add(Path(name))
    for case in manifest["cases"]:
        completed = list((root / case["case_id"]).glob("*/decomposition.json"))
        if len(completed) > 1:
            raise ValueError(f"Duplicate completed capture: {case['case_id']}")
        if not completed:
            continue
        run = completed[0].parent
        for name in ("decomposition.json", "decomposition.csv", "baseline.json",
                     "profiled.json", "profile_megamoe_iket.sbatch",
                     "ranks/rank-0/nsys/megamoe-rank0.cuda_gpu_kern_sum.csv",
                     "ranks/rank-0/nsys/megamoe-rank0.nsys-rep"):
            path = run / name
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.add(path.relative_to(root))
        if (run / "source-snapshot.txt").is_file():
            paths.add((run / "source-snapshot.txt").relative_to(root))
    print("\n".join(str(path) for path in sorted(paths)))


def fetch(root: Path, remote_root: str, remote_script: str, host: str) -> None:
    """Fetch completed rank-0 captures, then validate the actual local copies."""
    command = ["python3", remote_script, "files", remote_root + "/manifest.json", remote_root]
    listing = subprocess.check_output(["ssh", host, shlex.join(command)], text=True)
    root.mkdir(parents=True, exist_ok=True)
    files = root / "artifact_files.txt"
    files.write_text(listing)
    subprocess.run(["rsync", "-a", "--files-from=" + str(files),
                    host + ":" + remote_root.rstrip("/") + "/", str(root) + "/"], check=True)
    collect(root / "manifest.json", root)
    report = audit(root / "manifest.json", root, quiet=True)
    print(json.dumps({key: report[key] for key in (
        "complete", "artifact_contracts_checked", "samples_checked", "expected_cases")}))


def audit(manifest_path: Path, root: Path, *, quiet: bool = False) -> dict:
    """Check capture completeness and contracts without another GPU run."""
    manifest = json.loads(manifest_path.read_text())
    planned = manifest["contract"]
    eps = planned.get("eps", [2, 4, 8, 16, 32, 64])
    batches = range(planned["batch_min"], planned["batch_max"] + 1)
    expected = {(ep, f"{ratio}:1", batch) for ep in eps if ep != 64
                for ratio in range(1, 72 // ep) for batch in batches}
    if 64 in eps:
        expected.update((64, "1:8", batch) for batch in batches)
    for extra in manifest.get("supplemental_inputs", []):
        key = (extra["ep"], extra["attention_to_ffn_ratio"],
               extra["batch_per_attention_gpu"])
        assert key not in expected, (key, "duplicate supplemental input")
        assert key[0] in eps and key[2] > 0
        expected.add(key)
    required_iterations = manifest.get("final_validation", {}).get("baseline_iterations")
    required_source = manifest.get("final_validation", {}).get("source")
    required_samples = manifest.get("final_validation", {}).get("phase_samples", 3)
    assert isinstance(required_samples, int) and required_samples > 0
    experts, top_k = planned["experts"], planned["top_k"]
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
        if required_source:
            source = (path.parent / "source-snapshot.txt").read_text().strip()
            assert Path(source).name == required_source, (case["case_id"], source)
        assert (contract["ep_size"], contract["global_tokens_per_launch"]) == key
        assert contract["bucket"] == planned["bucket"]
        assert tuple(contract[k] for k in ("hidden", "intermediate", "experts", "top_k", "num_sms")) == tuple(planned[k] for k in ("hidden", "intermediate", "experts", "top_k", "num_sms"))
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
        assert contract["routes_per_expert_min"] == tokens * top_k // experts
        assert contract["routes_per_expert_max"] == (tokens * top_k + experts - 1) // experts
        with path.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(result["samples"]) == required_samples
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
            if name == "baseline.json":
                iterations = info["timing_iterations"]
                if required_iterations is not None:
                    assert iterations == required_iterations, (case["case_id"], iterations)
                csv_path = path.parent / "ranks/rank-0/nsys/megamoe-rank0.cuda_gpu_kern_sum.csv"
                text = csv_path.read_text()
                kernels = [row for row in csv.DictReader(io.StringIO(text[text.index("Time (%),"):]))
                           if "sm100_fp8_fp4_mega_moe_impl" in row["Name"]]
                assert len(kernels) == 1 and int(kernels[0]["Instances"]) == iterations
        checked += 1
    assert set(aliases) == expected and len(aliases) == len(expected)
    report = {"complete": not missing, "artifact_contracts_checked": checked,
              "samples_checked": checked * required_samples, "expected_cases": len(manifest["cases"]),
              "equivalent_inputs_checked": len(aliases), "missing": missing,
              "checks": ["scope and deduplication", "input and balance contracts",
                         *(["final source archive"] if required_source else []),
                         f"{required_samples} samples and six markers", "per-sample phase closure",
                         "normal/profiled success", "ordinary kernel sample count",
                         "EP64 exact-combine results"]}
    (root / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not quiet:
        print(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("output", type=Path)
    p.add_argument("--model", choices=("qwen3", "qwen35"), default="qwen3")
    p.add_argument("--batch-max", type=int, default=7)
    p.add_argument("--capacity", type=Path, help="Capacity basis JSON; calculates batch maximum")
    p = sub.add_parser("collect")
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    p = sub.add_parser("audit")
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    p = sub.add_parser("files", help="List completed capture artifacts for rsync --files-from")
    p.add_argument("manifest", type=Path)
    p.add_argument("root", type=Path)
    p = sub.add_parser("fetch", help="Fetch completed rank-0 captures and audit local copies")
    p.add_argument("root", type=Path)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--remote-script", required=True)
    p.add_argument("--host", default="oci-hsg")
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.output, args.model, args.batch_max, args.capacity)
    elif args.command == "collect":
        collect(args.manifest, args.root)
    elif args.command == "files":
        artifact_files(args.manifest, args.root)
    elif args.command == "fetch":
        fetch(args.root, args.remote_root, args.remote_script, args.host)
    else:
        audit(args.manifest, args.root)
