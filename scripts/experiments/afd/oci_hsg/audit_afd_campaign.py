#!/usr/bin/env python3
"""Strictly audit an FMHA-only campaign and report E2E latency deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--pool-tool", type=Path, required=True)
    parser.add_argument("--frontier-manifest", type=Path, required=True)
    parser.add_argument("--source-html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="audit only durable completions while other cases remain active",
    )
    parser.add_argument(
        "--tray-job",
        action="append",
        default=[],
        metavar="TRAYS=JOB_ID",
        help="required completion job; repeat exactly once per tray group",
    )
    return parser.parse_args()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool_tool(path: Path):
    spec = importlib.util.spec_from_file_location("afd_online_case_pool", path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tray_jobs(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        trays_text, separator, job_id = value.partition("=")
        require(
            separator == "=" and trays_text.isdigit() and job_id.isdigit(),
            f"invalid --tray-job: {value}",
        )
        trays = int(trays_text)
        require(trays not in result, f"duplicate tray mapping: {trays}")
        result[trays] = job_id
    return result


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve()
    state_root = args.state_root.resolve()
    manifest = json.loads(args.frontier_manifest.resolve().read_text(encoding="utf-8"))
    require(manifest["status"] == "validated", "frontier manifest is not validated")
    require(sha256(args.source_html.resolve()) == manifest["source_html_sha256"], "source HTML hash drift")
    require(sha256(plan_path) == manifest["plan_sha256"], "plan hash drift")

    with plan_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, "empty plan")
    manifest_cases = [str(case["case_id"]) for case in manifest["cases"]]
    require([row["case_id"] for row in rows] == manifest_cases, "plan/frontier order mismatch")
    require(len(rows) == int(manifest["frontier_case_count"]), "frontier count mismatch")
    require(len(rows) == 30, "optimized-attention 128K frontier must contain 30 points")
    require({row["pareto_source_html_sha256"] for row in rows} == {manifest["source_html_sha256"]}, "row HTML hash mismatch")

    tray_jobs = parse_tray_jobs(args.tray_job)
    expected_trays = {int(row["allocated_trays"]) for row in rows}
    require(expected_trays == set(range(8, 19)), "unexpected tray groups")
    if args.allow_partial:
        require(set(tray_jobs) <= expected_trays, "unknown partial tray-job mapping")
    else:
        require(set(tray_jobs) == expected_trays, "tray-job mapping does not match plan")
        require(len(tray_jobs) == 11, "expected one job for each tray group 8 through 18")
        require(not list((state_root / "claims").glob("*.json")), "claims remain")
    require(not list((state_root / "failed").glob("*.json")), "failed cases remain")

    pool_tool = load_pool_tool(args.pool_tool.resolve())
    cases: list[dict[str, object]] = []
    for row in rows:
        case_id = row["case_id"]
        completion_path = state_root / "completed" / f"{case_id}.json"
        if args.allow_partial and not completion_path.is_file():
            continue
        require(completion_path.is_file(), f"missing completion: {case_id}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        trays = int(row["allocated_trays"])
        require(trays in tray_jobs, f"missing tray-job mapping: {trays}")
        expected_job = tray_jobs[trays]
        require(completion["case_id"] == case_id, f"completion case: {case_id}")
        require(int(completion["allocated_trays"]) == trays, f"completion trays: {case_id}")
        require(completion["job_id"] == expected_job, f"completion job: {case_id}")

        metric_path = Path(completion["metric_path"])
        result_path = Path(completion["result_path"])
        require(sha256(metric_path) == completion["metric_sha256"], f"metric hash: {case_id}")
        require(sha256(result_path) == completion["result_sha256"], f"result hash: {case_id}")
        metric = pool_tool.validate_metric(metric_path, result_path, row)
        result = json.loads(result_path.read_text(encoding="utf-8"))

        batch = int(row["batch_per_attention_dp_lane"])
        expected_num_microbatches = min(2, batch)
        allocated_gpus = int(row["allocated_gpus"])
        attention_gpus = int(result["attention_gpus"])
        ffn_gpus = int(result["ffn_gpus"])
        active_gpus = attention_gpus + ffn_gpus
        expected_prompts = int(result["attention_dp_size"]) * batch
        expected_tokens = expected_prompts * int(result["tokens_per_sample"])

        require(result["afd_model_placement"] == "fmha-only", f"placement: {case_id}")
        require(result["sweep_contract"] == "comprehensive", f"contract: {case_id}")
        require(int(result["allocated_trays"]) == trays, f"result trays: {case_id}")
        require(int(result["allocated_gpus"]) == allocated_gpus, f"allocated GPUs: {case_id}")
        require(int(result["active_gpus"]) == active_gpus, f"active GPUs: {case_id}")
        require(int(result["idle_allocated_gpus"]) == allocated_gpus - active_gpus, f"idle GPUs: {case_id}")
        require(int(result["attention_tp_size"]) == int(row["attention_tp"]), f"ATP: {case_id}")
        require(int(result["ffn_ep_size"]) == int(row["ffn_ep"]), f"FEP: {case_id}")
        require(result["normalized_attention_to_ffn_active_gpu_ratio"] == f"{row['normalized_af_ratio']}:1", f"ratio: {case_id}")
        require(int(result["batch_per_attention_dp_lane"]) == batch, f"batch: {case_id}")

        microbatches = [int(value) for value in result["microbatch_real_sizes_per_attention_dp_lane"]]
        require(len(microbatches) == expected_num_microbatches, f"microbatch count: {case_id}")
        require(sum(microbatches) == batch and all(value > 0 for value in microbatches), f"microbatch split: {case_id}")
        require(int(metric["num_microbatches"]) == expected_num_microbatches, f"metric N: {case_id}")

        prompt_contract = result["prompt_contract"]
        require(int(prompt_contract["target_context_tokens"]) == int(row["isl_tokens"]), f"ISL: {case_id}")
        require(int(prompt_contract["output_prompt_count"]) == expected_prompts, f"prompt contract: {case_id}")
        require(int(result["samples"]) == expected_prompts, f"sample count: {case_id}")
        require(int(metric["global_prompts"]) == expected_prompts, f"metric prompts: {case_id}")
        require(int(result["tokens_per_sample"]) == 17, f"tokens/sample: {case_id}")

        decode_steps = [int(value) for value in result["decode_step_ids"]]
        trace_steps = [int(value) for value in result["trace_decode_step_ids"]]
        warmup = int(result["warmup_decode_step_id"])
        require(len(decode_steps) == 15, f"decode steps: {case_id}")
        require(decode_steps == list(range(decode_steps[0], decode_steps[0] + 15)), f"decode sequence: {case_id}")
        require(warmup == decode_steps[0] - 1, f"warmup: {case_id}")
        require(trace_steps == [warmup, *decode_steps], f"trace steps: {case_id}")

        run_dir = result_path.parent
        reports = sorted((run_dir / "ray_logs").glob("minisgl_rank*.nsys-rep"))
        require(len(reports) == active_gpus, f"report count: {case_id}")
        require((run_dir / "SUCCESS").is_file(), f"missing SUCCESS: {case_id}")
        afd_log = (run_dir / "afd.log").read_text(errors="replace")
        require("hot_rpc_loop:error" not in afd_log, f"hot-loop error: {case_id}")
        require("CUDA error" not in afd_log, f"CUDA error: {case_id}")

        prior_e2e_ms = float(row["prior_e2e_median_interval_ms"])
        current_e2e_ms = float(result["median_interval_ms"])
        require(
            all(math.isfinite(value) and value > 0 for value in (prior_e2e_ms, current_e2e_ms)),
            f"invalid E2E latency: {case_id}",
        )
        cases.append(
            {
                "case_id": case_id,
                "job_id": completion["job_id"],
                "trays": trays,
                "allocated_gpus": allocated_gpus,
                "active_gpus": active_gpus,
                "idle_gpus": allocated_gpus - active_gpus,
                "batch": batch,
                "num_microbatches": expected_num_microbatches,
                "microbatch_sizes": microbatches,
                "prompts": expected_prompts,
                "generated_tokens": expected_tokens,
                "reports": len(reports),
                "prior_e2e_median_interval_ms": prior_e2e_ms,
                "current_e2e_median_interval_ms": current_e2e_ms,
                "e2e_latency_delta_ms": current_e2e_ms - prior_e2e_ms,
                "e2e_latency_delta_percent": (current_e2e_ms / prior_e2e_ms - 1.0) * 100.0,
                "strict_profile_latency_ms": float(metric["cuda_wall_time_mean_ms"]),
                "strict_profile_tps_per_active_gpu": float(metric["tps_per_active_gpu"]),
                "metric_path": str(metric_path),
                "result_path": str(result_path),
            }
        )

    if args.allow_partial:
        require(cases, "partial audit found no completions")
    else:
        require(len(cases) == len(rows), "audit count mismatch")
    weighted_prior = sum(float(case["prior_e2e_median_interval_ms"]) for case in cases) / len(cases)
    weighted_current = sum(float(case["current_e2e_median_interval_ms"]) for case in cases) / len(cases)
    ratios = [
        float(case["current_e2e_median_interval_ms"])
        / float(case["prior_e2e_median_interval_ms"])
        for case in cases
    ]
    geometric_mean_ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    delta_percentages = [(value - 1.0) * 100.0 for value in ratios]
    report = {
        "status": "partial-pass" if args.allow_partial else "pass",
        "comparison_basis": "afd-result.json median_interval_ms (E2E decode-step latency)",
        "case_count": len(cases),
        "planned_case_count": len(rows),
        "tray_jobs": {str(key): value for key, value in sorted(tray_jobs.items())},
        "arithmetic_mean_prior_e2e_ms": weighted_prior,
        "arithmetic_mean_current_e2e_ms": weighted_current,
        "arithmetic_mean_e2e_delta_percent": (weighted_current / weighted_prior - 1.0) * 100.0,
        "geometric_mean_e2e_latency_delta_percent": (geometric_mean_ratio - 1.0) * 100.0,
        "median_case_e2e_latency_delta_percent": statistics.median(delta_percentages),
        "faster_case_count": sum(value < 1.0 - 1e-12 for value in ratios),
        "unchanged_case_count": sum(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12) for value in ratios),
        "slower_case_count": sum(value > 1.0 + 1e-12 for value in ratios),
        "minimum_case_e2e_latency_delta_percent": min(delta_percentages),
        "maximum_case_e2e_latency_delta_percent": max(delta_percentages),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
