#!/usr/bin/env python3
"""Validate completed AFD pool cases and write a paired sweep table."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

from compare_afd_sweep_metrics import geometric_delta, load_metrics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_runtime(result_path: Path, attention: int, ffn: int) -> None:
    logs = result_path.parent / "ray_logs"
    coordinator = (logs / "afd_coordinator.log").read_text()
    require("disable_overlap=False" in coordinator and "mode=dp_lockstep_fixed_lead" in coordinator,
            f"missing overlap runtime proof: {result_path}")
    capacity = math.ceil(attention / ffn)
    mapped = []
    max_rows = set()
    for rank in range(ffn):
        path = logs / f"mlp_dp{rank}_rank0.log"
        startup = []
        with path.open() as stream:
            for line in stream:
                startup.append(line)
                if "fmha_only_ready " in line:
                    break
        ready = startup[-1] if startup else ""
        runtime = next((line for line in startup if " runtime_ready " in line), "")
        require("fmha_only_ready " in ready and "decode_lane_streams=2 " in ready,
                f"missing two-lane readiness: {path}")
        require(all(flag in runtime for flag in ("afd_num_mb=2 ", "perf_mode=1 ", "disable_overlap=False ")),
                f"runtime configuration drift: {path}")
        values = dict(re.findall(r"\b(max_rows|fanin|fanin_capacity)=(\d+)", ready))
        sources = ast.literal_eval(re.search(r"mapped_attn_dps=(\([^)]*\))", ready).group(1))
        require(int(values["fanin_capacity"]) == capacity and int(values["fanin"]) == len(sources)
                and len(sources) in {attention // ffn, capacity}, f"fan-in mismatch: {path}")
        mapped.extend(sources)
        max_rows.add(int(values["max_rows"]))
    require(sorted(mapped) == list(range(attention)), f"attention mapping mismatch: {result_path}")
    require(len(max_rows) == 1 and next(iter(max_rows)) % capacity == 0,
            f"nonuniform symmetric buffer capacity: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_root", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.task_root.resolve()
    tool_path = Path(__file__).resolve().parents[1] / "scripts/experiments/afd/oci_hsg/afd_online_case_pool.py"
    spec = importlib.util.spec_from_file_location("afd_pool", tool_path)
    assert spec is not None and spec.loader is not None
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    plan = pool.read_plan(root / "CASES.csv")
    state = root / "state/pools/CASES"
    job_map = {}
    jobs = {}
    job_logs = {}
    for line in (root / "jobs.jsonl").read_text().splitlines():
        record = json.loads(line)
        job_map[int(record["trays"])] = record["job_id"]
        jobs[record["job_id"]] = record
    require(set(job_map) == {int(row["allocated_trays"]) for row in plan}, "job/plan tray mismatch")
    baseline = load_metrics(args.baseline) if args.baseline else {}
    if args.baseline:
        require(bool(baseline), "empty baseline")
        require(set(baseline) <= {row["case_id"] for row in plan}, "baseline cases absent from plan")
    rows = []
    for case in plan:
        cid = case["case_id"]
        path = state / "completed" / f"{cid}.json"
        if args.allow_partial and not path.exists():
            continue
        require(path.is_file(), f"missing completion: {cid}")
        done = json.loads(path.read_text())
        job_id = done["job_id"]
        require(job_id in jobs and int(jobs[job_id]["trays"]) == int(case["allocated_trays"]),
                f"unrecorded or wrong-tray job: {cid}")
        if job_id not in job_logs:
            log = root / "logs" / f"trays{case['allocated_trays']}-{job_id}.out"
            require(log.is_file(), f"missing execution log: {job_id}")
            job_logs[job_id] = log.read_text(errors="replace")
        terminal = "failed" if done.get("recovered_from_failed_archive") else "completed"
        require(f"{terminal} case={cid} job={job_id}" in job_logs[job_id],
                f"missing case/job execution evidence: {cid}")
        require(pool.case_execution_identity(done["case"]) == pool.case_execution_identity(case), f"completed plan drift: {cid}")
        result_path = Path(done["result_path"])
        metric = pool.validate_metric(Path(done["metric_path"]), result_path, case)
        result = json.loads(result_path.read_text())
        require(int(metric["num_microbatches"]) == 2, f"microbatch count drift: {cid}")
        require(int(result["afd_perf_context_len"]) == int(case["isl_tokens"]),
                f"resident context length drift: {cid}")
        attention = int(case.get("attention_gpus_override") or int(case["normalized_af_ratio"]) * int(case["ffn_ep"]))
        ffn = int(case["ffn_ep"])
        require(int(case["attention_tp"]) == 1, f"runtime audit requires ATP1: {cid}")
        validate_runtime(result_path, attention, ffn)
        active = attention + ffn
        batch = int(case["batch_per_attention_dp_lane"])
        dp = attention // int(case["attention_tp"])
        for field, expected in {
            "attention_gpus": attention,
            "ffn_gpus": ffn,
            "active_gpus": active,
            "allocated_gpus": int(case["allocated_gpus"]),
            "attention_dp_size": dp,
            "batch_per_attention_dp_lane": batch,
            "context_tokens": int(case["isl_tokens"]),
        }.items():
            require(int(result[field]) == expected, f"result {field} mismatch: {cid}")
        require(result["afd_model_placement"] == "attn-only", f"wrong placement: {cid}")
        require(result["megamoe_expert_weight_dtype"] == case["megamoe_expert_weight_dtype"], f"wrong weight dtype: {cid}")
        require(float(result["afd_memory_ratio"]) == float(case["afd_memory_ratio"]), f"memory ratio drift: {cid}")
        if case["afd_num_pages"] == "none":
            require(result["afd_num_pages_override"] is None, f"unexpected page override: {cid}")
        latency = float(metric["cuda_wall_time_mean_ms"])
        tps = float(metric["tps_per_active_gpu"])
        require(latency > 0 and math.isfinite(latency), f"invalid latency: {cid}")
        require(math.isclose(tps, 1000 / latency * dp * batch / active, rel_tol=1e-8), f"throughput arithmetic: {cid}")
        old = baseline.get(cid)
        rows.append({
            "case_id": cid, "job_id": done["job_id"],
            "trays": int(case["allocated_trays"]), "attention_gpus": attention,
            "ffn_gpus": ffn, "ep": ffn, "batch": batch,
            "latency_ms": latency, "tps_per_user": 1000 / latency,
            "tps_per_gpu": tps, "samples": metric["sample_count"],
            "outliers": metric["outlier_count"],
            "old_latency_ms": old["cuda_wall_time_mean_ms"] if old else "",
            "old_tps_per_gpu": old["tps_per_active_gpu"] if old else "",
            "latency_delta_pct": (latency / old["cuda_wall_time_mean_ms"] - 1) * 100 if old else "",
            "tps_delta_pct": (tps / old["tps_per_active_gpu"] - 1) * 100 if old else "",
        })
    counts = {kind: len(list((state / kind).glob("*.json"))) for kind in ("completed", "claims", "failed")}
    if not args.allow_partial:
        require(counts == {"completed": len(plan), "claims": 0, "failed": 0}, f"unfinished pool: {counts}")
    paired = [row for row in rows if row["old_latency_ms"] != ""]
    summary = {
        "status": "complete" if len(rows) == len(plan) and not counts["failed"] and not counts["claims"] else "partial",
        "planned": len(plan), "validated": len(rows), "pool_counts": counts,
        "baseline_cases": len(baseline), "paired_cases": len(paired),
        "validated_by_ep": dict(Counter(row["ep"] for row in rows)),
        "validated_by_trays": dict(Counter(row["trays"] for row in rows)),
        "job_by_trays": job_map,
        "accepted_by_job": dict(Counter(row["job_id"] for row in rows)),
    }
    if paired:
        for name, new, old in [("latency", "latency_ms", "old_latency_ms"), ("tps", "tps_per_gpu", "old_tps_per_gpu")]:
            summary[f"{name}_geomean_delta_pct"] = geometric_delta([
                row[new] / row[old] for row in paired
            ])
        summary["paired_faster"] = sum(row["latency_delta_pct"] < 0 for row in paired)
        summary["paired_by_batch"] = {}
        for batch in sorted({row["batch"] for row in paired}):
            group = [row for row in paired if row["batch"] == batch]
            summary["paired_by_batch"][batch] = {
                "cases": len(group),
                "faster": sum(row["latency_delta_pct"] < 0 for row in group),
                "tps_geomean_delta_pct": geometric_delta([
                    row["tps_per_gpu"] / row["old_tps_per_gpu"]
                    for row in group
                ]),
            }
    output = root / "report"
    output.mkdir(exist_ok=True)
    if rows:
        with (output / "sweep.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
