#!/usr/bin/env python3
"""Extract stable target-batch latency from per-step AFD CUDA spans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import statistics
import tempfile


METRIC_VERSION = "20260804-attention-cuda-execution-span-v14"
LATENCY_BASIS = (
    "afd_attention_cuda_execution_critical_target_batch_dominant_range_mean"
)
CUDA_SPAN_API_VERSION = "20260817-attention-cuda-execution-span-v1"
CUDA_SPAN_BASIS = "first_attention_cuda_start_to_last_attention_cuda_end"
MAX_MEDIAN_DIFF_PERCENT_LIMIT = 10.0
DOMINANT_RANGE_PERCENT_LIMIT = 10.0
MAX_OUTLIER_COUNT = 5
EXPECTED_CUDA_SPAN_MODULE_SHA256 = (
    "44669725b76e806aef1f73de732748ea5944f8b4a05dc1a1fade0e2f3566d2ce"
)
SELECTED_STEPS = 15
MIN_RETAINED_SAMPLES = SELECTED_STEPS - MAX_OUTLIER_COUNT
TRACE_WARMUP_STEPS = 1
SELECTED_WINDOW_POLICY = (
    "trace first exact full-resident decode as warmup, then measure the next "
    "exact 15-step decode window"
)
LATENCY_SAMPLE_POLICY = (
    "use the largest position-independent latency cluster among the 15 "
    "trace-selected target-batch steps whose full max-minus-min range is <=10% "
    "of its median; retain at least 10 steps, break equal-size ties by tighter "
    "range then higher mean, and use the retained arithmetic mean as latency"
)
FMHA_METRIC_VERSION = "20260820-fmha-only-dual-role-cuda-graph-span-v1"
FMHA_LATENCY_BASIS = "max_complete_cuda_graph_span_across_profiled_attention_and_model_roles"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_latency_samples(
    measured_steps: list[int],
    critical_ms: list[float],
    max_outlier_count: int = MAX_OUTLIER_COUNT,
) -> dict[str, object]:
    if len(measured_steps) != SELECTED_STEPS or len(critical_ms) != SELECTED_STEPS:
        raise RuntimeError(
            "latency policy requires the complete 15-step target-batch window: "
            f"steps={measured_steps} critical_ms={critical_ms}"
        )
    if not all(math.isfinite(value) and value > 0 for value in critical_ms):
        raise RuntimeError(f"invalid CUDA critical spans {critical_ms}")
    min_retained_samples = SELECTED_STEPS - max_outlier_count

    initial_median_ms = statistics.median(critical_ms)
    initial_min_ms = min(critical_ms)
    initial_max_ms = max(critical_ms)
    initial_diff_ms = initial_max_ms - initial_median_ms
    initial_diff_percent = initial_diff_ms * 100 / initial_median_ms
    initial_range_ms = initial_max_ms - initial_min_ms
    initial_range_percent = initial_range_ms * 100 / initial_median_ms

    ordered = sorted(
        zip(critical_ms, measured_steps), key=lambda item: (item[0], item[1])
    )
    candidates: list[dict[str, object]] = []
    for start in range(len(ordered)):
        for stop in range(start + 1, len(ordered) + 1):
            group = ordered[start:stop]
            group_ms = [value for value, _ in group]
            group_median_ms = statistics.median(group_ms)
            group_range_ms = group_ms[-1] - group_ms[0]
            group_range_percent = group_range_ms * 100 / group_median_ms
            if group_range_percent <= DOMINANT_RANGE_PERCENT_LIMIT:
                candidates.append(
                    {
                        "steps": [step for _, step in group],
                        "values": group_ms,
                        "range_percent": group_range_percent,
                        "mean_ms": statistics.fmean(group_ms),
                    }
                )
    if not candidates:
        raise RuntimeError(f"no finite target-batch latency cluster: {critical_ms}")
    dominant = min(
        candidates,
        key=lambda candidate: (
            -len(candidate["steps"]),
            candidate["range_percent"],
            -candidate["mean_ms"],
            candidate["steps"],
        ),
    )
    if len(dominant["steps"]) < min_retained_samples:
        raise RuntimeError(
            "target-batch latency has too many outliers for a dominant range: "
            f"required_retained={min_retained_samples} "
            f"largest_cluster={len(dominant['steps'])} "
            f"range_limit={DOMINANT_RANGE_PERCENT_LIMIT:.3f}% "
            f"steps={measured_steps} critical_ms={critical_ms}"
        )
    retained_step_set = set(dominant["steps"])
    retained_steps = [step for step in measured_steps if step in retained_step_set]
    retained_ms = [
        value
        for step, value in zip(measured_steps, critical_ms)
        if step in retained_step_set
    ]
    excluded_steps = [
        step for step in measured_steps if step not in retained_step_set
    ]

    median_ms = statistics.median(retained_ms)
    min_ms = min(retained_ms)
    mean_ms = statistics.fmean(retained_ms)
    max_ms = max(retained_ms)
    range_ms = max_ms - min_ms
    range_percent = range_ms * 100 / median_ms
    diff_ms = max_ms - median_ms
    diff_percent = diff_ms * 100 / median_ms
    if (
        range_percent > DOMINANT_RANGE_PERCENT_LIMIT
        or diff_percent > MAX_MEDIAN_DIFF_PERCENT_LIMIT
        or len(excluded_steps) > max_outlier_count
    ):
        raise RuntimeError(
            "invalid retained target-batch dominant range: "
            f"range_percent={range_percent:.3f} "
            f"range_limit={DOMINANT_RANGE_PERCENT_LIMIT:.3f} "
            f"max_median_diff_percent={diff_percent:.3f} "
            f"max_median_limit={MAX_MEDIAN_DIFF_PERCENT_LIMIT:.3f} "
            f"eligible_steps={measured_steps} retained_steps={retained_steps} "
            f"critical_ms={critical_ms} retained_critical_ms={retained_ms}"
        )
    return {
        "initial_median_ms": initial_median_ms,
        "initial_min_ms": initial_min_ms,
        "initial_max_ms": initial_max_ms,
        "initial_diff_ms": initial_diff_ms,
        "initial_diff_percent": initial_diff_percent,
        "initial_range_ms": initial_range_ms,
        "initial_range_percent": initial_range_percent,
        "retained_steps": retained_steps,
        "retained_ms": retained_ms,
        "excluded_steps": excluded_steps,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "mean_ms": mean_ms,
        "max_ms": max_ms,
        "range_ms": range_ms,
        "range_percent": range_percent,
        "diff_ms": diff_ms,
        "diff_percent": diff_percent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument(
        "--cuda-span-module", "--cuda-module", dest="cuda_span_module",
        type=Path, required=True,
    )
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_cuda_span_module(path: Path):
    actual = sha256(path)
    if actual != EXPECTED_CUDA_SPAN_MODULE_SHA256:
        raise RuntimeError(
            f"CUDA span module drift: {actual} != "
            f"{EXPECTED_CUDA_SPAN_MODULE_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("cuda_execution_span", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA span module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.CUDA_SPAN_API_VERSION != CUDA_SPAN_API_VERSION:
        raise RuntimeError("CUDA span API-version drift")
    if module.AFD_SPAN_BASIS != CUDA_SPAN_BASIS:
        raise RuntimeError("CUDA span basis drift")
    return module


def load_matching_plan(
    path: Path, data: dict[str, object], case_id: str
) -> dict[str, str]:
    ratio = int(str(data["normalized_attention_to_ffn_active_gpu_ratio"]).split(":")[0])
    ffn_ep = int(data["ffn_ep_size"])
    batch = int(data["batch_per_attention_dp_lane"])
    context_range = data["context_range_tokens"]
    if not isinstance(context_range, dict):
        raise RuntimeError("result has no context range")
    if data.get("prompt_mode") == "irregular":
        shape = f"r{int(context_range['min'])}-{int(context_range['max'])}"
    else:
        shape = f"i{int(data['context_tokens'])}"
    default_case_id = f"{shape}-fep{ffn_ep}-r{ratio}-atp1-b{batch}"
    precision = str(data.get("megamoe_expert_weight_dtype", ""))
    if case_id not in {default_case_id, f"{default_case_id}-{precision}"}:
        raise RuntimeError(f"case ID does not match result: {case_id}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames and "rerun_index" in reader.fieldnames:
            rows = [row for row in reader if row["case_id"] == case_id]
            if len(rows) != 1 or rows[0]["status"] != "needs_corrected_target_batch_rerun":
                raise RuntimeError(f"future plan match is not unique: {case_id} {rows}")
            rows[0]["plan_index"] = rows[0]["rerun_index"]
            rows[0]["active_gpus"] = str((ratio + 1) * ffn_ep)
            return rows[0]
    coordinate = (
        int(data["context_tokens"]),
        ffn_ep,
        ratio,
        batch,
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if (
                int(row["context_tokens"]),
                int(row["ffn_ep"]),
                int(row["normalized_af_ratio"]),
                int(row["batch_per_attention_dp_lane"]),
            )
            == coordinate
        ]
    if len(rows) != 1:
        raise RuntimeError(f"plan coordinate match is not unique: {coordinate} {rows}")
    if rows[0]["status"] != "pending":
        raise RuntimeError(
            f"new CUDA extraction targeted non-pending coordinate {coordinate}"
        )
    return rows[0]


def representative_ranks(data: dict[str, object]) -> tuple[list[int], list[str]]:
    if int(data["attention_tp_size"]) != 1:
        raise RuntimeError("this goal fixes attention TP at 1")
    placement = data["placement"]
    if not isinstance(placement, dict):
        raise RuntimeError("result has no placement mapping")
    rows = placement["attention_dp_rank_nodes"]
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != 1 for row in rows
    ):
        raise RuntimeError("TP1 placement must have one node per attention DP rank")
    if len(rows) != int(data["attention_dp_size"]):
        raise RuntimeError("placement/attention-DP mismatch")
    ranks: list[int] = []
    nodes: list[str] = []
    for dp_index, row in enumerate(rows):
        node = str(row[0])
        if node not in nodes:
            nodes.append(node)
            ranks.append(dp_index + 1)
    expected_attention_nodes = len(
        {
            str(node)
            for row in rows
            for node in row
        }
    )
    if len(ranks) != expected_attention_nodes:
        raise RuntimeError("attention-node representative selection drift")
    return ranks, nodes


def fmha_graph_spans_ms(path: Path) -> tuple[list[float], list[int]]:
    """Return ordered complete CUDA-graph spans and node counts."""

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "StringIds",
        }
        if not required <= tables:
            raise RuntimeError(f"{path} lacks tables {sorted(required - tables)}")
        launches = connection.execute(
            """
            SELECT r.correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id=r.nameId
            WHERE s.value LIKE 'cudaGraphLaunch%'
            ORDER BY r.start
            """
        ).fetchall()
        durations: list[float] = []
        counts: list[int] = []
        for (correlation_id,) in launches:
            row = (None, None, 0)
            if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
                row = connection.execute(
                    """
                    SELECT min(start),max(end),count(*)
                    FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                    WHERE correlationId=?
                    """,
                    (correlation_id,),
                ).fetchone()
            if not int(row[2]):
                row = connection.execute(
                    """
                    SELECT min(start),max(end),count(*)
                    FROM CUPTI_ACTIVITY_KIND_KERNEL
                    WHERE correlationId=? AND graphId IS NOT NULL
                    """,
                    (correlation_id,),
                ).fetchone()
            start, end, count = row
            if not int(count) or start is None or end is None or int(end) <= int(start):
                raise RuntimeError(
                    f"{path} graph launch {correlation_id} has no positive CUDA span"
                )
            durations.append((int(end) - int(start)) / 1e6)
            counts.append(int(count))
    return durations, counts


def collapse_fmha_graph_spans(
    durations_ms: list[float],
    graph_node_counts: list[int],
    trace_steps: list[int],
    launches_per_step: int,
) -> tuple[list[float], int, list[list[float]]]:
    """Collapse ordered graph launches into one CUDA span per trace step."""

    if launches_per_step < 1:
        raise RuntimeError(f"invalid graph launches per trace step: {launches_per_step}")
    expected = len(trace_steps) * launches_per_step
    if len(durations_ms) != expected or len(graph_node_counts) != expected:
        raise RuntimeError(
            "FMHA graph launch count does not match trace topology: "
            f"durations={len(durations_ms)} counts={len(graph_node_counts)} "
            f"steps={len(trace_steps)} launches_per_step={launches_per_step} "
            f"expected={expected}"
        )
    if not all(math.isfinite(value) and value > 0 for value in durations_ms):
        raise RuntimeError(f"invalid FMHA graph spans: {durations_ms}")
    if len(set(graph_node_counts)) != 1:
        raise RuntimeError(
            f"FMHA graph node counts are not stable: {graph_node_counts}"
        )
    grouped = [
        durations_ms[index : index + launches_per_step]
        for index in range(0, expected, launches_per_step)
    ]
    return [sum(group) for group in grouped], graph_node_counts[0], grouped


def fmha_profile_launches_per_step(
    attention_dp_size: int,
    mlp_dp_size: int,
) -> tuple[int, int]:
    """Return attention/model graph launches for the grouped FMHA-only path."""

    if (
        attention_dp_size < 1
        or mlp_dp_size < 1
        or attention_dp_size % mlp_dp_size
    ):
        raise RuntimeError(
            "FMHA attention/model DP fan-in is not integral: "
            f"attention_dp={attention_dp_size} model_dp={mlp_dp_size}"
        )
    # Every attention DP owns one graph. Each model DP now receives one grouped
    # command containing its complete source fan-in, so it also owns one graph
    # per step independent of A:F ratio.
    return 1, 1


def extract_fmha_only(
    *,
    args: argparse.Namespace,
    data: dict[str, object],
    plan: dict[str, str],
    span_module: object,
    measured_steps: list[int],
    warmup_step: int,
    max_outlier_count: int,
) -> None:
    """Extract the critical span across one attention and one model graph."""

    active_gpus = int(data["active_gpus"])
    attention_workers = int(data["attention_workers"])
    attention_dp_size = int(data["attention_dp_size"])
    placement = data.get("placement")
    if not isinstance(placement, dict):
        raise RuntimeError("FMHA result has no placement proof")
    mlp_dp_rank_nodes = placement.get("mlp_dp_rank_nodes")
    if not isinstance(mlp_dp_rank_nodes, list) or not mlp_dp_rank_nodes:
        raise RuntimeError("FMHA placement has no model-DP mapping")
    mlp_dp_size = len(mlp_dp_rank_nodes)
    attention_launches_per_step, model_launches_per_step = (
        fmha_profile_launches_per_step(attention_dp_size, mlp_dp_size)
    )
    profile_specs = (
        ("attention", 1, attention_launches_per_step),
        ("model", attention_workers + 1, model_launches_per_step),
    )
    reports = span_module.gpu_worker_reports(  # type: ignore[attr-defined]
        args.result.parent / "ray_logs", active_gpus
    )
    trace_steps = [warmup_step, *measured_steps]
    per_profile: dict[str, object] = {}
    per_step: dict[int, list[float]] = {step: [] for step in trace_steps}
    with tempfile.TemporaryDirectory(prefix="afd-fmha-", dir=args.temp_root) as temporary:
        temporary_path = Path(temporary)
        for role, rank, launches_per_step in profile_specs:
            report = reports[rank - 1]
            sqlite_path = temporary_path / f"{role}-rank-{rank}.sqlite"
            span_module.export_nsys(  # type: ignore[attr-defined]
                args.nsys,
                report,
                sqlite_path,
                span_module.AFD_TABLES,  # type: ignore[attr-defined]
            )
            raw_durations, counts = fmha_graph_spans_ms(sqlite_path)
            durations, node_count, grouped_durations = collapse_fmha_graph_spans(
                raw_durations,
                counts,
                trace_steps,
                launches_per_step,
            )
            label = f"{role}:rank{rank}"
            per_profile[label] = {
                "role": role,
                "rank": rank,
                "report_path": str(report),
                "graph_launches_per_trace_step": launches_per_step,
                "graph_node_count_per_launch": node_count,
                "per_trace_step_cuda_graph_launch_ms": dict(
                    zip(map(str, trace_steps), grouped_durations)
                ),
                "per_trace_step_cuda_graph_ms": dict(
                    zip(map(str, trace_steps), durations)
                ),
            }
            for step, duration in zip(trace_steps, durations):
                per_step[step].append(duration)

    critical_ms = [max(per_step[step]) for step in measured_steps]
    latency = select_latency_samples(
        measured_steps,
        critical_ms,
        max_outlier_count=max_outlier_count,
    )
    mean_ms = float(latency["mean_ms"])
    allocated_gpus = int(data["allocated_gpus"])
    global_prompts = int(data["samples"])
    output = {
        "case_id": plan["case_id"],
        "plan_index": int(plan["plan_index"]),
        "metric_version": FMHA_METRIC_VERSION,
        "latency_basis": FMHA_LATENCY_BASIS,
        "cuda_wall_time_ms": mean_ms,
        "cuda_wall_time_mean_ms": mean_ms,
        "cuda_wall_time_median_ms": latency["median_ms"],
        "cuda_wall_time_max_ms": latency["max_ms"],
        "dominant_range_percent": latency["range_percent"],
        "dominant_range_percent_limit": DOMINANT_RANGE_PERCENT_LIMIT,
        "max_median_diff_percent": latency["diff_percent"],
        "max_median_diff_percent_limit": MAX_MEDIAN_DIFF_PERCENT_LIMIT,
        "max_outlier_count": max_outlier_count,
        "outlier_count": len(latency["excluded_steps"]),
        "sample_count": len(latency["retained_ms"]),
        "eligible_target_batch_sample_count": len(critical_ms),
        "excluded_decode_step_ids": latency["excluded_steps"],
        "latency_decode_step_ids": latency["retained_steps"],
        "warmup_decode_step_id": warmup_step,
        "trace_decode_step_ids": trace_steps,
        "warmup_cuda_execution_critical_ms": max(per_step[warmup_step]),
        "decode_step_ids": measured_steps,
        "per_step_cuda_execution_critical_ms": critical_ms,
        "per_step_per_profiled_role_ms": {
            str(step): per_step[step] for step in trace_steps
        },
        "profiled_roles": per_profile,
        "target_batch_per_attention_dp_lane": int(
            data["batch_per_attention_dp_lane"]
        ),
        "num_microbatches": int(data["cuda_graph"]["num_microbatches"]),
        "global_prompts": global_prompts,
        "active_gpus": active_gpus,
        "allocated_gpus": allocated_gpus,
        "tps_per_active_gpu": global_prompts * 1000 / (active_gpus * mean_ms),
        "tps_per_allocated_gpu_diagnostic": (
            global_prompts * 1000 / (allocated_gpus * mean_ms)
        ),
        "result_path": str(args.result),
        "result_sha256": sha256(args.result),
        "mapping_proof": (
            "the selected attention and model reports each contain exactly the "
            "declared warmup plus 15 ordered graph launches"
        ),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case_id": output["case_id"],
                "cuda_wall_time_ms": mean_ms,
                "dominant_range_percent": latency["range_percent"],
                "sample_count": len(latency["retained_ms"]),
                "tps_per_active_gpu": output["tps_per_active_gpu"],
                "profiled_gpu_ranks": [rank for _, rank, _ in profile_specs],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    args.result = args.result.resolve()
    args.plan = args.plan.resolve()
    args.nsys = args.nsys.resolve()
    args.cuda_span_module = args.cuda_span_module.resolve()
    args.temp_root = args.temp_root.resolve()
    args.output = args.output.resolve()
    for path in (args.result, args.plan, args.nsys, args.cuda_span_module):
        if not path.is_file():
            raise RuntimeError(f"missing required input {path}")
    args.temp_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    span_module = load_cuda_span_module(args.cuda_span_module)
    data = json.loads(args.result.read_text())
    plan = load_matching_plan(args.plan, data, args.case_id)
    max_outlier_count = int(
        plan.get("required_max_outliers", MAX_OUTLIER_COUNT)
    )
    required_retained_steps = int(
        plan.get(
            "required_retained_steps_min",
            SELECTED_STEPS - max_outlier_count,
        )
    )
    if (
        not 0 <= max_outlier_count < SELECTED_STEPS
        or required_retained_steps != SELECTED_STEPS - max_outlier_count
    ):
        raise RuntimeError(
            "plan outlier policy must partition the complete 15-step window: "
            f"retained={required_retained_steps} outliers={max_outlier_count}"
        )
    if data.get("sweep_contract") != "comprehensive":
        raise RuntimeError("result is not from the comprehensive sweep contract")
    cuda_graph = data.get("cuda_graph")
    if not isinstance(cuda_graph, dict):
        raise RuntimeError("result lacks CUDA graph contract")
    num_microbatches = int(cuda_graph.get("num_microbatches", 0))
    batch_per_attention_dp_lane = int(data["batch_per_attention_dp_lane"])
    if (
        cuda_graph.get("enabled") is not True
        or not 1 <= num_microbatches <= batch_per_attention_dp_lane
        or cuda_graph.get("nsys_cuda_graph_trace") != "node"
        or int(cuda_graph.get("nsys_target_batch_per_attention_dp_lane", 0))
        != int(data["batch_per_attention_dp_lane"])
        or int(cuda_graph.get("nsys_capture_decode_steps", 0))
        != SELECTED_STEPS
        or int(cuda_graph.get("nsys_trace_warmup_decode_steps", 0))
        != TRACE_WARMUP_STEPS
        or int(cuda_graph.get("nsys_trace_decode_launches", 0))
        != SELECTED_STEPS + TRACE_WARMUP_STEPS
    ):
        raise RuntimeError(f"result CUDA graph contract mismatch: {cuda_graph}")
    if data.get("selected_window_policy") != SELECTED_WINDOW_POLICY:
        raise RuntimeError(
            "result selected-window policy mismatch: "
            f"{data.get('selected_window_policy')!r}"
        )

    selected = [int(step) for step in data.get("decode_step_ids", [])]
    warmup_step = int(data.get("warmup_decode_step_id", -1))
    traced = [int(step) for step in data.get("trace_decode_step_ids", [])]
    windows = [
        [int(step) for step in window]
        for window in data.get("full_bucket_decode_windows", [])
    ]
    if len(selected) != SELECTED_STEPS or selected not in windows:
        raise RuntimeError(
            f"result lacks an exact selected 15-step full-bucket wave: {selected}"
        )
    if traced != [warmup_step, *selected] or selected[0] != warmup_step + 1:
        raise RuntimeError(
            f"result lacks first-decode warmup followed by 15 measured steps: "
            f"warmup={warmup_step} selected={selected} traced={traced}"
        )

    active_gpus = int(data["active_gpus"])
    allocated_gpus = int(data["allocated_gpus"])
    attention_gpus = int(data["attention_gpus"])
    global_prompts = int(data["samples"])
    if active_gpus != attention_gpus + int(data["ffn_gpus"]):
        raise RuntimeError("active-GPU arithmetic mismatch")
    if allocated_gpus != 4 * int(data["allocated_trays"]):
        raise RuntimeError("allocated-GPU arithmetic mismatch")
    if active_gpus != int(plan["active_gpus"]):
        raise RuntimeError("plan/result active-GPU mismatch")

    if data.get("afd_model_placement") == "fmha-only":
        extract_fmha_only(
            args=args,
            data=data,
            plan=plan,
            span_module=span_module,
            measured_steps=selected,
            warmup_step=warmup_step,
            max_outlier_count=max_outlier_count,
        )
        return

    all_reports = span_module.gpu_worker_reports(
        args.result.parent / "ray_logs", active_gpus
    )
    ranks, nodes = representative_ranks(data)
    reports = [all_reports[rank - 1] for rank in ranks]
    if len(reports) != len(nodes):
        raise RuntimeError("representative report/node mismatch")

    per_rank_spans: list[dict[int, tuple[int, int]]] = []
    diagnostics: list[dict[str, int]] = []
    with tempfile.TemporaryDirectory(
        prefix="afd3d-cuda-", dir=args.temp_root
    ) as temporary:
        temporary_path = Path(temporary)
        for rank, report in zip(ranks, reports):
            sqlite_path = temporary_path / f"attention-rank-{rank}.sqlite"
            span_module.export_nsys(
                args.nsys,
                report,
                sqlite_path,
                span_module.AFD_TABLES,
            )
            spans, rank_diagnostics = span_module.afd_attention_cuda_spans(
                sqlite_path
            )
            if (
                int(rank_diagnostics["graph_trace_record_count_min"]) <= 0
                and int(rank_diagnostics["graph_kernel_count_min"]) <= 0
            ):
                raise RuntimeError(
                    f"rank {rank} lacks both CUDA graph-trace rows and "
                    "graph-ID kernel rows"
                )
            per_rank_spans.append(spans)
            diagnostics.append(rank_diagnostics)
            sqlite_path.unlink()

    common_steps = set.intersection(*(set(spans) for spans in per_rank_spans))
    if warmup_step not in common_steps:
        raise RuntimeError(
            f"first full-resident warmup step {warmup_step} is absent from a "
            "profiled attention-node representative"
        )
    measured, selection_policy = span_module.choose_afd_steps(
        windows, selected, common_steps
    )
    if measured != selected:
        raise RuntimeError(
            "new runs require all 15 selected steps on every attention-node "
            f"representative: selected={selected} measured={measured} "
            f"common={sorted(common_steps)}"
        )
    if selection_policy != "captured subset of run-selected full-bucket wave":
        raise RuntimeError(f"unexpected selection policy {selection_policy}")

    per_step_per_representative_ms = {
        str(step): [
            (spans[step][1] - spans[step][0]) / 1e6
            for spans in per_rank_spans
        ]
        for step in measured
    }
    warmup_per_representative_ms = [
        (spans[warmup_step][1] - spans[warmup_step][0]) / 1e6
        for spans in per_rank_spans
    ]
    warmup_critical_ms = max(warmup_per_representative_ms)
    critical_ms = [
        max(per_step_per_representative_ms[str(step)]) for step in measured
    ]
    latency = select_latency_samples(measured, critical_ms, max_outlier_count)
    latency_steps = list(latency["retained_steps"])
    latency_critical_ms = list(latency["retained_ms"])
    cuda_wall_median_ms = float(latency["median_ms"])
    cuda_wall_mean_ms = float(latency["mean_ms"])
    cuda_wall_max_ms = float(latency["max_ms"])
    max_median_diff_ms = float(latency["diff_ms"])
    max_median_diff_percent = float(latency["diff_percent"])
    tps_per_active_gpu = (
        global_prompts * 1000 / (active_gpus * cuda_wall_mean_ms)
    )
    tps_per_allocated_gpu = (
        global_prompts * 1000 / (allocated_gpus * cuda_wall_mean_ms)
    )
    output = {
        "case_id": plan["case_id"],
        "plan_index": int(plan["plan_index"]),
        "metric_version": METRIC_VERSION,
        "latency_basis": LATENCY_BASIS,
        "cuda_wall_time_ms": cuda_wall_mean_ms,
        "cuda_wall_time_mean_ms": cuda_wall_mean_ms,
        "cuda_wall_time_median_ms": cuda_wall_median_ms,
        "cuda_wall_time_max_ms": cuda_wall_max_ms,
        "max_median_diff_ms": max_median_diff_ms,
        "max_median_diff_percent": max_median_diff_percent,
        "max_median_diff_percent_limit": MAX_MEDIAN_DIFF_PERCENT_LIMIT,
        "max_median_stability_check_passed": True,
        "initial_cuda_wall_time_median_ms": latency["initial_median_ms"],
        "initial_cuda_wall_time_min_ms": latency["initial_min_ms"],
        "initial_cuda_wall_time_max_ms": latency["initial_max_ms"],
        "initial_max_median_diff_ms": latency["initial_diff_ms"],
        "initial_max_median_diff_percent": latency["initial_diff_percent"],
        "initial_range_ms": latency["initial_range_ms"],
        "initial_range_percent": latency["initial_range_percent"],
        "dominant_range_min_ms": latency["min_ms"],
        "dominant_range_max_ms": latency["max_ms"],
        "dominant_range_ms": latency["range_ms"],
        "dominant_range_percent": latency["range_percent"],
        "dominant_range_percent_limit": DOMINANT_RANGE_PERCENT_LIMIT,
        "max_outlier_count": max_outlier_count,
        "outlier_count": len(latency["excluded_steps"]),
        "excluded_decode_step_ids": latency["excluded_steps"],
        "sample_count": len(latency_critical_ms),
        "eligible_target_batch_sample_count": len(critical_ms),
        "complete_selected_wave": True,
        "decode_step_ids": measured,
        "eligible_target_batch_decode_step_ids": measured,
        "latency_decode_step_ids": latency_steps,
        "warmup_decode_step_id": warmup_step,
        "trace_decode_step_ids": [warmup_step, *measured],
        "warmup_cuda_execution_critical_ms": warmup_critical_ms,
        "warmup_per_attention_node_representative_ms": (
            warmup_per_representative_ms
        ),
        "decode_step_selection": (
            "15 exact target-batch full-resident steps immediately following "
            "the separately reported first-decode transition warmup"
        ),
        "latency_sample_policy": LATENCY_SAMPLE_POLICY.replace(
            "retain at least 10", f"retain at least {SELECTED_STEPS - max_outlier_count}"
        ),
        "outlier_policy_differs_from_default": (
            max_outlier_count != MAX_OUTLIER_COUNT
        ),
        "target_batch_filter_passed": True,
        "target_batch_per_attention_dp_lane": int(
            data["batch_per_attention_dp_lane"]
        ),
        "target_batch_proof": (
            "the coordinator target-batch marker and every attention worker "
            "agree on the warmup plus all 15 selected full-bucket steps"
        ),
        "per_step_cuda_execution_critical_ms": critical_ms,
        "latency_sample_cuda_execution_critical_ms": latency_critical_ms,
        "per_step_per_attention_node_representative_ms": (
            per_step_per_representative_ms
        ),
        "profiled_gpu_ranks": ranks,
        "profiled_attention_nodes": nodes,
        "per_rank_mapping_diagnostics": diagnostics,
        "pipeline_reply_to_launch_lead_steps": span_module.AFD_PIPELINE_LEAD_STEPS,
        "cuda_span_definition": (
            "first attention CUDA graph/kernel start through final "
            "attention sampling/writeback kernel end; inter-step gap excluded"
        ),
        "global_prompts": global_prompts,
        "active_gpus": active_gpus,
        "allocated_gpus": allocated_gpus,
        "tps_per_active_gpu": tps_per_active_gpu,
        "tps_per_allocated_gpu_diagnostic": tps_per_allocated_gpu,
        "result_path": str(args.result),
        "result_sha256": sha256(args.result),
        "cuda_span_module_path": str(args.cuda_span_module),
        "cuda_span_module_sha256": sha256(args.cuda_span_module),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case_id": output["case_id"],
                "cuda_wall_time_ms": cuda_wall_mean_ms,
                "cuda_wall_time_median_ms": cuda_wall_median_ms,
                "cuda_wall_time_max_ms": cuda_wall_max_ms,
                "max_median_diff_percent": max_median_diff_percent,
                "sample_count": len(latency_critical_ms),
                "eligible_target_batch_sample_count": len(critical_ms),
                "outlier_count": len(latency["excluded_steps"]),
                "dominant_range_percent": latency["range_percent"],
                "tps_per_active_gpu": tps_per_active_gpu,
                "profiled_gpu_ranks": ranks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
