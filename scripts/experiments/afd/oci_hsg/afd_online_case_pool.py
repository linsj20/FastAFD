#!/usr/bin/env python3
"""Atomic online case selection for AFD allocation workers."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys


METRIC_VERSION = "20260804-attention-cuda-execution-span-v14"
FMHA_METRIC_VERSION = "20260820-fmha-only-dual-role-cuda-graph-span-v1"
CASE_ID_PATTERN = re.compile(
    r"(?:i[1-9][0-9]*|r[1-9][0-9]*-[1-9][0-9]*)-"
    r"fep[1-9][0-9]*-r[1-9][0-9]*-atp[1-9][0-9]*-b[1-9][0-9]*"
    r"(?:-(?:fp8|fp4))?"
)
STATE_KINDS = ("claims", "completed", "failed")
CASE_EXECUTION_FIELDS = (
    "case_id",
    "legacy_isl_mode",
    "context_spec",
    "isl_tokens",
    "isl_min_tokens",
    "isl_max_tokens",
    "batch_per_attention_dp_lane",
    "normalized_af_ratio",
    "attention_tp",
    "ffn_ep",
    "allocated_trays",
    "allocated_gpus",
    "prompt_mode",
    "run_afd_argv",
    "status",
    "required_capture_policy",
    "required_target_batch_per_attention_dp_lane",
    "required_trace_decode_steps",
    "required_retained_steps_min",
    "required_max_outliers",
    "required_dominant_range_percent_limit",
    "required_max_median_diff_percent_limit",
    "afd_memory_ratio",
    "afd_num_pages",
    "afd_kv_capacity_tokens",
    "require_capacity_max",
    "megamoe_expert_weight_dtype",
)
LEGACY_SELECTION_POLICY = (
    "atomically claim the lowest rerun_index matching allocated_trays; "
    "unclaimed cases remain pending"
)
ISL_DESC_SELECTION_POLICY = (
    "atomically claim fixed-ISL cases before irregular cases, with fixed ISL "
    "descending from 131072 tokens and rerun_index as the stable tie-break; "
    "unclaimed cases remain pending"
)
RATIO_BATCH_SELECTION_POLICY = (
    "atomically claim fixed-ISL cases before irregular cases, with fixed ISL, "
    "normalized A:F ratio, and batch size all descending and rerun_index as "
    "the stable tie-break; unclaimed cases remain pending"
)
EP4_PHASE_SELECTION_POLICY = (
    "atomically gate all claims on the global pending fixed-ISL FFN-EP4 "
    "phase; within each EP prioritize fixed ISL, batch size, and allocated "
    "tray count descending, with rerun_index as the stable tie-break; other "
    "fixed-ISL EPs follow and irregular cases remain last"
)
PHASE_GATED_SELECTION_POLICY = (
    "atomically gate claims on the global pending mode and FFN-EP phase; "
    "prioritize regular cases before irregular cases and FFN-EP 4, 8, 16, "
    "32, then 2; within each regular EP prioritize fixed ISL, batch size, and "
    "allocated tray count descending, with rerun_index as the stable tie-break"
)
SELECTION_POLICY = (
    "atomically claim any pending case matching the worker tray count without "
    "a global mode or FFN-EP phase boundary; within each tray prioritize "
    "regular cases before irregular cases and FFN-EP 4, 8, 16, 32, then 2, "
    "then fixed ISL and batch size descending with rerun_index as the stable "
    "tie-break"
)
SHARED_REGULAR_EP8_32_SELECTION_POLICY_V1 = (
    "atomically claim from one shared regular FFN-EP8/16/32 pool without an "
    "FFN-EP phase boundary; prioritize fixed ISL, batch size, and allocated "
    "tray count descending, then FFN-EP 8, 16, 32 and rerun_index as stable "
    "tie-breaks"
)
SHARED_REGULAR_EP8_32_SELECTION_POLICY = (
    "atomically claim from one shared regular FFN-EP8/16/32 pool without an "
    "FFN-EP phase boundary; exhaust the maximum-batch point for every fixed "
    "ISL, FFN-EP, A:F ratio, and attention-TP topology before non-maximum "
    "batches, then prioritize fixed ISL, batch size, allocated tray count, "
    "FFN-EP 8/16/32, and rerun_index"
)
EP_RANK = {4: 0, 8: 1, 16: 2, 32: 3, 2: 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")
    next_worker = subparsers.add_parser("next-worker")
    next_worker.add_argument("--max-allocated-trays", type=int, default=18)
    claim = subparsers.add_parser("claim")
    claim.add_argument("--allocated-trays", type=int, required=True)
    claim.add_argument("--job-id", required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--case-id", required=True)
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--metric", type=Path, required=True)
    complete.add_argument("--result", type=Path, required=True)

    fail = subparsers.add_parser("fail")
    fail.add_argument("--case-id", required=True)
    fail.add_argument("--job-id", required=True)
    fail.add_argument("--exit-code", type=int, required=True)
    fail.add_argument("--reason", required=True)

    recover = subparsers.add_parser("recover-claim")
    recover.add_argument("--case-id", required=True)
    recover.add_argument("--job-id", required=True)
    recover.add_argument("--reason", required=True)

    release = subparsers.add_parser("release-failed")
    release.add_argument("--case-id", required=True)
    release.add_argument("--reason", required=True)

    recover_completion = subparsers.add_parser("recover-failed-completion")
    recover_completion.add_argument("--case-id", required=True)
    recover_completion.add_argument("--job-id", required=True)
    recover_completion.add_argument("--metric", type=Path, required=True)
    recover_completion.add_argument("--result", type=Path, required=True)
    recover_completion.add_argument("--reason", required=True)

    migrate = subparsers.add_parser("migrate-plan")
    migrate.add_argument("--new-plan", type=Path, required=True)
    migrate.add_argument("--manifest", type=Path, required=True)
    migrate.add_argument("--apply", action="store_true")

    import_terminal = subparsers.add_parser("import-terminal")
    import_terminal.add_argument("--source-state-root", type=Path, required=True)
    import_terminal.add_argument("--case-id", required=True)

    subparsers.add_parser("summary")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_job_id(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise RuntimeError(f"invalid Slurm job ID: {value!r}")
    return value


def validate_case_id(value: str) -> str:
    if CASE_ID_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"invalid case ID: {value!r}")
    return value


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "rerun_index",
        "case_id",
        "allocated_trays",
        "status",
        "required_capture_policy",
        "required_target_batch_per_attention_dp_lane",
        "batch_per_attention_dp_lane",
        "required_trace_decode_steps",
        "required_retained_steps_min",
        "required_max_outliers",
        "required_dominant_range_percent_limit",
        "required_max_median_diff_percent_limit",
        "legacy_isl_mode",
        "isl_tokens",
        "normalized_af_ratio",
        "afd_memory_ratio",
        "afd_num_pages",
        "afd_kv_capacity_tokens",
        "require_capacity_max",
        "megamoe_expert_weight_dtype",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"online plan schema mismatch: {path}")
    validate_plan_rows(rows, str(path))
    return rows


def validate_plan_rows(rows: list[dict[str, str]], source: str) -> None:
    case_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        case_id = validate_case_id(row["case_id"])
        retained_steps = int(row["required_retained_steps_min"])
        max_outliers = int(row["required_max_outliers"])
        if int(row["rerun_index"]) != index or case_id in case_ids:
            raise RuntimeError(
                f"invalid rerun index/duplicate case at row {index} in {source}"
            )
        case_ids.add(case_id)
        if (
            row["status"] != "needs_corrected_target_batch_rerun"
            or row["required_capture_policy"]
            != "trace_first_exact_then_measure_next15"
            or row["required_target_batch_per_attention_dp_lane"]
            != row["batch_per_attention_dp_lane"]
            or int(row["required_trace_decode_steps"]) != 15
            or retained_steps + max_outliers != 15
            or not 0 <= max_outliers < 15
            or float(row["required_dominant_range_percent_limit"]) != 10.0
            or float(row["required_max_median_diff_percent_limit"]) != 10.0
            or not 1 <= int(row["allocated_trays"]) <= 18
            or row["legacy_isl_mode"] not in {"uniform", "irregular"}
            or (row["legacy_isl_mode"] == "uniform") != bool(row["isl_tokens"])
            or int(row["normalized_af_ratio"]) < 1
            or row["megamoe_expert_weight_dtype"] != "fp4"
            or not case_id.endswith(f"-{row['megamoe_expert_weight_dtype']}")
            or not 0 < float(row["afd_memory_ratio"]) <= 1
            or (
                row["afd_num_pages"] != "none"
                and int(row["afd_num_pages"]) < 1
            )
            or int(row["afd_kv_capacity_tokens"]) < 1
            or row["require_capacity_max"] not in {"0", "1"}
        ):
            raise RuntimeError(f"invalid corrected contract for {case_id} in {source}")


def case_execution_identity(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in CASE_EXECUTION_FIELDS)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_plan(path: Path, rows: list[dict[str, str]]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        stream.write(output.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def pool_selection_policy(rows: list[dict[str, str]]) -> str:
    selection_mode = os.environ.get("FASTAFD_POOL_SELECTION_MODE", "auto")
    if selection_mode not in {"auto", "isl-desc"}:
        raise RuntimeError(
            "FASTAFD_POOL_SELECTION_MODE must be auto or isl-desc, got "
            f"{selection_mode!r}"
        )
    if selection_mode == "isl-desc":
        return SELECTION_POLICY
    if all(
        row["legacy_isl_mode"] == "uniform"
        and int(row["ffn_ep"]) in {8, 16, 32}
        for row in rows
    ):
        return SHARED_REGULAR_EP8_32_SELECTION_POLICY
    return SELECTION_POLICY


def initialize(plan: Path, state_root: Path, rows: list[dict[str, str]]) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    for kind in (*STATE_KINDS, "attempts"):
        (state_root / kind).mkdir(exist_ok=True)
    metadata_path = state_root / "pool.json"
    selection_policy = pool_selection_policy(rows)
    expected = {
        "plan_path": str(plan),
        "plan_sha256": sha256(plan),
        "case_count": len(rows),
        "metric_version": METRIC_VERSION,
        "selection_policy": selection_policy,
    }
    if metadata_path.exists():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_expected = [
            {**expected, "selection_policy": policy}
            for policy in (
                LEGACY_SELECTION_POLICY,
                ISL_DESC_SELECTION_POLICY,
                RATIO_BATCH_SELECTION_POLICY,
                EP4_PHASE_SELECTION_POLICY,
                PHASE_GATED_SELECTION_POLICY,
                SELECTION_POLICY,
                SHARED_REGULAR_EP8_32_SELECTION_POLICY_V1,
                SHARED_REGULAR_EP8_32_SELECTION_POLICY,
            )
            if policy != selection_policy
        ]
        if observed in previous_expected:
            with locked(state_root):
                current = json.loads(metadata_path.read_text(encoding="utf-8"))
                if current in previous_expected:
                    atomic_write_json(metadata_path, expected)
                elif current != expected:
                    raise RuntimeError(
                        f"online pool plan drift during policy migration: {current}"
                    )
        elif observed != expected:
            raise RuntimeError(
                f"online pool plan drift: observed={observed} expected={expected}"
            )
    else:
        atomic_write_json(metadata_path, expected)


def locked(state_root: Path):
    lock_path = state_root / "pool.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def state_path(state_root: Path, kind: str, case_id: str) -> Path:
    if kind not in STATE_KINDS:
        raise RuntimeError(f"invalid state kind: {kind}")
    return state_root / kind / f"{validate_case_id(case_id)}.json"


def load_claim(state_root: Path, case_id: str, job_id: str) -> tuple[Path, dict[str, object]]:
    path = state_path(state_root, "claims", case_id)
    if not path.is_file():
        raise RuntimeError(f"case is not actively claimed: {case_id}")
    claim = json.loads(path.read_text(encoding="utf-8"))
    if claim.get("case_id") != case_id or claim.get("job_id") != job_id:
        raise RuntimeError(
            f"claim ownership mismatch for {case_id}: {claim.get('job_id')} != {job_id}"
        )
    return path, claim


def archive_record(state_root: Path, case_id: str, source: Path, label: str) -> Path:
    case_dir = state_root / "attempts" / validate_case_id(case_id)
    case_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = case_dir / f"{timestamp}-{label}.json"
    if destination.exists():
        raise RuntimeError(f"attempt archive collision: {destination}")
    source.rename(destination)
    return destination


def fmha_profile_topology_valid(
    profiled_roles: object,
    result: dict[str, object],
    trace_steps: list[int],
) -> bool:
    if not isinstance(profiled_roles, dict) or len(profiled_roles) != 2:
        return False
    profiles_by_role = {
        str(profile.get("role")): profile
        for profile in profiled_roles.values()
        if isinstance(profile, dict)
    }
    if set(profiles_by_role) != {"attention", "model"}:
        return False
    placement = result.get("placement")
    if not isinstance(placement, dict):
        return False
    model_dp_nodes = placement.get("mlp_dp_rank_nodes")
    if not isinstance(model_dp_nodes, list) or not model_dp_nodes:
        return False
    attention_dp_size = int(result.get("attention_dp_size", 0))
    model_dp_size = len(model_dp_nodes)
    if attention_dp_size < 1 or attention_dp_size % model_dp_size:
        return False
    # Fan-in changes the rows serviced by the grouped model graph, never the
    # number of graph replays. Every attention and model DP launches exactly
    # one graph for each logical decode step.
    expected_launches = {"attention": 1, "model": 1}
    expected_step_keys = set(map(str, trace_steps))
    for role, launch_count in expected_launches.items():
        profile = profiles_by_role[role]
        if int(profile.get("graph_launches_per_trace_step", 0)) != launch_count:
            return False
        raw_by_step = profile.get("per_trace_step_cuda_graph_launch_ms")
        collapsed_by_step = profile.get("per_trace_step_cuda_graph_ms")
        if not isinstance(raw_by_step, dict) or not isinstance(collapsed_by_step, dict):
            return False
        if set(raw_by_step) != expected_step_keys or set(collapsed_by_step) != expected_step_keys:
            return False
        for step in expected_step_keys:
            raw = raw_by_step[step]
            if not isinstance(raw, list) or len(raw) != launch_count:
                return False
            durations = [float(value) for value in raw]
            collapsed = float(collapsed_by_step[step])
            if not all(math.isfinite(value) and value > 0 for value in durations):
                return False
            if not math.isfinite(collapsed) or not math.isclose(
                collapsed, sum(durations), rel_tol=1e-9, abs_tol=1e-9
            ):
                return False
    return True


def validate_metric(
    metric_path: Path,
    result_path: Path,
    case: dict[str, str],
) -> dict[str, object]:
    case_id = case["case_id"]
    if not metric_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"missing metric/result proof for {case_id}")
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    min_retained = int(case["required_retained_steps_min"])
    max_outliers = int(case["required_max_outliers"])
    range_limit = float(case["required_dominant_range_percent_limit"])
    median_limit = float(case["required_max_median_diff_percent_limit"])
    required_steps = int(case["required_trace_decode_steps"])
    required_batch = int(case["required_target_batch_per_attention_dp_lane"])
    placement = result.get("afd_model_placement")
    if placement == "fmha-only":
        decode_steps = [int(step) for step in metric.get("decode_step_ids", [])]
        warmup_step = int(metric.get("warmup_decode_step_id", -1))
        trace_steps = [
            int(step) for step in metric.get("trace_decode_step_ids", [])
        ]
        profiled_roles = metric.get("profiled_roles")
        placement_contract_valid = (
            metric.get("metric_version") == FMHA_METRIC_VERSION
            and int(metric.get("target_batch_per_attention_dp_lane", 0))
            == required_batch
            and int(metric.get("eligible_target_batch_sample_count", 0))
            == required_steps
            and len(decode_steps) == required_steps
            and decode_steps
            == list(range(decode_steps[0], decode_steps[0] + required_steps))
            and warmup_step == decode_steps[0] - 1
            and trace_steps == [warmup_step, *decode_steps]
            and fmha_profile_topology_valid(profiled_roles, result, trace_steps)
            and 1 <= int(metric.get("num_microbatches", 0)) <= required_batch
        )
    elif placement == "legacy":
        placement_contract_valid = (
            metric.get("metric_version") == METRIC_VERSION
            and metric.get("target_batch_filter_passed") is True
            and metric.get("max_median_stability_check_passed") is True
        )
    else:
        placement_contract_valid = False
    if (
        metric.get("case_id") != case_id
        or not placement_contract_valid
        or metric.get("result_sha256") != sha256(result_path)
        or int(metric.get("sample_count", 0)) < min_retained
        or int(metric.get("outlier_count", 99)) > max_outliers
        or float(metric.get("dominant_range_percent", 999)) > range_limit
        or float(metric.get("max_median_diff_percent", 999)) > median_limit
    ):
        raise RuntimeError(f"strict corrected metric validation failed: {metric_path}")
    return metric


def command_claim(
    rows: list[dict[str, str]], state_root: Path, trays: int, job_id: str
) -> None:
    if not 1 <= trays <= 18:
        raise RuntimeError(f"allocated trays must be 1 through 18, got {trays}")
    validate_job_id(job_id)
    with locked(state_root):
        occupied_ids = set().union(
            *(
                {path.stem for path in (state_root / kind).glob("*.json")}
                for kind in STATE_KINDS
            )
        )
        shared_regular_pool = (
            pool_selection_policy(rows) == SHARED_REGULAR_EP8_32_SELECTION_POLICY
        )
        maximum_batch_ids = (
            maximum_batch_case_ids(rows) if shared_regular_pool else set()
        )
        for row in sorted(
            rows,
            key=lambda value: claim_priority(
                value, shared_regular_pool, maximum_batch_ids
            ),
        ):
            if int(row["allocated_trays"]) != trays:
                continue
            case_id = row["case_id"]
            if case_id in occupied_ids:
                continue
            claim = {
                "case_id": case_id,
                "job_id": job_id,
                "allocated_trays": trays,
                "rerun_index": int(row["rerun_index"]),
                "claimed_at_utc": utc_now(),
                "case": row,
            }
            atomic_write_json(state_path(state_root, "claims", case_id), claim)
            json.dump(claim, sys.stdout, sort_keys=True)
            sys.stdout.write("\n")
            return
    # Empty stdout means that this tray-sized worker has no pending case.


def command_next_worker(
    rows: list[dict[str, str]], state_root: Path, max_allocated_trays: int
) -> None:
    if not 1 <= max_allocated_trays <= 18:
        raise RuntimeError(
            "maximum allocated trays must be 1 through 18, "
            f"got {max_allocated_trays}"
        )
    with locked(state_root):
        occupied_ids = set().union(
            *(
                {path.stem for path in (state_root / kind).glob("*.json")}
                for kind in STATE_KINDS
            )
        )
        shared_regular_pool = (
            pool_selection_policy(rows) == SHARED_REGULAR_EP8_32_SELECTION_POLICY
        )
        maximum_batch_ids = (
            maximum_batch_case_ids(rows) if shared_regular_pool else set()
        )
        for row in sorted(
            rows,
            key=lambda value: claim_priority(
                value, shared_regular_pool, maximum_batch_ids
            ),
        ):
            if int(row["allocated_trays"]) > max_allocated_trays:
                continue
            case_id = row["case_id"]
            if case_id in occupied_ids:
                continue
            json.dump(
                {
                    "case_id": case_id,
                    "allocated_trays": int(row["allocated_trays"]),
                    "isl_mode": row["legacy_isl_mode"],
                    "isl_tokens": int(row["isl_tokens"]) if row["isl_tokens"] else None,
                    "normalized_af_ratio": int(row["normalized_af_ratio"]),
                    "batch_per_attention_dp_lane": int(
                        row["batch_per_attention_dp_lane"]
                    ),
                },
                sys.stdout,
                sort_keys=True,
            )
            sys.stdout.write("\n")
            return
    # Empty stdout means that the sweep has no unclaimed pending case.


def case_phase(row: dict[str, str]) -> tuple[int, int]:
    return (
        1 if row["legacy_isl_mode"] == "irregular" else 0,
        EP_RANK[int(row["ffn_ep"])],
    )


def maximum_batch_case_ids(rows: list[dict[str, str]]) -> set[str]:
    maximum_by_topology: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        topology = (
            row["legacy_isl_mode"],
            row["isl_tokens"],
            row["ffn_ep"],
            row["normalized_af_ratio"],
            row["attention_tp"],
        )
        current = maximum_by_topology.get(topology)
        if current is None or int(row["batch_per_attention_dp_lane"]) > int(
            current["batch_per_attention_dp_lane"]
        ):
            maximum_by_topology[topology] = row
    return {row["case_id"] for row in maximum_by_topology.values()}


def claim_priority(
    row: dict[str, str],
    shared_regular_pool: bool = False,
    maximum_batch_ids: set[str] | None = None,
) -> tuple[int, int, int, int, int, int]:
    rerun_index = int(row["rerun_index"])
    if shared_regular_pool:
        return (
            0 if row["case_id"] in (maximum_batch_ids or set()) else 1,
            -int(row["isl_tokens"]) if row["isl_tokens"] else 0,
            -int(row["batch_per_attention_dp_lane"]),
            -int(row["allocated_trays"]),
            EP_RANK[int(row["ffn_ep"])],
            rerun_index,
        )
    if row["legacy_isl_mode"] == "irregular":
        return (*case_phase(row), 0, 0, 0, rerun_index)
    return (
        *case_phase(row),
        -int(row["isl_tokens"]),
        -int(row["batch_per_attention_dp_lane"]),
        -int(row["allocated_trays"]),
        rerun_index,
    )


def command_complete(
    rows: list[dict[str, str]],
    state_root: Path,
    case_id: str,
    job_id: str,
    metric_path: Path,
    result_path: Path,
) -> None:
    validate_job_id(job_id)
    metric_path = metric_path.resolve()
    result_path = result_path.resolve()
    cases = {row["case_id"]: row for row in rows}
    if case_id not in cases:
        raise RuntimeError(f"completed case is absent from plan: {case_id}")
    metric = validate_metric(metric_path, result_path, cases[case_id])
    with locked(state_root):
        claim_path, claim = load_claim(state_root, case_id, job_id)
        completed_path = state_path(state_root, "completed", case_id)
        if completed_path.exists() or state_path(state_root, "failed", case_id).exists():
            raise RuntimeError(f"case already has terminal pool state: {case_id}")
        record = {
            **claim,
            "completed_at_utc": utc_now(),
            "metric_path": str(metric_path),
            "metric_sha256": sha256(metric_path),
            "result_path": str(result_path),
            "result_sha256": metric["result_sha256"],
        }
        atomic_write_json(completed_path, record)
        claim_path.unlink()
    print(f"completed case={case_id} job={job_id}")


def command_fail(
    state_root: Path, case_id: str, job_id: str, exit_code: int, reason: str
) -> None:
    validate_job_id(job_id)
    if exit_code == 0 or not reason.strip():
        raise RuntimeError("failed case requires a nonzero exit code and reason")
    with locked(state_root):
        claim_path, claim = load_claim(state_root, case_id, job_id)
        failed_path = state_path(state_root, "failed", case_id)
        if failed_path.exists() or state_path(state_root, "completed", case_id).exists():
            raise RuntimeError(f"case already has terminal pool state: {case_id}")
        record = {
            **claim,
            "failed_at_utc": utc_now(),
            "exit_code": exit_code,
            "reason": reason.strip(),
            "retry_policy": "blocked until explicit release-failed after root-cause fix",
        }
        atomic_write_json(failed_path, record)
        claim_path.unlink()
    print(f"failed case={case_id} job={job_id} exit_code={exit_code}")


def command_recover_claim(
    state_root: Path, case_id: str, job_id: str, reason: str
) -> None:
    validate_job_id(job_id)
    if not reason.strip():
        raise RuntimeError("claim recovery requires an audit reason")
    with locked(state_root):
        claim_path, claim = load_claim(state_root, case_id, job_id)
        claim["recovered_at_utc"] = utc_now()
        claim["recovery_reason"] = reason.strip()
        temporary = claim_path.with_name(f".{claim_path.name}.recover.{os.getpid()}")
        atomic_write_json(temporary, claim)
        archived = archive_record(state_root, case_id, temporary, f"job-{job_id}-claim")
        claim_path.unlink()
    print(f"recovered case={case_id} job={job_id} archive={archived}")


def command_release_failed(state_root: Path, case_id: str, reason: str) -> None:
    if not reason.strip():
        raise RuntimeError("failed-case release requires a root-cause/fix reason")
    with locked(state_root):
        failed_path = state_path(state_root, "failed", case_id)
        if not failed_path.is_file():
            raise RuntimeError(f"case has no failed state: {case_id}")
        failed = json.loads(failed_path.read_text(encoding="utf-8"))
        failed["released_at_utc"] = utc_now()
        failed["release_reason"] = reason.strip()
        temporary = failed_path.with_name(f".{failed_path.name}.release.{os.getpid()}")
        atomic_write_json(temporary, failed)
        archived = archive_record(state_root, case_id, temporary, "failed-release")
        failed_path.unlink()
    print(f"released case={case_id} archive={archived}")


def command_recover_failed_completion(
    rows: list[dict[str, str]],
    state_root: Path,
    case_id: str,
    job_id: str,
    metric_path: Path,
    result_path: Path,
    reason: str,
) -> None:
    """Promote durable, strictly valid output after bookkeeping-only failure."""
    validate_job_id(job_id)
    if not reason.strip():
        raise RuntimeError("failed-completion recovery requires an audit reason")
    metric_path = metric_path.resolve()
    result_path = result_path.resolve()
    cases = {row["case_id"]: row for row in rows}
    if case_id not in cases:
        raise RuntimeError(f"recovered case is absent from plan: {case_id}")
    metric = validate_metric(metric_path, result_path, cases[case_id])
    with locked(state_root):
        failed_path = state_path(state_root, "failed", case_id)
        if not failed_path.is_file():
            raise RuntimeError(f"case has no failed state: {case_id}")
        if state_path(state_root, "claims", case_id).exists() or state_path(
            state_root, "completed", case_id
        ).exists():
            raise RuntimeError(f"case has conflicting pool state: {case_id}")
        failed = json.loads(failed_path.read_text(encoding="utf-8"))
        if failed.get("case_id") != case_id or failed.get("job_id") != job_id:
            raise RuntimeError(
                f"failed ownership mismatch for {case_id}: "
                f"{failed.get('job_id')} != {job_id}"
            )
        recovered_at = utc_now()
        failed["completion_recovered_at_utc"] = recovered_at
        failed["completion_recovery_reason"] = reason.strip()
        atomic_write_json(failed_path, failed)
        archived = archive_record(
            state_root, case_id, failed_path, "failed-completion-recovery"
        )
        record = {
            key: value
            for key, value in failed.items()
            if key
            not in {
                "failed_at_utc",
                "exit_code",
                "reason",
                "retry_policy",
            }
        }
        record.update(
            {
                "completed_at_utc": recovered_at,
                "metric_path": str(metric_path),
                "metric_sha256": sha256(metric_path),
                "result_path": str(result_path),
                "result_sha256": metric["result_sha256"],
                "recovered_from_failed_archive": str(archived),
            }
        )
        try:
            atomic_write_json(state_path(state_root, "completed", case_id), record)
        except BaseException:
            archived.rename(failed_path)
            raise
    print(
        f"recovered completion case={case_id} job={job_id} archive={archived}"
    )


def command_migrate_plan(
    plan: Path,
    state_root: Path,
    old_rows: list[dict[str, str]],
    desired_rows: list[dict[str, str]],
    manifest_path: Path,
    apply: bool,
) -> None:
    old_by_id = {row["case_id"]: row for row in old_rows}
    desired_by_id = {row["case_id"]: row for row in desired_rows}
    with locked(state_root):
        metadata_path = state_root / "pool.json"
        observed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        allowed_policies = (
            LEGACY_SELECTION_POLICY,
            ISL_DESC_SELECTION_POLICY,
            RATIO_BATCH_SELECTION_POLICY,
            EP4_PHASE_SELECTION_POLICY,
            SELECTION_POLICY,
            SHARED_REGULAR_EP8_32_SELECTION_POLICY_V1,
            SHARED_REGULAR_EP8_32_SELECTION_POLICY,
        )
        allowed_metadata = [
            {
                "plan_path": str(plan),
                "plan_sha256": sha256(plan),
                "case_count": len(old_rows),
                "metric_version": METRIC_VERSION,
                "selection_policy": policy,
            }
            for policy in allowed_policies
        ]
        if observed_metadata not in allowed_metadata:
            raise RuntimeError(
                "online pool metadata/current-plan mismatch before migration: "
                f"{observed_metadata}"
            )

        state_records: list[tuple[Path, dict[str, object]]] = []
        occupied_ids: set[str] = set()
        for kind in STATE_KINDS:
            for record_path in sorted((state_root / kind).glob("*.json")):
                case_id = record_path.stem
                if case_id in occupied_ids or case_id not in old_by_id:
                    raise RuntimeError(f"invalid occupied case during migration: {case_id}")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                old_row = old_by_id[case_id]
                if (
                    record.get("case_id") != case_id
                    or not isinstance(record.get("case"), dict)
                    or case_execution_identity(record["case"])
                    != case_execution_identity(old_row)
                    or int(record.get("allocated_trays", 0))
                    != int(old_row["allocated_trays"])
                ):
                    raise RuntimeError(f"occupied record/current-plan drift: {record_path}")
                occupied_ids.add(case_id)
                state_records.append((record_path, record))

        preserved_terminal_exceptions = len(occupied_ids - desired_by_id.keys())
        merged_rows = [dict(row) for row in desired_rows]
        for case_id in sorted(
            occupied_ids - desired_by_id.keys(),
            key=lambda value: int(old_by_id[value]["rerun_index"]),
        ):
            merged_rows.append(dict(old_by_id[case_id]))
        for index, output in enumerate(merged_rows, start=1):
            output["rerun_index"] = str(index)
        validate_plan_rows(merged_rows, "merged migration plan")
        merged_by_id = {row["case_id"]: row for row in merged_rows}
        for case_id in occupied_ids:
            if case_execution_identity(merged_by_id[case_id]) != case_execution_identity(
                old_by_id[case_id]
            ):
                raise RuntimeError(f"occupied execution identity changed for {case_id}")

        summary = {
            "apply": apply,
            "old_cases": len(old_rows),
            "desired_cases": len(desired_rows),
            "merged_cases": len(merged_rows),
            "occupied_cases": len(occupied_ids),
            "preserved_occupied_exceptions": preserved_terminal_exceptions,
            "removed_unclaimed_cases": len(old_by_id.keys() - merged_by_id.keys()),
            "added_cases": len(merged_by_id.keys() - old_by_id.keys()),
        }
        if not apply:
            print(json.dumps(summary, sort_keys=True))
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_sha256") != sha256(plan):
            raise RuntimeError("overlay manifest/current-plan mismatch before migration")
        atomic_write_plan(plan, merged_rows)
        new_sha256 = sha256(plan)
        atomic_write_json(
            metadata_path,
            {
                "plan_path": str(plan),
                "plan_sha256": new_sha256,
                "case_count": len(merged_rows),
                "metric_version": METRIC_VERSION,
                "selection_policy": pool_selection_policy(merged_rows),
            },
        )
        manifest["plan_path"] = str(plan)
        manifest["plan_sha256"] = new_sha256
        atomic_write_json(manifest_path, manifest)
        summary["plan_sha256"] = new_sha256
    print(json.dumps(summary, sort_keys=True))


def command_import_terminal(
    plan: Path,
    state_root: Path,
    rows: list[dict[str, str]],
    source_state_root: Path,
    case_id: str,
) -> None:
    expected_case = {row["case_id"]: row for row in rows}.get(case_id)
    if expected_case is None:
        raise RuntimeError(f"imported case is absent from destination plan: {case_id}")
    sources = [
        (kind, state_path(source_state_root, kind, case_id))
        for kind in ("completed", "failed")
    ]
    sources = [(kind, path) for kind, path in sources if path.is_file()]
    if len(sources) != 1:
        raise RuntimeError(
            f"import requires exactly one terminal source for {case_id}: {sources}"
        )
    kind, source_path = sources[0]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_case = source.get("case")
    if (
        source.get("case_id") != case_id
        or not isinstance(source_case, dict)
        or case_execution_identity(source_case)
        != case_execution_identity(expected_case)
        or int(source.get("allocated_trays", 0))
        != int(expected_case["allocated_trays"])
    ):
        raise RuntimeError(f"terminal import identity drift: {source_path}")
    if kind == "completed":
        metric_path = Path(str(source["metric_path"]))
        result_path = Path(str(source["result_path"]))
        metric = validate_metric(metric_path, result_path, expected_case)
        if (
            source.get("metric_sha256") != sha256(metric_path)
            or source.get("result_sha256") != metric["result_sha256"]
        ):
            raise RuntimeError(f"terminal import proof hash drift: {source_path}")
    elif int(source.get("exit_code", 0)) == 0 or not str(source.get("reason", "")):
        raise RuntimeError(f"invalid failed terminal import: {source_path}")

    with locked(state_root):
        destination = state_path(state_root, kind, case_id)
        occupied = [
            state_path(state_root, state_kind, case_id)
            for state_kind in STATE_KINDS
            if state_path(state_root, state_kind, case_id).exists()
        ]
        if occupied:
            if len(occupied) == 1:
                current = json.loads(occupied[0].read_text(encoding="utf-8"))
                if (
                    occupied[0] == destination
                    and current.get("imported_from") == str(source_path)
                    and isinstance(current.get("case"), dict)
                    and case_execution_identity(current["case"])
                    == case_execution_identity(expected_case)
                ):
                    print(f"terminal already imported case={case_id} kind={kind}")
                    return
            raise RuntimeError(f"destination case is already occupied: {occupied}")
        imported = {
            **source,
            "case": expected_case,
            "rerun_index": int(expected_case["rerun_index"]),
            "imported_at_utc": utc_now(),
            "imported_from": str(source_path),
        }
        atomic_write_json(destination, imported)
    print(f"imported terminal case={case_id} kind={kind}")


def command_summary(rows: list[dict[str, str]], state_root: Path) -> None:
    with locked(state_root):
        state_ids = {
            kind: {path.stem for path in (state_root / kind).glob("*.json")}
            for kind in STATE_KINDS
        }
        counts = {kind: len(case_ids) for kind, case_ids in state_ids.items()}
        occupied = sum(counts.values())
        occupied_ids = set().union(*state_ids.values())
        if len(occupied_ids) != occupied:
            raise RuntimeError("case has duplicate online pool states")
        summary = {
            "case_count": len(rows),
            **counts,
            "pending": len(rows) - occupied,
            "counts_by_pending_trays": {},
        }
        pending_by_trays: dict[str, int] = {}
        for row in rows:
            if row["case_id"] in occupied_ids:
                continue
            trays = row["allocated_trays"]
            pending_by_trays[trays] = pending_by_trays.get(trays, 0) + 1
        summary["counts_by_pending_trays"] = dict(
            sorted(pending_by_trays.items(), key=lambda item: int(item[0]))
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    plan = args.plan.resolve()
    state_root = args.state_root.resolve()
    rows = read_plan(plan)
    if args.command == "migrate-plan":
        command_migrate_plan(
            plan,
            state_root,
            rows,
            read_plan(args.new_plan.resolve()),
            args.manifest.resolve(),
            args.apply,
        )
        return
    initialize(plan, state_root, rows)
    if args.command == "init":
        print(f"initialized cases={len(rows)} state_root={state_root}")
    elif args.command == "next-worker":
        command_next_worker(rows, state_root, args.max_allocated_trays)
    elif args.command == "claim":
        command_claim(rows, state_root, args.allocated_trays, args.job_id)
    elif args.command == "complete":
        command_complete(
            rows,
            state_root,
            validate_case_id(args.case_id),
            args.job_id,
            args.metric,
            args.result,
        )
    elif args.command == "fail":
        command_fail(
            state_root,
            validate_case_id(args.case_id),
            args.job_id,
            args.exit_code,
            args.reason,
        )
    elif args.command == "recover-claim":
        command_recover_claim(
            state_root,
            validate_case_id(args.case_id),
            args.job_id,
            args.reason,
        )
    elif args.command == "release-failed":
        command_release_failed(
            state_root, validate_case_id(args.case_id), args.reason
        )
    elif args.command == "recover-failed-completion":
        command_recover_failed_completion(
            rows,
            state_root,
            validate_case_id(args.case_id),
            args.job_id,
            args.metric,
            args.result,
            args.reason,
        )
    elif args.command == "import-terminal":
        command_import_terminal(
            plan,
            state_root,
            rows,
            args.source_state_root.resolve(),
            validate_case_id(args.case_id),
        )
    elif args.command == "summary":
        command_summary(rows, state_root)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
