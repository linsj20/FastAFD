#!/usr/bin/env bash
# Qwen3 fixed-global-work MegaMoE Nsight timing and six-point IKET trace.
# Run from an OCI-HSG login node; with no allocation this script submits itself.
# Optional overrides: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV
#                     FASTAFD_IKET_INCLUDE FASTAFD_IKET_PYTHON_DEPS
#                     FASTAFD_IKET_ARTIFACT_ROOT FASTAFD_IKET_DEEPGEMM_CACHE_ROOT
#                     DG_JIT_CACHE_DIR
#                     MEGAMOE_IKET_EP MEGAMOE_IKET_GLOBAL_TOKENS
#                     MEGAMOE_IKET_DISPATCH_DETAIL (diagnostic source required)
#                     MEGAMOE_BASELINE_{WARMUPS,ITERATIONS}
#                     MEGAMOE_IKET_{WARMUPS,ITERATIONS,PROFILE_RANK}
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --segment=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=fastafd-megamoe-iket

set -euo pipefail

FASTAFD_USER_ROOT=${FASTAFD_USER_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel}
FASTAFD_SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$FASTAFD_USER_ROOT/FastAFD/local-mirror}
FASTAFD_IMAGE=${FASTAFD_IMAGE:-$FASTAFD_USER_ROOT/trtllm_qwen3_235b_megamoe/images/pytorch-26.05-py3-arm64-trtllm-fa778397186c09dd57bcaa5020e66270768b675b.sqsh}
FASTAFD_VENV=${FASTAFD_VENV:-$FASTAFD_USER_ROOT/fastafd_reproduce/envs/minisgl-3c716194-cuda130.artifact-5175891}
FASTAFD_IKET_ARTIFACT_ROOT=${FASTAFD_IKET_ARTIFACT_ROOT:-$FASTAFD_USER_ROOT/FastAFD/artifacts/megamoe_iket_qwen3}
FASTAFD_IKET_DEEPGEMM_CACHE_ROOT=${FASTAFD_IKET_DEEPGEMM_CACHE_ROOT:-}
FASTAFD_IKET_INCLUDE=${FASTAFD_IKET_INCLUDE:-$FASTAFD_USER_ROOT/FastAFD/source/iket/include}
FASTAFD_IKET_PYTHON_DEPS=${FASTAFD_IKET_PYTHON_DEPS:-}
MEGAMOE_IKET_DISPATCH_DETAIL=${MEGAMOE_IKET_DISPATCH_DETAIL:-0}
[[ $MEGAMOE_IKET_DISPATCH_DETAIL == 0 || $MEGAMOE_IKET_DISPATCH_DETAIL == 1 ]] || {
    echo "MEGAMOE_IKET_DISPATCH_DETAIL must be 0 or 1" >&2
    exit 2
}
MEGAMOE_IKET_EP=${MEGAMOE_IKET_EP:-8}
MEGAMOE_IKET_GLOBAL_TOKENS=${MEGAMOE_IKET_GLOBAL_TOKENS:-192}
case "$MEGAMOE_IKET_EP" in
    2|4|8|16|32|64) ;;
    *) echo "MEGAMOE_IKET_EP must be 2, 4, 8, 16, 32, or 64" >&2; exit 2 ;;
esac
[[ $MEGAMOE_IKET_GLOBAL_TOKENS =~ ^[1-9][0-9]*$ ]] || {
    echo "global tokens must be positive" >&2
    exit 2
}
MEGAMOE_IKET_ROWS=$(((MEGAMOE_IKET_GLOBAL_TOKENS + MEGAMOE_IKET_EP - 1) / MEGAMOE_IKET_EP))
MEGAMOE_IKET_NODES=$(((MEGAMOE_IKET_EP + 3) / 4))
MEGAMOE_IKET_TASKS_PER_NODE=$((MEGAMOE_IKET_EP / MEGAMOE_IKET_NODES))
MEGAMOE_BASELINE_WARMUPS=${MEGAMOE_BASELINE_WARMUPS:-1}
MEGAMOE_BASELINE_ITERATIONS=${MEGAMOE_BASELINE_ITERATIONS:-3}
MEGAMOE_IKET_WARMUPS=${MEGAMOE_IKET_WARMUPS:-1}
MEGAMOE_IKET_ITERATIONS=${MEGAMOE_IKET_ITERATIONS:-3}
MEGAMOE_IKET_PROFILE_RANK=${MEGAMOE_IKET_PROFILE_RANK:-0}
export MEGAMOE_IKET_EP MEGAMOE_IKET_GLOBAL_TOKENS MEGAMOE_IKET_DISPATCH_DETAIL
export FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV
export FASTAFD_IKET_ARTIFACT_ROOT MEGAMOE_BASELINE_WARMUPS
export FASTAFD_IKET_DEEPGEMM_CACHE_ROOT
export FASTAFD_IKET_INCLUDE FASTAFD_IKET_PYTHON_DEPS
export MEGAMOE_BASELINE_ITERATIONS MEGAMOE_IKET_WARMUPS
export MEGAMOE_IKET_ITERATIONS MEGAMOE_IKET_PROFILE_RANK

if [[ ${FASTAFD_IKET_RANK_MODE:-} == normal || \
      ${FASTAFD_IKET_RANK_MODE:-} == profile ]]; then
    rank=${SLURM_PROCID:?}
    rank_root=$FASTAFD_IKET_RUN_ROOT/ranks/rank-$rank
    mkdir -p "$rank_root" "/dev/shm/fastafd-megamoe-iket-$SLURM_JOB_ID-$rank"
    export TMPDIR=/dev/shm/fastafd-megamoe-iket-$SLURM_JOB_ID-$rank
    if [[ $rank == "$MEGAMOE_IKET_PROFILE_RANK" ]]; then
        if [[ $FASTAFD_IKET_RANK_MODE == normal ]]; then
            export MEGAMOE_SMOKE_WEIGHT_CACHE_OUTPUT=$FASTAFD_IKET_RUN_ROOT/profile-rank-weights.pt
        else
            export MEGAMOE_SMOKE_WEIGHT_CACHE_INPUT=$FASTAFD_IKET_RUN_ROOT/profile-rank-weights.pt
        fi
    fi
    if [[ $FASTAFD_IKET_RANK_MODE == normal &&
          $rank == "$MEGAMOE_IKET_PROFILE_RANK" ]]; then
        export OUTPUT_ROOT=$rank_root
        export MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX=$rank_root/nsys/megamoe
        export MEGAMOE_SMOKE_NSYS_PROFILE_RANK=$MEGAMOE_IKET_PROFILE_RANK
        export MEGAMOE_SMOKE_RANK_MODE=run
        exec bash "$FASTAFD_SOURCE_REPO/scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch"
    fi
    if [[ $FASTAFD_IKET_RANK_MODE == normal || \
          $rank != "$MEGAMOE_IKET_PROFILE_RANK" ]]; then
        # Source-mode IKET runs discovery and capture in separate processes.
        # Peers must join both worlds, with identical rank IDs and inputs.
        if [[ $FASTAFD_IKET_RANK_MODE == profile ]]; then
            "$FASTAFD_VENV/bin/python" \
                "$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py"
        fi
        exec "$FASTAFD_VENV/bin/python" \
            "$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py"
    fi
    # Add optional SModel dependencies only to the profiler interpreter.
    # Its child workload retains the original Python environment.
    exec /usr/bin/python3 -c '
import os, runpy, shutil, sys
deps = os.environ["FASTAFD_IKET_PYTHON_DEPS"]
if deps:
    sys.path.insert(0, deps)
runpy.run_path(shutil.which("run-iket"), run_name="__main__")
' --output-dir "$rank_root/iket" --clobber \
        --context-buffer-size 268435456 --log-level info \
        profile --max-ts-cnt-per-warp 1024 --postprocess json -- \
        "$FASTAFD_VENV/bin/python" \
        "$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py"
fi

script_path=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
analyze_only=0
if [[ ${1:-} == analyze ]]; then
    analyze_only=1
    run_root=${2:?analysis requires an existing run root}
fi
if (( ! analyze_only )) && [[ -z ${SLURM_JOB_ID:-} ]]; then
    mkdir -p "$FASTAFD_IKET_ARTIFACT_ROOT"
    job_id=$(sbatch --parsable \
        --nodes="$MEGAMOE_IKET_NODES" --segment="$MEGAMOE_IKET_NODES" \
        --ntasks="$MEGAMOE_IKET_EP" \
        --ntasks-per-node="$MEGAMOE_IKET_TASKS_PER_NODE" \
        --output="$FASTAFD_IKET_ARTIFACT_ROOT/slurm-%j.out" \
        --error="$FASTAFD_IKET_ARTIFACT_ROOT/slurm-%j.err" \
        "$script_path")
    printf 'JOB_ID=%s\nLOG=%s/slurm-%s.out\n' \
        "$job_id" "$FASTAFD_IKET_ARTIFACT_ROOT" "$job_id"
    exit 0
fi

if (( ! analyze_only )); then
[[ $SLURM_NNODES == "$MEGAMOE_IKET_NODES" &&
   $SLURM_NTASKS == "$MEGAMOE_IKET_EP" ]] || {
    echo "expected $MEGAMOE_IKET_NODES trays and $MEGAMOE_IKET_EP EP ranks" >&2
    exit 2
}
[[ $MEGAMOE_IKET_PROFILE_RANK =~ ^[0-9]+$ ]] &&
    (( MEGAMOE_IKET_PROFILE_RANK < MEGAMOE_IKET_EP )) || {
    echo "MEGAMOE_IKET_PROFILE_RANK must be within the EP world" >&2
    exit 2
}
[[ -f $FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py ]] || {
    echo "missing FastAFD MegaMoE smoke: $FASTAFD_SOURCE_REPO" >&2
    exit 2
}
[[ -x $FASTAFD_VENV/bin/python ]] || {
    echo "missing FastAFD Python environment: $FASTAFD_VENV" >&2
    exit 2
}
[[ -f $FASTAFD_IMAGE ]] || {
    echo "missing IKET-capable container image: $FASTAFD_IMAGE" >&2
    exit 2
}
[[ -f $FASTAFD_IKET_INCLUDE/iket/iket_device_apis.cuh ]] || {
    echo "missing IKET device API headers: $FASTAFD_IKET_INCLUDE" >&2
    exit 2
}

mapfile -t fabric_blocks < <(
    scontrol show hostnames "$SLURM_JOB_NODELIST" |
        sed -E 's/-T[0-9]+$//' | sort -u
)
(( ${#fabric_blocks[@]} == 1 )) || {
    printf 'allocation crosses NVL72 fabric blocks: %s\n' \
        "${fabric_blocks[*]}" >&2
    exit 2
}

run_root=$FASTAFD_IKET_ARTIFACT_ROOT/$SLURM_JOB_ID
deepgemm_cache=${FASTAFD_IKET_DEEPGEMM_CACHE_ROOT:-$run_root/cache/deepgemm}
mkdir -p "$run_root/ranks" "$deepgemm_cache" "$run_root/cache/tvm_ffi"
export FASTAFD_IKET_RUN_ROOT=$run_root
cp "$script_path" "$run_root/profile_megamoe_iket.sbatch"
git -C "$FASTAFD_SOURCE_REPO" rev-parse HEAD > "$run_root/source-head.txt"
git -C "$FASTAFD_SOURCE_REPO" status --short > "$run_root/source-status.txt"
git -C "$FASTAFD_SOURCE_REPO" diff -- \
    tests/afd_megamoe_same_rank_smoke.py \
    python/minisgl/kernel/csrc/deepgemm/csrc/jit/compiler.hpp \
    python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh \
    python/minisgl/kernel/csrc/deepgemm/csrc/jit_kernels/heuristics/mega_moe.hpp \
    > "$run_root/source.patch"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
master_port=${MASTER_PORT:-29631}

run_world() {
    local mode=$1 warmups=$2 iterations=$3 output=$4
    local iket_include=""
    local compare_combine=0
    [[ $mode != profile ]] || iket_include=$FASTAFD_IKET_INCLUDE
    [[ $mode != normal ]] || compare_combine=${MEGAMOE_SMOKE_COMPARE_COMBINE:-0}
    srun --nodes="$MEGAMOE_IKET_NODES" --ntasks="$MEGAMOE_IKET_EP" --ntasks-per-node="$MEGAMOE_IKET_TASKS_PER_NODE" --kill-on-bad-exit=1 \
        --container-image="$FASTAFD_IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env -u CUDA_VISIBLE_DEVICES \
        FASTAFD_IKET_RANK_MODE="$mode" \
        FASTAFD_IKET_RUN_ROOT="$run_root" \
        PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
        PYTHONPATH="$FASTAFD_SOURCE_REPO/python" \
        MINISGL_DEEPGEMM_BUILD_DIR="$deepgemm_cache" \
        DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$deepgemm_cache/jit}" \
        DG_JIT_IKET_INCLUDE="$iket_include" \
        TVM_FFI_CACHE_DIR="$run_root/cache/tvm_ffi" \
        MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4 \
        MEGAMOE_SMOKE_PROFILE_ONLY=1 \
        MEGAMOE_SMOKE_FIXED_PERFECT_ONLY=1 \
        MEGAMOE_SMOKE_PRODUCTION_ROUTES=1 \
        MEGAMOE_SMOKE_BALANCE_RANKS="${MEGAMOE_SMOKE_BALANCE_RANKS:-0}" \
        MEGAMOE_SMOKE_COMPARE_COMBINE="$compare_combine" \
        MEGAMOE_SMOKE_ROWS="$MEGAMOE_IKET_ROWS" \
        MEGAMOE_SMOKE_GLOBAL_TOKENS="$MEGAMOE_IKET_GLOBAL_TOKENS" \
        MEGAMOE_SMOKE_SMS=150 \
        MEGAMOE_SMOKE_BUCKET=3840 \
        MEGAMOE_SMOKE_WARMUPS="$warmups" \
        MEGAMOE_SMOKE_ITERATIONS="$iterations" \
        MEGAMOE_SMOKE_DEBUG_TIMINGS=0 \
        MEGAMOE_SMOKE_CONTEND_CYCLES=0 \
        MEGAMOE_SMOKE_NUM_EXPERTS=128 \
        MEGAMOE_SMOKE_HIDDEN=4096 \
        MEGAMOE_SMOKE_INTERMEDIATE=1536 \
        MEGAMOE_SMOKE_TOP_K=8 \
        MEGAMOE_SMOKE_OUTPUT="$output" \
        MASTER_ADDR="$master_addr" MASTER_PORT="$master_port" \
        bash "$run_root/profile_megamoe_iket.sbatch"
}

printf 'FASTAFD_MEGAMOE_IKET_CONFIG=job=%s nodes=%s source=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$FASTAFD_SOURCE_REPO"
run_world normal "$MEGAMOE_BASELINE_WARMUPS" \
    "$MEGAMOE_BASELINE_ITERATIONS" "$run_root/baseline.json"
run_world profile "$MEGAMOE_IKET_WARMUPS" \
    "$MEGAMOE_IKET_ITERATIONS" "$run_root/profiled.json"
fi

python3 - "$run_root" "$MEGAMOE_IKET_ITERATIONS" \
    "$MEGAMOE_IKET_PROFILE_RANK" "$MEGAMOE_IKET_EP" \
    "$MEGAMOE_IKET_GLOBAL_TOKENS" "$MEGAMOE_IKET_DISPATCH_DETAIL" <<'PY'
from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path


run_root = Path(sys.argv[1])
measured_iterations = int(sys.argv[2])
profile_rank = int(sys.argv[3])
ep_size = int(sys.argv[4])
global_tokens = int(sys.argv[5])
dispatch_detail = bool(int(sys.argv[6]))
rows_per_rank = (global_tokens + ep_size - 1) // ep_size
balance_ranks = bool(int(os.environ.get("MEGAMOE_SMOKE_BALANCE_RANKS", "0")))
case_key = f"bucket3840_rows{rows_per_rank}_sms150"


def load_benchmark(name: str) -> tuple[dict, dict]:
    payload = json.loads((run_root / name).read_text())
    if payload.get("status") != "ok" or payload.get("expert_weight_dtype") != "fp4":
        raise SystemExit(f"invalid {name}: status or precision mismatch")
    record = payload.get("exact_qwen", {}).get(case_key)
    if not record:
        raise SystemExit(f"invalid {name}: missing {case_key}")
    if (record["ep_size"], record["global_tokens"], record["rows_per_rank"]) != (
        ep_size, global_tokens, rows_per_rank
    ):
        raise SystemExit(f"invalid {name}: EP/global-work contract mismatch")
    expected_active = min(128 // ep_size, (global_tokens * 8 + ep_size - 1) // ep_size
                          if balance_ranks else global_tokens * 8)
    if record.get("max_active_experts") != expected_active:
        raise SystemExit(f"invalid {name}: unexpected active local expert count")
    if record.get("timed_scope") != "route_producer_and_megamoe":
        raise SystemExit(f"invalid {name}: expected production route preparation")
    if record.get("routes_prepared") is not True:
        raise SystemExit(f"invalid {name}: routes were not prepared")
    expected_routing = ("fixed_global_sequence_rank_interleaved_balance" if balance_ranks
                        else "fixed_global_sequence_perfect_balance")
    if record.get("routing") != expected_routing:
        raise SystemExit(f"invalid {name}: expected fixed perfect routing")
    return payload, record


def indexed_value(table, index):
    if isinstance(table, list):
        return table[int(index)]
    return table.get(str(index), table.get(index))


def kernel_name(launch: dict, strings) -> str:
    name = launch.get("kernelName", "")
    if isinstance(name, int):
        name = indexed_value(strings, name)
    return str(name or "")


def lifetime_location(lifetime: dict, locations) -> dict:
    index = lifetime.get("locIdx", lifetime.get("locationIdx"))
    if index is None:
        raise SystemExit("IKET warp lifetime has no location index")
    location = indexed_value(locations, index)
    if not isinstance(location, dict):
        raise SystemExit(f"IKET location {index} is missing")
    return location


def find_trace(rank_dir: Path) -> tuple[Path, dict]:
    candidates = []
    for path in rank_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "launches" in payload and "locationTable" in payload:
            candidates.append((path, payload))
    if len(candidates) != 1:
        names = [str(path) for path, _ in candidates]
        raise SystemExit(f"expected one IKET trace under {rank_dir}, got {names}")
    return candidates[0]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


roles = {
    "dispatch": set(range(0, 4)),
    "tma_a": {4},
    "tma_b": {5},
    "mma": {6},
    "scheduler": {7},
    "epilogue_combine": set(range(8, 16)),
}
samples = []
trace_files = {}
rank_dirs = [run_root / "ranks" / f"rank-{profile_rank}"]
if not rank_dirs[0].is_dir():
    raise SystemExit(f"missing profiled rank directory: {rank_dirs[0]}")

for rank_dir in rank_dirs:
    match = re.fullmatch(r"rank-(\d+)", rank_dir.name)
    if match is None:
        continue
    rank = int(match.group(1))
    trace_path, trace = find_trace(rank_dir / "iket")
    trace_files[str(rank)] = str(trace_path.relative_to(run_root))
    launch_records = [("direct", launch) for launch in trace["launches"]]
    graph_launches = trace.get("graphLaunches", {})
    if isinstance(graph_launches, dict):
        def graph_order(item):
            match = re.search(r":(\d+)$", item[0])
            return int(match.group(1)) if match else -1

        for graph_key, graph_group in sorted(
            graph_launches.items(), key=graph_order
        ):
            if isinstance(graph_group, dict):
                graph_group = graph_group.get("launches", [])
            if isinstance(graph_group, list):
                launch_records.extend(
                    (f"graph:{graph_key}", launch) for launch in graph_group
                )
    elif isinstance(graph_launches, list):
        launch_records.extend(("graph", launch) for launch in graph_launches)

    launches = []
    for source, launch in launch_records:
        name = kernel_name(launch, trace.get("stringTable", []))
        lowered = name.lower()
        if "mega_moe" in lowered and "fp4" in lowered:
            launches.append((source, name, launch))
    if len(launches) < measured_iterations:
        raise SystemExit(
            f"rank {rank}: found {len(launches)} FP4 MegaMoE launches, "
            f"need {measured_iterations}"
        )
    for ordinal, (source, name, launch) in enumerate(
        launches[-measured_iterations:]
    ):
        lifetimes = launch.get("warpLifetimes", [])
        if not lifetimes:
            raise SystemExit(f"rank {rank}: MegaMoE launch has no warp lifetimes")
        rows = []
        for lifetime in lifetimes:
            location = lifetime_location(lifetime, trace["locationTable"])
            rows.append(
                (
                    int(location["warpId"]),
                    int(lifetime["startTs"]),
                    int(lifetime["endTs"]),
                )
            )
        base = min(start for _, start, _ in rows)
        kernel_end = max(end for _, _, end in rows)
        metric = {
            "kernel_envelope_us": (kernel_end - base) / 1000.0,
        }
        role_ends = {}
        for role, warp_ids in roles.items():
            selected = [(start, end) for warp, start, end in rows if warp in warp_ids]
            if not selected:
                raise SystemExit(f"rank {rank}: launch is missing {role} warps")
            role_start = min(start for start, _ in selected)
            role_end = max(end for _, end in selected)
            metric[f"{role}_lifetime_us"] = (role_end - role_start) / 1000.0
            metric[f"{role}_tail_us"] = (role_end - base) / 1000.0
            role_ends[role] = role_end
        compute_tail = max(
            role_ends["tma_a"], role_ends["tma_b"],
            role_ends["mma"], role_ends["scheduler"],
        )
        post_compute_tail = max(0, role_ends["epilogue_combine"] - compute_tail)
        metric["compute_pipeline_tail_us"] = (compute_tail - base) / 1000.0
        metric["post_compute_combine_tail_us"] = post_compute_tail / 1000.0
        # Explicit source markers define overlapping phase envelopes. The
        # combine phase starts with L2 BF16 packing/return, before all compute
        # tiles finish; its duration must not be added to the compute duration.
        points = {}
        marker_counts = {}
        for phase in ("Dispatch", "Compute", "Combine"):
            for boundary, reduce_ts in (("Start", min), ("End", max)):
                event_name = f"Mega{phase}{boundary}"
                times = [
                    int(marker["timestamp"])
                    for marker in launch.get("markers", [])
                    if indexed_value(trace["stringTable"], marker["markerNameIdx"])
                    == event_name
                ]
                if not times:
                    raise SystemExit(f"rank {rank}: missing source marker {event_name}")
                if min(times) < base or max(times) > kernel_end:
                    raise SystemExit(f"rank {rank}: {event_name} is outside the kernel envelope")
                marker_counts[event_name] = len(times)
                points[f"{phase.lower()}_{boundary.lower()}_us"] = (
                    reduce_ts(times) - base
                ) / 1000.0
        metric.update(points)
        for phase in ("dispatch", "compute", "combine"):
            if points[f"{phase}_start_us"] > points[f"{phase}_end_us"]:
                raise SystemExit(f"rank {rank}: reversed {phase} boundaries")
            metric[f"{phase}_span_us"] = (
                points[f"{phase}_end_us"] - points[f"{phase}_start_us"]
            )
        overlap = max(0.0, min(points["dispatch_end_us"], points["combine_end_us"])
                      - max(points["dispatch_start_us"], points["combine_start_us"]))
        metric["communication_union_us"] = (
            metric["dispatch_span_us"] + metric["combine_span_us"] - overlap
        )
        metric["pre_compute_dispatch_exposure_us"] = max(
            0.0, min(points["dispatch_end_us"], points["compute_start_us"])
            - points["dispatch_start_us"]
        )
        metric["post_compute_combine_exposure_us"] = max(
            0.0, points["combine_end_us"]
            - max(points["combine_start_us"], points["compute_end_us"])
        )
        metric["pre_compute_gap_us"] = max(
            0.0, points["compute_start_us"] - points["dispatch_end_us"]
        )
        metric["post_compute_gap_us"] = max(
            0.0, points["combine_start_us"] - points["compute_end_us"]
        )
        metric["post_combine_tail_us"] = (
            metric["kernel_envelope_us"] - points["combine_end_us"]
        )
        messages = []
        if dispatch_detail:
            names = ("MegaDispatchRemoteIssue", "MegaDispatchRemoteArrived", "MegaDispatchRemotePublished")
            by_location = {}
            for marker in launch.get("markers", []):
                event = indexed_value(trace["stringTable"], marker["markerNameIdx"])
                if event in names:
                    entry = by_location.setdefault(marker["locIdx"], {name: [] for name in names})
                    entry[event].append(int(marker["timestamp"]))
            if not by_location:
                raise SystemExit("dispatch detail requested but remote-copy markers are missing")
            for loc, events in by_location.items():
                if len({len(events[name]) for name in names}) != 1:
                    raise SystemExit(f"unpaired dispatch detail events at location {loc}")
                for issue, arrived, published in zip(*(sorted(events[name]) for name in names)):
                    if not base <= issue <= arrived <= published <= kernel_end:
                        raise SystemExit(f"invalid dispatch message ordering at location {loc}")
                    messages.append({
                        "locIdx": loc,
                        "issue_us": (issue - base) / 1000.0,
                        "arrived_us": (arrived - base) / 1000.0,
                        "published_us": (published - base) / 1000.0,
                    })
            for boundary in ("issue", "arrived", "published"):
                times = [message[boundary + "_us"] for message in messages]
                metric[f"first_remote_{boundary}_us"] = min(times)
                metric[f"last_remote_{boundary}_us"] = max(times)
                metric[f"remote_{boundary}_spread_us"] = max(times) - min(times)
            metric["dispatch_to_first_remote_arrival_us"] = (
                metric["first_remote_arrived_us"] - points["dispatch_start_us"]
            )
            for label, start, end in (
                ("remote_request", "issue", "arrived"),
                ("remote_local_staging", "arrived", "published"),
            ):
                for stat, value in stats([m[end + "_us"] - m[start + "_us"] for m in messages]).items():
                    metric[f"{label}_{stat}_us"] = value
        samples.append(
            {
                "rank": rank,
                "measured_ordinal": ordinal,
                "launch_source": source,
                "kernel_name": name,
                "marker_counts": marker_counts,
                "dispatch_messages": messages,
                **metric,
            }
        )

_, baseline_record = load_benchmark("baseline.json")
load_benchmark("profiled.json")
csv_path = run_root / "ranks" / f"rank-{profile_rank}" / "nsys" / f"megamoe-rank{profile_rank}.cuda_gpu_kern_sum.csv"
csv_text = csv_path.read_text()
header = csv_text.index("Time (%),")
nsys_rows = [
    row for row in csv.DictReader(io.StringIO(csv_text[header:]))
    if "sm100_fp8_fp4_mega_moe_impl" in row["Name"]
]
if len(nsys_rows) != 1:
    raise SystemExit(f"expected one Nsight MegaMoE specialization, got {len(nsys_rows)}")
nsys_row = nsys_rows[0]
if int(nsys_row["Instances"]) != baseline_record["timing_iterations"]:
    raise SystemExit("Nsight MegaMoE instance count does not match the measured iterations")
nsys_us = {
    key: float(nsys_row[column]) / 1000.0
    for key, column in (
        ("mean", "Avg (ns)"), ("median", "Med (ns)"),
        ("min", "Min (ns)"), ("max", "Max (ns)"),
    )
}
metric_names = [name for name in samples[0] if name.endswith("_us")]
summary = {name: stats([sample[name] for sample in samples]) for name in metric_names}
result = {
    "contract": {
        "model_shape": "Qwen3-235B-A22B",
        "topology": f"EP{ep_size}, {(ep_size + 3) // 4} trays",
        "ep_size": ep_size,
        "reference_input": "see sweep manifest for equivalent AFD inputs",
        "microbatches": 2,
        "rows_per_ep_rank_per_microbatch": rows_per_rank,
        "valid_rows_per_rank": [global_tokens // ep_size + int(r < global_tokens % ep_size)
                                for r in range(ep_size)],
        "global_tokens_per_launch": global_tokens,
        "routed_slots_per_launch": global_tokens * 8,
        "routes_per_expert_min": global_tokens * 8 // 128,
        "routes_per_expert_max": (global_tokens * 8 + 127) // 128,
        "hidden": 4096,
        "intermediate": 1536,
        "experts": 128,
        "top_k": 8,
        "activation_dtype": "fp8",
        "weight_dtype": "fp4",
        "num_sms": 150,
        "routes_prepared": True,
        "rank_gated_combine": True,
        "route_ready_dispatch": True,
        "rank_ready_route_publish": True,
        "input_route_distribution": "perfect_balance",
        "rank_interleaved_expert_mapping": balance_ranks,
        "kernel_assumes_perfect_routing": False,
        "route_producer_launched": True,
        "warp_summary_selects_megamoe_only": True,
        "kernel_debug_timing": False,
        "iket_mode": "source-markers",
        "iket_profile_rank": profile_rank,
        "dispatch_detail": dispatch_detail,
        "profile_rank_weights": "profile-rank-weights.pt",
    },
    "interpretation": {
        "epilogue_combine_lifetime_us": (
            "epilogue/combine warp entry-to-exit; includes its compute wait"
        ),
        "post_compute_combine_tail_us": (
            "last epilogue/combine exit minus last TMA/MMA/scheduler exit"
        ),
        "dispatch_tail_us": (
            "kernel entry to last dispatch-warp exit; includes end-of-kernel "
            "cleanup and is not dispatch latency"
        ),
        "source_markers": {
            "dispatch": "dispatch warp entry through local payload publication, excluding cleanup",
            "compute": "first MMA operands ready through last L2 accumulator ready",
            "combine": "first L2 BF16 epilogue through source-side reduction completion",
            "communication_union_us": "union of dispatch and combine envelopes, including readiness and local work",
            "exposure": "temporal overlap with the pre/post-compute interval; not causal attribution of stalls",
            "post_combine_tail_us": "last warp exit minus final CombineEnd marker, including concurrent cleanup",
            "dispatch_detail": "remote/self-excluded copies: issue, observed TMA completion after scale/weight work, local publication; diagnostic arrival marker adds one warp sync",
        },
    },
    "trace_files": trace_files,
    "nsys_kernel_us": nsys_us,
    "summary_us": summary,
    "samples": samples,
}
(run_root / "decomposition.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
csv_records = []
for sample in samples:
    record = {
        "ep": ep_size,
        "rows_per_rank": rows_per_rank,
        "global_tokens": global_tokens,
        "rank": sample["rank"],
        "measured_ordinal": sample["measured_ordinal"],
        "nsys_kernel_median_us": nsys_us["median"],
        "initial_interval_us": sample["dispatch_start_us"],
        "dispatch_compute_overlap_us": max(
            0.0, min(sample["dispatch_end_us"], sample["compute_end_us"])
            - max(sample["dispatch_start_us"], sample["compute_start_us"])
        ),
        "combine_compute_overlap_us": max(
            0.0, min(sample["combine_end_us"], sample["compute_end_us"])
            - max(sample["combine_start_us"], sample["compute_start_us"])
        ),
        **{name: sample[name] for name in metric_names},
    }
    csv_records.append(record)
with (run_root / "decomposition.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_records[0]))
    writer.writeheader()
    writer.writerows(csv_records)

print(f"FASTAFD_MEGAMOE_IKET_RESULT={run_root / 'decomposition.json'}")
print(f"nsys_kernel_us={nsys_us}")
for name in (
    "kernel_envelope_us",
    "compute_pipeline_tail_us",
    "post_compute_combine_tail_us",
    "dispatch_span_us", "compute_span_us", "combine_span_us",
    "communication_union_us",
    "pre_compute_dispatch_exposure_us", "post_compute_combine_exposure_us",
    "post_combine_tail_us",
):
    values = summary[name]
    print(
        f"{name}=min:{values['min']:.3f},"
        f"median:{values['median']:.3f},max:{values['max']:.3f}"
    )
PY
