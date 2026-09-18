#!/usr/bin/env bash
# Model-configurable fixed-global-work MegaMoE Nsight timing and six-point IKET trace.
# Run from an OCI-HSG login node; with no allocation this script submits itself.
# Optional overrides: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV
#                     FASTAFD_IKET_INCLUDE FASTAFD_IKET_PYTHON_DEPS
#                     FASTAFD_IKET_ARTIFACT_ROOT FASTAFD_IKET_DEEPGEMM_CACHE_ROOT
#                     DG_JIT_CACHE_DIR
#                     FASTAFD_IKET_CAPTURE_ONLY (1: analyze later on a CPU allocation)
#                     FASTAFD_IKET_VERIFY_SOURCE (0 for archive-authoritative sources)
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
FASTAFD_SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$FASTAFD_USER_ROOT/FastAFD/source/local-mirror}
FASTAFD_IMAGE=${FASTAFD_IMAGE:-$FASTAFD_USER_ROOT/trtllm_qwen3_235b_megamoe/images/pytorch-26.05-py3-arm64-trtllm-fa778397186c09dd57bcaa5020e66270768b675b.sqsh}
FASTAFD_VENV=${FASTAFD_VENV:-$FASTAFD_USER_ROOT/fastafd_reproduce/envs/minisgl-3c716194-cuda130.artifact-5175891}
FASTAFD_IKET_ARTIFACT_ROOT=${FASTAFD_IKET_ARTIFACT_ROOT:-$FASTAFD_USER_ROOT/FastAFD/artifacts/megamoe_iket_qwen3}
FASTAFD_IKET_DEEPGEMM_CACHE_ROOT=${FASTAFD_IKET_DEEPGEMM_CACHE_ROOT:-}
FASTAFD_IKET_INCLUDE=${FASTAFD_IKET_INCLUDE:-$FASTAFD_USER_ROOT/FastAFD/source/iket/include}
FASTAFD_IKET_PYTHON_DEPS=${FASTAFD_IKET_PYTHON_DEPS:-}
MEGAMOE_IKET_DISPATCH_DETAIL=${MEGAMOE_IKET_DISPATCH_DETAIL:-0}
export MEGAMOE_IKET_METADATA_DETAIL=${MEGAMOE_IKET_METADATA_DETAIL:-0}
export MEGAMOE_IKET_CRITICAL=${MEGAMOE_IKET_CRITICAL:-0}
export MEGAMOE_IKET_MAX_TS_PER_WARP=${MEGAMOE_IKET_MAX_TS_PER_WARP:-1024}
[[ $MEGAMOE_IKET_MAX_TS_PER_WARP =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $MEGAMOE_IKET_CRITICAL == 0 || $MEGAMOE_IKET_CRITICAL == 1 ]] || exit 2
if [[ $MEGAMOE_IKET_CRITICAL == 1 && $MEGAMOE_IKET_METADATA_DETAIL != 0 ]]; then
    echo "critical IKET excludes metadata detail markers" >&2; exit 2
fi
[[ $MEGAMOE_IKET_METADATA_DETAIL == 0 || $MEGAMOE_IKET_METADATA_DETAIL == 1 ]] || exit 2
[[ $MEGAMOE_IKET_DISPATCH_DETAIL == 0 || $MEGAMOE_IKET_DISPATCH_DETAIL == 1 ]] || {
    echo "MEGAMOE_IKET_DISPATCH_DETAIL must be 0 or 1" >&2
    exit 2
}
MEGAMOE_IKET_EP=${MEGAMOE_IKET_EP:-8}
MEGAMOE_IKET_GLOBAL_TOKENS=${MEGAMOE_IKET_GLOBAL_TOKENS:-192}
export MEGAMOE_IKET_MODEL=${MEGAMOE_IKET_MODEL:-Qwen3-235B-A22B}
export MEGAMOE_IKET_HIDDEN=${MEGAMOE_IKET_HIDDEN:-4096}
export MEGAMOE_IKET_INTERMEDIATE=${MEGAMOE_IKET_INTERMEDIATE:-1536}
export MEGAMOE_IKET_EXPERTS=${MEGAMOE_IKET_EXPERTS:-128}
export MEGAMOE_IKET_TOP_K=${MEGAMOE_IKET_TOP_K:-8}
export MEGAMOE_IKET_SMS=${MEGAMOE_IKET_SMS:-150}
export MEGAMOE_IKET_BUCKET=${MEGAMOE_IKET_BUCKET:-3840}
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
    # A recorded ordinary result is also the reference for instrumented replay.
    if [[ $FASTAFD_IKET_RANK_MODE == profile && ${MEGAMOE_RESULT_MODE:-} == record ]]; then
        export MEGAMOE_RESULT_MODE=compare
    fi
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
                "${MEGAMOE_SMOKE_TEST_SCRIPT:-$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py}"
        fi
        exec "$FASTAFD_VENV/bin/python" \
            "${MEGAMOE_SMOKE_TEST_SCRIPT:-$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py}"
    fi
    # Add optional SModel dependencies only to the profiler interpreter.
    # Its child workload retains the original Python environment.
    exec /usr/bin/python3 -c '
import os, runpy, shutil, sys
deps = os.environ["FASTAFD_IKET_PYTHON_DEPS"]
if deps:
    sys.path.insert(0, deps)
if os.environ.get("FASTAFD_IKET_EVENT_REGISTER_LIMITS"):
    runpy.run_path(os.environ["FASTAFD_IKET_REGISTER_LIMIT_SCRIPT"], run_name="__main__")
runpy.run_path(shutil.which("run-iket"), run_name="__main__")
' --output-dir "$rank_root/iket" --clobber \
        --context-buffer-size 268435456 --log-level info \
        profile --max-ts-cnt-per-warp "$MEGAMOE_IKET_MAX_TS_PER_WARP" --postprocess json -- \
        "$FASTAFD_VENV/bin/python" \
        "${MEGAMOE_SMOKE_TEST_SCRIPT:-$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py}"
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
[[ -f "${MEGAMOE_SMOKE_TEST_SCRIPT:-$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py}" ]] || {
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

if [[ ${FASTAFD_IKET_IN_CONTAINER:-0} == 1 ]]; then
    [[ ${FASTAFD_IKET_SINGLE_FABRIC:-0} == 1 ]] || {
        echo 'container worker requires parent fabric validation' >&2; exit 2;
    }
else
    mapfile -t fabric_blocks < <(
        scontrol show hostnames "$SLURM_JOB_NODELIST" |
            sed -E 's/-T[0-9]+$//' | sort -u
    )
    (( ${#fabric_blocks[@]} == 1 )) || {
        printf 'allocation crosses NVL72 fabric blocks: %s\n' \
            "${fabric_blocks[*]}" >&2
        exit 2
    }
fi

run_root=$FASTAFD_IKET_ARTIFACT_ROOT/$SLURM_JOB_ID
deepgemm_cache=${FASTAFD_IKET_DEEPGEMM_CACHE_ROOT:-$run_root/cache/deepgemm}
tvm_cache=${FASTAFD_IKET_TVM_CACHE_ROOT:-$deepgemm_cache/tvm_ffi}
mkdir -p "$run_root/ranks" "$deepgemm_cache" "$tvm_cache"
export FASTAFD_IKET_RUN_ROOT=$run_root
if [[ ${FASTAFD_IKET_IN_CONTAINER:-0} != 1 || $SLURM_PROCID == "$MEGAMOE_IKET_PROFILE_RANK" ]]; then
    cp "$script_path" "$run_root/profile_megamoe_iket.sbatch"
    if [[ ${FASTAFD_IKET_VERIFY_SOURCE:-1} == 1 ]]; then
        git -C "$FASTAFD_SOURCE_REPO" rev-parse HEAD > "$run_root/source-head.txt"
        git -C "$FASTAFD_SOURCE_REPO" status --short > "$run_root/source-status.txt"
        git -C "$FASTAFD_SOURCE_REPO" diff -- \
            tests/afd_megamoe_same_rank_smoke.py \
            python/minisgl/kernel/csrc/deepgemm/csrc/jit/compiler.hpp \
            python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh \
            python/minisgl/kernel/csrc/deepgemm/csrc/jit_kernels/heuristics/mega_moe.hpp \
            > "$run_root/source.patch"
    elif [[ ${FASTAFD_IKET_VERIFY_SOURCE:-1} == 0 ]]; then
        printf '%s\n' "$FASTAFD_SOURCE_REPO" > "$run_root/source-snapshot.txt"
    else
        echo 'FASTAFD_IKET_VERIFY_SOURCE must be 0 or 1' >&2
        exit 2
    fi
fi

if [[ ${FASTAFD_IKET_IN_CONTAINER:-0} == 1 ]]; then
    master_addr=${MASTER_ADDR:?}
else
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
fi
master_port=${MASTER_PORT:-29631}

run_world() {
    local mode=$1 warmups=$2 iterations=$3 output=$4
    local iket_include=""
    local compare_combine=0
    local launch=()
    if [[ ${FASTAFD_IKET_IN_CONTAINER:-0} != 1 ]]; then
        launch=(srun --nodes="$MEGAMOE_IKET_NODES" --ntasks="$MEGAMOE_IKET_EP"
            --ntasks-per-node="$MEGAMOE_IKET_TASKS_PER_NODE" --kill-on-bad-exit=1
            --container-image="$FASTAFD_IMAGE" --container-mount-home
            --container-mounts=/lustre:/lustre --no-container-remap-root)
    fi
    [[ $mode != profile ]] || iket_include=$FASTAFD_IKET_INCLUDE
    if [[ $mode == normal ]]; then
        compare_combine=${MEGAMOE_SMOKE_COMPARE_COMBINE:-0}
    fi
    "${launch[@]}" \
        env -u CUDA_VISIBLE_DEVICES \
        FASTAFD_IKET_RANK_MODE="$mode" \
        FASTAFD_IKET_RUN_ROOT="$run_root" \
        PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
        PYTHONPATH="$FASTAFD_SOURCE_REPO/python" \
        MINISGL_DEEPGEMM_BUILD_DIR="$deepgemm_cache" \
        DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$deepgemm_cache/jit}" \
        DG_JIT_IKET_INCLUDE="$iket_include" \
        DG_JIT_IKET_CRITICAL="$MEGAMOE_IKET_CRITICAL" \
        TVM_FFI_CACHE_DIR="$tvm_cache" \
        MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4 \
        MEGAMOE_SMOKE_PROFILE_ONLY=1 \
        MEGAMOE_SMOKE_FIXED_PERFECT_ONLY=1 \
        MEGAMOE_SMOKE_PRODUCTION_ROUTES=1 \
        MEGAMOE_SMOKE_BALANCE_RANKS="${MEGAMOE_SMOKE_BALANCE_RANKS:-0}" \
        MEGAMOE_SMOKE_COMPARE_COMBINE="$compare_combine" \
        MEGAMOE_SMOKE_ROWS="$MEGAMOE_IKET_ROWS" \
        MEGAMOE_SMOKE_GLOBAL_TOKENS="$MEGAMOE_IKET_GLOBAL_TOKENS" \
        MEGAMOE_SMOKE_SMS="$MEGAMOE_IKET_SMS" \
        MEGAMOE_SMOKE_BUCKET="$MEGAMOE_IKET_BUCKET" \
        MEGAMOE_SMOKE_WARMUPS="$warmups" \
        MEGAMOE_SMOKE_ITERATIONS="$iterations" \
        MEGAMOE_SMOKE_DEBUG_TIMINGS=0 \
        MEGAMOE_SMOKE_CONTEND_CYCLES=0 \
        MEGAMOE_SMOKE_NUM_EXPERTS="$MEGAMOE_IKET_EXPERTS" \
        MEGAMOE_SMOKE_HIDDEN="$MEGAMOE_IKET_HIDDEN" \
        MEGAMOE_SMOKE_INTERMEDIATE="$MEGAMOE_IKET_INTERMEDIATE" \
        MEGAMOE_SMOKE_TOP_K="$MEGAMOE_IKET_TOP_K" \
        MEGAMOE_SMOKE_OUTPUT="$output" \
        MASTER_ADDR="$master_addr" MASTER_PORT="$master_port" \
        bash "$script_path"
}

printf 'FASTAFD_MEGAMOE_IKET_CONFIG=job=%s nodes=%s source=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$FASTAFD_SOURCE_REPO"
capture_phase=${FASTAFD_IKET_CAPTURE_PHASE:-all}
[[ $capture_phase == all || $capture_phase == normal || $capture_phase == profile ]] || {
    echo 'FASTAFD_IKET_CAPTURE_PHASE must be all, normal, or profile' >&2; exit 2;
}
if [[ $capture_phase != all && ${FASTAFD_IKET_CAPTURE_ONLY:-0} != 1 ]]; then
    echo 'separate capture phases require FASTAFD_IKET_CAPTURE_ONLY=1' >&2; exit 2
fi
if [[ $capture_phase == profile && ! -s $run_root/baseline.json ]]; then
    echo 'profile phase requires a completed ordinary capture in this run' >&2; exit 2
fi
if [[ $capture_phase != profile ]]; then
    run_world normal "$MEGAMOE_BASELINE_WARMUPS" \
        "$MEGAMOE_BASELINE_ITERATIONS" "$run_root/baseline.json"
fi
if [[ $capture_phase != normal ]]; then
    run_world profile "$MEGAMOE_IKET_WARMUPS" \
        "$MEGAMOE_IKET_ITERATIONS" "$run_root/profiled.json"
fi
if [[ ${FASTAFD_IKET_IN_CONTAINER:-0} == 1 && $SLURM_PROCID != "$MEGAMOE_IKET_PROFILE_RANK" ]]; then
    exit 0
fi
if [[ ${FASTAFD_IKET_CAPTURE_ONLY:-0} == 1 ]]; then
    printf 'IKET_CAPTURE_COMPLETE=%s\n' "$run_root"
    exit 0
fi
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
metadata_materialized_value = os.environ.get("MEGAMOE_IKET_METADATA_MATERIALIZED", "1")
if metadata_materialized_value not in ("0", "1"):
    raise SystemExit("MEGAMOE_IKET_METADATA_MATERIALIZED must be 0 or 1")
metadata_materialized = metadata_materialized_value == "1"
rows_per_rank = (global_tokens + ep_size - 1) // ep_size
balance_ranks = bool(int(os.environ.get("MEGAMOE_SMOKE_BALANCE_RANKS", "0")))
hidden, intermediate, experts, top_k, num_sms, bucket = (
    int(os.environ[f"MEGAMOE_IKET_{key}"])
    for key in ("HIDDEN", "INTERMEDIATE", "EXPERTS", "TOP_K", "SMS", "BUCKET")
)
case_key = f"bucket{bucket}_rows{rows_per_rank}_sms{num_sms}"


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
    dimensions = {"hidden": hidden, "intermediate": intermediate,
                  "experts": experts, "top_k": top_k}
    if record.get("model_dimensions") != dimensions:
        raise SystemExit(f"invalid {name}: model dimensions mismatch")
    expected_active = min(experts // ep_size, (global_tokens * top_k + ep_size - 1) // ep_size
                          if balance_ranks else global_tokens * top_k)
    if record.get("max_active_experts") != expected_active:
        raise SystemExit(f"invalid {name}: unexpected active local expert count")
    scope = os.environ.get("MEGAMOE_SMOKE_TIMING_SCOPE", "pipeline")
    expected_scope = "megamoe_kernel_only" if scope == "megamoe" else "route_producer_and_megamoe"
    if scope not in ("pipeline", "megamoe") or record.get("timed_scope") != expected_scope:
        raise SystemExit(f"invalid {name}: timing scope mismatch")
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
producer_samples = []
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
    producers = []
    for source, launch in launch_records:
        name = kernel_name(launch, trace.get("stringTable", []))
        if any(indexed_value(trace["stringTable"], m["markerNameIdx"]) == "RouteMetadataTransferStart"
               for m in launch.get("markers", [])):
            producers.append((source, name, launch))
        lowered = name.lower()
        if "mega_moe" in lowered and "fp4" in lowered:
            launches.append((source, name, launch))
    if len(launches) < measured_iterations:
        raise SystemExit(
            f"rank {rank}: found {len(launches)} FP4 MegaMoE launches, "
            f"need {measured_iterations}"
        )
    if producers:
        if len(producers) != len(launches):
            raise SystemExit("producer/MegaMoE launch counts differ; cannot pair startup traces")
        for ordinal, (source, name, launch) in enumerate(producers[-measured_iterations:]):
            producer_lifetimes = launch.get("warpLifetimes", [])
            if not producer_lifetimes:
                raise SystemExit("metadata producer has no warp lifetimes for common-origin timing")
            producer_samples.append({
                "rank": rank, "measured_ordinal": ordinal, "launch_source": source,
                "kernel_name": name,
                "kernel_start_timestamp_ns": min(int(w["startTs"]) for w in producer_lifetimes),
                "kernel_end_timestamp_ns": max(int(w["endTs"]) for w in producer_lifetimes),
                "markers": [
                    {"name": indexed_value(trace["stringTable"], m["markerNameIdx"]),
                     "location": lifetime_location(m, trace["locationTable"]),
                     "timestamp_ns": int(m["timestamp"])}
                    for m in launch.get("markers", [])
                ],
            })
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
        event_names = {indexed_value(trace["stringTable"], m["markerNameIdx"])
                       for m in launch.get("markers", [])}
        compute_events = event_names & {"MegaComputeStart", "MegaMmaIssue"}
        if len(compute_events) != 1:
            raise SystemExit("require exactly one compute-start marker contract")
        compute_start_event = next(iter(compute_events))
        metric["compute_start_semantics"] = (
            "actual_mma_issue" if compute_start_event == "MegaMmaIssue" else "operands_ready_proxy")
        phases = ["Dispatch", "Compute", "Combine"]
        metadata_detail = os.environ["MEGAMOE_IKET_METADATA_DETAIL"] == "1"
        if metadata_detail:
            phases.extend(("Metadata", "Payload"))
        for phase in phases:
            boundaries = (("Start", min),) if phase == "Payload" else (("Start", min), ("End", max))
            for boundary, reduce_ts in boundaries:
                event_name = f"Mega{phase}{boundary}"
                if phase == "Compute" and boundary == "Start":
                    event_name = compute_start_event
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
        sf_owner_events = [marker for marker in launch.get("markers", [])
                           if indexed_value(trace["stringTable"], marker["markerNameIdx"]) == "MegaSFByTMAA"]
        if sf_owner_events:
            locations = {marker.get("locIdx", marker.get("locationIdx")) for marker in sf_owner_events}
            if None in locations or len(locations) != num_sms:
                raise SystemExit("incomplete TMA-A scale-factor ownership markers")
            marker_counts["MegaSFByTMAA"] = len(sf_owner_events)
        payload_acquires = [m for m in launch.get("markers", [])
                            if indexed_value(trace["stringTable"], m["markerNameIdx"]) == "MegaPayloadAcquired"]
        if payload_acquires:
            locations = {m.get("locIdx", m.get("locationIdx")) for m in payload_acquires}
            times = [int(m["timestamp"]) for m in payload_acquires]
            if None in locations or len(locations) != num_sms or not base <= min(times) <= max(times) <= kernel_end:
                raise SystemExit("incomplete or invalid consumer payload acquisitions")
            marker_counts["MegaPayloadAcquired"] = len(times)
            metric["payload_acquire_first_us"] = (min(times) - base) / 1000.0
            metric["payload_acquire_last_us"] = (max(times) - base) / 1000.0
            # Removing a helper must not remove payload readiness from dispatch.
            # The first consumer acquires the same all-source completion value;
            # later consumer markers also contain their own scheduling delays.
            metric["dispatch_roles_end_us"] = points["dispatch_end_us"]
            points["dispatch_end_us"] = max(points["dispatch_end_us"], metric["payload_acquire_first_us"])
            metric.update(points)
        genphase = None
        if any(indexed_value(trace["stringTable"], marker["markerNameIdx"]) == "GenphaseStart"
               for marker in launch.get("markers", [])):
            communication = launch
            genphase = {}
            events_present = {indexed_value(trace["stringTable"], marker["markerNameIdx"])
                              for marker in communication.get("markers", [])}
            send_detail = [name for name in ("TopkStaged", "TopkWritten", "TopkSent")
                           if "Genphase" + name in events_present]
            input_detail = ["InputsReady"] if "GenphaseInputsReady" in events_present else []
            producer_metadata = ["MetadataFromProducer"] if "GenphaseMetadataFromProducer" in events_present else []
            cleanup_retirement = ["CleanupOwnsRetirement"] if "GenphaseCleanupOwnsRetirement" in events_present else []
            if bool(cleanup_retirement) != bool(payload_acquires):
                raise SystemExit("cleanup retirement requires complete consumer payload markers")
            expert_lists = ["ExpertLists"] if "GenphaseExpertLists" in events_present else []
            fused_scatter = ["FusedScatter"] if "GenphaseFusedScatter" in events_present else []
            index_scatter = ["IndexScatter"] if "GenphaseIndexScatter" in events_present else []
            refine_events = [] if index_scatter else ["RefineStart"]
            payload_events = [name for name in ("PayloadPushed", "PayloadIssued", "PayloadDrained", "PayloadPartitioned", "PayloadFromProducer", "PayloadSenderDone", "PayloadStoresIssued")
                              if "Genphase" + name in events_present]
            # Direct startup markers distinguish drained outgoing stores from
            # the subsequent readiness publication. Older paths mark only one.
            transport_detail = []
            if ("PayloadPushed" in payload_events or "PayloadSenderDone" in payload_events) and "PayloadDrained" in payload_events:
                payload_events.remove("PayloadDrained")
                transport_detail = ["PayloadDrained"]
            if "GenphasePayloadSendStart" in events_present:
                transport_detail.append("PayloadSendStart")
            if "GenphaseTailPublishStart" in events_present:
                transport_detail.append("TailPublishStart")
            if "GenphasePayloadValidateStart" in events_present:
                transport_detail.append("PayloadValidateStart")
            if "GenphaseRemainderValidateStart" in events_present:
                transport_detail.append("RemainderValidateStart")
            readiness_detail = [name for name in (
                "SfStoresIssued", "MetadataStoresIssued", "SfReadyIssue", "FirstPayloadIssue")
                if "Genphase" + name in events_present]
            if "SfStoresIssued" in readiness_detail:
                if ("MetadataStoresIssued" in readiness_detail) != metadata_materialized:
                    raise SystemExit("receiver metadata materialization marker contract mismatch")
            elif not metadata_materialized:
                raise SystemExit("direct producer metadata requires SF store detail markers")
            transport_detail.extend(readiness_detail)
            index_release = "FullPayloadIssue" if "GenphaseFullPayloadIssue" in events_present else "IndexReady"
            metadata_ctas = int(os.environ.get("MEGAMOE_IKET_METADATA_CTAS", ep_size))
            index_ctas = int(os.environ.get("MEGAMOE_IKET_METADATA_INDEX_CTAS", metadata_ctas))
            validation_ctas = int(os.environ.get("MEGAMOE_IKET_METADATA_PAYLOAD_CTAS", metadata_ctas))
            if not 0 < index_ctas <= metadata_ctas or not 0 < validation_ctas <= metadata_ctas:
                raise SystemExit("invalid metadata role CTA counts")
            if index_release == "FullPayloadIssue" and "GenphaseIndexReady" in events_present:
                raise SystemExit("ambiguous full-payload/index publication marker contract")
            if len(payload_events) > 1 or (not payload_events and not cleanup_retirement):
                raise SystemExit("ambiguous or missing generation-specialized payload marker")
            payload_event = payload_events[0] if payload_events else None
            if send_detail and (len(send_detail) != 2 or "TopkSent" not in send_detail):
                raise SystemExit("incomplete generation-specialized metadata send markers")
            for event in ("Start", *input_detail, *producer_metadata, *expert_lists, *cleanup_retirement, *send_detail, "TopkReceived", "SizesReady", index_release,
                          *payload_events, *transport_detail, *refine_events, *fused_scatter, *index_scatter, "End"):
                events = [marker for marker in communication.get("markers", [])
                          if indexed_value(trace["stringTable"], marker["markerNameIdx"]) == "Genphase" + event]
                times = [int(marker["timestamp"]) for marker in events]
                if not times:
                    raise SystemExit(f"rank {rank}: missing Genphase{event}")
                # A marker following lane-predicated peer publication may be
                # recorded twice by one warp. Count participating locations,
                # retaining both timestamps and the raw record count.
                locations = {marker.get("locIdx", marker.get("locationIdx")) for marker in events}
                if None in locations:
                    raise SystemExit(f"rank {rank}: Genphase{event} has no location")
                genphase[event] = {"first_us": (min(times) - base) / 1000,
                                   "last_us": (max(times) - base) / 1000,
                                   "count": len(locations), "raw_count": len(times)}
            begin, end = genphase["Start"]["first_us"], genphase["End"]["last_us"]
            if begin > end or any(not (begin <= value["first_us"] <= value["last_us"] <= end)
                                  for value in genphase.values()):
                raise SystemExit("invalid generation-specialized dispatch marker envelope")
            producer_payload = payload_event in (None, "PayloadFromProducer")
            if cleanup_retirement:
                refiners = 0 if producer_payload else genphase[payload_event]["count"]
            else:
                refiners = (1 if producer_payload else genphase[payload_event]["count"] + 1) if fused_scatter or index_scatter else genphase["RefineStart"]["count"]
            expected_counts = {"TopkReceived": metadata_ctas,
                               index_release: validation_ctas if index_release == "FullPayloadIssue" else index_ctas,
                               "SizesReady": 1,
                               "Start": metadata_ctas + refiners, "End": metadata_ctas + refiners}
            if index_release == "FullPayloadIssue":
                expected_counts[index_release] *= int(os.environ.get("MEGAMOE_IKET_FULL_PAYLOAD_PUBLISHERS_PER_CTA", "1"))
            if payload_event:
                expected_counts[payload_event] = 1 if producer_payload else refiners - (0 if cleanup_retirement else 1)
            if "TailPublishStart" in transport_detail:
                expected_counts["TailPublishStart"] = refiners - 1
            if "PayloadValidateStart" in transport_detail:
                expected_counts["PayloadValidateStart"] = validation_ctas
            if "RemainderValidateStart" in transport_detail:
                expected_counts["RemainderValidateStart"] = validation_ctas
            for event in readiness_detail:
                count = validation_ctas if event == "FirstPayloadIssue" else index_ctas
                if event == "SfStoresIssued":
                    count *= int(os.environ.get("MEGAMOE_IKET_SF_WORK_WARPS", "8"))
                if event == "FirstPayloadIssue":
                    count *= int(os.environ.get("MEGAMOE_IKET_FIRST_PAYLOAD_PUBLISHERS_PER_CTA", "1"))
                expected_counts[event] = count
            expected_counts.update({name: ep_size if fused_scatter else refiners for name in refine_events})
            expected_counts.update({name: metadata_ctas for name in fused_scatter})
            expected_counts.update({name: index_ctas for name in index_scatter})
            expected_counts.update({name: ep_size for name in send_detail})
            expected_counts.update({name: refiners if producer_metadata else metadata_ctas + refiners for name in input_detail})
            expected_counts.update({name: metadata_ctas for name in producer_metadata + expert_lists + cleanup_retirement})
            if refiners < (0 if cleanup_retirement else 1 if producer_payload else 2) or any(genphase[key]["count"] != count for key, count in expected_counts.items()):
                raise SystemExit("incomplete generation-specialized dispatch role markers")
            metric["genphase_start_us"] = begin
            metric["genphase_end_us"] = end
            metric["genphase_span_us"] = end - begin
        if metadata_detail:
            if not (points["dispatch_start_us"] <= points["metadata_start_us"]
                    <= points["metadata_end_us"] <= points["dispatch_end_us"]
                    and points["dispatch_start_us"] <= points["payload_start_us"] <= points["dispatch_end_us"]):
                raise SystemExit("invalid metadata/payload marker envelopes")
            metric["metadata_span_us"] = points["metadata_end_us"] - points["metadata_start_us"]
            metric["payload_end_us"] = points["dispatch_end_us"]
            metric["payload_span_us"] = metric["payload_end_us"] - points["payload_start_us"]
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
        # Direct sparse markers retain the operand-ready proxy while measuring
        # actual first MMA issue and the independent epilogue wait intervals.
        for event, label in (("MegaMmaIssue", "mma_issue"),
                             ("MegaGatherSfStart", "gather_sf_start"),
                             ("MegaEpiL1Wait", "epi_l1_wait"),
                             ("MegaEpiL1Ready", "epi_l1_ready"),
                             ("MegaEpiL2Wait", "epi_l2_wait"),
                             ("MegaEpiL2Ready", "epi_l2_ready")):
            times = [int(m["timestamp"]) for m in launch.get("markers", [])
                     if indexed_value(trace["stringTable"], m["markerNameIdx"]) == event]
            if times:
                if not base <= min(times) <= max(times) <= kernel_end:
                    raise SystemExit(f"invalid {event} timestamps")
                metric[label + "_first_us"] = (min(times) - base) / 1000.0
                metric[label + "_last_us"] = (max(times) - base) / 1000.0
                marker_counts[event] = len(times)
        remainder_warps = int(os.environ.get("MEGAMOE_IKET_REMAINDER_WARPS_PER_CTA", "0"))
        if remainder_warps:
            expected = {(x, 0, 0, w) for x in range(num_sms) for w in range(remainder_warps)}
            remainder = {}
            for event in ("MegaRemainderWaitStart", "MegaRemainderReady"):
                observed = {}
                for m in launch.get("markers", []):
                    if indexed_value(trace["stringTable"], m["markerNameIdx"]) != event:
                        continue
                    loc = lifetime_location(m, trace["locationTable"])
                    key = (*loc["ctaId"], loc["warpId"])
                    if key in observed:
                        raise SystemExit(f"duplicate {event} at {key}")
                    observed[key] = int(m["timestamp"])
                if observed.keys() != expected:
                    raise SystemExit(f"incomplete active-gather remainder markers: {event}")
                remainder[event] = observed
                marker_counts[event] = len(observed)
            for key in expected:
                if not base <= remainder["MegaRemainderWaitStart"][key] <= remainder["MegaRemainderReady"][key] <= kernel_end:
                    raise SystemExit(f"invalid remainder-wait interval at {key}")
        if producers:
            producer = next(p for p in producer_samples
                            if p["rank"] == rank and p["measured_ordinal"] == ordinal)
            producer_start = producer["kernel_start_timestamp_ns"]
            metadata_start = min(m["timestamp_ns"] for m in producer["markers"]
                                 if m["name"] == "RouteMetadataTransferStart")
            metric.update({
                "metadata_transfer_start_us": (metadata_start - base) / 1000.0,
                "producer_entry_to_metadata_start_us": (metadata_start - producer_start) / 1000.0,
                "producer_entry_to_megamoe_entry_us": (base - producer_start) / 1000.0,
                "producer_entry_to_first_mma_us": (base - producer_start) / 1000.0 + points["compute_start_us"],
                "producer_entry_to_megamoe_end_us": (kernel_end - producer_start) / 1000.0,
                "metadata_start_to_first_mma_us": (base - metadata_start) / 1000.0 + points["compute_start_us"],
                "metadata_start_to_megamoe_end_us": (kernel_end - metadata_start) / 1000.0,
            })
            if "mma_issue_first_us" in metric:
                metric["producer_entry_to_mma_issue_us"] = (base - producer_start) / 1000.0 + metric["mma_issue_first_us"]
                metric["metadata_start_to_mma_issue_us"] = (base - metadata_start) / 1000.0 + metric["mma_issue_first_us"]
        samples.append(
            {
                "rank": rank,
                "measured_ordinal": ordinal,
                "launch_source": source,
                "kernel_name": name,
                "kernel_start_timestamp_ns": base,
                "source_markers": [
                    {"name": indexed_value(trace["stringTable"], m["markerNameIdx"]),
                     "location": lifetime_location(m, trace["locationTable"]),
                     "time_us": (int(m["timestamp"]) - base) / 1000.0}
                    for m in launch.get("markers", [])
                ],
                "marker_counts": marker_counts,
                "dispatch_messages": messages,
                "genphase_markers_relative_to_megamoe": genphase,
                "payload_acquire_markers": [
                    {"location": indexed_value(trace["locationTable"], m["locIdx"]),
                     "time_us": (int(m["timestamp"]) - base) / 1000.0}
                    for m in payload_acquires
                ],
                "first_compute_markers": [
                    {"location": indexed_value(trace["locationTable"], m["locIdx"]),
                     "time_us": (int(m["timestamp"]) - base) / 1000.0}
                    for m in launch.get("markers", [])
                    if indexed_value(trace["stringTable"], m["markerNameIdx"]) == compute_start_event
                ],
                **metric,
            }
        )

_, baseline_record = load_benchmark("baseline.json")
load_benchmark("profiled.json")
csv_path = run_root / "ranks" / f"rank-{profile_rank}" / "nsys" / f"megamoe-rank{profile_rank}.cuda_gpu_kern_sum.csv"
csv_text = csv_path.read_text()
header = csv_text.index("Time (%),")
all_nsys_rows = list(csv.DictReader(io.StringIO(csv_text[header:])))
nsys_rows = [
    row for row in all_nsys_rows
    if "sm100_fp8_fp4_mega_moe_impl" in row["Name"]
]
if len(nsys_rows) != 1:
    raise SystemExit(f"expected one Nsight MegaMoE specialization, got {len(nsys_rows)}")
nsys_row = nsys_rows[0]
if int(nsys_row["Instances"]) != baseline_record["timing_iterations"]:
    raise SystemExit("Nsight MegaMoE instance count does not match the measured iterations")
if baseline_record["timed_scope"] == "megamoe_kernel_only" and (
    sum(int(row["Instances"]) for row in all_nsys_rows) != baseline_record["timing_iterations"]
):
    raise SystemExit("kernel-only trace contains extra kernel launches")
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
    "producer_metadata_samples": producer_samples,
    "contract": {
        "model_shape": os.environ["MEGAMOE_IKET_MODEL"],
        "topology": f"EP{ep_size}, {(ep_size + 3) // 4} trays",
        "ep_size": ep_size,
        "reference_input": "see sweep manifest for equivalent AFD inputs",
        "microbatches": 2,
        "rows_per_ep_rank_per_microbatch": rows_per_rank,
        "valid_rows_per_rank": [global_tokens // ep_size + int(r < global_tokens % ep_size)
                                for r in range(ep_size)],
        "global_tokens_per_launch": global_tokens,
        "routed_slots_per_launch": global_tokens * top_k,
        "routes_per_expert_min": global_tokens * top_k // experts,
        "routes_per_expert_max": (global_tokens * top_k + experts - 1) // experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "top_k": top_k,
        "activation_dtype": "fp8",
        "weight_dtype": "fp4",
        "num_sms": num_sms,
        "bucket": bucket,
        "routes_prepared": True,
        "rank_gated_combine": True,
        "route_ready_dispatch": True,
        "rank_ready_route_publish": True,
        "input_route_distribution": "perfect_balance",
        "rank_interleaved_expert_mapping": balance_ranks,
        "kernel_assumes_perfect_routing": False,
        "timed_scope": baseline_record["timed_scope"],
        "route_producer_in_timed_region": baseline_record["timed_scope"] == "route_producer_and_megamoe",
        "warp_summary_selects_megamoe_only": True,
        "genphase_dispatch": bool(genphase),
        "genphase_direct_sf_gather": bool(genphase and "IndexScatter" in genphase),
        "genphase_sf_by_tma_a": bool(sf_owner_events),
        "genphase_metadata_from_producer": bool(genphase and "MetadataFromProducer" in genphase),
        "genphase_direct_expert_lists": bool(genphase and "ExpertLists" in genphase),
        "genphase_payload_from_producer": bool(genphase and producer_payload),
        "genphase_cleanup_owns_retirement": bool(genphase and "CleanupOwnsRetirement" in genphase),
        "genphase_payload_issue_only": bool(genphase and "PayloadIssued" in genphase),
        "genphase_payload_by_source": bool(genphase and "PayloadPartitioned" in genphase),
        "genphase_payload_marker": payload_event if genphase else None,
        "kernel_debug_timing": False,
        "iket_mode": "source-markers",
        "iket_profile_rank": profile_rank,
        "dispatch_detail": dispatch_detail,
        "metadata_detail": metadata_detail,
        "critical_points": os.environ["MEGAMOE_IKET_CRITICAL"] == "1",
        "profile_rank_weights": "profile-rank-weights.pt",
    },
    "interpretation": {
        "receiver_metadata_materialized": metadata_materialized,
        "first_fc1_accumulator_probe": os.environ.get("MEGAMOE_IKET_ACCUMULATOR_PROBE_SEMANTICS", "Absent unless explicitly enabled by the captured source."),
        "remainder_wait": os.environ.get("MEGAMOE_IKET_REMAINDER_SEMANTICS", "Absent unless explicitly enabled by the captured source."),
        "producer_origin": "Producer-entry metrics use the paired selector's first observed warp entry, excluding the preceding gate GEMM/quantization kernel. They include selector work and PDL overlap through MegaMoE completion. These are instrumented envelopes, not ordinary Nsight kernel durations. Metadata-start metrics begin at the first destination transfer marker, whose operation differs between descriptor allocation and broadcast. Compare common-origin timing as well as kernel-relative starts when PDL signal placement changes.",
        "genphase": "Genphase markers are inside the single MegaMoE kernel. MetadataFromProducer means the preceding selector publishes metadata; its cost must be included in full-pipeline timing. Otherwise metadata sends occur in MegaMoE. PayloadFromProducer also places payload push in the selector; otherwise payload push remains in MegaMoE. Local index/SF work stays in MegaMoE. Phase envelopes overlap and must not be added.",
        "genphase_cleanup_retirement": "CleanupOwnsRetirement identifies reuse of the existing last-CTA cleanup ticket; metadata/sender CTAs do not publish a second dispatch-done handoff and no early payload helper exists. Each CTA acquires payload at first FC1 consumption, or before retirement if it has no FC1 work. Dispatch end is the later of the raw dispatch-role end and earliest consumer payload acquisition; both original boundary and all consumer timestamps are retained. Later acquisitions include CTA scheduling delays, not extra data transfers.",
        "genphase_expert_lists": "ExpertLists identifies destination expert lists filled by producer remote slot allocation, using the existing generation banks. Local index/score scatter replaces source histogram folding; the marker itself is not a completion boundary.",
        "genphase_producer_inputs": "With MetadataFromProducer, InputsReady covers the helper and payload CTAs acquiring the selector's local row counter, followed by a generic/async proxy fence. Metadata CTAs instead acquire incoming SYS flags; no full selector-completion wait is assumed.",
        "genphase_metadata_send": "Optional InputsReady follows the input dependency. TopkStaged follows shared staging, or TopkWritten follows generic peer-store issue and a CTA join. TopkSent follows notification issue (and TMA completion in staged variants); TopkReceived also includes peer readiness acquisition. These are local milestones, not NVLink flight-time measurements.",
        "genphase_index_scatter": os.environ.get("MEGAMOE_IKET_INDEX_SCATTER_SEMANTICS", "IndexScatter to IndexReady covers index/score scatter and publication only. SF is gathered within FC1 stages, and payload readiness is acquired at FC1 consumption. Earlier variants include SF scatter in dispatch; compare whole-kernel time and compute span as well as exposed dispatch."),
        "genphase_tail_publication": "Optional TailPublishStart is immediately before row-tail publication instructions. In the seq-cst parent it follows CTA join/SYS fence/join; in the release-store candidate it follows the first join and precedes release ordering inside each store. Thus compare drain-to-retirement as well as both subintervals; a shifted fence boundary alone is not a gain. PayloadDrained samples one warp and includes waiting for other senders before publication. GenphaseEnd retains sender retirement; no marker proves remote visibility.",
        "genphase_direct_sf_startup": "With IndexScatter, the gap before first MMA can include first-stage SF gathering and activation readiness. Compare compute_start_us as well as pre_compute_dispatch_exposure_us; moving SF work out of dispatch is not necessarily a reduction in total startup latency.",
        "genphase_payload_from_producer": "PayloadFromProducer follows the helper acquiring all source payload publications, including empty valid sets. Producer warp drains, proxy fences, CTA joins and a final source SYS fence precede publication. This receiver milestone is not a source send time; include the producer in full-pipeline timing.",
        "genphase_payload_by_source": "PayloadPartitioned has the same drain/fence boundary as PayloadPushed but publishes a per-source pusher bit. FC1 acquires only the bits owning its rows. The helper still acquires every source's full mask before cleanup, including empty destinations.",
        "sf_ready_contract": os.environ.get("MEGAMOE_IKET_SF_READY_SEMANTICS", "MegaGatherSfReady follows first-stage shared SF writes and proxy fence, sampled in warp0 or split-SF warp2; it does not mean all producers completed."),
        "index_path": os.environ.get("MEGAMOE_IKET_INDEX_PATH_SEMANTICS", "MegaIndicesReady is a variant-specific index-path boundary. In receiver-materialization parents it follows the SF/index acquire and precedes HBM index loads, not their completion. Consult source and the run contract."),
        "activation_load": os.environ.get("MEGAMOE_IKET_ACTIVATION_LOAD_SEMANTICS", "MegaGatherIssued marks activation-load issue; the source and run contract distinguish indexed gather from regular TMA and specify active issuer warps and completion bytes."),
        "genphase_sf_owner": "MegaSFByTMAA identifies SF gather by the TMA-A warp: it acquires payload, prefetches into registers, acquires the shared stage, writes/fences SF, and releases activation gather through the existing rendezvous. This marker identifies ownership, not a completion or transfer-latency boundary.",
        "genphase_payload_publication": "PayloadIssued follows generic stores, writer proxy fences, CTA joins and async SYS release issue; it does not assert remote visibility. PayloadDrained follows full TMA write completion, a CTA join and async SYS release issue. PayloadPushed in older variants follows TMA write drains and a separate sender system fence. Compare receiver-dependent milestones and first MMA; these source milestones have different ordering work.",
        "payload_source_phase": os.environ.get("MEGAMOE_IKET_PAYLOAD_SOURCE_SEMANTICS", "Source-load and send-start markers delimit phases whose instruction boundaries depend on the transport. Consult source locations and the run contract; compare receiver readiness and actual MMA as well as source intervals."),
        "receiver_validation": os.environ.get("MEGAMOE_IKET_RECEIVER_VALIDATION_SEMANTICS", "Optional PayloadValidateStart marks first-tile activation validation. Separate validation warpgroups may start before SF/index phase1; only phase2 publication joins those dependencies. Consult variant ownership and marker location. Each generic vector load validates all four scalar words and materializes observed values in receiver-owned HBM; phase2 releases gather. PayloadStoresIssued precedes sender completion/retirement; GenphaseEnd remains the retirement boundary. All-four-gather instrumentation records indices, activation readiness and gather issue on warps0..3; earlier traces sampled warp0 only. Buffer capacity is guarded and reset stays outside timing."),
        "lamport_payload_readiness": "PayloadSenderDone is sender retirement, not a separate payload-ready publication. For row-tail Lamport variants, MegaPayloadReady follows acquisition of the first tile's source-row final FP8 words; bulk activation/SF writes precede those in-band completion words. MegaGatherIssued samples activation warp0. MegaGatherSfReady samples SF warp0 in shared-role variants or SF warp2 in split-role variants; neither marker covers every producer. Their order depends on the variant. Consult source marker locations and the run contract for granularity. Payload preparation and replay resets are outside MegaMoE and excluded from prepared-harness event timing.",
        "mma_issue_semantics": "MegaComputeStart and legacy first_mma metrics mark operand-barrier completion before SF UTCCP and MMA. Where available, MegaMmaIssue and mma_issue metrics mark immediately before the first fma call after SF copying and descriptor preparation. Compare matched instrumentation; do not silently treat the legacy proxy as actual issue.",
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
            "metadata": "consumer warp-0 readiness acquire, count loads and CTA join; cross-CTA envelope, not pure network transfer",
            "payload": "dispatch work after cached count setup through local payload publication; includes descriptor reads and local staging",
            "compute": "compute_start_semantics identifies actual first MMA issue or the historical operands-ready proxy; end is last L2 accumulator ready",
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
