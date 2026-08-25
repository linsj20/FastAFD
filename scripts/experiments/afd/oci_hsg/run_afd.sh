#!/usr/bin/env bash
# Run a reproducible Qwen3 AFD parallel-mapping or comprehensive-sweep point.
# Usage: ./run_afd.sh qwen3 [1k|...|128k] [input-batch/attention-DP-lane] [normalized-A:F-ratio] [attention-TP] [FFN-EP]
#
# In the original pmap contract, the ratio is normalized by full FFN trays.
# In the comprehensive contract, it is an active-GPU ratio:
# attention_gpus=ratio*FFN_EP. This is identical for EP>=4 and permits EP2 to
# share a four-GPU tray with attention workers. The batch is the total real
# input batch per attention-DP lane before the AFD microbatch split.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:afd

set -eEuo pipefail
trap 'status=$?; printf "run_afd_error line=%s status=%s\n" "$LINENO" "$status" >&2' ERR
ulimit -s 8192

preflight_error() {
    printf 'run_afd_preflight_error check=%s detail=%s\n' "$1" "$2" >&2
    exit 2
}

MODEL_KEY=${MODEL_KEY:-${1:-qwen3}}
export AFD_MODEL_PLACEMENT_REQUESTED=${FASTAFD_AFD_MODEL_PLACEMENT:-${AFD_MODEL_PLACEMENT:-legacy}}
case "$AFD_MODEL_PLACEMENT_REQUESTED" in
    legacy|fmha-only|qwen3-128k-adaptive) ;;
    *) echo "AFD_MODEL_PLACEMENT must be legacy, fmha-only, or qwen3-128k-adaptive" >&2; exit 2 ;;
esac
export AFD_MODEL_PLACEMENT=$AFD_MODEL_PLACEMENT_REQUESTED
export AFD_MODEL_PLACEMENT_POLICY=explicit
export AFD_NUM_MB=${FASTAFD_AFD_NUM_MB:-${AFD_NUM_MB:-2}}
[[ "$AFD_NUM_MB" =~ ^[1-9][0-9]*$ ]] || {
    echo "AFD_NUM_MB must be a positive integer, got $AFD_NUM_MB" >&2
    exit 2
}
SUBMIT_QOS=${FASTAFD_SUBMIT_QOS:-short}
case "$SUBMIT_QOS" in
    short|normal) ;;
    *) echo "FASTAFD_SUBMIT_QOS must be short or normal" >&2; exit 2 ;;
esac
JOB_TIME_LIMIT=${FASTAFD_JOB_TIME_LIMIT:-02:00:00}
[[ "$JOB_TIME_LIMIT" =~ ^([0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
    echo "FASTAFD_JOB_TIME_LIMIT must use [days-]HH:MM:SS" >&2
    exit 2
}
SWEEP_CONTRACT=${FASTAFD_SWEEP_CONTRACT:-pmap}
case "$SWEEP_CONTRACT" in
    pmap|comprehensive) ;;
    *) echo "FASTAFD_SWEEP_CONTRACT must be pmap or comprehensive" >&2; exit 2 ;;
esac
CONTROL_DIR=$(dirname "$(realpath "$0")")
TASK_ROOT=${FASTAFD_TASK_ROOT:-$PWD}
CUDA_EXTRACT_SCRIPT=${FASTAFD_CUDA_EXTRACT_SCRIPT:-$CONTROL_DIR/extract_cuda_wall.py}
CUDA_METRIC_PLAN=${FASTAFD_CUDA_METRIC_PLAN:-$TASK_ROOT/CASES.csv}
CUDA_SPAN_MODULE=${FASTAFD_CUDA_SPAN_MODULE:-$CONTROL_DIR/cuda_execution_span.py}
GPU_PROCESS_EXIT_SCRIPT=${FASTAFD_GPU_PROCESS_EXIT_SCRIPT:-$CONTROL_DIR/wait_gpu_processes_exit.py}
CUDA_METRICS_ROOT=${FASTAFD_CUDA_METRICS_ROOT:-$TASK_ROOT/report/metrics}
CUDA_EXTRACT_TEMP_ROOT=${FASTAFD_CUDA_EXTRACT_TEMP_ROOT:-$TASK_ROOT/cuda_extract/tmp}
CONTEXT_SPEC=${CONTEXT_SPEC:-${CONTEXT:-${2:-8k}}}
BATCH_ARG=${BATCH:-${3:-}}
RATIO_ARG=${AFD_ATTENTION_TO_FFN_TRAY_RATIO:-${NORMALIZED_AF_RATIO:-${4:-}}}
ATTENTION_TP_ARG=${AFD_ATTN_TP_SIZE:-${ATTENTION_TP:-${5:-1}}}
FFN_EP_ARG=${AFD_MLP_EP_SIZE:-${FFN_EP:-${6:-4}}}
PROMPT_MODE=${PROMPT_MODE:-uniform}
NSYS_CUDA_GRAPH_TRACE=${NSYS_CUDA_GRAPH_TRACE:-node}
case "$NSYS_CUDA_GRAPH_TRACE" in
    node|none) ;;
    *) echo "NSYS_CUDA_GRAPH_TRACE must be node or none" >&2; exit 2 ;;
esac
NSYS_CAPTURE_DECODE_STEPS=${NSYS_CAPTURE_DECODE_STEPS:-15}
[[ "$NSYS_CAPTURE_DECODE_STEPS" == 15 ]] || {
    echo "NSYS_CAPTURE_DECODE_STEPS must be exactly 15" >&2
    exit 2
}
OUTPUT_TOKENS=17
TRACE_WARMUP_DECODE_STEPS=1
AFD_MEMORY_RATIO=${FASTAFD_AFD_MEMORY_RATIO:-0.82}
[[ "$AFD_MEMORY_RATIO" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]] || {
    echo "FASTAFD_AFD_MEMORY_RATIO must be greater than zero and at most one" >&2
    exit 2
}
AFD_NUM_PAGES=${FASTAFD_AFD_NUM_PAGES:-}
[[ -z "$AFD_NUM_PAGES" || "$AFD_NUM_PAGES" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_AFD_NUM_PAGES must be empty or a positive integer" >&2
    exit 2
}
# The inner capture runner may wait up to 600 seconds to make sample.json
# durable before entering Ray teardown. Keep the capture-only watchdog longer
# than that publication window. Once capture, sample, and the coordinator trace
# are all durable, terminate the allocation-scoped driver immediately because
# no metric input can improve by retaining an allocation after that point.
POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS=${FASTAFD_POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS:-900}
[[ "$POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS must be positive" >&2
    exit 2
}
POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS=${FASTAFD_POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS:-10}
[[ "$POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS must be positive" >&2
    exit 2
}
CAPTURE_WORKER_STOP_TIMEOUT_S=60
case "$CONTEXT_SPEC" in
    1k|1024) CONTEXT_MIN=1024; CONTEXT_MAX=1024 ;;
    2k|2048) CONTEXT_MIN=2048; CONTEXT_MAX=2048 ;;
    4k|4096) CONTEXT_MIN=4096; CONTEXT_MAX=4096 ;;
    8k|8192) CONTEXT_MIN=8192; CONTEXT_MAX=8192 ;;
    16k|16384) CONTEXT_MIN=16384; CONTEXT_MAX=16384 ;;
    32k|32768) CONTEXT_MIN=32768; CONTEXT_MAX=32768 ;;
    64k|65536) CONTEXT_MIN=65536; CONTEXT_MAX=65536 ;;
    128k|131072) CONTEXT_MIN=131072; CONTEXT_MAX=131072 ;;
    1k-4k|1024-4096) CONTEXT_MIN=1024; CONTEXT_MAX=4096 ;;
    4k-8k|4096-8192) CONTEXT_MIN=4096; CONTEXT_MAX=8192 ;;
    8k-32k|8192-32768) CONTEXT_MIN=8192; CONTEXT_MAX=32768 ;;
    32k-128k|32768-131072) CONTEXT_MIN=32768; CONTEXT_MAX=131072 ;;
    1k-16k|1024-16384) CONTEXT_MIN=1024; CONTEXT_MAX=16384 ;;
    4k-64k|4096-65536) CONTEXT_MIN=4096; CONTEXT_MAX=65536 ;;
    8k-128k|8192-131072) CONTEXT_MIN=8192; CONTEXT_MAX=131072 ;;
    1k-128k|1024-131072) CONTEXT_MIN=1024; CONTEXT_MAX=131072 ;;
    *) echo "context must be a supported uniform ISL or irregular ISL range" >&2; exit 2 ;;
esac
CONTEXT=$CONTEXT_MAX
case "$PROMPT_MODE" in
    uniform)
        (( CONTEXT_MIN == CONTEXT_MAX )) || {
            echo "uniform mode requires a single ISL, not a range" >&2
            exit 2
        }
        IRREGULAR=0 ;;
    irregular)
        (( CONTEXT_MIN < CONTEXT_MAX )) || {
            echo "irregular mode requires one of the supported ISL ranges" >&2
            exit 2
        }
        IRREGULAR=1 ;;
    *) echo "prompt mode must be uniform or irregular" >&2; exit 2 ;;
esac
if (( IRREGULAR )); then
    SHAPE_ID=r${CONTEXT_MIN}-${CONTEXT_MAX}
else
    SHAPE_ID=$CONTEXT
fi

ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
RESULTS_ROOT=${FASTAFD_RESULTS_ROOT:-$ROOT}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$HOME/scratch/github/FastAFD}
EXPECTED_HEAD=${FASTAFD_EXPECTED_HEAD:-$(git -C "$SOURCE_REPO" rev-parse HEAD)}
EXPECTED_SOURCE_MANIFEST=${FASTAFD_EXPECTED_SOURCE_MANIFEST:-}
ALLOW_DIRTY_SOURCE=${FASTAFD_ALLOW_DIRTY_SOURCE:-0}
[[ "$ALLOW_DIRTY_SOURCE" == 0 || "$ALLOW_DIRTY_SOURCE" == 1 ]]
SOURCE_COORDINATOR_REL=python/minisgl/afd_coordinator.py
SOURCE_PROFILER_REL=python/minisgl/afd_profiler.py
SOURCE_PROTOCOL_REL=python/minisgl/afd_protocol.py
SOURCE_PLACEMENT_REL=python/minisgl/afd_support.py
SOURCE_WORKER_BASE_REL=python/minisgl/afd_worker_base.py
SOURCE_MEGAMOE_ADAPTER_REL=python/minisgl/moe/megamoe_m2n_afd.py
SOURCE_API_REL=python/minisgl/server/api_server.py
SOURCE_ARGS_REL=python/minisgl/server/args.py
EXPECTED_SOURCE_COORDINATOR_SHA256=${FASTAFD_EXPECTED_SOURCE_COORDINATOR_SHA256:-${EXPECTED_SOURCE_COORDINATOR_SHA256:-}}
EXPECTED_SOURCE_PROFILER_SHA256=${FASTAFD_EXPECTED_SOURCE_PROFILER_SHA256:-${EXPECTED_SOURCE_PROFILER_SHA256:-}}
EXPECTED_SOURCE_PROTOCOL_SHA256=${FASTAFD_EXPECTED_SOURCE_PROTOCOL_SHA256:-${EXPECTED_SOURCE_PROTOCOL_SHA256:-}}
EXPECTED_SOURCE_PLACEMENT_SHA256=${FASTAFD_EXPECTED_SOURCE_PLACEMENT_SHA256:-${EXPECTED_SOURCE_PLACEMENT_SHA256:-}}
EXPECTED_SOURCE_WORKER_BASE_SHA256=${FASTAFD_EXPECTED_SOURCE_WORKER_BASE_SHA256:-${EXPECTED_SOURCE_WORKER_BASE_SHA256:-}}
EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256=${FASTAFD_EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256:-${EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256:-}}
EXPECTED_SOURCE_API_SHA256=${FASTAFD_EXPECTED_SOURCE_API_SHA256:-${EXPECTED_SOURCE_API_SHA256:-}}
EXPECTED_SOURCE_ARGS_SHA256=${FASTAFD_EXPECTED_SOURCE_ARGS_SHA256:-${EXPECTED_SOURCE_ARGS_SHA256:-}}
IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-${EXPECTED_HEAD:0:8}-cuda130-vllm-ep}
HF_CACHE=$ROOT/models/huggingface/hub

validate_source_repo() {
    SOURCE_VALIDATION_DETAIL=unknown
    local actual_head
    actual_head=$(git -C "$SOURCE_REPO" rev-parse HEAD 2>&1) || {
        SOURCE_VALIDATION_DETAIL="git_head repo=$SOURCE_REPO output=$actual_head"
        return 1
    }
    [[ "$actual_head" == "$EXPECTED_HEAD" ]] || {
        SOURCE_VALIDATION_DETAIL="head_mismatch expected=$EXPECTED_HEAD observed=$actual_head"
        return 1
    }
    if [[ "$ALLOW_DIRTY_SOURCE" == 1 ]]; then
        [[ -z "$EXPECTED_SOURCE_MANIFEST" ]] || {
            SOURCE_VALIDATION_DETAIL="dirty_source_with_manifest"
            return 1
        }
        SOURCE_VALIDATION_DETAIL=ok
        return 0
    fi
    if [[ -n "$EXPECTED_SOURCE_MANIFEST" ]]; then
        [[ -f "$EXPECTED_SOURCE_MANIFEST" ]] || {
            SOURCE_VALIDATION_DETAIL="missing_manifest path=$EXPECTED_SOURCE_MANIFEST"
            return 1
        }
        awk '
            NF != 2 || $1 !~ /^[0-9a-f]{64}$/ || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }
        ' "$EXPECTED_SOURCE_MANIFEST" || {
            SOURCE_VALIDATION_DETAIL="invalid_manifest path=$EXPECTED_SOURCE_MANIFEST"
            return 1
        }
        local status_output tracked_status_paths manifest_paths sha_output
        if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
            status_output=$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=no 2>&1) || {
                SOURCE_VALIDATION_DETAIL="git_status repo=$SOURCE_REPO output=$status_output"
                return 1
            }
            tracked_status_paths=$(printf '%s\n' "$status_output" | sed 's/^...//' | LC_ALL=C sort)
            manifest_paths=$(awk '{print $2}' "$EXPECTED_SOURCE_MANIFEST" | LC_ALL=C sort) || {
                SOURCE_VALIDATION_DETAIL="manifest_paths path=$EXPECTED_SOURCE_MANIFEST"
                return 1
            }
            [[ -n "$manifest_paths" && "$tracked_status_paths" == "$manifest_paths" ]] || {
                SOURCE_VALIDATION_DETAIL="status_manifest_mismatch tracked_count=$(printf '%s\n' "$tracked_status_paths" | sed '/^$/d' | wc -l) manifest_count=$(printf '%s\n' "$manifest_paths" | sed '/^$/d' | wc -l)"
                return 1
            }
        fi
        sha_output=$(cd "$SOURCE_REPO" && sha256sum --check --strict "$EXPECTED_SOURCE_MANIFEST" 2>&1) || {
            SOURCE_VALIDATION_DETAIL="sha256_check $(printf '%s\n' "$sha_output" | tail -1)"
            return 1
        }
        SOURCE_VALIDATION_DETAIL=ok
        return 0
    fi
    local status expected_status
    status=$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)
    expected_status=""
    if [[ -n "$EXPECTED_SOURCE_COORDINATOR_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_COORDINATOR_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_COORDINATOR_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_COORDINATOR_SHA256" ]]
        expected_status=" M $SOURCE_COORDINATOR_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_PROFILER_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_PROFILER_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_PROFILER_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_PROFILER_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_PROFILER_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_PROTOCOL_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_PROTOCOL_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_PROTOCOL_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_PROTOCOL_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_PROTOCOL_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_PLACEMENT_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_PLACEMENT_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_PLACEMENT_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_PLACEMENT_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_PLACEMENT_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_WORKER_BASE_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_WORKER_BASE_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_WORKER_BASE_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_WORKER_BASE_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_WORKER_BASE_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_MEGAMOE_ADAPTER_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_MEGAMOE_ADAPTER_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_MEGAMOE_ADAPTER_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_API_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_API_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_API_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_API_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_API_REL"
    fi
    if [[ -n "$EXPECTED_SOURCE_ARGS_SHA256" ]]; then
        [[ "$EXPECTED_SOURCE_ARGS_SHA256" =~ ^[0-9a-f]{64}$ ]]
        [[ "$(sha256sum "$SOURCE_REPO/$SOURCE_ARGS_REL" | awk '{print $1}')" == "$EXPECTED_SOURCE_ARGS_SHA256" ]]
        expected_status+="${expected_status:+$'\n'} M $SOURCE_ARGS_REL"
    fi
    [[ "$status" == "$expected_status" ]] || {
        SOURCE_VALIDATION_DETAIL=legacy_status_mismatch
        return 1
    }
    SOURCE_VALIDATION_DETAIL=ok
}

DEFAULT_BATCH=""
DEFAULT_NODES=""
case "$MODEL_KEY" in
    qwen3)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        MODEL_SOURCE_CONFIG_SHA256=702c46d431bb984db9035a1225186bbfdb52c0d19c82104df4a37cd005e0369e
        PRESET=scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SOURCE_CONTEXT=8192
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        MAX_SEQ_LEN=$((CONTEXT_MAX + 64))
        MODEL_PROFILE_ID=native
        ROPE_SCALING_FACTOR=0
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=32768
        case "$CONTEXT" in
            65536)
                MODEL_PROFILE_ID=qwen3-235b-a22b-fp8-39eb2b06-yarn-f2p001953125-original32768-max65600
                ROPE_SCALING_FACTOR=2.001953125 ;;
            131072)
                MODEL_PROFILE_ID=qwen3-235b-a22b-fp8-39eb2b06-yarn-f4p001953125-original32768-max131136
                ROPE_SCALING_FACTOR=4.001953125 ;;
        esac
        case "$CONTEXT" in
            8192) DEFAULT_BATCH=96; DEFAULT_NODES=8 ;;
            16384) DEFAULT_BATCH=48; DEFAULT_NODES=12 ;;
            *) DEFAULT_NODES=8 ;;
        esac ;;
    minimax)
        case "$CONTEXT" in
        8192)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        DEFAULT_NODES=18; DEFAULT_BATCH=72; MAX_SEQ_LEN=8320 ;;
        16384)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_16k_b36_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SHA256=918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca
        DEFAULT_NODES=18; DEFAULT_BATCH=36; MAX_SEQ_LEN=16640 ;;
        *) echo "MiniMax currently supports only the pinned 8K and 16K prompt contracts" >&2; exit 2 ;;
        esac
        PROMPT_SOURCE_CONTEXT=$CONTEXT
        MODEL_SOURCE_CONFIG_SHA256=""
        MODEL_PROFILE_ID=native
        ROPE_SCALING_FACTOR=0
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=0 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac
[[ "$MODEL_KEY" == qwen3 ]] || {
    echo "the parallel-mapping sweep supports only qwen3" >&2
    exit 2
}
if (( IRREGULAR )); then
    CURRENT_BEST_RATIO=7
else
    case "$CONTEXT" in
        1024) CURRENT_BEST_RATIO=1 ;;
        2048) CURRENT_BEST_RATIO=1 ;;
        4096) CURRENT_BEST_RATIO=2 ;;
        8192) CURRENT_BEST_RATIO=3 ;;
        16384) CURRENT_BEST_RATIO=5 ;;
        32768) CURRENT_BEST_RATIO=6 ;;
        65536) CURRENT_BEST_RATIO=7 ;;
        131072) CURRENT_BEST_RATIO=15 ;;
        *) echo "unsupported context" >&2; exit 2 ;;
    esac
fi
PRESET=${FASTAFD_AFD_PRESET:-$(dirname "$(realpath "$0")")/run_afd_qwen3_parallel.sh}
PRESET=$(realpath "$PRESET")
RATIO_ARG=${RATIO_ARG:-$CURRENT_BEST_RATIO}
[[ "$RATIO_ARG" =~ ^([1-9][0-9]*)(:1)?$ ]] || {
    echo "normalized A:F tray ratio must be a positive integer or N:1" >&2
    exit 2
}
NORMALIZED_AF_RATIO=${BASH_REMATCH[1]}
ATTENTION_TP=$ATTENTION_TP_ARG
FFN_EP=$FFN_EP_ARG
case "$ATTENTION_TP" in 1|2|4|8) ;; *) echo "attention TP must be 1, 2, 4, or 8" >&2; exit 2 ;; esac
if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
    [[ "$ATTENTION_TP" == 1 ]] || {
        echo "the comprehensive AFD sweep fixes attention TP at 1" >&2
        exit 2
    }
    case "$FFN_EP" in
        2|4|8|16|32) ;;
        *) echo "comprehensive FFN EP must be 2, 4, 8, 16, or 32" >&2; exit 2 ;;
    esac
else
    case "$FFN_EP" in
        4|8|16) ;;
        *) echo "pmap FFN EP must be 4, 8, or 16" >&2; exit 2 ;;
    esac
fi
# These are the minimum per-attention-DP-lane capacities measured by the
# current TP1 results and the topology pilots. Qwen3 has four KV heads, so KV
# sharding increases capacity through TP4 and then saturates at TP8.
case "$ATTENTION_TP" in
    1) AFD_KV_CAPACITY_TOKENS=839424 ;;
    2) AFD_KV_CAPACITY_TOKENS=1674560 ;;
    4|8) AFD_KV_CAPACITY_TOKENS=3349120 ;;
esac
BASELINE_AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS
AFD_KV_CAPACITY_TOKENS=${FASTAFD_AFD_KV_CAPACITY_TOKENS:-$AFD_KV_CAPACITY_TOKENS}
[[ "$AFD_KV_CAPACITY_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_AFD_KV_CAPACITY_TOKENS must be a positive integer" >&2
    exit 2
}
(( AFD_KV_CAPACITY_TOKENS % 64 == 0 )) || {
    echo "FASTAFD_AFD_KV_CAPACITY_TOKENS must be page-aligned to 64 tokens" >&2
    exit 2
}
if [[ "$AFD_MEMORY_RATIO" != 0.82 || -n "${FASTAFD_AFD_KV_CAPACITY_TOKENS:-}" ]]; then
    [[ -n "${FASTAFD_AFD_MEMORY_RATIO:-}" && -n "${FASTAFD_AFD_KV_CAPACITY_TOKENS:-}" ]] || {
        echo "tuned memory requires both FASTAFD_AFD_MEMORY_RATIO and FASTAFD_AFD_KV_CAPACITY_TOKENS" >&2
        exit 2
    }
    (( AFD_KV_CAPACITY_TOKENS >= BASELINE_AFD_KV_CAPACITY_TOKENS )) || {
        echo "tuned KV capacity cannot be below the measured 0.82 baseline" >&2
        exit 2
    }
fi
if [[ -n "$AFD_NUM_PAGES" ]]; then
    (( AFD_KV_CAPACITY_TOKENS == AFD_NUM_PAGES * 64 )) || {
        echo "FASTAFD_AFD_KV_CAPACITY_TOKENS must equal FASTAFD_AFD_NUM_PAGES * 64" >&2
        exit 2
    }
fi
NOMINAL_AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS
RESIDENT_PROMPT_TOKENS=$(( ((CONTEXT + 63) / 64) * 64 ))
FRESH_REQUEST_ESTIMATED_TOKENS=$(( CONTEXT + OUTPUT_TOKENS ))
SCHEDULER_RUNNING_RESERVATION_TOKENS=$(( OUTPUT_TOKENS - 1 + 63 ))
CAPACITY_BATCH=$((
    (AFD_KV_CAPACITY_TOKENS - FRESH_REQUEST_ESTIMATED_TOKENS) /
    (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) + 1
))
BATCH=${BATCH_ARG:-$CAPACITY_BATCH}
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || {
    echo "batch/attention-DP-lane must be a positive integer" >&2
    exit 2
}
(( AFD_NUM_MB <= BATCH )) || {
    echo "AFD_NUM_MB must not exceed batch/attention-DP-lane: $AFD_NUM_MB > $BATCH" >&2
    exit 2
}
if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
    ATTENTION_GPUS=$(( NORMALIZED_AF_RATIO * FFN_EP ))
    FFN_GPUS=$FFN_EP
    ACTIVE_GPUS=$(( ATTENTION_GPUS + FFN_GPUS ))
    NODES=$(( (ACTIVE_GPUS + 3) / 4 ))
    ALLOCATED_GPUS=$(( NODES * 4 ))
    ATTENTION_TRAYS=$(( (ATTENTION_GPUS + 3) / 4 ))
    FFN_TRAYS=$(( (FFN_GPUS + 3) / 4 ))
    SHARED_ROLE_TRAYS=$(( ATTENTION_TRAYS + FFN_TRAYS - NODES ))
else
    FFN_TRAYS=$(( FFN_EP / 4 ))
    ATTENTION_TRAYS=$(( NORMALIZED_AF_RATIO * FFN_TRAYS ))
    NODES=$(( ATTENTION_TRAYS + FFN_TRAYS ))
    ATTENTION_GPUS=$(( ATTENTION_TRAYS * 4 ))
    FFN_GPUS=$FFN_EP
    ACTIVE_GPUS=$(( ATTENTION_GPUS + FFN_GPUS ))
    ALLOCATED_GPUS=$(( NODES * 4 ))
    SHARED_ROLE_TRAYS=0
fi
if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
    MAX_TOTAL_TRAYS=${FASTAFD_MAX_TOTAL_TRAYS:-18}
else
    MAX_TOTAL_TRAYS=${FASTAFD_MAX_TOTAL_TRAYS:-17}
fi
[[ "$MAX_TOTAL_TRAYS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_MAX_TOTAL_TRAYS must be a positive integer" >&2
    exit 2
}
(( NODES <= MAX_TOTAL_TRAYS )) || {
    echo "topology requires $NODES trays, exceeding limit $MAX_TOTAL_TRAYS" >&2
    exit 2
}
if [[ "$SWEEP_CONTRACT" != comprehensive ]]; then
    RATIO_MIN=$CURRENT_BEST_RATIO
    RATIO_MAX=$(( CURRENT_BEST_RATIO * FFN_TRAYS ))
    (( NORMALIZED_AF_RATIO >= RATIO_MIN && NORMALIZED_AF_RATIO <= RATIO_MAX )) || {
        echo "ratio $NORMALIZED_AF_RATIO:1 is outside ${RATIO_MIN}:1-${RATIO_MAX}:1 for ISL $CONTEXT and EP $FFN_EP under $SWEEP_CONTRACT" >&2
        exit 2
    }
fi
(( ATTENTION_GPUS % ATTENTION_TP == 0 )) || {
    echo "$ATTENTION_GPUS attention GPUs cannot be divided into TP$ATTENTION_TP groups" >&2
    exit 2
}
ATTENTION_DP_SIZE=$(( ATTENTION_GPUS / ATTENTION_TP ))
ATTENTION_WORKERS=$ATTENTION_GPUS
MLP_DP_SIZE=$FFN_EP
MLP_TP_SIZE=1
if [[ "$AFD_MODEL_PLACEMENT_REQUESTED" == qwen3-128k-adaptive ]]; then
    ADAPTIVE_CONTRACT="$MODEL_KEY:$CONTEXT:$BATCH:$ATTENTION_TP:$FFN_EP:$AFD_NUM_MB"
    [[ "$ADAPTIVE_CONTRACT" == qwen3:131072:6:1:4:2 ]] || {
        echo "qwen3-128k-adaptive requires qwen3, 128K, batch 6, ATP1, FEP4, and MB2; got $ADAPTIVE_CONTRACT" >&2
        exit 2
    }
    if (( NORMALIZED_AF_RATIO >= 8 )); then
        AFD_MODEL_PLACEMENT=legacy
    else
        AFD_MODEL_PLACEMENT=fmha-only
    fi
    export AFD_MODEL_PLACEMENT
    export AFD_MODEL_PLACEMENT_POLICY=qwen3-128k-b6-atp1-fep4-mb2-r8-crossover
    printf "afd_model_placement requested=%s resolved=%s policy=%s ratio=%s:1\n" \
        "$AFD_MODEL_PLACEMENT_REQUESTED" "$AFD_MODEL_PLACEMENT" \
        "$AFD_MODEL_PLACEMENT_POLICY" "$NORMALIZED_AF_RATIO"
fi
if (( IRREGULAR )) && [[ "$MODEL_KEY" != qwen3 ]]; then
    echo "irregular prompt-length ranges are supported only for Qwen3" >&2
    exit 2
fi
if (( IRREGULAR )) && [[ "$SWEEP_CONTRACT" != comprehensive ]] && (( NODES != 8 )); then
    echo "the legacy Qwen3 irregular sweep fixes AFD at eight nodes" >&2
    exit 2
fi
# Construct a symmetric integer linspace. The lower half is rounded to nearest
# token and the upper half is its exact complement, preserving both endpoints
# and an exact (min + max) / 2 mean for every batch size.
PROMPT_LENGTHS=()
PROMPT_LENGTH_SUM=0
AFD_REQUIRED_KV_TOKENS=0
for ((index=0; index<BATCH; index++)); do
    if (( BATCH == 1 )); then
        prompt_length=$CONTEXT_MAX
    elif (( 2 * index <= BATCH - 1 )); then
        prompt_length=$(( CONTEXT_MIN + (index * (CONTEXT_MAX - CONTEXT_MIN) + (BATCH - 1) / 2) / (BATCH - 1) ))
    else
        prompt_length=$(( CONTEXT_MAX - ((BATCH - 1 - index) * (CONTEXT_MAX - CONTEXT_MIN) + (BATCH - 1) / 2) / (BATCH - 1) ))
    fi
    PROMPT_LENGTHS+=("$prompt_length")
    PROMPT_LENGTH_SUM=$(( PROMPT_LENGTH_SUM + prompt_length ))
    AFD_REQUIRED_KV_TOKENS=$(( AFD_REQUIRED_KV_TOKENS + ((prompt_length + 64 + 63) / 64) * 64 ))
done
PROMPT_LENGTHS_CSV=$(IFS=,; echo "${PROMPT_LENGTHS[*]}")
if (( AFD_REQUIRED_KV_TOKENS > AFD_KV_CAPACITY_TOKENS )); then
    echo "batch requires $AFD_REQUIRED_KV_TOKENS AFD KV tokens/attention-DP lane, capacity is $AFD_KV_CAPACITY_TOKENS" >&2
    exit 2
fi
CAPACITY_MAX_BATCH=0
RAW_KV_MAX_BATCH=0
SCHEDULER_ADMISSION_REQUIRED_TOKENS=0
SCHEDULER_ADMISSION_UNUSED_TOKENS=0
REQUIRE_CAPACITY_MAX=${FASTAFD_REQUIRE_CAPACITY_MAX:-0}
ALLOW_OBSERVED_CAPACITY_PROBE=${FASTAFD_ALLOW_OBSERVED_CAPACITY_PROBE:-0}
[[ "$REQUIRE_CAPACITY_MAX" == 0 || "$REQUIRE_CAPACITY_MAX" == 1 ]]
[[ "$ALLOW_OBSERVED_CAPACITY_PROBE" == 0 || "$ALLOW_OBSERVED_CAPACITY_PROBE" == 1 ]]
if (( ! IRREGULAR )); then
    TOKENS_PER_SEQUENCE=$(( ((CONTEXT + 64 + 63) / 64) * 64 ))
    RAW_KV_MAX_BATCH=$(( AFD_KV_CAPACITY_TOKENS / TOKENS_PER_SEQUENCE ))
    CAPACITY_MAX_BATCH=$((
        (AFD_KV_CAPACITY_TOKENS - FRESH_REQUEST_ESTIMATED_TOKENS) /
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) + 1
    ))
    SCHEDULER_ADMISSION_REQUIRED_TOKENS=$((
        (CAPACITY_MAX_BATCH - 1) *
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) +
        FRESH_REQUEST_ESTIMATED_TOKENS
    ))
    SCHEDULER_ADMISSION_UNUSED_TOKENS=$((
        AFD_KV_CAPACITY_TOKENS - SCHEDULER_ADMISSION_REQUIRED_TOKENS
    ))
    (( CAPACITY_MAX_BATCH > 0 && CAPACITY_MAX_BATCH <= RAW_KV_MAX_BATCH ))
    (( SCHEDULER_ADMISSION_REQUIRED_TOKENS <= AFD_KV_CAPACITY_TOKENS ))
    ((
        CAPACITY_MAX_BATCH *
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) +
        FRESH_REQUEST_ESTIMATED_TOKENS > AFD_KV_CAPACITY_TOKENS
    ))
    NOMINAL_CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH
    NOMINAL_RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH
    if (( REQUIRE_CAPACITY_MAX )); then
        if (( BATCH > CAPACITY_MAX_BATCH )); then
            echo "capacity-max sweep cannot exceed nominal batch $CAPACITY_MAX_BATCH at ISL $CONTEXT (raw KV ceiling $RAW_KV_MAX_BATCH), got $BATCH" >&2
            exit 2
        fi
        if (( BATCH < CAPACITY_MAX_BATCH && ! ALLOW_OBSERVED_CAPACITY_PROBE )); then
            echo "a below-nominal observed-capacity probe requires FASTAFD_ALLOW_OBSERVED_CAPACITY_PROBE=1: nominal batch $CAPACITY_MAX_BATCH at ISL $CONTEXT, got $BATCH" >&2
            exit 2
        fi
    fi
fi
if [[ "$SWEEP_CONTRACT" == comprehensive ]] && (( ! IRREGULAR )); then
    (( BATCH <= CAPACITY_MAX_BATCH )) || {
        echo "comprehensive batch $BATCH exceeds capacity $CAPACITY_MAX_BATCH" >&2
        exit 2
    }
fi
REFERENCE_TPS=0
if [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == minimax:8192:72:18 ]]; then
    REFERENCE_TPS=2198
elif [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == minimax:16384:36:18 ]]; then
    REFERENCE_TPS=1006
fi
MODEL_SOURCE_PATH=${FASTAFD_MODEL_PATH:-$HF_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION}
MODEL_PATH=$MODEL_SOURCE_PATH
MODEL_PROFILE_MANIFEST=""
if [[ "$MODEL_PROFILE_ID" != native ]]; then
    MODEL_PATH=$ROOT/model-profiles/$MODEL_PROFILE_ID
    MODEL_PROFILE_MANIFEST=$MODEL_PATH/fastafd-model-profile.json
fi
MODEL_PROFILE_SCRIPT=${FASTAFD_MODEL_PROFILE_SCRIPT:-$(dirname "$(realpath "$0")")/prepare_model_profile.py}
PROMPT_BASE=$SOURCE_REPO/prompts/prompts_512x${PROMPT_SOURCE_CONTEXT}_seed20260527.txt
NUM_PROMPTS=$(( ATTENTION_DP_SIZE * BATCH ))
MICROBATCH_UPPER=$(( (BATCH + AFD_NUM_MB - 1) / AFD_NUM_MB ))
MICROBATCH_BASE=$(( BATCH / AFD_NUM_MB ))
MICROBATCH_REMAINDER=$(( BATCH % AFD_NUM_MB ))
MICROBATCH_REAL_SIZES=""
MICROBATCH_SEPARATOR=""
for ((mb_index=0; mb_index<AFD_NUM_MB; mb_index++)); do
    if [[ "$AFD_MODEL_PLACEMENT" == fmha-only ]]; then
        mb_real_size=$(( MICROBATCH_BASE + (mb_index >= AFD_NUM_MB - MICROBATCH_REMAINDER ? 1 : 0) ))
    else
        mb_real_size=$(( MICROBATCH_BASE + (mb_index < MICROBATCH_REMAINDER ? 1 : 0) ))
    fi
    MICROBATCH_REAL_SIZES+="${MICROBATCH_SEPARATOR}${mb_real_size}"
    MICROBATCH_SEPARATOR=+
done
GRAPH_BATCH=$MICROBATCH_UPPER
PADDED_BATCH=$(( GRAPH_BATCH * AFD_NUM_MB ))
CASE_ORDINAL=${FASTAFD_CASE_ORDINAL:-single}
[[ "$CASE_ORDINAL" == single || "$CASE_ORDINAL" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_CASE_ORDINAL must be single or a positive integer" >&2
    exit 2
}
CASE_PORT_SLOT=${FASTAFD_CASE_PORT_SLOT:-$CASE_ORDINAL}
[[ "$CASE_PORT_SLOT" == single || "$CASE_PORT_SLOT" =~ ^[0-9]+$ ]] || {
    echo "FASTAFD_CASE_PORT_SLOT must be single or a non-negative integer" >&2
    exit 2
}

if [[ "${FASTAFD_DRY_RUN:-0}" == 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'mode=afd sweep_contract=%s submit_qos=%s model=%s prompt_mode=%s context_min=%s context_max=%s irregular=%s input_batch_per_attention_dp_lane=%s capacity_max_batch=%s raw_kv_max_batch=%s require_capacity_max=%s microbatch_real_sizes=%s nodes=%s allocated_gpus=%s active_gpus=%s shared_role_trays=%s normalized_af_ratio=%s:1 attention_trays=%s ffn_trays=%s attention_tp=%s ffn_ep=%s attention_dp=%s attention_gpus=%s ffn_gpus=%s prompts=%s prompt_lengths=%s prompt_length_sum=%s required_kv_tokens_per_attention_dp_lane=%s known_kv_capacity_tokens_per_attention_rank=%s afd_memory_ratio=%s afd_num_pages_override=%s max_seq_len=%s graph_batch_per_mb=%s graph_padded_input_batch=%s nsys_cuda_graph_trace=%s nsys_target_batch_per_attention_dp=%s nsys_capture_decode_steps=%s capture_policy=%s prompt_source=%s model_profile=%s rope_factor=%s\n' \
        "$SWEEP_CONTRACT" "$SUBMIT_QOS" "$MODEL_KEY" "$PROMPT_MODE" "$CONTEXT_MIN" "$CONTEXT_MAX" "$IRREGULAR" \
        "$BATCH" "$CAPACITY_MAX_BATCH" "$RAW_KV_MAX_BATCH" "$REQUIRE_CAPACITY_MAX" \
        "$MICROBATCH_REAL_SIZES" "$NODES" "$ALLOCATED_GPUS" \
        "$ACTIVE_GPUS" "$SHARED_ROLE_TRAYS" "$NORMALIZED_AF_RATIO" \
        "$ATTENTION_TRAYS" "$FFN_TRAYS" "$ATTENTION_TP" "$FFN_EP" "$ATTENTION_DP_SIZE" \
        "$ATTENTION_GPUS" "$FFN_GPUS" \
        "$NUM_PROMPTS" "$PROMPT_LENGTHS_CSV" "$PROMPT_LENGTH_SUM" \
        "$AFD_REQUIRED_KV_TOKENS" "$AFD_KV_CAPACITY_TOKENS" \
        "$AFD_MEMORY_RATIO" "${AFD_NUM_PAGES:-none}" \
        "$MAX_SEQ_LEN" "$GRAPH_BATCH" "$PADDED_BATCH" "$NSYS_CUDA_GRAPH_TRACE" \
        "$BATCH" "$NSYS_CAPTURE_DECODE_STEPS" trace_first_exact_then_measure_next15 \
        "$PROMPT_BASE" "$MODEL_PROFILE_ID" "$ROPE_SCALING_FACTOR"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]] || \
        preflight_error login_host "host=$(hostname -s)"
    validate_source_repo || \
        preflight_error source_repo "$SOURCE_VALIDATION_DETAIL"
    [[ -x "$EP_VENV_DIR/bin/python" ]] || preflight_error venv_python "$EP_VENV_DIR/bin/python"
    [[ -d "$MODEL_SOURCE_PATH" ]] || preflight_error model_source "$MODEL_SOURCE_PATH"
    [[ -f "$PROMPT_BASE" ]] || preflight_error prompt_source "$PROMPT_BASE"
    if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
        [[ -f "$CUDA_EXTRACT_SCRIPT" && -f "$CUDA_METRIC_PLAN" ]]
        [[ -f "$CUDA_SPAN_MODULE" && -d "$CUDA_EXTRACT_TEMP_ROOT" ]]
        mkdir -p "$CUDA_METRICS_ROOT"
    fi
    if [[ "$MODEL_PROFILE_ID" != native ]]; then
        "$EP_VENV_DIR/bin/python" "$MODEL_PROFILE_SCRIPT" \
            --source "$MODEL_SOURCE_PATH" --output "$MODEL_PATH" \
            --profile-id "$MODEL_PROFILE_ID" --source-revision "$MODEL_REVISION" \
            --expected-source-config-sha256 "$MODEL_SOURCE_CONFIG_SHA256" \
            --max-position-embeddings "$MAX_SEQ_LEN" \
            --rope-factor "$ROPE_SCALING_FACTOR" \
            --original-max-position-embeddings "$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS"
    fi
    MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
    [[ -f "$MODEL_PATH/model.safetensors.index.json" ]]
    [[ $(sha256sum "$PROMPT_BASE" | awk '{print $1}') == "$PROMPT_SHA256" ]]
    if [[ "${FASTAFD_VALIDATE_ONLY:-0}" == 1 ]]; then
        printf 'validated model=%s context=%s batch=%s normalized_ratio=%s:1 attention_tp=%s ffn_ep=%s nodes=%s source=%s preset=%s\n' \
            "$MODEL_KEY" "$CONTEXT" "$BATCH" "$NORMALIZED_AF_RATIO" \
            "$ATTENTION_TP" "$FFN_EP" "$NODES" "$SOURCE_REPO" "$PRESET"
        exit 0
    fi
    if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
        JOB_PREFIX=${FASTAFD_JOB_PREFIX:-fastafd:afd3d}
        JOB_NAME="${JOB_PREFIX}-${CONTEXT}-fep${FFN_EP}-r${NORMALIZED_AF_RATIO}-b${BATCH}"
    else
        JOB_NAME="fastafd:pmap-${CONTEXT}-r${NORMALIZED_AF_RATIO}-atp${ATTENTION_TP}-fep${FFN_EP}"
    fi
    ! squeue -h -u "$USER" -o '%j' | grep -qx "$JOB_NAME" || {
        echo "the requested AFD job is already active" >&2; exit 1;
    }
    if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
        ACTIVE_JOB_REGEX=${FASTAFD_ACTIVE_JOB_REGEX:-'^fastafd:afd3d-'}
        MAX_ACTIVE_JOBS=${FASTAFD_MAX_ACTIVE_JOBS:-5}
    else
        ACTIVE_JOB_REGEX=${FASTAFD_ACTIVE_JOB_REGEX:-'^fastafd:pmap-'}
        MAX_ACTIVE_JOBS=${FASTAFD_MAX_ACTIVE_JOBS:-4}
    fi
    [[ -n "$ACTIVE_JOB_REGEX" ]]
    [[ "$MAX_ACTIVE_JOBS" =~ ^[1-5]$ ]] || {
        echo "FASTAFD_MAX_ACTIVE_JOBS must be between 1 and 5" >&2; exit 2;
    }
    ACTIVE_FASTAFD=$(squeue -h -u "$USER" -o '%j' | \
        awk -v job_regex="$ACTIVE_JOB_REGEX" '$0 ~ job_regex {n++} END {print n+0}')
    (( ACTIVE_FASTAFD < MAX_ACTIVE_JOBS )) || {
        echo "$MAX_ACTIVE_JOBS AFD jobs matching $ACTIVE_JOB_REGEX are already active" >&2; exit 1;
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR=$RESULTS_ROOT/afd_${MODEL_KEY}_${CONTEXT}_r${NORMALIZED_AF_RATIO}_ag${ATTENTION_GPUS}_fg${FFN_GPUS}_atp${ATTENTION_TP}_fep${FFN_EP}_adp${ATTENTION_DP_SIZE}_b${BATCH}_n${NODES}_${STAMP}_manual_na
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable --nodes="$NODES" --segment="$NODES" --qos="$SUBMIT_QOS" --time="$JOB_TIME_LIMIT")
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,FASTAFD_SUBMIT_QOS=$SUBMIT_QOS,FASTAFD_SWEEP_CONTRACT=$SWEEP_CONTRACT,MODEL_KEY=$MODEL_KEY,PROMPT_MODE=$PROMPT_MODE,CONTEXT_SPEC=$CONTEXT_SPEC,CONTEXT=$CONTEXT,CONTEXT_MIN=$CONTEXT_MIN,CONTEXT_MAX=$CONTEXT_MAX,IRREGULAR=$IRREGULAR,SHAPE_ID=$SHAPE_ID,PROMPT_LENGTH_SUM=$PROMPT_LENGTH_SUM,AFD_REQUIRED_KV_TOKENS=$AFD_REQUIRED_KV_TOKENS,AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS,NOMINAL_AFD_KV_CAPACITY_TOKENS=$NOMINAL_AFD_KV_CAPACITY_TOKENS,CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH,RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH,SCHEDULER_ADMISSION_REQUIRED_TOKENS=$SCHEDULER_ADMISSION_REQUIRED_TOKENS,SCHEDULER_ADMISSION_UNUSED_TOKENS=$SCHEDULER_ADMISSION_UNUSED_TOKENS,SCHEDULER_RUNNING_RESERVATION_TOKENS=$SCHEDULER_RUNNING_RESERVATION_TOKENS,REQUIRE_CAPACITY_MAX=$REQUIRE_CAPACITY_MAX,ALLOW_OBSERVED_CAPACITY_PROBE=$ALLOW_OBSERVED_CAPACITY_PROBE,MAX_SEQ_LEN=$MAX_SEQ_LEN,NSYS_CUDA_GRAPH_TRACE=$NSYS_CUDA_GRAPH_TRACE,NSYS_CAPTURE_DECODE_STEPS=$NSYS_CAPTURE_DECODE_STEPS,MODEL_PATH=$MODEL_PATH,MODEL_SOURCE_PATH=$MODEL_SOURCE_PATH,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_SOURCE_CONFIG_SHA256=$MODEL_SOURCE_CONFIG_SHA256,MODEL_PROFILE_ID=$MODEL_PROFILE_ID,MODEL_PROFILE_MANIFEST=$MODEL_PROFILE_MANIFEST,ROPE_SCALING_FACTOR=$ROPE_SCALING_FACTOR,ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,MODEL_REVISION=$MODEL_REVISION,PRESET=$PRESET,NODES=$NODES,BATCH=$BATCH,GRAPH_BATCH=$GRAPH_BATCH,REFERENCE_TPS=$REFERENCE_TPS,AFD_ATTENTION_TO_FFN_TRAY_RATIO=$NORMALIZED_AF_RATIO,AFD_ATTN_TP_SIZE=$ATTENTION_TP,AFD_MLP_EP_SIZE=$FFN_EP,NORMALIZED_AF_RATIO=$NORMALIZED_AF_RATIO,ATTENTION_TRAYS=$ATTENTION_TRAYS,FFN_TRAYS=$FFN_TRAYS,ATTENTION_TP=$ATTENTION_TP,FFN_EP=$FFN_EP,ATTENTION_GPUS=$ATTENTION_GPUS,FFN_GPUS=$FFN_GPUS,ACTIVE_GPUS=$ACTIVE_GPUS,ALLOCATED_GPUS=$ALLOCATED_GPUS,SHARED_ROLE_TRAYS=$SHARED_ROLE_TRAYS,ATTENTION_DP_SIZE=$ATTENTION_DP_SIZE,ATTENTION_WORKERS=$ATTENTION_WORKERS,MLP_DP_SIZE=$MLP_DP_SIZE,MLP_TP_SIZE=$MLP_TP_SIZE,NUM_PROMPTS=$NUM_PROMPTS,PROMPT_BASE=$PROMPT_BASE,PROMPT_SHA256=$PROMPT_SHA256,PROMPT_SOURCE_CONTEXT=$PROMPT_SOURCE_CONTEXT,EXPECTED_HEAD=$EXPECTED_HEAD,ROOT=$ROOT,RESULTS_ROOT=$RESULTS_ROOT,SOURCE_REPO=$SOURCE_REPO,EXPECTED_SOURCE_MANIFEST=$EXPECTED_SOURCE_MANIFEST,EXPECTED_SOURCE_COORDINATOR_SHA256=$EXPECTED_SOURCE_COORDINATOR_SHA256,EXPECTED_SOURCE_PROFILER_SHA256=$EXPECTED_SOURCE_PROFILER_SHA256,EXPECTED_SOURCE_PROTOCOL_SHA256=$EXPECTED_SOURCE_PROTOCOL_SHA256,EXPECTED_SOURCE_PLACEMENT_SHA256=$EXPECTED_SOURCE_PLACEMENT_SHA256,EXPECTED_SOURCE_API_SHA256=$EXPECTED_SOURCE_API_SHA256,EXPECTED_SOURCE_ARGS_SHA256=$EXPECTED_SOURCE_ARGS_SHA256,CUDA_EXTRACT_SCRIPT=$CUDA_EXTRACT_SCRIPT,CUDA_METRIC_PLAN=$CUDA_METRIC_PLAN,CUDA_SPAN_MODULE=$CUDA_SPAN_MODULE,CUDA_METRICS_ROOT=$CUDA_METRICS_ROOT,CUDA_EXTRACT_TEMP_ROOT=$CUDA_EXTRACT_TEMP_ROOT,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s model=%s context=%s batch=%s normalized_ratio=%s:1 attention_trays=%s ffn_trays=%s attention_tp=%s ffn_ep=%s attention_dp=%s active_gpus=%s allocated_gpus=%s shared_role_trays=%s nodes=%s run_dir=%s\n' \
        "$JOB" "$MODEL_KEY" "$CONTEXT" "$BATCH" "$NORMALIZED_AF_RATIO" \
        "$ATTENTION_TRAYS" "$FFN_TRAYS" "$ATTENTION_TP" "$FFN_EP" \
        "$ATTENTION_DP_SIZE" "$ACTIVE_GPUS" "$ALLOCATED_GPUS" \
        "$SHARED_ROLE_TRAYS" "$NODES" "$RUN_DIR"
    exit 0
fi

if [[ -z "${MODEL_CONFIG_SHA256:-}" ]]; then
    [[ "${FASTAFD_IN_CONTAINER:-0}" == 1 ]] || {
        echo "MODEL_CONFIG_SHA256 is unset outside the container execution path" >&2
        exit 2
    }
    MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
fi

: "${RUN_DIR:?}" "${JOB_SCRIPT:?}"
if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == "$SUBMIT_QOS" ]]
    [[ "$SLURM_JOB_NUM_NODES" == "$NODES" ]]
    HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
    exec srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gres=gpu:4 --kill-on-bad-exit=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=CUDA_LAUNCH_BLOCKING,NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_IN_CONTAINER=1 FASTAFD_SWEEP_CONTRACT="$SWEEP_CONTRACT" \
        HEAD_HOST="$HEAD_HOST" MODEL_KEY="$MODEL_KEY" \
        PROMPT_MODE="$PROMPT_MODE" CONTEXT_SPEC="$CONTEXT_SPEC" \
        CONTEXT="$CONTEXT" CONTEXT_MIN="$CONTEXT_MIN" CONTEXT_MAX="$CONTEXT_MAX" \
        IRREGULAR="$IRREGULAR" SHAPE_ID="$SHAPE_ID" \
        PROMPT_LENGTHS_CSV="$PROMPT_LENGTHS_CSV" PROMPT_LENGTH_SUM="$PROMPT_LENGTH_SUM" \
        AFD_REQUIRED_KV_TOKENS="$AFD_REQUIRED_KV_TOKENS" \
        AFD_KV_CAPACITY_TOKENS="$AFD_KV_CAPACITY_TOKENS" \
        NOMINAL_AFD_KV_CAPACITY_TOKENS="$NOMINAL_AFD_KV_CAPACITY_TOKENS" \
        CAPACITY_MAX_BATCH="$CAPACITY_MAX_BATCH" RAW_KV_MAX_BATCH="$RAW_KV_MAX_BATCH" \
        SCHEDULER_ADMISSION_REQUIRED_TOKENS="$SCHEDULER_ADMISSION_REQUIRED_TOKENS" \
        SCHEDULER_ADMISSION_UNUSED_TOKENS="$SCHEDULER_ADMISSION_UNUSED_TOKENS" \
        SCHEDULER_RUNNING_RESERVATION_TOKENS="$SCHEDULER_RUNNING_RESERVATION_TOKENS" \
        REQUIRE_CAPACITY_MAX="$REQUIRE_CAPACITY_MAX" \
        ALLOW_OBSERVED_CAPACITY_PROBE="$ALLOW_OBSERVED_CAPACITY_PROBE" \
        MAX_SEQ_LEN="$MAX_SEQ_LEN" NSYS_CUDA_GRAPH_TRACE="$NSYS_CUDA_GRAPH_TRACE" \
        NSYS_CAPTURE_DECODE_STEPS="$NSYS_CAPTURE_DECODE_STEPS" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_SOURCE_PATH="$MODEL_SOURCE_PATH" MODEL_CONFIG_SHA256="$MODEL_CONFIG_SHA256" \
        MODEL_SOURCE_CONFIG_SHA256="$MODEL_SOURCE_CONFIG_SHA256" \
        MODEL_PROFILE_ID="$MODEL_PROFILE_ID" MODEL_PROFILE_MANIFEST="$MODEL_PROFILE_MANIFEST" \
        ROPE_SCALING_FACTOR="$ROPE_SCALING_FACTOR" \
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS="$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS" \
        MODEL_REVISION="$MODEL_REVISION" PRESET="$PRESET" NODES="$NODES" \
        BATCH="$BATCH" GRAPH_BATCH="$GRAPH_BATCH" REFERENCE_TPS="$REFERENCE_TPS" \
        NORMALIZED_AF_RATIO="$NORMALIZED_AF_RATIO" \
        ATTENTION_TRAYS="$ATTENTION_TRAYS" FFN_TRAYS="$FFN_TRAYS" \
        ATTENTION_TP="$ATTENTION_TP" FFN_EP="$FFN_EP" \
        ATTENTION_GPUS="$ATTENTION_GPUS" ATTENTION_DP_SIZE="$ATTENTION_DP_SIZE" \
        FFN_GPUS="$FFN_GPUS" ACTIVE_GPUS="$ACTIVE_GPUS" \
        ALLOCATED_GPUS="$ALLOCATED_GPUS" SHARED_ROLE_TRAYS="$SHARED_ROLE_TRAYS" \
        ATTENTION_WORKERS="$ATTENTION_WORKERS" \
        MLP_DP_SIZE="$MLP_DP_SIZE" MLP_TP_SIZE="$MLP_TP_SIZE" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        PROMPT_BASE="$PROMPT_BASE" PROMPT_SHA256="$PROMPT_SHA256" \
        PROMPT_SOURCE_CONTEXT="$PROMPT_SOURCE_CONTEXT" \
        EXPECTED_HEAD="$EXPECTED_HEAD" ROOT="$ROOT" RESULTS_ROOT="$RESULTS_ROOT" \
        SOURCE_REPO="$SOURCE_REPO" EXPECTED_SOURCE_MANIFEST="$EXPECTED_SOURCE_MANIFEST" \
        EXPECTED_SOURCE_COORDINATOR_SHA256="$EXPECTED_SOURCE_COORDINATOR_SHA256" \
        EXPECTED_SOURCE_PROFILER_SHA256="$EXPECTED_SOURCE_PROFILER_SHA256" \
        EXPECTED_SOURCE_PROTOCOL_SHA256="$EXPECTED_SOURCE_PROTOCOL_SHA256" \
        EXPECTED_SOURCE_PLACEMENT_SHA256="$EXPECTED_SOURCE_PLACEMENT_SHA256" \
        EXPECTED_SOURCE_API_SHA256="$EXPECTED_SOURCE_API_SHA256" \
        EXPECTED_SOURCE_ARGS_SHA256="$EXPECTED_SOURCE_ARGS_SHA256" \
        CUDA_EXTRACT_SCRIPT="$CUDA_EXTRACT_SCRIPT" \
        CUDA_METRIC_PLAN="$CUDA_METRIC_PLAN" \
        CUDA_SPAN_MODULE="$CUDA_SPAN_MODULE" \
        CUDA_METRICS_ROOT="$CUDA_METRICS_ROOT" \
        CUDA_EXTRACT_TEMP_ROOT="$CUDA_EXTRACT_TEMP_ROOT" \
        IMAGE="$IMAGE" EP_VENV_DIR="$EP_VENV_DIR" \
        RUN_DIR="$RUN_DIR" JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

RANK=${SLURM_PROCID:?}
[[ "$RANK" =~ ^[0-9]+$ && "$RANK" -lt "$NODES" ]] || \
    preflight_error rank "rank=$RANK nodes=$NODES"
PYTHON=$EP_VENV_DIR/bin/python
[[ -x "$PYTHON" ]] || preflight_error venv_python "$PYTHON"
export PATH=$EP_VENV_DIR/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$SOURCE_REPO/python${PYTHONPATH:+:$PYTHONPATH}
export VIRTUAL_ENV=$EP_VENV_DIR CONDA_PREFIX=$EP_VENV_DIR
export CONDA_DEFAULT_ENV=fastafd-venv ENV_NAME=fastafd-venv
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=32 NVCC_THREADS=4 TORCH_CUDA_ARCH_LIST=10.0
# Ray is only the control plane for four GPU actors per tray. Advertising all
# 140 CPUs invites an unnecessary worker pool. Native compiler and kernel
# threads can still use the complete Slurm CPU allocation independently of
# Ray logical resources. Let Ray size its object store from the task's actual
# cgroup rather than assuming the allocation-level memory is visible inside
# Pyxis.
RAY_NUM_CPUS_PER_NODE=${FASTAFD_RAY_NUM_CPUS_PER_NODE:-16}
[[ "$RAY_NUM_CPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
    preflight_error ray_num_cpus "$RAY_NUM_CPUS_PER_NODE"
(( RAY_NUM_CPUS_PER_NODE >= 8 && RAY_NUM_CPUS_PER_NODE <= 32 )) || \
    preflight_error ray_num_cpus "$RAY_NUM_CPUS_PER_NODE"
# NCCL RAS is not queried by this sweep.  Disable its default localhost:28028
# listener so unrelated jobs sharing a tray cannot produce bind warnings.
export NCCL_RAS_ENABLE=0

CASE_RUN_DIR=$RUN_DIR
CONTROL=$CASE_RUN_DIR/control
READY=$CONTROL/ready
DONE=$CONTROL/snapshot-done
STOP=$CONTROL/STOP
SNAPSHOT=$CONTROL/SNAPSHOT
GPU_PROGRESS=$CASE_RUN_DIR/gpu-progress
mkdir -p "$READY" "$DONE" "$CASE_RUN_DIR/gpu-snapshots" "$GPU_PROGRESS" "$CASE_RUN_DIR/tmp/rank-$RANK"
export TMPDIR=$CASE_RUN_DIR/tmp/rank-$RANK
export RAY_TMPDIR=/dev/shm/fastafd-ray-$SLURM_JOB_ID-$RANK-$CASE_ORDINAL
ray_cli() { "$PYTHON" -m ray.scripts.scripts "$@"; }
GPU_PROGRESS_PID=""
cleanup() {
    local cleanup_status=0
    [[ "$RANK" != 0 ]] || touch "$STOP"
    if [[ -n "$GPU_PROGRESS_PID" ]]; then
        kill "$GPU_PROGRESS_PID" >/dev/null 2>&1 || true
    fi
    ray_cli stop --force >/dev/null 2>&1 || true
    "$PYTHON" "$GPU_PROCESS_EXIT_SCRIPT" --timeout-seconds 120 \
        > "$CASE_RUN_DIR/gpu-snapshots/cleanup-rank-$RANK.json" || cleanup_status=$?
    return "$cleanup_status"
}
trap cleanup EXIT TERM INT

[[ "$(uname -m)" == aarch64 ]] || preflight_error architecture "observed=$(uname -m)"
validate_source_repo || \
    preflight_error source_repo "$SOURCE_VALIDATION_DETAIL"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || \
    preflight_error model_index "$MODEL_PATH/model.safetensors.index.json"
[[ -f "$PROMPT_BASE" ]] || preflight_error prompt_source "$PROMPT_BASE"
[[ -f "$GPU_PROCESS_EXIT_SCRIPT" ]] || \
    preflight_error gpu_process_exit_script "$GPU_PROCESS_EXIT_SCRIPT"
[[ $(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}') == "$MODEL_CONFIG_SHA256" ]] || \
    preflight_error model_config_sha256 "$MODEL_PATH/config.json"
[[ "$MODEL_PROFILE_ID" == native || -f "$MODEL_PROFILE_MANIFEST" ]] || \
    preflight_error model_profile_manifest "$MODEL_PROFILE_MANIFEST"
[[ $(sha256sum "$PROMPT_BASE" | awk '{print $1}') == "$PROMPT_SHA256" ]] || \
    preflight_error prompt_sha256 "$PROMPT_BASE"
ray_cli stop --force >/dev/null 2>&1 || true
"$PYTHON" "$GPU_PROCESS_EXIT_SCRIPT" --timeout-seconds 120 \
    > "$CASE_RUN_DIR/gpu-snapshots/preflight-cleanup-rank-$RANK.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$CASE_RUN_DIR/gpu-snapshots/initial-rank-$RANK.csv"
[[ $(wc -l < "$CASE_RUN_DIR/gpu-snapshots/initial-rank-$RANK.csv") -eq 1 ]]

# One nvidia-smi sample/tray/minute gives the controller a cheap independent
# progress signal. It is intentionally outside the profiled worker processes.
PROGRESS_FILE=$GPU_PROGRESS/rank-$RANK.tsv
printf 'epoch_seconds\tgpu_index,utilization_percent,memory_mib[;...]\n' > "$PROGRESS_FILE"
(
    while [[ ! -e "$STOP" ]]; do
        snapshot=$(nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits | paste -sd ';' -)
        printf '%s\t%s\n' "$(date +%s)" "$snapshot" >> "$PROGRESS_FILE"
        sleep 60
    done
) &
GPU_PROGRESS_PID=$!

HEAD_ADDRESS=$HEAD_HOST:6379
if [[ "$RANK" == 0 ]]; then
    printf '%s\n' "$HEAD_ADDRESS" > "$CONTROL/head-address"
    ray_cli start --head --port=6379 \
        --num-cpus="$RAY_NUM_CPUS_PER_NODE" --num-gpus=4 \
        --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMPDIR" \
        > "$RUN_DIR/ray-start-0.log" 2>&1
else
    deadline=$((SECONDS + 120))
    until [[ -s "$CONTROL/head-address" ]]; do
        [[ ! -e "$STOP" && $SECONDS -lt $deadline ]]; sleep 1
    done
    HEAD_ADDRESS=$(<"$CONTROL/head-address")
    ray_cli start --address="$HEAD_ADDRESS" \
        --num-cpus="$RAY_NUM_CPUS_PER_NODE" --num-gpus=4 \
        --disable-usage-stats > "$RUN_DIR/ray-start-$RANK.log" 2>&1
fi
touch "$READY/$RANK"

if [[ "$RANK" != 0 ]]; then
    until [[ -e "$SNAPSHOT" || -e "$STOP" ]]; do sleep 2; done
    if [[ -e "$SNAPSHOT" ]]; then
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv > "$RUN_DIR/gpu-snapshots/final-rank-$RANK.csv"
        touch "$DONE/$RANK"
    fi
    until [[ -e "$STOP" ]]; do sleep 2; done
    exit 0
fi

deadline=$((SECONDS + 180))
until [[ $(find "$READY" -maxdepth 1 -type f | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $deadline ]]; sleep 2
done
RAY_ADDRESS=$HEAD_ADDRESS NODES=$NODES "$PYTHON" - <<'PY'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0)]
    if len(nodes) == int(os.environ["NODES"]) and sum(int(n["Resources"]["GPU"]) for n in nodes) == 4 * len(nodes):
        break
    time.sleep(2)
else:
    raise RuntimeError([(n.get("NodeManagerAddress"), n.get("Resources")) for n in ray.nodes()])
PY

# AFD uses a contiguous seven-port control block. Validate the whole block
# after Ray starts so an adjacent service cannot strand the detokenizer.
MINISGL_PORT=$(
RAY_ADDRESS="$HEAD_ADDRESS" NODES="$NODES" "$PYTHON" - \
    "$SLURM_JOB_ID" "${FASTAFD_MINISGL_PORT:-}" "$CASE_PORT_SLOT" <<'PY'
import os
import socket
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

job_id = int(sys.argv[1].split("_")[0])
override = sys.argv[2]
case_port_slot = 0 if sys.argv[3] == "single" else int(sys.argv[3])
candidates = [int(override)] if override else [
    20000 + ((job_id + case_port_slot + offset) % 1000) * 8
    for offset in range(1000)
]
ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
nodes = [
    node
    for node in ray.nodes()
    if node.get("Alive") and node.get("Resources", {}).get("GPU", 0)
]
expected_nodes = int(os.environ["NODES"])
if len(nodes) != expected_nodes:
    raise SystemExit(
        f"expected {expected_nodes} live GPU nodes for port validation, got "
        f"{[(node.get('NodeID'), node.get('NodeManagerAddress')) for node in nodes]}"
    )


@ray.remote(num_cpus=0)
def block_is_free(base):
    sockets = []
    try:
        for port in range(base, base + 7):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
    except OSError as exc:
        return {"free": False, "error": str(exc)}
    finally:
        for sock in sockets:
            sock.close()
    return {"free": True, "error": None}


last_error = None
for base in candidates:
    if not 1024 <= base <= 65529:
        last_error = ValueError(f"invalid seven-port block {base}-{base + 6}")
        continue
    checks = ray.get(
        [
            block_is_free.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(base)
            for node in nodes
        ]
    )
    if all(check["free"] for check in checks):
        print(base)
        break
    last_error = {
        node["NodeManagerAddress"]: check["error"]
        for node, check in zip(nodes, checks)
        if not check["free"]
    }
else:
    raise SystemExit(
        f"no seven-port AFD control block free on every allocated node: {last_error}"
    )
PY
)
[[ "$MINISGL_PORT" =~ ^[0-9]+$ && "$MINISGL_PORT" -ge 1024 && "$MINISGL_PORT" -le 65529 ]]
export MINISGL_PORT
printf 'afd_control_port_block=%s-%s\n' "$MINISGL_PORT" "$((MINISGL_PORT + 6))"

export RAY_ADDRESS=$HEAD_ADDRESS MINISGL_RAY_ADDRESS=$HEAD_ADDRESS MODEL_PATH
export PROMPT_BASE_FILE=$PROMPT_BASE
export PROMPT_FILE=$RUN_DIR/prompts_${NUM_PROMPTS}x${SHAPE_ID}.txt
PROMPT_MANIFEST=$RUN_DIR/prompt-transform.json
PROMPT_CACHE_ROOT=${FASTAFD_PROMPT_CACHE_ROOT:-$RESULTS_ROOT/prompt-cache}
"$PYTHON" - "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$PROMPT_MANIFEST" \
    "$PROMPT_CACHE_ROOT" "$NUM_PROMPTS" "$CONTEXT_MIN" "$CONTEXT_MAX" \
    "$MODEL_KEY" "$MODEL_PATH" "$MODEL_REVISION" "$PROMPT_MODE" \
    "$PROMPT_SOURCE_CONTEXT" "$ATTENTION_DP_SIZE" "$PROMPT_LENGTHS_CSV" <<'PY'
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile

src, dst, manifest, cache_root = map(Path, sys.argv[1:5])
count, context_min, context_max = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
model_key, model_path, model_revision, prompt_mode = sys.argv[8:12]
source_context, attention_dp_size = int(sys.argv[12]), int(sys.argv[13])
prompt_lengths = [int(value) for value in sys.argv[14].split(",")]
if count != attention_dp_size * len(prompt_lengths):
    raise RuntimeError((count, attention_dp_size, len(prompt_lengths)))
if prompt_lengths[0] != context_min or prompt_lengths[-1] != context_max:
    raise RuntimeError(prompt_lengths)
if any(left > right for left, right in zip(prompt_lengths, prompt_lengths[1:])):
    raise RuntimeError("prompt lengths are not monotonic")
if any(left + right != context_min + context_max for left, right in zip(prompt_lengths, reversed(prompt_lengths))):
    raise RuntimeError("prompt lengths are not symmetric")
if prompt_mode not in {"uniform", "irregular"}:
    raise RuntimeError(f"invalid prompt mode: {prompt_mode}")
if prompt_mode == "uniform" and (
    context_min != context_max or any(value != context_min for value in prompt_lengths)
):
    raise RuntimeError("uniform prompt mode has nonuniform prompt lengths")
model_config_bytes = (Path(model_path) / "config.json").read_bytes()
model_config = json.loads(model_config_bytes)
source_bytes = src.read_bytes()
source_sha256 = hashlib.sha256(source_bytes).hexdigest()
lines = [x.strip() for x in source_bytes.decode().splitlines() if x.strip()]
if len(lines) != 512:
    raise RuntimeError(len(lines))

if prompt_mode == "uniform":
    prompt_layout = {"prompt_mode": "uniform", "target_tokens": context_min}
else:
    prompt_layout = {
        "prompt_mode": "irregular",
        "attention_dp_size": attention_dp_size,
        "per_attention_gpu_prompt_lengths": prompt_lengths,
    }
cache_spec = {
    "format_version": "20260814-shared-prompt-cache-v2",
    "model_key": model_key,
    "model_path": model_path,
    "model_revision": model_revision,
    "model_config_sha256": hashlib.sha256(model_config_bytes).hexdigest(),
    "source_file": str(src),
    "source_sha256": source_sha256,
    "source_context_tokens": source_context,
    "output_prompt_count": count,
    "target_context_min_tokens": context_min,
    "target_context_max_tokens": context_max,
    "prompt_layout": prompt_layout,
}
cache_key = hashlib.sha256(
    json.dumps(cache_spec, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
cache_entry = cache_root / cache_key
cache_file = cache_entry / "prompts.txt"
cache_manifest = cache_entry / "prompt-transform.json"
cache_lock = cache_root / f"{cache_key}.lock"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def qwen_chat_prompt(text, target_tokens, tokenizer):
    source_ids = tokenizer.encode(text, add_special_tokens=False)
    if not source_ids:
        raise RuntimeError("empty tokenized source prompt")
    template_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    initial_content_count = target_tokens - len(template_tokens)
    content_count = initial_content_count
    visited_counts = set()
    for _ in range(64):
        if content_count in visited_counts:
            break
        visited_counts.add(content_count)
        repeats = (content_count + len(source_ids) - 1) // len(source_ids)
        selected = (source_ids * repeats)[:content_count]
        content = tokenizer.decode(
            selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        delta = target_tokens - len(prompt_ids)
        if delta == 0:
            if "\n" in content or "\r" in content:
                raise RuntimeError("transformed prompt is not one physical line")
            return content, prompt_ids, len(source_ids), ""
        content_count += delta
        if content_count <= 0:
            break

    # Byte-pair merges can make an exact length unreachable by changing only
    # the source-token prefix (for example, a 1088 -> 1090 jump). Resolve such
    # gaps deterministically with a small fixed suffix search. Exact length is
    # still mandatory, and every repair is recorded in the prompt manifest.
    repair_suffixes = (" 0", " a", " x", " !", " ?", " .", " #", " |", "0", "!", "?", "_", "~")
    for distance in range(65):
        candidate_counts = [initial_content_count - distance]
        if distance:
            candidate_counts.append(initial_content_count + distance)
        for candidate_count in candidate_counts:
            if candidate_count <= 0:
                continue
            repeats = (candidate_count + len(source_ids) - 1) // len(source_ids)
            selected = (source_ids * repeats)[:candidate_count]
            base_content = tokenizer.decode(
                selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for suffix in repair_suffixes:
                content = base_content + suffix
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if len(prompt_ids) == target_tokens:
                    if "\n" in content or "\r" in content:
                        raise RuntimeError("transformed prompt is not one physical line")
                    return content, prompt_ids, len(source_ids), suffix
    raise RuntimeError(
        f"cannot construct exact {target_tokens}-token chat prompt after deterministic merge-gap search"
    )

def build_record(output_file):
    token_digest = hashlib.sha256()
    source_token_lengths = set()
    merge_gap_repairs = []
    if model_key == "qwen3":
        from minisgl.hf_support import load_tokenizer

        tokenizer = load_tokenizer(model_path)
        transformed = {}
        with output_file.open("w", encoding="utf-8") as out:
            for index in range(count):
                source_index = index % len(lines)
                target_tokens = prompt_lengths[index // attention_dp_size]
                transform_key = (source_index, target_tokens)
                if transform_key not in transformed:
                    transformed[transform_key] = qwen_chat_prompt(
                        lines[source_index], target_tokens, tokenizer
                    )
                content, prompt_ids, source_length, repair_suffix = transformed[transform_key]
                if repair_suffix and not any(
                    item["source_index"] == source_index
                    and item["target_tokens"] == target_tokens
                    for item in merge_gap_repairs
                ):
                    merge_gap_repairs.append(
                        {
                            "source_index": source_index,
                            "target_tokens": target_tokens,
                            "suffix": repair_suffix,
                        }
                    )
                source_token_lengths.add(source_length)
                for token in prompt_ids:
                    token_digest.update(int(token).to_bytes(4, "little"))
                out.write(content + "\n")
        transform = "cut_or_repeat_pinned_8k_content_tokens_then_qwen_chat_template_with_exact_merge_gap_correction"
        observed = {"min": min(prompt_lengths), "max": max(prompt_lengths)}
        prompt_token_sha256 = token_digest.hexdigest()
    else:
        output_file.write_text(
            "\n".join(lines[i % len(lines)] for i in range(count)) + "\n"
        )
        transform = "pinned_text_unchanged"
        observed = None
        prompt_token_sha256 = None
    return {
        "model_key": model_key,
        "source_file": str(src),
        "source_sha256": source_sha256,
        "source_prompt_count": len(lines),
        "source_context_tokens": source_context,
        "output_file": str(cache_file),
        "output_file_sha256": file_sha256(output_file),
        "output_file_size_bytes": output_file.stat().st_size,
        "output_prompt_count": count,
        "target_context_tokens": context_max if context_min == context_max else None,
        "target_context_range_tokens": {"min": context_min, "max": context_max},
        "distribution": "inclusive_uniform_symmetric_integer_linspace",
        "attention_dp_size": attention_dp_size,
        "batch_per_attention_dp_lane": len(prompt_lengths),
        "batch_per_attention_gpu": len(prompt_lengths),
        "per_attention_gpu_prompt_lengths": prompt_lengths,
        "per_attention_gpu_prompt_length_sum": sum(prompt_lengths),
        "global_prompt_order": "length-major, then attention-DP rank; coordinator round-robin gives every rank the listed distribution",
        "model_path": model_path,
        "model_config_sha256": hashlib.sha256(model_config_bytes).hexdigest(),
        "model_max_position_embeddings": model_config.get("max_position_embeddings"),
        "model_rope_scaling": model_config.get("rope_scaling"),
        "transform": transform,
        "observed_chat_prompt_tokens": observed,
        "aggregate_prompt_token_ids_sha256": prompt_token_sha256,
        "source_content_token_lengths": sorted(source_token_lengths),
        "exact_length_merge_gap_repairs": merge_gap_repairs,
        "prefix_caching": "not present: AFD uses cache-type=naive",
        "prompt_cache_key": cache_key,
        "prompt_cache_spec": cache_spec,
    }


cache_root.mkdir(parents=True, exist_ok=True)
with cache_lock.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    cache_hit = cache_entry.is_dir()
    if cache_hit:
        if not cache_file.is_file() or not cache_manifest.is_file():
            raise RuntimeError(f"incomplete prompt cache entry: {cache_entry}")
        record = json.loads(cache_manifest.read_text(encoding="utf-8"))
    else:
        if cache_entry.exists():
            raise RuntimeError(f"prompt cache entry is not a directory: {cache_entry}")
        build_dir = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=cache_root))
        try:
            build_file = build_dir / "prompts.txt"
            record = build_record(build_file)
            (build_dir / "prompt-transform.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.rename(build_dir, cache_entry)
        except BaseException:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise
    if record.get("prompt_cache_key") != cache_key:
        raise RuntimeError(f"prompt cache key mismatch: {cache_entry}")
    if record.get("prompt_cache_spec") != cache_spec:
        raise RuntimeError(f"prompt cache specification mismatch: {cache_entry}")
    if record.get("output_file") != str(cache_file):
        raise RuntimeError(f"prompt cache path mismatch: {cache_entry}")
    if record.get("output_file_size_bytes") != cache_file.stat().st_size:
        raise RuntimeError(f"prompt cache size mismatch: {cache_entry}")
    if record.get("output_file_sha256") != file_sha256(cache_file):
        raise RuntimeError(f"prompt cache digest mismatch: {cache_entry}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(cache_file, dst)

case_record = dict(record)
case_record["output_file"] = str(dst)
case_record["prompt_cache_file"] = str(cache_file)
case_record["prompt_cache_hit"] = cache_hit
case_record["prompt_mode"] = prompt_mode
case_record["attention_dp_size"] = attention_dp_size
case_record["batch_per_attention_dp_lane"] = len(prompt_lengths)
case_record["batch_per_attention_gpu"] = len(prompt_lengths)
case_record["per_attention_gpu_prompt_lengths"] = prompt_lengths
case_record["per_attention_gpu_prompt_length_sum"] = sum(prompt_lengths)
manifest.write_text(json.dumps(case_record, indent=2, sort_keys=True) + "\n")
print(json.dumps(case_record, indent=2, sort_keys=True))
PY
export FASTAFD_SOURCE_REPO=$SOURCE_REPO
export AFD_TOTAL_NODES=$NODES \
    RUN_VLLM_ALIGNMENT=${FASTAFD_RUN_VLLM_ALIGNMENT:-0} \
    NSYS_CUDA_GRAPH_TRACE
if [[ "$RUN_VLLM_ALIGNMENT" == 1 ]]; then
    export CAPTURE_EXIT_AFTER_WINDOW=0
fi
export AFD_ATTN_TP_SIZE=$ATTENTION_TP AFD_MLP_EP_SIZE=$FFN_EP
if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
    export AFD_ATTENTION_GPU_COUNT=$ATTENTION_GPUS
else
    unset AFD_ATTENTION_GPU_COUNT
fi
export AFD_ATTN_DP_SIZE=$ATTENTION_DP_SIZE MLP_DP_SIZE MLP_TP_SIZE
export PROMPT_LEN=$CONTEXT_MAX PER_ATTN_GPU_BSZ=$BATCH MAX_TOKENS=$OUTPUT_TOKENS
export MINISGL_MAX_SEQ_LEN=$MAX_SEQ_LEN VLLM_MAX_MODEL_LEN=$MAX_SEQ_LEN
export AFD_DECODE_GRAPH_BS=$GRAPH_BATCH
export AFD_MEMORY_RATIO
# Keep the proven sweep prefill contract for both placements. The apparent
# long-prefill stall was cold per-rank extension compilation followed by a
# CUDA-fabric mapping-limit failure, not useful 512-token inference work.
AFD_MAX_BATCHED_TOKENS=${FASTAFD_AFD_MAX_BATCHED_TOKENS:-512}
[[ "$AFD_MAX_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_AFD_MAX_BATCHED_TOKENS must be a positive integer" >&2
    exit 2
}
(( AFD_MAX_BATCHED_TOKENS <= 8192 )) || {
    echo "AFD max batched tokens exceeds the supported extend capacity: $AFD_MAX_BATCHED_TOKENS > 8192" >&2
    exit 2
}
export AFD_MAX_BATCHED_TOKENS
# A prefill scheduler step may carry one decode token for every resident
# request in addition to the prefill token budget. Reserve that exact combined
# bound for DeepEP; decode still dispatches with the much smaller graph bucket.
MINISGL_AFD_MAX_COMM_TOKENS=${FASTAFD_AFD_MAX_COMM_TOKENS:-$((AFD_MAX_BATCHED_TOKENS + BATCH))}
[[ "$MINISGL_AFD_MAX_COMM_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FASTAFD_AFD_MAX_COMM_TOKENS must be a positive integer" >&2
    exit 2
}
(( MINISGL_AFD_MAX_COMM_TOKENS >= AFD_MAX_BATCHED_TOKENS + BATCH )) || {
    echo "AFD max communication tokens must cover prefill budget plus lane batch: required=$((AFD_MAX_BATCHED_TOKENS + BATCH)) configured=$MINISGL_AFD_MAX_COMM_TOKENS" >&2
    exit 2
}
export MINISGL_AFD_MAX_COMM_TOKENS
if [[ -n "$AFD_NUM_PAGES" ]]; then
    export AFD_SERVER_EXTRA_ARGS="--attention-backend trtllm --memory-ratio $AFD_MEMORY_RATIO --num-pages $AFD_NUM_PAGES"
else
    export AFD_SERVER_EXTRA_ARGS="--attention-backend trtllm --memory-ratio $AFD_MEMORY_RATIO"
fi
export MINISGL_AFD_WORKER_HOT_LOOP_SHUTDOWN_TIMEOUT_S=${FASTAFD_AFD_WORKER_HOT_LOOP_SHUTDOWN_TIMEOUT_S:-300}
export MINISGL_AFD_WORKER_SHUTDOWN_TIMEOUT_S=${FASTAFD_AFD_WORKER_SHUTDOWN_TIMEOUT_S:-120}
export NSYS_TARGET_BATCH_PER_ATTN_DP=$BATCH
export NSYS_CAPTURE_DECODE_STEPS
export CAPTURE_WORKER_STOP_TIMEOUT_S
export MINISGL_RAY_NSYS_TARGET_BATCH_PER_DP=$BATCH
export MINISGL_RAY_NSYS_CAPTURE_DECODE_STEPS=$NSYS_CAPTURE_DECODE_STEPS
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$ROOT/cache/afd/flashinfer}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$ROOT/cache/afd/tvm_ffi}
export EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-$ROOT/cache/afd/deepep_jit}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$ROOT/cache/afd/deepgemm_jit}
export N2M_M2N_GIN_BUILD_DIR=${N2M_M2N_GIN_BUILD_DIR:-$ROOT/cache/afd/gin_comm}
export MINISGL_DEEPEP_BUILD_DIR=${MINISGL_DEEPEP_BUILD_DIR:-$ROOT/cache/afd/deepep_moe}
export MINISGL_DEEPGEMM_BUILD_DIR=${MINISGL_DEEPGEMM_BUILD_DIR:-$ROOT/cache/afd/deepgemm}
EXPERIMENT=$RUN_DIR/experiment
export RUN_DIR=$EXPERIMENT
mkdir -p "$RUN_DIR"
INPUTS=("$PRESET" "$SOURCE_REPO/$SOURCE_COORDINATOR_REL" \
    "$SOURCE_REPO/$SOURCE_PROFILER_REL" \
    "$SOURCE_REPO/python/minisgl/afd_scheduler.py" \
    "$SOURCE_REPO/$SOURCE_PLACEMENT_REL" \
    "$SOURCE_REPO/python/minisgl/afd_worker_launcher.py" \
    "$SOURCE_REPO/python/minisgl/server/supervisor_ray.py" \
    "$SOURCE_REPO/$SOURCE_PROTOCOL_REL" \
    "$SOURCE_REPO/$SOURCE_API_REL" \
    "$SOURCE_REPO/$SOURCE_ARGS_REL" \
    "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$PROMPT_MANIFEST" \
    "$MODEL_PATH/config.json" "$JOB_SCRIPT")
[[ -z "$EXPECTED_SOURCE_MANIFEST" ]] || INPUTS+=("$EXPECTED_SOURCE_MANIFEST")
[[ "$MODEL_PROFILE_ID" == native ]] || INPUTS+=("$MODEL_PROFILE_MANIFEST")
sha256sum "${INPUTS[@]}" > "$RUN_DIR/inputs.sha256"
cd "$SOURCE_REPO"
bash "$PRESET" > "$RUN_DIR/driver.stdout" 2> "$RUN_DIR/driver.stderr" &
DRIVER_PID=$!
WORKER_FAILURE_MARKER=$RUN_DIR/worker-hot-loop-failure.txt
POST_CAPTURE_DRIVER_TERMINATION_MARKER=$RUN_DIR/post-capture-driver-termination.txt
(
    capture_exit_deadline=0
    while kill -0 "$DRIVER_PID" 2>/dev/null; do
        if grep -q -m1 "zmq.error.ZMQError: Address already in use" \
            "$RUN_DIR/afd.log" 2>/dev/null; then
            {
                printf 'AFD control-port bind failed; source=%s detected_at=%s\n' \
                    "$RUN_DIR/afd.log" "$(date --iso-8601=seconds)"
                tail -n 120 "$RUN_DIR/afd.log"
            } > "$WORKER_FAILURE_MARKER"
            kill -TERM "$DRIVER_PID" 2>/dev/null || true
            exit 3
        fi
        offender=$(
            grep -l -m1 "hot_rpc_loop:error" "$RUN_DIR"/ray_logs/*.log \
                2>/dev/null | head -n 1 || true
        )
        if [[ -n "$offender" ]]; then
            {
                printf 'worker hot loop failed; source=%s detected_at=%s\n' \
                    "$offender" "$(date --iso-8601=seconds)"
                tail -n 120 "$offender"
            } > "$WORKER_FAILURE_MARKER"
            kill -TERM "$DRIVER_PID" 2>/dev/null || true
            break
        fi
        if [[ -s "$RUN_DIR/capture-complete.json" ]]; then
            if (( capture_exit_deadline == 0 )); then
                capture_exit_deadline=$((SECONDS + POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS))
            fi
            if [[ -s "$RUN_DIR/sample.json" && -s "$RUN_DIR/ray_logs/coordinator_nvtx_cpu.json" ]]; then
                printf 'terminating allocation-scoped driver immediately after capture, sample, and coordinator trace became durable; pid=%s detected_at=%s\n' \
                    "$DRIVER_PID" \
                    "$(date --iso-8601=seconds)" \
                    > "$POST_CAPTURE_DRIVER_TERMINATION_MARKER"
                kill -TERM "$DRIVER_PID" 2>/dev/null || true
                kill_deadline=$((SECONDS + POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS))
                while kill -0 "$DRIVER_PID" 2>/dev/null && (( SECONDS < kill_deadline )); do
                    sleep 1
                done
                kill -KILL "$DRIVER_PID" 2>/dev/null || true
                break
            elif (( SECONDS >= capture_exit_deadline )); then
                printf 'terminating allocation-scoped driver after completed capture failed to publish a sample and exit within %s seconds; pid=%s detected_at=%s\n' \
                    "$POST_CAPTURE_DRIVER_EXIT_TIMEOUT_SECONDS" "$DRIVER_PID" \
                    "$(date --iso-8601=seconds)" \
                    > "$POST_CAPTURE_DRIVER_TERMINATION_MARKER"
                kill -TERM "$DRIVER_PID" 2>/dev/null || true
                kill_deadline=$((SECONDS + POST_SAMPLE_DRIVER_TERM_TIMEOUT_SECONDS))
                while kill -0 "$DRIVER_PID" 2>/dev/null && (( SECONDS < kill_deadline )); do
                    sleep 1
                done
                kill -KILL "$DRIVER_PID" 2>/dev/null || true
                break
            fi
        fi
        sleep 2
    done
) &
DRIVER_WATCHDOG_PID=$!

# For a capacity-max point, hold the sampler until every attention worker has
# reported its actual post-init page count. The requested batch must equal the
# scheduler-admissible maximum under the allocation's minimum capacity.
if (( ! IRREGULAR && REQUIRE_CAPACITY_MAX )); then
    deadline=$((SECONDS + 900))
    while true; do
        observed_page_count=$(grep -c "AfdAttentionState allocating [0-9][0-9]* pages" "$RUN_DIR/afd.log" 2>/dev/null || true)
        if (( observed_page_count == ATTENTION_WORKERS )); then
            break
        fi
        if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
            set +e
            wait "$DRIVER_PID"
            driver_status=$?
            set -e
            if [[ -s "$WORKER_FAILURE_MARKER" ]]; then
                cat "$WORKER_FAILURE_MARKER" >&2
                exit 3
            fi
            (( driver_status != 0 )) || {
                echo "AFD driver exited without complete attention page-count evidence: $observed_page_count/$ATTENTION_WORKERS" >&2
                exit 3
            }
            exit "$driver_status"
        fi
        if (( SECONDS >= deadline )); then
            kill -TERM "$DRIVER_PID" 2>/dev/null || true
            set +e
            wait "$DRIVER_PID"
            set -e
            echo "timed out waiting for attention page-count evidence: $observed_page_count/$ATTENTION_WORKERS" >&2
            exit 3
        fi
        sleep 2
    done
    capacity_line=$("$PYTHON" - "$RUN_DIR/afd.log" "$RUN_DIR/capacity-preflight.json" \
        "$ATTENTION_WORKERS" "$CONTEXT" "$BATCH" "$OUTPUT_TOKENS" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
workers, context, requested, output_tokens = map(int, sys.argv[3:])
page_size = 64
pages = [
    int(value)
    for value in re.findall(
        r"AfdAttentionState allocating (\d+) pages",
        log_path.read_text(errors="replace"),
    )
]
if len(pages) != workers:
    raise RuntimeError(("attention page-count evidence", len(pages), workers))
capacity = min(pages) * page_size
per_sequence = ((context + output_tokens + page_size - 1) // page_size) * page_size
resident_prompt = ((context + page_size - 1) // page_size) * page_size
running_reservation = (output_tokens - 1) + (page_size - 1)
fresh_request = context + output_tokens
maximum = ((capacity - fresh_request) // (resident_prompt + running_reservation)) + 1
raw_maximum = capacity // per_sequence
admission_required = (
    (requested - 1) * (resident_prompt + running_reservation) + fresh_request
)
record = {
    "attention_workers": workers,
    "requested_batch_per_attention_gpu": requested,
    "observed_kv_pages_per_attention_gpu": pages,
    "observed_min_kv_capacity_tokens_per_attention_gpu": capacity,
    "observed_scheduler_admissible_max_batch": maximum,
    "observed_raw_kv_max_batch": raw_maximum,
    "requested_scheduler_admission_tokens": admission_required,
    "requested_batch_is_exact": requested == maximum,
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(f"{capacity}|{maximum}|{raw_maximum}|{admission_required}")
PY
    )
    IFS='|' read -r observed_capacity observed_max_batch observed_raw_max_batch observed_admission_required <<< "$capacity_line"
    printf 'observed_capacity_gate pages=%s workers=%s min_capacity_tokens=%s requested_batch=%s exact_max_batch=%s raw_max_batch=%s admission_required_tokens=%s\n' \
        "$observed_page_count" "$ATTENTION_WORKERS" "$observed_capacity" "$BATCH" \
        "$observed_max_batch" "$observed_raw_max_batch" "$observed_admission_required"
    if (( BATCH != observed_max_batch )); then
        kill -TERM "$DRIVER_PID" 2>/dev/null || true
        set +e
        wait "$DRIVER_PID"
        set -e
        echo "observed-capacity gate requires batch $observed_max_batch at ISL $CONTEXT under the allocation minimum $observed_capacity tokens, got $BATCH" >&2
        exit 3
    fi
    AFD_KV_CAPACITY_TOKENS=$observed_capacity
    CAPACITY_MAX_BATCH=$observed_max_batch
    RAW_KV_MAX_BATCH=$observed_raw_max_batch
    SCHEDULER_ADMISSION_REQUIRED_TOKENS=$observed_admission_required
    SCHEDULER_ADMISSION_UNUSED_TOKENS=$(( observed_capacity - observed_admission_required ))
fi

set +e
wait "$DRIVER_PID"
driver_status=$?
kill "$DRIVER_WATCHDOG_PID" 2>/dev/null || true
wait "$DRIVER_WATCHDOG_PID" 2>/dev/null
set -e
if [[ -s "$WORKER_FAILURE_MARKER" ]]; then
    cat "$WORKER_FAILURE_MARKER" >&2
    exit 3
fi
if (( driver_status != 0 )); then
    if [[ -s "$POST_CAPTURE_DRIVER_TERMINATION_MARKER" ]]; then
        cat "$POST_CAPTURE_DRIVER_TERMINATION_MARKER" >&2
    else
        exit "$driver_status"
    fi
fi
[[ -s "$RUN_DIR/sample.json" ]] || {
    echo "AFD driver completed without a nonempty sample.json" >&2
    exit 3
}
[[ -s "$RUN_DIR/ray_logs/coordinator_nvtx_cpu.json" ]] || {
    echo "AFD driver completed without a nonempty coordinator CPU trace; graceful shutdown may have timed out" >&2
    exit 3
}

RUN_DIR=$RUN_DIR SWEEP_CONTRACT=$SWEEP_CONTRACT NUM_PROMPTS=$NUM_PROMPTS BATCH=$BATCH \
AFD_MODEL_PLACEMENT_REQUESTED=$AFD_MODEL_PLACEMENT_REQUESTED \
AFD_MODEL_PLACEMENT_POLICY=$AFD_MODEL_PLACEMENT_POLICY \
AFD_MODEL_PLACEMENT=$AFD_MODEL_PLACEMENT AFD_NUM_MB=$AFD_NUM_MB \
ATTENTION_WORKERS=$ATTENTION_WORKERS ATTENTION_DP_SIZE=$ATTENTION_DP_SIZE \
ATTENTION_TP=$ATTENTION_TP FFN_EP=$FFN_EP NORMALIZED_AF_RATIO=$NORMALIZED_AF_RATIO \
NODES=$NODES REFERENCE_TPS=$REFERENCE_TPS \
CONTEXT=$CONTEXT CONTEXT_MIN=$CONTEXT_MIN CONTEXT_MAX=$CONTEXT_MAX \
IRREGULAR=$IRREGULAR PROMPT_LENGTHS_CSV=$PROMPT_LENGTHS_CSV \
PROMPT_LENGTH_SUM=$PROMPT_LENGTH_SUM AFD_REQUIRED_KV_TOKENS=$AFD_REQUIRED_KV_TOKENS \
AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS \
NOMINAL_AFD_KV_CAPACITY_TOKENS=$NOMINAL_AFD_KV_CAPACITY_TOKENS \
CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH \
NOMINAL_CAPACITY_MAX_BATCH=${NOMINAL_CAPACITY_MAX_BATCH:-$CAPACITY_MAX_BATCH} \
NOMINAL_RAW_KV_MAX_BATCH=${NOMINAL_RAW_KV_MAX_BATCH:-$RAW_KV_MAX_BATCH} \
SCHEDULER_ADMISSION_REQUIRED_TOKENS=$SCHEDULER_ADMISSION_REQUIRED_TOKENS \
SCHEDULER_ADMISSION_UNUSED_TOKENS=$SCHEDULER_ADMISSION_UNUSED_TOKENS \
SCHEDULER_RUNNING_RESERVATION_TOKENS=$SCHEDULER_RUNNING_RESERVATION_TOKENS \
REQUIRE_CAPACITY_MAX=$REQUIRE_CAPACITY_MAX ATTENTION_TRAYS=$ATTENTION_TRAYS \
FFN_TRAYS=$FFN_TRAYS ATTENTION_GPUS=$ATTENTION_GPUS FFN_GPUS=$FFN_GPUS \
ACTIVE_GPUS=$ACTIVE_GPUS ALLOCATED_GPUS=$ALLOCATED_GPUS \
SHARED_ROLE_TRAYS=$SHARED_ROLE_TRAYS \
PADDED_BATCH=$PADDED_BATCH PROMPT_MANIFEST=$PROMPT_MANIFEST \
AFD_MEMORY_RATIO=$AFD_MEMORY_RATIO \
AFD_NUM_PAGES=$AFD_NUM_PAGES \
OUTPUT_TOKENS=$OUTPUT_TOKENS TRACE_WARMUP_DECODE_STEPS=$TRACE_WARMUP_DECODE_STEPS \
"$PYTHON" - <<'PY'
import json, os, re, statistics
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
count, batch = int(os.environ["NUM_PROMPTS"]), int(os.environ["BATCH"])
padded_batch = int(os.environ["PADDED_BATCH"])
workers, nodes = int(os.environ["ATTENTION_WORKERS"]), int(os.environ["NODES"])
attention_dp = int(os.environ["ATTENTION_DP_SIZE"])
attention_tp = int(os.environ["ATTENTION_TP"])
ffn_ep = int(os.environ["FFN_EP"])
normalized_ratio = int(os.environ["NORMALIZED_AF_RATIO"])
reference = float(os.environ["REFERENCE_TPS"])
context = int(os.environ["CONTEXT"])
context_min, context_max = int(os.environ["CONTEXT_MIN"]), int(os.environ["CONTEXT_MAX"])
irregular = bool(int(os.environ["IRREGULAR"]))
prompt_lengths = [int(value) for value in os.environ["PROMPT_LENGTHS_CSV"].split(",")]
prompt_length_sum = int(os.environ["PROMPT_LENGTH_SUM"])
required_kv = int(os.environ["AFD_REQUIRED_KV_TOKENS"])
capacity_kv = int(os.environ["AFD_KV_CAPACITY_TOKENS"])
nominal_capacity_kv = int(os.environ["NOMINAL_AFD_KV_CAPACITY_TOKENS"])
capacity_max_batch = int(os.environ["CAPACITY_MAX_BATCH"])
raw_kv_max_batch = int(os.environ["RAW_KV_MAX_BATCH"])
nominal_capacity_max_batch = int(os.environ["NOMINAL_CAPACITY_MAX_BATCH"])
nominal_raw_kv_max_batch = int(os.environ["NOMINAL_RAW_KV_MAX_BATCH"])
admission_required = int(os.environ["SCHEDULER_ADMISSION_REQUIRED_TOKENS"])
admission_unused = int(os.environ["SCHEDULER_ADMISSION_UNUSED_TOKENS"])
running_reservation = int(os.environ["SCHEDULER_RUNNING_RESERVATION_TOKENS"])
require_capacity_max = bool(int(os.environ["REQUIRE_CAPACITY_MAX"]))
output_tokens = int(os.environ["OUTPUT_TOKENS"])
trace_warmup_steps = int(os.environ["TRACE_WARMUP_DECODE_STEPS"])
model_placement = os.environ["AFD_MODEL_PLACEMENT"]
model_placement_requested = os.environ["AFD_MODEL_PLACEMENT_REQUESTED"]
model_placement_policy = os.environ["AFD_MODEL_PLACEMENT_POLICY"]
num_mb = int(os.environ["AFD_NUM_MB"])
mb_base, mb_remainder = divmod(batch, num_mb)
if model_placement == "fmha-only":
    microbatch_real_sizes = [
        mb_base + (1 if mb >= num_mb - mb_remainder else 0)
        for mb in range(num_mb)
    ]
else:
    microbatch_real_sizes = [
        mb_base + (1 if mb < mb_remainder else 0)
        for mb in range(num_mb)
    ]
attention_trays = int(os.environ["ATTENTION_TRAYS"])
ffn_trays = int(os.environ["FFN_TRAYS"])
attention_gpus = int(os.environ["ATTENTION_GPUS"])
ffn_gpus = int(os.environ["FFN_GPUS"])
active_gpus = int(os.environ["ACTIVE_GPUS"])
allocated_gpus = int(os.environ["ALLOCATED_GPUS"])
shared_role_trays = int(os.environ["SHARED_ROLE_TRAYS"])
prompt_contract = json.loads(Path(os.environ["PROMPT_MANIFEST"]).read_text())
placement_path = run / "afd-placement.json"
if not placement_path.is_file():
    raise RuntimeError("missing explicit AFD placement record")
placement = json.loads(placement_path.read_text())
page_pattern = re.compile(r"AfdAttentionState allocating (\d+) pages")
observed_pages = []
with (run / "afd.log").open(encoding="utf-8", errors="replace") as afd_log:
    for line in afd_log:
        match = page_pattern.search(line)
        if match:
            observed_pages.append(int(match.group(1)))
if len(observed_pages) != workers:
    raise RuntimeError(("attention page-count evidence", len(observed_pages), workers))
observed_min_capacity = min(observed_pages) * 64
resident_prompt = ((context + 63) // 64) * 64
fresh_request = context + output_tokens
observed_max_batch = (
    (observed_min_capacity - fresh_request)
    // (resident_prompt + output_tokens - 1 + 63)
) + 1
if prompt_contract["per_attention_gpu_prompt_lengths"] != prompt_lengths:
    raise RuntimeError("prompt manifest distribution mismatch")
if prompt_contract["per_attention_gpu_prompt_length_sum"] != prompt_length_sum:
    raise RuntimeError("prompt manifest length sum mismatch")
if placement["attention_dp_size"] != attention_dp:
    raise RuntimeError(("placement attention DP", placement, attention_dp))
if placement["attention_tp_size"] != attention_tp:
    raise RuntimeError(("placement attention TP", placement, attention_tp))
if placement["ffn_ep_size"] != ffn_ep:
    raise RuntimeError(("placement FFN EP", placement, ffn_ep))
if len(placement["allocated_nodes"]) != nodes:
    raise RuntimeError(("placement allocated nodes", placement, nodes))
attention_rows = placement["attention_dp_rank_nodes"]
mlp_rows = placement["mlp_dp_rank_nodes"]
if len(attention_rows) != attention_dp or any(
    len(row) != attention_tp for row in attention_rows
):
    raise RuntimeError(("placement attention rows", placement, attention_dp, attention_tp))
if len(mlp_rows) != ffn_ep or any(len(row) != 1 for row in mlp_rows):
    raise RuntimeError(("placement FFN rows", placement, ffn_ep))
allocated_nodes = placement["allocated_nodes"]
allocated_node_set = set(allocated_nodes)
role_nodes = [node for row in attention_rows + mlp_rows for node in row]
if any(node not in allocated_node_set for node in role_nodes):
    raise RuntimeError(("placement references unallocated node", placement))
node_role_counts = {node: role_nodes.count(node) for node in allocated_nodes}
if any(count > 4 for count in node_role_counts.values()):
    raise RuntimeError(("placement overcommits a four-GPU tray", node_role_counts))
if sum(node_role_counts.values()) != active_gpus:
    raise RuntimeError(("placement active-GPU count", node_role_counts, active_gpus))
if required_kv > capacity_kv:
    raise RuntimeError((required_kv, capacity_kv))
if require_capacity_max and capacity_max_batch != batch:
    raise RuntimeError(("not capacity-max batch", batch, capacity_max_batch))
if require_capacity_max and observed_max_batch != batch:
    raise RuntimeError(("not observed scheduler-admissible max batch", batch, observed_max_batch))
samples = json.loads((run / "sample.json").read_text()).get("samples", [])
lengths = [len(x.get("generated_token_ids", [])) for x in samples]
if len(samples) != count or set(lengths) != {output_tokens}:
    raise RuntimeError((len(samples), sorted(set(lengths))))
if model_placement == "fmha-only":
    attention_trace_pattern = (
        "attention_dp*_rank*_nvtx_cpu.json"
        if attention_dp > 1
        else "attention_rank*_nvtx_cpu.json"
    )
    attention_traces = sorted((run / "ray_logs").glob(attention_trace_pattern))
    if len(attention_traces) != workers:
        raise RuntimeError(("attention CPU traces", len(attention_traces), workers))
    worker_steps = []
    for path in attention_traces:
        events = json.loads(path.read_text()).get("events", [])
        worker_steps.append([
            int(event["step_id"])
            for event in events
            if str(event.get("name", "")).startswith("AFD_AG_Materialize")
            and isinstance(event.get("step_id"), int)
        ])
else:
    attention_log_pattern = (
        "attention_dp*_rank*.log" if attention_dp > 1 else "attention_rank*.log"
    )
    logs = sorted((run / "ray_logs").glob(attention_log_pattern))
    if len(logs) != workers:
        raise RuntimeError((len(logs), workers))
    pattern = re.compile(
        r"afd_ag_decode_graph:replay step_id=(\d+) bs=(\d+) num_mb=(\d+)"
    )
    worker_steps = []
    for path in logs:
        captured = [
            int(s)
            for s, b, marker in pattern.findall(path.read_text(errors="replace"))
            if int(b) == padded_batch and int(marker) == num_mb
        ]
        worker_steps.append(captured)
coordinator_log = run / "ray_logs/afd_coordinator.log"
if not coordinator_log.is_file():
    raise RuntimeError("missing AFD coordinator log")
window_pattern = re.compile(
    r"nsys profiler:target_decode_window target_batch_per_dp=(\d+) "
    r"warmup_step_id=(\d+) step_ids=([0-9,]+) "
    r"count=(\d+) trace_count=(\d+)"
)
window_matches = window_pattern.findall(coordinator_log.read_text(errors="replace"))
if len(window_matches) != 1:
    raise RuntimeError(("expected one target decode-window marker", window_matches))
(
    target_batch_text,
    warmup_step_text,
    step_ids_text,
    step_count_text,
    trace_count_text,
) = window_matches[0]
warmup_step = int(warmup_step_text)
steps = [int(value) for value in step_ids_text.split(",")]
if (
    int(target_batch_text) != batch
    or int(step_count_text) != 15
    or int(trace_count_text) != 16
    or trace_warmup_steps != 1
):
    raise RuntimeError(("target decode-window contract", window_matches[0], batch))
if len(steps) != 15 or steps != list(range(steps[0], steps[0] + 15)):
    raise RuntimeError(("target decode steps are not exactly consecutive", steps))
trace_steps = [warmup_step, *steps]
if model_placement == "fmha-only":
    trace_step_set = set(trace_steps)
    worker_trace_steps = [
        [step for step in captured if step in trace_step_set]
        for captured in worker_steps
    ]
else:
    worker_trace_steps = worker_steps
if steps[0] != warmup_step + 1 or any(
    captured != trace_steps for captured in worker_trace_steps
):
    raise RuntimeError(
        (
            "attention worker trace differs from warmup+target window",
            worker_trace_steps,
            trace_steps,
        )
    )
full_windows = [steps]
partial_windows = []
trace = json.loads((run / "ray_logs/coordinator_nvtx_cpu.json").read_text())
complete = {
    e["step_id"]: e["end_perf_ns"] for e in trace["events"]
    if str(e.get("name", "")).startswith("AFD_Coordinator_CompleteCollect")
    and isinstance(e.get("step_id"), int)
}
ends = [complete[s] for s in steps]
intervals = [(b - a) / 1e6 for a, b in zip(ends, ends[1:])]
median_ms, mean_ms = statistics.median(intervals), statistics.fmean(intervals)
tps = count * 1000 / (active_gpus * median_ms)
allocated_tps = count * 1000 / (allocated_gpus * median_ms)
result = {
    "sweep_contract": os.environ["SWEEP_CONTRACT"],
    "afd_model_placement": model_placement,
    "afd_model_placement_requested": model_placement_requested,
    "afd_model_placement_policy": model_placement_policy,
    "model_key": os.environ.get("MODEL_KEY"),
    "prompt_mode": os.environ["PROMPT_MODE"],
    "context_tokens": context,
    "context_range_tokens": {"min": context_min, "max": context_max},
    "irregular_prompt_lengths": irregular,
    "prompt_length_distribution": "inclusive_uniform_symmetric_integer_linspace",
    "prompt_lengths_per_attention_gpu": prompt_lengths,
    "prompt_length_sum_per_attention_gpu": prompt_length_sum,
    "samples": count, "tokens_per_sample": output_tokens,
    "batch_per_attention_dp_lane": batch,
    "input_batch_per_attention_dp_lane": batch,
    "batch_per_attention_gpu": batch,
    "input_batch_per_attention_gpu": batch,
    "afd_memory_ratio": float(os.environ["AFD_MEMORY_RATIO"]),
    "afd_num_pages_override": (
        int(os.environ["AFD_NUM_PAGES"])
        if os.environ.get("AFD_NUM_PAGES")
        else None
    ),
    "batch_selection": (
        "exact scheduler-admissible ceiling under measured KV capacity"
        if require_capacity_max
        else "explicit sweep point; post-init KV capacity reported separately"
    ),
    "capacity_max_batch": capacity_max_batch,
    "raw_kv_capacity_max_batch": raw_kv_max_batch,
    "nominal_kv_capacity_tokens_per_attention_gpu": nominal_capacity_kv,
    "nominal_scheduler_admissible_max_batch": nominal_capacity_max_batch,
    "nominal_raw_kv_capacity_max_batch": nominal_raw_kv_max_batch,
    "observed_kv_pages_per_attention_gpu": observed_pages,
    "observed_min_kv_capacity_tokens_per_attention_gpu": observed_min_capacity,
    "observed_scheduler_admissible_max_batch": observed_max_batch,
    "scheduler_running_reservation_tokens_per_request": running_reservation,
    "scheduler_admission_required_tokens_per_attention_gpu": admission_required,
    "scheduler_admission_unused_tokens_per_attention_gpu": admission_unused,
    "scheduler_admission_method": f"largest B satisfying (B-1)*(ceil(prompt/64)*64 + ({output_tokens - 1}+63)) + (prompt+{output_tokens}) <= measured KV capacity",
    "microbatch_real_sizes_per_attention_dp_lane": microbatch_real_sizes,
    "graph_padded_batch_per_attention_dp_lane": padded_batch,
    "microbatch_real_sizes_per_attention_gpu": microbatch_real_sizes,
    "graph_padded_batch_per_attention_gpu": padded_batch,
    "attention_workers": workers,
    "attention_dp_size": attention_dp,
    "attention_tp_size": attention_tp,
    "ffn_ep_size": ffn_ep,
    "total_gpus": active_gpus,
    "active_gpus": active_gpus,
    "allocated_gpus": allocated_gpus,
    "allocated_trays": nodes,
    "idle_allocated_gpus": allocated_gpus - active_gpus,
    "attention_gpus": attention_gpus,
    "ffn_gpus": ffn_gpus,
    "shared_role_trays": shared_role_trays,
    "attention_trays": attention_trays, "ffn_trays": ffn_trays,
    "attention_to_ffn_tray_ratio": f"{attention_trays}:{ffn_trays}",
    "attention_to_ffn_active_gpu_ratio": f"{attention_gpus}:{ffn_gpus}",
    "normalized_attention_to_ffn_tray_ratio": f"{normalized_ratio}:1",
    "normalized_attention_to_ffn_active_gpu_ratio": f"{normalized_ratio}:1",
    "kv_capacity_tokens_per_attention_gpu": capacity_kv,
    "required_kv_tokens_per_attention_gpu": required_kv,
    "unused_kv_tokens_per_attention_gpu": capacity_kv - required_kv,
    "kv_preflight_method": "runtime evidence from every attention worker; capacity-max mode gates sampling on the observed minimum page count",
    "full_bucket_decode_windows": full_windows,
    "partial_padded_batch_decode_windows": partial_windows,
    "decode_step_ids": steps,
    "warmup_decode_step_id": warmup_step,
    "trace_decode_step_ids": trace_steps,
    "selected_window_policy": "trace first exact full-resident decode as warmup, then measure the next exact 15-step decode window",
    "completion_interval_ms": intervals,
    "mean_interval_ms": mean_ms, "median_interval_ms": median_ms,
    "median_tokens_per_second_per_gpu": tps,
    "coordinator_median_tokens_per_second_per_active_gpu": tps,
    "coordinator_median_tokens_per_second_per_allocated_gpu": allocated_tps,
    "prompt_contract": prompt_contract,
    "placement": placement,
    "cuda_graph": {
        "enabled": True,
        "num_microbatches": num_mb,
        "configured_batch_per_microbatch": (batch + num_mb - 1) // num_mb,
        "observed_padded_batch_per_attention_gpu": padded_batch,
        "nsys_cuda_graph_trace": os.environ["NSYS_CUDA_GRAPH_TRACE"],
        "nsys_target_batch_per_attention_dp_lane": int(os.environ["NSYS_TARGET_BATCH_PER_ATTN_DP"]),
        "nsys_capture_decode_steps": int(os.environ["NSYS_CAPTURE_DECODE_STEPS"]),
        "nsys_trace_warmup_decode_steps": trace_warmup_steps,
        "nsys_trace_decode_launches": len(trace_steps),
        "proof": "the coordinator emitted one exact warmup-plus-target_decode_window marker and every attention-worker log contains the first full-resident warmup followed by exactly the same 15 measured decode-graph replay steps",
    },
    "measurement": (
        "coordinator completion interval diagnostic only; final latency and "
        "TPS/GPU use the arithmetic mean of the dominant range among the 15 "
        "target-batch post-warmup critical CUDA spans, with its median also "
        "reported; FMHA-only placement takes the maximum complete graph span "
        "across one attention and one model representative"
    ),
    "primary_performance_metric": {
        "status": "pending_postprocess",
        "metric_version": (
            "20260820-fmha-only-dual-role-cuda-graph-span-v1"
            if model_placement == "fmha-only"
            else "20260804-attention-cuda-execution-span-v14"
        ),
        "latency_basis": (
            "max_complete_cuda_graph_span_across_profiled_attention_and_model_roles"
            if model_placement == "fmha-only"
            else "afd_attention_cuda_execution_critical_target_batch_dominant_range_mean"
        ),
        "required_target_batch_selected_steps": 15,
        "minimum_retained_latency_sample_count": 10,
        "maximum_outlier_count": 5,
        "dominant_range_percent_limit": 10.0,
        "dominant_range_tie_break": "largest cluster, then tightest range, then higher mean",
        "max_median_diff_percent_limit": 10.0,
        "tps_denominator": "active_gpus",
    },
}
if reference > 0:
    result["reference_tokens_per_second_per_gpu"] = reference
    result["reference_delta_percent"] = (tps / reference - 1) * 100
(run / "afd-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

if [[ "$SWEEP_CONTRACT" == comprehensive ]]; then
    if (( IRREGULAR )); then
        CASE_ID=r${CONTEXT_MIN}-${CONTEXT_MAX}-fep${FFN_EP}-r${NORMALIZED_AF_RATIO}-atp${ATTENTION_TP}-b${BATCH}
    else
        CASE_ID=i${CONTEXT}-fep${FFN_EP}-r${NORMALIZED_AF_RATIO}-atp${ATTENTION_TP}-b${BATCH}
    fi
    CUDA_CASE_TEMP_ROOT=$CUDA_EXTRACT_TEMP_ROOT/$SLURM_JOB_ID-$CASE_ORDINAL
    mkdir -p "$CUDA_METRICS_ROOT" "$CUDA_CASE_TEMP_ROOT"
    "$PYTHON" "$CUDA_EXTRACT_SCRIPT" \
        --result "$RUN_DIR/afd-result.json" \
        --plan "$CUDA_METRIC_PLAN" \
        --nsys /usr/local/cuda/bin/nsys \
        --cuda-span-module "$CUDA_SPAN_MODULE" \
        --temp-root "$CUDA_CASE_TEMP_ROOT" \
        --output "$CUDA_METRICS_ROOT/$CASE_ID.json"
fi

touch "$SNAPSHOT"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$EXPERIMENT/../gpu-snapshots/final-rank-0.csv"
touch "$DONE/0"
deadline=$((SECONDS + 120))
until [[ $(find "$DONE" -maxdepth 1 -type f | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $deadline ]]; sleep 2
done
for ((rank=0; rank<NODES; rank++)); do
    # This snapshot is taken before the allocation-scoped EXIT trap stops Ray,
    # so surviving worker rows are cleanup diagnostics, not evidence that the
    # already extracted strict metric is invalid. The next case's initial
    # snapshot remains the fail-fast proof that cleanup actually completed.
    [[ -s "$EXPERIMENT/../gpu-snapshots/final-rank-$rank.csv" ]]
done
printf 'FASTAFD_AFD_SUCCESS result=%s\n' "$EXPERIMENT/afd-result.json" | tee "$EXPERIMENT/SUCCESS"
