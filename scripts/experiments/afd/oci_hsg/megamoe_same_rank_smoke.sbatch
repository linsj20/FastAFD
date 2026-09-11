#!/usr/bin/env bash
# Reusable four-GPU FP8 same-rank MegaMoE correctness/exact-shape smoke.
# Required: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV OUTPUT_ROOT
# Optional: MINISGL_DEEPGEMM_BUILD_DIR, DG_JIT_CACHE_DIR, TVM_FFI_CACHE_DIR, MEGAMOE_SMOKE_ROWS,
#           MEGAMOE_SMOKE_SMS, MEGAMOE_SMOKE_BUCKET, MEGAMOE_SMOKE_WARMUPS,
#           MEGAMOE_SMOKE_ITERATIONS, MEGAMOE_SMOKE_CONTEND_CYCLES
#           MEGAMOE_SMOKE_DEBUG_TIMINGS, MEGAMOE_SMOKE_ROUTE_PREPARE_MODES,
#           MEGAMOE_SMOKE_PROFILE_ONLY, MEGAMOE_SMOKE_FIXED_PERFECT_ONLY,
#           MEGAMOE_SMOKE_PRODUCTION_ROUTES,
#           MEGAMOE_SMOKE_TEST_SCRIPT,
#           MEGAMOE_SMOKE_MEMCHECK (1 enables per-rank Compute Sanitizer logs),
#           MEGAMOE_SMOKE_GPU_HEALTH (1 records nvidia-smi XML per node),
#           MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX,
#           MEGAMOE_SMOKE_COMBINE_MODES,
#           MEGAMOE_SMOKE_ROUTE_READY_DISPATCH_MODES,
#           MEGAMOE_SMOKE_SPECIALIZATION_MODES[_SBATCH],
#           MEGAMOE_SMOKE_NUM_EXPERTS, MEGAMOE_SMOKE_HIDDEN,
#           MEGAMOE_SMOKE_INTERMEDIATE, MEGAMOE_SMOKE_TOP_K,
#           MEGAMOE_SMOKE_PERFECT_ROUTING, MEGAMOE_NUMERIC_ROWS,
#           MEGAMOE_NUMERIC_BUCKETS, MEGAMOE_NUMERIC_NUM_EXPERTS,
#           MEGAMOE_NUMERIC_HIDDEN, MEGAMOE_NUMERIC_INTERMEDIATE,
#           MEGAMOE_NUMERIC_TOP_K
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --segment=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=megamoe-same-rank-smoke

set -euo pipefail

: "${FASTAFD_SOURCE_REPO:?}" "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" \
  "${OUTPUT_ROOT:?}"

if [[ ${MEGAMOE_SMOKE_RANK_MODE:-} == run ]]; then
    rank=${SLURM_PROCID:?}
    case "${MEGAMOE_SMOKE_GPU_HEALTH:-0}" in
        0) ;;
        1)
            if [[ ${SLURM_LOCALID:?} == 0 ]]; then
                mkdir -p "$OUTPUT_ROOT/logs"
                nvidia-smi -q -x > "$OUTPUT_ROOT/logs/gpu-health-${SLURMD_NODENAME:?}.xml"
            fi
            ;;
        *) echo "MEGAMOE_SMOKE_GPU_HEALTH must be 0 or 1" >&2; exit 2 ;;
    esac
    python_cmd=(
        "$FASTAFD_VENV/bin/python"
        "${MEGAMOE_SMOKE_TEST_SCRIPT:-$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py}"
    )
    case "${MEGAMOE_SMOKE_MEMCHECK:-0}" in
        0) ;;
        1)
            [[ -z ${MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX:-} ]] || {
                echo "Choose either memcheck or Nsight capture" >&2
                exit 2
            }
            sanitizer=/usr/local/cuda/bin/compute-sanitizer
            [[ -x $sanitizer ]] || {
                echo "Compute Sanitizer is missing: $sanitizer" >&2
                exit 2
            }
            mkdir -p "$OUTPUT_ROOT/logs"
            exec "$sanitizer" --tool memcheck --error-exitcode 99 \
                --log-file "$OUTPUT_ROOT/logs/memcheck-rank$rank.log" \
                "${python_cmd[@]}"
            ;;
        *) echo "MEGAMOE_SMOKE_MEMCHECK must be 0 or 1" >&2; exit 2 ;;
    esac
    if [[ -n ${MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX:-} &&
          $rank == ${MEGAMOE_SMOKE_NSYS_PROFILE_RANK:-0} ]]; then
        nsys=/usr/local/cuda/bin/nsys
        [[ -x $nsys ]] || {
            echo "Nsight Systems is missing: $nsys" >&2
            exit 2
        }
        mkdir -p "$(dirname "$MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX")" \
            "/dev/shm/fastafd-megamoe-nsys-$SLURM_JOB_ID-$rank"
        export TMPDIR=/dev/shm/fastafd-megamoe-nsys-$SLURM_JOB_ID-$rank
        export MEGAMOE_SMOKE_CUDA_PROFILER_RANGE=1
        report_prefix=$MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX-rank$rank
        "$nsys" profile --trace=cuda --sample=none --cpuctxsw=none \
            --cuda-graph-trace=node --cuda-event-trace=false \
            --capture-range=cudaProfilerApi --capture-range-end=stop \
            --force-overwrite=true --output="$report_prefix" \
            "${python_cmd[@]}"
        "$nsys" stats --force-export=true --report cuda_gpu_kern_sum \
            --format csv "$report_prefix.nsys-rep" \
            > "$report_prefix.cuda_gpu_kern_sum.csv"
        exit 0
    fi
    exec "${python_cmd[@]}"
fi

CACHE_ROOT=${MINISGL_DEEPGEMM_BUILD_DIR:-$OUTPUT_ROOT/cache/deepgemm}
TVM_CACHE_ROOT=${TVM_FFI_CACHE_DIR:-$OUTPUT_ROOT/cache/tvm_ffi}
JOB_SCRIPT=${MEGAMOE_SMOKE_JOB_SCRIPT:-$(realpath "$0")}
[[ -f $JOB_SCRIPT ]] || {
    echo "MegaMoE smoke job script is missing: $JOB_SCRIPT" >&2
    exit 2
}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
MASTER_PORT=${MASTER_PORT:-29573}
TASKS_PER_NODE=$((SLURM_NTASKS / SLURM_NNODES))
if (( TASKS_PER_NODE * SLURM_NNODES != SLURM_NTASKS )); then
    echo "MegaMoE smoke requires equal task counts per node" >&2
    exit 2
fi
SPECIALIZATION_MODES=${MEGAMOE_SMOKE_SPECIALIZATION_MODES:-}
if [[ -n ${MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH:-} ]]; then
    SPECIALIZATION_MODES=${MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH//\//,}
fi

mkdir -p "$CACHE_ROOT" "$TVM_CACHE_ROOT" "$OUTPUT_ROOT/logs"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NTASKS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    --kill-on-bad-exit=1 \
    --container-image="$FASTAFD_IMAGE" --container-mount-home \
    --container-mounts=/lustre:/lustre --no-container-remap-root \
    env -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$FASTAFD_SOURCE_REPO/python" \
    MINISGL_DEEPGEMM_BUILD_DIR="$CACHE_ROOT" \
    DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$CACHE_ROOT/jit}" \
    TVM_FFI_CACHE_DIR="$TVM_CACHE_ROOT" \
    MEGAMOE_SMOKE_ROWS="${MEGAMOE_SMOKE_ROWS:-3}" \
    MEGAMOE_SMOKE_SMS="${MEGAMOE_SMOKE_SMS:-152}" \
    MEGAMOE_SMOKE_WARMUPS="${MEGAMOE_SMOKE_WARMUPS:-10}" \
    MEGAMOE_SMOKE_ITERATIONS="${MEGAMOE_SMOKE_ITERATIONS:-100}" \
    MEGAMOE_SMOKE_DEBUG_TIMINGS="${MEGAMOE_SMOKE_DEBUG_TIMINGS:-0}" \
    MEGAMOE_SMOKE_PROFILE_ONLY="${MEGAMOE_SMOKE_PROFILE_ONLY:-0}" \
    MEGAMOE_SMOKE_FIXED_PERFECT_ONLY="${MEGAMOE_SMOKE_FIXED_PERFECT_ONLY:-0}" \
    MEGAMOE_SMOKE_PRODUCTION_ROUTES="${MEGAMOE_SMOKE_PRODUCTION_ROUTES:-0}" \
    MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX="${MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX:-}" \
    MEGAMOE_SMOKE_NSYS_PROFILE_RANK="${MEGAMOE_SMOKE_NSYS_PROFILE_RANK:-0}" \
    MEGAMOE_SMOKE_ROUTE_PREPARE_MODES="${MEGAMOE_SMOKE_ROUTE_PREPARE_MODES:-0}" \
    MEGAMOE_SMOKE_COMBINE_MODES="${MEGAMOE_SMOKE_COMBINE_MODES:-0}" \
    MEGAMOE_SMOKE_ROUTE_READY_DISPATCH_MODES="${MEGAMOE_SMOKE_ROUTE_READY_DISPATCH_MODES:-0}" \
    MEGAMOE_SMOKE_SPECIALIZATION_MODES="$SPECIALIZATION_MODES" \
    MEGAMOE_SMOKE_NUM_EXPERTS="${MEGAMOE_SMOKE_NUM_EXPERTS:-128}" \
    MEGAMOE_SMOKE_HIDDEN="${MEGAMOE_SMOKE_HIDDEN:-4096}" \
    MEGAMOE_SMOKE_INTERMEDIATE="${MEGAMOE_SMOKE_INTERMEDIATE:-1536}" \
    MEGAMOE_SMOKE_TOP_K="${MEGAMOE_SMOKE_TOP_K:-8}" \
    MEGAMOE_SMOKE_PERFECT_ROUTING="${MEGAMOE_SMOKE_PERFECT_ROUTING:-0}" \
    MEGAMOE_NUMERIC_ROWS="${MEGAMOE_NUMERIC_ROWS:-3,6}" \
    MEGAMOE_NUMERIC_BUCKETS="${MEGAMOE_NUMERIC_BUCKETS:-518,1036}" \
    MEGAMOE_NUMERIC_NUM_EXPERTS="${MEGAMOE_NUMERIC_NUM_EXPERTS:-$((2 * SLURM_NTASKS))}" \
    MEGAMOE_NUMERIC_HIDDEN="${MEGAMOE_NUMERIC_HIDDEN:-1024}" \
    MEGAMOE_NUMERIC_INTERMEDIATE="${MEGAMOE_NUMERIC_INTERMEDIATE:-512}" \
    MEGAMOE_NUMERIC_TOP_K="${MEGAMOE_NUMERIC_TOP_K:-2}" \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    MEGAMOE_SMOKE_RANK_MODE=run \
    bash "$JOB_SCRIPT"
