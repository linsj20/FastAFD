#!/usr/bin/env bash
# Reusable four-GPU FP8 same-rank MegaMoE correctness/exact-shape smoke.
# Required: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV OUTPUT_ROOT
# Optional: MINISGL_DEEPGEMM_BUILD_DIR, TVM_FFI_CACHE_DIR, MEGAMOE_SMOKE_ROWS,
#           MEGAMOE_SMOKE_SMS, MEGAMOE_SMOKE_BUCKET, MEGAMOE_SMOKE_WARMUPS,
#           MEGAMOE_SMOKE_ITERATIONS, MEGAMOE_SMOKE_CONTEND_CYCLES
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

CACHE_ROOT=${MINISGL_DEEPGEMM_BUILD_DIR:-$OUTPUT_ROOT/cache/deepgemm}
TVM_CACHE_ROOT=${TVM_FFI_CACHE_DIR:-$OUTPUT_ROOT/cache/tvm_ffi}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
MASTER_PORT=${MASTER_PORT:-29573}
mkdir -p "$CACHE_ROOT" "$TVM_CACHE_ROOT" "$OUTPUT_ROOT/logs"

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 \
    --kill-on-bad-exit=1 \
    --container-image="$FASTAFD_IMAGE" --container-mount-home \
    --container-mounts=/lustre:/lustre --no-container-remap-root \
    env -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$FASTAFD_SOURCE_REPO/python" \
    MINISGL_DEEPGEMM_BUILD_DIR="$CACHE_ROOT" \
    TVM_FFI_CACHE_DIR="$TVM_CACHE_ROOT" \
    MEGAMOE_SMOKE_ROWS="${MEGAMOE_SMOKE_ROWS:-3}" \
    MEGAMOE_SMOKE_SMS="${MEGAMOE_SMOKE_SMS:-152}" \
    MEGAMOE_SMOKE_WARMUPS="${MEGAMOE_SMOKE_WARMUPS:-10}" \
    MEGAMOE_SMOKE_ITERATIONS="${MEGAMOE_SMOKE_ITERATIONS:-100}" \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$FASTAFD_VENV/bin/python" \
    "$FASTAFD_SOURCE_REPO/tests/afd_megamoe_same_rank_smoke.py"
