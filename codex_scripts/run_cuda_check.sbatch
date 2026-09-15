#!/usr/bin/env bash
# Reusable short CUDA check in the project's execution container.
# Required: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV FASTAFD_CHECK_SCRIPT
#           OUTPUT_ROOT FASTAFD_CHECK_JOB_SCRIPT
# Script arguments are forwarded, e.g. --rows 19 --bench-iters 500 for GDN.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=fastafd:cuda-check
set -euo pipefail
: "${FASTAFD_SOURCE_REPO:?}" "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" \
  "${FASTAFD_CHECK_SCRIPT:?}" "${OUTPUT_ROOT:?}" "${FASTAFD_CHECK_JOB_SCRIPT:?}"
if [[ ${FASTAFD_CHECK_IN_CONTAINER:-0} != 1 ]]; then
    exec srun --ntasks=1 --kill-on-bad-exit=1 \
        --container-image="$FASTAFD_IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env FASTAFD_CHECK_IN_CONTAINER=1 bash "$FASTAFD_CHECK_JOB_SCRIPT" "$@"
fi
export PATH="$FASTAFD_VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$FASTAFD_SOURCE_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export TVM_FFI_CACHE_DIR="$OUTPUT_ROOT/cache/tvm_ffi"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/cache/triton"
export MINISGL_DEEPGEMM_BUILD_DIR=${MINISGL_DEEPGEMM_BUILD_DIR:-$OUTPUT_ROOT/cache/deepgemm}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$OUTPUT_ROOT/cache/deepgemm_jit}
mkdir -p "$TVM_FFI_CACHE_DIR" "$TRITON_CACHE_DIR" "$MINISGL_DEEPGEMM_BUILD_DIR" "$DG_JIT_CACHE_DIR"
exec "$FASTAFD_VENV/bin/python" "$FASTAFD_CHECK_SCRIPT" "$@"
