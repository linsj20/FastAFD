#!/usr/bin/env bash
# Validate attn-only BF16/FP8 transport publication.
# Required: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV OUTPUT_ROOT

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
#SBATCH --job-name=attn-transport-smoke

set -euo pipefail

: "${FASTAFD_SOURCE_REPO:?}" "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" \
  "${OUTPUT_ROOT:?}"
RANKS=${AFD_FABRIC_SMOKE_RANKS:-4}
[[ "$RANKS" == 2 || "$RANKS" == 4 ]]

CACHE_ROOT=${MINISGL_DEEPGEMM_BUILD_DIR:-$OUTPUT_ROOT/cache/deepgemm}
DEEPEP_CACHE_ROOT=${MINISGL_DEEPEP_BUILD_DIR:-$OUTPUT_ROOT/cache/deepep_moe}
TVM_CACHE_ROOT=${TVM_FFI_CACHE_DIR:-$OUTPUT_ROOT/cache/tvm_ffi}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
COMMON_SRUN=(
  --nodes=1
  --container-image="$FASTAFD_IMAGE"
  --container-mount-home
  --container-mounts=/lustre:/lustre
  --no-container-remap-root
  --kill-on-bad-exit=1
)
COMMON_ENV=(
  env -u CUDA_VISIBLE_DEVICES
  PYTHONPATH="$FASTAFD_SOURCE_REPO/python:$FASTAFD_SOURCE_REPO"
  MINISGL_DEEPGEMM_BUILD_DIR="$CACHE_ROOT"
  MINISGL_DEEPEP_BUILD_DIR="$DEEPEP_CACHE_ROOT"
  TVM_FFI_CACHE_DIR="$TVM_CACHE_ROOT"
)
mkdir -p "$CACHE_ROOT" "$DEEPEP_CACHE_ROOT" "$TVM_CACHE_ROOT" \
  "$OUTPUT_ROOT/logs"

# Build the transport JIT once before its four ranks exchange fabric handles.
srun "${COMMON_SRUN[@]}" --ntasks=1 --ntasks-per-node=1 \
  "${COMMON_ENV[@]}" AFD_FABRIC_SMOKE_PREBUILD=1 \
  "$FASTAFD_VENV/bin/python" \
  "$FASTAFD_SOURCE_REPO/tests/afd_fabric_transport_smoke.py"

# Exercise repeated turn/counter reuse while checking remote FP8 bytes/scales.
srun "${COMMON_SRUN[@]}" --ntasks="$RANKS" --ntasks-per-node="$RANKS" \
  "${COMMON_ENV[@]}" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=29671 \
  AFD_FABRIC_SMOKE_ROWS="${AFD_FABRIC_SMOKE_ROWS:-3}" \
  AFD_FABRIC_SMOKE_ITERATIONS="${AFD_FABRIC_SMOKE_ITERATIONS:-9}" \
  "$FASTAFD_VENV/bin/python" \
  "$FASTAFD_SOURCE_REPO/tests/afd_fabric_transport_smoke.py"
