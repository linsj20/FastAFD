#!/bin/bash
# Build the host DeepGEMM extension once before multi-rank GPU validation.
# Required: FASTAFD_SOURCE_REPO, FASTAFD_IMAGE, FASTAFD_VENV, OUTPUT_ROOT.
# Optional: MINISGL_DEEPGEMM_BUILD_DIR.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=fastafd:deepgemm-prebuild

set -euo pipefail

: "${FASTAFD_SOURCE_REPO:?FASTAFD_SOURCE_REPO is required}"
: "${FASTAFD_IMAGE:?FASTAFD_IMAGE is required}"
: "${FASTAFD_VENV:?FASTAFD_VENV is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

CACHE_ROOT=${MINISGL_DEEPGEMM_BUILD_DIR:-$OUTPUT_ROOT/cache/deepgemm}
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT/logs"

srun --nodes=1 --ntasks=1 \
    --container-image="$FASTAFD_IMAGE" --container-mount-home \
    --container-mounts=/lustre:/lustre --no-container-remap-root \
    env -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$FASTAFD_SOURCE_REPO/python" \
    MINISGL_DEEPGEMM_BUILD_DIR="$CACHE_ROOT" \
    "$FASTAFD_VENV/bin/python" -c \
    'from minisgl.kernel import deepgemm; deepgemm._load_extension()'
