#!/usr/bin/env bash
# Byte-exact comparison: record control, repeat control, then each candidate.
# Required: FASTAFD_COMPARE_SOURCES (colon-separated, control first),
# FASTAFD_IMAGE, FASTAFD_VENV, OUTPUT_ROOT, MEGAMOE_SMOKE_JOB_SCRIPT,
# MEGAMOE_SMOKE_TEST_SCRIPT. Shape variables use the standard smoke interface.
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
#SBATCH --job-name=megamoe-result-compare

set -euo pipefail
: "${FASTAFD_COMPARE_SOURCES:?}" "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" \
  "${OUTPUT_ROOT:?}" "${MEGAMOE_SMOKE_JOB_SCRIPT:?}" "${MEGAMOE_SMOKE_TEST_SCRIPT:?}"
IFS=: read -r -a sources <<< "$FASTAFD_COMPARE_SOURCES"
(( ${#sources[@]} >= 2 )) || { echo 'at least two sources required' >&2; exit 2; }
mapfile -t blocks < <(scontrol show hostnames "$SLURM_JOB_NODELIST" |
    sed -E 's/-T[0-9]+$//' | sort -u)
(( ${#blocks[@]} == 1 )) || { echo 'allocation crosses NVL72 blocks' >&2; exit 2; }
comparison_root=$OUTPUT_ROOT
export MEGAMOE_RESULT_REFERENCE_DIR="$comparison_root/reference"
export MEGAMOE_SMOKE_PROFILE_ONLY=${MEGAMOE_SMOKE_PROFILE_ONLY:-0}

run_source() {
    local index=$1 mode=$2 label=$3
    echo "RESULT_SOURCE=$label path=${sources[index]} mode=$mode"
    FASTAFD_SOURCE_REPO="${sources[index]}" \
    OUTPUT_ROOT="$comparison_root/$label" \
    MEGAMOE_SMOKE_OUTPUT="$comparison_root/$label/result.json" \
    MINISGL_DEEPGEMM_BUILD_DIR="$comparison_root/cache/source$index/deepgemm" \
    DG_JIT_CACHE_DIR="$comparison_root/cache/source$index/jit" \
    TVM_FFI_CACHE_DIR="$comparison_root/cache/source$index/tvm_ffi" \
    MEGAMOE_RESULT_MODE="$mode" \
    bash "$MEGAMOE_SMOKE_JOB_SCRIPT"
}

run_source 0 record control
run_source 0 compare control_repeat
for ((index=1; index<${#sources[@]}; ++index)); do
    run_source "$index" compare "candidate$index"
done
echo MEGAMOE_ALL_SOURCE_RESULTS_IDENTICAL
