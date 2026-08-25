#!/usr/bin/env bash
# CPU-only strict metric extraction from an already completed AFD capture.
# Required exports: RESULT_JSON PLAN OUTPUT_JSON SOURCE_REPO TEMP_ROOT IMAGE
# Optional: PYTHON (defaults to python3 inside the container)
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fastafd:extract-existing-metric

set -eEuo pipefail
trap 'status=$?; printf "extract_existing_afd_metric_error line=%s status=%s\n" "$LINENO" "$status" >&2' ERR

: "${RESULT_JSON:?}" "${PLAN:?}" "${OUTPUT_JSON:?}" "${SOURCE_REPO:?}"
: "${TEMP_ROOT:?}" "${IMAGE:?}"
JOB_SCRIPT=${JOB_SCRIPT:-$SOURCE_REPO/scripts/experiments/afd/oci_hsg/extract_existing_afd_metric.sbatch}

[[ -s "$RESULT_JSON" ]]
[[ -s "$PLAN" ]]
[[ -f "$JOB_SCRIPT" ]]
[[ -f "$SOURCE_REPO/scripts/experiments/afd/oci_hsg/extract_cuda_wall.py" ]]
[[ -f "$SOURCE_REPO/scripts/experiments/afd/oci_hsg/cuda_execution_span.py" ]]

if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env FASTAFD_IN_CONTAINER=1 RESULT_JSON="$RESULT_JSON" PLAN="$PLAN" \
        OUTPUT_JSON="$OUTPUT_JSON" SOURCE_REPO="$SOURCE_REPO" \
        TEMP_ROOT="$TEMP_ROOT" PYTHON="${PYTHON:-python3}" \
        JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

PYTHON=${PYTHON:-python3}
mkdir -p "$(dirname "$OUTPUT_JSON")" "$TEMP_ROOT"
"$PYTHON" "$SOURCE_REPO/scripts/experiments/afd/oci_hsg/extract_cuda_wall.py" \
    --result "$RESULT_JSON" \
    --plan "$PLAN" \
    --nsys /usr/local/cuda/bin/nsys \
    --cuda-span-module \
    "$SOURCE_REPO/scripts/experiments/afd/oci_hsg/cuda_execution_span.py" \
    --temp-root "$TEMP_ROOT" \
    --output "$OUTPUT_JSON"
[[ -s "$OUTPUT_JSON" ]]
printf 'FASTAFD_EXISTING_METRIC_SUCCESS output=%s\n' "$OUTPUT_JSON"
