#!/usr/bin/env bash
# CPU-only kernel summaries for existing Nsight Systems reports.
# Required exports: NSYS_REPORTS_FILE OUTPUT_ROOT IMAGE
# Manifest format: one safe label<TAB>absolute .nsys-rep path per line.
# Optional: SOURCE_REPO (used to resolve this script inside the container)
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fastafd:nsys-kernel-summaries

set -eEuo pipefail
trap 'status=$?; printf "extract_nsys_kernel_summaries_error line=%s status=%s\n" "$LINENO" "$status" >&2' ERR

: "${NSYS_REPORTS_FILE:?}" "${OUTPUT_ROOT:?}" "${IMAGE:?}"
if [[ -n "${SOURCE_REPO:-}" ]]; then
    JOB_SCRIPT=${JOB_SCRIPT:-$SOURCE_REPO/scripts/experiments/afd/oci_hsg/extract_nsys_kernel_summaries.sbatch}
else
    JOB_SCRIPT=${JOB_SCRIPT:-$(realpath "$0")}
fi
[[ -s "$NSYS_REPORTS_FILE" && -f "$JOB_SCRIPT" ]]

if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env FASTAFD_IN_CONTAINER=1 NSYS_REPORTS_FILE="$NSYS_REPORTS_FILE" \
        OUTPUT_ROOT="$OUTPUT_ROOT" IMAGE="$IMAGE" JOB_SCRIPT="$JOB_SCRIPT" \
        bash "$JOB_SCRIPT"
fi

NSYS=/usr/local/cuda/bin/nsys
[[ -x "$NSYS" ]]
mkdir -p "$OUTPUT_ROOT/reports" "$OUTPUT_ROOT/summaries"
count=0
while IFS=$'\t' read -r label report extra; do
    [[ -n "$label" || -n "$report" ]] || continue
    [[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
    [[ -z "$extra" && "$report" = /* && "$report" == *.nsys-rep && -s "$report" ]]
    copied_report=$OUTPUT_ROOT/reports/$label.nsys-rep
    cp -p "$report" "$copied_report"
    "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
        "$copied_report" > "$OUTPUT_ROOT/summaries/$label.cuda_gpu_kern_sum.csv"
    [[ -s "$OUTPUT_ROOT/summaries/$label.cuda_gpu_kern_sum.csv" ]]
    count=$((count + 1))
done < "$NSYS_REPORTS_FILE"
(( count > 0 ))
printf 'FASTAFD_NSYS_KERNEL_SUMMARIES_SUCCESS reports=%s output=%s\n' \
    "$count" "$OUTPUT_ROOT"
