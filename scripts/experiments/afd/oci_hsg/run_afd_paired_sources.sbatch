#!/usr/bin/env bash
# Run two one-case AFD task roots sequentially in one Slurm allocation.
# Required export: FASTAFD_PAIR_CONFIG.  Each TSV row is:
# label<TAB>task_root<TAB>plan<TAB>pool<TAB>source_repo<TAB>expected_head
# All ordinary run_afd_bundle.sh exports remain required and are shared.
# Optional FASTAFD_PAIRED_CACHE_ROOT defaults beside the pair config.  Each
# source gets separate DeepGEMM and TVM-FFI caches so differing bindings cannot
# reuse an ABI-incompatible extension within the shared allocation.
# This wrapper exists to remove allocation-to-allocation hardware variance from
# source A/B comparisons; it does not change either source tree or case policy.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=fastafd:paired-sources

set -eEuo pipefail
trap 'status=$?; printf "run_afd_paired_sources_error line=%s status=%s job=%s\n" "$LINENO" "$status" "${SLURM_JOB_ID:-unknown}" >&2' ERR

: "${FASTAFD_PAIR_CONFIG:?}" "${FASTAFD_IMAGE:?}" \
  "${FASTAFD_ALLOCATED_TRAYS:?}" "${FASTAFD_SUBMIT_QOS:?}" \
  "${FASTAFD_REPRO_ROOT:?}" "${FASTAFD_EP_VENV_DIR:?}"
[[ -s "$FASTAFD_PAIR_CONFIG" && -f "$FASTAFD_IMAGE" ]]
[[ "$FASTAFD_ALLOCATED_TRAYS" =~ ^([1-9]|1[0-8])$ ]]
[[ "$FASTAFD_SUBMIT_QOS" == short || "$FASTAFD_SUBMIT_QOS" == normal ]]
PAIR_CACHE_ROOT=${FASTAFD_PAIRED_CACHE_ROOT:-$(dirname "$FASTAFD_PAIR_CONFIG")/cache}
[[ "$PAIR_CACHE_ROOT" = /* ]]
mkdir -p "$PAIR_CACHE_ROOT"

declare -a labels task_roots plans pools source_repos expected_heads
while IFS=$'\t' read -r label task_root plan pool source_repo expected_head extra; do
    [[ -n "$label" || -n "$task_root" ]] || continue
    [[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
    [[ -z "$extra" ]]
    [[ "$task_root" = /* && "$plan" = /* && "$pool" = /* && "$source_repo" = /* ]]
    [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]]
    [[ -f "$plan" && -d "$pool" && -d "$source_repo" ]]
    [[ -x "$source_repo/scripts/experiments/afd/oci_hsg/run_afd_bundle.sh" ]]
    labels+=("$label")
    task_roots+=("$task_root")
    plans+=("$plan")
    pools+=("$pool")
    source_repos+=("$source_repo")
    expected_heads+=("$expected_head")
done < "$FASTAFD_PAIR_CONFIG"

(( ${#labels[@]} == 2 ))
[[ "${labels[0]}" != "${labels[1]}" ]]

for index in "${!labels[@]}"; do
    label=${labels[$index]}
    source_repo=${source_repos[$index]}
    control_dir=$source_repo/scripts/experiments/afd/oci_hsg
    cache_root=$PAIR_CACHE_ROOT/$label
    mkdir -p "$cache_root/deepgemm" "$cache_root/tvm_ffi"
    printf 'FASTAFD_PAIRED_SOURCE_BEGIN label=%s index=%s source=%s head=%s\n' \
        "$label" "$index" "$source_repo" "${expected_heads[$index]}"
    FASTAFD_TASK_ROOT="${task_roots[$index]}" \
    FASTAFD_PLAN="${plans[$index]}" \
    FASTAFD_POOL_ROOT="${pools[$index]}" \
    FASTAFD_SOURCE_REPO="$source_repo" \
    FASTAFD_CONTROL_DIR="$control_dir" \
    FASTAFD_EXPECTED_HEAD="${expected_heads[$index]}" \
    FASTAFD_IMAGE="$FASTAFD_IMAGE" \
    MINISGL_DEEPGEMM_BUILD_DIR="$cache_root/deepgemm" \
    TVM_FFI_CACHE_DIR="$cache_root/tvm_ffi" \
        bash "$control_dir/run_afd_bundle.sh"
    printf 'FASTAFD_PAIRED_SOURCE_COMPLETE label=%s index=%s\n' "$label" "$index"
done

printf 'FASTAFD_PAIRED_SOURCES_SUCCESS first=%s second=%s\n' \
    "${labels[0]}" "${labels[1]}"
