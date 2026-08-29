#!/usr/bin/env bash
# Same-fabric ABBA comparison for two clean MegaMoE source trees.
# Required: FASTAFD_CONTROL_SOURCE FASTAFD_CONTROL_HEAD
#           FASTAFD_CANDIDATE_SOURCE FASTAFD_CANDIDATE_HEAD
#           FASTAFD_IMAGE FASTAFD_VENV OUTPUT_ROOT
# Optional: MEGAMOE_ABBA_CONTROL_CACHE_ROOT MEGAMOE_ABBA_CANDIDATE_CACHE_ROOT
# Smoke-shape and precision variables must be explicit and are forwarded.
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
#SBATCH --job-name=megamoe-source-abba

set -euo pipefail

: "${FASTAFD_CONTROL_SOURCE:?}" "${FASTAFD_CONTROL_HEAD:?}" \
  "${FASTAFD_CANDIDATE_SOURCE:?}" "${FASTAFD_CANDIDATE_HEAD:?}" \
  "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" "${OUTPUT_ROOT:?}" \
  "${MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE:?}" \
  "${MEGAMOE_SMOKE_ROWS:?}" "${MEGAMOE_SMOKE_SMS:?}" \
  "${MEGAMOE_SMOKE_BUCKET:?}" "${MEGAMOE_SMOKE_WARMUPS:?}" \
  "${MEGAMOE_SMOKE_ITERATIONS:?}" \
  "${MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH:?}"

case "$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" in
    fp4|fp8) ;;
    *)
        echo "unsupported MegaMoE expert weight dtype: $MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" >&2
        exit 2
        ;;
esac

MEGAMOE_ABBA_SPECIALIZATION_MODES=${MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH//\//,}
IFS=, read -r -a MEGAMOE_ABBA_SPECIALIZATION_ENTRIES <<< \
    "$MEGAMOE_ABBA_SPECIALIZATION_MODES"
for mode in "${MEGAMOE_ABBA_SPECIALIZATION_ENTRIES[@]}"; do
    if [[ ! $mode =~ ^[01]:[01](:[01](:[01])?)?$ ]]; then
        echo "invalid MegaMoE specialization tuple: $mode" >&2
        exit 2
    fi
    IFS=: read -r route_mode combine_mode dispatch_mode rank_ready_mode <<< "$mode"
    dispatch_mode=${dispatch_mode:-0}
    rank_ready_mode=${rank_ready_mode:-0}
    if (( dispatch_mode && ! route_mode )); then
        echo "route-ready dispatch requires prepared routes: $mode" >&2
        exit 2
    fi
    if (( rank_ready_mode && (! route_mode || ! dispatch_mode) )); then
        echo "rank-ready publication requires prepared, route-ready dispatch: $mode" >&2
        exit 2
    fi
done

printf 'ABBA_CONFIG=dtype=%s rows=%s sms=%s bucket=%s warmups=%s iterations=%s specialization=%s\n' \
    "$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" \
    "$MEGAMOE_SMOKE_ROWS" "$MEGAMOE_SMOKE_SMS" "$MEGAMOE_SMOKE_BUCKET" \
    "$MEGAMOE_SMOKE_WARMUPS" "$MEGAMOE_SMOKE_ITERATIONS" \
    "$MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH"

mapfile -t MEGAMOE_ABBA_FABRIC_BLOCKS < <(
    scontrol show hostnames "$SLURM_JOB_NODELIST" |
        sed -E 's/-T[0-9]+$//' | sort -u
)
(( ${#MEGAMOE_ABBA_FABRIC_BLOCKS[@]} == 1 )) || {
    printf 'MegaMoE ABBA allocation crosses NVL72 fabric blocks: %s\n' \
        "${MEGAMOE_ABBA_FABRIC_BLOCKS[*]}" >&2
    exit 2
}

verify_source() {
    local source_path=$1 expected_head=$2 actual_head status
    actual_head=$(git -C "$source_path" rev-parse HEAD)
    [[ "$actual_head" == "$expected_head" ]] || {
        echo "unexpected source head: path=$source_path expected=$expected_head actual=$actual_head" >&2
        exit 2
    }
    status=$(git -C "$source_path" status --porcelain)
    [[ -z "$status" ]] || {
        echo "source is dirty: $source_path" >&2
        printf '%s\n' "$status" >&2
        exit 2
    }
}

run_case() {
    local label=$1 source_path=$2 cache_label=$3
    local run_root="$OUTPUT_ROOT/$label"
    local cache_root
    if [[ "$cache_label" == control ]]; then
        cache_root=${MEGAMOE_ABBA_CONTROL_CACHE_ROOT:-$MEGAMOE_ABBA_CACHE_ROOT/control}
    else
        cache_root=${MEGAMOE_ABBA_CANDIDATE_CACHE_ROOT:-$MEGAMOE_ABBA_CACHE_ROOT/candidate}
    fi
    echo "CASE=$label SOURCE=$source_path HEAD=$(git -C "$source_path" rev-parse --short HEAD)"
    FASTAFD_SOURCE_REPO="$source_path" \
    FASTAFD_IMAGE="$FASTAFD_IMAGE" \
    FASTAFD_VENV="$FASTAFD_VENV" \
    OUTPUT_ROOT="$run_root" \
    MINISGL_DEEPGEMM_BUILD_DIR="$cache_root/deepgemm" \
    TVM_FFI_CACHE_DIR="$cache_root/tvm_ffi" \
    MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" \
    MEGAMOE_SMOKE_ROWS="$MEGAMOE_SMOKE_ROWS" \
    MEGAMOE_SMOKE_SMS="$MEGAMOE_SMOKE_SMS" \
    MEGAMOE_SMOKE_BUCKET="$MEGAMOE_SMOKE_BUCKET" \
    MEGAMOE_SMOKE_WARMUPS="$MEGAMOE_SMOKE_WARMUPS" \
    MEGAMOE_SMOKE_ITERATIONS="$MEGAMOE_SMOKE_ITERATIONS" \
    MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH="$MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH" \
    bash "$source_path/scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch"
}

verify_source "$FASTAFD_CONTROL_SOURCE" "$FASTAFD_CONTROL_HEAD"
verify_source "$FASTAFD_CANDIDATE_SOURCE" "$FASTAFD_CANDIDATE_HEAD"
mkdir -p "$OUTPUT_ROOT"
MEGAMOE_ABBA_CACHE_ROOT=${MEGAMOE_ABBA_CACHE_ROOT:-$OUTPUT_ROOT/cache}

run_case control_trial0 "$FASTAFD_CONTROL_SOURCE" control
run_case candidate_trial1 "$FASTAFD_CANDIDATE_SOURCE" candidate
run_case candidate_trial2 "$FASTAFD_CANDIDATE_SOURCE" candidate
run_case control_trial3 "$FASTAFD_CONTROL_SOURCE" control
