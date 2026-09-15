#!/usr/bin/env bash
# Same-fabric comparison for MegaMoE source/configuration variants.
# Required: FASTAFD_CONTROL_SOURCE FASTAFD_CONTROL_HEAD
#           FASTAFD_CANDIDATE_SOURCE FASTAFD_CANDIDATE_HEAD
#           FASTAFD_IMAGE FASTAFD_VENV OUTPUT_ROOT
# Optional: MEGAMOE_ABBA_CONTROL_CACHE_ROOT MEGAMOE_ABBA_CANDIDATE_CACHE_ROOT
#           FASTAFD_{REFERENCE,ALTERNATE}_SOURCE FASTAFD_{REFERENCE,ALTERNATE}_HEAD
#           MEGAMOE_ABBA_{REFERENCE,ALTERNATE}_ENV_FILE
#           MEGAMOE_ABBA_{REFERENCE,ALTERNATE}_CACHE_ROOT
#           MEGAMOE_ABBA_CONTROL_ENV_FILE MEGAMOE_ABBA_CANDIDATE_ENV_FILE
#           (trusted shell files exporting per-variant configuration)
#           MEGAMOE_ABBA_ORDER (control_*/candidate_*, optional reference_*/alternate_* labels)
#           MEGAMOE_ABBA_VERIFY_SOURCE (0 for archive-authoritative snapshots)
#           MEGAMOE_ABBA_ALLOW_DIRTY, MEGAMOE_SMOKE_PROFILE_ONLY,
#           MEGAMOE_SMOKE_FIXED_PERFECT_ONLY,
#           MEGAMOE_SMOKE_PRODUCTION_ROUTES, MEGAMOE_ABBA_NSYS,
#           MEGAMOE_ABBA_NSYS_{WARMUPS,ITERATIONS}
#           MEGAMOE_ABBA_PROFILE_VARIANTS (default: control candidate)
#           MEGAMOE_ABBA_IKET_SCRIPT (optional paired decomposition after timing;
#           consumes the ordinary MEGAMOE_IKET_* and FASTAFD_IKET_* settings)
# Smoke-shape and precision variables must be explicit and are forwarded.
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
MEGAMOE_ABBA_ROWS=${MEGAMOE_SMOKE_ROWS//\//,}
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
    "$MEGAMOE_ABBA_ROWS" "$MEGAMOE_SMOKE_SMS" "$MEGAMOE_SMOKE_BUCKET" \
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
    [[ -z "$status" || ${MEGAMOE_ABBA_ALLOW_DIRTY:-0} == 1 ]] || {
        echo "source is dirty: $source_path" >&2
        printf '%s\n' "$status" >&2
        exit 2
    }
    if [[ -n "$status" ]]; then
        echo "SOURCE_STATUS_BEGIN=$source_path"
        printf '%s\n' "$status"
        echo "SOURCE_STATUS_END=$source_path"
    fi
}

# Keep arm-specific exports from changing later trials or the profile schedule.
run_case() (
    local label=$1 source_path=$2 cache_label=$3
    local env_var=MEGAMOE_ABBA_${cache_label^^}_ENV_FILE
    local env_file=${!env_var:-}
    if [[ -n "$env_file" ]]; then
        [[ -f "$env_file" ]]
        source "$env_file"
    fi
    local run_root="$OUTPUT_ROOT/$label"
    local cache_root warmups iterations nsys_output
    local cache_var=MEGAMOE_ABBA_${cache_label^^}_CACHE_ROOT
    cache_root=${!cache_var:-$MEGAMOE_ABBA_CACHE_ROOT/$cache_label}
    warmups=$MEGAMOE_SMOKE_WARMUPS
    iterations=$MEGAMOE_SMOKE_ITERATIONS
    nsys_output=
    if [[ ${MEGAMOE_ABBA_NSYS:-0} == 1 ]]; then
        warmups=${MEGAMOE_ABBA_NSYS_WARMUPS:-1}
        iterations=${MEGAMOE_ABBA_NSYS_ITERATIONS:-3}
        nsys_output=$run_root/nsys/megamoe
    fi
    echo "CASE=$label SOURCE=$source_path"
    FASTAFD_SOURCE_REPO="$source_path" \
    FASTAFD_IMAGE="$FASTAFD_IMAGE" \
    FASTAFD_VENV="$FASTAFD_VENV" \
    OUTPUT_ROOT="$run_root" \
    MINISGL_DEEPGEMM_BUILD_DIR="$cache_root/deepgemm" \
    DG_JIT_CACHE_DIR="$cache_root/jit" \
    TVM_FFI_CACHE_DIR="$cache_root/tvm_ffi" \
    MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" \
    MEGAMOE_SMOKE_ROWS="$MEGAMOE_ABBA_ROWS" \
    MEGAMOE_SMOKE_SMS="$MEGAMOE_SMOKE_SMS" \
    MEGAMOE_SMOKE_BUCKET="$MEGAMOE_SMOKE_BUCKET" \
    MEGAMOE_SMOKE_WARMUPS="$warmups" \
    MEGAMOE_SMOKE_ITERATIONS="$iterations" \
    MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH="$MEGAMOE_SMOKE_SPECIALIZATION_MODES_SBATCH" \
    MEGAMOE_SMOKE_PROFILE_ONLY="${MEGAMOE_SMOKE_PROFILE_ONLY:-0}" \
    MEGAMOE_SMOKE_FIXED_PERFECT_ONLY="${MEGAMOE_SMOKE_FIXED_PERFECT_ONLY:-0}" \
    MEGAMOE_SMOKE_PRODUCTION_ROUTES="${MEGAMOE_SMOKE_PRODUCTION_ROUTES:-0}" \
    MEGAMOE_SMOKE_NSYS_OUTPUT_PREFIX="$nsys_output" \
    bash "$source_path/scripts/experiments/afd/oci_hsg/megamoe_same_rank_smoke.sbatch"
)

case "${MEGAMOE_ABBA_VERIFY_SOURCE:-1}" in
    1)
        verify_source "$FASTAFD_CONTROL_SOURCE" "$FASTAFD_CONTROL_HEAD"
        verify_source "$FASTAFD_CANDIDATE_SOURCE" "$FASTAFD_CANDIDATE_HEAD"
        for optional_variant in REFERENCE ALTERNATE; do
            source_var=FASTAFD_${optional_variant}_SOURCE
            head_var=FASTAFD_${optional_variant}_HEAD
            if [[ -n ${!source_var:-} ]]; then
                verify_source "${!source_var}" "${!head_var:?}"
            fi
        done
        ;;
    0) ;;
    *) echo "MEGAMOE_ABBA_VERIFY_SOURCE must be 0 or 1" >&2; exit 2 ;;
esac
mkdir -p "$OUTPUT_ROOT"
MEGAMOE_ABBA_CACHE_ROOT=${MEGAMOE_ABBA_CACHE_ROOT:-$OUTPUT_ROOT/cache}

read -r -a case_order <<< "${MEGAMOE_ABBA_ORDER:-control_trial0 candidate_trial1 candidate_trial2 control_trial3}"
for label in "${case_order[@]}"; do
    [[ "$label" =~ ^(control|candidate|reference|alternate)_[a-zA-Z0-9_]+$ ]] || {
        echo "invalid comparison label: $label" >&2; exit 2;
    }
    variant=${label%%_*}
    source_var=FASTAFD_${variant^^}_SOURCE
    run_case "$label" "${!source_var:?}" "$variant"
done

if [[ -n ${MEGAMOE_ABBA_IKET_SCRIPT:-} ]]; then
    [[ -f "$MEGAMOE_ABBA_IKET_SCRIPT" ]]
    read -r -a profile_variants <<< "${MEGAMOE_ABBA_PROFILE_VARIANTS:-control candidate}"
    for variant in "${profile_variants[@]}"; do
        [[ $variant =~ ^(reference|control|candidate|alternate)$ ]] || {
            echo "invalid profile variant: $variant" >&2; exit 2;
        }
        (
            label=${variant}_profile
            source_var=FASTAFD_${variant^^}_SOURCE
            env_var=MEGAMOE_ABBA_${variant^^}_ENV_FILE
            cache_var=MEGAMOE_ABBA_${variant^^}_CACHE_ROOT
            export FASTAFD_SOURCE_REPO=${!source_var:?}
            env_file=${!env_var:-}
            cache_root=${!cache_var:-$MEGAMOE_ABBA_CACHE_ROOT/$variant}
            if [[ -n $env_file ]]; then source "$env_file"; fi
            # Dedicated profile samples use one invocation per replay. The
            # preceding uninstrumented comparison retains its batched graphs.
            export MEGAMOE_SMOKE_GRAPH_BATCH=1
            if [[ -n ${MEGAMOE_RESULT_REFERENCE_DIR:-} ]]; then
                export MEGAMOE_RESULT_MODE=compare
            fi
            export FASTAFD_IKET_ARTIFACT_ROOT=$OUTPUT_ROOT/decomposition/$variant
            export FASTAFD_IKET_DEEPGEMM_CACHE_ROOT=$cache_root/deepgemm
            export FASTAFD_IKET_TVM_CACHE_ROOT=$cache_root/tvm_ffi
            export DG_JIT_CACHE_DIR=$cache_root/jit
            export FASTAFD_IKET_VERIFY_SOURCE=${MEGAMOE_ABBA_VERIFY_SOURCE:-1}
            bash "$MEGAMOE_ABBA_IKET_SCRIPT" </dev/null
        )
    done
fi
