#!/usr/bin/env bash
# Manually submit all remaining cases of one EP. No self-submission or job chaining.
# Required: MEGAMOE_SWEEP_ROOT MEGAMOE_IKET_EP MEGAMOE_SWEEP_CASES
# sbatch --nodes=N --segment=N --ntasks=EP --ntasks-per-node=min(EP,4) ...
# Each row is profiled once; completed cases are rejected, never overwritten.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=megamoe-input-sweep
set -euo pipefail
: "${SLURM_JOB_ID:?submit manually with sbatch}" "${MEGAMOE_SWEEP_ROOT:?}"
: "${MEGAMOE_IKET_EP:?}" "${MEGAMOE_SWEEP_CASES:?}" "${FASTAFD_SOURCE_REPO:?}"
export FASTAFD_IKET_DEEPGEMM_CACHE_ROOT="${FASTAFD_IKET_DEEPGEMM_CACHE_ROOT:-$MEGAMOE_SWEEP_ROOT/cache/ep$MEGAMOE_IKET_EP/deepgemm}"
export FASTAFD_IKET_PYTHON_DEPS="${FASTAFD_IKET_PYTHON_DEPS:?}"
export MEGAMOE_SMOKE_BALANCE_RANKS=1
export FASTAFD_IKET_TVM_CACHE_ROOT="$FASTAFD_IKET_DEEPGEMM_CACHE_ROOT/tvm_ffi"
if [[ ${FASTAFD_SWEEP_RANK_WORKER:-0} != 1 ]]; then
    case_specs=$(python3 - "$MEGAMOE_SWEEP_ROOT/manifest.json" "$MEGAMOE_SWEEP_CASES" "$MEGAMOE_IKET_EP" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.load(open(sys.argv[1]))
cases = {case["case_id"]: case for case in manifest["cases"]}
requested = sys.argv[2].split(",")
assert len(requested) == len(set(requested)), "duplicate requested case"
for case_id in requested:
    case = cases[case_id]
    assert not list((Path(sys.argv[1]).parent / case_id).glob("*/decomposition.json")), f"case already complete: {case_id}"
    assert case["ep"] == int(sys.argv[3]), "case EP does not match allocation"
    c = manifest["contract"]
    print(case_id, case["rows_per_rank"], case["global_tokens"],
          *(c[k] for k in ("model", "hidden", "intermediate", "experts", "top_k", "num_sms", "bucket")),
          sep="\t")
PY
    )
    export FASTAFD_SWEEP_CASE_SPECS="$case_specs"
    [[ ${MEGAMOE_IKET_PROFILE_RANK:-0} == 0 ]] || { echo 'sweep requires profile rank zero' >&2; exit 2; }
    tasks_per_node=$((MEGAMOE_IKET_EP / SLURM_NNODES))
    (( tasks_per_node * SLURM_NNODES == MEGAMOE_IKET_EP && tasks_per_node <= 4 )) || exit 2
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t blocks < <(printf '%s\n' "${nodes[@]}" | sed -E 's/-T[0-9]+$//' | sort -u)
    (( ${#blocks[@]} == 1 )) || { echo 'allocation crosses NVL72 blocks' >&2; exit 2; }
    export MASTER_ADDR=${nodes[0]} MASTER_PORT=${MASTER_PORT:-29631}
    export FASTAFD_IKET_SINGLE_FABRIC=1
    user_root=${FASTAFD_USER_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel}
    export FASTAFD_IMAGE=${FASTAFD_IMAGE:-$user_root/trtllm_qwen3_235b_megamoe/images/pytorch-26.05-py3-arm64-trtllm-fa778397186c09dd57bcaa5020e66270768b675b.sqsh}
    # One container launch per node, with rank workers alive for the whole chunk.
    # All normal, IKET discovery, and IKET capture subprocesses run inside it.
    exec srun --nodes="$SLURM_NNODES" --ntasks="$MEGAMOE_IKET_EP" --ntasks-per-node="$tasks_per_node" \
        --kill-on-bad-exit=1 --container-image="$FASTAFD_IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env -u CUDA_VISIBLE_DEVICES FASTAFD_SWEEP_RANK_WORKER=1 FASTAFD_IKET_IN_CONTAINER=1 \
        bash "$FASTAFD_SOURCE_REPO/codex_scripts/megamoe_input_sweep.sbatch"
fi
: "${FASTAFD_SWEEP_CASE_SPECS:?}" "${SLURM_PROCID:?}"
log_case() {
    if [[ $SLURM_PROCID == 0 ]]; then
        printf 'SWEEP_CASE_%s case=%s ep=%s rows=%s global_tokens=%s job=%s time=%s\n' \
            "$1" "$case_id" "$MEGAMOE_IKET_EP" "$rows" "$global_tokens" \
            "$SLURM_JOB_ID" "$(date +%s.%N)"
    fi
}

while IFS=$'\t' read -r case_id rows global_tokens model hidden intermediate experts top_k sms bucket; do
    export MEGAMOE_IKET_MODEL=$model MEGAMOE_IKET_HIDDEN=$hidden
    export MEGAMOE_IKET_INTERMEDIATE=$intermediate MEGAMOE_IKET_EXPERTS=$experts
    export MEGAMOE_IKET_TOP_K=$top_k MEGAMOE_IKET_SMS=$sms MEGAMOE_IKET_BUCKET=$bucket
    export MEGAMOE_IKET_GLOBAL_TOKENS=$global_tokens
    export FASTAFD_IKET_ARTIFACT_ROOT="$MEGAMOE_SWEEP_ROOT/$case_id"
    log_case START
    # Child programs must not consume the remaining case list from stdin.
    bash "$FASTAFD_SOURCE_REPO/codex_scripts/profile_megamoe_iket.sbatch" </dev/null
    log_case COMPLETE
done <<< "$FASTAFD_SWEEP_CASE_SPECS"
