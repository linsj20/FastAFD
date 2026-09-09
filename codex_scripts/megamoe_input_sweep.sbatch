#!/usr/bin/env bash
# Manually submit one bounded EP/row chunk. No self-submission or job chaining.
# Required: MEGAMOE_SWEEP_ROOT MEGAMOE_IKET_EP MEGAMOE_SWEEP_CASES
# sbatch --nodes=N --segment=N --ntasks=EP --ntasks-per-node=min(EP,4) ...
# Each row is profiled once; completed cases are rejected, never overwritten.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:00:00
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
case_specs=$(python3 - "$MEGAMOE_SWEEP_ROOT/manifest.json" "$MEGAMOE_SWEEP_CASES" "$MEGAMOE_IKET_EP" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
cases = {case["case_id"]: case for case in manifest["cases"]}
requested = sys.argv[2].split(",")
assert len(requested) == len(set(requested)), "duplicate requested case"
for case_id in requested:
    case = cases[case_id]
    assert case["ep"] == int(sys.argv[3]), "case EP does not match allocation"
    print(case_id, case["rows_per_rank"], case["global_tokens"], sep="\t")
PY
)
while IFS=$'\t' read -r case_id rows global_tokens; do
    export MEGAMOE_IKET_GLOBAL_TOKENS=$global_tokens
    export FASTAFD_IKET_ARTIFACT_ROOT="$MEGAMOE_SWEEP_ROOT/$case_id"
    if compgen -G "$FASTAFD_IKET_ARTIFACT_ROOT/*/decomposition.json" >/dev/null; then
        echo "case already complete: $FASTAFD_IKET_ARTIFACT_ROOT" >&2
        exit 2
    fi
    printf 'SWEEP_CASE_START case=%s ep=%s rows=%s global_tokens=%s job=%s\n' \
        "$case_id" "$MEGAMOE_IKET_EP" "$rows" "$global_tokens" "$SLURM_JOB_ID"
    # srun must not consume the remaining case list from this loop's stdin.
    bash "$FASTAFD_SOURCE_REPO/codex_scripts/profile_megamoe_iket.sbatch" </dev/null
    printf 'SWEEP_CASE_COMPLETE case=%s ep=%s rows=%s global_tokens=%s job=%s\n' \
        "$case_id" "$MEGAMOE_IKET_EP" "$rows" "$global_tokens" "$SLURM_JOB_ID"
done <<< "$case_specs"
