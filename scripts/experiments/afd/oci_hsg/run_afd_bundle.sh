#!/usr/bin/env bash
# Online, same-tray AFD worker. No per-job case limit.
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

set -eEuo pipefail
trap 'status=$?; printf "run_afd_bundle_error line=%s status=%s job=%s rank=%s qos=%s trays=%s\n" "$LINENO" "$status" "${SLURM_JOB_ID:-unknown}" "${SLURM_PROCID:-outer}" "${FASTAFD_SUBMIT_QOS:-unknown}" "${FASTAFD_ALLOCATED_TRAYS:-unknown}" >&2' ERR

SCRIPT_CONTROL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK_ROOT=${FASTAFD_TASK_ROOT:-$PWD}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$HOME/scratch/github/FastAFD}
if [[ -n "${FASTAFD_CONTROL_DIR:-}" ]]; then
    CONTROL_DIR=$FASTAFD_CONTROL_DIR
elif [[ -n "${FASTAFD_SOURCE_REPO:-}" ]]; then
    # Slurm executes a copied spool script, so BASH_SOURCE does not identify
    # the submitted repository until the inner srun invokes this file again.
    CONTROL_DIR=$SOURCE_REPO/scripts/experiments/afd/oci_hsg
else
    CONTROL_DIR=$SCRIPT_CONTROL_DIR
fi
PLAN=${FASTAFD_PLAN:-$TASK_ROOT/CASES.csv}
PLAN_NAME=$(basename "$PLAN" .csv)
POOL=${FASTAFD_POOL_ROOT:-$TASK_ROOT/state/pools/$PLAN_NAME}
FOLLOWUP_PLAN=${FASTAFD_FOLLOWUP_PLAN:-}
FOLLOWUP_POOL=${FASTAFD_FOLLOWUP_POOL:-}
CASE_RUNNER=${FASTAFD_CASE_RUNNER:-$CONTROL_DIR/run_afd.sh}
POOL_TOOL=${FASTAFD_POOL_TOOL:-$CONTROL_DIR/afd_online_case_pool.py}
NODES=${FASTAFD_ALLOCATED_TRAYS:?}
SUBMIT_QOS=${FASTAFD_SUBMIT_QOS:?}
MIN_REMAINING_SECONDS=${FASTAFD_MIN_REMAINING_SECONDS:-900}
MAX_FAILURES=${FASTAFD_MAX_FAILURES:-2}
CASE_TIMEOUT_SECONDS=${FASTAFD_CASE_TIMEOUT_SECONDS:-1800}
MODEL_PLACEMENT=${FASTAFD_AFD_MODEL_PLACEMENT:-legacy}
MAX_NUM_MB=${FASTAFD_AFD_NUM_MB:-2}
POOL_SELECTION_MODE=${FASTAFD_POOL_SELECTION_MODE:-auto}
AUTO_RELEASE_FAILED=${FASTAFD_AUTO_RELEASE_FAILED:-0}
TEST_MODE=${FASTAFD_BUNDLE_TEST_MODE:-0}
REPRO_ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
SOURCE_MANIFEST=${FASTAFD_EXPECTED_SOURCE_MANIFEST:-}
ALLOW_DIRTY_SOURCE=${FASTAFD_ALLOW_DIRTY_SOURCE:-0}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$REPRO_ROOT/envs/minisgl-3c716194-cuda130-vllm-ep.artifact-5179989}
EXPECTED_HEAD=${FASTAFD_EXPECTED_HEAD:-}
MOE_BACKEND=${MINISGL_AFD_MOE_BACKEND:-}

[[ "$NODES" =~ ^([1-9]|1[0-8])$ ]]
[[ "$SUBMIT_QOS" == short || "$SUBMIT_QOS" == normal ]]
[[ "$MIN_REMAINING_SECONDS" == 900 ]]
[[ "$MAX_FAILURES" == 2 ]]
[[ "$CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
(( CASE_TIMEOUT_SECONDS >= 600 && CASE_TIMEOUT_SECONDS <= 3600 ))
[[ "$MODEL_PLACEMENT" == legacy || "$MODEL_PLACEMENT" == fmha-only ]]
[[ "$MAX_NUM_MB" =~ ^[1-9][0-9]*$ ]]
[[ "$POOL_SELECTION_MODE" == auto || "$POOL_SELECTION_MODE" == isl-desc ]]
[[ "$AUTO_RELEASE_FAILED" == 0 || "$AUTO_RELEASE_FAILED" == 1 ]]
[[ "$ALLOW_DIRTY_SOURCE" == 0 || "$ALLOW_DIRTY_SOURCE" == 1 ]]
[[ -z "$EXPECTED_HEAD" || "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$MOE_BACKEND" || "$MOE_BACKEND" == deepep || "$MOE_BACKEND" == megamoe || "$MOE_BACKEND" == megamoe_m2n ]]
[[ "$TEST_MODE" == 0 || "$TEST_MODE" == 1 ]]
[[ -x "$CASE_RUNNER" && -f "$POOL_TOOL" && -f "$PLAN" ]]
if [[ -n "$FOLLOWUP_PLAN" || -n "$FOLLOWUP_POOL" ]]; then
    [[ -n "$FOLLOWUP_PLAN" && -n "$FOLLOWUP_POOL" && -f "$FOLLOWUP_PLAN" ]]
fi
[[ -x "$EP_VENV_DIR/bin/python" ]]

slurm_duration_seconds() {
    local value=$1 days=0 hours=0 minutes=0 seconds=0
    [[ -n "$value" && "$value" != UNLIMITED && "$value" != INVALID ]]
    if [[ "$value" == *-* ]]; then
        days=${value%%-*}
        value=${value#*-}
    fi
    IFS=: read -r -a parts <<< "$value"
    case ${#parts[@]} in
        3) hours=${parts[0]}; minutes=${parts[1]}; seconds=${parts[2]} ;;
        2) minutes=${parts[0]}; seconds=${parts[1]} ;;
        *) return 1 ;;
    esac
    [[ "$days" =~ ^[0-9]+$ && "$hours" =~ ^[0-9]+$ ]]
    [[ "$minutes" =~ ^[0-5][0-9]$ && "$seconds" =~ ^[0-5][0-9]$ ]]
    printf '%s\n' "$((
        10#$days * 86400 + 10#$hours * 3600 +
        10#$minutes * 60 + 10#$seconds
    ))"
}

if [[ "$TEST_MODE" == 0 && "${FASTAFD_BUNDLE_INSIDE:-0}" != 1 ]]; then
    : "${SLURM_JOB_ID:?}"
    [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%q')" == "$SUBMIT_QOS" ]]
    [[ "$SLURM_JOB_NUM_NODES" == "$NODES" ]]
    SLURM_REMAINING=$(squeue -h -j "$SLURM_JOB_ID" -o '%L')
    SLURM_REMAINING_SECONDS=$(slurm_duration_seconds "$SLURM_REMAINING")
    JOB_DEADLINE_EPOCH=$(($(date +%s) + SLURM_REMAINING_SECONDS))
    IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
    mapfile -t ALLOCATED_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    (( ${#ALLOCATED_HOSTS[@]} == NODES ))
    HEAD_HOST=${ALLOCATED_HOSTS[0]}
    if [[ "$MODEL_PLACEMENT" == fmha-only ]]; then
        mapfile -t FABRIC_BLOCKS < <(
            printf '%s\n' "${ALLOCATED_HOSTS[@]}" |
                sed -E 's/-T[0-9]+$//' | sort -u
        )
        (( ${#FABRIC_BLOCKS[@]} == 1 )) || {
            printf 'FMHA-only allocation must remain in one NVL72 fabric block; blocks=%s nodes=%s\n' \
                "${FABRIC_BLOCKS[*]}" "${ALLOCATED_HOSTS[*]}" >&2
            exit 2
        }
        printf 'fmha_fabric_block=%s trays=%s\n' "${FABRIC_BLOCKS[0]}" "$NODES"
    fi
    set +e
    srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gres=gpu:4 --mem=0 --kill-on-bad-exit=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=CUDA_LAUNCH_BLOCKING,NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_BUNDLE_INSIDE=1 FASTAFD_IN_CONTAINER=1 \
        FASTAFD_TASK_ROOT="$TASK_ROOT" FASTAFD_PLAN="$PLAN" \
        FASTAFD_POOL_ROOT="$POOL" FASTAFD_ALLOCATED_TRAYS="$NODES" \
        FASTAFD_FOLLOWUP_PLAN="$FOLLOWUP_PLAN" \
        FASTAFD_FOLLOWUP_POOL="$FOLLOWUP_POOL" \
        FASTAFD_SUBMIT_QOS="$SUBMIT_QOS" \
        FASTAFD_JOB_DEADLINE_EPOCH="$JOB_DEADLINE_EPOCH" \
        FASTAFD_REPRO_ROOT="$REPRO_ROOT" \
        FASTAFD_SOURCE_REPO="$SOURCE_REPO" \
        FASTAFD_EXPECTED_HEAD="$EXPECTED_HEAD" \
        FASTAFD_EXPECTED_SOURCE_MANIFEST="$SOURCE_MANIFEST" \
        FASTAFD_ALLOW_DIRTY_SOURCE="$ALLOW_DIRTY_SOURCE" \
        FASTAFD_EP_VENV_DIR="$EP_VENV_DIR" \
        FASTAFD_MIN_REMAINING_SECONDS="$MIN_REMAINING_SECONDS" \
        FASTAFD_MAX_FAILURES="$MAX_FAILURES" HEAD_HOST="$HEAD_HOST" \
        FASTAFD_CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
        FASTAFD_AFD_MODEL_PLACEMENT="$MODEL_PLACEMENT" \
        FASTAFD_AFD_NUM_MB="$MAX_NUM_MB" \
        MINISGL_AFD_MOE_BACKEND="$MOE_BACKEND" \
        FASTAFD_POOL_SELECTION_MODE="$POOL_SELECTION_MODE" \
        FASTAFD_AUTO_RELEASE_FAILED="$AUTO_RELEASE_FAILED" \
        bash "$CONTROL_DIR/run_afd_bundle.sh"
    srun_status=$?
    set -e
    if (( srun_status != 0 )); then
        # A node/bootstrap failure can terminate the step before rank 0 records
        # a case outcome. Recover only this allocation's surviving claim; a
        # normally recorded case failure has already moved it out of claims.
        recovery_plans=("$PLAN")
        recovery_pools=("$POOL")
        if [[ -n "$FOLLOWUP_PLAN" ]]; then
            recovery_plans+=("$FOLLOWUP_PLAN")
            recovery_pools+=("$FOLLOWUP_POOL")
        fi
        for recovery_index in "${!recovery_plans[@]}"; do
            recovery_plan=${recovery_plans[$recovery_index]}
            recovery_pool=${recovery_pools[$recovery_index]}
            mapfile -t orphaned_case_ids < <(
                python3 - "$recovery_pool" "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

claims, job_id = Path(sys.argv[1]) / "claims", sys.argv[2]
for path in sorted(claims.glob("*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if str(record.get("job_id")) == job_id:
        print(record["case_id"])
PY
            )
            for case_id in "${orphaned_case_ids[@]}"; do
                python3 "$POOL_TOOL" --plan "$recovery_plan" \
                    --state-root "$recovery_pool" recover-claim \
                    --case-id "$case_id" --job-id "$SLURM_JOB_ID" \
                    --reason "allocation step failed before case outcome bookkeeping"
            done
        done
    fi
    exit "$srun_status"
fi

RANK=${SLURM_PROCID:-0}
JOB_ID=${SLURM_JOB_ID:-${FASTAFD_TEST_JOB_ID:-9001}}
[[ "$RANK" =~ ^[0-9]+$ && "$RANK" -lt "$NODES" ]]
[[ "$JOB_ID" =~ ^[1-9][0-9]*$ ]]
JOB_ROOT=$TASK_ROOT/state/jobs/$JOB_ID

wait_for() {
    local path=$1 deadline=$((SECONDS + 180))
    until [[ -e "$path" ]]; do
        (( SECONDS < deadline )) || return 1
        sleep 1
    done
}

remaining_seconds() {
    local value now deadline
    if [[ "$TEST_MODE" == 1 ]]; then
        value=${FASTAFD_TEST_REMAINING_TIME:-02:00:00}
    else
        deadline=${FASTAFD_JOB_DEADLINE_EPOCH:?}
        [[ "$deadline" =~ ^[1-9][0-9]*$ ]]
        now=$(date +%s)
        if (( deadline <= now )); then
            printf '0\n'
        else
            printf '%s\n' "$((deadline - now))"
        fi
        return
    fi
    slurm_duration_seconds "$value"
}

if [[ "$RANK" == 0 ]]; then
    mkdir -p "$JOB_ROOT" "$TASK_ROOT/results" "$TASK_ROOT/report/metrics" "$TASK_ROOT/cuda_extract/tmp"
    python3 "$POOL_TOOL" --plan "$PLAN" --state-root "$POOL" init >/dev/null
    touch "$JOB_ROOT/READY"
else
    wait_for "$JOB_ROOT/READY"
fi

failures=0
ordinal=0
while true; do
    ordinal=$((ordinal + 1))
    action=$JOB_ROOT/action-$ordinal
    if [[ "$RANK" == 0 ]]; then
        left=$(remaining_seconds)
        if (( left <= MIN_REMAINING_SECONDS )); then
            printf 'STOP\ttime\t%s\n' "$left" > "$action.tmp"
            mv "$action.tmp" "$action"
        else
            python3 "$POOL_TOOL" --plan "$PLAN" --state-root "$POOL" claim \
                --allocated-trays "$NODES" --job-id "$JOB_ID" > "$action.tmp"
            if [[ ! -s "$action.tmp" ]]; then
                if [[ -n "$FOLLOWUP_PLAN" ]]; then
                    python3 "$POOL_TOOL" --plan "$FOLLOWUP_PLAN" \
                        --state-root "$FOLLOWUP_POOL" init >/dev/null
                    mapfile -t terminal_case_ids < <(
                        python3 - "$PLAN" "$POOL" "$NODES" <<'PY'
import csv
import sys
from pathlib import Path

plan, pool, trays = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
with plan.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if int(row["allocated_trays"]) != trays:
            continue
        case_id = row["case_id"]
        terminal = sum(
            (pool / kind / f"{case_id}.json").is_file()
            for kind in ("completed", "failed")
        )
        if terminal == 1:
            print(case_id)
        elif terminal > 1:
            raise SystemExit(f"duplicate focused terminal state: {case_id}")
PY
                    )
                    for terminal_case_id in "${terminal_case_ids[@]}"; do
                        python3 "$POOL_TOOL" --plan "$FOLLOWUP_PLAN" \
                            --state-root "$FOLLOWUP_POOL" import-terminal \
                            --source-state-root "$POOL" \
                            --case-id "$terminal_case_id"
                    done
                    printf 'SWITCH\t%s\t%s\t%s\n' \
                        "$FOLLOWUP_PLAN" "$FOLLOWUP_POOL" "$left" > "$action.tmp"
                else
                    printf 'STOP\tempty\t%s\n' "$left" > "$action.tmp"
                fi
            fi
            mv "$action.tmp" "$action"
        fi
    else
        wait_for "$action"
    fi
    if [[ $(head -c 4 "$action") == STOP ]]; then
        break
    fi
    if [[ $(head -c 6 "$action") == SWITCH ]]; then
        IFS=$'\t' read -r _ PLAN POOL _ < "$action"
        FOLLOWUP_PLAN=""
        FOLLOWUP_POOL=""
        continue
    fi

    IFS=$'\t' read -r case_id context batch ratio attention_tp ffn_ep prompt_mode claimed_trays case_memory_ratio case_num_pages case_kv_capacity case_require_capacity_max case_expert_weight_dtype < <(
        python3 - "$action" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
r = c["case"]
expected = f"qwen3 {r['context_spec']} {r['batch_per_attention_dp_lane']} {r['normalized_af_ratio']}:1 {r['attention_tp']} {r['ffn_ep']}"
if r["run_afd_argv"] != expected:
    raise SystemExit("run_afd_argv drift")
memory = [
    r.get("afd_memory_ratio", ""), r.get("afd_num_pages", ""),
    r.get("afd_kv_capacity_tokens", ""), r.get("require_capacity_max", ""),
]
if not all(memory):
    raise SystemExit("per-case AFD memory contract is incomplete")
precision = r.get("megamoe_expert_weight_dtype", "")
if precision not in {"fp8", "fp4"}:
    raise SystemExit("per-case MegaMoE expert weight dtype must be fp8 or fp4")
if not c["case_id"].endswith(f"-{precision}"):
    raise SystemExit("case_id must end with its MegaMoE expert weight dtype")
print("\t".join([
    c["case_id"], r["context_spec"], r["batch_per_attention_dp_lane"],
    r["normalized_af_ratio"], r["attention_tp"], r["ffn_ep"],
    r["prompt_mode"], r["allocated_trays"], *memory, precision,
]))
PY
    )
    [[ "$claimed_trays" == "$NODES" ]]
    [[ "$case_memory_ratio" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]]
    [[ "$case_num_pages" == none || "$case_num_pages" =~ ^[1-9][0-9]*$ ]]
    [[ "$case_kv_capacity" =~ ^[1-9][0-9]*$ ]]
    [[ "$case_require_capacity_max" == 0 || "$case_require_capacity_max" == 1 ]]
    [[ "$case_expert_weight_dtype" == fp8 || "$case_expert_weight_dtype" == fp4 ]]
    if [[ "$case_num_pages" == none ]]; then
        case_num_pages=""
    fi
    case_num_mb=$MAX_NUM_MB
    if (( case_num_mb > batch )); then
        case_num_mb=$batch
    fi
    run_dir=$TASK_ROOT/results/$case_id
    status_dir=$JOB_ROOT/case-$ordinal
    mkdir -p "$status_dir"
    run_dir_ready=$status_dir/RUN_DIR_READY
    if [[ "$RANK" == 0 ]]; then
        if [[ -e "$run_dir" ]]; then
            archived_run_dir=$TASK_ROOT/state/attempt-results/$case_id/preexisting-before-job-$JOB_ID-case-$ordinal
            mkdir -p "$(dirname "$archived_run_dir")"
            [[ ! -e "$archived_run_dir" ]]
            mv "$run_dir" "$archived_run_dir"
            printf 'archived_preexisting_run case=%s source=%s destination=%s\n' \
                "$case_id" "$run_dir" "$archived_run_dir"
        fi
        touch "$run_dir_ready"
    else
        wait_for "$run_dir_ready"
    fi
    set +e
    timeout --foreground --signal=TERM --kill-after=60s "$CASE_TIMEOUT_SECONDS" \
        env FASTAFD_IN_CONTAINER=1 FASTAFD_SWEEP_CONTRACT=comprehensive \
        FASTAFD_SUBMIT_QOS="$SUBMIT_QOS" FASTAFD_TASK_ROOT="$TASK_ROOT" \
        FASTAFD_REPRO_ROOT="$REPRO_ROOT" \
        FASTAFD_RESULTS_ROOT="$TASK_ROOT/results" \
        FASTAFD_SOURCE_REPO="$SOURCE_REPO" \
        FASTAFD_EXPECTED_HEAD="$EXPECTED_HEAD" \
        FASTAFD_EXPECTED_SOURCE_MANIFEST="$SOURCE_MANIFEST" \
        FASTAFD_ALLOW_DIRTY_SOURCE="$ALLOW_DIRTY_SOURCE" \
        FASTAFD_CUDA_METRIC_PLAN="$PLAN" FASTAFD_CUDA_METRICS_ROOT="$TASK_ROOT/report/metrics" \
        FASTAFD_CUDA_EXTRACT_TEMP_ROOT="$TASK_ROOT/cuda_extract/tmp" \
        FASTAFD_CASE_ORDINAL="$ordinal" FASTAFD_CASE_PORT_SLOT="$ordinal" \
        FASTAFD_CASE_ID="$case_id" \
        FASTAFD_MAX_TOTAL_TRAYS=18 \
        FASTAFD_REQUIRE_CAPACITY_MAX="$case_require_capacity_max" \
        FASTAFD_AFD_MEMORY_RATIO="$case_memory_ratio" \
        FASTAFD_AFD_NUM_PAGES="$case_num_pages" \
        FASTAFD_AFD_KV_CAPACITY_TOKENS="$case_kv_capacity" \
        FASTAFD_AFD_MODEL_PLACEMENT="$MODEL_PLACEMENT" \
        FASTAFD_AFD_NUM_MB="$case_num_mb" \
        MINISGL_AFD_MOE_BACKEND="$MOE_BACKEND" \
        MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="$case_expert_weight_dtype" \
        PROMPT_MODE="$prompt_mode" \
        RUN_DIR="$run_dir" JOB_SCRIPT="$CASE_RUNNER" \
        bash "$CASE_RUNNER" qwen3 "$context" "$batch" "$ratio:1" "$attention_tp" "$ffn_ep"
    case_exit=$?
    set -e
    printf '%s\n' "$case_exit" > "$status_dir/rank-$RANK.exit"

    outcome=$status_dir/outcome
    if [[ "$RANK" == 0 ]]; then
        watchdog_timeout=0
        for ((rank=0; rank<NODES; rank++)); do
            wait_for "$status_dir/rank-$rank.exit"
        done
        aggregate=0
        for ((rank=0; rank<NODES; rank++)); do
            code=$(<"$status_dir/rank-$rank.exit")
            (( code == 0 )) || aggregate=$code
        done
        if (( aggregate == 0 )); then
            metric=$TASK_ROOT/report/metrics/$case_id.json
            result=$run_dir/experiment/afd-result.json
            if ! python3 "$POOL_TOOL" --plan "$PLAN" --state-root "$POOL" complete \
                --case-id "$case_id" --job-id "$JOB_ID" --metric "$metric" --result "$result"; then
                aggregate=70
            fi
        fi
        if (( aggregate != 0 )); then
            failure_reason="case failed in same-tray online worker"
            if (( aggregate == 124 || aggregate == 137 )); then
                failure_reason="case exceeded ${CASE_TIMEOUT_SECONDS}s historical-runtime watchdog"
                watchdog_timeout=1
            fi
            python3 "$POOL_TOOL" --plan "$PLAN" --state-root "$POOL" fail \
                --case-id "$case_id" --job-id "$JOB_ID" --exit-code "$aggregate" \
                --reason "$failure_reason"
            if [[ "$AUTO_RELEASE_FAILED" == 1 ]]; then
                python3 "$POOL_TOOL" --plan "$PLAN" --state-root "$POOL" release-failed \
                    --case-id "$case_id" \
                    --reason "continuous focused-sweep retry requested by user"
            fi
            failures=$((failures + 1))
        fi
        if (( watchdog_timeout || failures >= MAX_FAILURES )); then
            printf 'EXIT\t%s\n' "$failures" > "$outcome.tmp"
        else
            printf 'CONTINUE\t%s\n' "$failures" > "$outcome.tmp"
        fi
        mv "$outcome.tmp" "$outcome"
    else
        wait_for "$outcome"
    fi
    IFS=$'\t' read -r decision failures < "$outcome"
    [[ "$decision" == CONTINUE ]] || exit 1
done

printf 'online_bundle_complete job=%s trays=%s failures=%s\n' "$JOB_ID" "$NODES" "$failures"
