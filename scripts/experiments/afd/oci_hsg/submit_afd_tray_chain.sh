#!/usr/bin/env bash
# Submit one same-tray AFD allocation at a time from a dependency-driven chain.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --no-requeue

set -eEuo pipefail

COMMAND=${1:?usage: submit_afd_tray_chain.sh start|advance}
TASK_ROOT=${FASTAFD_TASK_ROOT:?}
PLAN=${FASTAFD_PLAN:-$TASK_ROOT/CASES.csv}
POOL=${FASTAFD_POOL_ROOT:-$TASK_ROOT/state/pools/CASES}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:?}
CONTROL_DIR=${FASTAFD_CONTROL_DIR:-$SOURCE_REPO/scripts/experiments/afd/oci_hsg}
REPRO_ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$REPRO_ROOT/envs/minisgl-3c716194-cuda130-vllm-ep}
IMAGE=${FASTAFD_IMAGE:?}
SOURCE_MANIFEST=${FASTAFD_EXPECTED_SOURCE_MANIFEST:?}
MAX_BATCHED_TOKENS=${FASTAFD_AFD_MAX_BATCHED_TOKENS:?}
CASE_TIMEOUT_SECONDS=${FASTAFD_CASE_TIMEOUT_SECONDS:?}
CHAIN_ID=${FASTAFD_CHAIN_ID:?}
CHAIN_TRAYS_ENCODED=${FASTAFD_CHAIN_TRAYS_ENCODED:-}
if [[ -n "$CHAIN_TRAYS_ENCODED" ]]; then
    [[ -z "${FASTAFD_CHAIN_TRAYS:-}" ]]
    [[ "$CHAIN_TRAYS_ENCODED" =~ ^([1-9]|1[0-8])(:([1-9]|1[0-8]))*$ ]]
    CHAIN_TRAYS=${CHAIN_TRAYS_ENCODED//:/,}
else
    CHAIN_TRAYS=${FASTAFD_CHAIN_TRAYS:?}
fi
PREVIOUS_JOB_ID=${FASTAFD_PREVIOUS_JOB_ID:-}
MAX_SHORT_JOBS=${FASTAFD_MAX_SHORT_JOBS:-4}
EXPECTED_CHAIN_COUNT=${FASTAFD_EXPECTED_CHAIN_COUNT:-4}
FRONTIER_MANIFEST=${FASTAFD_FRONTIER_MANIFEST:-$TASK_ROOT/frontier_manifest.json}
SOURCE_HTML=${FASTAFD_SOURCE_HTML:-$TASK_ROOT/source/afd-baseline-user-tps-vs-tps-gpu-pareto.html}
AUDITOR=${FASTAFD_CAMPAIGN_AUDITOR:-$CONTROL_DIR/audit_afd_campaign.py}
STATE_ROOT=$TASK_ROOT/state/chains
SCRIPT_PATH=$CONTROL_DIR/submit_afd_tray_chain.sh

[[ "$COMMAND" == start || "$COMMAND" == advance ]]
[[ "$CHAIN_ID" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$CHAIN_TRAYS" =~ ^([1-9]|1[0-8])(,([1-9]|1[0-8]))*$ ]]
[[ "$MAX_SHORT_JOBS" == 4 ]]
[[ "$EXPECTED_CHAIN_COUNT" == 4 ]]
[[ -f "$PLAN" && -f "$CONTROL_DIR/afd_online_case_pool.py" ]]
[[ -f "$FRONTIER_MANIFEST" && -f "$SOURCE_HTML" && -f "$AUDITOR" ]]
[[ -x "$CONTROL_DIR/run_afd_bundle.sh" ]]
[[ -x "$SCRIPT_PATH" ]]
[[ -x "$EP_VENV_DIR/bin/python" ]]
[[ -f "$SOURCE_MANIFEST" ]]
[[ "$MAX_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]]
(( MAX_BATCHED_TOKENS >= 512 && MAX_BATCHED_TOKENS <= 8192 ))
[[ "$CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
(( CASE_TIMEOUT_SECONDS >= 1800 && CASE_TIMEOUT_SECONDS <= 3600 ))
mkdir -p "$TASK_ROOT/logs" "$STATE_ROOT/jobs" "$STATE_ROOT/controllers" "$STATE_ROOT/done"

validate_completed_group() {
    local trays=$1 job_id=$2
    "$EP_VENV_DIR/bin/python" - "$PLAN" "$POOL" "$trays" "$job_id" <<'PY'
import csv
import json
import sys
from pathlib import Path

plan, pool, trays, job_id = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with plan.open(newline="", encoding="utf-8") as stream:
    cases = [row["case_id"] for row in csv.DictReader(stream) if int(row["allocated_trays"]) == trays]
if not cases:
    raise SystemExit(f"no planned cases for tray group {trays}")
for case_id in cases:
    if (pool / "claims" / f"{case_id}.json").exists():
        raise SystemExit(f"claim remains for {case_id}")
    if (pool / "failed" / f"{case_id}.json").exists():
        raise SystemExit(f"failed case remains for {case_id}")
    path = pool / "completed" / f"{case_id}.json"
    if not path.is_file():
        raise SystemExit(f"missing completion for {case_id}")
    completion = json.loads(path.read_text(encoding="utf-8"))
    if completion.get("job_id") != job_id:
        raise SystemExit(
            f"completion job mismatch for {case_id}: {completion.get('job_id')} != {job_id}"
        )
print(f"validated tray_group={trays} cases={len(cases)} job={job_id}")
PY
    local accounting
    accounting=$(sacct -n -X -j "$job_id" -o State,ExitCode -P | sed -n '1p')
    [[ "$accounting" == COMPLETED\|0:0 ]]
}

record_submission() {
    local trays=$1 gpu_job_id=$2 controller_job_id=$3 remaining=$4
    "$EP_VENV_DIR/bin/python" - "$STATE_ROOT" "$CHAIN_ID" "$trays" \
        "$gpu_job_id" "$controller_job_id" "$remaining" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

root, chain_id, trays, gpu_job, controller_job, remaining = (
    Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
)
path = root / "jobs" / f"trays{trays}.json"
if path.exists():
    raise SystemExit(f"tray group already submitted: {trays}")
record = {
    "chain_id": chain_id,
    "trays": trays,
    "gpu_job_id": gpu_job,
    "controller_job_id": controller_job,
    "remaining_trays": remaining,
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

mark_done() {
    "$EP_VENV_DIR/bin/python" - "$STATE_ROOT" "$CHAIN_ID" "$PREVIOUS_JOB_ID" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

root, chain_id, previous_job = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = root / "done" / f"{chain_id}.json"
if path.exists():
    raise SystemExit(f"chain already done: {chain_id}")
record = {
    "chain_id": chain_id,
    "last_gpu_job_id": previous_job,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(record, sort_keys=True))
PY
}

finalize_if_ready() {
    exec 8>"$STATE_ROOT/terminal.lock"
    flock 8
    if [[ -f "$STATE_ROOT/terminal-complete.json" ]]; then
        return
    fi
    local done_count
    done_count=$(find "$STATE_ROOT/done" -maxdepth 1 -type f -name '*.json' | wc -l)
    if (( done_count != EXPECTED_CHAIN_COUNT )); then
        printf 'terminal_audit_waiting complete_chains=%s expected=%s\n' \
            "$done_count" "$EXPECTED_CHAIN_COUNT"
        return
    fi

    local tray_job_args=() record trays gpu_job expected_tray_groups
    while IFS= read -r record; do
        trays=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trays"])' "$record")
        gpu_job=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gpu_job_id"])' "$record")
        tray_job_args+=(--tray-job "$trays=$gpu_job")
    done < <(find "$STATE_ROOT/jobs" -maxdepth 1 -type f -name 'trays*.json' | sort -V)
    expected_tray_groups=$("$EP_VENV_DIR/bin/python" - "$PLAN" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    print(len({int(row["allocated_trays"]) for row in csv.DictReader(stream)}))
PY
    )
    [[ "$expected_tray_groups" =~ ^[1-9][0-9]*$ ]]
    [[ ${#tray_job_args[@]} == "$expected_tray_groups" ]]

    mkdir -p "$TASK_ROOT/report"
    "$EP_VENV_DIR/bin/python" "$CONTROL_DIR/afd_online_case_pool.py" \
        --plan "$PLAN" --state-root "$POOL" summary \
        > "$TASK_ROOT/report/terminal-pool-summary.json"
    "$EP_VENV_DIR/bin/python" "$AUDITOR" \
        --plan "$PLAN" --state-root "$POOL" \
        --pool-tool "$CONTROL_DIR/afd_online_case_pool.py" \
        --frontier-manifest "$FRONTIER_MANIFEST" \
        --source-html "$SOURCE_HTML" \
        --output "$TASK_ROOT/report/terminal-audit.json" \
        "${tray_job_args[@]}"
    "$EP_VENV_DIR/bin/python" - "$STATE_ROOT/terminal-complete.json" \
        "$TASK_ROOT/report/terminal-audit.json" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path, audit_path = Path(sys.argv[1]), Path(sys.argv[2])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "pass" or int(audit.get("case_count", 0)) != 30:
    raise SystemExit("terminal audit did not pass all 30 cases")
record = {
    "status": "pass",
    "case_count": 30,
    "audit_path": str(audit_path),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(record, sort_keys=True))
PY
}

defer_until_short_slot() {
    local target_job retry_job_id encoded_trays
    target_job=$(
        squeue -h -u "$USER" -o '%i|%q|%T' |
            awk -F'|' '$2 == "short" && ($3 == "PENDING" || $3 == "RUNNING" || $3 == "CONFIGURING" || $3 == "COMPLETING") {print $1; exit}'
    )
    [[ "$target_job" =~ ^[1-9][0-9]*$ ]]
    encoded_trays=${CHAIN_TRAYS//,/:}
    retry_job_id=$(sbatch --parsable \
        --job-name="fastafd:optattn128-chain-${CHAIN_ID}" \
        --dependency="afterany:$target_job" --kill-on-invalid-dep=yes \
        --output="$TASK_ROOT/logs/chain-${CHAIN_ID}-%j.out" \
        --error="$TASK_ROOT/logs/chain-${CHAIN_ID}-%j.err" \
        --export="ALL,FASTAFD_TASK_ROOT=$TASK_ROOT,FASTAFD_PLAN=$PLAN,FASTAFD_POOL_ROOT=$POOL,FASTAFD_SOURCE_REPO=$SOURCE_REPO,FASTAFD_CONTROL_DIR=$CONTROL_DIR,FASTAFD_REPRO_ROOT=$REPRO_ROOT,FASTAFD_EP_VENV_DIR=$EP_VENV_DIR,FASTAFD_IMAGE=$IMAGE,FASTAFD_EXPECTED_SOURCE_MANIFEST=$SOURCE_MANIFEST,FASTAFD_AFD_MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS,FASTAFD_CASE_TIMEOUT_SECONDS=$CASE_TIMEOUT_SECONDS,FASTAFD_CHAIN_ID=$CHAIN_ID,FASTAFD_CHAIN_TRAYS_ENCODED=$encoded_trays,FASTAFD_PREVIOUS_JOB_ID=$PREVIOUS_JOB_ID,FASTAFD_MAX_SHORT_JOBS=4" \
        "$SCRIPT_PATH" "$COMMAND")
    retry_job_id=${retry_job_id%%;*}
    [[ "$retry_job_id" =~ ^[1-9][0-9]*$ ]]
    printf 'deferred chain=%s controller_job=%s until_short_job=%s queued_short_jobs=%s\n' \
        "$CHAIN_ID" "$retry_job_id" "$target_job" "$MAX_SHORT_JOBS"
}

submit_next() {
    local trays=${CHAIN_TRAYS%%,*}
    local remaining=""
    local encoded_remaining
    if [[ "$CHAIN_TRAYS" == *,* ]]; then
        remaining=${CHAIN_TRAYS#*,}
    fi
    encoded_remaining=${remaining:-$trays}
    encoded_remaining=${encoded_remaining//,/:}

    exec 9>"$STATE_ROOT/submit.lock"
    flock 9
    [[ ! -e "$STATE_ROOT/jobs/trays${trays}.json" ]]
    local short_jobs
    short_jobs=$(squeue -h -u "$USER" -o '%q' | awk '$1 == "short" {count++} END {print count + 0}')
    if (( short_jobs >= MAX_SHORT_JOBS )); then
        defer_until_short_slot
        return
    fi

    local gpu_job_id controller_job_id
    gpu_job_id=$(sbatch --parsable \
        --job-name="fastafd:optattn128-t${trays}" \
        --nodes="$trays" --segment="$trays" --qos=short --time=02:00:00 \
        --output="$TASK_ROOT/logs/trays${trays}-%j.out" \
        --error="$TASK_ROOT/logs/trays${trays}-%j.err" \
        --export="ALL,FASTAFD_TASK_ROOT=$TASK_ROOT,FASTAFD_PLAN=$PLAN,FASTAFD_POOL_ROOT=$POOL,FASTAFD_ALLOCATED_TRAYS=$trays,FASTAFD_SUBMIT_QOS=short,FASTAFD_REPRO_ROOT=$REPRO_ROOT,FASTAFD_SOURCE_REPO=$SOURCE_REPO,FASTAFD_CONTROL_DIR=$CONTROL_DIR,FASTAFD_EP_VENV_DIR=$EP_VENV_DIR,FASTAFD_IMAGE=$IMAGE,FASTAFD_EXPECTED_SOURCE_MANIFEST=$SOURCE_MANIFEST,FASTAFD_AFD_MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS,FASTAFD_CASE_TIMEOUT_SECONDS=$CASE_TIMEOUT_SECONDS,FASTAFD_AFD_MODEL_PLACEMENT=fmha-only,FASTAFD_AFD_NUM_MB=2,FASTAFD_POOL_SELECTION_MODE=auto,FASTAFD_AUTO_RELEASE_FAILED=0,FASTAFD_MIN_REMAINING_SECONDS=900,FASTAFD_MAX_FAILURES=2" \
        "$CONTROL_DIR/run_afd_bundle.sh")
    gpu_job_id=${gpu_job_id%%;*}
    [[ "$gpu_job_id" =~ ^[1-9][0-9]*$ ]]

    controller_job_id=$(sbatch --parsable \
        --job-name="fastafd:optattn128-chain-${CHAIN_ID}" \
        --dependency="afterany:$gpu_job_id" --kill-on-invalid-dep=yes \
        --output="$TASK_ROOT/logs/chain-${CHAIN_ID}-%j.out" \
        --error="$TASK_ROOT/logs/chain-${CHAIN_ID}-%j.err" \
        --export="ALL,FASTAFD_TASK_ROOT=$TASK_ROOT,FASTAFD_PLAN=$PLAN,FASTAFD_POOL_ROOT=$POOL,FASTAFD_SOURCE_REPO=$SOURCE_REPO,FASTAFD_CONTROL_DIR=$CONTROL_DIR,FASTAFD_REPRO_ROOT=$REPRO_ROOT,FASTAFD_EP_VENV_DIR=$EP_VENV_DIR,FASTAFD_IMAGE=$IMAGE,FASTAFD_EXPECTED_SOURCE_MANIFEST=$SOURCE_MANIFEST,FASTAFD_AFD_MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS,FASTAFD_CASE_TIMEOUT_SECONDS=$CASE_TIMEOUT_SECONDS,FASTAFD_CHAIN_ID=$CHAIN_ID,FASTAFD_CHAIN_TRAYS_ENCODED=$encoded_remaining,FASTAFD_PREVIOUS_JOB_ID=$gpu_job_id,FASTAFD_MAX_SHORT_JOBS=4" \
        "$SCRIPT_PATH" advance)
    controller_job_id=${controller_job_id%%;*}
    [[ "$controller_job_id" =~ ^[1-9][0-9]*$ ]]
    record_submission "$trays" "$gpu_job_id" "$controller_job_id" "$remaining"
    printf 'submitted chain=%s trays=%s gpu_job=%s controller_job=%s remaining=%s\n' \
        "$CHAIN_ID" "$trays" "$gpu_job_id" "$controller_job_id" "$remaining"
}

case "$COMMAND" in
    start)
        [[ -z "$PREVIOUS_JOB_ID" ]]
        submit_next
        ;;
    advance)
        [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%q')" == cpu-short ]]
        [[ "$PREVIOUS_JOB_ID" =~ ^[1-9][0-9]*$ ]]
        previous_record=$(
            find "$STATE_ROOT/jobs" -maxdepth 1 -type f -name 'trays*.json' \
                -exec grep -l "\"gpu_job_id\": \"$PREVIOUS_JOB_ID\"" {} +
        )
        [[ -n "$previous_record" && $(wc -l <<< "$previous_record") == 1 ]]
        previous_trays=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trays"])' "$previous_record")
        validate_completed_group "$previous_trays" "$PREVIOUS_JOB_ID"
        if [[ -n "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["remaining_trays"])' "$previous_record")" ]]; then
            submit_next
        else
            mark_done
            finalize_if_ready
        fi
        ;;
esac
