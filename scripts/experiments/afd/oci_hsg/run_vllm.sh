#!/usr/bin/env bash
# Run a reproducible wide-EP vLLM decode point.
# Usage: ./run_vllm.sh [qwen3|minimax] [1k|...|128k|1k-4k|...] [2|4|8|16|32|64] [batch/lane] [uniform|irregular]
#
# Omitting batch preserves the published/capacity-ceiling 8K/16K presets.
# Arbitrary Qwen3 ISLs and irregular ranges require an explicit batch. Prompts
# are deterministically cut/repeated in token space from the pinned 8K corpus;
# prefix caching is off. Irregular mode applies the same inclusive uniform
# prompt-length distribution independently to every DP/EP lane.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:vllm

set -euo pipefail
ulimit -s 8192

# A packed sweep allocation keeps one Pyxis container alive while executing up
# to ten independent cases.  The manifest is a tab-separated file with this
# exact header:
# point_id context_tokens ep batch_per_lane nodes run_dir
# require_capacity_max known_kv_capacity_tokens
# Every case in a pack must request the same number of nodes/trays.  Per-node
# exit files and a shared-filesystem barrier prevent any node from advancing to
# the next case until the previous case has exited (or timed out) everywhere.
BATCH_CASES_FILE=${FASTAFD_BATCH_CASES_FILE:-}
if [[ -n "$BATCH_CASES_FILE" ]]; then
    ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
    RESULTS_ROOT=${FASTAFD_RESULTS_ROOT:-$ROOT}
    SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$ROOT/source/FastAFD-3c716194}
    IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
    EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-3c716194-cuda130-vllm-ep}
    JOB_SCRIPT=$(realpath "${JOB_SCRIPT:-$0}")
    SLURM_QOS_REQUEST=${FASTAFD_SLURM_QOS:-normal}
    BATCH_CASE_TIMEOUT_SECONDS=${FASTAFD_BATCH_CASE_TIMEOUT_SECONDS:-600}
    case "$SLURM_QOS_REQUEST" in
        normal|short) ;;
        *) echo "FASTAFD_SLURM_QOS must be normal or short" >&2; exit 2 ;;
    esac
    [[ "$BATCH_CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
    (( BATCH_CASE_TIMEOUT_SECONDS >= 60 && BATCH_CASE_TIMEOUT_SECONDS <= 600 ))
    [[ -f "$BATCH_CASES_FILE" ]]
    expected_header=$'point_id\tcontext_tokens\tep\tbatch_per_lane\tnodes\trun_dir\trequire_capacity_max\tknown_kv_capacity_tokens'
    [[ $(sed -n '1p' "$BATCH_CASES_FILE") == "$expected_header" ]]
    BATCH_CASE_ROWS=()
    while IFS= read -r case_row; do
        [[ -n "$case_row" ]]
        BATCH_CASE_ROWS[${#BATCH_CASE_ROWS[@]}]=$case_row
    done < <(sed -n '2,$p' "$BATCH_CASES_FILE")
    BATCH_CASE_COUNT=${#BATCH_CASE_ROWS[@]}
    (( BATCH_CASE_COUNT >= 1 && BATCH_CASE_COUNT <= 10 ))
    BATCH_NODES=""
    BATCH_POINT_IDS=$'\n'
    for case_row in "${BATCH_CASE_ROWS[@]}"; do
        IFS=$'\t' read -r point_id context_tokens ep_size batch_per_lane \
            case_nodes run_dir require_capacity_max known_kv_capacity_tokens \
            extra <<< "$case_row"
        [[ -z "${extra:-}" && "$point_id" =~ ^i[0-9]+-e[0-9]+-b[0-9]+$ ]]
        [[ "$context_tokens" =~ ^[1-9][0-9]*$ && "$batch_per_lane" =~ ^[1-9][0-9]*$ ]]
        [[ "$case_nodes" =~ ^[1-9][0-9]*$ && "$known_kv_capacity_tokens" =~ ^[0-9]+$ ]]
        [[ "$require_capacity_max" == 0 || "$require_capacity_max" == 1 ]]
        case "$BATCH_POINT_IDS" in
            *$'\n'"$point_id"$'\n'*)
                echo "duplicate point in packed manifest: $point_id" >&2
                exit 2 ;;
        esac
        BATCH_POINT_IDS+="$point_id"$'\n'
        case "$ep_size" in
            2|4) expected_nodes=1 ;;
            8) expected_nodes=2 ;;
            16) expected_nodes=4 ;;
            32) expected_nodes=8 ;;
            64) expected_nodes=16 ;;
            *) echo "invalid packed EP size: $ep_size" >&2; exit 2 ;;
        esac
        [[ "$case_nodes" == "$expected_nodes" ]]
        if [[ -z "$BATCH_NODES" ]]; then
            BATCH_NODES=$case_nodes
        else
            [[ "$case_nodes" == "$BATCH_NODES" ]] || {
                echo "all packed cases must use the same number of trays" >&2
                exit 2
            }
        fi
        case "$run_dir" in
            "$RESULTS_ROOT"/*) ;;
            *) echo "packed run directory is outside results root: $run_dir" >&2; exit 2 ;;
        esac
    done

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
        [[ -x "$EP_VENV_DIR/bin/python" ]]
        JOB_PREFIX=${FASTAFD_JOB_PREFIX:-fastafd:vllm}
        [[ "$JOB_PREFIX" =~ ^[A-Za-z0-9:_-]+$ ]]
        first_point=${BATCH_CASE_ROWS[0]%%$'\t'*}
        last_row=${BATCH_CASE_ROWS[$((BATCH_CASE_COUNT - 1))]}
        last_point=${last_row%%$'\t'*}
        JOB_NAME="${JOB_PREFIX}-qwen3-pack${BATCH_CASE_COUNT}-n${BATCH_NODES}-${first_point}-${last_point}"
        ! squeue -h -u "$USER" -o '%j' | grep -qx "$JOB_NAME" || {
            echo "the requested packed baseline job is already active" >&2
            exit 1
        }
        ACTIVE_JOB_REGEX=${FASTAFD_ACTIVE_JOB_REGEX:-'^fastafd:'}
        MAX_ACTIVE_JOBS=${FASTAFD_MAX_ACTIVE_JOBS:-4}
        [[ -n "$ACTIVE_JOB_REGEX" && "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]]
        ACTIVE_FASTAFD=$(squeue -h -u "$USER" -o '%j' | \
            awk -v job_regex="$ACTIVE_JOB_REGEX" '$0 ~ job_regex {n++} END {print n+0}')
        (( ACTIVE_FASTAFD < MAX_ACTIVE_JOBS )) || {
            echo "$MAX_ACTIVE_JOBS FastAFD experiment jobs matching $ACTIVE_JOB_REGEX are already active" >&2
            exit 1
        }
        for case_row in "${BATCH_CASE_ROWS[@]}"; do
            IFS=$'\t' read -r point_id context_tokens ep_size batch_per_lane \
                case_nodes run_dir require_capacity_max known_kv_capacity_tokens \
                extra <<< "$case_row"
            env -u FASTAFD_BATCH_CASES_FILE \
                FASTAFD_PREFLIGHT_ONLY=1 \
                MODEL_KEY=qwen3 CONTEXT_SPEC="$context_tokens" \
                EP_SIZE="$ep_size" BATCH="$batch_per_lane" \
                PROMPT_MODE=uniform \
                FASTAFD_REQUIRE_CAPACITY_MAX="$require_capacity_max" \
                FASTAFD_VLLM_KNOWN_KV_CAPACITY_TOKENS="$known_kv_capacity_tokens" \
                "$JOB_SCRIPT" qwen3 "$context_tokens" "$ep_size" \
                "$batch_per_lane" uniform
        done
        if [[ "${FASTAFD_BATCH_PREFLIGHT_ONLY:-0}" == 1 ]]; then
            printf 'packed preflight valid cases=%s nodes=%s qos=%s time_limit=01:00:00 case_timeout_seconds=%s\n' \
                "$BATCH_CASE_COUNT" "$BATCH_NODES" "$SLURM_QOS_REQUEST" \
                "$BATCH_CASE_TIMEOUT_SECONDS"
            exit 0
        fi
        for case_row in "${BATCH_CASE_ROWS[@]}"; do
            IFS=$'\t' read -r point_id context_tokens ep_size batch_per_lane \
                case_nodes run_dir require_capacity_max known_kv_capacity_tokens \
                extra <<< "$case_row"
            mkdir -p "$run_dir"
        done
        STAMP=$(date +%Y%m%d_%H%M%S)
        BATCH_JOB_DIR=${FASTAFD_BATCH_JOB_DIR:-$ROOT/status/case-packs/${STAMP}_${first_point}_${last_point}}
        mkdir -p "$BATCH_JOB_DIR/control"
        cp "$BATCH_CASES_FILE" "$BATCH_JOB_DIR/cases.tsv"
        BATCH_CASES_FILE=$BATCH_JOB_DIR/cases.tsv
        SBATCH_ARGS=(--parsable --nodes="$BATCH_NODES" --segment="$BATCH_NODES" \
            --gres=gpu:4 --qos="$SLURM_QOS_REQUEST" --time=01:00:00)
        if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
            SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
        fi
        JOB=$(sbatch "${SBATCH_ARGS[@]}" \
            --job-name="$JOB_NAME" \
            --output="$BATCH_JOB_DIR/slurm-%j.out" \
            --error="$BATCH_JOB_DIR/slurm-%j.err" \
            --export="ALL,FASTAFD_BATCH_CASES_FILE=$BATCH_CASES_FILE,FASTAFD_BATCH_JOB_DIR=$BATCH_JOB_DIR,FASTAFD_BATCH_CASE_TIMEOUT_SECONDS=$BATCH_CASE_TIMEOUT_SECONDS,FASTAFD_SLURM_QOS=$SLURM_QOS_REQUEST,FASTAFD_REPRO_ROOT=$ROOT,FASTAFD_RESULTS_ROOT=$RESULTS_ROOT,FASTAFD_SOURCE_REPO=$SOURCE_REPO,FASTAFD_IMAGE=$IMAGE,FASTAFD_EP_VENV_DIR=$EP_VENV_DIR,JOB_SCRIPT=$JOB_SCRIPT" \
            "$JOB_SCRIPT")
        printf 'submitted batch job=%s cases=%s nodes=%s qos=%s cases_file=%s job_dir=%s\n' \
            "$JOB" "$BATCH_CASE_COUNT" "$BATCH_NODES" "$SLURM_QOS_REQUEST" \
            "$BATCH_CASES_FILE" "$BATCH_JOB_DIR"
        exit 0
    fi

    : "${FASTAFD_BATCH_JOB_DIR:?}"
    if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
        [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == "$SLURM_QOS_REQUEST" ]]
        [[ "$SLURM_JOB_NUM_NODES" == "$BATCH_NODES" ]]
        HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
        exec srun --nodes="$BATCH_NODES" --ntasks="$BATCH_NODES" --ntasks-per-node=1 \
            --gres=gpu:4 --kill-on-bad-exit=1 \
            --container-image="$IMAGE" --container-mount-home \
            --container-mounts=/lustre:/lustre --no-container-remap-root \
            --container-env=NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
            env FASTAFD_IN_CONTAINER=1 HEAD_HOST="$HEAD_HOST" \
            FASTAFD_BATCH_CASES_FILE="$BATCH_CASES_FILE" \
            FASTAFD_BATCH_JOB_DIR="$FASTAFD_BATCH_JOB_DIR" \
            FASTAFD_BATCH_CASE_TIMEOUT_SECONDS="$BATCH_CASE_TIMEOUT_SECONDS" \
            FASTAFD_SLURM_QOS="$SLURM_QOS_REQUEST" \
            FASTAFD_REPRO_ROOT="$ROOT" FASTAFD_RESULTS_ROOT="$RESULTS_ROOT" \
            FASTAFD_SOURCE_REPO="$SOURCE_REPO" FASTAFD_IMAGE="$IMAGE" \
            FASTAFD_EP_VENV_DIR="$EP_VENV_DIR" JOB_SCRIPT="$JOB_SCRIPT" \
            bash "$JOB_SCRIPT"
    fi

    NODE_RANK=${SLURM_PROCID:?}
    [[ "$NODE_RANK" =~ ^[0-9]+$ && "$NODE_RANK" -lt "$BATCH_NODES" ]]
    [[ "$(uname -m)" == aarch64 ]]
    mkdir -p "$FASTAFD_BATCH_JOB_DIR/control"
    if [[ "$NODE_RANK" == 0 ]]; then
        if ! printf 'case_index\tpoint_id\trun_dir\tnode_exit_codes\n' \
            > "$FASTAFD_BATCH_JOB_DIR/case-status.tsv"; then
            echo "warning: could not initialize aggregate case-status.tsv" >&2
        fi
    fi
    case_index=0
    for case_row in "${BATCH_CASE_ROWS[@]}"; do
        IFS=$'\t' read -r point_id context_tokens ep_size batch_per_lane \
            case_nodes run_dir require_capacity_max known_kv_capacity_tokens \
            extra <<< "$case_row"
        case_dir=$FASTAFD_BATCH_JOB_DIR/control/case-$case_index
        mkdir -p "$case_dir/exits"
        set +e
        timeout --signal=TERM --kill-after=30s "$BATCH_CASE_TIMEOUT_SECONDS" \
            env -u FASTAFD_BATCH_CASES_FILE -u FASTAFD_BATCH_JOB_DIR \
            FASTAFD_IN_CONTAINER=1 FASTAFD_MASTER_PORT_OFFSET="$case_index" \
            FASTAFD_REPRO_ROOT="$ROOT" FASTAFD_RESULTS_ROOT="$RESULTS_ROOT" \
            FASTAFD_SOURCE_REPO="$SOURCE_REPO" FASTAFD_IMAGE="$IMAGE" \
            FASTAFD_EP_VENV_DIR="$EP_VENV_DIR" \
            FASTAFD_SLURM_QOS="$SLURM_QOS_REQUEST" \
            MODEL_KEY=qwen3 CONTEXT_SPEC="$context_tokens" \
            EP_SIZE="$ep_size" BATCH="$batch_per_lane" \
            PROMPT_MODE=uniform \
            FASTAFD_REQUIRE_CAPACITY_MAX="$require_capacity_max" \
            FASTAFD_VLLM_KNOWN_KV_CAPACITY_TOKENS="$known_kv_capacity_tokens" \
            RUN_DIR="$run_dir" JOB_SCRIPT="$JOB_SCRIPT" HEAD_HOST="$HEAD_HOST" \
            "$JOB_SCRIPT" qwen3 "$context_tokens" "$ep_size" \
            "$batch_per_lane" uniform
        case_rc=$?
        set -e
        exit_tmp=$case_dir/exits/node-$NODE_RANK.tmp.$$
        exit_recorded=0
        for ((attempt=1; attempt<=60; attempt++)); do
            if printf '%s\n' "$case_rc" > "$exit_tmp" && \
               mv "$exit_tmp" "$case_dir/exits/node-$NODE_RANK.exit"; then
                exit_recorded=1
                break
            fi
            sleep 1
        done
        (( exit_recorded )) || {
            echo "could not record node exit after 60 attempts" >&2
            exit 1
        }
        barrier_deadline=$((SECONDS + BATCH_CASE_TIMEOUT_SECONDS + 60))
        until [[ $(find "$case_dir/exits" -maxdepth 1 -name 'node-*.exit' | wc -l) -eq "$BATCH_NODES" ]]; do
            [[ $SECONDS -lt $barrier_deadline ]]
            sleep 2
        done
        if [[ "$NODE_RANK" == 0 ]]; then
            exit_codes=$(for ((node=0; node<BATCH_NODES; node++)); do \
                tr -d '\n' < "$case_dir/exits/node-$node.exit"; \
                (( node + 1 == BATCH_NODES )) || printf ','; \
            done)
            status_recorded=0
            for ((attempt=1; attempt<=10; attempt++)); do
                if printf '%s\t%s\t%s\t%s\n' "$case_index" "$point_id" \
                    "$run_dir" "$exit_codes" \
                    >> "$FASTAFD_BATCH_JOB_DIR/case-status.tsv"; then
                    status_recorded=1
                    break
                fi
                sleep 1
            done
            (( status_recorded )) || echo \
                "warning: case $case_index aggregate status was not recorded" >&2
        fi
        case_index=$((case_index + 1))
    done
    if [[ "$NODE_RANK" == 0 ]]; then
        touch "$FASTAFD_BATCH_JOB_DIR/COMPLETE"
    fi
    exit 0
fi

MODEL_KEY=${MODEL_KEY:-${1:-qwen3}}
CONTEXT_SPEC=${CONTEXT_SPEC:-${CONTEXT:-${2:-8k}}}
EP_SIZE=${EP_SIZE:-${3:-4}}
BATCH_ARG=${BATCH:-${4:-}}
PROMPT_MODE=${PROMPT_MODE:-${5:-uniform}}
SLURM_QOS_REQUEST=${FASTAFD_SLURM_QOS:-normal}
case "$SLURM_QOS_REQUEST" in
    normal|short) ;;
    *) echo "FASTAFD_SLURM_QOS must be normal or short" >&2; exit 2 ;;
esac
ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
RESULTS_ROOT=${FASTAFD_RESULTS_ROOT:-$ROOT}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$ROOT/source/FastAFD-3c716194}
IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-3c716194-cuda130-vllm-ep}
case "$CONTEXT_SPEC" in
    1k|1024) CONTEXT_MIN=1024; CONTEXT_MAX=1024 ;;
    2k|2048) CONTEXT_MIN=2048; CONTEXT_MAX=2048 ;;
    4k|4096) CONTEXT_MIN=4096; CONTEXT_MAX=4096 ;;
    8k|8192) CONTEXT_MIN=8192; CONTEXT_MAX=8192 ;;
    16k|16384) CONTEXT_MIN=16384; CONTEXT_MAX=16384 ;;
    32k|32768) CONTEXT_MIN=32768; CONTEXT_MAX=32768 ;;
    64k|65536) CONTEXT_MIN=65536; CONTEXT_MAX=65536 ;;
    128k|131072) CONTEXT_MIN=131072; CONTEXT_MAX=131072 ;;
    1k-4k|1024-4096) CONTEXT_MIN=1024; CONTEXT_MAX=4096 ;;
    4k-8k|4096-8192) CONTEXT_MIN=4096; CONTEXT_MAX=8192 ;;
    8k-32k|8192-32768) CONTEXT_MIN=8192; CONTEXT_MAX=32768 ;;
    32k-128k|32768-131072) CONTEXT_MIN=32768; CONTEXT_MAX=131072 ;;
    1k-16k|1024-16384) CONTEXT_MIN=1024; CONTEXT_MAX=16384 ;;
    4k-64k|4096-65536) CONTEXT_MIN=4096; CONTEXT_MAX=65536 ;;
    8k-128k|8192-131072) CONTEXT_MIN=8192; CONTEXT_MAX=131072 ;;
    1k-128k|1024-131072) CONTEXT_MIN=1024; CONTEXT_MAX=131072 ;;
    *) echo "context must be a supported uniform ISL or irregular ISL range" >&2; exit 2 ;;
esac
CONTEXT=$CONTEXT_MAX
case "$PROMPT_MODE" in
    uniform)
        (( CONTEXT_MIN == CONTEXT_MAX )) || {
            echo "uniform mode requires a single ISL, not a range" >&2
            exit 2
        }
        IRREGULAR=0 ;;
    irregular)
        (( CONTEXT_MIN < CONTEXT_MAX )) || {
            echo "irregular mode requires one of the supported ISL ranges" >&2
            exit 2
        }
        IRREGULAR=1 ;;
    *) echo "prompt mode must be uniform or irregular" >&2; exit 2 ;;
esac
if (( IRREGULAR )); then
    SHAPE_ID=r${CONTEXT_MIN}-${CONTEXT_MAX}
else
    SHAPE_ID=$CONTEXT
fi
case "$EP_SIZE" in
    # This launcher reserves a full four-GPU tray. EP2 launches two worker
    # processes on that exclusive tray and intentionally leaves the other two
    # allocation GPUs idle.
    2) NODES=1; GPUS_PER_NODE=2; SLURM_GPUS_PER_NODE=4 ;;
    4) NODES=1; GPUS_PER_NODE=4; SLURM_GPUS_PER_NODE=4 ;;
    8) NODES=2; GPUS_PER_NODE=4; SLURM_GPUS_PER_NODE=4 ;;
    16) NODES=4; GPUS_PER_NODE=4; SLURM_GPUS_PER_NODE=4 ;;
    32) NODES=8; GPUS_PER_NODE=4; SLURM_GPUS_PER_NODE=4 ;;
    64) NODES=16; GPUS_PER_NODE=4; SLURM_GPUS_PER_NODE=4 ;;
    *) echo "EP size must be 2, 4, 8, 16, 32, or 64" >&2; exit 2 ;;
esac

DEFAULT_BATCH=""
case "$MODEL_KEY:$CONTEXT:$EP_SIZE" in
    qwen3:8192:4) DEFAULT_BATCH=64 ;;
    qwen3:8192:8) DEFAULT_BATCH=88 ;;
    qwen3:8192:16) DEFAULT_BATCH=96 ;;
    qwen3:8192:32) DEFAULT_BATCH=101 ;;
    qwen3:8192:64) DEFAULT_BATCH=103 ;;
    qwen3:16384:4) DEFAULT_BATCH=32 ;;
    qwen3:16384:8) DEFAULT_BATCH=44 ;;
    qwen3:16384:16) DEFAULT_BATCH=48 ;;
    qwen3:16384:32) DEFAULT_BATCH=50 ;;
    qwen3:16384:64) DEFAULT_BATCH=52 ;;
    minimax:8192:4) DEFAULT_BATCH=48 ;;
    minimax:8192:8) DEFAULT_BATCH=68 ;;
    minimax:8192:16) DEFAULT_BATCH=74 ;;
    minimax:8192:32) DEFAULT_BATCH=78 ;;
    minimax:8192:64) DEFAULT_BATCH=79 ;;
    minimax:16384:4) DEFAULT_BATCH=24 ;;
    minimax:16384:8) DEFAULT_BATCH=34 ;;
    minimax:16384:16) DEFAULT_BATCH=37 ;;
    minimax:16384:32) DEFAULT_BATCH=39 ;;
    minimax:16384:64) DEFAULT_BATCH=39 ;;
esac
BATCH=${BATCH_ARG:-$DEFAULT_BATCH}
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || {
    echo "an explicit positive batch/lane is required for this model/context/EP" >&2
    exit 2
}
if [[ "$MODEL_KEY" == minimax && "$CONTEXT" != 8192 && "$CONTEXT" != 16384 ]]; then
    echo "MiniMax currently supports only the pinned 8K and 16K prompt contracts" >&2
    exit 2
fi
if (( IRREGULAR )) && [[ "$MODEL_KEY" != qwen3 ]]; then
    echo "irregular prompt-length ranges are supported only for Qwen3" >&2
    exit 2
fi
if (( IRREGULAR )) && (( EP_SIZE != 16 )); then
    echo "the Qwen3 irregular baseline sweep fixes DP=EP16" >&2
    exit 2
fi
if (( IRREGULAR )); then
    valid_sweep_batch=0
    for ((power=2; power<=BATCH; power*=2)); do
        if (( BATCH == power || BATCH == power + power / 2 )); then
            valid_sweep_batch=1
            break
        fi
    done
    (( valid_sweep_batch )) || {
        echo "irregular batch must follow 2^x or 2^x + 2^(x-1), x >= 1" >&2
        exit 2
    }
fi

case "$MODEL_KEY" in
    qwen3)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        MODEL_SOURCE_CONFIG_SHA256=702c46d431bb984db9035a1225186bbfdb52c0d19c82104df4a37cd005e0369e
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        MAX_MODEL_LEN=$((CONTEXT_MAX + 64))
        PROMPT_SOURCE_CONTEXT=8192
        REFERENCE_EP4_TPS=0
        MODEL_PROFILE_ID=native
        ROPE_SCALING_FACTOR=0
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=32768
        case "$CONTEXT" in
            65536)
                MODEL_PROFILE_ID=qwen3-235b-a22b-fp8-39eb2b06-yarn-f2p001953125-original32768-max65600
                ROPE_SCALING_FACTOR=2.001953125 ;;
            131072)
                MODEL_PROFILE_ID=qwen3-235b-a22b-fp8-39eb2b06-yarn-f4p001953125-original32768-max131136
                ROPE_SCALING_FACTOR=4.001953125 ;;
        esac ;;
    minimax)
        case "$CONTEXT" in
        8192)
            MAX_MODEL_LEN=8320
            PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
            REFERENCE_EP4_TPS=1512.5715032858584 ;;
        16384)
            MAX_MODEL_LEN=16640
            PROMPT_SHA256=918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca
            REFERENCE_EP4_TPS=735.3254410690338 ;;
        esac
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PROMPT_SOURCE_CONTEXT=$CONTEXT
        MODEL_SOURCE_CONFIG_SHA256=""
        MODEL_PROFILE_ID=native
        ROPE_SCALING_FACTOR=0
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=0 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac

# Keep the integer grid symmetric so every batch includes both endpoints and
# has the exact midpoint mean requested by the irregular-sweep contract.
PROMPT_LENGTHS=()
PROMPT_LENGTH_SUM=0
VLLM_REQUIRED_KV_TOKENS=0
for ((index=0; index<BATCH; index++)); do
    if (( BATCH == 1 )); then
        prompt_length=$CONTEXT_MAX
    elif (( 2 * index <= BATCH - 1 )); then
        prompt_length=$(( CONTEXT_MIN + (index * (CONTEXT_MAX - CONTEXT_MIN) + (BATCH - 1) / 2) / (BATCH - 1) ))
    else
        prompt_length=$(( CONTEXT_MAX - ((BATCH - 1 - index) * (CONTEXT_MAX - CONTEXT_MIN) + (BATCH - 1) / 2) / (BATCH - 1) ))
    fi
    PROMPT_LENGTHS+=("$prompt_length")
    PROMPT_LENGTH_SUM=$(( PROMPT_LENGTH_SUM + prompt_length ))
    VLLM_REQUIRED_KV_TOKENS=$(( VLLM_REQUIRED_KV_TOKENS + ((prompt_length + 64 + 15) / 16) * 16 ))
done
PROMPT_LENGTHS_CSV=$(IFS=,; echo "${PROMPT_LENGTHS[*]}")
case "$EP_SIZE" in
    2) VLLM_KNOWN_KV_CAPACITY_TOKENS=${FASTAFD_VLLM_KNOWN_KV_CAPACITY_TOKENS:-0} ;;
    4) VLLM_KNOWN_KV_CAPACITY_TOKENS=577536 ;;
    8) VLLM_KNOWN_KV_CAPACITY_TOKENS=726960 ;;
    16) VLLM_KNOWN_KV_CAPACITY_TOKENS=799776 ;;
    32) VLLM_KNOWN_KV_CAPACITY_TOKENS=837056 ;;
    64) VLLM_KNOWN_KV_CAPACITY_TOKENS=855632 ;;
esac
[[ "$VLLM_KNOWN_KV_CAPACITY_TOKENS" =~ ^[0-9]+$ ]]
if (( VLLM_KNOWN_KV_CAPACITY_TOKENS > 0 &&
      VLLM_REQUIRED_KV_TOKENS > VLLM_KNOWN_KV_CAPACITY_TOKENS )); then
    echo "batch requires $VLLM_REQUIRED_KV_TOKENS vLLM KV tokens/lane, conservative measured capacity is $VLLM_KNOWN_KV_CAPACITY_TOKENS" >&2
    exit 2
fi
REQUIRE_CAPACITY_MAX=${FASTAFD_REQUIRE_CAPACITY_MAX:-0}
[[ "$REQUIRE_CAPACITY_MAX" == 0 || "$REQUIRE_CAPACITY_MAX" == 1 ]]
CAPACITY_ESTIMATE_MAX_BATCH=0
if (( ! IRREGULAR )); then
    TOKENS_PER_SEQUENCE=$(( ((CONTEXT + 64 + 15) / 16) * 16 ))
    if (( VLLM_KNOWN_KV_CAPACITY_TOKENS > 0 )); then
        CAPACITY_ESTIMATE_MAX_BATCH=$(( VLLM_KNOWN_KV_CAPACITY_TOKENS / TOKENS_PER_SEQUENCE ))
    fi
    if (( REQUIRE_CAPACITY_MAX && CAPACITY_ESTIMATE_MAX_BATCH == 0 )); then
        echo "capacity-max validation requires a positive measured KV-capacity override" >&2
        exit 2
    fi
    if (( REQUIRE_CAPACITY_MAX && BATCH != CAPACITY_ESTIMATE_MAX_BATCH )); then
        echo "capacity-max sweep requires estimated batch $CAPACITY_ESTIMATE_MAX_BATCH at ISL $CONTEXT EP $EP_SIZE, got $BATCH" >&2
        exit 2
    fi
fi

MODEL_SOURCE_PATH=${FASTAFD_MODEL_PATH:-$ROOT/models/huggingface/hub/$MODEL_CACHE/snapshots/$MODEL_REVISION}
MODEL_PATH=$MODEL_SOURCE_PATH
MODEL_PROFILE_MANIFEST=""
if [[ "$MODEL_PROFILE_ID" != native ]]; then
    MODEL_PATH=$ROOT/model-profiles/$MODEL_PROFILE_ID
    MODEL_PROFILE_MANIFEST=$MODEL_PATH/fastafd-model-profile.json
fi
MODEL_PROFILE_SCRIPT=${FASTAFD_MODEL_PROFILE_SCRIPT:-$(dirname "$(realpath "$0")")/prepare_model_profile.py}
PROMPT_FILE=${FASTAFD_PROMPT_FILE:-$SOURCE_REPO/prompts/prompts_512x${PROMPT_SOURCE_CONTEXT}_seed20260527.txt}
GPU_MEMORY_UTILIZATION=${FASTAFD_VLLM_GPU_MEMORY_UTILIZATION:-0.94}
[[ "$GPU_MEMORY_UTILIZATION" == 0.94 ]] || {
    echo "published baseline comparison requires GPU memory utilization 0.94" >&2
    exit 2
}
if [[ -n "${SLURM_JOB_ID:-}" && "${FASTAFD_IN_CONTAINER:-0}" == 1 &&
      -z "${MODEL_CONFIG_SHA256:-}" ]]; then
    [[ -f "$MODEL_PATH/config.json" ]]
    MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
fi

if [[ "${FASTAFD_DRY_RUN:-0}" == 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'mode=vllm model=%s prompt_mode=%s context_min=%s context_max=%s irregular=%s ep=%s nodes=%s batch_per_lane=%s capacity_estimate_max_batch=%s require_capacity_max=%s prompt_lengths=%s prompt_length_sum=%s required_kv_tokens_per_lane=%s known_kv_capacity_tokens_per_lane=%s max_model_len=%s prompt_source=%s prefix_caching=off model_profile=%s rope_factor=%s\n' \
        "$MODEL_KEY" "$PROMPT_MODE" "$CONTEXT_MIN" "$CONTEXT_MAX" "$IRREGULAR" \
        "$EP_SIZE" "$NODES" "$BATCH" "$CAPACITY_ESTIMATE_MAX_BATCH" "$REQUIRE_CAPACITY_MAX" \
        "$PROMPT_LENGTHS_CSV" "$PROMPT_LENGTH_SUM" \
        "$VLLM_REQUIRED_KV_TOKENS" "$VLLM_KNOWN_KV_CAPACITY_TOKENS" \
        "$MAX_MODEL_LEN" "$PROMPT_FILE" "$MODEL_PROFILE_ID" "$ROPE_SCALING_FACTOR"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
    [[ -x "$EP_VENV_DIR/bin/python" && -f "$MODEL_SOURCE_PATH/model.safetensors.index.json" ]]
    if [[ "$MODEL_PROFILE_ID" != native ]]; then
        "$EP_VENV_DIR/bin/python" "$MODEL_PROFILE_SCRIPT" \
            --source "$MODEL_SOURCE_PATH" --output "$MODEL_PATH" \
            --profile-id "$MODEL_PROFILE_ID" --source-revision "$MODEL_REVISION" \
            --expected-source-config-sha256 "$MODEL_SOURCE_CONFIG_SHA256" \
            --max-position-embeddings "$MAX_MODEL_LEN" \
            --rope-factor "$ROPE_SCALING_FACTOR" \
            --original-max-position-embeddings "$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS"
    fi
    MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
    [[ -f "$MODEL_PATH/model.safetensors.index.json" ]]
    [[ -f "$PROMPT_FILE" ]]
    [[ $(sha256sum "$PROMPT_FILE" | awk '{print $1}') == "$PROMPT_SHA256" ]]
    if [[ "${FASTAFD_PREFLIGHT_ONLY:-0}" == 1 ]]; then
        printf 'preflight valid model=%s context=%s ep=%s nodes=%s batch=%s run_dir=%s\n' \
            "$MODEL_KEY" "$CONTEXT" "$EP_SIZE" "$NODES" "$BATCH" \
            "${RUN_DIR:-not-created}"
        exit 0
    fi
    JOB_PREFIX=${FASTAFD_JOB_PREFIX:-fastafd:vllm}
    [[ "$JOB_PREFIX" =~ ^[A-Za-z0-9:_-]+$ ]]
    if (( IRREGULAR )); then
        JOB_NAME="${JOB_PREFIX}-irregular-${MODEL_KEY}-${SHAPE_ID}-ep${EP_SIZE}-b${BATCH}"
    else
        JOB_NAME="${JOB_PREFIX}-${MODEL_KEY}-${CONTEXT}-ep${EP_SIZE}-b${BATCH}"
    fi
    ! squeue -h -u "$USER" -o '%j' | grep -qx "$JOB_NAME" || {
        echo "the requested baseline job is already active" >&2
        exit 1
    }
    ACTIVE_JOB_REGEX=${FASTAFD_ACTIVE_JOB_REGEX:-'^fastafd:'}
    [[ -n "$ACTIVE_JOB_REGEX" ]]
    MAX_ACTIVE_JOBS=${FASTAFD_MAX_ACTIVE_JOBS:-4}
    [[ "$MAX_ACTIVE_JOBS" =~ ^[1-9][0-9]*$ ]]
    ACTIVE_FASTAFD=$(squeue -h -u "$USER" -o '%j' | \
        awk -v job_regex="$ACTIVE_JOB_REGEX" '$0 ~ job_regex {n++} END {print n+0}')
    (( ACTIVE_FASTAFD < MAX_ACTIVE_JOBS )) || {
        echo "$MAX_ACTIVE_JOBS FastAFD experiment jobs matching $ACTIVE_JOB_REGEX are already active" >&2
        exit 1
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    if (( IRREGULAR )); then
        RUN_DIR=$RESULTS_ROOT/vllm_irregular_${MODEL_KEY}_${CONTEXT_MIN}-${CONTEXT_MAX}_dp${EP_SIZE}_ep${EP_SIZE}_b${BATCH}_$STAMP
    else
        RUN_DIR=$RESULTS_ROOT/vllm_${MODEL_KEY}_${CONTEXT}_dp${EP_SIZE}_ep${EP_SIZE}_b${BATCH}_${STAMP}_manual_na
    fi
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable --nodes="$NODES" --segment="$NODES" --gres="gpu:$SLURM_GPUS_PER_NODE" --qos="$SLURM_QOS_REQUEST")
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,FASTAFD_SLURM_QOS=$SLURM_QOS_REQUEST,MODEL_KEY=$MODEL_KEY,PROMPT_MODE=$PROMPT_MODE,CONTEXT_SPEC=$CONTEXT_SPEC,EP_SIZE=$EP_SIZE,NODES=$NODES,GPUS_PER_NODE=$GPUS_PER_NODE,SLURM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE,BATCH=$BATCH,CONTEXT=$CONTEXT,CONTEXT_MIN=$CONTEXT_MIN,CONTEXT_MAX=$CONTEXT_MAX,IRREGULAR=$IRREGULAR,SHAPE_ID=$SHAPE_ID,PROMPT_LENGTH_SUM=$PROMPT_LENGTH_SUM,VLLM_REQUIRED_KV_TOKENS=$VLLM_REQUIRED_KV_TOKENS,VLLM_KNOWN_KV_CAPACITY_TOKENS=$VLLM_KNOWN_KV_CAPACITY_TOKENS,CAPACITY_ESTIMATE_MAX_BATCH=$CAPACITY_ESTIMATE_MAX_BATCH,REQUIRE_CAPACITY_MAX=$REQUIRE_CAPACITY_MAX,MAX_MODEL_LEN=$MAX_MODEL_LEN,MODEL_REVISION=$MODEL_REVISION,MODEL_PATH=$MODEL_PATH,MODEL_SOURCE_PATH=$MODEL_SOURCE_PATH,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_SOURCE_CONFIG_SHA256=$MODEL_SOURCE_CONFIG_SHA256,MODEL_PROFILE_ID=$MODEL_PROFILE_ID,MODEL_PROFILE_MANIFEST=$MODEL_PROFILE_MANIFEST,ROPE_SCALING_FACTOR=$ROPE_SCALING_FACTOR,ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,PROMPT_SHA256=$PROMPT_SHA256,PROMPT_SOURCE_CONTEXT=$PROMPT_SOURCE_CONTEXT,REFERENCE_EP4_TPS=$REFERENCE_EP4_TPS,ROOT=$ROOT,RESULTS_ROOT=$RESULTS_ROOT,SOURCE_REPO=$SOURCE_REPO,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,PROMPT_FILE=$PROMPT_FILE,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s model=%s context_min=%s context_max=%s ep=%s nodes=%s batch=%s run_dir=%s\n' \
        "$JOB" "$MODEL_KEY" "$CONTEXT_MIN" "$CONTEXT_MAX" "$EP_SIZE" "$NODES" "$BATCH" "$RUN_DIR"
    exit 0
fi

: "${RUN_DIR:?}" "${JOB_SCRIPT:?}"
if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == "$SLURM_QOS_REQUEST" ]]
    [[ "$SLURM_JOB_NUM_NODES" == "$NODES" ]]
    HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
    exec srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gres="gpu:$GPUS_PER_NODE" --kill-on-bad-exit=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_IN_CONTAINER=1 HEAD_HOST="$HEAD_HOST" MODEL_KEY="$MODEL_KEY" \
        PROMPT_MODE="$PROMPT_MODE" CONTEXT_SPEC="$CONTEXT_SPEC" \
        EP_SIZE="$EP_SIZE" \
        NODES="$NODES" GPUS_PER_NODE="$GPUS_PER_NODE" \
        SLURM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE" \
        BATCH="$BATCH" CONTEXT="$CONTEXT" \
        CONTEXT_MIN="$CONTEXT_MIN" CONTEXT_MAX="$CONTEXT_MAX" \
        IRREGULAR="$IRREGULAR" SHAPE_ID="$SHAPE_ID" \
        PROMPT_LENGTHS_CSV="$PROMPT_LENGTHS_CSV" PROMPT_LENGTH_SUM="$PROMPT_LENGTH_SUM" \
        VLLM_REQUIRED_KV_TOKENS="$VLLM_REQUIRED_KV_TOKENS" \
        VLLM_KNOWN_KV_CAPACITY_TOKENS="$VLLM_KNOWN_KV_CAPACITY_TOKENS" \
        FASTAFD_VLLM_KNOWN_KV_CAPACITY_TOKENS="$VLLM_KNOWN_KV_CAPACITY_TOKENS" \
        CAPACITY_ESTIMATE_MAX_BATCH="$CAPACITY_ESTIMATE_MAX_BATCH" \
        REQUIRE_CAPACITY_MAX="$REQUIRE_CAPACITY_MAX" \
        FASTAFD_REQUIRE_CAPACITY_MAX="$REQUIRE_CAPACITY_MAX" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" MODEL_REVISION="$MODEL_REVISION" \
        MODEL_SOURCE_PATH="$MODEL_SOURCE_PATH" MODEL_CONFIG_SHA256="$MODEL_CONFIG_SHA256" \
        MODEL_SOURCE_CONFIG_SHA256="$MODEL_SOURCE_CONFIG_SHA256" \
        MODEL_PROFILE_ID="$MODEL_PROFILE_ID" MODEL_PROFILE_MANIFEST="$MODEL_PROFILE_MANIFEST" \
        ROPE_SCALING_FACTOR="$ROPE_SCALING_FACTOR" \
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS="$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS" \
        PROMPT_SHA256="$PROMPT_SHA256" PROMPT_SOURCE_CONTEXT="$PROMPT_SOURCE_CONTEXT" \
        REFERENCE_EP4_TPS="$REFERENCE_EP4_TPS" \
        ROOT="$ROOT" RESULTS_ROOT="$RESULTS_ROOT" SOURCE_REPO="$SOURCE_REPO" IMAGE="$IMAGE" \
        EP_VENV_DIR="$EP_VENV_DIR" MODEL_PATH="$MODEL_PATH" \
        PROMPT_FILE="$PROMPT_FILE" GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        RUN_DIR="$RUN_DIR" JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

NODE_RANK=${SLURM_PROCID:?}
[[ "$NODE_RANK" =~ ^[0-9]+$ && "$NODE_RANK" -lt "$NODES" ]]
[[ "$(uname -m)" == aarch64 ]]
PYTHON=$EP_VENV_DIR/bin/python
[[ -x "$PYTHON" && -f "$MODEL_PATH/model.safetensors.index.json" && -f "$PROMPT_FILE" ]]
[[ $(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}') == "$MODEL_CONFIG_SHA256" ]]
[[ "$MODEL_PROFILE_ID" == native || -f "$MODEL_PROFILE_MANIFEST" ]]
[[ $(sha256sum "$PROMPT_FILE" | awk '{print $1}') == "$PROMPT_SHA256" ]]

export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export EP_SIZE GPUS_PER_NODE VLLM_REQUIRED_KV_TOKENS REQUIRE_CAPACITY_MAX
export PROMPT_SOURCE_CONTEXT
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1 VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_DEEP_GEMM_WARMUP=skip VLLM_DEEPEPLL_FP8_DISPATCH=1
export VLLM_DEEPEPLL_UE8M0_DISPATCH=0
# DeepEP's low-latency reference path uses NVSHMEM rather than the generic
# NVLink IPC buffer.  The latter assumes eight GPUs in one host and cannot open
# cross-host CUDA IPC handles on OCI-HSG's four-GPU nodes.  Use GB200 MNNVL for
# the NVSHMEM transport.
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1
export TMPDIR=$RUN_DIR/tmp/node-$NODE_RANK
mkdir -p "$TMPDIR" "$RUN_DIR/nsys" "$RUN_DIR/control/profile-done"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RUN_DIR/gpu-processes-initial-node-$NODE_RANK.csv"
[[ $(wc -l < "$RUN_DIR/gpu-processes-initial-node-$NODE_RANK.csv") -eq 1 ]]
if [[ "$NODE_RANK" == 0 ]]; then
    INPUTS=("$PROMPT_FILE" "$MODEL_PATH/config.json" "$JOB_SCRIPT")
    [[ "$MODEL_PROFILE_ID" == native ]] || INPUTS+=("$MODEL_PROFILE_MANIFEST")
    sha256sum "${INPUTS[@]}" > "$RUN_DIR/inputs.sha256"
fi

# Fail a case coherently before rendezvous if any tray sees an incomplete
# shared Python environment or cannot launch Nsight Systems.  Without this
# barrier, healthy nodes can wait for the full case timeout after another node
# exits before torchrun rendezvous.
IMPORT_PREFLIGHT_DIR=$RUN_DIR/control/import-preflight
mkdir -p "$IMPORT_PREFLIGHT_DIR"
set +e
/usr/local/cuda/bin/nsys profile \
    --trace=osrt --sample=none --cpuctxsw=none --force-overwrite=true \
    --output="$TMPDIR/nsys-smoke" \
    "$PYTHON" -c \
    'import numpy.lib._array_utils_impl; import torch.fx.experimental._constant_symnode; import vllm' \
    > "$RUN_DIR/import-preflight-node-$NODE_RANK.log" 2>&1
import_preflight_rc=$?
set -e
import_preflight_tmp=$IMPORT_PREFLIGHT_DIR/node-$NODE_RANK.tmp.$$
printf '%s\n' "$import_preflight_rc" > "$import_preflight_tmp"
mv "$import_preflight_tmp" "$IMPORT_PREFLIGHT_DIR/node-$NODE_RANK.exit"
import_preflight_deadline=$((SECONDS + 60))
until [[ $(find "$IMPORT_PREFLIGHT_DIR" -maxdepth 1 -name 'node-*.exit' | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $import_preflight_deadline ]]
    sleep 2
done
for ((node=0; node<NODES; node++)); do
    [[ $(tr -d '\n' < "$IMPORT_PREFLIGHT_DIR/node-$node.exit") == 0 ]]
done

read -r -d '' BENCHMARK <<'PY' || true
import argparse
import faulthandler
import hashlib
import json
import os
import signal
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_world_group
from vllm.sampling_params import RequestOutputKind

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--model-key", choices=("qwen3", "minimax"), required=True)
p.add_argument("--revision", required=True)
p.add_argument("--prompts", type=Path, required=True)
p.add_argument("--run-dir", type=Path, required=True)
p.add_argument("--context", type=int, required=True)
p.add_argument("--context-min", type=int, required=True)
p.add_argument("--context-max", type=int, required=True)
p.add_argument("--prompt-mode", choices=("uniform", "irregular"), required=True)
p.add_argument("--prompt-lengths", required=True)
p.add_argument("--max-model-len", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--memory", type=float, required=True)
p.add_argument("--reference-ep4-tps", type=float, required=True)
a = p.parse_args()

rank, local_rank, world = (
    int(os.environ[x]) for x in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
)
faulthandler.register(signal.SIGUSR1, all_threads=True)

prompt_lengths = [int(value) for value in a.prompt_lengths.split(",")]
if len(prompt_lengths) != a.batch:
    raise RuntimeError((len(prompt_lengths), a.batch))
if prompt_lengths[0] != a.context_min or prompt_lengths[-1] != a.context_max:
    raise RuntimeError(prompt_lengths)
if a.context != a.context_max or a.max_model_len != a.context_max + 64:
    raise RuntimeError((a.context, a.context_max, a.max_model_len))
if any(left > right for left, right in zip(prompt_lengths, prompt_lengths[1:])):
    raise RuntimeError("prompt lengths are not monotonic")
if any(left + right != a.context_min + a.context_max for left, right in zip(prompt_lengths, reversed(prompt_lengths))):
    raise RuntimeError("prompt lengths are not symmetric")
if (a.prompt_mode == "irregular") != (a.context_min < a.context_max):
    raise RuntimeError((a.prompt_mode, a.context_min, a.context_max))

def stage(name):
    print(f"SWEEP_STAGE rank={rank} name={name}", flush=True)

stage("process_started")
gpus_per_node = int(os.environ["GPUS_PER_NODE"])
if (
    world not in (2, 4, 8, 16, 32, 64)
    or rank % gpus_per_node != local_rank
    or world != int(os.environ["EP_SIZE"])
    or a.memory != 0.94
):
    raise RuntimeError((rank, local_rank, world, a.memory))

capture_sizes = sorted({1, 2, 4, 8, 16, 32, 40, 64, 80, 128, 160, 256, a.batch})
stage("llm_init_start")
llm = LLM(
    model=a.model,
    tokenizer=a.model,
    trust_remote_code=a.model_key == "minimax",
    tensor_parallel_size=1,
    data_parallel_size=world,
    pipeline_parallel_size=1,
    distributed_executor_backend="external_launcher",
    enable_expert_parallel=True,
    all2all_backend="deepep_low_latency",
    moe_backend="deep_gemm",
    max_model_len=a.max_model_len,
    max_num_batched_tokens=40000,
    max_num_seqs=a.batch,
    gpu_memory_utilization=a.memory,
    enable_prefix_caching=False,
    seed=0,
    enforce_eager=False,
    compilation_config={
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": capture_sizes,
    },
    disable_log_stats=True,
)
stage("llm_init_done")

compilation = llm.llm_engine.vllm_config.compilation_config
cudagraph_mode = str(getattr(compilation, "cudagraph_mode", ""))
if bool(llm.model_config.enforce_eager) or "FULL_AND_PIECEWISE" not in cudagraph_mode:
    raise RuntimeError((llm.model_config.enforce_eager, cudagraph_mode))

cache = llm.llm_engine.vllm_config.cache_config
available = cache.num_gpu_blocks * cache.block_size
if cache.block_size != 16:
    raise RuntimeError(f"expected 16-token vLLM KV blocks, got {cache.block_size}")

def required_kv_tokens(lengths):
    return sum(
        ((length + 64 + cache.block_size - 1) // cache.block_size) * cache.block_size
        for length in lengths
    )

required = required_kv_tokens(prompt_lengths)
preflight_required = int(os.environ["VLLM_REQUIRED_KV_TOKENS"])
if required != preflight_required:
    raise RuntimeError((required, preflight_required))
if required > available:
    raise RuntimeError(
        f"configured irregular batch needs {required} KV tokens/lane; only {available} available"
    )
batch = a.batch

def distribution_for_batch(size):
    if size == 1:
        return [a.context_max]
    denominator = size - 1
    delta = a.context_max - a.context_min
    values = []
    for index in range(size):
        if 2 * index <= denominator:
            values.append(a.context_min + (index * delta + denominator // 2) // denominator)
        else:
            values.append(a.context_max - ((denominator - index) * delta + denominator // 2) // denominator)
    return values

capacity_batch = 1
while required_kv_tokens(distribution_for_batch(capacity_batch + 1)) <= available:
    capacity_batch += 1
require_capacity_max = bool(int(os.environ["REQUIRE_CAPACITY_MAX"]))
if require_capacity_max and batch != capacity_batch:
    raise RuntimeError(
        f"configured batch {batch} is not the post-init capacity maximum {capacity_batch}"
    )

source_bytes = a.prompts.read_bytes()
source = [x.strip() for x in source_bytes.decode().splitlines() if x.strip()]
if len(source) != 512:
    raise RuntimeError(len(source))
stage("source_loaded")

def qwen_chat_prompt_ids(text, target, tokenizer):
    source_ids = tokenizer.encode(text, add_special_tokens=False)
    if not source_ids:
        raise RuntimeError("empty tokenized source prompt")
    # Keep the prompt content derived only by cutting/repeating the pinned
    # source token stream. Adjust for the chat-template envelope and any BPE
    # boundary merge, then prove the final server-visible ISL exactly.
    template_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    content_count = target - len(template_tokens)
    for _ in range(64):
        repeats = (content_count + len(source_ids) - 1) // len(source_ids)
        selected = (source_ids * repeats)[:content_count]
        content = tokenizer.decode(
            selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        delta = target - len(prompt_ids)
        if delta == 0:
            return prompt_ids
        content_count += delta
        if content_count <= 0:
            break
    raise RuntimeError(f"cannot construct exact {target}-token chat prompt")

global_source = [source[i % len(source)] for i in range(world * batch)]
local_source = global_source[rank::world]
if a.model_key == "qwen3":
    tokenizer = llm.get_tokenizer()
    stage("prompt_transform_start")
    local_prompts = [
        {"prompt_token_ids": qwen_chat_prompt_ids(text, target, tokenizer)}
        for text, target in zip(local_source, prompt_lengths, strict=True)
    ]
    stage("prompt_transform_done")
    prompt_transform = "cut_or_repeat_pinned_8k_content_tokens_then_qwen_chat_template"
else:
    local_prompts = local_source
    prompt_transform = "pinned_text_unchanged"
inputs = list(llm._preprocess_cmpl(local_prompts))
stage("prompt_preprocess_done")
lengths = [len(x["prompt_token_ids"]) for x in inputs]
if a.model_key == "qwen3" and lengths != prompt_lengths:
    raise RuntimeError((lengths, prompt_lengths))
if a.model_key == "minimax" and (
    min(lengths) <= 0 or max(lengths) + 64 > a.max_model_len
):
    raise RuntimeError(sorted(set(lengths)))

params = SamplingParams(
    temperature=0,
    max_tokens=64,
    min_tokens=64,
    ignore_eos=True,
    detokenize=False,
    output_kind=RequestOutputKind.DELTA,
)
request_ids = {f"r{rank}-{i}" for i in range(batch)}
for request_id, engine_input in zip(sorted(request_ids), inputs, strict=True):
    llm.llm_engine.add_request(request_id, engine_input, params)
stage("requests_added")

generated = {request_id: 0 for request_id in request_ids}
group = get_world_group().cpu_group
for warmup_step in range(64):
    stage(f"warmup_step_{warmup_step}_start")
    outputs = llm.llm_engine.step()
    stage(f"warmup_step_{warmup_step}_done")
    for output in outputs:
        if output.request_id not in generated or len(output.outputs[0].token_ids) != 1:
            raise RuntimeError(output)
        generated[output.request_id] += 1
    # EP ranks must execute the same number of engine steps. A local early exit
    # can strand peers inside the next DeepEP collective when prompt scheduling
    # completes one rank a step earlier (observed deterministically at batch 3).
    globally_ready = torch.tensor(
        int(min(generated.values()) >= 1), dtype=torch.int32, device="cpu"
    )
    dist.all_reduce(globally_ready, op=dist.ReduceOp.MIN, group=group)
    stage(f"warmup_step_{warmup_step}_globally_ready_{globally_ready.item()}")
    if globally_ready.item():
        break
else:
    raise RuntimeError("resident-batch warmup did not finish")
if max(generated.values()) + 15 >= 64:
    raise RuntimeError(generated)

stage("measurement_barrier_start")
dist.barrier(group=group)
stage("measurement_barrier_done")
torch.cuda.synchronize()
cudart = torch.cuda.cudart()
if int(cudart.cudaProfilerStart()) != 0:
    raise RuntimeError(f"rank {rank}: cudaProfilerStart failed")
dist.barrier(group=group)
step_ms = []
for step in range(15):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    outputs = llm.llm_engine.step()
    torch.cuda.synchronize()
    step_ms.append((time.perf_counter_ns() - start) / 1e6)
    if len(outputs) != batch or {x.request_id for x in outputs} != request_ids:
        raise RuntimeError((step, len(outputs)))
    if any(len(x.outputs[0].token_ids) != 1 or x.finished for x in outputs):
        raise RuntimeError(f"invalid decode output at step {step}")
dist.barrier(group=group)
if int(cudart.cudaProfilerStop()) != 0:
    raise RuntimeError(f"rank {rank}: cudaProfilerStop failed")
dist.barrier(group=group)

all_ms = [None] * world
dist.all_gather_object(all_ms, step_ms, group=group)
all_lengths = [None] * world
dist.all_gather_object(all_lengths, lengths, group=group)
if any(rank_lengths != prompt_lengths for rank_lengths in all_lengths):
    raise RuntimeError("DP ranks do not share the requested prompt-length distribution")
local_token_hashes = [
    hashlib.sha256(
        b"".join(int(token).to_bytes(4, "little") for token in x["prompt_token_ids"])
    ).hexdigest()
    for x in inputs
]
all_token_hashes = [None] * world
dist.all_gather_object(all_token_hashes, local_token_hashes, group=group)
(a.run_dir / f"rank-{rank}.json").write_text(json.dumps({
    "rank": rank,
    "batch": batch,
    "step_ms": step_ms,
    "warmup_token_counts": sorted(generated.values()),
    "prompt_lengths": sorted(set(lengths)),
}, indent=2) + "\n")
if rank == 0:
    synchronized = [max(values[i] for values in all_ms) for i in range(15)]
    observed_lengths = [length for values in all_lengths for length in values]
    mean_ms = statistics.fmean(synchronized)
    median_ms = statistics.median(synchronized)
    result = {
        "model": a.model,
        "model_key": a.model_key,
        "model_revision": a.revision,
        "model_config_sha256": hashlib.sha256(
            (Path(a.model) / "config.json").read_bytes()
        ).hexdigest(),
        "model_max_position_embeddings": int(
            llm.model_config.hf_config.max_position_embeddings
        ),
        "model_rope_scaling": getattr(
            llm.model_config.hf_config, "rope_scaling", None
        ),
        "context_tokens": a.context,
        "context_range_tokens": {"min": a.context_min, "max": a.context_max},
        "prompt_mode": a.prompt_mode,
        "irregular_prompt_lengths": a.prompt_mode == "irregular",
        "prompt_length_distribution": "inclusive_uniform_symmetric_integer_linspace",
        "prompt_lengths_per_dp_lane": prompt_lengths,
        "prompt_length_sum_per_dp_lane": sum(prompt_lengths),
        "data_parallel_size": world,
        "expert_parallel_size": world,
        "batch_per_dp_lane": batch,
        "batch_selection": (
            "exact post-init KV-capacity ceiling"
            if require_capacity_max
            else "explicit sweep point; post-init KV capacity reported separately"
        ),
        "capacity_max_batch": capacity_batch,
        "capacity_max_batch_semantics": "largest inclusive-uniform distribution batch fitting the post-init KV cache",
        "global_prompts": world * batch,
        "gpu_memory_utilization": a.memory,
        "allow_long_max_model_len": bool(
            int(os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"])
        ),
        "deepep_buffer_size_mb": int(os.environ["VLLM_DEEPEP_BUFFER_SIZE_MB"]),
        "deepep_low_latency_use_mnnvl": bool(
            int(os.environ["VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL"])
        ),
        "kv_capacity_tokens_per_gpu": available,
        "required_kv_tokens_per_gpu": required,
        "unused_kv_tokens_per_gpu": available - required,
        "kv_block_size_tokens": cache.block_size,
        "kv_preflight_method": "sum ceil((prompt_tokens + 64) / 16) * 16",
        "prompt_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prompt_source_context_tokens": int(os.environ["PROMPT_SOURCE_CONTEXT"]),
        "prompt_transform": prompt_transform,
        "prompt_assignment": "global prompt i uses source prompt i modulo 512",
        "prompt_token_hashes_by_rank": all_token_hashes,
        "observed_prompt_tokens": {
            "min": min(observed_lengths),
            "max": max(observed_lengths),
        },
        "observed_prompt_lengths_by_rank": all_lengths,
        "cuda_graph": {
            "enabled": True,
            "enforce_eager": bool(llm.model_config.enforce_eager),
            "mode": cudagraph_mode,
            "requested_capture_batch": batch,
            "capture_sizes": capture_sizes,
        },
        "wall": {
            "synchronized_step_ms": synchronized,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "mean_tokens_per_second_per_gpu": batch * 1000 / mean_ms,
            "median_tokens_per_second_per_gpu": batch * 1000 / median_ms,
        },
        "reference_dp4_ep4_cuda_tokens_per_second_per_gpu": a.reference_ep4_tps,
        "measurement": (
            "15 full-resident decode steps; request wall=max across all DP ranks; "
            "primary latency/TPS use first-to-last CUDA graph kernel spans from "
            "all DP GPUs and steps"
        ),
    }
    (a.run_dir / "baseline-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
PY

# Derive a high rendezvous port from the allocation rather than EP size so
# unrelated jobs with the same topology cannot collide on a shared host. Each
# case in a packed allocation gets a distinct offset so a failed case cannot
# leave a rendezvous socket that interferes with the next case.
MASTER_PORT_OFFSET=${FASTAFD_MASTER_PORT_OFFSET:-0}
[[ "$MASTER_PORT_OFFSET" =~ ^[0-9]+$ && "$MASTER_PORT_OFFSET" -le 9 ]]
MASTER_PORT=$((20000 + (SLURM_JOB_ID + MASTER_PORT_OFFSET) % 30000))
/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --trace-fork-before-exec=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="$RUN_DIR/nsys/decode-node-$NODE_RANK" \
    "$PYTHON" -m torch.distributed.run \
    --nnodes="$NODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_HOST" --master-port="$MASTER_PORT" \
    --no-python "$PYTHON" -c "$BENCHMARK" \
    --model "$MODEL_PATH" --model-key "$MODEL_KEY" --revision "$MODEL_REVISION" \
    --prompts "$PROMPT_FILE" --run-dir "$RUN_DIR" --context "$CONTEXT" \
    --context-min "$CONTEXT_MIN" --context-max "$CONTEXT_MAX" \
    --prompt-mode "$PROMPT_MODE" --prompt-lengths "$PROMPT_LENGTHS_CSV" \
    --max-model-len "$MAX_MODEL_LEN" --batch "$BATCH" \
    --memory "$GPU_MEMORY_UTILIZATION" --reference-ep4-tps "$REFERENCE_EP4_TPS" \
    2>&1 | tee "$RUN_DIR/torchrun-node-$NODE_RANK.log"

REPORT="$RUN_DIR/nsys/decode-node-$NODE_RANK.nsys-rep"
[[ -s "$REPORT" && -s "$RUN_DIR/baseline-result.json" ]]
/usr/local/cuda/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum \
    "$REPORT" > "$RUN_DIR/nsys/stats-node-$NODE_RANK.txt"
SQLITE="$RUN_DIR/nsys/decode-node-$NODE_RANK.sqlite"
[[ -s "$SQLITE" ]]
printf '%s\n' "$SQLITE" > "$RUN_DIR/control/profile-done/$NODE_RANK"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RUN_DIR/gpu-processes-final-node-$NODE_RANK.csv"
[[ $(wc -l < "$RUN_DIR/gpu-processes-final-node-$NODE_RANK.csv") -eq 1 ]]

if [[ "$NODE_RANK" != 0 ]]; then
    exit 0
fi

deadline=$((SECONDS + 180))
until [[ $(find "$RUN_DIR/control/profile-done" -maxdepth 1 -type f | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $deadline ]]
    sleep 2
done

"$PYTHON" - "$RUN_DIR/baseline-result.json" "$EP_SIZE" "$REFERENCE_EP4_TPS" \
    "$RUN_DIR/control/profile-done" "$NODES" "$GPUS_PER_NODE" <<'PY'
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import sqlite3
import statistics
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
world = int(sys.argv[2])
reference = float(sys.argv[3])
done_dir = Path(sys.argv[4])
nodes = int(sys.argv[5])
gpus_per_node = int(sys.argv[6])
if nodes * gpus_per_node != world:
    raise RuntimeError((nodes, gpus_per_node, world))
sqlite_paths = [
    Path((done_dir / str(node)).read_text().strip()) for node in range(nodes)
]
rows = []
spans_by_gpu = {}
for node, sqlite_path in enumerate(sqlite_paths):
    con = sqlite3.connect(sqlite_path)
    node_rows = list(con.execute(
        "select deviceId, sum(end-start), count(*) "
        "from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId order by deviceId"
    ))
    if len(node_rows) != gpus_per_node or any(
        total <= 0 for _, total, _ in node_rows
    ):
        raise RuntimeError((node, node_rows))
    rows.extend((node, device, total, count) for device, total, count in node_rows)
    per_process = defaultdict(list)
    for (
        global_pid,
        device_id,
        context_id,
        graph_id,
        graph_node_id,
        start,
        end,
    ) in con.execute(
        """
        SELECT globalPid,deviceId,contextId,graphId,graphNodeId,start,end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE graphId IS NOT NULL
        """
    ):
        per_process[(int(global_pid), int(device_id), int(context_id))].append(
            (int(graph_id), int(graph_node_id), int(start), int(end))
        )
    con.close()
    if len(per_process) != gpus_per_node:
        raise RuntimeError(
            f"node {node}: expected {gpus_per_node} profiled GPU processes, "
            f"found {len(per_process)}"
        )
    for process, events in sorted(per_process.items()):
        graph_counts = Counter(graph_id for graph_id, _, _, _ in events)
        graph_id = min(
            graph_counts,
            key=lambda candidate: (-graph_counts[candidate], candidate),
        )
        graph_events = [event for event in events if event[0] == graph_id]
        first_by_node = {}
        for _, graph_node_id, start, _ in graph_events:
            first_by_node[graph_node_id] = min(
                first_by_node.get(graph_node_id, start), start
            )
        first_node = min(
            first_by_node, key=lambda node_id: (first_by_node[node_id], node_id)
        )
        starts = sorted(
            start
            for _, graph_node_id, start, _ in graph_events
            if graph_node_id == first_node
        )
        if len(starts) != 15:
            raise RuntimeError(
                f"node {node} process {process}: expected 15 graph replays, "
                f"found {len(starts)}"
            )
        replay_min = [None] * 15
        replay_max = [None] * 15
        for _, _, start, end in graph_events:
            replay = bisect_right(starts, start) - 1
            if replay < 0:
                continue
            replay_min[replay] = (
                start if replay_min[replay] is None
                else min(replay_min[replay], start)
            )
            replay_max[replay] = (
                end if replay_max[replay] is None
                else max(replay_max[replay], end)
            )
        if any(start is None or end is None for start, end in zip(replay_min, replay_max)):
            raise RuntimeError(f"node {node} process {process}: empty graph replay")
        key = f"node{node}:pid{process[0]}:device{process[1]}:context{process[2]}"
        spans_by_gpu[key] = [
            (end - start) / 1e6
            for start, end in zip(replay_min, replay_max)
        ]
if len(rows) != world:
    raise RuntimeError(len(rows))
if len(spans_by_gpu) != world:
    raise RuntimeError(len(spans_by_gpu))

result = json.loads(result_path.read_text())
batch = int(result["batch_per_dp_lane"])
mean_ns = sum(total for _, _, total, _ in rows) / (world * 15)
tps = batch * 1e9 / mean_ns
cuda_result = {
    "per_device_total_ns": {
        f"node{node}:device{device}": total for node, device, total, _ in rows
    },
    "per_device_kernel_count": {
        f"node{node}:device{device}": count for node, device, _, count in rows
    },
    "mean_ms_per_step_per_gpu": mean_ns / 1e6,
    "tokens_per_second_per_gpu": tps,
    "method": (
        "sum of CUDA kernel durations in every node's 15-step profiler range "
        "/ (world GPUs x 15 steps)"
    ),
}
if reference > 0:
    cuda_result["dp4_ep4_speed_ratio"] = tps / reference
    cuda_result["dp4_ep4_delta_percent"] = (tps / reference - 1) * 100
result["cuda_kernels"] = cuda_result
all_spans = [span for spans in spans_by_gpu.values() for span in spans]
if len(all_spans) != world * 15:
    raise RuntimeError((len(all_spans), world * 15))
cuda_wall_ms = statistics.fmean(all_spans)
result["cuda_wall"] = {
    "latency_basis": "vllm_cuda_graph_span_mean_per_step_per_gpu",
    "latency_ms": cuda_wall_ms,
    "tokens_per_second_per_gpu": batch * 1000 / cuda_wall_ms,
    "sample_count": len(all_spans),
    "steps_per_gpu": 15,
    "gpu_processes": world,
    "per_gpu_step_spans_ms": spans_by_gpu,
    "span_min_ms": min(all_spans),
    "span_max_ms": max(all_spans),
    "method": (
        "for each DP GPU and each of 15 captured decode replays, end of last "
        "CUDA graph kernel minus start of first CUDA graph kernel; arithmetic "
        "mean across GPUs and replays"
    ),
}
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

printf 'FASTAFD_VLLM_SUCCESS result=%s\n' "$RUN_DIR/baseline-result.json" \
    | tee "$RUN_DIR/SUCCESS"
