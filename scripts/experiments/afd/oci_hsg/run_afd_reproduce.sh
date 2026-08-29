#!/usr/bin/env bash
# Run a reproducible multi-tray FastAFD decode point.
# Usage: ./run_afd.sh [qwen3|minimax] [1k|...|128k|1k-4k|...] [input-batch/attention-GPU] [nodes] [uniform|irregular]
#
# Omitting batch/nodes selects the published 8K/16K shape. Arbitrary Qwen3 ISLs
# and irregular ranges require an explicit batch and use a fixed caller-selected
# topology. Irregular Qwen3 ranges use an inclusive, evenly spaced prompt-length
# distribution independently on every attention-DP worker. The batch argument
# is the total real input batch per attention-DP lane before AFD's two-way
# microbatch split; it is never a per-microbatch batch size.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:afd

set -euo pipefail
ulimit -s 8192

MODEL_KEY=${MODEL_KEY:-${1:-qwen3}}
CONTEXT_SPEC=${CONTEXT_SPEC:-${CONTEXT:-${2:-8k}}}
BATCH_ARG=${BATCH:-${3:-}}
NODES_ARG=${NODES:-${4:-}}
PROMPT_MODE=${PROMPT_MODE:-${5:-uniform}}
SLURM_QOS_REQUEST=${FASTAFD_SLURM_QOS:-normal}
case "$SLURM_QOS_REQUEST" in
    normal|short) ;;
    *) echo "FASTAFD_SLURM_QOS must be normal or short" >&2; exit 2 ;;
esac
JOB_TIME_LIMIT=${FASTAFD_JOB_TIME_LIMIT:-01:00:00}
[[ "$JOB_TIME_LIMIT" =~ ^([0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
    echo "FASTAFD_JOB_TIME_LIMIT must use [days-]HH:MM:SS" >&2
    exit 2
}
AFD_MODEL_PLACEMENT=${FASTAFD_AFD_MODEL_PLACEMENT:-${AFD_MODEL_PLACEMENT:-legacy}}
case "$AFD_MODEL_PLACEMENT" in
    legacy) DEFAULT_AFD_MOE_BACKEND=megamoe_m2n ;;
    fmha-only) DEFAULT_AFD_MOE_BACKEND=megamoe ;;
    *) echo "FASTAFD_AFD_MODEL_PLACEMENT must be legacy or fmha-only" >&2; exit 2 ;;
esac
MINISGL_AFD_MOE_BACKEND=${MINISGL_AFD_MOE_BACKEND:-$DEFAULT_AFD_MOE_BACKEND}
case "$MINISGL_AFD_MOE_BACKEND" in
    deepep|megamoe|megamoe_m2n) ;;
    *) echo "MINISGL_AFD_MOE_BACKEND must be deepep, megamoe, or megamoe_m2n" >&2; exit 2 ;;
esac
if [[ "$MINISGL_AFD_MOE_BACKEND" == megamoe && "$AFD_MODEL_PLACEMENT" != fmha-only ]]; then
    echo "MINISGL_AFD_MOE_BACKEND=megamoe requires fmha-only placement" >&2
    exit 2
fi
if [[ "$MINISGL_AFD_MOE_BACKEND" == megamoe_m2n && "$AFD_MODEL_PLACEMENT" != legacy ]]; then
    echo "MINISGL_AFD_MOE_BACKEND=megamoe_m2n requires legacy placement" >&2
    exit 2
fi
MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=${MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE:-fp4}
if [[ "$MINISGL_AFD_MOE_BACKEND" == megamoe && "$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" != fp4 ]]; then
    echo "FP8xFP4 MegaMoE requires MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4" >&2
    exit 2
fi
export AFD_MODEL_PLACEMENT MINISGL_AFD_MOE_BACKEND
export MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE
NSYS_CUDA_GRAPH_TRACE=${NSYS_CUDA_GRAPH_TRACE:-node}
case "$NSYS_CUDA_GRAPH_TRACE" in
    node|none) ;;
    *) echo "NSYS_CUDA_GRAPH_TRACE must be node or none" >&2; exit 2 ;;
esac
NSYS_CAPTURE_DECODE_STEPS=15
TRACE_WARMUP_DECODE_STEPS=1
# Prefill produces the first sampled token.  Keep one full-batch decode warmup
# plus the complete measured window available afterward.
OUTPUT_TOKENS=$((1 + TRACE_WARMUP_DECODE_STEPS + NSYS_CAPTURE_DECODE_STEPS))
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

EXPECTED_HEAD=${FASTAFD_EXPECTED_HEAD:-3c7161949310b6d59d6b4cf9bf997a4935c8113b}
ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
RESULTS_ROOT=${FASTAFD_RESULTS_ROOT:-$ROOT}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$ROOT/source/FastAFD-${EXPECTED_HEAD:0:8}}
EXPECTED_SOURCE_MANIFEST=${FASTAFD_EXPECTED_SOURCE_MANIFEST:-}
IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-${EXPECTED_HEAD:0:8}-cuda130-vllm-ep}
HF_CACHE=$ROOT/models/huggingface/hub

validate_source_repo() {
    [[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
    local status_paths expected_paths
    status_paths=$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all | \
        sed 's/^...//' | LC_ALL=C sort)
    if [[ -n "$EXPECTED_SOURCE_MANIFEST" ]]; then
        [[ -f "$EXPECTED_SOURCE_MANIFEST" ]]
        awk '
            NF != 2 || $1 !~ /^[0-9a-f]{64}$/ || $2 ~ /^\// || $2 ~ /(^|\/)\.\.($|\/)/ { exit 1 }
        ' "$EXPECTED_SOURCE_MANIFEST"
        expected_paths=$(awk '{print $2}' "$EXPECTED_SOURCE_MANIFEST" | LC_ALL=C sort)
        [[ -n "$expected_paths" && "$status_paths" == "$expected_paths" ]]
        (cd "$SOURCE_REPO" && sha256sum --check --strict "$EXPECTED_SOURCE_MANIFEST")
    else
        [[ -z "$status_paths" ]]
    fi
}

DEFAULT_BATCH=""
DEFAULT_NODES=""
case "$MODEL_KEY" in
    qwen3)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        MODEL_SOURCE_CONFIG_SHA256=702c46d431bb984db9035a1225186bbfdb52c0d19c82104df4a37cd005e0369e
        PRESET=scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SOURCE_CONTEXT=8192
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        MAX_SEQ_LEN=$((CONTEXT_MAX + 64))
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
        esac
        case "$CONTEXT" in
            8192) DEFAULT_BATCH=96; DEFAULT_NODES=8 ;;
            16384) DEFAULT_BATCH=48; DEFAULT_NODES=12 ;;
            *) DEFAULT_NODES=8 ;;
        esac ;;
    minimax)
        case "$CONTEXT" in
        8192)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        DEFAULT_NODES=18; DEFAULT_BATCH=72; MAX_SEQ_LEN=8320 ;;
        16384)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_16k_b36_dynamicnode_mb2_nsys_alignment.sh
        PROMPT_SHA256=918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca
        DEFAULT_NODES=18; DEFAULT_BATCH=36; MAX_SEQ_LEN=16640 ;;
        *) echo "MiniMax currently supports only the pinned 8K and 16K prompt contracts" >&2; exit 2 ;;
        esac
        PROMPT_SOURCE_CONTEXT=$CONTEXT
        MODEL_SOURCE_CONFIG_SHA256=""
        MODEL_PROFILE_ID=native
        ROPE_SCALING_FACTOR=0
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=0 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac
BATCH=${BATCH_ARG:-$DEFAULT_BATCH}
NODES=${NODES_ARG:-$DEFAULT_NODES}
[[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || {
    echo "an explicit positive batch/attention-GPU is required for this context" >&2
    exit 2
}
[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 2 )) || {
    echo "nodes must be an integer of at least two" >&2
    exit 2
}
if (( IRREGULAR )) && [[ "$MODEL_KEY" != qwen3 ]]; then
    echo "irregular prompt-length ranges are supported only for Qwen3" >&2
    exit 2
fi
if (( IRREGULAR )) && (( NODES != 8 )); then
    echo "the Qwen3 irregular sweep fixes AFD at eight nodes" >&2
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

# Construct a symmetric integer linspace. The lower half is rounded to nearest
# token and the upper half is its exact complement, preserving both endpoints
# and an exact (min + max) / 2 mean for every batch size.
PROMPT_LENGTHS=()
PROMPT_LENGTH_SUM=0
AFD_REQUIRED_KV_TOKENS=0
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
    AFD_REQUIRED_KV_TOKENS=$(( AFD_REQUIRED_KV_TOKENS + ((prompt_length + 64 + 63) / 64) * 64 ))
done
PROMPT_LENGTHS_CSV=$(IFS=,; echo "${PROMPT_LENGTHS[*]}")
AFD_KV_CAPACITY_TOKENS=839424
NOMINAL_AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS
if (( AFD_REQUIRED_KV_TOKENS > AFD_KV_CAPACITY_TOKENS )); then
    echo "batch requires $AFD_REQUIRED_KV_TOKENS AFD KV tokens/attention-GPU, capacity is $AFD_KV_CAPACITY_TOKENS" >&2
    exit 2
fi
CAPACITY_MAX_BATCH=0
RAW_KV_MAX_BATCH=0
SCHEDULER_ADMISSION_REQUIRED_TOKENS=0
SCHEDULER_ADMISSION_UNUSED_TOKENS=0
SCHEDULER_RUNNING_RESERVATION_TOKENS=0
REQUIRE_CAPACITY_MAX=${FASTAFD_REQUIRE_CAPACITY_MAX:-0}
ALLOW_OBSERVED_CAPACITY_PROBE=${FASTAFD_ALLOW_OBSERVED_CAPACITY_PROBE:-0}
[[ "$REQUIRE_CAPACITY_MAX" == 0 || "$REQUIRE_CAPACITY_MAX" == 1 ]]
[[ "$ALLOW_OBSERVED_CAPACITY_PROBE" == 0 || "$ALLOW_OBSERVED_CAPACITY_PROBE" == 1 ]]
if (( ! IRREGULAR )); then
    TOKENS_PER_SEQUENCE=$(( ((CONTEXT + 64 + 63) / 64) * 64 ))
    RAW_KV_MAX_BATCH=$(( AFD_KV_CAPACITY_TOKENS / TOKENS_PER_SEQUENCE ))
    RESIDENT_PROMPT_TOKENS=$(( ((CONTEXT + 63) / 64) * 64 ))
    FRESH_REQUEST_ESTIMATED_TOKENS=$(( CONTEXT + OUTPUT_TOKENS ))
    SCHEDULER_RUNNING_RESERVATION_TOKENS=$(( OUTPUT_TOKENS - 1 + 63 ))
    CAPACITY_MAX_BATCH=$((
        (AFD_KV_CAPACITY_TOKENS - FRESH_REQUEST_ESTIMATED_TOKENS) /
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) + 1
    ))
    SCHEDULER_ADMISSION_REQUIRED_TOKENS=$((
        (CAPACITY_MAX_BATCH - 1) *
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) +
        FRESH_REQUEST_ESTIMATED_TOKENS
    ))
    SCHEDULER_ADMISSION_UNUSED_TOKENS=$((
        AFD_KV_CAPACITY_TOKENS - SCHEDULER_ADMISSION_REQUIRED_TOKENS
    ))
    (( CAPACITY_MAX_BATCH > 0 && CAPACITY_MAX_BATCH <= RAW_KV_MAX_BATCH ))
    (( SCHEDULER_ADMISSION_REQUIRED_TOKENS <= AFD_KV_CAPACITY_TOKENS ))
    ((
        CAPACITY_MAX_BATCH *
        (RESIDENT_PROMPT_TOKENS + SCHEDULER_RUNNING_RESERVATION_TOKENS) +
        FRESH_REQUEST_ESTIMATED_TOKENS > AFD_KV_CAPACITY_TOKENS
    ))
    NOMINAL_CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH
    NOMINAL_RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH
    if (( REQUIRE_CAPACITY_MAX )); then
        if (( BATCH > CAPACITY_MAX_BATCH )); then
            echo "capacity-max sweep cannot exceed nominal batch $CAPACITY_MAX_BATCH at ISL $CONTEXT (raw KV ceiling $RAW_KV_MAX_BATCH), got $BATCH" >&2
            exit 2
        fi
        if (( BATCH < CAPACITY_MAX_BATCH && ! ALLOW_OBSERVED_CAPACITY_PROBE )); then
            echo "a below-nominal observed-capacity probe requires FASTAFD_ALLOW_OBSERVED_CAPACITY_PROBE=1: nominal batch $CAPACITY_MAX_BATCH at ISL $CONTEXT, got $BATCH" >&2
            exit 2
        fi
    fi
fi
REFERENCE_TPS=0
if [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == qwen3:8192:96:8 ]]; then
    REFERENCE_TPS=2518
elif [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == qwen3:16384:48:12 ]]; then
    REFERENCE_TPS=1377
elif [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == minimax:8192:72:18 ]]; then
    REFERENCE_TPS=2198
elif [[ "$MODEL_KEY:$CONTEXT:$BATCH:$NODES" == minimax:16384:36:18 ]]; then
    REFERENCE_TPS=1006
fi
MODEL_SOURCE_PATH=${FASTAFD_MODEL_PATH:-$HF_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION}
MODEL_PATH=$MODEL_SOURCE_PATH
MODEL_PROFILE_MANIFEST=""
if [[ "$MODEL_PROFILE_ID" != native ]]; then
    MODEL_PATH=$ROOT/model-profiles/$MODEL_PROFILE_ID
    MODEL_PROFILE_MANIFEST=$MODEL_PATH/fastafd-model-profile.json
fi
MODEL_PROFILE_SCRIPT=${FASTAFD_MODEL_PROFILE_SCRIPT:-$(dirname "$(realpath "$0")")/prepare_model_profile.py}
PROMPT_BASE=$SOURCE_REPO/prompts/prompts_512x${PROMPT_SOURCE_CONTEXT}_seed20260527.txt
ATTENTION_WORKERS=$(( (NODES - 1) * 4 ))
ATTENTION_TRAYS=$(( NODES - 1 ))
FFN_TRAYS=1
NUM_PROMPTS=$(( ATTENTION_WORKERS * BATCH ))
MICROBATCH_UPPER=$(( (BATCH + 1) / 2 ))
MICROBATCH_LOWER=$(( BATCH / 2 ))
GRAPH_BATCH=$MICROBATCH_UPPER
PADDED_BATCH=$(( GRAPH_BATCH * 2 ))

if [[ "${FASTAFD_DRY_RUN:-0}" == 1 && -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'mode=afd model=%s afd_model_placement=%s afd_moe_backend=%s prompt_mode=%s context_min=%s context_max=%s irregular=%s input_batch_per_attention_gpu=%s capacity_max_batch=%s raw_kv_max_batch=%s require_capacity_max=%s microbatch_real_sizes=%s+%s nodes=%s attention_trays=%s ffn_trays=%s attention_workers=%s prompts=%s prompt_lengths=%s prompt_length_sum=%s required_kv_tokens_per_attention_gpu=%s known_kv_capacity_tokens_per_attention_gpu=%s max_seq_len=%s graph_batch_per_mb=%s graph_padded_input_batch=%s nsys_cuda_graph_trace=%s nsys_target_batch_per_attention_gpu=%s nsys_capture_decode_steps=%s prompt_source=%s model_profile=%s rope_factor=%s\n' \
        "$MODEL_KEY" "$AFD_MODEL_PLACEMENT" "$MINISGL_AFD_MOE_BACKEND" \
        "$PROMPT_MODE" "$CONTEXT_MIN" "$CONTEXT_MAX" "$IRREGULAR" \
        "$BATCH" "$CAPACITY_MAX_BATCH" "$RAW_KV_MAX_BATCH" "$REQUIRE_CAPACITY_MAX" \
        "$MICROBATCH_UPPER" "$MICROBATCH_LOWER" "$NODES" "$ATTENTION_TRAYS" "$FFN_TRAYS" "$ATTENTION_WORKERS" \
        "$NUM_PROMPTS" "$PROMPT_LENGTHS_CSV" "$PROMPT_LENGTH_SUM" \
        "$AFD_REQUIRED_KV_TOKENS" "$AFD_KV_CAPACITY_TOKENS" \
        "$MAX_SEQ_LEN" "$GRAPH_BATCH" "$PADDED_BATCH" "$NSYS_CUDA_GRAPH_TRACE" \
        "$BATCH" "$NSYS_CAPTURE_DECODE_STEPS" \
        "$PROMPT_BASE" "$MODEL_PROFILE_ID" "$ROPE_SCALING_FACTOR"
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
    validate_source_repo
    [[ -x "$EP_VENV_DIR/bin/python" && -d "$MODEL_SOURCE_PATH" && -f "$PROMPT_BASE" ]]
    if [[ "$MODEL_PROFILE_ID" != native ]]; then
        "$EP_VENV_DIR/bin/python" "$MODEL_PROFILE_SCRIPT" \
            --source "$MODEL_SOURCE_PATH" --output "$MODEL_PATH" \
            --profile-id "$MODEL_PROFILE_ID" --source-revision "$MODEL_REVISION" \
            --expected-source-config-sha256 "$MODEL_SOURCE_CONFIG_SHA256" \
            --max-position-embeddings "$MAX_SEQ_LEN" \
            --rope-factor "$ROPE_SCALING_FACTOR" \
            --original-max-position-embeddings "$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS"
    fi
    MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')
    [[ -f "$MODEL_PATH/model.safetensors.index.json" ]]
    [[ $(sha256sum "$PROMPT_BASE" | awk '{print $1}') == "$PROMPT_SHA256" ]]
    if (( IRREGULAR )); then
        JOB_NAME="fastafd:afd-irregular-${MODEL_KEY}-${SHAPE_ID}-b${BATCH}-n${NODES}"
    else
        JOB_NAME="fastafd:afd-${MODEL_KEY}-${CONTEXT}-b${BATCH}-n${NODES}"
    fi
    ! squeue -h -u "$USER" -o '%j' | grep -qx "$JOB_NAME" || {
        echo "the requested AFD job is already active" >&2; exit 1;
    }
    ACTIVE_JOB_REGEX=${FASTAFD_ACTIVE_JOB_REGEX:-'^fastafd:'}
    [[ -n "$ACTIVE_JOB_REGEX" ]]
    ACTIVE_FASTAFD=$(squeue -h -u "$USER" -o '%j' | \
        awk -v job_regex="$ACTIVE_JOB_REGEX" '$0 ~ job_regex {n++} END {print n+0}')
    (( ACTIVE_FASTAFD < 4 )) || {
        echo "four FastAFD experiment jobs matching $ACTIVE_JOB_REGEX are already active" >&2; exit 1;
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    if (( IRREGULAR )); then
        RUN_DIR=$RESULTS_ROOT/afd_irregular_${MODEL_KEY}_${CONTEXT_MIN}-${CONTEXT_MAX}_b${BATCH}_n${NODES}_$STAMP
    else
        RUN_DIR=$RESULTS_ROOT/afd_${MODEL_KEY}_${CONTEXT}_a${ATTENTION_TRAYS}_f${FFN_TRAYS}_b${BATCH}_n${NODES}_${STAMP}_manual_na
    fi
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable --nodes="$NODES" --segment="$NODES" --qos="$SLURM_QOS_REQUEST" --time="$JOB_TIME_LIMIT")
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,FASTAFD_SLURM_QOS=$SLURM_QOS_REQUEST,AFD_MODEL_PLACEMENT=$AFD_MODEL_PLACEMENT,MINISGL_AFD_MOE_BACKEND=$MINISGL_AFD_MOE_BACKEND,MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE,MODEL_KEY=$MODEL_KEY,PROMPT_MODE=$PROMPT_MODE,CONTEXT_SPEC=$CONTEXT_SPEC,CONTEXT=$CONTEXT,CONTEXT_MIN=$CONTEXT_MIN,CONTEXT_MAX=$CONTEXT_MAX,IRREGULAR=$IRREGULAR,SHAPE_ID=$SHAPE_ID,PROMPT_LENGTH_SUM=$PROMPT_LENGTH_SUM,AFD_REQUIRED_KV_TOKENS=$AFD_REQUIRED_KV_TOKENS,AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS,NOMINAL_AFD_KV_CAPACITY_TOKENS=$NOMINAL_AFD_KV_CAPACITY_TOKENS,CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH,RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH,SCHEDULER_ADMISSION_REQUIRED_TOKENS=$SCHEDULER_ADMISSION_REQUIRED_TOKENS,SCHEDULER_ADMISSION_UNUSED_TOKENS=$SCHEDULER_ADMISSION_UNUSED_TOKENS,SCHEDULER_RUNNING_RESERVATION_TOKENS=$SCHEDULER_RUNNING_RESERVATION_TOKENS,REQUIRE_CAPACITY_MAX=$REQUIRE_CAPACITY_MAX,ALLOW_OBSERVED_CAPACITY_PROBE=$ALLOW_OBSERVED_CAPACITY_PROBE,MAX_SEQ_LEN=$MAX_SEQ_LEN,NSYS_CUDA_GRAPH_TRACE=$NSYS_CUDA_GRAPH_TRACE,NSYS_CAPTURE_DECODE_STEPS=$NSYS_CAPTURE_DECODE_STEPS,MODEL_PATH=$MODEL_PATH,MODEL_SOURCE_PATH=$MODEL_SOURCE_PATH,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_SOURCE_CONFIG_SHA256=$MODEL_SOURCE_CONFIG_SHA256,MODEL_PROFILE_ID=$MODEL_PROFILE_ID,MODEL_PROFILE_MANIFEST=$MODEL_PROFILE_MANIFEST,ROPE_SCALING_FACTOR=$ROPE_SCALING_FACTOR,ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS=$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,MODEL_REVISION=$MODEL_REVISION,PRESET=$PRESET,NODES=$NODES,BATCH=$BATCH,GRAPH_BATCH=$GRAPH_BATCH,REFERENCE_TPS=$REFERENCE_TPS,ATTENTION_TRAYS=$ATTENTION_TRAYS,FFN_TRAYS=$FFN_TRAYS,ATTENTION_WORKERS=$ATTENTION_WORKERS,NUM_PROMPTS=$NUM_PROMPTS,PROMPT_BASE=$PROMPT_BASE,PROMPT_SHA256=$PROMPT_SHA256,PROMPT_SOURCE_CONTEXT=$PROMPT_SOURCE_CONTEXT,EXPECTED_HEAD=$EXPECTED_HEAD,ROOT=$ROOT,RESULTS_ROOT=$RESULTS_ROOT,SOURCE_REPO=$SOURCE_REPO,EXPECTED_SOURCE_MANIFEST=$EXPECTED_SOURCE_MANIFEST,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s model=%s context_min=%s context_max=%s batch=%s nodes=%s run_dir=%s\n' \
        "$JOB" "$MODEL_KEY" "$CONTEXT_MIN" "$CONTEXT_MAX" "$BATCH" "$NODES" "$RUN_DIR"
    exit 0
fi

: "${RUN_DIR:?}" "${JOB_SCRIPT:?}"
if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == "$SLURM_QOS_REQUEST" ]]
    [[ "$SLURM_JOB_NUM_NODES" == "$NODES" ]]
    HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
    exec srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gres=gpu:4 --kill-on-bad-exit=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_IN_CONTAINER=1 HEAD_HOST="$HEAD_HOST" MODEL_KEY="$MODEL_KEY" \
        PROMPT_MODE="$PROMPT_MODE" CONTEXT_SPEC="$CONTEXT_SPEC" \
        CONTEXT="$CONTEXT" CONTEXT_MIN="$CONTEXT_MIN" CONTEXT_MAX="$CONTEXT_MAX" \
        IRREGULAR="$IRREGULAR" SHAPE_ID="$SHAPE_ID" \
        PROMPT_LENGTHS_CSV="$PROMPT_LENGTHS_CSV" PROMPT_LENGTH_SUM="$PROMPT_LENGTH_SUM" \
        AFD_REQUIRED_KV_TOKENS="$AFD_REQUIRED_KV_TOKENS" \
        AFD_KV_CAPACITY_TOKENS="$AFD_KV_CAPACITY_TOKENS" \
        NOMINAL_AFD_KV_CAPACITY_TOKENS="$NOMINAL_AFD_KV_CAPACITY_TOKENS" \
        CAPACITY_MAX_BATCH="$CAPACITY_MAX_BATCH" RAW_KV_MAX_BATCH="$RAW_KV_MAX_BATCH" \
        SCHEDULER_ADMISSION_REQUIRED_TOKENS="$SCHEDULER_ADMISSION_REQUIRED_TOKENS" \
        SCHEDULER_ADMISSION_UNUSED_TOKENS="$SCHEDULER_ADMISSION_UNUSED_TOKENS" \
        SCHEDULER_RUNNING_RESERVATION_TOKENS="$SCHEDULER_RUNNING_RESERVATION_TOKENS" \
        REQUIRE_CAPACITY_MAX="$REQUIRE_CAPACITY_MAX" \
        ALLOW_OBSERVED_CAPACITY_PROBE="$ALLOW_OBSERVED_CAPACITY_PROBE" \
        MAX_SEQ_LEN="$MAX_SEQ_LEN" NSYS_CUDA_GRAPH_TRACE="$NSYS_CUDA_GRAPH_TRACE" \
        NSYS_CAPTURE_DECODE_STEPS="$NSYS_CAPTURE_DECODE_STEPS" \
        AFD_MODEL_PLACEMENT="$AFD_MODEL_PLACEMENT" \
        MINISGL_AFD_MOE_BACKEND="$MINISGL_AFD_MOE_BACKEND" \
        MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE="$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE" \
        MODEL_PATH="$MODEL_PATH" \
        MODEL_SOURCE_PATH="$MODEL_SOURCE_PATH" MODEL_CONFIG_SHA256="$MODEL_CONFIG_SHA256" \
        MODEL_SOURCE_CONFIG_SHA256="$MODEL_SOURCE_CONFIG_SHA256" \
        MODEL_PROFILE_ID="$MODEL_PROFILE_ID" MODEL_PROFILE_MANIFEST="$MODEL_PROFILE_MANIFEST" \
        ROPE_SCALING_FACTOR="$ROPE_SCALING_FACTOR" \
        ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS="$ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS" \
        MODEL_REVISION="$MODEL_REVISION" PRESET="$PRESET" NODES="$NODES" \
        BATCH="$BATCH" GRAPH_BATCH="$GRAPH_BATCH" REFERENCE_TPS="$REFERENCE_TPS" \
        ATTENTION_TRAYS="$ATTENTION_TRAYS" FFN_TRAYS="$FFN_TRAYS" \
        ATTENTION_WORKERS="$ATTENTION_WORKERS" NUM_PROMPTS="$NUM_PROMPTS" \
        PROMPT_BASE="$PROMPT_BASE" PROMPT_SHA256="$PROMPT_SHA256" \
        PROMPT_SOURCE_CONTEXT="$PROMPT_SOURCE_CONTEXT" \
        EXPECTED_HEAD="$EXPECTED_HEAD" ROOT="$ROOT" RESULTS_ROOT="$RESULTS_ROOT" \
        SOURCE_REPO="$SOURCE_REPO" \
        EXPECTED_SOURCE_MANIFEST="$EXPECTED_SOURCE_MANIFEST" \
        IMAGE="$IMAGE" EP_VENV_DIR="$EP_VENV_DIR" \
        RUN_DIR="$RUN_DIR" JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

RANK=${SLURM_PROCID:?}
[[ "$RANK" =~ ^[0-9]+$ && "$RANK" -lt "$NODES" ]]
PYTHON=$EP_VENV_DIR/bin/python
[[ -x "$PYTHON" ]]
export PATH=$EP_VENV_DIR/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$SOURCE_REPO/python${PYTHONPATH:+:$PYTHONPATH}
export VIRTUAL_ENV=$EP_VENV_DIR CONDA_PREFIX=$EP_VENV_DIR
export CONDA_DEFAULT_ENV=fastafd-venv ENV_NAME=fastafd-venv
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export MAX_JOBS=32 NVCC_THREADS=4 TORCH_CUDA_ARCH_LIST=10.0

CONTROL=$RUN_DIR/control
READY=$CONTROL/ready
DONE=$CONTROL/snapshot-done
STOP=$CONTROL/STOP
SNAPSHOT=$CONTROL/SNAPSHOT
mkdir -p "$READY" "$DONE" "$RUN_DIR/gpu-snapshots" "$RUN_DIR/tmp/rank-$RANK"
export TMPDIR=$RUN_DIR/tmp/rank-$RANK
export RAY_TMPDIR=/dev/shm/fastafd-ray-$SLURM_JOB_ID-$RANK
ray_cli() { "$PYTHON" -m ray.scripts.scripts "$@"; }
cleanup() {
    [[ "$RANK" != 0 ]] || touch "$STOP"
    ray_cli stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT TERM INT

[[ "$(uname -m)" == aarch64 ]]
validate_source_repo
[[ -f "$MODEL_PATH/model.safetensors.index.json" && -f "$PROMPT_BASE" ]]
[[ $(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}') == "$MODEL_CONFIG_SHA256" ]]
[[ "$MODEL_PROFILE_ID" == native || -f "$MODEL_PROFILE_MANIFEST" ]]
[[ $(sha256sum "$PROMPT_BASE" | awk '{print $1}') == "$PROMPT_SHA256" ]]
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RUN_DIR/gpu-snapshots/initial-rank-$RANK.csv"
[[ $(wc -l < "$RUN_DIR/gpu-snapshots/initial-rank-$RANK.csv") -eq 1 ]]
ray_cli stop --force >/dev/null 2>&1 || true

HEAD_ADDRESS=$HEAD_HOST:6379
if [[ "$RANK" == 0 ]]; then
    printf '%s\n' "$HEAD_ADDRESS" > "$CONTROL/head-address"
    ray_cli start --head --port=6379 --num-cpus=140 --num-gpus=4 \
        --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMPDIR" \
        > "$RUN_DIR/ray-start-0.log" 2>&1
else
    deadline=$((SECONDS + 120))
    until [[ -s "$CONTROL/head-address" ]]; do
        [[ ! -e "$STOP" && $SECONDS -lt $deadline ]]; sleep 1
    done
    HEAD_ADDRESS=$(<"$CONTROL/head-address")
    ray_cli start --address="$HEAD_ADDRESS" --num-cpus=140 --num-gpus=4 \
        --disable-usage-stats > "$RUN_DIR/ray-start-$RANK.log" 2>&1
fi
touch "$READY/$RANK"

if [[ "$RANK" != 0 ]]; then
    until [[ -e "$SNAPSHOT" || -e "$STOP" ]]; do sleep 2; done
    if [[ -e "$SNAPSHOT" ]]; then
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv > "$RUN_DIR/gpu-snapshots/final-rank-$RANK.csv"
        touch "$DONE/$RANK"
    fi
    until [[ -e "$STOP" ]]; do sleep 2; done
    exit 0
fi

deadline=$((SECONDS + 180))
until [[ $(find "$READY" -maxdepth 1 -type f | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $deadline ]]; sleep 2
done
RAY_ADDRESS=$HEAD_ADDRESS NODES=$NODES "$PYTHON" - <<'PY'
import os, time, ray
ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0)]
    if len(nodes) == int(os.environ["NODES"]) and sum(int(n["Resources"]["GPU"]) for n in nodes) == 4 * len(nodes):
        break
    time.sleep(2)
else:
    raise RuntimeError([(n.get("NodeManagerAddress"), n.get("Resources")) for n in ray.nodes()])
PY

# AFD uses a contiguous seven-port control block. Validate the whole block
# after Ray starts so an adjacent service cannot strand the detokenizer.
MINISGL_PORT=$("$PYTHON" - "$SLURM_JOB_ID" "${FASTAFD_MINISGL_PORT:-}" <<'PY'
import socket
import sys

job_id = int(sys.argv[1].split("_")[0])
override = sys.argv[2]
candidates = [int(override)] if override else [
    20000 + ((job_id + offset) % 1000) * 8 for offset in range(1000)
]
last_error = None
for base in candidates:
    if not 1024 <= base <= 65529:
        last_error = ValueError(f"invalid seven-port block {base}-{base + 6}")
        continue
    sockets = []
    try:
        for port in range(base, base + 7):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
    except OSError as exc:
        last_error = exc
    else:
        print(base)
        break
    finally:
        for sock in sockets:
            sock.close()
else:
    raise SystemExit(f"no free seven-port AFD control block: {last_error}")
PY
)
[[ "$MINISGL_PORT" =~ ^[0-9]+$ && "$MINISGL_PORT" -ge 1024 && "$MINISGL_PORT" -le 65529 ]]
export MINISGL_PORT
printf 'afd_control_port_block=%s-%s\n' "$MINISGL_PORT" "$((MINISGL_PORT + 6))"

export RAY_ADDRESS=$HEAD_ADDRESS MINISGL_RAY_ADDRESS=$HEAD_ADDRESS MODEL_PATH
export PROMPT_BASE_FILE=$PROMPT_BASE
export PROMPT_FILE=$RUN_DIR/prompts_${NUM_PROMPTS}x${SHAPE_ID}.txt
PROMPT_MANIFEST=$RUN_DIR/prompt-transform.json
"$PYTHON" - "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$PROMPT_MANIFEST" \
    "$NUM_PROMPTS" "$CONTEXT_MIN" "$CONTEXT_MAX" "$MODEL_KEY" "$MODEL_PATH" \
    "$PROMPT_SOURCE_CONTEXT" "$ATTENTION_WORKERS" "$PROMPT_LENGTHS_CSV" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

from minisgl.hf_support import load_tokenizer

src, dst, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
count, context_min, context_max = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
model_key, model_path, source_context = sys.argv[7], sys.argv[8], int(sys.argv[9])
attention_workers = int(sys.argv[10])
prompt_lengths = [int(value) for value in sys.argv[11].split(",")]
if count != attention_workers * len(prompt_lengths):
    raise RuntimeError((count, attention_workers, len(prompt_lengths)))
if prompt_lengths[0] != context_min or prompt_lengths[-1] != context_max:
    raise RuntimeError(prompt_lengths)
if any(left > right for left, right in zip(prompt_lengths, prompt_lengths[1:])):
    raise RuntimeError("prompt lengths are not monotonic")
if any(left + right != context_min + context_max for left, right in zip(prompt_lengths, reversed(prompt_lengths))):
    raise RuntimeError("prompt lengths are not symmetric")
model_config_bytes = (Path(model_path) / "config.json").read_bytes()
model_config = json.loads(model_config_bytes)
lines = [x.strip() for x in src.read_text().splitlines() if x.strip()]
if len(lines) != 512:
    raise RuntimeError(len(lines))

def qwen_chat_prompt(text, target_tokens, tokenizer):
    source_ids = tokenizer.encode(text, add_special_tokens=False)
    if not source_ids:
        raise RuntimeError("empty tokenized source prompt")
    template_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    initial_content_count = target_tokens - len(template_tokens)
    content_count = initial_content_count
    visited_counts = set()
    for _ in range(64):
        if content_count in visited_counts:
            break
        visited_counts.add(content_count)
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
        delta = target_tokens - len(prompt_ids)
        if delta == 0:
            if "\n" in content or "\r" in content:
                raise RuntimeError("transformed prompt is not one physical line")
            return content, prompt_ids, len(source_ids), ""
        content_count += delta
        if content_count <= 0:
            break

    # Byte-pair merges can make an exact length unreachable by changing only
    # the source-token prefix (for example, a 1088 -> 1090 jump). Resolve such
    # gaps deterministically with a small fixed suffix search. Exact length is
    # still mandatory, and every repair is recorded in the prompt manifest.
    repair_suffixes = (" 0", " a", " x", " !", " ?", " .", " #", " |", "0", "!", "?", "_", "~")
    for distance in range(65):
        candidate_counts = [initial_content_count - distance]
        if distance:
            candidate_counts.append(initial_content_count + distance)
        for candidate_count in candidate_counts:
            if candidate_count <= 0:
                continue
            repeats = (candidate_count + len(source_ids) - 1) // len(source_ids)
            selected = (source_ids * repeats)[:candidate_count]
            base_content = tokenizer.decode(
                selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for suffix in repair_suffixes:
                content = base_content + suffix
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if len(prompt_ids) == target_tokens:
                    if "\n" in content or "\r" in content:
                        raise RuntimeError("transformed prompt is not one physical line")
                    return content, prompt_ids, len(source_ids), suffix
    raise RuntimeError(
        f"cannot construct exact {target_tokens}-token chat prompt after deterministic merge-gap search"
    )

dst.parent.mkdir(parents=True, exist_ok=True)
token_digest = hashlib.sha256()
source_token_lengths = set()
merge_gap_repairs = []
if model_key == "qwen3":
    tokenizer = load_tokenizer(model_path)
    cache = {}
    with dst.open("w", encoding="utf-8") as out:
        for index in range(count):
            source_index = index % len(lines)
            target_tokens = prompt_lengths[index // attention_workers]
            cache_key = (source_index, target_tokens)
            if cache_key not in cache:
                cache[cache_key] = qwen_chat_prompt(
                    lines[source_index], target_tokens, tokenizer
                )
            content, prompt_ids, source_length, repair_suffix = cache[cache_key]
            if repair_suffix and not any(
                item["source_index"] == source_index and item["target_tokens"] == target_tokens
                for item in merge_gap_repairs
            ):
                merge_gap_repairs.append(
                    {
                        "source_index": source_index,
                        "target_tokens": target_tokens,
                        "suffix": repair_suffix,
                    }
                )
            source_token_lengths.add(source_length)
            for token in prompt_ids:
                token_digest.update(int(token).to_bytes(4, "little"))
            out.write(content + "\n")
    transform = "cut_or_repeat_pinned_8k_content_tokens_then_qwen_chat_template_with_exact_merge_gap_correction"
    observed = {"min": min(prompt_lengths), "max": max(prompt_lengths)}
    prompt_token_sha256 = token_digest.hexdigest()
else:
    dst.write_text("\n".join(lines[i % len(lines)] for i in range(count)) + "\n")
    transform = "pinned_text_unchanged"
    observed = None
    prompt_token_sha256 = None

record = {
    "model_key": model_key,
    "source_file": str(src),
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "source_prompt_count": len(lines),
    "source_context_tokens": source_context,
    "output_file": str(dst),
    "output_prompt_count": count,
    "target_context_tokens": context_max if context_min == context_max else None,
    "target_context_range_tokens": {"min": context_min, "max": context_max},
    "distribution": "inclusive_uniform_symmetric_integer_linspace",
    "batch_per_attention_gpu": len(prompt_lengths),
    "per_attention_gpu_prompt_lengths": prompt_lengths,
    "per_attention_gpu_prompt_length_sum": sum(prompt_lengths),
    "global_prompt_order": "length-major, then attention-DP rank; coordinator round-robin gives every rank the listed distribution",
    "model_path": model_path,
    "model_config_sha256": hashlib.sha256(model_config_bytes).hexdigest(),
    "model_max_position_embeddings": model_config.get("max_position_embeddings"),
    "model_rope_scaling": model_config.get("rope_scaling"),
    "transform": transform,
    "observed_chat_prompt_tokens": observed,
    "aggregate_prompt_token_ids_sha256": prompt_token_sha256,
    "source_content_token_lengths": sorted(source_token_lengths),
    "exact_length_merge_gap_repairs": merge_gap_repairs,
    "prefix_caching": "not present: AFD uses cache-type=naive",
}
manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
export AFD_TOTAL_NODES=$NODES RUN_VLLM_ALIGNMENT=0 NSYS_CUDA_GRAPH_TRACE
export PROMPT_LEN=$CONTEXT_MAX PER_ATTN_GPU_BSZ=$BATCH MAX_TOKENS=$OUTPUT_TOKENS
export MINISGL_MAX_SEQ_LEN=$MAX_SEQ_LEN VLLM_MAX_MODEL_LEN=$MAX_SEQ_LEN
export AFD_NUM_MB=2 AFD_DECODE_GRAPH_BS=$GRAPH_BATCH
printf 'afd_model_placement=%s afd_moe_backend=%s\n' \
    "$AFD_MODEL_PLACEMENT" "$MINISGL_AFD_MOE_BACKEND"
export AFD_MEMORY_RATIO=0.82 AFD_MAX_BATCHED_TOKENS=512
export MINISGL_AFD_WORKER_HOT_LOOP_SHUTDOWN_TIMEOUT_S=${FASTAFD_AFD_WORKER_HOT_LOOP_SHUTDOWN_TIMEOUT_S:-300}
export MINISGL_AFD_WORKER_SHUTDOWN_TIMEOUT_S=${FASTAFD_AFD_WORKER_SHUTDOWN_TIMEOUT_S:-120}
export NSYS_TARGET_BATCH_PER_ATTN_DP=$BATCH
export NSYS_CAPTURE_DECODE_STEPS
export MINISGL_RAY_NSYS_TARGET_BATCH_PER_DP=$BATCH
export MINISGL_RAY_NSYS_CAPTURE_DECODE_STEPS=$NSYS_CAPTURE_DECODE_STEPS
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/afd/flashinfer
export TVM_FFI_CACHE_DIR=$ROOT/cache/afd/tvm_ffi
export EP_JIT_CACHE_DIR=$ROOT/cache/afd/deepep_jit
export DG_JIT_CACHE_DIR=$ROOT/cache/afd/deepgemm_jit
export N2M_M2N_GIN_BUILD_DIR=$ROOT/cache/afd/gin_comm
export MINISGL_DEEPEP_BUILD_DIR=$ROOT/cache/afd/deepep_moe
export MINISGL_DEEPGEMM_BUILD_DIR=${MINISGL_DEEPGEMM_BUILD_DIR:-$ROOT/cache/afd/deepgemm}
EXPERIMENT=$RUN_DIR/experiment
export RUN_DIR=$EXPERIMENT
mkdir -p "$RUN_DIR"
INPUTS=("$SOURCE_REPO/$PRESET" "$SOURCE_REPO/python/minisgl/afd_coordinator.py" \
    "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$PROMPT_MANIFEST" \
    "$MODEL_PATH/config.json" "$JOB_SCRIPT")
[[ "$MODEL_PROFILE_ID" == native ]] || INPUTS+=("$MODEL_PROFILE_MANIFEST")
[[ -z "$EXPECTED_SOURCE_MANIFEST" ]] || INPUTS+=("$EXPECTED_SOURCE_MANIFEST")
sha256sum "${INPUTS[@]}" > "$RUN_DIR/inputs.sha256"
cd "$SOURCE_REPO"
bash "$PRESET" > "$RUN_DIR/driver.stdout" 2> "$RUN_DIR/driver.stderr" &
DRIVER_PID=$!

# For a capacity-max point, hold the sampler until every attention worker has
# reported its actual post-init page count. The requested batch must equal the
# scheduler-admissible maximum under the allocation's minimum capacity.
if (( ! IRREGULAR && REQUIRE_CAPACITY_MAX )); then
    deadline=$((SECONDS + 900))
    while true; do
        observed_page_count=$(grep -c "AfdAttentionState allocating [0-9][0-9]* pages" "$RUN_DIR/afd.log" 2>/dev/null || true)
        if (( observed_page_count == ATTENTION_WORKERS )); then
            break
        fi
        if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
            set +e
            wait "$DRIVER_PID"
            driver_status=$?
            set -e
            (( driver_status != 0 )) || {
                echo "AFD driver exited without complete attention page-count evidence: $observed_page_count/$ATTENTION_WORKERS" >&2
                exit 3
            }
            exit "$driver_status"
        fi
        if (( SECONDS >= deadline )); then
            kill -TERM "$DRIVER_PID" 2>/dev/null || true
            set +e
            wait "$DRIVER_PID"
            set -e
            echo "timed out waiting for attention page-count evidence: $observed_page_count/$ATTENTION_WORKERS" >&2
            exit 3
        fi
        sleep 2
    done
    capacity_line=$("$PYTHON" - "$RUN_DIR/afd.log" "$RUN_DIR/capacity-preflight.json" \
        "$ATTENTION_WORKERS" "$CONTEXT" "$BATCH" "$OUTPUT_TOKENS" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
workers, context, requested, output_tokens = map(int, sys.argv[3:])
page_size = 64
pages = [
    int(value)
    for value in re.findall(
        r"AfdAttentionState allocating (\d+) pages",
        log_path.read_text(errors="replace"),
    )
]
if len(pages) != workers:
    raise RuntimeError(("attention page-count evidence", len(pages), workers))
capacity = min(pages) * page_size
per_sequence = ((context + output_tokens + page_size - 1) // page_size) * page_size
resident_prompt = ((context + page_size - 1) // page_size) * page_size
running_reservation = (output_tokens - 1) + (page_size - 1)
fresh_request = context + output_tokens
maximum = ((capacity - fresh_request) // (resident_prompt + running_reservation)) + 1
raw_maximum = capacity // per_sequence
admission_required = (
    (requested - 1) * (resident_prompt + running_reservation) + fresh_request
)
record = {
    "attention_workers": workers,
    "requested_batch_per_attention_gpu": requested,
    "observed_kv_pages_per_attention_gpu": pages,
    "observed_min_kv_capacity_tokens_per_attention_gpu": capacity,
    "observed_scheduler_admissible_max_batch": maximum,
    "observed_raw_kv_max_batch": raw_maximum,
    "requested_scheduler_admission_tokens": admission_required,
    "requested_batch_is_exact": requested == maximum,
}
output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(f"{capacity}|{maximum}|{raw_maximum}|{admission_required}")
PY
    )
    IFS='|' read -r observed_capacity observed_max_batch observed_raw_max_batch observed_admission_required <<< "$capacity_line"
    printf 'observed_capacity_gate pages=%s workers=%s min_capacity_tokens=%s requested_batch=%s exact_max_batch=%s raw_max_batch=%s admission_required_tokens=%s\n' \
        "$observed_page_count" "$ATTENTION_WORKERS" "$observed_capacity" "$BATCH" \
        "$observed_max_batch" "$observed_raw_max_batch" "$observed_admission_required"
    if (( BATCH != observed_max_batch )); then
        kill -TERM "$DRIVER_PID" 2>/dev/null || true
        set +e
        wait "$DRIVER_PID"
        set -e
        echo "observed-capacity gate requires batch $observed_max_batch at ISL $CONTEXT under the allocation minimum $observed_capacity tokens, got $BATCH" >&2
        exit 3
    fi
    AFD_KV_CAPACITY_TOKENS=$observed_capacity
    CAPACITY_MAX_BATCH=$observed_max_batch
    RAW_KV_MAX_BATCH=$observed_raw_max_batch
    SCHEDULER_ADMISSION_REQUIRED_TOKENS=$observed_admission_required
    SCHEDULER_ADMISSION_UNUSED_TOKENS=$(( observed_capacity - observed_admission_required ))
fi

set +e
wait "$DRIVER_PID"
driver_status=$?
set -e
(( driver_status == 0 )) || exit "$driver_status"
[[ -s "$RUN_DIR/sample.json" ]] || {
    echo "AFD driver completed without a nonempty sample.json" >&2
    exit 3
}
[[ -s "$RUN_DIR/ray_logs/coordinator_nvtx_cpu.json" ]] || {
    echo "AFD driver completed without a nonempty coordinator CPU trace; graceful shutdown may have timed out" >&2
    exit 3
}

RUN_DIR=$RUN_DIR NUM_PROMPTS=$NUM_PROMPTS BATCH=$BATCH \
ATTENTION_WORKERS=$ATTENTION_WORKERS NODES=$NODES REFERENCE_TPS=$REFERENCE_TPS \
CONTEXT=$CONTEXT CONTEXT_MIN=$CONTEXT_MIN CONTEXT_MAX=$CONTEXT_MAX \
IRREGULAR=$IRREGULAR PROMPT_LENGTHS_CSV=$PROMPT_LENGTHS_CSV \
PROMPT_LENGTH_SUM=$PROMPT_LENGTH_SUM AFD_REQUIRED_KV_TOKENS=$AFD_REQUIRED_KV_TOKENS \
OUTPUT_TOKENS=$OUTPUT_TOKENS \
AFD_KV_CAPACITY_TOKENS=$AFD_KV_CAPACITY_TOKENS \
NOMINAL_AFD_KV_CAPACITY_TOKENS=$NOMINAL_AFD_KV_CAPACITY_TOKENS \
CAPACITY_MAX_BATCH=$CAPACITY_MAX_BATCH RAW_KV_MAX_BATCH=$RAW_KV_MAX_BATCH \
NOMINAL_CAPACITY_MAX_BATCH=${NOMINAL_CAPACITY_MAX_BATCH:-$CAPACITY_MAX_BATCH} \
NOMINAL_RAW_KV_MAX_BATCH=${NOMINAL_RAW_KV_MAX_BATCH:-$RAW_KV_MAX_BATCH} \
SCHEDULER_ADMISSION_REQUIRED_TOKENS=$SCHEDULER_ADMISSION_REQUIRED_TOKENS \
SCHEDULER_ADMISSION_UNUSED_TOKENS=$SCHEDULER_ADMISSION_UNUSED_TOKENS \
SCHEDULER_RUNNING_RESERVATION_TOKENS=$SCHEDULER_RUNNING_RESERVATION_TOKENS \
REQUIRE_CAPACITY_MAX=$REQUIRE_CAPACITY_MAX ATTENTION_TRAYS=$ATTENTION_TRAYS \
FFN_TRAYS=$FFN_TRAYS \
PADDED_BATCH=$PADDED_BATCH PROMPT_MANIFEST=$PROMPT_MANIFEST \
AFD_MODEL_PLACEMENT=$AFD_MODEL_PLACEMENT \
MINISGL_AFD_MOE_BACKEND=$MINISGL_AFD_MOE_BACKEND \
MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=$MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE \
"$PYTHON" - <<'PY'
import json, os, re, statistics
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
count, batch = int(os.environ["NUM_PROMPTS"]), int(os.environ["BATCH"])
padded_batch = int(os.environ["PADDED_BATCH"])
workers, nodes = int(os.environ["ATTENTION_WORKERS"]), int(os.environ["NODES"])
reference = float(os.environ["REFERENCE_TPS"])
context = int(os.environ["CONTEXT"])
output_tokens = int(os.environ["OUTPUT_TOKENS"])
context_min, context_max = int(os.environ["CONTEXT_MIN"]), int(os.environ["CONTEXT_MAX"])
irregular = bool(int(os.environ["IRREGULAR"]))
prompt_lengths = [int(value) for value in os.environ["PROMPT_LENGTHS_CSV"].split(",")]
prompt_length_sum = int(os.environ["PROMPT_LENGTH_SUM"])
required_kv = int(os.environ["AFD_REQUIRED_KV_TOKENS"])
capacity_kv = int(os.environ["AFD_KV_CAPACITY_TOKENS"])
nominal_capacity_kv = int(os.environ["NOMINAL_AFD_KV_CAPACITY_TOKENS"])
capacity_max_batch = int(os.environ["CAPACITY_MAX_BATCH"])
raw_kv_max_batch = int(os.environ["RAW_KV_MAX_BATCH"])
nominal_capacity_max_batch = int(os.environ["NOMINAL_CAPACITY_MAX_BATCH"])
nominal_raw_kv_max_batch = int(os.environ["NOMINAL_RAW_KV_MAX_BATCH"])
admission_required = int(os.environ["SCHEDULER_ADMISSION_REQUIRED_TOKENS"])
admission_unused = int(os.environ["SCHEDULER_ADMISSION_UNUSED_TOKENS"])
running_reservation = int(os.environ["SCHEDULER_RUNNING_RESERVATION_TOKENS"])
require_capacity_max = bool(int(os.environ["REQUIRE_CAPACITY_MAX"]))
attention_trays = int(os.environ["ATTENTION_TRAYS"])
ffn_trays = int(os.environ["FFN_TRAYS"])
prompt_contract = json.loads(Path(os.environ["PROMPT_MANIFEST"]).read_text())
observed_pages = [
    int(value)
    for value in re.findall(
        r"AfdAttentionState allocating (\d+) pages",
        (run / "afd.log").read_text(errors="replace"),
    )
]
if len(observed_pages) != workers:
    raise RuntimeError(("attention page-count evidence", len(observed_pages), workers))
observed_min_capacity = min(observed_pages) * 64
resident_prompt = ((context + 63) // 64) * 64
fresh_request = context + output_tokens
observed_max_batch = (
    (observed_min_capacity - fresh_request)
    // (resident_prompt + output_tokens - 1 + 63)
) + 1
if prompt_contract["per_attention_gpu_prompt_lengths"] != prompt_lengths:
    raise RuntimeError("prompt manifest distribution mismatch")
if prompt_contract["per_attention_gpu_prompt_length_sum"] != prompt_length_sum:
    raise RuntimeError("prompt manifest length sum mismatch")
if required_kv > capacity_kv:
    raise RuntimeError((required_kv, capacity_kv))
if require_capacity_max and capacity_max_batch != batch:
    raise RuntimeError(("not capacity-max batch", batch, capacity_max_batch))
if require_capacity_max and observed_max_batch != batch:
    raise RuntimeError(("not observed scheduler-admissible max batch", batch, observed_max_batch))
samples = json.loads((run / "sample.json").read_text()).get("samples", [])
lengths = [len(x.get("generated_token_ids", [])) for x in samples]
if len(samples) != count or set(lengths) != {output_tokens}:
    raise RuntimeError((len(samples), sorted(set(lengths))))
attention_logs = sorted((run / "ray_logs").glob("attention_dp*_rank*.log"))
model_logs = sorted((run / "ray_logs").glob("mlp_dp*_rank*.log"))
if len(attention_logs) != workers or len(attention_logs) + len(model_logs) != nodes * 4:
    raise RuntimeError((len(attention_logs), len(model_logs), workers, nodes * 4))
for path in attention_logs + model_logs:
    worker_log = path.read_text(errors="replace")
    if worker_log.count("nsys profiler:start sync=1") != 1:
        raise RuntimeError(("invalid profiler start evidence", path))
    if worker_log.count("nsys profiler:stop sync=1") != 1:
        raise RuntimeError(("invalid profiler stop evidence", path))
coordinator_log = (run / "ray_logs/afd_coordinator.log").read_text(errors="replace")
capture_matches = re.findall(
    r"nsys profiler:target_decode_window "
    r"target_batch_per_dp=(\d+) warmup_step_id=(\d+) "
    r"step_ids=([0-9,]+) count=(\d+) trace_count=(\d+)",
    coordinator_log,
)
if len(capture_matches) != 1:
    raise RuntimeError(("expected one exact target-decode capture", capture_matches))
captured_target, captured_warmup, captured_csv, captured_count, trace_count = (
    capture_matches[0]
)
steps = [int(value) for value in captured_csv.split(",")]
if (
    int(captured_target) != batch
    or int(captured_count) != 15
    or len(steps) != 15
    or int(trace_count) != 16
    or int(captured_warmup) != steps[0] - 1
):
    raise RuntimeError(("invalid target-decode capture", capture_matches[0], batch))
if any(right != left + 1 for left, right in zip(steps, steps[1:])):
    raise RuntimeError(("non-consecutive target-decode capture", steps))
capture = json.loads((run / "capture-complete.json").read_text())
expected_trace_steps = [int(captured_warmup), *steps]
if (
    int(capture.get("target_batch_per_attention_dp", -1)) != batch
    or int(capture.get("capture_decode_steps", -1)) != 15
    or capture.get("decode_step_ids") != steps
    or int(capture.get("warmup_decode_step_id", -1)) != int(captured_warmup)
    or int(capture.get("trace_decode_launches", -1)) != 16
    or capture.get("trace_decode_step_ids") != expected_trace_steps
    or int(capture.get("gpu_worker_profiler_stop_logs", -1)) != nodes * 4
):
    raise RuntimeError(("invalid capture-complete evidence", capture))
full_windows = [steps]
partial_windows = []
trace = json.loads((run / "ray_logs/coordinator_nvtx_cpu.json").read_text())
complete = {
    e["step_id"]: e["end_perf_ns"] for e in trace["events"]
    if str(e.get("name", "")).startswith("AFD_Coordinator_CompleteCollect")
    and isinstance(e.get("step_id"), int)
}
ends = [complete[s] for s in steps]
intervals = [(b - a) / 1e6 for a, b in zip(ends, ends[1:])]
median_ms, mean_ms = statistics.median(intervals), statistics.fmean(intervals)
tps = count * 1000 / (nodes * 4 * median_ms)
result = {
    "model_key": os.environ.get("MODEL_KEY"),
    "afd_model_placement": os.environ["AFD_MODEL_PLACEMENT"],
    "afd_moe_backend": os.environ["MINISGL_AFD_MOE_BACKEND"],
    "megamoe_expert_weight_dtype": os.environ["MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE"],
    "prompt_mode": os.environ["PROMPT_MODE"],
    "context_tokens": context,
    "context_range_tokens": {"min": context_min, "max": context_max},
    "irregular_prompt_lengths": irregular,
    "prompt_length_distribution": "inclusive_uniform_symmetric_integer_linspace",
    "prompt_lengths_per_attention_gpu": prompt_lengths,
    "prompt_length_sum_per_attention_gpu": prompt_length_sum,
    "samples": count, "tokens_per_sample": output_tokens,
    "batch_per_attention_gpu": batch,
    "input_batch_per_attention_gpu": batch,
    "batch_selection": (
        "exact scheduler-admissible ceiling under measured KV capacity"
        if require_capacity_max
        else "explicit sweep point; post-init KV capacity reported separately"
    ),
    "capacity_max_batch": capacity_max_batch,
    "raw_kv_capacity_max_batch": raw_kv_max_batch,
    "nominal_kv_capacity_tokens_per_attention_gpu": nominal_capacity_kv,
    "nominal_scheduler_admissible_max_batch": nominal_capacity_max_batch,
    "nominal_raw_kv_capacity_max_batch": nominal_raw_kv_max_batch,
    "observed_kv_pages_per_attention_gpu": observed_pages,
    "observed_min_kv_capacity_tokens_per_attention_gpu": observed_min_capacity,
    "observed_scheduler_admissible_max_batch": observed_max_batch,
    "scheduler_running_reservation_tokens_per_request": running_reservation,
    "scheduler_admission_required_tokens_per_attention_gpu": admission_required,
    "scheduler_admission_unused_tokens_per_attention_gpu": admission_unused,
    "scheduler_admission_method": (
        "largest B satisfying (B-1)*(ceil(prompt/64)*64 + "
        f"({output_tokens - 1}+63)) + (prompt+{output_tokens}) "
        "<= measured KV capacity"
    ),
    "microbatch_real_sizes_per_attention_gpu": [(batch + 1) // 2, batch // 2],
    "graph_padded_batch_per_attention_gpu": padded_batch,
    "attention_workers": workers, "total_gpus": nodes * 4,
    "attention_trays": attention_trays, "ffn_trays": ffn_trays,
    "attention_to_ffn_tray_ratio": f"{attention_trays}:{ffn_trays}",
    "kv_capacity_tokens_per_attention_gpu": capacity_kv,
    "required_kv_tokens_per_attention_gpu": required_kv,
    "unused_kv_tokens_per_attention_gpu": capacity_kv - required_kv,
    "kv_preflight_method": "runtime evidence from every attention worker; capacity-max mode gates sampling on the observed minimum page count",
    "full_bucket_decode_windows": full_windows,
    "partial_padded_batch_decode_windows": partial_windows,
    "decode_step_ids": steps,
    "selected_window_policy": "first 15 full-real-batch decode steps; capture starts from observed scheduler state with no predicted step window",
    "completion_interval_ms": intervals,
    "mean_interval_ms": mean_ms, "median_interval_ms": median_ms,
    "median_tokens_per_second_per_gpu": tps,
    "prompt_contract": prompt_contract,
    "cuda_graph": {
        "enabled": True,
        "num_microbatches": 2,
        "configured_batch_per_microbatch": (batch + 1) // 2,
        "observed_padded_batch_per_attention_gpu": padded_batch,
        "nsys_cuda_graph_trace": os.environ["NSYS_CUDA_GRAPH_TRACE"],
        "nsys_target_batch_per_attention_gpu": int(os.environ["MINISGL_RAY_NSYS_TARGET_BATCH_PER_DP"]),
        "nsys_capture_decode_steps": int(os.environ["MINISGL_RAY_NSYS_CAPTURE_DECODE_STEPS"]),
        "proof": "capture-complete records one warmup plus 15 consecutive measured full-real-batch steps; every GPU-worker log records one synchronous profiler start and stop; strict extraction verifies exactly the declared graph launches in representative attention and model reports",
    },
    "measurement": "median of all 14 consecutive coordinator completion intervals in the first 15-step full-real-batch decode window selected by observed scheduler state; no prefill-step prediction",
}
if reference > 0:
    result["reference_tokens_per_second_per_gpu"] = reference
    result["reference_delta_percent"] = (tps / reference - 1) * 100
(run / "afd-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

touch "$SNAPSHOT"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$EXPERIMENT/../gpu-snapshots/final-rank-0.csv"
touch "$DONE/0"
deadline=$((SECONDS + 120))
until [[ $(find "$DONE" -maxdepth 1 -type f | wc -l) -eq "$NODES" ]]; do
    [[ $SECONDS -lt $deadline ]]; sleep 2
done
for ((rank=0; rank<NODES; rank++)); do
    # This snapshot is taken before the allocation-scoped EXIT trap stops Ray,
    # so surviving worker rows are cleanup diagnostics, not a run failure. A
    # subsequent allocation's initial snapshot remains the fail-fast proof
    # that cleanup completed.
    [[ -s "$EXPERIMENT/../gpu-snapshots/final-rank-$rank.csv" ]]
done
printf 'FASTAFD_AFD_SUCCESS result=%s\n' "$EXPERIMENT/afd-result.json" | tee "$EXPERIMENT/SUCCESS"
