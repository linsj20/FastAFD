#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="${FASTAFD_SOURCE_REPO:-$SCRIPT_DIR}"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
[[ -f "$_common_dir/scripts/lib/common.sh" ]] || {
  echo "could not locate scripts/lib/common.sh from $_common_dir" >&2
  exit 2
}
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "${FASTAFD_SOURCE_REPO:-$SCRIPT_DIR}"
RUNNER_DIR="$CALIB_DIR/scripts"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# Qwen3-235B-A22B-FP8 AFD AG/EG large-node 8k decode replay workload.
# PER_ATTN_GPU_BSZ is the inherited environment name; semantically it is the
# input batch per attention-DP lane, independent of that lane's TP degree.
#
# Node policy:
#   - AFD_TOTAL_NODES is attention_trays + FFN_trays.
#   - EP4/8/16/32 occupies full FFN trays in the pmap contract.
#   - The comprehensive contract also supports EP2 and packs contiguous
#     attention/FFN role slots onto the allocated four-GPU trays.
#   - Attention TP1/2/4 stays within a tray; TP8 spans adjacent tray pairs.
#   - Explicit rank-node rows avoid the generic combinatorial placement search.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-235B-A22B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-8192}"
export PER_ATTN_GPU_BSZ="${PER_ATTN_GPU_BSZ:-96}"
export MAX_TOKENS="${MAX_TOKENS:-16}"

export AFD_TOTAL_NODES="${AFD_TOTAL_NODES:-${NUM_NODES:-4}}"
export AFD_GPUS_PER_NODE="${AFD_GPUS_PER_NODE:-4}"
export AFD_NUM_MB="${AFD_NUM_MB:-2}"

if [[ -z "${AFD_NODE_LIST:-}" ]]; then
  export AFD_NODE_LIST="$(
    python - <<'PY'
import os
import ray

total = int(os.environ["AFD_TOTAL_NODES"])
ray.init(
    address=os.environ.get("RAY_ADDRESS", "auto"),
    ignore_reinit_error=True,
    logging_level="ERROR",
)
nodes = [
    node["NodeManagerAddress"]
    for node in ray.nodes()
    if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0)) > 0
]
nodes = sorted(dict.fromkeys(nodes))
if len(nodes) < total:
    raise SystemExit(f"need {total} Ray GPU nodes, found {len(nodes)}: {nodes}")
print(",".join(nodes[:total]))
PY
  )"
fi

IFS=',' read -r -a _AFD_NODES <<< "$AFD_NODE_LIST"
if (( ${#_AFD_NODES[@]} != AFD_TOTAL_NODES )); then
  echo "AFD_NODE_LIST must contain exactly AFD_TOTAL_NODES=$AFD_TOTAL_NODES nodes, got ${#_AFD_NODES[@]}" >&2
  exit 1
fi

export AFD_ATTN_TP_SIZE="${AFD_ATTN_TP_SIZE:-1}"
export AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
case "$AFD_ATTN_TP_SIZE" in 1|2|4|8) ;; *) echo "AFD_ATTN_TP_SIZE must be 1, 2, 4, or 8" >&2; exit 2 ;; esac
case "$AFD_MLP_EP_SIZE" in
  2|4|8|16|32) ;;
  *) echo "AFD_MLP_EP_SIZE must be 2, 4, 8, 16, or 32" >&2; exit 2 ;;
esac

_rank_rows() {
  local tp_size="$1"
  shift
  python - "$tp_size" "$AFD_GPUS_PER_NODE" "$@" <<'PY'
import sys
tp_size = int(sys.argv[1])
gpus_per_node = int(sys.argv[2])
nodes = sys.argv[3:]
slots = []
for node in nodes:
    slots.extend([node] * gpus_per_node)
if len(slots) % tp_size:
    raise SystemExit(
        f"{len(slots)} GPUs on nodes={nodes} cannot form TP{tp_size} groups"
    )
rows = [slots[i : i + tp_size] for i in range(0, len(slots), tp_size)]
print(",".join("+".join(row) for row in rows))
PY
}

_ATTN_NODES=()
_FFN_NODES=()
if [[ -n "${AFD_ATTENTION_GPU_COUNT:-}" ]]; then
  [[ "$AFD_ATTENTION_GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "AFD_ATTENTION_GPU_COUNT must be a positive integer" >&2
    exit 2
  }
  _TOTAL_ACTIVE_GPUS=$(( AFD_ATTENTION_GPU_COUNT + AFD_MLP_EP_SIZE ))
  _EXPECTED_NODES=$(( (_TOTAL_ACTIVE_GPUS + AFD_GPUS_PER_NODE - 1) / AFD_GPUS_PER_NODE ))
  (( _EXPECTED_NODES == AFD_TOTAL_NODES )) || {
    echo "packed topology requires $_EXPECTED_NODES nodes for $_TOTAL_ACTIVE_GPUS active GPUs, got $AFD_TOTAL_NODES" >&2
    exit 2
  }
  _PACKED_ROWS=$(
    python - "$AFD_ATTN_TP_SIZE" "$AFD_ATTENTION_GPU_COUNT" \
      "$AFD_MLP_EP_SIZE" "$AFD_GPUS_PER_NODE" "${_AFD_NODES[@]}" <<'PY'
import sys

attn_tp = int(sys.argv[1])
attn_gpus = int(sys.argv[2])
ffn_gpus = int(sys.argv[3])
gpus_per_node = int(sys.argv[4])
nodes = sys.argv[5:]
slots = [node for node in nodes for _ in range(gpus_per_node)]
if attn_gpus % attn_tp:
    raise SystemExit(f"{attn_gpus} attention GPUs cannot form TP{attn_tp}")
if attn_gpus + ffn_gpus > len(slots):
    raise SystemExit(
        f"{attn_gpus}+{ffn_gpus} active GPUs exceed {len(slots)} allocated slots"
    )
attn_slots = slots[:attn_gpus]
ffn_slots = slots[attn_gpus : attn_gpus + ffn_gpus]
attn_rows = [
    attn_slots[index : index + attn_tp]
    for index in range(0, len(attn_slots), attn_tp)
]
print(",".join("+".join(row) for row in attn_rows))
print(",".join(ffn_slots))
PY
  )
  _PACKED_ATTN_ROWS=$(printf '%s\n' "$_PACKED_ROWS" | sed -n '1p')
  _PACKED_FFN_ROWS=$(printf '%s\n' "$_PACKED_ROWS" | sed -n '2p')
  [[ -n "$_PACKED_ATTN_ROWS" && -n "$_PACKED_FFN_ROWS" ]]
  export MINISGL_AFD_ATTN_NODES="${MINISGL_AFD_ATTN_NODES:-$_PACKED_ATTN_ROWS}"
  export MINISGL_AFD_MLP_NODES="${MINISGL_AFD_MLP_NODES:-$_PACKED_FFN_ROWS}"
  export AFD_ATTN_DP_SIZE="$(( AFD_ATTENTION_GPU_COUNT / AFD_ATTN_TP_SIZE ))"
else
  _FFN_TRAYS=$(( AFD_MLP_EP_SIZE / AFD_GPUS_PER_NODE ))
  _ATTN_TRAYS=$(( AFD_TOTAL_NODES - _FFN_TRAYS ))
  (( _ATTN_TRAYS >= 1 )) || { echo "topology leaves no attention tray" >&2; exit 2; }
  _ATTN_NODES=("${_AFD_NODES[@]:0:_ATTN_TRAYS}")
  _FFN_NODES=("${_AFD_NODES[@]:_ATTN_TRAYS:_FFN_TRAYS}")
  export MINISGL_AFD_ATTN_NODES="${MINISGL_AFD_ATTN_NODES:-$(_rank_rows "$AFD_ATTN_TP_SIZE" "${_ATTN_NODES[@]}")}"
  export MINISGL_AFD_MLP_NODES="${MINISGL_AFD_MLP_NODES:-$(_rank_rows 1 "${_FFN_NODES[@]}")}"
  export AFD_ATTN_DP_SIZE="$(( ${#_ATTN_NODES[@]} * AFD_GPUS_PER_NODE / AFD_ATTN_TP_SIZE ))"
fi
export ATTN_DP_SIZE="$AFD_ATTN_DP_SIZE"
export ATTN_TP_SIZE="$AFD_ATTN_TP_SIZE"
export MLP_DP_SIZE="$AFD_MLP_EP_SIZE"
export MLP_TP_SIZE="1"

if [[ -n "${RUN_DIR:-}" ]]; then
  MINISGL_AFD_ATTN_NODES="$MINISGL_AFD_ATTN_NODES" \
  MINISGL_AFD_MLP_NODES="$MINISGL_AFD_MLP_NODES" \
  AFD_NODE_LIST="$AFD_NODE_LIST" \
  AFD_ATTN_DP_SIZE="$AFD_ATTN_DP_SIZE" \
  AFD_ATTN_TP_SIZE="$AFD_ATTN_TP_SIZE" \
  AFD_MLP_EP_SIZE="$AFD_MLP_EP_SIZE" \
  python - "$RUN_DIR/afd-placement.json" <<'PY'
import json
import os
import sys
from pathlib import Path

def rows(value):
    return [
        row.split("+")
        for row in value.split(",")
        if row
    ]

record = {
    "sweep_contract": os.environ.get("FASTAFD_SWEEP_CONTRACT", "pmap"),
    "allocated_nodes": os.environ["AFD_NODE_LIST"].split(","),
    "attention_dp_rank_nodes": rows(os.environ["MINISGL_AFD_ATTN_NODES"]),
    "mlp_dp_rank_nodes": rows(os.environ["MINISGL_AFD_MLP_NODES"]),
    "attention_dp_size": int(os.environ["AFD_ATTN_DP_SIZE"]),
    "attention_tp_size": int(os.environ["AFD_ATTN_TP_SIZE"]),
    "ffn_ep_size": int(os.environ["AFD_MLP_EP_SIZE"]),
    "placement_contract": "explicit contiguous active-GPU slots",
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
fi

export NUM_PROMPTS="${NUM_PROMPTS:-$(( PER_ATTN_GPU_BSZ * AFD_ATTN_DP_SIZE ))}"
export AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
export MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-8256}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-512}"

# Profiling is state driven: wait until every attention-DP lane has the full
# requested batch resident, then capture exactly the first 15 decode launches.
export NSYS_TARGET_BATCH_PER_ATTN_DP="${NSYS_TARGET_BATCH_PER_ATTN_DP:-$PER_ATTN_GPU_BSZ}"
export NSYS_CAPTURE_DECODE_STEPS="${NSYS_CAPTURE_DECODE_STEPS:-15}"
[[ "$NSYS_TARGET_BATCH_PER_ATTN_DP" == "$PER_ATTN_GPU_BSZ" ]] || {
  echo "NSYS_TARGET_BATCH_PER_ATTN_DP must equal PER_ATTN_GPU_BSZ" >&2
  exit 2
}
[[ "$NSYS_CAPTURE_DECODE_STEPS" == 15 ]] || {
  echo "NSYS_CAPTURE_DECODE_STEPS must be exactly 15" >&2
  exit 2
}

# Per-microbatch decode graph buckets. The default b96 workload replays bucket 48.
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,32,48,64}"
# Keep the original 4-node tuned KV-cache ratio by default. Larger node-count
# debug runs can override this together with graph settings from the environment.
export AFD_MEMORY_RATIO="${AFD_MEMORY_RATIO:-0.82}"
export AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend trtllm --memory-ratio $AFD_MEMORY_RATIO}"

export MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-megamoe_m2n}"
export MINISGL_MEGAMOE_AG_SMS="${MINISGL_MEGAMOE_AG_SMS:-24}"

# The shared-host cleanup helper scans and kills matching processes outside the
# current Slurm allocation. Bundle cases rely on run_afd.sh's allocation-scoped
# Ray and process cleanup instead.
export CLEAN_FIRST=0
export CLEAN_AFTER=0

export NSYS="${NSYS:-1}"
export NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
if [[ -z "${MINISGL_RAY_NSYS_BIN:-}" ]]; then
  if [[ -n "${CUDA_13_HOME:-}" && -x "$CUDA_13_HOME/bin/nsys" ]]; then
    export MINISGL_RAY_NSYS_BIN="$CUDA_13_HOME/bin/nsys"
  elif [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nsys" ]]; then
    export MINISGL_RAY_NSYS_BIN="$CUDA_HOME/bin/nsys"
  else
    export MINISGL_RAY_NSYS_BIN="$(command -v nsys || true)"
  fi
fi

export RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-0}"
export VLLM_SHARDED_SCORE="${VLLM_SHARDED_SCORE:-1}"
export VLLM_SHARDED_NODE_LIST="${VLLM_SHARDED_NODE_LIST:-$AFD_NODE_LIST}"
export VLLM_SHARDED_GPUS_PER_SERVER="${VLLM_SHARDED_GPUS_PER_SERVER:-$AFD_GPUS_PER_NODE}"
export VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
export VLLM_PROMPT_LOGPROBS="${VLLM_PROMPT_LOGPROBS:-0}"
export VLLM_SCORE_CONCURRENCY="${VLLM_SCORE_CONCURRENCY:-16}"
if [[ -z "${VLLM_EXTRA_ARGS:-}" ]]; then
  export VLLM_EXTRA_ARGS="--distributed-executor-backend mp --data-parallel-backend mp --data-parallel-size 4 --data-parallel-size-local 4 --enable-expert-parallel --all2all-backend deepep_low_latency --moe-backend deep_gemm --max-num-batched-tokens 40000 --max-num-seqs 16 --gpu-memory-utilization 0.97 --no-enable-prefix-caching --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,32,40,64,80,128,160,256]}'"
fi
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-1}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export VLLM_DEEPEPLL_FP8_DISPATCH="${VLLM_DEEPEPLL_FP8_DISPATCH:-1}"
export VLLM_DEEPEPLL_UE8M0_DISPATCH="${VLLM_DEEPEPLL_UE8M0_DISPATCH:-0}"
export ALIGN_MIN_TOP10_RATE="${ALIGN_MIN_TOP10_RATE:-1.0}"

export CLEAN_FIRST="${CLEAN_FIRST:-1}"
export CLEAN_AFTER="${CLEAN_AFTER:-1}"

PROMPT_BASE_FILE="${PROMPT_BASE_FILE:-$CALIB_DIR/prompts/prompts_512x8192_seed20260527.txt}"
DEFAULT_PROMPT_FILE="$CALIB_DIR/reports/prompts_${NUM_PROMPTS}x${PROMPT_LEN}_from512_seed20260527.txt"
if [[ -z "${PROMPT_FILE:-}" ]]; then
  export PROMPT_FILE="$DEFAULT_PROMPT_FILE"
  if [[ ! -f "$PROMPT_FILE" ]]; then
    python - "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$NUM_PROMPTS" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
count = int(sys.argv[3])
lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"no prompts in {src}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(lines[i % len(lines)] for i in range(count)) + "\n", encoding="utf-8")
print(f"wrote {count} prompts to {dst} from {len(lines)} source prompts")
PY
  fi
fi
export PROMPT_KIND="${PROMPT_KIND:-existing}"

AFD_MEMORY_TAG="${AFD_MEMORY_RATIO/./}"
export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen3_235b_a22b_fp8_8k_b${PER_ATTN_GPU_BSZ}_perlane_attn${ATTN_DP_SIZE}dp_tp${ATTN_TP_SIZE}_mlp${MLP_DP_SIZE}dp_ep${AFD_MLP_EP_SIZE}_${#_AFD_NODES[@]}node_mb${AFD_NUM_MB}_mem${AFD_MEMORY_TAG}_bucket${AFD_DECODE_GRAPH_BS}_megamoe_nsys_${TIMESTAMP}}"

echo "AFD parallel-mapping Qwen3-235B b${PER_ATTN_GPU_BSZ}/attention-DP-lane"
echo "  total_nodes: ${#_AFD_NODES[@]}"
if [[ -n "${AFD_ATTENTION_GPU_COUNT:-}" ]]; then
  echo "  role_node_policy: packed contiguous active-GPU slots"
else
  echo "  attn_nodes: ${_ATTN_NODES[*]}"
  echo "  ffn_nodes: ${_FFN_NODES[*]}"
fi
echo "  explicit_attn_rank_rows: $MINISGL_AFD_ATTN_NODES"
echo "  explicit_ffn_rank_rows: $MINISGL_AFD_MLP_NODES"
echo "  attn_dp/tp: $ATTN_DP_SIZE/$ATTN_TP_SIZE"
echo "  mlp_dp/tp/ep: $MLP_DP_SIZE/$MLP_TP_SIZE/$AFD_MLP_EP_SIZE"
echo "  num_prompts: $NUM_PROMPTS"
echo "  per_attn_dp_lane_bsz: $PER_ATTN_GPU_BSZ"
echo "  per_attn_dp_lane_mb_bsz: $(( (PER_ATTN_GPU_BSZ + AFD_NUM_MB - 1) / AFD_NUM_MB ))"
echo "  nsys_target_batch_per_attn_dp: $NSYS_TARGET_BATCH_PER_ATTN_DP"
echo "  nsys_capture_decode_steps: $NSYS_CAPTURE_DECODE_STEPS"
echo "  nsys_capture_policy: first exact full-batch decode window"
echo "  run_vllm_alignment: $RUN_VLLM_ALIGNMENT"
if [[ "$RUN_VLLM_ALIGNMENT" == "1" ]]; then
  echo "  vllm_sharded_score: $VLLM_SHARDED_SCORE"
  echo "  vllm_shard_nodes: $VLLM_SHARDED_NODE_LIST"
  echo "  vllm_dp/ep/tp_per_shard: 4/4/$VLLM_TP_SIZE"
  echo "  vllm_score_concurrency_per_shard: $VLLM_SCORE_CONCURRENCY"
fi

exec bash "$RUNNER_DIR/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh"
