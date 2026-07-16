#!/usr/bin/env bash
# Reproduce a published vLLM baseline at DP=EP4/8/16/32/64.
# Usage: ./run_vllm.sh [qwen3|minimax] [8k|16k] [4|8|16|32|64]
#
# EP8+ per-lane batches are estimated from measured EP4 weight/KV footprints.
# Post-initialization KV capacity must prove that the selected value is exactly
# the maximum full-length batch. EP4 retains the published batch for reference.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:vllm

set -euo pipefail
ulimit -s 8192

MODEL_KEY=${MODEL_KEY:-${1:-qwen3}}
CONTEXT=${CONTEXT:-${2:-8k}}
EP_SIZE=${EP_SIZE:-${3:-4}}
case "$CONTEXT" in
    8k|8192) CONTEXT=8192 ;;
    16k|16384) CONTEXT=16384 ;;
    *) echo "context must be 8k or 16k" >&2; exit 2 ;;
esac
case "$EP_SIZE" in
    4) NODES=1 ;;
    8) NODES=2 ;;
    16) NODES=4 ;;
    32) NODES=8 ;;
    64) NODES=16 ;;
    *) echo "EP size must be 4, 8, 16, 32, or 64" >&2; exit 2 ;;
esac

case "$MODEL_KEY:$CONTEXT:$EP_SIZE" in
    qwen3:8192:4) BATCH=64 ;;
    qwen3:8192:8) BATCH=88 ;;
    qwen3:8192:16) BATCH=96 ;;
    qwen3:8192:32) BATCH=101 ;;
    qwen3:8192:64) BATCH=103 ;;
    qwen3:16384:4) BATCH=32 ;;
    qwen3:16384:8) BATCH=44 ;;
    qwen3:16384:16) BATCH=48 ;;
    qwen3:16384:32) BATCH=50 ;;
    qwen3:16384:64) BATCH=52 ;;
    minimax:8192:4) BATCH=48 ;;
    minimax:8192:8) BATCH=68 ;;
    minimax:8192:16) BATCH=74 ;;
    minimax:8192:32) BATCH=78 ;;
    minimax:8192:64) BATCH=79 ;;
    minimax:16384:4) BATCH=24 ;;
    minimax:16384:8) BATCH=34 ;;
    minimax:16384:16) BATCH=37 ;;
    minimax:16384:32) BATCH=39 ;;
    minimax:16384:64) BATCH=39 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac

case "$MODEL_KEY:$CONTEXT" in
    qwen3:8192)
        MAX_MODEL_LEN=8256
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        REFERENCE_EP4_TPS=1779.83 ;;
    qwen3:16384)
        MAX_MODEL_LEN=16448
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        PROMPT_SHA256=918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca
        REFERENCE_EP4_TPS=935.7644397953262 ;;
    minimax:8192)
        MAX_MODEL_LEN=8320
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PROMPT_SHA256=26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb
        REFERENCE_EP4_TPS=1512.5715032858584 ;;
    minimax:16384)
        MAX_MODEL_LEN=16640
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PROMPT_SHA256=918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca
        REFERENCE_EP4_TPS=735.3254410690338 ;;
esac

ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$HOME/scratch/github/FastAFD}
IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-3c716194-cuda130-vllm-ep}
MODEL_PATH=${FASTAFD_MODEL_PATH:-$ROOT/models/huggingface/hub/$MODEL_CACHE/snapshots/$MODEL_REVISION}
PROMPT_FILE=${FASTAFD_PROMPT_FILE:-$SOURCE_REPO/prompts/prompts_512x${CONTEXT}_seed20260527.txt}
GPU_MEMORY_UTILIZATION=${FASTAFD_VLLM_GPU_MEMORY_UTILIZATION:-0.94}
[[ "$GPU_MEMORY_UTILIZATION" == 0.94 ]] || {
    echo "published baseline comparison requires GPU memory utilization 0.94" >&2
    exit 2
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
    [[ -x "$EP_VENV_DIR/bin/python" && -f "$MODEL_PATH/model.safetensors.index.json" ]]
    [[ -f "$PROMPT_FILE" ]]
    [[ $(sha256sum "$PROMPT_FILE" | awk '{print $1}') == "$PROMPT_SHA256" ]]
    JOB_NAME="fastafd:vllm-${MODEL_KEY}-${CONTEXT}-ep${EP_SIZE}"
    ! squeue -h -u "$USER" -o '%j' | grep -qx "$JOB_NAME" || {
        echo "the requested baseline job is already active" >&2
        exit 1
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR=$ROOT/vllm_${MODEL_KEY}_${CONTEXT}_dp${EP_SIZE}_ep${EP_SIZE}_b${BATCH}_$STAMP
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable --nodes="$NODES" --segment="$NODES")
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,MODEL_KEY=$MODEL_KEY,EP_SIZE=$EP_SIZE,NODES=$NODES,BATCH=$BATCH,CONTEXT=$CONTEXT,MAX_MODEL_LEN=$MAX_MODEL_LEN,MODEL_REVISION=$MODEL_REVISION,PROMPT_SHA256=$PROMPT_SHA256,REFERENCE_EP4_TPS=$REFERENCE_EP4_TPS,ROOT=$ROOT,SOURCE_REPO=$SOURCE_REPO,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,MODEL_PATH=$MODEL_PATH,PROMPT_FILE=$PROMPT_FILE,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s model=%s context=%s ep=%s nodes=%s batch=%s run_dir=%s\n' \
        "$JOB" "$MODEL_KEY" "$CONTEXT" "$EP_SIZE" "$NODES" "$BATCH" "$RUN_DIR"
    exit 0
fi

: "${RUN_DIR:?}" "${JOB_SCRIPT:?}"
if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == short ]]
    [[ "$SLURM_JOB_NUM_NODES" == "$NODES" ]]
    HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
    exec srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
        --gres=gpu:4 --kill-on-bad-exit=1 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_IN_CONTAINER=1 HEAD_HOST="$HEAD_HOST" MODEL_KEY="$MODEL_KEY" \
        EP_SIZE="$EP_SIZE" \
        NODES="$NODES" BATCH="$BATCH" CONTEXT="$CONTEXT" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" MODEL_REVISION="$MODEL_REVISION" \
        PROMPT_SHA256="$PROMPT_SHA256" REFERENCE_EP4_TPS="$REFERENCE_EP4_TPS" \
        ROOT="$ROOT" SOURCE_REPO="$SOURCE_REPO" IMAGE="$IMAGE" \
        EP_VENV_DIR="$EP_VENV_DIR" MODEL_PATH="$MODEL_PATH" \
        PROMPT_FILE="$PROMPT_FILE" GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        RUN_DIR="$RUN_DIR" JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

NODE_RANK=${SLURM_PROCID:?}
[[ "$NODE_RANK" =~ ^[0-9]+$ && "$NODE_RANK" -lt "$NODES" ]]
[[ "$(uname -m)" == aarch64 ]]
PYTHON=$EP_VENV_DIR/bin/python
[[ -x "$PYTHON" && -f "$MODEL_PATH/model.safetensors.index.json" && -f "$PROMPT_FILE" ]]
[[ $(sha256sum "$PROMPT_FILE" | awk '{print $1}') == "$PROMPT_SHA256" ]]

export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
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
    sha256sum "$PROMPT_FILE" "$JOB_SCRIPT" > "$RUN_DIR/inputs.sha256"
fi

read -r -d '' BENCHMARK <<'PY' || true
import argparse
import hashlib
import json
import os
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
p.add_argument("--max-model-len", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--memory", type=float, required=True)
p.add_argument("--reference-ep4-tps", type=float, required=True)
a = p.parse_args()

rank, local_rank, world = (
    int(os.environ[x]) for x in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
)
if world not in (4, 8, 16, 32, 64) or rank % 4 != local_rank or a.memory != 0.94:
    raise RuntimeError((rank, local_rank, world, a.memory))

capture_sizes = sorted({1, 2, 4, 8, 16, 32, 40, 64, 80, 128, 160, 256, a.batch})
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

cache = llm.llm_engine.vllm_config.cache_config
available = cache.num_gpu_blocks * cache.block_size
capacity_batch = available // a.max_model_len
if (world == 4 and capacity_batch < a.batch) or (
    world > 4 and capacity_batch != a.batch
):
    raise RuntimeError(
        f"configured batch {a.batch} does not satisfy KV maximum {capacity_batch}"
    )
batch = a.batch
required = batch * a.max_model_len

source_bytes = a.prompts.read_bytes()
source = [x.strip() for x in source_bytes.decode().splitlines() if x.strip()]
if len(source) != 512:
    raise RuntimeError(len(source))
global_prompts = [source[i % len(source)] for i in range(world * batch)]
local_prompts = global_prompts[rank::world]
inputs = list(llm._preprocess_cmpl(local_prompts))
lengths = [len(x["prompt_token_ids"]) for x in inputs]
if a.model_key == "qwen3" and lengths != [a.context] * batch:
    raise RuntimeError(sorted(set(lengths)))
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

generated = {request_id: 0 for request_id in request_ids}
for _ in range(64):
    outputs = llm.llm_engine.step()
    for output in outputs:
        if output.request_id not in generated or len(output.outputs[0].token_ids) != 1:
            raise RuntimeError(output)
        generated[output.request_id] += 1
    if min(generated.values()) >= 1:
        break
else:
    raise RuntimeError("resident-batch warmup did not finish")
if max(generated.values()) + 15 >= 64:
    raise RuntimeError(generated)

group = get_world_group().cpu_group
dist.barrier(group=group)
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
        "context_tokens": a.context,
        "data_parallel_size": world,
        "expert_parallel_size": world,
        "batch_per_dp_lane": batch,
        "batch_selection": (
            "published EP4 batch; capacity reported separately"
            if world == 4
            else "estimated from EP4 weight/KV footprint and proven by KV capacity"
        ),
        "capacity_max_batch": capacity_batch,
        "global_prompts": world * batch,
        "gpu_memory_utilization": a.memory,
        "deepep_buffer_size_mb": int(os.environ["VLLM_DEEPEP_BUFFER_SIZE_MB"]),
        "deepep_low_latency_use_mnnvl": bool(
            int(os.environ["VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL"])
        ),
        "kv_capacity_tokens_per_gpu": available,
        "required_kv_tokens_per_gpu": required,
        "unused_kv_tokens_per_gpu": available - required,
        "prompt_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prompt_repetition": "global prompt i uses source prompt i modulo 512",
        "observed_prompt_tokens": {
            "min": min(observed_lengths),
            "max": max(observed_lengths),
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
            "15 full-resident decode steps; wall=max across all DP ranks; "
            "CUDA metric added from all per-node Nsight reports"
        ),
    }
    (a.run_dir / "baseline-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
PY

# Derive a high rendezvous port from the allocation rather than EP size so
# unrelated jobs with the same topology cannot collide on a shared host.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --trace-fork-before-exec=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="$RUN_DIR/nsys/decode-node-$NODE_RANK" \
    "$PYTHON" -m torch.distributed.run \
    --nnodes="$NODES" --nproc-per-node=4 --node-rank="$NODE_RANK" \
    --master-addr="$HEAD_HOST" --master-port="$MASTER_PORT" \
    --no-python "$PYTHON" -c "$BENCHMARK" \
    --model "$MODEL_PATH" --model-key "$MODEL_KEY" --revision "$MODEL_REVISION" \
    --prompts "$PROMPT_FILE" --run-dir "$RUN_DIR" --context "$CONTEXT" \
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
    "$RUN_DIR/control/profile-done" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
world = int(sys.argv[2])
reference = float(sys.argv[3])
done_dir = Path(sys.argv[4])
sqlite_paths = [Path((done_dir / str(node)).read_text().strip()) for node in range(world // 4)]
rows = []
for node, sqlite_path in enumerate(sqlite_paths):
    con = sqlite3.connect(sqlite_path)
    node_rows = list(con.execute(
        "select deviceId, sum(end-start), count(*) "
        "from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId order by deviceId"
    ))
    if len(node_rows) != 4 or any(total <= 0 for _, total, _ in node_rows):
        raise RuntimeError((node, node_rows))
    rows.extend((node, device, total, count) for device, total, count in node_rows)
if len(rows) != world:
    raise RuntimeError(len(rows))

result = json.loads(result_path.read_text())
batch = int(result["batch_per_dp_lane"])
mean_ns = sum(total for _, _, total, _ in rows) / (world * 15)
tps = batch * 1e9 / mean_ns
result["cuda_kernels"] = {
    "per_device_total_ns": {
        f"node{node}:device{device}": total for node, device, total, _ in rows
    },
    "per_device_kernel_count": {
        f"node{node}:device{device}": count for node, device, _, count in rows
    },
    "mean_ms_per_step_per_gpu": mean_ns / 1e6,
    "tokens_per_second_per_gpu": tps,
    "dp4_ep4_speed_ratio": tps / reference,
    "dp4_ep4_delta_percent": (tps / reference - 1) * 100,
    "method": (
        "sum of CUDA kernel durations in every node's 15-step profiler range "
        "/ (world GPUs x 15 steps)"
    ),
}
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

printf 'FASTAFD_VLLM_SUCCESS result=%s\n' "$RUN_DIR/baseline-result.json" \
    | tee "$RUN_DIR/SUCCESS"
