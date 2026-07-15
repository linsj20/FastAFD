#!/usr/bin/env bash
# Reproduce a published one-tray vLLM decode point.
# Usage: ./run_vllm.sh [qwen3|minimax] [8k|16k]
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:vllm

set -euo pipefail
ulimit -s 8192

MODEL_KEY=${MODEL_KEY:-${1:-qwen3}}
CONTEXT=${CONTEXT:-${2:-8k}}
case "$CONTEXT" in
    8k|8192) CONTEXT=8192 ;;
    16k|16384) CONTEXT=16384 ;;
    *) echo "context must be 8k or 16k" >&2; exit 2 ;;
esac

EXPECTED_HEAD=${FASTAFD_EXPECTED_HEAD:-3c7161949310b6d59d6b4cf9bf997a4935c8113b}
ROOT=${FASTAFD_REPRO_ROOT:-$HOME/scratch/fastafd_reproduce}
SOURCE_REPO=${FASTAFD_SOURCE_REPO:-$HOME/scratch/github/FastAFD}
IMAGE=${FASTAFD_IMAGE:-$HOME/scratch/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh}
EP_VENV_DIR=${FASTAFD_EP_VENV_DIR:-$ROOT/envs/minisgl-${EXPECTED_HEAD:0:8}-cuda130-vllm-ep}
HF_CACHE=$ROOT/models/huggingface/hub

case "$MODEL_KEY:$CONTEXT" in
    qwen3:8192)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        BATCH=64; MAX_MODEL_LEN=8256; REFERENCE_TPS=1781 ;;
    qwen3:16384)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        BATCH=32; MAX_MODEL_LEN=16448; REFERENCE_TPS=954 ;;
    minimax:8192)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        BATCH=48; MAX_MODEL_LEN=8320; REFERENCE_TPS=1516 ;;
    minimax:16384)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        BATCH=24; MAX_MODEL_LEN=16640; REFERENCE_TPS=745 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac
if [[ "$MODEL_KEY" == qwen3 ]]; then
    MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
else
    MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
fi
MODEL_PATH=${FASTAFD_MODEL_PATH:-$HF_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION}
PROMPT_FILE=$SOURCE_REPO/prompts/prompts_512x${CONTEXT}_seed20260527.txt
GPU_MEMORY_UTILIZATION=${FASTAFD_VLLM_GPU_MEMORY_UTILIZATION:-0.94}
[[ "$GPU_MEMORY_UTILIZATION" == 0.94 ]] || {
    echo "published resident batches require gpu memory utilization 0.94" >&2
    exit 2
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
    [[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
    [[ -z "$(git -C "$SOURCE_REPO" status --porcelain)" ]]
    [[ -x "$EP_VENV_DIR/bin/python" && -d "$MODEL_PATH" && -f "$PROMPT_FILE" ]]
    ! squeue -h -u "$USER" -o '%j' | grep -q '^fastafd:' || {
        echo "another FastAFD job is active" >&2; exit 1;
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR=$ROOT/vllm_${MODEL_KEY}_${CONTEXT}_b${BATCH}_$STAMP
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable)
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,MODEL_KEY=$MODEL_KEY,CONTEXT=$CONTEXT,MODEL_PATH=$MODEL_PATH,MODEL_REVISION=$MODEL_REVISION,BATCH=$BATCH,MAX_MODEL_LEN=$MAX_MODEL_LEN,REFERENCE_TPS=$REFERENCE_TPS,PROMPT_FILE=$PROMPT_FILE,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION,EXPECTED_HEAD=$EXPECTED_HEAD,ROOT=$ROOT,SOURCE_REPO=$SOURCE_REPO,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s run_dir=%s\n' "$JOB" "$RUN_DIR"
    exit 0
fi

: "${RUN_DIR:?}" "${JOB_SCRIPT:?}"
if [[ "${FASTAFD_IN_CONTAINER:-0}" != 1 ]]; then
    [[ "$SLURM_JOB_PARTITION" == batch && "$SLURM_JOB_QOS" == short ]]
    exec srun --nodes=1 --ntasks=1 --gres=gpu:4 \
        --container-image="$IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        --container-env=NCCL_IB_TIMEOUT,NCCL_IB_SL,NCCL_DEBUG,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NET_GDR_C2C,NCCL_IB_HCA,NCCL_SOCKET_IFNAME,UCX_TLS,UCX_NET_DEVICES \
        env FASTAFD_IN_CONTAINER=1 MODEL_KEY="$MODEL_KEY" CONTEXT="$CONTEXT" \
        MODEL_PATH="$MODEL_PATH" MODEL_REVISION="$MODEL_REVISION" \
        BATCH="$BATCH" MAX_MODEL_LEN="$MAX_MODEL_LEN" REFERENCE_TPS="$REFERENCE_TPS" \
        PROMPT_FILE="$PROMPT_FILE" GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        EXPECTED_HEAD="$EXPECTED_HEAD" ROOT="$ROOT" SOURCE_REPO="$SOURCE_REPO" \
        IMAGE="$IMAGE" EP_VENV_DIR="$EP_VENV_DIR" RUN_DIR="$RUN_DIR" \
        JOB_SCRIPT="$JOB_SCRIPT" bash "$JOB_SCRIPT"
fi

[[ "$(uname -m)" == aarch64 ]]
[[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
[[ -z "$(git -C "$SOURCE_REPO" status --porcelain)" ]]
[[ -x "$EP_VENV_DIR/bin/python" && -f "$MODEL_PATH/model.safetensors.index.json" ]]
[[ -f "$PROMPT_FILE" ]]
PYTHON=$EP_VENV_DIR/bin/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1 VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_DEEP_GEMM_WARMUP=skip VLLM_DEEPEPLL_FP8_DISPATCH=1
export VLLM_DEEPEPLL_UE8M0_DISPATCH=0 TMPDIR=$RUN_DIR/tmp
mkdir -p "$TMPDIR" "$RUN_DIR/nsys"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RUN_DIR/gpu-processes-initial.csv"
[[ $(wc -l < "$RUN_DIR/gpu-processes-initial.csv") -eq 1 ]]
sha256sum "$PROMPT_FILE" "$JOB_SCRIPT" > "$RUN_DIR/inputs.sha256"

read -r -d '' BENCHMARK <<'PY' || true
import argparse, hashlib, json, os, statistics, time
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
p.add_argument("--batch", type=int, required=True)
p.add_argument("--max-model-len", type=int, required=True)
p.add_argument("--reference-tps", type=float, required=True)
p.add_argument("--memory", type=float, required=True)
a = p.parse_args()
rank, local_rank, world = (int(os.environ[x]) for x in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
if world != 4 or rank != local_rank or a.memory != 0.94:
    raise RuntimeError((rank, local_rank, world, a.memory))

source_bytes = a.prompts.read_bytes()
source = [x.strip() for x in source_bytes.decode().splitlines() if x.strip()]
if len(source) != 512:
    raise RuntimeError(len(source))
global_prompts = source[: 4 * a.batch]
local_prompts = global_prompts[rank::4]
llm = LLM(
    model=a.model, tokenizer=a.model, tensor_parallel_size=1, data_parallel_size=4,
    trust_remote_code=a.model_key == "minimax",
    pipeline_parallel_size=1, distributed_executor_backend="external_launcher",
    enable_expert_parallel=True, all2all_backend="deepep_low_latency",
    moe_backend="deep_gemm", max_model_len=a.max_model_len,
    max_num_batched_tokens=40000, max_num_seqs=a.batch,
    gpu_memory_utilization=a.memory, enable_prefix_caching=False, seed=0,
    enforce_eager=False, compilation_config={
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": sorted({
            1, 2, 4, 8, 16, 32, 40, 64, 80, 128, 160, 256, a.batch
        }),
    }, disable_log_stats=True,
)
cfg = llm.llm_engine.vllm_config
cache = cfg.cache_config
available = cache.num_gpu_blocks * cache.block_size
required = a.batch * a.max_model_len
if available < required:
    raise RuntimeError(f"KV capacity {available} < {required}")
inputs = list(llm._preprocess_cmpl(local_prompts))
lengths = [len(x["prompt_token_ids"]) for x in inputs]
if a.model_key == "qwen3" and lengths != [a.context] * a.batch:
    raise RuntimeError(sorted(set(lengths)))
if a.model_key == "minimax" and (
    min(lengths) <= 0 or max(lengths) + 64 > a.max_model_len
):
    raise RuntimeError(sorted(set(lengths)))
params = SamplingParams(
    temperature=0, max_tokens=64, min_tokens=64, ignore_eos=True,
    detokenize=False, output_kind=RequestOutputKind.DELTA,
)
request_ids = {f"r{rank}-{i}" for i in range(a.batch)}
for request_id, engine_input in zip(sorted(request_ids), inputs, strict=True):
    llm.llm_engine.add_request(request_id, engine_input, params)

generated = {x: 0 for x in request_ids}
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
if rank == 0 and int(cudart.cudaProfilerStart()) != 0:
    raise RuntimeError("cudaProfilerStart failed")
dist.barrier(group=group)
step_ms = []
for step in range(15):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    outputs = llm.llm_engine.step()
    torch.cuda.synchronize()
    step_ms.append((time.perf_counter_ns() - start) / 1e6)
    if len(outputs) != a.batch or {x.request_id for x in outputs} != request_ids:
        raise RuntimeError((step, len(outputs)))
    if any(len(x.outputs[0].token_ids) != 1 or x.finished for x in outputs):
        raise RuntimeError(f"invalid decode output at step {step}")
dist.barrier(group=group)
if rank == 0 and int(cudart.cudaProfilerStop()) != 0:
    raise RuntimeError("cudaProfilerStop failed")
dist.barrier(group=group)

all_ms = [None] * 4
dist.all_gather_object(all_ms, step_ms, group=group)
all_lengths = [None] * 4
dist.all_gather_object(all_lengths, lengths, group=group)
(a.run_dir / f"rank-{rank}.json").write_text(json.dumps({
    "rank": rank, "step_ms": step_ms, "warmup_token_counts": sorted(generated.values()),
    "prompt_lengths": sorted(set(lengths)),
}, indent=2) + "\n")
if rank == 0:
    synchronized = [max(values[i] for values in all_ms) for i in range(15)]
    observed_lengths = [length for values in all_lengths for length in values]
    mean_ms, median_ms = statistics.fmean(synchronized), statistics.median(synchronized)
    result = {
        "model": a.model, "model_revision": a.revision,
        "context_tokens": a.context, "batch_per_gpu": a.batch,
        "global_prompts": 4 * a.batch, "gpu_memory_utilization": a.memory,
        "kv_capacity_tokens_per_gpu": available, "required_kv_tokens_per_gpu": required,
        "prompt_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "observed_prompt_tokens": {
            "min": min(observed_lengths), "max": max(observed_lengths),
            "rank_major_sha256": hashlib.sha256(
                json.dumps(observed_lengths).encode()
            ).hexdigest(),
        },
        "wall": {
            "synchronized_step_ms": synchronized,
            "mean_ms": mean_ms, "median_ms": median_ms,
            "mean_tokens_per_second_per_gpu": a.batch * 1000 / mean_ms,
            "median_tokens_per_second_per_gpu": a.batch * 1000 / median_ms,
        },
        "reference_tokens_per_second_per_gpu": a.reference_tps,
        "measurement": "15 full-resident decode steps; wall=max across four DP ranks",
    }
    (a.run_dir / "baseline-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
dist.barrier(group=group)
PY

/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --trace-fork-before-exec=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="$RUN_DIR/nsys/decode" \
    "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node=4 --no-python "$PYTHON" -c "$BENCHMARK" \
    --model "$MODEL_PATH" --model-key "$MODEL_KEY" --revision "$MODEL_REVISION" \
    --prompts "$PROMPT_FILE" --run-dir "$RUN_DIR" --context "$CONTEXT" \
    --batch "$BATCH" --max-model-len "$MAX_MODEL_LEN" \
    --reference-tps "$REFERENCE_TPS" --memory "$GPU_MEMORY_UTILIZATION" \
    2>&1 | tee "$RUN_DIR/torchrun.log"

REPORT=$(find "$RUN_DIR/nsys" -maxdepth 1 -name '*.nsys-rep' -type f)
[[ -n "$REPORT" && -s "$RUN_DIR/baseline-result.json" ]]
/usr/local/cuda/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum \
    "$REPORT" > "$RUN_DIR/nsys/stats.txt"
SQLITE=$(find "$RUN_DIR/nsys" -maxdepth 1 -name '*.sqlite' -type f)
"$PYTHON" - "$RUN_DIR/baseline-result.json" "$SQLITE" "$BATCH" "$REFERENCE_TPS" <<'PY'
import json, sqlite3, sys
from pathlib import Path
result_path, sqlite_path, batch, reference = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
con = sqlite3.connect(sqlite_path)
rows = list(con.execute(
    "select deviceId, sum(end-start), count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId order by deviceId"
))
if len(rows) != 4 or any(total <= 0 for _, total, _ in rows):
    raise RuntimeError(rows)
mean_ns = sum(total for _, total, _ in rows) / (4 * 15)
tps = batch * 1e9 / mean_ns
result = json.loads(result_path.read_text())
result["cuda_kernels"] = {
    "per_device_total_ns": {str(device): total for device, total, _ in rows},
    "per_device_kernel_count": {str(device): count for device, _, count in rows},
    "mean_ms_per_step_per_gpu": mean_ns / 1e6,
    "tokens_per_second_per_gpu": tps,
    "reference_delta_percent": (tps / reference - 1) * 100,
    "method": "sum of CUDA kernel durations in the 15-step profiler range / (4 GPUs x 15 steps)",
}
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$RUN_DIR/gpu-processes-final.csv"
[[ $(wc -l < "$RUN_DIR/gpu-processes-final.csv") -eq 1 ]]
printf 'FASTAFD_VLLM_SUCCESS result=%s\n' "$RUN_DIR/baseline-result.json" | tee "$RUN_DIR/SUCCESS"
