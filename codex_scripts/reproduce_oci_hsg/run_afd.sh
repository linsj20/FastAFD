#!/usr/bin/env bash
# Reproduce a published multi-tray FastAFD decode point.
# Usage: ./run_afd.sh [qwen3|minimax] [8k|16k]
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=fastafd:afd

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
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        PRESET=scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
        NODES=8; BATCH=96; REFERENCE_TPS=2518 ;;
    qwen3:16384)
        MODEL_REVISION=39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35
        MODEL_CACHE=models--Qwen--Qwen3-235B-A22B-FP8
        PRESET=scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_16k_b48_dynamicnode_mb2_nsys_alignment.sh
        NODES=12; BATCH=48; REFERENCE_TPS=1377 ;;
    minimax:8192)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_8k_b72_dynamicnode_mb2_nsys_alignment.sh
        NODES=18; BATCH=72; REFERENCE_TPS=2198 ;;
    minimax:16384)
        MODEL_REVISION=f710177d938eff80b684d42c5aa84b382612f21f
        MODEL_CACHE=models--MiniMaxAI--MiniMax-M2.5
        PRESET=scripts/experiments/afd/minimax_m25/run_afd_minimax_m25_fp8_16k_b36_dynamicnode_mb2_nsys_alignment.sh
        NODES=18; BATCH=36; REFERENCE_TPS=1006 ;;
    *) echo "model must be qwen3 or minimax" >&2; exit 2 ;;
esac
MODEL_PATH=${FASTAFD_MODEL_PATH:-$HF_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION}
PROMPT_BASE=$SOURCE_REPO/prompts/prompts_512x${CONTEXT}_seed20260527.txt
ATTENTION_WORKERS=$(( (NODES - 1) * 4 ))
NUM_PROMPTS=$(( ATTENTION_WORKERS * BATCH ))

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ $(hostname -s) == oci-hsg-cs-001-login-* ]]
    [[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
    [[ -z "$(git -C "$SOURCE_REPO" status --porcelain)" ]]
    [[ -x "$EP_VENV_DIR/bin/python" && -d "$MODEL_PATH" && -f "$PROMPT_BASE" ]]
    ! squeue -h -u "$USER" -o '%j' | grep -q '^fastafd:' || {
        echo "another FastAFD job is active" >&2; exit 1;
    }
    STAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR=$ROOT/afd_${MODEL_KEY}_${CONTEXT}_b${BATCH}_$STAMP
    mkdir -p "$RUN_DIR"
    SBATCH_ARGS=(--parsable --nodes="$NODES" --segment="$NODES")
    if [[ -n "${FASTAFD_EXCLUDE_NODE:-}" ]]; then
        SBATCH_ARGS+=(--exclude="$FASTAFD_EXCLUDE_NODE")
    fi
    JOB=$(sbatch "${SBATCH_ARGS[@]}" \
        --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
        --export="ALL,MODEL_KEY=$MODEL_KEY,CONTEXT=$CONTEXT,MODEL_PATH=$MODEL_PATH,MODEL_REVISION=$MODEL_REVISION,PRESET=$PRESET,NODES=$NODES,BATCH=$BATCH,REFERENCE_TPS=$REFERENCE_TPS,ATTENTION_WORKERS=$ATTENTION_WORKERS,NUM_PROMPTS=$NUM_PROMPTS,PROMPT_BASE=$PROMPT_BASE,EXPECTED_HEAD=$EXPECTED_HEAD,ROOT=$ROOT,SOURCE_REPO=$SOURCE_REPO,IMAGE=$IMAGE,EP_VENV_DIR=$EP_VENV_DIR,RUN_DIR=$RUN_DIR,JOB_SCRIPT=$(realpath "$0")" \
        "$(realpath "$0")")
    printf 'submitted job=%s run_dir=%s\n' "$JOB" "$RUN_DIR"
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
        CONTEXT="$CONTEXT" MODEL_PATH="$MODEL_PATH" MODEL_REVISION="$MODEL_REVISION" \
        PRESET="$PRESET" NODES="$NODES" BATCH="$BATCH" REFERENCE_TPS="$REFERENCE_TPS" \
        ATTENTION_WORKERS="$ATTENTION_WORKERS" NUM_PROMPTS="$NUM_PROMPTS" \
        PROMPT_BASE="$PROMPT_BASE" EXPECTED_HEAD="$EXPECTED_HEAD" ROOT="$ROOT" \
        SOURCE_REPO="$SOURCE_REPO" IMAGE="$IMAGE" EP_VENV_DIR="$EP_VENV_DIR" \
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
[[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]]
[[ -z "$(git -C "$SOURCE_REPO" status --porcelain)" ]]
[[ -f "$MODEL_PATH/model.safetensors.index.json" && -f "$PROMPT_BASE" ]]
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

export RAY_ADDRESS=$HEAD_ADDRESS MINISGL_RAY_ADDRESS=$HEAD_ADDRESS MODEL_PATH
export PROMPT_BASE_FILE=$PROMPT_BASE
export PROMPT_FILE=$RUN_DIR/prompts_${NUM_PROMPTS}x${CONTEXT}.txt
"$PYTHON" - "$PROMPT_BASE_FILE" "$PROMPT_FILE" "$NUM_PROMPTS" <<'PY'
from pathlib import Path
import sys
src, dst, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
lines = [x.strip() for x in src.read_text().splitlines() if x.strip()]
if len(lines) != 512:
    raise RuntimeError(len(lines))
dst.write_text("\n".join(lines[i % 512] for i in range(count)) + "\n")
PY
export AFD_TOTAL_NODES=$NODES RUN_VLLM_ALIGNMENT=0
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/afd/flashinfer
export TVM_FFI_CACHE_DIR=$ROOT/cache/afd/tvm_ffi
export EP_JIT_CACHE_DIR=$ROOT/cache/afd/deepep_jit
export DG_JIT_CACHE_DIR=$ROOT/cache/afd/deepgemm_jit
export N2M_M2N_GIN_BUILD_DIR=$ROOT/cache/afd/gin_comm
export MINISGL_DEEPEP_BUILD_DIR=$ROOT/cache/afd/deepep_moe
export MINISGL_DEEPGEMM_BUILD_DIR=$ROOT/cache/afd/deepgemm
EXPERIMENT=$RUN_DIR/experiment
export RUN_DIR=$EXPERIMENT
mkdir -p "$RUN_DIR"
sha256sum "$SOURCE_REPO/$PRESET" "$PROMPT_BASE_FILE" "$JOB_SCRIPT" \
    > "$RUN_DIR/inputs.sha256"
cd "$SOURCE_REPO"
bash "$PRESET" > "$RUN_DIR/driver.stdout" 2> "$RUN_DIR/driver.stderr"
[[ -s "$RUN_DIR/sample.json" && -f "$RUN_DIR/ray_logs/coordinator_nvtx_cpu.json" ]]

RUN_DIR=$RUN_DIR NUM_PROMPTS=$NUM_PROMPTS BATCH=$BATCH \
ATTENTION_WORKERS=$ATTENTION_WORKERS NODES=$NODES REFERENCE_TPS=$REFERENCE_TPS \
"$PYTHON" - <<'PY'
import json, os, re, statistics
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
count, batch = int(os.environ["NUM_PROMPTS"]), int(os.environ["BATCH"])
workers, nodes = int(os.environ["ATTENTION_WORKERS"]), int(os.environ["NODES"])
reference = float(os.environ["REFERENCE_TPS"])
samples = json.loads((run / "sample.json").read_text()).get("samples", [])
lengths = [len(x.get("generated_token_ids", [])) for x in samples]
if len(samples) != count or set(lengths) != {16}:
    raise RuntimeError((len(samples), sorted(set(lengths))))
logs = sorted((run / "ray_logs").glob("attention_dp*_rank*.log"))
if len(logs) != workers:
    raise RuntimeError((len(logs), workers))
pattern = re.compile(r"afd_ag_decode_graph:replay step_id=(\d+) bs=(\d+) num_mb=(\d+)")
worker_steps = []
for path in logs:
    steps = sorted(int(s) for s, b, mb in pattern.findall(path.read_text(errors="replace"))
                   if int(b) == batch and int(mb) == 2)
    worker_steps.append(steps)
if len({tuple(x) for x in worker_steps}) != 1:
    raise RuntimeError("attention worker decode windows differ")
all_steps = worker_steps[0]
windows = []
for step in all_steps:
    if not windows or step != windows[-1][-1] + 1:
        windows.append([step])
    else:
        windows[-1].append(step)
if not windows or any(len(window) != 15 for window in windows):
    raise RuntimeError(("invalid full-bucket decode windows", windows))
# Reused MiniMax prompts form multiple length cohorts. Select the final complete
# full-bucket wave deterministically; never select a window by its timing.
steps = windows[-1]
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
    "samples": count, "tokens_per_sample": 16, "batch_per_attention_gpu": batch,
    "attention_workers": workers, "total_gpus": nodes * 4,
    "full_bucket_decode_windows": windows, "decode_step_ids": steps,
    "selected_window_policy": "final complete 15-step full-bucket decode wave",
    "completion_interval_ms": intervals,
    "mean_interval_ms": mean_ms, "median_interval_ms": median_ms,
    "median_tokens_per_second_per_gpu": tps,
    "reference_tokens_per_second_per_gpu": reference,
    "reference_delta_percent": (tps / reference - 1) * 100,
    "measurement": "median of all 14 consecutive coordinator completion intervals in the deterministic final full-bucket wave; every wave proven by every attention-worker b/mb replay log",
}
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
    [[ $(wc -l < "$EXPERIMENT/../gpu-snapshots/final-rank-$rank.csv") -eq 1 ]]
done
printf 'FASTAFD_AFD_SUCCESS result=%s\n' "$EXPERIMENT/afd-result.json" | tee "$EXPERIMENT/SUCCESS"
