#!/usr/bin/env bash
# Score completed AFD samples with the existing vLLM alignment scorer.
# Required: FASTAFD_SOURCE_REPO FASTAFD_IMAGE FASTAFD_VENV MODEL_PATH
#           SAMPLE_JSON OUTPUT_ROOT. Run after releasing the AFD allocation.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=fastafd-sample-alignment

set -euo pipefail
: "${FASTAFD_SOURCE_REPO:?}" "${FASTAFD_IMAGE:?}" "${FASTAFD_VENV:?}" \
  "${MODEL_PATH:?}" "${SAMPLE_JSON:?}" "${OUTPUT_ROOT:?}"
if [[ ${FASTAFD_SCORE_INSIDE_CONTAINER:-0} != 1 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    script=${FASTAFD_SCORE_SCRIPT:-$FASTAFD_SOURCE_REPO/codex_scripts/score_afd_samples.sbatch}
    exec srun --ntasks=1 --kill-on-bad-exit=1 \
        --container-image="$FASTAFD_IMAGE" --container-mount-home \
        --container-mounts=/lustre:/lustre --no-container-remap-root \
        env FASTAFD_SCORE_INSIDE_CONTAINER=1 bash "$script"
fi

export PATH="$FASTAFD_VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CONDA_DEFAULT_ENV=fastafd-venv
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
# vLLM places Unix-domain sockets here; sockaddr_un limits paths to 107 bytes.
# Keep transient IPC on local shared memory and durable outputs under OUTPUT_ROOT.
export TMPDIR="/dev/shm/afd-score-${SLURM_JOB_ID:?}"
export HF_HOME="$OUTPUT_ROOT/cache/huggingface"
export VLLM_CACHE_ROOT="$OUTPUT_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$OUTPUT_ROOT/cache/torch_extensions"
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
port=${VLLM_PORT:-29731}
bash "$FASTAFD_SOURCE_REPO/scripts/serve/vllm_server.sh" \
    --env fastafd-venv --model "$MODEL_PATH" --gpus 0,1,2,3 --tp-size 4 \
    --port "$port" --host 127.0.0.1 --max-model-len "${VLLM_MAX_MODEL_LEN:-131136}" \
    --extra-args "${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}" \
    > "$OUTPUT_ROOT/vllm.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 900))
until curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || { tail -40 "$OUTPUT_ROOT/vllm.log"; exit 2; }
    (( SECONDS < deadline )) || { echo 'vLLM readiness timed out' >&2; exit 2; }
    sleep 2
done
python "$FASTAFD_SOURCE_REPO/scripts/validate/compare_minisgl_vllm.py" score \
    --sample-json "$SAMPLE_JSON" --model "$MODEL_PATH" \
    --vllm-url "http://127.0.0.1:$port" --prompt-logprobs 0 \
    --score-concurrency "${VLLM_SCORE_CONCURRENCY:-16}" --timeout 1800 \
    --output-json "$OUTPUT_ROOT/alignment.json" > "$OUTPUT_ROOT/score.log" 2>&1
python - "$OUTPUT_ROOT/alignment.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))["summary"]
print(json.dumps(summary, indent=2))
if (summary.get("overall_top10_hit_rate") or 0.0) < 1.0:
    raise SystemExit("official alignment failed: top-10 hit rate below 1.0")
PY
