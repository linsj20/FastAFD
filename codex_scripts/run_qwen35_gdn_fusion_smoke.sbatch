#!/usr/bin/env bash
# Focused one-tray CUDA validation/benchmark for Qwen3.5 GDN fusion work.
# Required: GDN_SOURCE_REPO GDN_EXPECTED_HEAD GDN_IMAGE GDN_VENV GDN_JOB_SCRIPT
# Optional: GDN_ROWS (default 19), GDN_BENCH_ITERS (default 500)
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
#SBATCH --no-requeue
#SBATCH --job-name=fastafd:q35-gdn-smoke

set -euo pipefail

: "${GDN_SOURCE_REPO:?}" "${GDN_EXPECTED_HEAD:?}" "${GDN_IMAGE:?}" \
  "${GDN_VENV:?}" "${GDN_JOB_SCRIPT:?}"
GDN_ROWS=${GDN_ROWS:-19}
GDN_BENCH_ITERS=${GDN_BENCH_ITERS:-500}
[[ "$GDN_ROWS" =~ ^[1-9][0-9]*$ ]]
[[ "$GDN_BENCH_ITERS" =~ ^[1-9][0-9]*$ ]]

if [[ ${FASTAFD_GDN_SMOKE_IN_CONTAINER:-0} != 1 ]]; then
    exec srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:4 \
      --kill-on-bad-exit=1 --container-image="$GDN_IMAGE" \
      --container-mount-home --container-mounts=/lustre:/lustre \
      --no-container-remap-root \
      env FASTAFD_GDN_SMOKE_IN_CONTAINER=1 \
          GDN_SOURCE_REPO="$GDN_SOURCE_REPO" \
          GDN_EXPECTED_HEAD="$GDN_EXPECTED_HEAD" GDN_IMAGE="$GDN_IMAGE" \
          GDN_VENV="$GDN_VENV" GDN_JOB_SCRIPT="$GDN_JOB_SCRIPT" \
          GDN_ROWS="$GDN_ROWS" GDN_BENCH_ITERS="$GDN_BENCH_ITERS" \
          bash "$GDN_JOB_SCRIPT"
fi

[[ $(git -C "$GDN_SOURCE_REPO" rev-parse HEAD) == "$GDN_EXPECTED_HEAD" ]]
[[ -z $(git -C "$GDN_SOURCE_REPO" status --porcelain) ]]
[[ -x "$GDN_VENV/bin/python" ]]
export PATH="$GDN_VENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$GDN_SOURCE_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export VIRTUAL_ENV="$GDN_VENV" CONDA_PREFIX="$GDN_VENV"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas

exec "$GDN_VENV/bin/python" \
  "$GDN_SOURCE_REPO/codex_scripts/validate_qwen35_gdn_fusion.py" \
  --rows "$GDN_ROWS" --bench-iters "$GDN_BENCH_ITERS"
