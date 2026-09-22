#!/usr/bin/env bash
# One EP group per short allocation, all cases sequential in persistent containers.
# Modes: prepare (CPU), gpu, worker (inside container), analyze (CPU).
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --no-requeue
#SBATCH --job-name=trt-original-megamoe
set -euo pipefail
mode=${1:?mode} root=${2:?artifact root}
script=$root/control/trtllm_megamoe_sweep.sbatch
workload=$root/control/trtllm_megamoe_sweep.py
user_root=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel
image=$user_root/trtllm_qwen3_235b_megamoe/images/pytorch-26.05-py3-arm64-trtllm-fa778397186c09dd57bcaa5020e66270768b675b.sqsh
export TRT_SWEEP_ROOT=$root
export TRT_IKET_INCLUDE=$user_root/FastAFD/source/iket/include
export FASTAFD_IKET_PYTHON_DEPS=$user_root/FastAFD/source/iket-smodel
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
if [[ $mode == prepare ]]; then
    exec srun --container-image="$image" --container-mount-home --container-mounts=/lustre:/lustre \
        --no-container-remap-root bash "$script" prepare-container "$root"
fi
if [[ $mode == prepare-container ]]; then
    export TMPDIR=$root/preparation/tmp
    mkdir -p "$TMPDIR"
    # Verify the kernel is the original bundled implementation, not the local port.
    cmp "$root/source/sm100_fp8_fp4_mega_moe.cuh" \
        /usr/local/lib/python3.12/dist-packages/tensorrt_llm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
    /usr/bin/python3 -c 'import ast; from pathlib import Path; p=Path("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/deep_gemm/mega/__init__.py"); names={n.name for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}; assert {"fp8_fp4_mega_moe","get_symm_buffer_for_mega_moe","transform_weights_for_mega_moe"} <= names; print("ORIGINAL_DEEPGEMM_SOURCE_API_OK")'
    /usr/bin/python3 -c 'import os,sys; sys.path.insert(0,os.environ["FASTAFD_IKET_PYTHON_DEPS"]); import iket, smodel; print("IKET_IMPORT_OK")'
    command -v nsys
    command -v run-iket
    touch "$root/preparation/complete"
    exit
fi
ep=${3:?EP}
if [[ $mode == analyze ]]; then
    export TMPDIR=$root/ep$ep/analysis-tmp
    mkdir -p "$TMPDIR"
    exec srun --container-image="$image" --container-mount-home --container-mounts=/lustre:/lustre \
        --no-container-remap-root /usr/bin/python3 "$workload" analyze "$root" "$ep"
fi
if [[ $mode == gpu ]]; then
    [[ -f $root/preparation/complete ]]
    mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t blocks < <(printf '%s\n' "${nodes[@]}" | sed -E 's/-T[0-9]+$//' | sort -u)
    (( ${#blocks[@]} == 1 )) || { echo 'allocation crosses NVL72 racks' >&2; exit 2; }
    printf '%s\n' "${nodes[@]}" > "$root/ep$ep/nodes.txt"
    export MASTER_ADDR=${nodes[0]} MASTER_PORT=29671
    exec srun --ntasks="$ep" --ntasks-per-node="$((ep < 4 ? ep : 4))" --kill-on-bad-exit=1 \
        --container-image="$image" --container-mount-home --container-mounts=/lustre:/lustre \
        --no-container-remap-root env -u CUDA_VISIBLE_DEVICES bash "$script" worker "$root" "$ep"
fi
[[ $mode == worker ]]
rank=${SLURM_PROCID:?}
export TMPDIR=$root/cache/ep$ep/tmp-rank$rank
mkdir -p "$TMPDIR"
export FASTAFD_IKET_EVENT_REGISTER_LIMITS='{"MegaDispatchStart":48,"MegaDispatchEnd":48,"MegaComputeStart":40,"MegaComputeEnd":208,"MegaCombineStart":208,"MegaCombineEnd":208}'
while read -r case_id; do
    case_root=$root/cases/$case_id
    export DG_JIT_CACHE_DIR=$root/cache/ep$ep/normal-rank$rank
    unset DG_JIT_NVCC_COMPILER
    if [[ $rank == 0 ]]; then
        echo "CASE_START $case_id job=$SLURM_JOB_ID"
        nsys profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
            --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
            --output="$case_root/nsys/megamoe" \
            /usr/bin/python3 "$workload" workload "$root" "$case_id" normal </dev/null
    else
        /usr/bin/python3 "$workload" workload "$root" "$case_id" normal </dev/null
    fi
    export DG_JIT_CACHE_DIR=$root/cache/ep$ep/marked-rank$rank
    export DG_JIT_NVCC_COMPILER=$root/control/trtllm_megamoe_iket_nvcc.sh
    if [[ $rank == 0 ]]; then
        /usr/bin/python3 -c '
import os,runpy,shutil,sys
sys.path.insert(0,os.environ["FASTAFD_IKET_PYTHON_DEPS"])
runpy.run_path(os.environ["TRT_SWEEP_ROOT"]+"/control/iket_event_register_limits.py",run_name="__main__")
runpy.run_path(shutil.which("run-iket"),run_name="__main__")
' --output-dir "$case_root/iket" --context-buffer-size 268435456 --log-level info \
            profile --max-ts-cnt-per-warp 1024 --postprocess json -- \
            /usr/bin/python3 "$workload" workload "$root" "$case_id" profile </dev/null
        echo "CASE_CAPTURE_COMPLETE $case_id"
    else
        # IKET source discovery and capture are separate workload processes.
        /usr/bin/python3 "$workload" workload "$root" "$case_id" profile </dev/null
        /usr/bin/python3 "$workload" workload "$root" "$case_id" profile </dev/null
    fi
done < "$root/ep$ep/cases.txt"
