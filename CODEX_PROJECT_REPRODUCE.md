# FastAFD performance reproduction

## Status

Goal achieved on OCI HSG. Qwen3-235B 8K/16K and MiniMax-M2.5 8K FastAFD and
vLLM results reproduce the README performance points. The workspace retains
exactly two experiment launchers.

## Reproduction entry points

Run on the `oci-hsg` login node:

```bash
codex_scripts/reproduce_oci_hsg/run_afd.sh [qwen3|minimax] [8k|16k]
codex_scripts/reproduce_oci_hsg/run_vllm.sh [qwen3|minimax] [8k|16k]
```

Both scripts self-submit to `batch+short` with a one-hour limit, reject a
concurrent `fastafd:` job, pin the source/model/prompt presets, validate clean
GPU entry/exit, and write results beneath
`~/scratch/fastafd_reproduce`. `FASTAFD_MODEL_PATH` may override the model
snapshot. `FASTAFD_EXCLUDE_NODE` is an infrastructure-only escape hatch for a
bad Slurm node.

| Model/context | AFD trays (A:F) | AFD batch/attention GPU | vLLM batch/GPU | README AFD | README vLLM |
|---|---:|---:|---:|---:|---:|
| Qwen3 8K | 8 (7:1) | 96 | 64 | 2518 | 1781 |
| Qwen3 16K | 12 (11:1) | 48 | 32 | 1377 | 954 |
| MiniMax 8K | 18 (17:1) | 72 | 48 | 2198 | 1516 |
| MiniMax 16K | 18 (17:1) | 36 | 24 | 1006 | 745 |

The one-tray baseline retains the published per-GPU batch and uses one eighth
of an eight-tray duplicated workload. Baseline memory utilization defaults to
and is guarded at `0.94` so the full resident batch fits.

## Pinned provenance

- FastAFD commit: `3c7161949310b6d59d6b4cf9bf997a4935c8113b`.
- Qwen model: `Qwen/Qwen3-235B-A22B-FP8` revision
  `39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35`.
- MiniMax model: `MiniMaxAI/MiniMax-M2.5` revision
  `f710177d938eff80b684d42c5aa84b382612f21f`.
- Prompt SHA-256: 8K `26482bc14fe61372c30eed8731fae1103fe477cdf03c70e8a808c3723ede5fdb`;
  16K `918f24cde353525d62d7a0493912719ca97eafd9201157067d9ed29a93d29fca`.
- Runtime: Python 3.12, CUDA 13.0, vLLM `0.19.0+cu130`, Torch
  `2.10.0+cu130`, Triton `3.6.0`, FlashInfer `0.6.6`, NCCL `2.30.4`.
- External vLLM kernels: DeepEP
  `73b6ea4a439ba03a695563f9fd242c8e4b02b37c` and DeepGEMM
  `891d57b4db1071624b5c8fa0d1e51cb317fa709f`.
- Official CUDA 13 ARM64 wheels were used where available; pinned DeepEP and
  DeepGEMM source builds were required because compatible wheels were absent.
- Remote source: `~/scratch/github/FastAFD`. Runtime overlay:
  `~/scratch/fastafd_reproduce/envs/minisgl-3c716194-cuda130-vllm-ep`.

## Reproduced results

| Workload | Job | Measurement | Reproduced | README | Delta |
|---|---:|---|---:|---:|---:|
| Qwen3 8K AFD | 5184791 | median coordinator interval, 28 workers | 2522.64 | 2518 | +0.184% |
| Qwen3 8K vLLM | 5202507 | mean CUDA-kernel time, 4 GPUs | 1779.83 | 1781 | -0.066% |
| Qwen3 16K AFD | 5206679 | median coordinator interval, 44 workers | 1395.78 | 1377 | +1.364% |
| Qwen3 16K vLLM | 5209328 | mean CUDA-kernel time, 4 GPUs | 935.76 | 954 | -1.911% |
| MiniMax 8K AFD | 5216240 | final median coordinator wave, 68 workers | 2254.44 | 2198 | +2.568% |
| MiniMax 8K vLLM | 5215344 | mean CUDA-kernel time, 4 GPUs | 1512.57 | 1516 | -0.226% |

Units are generated tokens/second/GPU. Reproduced AFD/vLLM device-throughput
speedups are `1.4916x` for Qwen3 16K and `1.4905x` for MiniMax 8K.

Qwen3-16K AFD used the repo b48/mb2 preset unchanged: 12 trays, 44 attention
workers, four FFN workers, 2,112 prompts, 16 generated tokens, memory ratio
0.82, TRT-LLM attention, MegaMoE, and all published graph buckets. All outputs,
44 attention logs, four FFN logs, 51 Nsight reports, and 12 empty final GPU
snapshots passed. Result:
`~/scratch/fastafd_reproduce/afd_qwen3_16384_b48_20260714_151648/experiment/afd-result.json`
(SHA-256 `83a2033733164b7f2ffa83886bbb86813dbad360bd22047b84580719ed47de13`).

Qwen3-16K vLLM used one tray with external-launcher DP4/EP4, DeepEP,
DeepGEMM, b32/GPU, 128 total prompts, max length 16,448, prefix caching off,
the repo graph-size list, and memory 0.94. Capacity was 571,968 KV tokens/GPU
against 526,336 required. CUDA-kernel throughput was 935.76; synchronized wall
median was separately reported as 782.17. Result:
`~/scratch/fastafd_reproduce/vllm_qwen3_16384_b32_20260714_153912/baseline-result.json`
(SHA-256 `ca07e1c44cbda1840f4160f8ba2b1dcc3653f7d119e091f1e60afebe2e156791`).

MiniMax-8K AFD used the unchanged repo b72/mb2 preset: 18 trays, 68 attention
workers, four FFN workers, 4,896 prompts, and 16 generated tokens. All outputs,
worker replay logs, graph captures, and 75 Nsight reports passed. The workload
completed in job 5216240; its `1:0` exit was solely the old task-owned validator
assuming one decode wave. The corrected validator proved the two unanimous
15-step waves and deterministically selected the final wave, steps 1143--1157.
Result:
`~/scratch/fastafd_reproduce/afd_minimax_8192_b72_20260714_164032/experiment/afd-result.postvalidated.json`
(SHA-256 `9a96cee482ca964613b9047b19db2149e5d37986b72881785b9d19b9b2d0a9b3`).

MiniMax-8K vLLM used one tray with DP4/EP4, b48/GPU, 192 total prompts,
max length 8,320, memory 0.94, and a full CUDA graph at b48. Capacity was
451,328 KV tokens/GPU against 399,360 required. Result:
`~/scratch/fastafd_reproduce/vllm_minimax_8192_b48_20260714_163239/baseline-result.json`
(SHA-256 `e01a4ceb4d85e14dbdd3337cefb9b9511e3f1bc9fd8999c09e1f446ce74c20c5`).

## Measurement contract

AFD validates every generated output and requires every attention-worker log
to identify identical contiguous 15-step batch/microbatch replay waves.
Throughput uses all 14 coordinator completion intervals from the deterministic
final complete wave; no interval or wave is selected by timing.

The public repo has no vLLM throughput launcher, so the baseline measurement is
task-owned and labeled as such. It warms until every request is resident,
measures 15 synchronized full-batch decode steps, and reports both the maximum
host-wall latency across four ranks and the sum of CUDA kernel durations per
GPU. The README match uses the device-kernel metric; wall timing remains visible
instead of being conflated with kernel throughput.

Excluded attempts were fail-fast and informed the final scripts: two nodes had
infrastructure-only Nsight/Slurm faults; MiniMax required remote-code loading,
the repo's reused variable-length Qwen prompt contract, and literal b48 CUDA
graph capture. None contributes to the reported measurements.

## Final controls

| File | SHA-256 |
|---|---|
| `run_afd.sh` | `026af99c3823a2e91916a8b983b83028e2d7bc3a526b0dff7c44a1587d45b348` |
| `run_vllm.sh` | `e697a423ce8008adc95b3ae4daff0cde65974b3277bb1a4f79d493fca3561ff5` |

Both shell files pass `bash -n`; all five embedded Python blocks compile.
