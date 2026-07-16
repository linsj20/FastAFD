# FastAFD performance reproduction

## Status

Goal achieved on OCI HSG. Qwen3-235B and MiniMax-M2.5 at both 8K and 16K now
have DP=EP4/8/16/32/64 vLLM tables. Every EP8+ point uses the exact observed
post-graph full-length KV-cache ceiling and captures a corresponding full CUDA
graph from the sparse bucket set. Across all four workloads, EP16 is the best
device-throughput topology.

| Workload | AFD comparison | EP4 CUDA | Best wide EP | Best wide CUDA | AFD vs. best wide |
|---|---:|---:|---:|---:|---:|
| Qwen3 8K | 2522.64 reproduced | 1779.83 | 16 | 2331.65 | +8.19% |
| Qwen3 16K | 1395.78 reproduced | 935.76 | 16 | 1204.08 | +15.92% |
| MiniMax 8K | 2254.44 reproduced | 1512.57 | 16 | 2218.89 | +1.60% |
| MiniMax 16K | 1006 README | 735.33 | 16 | 1118.73 | **-10.08%** |

Positive final-column values mean AFD is faster. MiniMax 16K is the exception:
the README AFD point is 10.08% below the measured EP16 vLLM device throughput;
no local MiniMax 16K AFD reproduction was run.

## Reproduction entry points

Run on the `oci-hsg` login node:

```bash
codex_scripts/reproduce_oci_hsg/run_afd.sh [qwen3|minimax] [8k|16k]
codex_scripts/reproduce_oci_hsg/run_vllm.sh [qwen3|minimax] [8k|16k] [4|8|16|32|64]
```

The launchers self-submit to `batch+short` with a one-hour limit, pin their
runtime/model/prompt presets, validate clean GPU entry/exit, and write results
beneath `~/scratch/fastafd_reproduce`. AFD rejects any concurrent `fastafd:`
job; vLLM rejects a duplicate model/context/EP job. `FASTAFD_MODEL_PATH` may
override the model snapshot. `FASTAFD_EXCLUDE_NODE` is an infrastructure-only
escape hatch for a bad Slurm node.

`run_vllm.sh` defaults to Qwen3 8K EP4, so the original no-argument and
two-argument forms are unchanged. Its optional third argument selects EP4, 8,
16, 32, or 64. EP4 retains the published batch; every EP8+ preset uses the
exact observed full-length KV-cache ceiling and a sparse CUDA graph capture set
that includes the selected batch.

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
| MiniMax 16K vLLM | 5388859 | mean CUDA-kernel time, 4 GPUs | 735.33 | 745 | -1.299% |

Units are generated tokens/second/GPU. Reproduced AFD/vLLM device-throughput
speedups are `1.4916x` for Qwen3 16K and `1.4905x` for MiniMax 8K.

## Qwen3 8K wide-EP baseline sweep

The wide-EP sweep keeps the pinned Qwen3 8K workload and vLLM measurement
contract, uses one DP lane per GPU, and raises the global request count to
`world_size * batch_per_lane`. Every EP8+ batch is the exact full 8,256-token
KV-cache ceiling observed after model load and CUDA graph allocation.

| DP=EP | Job | Nodes / GPUs | Batch/lane (max) | Requests | CUDA tok/s/GPU | vs. EP4 | Wall mean tok/s/GPU | AFD lead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5202507 | 1 / 4 | 64 (69) | 256 | 1779.83 | reference | 1651.58 | 41.73% |
| 8 | 5369203 | 2 / 8 | 88 (88) | 704 | 2271.76 | +27.64% | 1863.75 | 11.04% |
| 16 | 5370079 | 4 / 16 | 96 (96) | 1,536 | **2331.65** | **+31.00%** | 1771.55 | **8.19%** |
| 32 | 5370692 | 8 / 32 | 101 (101) | 3,232 | 2304.92 | +29.50% | 1563.07 | 9.45% |
| 64 | 5370693 | 16 / 64 | 103 (103) | 6,592 | 1811.48 | +1.78% | 980.33 | 39.26% |

The comparison AFD result is job 5184791 at 2522.64 tok/s/GPU. EP16 is the
best wide-vLLM device-throughput point: EP32 is 1.15% slower than EP16 and
EP64 is 22.31% slower. Consequently, wide EP narrows the AFD lead from 41.73%
at the published EP4 baseline to 8.19% at EP16, but does not overtake AFD.
Synchronized wall throughput deteriorates with node count; EP64 includes two
cross-rank tail steps at 154.02 and 268.84 ms, so its wall median is separately
preserved at 1162.94 tok/s/GPU.

Measured KV capacities for EP8/16/32/64 were respectively 726,960, 799,920,
837,056, and 855,632 tokens/GPU. After placing 88/96/101/103 full sequences,
only 432/7,344/3,200/5,264 tokens remained, proving that another full sequence
would not fit. Model memory fell from 35.58 GiB/GPU at EP8 to 22.26, 15.60,
and 12.27 GiB/GPU, but the batch ceiling saturates quickly because replicated
non-weight allocations remain.

CUDA graph capture is sparse rather than a dense `65..batch` range. Each run
requests `{1,2,4,8,16,32,40,64,80,128,160,256,batch}`; the exact resident batch
88/96/101/103 is therefore captured while graph-pool memory stays at
0.51--0.52 GiB. Logs prove 13 mixed prefill/decode captures and 10 full-decode
captures per GPU. Multi-node DeepEP uses its low-latency NVSHMEM path with the
generic IPC buffer disabled (`VLLM_DEEPEP_BUFFER_SIZE_MB=0`) and GB200 MNNVL
enabled; the generic buffer assumes eight same-host GPUs and cannot open
cross-host CUDA IPC handles on OCI HSG's four-GPU nodes.

Result evidence:

- EP8: `~/scratch/fastafd_reproduce/vllm_qwen3_8192_dp8_ep8_b88_20260715_181958/baseline-result.json`, SHA-256 `abbcd5735e3759de0333df760acaec340d4d440c77a0e8e58d1112ea7a926ff1`.
- EP16: `~/scratch/fastafd_reproduce/vllm_qwen3_8192_dp16_ep16_b96_20260715_183436/baseline-result.json`, SHA-256 `f9c346338824cb5b0fe429821f0f77338b5d50b1fe468dca232a3fbd6c07c8bf`.
- EP32: `~/scratch/fastafd_reproduce/vllm_qwen3_8192_dp32_ep32_b101_20260715_184532/baseline-result.json`, SHA-256 `0144f921b7e2b91697d5fae6a228777590b096a356bb707d0b8ee601c995ca05`.
- EP64: `~/scratch/fastafd_reproduce/vllm_qwen3_8192_dp64_ep64_b103_20260715_184532/baseline-result.json`, SHA-256 `3194c0bb5d9ada2454f5f21c90fe807b4b46adfa379f31040e0b630a76a3736d`.

EP8/16/32 completed normally with 8/16/32 rank artifacts, 2/4/8 Nsight
reports, matching SQLite exports, and clean initial/final GPU snapshots. EP64
completed all 64 ranks and all 16 captures, but its original post-processing
shell used a `node-1*` glob that also selected node 10--15 reports; `nsys stats`
failed after model exit and therefore omitted one of 16 final snapshots.
Recovery job 5371830 exported only the missing node-1 SQLite and validated all
64 rank artifacts, 16 initial clean snapshots, 15 available clean final
snapshots, and 16 reports before aggregating. The model was not rerun. The
launcher now addresses exact report/SQLite paths, preventing recurrence.

Excluded wide-EP attempts contributed no measurements: jobs 5362448/5362449
were canceled when the dense graph-size range was rejected in favor of sparse
capture; 5362531/5362532 never launched their batch shells; 5368432/5368433
exposed the invalid cross-host generic DeepEP IPC path; 5369204 exposed a
static rendezvous-port collision; and diagnostic 5369635 proved that EP16 b97
exceeded the observed exact b96 capacity. The final launcher uses MNNVL, a
job-specific rendezvous port, exact capacity guards, and sparse graphs.

## Qwen3 16K wide-EP baseline sweep

| DP=EP | Job | Nodes / GPUs | Batch/lane (max) | Requests | CUDA tok/s/GPU | vs. EP4 | Wall mean tok/s/GPU | AFD lead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5209328 | 1 / 4 | 32 (34) | 128 | 935.76 | reference | 775.26 | 49.16% |
| 8 | 5385780 | 2 / 8 | 44 (44) | 352 | 1094.29 | +16.94% | 914.44 | 27.55% |
| 16 | 5385781 | 4 / 16 | 48 (48) | 768 | **1204.08** | **+28.67%** | 918.28 | **15.92%** |
| 32 | 5386396 | 8 / 32 | 50 (50) | 1,600 | 1164.91 | +24.49% | 796.73 | 19.82% |
| 64 | 5386397 | 16 / 64 | 52 (52) | 3,328 | 1126.48 | +20.38% | 585.50 | 23.91% |

The AFD comparison is reproduced job 5206679 at 1395.78 tok/s/GPU. Wide EP
narrows its advantage from 49.16% at EP4 to 15.92% at EP16. EP8/16/32/64 KV
capacities are 726,960/799,920/837,056/855,632 tokens/GPU; the exact
b44/b48/b50/b52 graphs leave only 3,248/10,416/14,656/336 tokens. Each run
records 13 piecewise graphs and eight full graphs, with the exact batch as the
largest full graph.

Result directories and SHA-256 values:

- EP8: `~/scratch/fastafd_reproduce/vllm_qwen3_16384_dp8_ep8_b44_20260715_223351`, `94ac72227691114c18715587253d0a310f51689dbf0bebaea59b7c74829b7092`.
- EP16: `~/scratch/fastafd_reproduce/vllm_qwen3_16384_dp16_ep16_b48_20260715_223351`, `4f77b137381d1e6d5fd4ae9dd0c51d3108de9a6450950502354db5c115ad5f84`.
- EP32: `~/scratch/fastafd_reproduce/vllm_qwen3_16384_dp32_ep32_b50_20260715_224412`, `43351caeb7e9931608f5fe79fdf023df42c35a64d5cdbf2c92297c39c0484d49`.
- EP64: `~/scratch/fastafd_reproduce/vllm_qwen3_16384_dp64_ep64_b52_20260715_224412`, `2941e6a6ee371744531d0166b208d9f284cb5240aff917e7de9e25163427f9b9`.

## MiniMax 8K wide-EP baseline sweep

| DP=EP | Job | Nodes / GPUs | Batch/lane (max) | Requests | CUDA tok/s/GPU | vs. EP4 | Wall mean tok/s/GPU | AFD lead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5215344 | 1 / 4 | 48 (54) | 192 | 1512.57 | reference | 1274.87 | 49.05% |
| 8 | 5387563 | 2 / 8 | 68 (68) | 544 | 1991.72 | +31.68% | 1629.81 | 13.19% |
| 16 | 5387564 | 4 / 16 | 74 (74) | 1,184 | **2218.89** | **+46.70%** | 1593.80 | **1.60%** |
| 32 | 5388167 | 8 / 32 | 78 (78) | 2,496 | 2162.36 | +42.96% | 1352.48 | 4.26% |
| 64 | 5388168 | 16 / 64 | 79 (79) | 5,056 | 2151.51 | +42.24% | 1032.53 | 4.78% |

The AFD comparison is reproduced job 5216240 at 2254.44 tok/s/GPU. EP16
closes the AFD advantage to 1.60%, but EP32/64 do not overtake it. EP8/16/32/64
KV capacities are 567,616/622,320/650,192/664,112 tokens/GPU, proving exact
b68/b74/b78/b79. Each run records 13 piecewise and nine full graphs.

Result directories and SHA-256 values:

- EP8: `~/scratch/fastafd_reproduce/vllm_minimax_8192_dp8_ep8_b68_20260715_230250`, `cf6a61f3a23530603d32ef13156c8902ac634aa20884cf33cf38d41cd9cffb14`.
- EP16: `~/scratch/fastafd_reproduce/vllm_minimax_8192_dp16_ep16_b74_20260715_230250`, `43b5f0950336f5f7aad175ba891fccbb77e9530cd949dc81aeb53085d94d30f2`.
- EP32: `~/scratch/fastafd_reproduce/vllm_minimax_8192_dp32_ep32_b78_20260715_231042`, `b61637deadebcfbce3f1cef32edb82f5705076412686dca27a057e602c2ecf8d`.
- EP64: `~/scratch/fastafd_reproduce/vllm_minimax_8192_dp64_ep64_b79_20260715_231043`, `2c71c94ea1f9f1ad9c8ef146667889288da033c206788a3f279939cb46de2d35`.

Capacity-probe jobs 5386994/5386995 used conservative b66/b72 estimates.
The exact guard observed b68/b74 and stopped before benchmarking; only the
corrected jobs above contribute measurements.

## MiniMax 16K wide-EP baseline sweep

| DP=EP | Job | Nodes / GPUs | Batch/lane (max) | Requests | CUDA tok/s/GPU | vs. EP4 | Wall mean tok/s/GPU | AFD lead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5388859 | 1 / 4 | 24 (27) | 96 | 735.33 | reference | 630.32 | 36.81% |
| 8 | 5389372 | 2 / 8 | 34 (34) | 272 | 1027.35 | +39.71% | 844.99 | -2.08% |
| 16 | 5389373 | 4 / 16 | 37 (37) | 592 | **1118.73** | **+52.14%** | 623.45 | **-10.08%** |
| 32 | 5389906 | 8 / 32 | 39 (39) | 1,248 | 1051.05 | +42.94% | 625.20 | -4.29% |
| 64 | 5390375 | 16 / 64 | 39 (39) | 2,496 | 1033.52 | +40.55% | 524.63 | -2.66% |

The AFD comparison is the README value 1006 tok/s/GPU; it has not been locally
reproduced. Negative AFD-lead values mean wide vLLM is faster: EP8 is 2.12%,
EP16 11.21%, EP32 4.48%, and EP64 2.74% above README AFD. EP8/16/32/64 KV
capacities are 567,616/622,320/650,192/664,112 tokens/GPU, proving exact
b34/b37/b39/b39. Each run records 13 piecewise and seven full graphs.

Result directories and SHA-256 values:

- EP4: `~/scratch/fastafd_reproduce/vllm_minimax_16384_dp4_ep4_b24_20260715_232022`, `fb0912f97e69d6f43c223117537eaae3f26e910c1586aaf20da6245754fdb624`.
- EP8: `~/scratch/fastafd_reproduce/vllm_minimax_16384_dp8_ep8_b34_20260715_232829`, `d238508b14a71599170208a0105aae925aa3baedb3e2e6ee0fa5ccac9688f8dc`.
- EP16: `~/scratch/fastafd_reproduce/vllm_minimax_16384_dp16_ep16_b37_20260715_232829`, `504de92f77115462f60ae3ca3015f150689bd85b868edd691068e8b51bf8ac45`.
- EP32: `~/scratch/fastafd_reproduce/vllm_minimax_16384_dp32_ep32_b39_20260715_233626`, `a480c656b77746172b55231886444779fde0cc1f6d1f972ac032c51d58ab6bbf`.
- EP64: `~/scratch/fastafd_reproduce/vllm_minimax_16384_dp64_ep64_b39_20260715_234444`, `bb44dd5be490b5de5c6773d2210f5c91911787b47c21f344f6898786a1843ca2`.

Probe job 5389907 used b40 at EP64; the exact guard observed a b39 ceiling and
stopped before benchmarking. Corrected job 5390375 is the retained result.
Across the three new wide sweeps, the matrix validator passed all 360 rank
files, 90 Nsight reports/SQLite exports, 90 clean initial node snapshots, 90
clean final node snapshots, and every exact-batch graph-capture record.

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
| `run_vllm.sh` | `44c0fa9cfe588fc413b237fd70e35b3058d683e3b564a56713beb224913bae67` |

Both shell files pass `bash -n`, and all embedded Python blocks compile in
memory. `run_vllm.sh` has byte-identical local and remote control copies.
