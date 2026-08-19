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
scripts/experiments/afd/oci_hsg/run_afd_reproduce.sh [qwen3|minimax] [8k|16k]
scripts/experiments/afd/oci_hsg/run_vllm.sh [qwen3|minimax] [8k|16k] [4|8|16|32|64]
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

The public repo can launch vLLM and use it as a correctness baseline:
`scripts/serve/vllm_server.sh` wraps `vllm serve`, while
`scripts/validate/fastafd_vllm_alignment.sh` and the optional sharded scorer
sample deterministic FastAFD outputs and rescore the same tokens with vLLM
`prompt_logprobs`. That is an alignment workflow, not the throughput collector
behind the README figure. No checked-in script reproduces the README vLLM TPS
measurement. Therefore this task's throughput baseline is task-owned and
labeled as such. It warms until every request is resident, measures 15
synchronized full-batch decode steps, and reports both the maximum host-wall
latency across four ranks and the sum of CUDA kernel durations per GPU. The
README match uses the device-kernel metric; wall timing remains visible instead
of being conflated with kernel throughput. The generic
`python/minisgl/benchmark/perf.py::perf_cuda` CUDA-event helper times callable
microbenchmarks and is not wired into the vLLM serving/alignment path.

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

## Qwen3 ISL / batch sweep (complete)

Session objective: hold model/topology fixed and collect decode-throughput data
across ISL and resident batch for both AFD and the best measured wide-EP vLLM
topology. The fixed contracts are Qwen3-235B-A22B-FP8, AFD on eight nodes
(28 attention GPUs plus four FFN GPUs, MB2, memory ratio 0.82), and vLLM
DP=EP16 on four nodes (memory utilization 0.94). This isolates ISL and batch;
it does not retune AFD's attention/FFN node split at each ISL.

Sweep reporting contract (confirmed by the user on 2026-07-16): the primary
performance metrics for every point are **request decode-step latency** and the
corresponding **TPS/GPU**. For vLLM, latency is the synchronized host-wall step
(maximum across all 16 DP/EP ranks); report its complete 15-step series, median,
mean, and the TPS/GPU derived from each. For AFD, latency is the coordinator
completion interval in the deterministic final 15-step full-bucket wave; report
all 14 intervals, median, mean, and median-derived TPS/GPU across all 32
allocated GPUs. Nsight CUDA-kernel TPS/GPU remains a secondary diagnostic and
must not replace the request-level headline metric.

ISLs are 1K/2K/4K/8K/16K/32K/64K/128K. Per-attention-GPU (AFD) and per-DP-lane
(vLLM) batches follow `2^x` and `2^x + 2^(x-1)` for integer `x >= 1`. The
initial expected upper candidates, derived from the measured EP16 KV capacity
and the AFD 8K shape, are respectively 512/256/192/96/48/24/12/6; each runtime
reports or proves its actual capacity, and a mode stops only when the next
sequence value is unsupported. The full candidate prefixes are:

- 1K: 2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512.
- 2K: the same prefix through 256.
- 4K: the same prefix through 192.
- 8K: the same prefix through 96.
- 16K: the same prefix through 48.
- 32K: the same prefix through 24.
- 64K: the same prefix through 12.
- 128K: 2,3,4,6.

The two retained launchers are generalized without adding another runner:

```bash
run_afd.sh qwen3 <isl> <batch-per-attention-gpu> 8
run_vllm.sh qwen3 <isl> 16 <batch-per-dp-lane>
```

Both accept all eight ISLs and explicit batches, expose a no-submit
`FASTAFD_DRY_RUN=1` contract, enforce the current active-`fastafd:` job cap, use
shape-specific job/run names, and record the post-init capacity and prompt
contract. Qwen prompts come from the pinned 8K corpus: content tokens are cut
for shorter ISLs or repeated for longer ISLs, then the Qwen chat template is
applied and the exact final server-visible ISL is validated. vLLM consumes the
resulting token IDs directly with prefix caching disabled. AFD writes exact
text round-trips plus a prompt-token hash; its required `naive` cache has no
prefix reuse. The 64K/128K points exceed Qwen's configured 40,960 trained
positions and are explicitly recorded as long-context extrapolation; vLLM uses
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` rather than hiding that distinction.

Local validation on 2026-07-16: both scripts pass `bash -n`, every embedded
Python block compiles, `git diff --check` passes, and dry-run resolution passes
for all eight expected upper points. A tokenizer-only remote check proved the
cut/repeat construction yields exactly every requested ISL from 1,024 through
131,072 tokens with no physical newlines in the generated AFD prompt. No sweep
jobs were active at inspection time. The remote source checkout contains
unrelated co-batching changes, so sweep jobs must use a separate clean worktree
at commit `3c716194` and separate control-script copies; never clean or replace
the co-batching tree.

Remote installation uses clean worktree
`~/scratch/fastafd_reproduce/source/FastAFD-3c716194` and controls under
`~/scratch/fastafd_reproduce/control`. Installed launcher SHA-256 values are
`78a84714ab78b36bb0ea1c57f0d3697c8a38c8a7746cc023fcac708a49f4045a`
(AFD) and
`6bab5df8fb82b9162c86ce1f784fc4674e06817afc90f9daf972256b503ae296`
(vLLM); remote `bash -n`, dry-run resolution, clean source checks, and
local/remote byte identity passed.

First smoke pair submitted 2026-07-16 18:04 PDT, with exactly two active jobs:

- AFD job `5438790`: Qwen3, ISL 1,024, batch 2/attention GPU, eight nodes;
  `~/scratch/fastafd_reproduce/afd_qwen3_1024_b2_n8_20260716_180459`.
- vLLM job `5438791`: Qwen3, ISL 1,024, DP=EP16, batch 2/lane, four nodes;
  `~/scratch/fastafd_reproduce/vllm_qwen3_1024_dp16_ep16_b2_20260716_180459`.

Both initially queued with one-hour `batch+short` allocations. This pair tests
the new exact-token path and AFD's smallest MB2 graph bucket before advancing
the requested sequence.

AFD smoke job `5438790` completed `0:0` in 7:36. All 56 requests returned 16
tokens, every one of 28 attention workers proved the same two complete 15-step
batch-2/MB2 graph windows, and the deterministic final window measured 15.3595
ms median coordinator interval or **113.9359 generated tok/s/GPU** across all
32 allocated GPUs. Result:
`afd_qwen3_1024_b2_n8_20260716_180459/experiment/afd-result.json`, SHA-256
`43e13da266654188cf2f7e839f2d4414d048dcf36c65c5d4184b04d1ff0122b0`.
The prompt manifest proves 56 exact 1,024-token chat inputs and aggregate token
hash `1f24959d0c254736db8404f4ae0e58d73204337856fbd376c3f947e803e75cb6`.

vLLM smoke job `5438791` initialized normally and reported 799,904 KV tokens/GPU
(capacity 735 full 1,088-token sequences), captured 12 piecewise plus two full
graphs including batch 2, then stopped producing artifacts after 18:10:22.
At 18:16--18:19 all 16 rank processes remained at roughly one CPU core each,
all 16 GPUs were resident at ~179--181 GiB but 0% utilized, all four node logs
had identical terminal timestamps, and no rank/result files existed. A
read-only overlapping scheduler step and GDB attach confirmed living rank/NCCL
threads rather than a dead Slurm shell. Leave it until its built-in distributed
timeout produces an actionable error; do not count it as a measurement.

With AFD batch 2 complete and exactly one job still active, AFD batch 3 job
`5438918` was submitted at 18:19 PDT on eight nodes. Run directory:
`~/scratch/fastafd_reproduce/afd_qwen3_1024_b3_n8_20260716_181921`.

No timeout fired; job `5438791` remained unchanged and GPU-idle, so it was
cancelled deliberately at 18:20 after 16:01 to avoid wasting four nodes. A
five-second `perf` sample attributed the apparent CPU load to NVSHMEM IBRC proxy
progress (`progress_send`/`progress_recv`) while model GPUs remained idle. The
vLLM control now prints a flushed stage marker before/after every setup and
warmup boundary and registers `SIGUSR1` faulthandler stack dumps. Updated vLLM
control SHA-256 is
`8f4a7d1f10b25afa2f12b562ca5a01a3e72d410b90ea6b9cb3716990ddb9f3f5`;
shell and embedded-Python validation pass. Instrumented replacement job
`5438972` was submitted at 18:22 PDT for the same 1K/b2/EP16 point, directory
`vllm_qwen3_1024_dp16_ep16_b2_20260716_182250`. Together with AFD b3 there are
again exactly two active jobs.

Instrumented vLLM replacement `5438972` completed `0:0` in 5:04, proving the
first attempt was a transient distributed/NVSHMEM stall rather than a prompt or
runner defect. Every rank completed exact prompt transform, two warmup steps,
and the measurement barrier. Capacity was 799,920 tokens/GPU or 735 full
1,088-token requests/lane. The primary synchronized request-level result was
**25.853 ms median decode-step latency and 77.3610 tok/s/GPU**; wall mean was
48.7387 tok/s/GPU due one 241.16 ms tail step. CUDA-kernel throughput was
136.0212 tok/s/GPU. Result SHA-256:
`67a71d04ec763320dcb869cc7b912b58c1264e65f8d7c16a44472a8c1e7af7ba`.

AFD batch-3 job `5438918` returned all 84 exact outputs and every attention
worker logged two identical 15-step graph windows, but the original task
validator exited `1:0`: odd MB2 batches pad resident batch 3 to graph batch 4.
The validator now compares replay logs against the padded batch and records
both values. Updated AFD control SHA-256 is
`282ad72041d943b4e42de62b2c91ade3bb11f25ae56bb0b5a7a0097526e07b4b`.
Corrected postvalidation deterministically selected steps 22--36 and measured
17.1447 ms median or **153.1084 tok/s/GPU**. Result:
`afd_qwen3_1024_b3_n8_20260716_181921/experiment/afd-result.postvalidated.json`,
SHA-256 `e7880736c90db75cceb92eddaf655d6d6f743cadc14ab3f8f429c7239b8ee7cf`.
The workload need not be rerun; its job exit reflects only the corrected
task-owned validator, though final GPU snapshots were skipped after that exit.

Next pair submitted at 18:30 PDT with exactly two jobs: AFD 1K/b4 job `5439031`
(`afd_qwen3_1024_b4_n8_20260716_183019`) and vLLM 1K/b3/EP16 job `5439032`
(`vllm_qwen3_1024_dp16_ep16_b3_20260716_183019`). Both use the corrected,
remote-hash-verified controls above.

AFD batch-4 job `5439031` completed `0:0` in 4:08. All 112 requests had exact
1,024-token inputs and returned 16 output tokens; all 28 attention workers
proved two complete 15-step batch-4 graph windows. The deterministic final
window measured 16.1068 ms median coordinator interval or **217.2996 generated
tok/s/GPU** across 32 allocated GPUs. Result:
`afd_qwen3_1024_b4_n8_20260716_183019/experiment/afd-result.json`, SHA-256
`740eba547c797b041d7b4e1fb5bb976bee3fd353c423683a5acbf19aae94009d`;
aggregate prompt-token hash
`b0ff398ab81bfea0803b14328425c4cdb355739d323c34be3fa0273ed305e3e7`.

At 18:39 PDT, vLLM batch-3 job `5439032` was still active after 8:36. All ranks
had entered warmup, but rank progress was skewed: one rank had reached the
measurement barrier while several remained at warmup step 1. It remains an
unvalidated active attempt. The freed slot was filled with AFD 1K/b6 job
`5439139`, run directory `afd_qwen3_1024_b6_n8_20260716_183930`; exactly two
FastAFD sweep jobs are active.

At 18:45 PDT, vLLM batch-3 attempt `5439032` was conclusively stalled: every
rank log had remained unchanged since 18:36:25, ranks 0--13 and 15 were frozen
at `warmup_step_1_start`, rank 14 alone was at `measurement_barrier_start`, and
all 16 GPUs were resident at 179--181 GiB but 0% utilized. The job was cancelled
after 15:34 and is not a measurement. Once Slurm teardown completed, replacement
job `5439248` was submitted at 18:47 with the same exact control and point while
excluding the affected `nvl72082-T[02-05]` nodes. Run directory:
`vllm_qwen3_1024_dp16_ep16_b3_20260716_184712`. Together with AFD b6, exactly
two jobs are active.

AFD batch-6 job `5439139` completed `0:0` in 8:05; the longer startup was active
compilation of the new graph bucket. All 168 exact 1,024-token requests returned
16 tokens, and all 28 attention workers proved both 15-step batch-6 windows.
The final window measured 16.1127 ms median or **325.8304 tok/s/GPU**. Result:
`afd_qwen3_1024_b6_n8_20260716_183930/experiment/afd-result.json`, SHA-256
`01e3d31e37f0c0915fb29ef42463bd8371d5c92e095736d58f80e5ffd278f8c9`;
aggregate prompt-token hash
`c61772109ff30198a5e419ed1ae5a39be612eb1dd2f7416ab4ee27c45f552a39`.
AFD 1K/b8 job `5439272` was submitted at 18:50 PDT, run directory
`afd_qwen3_1024_b8_n8_20260716_185007`. It and vLLM replacement b3 job
`5439248` are the two active jobs.

The second vLLM batch-3 attempt `5439248` reproduced the first attempt's exact
rank divergence on different nodes: after every rank completed warmup step 0,
rank 14 alone entered the measurement barrier while ranks 0--13 and 15 entered
warmup step 1 and froze. Root cause was the control's rank-local warmup exit:
rank 14 locally produced all first tokens one engine call earlier and stopped
participating in EP execution. The other ranks then required it for their next
DeepEP collective. This was cancelled after 10:22 and is not a measurement.

The vLLM control now computes warmup readiness with `MIN` across the vLLM CPU
world group after each engine call, guaranteeing every EP rank executes the same
number of calls before measurement. There is no padded batch or alternate
backend. Local shell, embedded-Python, and diff checks passed; remote shell and
dry-run checks plus byte identity passed. Updated vLLM control SHA-256:
`0e46c1193bc8c79ac2bd1af7cdfbcc2bdc9502cba7b7f53fbd40d8ae639df224`.

AFD batch-8 job `5439272` completed `0:0` in 8:17. All 224 exact 1,024-token
requests returned 16 tokens, and all 28 attention workers proved two complete
15-step batch-8 windows. The final window measured 16.9594 ms median or
**412.7512 tok/s/GPU**. Result:
`afd_qwen3_1024_b8_n8_20260716_185007/experiment/afd-result.json`, SHA-256
`c4d019a22a7a2b88b7913c2470a8e1f73bff62ee320d6c017e45682efc1e8f3f`;
aggregate prompt-token hash
`d8432f616781a64659a42352474bfaab299633e7356f26f2c52d65ea3f100833`.

Corrected vLLM 1K/b3 job `5439364` and AFD 1K/b12 job `5439365` were submitted
at 18:59 PDT. Run directories are respectively
`vllm_qwen3_1024_dp16_ep16_b3_20260716_185942` and
`afd_qwen3_1024_b12_n8_20260716_185942`; exactly two jobs are active.

Corrected vLLM batch-3 job `5439364` completed `0:0` in 5:11. Every rank
reported global readiness 0 after warmup step 0 and 1 after step 1, then crossed
the measurement barrier together. Its 48 prompts were all exactly 1,024 tokens.
The primary synchronized request result was **23.5755 ms median decode-step
latency and 127.2507 tok/s/GPU**; wall mean was 25.3217 ms and 118.4754
tok/s/GPU. CUDA-kernel throughput was 227.6473 tok/s/GPU. Result:
`vllm_qwen3_1024_dp16_ep16_b3_20260716_185942/baseline-result.json`, SHA-256
`f84b273fc086dcb46ebaeeea9ff32d970030e619fad0edc22de40f3dd2ab3885`.

AFD batch-12 job `5439365` completed `0:0` in 5:00. All 336 exact prompts
returned 16 tokens, and the final proven 15-step window measured 17.4330 ms
median or **602.3043 tok/s/GPU**. Result:
`afd_qwen3_1024_b12_n8_20260716_185942/experiment/afd-result.json`, SHA-256
`6b68f9be19462e1b6f928dca2a83b9d1a3b5211749ce59d9bc09f0e7f2421661`;
aggregate prompt-token hash
`26b26df2a51f2c7db5b639b2ed51042bda6d3eb8eff71b53fa2b76a4553a4281`.

vLLM 1K/b4 job `5439475` and AFD 1K/b16 job `5439476` were submitted at
19:08 PDT, in `vllm_qwen3_1024_dp16_ep16_b4_20260716_190806` and
`afd_qwen3_1024_b16_n8_20260716_190806`. Exactly two jobs are active.

vLLM batch-4 job `5439475` completed `0:0` in 5:08. Its primary synchronized
request result was **23.2567 ms median decode-step latency and 171.9933
tok/s/GPU**; the complete 15-step series has 23.7462 ms mean and 168.4482
tok/s/GPU. Secondary CUDA-kernel throughput was 309.9584 tok/s/GPU. All 64
prompts were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b4_20260716_190806/baseline-result.json`, SHA-256
`bdd8334bbc196c46df6518b18ab11981e8a7559882ea033b1e16638161b97ece`.

AFD batch-16 job `5439476` completed `0:0` in 5:32. All 448 exact prompts
returned 16 tokens. The primary request result was **18.2054 ms median decode
interval and 769.0038 tok/s/GPU**; mean interval was 17.2246 ms. Result:
`afd_qwen3_1024_b16_n8_20260716_190806/experiment/afd-result.json`, SHA-256
`b7096175a6c5cb3adc20220e176d2dda329a6b8b06adec5e4f3145d0646fe4ac`;
aggregate prompt-token hash
`020769f1816656659619fd7760a0dcdb9e76d7377a993ebcf851d35dfc91ed12`.

vLLM 1K/b6 job `5439540` and AFD 1K/b24 job `5439541` were submitted at
19:17 PDT, in `vllm_qwen3_1024_dp16_ep16_b6_20260716_191714` and
`afd_qwen3_1024_b24_n8_20260716_191715`. Exactly two jobs are active.

vLLM batch-6 job `5439540` completed `0:0` in 5:55. Its primary synchronized
request result was **24.2796 ms median decode-step latency and 247.1209
tok/s/GPU**; the complete series has 24.5578 ms mean and 244.3212 tok/s/GPU.
Secondary CUDA-kernel throughput was 457.0318 tok/s/GPU. All 96 prompts were
exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b6_20260716_191714/baseline-result.json`, SHA-256
`9e2f58bf6dc085eb873d8b15cdc0b1d5da4cda0fdabe2cefbd6a6ec113b96d1c`.
AFD b24 remained active at 7:53, so the free slot was filled with vLLM 1K/b8
job `5439677`, run directory
`vllm_qwen3_1024_dp16_ep16_b8_20260716_192513`. Exactly two jobs are active.

AFD batch-24 job `5439541` completed `0:0` in 8:56. All 672 exact prompts
returned 16 tokens. Its primary request result was **18.0581 ms median decode
interval and 1,162.9151 tok/s/GPU**; mean interval was 16.7944 ms. Result:
`afd_qwen3_1024_b24_n8_20260716_191715/experiment/afd-result.json`, SHA-256
`670e183f8b34964926c18f22f8f5668882f8b23efb2fd770072c04730ab547e6`;
aggregate prompt-token hash
`7ea195fdb17c728e62829ef3e2d09cebdfc2e5bc53a048c37b6aff1aef148c59`.
AFD 1K/b32 job `5439701` was submitted at 19:27 PDT, run directory
`afd_qwen3_1024_b32_n8_20260716_192759`. It and vLLM b8 are the two active
jobs.

vLLM batch-8 job `5439677` completed `0:0` in 3:59. Its primary synchronized
request result was **25.9563 ms median decode-step latency and 308.2100
tok/s/GPU**; the complete series has 26.1891 ms mean and 305.4702 tok/s/GPU.
Secondary CUDA-kernel throughput was 569.4306 tok/s/GPU. All 128 prompts were
exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b8_20260716_192513/baseline-result.json`, SHA-256
`8a0f108c13697a2b7cfef5bb00dc285eb0820bf280c1f7d9e4325d8decb59e61`.
AFD b32 remained active at 5:32, so the free slot was filled with vLLM 1K/b12
job `5439747`, run directory
`vllm_qwen3_1024_dp16_ep16_b12_20260716_193338`. Exactly two jobs are active.

AFD batch-32 job `5439701` completed `0:0` in 9:07. All 896 exact prompts
returned 16 tokens. Its primary request result was **19.8281 ms median decode
interval and 1,412.1403 tok/s/GPU**; mean interval was 18.3858 ms. Result:
`afd_qwen3_1024_b32_n8_20260716_192759/experiment/afd-result.json`, SHA-256
`008ddda525f72d4c1cfe9cf98bac8ebc03a075664439988d53a8f62e5796e678`;
aggregate prompt-token hash
`f3c65bcd36ecf5396de06f589a3690019781fa0e96edb92a3c80c0b4706e2e19`.
AFD 1K/b48 job `5439806` was submitted at 19:38 PDT, run directory
`afd_qwen3_1024_b48_n8_20260716_193830`. It and vLLM b12 are the two active
jobs.

vLLM batch-12 job `5439747` completed `0:0` in 6:07. Its primary synchronized
request result was **25.9905 ms median decode-step latency and 461.7079
tok/s/GPU**; the complete series has 26.4139 ms mean and 454.3057 tok/s/GPU.
Secondary CUDA-kernel throughput was 845.7934 tok/s/GPU. All 192 prompts were
exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b12_20260716_193338/baseline-result.json`, SHA-256
`d9ab6e9b4309420031076140b6af025622be421f71d3a28cf10348f20e41479b`.
The free slot was filled with vLLM 1K/b16 job `5439835`, run directory
`vllm_qwen3_1024_dp16_ep16_b16_20260716_194114`. It and AFD b48 are the two
active jobs.

vLLM batch-16 job `5439835` completed `0:0` in 5:10. Its primary synchronized
request result was **24.3421 ms median decode-step latency and 657.2981
tok/s/GPU**; the complete series has 24.5767 ms mean and 651.0225 tok/s/GPU.
Secondary CUDA-kernel throughput was 1,152.0144 tok/s/GPU. All 256 prompts were
exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b16_20260716_194114/baseline-result.json`, SHA-256
`23b0d1b3d003bcec85409f1fa705c3f3ae3e5c58c8c42bff41917a4ba59f3dce`.

AFD batch-48 job `5439806` completed `0:0` in 6:02. All 1,344 exact prompts
returned 16 tokens. Its primary request result was **23.8292 ms median decode
interval and 1,762.5435 tok/s/GPU**; mean interval was 22.3400 ms. Result:
`afd_qwen3_1024_b48_n8_20260716_193830/experiment/afd-result.json`, SHA-256
`b58b3323bbc6323dbebcf0492a4957c2d21cb9cf511fcdf1e80bda3fd3e66130`;
aggregate prompt-token hash
`30366c7c9a2e267c25d51953f75c0087e1ed0a62d31c2c287f5cdd35cb83dae9`.

vLLM 1K/b24 job `5439947` and AFD 1K/b64 job `5439948` were submitted at
19:49 PDT, in `vllm_qwen3_1024_dp16_ep16_b24_20260716_194915` and
`afd_qwen3_1024_b64_n8_20260716_194916`. Exactly two jobs are active.

vLLM batch-24 job `5439947` completed `0:0` in 4:47. Its primary synchronized
request result was **26.2679 ms median decode-step latency and 913.6624
tok/s/GPU**. One 779.6538 ms tail step raises the full-series mean to 76.6184 ms
and reduces mean-derived throughput to 313.2407 tok/s/GPU; the complete vector
is retained. Secondary CUDA-kernel throughput was 1,575.7764 tok/s/GPU. All 384
prompts were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b24_20260716_194915/baseline-result.json`, SHA-256
`3a820c98ef03abdcb2e252e9c464fee5fd2a767eb1e25b6f8f9fbf7dd8c259e9`.

AFD batch-64 job `5439948` completed `0:0` in 4:50. All 1,792 exact prompts
returned 16 tokens. Its primary robust result was **28.0778 ms median decode
interval and 1,994.4587 tok/s/GPU**. The retained 14-interval vector contains a
5,794.8244 ms first interval and 0.6498 ms final interval; consequently its mean
is 437.9256 ms and mean-derived throughput is 127.8756 tok/s/GPU. Result:
`afd_qwen3_1024_b64_n8_20260716_194916/experiment/afd-result.json`, SHA-256
`26567f9bb1809a180e1de9d74366fae0c39fba433c5b6b8203971b305ea80774`;
aggregate prompt-token hash
`b077a3ce4cd075b8bd9d599c7f3296cbd6f2d1ed6061b9976084dee80cc272d4`.

vLLM 1K/b32 job `5440030` and AFD 1K/b96 job `5440031` were submitted at
19:57 PDT, in `vllm_qwen3_1024_dp16_ep16_b32_20260716_195734` and
`afd_qwen3_1024_b96_n8_20260716_195735`. Exactly two jobs are active.

vLLM batch-32 job `5440030` completed `0:0` in 5:14. Its primary synchronized
request result was **25.9824 ms median decode-step latency and 1,231.6040
tok/s/GPU**; the complete series was stable at 26.2971 ms mean and 1,216.8621
tok/s/GPU. Secondary CUDA-kernel throughput was 2,029.7186 tok/s/GPU. All 512
prompts were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b32_20260716_195734/baseline-result.json`, SHA-256
`03991a96fcfad731c7fcebbc607bace981f5a6727d49ef841c03c805da74dbcd`.
AFD b96 remained active at 8:25, so the free slot was filled with vLLM 1K/b48
job `5440100`, run directory
`vllm_qwen3_1024_dp16_ep16_b48_20260716_200616`. Exactly two jobs are active.

AFD batch-96 job `5440031` completed `0:0` in 9:28. All 2,688 exact prompts
returned 16 tokens. Its primary robust result was **33.2956 ms median decode
interval and 2,522.8583 tok/s/GPU**. The retained vector contains a 7,263.2312
ms first interval and 0.5004 ms final interval; mean latency is therefore
547.5226 ms and mean-derived throughput 153.4183 tok/s/GPU. Result:
`afd_qwen3_1024_b96_n8_20260716_195735/experiment/afd-result.json`, SHA-256
`8e048343746666c8abae2fc9eb90260158f5fc7b70f0bd941f8164217eec68df`;
aggregate prompt-token hash
`059d338e42ae7a68881b388614f55e06d72f1bc85a622f34978894fb84f267e6`.
AFD 1K/b128 job `5440132` was submitted at 20:08 PDT, run directory
`afd_qwen3_1024_b128_n8_20260716_200855`. It and vLLM b48 are the two active
jobs.

vLLM batch-48 job `5440100` completed `0:0` in 6:26. Its primary synchronized
request result was **30.0366 ms median decode-step latency and 1,598.0497
tok/s/GPU**; 72.0082 and 42.5449 ms tails raise the full-series mean to 34.2351
ms and reduce mean throughput to 1,402.0710 tok/s/GPU. Secondary CUDA-kernel
throughput was 2,554.8398 tok/s/GPU. All 768 prompts were exactly 1,024 tokens.
Result: `vllm_qwen3_1024_dp16_ep16_b48_20260716_200616/baseline-result.json`,
SHA-256 `4eabf208c6fd4d9e6fade05c359651084533a5454c823d088e80ca01fe616b5e`.

AFD batch-128 job `5440132` completed `0:0` in 5:08. All 3,584 exact prompts
returned 16 tokens. Its primary robust result was **39.6706 ms median decode
interval and 2,823.2487 tok/s/GPU**. The retained vector contains a 5,787.3528
ms interval and a 0.5816 ms final interval; mean latency is 445.5361 ms and
mean-derived throughput 251.3826 tok/s/GPU. Result:
`afd_qwen3_1024_b128_n8_20260716_200855/experiment/afd-result.json`, SHA-256
`6613b2867cfe23a46030a0e9a43f18d301eff0429608bf1d4a3c59d65521dfa0`;
aggregate prompt-token hash
`fbff5d61bce5d3767fd2aa86772aef5817eda11c40bf48d241dbb08ed7bc9939`.

vLLM 1K/b64 job `5440210` and AFD 1K/b192 job `5440211` were submitted at
20:15 PDT, in `vllm_qwen3_1024_dp16_ep16_b64_20260716_201517` and
`afd_qwen3_1024_b192_n8_20260716_201518`. Exactly two jobs are active.

vLLM batch-64 job `5440210` completed `0:0` in 4:11. Its primary synchronized
request result was stable: **30.7211 ms median decode-step latency and
2,083.2569 tok/s/GPU**, with 30.7836 ms mean and 2,079.0265 tok/s/GPU.
Secondary CUDA-kernel throughput was 3,322.4779 tok/s/GPU. All 1,024 prompts
were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b64_20260716_201517/baseline-result.json`, SHA-256
`ceb1a4fa173153fbef25b4df34e5a6783fe8837f8b873e949f40d96d915dff97`.

AFD batch-192 job `5440211` completed `0:0` in 5:39. All 5,376 exact prompts
returned 16 tokens. Its primary robust result was **52.1855 ms median decode
interval and 3,219.2825 tok/s/GPU**. The retained vector contains a 5,753.9065
ms interval and a 0.6461 ms final interval; mean latency is 453.5490 ms and
mean-derived throughput 370.4120 tok/s/GPU. Result:
`afd_qwen3_1024_b192_n8_20260716_201518/experiment/afd-result.json`, SHA-256
`8b2497bad0edc6788a15199e5c8a7ef3f1cadd20527b4398ea89edaf785ffb50`;
aggregate prompt-token hash
`f97a35a9e36720f3a5f16222bfed8b3f3f790488cd475ba1ec318f8fa826fd0a`.

vLLM 1K/b96 job `5440311` and AFD 1K/b256 job `5440312` were submitted at
20:23 PDT, in `vllm_qwen3_1024_dp16_ep16_b96_20260716_202326` and
`afd_qwen3_1024_b256_n8_20260716_202326`. Exactly two jobs are active.

vLLM batch-96 job `5440311` completed `0:0` in 6:24. Its primary synchronized
request result was **35.4061 ms median decode-step latency and 2,711.3997
tok/s/GPU**. One 119.8320 ms tail raises the complete-series mean to 40.9038 ms
and reduces mean throughput to 2,346.9685 tok/s/GPU. Secondary CUDA-kernel
throughput was 4,116.7665 tok/s/GPU. All 1,536 prompts were exactly 1,024
tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b96_20260716_202326/baseline-result.json`, SHA-256
`b1eeb72d759059dd40d4f26a23f304dbd2c127a3dcdeff3e0a22ee4d1b60f253`.
AFD b256 remained active at 8:04, so the free slot was filled with vLLM 1K/b128
job `5440410`, run directory
`vllm_qwen3_1024_dp16_ep16_b128_20260716_203134`. Exactly two jobs are active.

AFD batch-256 job `5440312` completed `0:0` in 10:16. All 7,168 exact prompts
returned 16 tokens. Its primary robust result was **64.7658 ms median decode
interval and 3,458.6144 tok/s/GPU**. The retained vector contains a 5,672.5161
ms interval and a 0.6371 ms final interval; mean latency is 457.4009 ms and
mean-derived throughput 489.7235 tok/s/GPU. Result:
`afd_qwen3_1024_b256_n8_20260716_202326/experiment/afd-result.json`, SHA-256
`0b360e7cf3d3af8bfaac55dc115399af537b0d65e8a4fe2cc9f7571063c7d67a`;
aggregate prompt-token hash
`443e6d701faf37bb4e089447ad7149a0c3c063fd896767175c5ec3d37a766ece`.
AFD 1K/b384 job `5440430` was submitted at 20:34 PDT, run directory
`afd_qwen3_1024_b384_n8_20260716_203424`. It and vLLM b128 are the two active
jobs.

vLLM batch-128 job `5440410` completed `0:0` in 4:14. Its primary synchronized
request result was **39.6523 ms median decode-step latency and 3,228.0612
tok/s/GPU**; a 55.4802 ms tail raises the complete-series mean to 40.7234 ms and
reduces mean throughput to 3,143.1582 tok/s/GPU. Secondary CUDA-kernel
throughput was 4,529.4278 tok/s/GPU. All 2,048 prompts were exactly 1,024
tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b128_20260716_203134/baseline-result.json`, SHA-256
`3ea23c8e735c41c043653f71cbee9b61670994d00f4ebcb008c08fcd2f74291a`.
AFD b384 remained active at 6:18, so the free slot was filled with vLLM 1K/b192
job `5440508`, run directory
`vllm_qwen3_1024_dp16_ep16_b192_20260716_204043`. Exactly two jobs are active.

AFD batch-384 job `5440430` completed `0:0` in 7:14. All 10,752 exact prompts
returned 16 tokens. Its primary robust result was **87.6525 ms median decode
interval and 3,833.3169 tok/s/GPU**. The retained vector contains a 5,845.3817
ms interval and a 0.7600 ms final interval; mean latency is 489.3894 ms and
mean-derived throughput 686.5699 tok/s/GPU. Result:
`afd_qwen3_1024_b384_n8_20260716_203424/experiment/afd-result.json`, SHA-256
`52a51e3dd291aec760fc4621cb8c035768e12a90cd6d30a62f2ca82f0f800656`;
aggregate prompt-token hash
`1375e1f9663e1f8636f6a32608fe3b1c80aeb18f9d449b51b67ab5062575b255`.
The final planned 1K AFD point, b512 job `5440556`, was submitted at 20:43 PDT,
run directory `afd_qwen3_1024_b512_n8_20260716_204336`. It and vLLM b192 are
the two active jobs.

vLLM batch-192 job `5440508` completed `0:0` in 6:57. Its primary synchronized
request result was stable: **47.8515 ms median decode-step latency and
4,012.4113 tok/s/GPU**, with 47.3710 ms mean and 4,053.1107 tok/s/GPU.
Secondary CUDA-kernel throughput was 5,578.4458 tok/s/GPU. All 3,072 prompts
were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b192_20260716_204043/baseline-result.json`, SHA-256
`61f34d4baa9c1434d1bf701d4d673283b76b1e537b209a4f19276c6f573f8f85`.
AFD b512 remained active at 5:40, so the free slot was filled with vLLM 1K/b256
job `5440622`, run directory
`vllm_qwen3_1024_dp16_ep16_b256_20260716_204929`. Exactly two jobs are active.

vLLM batch-256 job `5440622` completed `0:0` in 5:02. Its primary synchronized
request result was stable: **56.4194 ms median decode-step latency and
4,537.4466 tok/s/GPU**, with 56.1630 ms mean and 4,558.1600 tok/s/GPU.
Secondary CUDA-kernel throughput was 6,137.5789 tok/s/GPU. All 4,096 prompts
were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b256_20260716_204929/baseline-result.json`, SHA-256
`b85c2c0a3a70f50228c7c7884afec3e7c062540d522cb62e66a544e1daa7077a`.

AFD batch-512 job `5440556`, the terminal planned 1K AFD point, completed `0:0`
in 11:18. All 14,336 exact prompts returned 16 tokens. Its primary robust
result was **112.5493 ms median decode interval and 3,980.4762 tok/s/GPU**. The
retained vector contains a 5,972.5533 ms interval and a 0.6956 ms final
interval; mean latency is 520.7332 ms and mean-derived throughput 860.3254
tok/s/GPU. Result:
`afd_qwen3_1024_b512_n8_20260716_204336/experiment/afd-result.json`, SHA-256
`8ef13661e768c56b72fe2730c61c4bbe3cc16fb9a9743d18e414cb0c76dc1c29`;
aggregate prompt-token hash
`8a945786121675ea6ce555e16dc1b7bb7fce5a229348985dba6f129d35e429bb`.

vLLM 1K/b384 job `5440681` was submitted at 20:55 PDT, run directory
`vllm_qwen3_1024_dp16_ep16_b384_20260716_205535`. After AFD b512 teardown,
AFD 2K/b2 job `5440686` was submitted at 20:56 PDT, run directory
`afd_qwen3_2048_b2_n8_20260716_205604`. Exactly two jobs are active; the 1K AFD
sequence is complete and the AFD slot has advanced to 2K.

vLLM batch-384 job `5440681` completed `0:0` in 6:29. Its primary synchronized
request result was stable: **81.0179 ms median decode-step latency and
4,739.6942 tok/s/GPU**, with 81.5758 ms mean and 4,707.2781 tok/s/GPU.
Secondary CUDA-kernel throughput was 6,053.9090 tok/s/GPU. All 6,144 prompts
were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b384_20260716_205535/baseline-result.json`, SHA-256
`f7a0a2ff848c43af3465fb7bba3f3da8e90bc9f2ca78adfa7667ebf0dc6e314a`.

AFD 2K/b2 job `5440686` completed `0:0` in 4:30. All 56 prompts were proven
exactly 2,048 tokens and returned 16 tokens. Its primary result was **15.5553 ms
median decode interval and 112.5016 tok/s/GPU**; the complete 14-interval series
has 14.8105 ms mean and is retained. Result:
`afd_qwen3_2048_b2_n8_20260716_205604/experiment/afd-result.json`, SHA-256
`6e914117964897bc3179501d8a02edc881da1d5ef40b88a134cd2c9a3ef6e1b9`;
aggregate prompt-token hash
`251aa026419a121916bcf4be088d30566eff6e91757d6c5d6e3296005b27e331`.

The terminal planned 1K vLLM point, b512 job `5440790`, and AFD 2K/b3 job
`5440791` were submitted at 21:04 PDT. Run directories are respectively
`vllm_qwen3_1024_dp16_ep16_b512_20260716_210437` and
`afd_qwen3_2048_b3_n8_20260716_210437`; exactly two jobs are active.

vLLM batch-512 job `5440790`, the terminal planned 1K baseline point,
completed `0:0` in 7:35. Its primary synchronized request result was **92.4569
ms median decode-step latency and 5,537.7146 tok/s/GPU**. One 240.5355 ms tail
raises the complete-series mean to 103.0223 ms and reduces mean throughput to
4,969.8002 tok/s/GPU. Secondary CUDA-kernel throughput was 6,172.7024
tok/s/GPU. All 8,192 prompts were exactly 1,024 tokens. Result:
`vllm_qwen3_1024_dp16_ep16_b512_20260716_210437/baseline-result.json`, SHA-256
`896f269a32dac31fb8fd7c9b5a8678597175a8f7de8698f700b615abd57bfc91`.

AFD 2K/b3 job `5440791` completed `0:0` in 7:42. All 84 exact 2,048-token
prompts returned 16 tokens. Its primary result was **16.5991 ms median decode
interval and 158.1414 tok/s/GPU**; the full 14-interval vector has 15.6874 ms
mean and is retained. Result:
`afd_qwen3_2048_b3_n8_20260716_210437/experiment/afd-result.json`, SHA-256
`82a7f01be8f925621a292c9ae1a81dff356caa486c7baec622ebaa47633388fb`;
aggregate prompt-token hash
`236559b7fcc658b1926ce0c9a3cb86a9438da73fe2e731ce24ccf4edc5cf09a1`.

The 1K sweeps are complete for both modes. vLLM 2K/b2 job `5440858` and AFD
2K/b4 job `5440859` were submitted at 21:13 PDT, in
`vllm_qwen3_2048_dp16_ep16_b2_20260716_211313` and
`afd_qwen3_2048_b4_n8_20260716_211313`. Exactly two jobs are active.

vLLM 2K/b2 job `5440858` completed `0:0` in 5:10. Its primary synchronized
request result was **24.9861 ms median decode-step latency and 80.0446
tok/s/GPU**, with 25.2736 ms mean and 79.1339 tok/s/GPU. Secondary CUDA-kernel
throughput was 143.9759 tok/s/GPU. All 32 prompts were exactly 2,048 tokens.
Result: `vllm_qwen3_2048_dp16_ep16_b2_20260716_211313/baseline-result.json`,
SHA-256 `9c469d5a05e1f005de4f6b0629a713092d222347e533216c40ed2c638075493c`.

AFD 2K/b4 job `5440859` completed `0:0` in 4:28. All 112 exact prompts
returned 16 tokens. Its primary result was **16.7768 ms median decode interval
and 208.6213 tok/s/GPU**; the full vector has 15.8034 ms mean. Result:
`afd_qwen3_2048_b4_n8_20260716_211313/experiment/afd-result.json`, SHA-256
`7a4cddef584d7f4386a3af3cc7e0ace4f18ff15303f6af3138bf7e8d5a7ed987`;
aggregate prompt-token hash
`96df67a9bbdf1faebcb56243a51a13fff12e3d8b0bb9976cca0d947d61bfcea7`.

vLLM 2K/b3 job `5440964` and AFD 2K/b6 job `5440965` were submitted at
21:21 PDT, in `vllm_qwen3_2048_dp16_ep16_b3_20260716_212152` and
`afd_qwen3_2048_b6_n8_20260716_212153`. Exactly two jobs are active.

vLLM 2K/b3 job `5440964` completed `0:0` in 4:57. Its globally coordinated
odd-batch warmup completed on every rank. The primary request result was stable:
**23.7990 ms median decode-step latency and 126.0555 tok/s/GPU**, with 23.8338
ms mean and 125.8714 tok/s/GPU. Secondary CUDA-kernel throughput was 223.4760
tok/s/GPU. All 48 prompts were exactly 2,048 tokens. Result:
`vllm_qwen3_2048_dp16_ep16_b3_20260716_212152/baseline-result.json`, SHA-256
`aaea475e59a3a0feb6d17b47618d43ab0e28b8b7515d6a055caafabd26162014`.

AFD 2K/b6 job `5440965` completed `0:0` in 7:46. All 168 exact prompts
returned 16 tokens. Its primary result was **16.3881 ms median decode interval
and 320.3544 tok/s/GPU**; the full vector has 15.0782 ms mean. Result:
`afd_qwen3_2048_b6_n8_20260716_212153/experiment/afd-result.json`, SHA-256
`a83ad1fd677217a8d4f61ea395a5156652d3bd142d8544be549a398641284f31`;
aggregate prompt-token hash
`44a4570781bedaadb3699001d2cd2dbf28e53841ccfe434e66f23a1257ddf51e`.

vLLM 2K/b4 job `5441060` and, after b6 teardown, AFD 2K/b8 job `5441064` were
submitted at 21:30 PDT. Run directories are
`vllm_qwen3_2048_dp16_ep16_b4_20260716_213015` and
`afd_qwen3_2048_b8_n8_20260716_213045`; exactly two jobs are active.

vLLM 2K/b4 job `5441060` completed `0:0` in 3:52. Its primary synchronized
request result was **23.6917 ms median decode-step latency and 168.8357
tok/s/GPU**, with 24.3161 ms mean and 164.5003 tok/s/GPU. Secondary CUDA-kernel
throughput was 302.3437 tok/s/GPU. All 64 prompts were exactly 2,048 tokens;
the complete 15-step wall-latency vector is retained in the result. Result:
`vllm_qwen3_2048_dp16_ep16_b4_20260716_213015/baseline-result.json`, SHA-256
`defd047896703fb35a8cf275a950c0bd0f46ccece2700ef537ccc1b84073d890`.

AFD 2K/b8 job `5441064` completed `0:0` in 4:11. All 224 exact prompts
returned 16 tokens. Its primary result was **16.7831 ms median decode interval
and 417.0863 tok/s/GPU**; the complete 14-interval vector has 15.8875 ms mean
and mean-derived throughput 440.5989 tok/s/GPU. Result:
`afd_qwen3_2048_b8_n8_20260716_213045/experiment/afd-result.json`, SHA-256
`392bb1e58ee6f36b2386634bba7bfb184dc33b61b6cd83fbf7ac52241c4c3417`;
aggregate prompt-token hash
`bbe99bc9b72563b2249ffa9b4cd9f4e531d84da008c031f0a1d5d17f980f209b`.

vLLM 2K/b6 job `5441173` and AFD 2K/b12 job `5441174` were submitted at
21:40 PDT, in `vllm_qwen3_2048_dp16_ep16_b6_20260716_214044` and
`afd_qwen3_2048_b12_n8_20260716_214044`. Exactly two jobs are active.

vLLM 2K/b6 job `5441173` completed `0:0` in 6:29. Its primary synchronized
request result was **25.3858 ms median decode-step latency and 236.3530
tok/s/GPU**, with 25.6241 ms mean and 234.1549 tok/s/GPU. Secondary CUDA-kernel
throughput was 441.1151 tok/s/GPU. All 96 prompts were exactly 2,048 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b6_20260716_214044/baseline-result.json`, SHA-256
`bcfd3f8b2092e556ade6f1b532bc3b22052a8b1232ba5ca09796522e53de997c`.
AFD b12 remained active at 7:59, so the free slot was filled with vLLM 2K/b8
job `5441344`, run directory
`vllm_qwen3_2048_dp16_ep16_b8_20260716_214933`. Exactly two jobs are active.

AFD 2K/b12 job `5441174` completed `0:0` in 9:20. All 336 exact prompts
returned 16 tokens. Its primary result was **16.9834 ms median decode interval
and 618.2500 tok/s/GPU**; the complete 14-interval vector has 16.0556 ms mean
and mean-derived throughput 653.9777 tok/s/GPU. Result:
`afd_qwen3_2048_b12_n8_20260716_214044/experiment/afd-result.json`, SHA-256
`3b58a41ebf5232487fe4503a1ff9591a7f77900e1d3d26ae4884fb1f4408562f`;
aggregate prompt-token hash
`84c9cc1123a62e741560d4e229474f8bc0b554b25ec5062fe634a44c84c2f33d`.
AFD 2K/b16 job `5441384` was submitted at 21:52 PDT, run directory
`afd_qwen3_2048_b16_n8_20260716_215212`. It and vLLM b8 are the two active
jobs.

vLLM 2K/b8 job `5441344` completed `0:0` in 4:06. Its primary synchronized
request result was **24.2345 ms median decode-step latency and 330.1075
tok/s/GPU**, with 24.8131 ms mean and 322.4105 tok/s/GPU. Secondary CUDA-kernel
throughput was 584.3956 tok/s/GPU. All 128 prompts were exactly 2,048 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b8_20260716_214933/baseline-result.json`, SHA-256
`9d4f3812c1da3e369675c0d2db567428d24d02633a027aa9c34cba1efa092124`.
After b8 scheduler teardown, vLLM 2K/b12 job `5441441` was submitted at 21:57
PDT, run directory `vllm_qwen3_2048_dp16_ep16_b12_20260716_215707`. It and AFD
b16 are the two active jobs.

AFD 2K/b16 job `5441384` completed `0:0` in 8:47. All 448 exact prompts
returned 16 tokens. Its primary result was **18.4434 ms median decode interval
and 759.0777 tok/s/GPU**; the complete 14-interval vector has 17.2874 ms mean
and mean-derived throughput 809.8383 tok/s/GPU. Result:
`afd_qwen3_2048_b16_n8_20260716_215212/experiment/afd-result.json`, SHA-256
`ce3b0a77319055daff3a116924d7729f125e581e1b81826331af856f6826bdf1`;
aggregate prompt-token hash
`159b64afe755f87a91c0c530919c8860784b8c29fb0a4f2290f319d2587bb254`.

vLLM 2K/b12 job `5441441` completed `0:0` in 6:21. Its primary synchronized
request result was **25.9435 ms median decode-step latency and 462.5436
tok/s/GPU**, with 26.1402 ms mean and 459.0638 tok/s/GPU. Secondary CUDA-kernel
throughput was 843.3466 tok/s/GPU. All 192 prompts were exactly 2,048 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b12_20260716_215707/baseline-result.json`, SHA-256
`9a4f5c44451ef09e28920f8adc43be660d955a707bf419c8bed6d8ce500e20b6`.

vLLM 2K/b16 job `5441535` and AFD 2K/b24 job `5441536` were submitted at
22:05 PDT, in `vllm_qwen3_2048_dp16_ep16_b16_20260716_220509` and
`afd_qwen3_2048_b24_n8_20260716_220510`. Exactly two jobs are active.

vLLM 2K/b16 job `5441535` completed `0:0` in 5:02. Its primary synchronized
request result was **25.9929 ms median decode-step latency and 615.5526
tok/s/GPU**, with 26.1556 ms mean and 611.7227 tok/s/GPU. Secondary CUDA-kernel
throughput was 1,094.2354 tok/s/GPU. All 256 prompts were exactly 2,048 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b16_20260716_220509/baseline-result.json`, SHA-256
`0b0b188beabada664046fe95f5ccbf910f902c321071d0dc84f5e2683402a03b`.
AFD b24 remained active at 8:12, so the free slot was filled with vLLM 2K/b24
job `5441651`, run directory
`vllm_qwen3_2048_dp16_ep16_b24_20260716_221327`. Exactly two jobs are active.

AFD 2K/b24 attempt job `5441536` did not produce a result. At 22:16:11 PDT,
attention worker PID `1448552` on `10.109.19.12` aborted during MB12 decode
graph warmup with `torch.AcceleratorError: CUDA error: unspecified launch
failure`; Ray reported a dead worker and the job remained running without a
viable coordinator. The job was cancelled at 12:59 elapsed, after preserving
the complete crash log under
`afd_qwen3_2048_b24_n8_20260716_220510/experiment/afd.log`. The exact b24
configuration will be retried once on a fresh allocation to distinguish a
transient node/GPU failure from a reproducible graph-size fault; no fallback or
configuration change is being introduced.

vLLM 2K/b24 job `5441651` completed `0:0` in 5:52. Its primary synchronized
request result was **27.1070 ms median decode-step latency and 885.3794
tok/s/GPU**. A final 47.3563 ms sample raises the complete-series mean to
28.8055 ms and reduces mean throughput to 833.1742 tok/s/GPU. Secondary
CUDA-kernel throughput was 1,514.0289 tok/s/GPU. All 384 prompts were exactly
2,048 tokens. Result:
`vllm_qwen3_2048_dp16_ep16_b24_20260716_221327/baseline-result.json`, SHA-256
`e361a800fc479d1ad73b6590a1f959ca58b04ab5899cdddb50c75011ce2877b7`.

The identical AFD 2K/b24 retry job `5441755` and vLLM 2K/b32 job `5441756`
were submitted at 22:22 PDT, in
`afd_qwen3_2048_b24_n8_20260716_222244` and
`vllm_qwen3_2048_dp16_ep16_b32_20260716_222244`. Exactly two jobs are active.

vLLM 2K/b32 job `5441756` completed `0:0` in 5:28. Its primary synchronized
request result was **27.5281 ms median decode-step latency and 1,162.4473
tok/s/GPU**, with 27.6873 ms mean and 1,155.7648 tok/s/GPU. Secondary
CUDA-kernel throughput was 1,921.7695 tok/s/GPU. All 512 prompts were exactly
2,048 tokens; the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b32_20260716_222244/baseline-result.json`, SHA-256
`dfd9c6b1a6460761ad1314e06aa98f68f2690cb33cf447029236f2039def5301`.
The AFD b24 retry remained healthy at 7:38, so the free slot was filled with
vLLM 2K/b48 job `5441829`, run directory
`vllm_qwen3_2048_dp16_ep16_b48_20260716_223044`. Exactly two jobs are active.

The identical AFD 2K/b24 retry job `5441755` completed `0:0` in 8:38 on a
fresh allocation, establishing that attempt `5441536` was a transient
node/GPU failure rather than a reproducible MB12 graph-size boundary. All 672
exact prompts returned 16 tokens. Its primary result was **18.2045 ms median
decode interval and 1,153.5630 tok/s/GPU**; the complete 14-interval vector has
17.1487 ms mean and mean-derived throughput 1,224.5855 tok/s/GPU. Result:
`afd_qwen3_2048_b24_n8_20260716_222244/experiment/afd-result.json`, SHA-256
`89860d0f330861253c30704433699ef8c32899f6929036ea84b93ec97e8924ee`;
aggregate prompt-token hash
`2b1d69929cf4de9ef7f019e476b434abb0c03456ca8a43ae020c79d571486167`.
AFD 2K/b32 job `5441866` was submitted at 22:33 PDT, run directory
`afd_qwen3_2048_b32_n8_20260716_223331`. It and vLLM b48 are the two active
jobs.

vLLM 2K/b48 job `5441829` completed `0:0` in 5:43. Its primary synchronized
request result was **31.4183 ms median decode-step latency and 1,527.7740
tok/s/GPU**. One 72.2188 ms sample raises the complete-series mean to 34.6328
ms and reduces mean throughput to 1,385.9703 tok/s/GPU. Secondary CUDA-kernel
throughput was 2,441.3748 tok/s/GPU. All 768 prompts were exactly 2,048 tokens.
Result: `vllm_qwen3_2048_dp16_ep16_b48_20260716_223044/baseline-result.json`,
SHA-256 `6545edca0c16d6d9143c1429e3159c1ab2e650449b7d11351550685929138c12`.

AFD 2K/b32 job `5441866` completed `0:0` in 4:31. All 896 exact prompts
returned 16 tokens. Its primary result was **20.1902 ms median decode interval
and 1,386.8084 tok/s/GPU**; the complete 14-interval vector has 18.9219 ms mean
and mean-derived throughput 1,479.7673 tok/s/GPU. Result:
`afd_qwen3_2048_b32_n8_20260716_223331/experiment/afd-result.json`, SHA-256
`5f9ee1e8642b53c48c851b8279f1ecda33466b8ee0d7d646cfca36fa140493e6`;
aggregate prompt-token hash
`4709a55a725ff4936077cd0633fde15f6b337c126d812ec5b48899beb2045232`.

vLLM 2K/b64 job `5442078` and AFD 2K/b48 job `5442079` were submitted at
22:43 PDT, in `vllm_qwen3_2048_dp16_ep16_b64_20260716_224316` and
`afd_qwen3_2048_b48_n8_20260716_224316`. Exactly two jobs are active.

vLLM 2K/b64 job `5442078` completed `0:0` in 5:49. Its primary synchronized
request result was **32.4527 ms median decode-step latency and 1,972.0991
tok/s/GPU**, with 32.9406 ms mean and 1,942.8938 tok/s/GPU. Secondary
CUDA-kernel throughput was 3,003.8450 tok/s/GPU. All 1,024 prompts were exactly
2,048 tokens; the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b64_20260716_224316/baseline-result.json`, SHA-256
`999dc128fa6be299e9888e8bce687a4bfd676d56a2e1d32e98edbe71fed553c2`.
AFD b48 remained active at 7:57, so the free slot was filled with vLLM 2K/b96
job `5442216`, run directory
`vllm_qwen3_2048_dp16_ep16_b96_20260716_225125`. Exactly two jobs are active.

AFD 2K/b48 job `5442079` completed `0:0` in 8:59. All 1,344 exact prompts
returned 16 tokens. Its primary result was **23.5467 ms median decode interval
and 1,783.6909 tok/s/GPU**; the complete 14-interval vector has 22.1302 ms mean
and mean-derived throughput 1,897.8568 tok/s/GPU. At the matched b48 point,
this is lower request latency and higher normalized throughput than vLLM's
31.4183 ms / 1,527.7740 tok/s/GPU median result. Result:
`afd_qwen3_2048_b48_n8_20260716_224316/experiment/afd-result.json`, SHA-256
`1987713f1358bebca65a50d762915593ab19b636f7878a23f4f6b3e9ca1baad2`;
aggregate prompt-token hash
`f53c2044e8912cad5e0bf4baa1315e863d0fb9b24920344ecee455aa8496372f`.
AFD 2K/b64 job `5442277` was submitted at 22:54 PDT, run directory
`afd_qwen3_2048_b64_n8_20260716_225406`. It and vLLM b96 are the two active
jobs.

vLLM 2K/b96 job `5442216` completed `0:0` in 6:44. Its primary synchronized
request result was **38.3424 ms median decode-step latency and 2,503.7551
tok/s/GPU**, with 38.2728 ms mean and 2,508.3085 tok/s/GPU. Secondary
CUDA-kernel throughput was 3,711.2742 tok/s/GPU. All 1,536 prompts were exactly
2,048 tokens; the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b96_20260716_225125/baseline-result.json`, SHA-256
`9ddb2771e15be745399a50c4aad128601f423f707bf350d24e0a63d329fdaa7d`.
AFD b64 remained active at 8:02, so the free slot was filled with vLLM 2K/b128
job `5442381`, run directory
`vllm_qwen3_2048_dp16_ep16_b128_20260716_230212`. Exactly two jobs are active.

AFD 2K/b64 job `5442277` completed `0:0` in 9:11. All 1,792 exact prompts
returned 16 tokens. Its primary robust result was **26.5490 ms median decode
interval and 2,109.3095 tok/s/GPU**. The retained vector contains a 5,740.7810
ms first interval and a 0.6299 ms final coordinator-boundary interval; its
complete-series mean is therefore 432.8738 ms and mean-derived throughput is
129.3680 tok/s/GPU. Result:
`afd_qwen3_2048_b64_n8_20260716_225406/experiment/afd-result.json`, SHA-256
`4767621117b79acce9e217edd9a762c151be95467076742a83a9a445c1dc10a0`;
aggregate prompt-token hash
`0a7190f6284a0d59ebbfb329761c6cb222782b938cec609299061d8b5312a49d`.
AFD 2K/b96 job `5442417` was submitted at 23:04 PDT, run directory
`afd_qwen3_2048_b96_n8_20260716_230458`. vLLM b128 reached distributed engine
initialization on all nodes; exactly two jobs are active.

vLLM 2K/b128 job `5442381` completed `0:0` in 7:09. Its primary synchronized
request result was **41.6734 ms median decode-step latency and 3,071.5003
tok/s/GPU**, with 42.3885 ms mean and 3,019.6837 tok/s/GPU. Secondary
CUDA-kernel throughput was 4,246.5577 tok/s/GPU. All 2,048 prompts were exactly
2,048 tokens; the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b128_20260716_230212/baseline-result.json`, SHA-256
`9aa79164d156931ff4df3b3de2ce8a8f67bbb3a32c88ac99f4b3c1745ea2807a`.

AFD 2K/b96 job `5442417` completed `0:0` in 7:15. All 2,688 exact prompts
returned 16 tokens. Its primary robust result was **32.7636 ms median decode
interval and 2,563.8182 tok/s/GPU**. At matched b96, this is lower latency and
slightly higher median throughput than vLLM's 38.3424 ms / 2,503.7551
tok/s/GPU. The retained vector contains a 5,847.5695 ms first interval and a
0.7233 ms final interval, yielding 447.1184 ms mean and 187.8697 tok/s/GPU
mean-derived throughput. Result:
`afd_qwen3_2048_b96_n8_20260716_230458/experiment/afd-result.json`, SHA-256
`d6a07c1bc1a203db5c4b40e59c44d651afce04923e9afe74d0ed0f51168156af`;
aggregate prompt-token hash
`f7584fd2aa2a9ac4a907c4b85c9ee7f29b49ef793073cf2230f7dcd465bbc759`.

vLLM 2K/b192 job `5442564` and AFD 2K/b128 job `5442565` were submitted at
23:13 PDT, in `vllm_qwen3_2048_dp16_ep16_b192_20260716_231316` and
`afd_qwen3_2048_b128_n8_20260716_231317`. Exactly two jobs are active.

vLLM 2K/b192 job `5442564` completed `0:0` in 7:17. Its primary synchronized
request result was **51.5226 ms median decode-step latency and 3,726.5221
tok/s/GPU**, with 51.8800 ms mean and 3,700.8489 tok/s/GPU. Secondary
CUDA-kernel throughput was 4,902.0598 tok/s/GPU. All 3,072 prompts were exactly
2,048 tokens; the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b192_20260716_231316/baseline-result.json`, SHA-256
`e49f77554c27a6853e9bffaef2c14e9343315a16470b45a02772ea88594c3711`.

AFD 2K/b128 job `5442565` completed `0:0` in 5:57. All 3,584 exact prompts
returned 16 tokens. Its primary robust result was **39.7427 ms median decode
interval and 2,818.1266 tok/s/GPU**. At matched b128, AFD has lower median
latency but lower normalized throughput than vLLM's 41.6734 ms / 3,071.5003
tok/s/GPU. The retained vector contains a 5,808.4503 ms interval and a 0.6133
ms final interval, yielding 449.0111 ms mean and 249.4370 tok/s/GPU
mean-derived throughput. Result:
`afd_qwen3_2048_b128_n8_20260716_231317/experiment/afd-result.json`, SHA-256
`7ebfd90ee005aa38bba9b4c319252ff4ff65e6e2141f7a9432575a8448ef1ff5`;
aggregate prompt-token hash
`b5a9304834caf029e7fe152ff3264888140ab66163fe9414e6acfab9ac16699c`.

The terminal planned vLLM 2K point, b256 job `5442717`, and AFD 2K/b192 job
`5442718` were submitted at 23:21 PDT, in
`vllm_qwen3_2048_dp16_ep16_b256_20260716_232144` and
`afd_qwen3_2048_b192_n8_20260716_232145`. Exactly two jobs are active.

vLLM 2K/b256 job `5442717`, the terminal planned 2K baseline point, completed
`0:0` in 6:47. Its primary synchronized request result was **60.7415 ms median
decode-step latency and 4,214.5823 tok/s/GPU**, with 61.0066 ms mean and
4,196.2698 tok/s/GPU. Secondary CUDA-kernel throughput was 5,449.8986
tok/s/GPU. All 4,096 prompts were exactly 2,048 tokens; the complete 15-step
wall-latency vector is retained. Result:
`vllm_qwen3_2048_dp16_ep16_b256_20260716_232144/baseline-result.json`, SHA-256
`392716c1283410c38c9a86415c26d054f49ca8ddf638f4086bcdf90b0dec03dd`.
AFD b192 remained active at 8:09, so the baseline stream advanced to vLLM
4K/b2 job `5442802`, run directory
`vllm_qwen3_4096_dp16_ep16_b2_20260716_233009`. Exactly two jobs are active.

AFD 2K/b192 job `5442718` completed `0:0` in 10:39. All 5,376 exact prompts
returned 16 tokens. Its primary robust result was **50.9956 ms median decode
interval and 3,294.4026 tok/s/GPU**. At matched b192, AFD has slightly lower
median latency but lower normalized throughput than vLLM's 51.5226 ms /
3,726.5221 tok/s/GPU. The retained vector contains a 5,664.0464 ms interval
and a 0.6806 ms final interval, yielding 448.0267 ms mean and 374.9777
tok/s/GPU mean-derived throughput. Result:
`afd_qwen3_2048_b192_n8_20260716_232145/experiment/afd-result.json`, SHA-256
`80be95f38a4e21abd21b53084f711968f67b12af804700b3303ad2f171b18717`;
aggregate prompt-token hash
`16f272cbe255bbb151bd81ab6c60b41aa4f3d2bcb581456b6637ad6a1792986b`.
After scheduler teardown, the terminal planned AFD 2K point, b256 job
`5442836`, was submitted at 23:33 PDT, run directory
`afd_qwen3_2048_b256_n8_20260716_233340`. It and vLLM 4K/b2 are the two active
jobs.

At 23:35 PDT the user raised the concurrency cap from two to four distinct
sweep jobs. Both retained launchers now fail closed at four active `fastafd:`
jobs; local/remote `bash -n`, remote 4K/b3 dry runs, and exact local/remote
hashes passed. Current control hashes are
`34bcb917ddaca850fae6090fd0c56a5054168832df0a5dde5c7731e0a75758ee`
for `run_afd.sh` and
`3d0b9ce26690d80055e281518dc202fee4e3fb5d3fe4597b4c6afdfcf2e3e928`
for `run_vllm.sh`. The two newly available slots were filled with AFD 4K/b2
job `5442848`, run directory `afd_qwen3_4096_b2_n8_20260716_233523`, and vLLM
4K/b3 job `5442849`, run directory
`vllm_qwen3_4096_dp16_ep16_b3_20260716_233523`. Together with AFD 2K/b256 job
`5442836` and vLLM 4K/b2 job `5442802`, exactly four distinct jobs are active.

vLLM 4K/b2 job `5442802` completed `0:0` in 4:45. Its primary synchronized
request result was **23.5177 ms median decode-step latency and 85.0424
tok/s/GPU**, with 23.6210 ms mean and 84.6704 tok/s/GPU. Secondary CUDA-kernel
throughput was 156.5206 tok/s/GPU. All 32 prompts were exactly 4,096 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_4096_dp16_ep16_b2_20260716_233009/baseline-result.json`, SHA-256
`8a168131a822f174e55c8fe894e2437d7fd994b6703ad700c40d266e5b824081`.

AFD 4K/b2 job `5442848` completed `0:0` in 4:22. All 56 exact prompts
returned 16 tokens. Its primary result was **16.3157 ms median decode interval
and 107.2588 tok/s/GPU**; the complete 14-interval vector has 15.3806 ms mean
and mean-derived throughput 113.7799 tok/s/GPU. Result:
`afd_qwen3_4096_b2_n8_20260716_233523/experiment/afd-result.json`, SHA-256
`32e1a5b081ebcb1025a541b7316e8d46c1f96a33a7964f1a42993f4a938bc2ed`;
aggregate prompt-token hash
`bd55521fa0fea1e654740739419ce5a506b488705131fb47718e4fc6e9680124`.

The cluster's `short` QoS still reports `MaxJobsPU=2`, so its third job was
held with `QOSMaxJobsPerUserLimit` despite the new four-job launcher guard. The
account's assigned `normal` QoS has no running-job cap (`MaxSubmitPU=2000`), so
future sweep submissions now explicitly use `normal`; local/remote `bash -n`,
4K/b4 dry run, and byte identity passed. Current control hashes are
`3995c4f68307b6642f7345e2dc1d369982fe27fc16a99568e389ea2383c3cdba`
(AFD) and
`b3dafa0344b8bfcebdb6db3e769eff9289c34adada5fcc49ade519b81b1ad632`
(vLLM). vLLM 4K/b4 job `5442910` and AFD 4K/b3 job `5442911` were submitted
at 23:41 PDT under `normal`, in
`vllm_qwen3_4096_dp16_ep16_b4_20260716_234116` and
`afd_qwen3_4096_b3_n8_20260716_234116`. They are eligible and pending on
priority; together with running AFD 2K/b256 and vLLM 4K/b3, exactly four
distinct sweep jobs are queued/running.

The terminal planned AFD 2K point, b256 job `5442836`, completed `0:0` in
11:18. All 7,168 exact prompts returned 16 tokens. Its primary robust result
was **64.3995 ms median decode interval and 3,478.2895 tok/s/GPU**. At matched
b256, this is lower normalized throughput than vLLM's 60.7415 ms / 4,214.5823
tok/s/GPU, and AFD also has slightly higher robust latency. The retained vector
contains a 5,913.1413 ms interval and a 0.6020 ms final interval, yielding
477.1273 ms mean and 469.4764 tok/s/GPU mean-derived throughput. Result:
`afd_qwen3_2048_b256_n8_20260716_233340/experiment/afd-result.json`, SHA-256
`d20dc03723da66ecb656ce3a26d3f340a83bcd10ff32c46573960782767cf18e`;
aggregate prompt-token hash
`b09c1a20ab39ec5c4ff882e086b0ff68bf318b040006d6be52a9992591719450`.
The planned 2K sweeps are now complete for both modes.

vLLM 4K/b3 job `5442849` completed `0:0` in 6:02. Its primary synchronized
request result was **24.6867 ms median decode-step latency and 121.5229
tok/s/GPU**, with 24.6994 ms mean and 121.4604 tok/s/GPU. Secondary CUDA-kernel
throughput was 229.0788 tok/s/GPU. All 48 prompts were exactly 4,096 tokens;
the complete 15-step wall-latency vector is retained. Result:
`vllm_qwen3_4096_dp16_ep16_b3_20260716_233523/baseline-result.json`, SHA-256
`37dee56e4ea94653ecaf139b9a9870a7e5d38b575acb85eb774bb2fb39faddaf`.

The first normal-QoS b4/b3 attempts, jobs `5442910` and `5442911`, each
received nodes but exited `1:0` before workload execution with empty Slurm
logs. Diagnosis found the retained scripts' fail-fast job-side invariant still
required `SLURM_JOB_QOS=short`; the switch to `normal` therefore terminated
both before their first `srun`. The invariant was updated to require `normal`,
local/remote shell checks and dry runs passed, and a no-longer-needed pending
normal-QoS probe `5443075` was cancelled. Current control hashes are
`e28253d9be7bc84d33ababa83f30f43fdd4d844450274c212fb523b59a2f775a`
(AFD) and
`0ccb2840eb87648c16e0fdf2280bdc29e9a3a2e2930178773255279e1a8cba84`
(vLLM). Four distinct normal-QoS jobs were submitted at 23:49 PDT: vLLM
4K/b4 `5443106` in
`vllm_qwen3_4096_dp16_ep16_b4_20260716_234952`, AFD 4K/b3 `5443107` in
`afd_qwen3_4096_b3_n8_20260716_234953`, vLLM 4K/b6 `5443108` in
`vllm_qwen3_4096_dp16_ep16_b6_20260716_234953`, and AFD 4K/b4 `5443109` in
`afd_qwen3_4096_b4_n8_20260716_234953`. Exactly four jobs are eligible under
the launcher cap, initially pending on scheduler priority/resources rather
than a per-user QoS limit.

The corrected normal-QoS path was proven when jobs `5443106`, `5443107`, and
`5443108` ran concurrently for more than five minutes and AFD b4 job `5443109`
joined them as the fourth simultaneous job. vLLM 4K/b4 job `5443106` completed
`0:0` in 5:22. Its primary synchronized request result was **23.3301 ms median
decode-step latency and 171.4525 tok/s/GPU**, with 23.6863 ms mean and 168.8740
tok/s/GPU. Secondary CUDA-kernel throughput was 298.7901 tok/s/GPU. All 64
prompts were exactly 4,096 tokens. Result:
`vllm_qwen3_4096_dp16_ep16_b4_20260716_234952/baseline-result.json`, SHA-256
`67670cb5e58987942cf4f9b23c02a5a942e385229100ffbb128a4c6fa9bd6315`.

vLLM 4K/b6 job `5443108` completed `0:0` in 6:22. Its primary synchronized
request result was **26.1409 ms median decode-step latency and 229.5254
tok/s/GPU**. One 79.4302 ms sample raises the complete-series mean to 29.8627
ms and reduces mean throughput to 200.9195 tok/s/GPU. Secondary CUDA-kernel
throughput was 338.1980 tok/s/GPU. All 96 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b6_20260716_234953/baseline-result.json`,
SHA-256 `f48aeb16d5fcd92a0c94f5eea4fbeb8ddb2de68edf55f7111d1f613807f071f0`.
Their freed slots were filled at 00:01 PDT with vLLM 4K/b8 job `5443241` in
`vllm_qwen3_4096_dp16_ep16_b8_20260717_000124` and vLLM 4K/b12 job `5443242`
in `vllm_qwen3_4096_dp16_ep16_b12_20260717_000124`; together with AFD b3
completing and AFD b4 running, four distinct jobs remain active.

AFD 4K/b3 job `5443107` completed `0:0` in 8:04. All 84 exact prompts
returned 16 tokens. Its primary result was **16.7992 ms median decode interval
and 156.2573 tok/s/GPU**; the complete 14-interval vector has 16.0680 ms mean
and mean-derived throughput 163.3678 tok/s/GPU. Result:
`afd_qwen3_4096_b3_n8_20260716_234953/experiment/afd-result.json`, SHA-256
`5898a637ae343a3cdc14d659535e84e3a1e9cb2c306fb0b75f6bcb252ff0be63`;
aggregate prompt-token hash
`537d7237990e808c29c9e3f49d4306871d8275c909d57074a0f2817aafac6e96`.
Its freed slot was filled at 00:04 PDT with AFD 4K/b6 job `5443296`, run
directory `afd_qwen3_4096_b6_n8_20260717_000452`. Together with AFD b4 and
vLLM b8/b12, four distinct jobs are queued/running.

AFD 4K/b4 job `5443109` completed `0:0` in 7:40. All 112 exact prompts
returned 16 tokens. Its primary robust result was **16.1964 ms median decode
interval and 216.0977 tok/s/GPU**. A 227.3462 ms first interval raises the
complete-series mean to 30.3360 ms and reduces mean-derived throughput to
115.3746 tok/s/GPU; the full vector is retained. Result:
`afd_qwen3_4096_b4_n8_20260716_234953/experiment/afd-result.json`, SHA-256
`f5655afbd808bb6446472bb067452a37042826c30046b62427bcc3b2e4aedced`;
aggregate prompt-token hash
`f145cc46e1b756ef99c1c79846dd40f4f43e58d872f571e7c5f455789da77e69`.
Its slot was filled at 00:07 PDT with AFD 4K/b8 job `5443325`, run directory
`afd_qwen3_4096_b8_n8_20260717_000756`. Together with AFD b6 and vLLM b8/b12,
four distinct jobs are eligible and pending on normal-QoS priority/resources.

vLLM 4K/b8 job `5443241` completed `0:0` in 4:26. Its primary synchronized
request result was **24.2327 ms median decode-step latency and 330.1325
tok/s/GPU**, with 24.4032 ms mean and 327.8262 tok/s/GPU. Secondary CUDA-kernel
throughput was 579.3623 tok/s/GPU. All 128 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b8_20260717_000124/baseline-result.json`,
SHA-256 `a9501dd40dbff4374f3ec1639985177f870562001b03e5339a8d76c5aea6746d`.

vLLM 4K/b12 job `5443242` completed `0:0` in 5:13. Its primary synchronized
request result was **26.2528 ms median decode-step latency and 457.0942
tok/s/GPU**, with 26.1643 ms mean and 458.6404 tok/s/GPU. Secondary CUDA-kernel
throughput was 811.3394 tok/s/GPU. All 192 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b12_20260717_000124/baseline-result.json`,
SHA-256 `d5a7518b8842b1bb75d3fd53878ce0f1c82a9ae46a85e5dbf1aec22a429eb9d2`.
Their slots were filled at 00:14 PDT with vLLM 4K/b16 job `5443418` in
`vllm_qwen3_4096_dp16_ep16_b16_20260717_001404` and vLLM 4K/b24 job `5443419`
in `vllm_qwen3_4096_dp16_ep16_b24_20260717_001404`. Together with AFD b6/b8,
four distinct jobs are queued/running.

AFD 4K/b6 job `5443296` completed `0:0` in 8:10. All 168 exact prompts
returned 16 tokens. Its primary result was **16.4864 ms median decode interval
and 318.4437 tok/s/GPU**; the complete 14-interval vector has 15.4728 ms mean
and mean-derived throughput 339.3060 tok/s/GPU. Result:
`afd_qwen3_4096_b6_n8_20260717_000452/experiment/afd-result.json`, SHA-256
`0b7be5ba77e10e00b516c99f3d427df53dec35dfe3ffcabbbf78bbbc1ca76739`;
aggregate prompt-token hash
`454c2c65878c855dd5c43d7ad8edbdb81dac3f00e57414daa5c8d83bd967342b`.
Its slot was filled at 00:20 PDT with AFD 4K/b12 job `5443524`, run directory
`afd_qwen3_4096_b12_n8_20260717_002024`. Together with AFD b8 and vLLM
b16/b24, four distinct jobs are queued/running.

AFD 4K/b8 job `5443325` completed `0:0` in 4:35. All 224 exact prompts
returned 16 tokens. Its primary result was **17.0460 ms median decode interval
and 410.6544 tok/s/GPU**; the complete 14-interval vector has 16.3296 ms mean
and mean-derived throughput 428.6702 tok/s/GPU. Result:
`afd_qwen3_4096_b8_n8_20260717_000756/experiment/afd-result.json`, SHA-256
`9d3037709b14246005072087241752ce3a2b58178b15038c24a55ba956a4a3d5`;
aggregate prompt-token hash
`695d7911a6c1a4b04468247d00d95d9db5bd780a3158d67ed4e92d07aa774972`.

vLLM 4K/b16 job `5443418` completed `0:0` in 5:29. Its primary synchronized
request result was **26.5652 ms median decode-step latency and 602.2927
tok/s/GPU**, with 26.6052 ms mean and 601.3873 tok/s/GPU. Secondary CUDA-kernel
throughput was 1,029.9882 tok/s/GPU. All 256 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b16_20260717_001404/baseline-result.json`,
SHA-256 `19924a658d89cc26f4127c9c9a73b35aa907d1d9a77361d75a83c2d071d96530`.
Their slots were filled at 00:26 PDT with AFD 4K/b16 job `5443583` in
`afd_qwen3_4096_b16_n8_20260717_002645` and vLLM 4K/b32 job `5443584` in
`vllm_qwen3_4096_dp16_ep16_b32_20260717_002645`. Together with AFD b12 and
vLLM b24, four distinct jobs are queued under normal QoS.

vLLM 4K/b24 job `5443419` completed `0:0` in 6:50. Its primary synchronized
request result was **27.6264 ms median decode-step latency and 868.7332
tok/s/GPU**, with 27.9265 ms mean and 859.3975 tok/s/GPU. Secondary CUDA-kernel
throughput was 1,412.8388 tok/s/GPU. All 384 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b24_20260717_001404/baseline-result.json`,
SHA-256 `3bd76577668957ce5f65f8b6573155bc90ea871321455810e4457b88efa9fe87`.

vLLM 4K/b32 job `5443584` completed `0:0` in 6:00. Its primary synchronized
request result was **29.6480 ms median decode-step latency and 1,079.3316
tok/s/GPU**, with 29.9241 ms mean and 1,069.3709 tok/s/GPU. Secondary
CUDA-kernel throughput was 1,716.7678 tok/s/GPU. All 512 prompts were exactly
4,096 tokens. Result:
`vllm_qwen3_4096_dp16_ep16_b32_20260717_002645/baseline-result.json`, SHA-256
`818af8f70ef312fed1c4c299295605528ff37e429d5a130b7979a6b5d9c57674`.
Their slots were filled at 00:35 PDT with vLLM 4K/b48 job `5443649` in
`vllm_qwen3_4096_dp16_ep16_b48_20260717_003551` and vLLM 4K/b64 job `5443650`
in `vllm_qwen3_4096_dp16_ep16_b64_20260717_003551`. Together with AFD b12/b16,
four distinct jobs are queued/running under normal QoS.

AFD 4K/b12 job `5443524` completed `0:0` in 8:20. All 336 exact prompts
returned 16 tokens. Its primary result was **17.3301 ms median decode interval
and 605.8819 tok/s/GPU**; the complete 14-interval vector has 16.2594 ms mean
and mean-derived throughput 645.7815 tok/s/GPU. Result:
`afd_qwen3_4096_b12_n8_20260717_002024/experiment/afd-result.json`, SHA-256
`98928750cc3e265864aa2bd947ab4ce1d6dbb3edad80c809e146dbdf7d1c57dc`;
aggregate prompt-token hash
`da7a954366898d5a6e9e8c671b81cbeca4fe2695b19d692ab152ee4534ba87e5`.

AFD 4K/b16 job `5443583` completed `0:0` in 5:27. All 448 exact prompts
returned 16 tokens. Its primary result was **18.4868 ms median decode interval
and 757.2953 tok/s/GPU**; the complete 14-interval vector has 17.7586 ms mean
and mean-derived throughput 788.3517 tok/s/GPU. Result:
`afd_qwen3_4096_b16_n8_20260717_002645/experiment/afd-result.json`, SHA-256
`95725f45466b048ce7a57b5a06d03186899447547c7a5da8e188baac5ce64f79`;
aggregate prompt-token hash
`c6ff86d50512d7b132551bbfd2e5caab0fc1b7f85c62009c4b45e08d479f2650`.

The first b24/b32 launcher calls after these completions produced no jobs:
preflight correctly rejected the dirty default development checkout
`/home/shengjiel/scratch/github/FastAFD`. Retrying with the sweep's explicit
clean pinned source
`/home/shengjiel/scratch/fastafd_reproduce/source/FastAFD-3c716194` succeeded.
AFD 4K/b24 job `5443684` runs in
`afd_qwen3_4096_b24_n8_20260717_003903`; AFD 4K/b32 job `5443686` runs in
`afd_qwen3_4096_b32_n8_20260717_003903`. Together with vLLM b48/b64, four
distinct jobs are queued/running under normal QoS.

The launcher source defaults were then hardened to the pinned clean checkout
under `$FASTAFD_REPRO_ROOT/source/FastAFD-3c716194` (the environment override
remains supported), eliminating dependence on the mutable development
checkout. Local and remote `bash -n` plus representative AFD/vLLM dry runs
passed; the byte-identical local/remote control hashes are now
`d42f63b14d90fd19f5f77c8ea017331e366141bd731df467a7dde4891b371444`
(AFD) and
`b76a4c2ce284c53d1701ded3a07811e19d08a168f1a9e1fe90caab8e674e03e0`
(vLLM).

vLLM 4K/b48 job `5443649` completed `0:0` in 7:00. Its primary synchronized
request result was **33.7099 ms median decode-step latency and 1,423.9137
tok/s/GPU**. One 51.7757 ms sample raises the full-series mean to 35.0469 ms
and reduces mean throughput to 1,369.5940 tok/s/GPU. Secondary CUDA-kernel
throughput was 2,093.3474 tok/s/GPU. All 768 prompts were exactly 4,096 tokens.
Result: `vllm_qwen3_4096_dp16_ep16_b48_20260717_003551/baseline-result.json`,
SHA-256 `9bf155fd7d42ef214325fd20cb201cc398d82b4fd98762ed2c8450179a7a0da0`.

vLLM 4K/b64 job `5443650` completed `0:0` in 6:03. Its primary synchronized
request result was **36.1583 ms median decode-step latency and 1,769.9948
tok/s/GPU**, with 36.2251 ms mean and 1,766.7288 tok/s/GPU. Secondary
CUDA-kernel throughput was 2,617.4504 tok/s/GPU. All 1,024 prompts were exactly
4,096 tokens. Result:
`vllm_qwen3_4096_dp16_ep16_b64_20260717_003551/baseline-result.json`, SHA-256
`210f7517eb801ebae9f1e2131bb63f0f550f90108e491f70d728542c9b97edba`.
Their slots were filled at 00:45 PDT with vLLM 4K/b96 job `5443730` in
`vllm_qwen3_4096_dp16_ep16_b96_20260717_004541` and vLLM 4K/b128 job `5443731`
in `vllm_qwen3_4096_dp16_ep16_b128_20260717_004541`. Together with AFD b24/b32,
four distinct jobs are queued/running under normal QoS.

AFD 4K/b32 job `5443686` completed `0:0` in 9:42. All 896 exact prompts
returned 16 tokens. Its primary result was **20.7677 ms median decode interval
and 1,348.2492 tok/s/GPU**; the complete 14-interval vector has 19.7487 ms mean
and mean-derived throughput 1,417.8173 tok/s/GPU. Result:
`afd_qwen3_4096_b32_n8_20260717_003903/experiment/afd-result.json`, SHA-256
`b98f9b84cbfb6abf52b48485eef35e618f7ad3025e25d0b223d5aa8859ac2d75`;
aggregate prompt-token hash
`ff898f710215ced98902b4300482884f193b4ca041588f59a3743cfec84900c7`.
The first b48 replacement call was correctly held by the four-job guard while
Slurm still listed b32 as `COMPLETING`; no job was created. Once teardown
cleared, AFD 4K/b48 job `5443814` was submitted in
`afd_qwen3_4096_b48_n8_20260717_005157`. Together with AFD b24 and vLLM
b96/b128, four distinct jobs are queued/running.

AFD 4K/b24 job `5443684` completed `0:0` in 10:46. All 672 exact prompts
returned 16 tokens. Its primary result was **18.7622 ms median decode interval
and 1,119.2732 tok/s/GPU**; the complete 14-interval vector has 17.8566 ms mean
and mean-derived throughput 1,176.0376 tok/s/GPU. Result:
`afd_qwen3_4096_b24_n8_20260717_003903/experiment/afd-result.json`, SHA-256
`ec4b4614fe963a84f925995a09f46949128dddfde4e2822cbb08acab82e9b4d1`;
aggregate prompt-token hash
`fd900ebf0d46b97b4e6903705c446636a583bbf9def0e01ac4464fdfce747b82`.
AFD b48 job `5443814` was already queued to consume this released AFD slot.

vLLM 4K/b128 job `5443731` completed `0:0` in 7:19. Its primary synchronized
request result was **47.5354 ms median decode-step latency and 2,692.7306
tok/s/GPU**. Several early samples up to 62.8399 ms raise the complete-series
mean to 49.7936 ms and reduce mean throughput to 2,570.6139 tok/s/GPU.
Secondary CUDA-kernel throughput was 3,451.0037 tok/s/GPU. All 2,048 prompts
were exactly 4,096 tokens. Result:
`vllm_qwen3_4096_dp16_ep16_b128_20260717_004541/baseline-result.json`, SHA-256
`544cddacea9da01bf79532241fd0da5777a2ce8f4227722764f44962bcac7ffa`.
Its slot was filled at 00:54 PDT with the final planned 4K vLLM point, b192 job
`5443848`, in `vllm_qwen3_4096_dp16_ep16_b192_20260717_005455`.

vLLM 4K/b96 job `5443730` completed `0:0` in 8:24. Its primary synchronized
request result was **43.1104 ms median decode-step latency and 2,226.8430
tok/s/GPU**, with 43.4147 ms mean and 2,211.2310 tok/s/GPU. Secondary
CUDA-kernel throughput was 3,123.8396 tok/s/GPU. All 1,536 prompts were exactly
4,096 tokens. Result:
`vllm_qwen3_4096_dp16_ep16_b96_20260717_004541/baseline-result.json`, SHA-256
`3dcea749d2376367127ffec2cb4da1f353f206554862ef0c9f02d73cb56f6bed`.
The two newly open scheduler slots were filled at 00:56 PDT with AFD 4K/b64
job `5443863` in `afd_qwen3_4096_b64_n8_20260717_005628` and AFD 4K/b96 job
`5443864` in `afd_qwen3_4096_b96_n8_20260717_005628`. Together with AFD b48
and vLLM b192, four distinct jobs are queued/running.

vLLM 4K/b192 job `5443848` completed `0:0` in 7:31, completing every planned
4K vLLM batch. Its primary synchronized request result was **61.9714 ms median
decode-step latency and 3,098.2047 tok/s/GPU**, with 62.0710 ms mean and
3,093.2314 tok/s/GPU. Secondary CUDA-kernel throughput was 3,926.1318
tok/s/GPU. All 3,072 prompts were exactly 4,096 tokens, and this point exactly
matches the post-init KV-capacity ceiling of 192 prompts/rank. Result:
`vllm_qwen3_4096_dp16_ep16_b192_20260717_005455/baseline-result.json`, SHA-256
`4f843df2b48abbaeb014772ebd0d3d39a931655f4e9715dc0913550d3db134a9`.
Its slot was filled at 01:04 PDT with AFD 4K/b128 job `5444152` in
`afd_qwen3_4096_b128_n8_20260717_010404`.

AFD 4K/b48 job `5443814` completed `0:0` in 9:52. All 1,344 exact prompts
returned 16 tokens. Its primary result was **23.6079 ms median decode interval
and 1,779.0667 tok/s/GPU**; the complete 14-interval vector has 21.9553 ms mean
and mean-derived throughput 1,912.9766 tok/s/GPU. Result:
`afd_qwen3_4096_b48_n8_20260717_005157/experiment/afd-result.json`, SHA-256
`78a72065692523d3e22cd44acfa9a2a5570cd79715545e91cc707b820979e95b`;
aggregate prompt-token hash
`d17dff0afdd0016d2580680404e1da096d3c901cc8ff260d262b847bd4c0eea6`.
Its slot was filled at 01:04 PDT with the final planned 4K AFD point, b192 job
`5444170`, in `afd_qwen3_4096_b192_n8_20260717_010459`.

AFD 4K/b64 job `5443863` completed `0:0` in 10:12. All 1,792 exact prompts
returned 16 tokens. Its primary robust result was **26.5870 ms median decode
interval and 2,106.2945 tok/s/GPU**. A 5,803.9454 ms wave-boundary interval
raises the retained complete-vector mean to 437.1619 ms and reduces its
mean-derived throughput to 128.0990 tok/s/GPU; the full vector is preserved.
Result: `afd_qwen3_4096_b64_n8_20260717_005628/experiment/afd-result.json`,
SHA-256 `8273a9938046449613280315dee322cb55607dd968157eb304b45f8eccc291ea`;
aggregate prompt-token hash
`ef73a3a7e9a5ec89671481e15a56718988060629a424b14c19972bd68168a3f9`.
With AFD b128/b192 already queued, its released fourth slot began the next ISL
tier: vLLM 8K/b2 job `5444196` in
`vllm_qwen3_8192_dp16_ep16_b2_20260717_010815`.

AFD 4K/b96 job `5443864` completed `0:0` in 11:55. All 2,688 exact prompts
returned 16 tokens. Its primary robust result was **34.9694 ms median decode
interval and 2,402.1012 tok/s/GPU**. A 5,820.7923 ms wave-boundary interval
raises the retained complete-vector mean to 445.4818 ms and reduces its
mean-derived throughput to 188.5599 tok/s/GPU; the full vector is preserved.
Result: `afd_qwen3_4096_b96_n8_20260717_005628/experiment/afd-result.json`,
SHA-256 `8f800b7bf2b9d735c04f2f297342c4aada52699a2e19be394cdc5a7b6d47496c`;
aggregate prompt-token hash
`cf8499a5d19dbe32bddbefcaee04692c9bc6d8298a444b4a261582c26d43c3c9`.
After its `COMPLETING` entry cleared, the released fourth slot started AFD
8K/b2 job `5444321` in `afd_qwen3_8192_b2_n8_20260717_011024` alongside the
already running vLLM 8K/b2 job.

vLLM 8K/b2 job `5444196` completed `0:0` in 5:51. Its primary synchronized
request result was **24.9923 ms median decode-step latency and 80.0248
tok/s/GPU**, with 25.0120 ms mean and 79.9617 tok/s/GPU. Secondary CUDA-kernel
throughput was 150.9899 tok/s/GPU. All 32 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b2_20260717_010815/baseline-result.json`,
SHA-256 `d583d2f0463331510b1ce8896e7a1e18e24fb18c6e93ba99927fd7d1e6bdaa46`.
After teardown cleared, its slot advanced at 01:16 PDT to vLLM 8K/b3 job
`5444381` in `vllm_qwen3_8192_dp16_ep16_b3_20260717_011656`.

AFD 8K/b2 job `5444321` completed `0:0` in 4:31. All 56 exact prompts
returned 16 tokens. Its primary result was **16.3832 ms median decode interval
and 106.8168 tok/s/GPU**; the complete 14-interval vector has 15.5353 ms mean
and mean-derived throughput 112.6466 tok/s/GPU. Result:
`afd_qwen3_8192_b2_n8_20260717_011024/experiment/afd-result.json`, SHA-256
`7f418aa90129257a70babede7fbfb62685359ffe539512f54ba93ae37bd40bef`;
aggregate prompt-token hash
`a165b03c659ff897d31336b83f54661562ef018ffa9a95ac78a5fb05a0c9651c`.
After teardown cleared, its slot advanced at 01:20 PDT to AFD 8K/b3 job
`5444412` in `afd_qwen3_8192_b3_n8_20260717_012037`.

AFD 4K/b192 job `5444170` completed `0:0` in 13:18. All 5,376 exact prompts
returned 16 tokens. Its primary robust result was **51.1881 ms median decode
interval and 3,282.0123 tok/s/GPU**. A 5,757.8588 ms wave-boundary interval
raises the retained complete-vector mean to 454.7819 ms and reduces its
mean-derived throughput to 369.4079 tok/s/GPU; the full vector is preserved.
Result: `afd_qwen3_4096_b192_n8_20260717_010459/experiment/afd-result.json`,
SHA-256 `13bbc74af82e2af94843495c8c01b2ba657759211c4ac2050bfd6254d72722c5`;
aggregate prompt-token hash
`e74fb0c90e109c7249f0ccb42380a0016d353f82ce8a4e16ed05733d1dae8d48`.
Its slot advanced at 01:29 PDT to vLLM 8K/b4 job `5444541` in
`vllm_qwen3_8192_dp16_ep16_b4_20260717_012914`. Only AFD b128 remains from the
4K sweep tier.

vLLM 8K/b3 job `5444381` completed `0:0` in 6:19. Its primary synchronized
request result was **23.5652 ms median decode-step latency and 127.3062
tok/s/GPU**. A 31.0189 ms first sample raises the complete-series mean to
24.5002 ms and reduces mean throughput to 122.4479 tok/s/GPU. Secondary
CUDA-kernel throughput was 217.9765 tok/s/GPU. All 48 prompts were exactly
8,192 tokens. Result:
`vllm_qwen3_8192_dp16_ep16_b3_20260717_011656/baseline-result.json`, SHA-256
`9abc92ece33bbc873f9da2caa110bea8b8ca0615d767066a5bc9c7bd40227738`.
After teardown cleared, its slot advanced at 01:33 PDT to AFD 8K/b4 job
`5444603` in `afd_qwen3_8192_b4_n8_20260717_013314`.

AFD 8K/b4 job `5444603` completed `0:0` in 4:57. All 112 exact prompts
returned 16 tokens. Its primary result was **16.5783 ms median decode interval
and 211.1192 tok/s/GPU**; the complete 14-interval vector has 15.5447 ms mean
and mean-derived throughput 225.1576 tok/s/GPU. Result:
`afd_qwen3_8192_b4_n8_20260717_013314/experiment/afd-result.json`, SHA-256
`b4946bebe8966fabcba6c92b39580a9b2981b17979c6370c6bd8b19b39074e54`;
aggregate prompt-token hash
`7b4257e5c1f096212b74efad3757191719b8bb4b3715ed897cb10cc2a50340ec`.

The first AFD 4K/b128 attempt, job `5444152`, was cancelled after 20:54 with
no result and is not a performance point. Live logs had stopped at 01:24 after
a 300-second c10d rendezvous timeout: attention rank 4 on `nvl72099-T02` GPU 0
never joined while every other rank (0--3 and 5--31) timed out. After Slurm
cleared the cancelled allocation, b128 was retried with that node excluded as
job `5444668` in `afd_qwen3_4096_b128_n8_20260717_014128`.

AFD 8K/b3 job `5444412` completed `0:0` in 7:43. All 84 exact prompts
returned 16 tokens. Its primary result was **17.7552 ms median decode interval
and 147.8439 tok/s/GPU**; the complete 14-interval vector has 16.8032 ms mean
and mean-derived throughput 156.2207 tok/s/GPU. Result:
`afd_qwen3_8192_b3_n8_20260717_012037/experiment/afd-result.json`, SHA-256
`cb3b9df6ac7bf1c4cf17fa9c5d2debc9d2635fdb4151076180955ee3f2f098e7`;
aggregate prompt-token hash
`fe34daac4c05825ca23f5db57e2526b1d89f5f34c528abaaf652a8055b07e76e`.

vLLM 8K/b4 job `5444541` completed `0:0` in 5:22. Its primary synchronized
request result was **25.3091 ms median decode-step latency and 158.0459
tok/s/GPU**, with 25.4893 ms mean and 156.9284 tok/s/GPU. Secondary CUDA-kernel
throughput was 281.4103 tok/s/GPU. All 64 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b4_20260717_012914/baseline-result.json`,
SHA-256 `2184270469fa3c7c0c31160ae8d9b274b13298a683a400cb6dc23e4344b83ba9`.
Their freed capacity was filled at 01:42 PDT with AFD 8K/b6 job `5444683` in
`afd_qwen3_8192_b6_n8_20260717_014233`, vLLM 8K/b6 job `5444684` in
`vllm_qwen3_8192_dp16_ep16_b6_20260717_014233`, and vLLM 8K/b8 job `5444685`
in `vllm_qwen3_8192_dp16_ep16_b8_20260717_014233`. Together with the AFD 4K
b128 retry, four distinct jobs are queued/running.

AFD 8K/b6 job `5444683` completed `0:0` in 4:41. All 168 exact prompts
returned 16 tokens. Its primary result was **16.7440 ms median decode interval
and 313.5453 tok/s/GPU**; the complete 14-interval vector has 15.7892 ms mean
and mean-derived throughput 332.5068 tok/s/GPU. Result:
`afd_qwen3_8192_b6_n8_20260717_014233/experiment/afd-result.json`, SHA-256
`8d7dad1afa8db4738c21c38f407090809999e507930e888db445cf20fbad0ca9`;
aggregate prompt-token hash
`6d6170698ed7a5ded51f1d750b46d8a31282f27498634c93665f90ce3f99a1fa`.

vLLM 8K/b6 job `5444684` completed `0:0` in 5:21. Its primary synchronized
request result was **25.6475 ms median decode-step latency and 233.9409
tok/s/GPU**, with 25.7598 ms mean and 232.9212 tok/s/GPU. Secondary CUDA-kernel
throughput was 415.3536 tok/s/GPU. All 96 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b6_20260717_014233/baseline-result.json`,
SHA-256 `c68cd47c9eed399fbc14a28571640201e4ef1f2102e5a7a22b0f074ef709e617`.

vLLM 8K/b8 job `5444685` completed `0:0` in 4:33. Its primary synchronized
request result was **26.5536 ms median decode-step latency and 301.2775
tok/s/GPU**, with 26.6800 ms mean and 299.8499 tok/s/GPU. Secondary CUDA-kernel
throughput was 530.6865 tok/s/GPU. All 128 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b8_20260717_014233/baseline-result.json`,
SHA-256 `1665934b5f6e3ca398d1895e8aca079b66bb1e65d4bd4f4de4c2bbcf12a81110`.
Their three released slots were filled at 01:49 PDT with AFD 8K/b8 job
`5444733` in `afd_qwen3_8192_b8_n8_20260717_014937`, vLLM 8K/b12 job `5444734`
in `vllm_qwen3_8192_dp16_ep16_b12_20260717_014937`, and AFD 8K/b12 job
`5444735` in `afd_qwen3_8192_b12_n8_20260717_014937`. Together with the AFD
4K/b128 retry, four distinct jobs are queued/running.

AFD 4K/b128 retry job `5444668` completed `0:0` in 7:23, completing every
planned 4K AFD batch. All 3,584 exact prompts returned 16 tokens. Its primary
robust result was **40.1521 ms median decode interval and 2,789.3944
tok/s/GPU**. A 5,843.8338 ms wave-boundary interval raises the retained
complete-vector mean to 451.6440 ms and reduces its mean-derived throughput to
247.9829 tok/s/GPU; the full vector is preserved. Result:
`afd_qwen3_4096_b128_n8_20260717_014128/experiment/afd-result.json`, SHA-256
`a7e9c6d03064cee3b60c3e46e4cd0944eed86a638836e761830f09b0c40d1ad3`;
aggregate prompt-token hash
`d5019698e6faa1850fa9f70cb0b49c7a555d7a057b39b1f65074af25bf06c4b1`.
Its released slot advanced at 01:51 PDT to vLLM 8K/b16 job `5444748` in
`vllm_qwen3_8192_dp16_ep16_b16_20260717_015142`.

vLLM 8K/b12 job `5444734` completed `0:0` in 6:35. Its primary synchronized
request result was **26.6271 ms median decode-step latency and 450.6688
tok/s/GPU**, with 26.9910 ms mean and 444.5925 tok/s/GPU. Secondary CUDA-kernel
throughput was 739.9897 tok/s/GPU. All 192 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b12_20260717_014937/baseline-result.json`,
SHA-256 `8aeffd9f866fafcf10de0a9cee6a746777b4793b0034c327a3131de75cef2d0f`.

vLLM 8K/b16 job `5444748` completed `0:0` in 5:50. Its primary synchronized
request result was **28.4497 ms median decode-step latency and 562.3967
tok/s/GPU**, with 28.5219 ms mean and 560.9715 tok/s/GPU. Secondary CUDA-kernel
throughput was 921.7895 tok/s/GPU. All 256 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b16_20260717_015142/baseline-result.json`,
SHA-256 `e12d906ebf411d58f1d20490d7d6d7557fe32a96cff6619ffe7d33eeba4cc538`.
Their slots advanced at 02:03 PDT to vLLM 8K/b24 job `5444858` in
`vllm_qwen3_8192_dp16_ep16_b24_20260717_020309` and vLLM 8K/b32 job `5444859`
in `vllm_qwen3_8192_dp16_ep16_b32_20260717_020310`. Together with AFD b8/b12,
four distinct jobs are queued/running.

AFD 8K/b8 job `5444733` completed `0:0` in 8:11. All 224 exact prompts
returned 16 tokens. Its primary result was **18.1440 ms median decode interval
and 385.8022 tok/s/GPU**; the complete 14-interval vector has 17.0112 ms mean
and mean-derived throughput 411.4942 tok/s/GPU. Result:
`afd_qwen3_8192_b8_n8_20260717_014937/experiment/afd-result.json`, SHA-256
`012b005ff4ee7f680190a2f53abd9c490306aa1119910d7337acfa9f5d764510`;
aggregate prompt-token hash
`3bff941b312b8b64efed7bcd021aef2328d0b5e1b2113f3ac2f74e30e557762d`.

AFD 8K/b12 job `5444735` completed `0:0` in 8:51. All 336 exact prompts
returned 16 tokens. Its primary result was **17.6274 ms median decode interval
and 595.6633 tok/s/GPU**; the complete 14-interval vector has 16.5700 ms mean
and mean-derived throughput 633.6754 tok/s/GPU. Result:
`afd_qwen3_8192_b12_n8_20260717_014937/experiment/afd-result.json`, SHA-256
`823117ebfc5355198c225143b2208395f2fee2f9c0b645566e81e7de043033aa`;
aggregate prompt-token hash
`3aad5654c0569f91a59320a3c24c569362bfc72419b343612f9ef6ea14595a73`.
Their AFD slots advanced as teardown permitted to AFD 8K/b16 job `5444891` in
`afd_qwen3_8192_b16_n8_20260717_020533` and AFD 8K/b24 job `5444895` in
`afd_qwen3_8192_b24_n8_20260717_020627`. Together with vLLM b24/b32, four
distinct jobs are queued/running.

vLLM 8K/b32 job `5444859` completed `0:0` in 4:43. Its primary synchronized
request result was **33.9574 ms median decode-step latency and 942.3577
tok/s/GPU**. A 156.2617 ms first sample raises the complete-series mean to
41.9524 ms and reduces mean throughput to 762.7684 tok/s/GPU. Secondary
CUDA-kernel throughput was 1,454.4532 tok/s/GPU. All 512 prompts were exactly
8,192 tokens. Result:
`vllm_qwen3_8192_dp16_ep16_b32_20260717_020310/baseline-result.json`, SHA-256
`76b37f30cb9a714977e91068ea11f6b66ac5e33a0a520e43b4b4c890cc863681`.
After teardown cleared, its slot advanced at 02:09 PDT to vLLM 8K/b48 job
`5444962` in `vllm_qwen3_8192_dp16_ep16_b48_20260717_020947`.

vLLM 8K/b24 job `5444858` completed `0:0` in 6:57. Its primary synchronized
request result was **31.8106 ms median decode-step latency and 754.4649
tok/s/GPU**, with 31.7656 ms mean and 755.5346 tok/s/GPU. Secondary CUDA-kernel
throughput was 1,209.2318 tok/s/GPU. All 384 prompts were exactly 8,192 tokens.
Result: `vllm_qwen3_8192_dp16_ep16_b24_20260717_020309/baseline-result.json`,
SHA-256 `52e56f0846ae0102a4088d6cfcf3df93c0d668faceba86dd2cf49009153879f6`.
After teardown cleared, its slot advanced at 02:12 PDT to vLLM 8K/b64 job
`5445007` in `vllm_qwen3_8192_dp16_ep16_b64_20260717_021234`.

AFD 8K/b16 job `5444891` completed `0:0` in 6:16. All 448 exact prompts
returned 16 tokens. Its primary result was **18.6142 ms median decode interval
and 752.1150 tok/s/GPU**; the complete 14-interval vector has 17.6096 ms mean
and mean-derived throughput 795.0226 tok/s/GPU. Result:
`afd_qwen3_8192_b16_n8_20260717_020533/experiment/afd-result.json`, SHA-256
`5a2a9c509c8830ac6ac313b4316604d4d24710204075eb656e5213831950a754`;
aggregate prompt-token hash
`0adb3bfa51c68af6e3a1264fbd241fc3fa022e5fd8f4322ac4fc901d041e70a0`.
After teardown cleared, its slot advanced at 02:15 PDT to AFD 8K/b32 job
`5445032` in `afd_qwen3_8192_b32_n8_20260717_021535`.

AFD 8K/b24 job `5444895` completed `0:0` in 9:51. All 672 exact prompts
returned 16 tokens. Its primary result was **19.5534 ms median decode interval
and 1,073.9831 tok/s/GPU**; the complete 14-interval vector has 18.0469 ms mean
and mean-derived throughput 1,163.6329 tok/s/GPU. Result:
`afd_qwen3_8192_b24_n8_20260717_020627/experiment/afd-result.json`, SHA-256
`05db3916fa88cf8577b6e31115843601d546cbb04af2034df9e1027da1d64d74`;
aggregate prompt-token hash
`7b2ebaebcbb54796be67ddef9cb26c13725b9044ba6d460acae7bc1d0f054e97`.

vLLM 8K/b48 job `5444962` completed `0:0` in 7:24. Its primary synchronized
request result was **40.7420 ms median decode-step latency and 1,178.1462
tok/s/GPU**, with 40.7450 ms mean and 1,178.0574 tok/s/GPU. Secondary
CUDA-kernel throughput was 1,692.4348 tok/s/GPU. All 768 prompts were exactly
8,192 tokens. Result:
`vllm_qwen3_8192_dp16_ep16_b48_20260717_020947/baseline-result.json`, SHA-256
`6d930606a6de1a433cff211a816c959a3c394a31a79c9cfb8e1eaaf49f2041c4`.
Their slots advanced at 02:19 PDT to AFD 8K/b48 job `5445051` in
`afd_qwen3_8192_b48_n8_20260717_021917` and the final planned 8K vLLM point,
b96 job `5445052`, in `vllm_qwen3_8192_dp16_ep16_b96_20260717_021917`.

vLLM 8K/b64 job `5445007` completed `0:0` in 6:33. Its primary synchronized
request result was **42.7650 ms median decode-step latency and 1,496.5509
tok/s/GPU**, with 42.7830 ms mean and 1,495.9215 tok/s/GPU. Secondary
CUDA-kernel throughput was 2,051.1154 tok/s/GPU. All 1,024 prompts were exactly
8,192 tokens. Result:
`vllm_qwen3_8192_dp16_ep16_b64_20260717_021234/baseline-result.json`, SHA-256
`deea126165b3ecbc14ac82ca32b092fb74a718d43a997c1a4ae7857fbc479e3e`.
Its slot advanced at 02:21 PDT to AFD 8K/b64 job `5445071` in
`afd_qwen3_8192_b64_n8_20260717_022146`.

AFD 8K/b48 job `5445051` completed `0:0` in 7:02. All 1,344 exact prompts
returned 16 tokens. Its primary result was **24.4294 ms median decode interval
and 1,719.2374 tok/s/GPU**; the complete 14-interval vector has 23.2863 ms mean
and mean-derived throughput 1,803.6333 tok/s/GPU. Result:
`afd_qwen3_8192_b48_n8_20260717_021917/experiment/afd-result.json`, SHA-256
`2eaf2f9f8d8cc8c7c852dfbfe0b7920b556fd347d71198d6793d6b7b53976201`;
aggregate prompt-token hash
`b9cac0e8976594c4a064e19cce4bd0e8e30efcb0791149461441a0518d6723c1`.

vLLM 8K/b96 job `5445052` completed `0:0` in 7:14, completing every planned
8K vLLM batch and exactly reaching its measured KV-capacity ceiling of 96
prompts/rank. Its primary synchronized request result was **53.1609 ms median
decode-step latency and 1,805.8390 tok/s/GPU**, with 53.9274 ms mean and
1,780.1726 tok/s/GPU. Secondary CUDA-kernel throughput was 2,322.8518
tok/s/GPU. All 1,536 prompts were exactly 8,192 tokens. Result:
`vllm_qwen3_8192_dp16_ep16_b96_20260717_021917/baseline-result.json`, SHA-256
`4148f47d4791fbf9342dba7e7c2dc4d2bfe1050b91250d591486a0c1c6a98317`.

The first AFD 8K/b32 attempt, job `5445032`, failed `1:0` after 11:13 and is
not a performance point. Generation itself completed for all 896 requests and
all 28 attention-worker logs proved their target replay, but strict metric
validation found `alignment_json_missing`, zero coordinator schedule steps,
and no alignment phases, so no result JSON was written. It was retried away
from head node `nvl72051-T01` as job `5445156` in
`afd_qwen3_8192_b32_n8_20260717_023046`. The other free slots launched final
AFD 8K/b96 job `5445157` in `afd_qwen3_8192_b96_n8_20260717_023046` and began
the 16K tier with vLLM 16K/b2 job `5445158` in
`vllm_qwen3_16384_dp16_ep16_b2_20260717_023046`. Together with AFD 8K/b64,
four distinct jobs are queued/running.

AFD 8K/b64 job `5445071` completed `0:0` in 8:57. All 1,792 exact prompts
returned 16 tokens. Its primary robust result was **28.2890 ms median decode
interval and 1,979.5699 tok/s/GPU**. A 5,830.3264 ms wave-boundary interval
raises the retained complete-vector mean to 440.3254 ms and reduces its
mean-derived throughput to 127.1787 tok/s/GPU; the full vector is preserved.
Result: `afd_qwen3_8192_b64_n8_20260717_022146/experiment/afd-result.json`,
SHA-256 `acc87c6570f7aebc34851315960d41a9ae9f644243816e240142887bd28a0ec1`;
aggregate prompt-token hash
`039308daa7e9110ecb13a0d5a6672e7b3dde4e2e20ce7641d3d8c73ffa962260`.
Its slot advanced at 02:32 PDT to AFD 16K/b2 job `5445207` in
`afd_qwen3_16384_b2_n8_20260717_023237`, pairing the pending vLLM 16K/b2 case.

AFD 8K/b32 retry job `5445156` completed `0:0` in 9:22. All 896 exact prompts
returned 16 tokens, and the same aggregate prompt-token hash as the failed
attempt proves identical input. Its primary result was **21.1003 ms median
decode interval and 1,326.9967 tok/s/GPU**; the complete 14-interval vector has
19.6536 ms mean and mean-derived throughput 1,424.6730 tok/s/GPU. Result:
`afd_qwen3_8192_b32_n8_20260717_023046/experiment/afd-result.json`, SHA-256
`b313ec68be547825479e480bac312d2df8e0277b2dd2c5cd1325c8158db28848`;
aggregate prompt-token hash
`ac8e1a9444570655f3d5d43c41b2fb96ac929a8d6f4242fc6afc3f4d2e59ae4f`.

AFD 16K/b2 job `5445207` completed `0:0` in 4:29. All 56 exact prompts
returned 16 tokens. Its primary result was **17.6895 ms median decode interval
and 98.9285 tok/s/GPU**; the complete 14-interval vector has 16.9509 ms mean
and mean-derived throughput 103.2391 tok/s/GPU. Result:
`afd_qwen3_16384_b2_n8_20260717_023237/experiment/afd-result.json`, SHA-256
`cef6bd3ea308cbb6583193bcb93ed22d84d82c92f1fbc5fb1057d25163eade2e`;
aggregate prompt-token hash
`4251642a9b7c38023aabc5ecccf8e6d42a034c2af76c993f0c1784d23810042d`.

vLLM 16K/b2 job `5445158` completed `0:0` in 6:19. Its primary synchronized
request result was **24.8746 ms median decode-step latency and 80.4032
tok/s/GPU**. A 41.4279 ms first sample raises the complete-series mean to
26.0437 ms and reduces mean throughput to 76.7940 tok/s/GPU. Secondary
CUDA-kernel throughput was 143.2767 tok/s/GPU. All 32 prompts were exactly
16,384 tokens. Result:
`vllm_qwen3_16384_dp16_ep16_b2_20260717_023046/baseline-result.json`, SHA-256
`51216ac1773d5e1234a510efb2776e425c7d1a7bc0b6afc5f127f6da834696e6`.
The three open slots advanced at 02:42 PDT to AFD 16K/b3 job `5445282` in
`afd_qwen3_16384_b3_n8_20260717_024243`, vLLM 16K/b3 job `5445283` in
`vllm_qwen3_16384_dp16_ep16_b3_20260717_024243`, and vLLM 16K/b4 job `5445284`
in `vllm_qwen3_16384_dp16_ep16_b4_20260717_024243`. Together with AFD 8K/b96,
four distinct jobs are queued/running.

AFD 8K/b96 job `5445157` completed `0:0` in 13:58, completing every planned
8K AFD batch. All 2,688 exact prompts returned 16 tokens. Its primary robust
result was **34.0722 ms median decode interval and 2,465.3500 tok/s/GPU**. A
5,879.8736 ms wave-boundary interval raises the retained complete-vector mean
to 449.1616 ms and reduces its mean-derived throughput to 187.0151 tok/s/GPU;
the full vector is preserved. Result:
`afd_qwen3_8192_b96_n8_20260717_023046/experiment/afd-result.json`, SHA-256
`9a93e56570e7ac7cfdd2391ba4a338b86749b913aa7329baf14b8eedfea3ac72`;
aggregate prompt-token hash
`70b04add59c7915d69ba866f46c8db90828feb91015bb76644c1f192f8736d76`.
After teardown cleared, its slot advanced at 02:46 PDT to AFD 16K/b4 job
`5445334` in `afd_qwen3_16384_b4_n8_20260717_024650`.

vLLM 16K/b3 job `5445283` completed `0:0` in 6:15. Its primary synchronized
request result was **25.8932 ms median decode-step latency and 115.8605
tok/s/GPU**, with 26.0457 ms mean and 115.1824 tok/s/GPU. Secondary CUDA-kernel
throughput was 207.8607 tok/s/GPU. All 48 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b3_20260717_024243/baseline-result.json`,
SHA-256 `ff64e71a1a3a1d42a8e9f0dcb17f0327d5d2d05e7da10ebf5c8e60e6d1aca540`.

vLLM 16K/b4 job `5445284` completed `0:0` in 5:37. Its primary synchronized
request result was **26.2944 ms median decode-step latency and 152.1234
tok/s/GPU**, with 26.6580 ms mean and 150.0487 tok/s/GPU. Secondary CUDA-kernel
throughput was 260.0757 tok/s/GPU. All 64 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b4_20260717_024243/baseline-result.json`,
SHA-256 `e558d2b4241a366db214dc4bf2c5c6532b5583918fa466fe987bd992ac926cf0`.

The first AFD 16K/b3 attempt, job `5445282`, failed `1:0` after 4:41 and is
not a performance point. All 84 requests generated 16 tokens and all 28
attention workers completed their target replay, but strict validation found
`alignment_json_missing`, zero coordinator schedule steps, and no phases; no
result JSON was written. This repeats the artifact-finalization defect seen on
the first 8K/b32 attempt, now with head node `nvl72051-T02`. The identical case
was resubmitted as job `5445360` in
`afd_qwen3_16384_b3_n8_20260717_025051`, excluding both known bad head nodes
`nvl72051-T01,T02`. The two completed vLLM slots advanced at 02:51 PDT to
vLLM 16K/b6 job `5445368` in
`vllm_qwen3_16384_dp16_ep16_b6_20260717_025153` and vLLM 16K/b8 job `5445369`
in `vllm_qwen3_16384_dp16_ep16_b8_20260717_025157`. Together with AFD 16K/b4,
four distinct cases are queued/running.

AFD 16K/b4 job `5445334` completed `0:0` in 7:58. All 112 exact prompts
returned 16 tokens. Its primary result was **16.8569 ms median decode interval
and 207.6305 tok/s/GPU**; the complete 14-interval vector has 15.9480 ms mean
and mean-derived throughput 219.4631 tok/s/GPU. Result:
`afd_qwen3_16384_b4_n8_20260717_024650/experiment/afd-result.json`, SHA-256
`eb9ec56b226e59d7625e3e7806498186d1d922fb8d7c0c7193fa75ae1e38264c`;
aggregate prompt-token hash
`d25f155fd3e217adccd3b5f87cfcab99872b23da73c5b7a374292c845ac57584`.

vLLM 16K/b8 job `5445369` completed `0:0` in 4:22. Its primary synchronized
request result was **28.3939 ms median decode-step latency and 281.7510
tok/s/GPU**, with 28.5878 ms mean and 279.8396 tok/s/GPU. Secondary CUDA-kernel
throughput was 471.8225 tok/s/GPU. All 128 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b8_20260717_025157/baseline-result.json`,
SHA-256 `024e49c43b258d11aa5054194cecff0724ae944cfd967ecefa343fe1d9a775e1`.
Their slots advanced at 02:57 PDT to AFD 16K/b6 job `5445423` in
`afd_qwen3_16384_b6_n8_20260717_025754` and AFD 16K/b8 job `5445424` in
`afd_qwen3_16384_b8_n8_20260717_025759`; both retain the known-bad head-node
exclusion.

AFD 16K/b3 retry job `5445360` completed `0:0` in 7:10. All 84 exact prompts
returned 16 tokens. Its primary result was **17.4499 ms median decode interval
and 150.4305 tok/s/GPU**; the complete 14-interval vector has 16.6491 ms mean
and mean-derived throughput 157.6661 tok/s/GPU. Result:
`afd_qwen3_16384_b3_n8_20260717_025051/experiment/afd-result.json`, SHA-256
`b9ae764c9b50de1af5f426c356a6486c3ec79980c73c65db927b07daca6fa69f`;
aggregate prompt-token hash
`096f9655821b986aae58ed9180158d87b3e871221330e3d4bc45d25d8b4007c7`.

vLLM 16K/b6 job `5445368` completed `0:0` in 6:18. Its primary synchronized
request result was **25.6618 ms median decode-step latency and 233.8108
tok/s/GPU**, with 26.1359 ms mean and 229.5692 tok/s/GPU. Secondary CUDA-kernel
throughput was 385.2681 tok/s/GPU. All 96 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b6_20260717_025153/baseline-result.json`,
SHA-256 `66c3babe47438082e362bc64abfb6b50ac65a54b577325d49f8bc056afb8f0b1`.
Their slots advanced at 03:01 PDT to vLLM 16K/b12 job `5445470` in
`vllm_qwen3_16384_dp16_ep16_b12_20260717_030123` and vLLM 16K/b16 job
`5445471` in `vllm_qwen3_16384_dp16_ep16_b16_20260717_030126`.

AFD 16K/b8 job `5445424` completed `0:0` in 5:13. All 224 exact prompts
returned 16 tokens. Its primary result was **17.9395 ms median decode interval
and 390.2004 tok/s/GPU**; the complete 14-interval vector has 17.7058 ms mean
and mean-derived throughput 395.3507 tok/s/GPU. Result:
`afd_qwen3_16384_b8_n8_20260717_025759/experiment/afd-result.json`, SHA-256
`b858ea470e168b3d63937dd0654abb73398759e465902fb7b89d398b49a156de`;
aggregate prompt-token hash
`0fa25ff1b47d906c0f2ad13d0413898229e992d7408cbe23f6989c9ea52ab6d6`.
Its slot advanced at 03:05 PDT to AFD 16K/b12 job `5445542` in
`afd_qwen3_16384_b12_n8_20260717_030500`, retaining the known-bad head-node
exclusion.

AFD 16K/b6 job `5445423` completed `0:0` in 8:10. The result artifact became
visible after a short filesystem-propagation delay and passed the result-level
contract: all 168 exact prompts returned 16 tokens, two complete 15-step
full-bucket waves were identified, and every attention-worker replay was
proven. Its primary result was **17.9955 ms median decode interval and 291.7402
tok/s/GPU**; the complete 14-interval vector has 16.8382 ms mean and
mean-derived throughput 311.7910 tok/s/GPU. Result:
`afd_qwen3_16384_b6_n8_20260717_025754/experiment/afd-result.json`, SHA-256
`07806a57cc1487acde987e4353a0fc3835b9fbe27e2b2f67b69ce78828fc6c81`;
aggregate prompt-token hash
`841344c3303282b7c2ae4994327714d4ace81ea7b6c0113a4f6690f9f13d99c5`.
Its slot advanced at 03:08 PDT to AFD 16K/b16 job `5445569` in
`afd_qwen3_16384_b16_n8_20260717_030853`, retaining the known-bad head-node
exclusion.

vLLM 16K/b16 job `5445471` completed `0:0` in 4:59. Its primary synchronized
request result was **32.7490 ms median decode-step latency and 488.5651
tok/s/GPU**, with 32.6997 ms mean and 489.3017 tok/s/GPU. Secondary CUDA-kernel
throughput was 762.3474 tok/s/GPU. All 256 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b16_20260717_030126/baseline-result.json`,
SHA-256 `4cf61e0fc4be849c0091cd03f07b74f65461d4b8ca4192d3acbf0a9a8164ecba`.
Its slot advanced at 03:14 PDT to vLLM 16K/b24 job `5445632` in
`vllm_qwen3_16384_dp16_ep16_b24_20260717_031414`.

vLLM 16K/b12 job `5445470` completed `0:0` in 6:44. Its primary synchronized
request result was **29.3730 ms median decode-step latency and 408.5390
tok/s/GPU**. One 229.3825 ms step raises the complete-series mean to 43.2220 ms
and reduces mean throughput to 277.6362 tok/s/GPU. Secondary CUDA-kernel
throughput was 638.6397 tok/s/GPU. All 192 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b12_20260717_030123/baseline-result.json`,
SHA-256 `9d69aa4983cb25f1ff3b9a55ae267dccf362ab99bd6b01b92038141e152b82c4`.

AFD 16K/b12 job `5445542` completed `0:0` in 8:18. All 336 exact prompts
returned 16 tokens. Its primary result was **18.0756 ms median decode interval
and 580.8947 tok/s/GPU**; the complete 14-interval vector has 16.9311 ms mean
and mean-derived throughput 620.1618 tok/s/GPU. Result:
`afd_qwen3_16384_b12_n8_20260717_030500/experiment/afd-result.json`, SHA-256
`299b571a47e6c355cd9f57dc420266764584eab672058a94d2075fafc04d2c99`;
aggregate prompt-token hash
`692daade35617e2fcdafd07e7cc2f659241f033c262ba8bf7dd2070caf2c753f`.
Their slots advanced at 03:17 PDT to AFD 16K/b24 job `5445677` in
`afd_qwen3_16384_b24_n8_20260717_031726`, retaining the known-bad head-node
exclusion, and vLLM 16K/b32 job `5445678` in
`vllm_qwen3_16384_dp16_ep16_b32_20260717_031729`.

AFD 16K/b16 job `5445569` completed `0:0` in 7:21. All 448 exact prompts
returned 16 tokens. Its primary result was **19.8289 ms median decode interval
and 706.0407 tok/s/GPU**; the complete 14-interval vector has 18.7632 ms mean
and mean-derived throughput 746.1416 tok/s/GPU. Result:
`afd_qwen3_16384_b16_n8_20260717_030853/experiment/afd-result.json`, SHA-256
`d94f510cb10de6263f4087a29fab6a67133c461325e63ca1508521c01ae38631`;
aggregate prompt-token hash
`4c6a531b88d1e040dacb6db657d4264f517e3f0ebc8882fe32220e6ee76b8ee7`.
Its slot advanced at 03:20 PDT to AFD 16K/b32 job `5445703` in
`afd_qwen3_16384_b32_n8_20260717_032041`, retaining the known-bad head-node
exclusion.

vLLM 16K/b24 job `5445632` completed `0:0` in 6:42. Its primary synchronized
request result was **36.2800 ms median decode-step latency and 661.5208
tok/s/GPU**, with 37.1449 ms mean and 646.1191 tok/s/GPU. Secondary CUDA-kernel
throughput was 965.6789 tok/s/GPU. All 384 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b24_20260717_031414/baseline-result.json`,
SHA-256 `05aa9707a01d2fed17465880c7598860400fcdc79469d2d7b89b13e3cf73b2f3`.
Its slot advanced at 03:23 PDT to the final planned 16K vLLM point, b48 job
`5445732` in `vllm_qwen3_16384_dp16_ep16_b48_20260717_032339`.

vLLM 16K/b32 job `5445678` completed `0:0` in 6:50. Its primary synchronized
request result was **39.6243 ms median decode-step latency and 807.5857
tok/s/GPU**, with 39.7385 ms mean and 805.2647 tok/s/GPU. Secondary CUDA-kernel
throughput was 1,124.9480 tok/s/GPU. All 512 prompts were exactly 16,384 tokens.
Result: `vllm_qwen3_16384_dp16_ep16_b32_20260717_031729/baseline-result.json`,
SHA-256 `74a4957c261f5a0b34718b86819ac4ee041e8bf3dc824d8212711ee7bf7f65b6`.
Its slot advanced at 03:26 PDT to the final planned 16K AFD point, b48 job
`5445758` in `afd_qwen3_16384_b48_n8_20260717_032649`, retaining the known-bad
head-node exclusion.

AFD 16K/b24 job `5445677` completed `0:0` in 11:07. All 672 exact prompts
returned 16 tokens. Its primary result was **21.1217 ms median decode interval
and 994.2383 tok/s/GPU**; the complete 14-interval vector has 19.6811 ms mean
and mean-derived throughput 1,067.0112 tok/s/GPU. Result:
`afd_qwen3_16384_b24_n8_20260717_031726/experiment/afd-result.json`, SHA-256
`b80750127a0e0a9ce8fc698658a13668c754b56d004736f684a6b8671b439e7c`;
aggregate prompt-token hash
`ecb2d008db61453b846020201b717e6e062277df623f032c69adbb42086440ac`.
Its slot began the 32K tier at 03:32 PDT with vLLM 32K/b2 job `5445835` in
`vllm_qwen3_32768_dp16_ep16_b2_20260717_033216`.

AFD 16K/b32 job `5445703` completed `0:0` in 8:05. All 896 exact prompts
returned 16 tokens. Its primary result was **22.3979 ms median decode interval
and 1,250.1166 tok/s/GPU**; the complete 14-interval vector has 20.8728 ms mean
and mean-derived throughput 1,341.4602 tok/s/GPU. Result:
`afd_qwen3_16384_b32_n8_20260717_032041/experiment/afd-result.json`, SHA-256
`e21f2358ac7c689b12709d595d7d182e3ca34b5f77e335c300ea8f7f62f9eb68`;
aggregate prompt-token hash
`4df3a40315a21042add71438552f3bcab06c8848b5da967a64d7406261964d0c`.
Its slot paired the first 32K point at 03:35 PDT with AFD 32K/b2 job `5445872`
in `afd_qwen3_32768_b2_n8_20260717_033530`, retaining the known-bad head-node
exclusion.

vLLM 16K/b48 job `5445732` completed `0:0` in 9:01, completing every planned
16K vLLM batch. Its primary synchronized request result was **52.4121 ms median
decode-step latency and 915.8197 tok/s/GPU**, with 52.9596 ms mean and 906.3504
tok/s/GPU. Secondary CUDA-kernel throughput was 1,180.2113 tok/s/GPU. All 768
prompts were exactly 16,384 tokens. Result:
`vllm_qwen3_16384_dp16_ep16_b48_20260717_032339/baseline-result.json`, SHA-256
`feaa37f939db3909333a3c5c3ba1f96e7cd29a33d70cb0dc33a5209e46f90a6e`.
Its slot advanced at 03:36 PDT to vLLM 32K/b3 job `5445886` in
`vllm_qwen3_32768_dp16_ep16_b3_20260717_033647`.

AFD 16K/b48 job `5445758` completed `0:0` in 10:32, completing every planned
16K AFD batch. All 1,344 exact prompts returned 16 tokens. Its primary result
was **26.7928 ms median decode interval and 1,567.5856 tok/s/GPU**; the complete
14-interval vector has 24.8539 ms mean and mean-derived throughput 1,689.8785
tok/s/GPU. Result:
`afd_qwen3_16384_b48_n8_20260717_032649/experiment/afd-result.json`, SHA-256
`1f9665ac00150950407fecf04a555706d533fbd425827b7406cd535ffe6994e9`;
aggregate prompt-token hash
`2869645d8d0719d8f27e7b3326038781c8e386fe2d1c07181d3ca453605bf46c`.
Its slot advanced at 03:40 PDT to AFD 32K/b3 job `5445910` in
`afd_qwen3_32768_b3_n8_20260717_034015`, retaining the known-bad head-node
exclusion. Both implementations' planned 16K sweeps are now complete.

vLLM 32K/b2 job `5445835` completed `0:0` in 6:16. Its primary synchronized
request result was **26.3364 ms median decode-step latency and 75.9405
tok/s/GPU**. A 38.1027 ms first sample raises the complete-series mean to
26.9276 ms and reduces mean throughput to 74.2732 tok/s/GPU. Secondary
CUDA-kernel throughput was 131.7529 tok/s/GPU. All 32 prompts were exactly
32,768 tokens. Result:
`vllm_qwen3_32768_dp16_ep16_b2_20260717_033216/baseline-result.json`, SHA-256
`fc93504005cf1a1927e4e4a165f5d4eda4149eadd85ceea686ca557d3e483c19`.
Its slot advanced at 03:43 PDT to vLLM 32K/b4 job `5445938` in
`vllm_qwen3_32768_dp16_ep16_b4_20260717_034324`.

vLLM 32K/b3 job `5445886` completed `0:0` in 5:21. Its primary synchronized
request result was **26.2765 ms median decode-step latency and 114.1703
tok/s/GPU**, with 26.5560 ms mean and 112.9690 tok/s/GPU. Secondary CUDA-kernel
throughput was 191.7374 tok/s/GPU. All 48 prompts were exactly 32,768 tokens.
Result: `vllm_qwen3_32768_dp16_ep16_b3_20260717_033647/baseline-result.json`,
SHA-256 `79c85d0fbe970efcd9805c3559560f58577a86b4eb2cbf0b01d57bb724c55139`.
Its slot advanced at 03:44 PDT to AFD 32K/b4 job `5445949` in
`afd_qwen3_32768_b4_n8_20260717_034415`, retaining the known-bad head-node
exclusion.

AFD 32K/b2 job `5445872` completed `0:0` in 7:54. All 56 exact prompts
returned 16 tokens. Its primary result was **17.3567 ms median decode interval
and 100.8257 tok/s/GPU**; the complete 14-interval vector has 16.2344 ms mean
and mean-derived throughput 107.7956 tok/s/GPU. Result:
`afd_qwen3_32768_b2_n8_20260717_033530/experiment/afd-result.json`, SHA-256
`4605977995637c126a3a83f8c0f457382278c2b520e542482bcee85d633f3bec`;
aggregate prompt-token hash
`74b05d5e42bab2d677066c6d088f0142d5cd5133c84f0caa1424fad1894f488e`.
Its slot advanced at 03:47 PDT to vLLM 32K/b6 job `5445988` in
`vllm_qwen3_32768_dp16_ep16_b6_20260717_034730`.

AFD 32K/b3 job `5445910` completed `0:0` in 7:52. All 84 exact prompts
returned 16 tokens. Its primary result was **17.5153 ms median decode interval
and 149.8692 tok/s/GPU**; the complete 14-interval vector has 16.2876 ms mean
and mean-derived throughput 161.1659 tok/s/GPU. Result:
`afd_qwen3_32768_b3_n8_20260717_034015/experiment/afd-result.json`, SHA-256
`5b459c289bb6e32e5e1910eb29cccbece6a92e19cbe37c6ef4d57d8cbd9d24a7`;
aggregate prompt-token hash
`9a3f584e57a8c48d36830f24aa759e65776013c44e1f2b2b1b02efd7b83e5942`.

vLLM 32K/b4 job `5445938` completed `0:0` in 4:29. Its primary synchronized
request result was **27.3393 ms median decode-step latency and 146.3096
tok/s/GPU**, with 27.4615 ms mean and 145.6586 tok/s/GPU. Secondary CUDA-kernel
throughput was 240.9564 tok/s/GPU. All 64 prompts were exactly 32,768 tokens.
Result: `vllm_qwen3_32768_dp16_ep16_b4_20260717_034324/baseline-result.json`,
SHA-256 `eee9d1f6be5a1ed6bcacb64d6af71fb456c7ae3c5b751b0261ab1192677af1e4`.
Their slots advanced at 03:50 PDT to AFD 32K/b6 job `5446015` in
`afd_qwen3_32768_b6_n8_20260717_035053`, retaining the known-bad head-node
exclusion, and vLLM 32K/b8 job `5446017` in
`vllm_qwen3_32768_dp16_ep16_b8_20260717_035058`.

AFD 32K/b4 job `5445949` completed `0:0` in 5:27. All 112 exact prompts
returned 16 tokens. Its primary robust result was **16.8201 ms median decode
interval and 208.0840 tok/s/GPU**. Two long intervals (32.0439 and 37.4578 ms)
and one short interval (7.1705 ms) give the retained complete-vector mean of
17.5043 ms and mean-derived throughput 199.9513 tok/s/GPU. Result:
`afd_qwen3_32768_b4_n8_20260717_034415/experiment/afd-result.json`, SHA-256
`4983f25a41c4eaa2725e1e8fb74a6a9e0f774c7e83fe6c7a5b347b53985ba608`;
aggregate prompt-token hash
`6936654aa10ddd6c700f71cb75a8f3b962e91e402b7447c9f9bbc6b217f426eb`.
Its slot advanced at 03:54 PDT to AFD 32K/b8 job `5446048` in
`afd_qwen3_32768_b8_n8_20260717_035426`, retaining the known-bad head-node
exclusion.

vLLM 32K/b6 job `5445988` completed `0:0` in 6:40. Its primary synchronized
request result was **29.4765 ms median decode-step latency and 203.5520
tok/s/GPU**, with 29.4254 ms mean and 203.9058 tok/s/GPU. Secondary CUDA-kernel
throughput was 324.2907 tok/s/GPU. All 96 prompts were exactly 32,768 tokens.
Result: `vllm_qwen3_32768_dp16_ep16_b6_20260717_034730/baseline-result.json`,
SHA-256 `36cbe7b738dbbd65f10aa3954ef025933caf8a23fdc916d6c16884abbaabb337`.
Its slot advanced at 03:57 PDT to vLLM 32K/b12 job `5446065` in
`vllm_qwen3_32768_dp16_ep16_b12_20260717_035744`.

AFD 32K/b6 job `5446015` completed `0:0` in 5:45. All 168 exact prompts
returned 16 tokens. Its primary result was **17.4851 ms median decode interval
and 300.2548 tok/s/GPU**; the complete 14-interval vector has 16.7988 ms mean
and mean-derived throughput 312.5217 tok/s/GPU. Result:
`afd_qwen3_32768_b6_n8_20260717_035053/experiment/afd-result.json`, SHA-256
`6fa0b48ad8b31113fcc60f2bc2c090396b4018073eae9ec4e1b2260fca08c484`;
aggregate prompt-token hash
`7c5f30adf75a46cfcea34ef204f321634ecd9fabfb82bc75673af472c7b62bb3`.

vLLM 32K/b8 job `5446017` completed `0:0` in 5:34. Its primary synchronized
request result was **31.5932 ms median decode-step latency and 253.2190
tok/s/GPU**. Early 89.4816 and 55.3368 ms tails raise the complete-series mean
to 37.6022 ms and reduce mean throughput to 212.7533 tok/s/GPU. Secondary
CUDA-kernel throughput was 395.6663 tok/s/GPU. All 128 prompts were exactly
32,768 tokens. Result:
`vllm_qwen3_32768_dp16_ep16_b8_20260717_035058/baseline-result.json`, SHA-256
`be8225d3de8c211426db47430203d741beec7b935f386260daaf84f26b55dcf2`.
Their slots advanced at 04:01 PDT to AFD 32K/b12 job `5446091` in
`afd_qwen3_32768_b12_n8_20260717_040107`, retaining the known-bad head-node
exclusion, and vLLM 32K/b16 job `5446092` in
`vllm_qwen3_32768_dp16_ep16_b16_20260717_040111`.

AFD 32K/b8 job `5446048` completed `0:0` in 9:24. All 224 exact prompts
returned 16 tokens. Its primary result was **18.0778 ms median decode interval
and 387.2144 tok/s/GPU**; the complete 14-interval vector has 16.9170 ms mean
and mean-derived throughput 413.7854 tok/s/GPU. Result:
`afd_qwen3_32768_b8_n8_20260717_035426/experiment/afd-result.json`, SHA-256
`3c2095b1c1b8b6a8108770fc3e3b98d047374f80e30b4c47a52cea4a408b5c46`;
aggregate prompt-token hash
`a16cc839b76d8e9276f9d239dc3e3f24c27d1da54ba2f5c98f69d3d0489ee893`.

vLLM 32K/b12 job `5446065` completed `0:0` in 7:26. Its primary synchronized
request result was **35.6489 ms median decode-step latency and 336.6167
tok/s/GPU**, with 35.8025 ms mean and 335.1719 tok/s/GPU. Secondary CUDA-kernel
throughput was 496.0175 tok/s/GPU. All 192 prompts were exactly 32,768 tokens.
Result: `vllm_qwen3_32768_dp16_ep16_b12_20260717_035744/baseline-result.json`,
SHA-256 `b06d7d939ca99e01698e3af5d59921104d64e9b36cc252bb6b9a801a39d9d002`.
Their slots advanced at 04:07 PDT to AFD 32K/b16 job `5446155` in
`afd_qwen3_32768_b16_n8_20260717_040700`, retaining the known-bad head-node
exclusion, and the final planned 32K vLLM point, b24 job `5446156` in
`vllm_qwen3_32768_dp16_ep16_b24_20260717_040704`.

vLLM 32K/b16 job `5446092` completed `0:0` in 8:05. Its primary synchronized
request result was **39.7921 ms median decode-step latency and 402.0902
tok/s/GPU**, with 40.4534 ms mean and 395.5171 tok/s/GPU. Secondary CUDA-kernel
throughput was 554.9761 tok/s/GPU. All 256 prompts were exactly 32,768 tokens.
Result: `vllm_qwen3_32768_dp16_ep16_b16_20260717_040111/baseline-result.json`,
SHA-256 `422abfbeb1dbc0ebb1183261b637ebe821c220b6161c218aca52a252c615ecc6`.
Its slot advanced at 04:12 PDT to the final planned 32K AFD point, b24 job
`5446240` in `afd_qwen3_32768_b24_n8_20260717_041231`, retaining the known-bad
head-node exclusion.

vLLM 32K/b24 job `5446156` completed `0:0` in 8:35, completing every planned
32K vLLM batch. Its primary synchronized request result was **47.9162 ms median
decode-step latency and 500.8747 tok/s/GPU**, with 48.5164 ms mean and 494.6786
tok/s/GPU. Secondary CUDA-kernel throughput was 673.9662 tok/s/GPU. All 384
prompts were exactly 32,768 tokens. Result:
`vllm_qwen3_32768_dp16_ep16_b24_20260717_040704/baseline-result.json`, SHA-256
`cb143697ad3913d371040eaf9821041a5bbb895106af73ffafba0806fd5d4265`.
Its slot began the 64K tier at 04:17 PDT with vLLM 64K/b2 job `5446308` in
`vllm_qwen3_65536_dp16_ep16_b2_20260717_041758`.

The first AFD 32K/b12 attempt, job `5446091`, failed `1:0` after 14:35 and is
not a performance point. All 336 requests generated 16 tokens and all 28
attention workers completed their target replay, but strict validation found
`alignment_json_missing`, zero coordinator schedule steps, and no phases; no
result JSON was written. The artifact-finalization defect occurred with head
node `nvl72130-T10`. The identical case was resubmitted as job `5446363` in
`afd_qwen3_32768_b12_n8_20260717_042125`, excluding all three known bad heads
`nvl72051-T01,T02,nvl72130-T10`.

AFD 32K/b16 job `5446155` completed `0:0` in 11:28. All 448 exact prompts
returned 16 tokens. Its primary result was **20.5300 ms median decode interval
and 681.9275 tok/s/GPU**; the complete 14-interval vector has 19.3125 ms mean
and mean-derived throughput 724.9174 tok/s/GPU. Result:
`afd_qwen3_32768_b16_n8_20260717_040700/experiment/afd-result.json`, SHA-256
`66fb002bb04f35ffa23e08ec3926f6daa273f83d7816498369c064511d5b749f`;
aggregate prompt-token hash
`afded5a46914cab1ce35d6955b736035af726e52c90d11496d441f09d2f2ad55`.
Its slot advanced at 04:21 PDT to vLLM 64K/b3 job `5446370` in
`vllm_qwen3_65536_dp16_ep16_b3_20260717_042131`.

AFD 32K/b24 job `5446240` completed `0:0` in 10:55, completing every planned
32K AFD batch except the in-flight b12 retry. All 672 exact prompts returned 16
tokens. Its primary result was **22.3691 ms median decode interval and 938.7931
tok/s/GPU**; the complete 14-interval vector has 20.8530 ms mean and
mean-derived throughput 1,007.0477 tok/s/GPU. Result:
`afd_qwen3_32768_b24_n8_20260717_041231/experiment/afd-result.json`, SHA-256
`d2460285815c9d304b2e00828d3ea115681c923be968e3c6bddaed57b0d8bf2c`;
aggregate prompt-token hash
`498bec883972b6e8667d11b24ce7c214dd637a77082bda87efa5f7fa2a608be6`.

The first vLLM 64K/b2 attempt, job `5446308`, failed `1:0` after 6:07 and is
not a performance point. The engine accepted `max_seq_len=65600` only because
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, but decode hit a deterministic CUDA index
assert requiring position indices `< 40960`, exactly matching the pinned
Qwen3 model's native `max_position_embeddings=40960`. This is a model-contract
limit, not an infrastructure failure; allowing a longer configured max length
does not extend model semantics. The queued duplicate 64K/b3 job `5446370` was
cancelled before measurement. This proved that 64K/128K are unsupported under
the unchanged-model contract; the explicit, hashed YaRN continuation below
supersedes that checkpoint.

At that checkpoint both launchers failed fast above 40,960 tokens, explicitly
requiring a model-semantic/RoPE-scaling change. Those guards were subsequently
replaced by the explicit shared-profile contract below. The checkpoint control
hashes were
`7b1cd45e3c658f2af09c3ab1a172458adf18419eb1ee5d66fc68c4ab542e57f1`
(AFD) and
`df6ebb3f6077117cf3d9ff8821ef57db60ffa4c020d02c34146c0cc5bbfdb561`
(vLLM).

AFD 32K/b12 retry job `5446363` completed `0:0` in 11:39, closing the last
supported sweep point. All 336 exact prompts returned 16 tokens, and the input
contract matches the failed attempt. Its primary robust result was **19.8708 ms
median decode interval and 528.4141 tok/s/GPU**. The retained complete vector
contains 53.4726 and 91.5176 ms tails plus a 6.3765 ms short interval, giving
26.5612 ms mean and mean-derived throughput 395.3129 tok/s/GPU. Result:
`afd_qwen3_32768_b12_n8_20260717_042125/experiment/afd-result.json`, SHA-256
`aa401e1b9677635cfc007af272db22e33290ddb8430f9ec87017989d0c5e0721`;
aggregate prompt-token hash
`a86730494bf5b287ba7457384a29845c959db73c101c39032e1b63627abacc3b`.

Supported unchanged-model sweep completion: AFD and vLLM both successfully ran
all planned batches at 1K (2..512 planned sequence), 2K (2..256), 4K (2..192),
8K (2..96), 16K (2..48), and 32K (2..24). Exact sparse sequences are recorded
in the sweep contract above. Under that unchanged-model checkpoint no valid
64K/128K point existed because the pinned revision's native position limit is
40,960 tokens; the YaRN continuation below completed both tiers.

### Explicit Qwen3 long-context continuation (2026-07-17)

The full requested 64K/128K scope is continuing under an explicit, shared
model-semantic extension rather than treating the native limit as the endpoint.
The pinned model card says Qwen3 is native to 32,768 tokens, validates YaRN to
131,072, specifies `original_max_position_embeddings=32768`, and recommends
static factor 2 for a typical 65,536-token context and factor 4 for 131,072.
Both AFD and vLLM now consume immutable derived model directories that symlink
every non-config artifact to revision `39eb2b067...` and replace only
`config.json`. The intended 64K/128K factors are 2/4; because the decode tests
reserve 64 positions beyond the exact ISL, the configured factors are the exact
extended/native ratios 2.001953125 and 4.001953125 for maxima 65,600 and
131,136. This avoids Transformers' explicit/implicit YaRN-factor mismatch
warning while changing the intended factor by only the required decode
headroom. Final profile config hashes are recorded after validation below.
The unchanged source config SHA-256 is
`702c46d431bb984db9035a1225186bbfdb52c0d19c82104df4a37cd005e0369e`.

`prepare_model_profile.py` builds these profiles atomically, validates them
idempotently, and fails on source/config/artifact drift. Both launchers record
the model config hash and full RoPE scaling in result provenance. Local and
remote syntax, dry-run selection, byte sync, profile creation, repeated profile
validation, config fields, and artifact symlinks passed. Current remote control
hashes are `3880a09b...` (AFD), `056e4df4...` (vLLM), and `10682c87...`
(profile builder).

With the user-authorized ceiling raised to four concurrent jobs, the first
paired correctness probes were submitted at 05:09 PDT: AFD 64K/b2 job
`5446930` in `afd_qwen3_65536_b2_n8_20260717_050930` (8 nodes/32 GPUs,
known-bad head nodes excluded) and vLLM 64K/b2 job `5446931` in
`vllm_qwen3_65536_dp16_ep16_b2_20260717_050930` (DP=EP16, 4 nodes/16 GPUs).
They were cancelled 42 seconds after allocation, before measurement, when a
read-only Transformers/FastAFD config interpretation probe warned that the
original factor-2/factor-4 draft did not include the 64-token decode headroom
in its scaling ratio. Those draft profiles (`d4b702...`, `f9c1df...`) are
rejected and must not be used as performance contracts. Corrected probes will
remain gated on warning-free interpretation before submission.

The corrected factor 2.001953125 and 4.001953125 profiles passed warning-free
Transformers and FastAFD `ModelConfig` interpretation with identical max
positions and `rope_scaling` dictionaries. Their final config SHA-256 values
are `fce24b515fbf65a2d0e986a2d135dbb5f793dbc1edfb5025bba6613fed4994c1`
(64K) and `7886de5c119e8eeb5eb72e825b18c1cf898e08bf1323932214610df331d1bc66`
(128K). The corrected 64K/b2 vLLM probe is job `5446958` in
`vllm_qwen3_65536_dp16_ep16_b2_20260717_051431`. Its AFD peer is deferred only
until Slurm releases cancelled job `5446930` from `COMPLETING`; an attempted
early resubmit was rejected by the duplicate-name guard and created no job.
Slurm released the allocation at 05:15 PDT, and the corrected AFD peer is job
`5446974` in `afd_qwen3_65536_b2_n8_20260717_051536`, retaining all three
known-bad head-node exclusions.

Corrected vLLM 64K/b2 job `5446958` completed `0:0` in 6:17, the first valid
long-context point. All 32 prompts were exactly 65,536 tokens under config hash
`fce24b...`; all 16 rank artifacts, four Nsight reports/SQLite exports, and four
clean final snapshots exist. Its primary synchronized request result is
**26.6468 ms median decode-step latency and 75.0560 tok/s/GPU**, with the
complete 15-step series
`[75.212438,26.646764,26.323473,26.271737,27.148013,25.963161,25.865003,25.941561,26.191953,26.181617,27.182637,26.802804,26.730349,26.705267,26.858683]`
ms, 29.7350 ms mean, and 67.2607 mean-derived tok/s/GPU. Secondary CUDA-kernel
throughput is 124.2268 tok/s/GPU. Post-init KV capacity is exactly 12 full
65,600-token sequences/lane, matching the terminal planned 64K batch. Result:
`vllm_qwen3_65536_dp16_ep16_b2_20260717_051431/baseline-result.json`, SHA-256
`a978b9d03ed855c4562196b422996d6b41a7b553db09d922e1d6846d9d10d8b8`.
Its freed slot plus an idle fourth slot launched vLLM 64K/b3 job `5447047` in
`vllm_qwen3_65536_dp16_ep16_b3_20260717_052345` and b4 job `5447048` in
`vllm_qwen3_65536_dp16_ep16_b4_20260717_052345`.
The fourth authorized slot was filled by vLLM 64K/b6 job `5447058` in
`vllm_qwen3_65536_dp16_ep16_b6_20260717_052501`; batch 6 is below the measured
exact capacity 12, so this is a performance point rather than a capacity probe.

Corrected AFD 64K/b2 job `5446974` completed `0:0` in 8:29, validating the
shared long-context contract in both runtimes. All 56 exact prompts returned 16
tokens; 28 attention logs and eight clean final GPU snapshots exist. Its
primary result is **18.4803 ms median decode interval and 94.6956 tok/s/GPU**.
The complete interval vector is
`[18.915173,18.271077,17.951525,18.629125,18.298405,20.061573,18.959525,18.414373,18.707653,19.548390,18.142341,18.546181,16.277284,0.520960]`
ms; the terminal short interval makes the arithmetic mean 17.2317 ms, so the
predeclared median remains the robust headline. Result:
`afd_qwen3_65536_b2_n8_20260717_051536/experiment/afd-result.json`, SHA-256
`9a01ae2293954ab8ea5cf5c28aa68e57e49d00754c24d9f1119d14d87e6332c6`;
aggregate prompt-token hash `33fe053e...`. Its slot advanced to AFD 64K/b3 job
`5447103` in `afd_qwen3_65536_b3_n8_20260717_053018`, retaining the known-bad
head-node exclusions.

vLLM 64K/b3 job `5447047` completed `0:0` in 5:31. Its primary synchronized
request result is **28.3596 ms median decode-step latency and 105.7842
tok/s/GPU**, with 28.5618 ms mean and 105.0352 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 166.0152 tok/s/GPU. Exact prompts and all
16 rank/four profile/four final-snapshot artifacts passed. Result:
`vllm_qwen3_65536_dp16_ep16_b3_20260717_052345/baseline-result.json`, SHA-256
`6a9c19f53eaaf17441bcf6642b6cee8aaada866529b8602cb3fef036d127c9e1`.
Its slot advanced to vLLM 64K/b8 job `5447133` in
`vllm_qwen3_65536_dp16_ep16_b8_20260717_053318`.

vLLM 64K/b4 job `5447048` completed `0:0` in 7:06. Its primary synchronized
request result is **31.7348 ms median decode-step latency and 126.0447
tok/s/GPU**, with 32.1502 ms mean and 124.4159 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 191.1564 tok/s/GPU. Exact prompts and all
artifacts passed. Result:
`vllm_qwen3_65536_dp16_ep16_b4_20260717_052345/baseline-result.json`, SHA-256
`4ebe1b9cded87c23c58a6a08b3ab052363531792b866d601664f9994177afdde`.
Its slot advanced to the measured 64K capacity ceiling, vLLM/b12 job `5447138`
in `vllm_qwen3_65536_dp16_ep16_b12_20260717_053417`.

vLLM 64K/b6 job `5447058` completed `0:0` in 7:46. Its primary synchronized
request result is **34.1989 ms median decode-step latency and 175.4442
tok/s/GPU**, with 34.5360 ms mean and 173.7315 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 254.0933 tok/s/GPU. Exact prompts and all
artifacts passed. Result:
`vllm_qwen3_65536_dp16_ep16_b6_20260717_052501/baseline-result.json`, SHA-256
`f342dcc31e471bf65fa8464a2934fff9d01eb3d19e382f60025903cd767c1410`.
Its freed slot launched AFD 64K/b4 job `5447163` in
`afd_qwen3_65536_b4_n8_20260717_053508`, retaining the head exclusions.

AFD 64K/b3 job `5447103` completed `0:0` in 8:57. All 84 exact prompts
returned 16 tokens; 28 attention logs and eight final snapshots passed. Its
primary result is **18.3084 ms median decode interval and 143.3765 tok/s/GPU**.
The full interval vector is
`[18.482570,18.290762,18.367722,18.288329,18.623498,18.689834,18.326122,18.245449,18.269066,18.486026,18.248361,18.361034,16.001545,0.942944]`
ms; the terminal short interval again makes the mean non-headline. Result:
`afd_qwen3_65536_b3_n8_20260717_053018/experiment/afd-result.json`, SHA-256
`8526b0ca7d8d6276209dbd6bd683d50596bf24aaa37f4896af93198208ccc4da`.
Its slot advanced to AFD 64K/b6 job `5447210` in
`afd_qwen3_65536_b6_n8_20260717_054224`, retaining head exclusions.

vLLM 64K/b8 job `5447133` completed `0:0` in 6:56. Its primary synchronized
request result is **37.5998 ms median decode-step latency and 212.7669
tok/s/GPU**, with 37.8710 ms mean and 211.2437 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 297.9914 tok/s/GPU. Exact prompts and all
artifacts passed. Result:
`vllm_qwen3_65536_dp16_ep16_b8_20260717_053318/baseline-result.json`, SHA-256
`620ca2581909c06d805e80b7a62cbece1829d19c66823630900b882ccedbebfa`.
Its slot began the 128K tier with vLLM/b2 job `5447218` in
`vllm_qwen3_131072_dp16_ep16_b2_20260717_054327`, using config SHA-256
`7886de5c...` and exact factor 4.001953125.

vLLM 64K/b12 job `5447138` completed `0:0` in 8:50, closing every planned
64K vLLM batch `2,3,4,6,8,12` at the measured capacity ceiling. Its primary
synchronized request result is **44.5272 ms median decode-step latency and
269.4980 tok/s/GPU**, with 45.0722 ms mean and 266.2393 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 353.7301 tok/s/GPU. All exact prompts,
16 ranks, four profiles/SQLite exports, and final snapshots passed. Capacity is
12 with 12,688 KV tokens/GPU unused, insufficient for another 65,600-token
sequence. Result:
`vllm_qwen3_65536_dp16_ep16_b12_20260717_053417/baseline-result.json`, SHA-256
`93fd92566caaf78659c61da771058a2b1d03fa5d01c2c2bce02fa8841cb18a6b`.

AFD 64K/b4 job `5447163` completed `0:0` in 8:59. All 112 exact prompts
returned 16 tokens and all worker/snapshot artifacts passed. Its primary result
is **18.4541 ms median decode interval and 189.6600 tok/s/GPU**. The full vector
is
`[25.279445,10.971561,17.878959,17.837551,20.856561,18.285199,19.835857,18.309327,20.296145,19.024016,18.598832,18.729711,15.699214,0.589792]`
ms; it contains both a high tail and a terminal short interval. Result:
`afd_qwen3_65536_b4_n8_20260717_053508/experiment/afd-result.json`, SHA-256
`8145101cde26584683c205bac498b9d7ea7cdf4938a340e046f2d5056db7edaf`.
Its slot advanced to AFD 64K/b8 job `5447253` in
`afd_qwen3_65536_b8_n8_20260717_054859`, retaining head exclusions.

The first vLLM 128K/b2 probe reached eight synchronized warmup/prefill steps
and the measurement barrier without position, capacity, or CUDA errors. On that
semantic gate, the fourth slot launched paired AFD 128K/b2 job `5447279` in
`afd_qwen3_131072_b2_n8_20260717_055005`, using the identical `7886de5c...`
config and retaining head exclusions.

vLLM 128K/b2 job `5447218` completed `0:0` in 5:43, the first valid 128K
point. All 32 prompts were exactly 131,072 tokens under config `7886de5c...`;
all 16 rank artifacts, four profiles/SQLite exports, and final snapshots passed.
Its primary synchronized request result is **29.9044 ms median decode-step
latency and 66.8798 tok/s/GPU**, with the complete series
`[49.230562,29.689320,29.565530,29.245292,29.573802,30.278524,29.415643,29.023149,29.833531,30.341736,30.037596,31.422750,30.072797,29.904412,29.944426]`
ms, 31.1719 ms mean, and 64.1603 mean-derived tok/s/GPU. Secondary CUDA-kernel
throughput is 102.0086 tok/s/GPU. Post-init capacity is exactly batch 6. Result:
`vllm_qwen3_131072_dp16_ep16_b2_20260717_054327/baseline-result.json`, SHA-256
`355ec8d01f74e44efc85fcea654acfc18574a1d539a7827ab9e7ba25185a0e73`.
Its slot advanced to vLLM 128K/b3 job `5447293` in
`vllm_qwen3_131072_dp16_ep16_b3_20260717_055109`.

AFD 64K/b6 job `5447210` completed `0:0` in 10:39. All 168 exact prompts
returned 16 tokens and all artifacts passed. Its primary result is **18.9173 ms
median decode interval and 277.5240 tok/s/GPU**. The full vector is
`[19.126790,18.351685,18.498437,18.486853,18.707782,22.379373,19.495911,18.587205,21.569548,19.241414,20.373385,21.904300,16.313441,0.597377]`
ms. Result: `afd_qwen3_65536_b6_n8_20260717_054224/experiment/afd-result.json`,
SHA-256 `4ff28375d42e4f8677b12e704246e9dc19f748e9bae2191cab16dfdb7196e827`.
Its slot advanced to the final planned AFD 64K point, b12 job `5447352` in
`afd_qwen3_65536_b12_n8_20260717_055650`, retaining head exclusions.

vLLM 128K/b3 job `5447293` completed `0:0` in 6:35. Its primary synchronized
request result is **34.9808 ms median decode-step latency and 85.7614
tok/s/GPU**, with 35.0063 ms mean and 85.6988 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 129.2750 tok/s/GPU. Exact prompts and all
artifacts passed. Result:
`vllm_qwen3_131072_dp16_ep16_b3_20260717_055109/baseline-result.json`, SHA-256
`c6864cffc8d9b67a1d5340bb8e751e11b628b305223cd3321cee0128386cb6f8`.
Its slot advanced to vLLM 128K/b4 job `5447411` in
`vllm_qwen3_131072_dp16_ep16_b4_20260717_060002`.

AFD 128K/b2 job `5447279` completed `0:0` in 9:34, the first valid AFD 128K
point. All 56 exact prompts returned 16 tokens; all 28 attention logs and eight
final snapshots passed. Its primary result is **20.2388 ms median decode
interval and 86.4677 tok/s/GPU**. The full vector is
`[24.245041,20.168008,19.777960,20.088040,20.094024,20.270792,20.328425,20.350601,20.801769,20.479977,20.206729,20.298312,18.060836,0.511169]`
ms. Result: `afd_qwen3_131072_b2_n8_20260717_055005/experiment/afd-result.json`,
SHA-256 `900a6cbfb4f4a027464eaa0740dba2ba039152b8ec1c9ba7b460d408900de9e6`.
Its slot advanced to AFD 128K/b3 job `5447467` in
`afd_qwen3_131072_b3_n8_20260717_060340`, retaining head exclusions.

AFD 64K/b8 job `5447253` completed `0:0` in 11:54. All 224 exact prompts
returned 16 tokens and all artifacts passed. Its primary result is **19.8967 ms
median decode interval and 351.8179 tok/s/GPU**. The full vector is
`[20.047343,19.716047,19.601518,19.618862,19.505293,23.722686,19.893872,20.386672,19.899440,20.210768,20.374289,21.467189,17.328517,0.623138]`
ms. Result: `afd_qwen3_65536_b8_n8_20260717_054859/experiment/afd-result.json`,
SHA-256 `868098ee75f282737cf1c314fcbd1917bf04c93fb2403bc84422cfbcdcb26766`.
Its slot advanced to AFD 128K/b4 job `5447477` in
`afd_qwen3_131072_b4_n8_20260717_060441`, retaining head exclusions.

vLLM 128K/b4 job `5447411` completed `0:0` in 7:36. Its primary synchronized
request result is **38.1154 ms median decode-step latency and 104.9444
tok/s/GPU**, with 38.6786 ms mean and 103.4165 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 146.0433 tok/s/GPU. Exact prompts and all
artifacts passed. Result:
`vllm_qwen3_131072_dp16_ep16_b4_20260717_060002/baseline-result.json`, SHA-256
`357669cdd1233549dc8cc69e868c67f79d76df03439470b52518f1fd063d4c08`.
Its slot advanced to the final planned vLLM 128K point, b6 job `5447543` in
`vllm_qwen3_131072_dp16_ep16_b6_20260717_060953`.

AFD 64K/b12 job `5447352` completed `0:0` in 13:28, closing every planned AFD
64K batch `2,3,4,6,8,12`. All 336 exact prompts returned 16 tokens and all
artifacts passed. Its primary result is **20.9625 ms median decode interval and
500.8946 tok/s/GPU**. The full vector is
`[21.272060,20.901596,20.902523,20.990396,21.457340,21.994845,20.887228,20.934588,20.754299,21.276476,21.772669,21.876925,19.142681,0.565761]`
ms. Result: `afd_qwen3_65536_b12_n8_20260717_055650/experiment/afd-result.json`,
SHA-256 `ac45746d4b09452b2081c42312edb9096acafc28629f30a08be7f4a090cd455d`.
Its slot advanced to the final planned AFD 128K point, b6 job `5447573` in
`afd_qwen3_131072_b6_n8_20260717_061313`, retaining head exclusions.

AFD 128K/b3 job `5447467` completed `0:0` in 10:51. All 84 exact prompts
returned 16 tokens and all artifacts passed. Its primary result is **20.4330 ms
median decode interval and 128.4687 tok/s/GPU**. The full vector is
`[23.018108,20.964191,20.487712,20.578048,20.799583,21.036320,19.892737,20.030464,20.013121,20.562656,20.265632,20.378272,17.288485,0.605887]`
ms. Result: `afd_qwen3_131072_b3_n8_20260717_060340/experiment/afd-result.json`,
SHA-256 `b8a9a539a2d57366c74c2ae537ef91b0db55a8c7af4d31cbd60961a838258a16`.

vLLM 128K/b6 job `5447543` completed `0:0` in 10:35, closing every planned
128K vLLM batch `2,3,4,6` at the exact capacity ceiling. Its primary
synchronized request result is **45.8928 ms median decode-step latency and
130.7393 tok/s/GPU**, with 46.2398 ms mean and 129.7584 mean-derived tok/s/GPU.
Secondary CUDA-kernel throughput is 171.4366 tok/s/GPU. Exact prompts and all
artifacts passed; only 12,960 KV tokens/GPU remain, insufficient for another
131,136-token sequence. Result:
`vllm_qwen3_131072_dp16_ep16_b6_20260717_060953/baseline-result.json`, SHA-256
`f4eb5bb50a3726e662af4ddcb127d4c667dd2a3eefd159ad7126a9790a0254d3`.

The first AFD 128K/b4 attempt, job `5447477`, failed `1:0` after 17:14 and is
not a performance point. All 112 requests returned 16 tokens and all 28
attention workers recorded 30 target replays, but strict artifact validation
found `alignment_json_missing`, zero coordinator schedule steps, and no phases.
This matches the known head-node artifact-finalization defect, now on head
`nvl72067-T01`. The identical case was resubmitted as job `5447705` in
`afd_qwen3_131072_b4_n8_20260717_062605`, excluding that head plus
`nvl72051-T01,T02` and `nvl72130-T10`.

AFD 128K/b6 job `5447573` completed `0:0` in 14:54. All 168 exact prompts
returned 16 tokens and all artifacts passed. Its primary result is **21.8021 ms
median decode interval and 240.8030 tok/s/GPU**. The full vector is
`[23.634798,21.489609,22.214571,21.546570,22.531340,23.803374,22.666924,21.104169,21.496778,21.771818,21.832298,21.898219,18.930564,0.569409]`
ms. Result: `afd_qwen3_131072_b6_n8_20260717_061313/experiment/afd-result.json`,
SHA-256 `ae3222b188da78aa3b6b1b306c2eb3a279e29126d80766afb9ff6d84c1dbd9a7`.

AFD 128K/b4 retry job `5447705` completed `0:0` in 11:36, closing every
planned AFD 128K batch `2,3,4,6`. All 112 exact prompts returned 16 tokens and
all 28 worker logs/eight final snapshots passed. Its primary result is
**22.5530 ms median decode interval and 155.1898 tok/s/GPU**. The complete
vector is
`[22.613839,22.001133,22.929679,24.703056,20.323821,22.492206,25.982609,20.594669,20.649869,23.957007,22.729871,26.970961,17.680779,0.585473]`
ms. Result: `afd_qwen3_131072_b4_n8_20260717_062605/experiment/afd-result.json`,
SHA-256 `e53c2443f670de5b1dc7288da5e51c1fe70d610ca51d3ccae956a1f93f2ecc93`.

### Final long-context request-level tables

All values below are the predeclared primary metrics: median request decode
step/interval and its TPS/GPU. A consolidated remote audit required exactly one
valid result per expected mode/context/batch, the correct profile hash, exact
prompt length, and complete rank/worker/profile/snapshot artifacts; all 20 rows
passed.

| 64K batch | AFD ms | AFD TPS/GPU | vLLM ms | vLLM TPS/GPU |
|---:|---:|---:|---:|---:|
| 2 | 18.4803 | 94.6956 | 26.6468 | 75.0560 |
| 3 | 18.3084 | 143.3765 | 28.3596 | 105.7842 |
| 4 | 18.4541 | 189.6600 | 31.7348 | 126.0447 |
| 6 | 18.9173 | 277.5240 | 34.1989 | 175.4442 |
| 8 | 19.8967 | 351.8179 | 37.5998 | 212.7669 |
| 12 | 20.9625 | 500.8946 | 44.5272 | 269.4980 |

| 128K batch | AFD ms | AFD TPS/GPU | vLLM ms | vLLM TPS/GPU |
|---:|---:|---:|---:|---:|
| 2 | 20.2388 | 86.4677 | 29.9044 | 66.8798 |
| 3 | 20.4330 | 128.4687 | 34.9808 | 85.7614 |
| 4 | 22.5530 | 155.1898 | 38.1154 | 104.9444 |
| 6 | 21.8021 | 240.8030 | 45.8928 | 130.7393 |

Final full-sweep success: both AFD and wide-EP vLLM successfully ran every
planned sparse batch at all eight ISLs. Batch lists are 1K through 512, 2K
through 256, 4K through 192, 8K through 96, 16K through 48, 32K through 24,
64K through 12, and 128K through 6, using the exact sequences listed in the
sweep contract. The scheduler was empty after completion.

Final local/remote control SHA-256 values are
`86032f08ed2ddfa7cee649a617ac1871a592710f349c3d6c9df05c589339a41f`
(`run_afd.sh`),
`49780b864e72cdb18ec56f9b8d4a1ec423187b4c2e55bdc7de0cf44eff0deb34`
(`run_vllm.sh`), and
`10682c878fc7f14d6608ce33d804fa5ea933006ec406dc0e49d89fd1172501d4`
(`prepare_model_profile.py`). Local and remote copies are byte-identical; both
shell scripts pass `bash -n`, the profile builder compiles in memory, 128K
terminal dry runs resolve the correct profile/factor, `git diff --check`
passes, the pinned remote source worktree is clean, and the scheduler is empty.

### Historical consolidated report and AFD KV capacity

The removed `QWEN3_ISL_BATCH_SWEEP_CONFIGS_AND_RESULTS.txt` was a plain-text,
Google-Docs-friendly consolidation of the fixed AFD/wide-EP configurations, batch and
metric semantics, all 172 result rows, terminal comparisons, capacity,
long-context provenance, notable excluded/retried attempts, the earlier Qwen3
8K/16K wide-EP topology tables, and both vLLM summed-kernel and synchronized
request metrics. Per the user's final report-format decision, its headline
ratio follows the README-compatible mixed contract: AFD median coordinator
TPS/GPU versus vLLM summed CUDA-kernel TPS/GPU; request-wall vLLM latency/TPS is
retained as secondary evidence. A retired one-off report generator produced it
from the remote result JSONs and failed closed unless it found exactly the
expected 86 rows per mode and validated topology, global request counts,
capacity guards, and TPS/GPU formulas. That case-specific generator is no
longer part of the source tree.

AFD terminal-run coordinator logs at every ISL reported the same measured
attention-side allocation: 13,116 pages at 64 tokens/page, or 839,424 KV-token
slots per attention GPU. Dividing by each configured maximum model length
(ISL + 64 decode-token slots) gives AFD KV-capacity batch ceilings of
771/397/201/101/51/25/12/6 for 1K through 128K. Wide-EP result JSONs report
799,888--799,920 slots/GPU through 64K and 799,776 at 128K, giving ceilings of
735/378/192/96/48/24/12/6. Thus AFD exposes about 4.9% more KV-token slots per
attention GPU; floor effects make the two capacity batches equal at 64K/128K.
The largest AFD performance-tested batches remain
512/256/192/96/48/24/12/6, so the first six AFD capacity ceilings are derived
from measured allocation rather than benchmarked at those exact batches.

The apparent 8K gap change is metric-only rather than a missing result. The
earlier EP16 comparison used AFD's 2,522.64 request/coordinator TPS/GPU against
vLLM's 2,331.65 CUDA-kernel TPS/GPU, an 8.19% AFD lead. The new 8K/b96 runs are
within -2.27% (AFD) and -0.38% (vLLM CUDA) of those values: 2,465.35 and
2,322.85, a 6.13% lead on the earlier basis. The underlying sweep's predeclared
request metric instead compares AFD 2,465.35 against vLLM synchronized request-
wall 1,805.84, giving 36.52%; the difference is host/distributed scheduling and
synchronization excluded from CUDA kernel time. The copy/paste report now
headlines the 6.13% README-compatible comparison at the user's request while
retaining the request metric. The remembered 15.92% lead is the earlier Qwen3
16K EP16 result, whose AFD topology was also 12 nodes/44 attention GPUs rather
than the fixed eight-node/28-attention-GPU topology used by the ISL sweep.

The summed-kernel metric itself is task-owned rather than emitted by vLLM:
Nsight Systems supplies raw `CUPTI_ACTIVITY_KIND_KERNEL` start/end timestamps
and SQLite exports, while `run_vllm.sh` sums duration per device across the
15-step profiler range, averages across world GPUs and steps, and derives
TPS/GPU. This matches the README vLLM number but is not E2E CUDA-event timing.
The repo's built-in vLLM path is instead a sequential or sharded correctness
cross-check: launch `vllm serve`, score FastAFD-sampled tokens with
`prompt_logprobs`, and compare alignment thresholds. It does not emit the
README-compatible summed-kernel throughput metric.

Representative 8K/b96 graph/idle audit (job 5445052): CUDA graphs were not a
silent fallback. The vLLM 0.19 log records `enforce_eager=False`,
`CUDAGraphMode.FULL_AND_PIECEWISE`, capture size 96, and completed decode-FULL
capture with `FULL=10 (largest=96)`. Its request-wall mean was 53.9274 ms/step
and summed-kernel time was 41.3285 ms/step/GPU. Direct interval-union analysis
of all four Nsight SQLite files gives an approximately 53.56 ms per-GPU kernel
timeline envelope, 41.2 ms busy union, and 12.3 ms idle gaps per step. Averaged
over all 16 GPUs, about 9.3 ms/step is in gaps >=5 ms, 1.3 ms in 1--5 ms gaps,
and 1.76 ms in sub-ms gaps. Thus most of the difference is a long host-side
pause before each full decode-graph replay, not accumulated tiny gaps inside
the graph and not merely the max-rank-versus-mean-GPU aggregation mismatch.
The log also proves vLLM uses a CPU all-reduce each step to synchronize DP
padding/cudagraph mode across the 16 DP/EP ranks; vLLM's `dp_utils.py` performs
that `dist.all_reduce` on the CPU group. Scheduler/metadata preparation,
sampling/output bookkeeping, and graph-launch/driver work also remain outside
the captured model graph. Nsight reports about 2.1 ms average host API duration
per `cudaGraphLaunch` in this profiled run, but these components overlap and
must not be added independently. The benchmark's output validation between
steps also creates some trace-visible idle time but is outside each recorded
request-wall interval.

The removed Google-Sheets-ready exports accompanied the prose report.
`QWEN3_ISL_BATCH_SWEEP_GOOGLE_SHEETS.csv` was the primary compact flat table:
86 paired ISL/batch rows and 24 key configuration, topology, KV-capacity,
primary summed-kernel, diagnostic request-wall, and comparison columns.
`QWEN3_HISTORICAL_WIDE_EP_GOOGLE_SHEETS.csv` contained the earlier 8K/16K
EP4/8/16/32/64 comparison as 10 rows by 16 key columns. Fixed backend details
are grouped into readable AFD/vLLM config cells rather than repeated as many
individual columns.
Both contained only a header and data rows, used standard CSV quoting, and were
generated by a retired one-off CSV builder from the same fail-closed 172-result
collection. CSV parsing, rectangularity,
unique sweep keys, topology/global-batch identities, CUDA TPS, ratio/lead
formulas, and `git diff --check` passed. Historical SHA-256 values are
`c2ea6360318d9ceb6d378295c018b3d4d65beee343bad72cdbea97b39f1913f8`
(primary sweep CSV) and
`b9fcf0a9005a54a3e84758b84943047bc0cdc6ea585013234c8599c6bce86141`
(historical CSV).

## Qwen3 irregular-ISL / batch sweep (in progress, 2026-07-17)

The follow-on sweep measures resident decode batches whose request prompt
lengths span one of eight inclusive ranges rather than using one uniform ISL.
The same two retained launchers remain the only AFD/baseline entry points. Both
now expose an explicit final `uniform|irregular` mode argument; `uniform` is the
backward-compatible default and requires a single ISL, while `irregular`
requires one of the supported ranges. Representative calls are:

```bash
run_afd.sh qwen3 1k-4k 3 8 irregular
run_vllm.sh qwen3 1k-4k 16 3 irregular
run_afd.sh qwen3 8k 96 8 uniform
run_vllm.sh qwen3 8k 16 96 uniform
```

For an irregular local batch of size `n`, both modes build the same monotonic,
inclusive, symmetric integer linspace between the range endpoints. The lower
half is rounded to the nearest token and the upper half is its exact
complement, so endpoints are exact and the mean is exactly `(low + high) / 2`.
Thus 1K--4K/b3 is exactly `[1024, 2560, 4096]`. AFD writes prompts in
length-major/attention-rank order; the coordinator's deterministic round-robin
assignment gives every one of 28 attention-DP workers the full list. vLLM
gives the identical list to every one of 16 DP=EP ranks. Both result schemas
retain the complete list and length sum, and both modes keep prefix reuse off.

Batch candidates remain exactly `2^x` and `2^x + 2^(x-1)` for `x >= 1`.
Preflight is page-aware and fail-fast: AFD uses the measured 13,116 x 64-token
pages (839,424 KV tokens/attention GPU) and sums
`ceil((prompt + 64)/64)*64`; vLLM uses the conservative minimum measured
799,776 tokens/lane and 16-token blocks. Runtime vLLM recomputes the requirement
from the post-init block size/capacity. The complete mode-specific candidate
plans are:

| Range | AFD batches | vLLM batches |
|---|---|---|
| 1K--4K | 2,3,4,6,8,12,16,24,32,48,64,96,128,192,256 | same |
| 4K--8K | 2,3,4,6,8,12,16,24,32,48,64,96,128 | same |
| 8K--32K | 2,3,4,6,8,12,16,24,32 | same |
| 32K--128K | 2,3,4,6,8 | same |
| 1K--16K | 2,3,4,6,8,12,16,24,32,48,64 | same |
| 4K--64K | 2,3,4,6,8,12,16,24 | 2,3,4,6,8,12,16 |
| 8K--128K | 2,3,4,6,8,12 | 2,3,4,6,8 |
| 1K--128K | 2,3,4,6,8,12 | same |

This is 73 AFD points plus 71 vLLM points, 144 successful measurements total.

The AFD-only terminal b24/b12 points above fit its larger measured cache and
follow the prior contract that each mode continues until its next sparse point
is unsupported. Ranges ending at 64K/128K select the same immutable YaRN model
profiles already validated by the uniform sweep. AFD sets `PROMPT_LEN` and
maximum sequence length from the upper endpoint but overrides the Nsight
schedule window from the exact per-worker sum of 512-token prefill chunks.

CUDA graphs remain mandatory. AFD still configures the exact per-microbatch
graph bucket and accepts a result only if every attention-worker log has the
same full `afd_ag_decode_graph:replay` window. vLLM still uses
`enforce_eager=False`, `FULL_AND_PIECEWISE`, and a capture-size set containing
the requested batch; the benchmark now asserts the resolved runtime config and
records it in the result.

Local validation before remote install: both launchers pass `bash -n`; all five
embedded Python blocks compile; `git diff --check` passes; explicit uniform and
irregular dry runs pass; all eight terminal capacity points pass and every next
sparse point fails; wrong mode/range combinations and non-sparse b5 fail. At
12:15 PDT the remote pinned source was clean, its prior controls still matched
the completed uniform-sweep hashes, and no FastAFD jobs were active. Remote
sync, tokenizer/runtime probes, submissions, and measurements remain pending.

Remote install then passed byte identity, `bash -n`, representative native and
128K dry runs, the AFD 4K--64K/b24 edge, the vLLM rejection of that unsupported
b24 edge, clean pinned source, and an empty scheduler. Installed control hashes
are `86b5f70e...` (AFD), `7a1c26c9...` (vLLM), and unchanged `10682c87...`
(profile builder).

The four-slot correctness/performance gate was submitted at 12:31 PDT:

- AFD 1K--4K/b3 job `5451938`, run
  `afd_irregular_qwen3_1024-4096_b3_n8_20260717_123108`.
- vLLM 1K--4K/b3 job `5451939`, run
  `vllm_irregular_qwen3_1024-4096_dp16_ep16_b3_20260717_123108`.
- AFD 32K--128K/b3 job `5451940`, run
  `afd_irregular_qwen3_32768-131072_b3_n8_20260717_123108`.
- vLLM 32K--128K/b3 job `5451941`, run
  `vllm_irregular_qwen3_32768-131072_dp16_ep16_b3_20260717_123109`.

The native pair uses exact lengths `[1024,2560,4096]`; the YaRN pair uses
`[32768,81920,131072]` and revalidated config hash `7886de5c...`. Both AFD jobs
exclude the four known artifact-finalization head nodes. All four initially
queued normally and are the only active FastAFD jobs.

Both baseline gates completed `0:0` with full artifact validation. Native
1K--4K/b3 job `5451939` finished in 5:29 with exact per-rank lengths
`[1024,2560,4096]`, 7,872/799,920 required/available KV tokens, verified
`FULL_AND_PIECEWISE` graph mode including capture batch 3, 24.6729 ms median
request step (121.5907 tok/s/GPU), and 12.9343 ms summed CUDA kernels
(231.9410 tok/s/GPU). Its result SHA-256 is `cd00f7ef...`.

YaRN 32K--128K/b3 job `5451941` finished in 6:28 with exact per-rank lengths
`[32768,81920,131072]`, 245,952/799,792 KV tokens, the same graph proof,
31.0918 ms median request step (96.4885 tok/s/GPU), and 20.0799 ms summed CUDA
kernels (149.4028 tok/s/GPU). Its result SHA-256 is `b74672b6...`. Each run has
16 rank JSONs, four `.nsys-rep`/SQLite pairs, four clean final GPU snapshots,
and `SUCCESS`. Both AFD peers remain active at the first five-minute boundary.

The two validated baseline slots advanced at 12:39 PDT to the endpoint-only
b2 cases: native 1K--4K job `5452019` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b2_20260717_123953` and YaRN
32K--128K job `5452038` in
`vllm_irregular_qwen3_32768-131072_dp16_ep16_b2_20260717_123959`. Together
with the two original AFD b3 probes, exactly four jobs are active.

AFD native 1K--4K/b3 job `5451938` completed `0:0` in 7:46. Its exact
per-attention-rank distribution `[1024,2560,4096]`, prompt manifest, 84 x
16-token outputs, two identical 15-step padded-b4 graph windows on all 28
attention workers, eight final snapshots, and `SUCCESS` passed. The primary
result is 17.2346 ms median coordinator interval and 152.3095 tok/s/GPU; the
full 14-interval vector is retained. KV preflight is 7,872/839,424 tokens. The
result SHA-256 is `c0b4faef...`. Its 32K--128K peer remained active at 9:44.

AFD YaRN 32K--128K/b3 job `5451940` then completed `0:0` in 9:54. Its exact
per-rank distribution `[32768,81920,131072]`, 84 outputs, two identical
15-step padded-b4 graph windows on all 28 workers, eight snapshots, and profile
contract passed. The primary result is 20.1553 ms median coordinator interval
and 130.2386 tok/s/GPU; KV preflight is 245,952/839,424. Result SHA-256 is
`d7441aa8...`.

The two AFD slots advanced to endpoint-only b2: native job `5452056` in
`afd_irregular_qwen3_1024-4096_b2_n8_20260717_124242` and YaRN job `5452059`
in `afd_irregular_qwen3_32768-131072_b2_n8_20260717_124313`, both retaining
the four known head-node exclusions. Together with vLLM b2 jobs `5452019` and
`5452038`, exactly four jobs are queued/running.

`build_qwen3_irregular_sweep_report.py` is the fail-closed collector for this
sweep. It encodes all 73 AFD/71 vLLM expected keys, regenerates and verifies
every integer-linspace distribution, validates topology/KV/graph/result fields
and complete worker/rank/profile/snapshot artifacts, rejects duplicates or
unexpected keys, and writes a flat 144-row CSV only when complete. Its
`--allow-partial` mode audited the four completed b3 results and wrote a valid
four-row scratch CSV. A first `py_compile` attempt was denied only because the
macOS system Python tried to create its cache under the non-writable user
Library; in-memory compilation passed, as did actual collector execution and
`git diff --check`.

Three b2 results completed and the collector's full artifact/schema audit
advanced cleanly from four to seven rows. AFD native job `5452056` finished in
4:26 at 16.2914 ms / 107.4183 tok/s/GPU (SHA `c9cec7ec...`). vLLM native job
`5452019` finished in 5:09 at 24.4901 ms / 81.6657 request tok/s/GPU and
155.8379 CUDA tok/s/GPU (SHA `782cc42e...`). vLLM YaRN job `5452038` finished
in 6:09 at 28.1363 ms / 71.0825 request tok/s/GPU and 113.4023 CUDA tok/s/GPU
(SHA `2c443e99...`). AFD YaRN b2 remained active.

Their three slots advanced at 12:48 PDT to AFD native 1K--4K/b4 job `5452125`,
vLLM native 1K--4K/b4 job `5452126`, and vLLM YaRN 32K--128K/b4 job `5452127`.
Together with AFD YaRN b2 job `5452059`, exactly four jobs are queued/running.

AFD YaRN 32K--128K/b2 job `5452059` produced complete valid artifacts at
8:28 and remained briefly in Ray/Slurm teardown. The eight-row collector audit
accepted its exact endpoint distribution, 56 outputs, all 28 graph logs and
eight snapshots. Its primary result is 20.9356 ms / 83.5897 tok/s/GPU, KV is
164,032/839,424, and result SHA-256 is `285af010...`. The 32K--128K b2 pair is
therefore complete in both modes.

The native 1K--4K/b4 pair then completed `0:0` and advanced the fail-closed
collector to ten audited rows. Both modes used exact lengths
`[1024,2048,3072,4096]` and required 10,496 KV tokens/GPU. AFD job `5452125`
measured 16.0570 ms / 217.9736 tok/s/GPU (SHA `bd327e6c...`); vLLM job
`5452126` measured 24.6271 ms / 162.4228 request tok/s/GPU and 13.5249 ms /
295.7510 CUDA tok/s/GPU (SHA `8263572c...`).

At 12:56 PDT, the three released slots advanced to AFD YaRN 32K--128K/b4 job
`5452218` in `afd_irregular_qwen3_32768-131072_b4_n8_20260717_125611`, AFD
native 1K--4K/b6 job `5452219` in
`afd_irregular_qwen3_1024-4096_b6_n8_20260717_125612`, and vLLM native
1K--4K/b6 job `5452220` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b6_20260717_125612`. AFD retained
the four artifact-finalization head-node exclusions. Together with active vLLM
YaRN 32K--128K/b4 job `5452127`, exactly four jobs are active.

vLLM YaRN b4 job `5452127` completed `0:0` in 6:32 and advanced the collector
to eleven audited rows. It used exact lengths
`[32768,65536,98304,131072]`, required 327,936/799,792 KV tokens, and measured
34.1145 ms / 117.2520 request tok/s/GPU and 23.0269 ms / 173.7096 CUDA
tok/s/GPU (SHA `443f6e3e...`). Its slot advanced at 12:59 PDT to vLLM YaRN
32K--128K/b6 job `5452260` in
`vllm_irregular_qwen3_32768-131072_dp16_ep16_b6_20260717_125903`, restoring
four active jobs.

vLLM native b6 job `5452220` completed `0:0` in 5:17 and advanced the collector
to twelve audited rows. It used exact symmetric lengths
`[1024,1638,2253,2867,3482,4096]`, required 15,776/799,920 KV tokens, and
measured 24.9570 ms / 240.4136 request tok/s/GPU and 13.9424 ms / 430.3417
CUDA tok/s/GPU (SHA `b74fb136...`). Its slot advanced at 13:04 PDT to vLLM
native b8 job `5452291` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b8_20260717_130358`.

AFD native b6 job `5452219`, AFD YaRN b4 job `5452218`, and vLLM YaRN b6 job
`5452260` completed `0:0`; the collector accepted all artifacts and advanced to
fifteen rows. AFD native b6 used the same exact six-length distribution as its
baseline peer and measured 16.3143 ms / 321.8027 tok/s/GPU with
15,872/839,424 KV tokens (SHA `f6836817...`). AFD YaRN b4 measured 20.0819 ms
/ 174.2859 tok/s/GPU with 327,936/839,424 KV tokens (SHA `5342e810...`). vLLM
YaRN b6 used `[32768,52429,72090,91750,111411,131072]`, required
491,936/799,792 KV tokens, and measured 38.7544 ms / 154.8213 request
tok/s/GPU and 27.7170 ms / 216.4734 CUDA tok/s/GPU (SHA `32be5ee3...`).

At 13:10 PDT, those slots advanced to AFD native b8 job `5452372` in
`afd_irregular_qwen3_1024-4096_b8_n8_20260717_130944`, AFD YaRN b6 job
`5452373` in `afd_irregular_qwen3_32768-131072_b6_n8_20260717_130945`, and
vLLM YaRN terminal b8 job `5452374` in
`vllm_irregular_qwen3_32768-131072_dp16_ep16_b8_20260717_130945`. Both AFD
submissions retain the four head-node exclusions. Together with active vLLM
native b8 job `5452291`, exactly four jobs are active.

vLLM native b8 job `5452291` completed `0:0` in 5:14 and advanced the
collector to sixteen audited rows. Its eight exact lengths are
`[1024,1463,1902,2341,2779,3218,3657,4096]`; KV is 21,040/799,920 and the
result is 25.1803 ms / 317.7086 request tok/s/GPU and 13.9460 ms / 573.6394
CUDA tok/s/GPU (SHA `3b19bb6f...`). The released slot advanced at 13:12 PDT
to AFD YaRN terminal b8 job `5452436` in
`afd_irregular_qwen3_32768-131072_b8_n8_20260717_131229`, retaining the head-
node exclusions.

An explicit sanity comparison against the completed uniform/single-ISL sweep
confirms the live paths are genuinely irregular. For 32K--128K, vLLM summed
CUDA latency at b2/b3/b4/b6 is respectively 17.636/20.080/23.027/27.717 ms,
strictly between the same-batch uniform 32K values
15.180/15.646/16.601/18.502 ms and uniform 128K values
19.606/23.206/27.389/34.998 ms. The irregular points occupy a stable
55.5%--59.6% of the endpoint latency span, consistent with their exact 80K
mean ISL rather than either endpoint. vLLM synchronized request latency shows
the same intermediate pattern. AFD long-context latency is noisier but is
15%--21% above uniform 32K for b2--b4 and moves away from uniform 128K by b4.
For every audited result, the stronger correctness evidence is the collector's
verification of each distinct per-rank/per-attention-worker prompt list and its
page-rounded KV sum; no irregular artifact contains a repeated single ISL.
The short 1K--4K latency delta is small relative to run noise, so it is not used
alone as proof even though its manifests also contain all exact distinct ISLs.

AFD native b8 job `5452372` completed `0:0` in 5:05 and vLLM YaRN terminal b8
job `5452374` completed `0:0` in 8:08; the collector accepted both and advanced
to eighteen rows. Both retained their exact eight-point distributions. AFD
native b8 used 21,184/839,424 KV tokens and measured 16.9815 ms / 412.2137
tok/s/GPU (SHA `f9cabd49...`). vLLM YaRN b8 used 655,920/799,792 KV tokens and
measured 44.0727 ms / 181.5181 request tok/s/GPU and 33.0072 ms / 242.3711
CUDA tok/s/GPU (SHA `8319a73c...`). The latter also preserves the intermediate
uniform-endpoint sanity trend.

At 13:20 PDT, those slots advanced to AFD native b12 job `5452656` in
`afd_irregular_qwen3_1024-4096_b12_n8_20260717_131937` and vLLM native b12 job
`5452657` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b12_20260717_131937`. AFD retains
the four head-node exclusions. Together with active AFD YaRN b6/b8 jobs
`5452373`/`5452436`, exactly four jobs are active; the vLLM 32K--128K range is
complete through its terminal b8.

vLLM native b12 job `5452657` completed `0:0` in 4:01 and advanced the
collector to nineteen audited rows. All 12 exact lengths and graph artifacts
passed; KV is 31,568/799,920 and the result is 25.4265 ms / 471.9482 request
tok/s/GPU and 14.6132 ms / 821.1751 CUDA tok/s/GPU (SHA `edae3e63...`). Its
slot advanced at 13:28 PDT to vLLM native b16 job `5452717` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b16_20260717_132753`.

AFD YaRN b6/b8 exceeded the earlier small-batch wall times but remained live:
their Nsight `.qdstrm` files continued changing and being converted/removed,
with b6 down to three outstanding streams and b8 producing final `.nsys-rep`
files at the 13:28 phase probe. No result assembly or failure marker existed,
so both were left running and no fallback or resubmission was introduced.

AFD YaRN terminal b8 job `5452436` completed `0:0` in 13:47, AFD native b12
job `5452656` completed `0:0` in 6:09, and vLLM native b16 job `5452717`
completed `0:0` in 4:12. The fail-closed collector accepted all three and
advanced to 22 rows. AFD YaRN b8 measured 22.2917 ms / 314.0177 tok/s/GPU
with 656,064/839,424 KV tokens (SHA `cc0cc4be...`). AFD native b12 measured
17.3926 ms / 603.7046 tok/s/GPU with 31,808/839,424 KV tokens (SHA
`1af10834...`). vLLM native b16 measured 26.1601 ms / 611.6194 request
tok/s/GPU and 15.2707 ms / 1,047.7590 CUDA tok/s/GPU with 42,080/799,920 KV
tokens (SHA `b7ff965a...`). The 32K--128K range is complete except for the AFD
b6 retry described below.

AFD YaRN b6 attempt `5452373` failed after 18:06 on nodes
`nvl72063-T[02-03,06,13-17]`. The workload itself completed: all 168 prompts
returned exactly 16 tokens, every one of 28 attention logs contained the same
30 target graph replays, and the generated distribution/KV contract was exact.
Post-run validation then found no `coordinator_nvtx_cpu.json`
(`alignment_json_missing`) although three coordinator `.nsys-rep` files and all
attention NVTX JSONs existed; the silent `[[ -f coordinator_nvtx_cpu.json ]]`
gate therefore exited 1 before result assembly. The tokenizer's separate
131,087 > 131,072 warning was nonfatal and came from full prompt-plus-output
inspection; generated chat prompts are exactly 131,072 and max sequence is
131,136. This is the same coordinator artifact-finalization class as the four
previously excluded AFD head nodes, not a runtime or capacity failure. No prior
successful AFD job used head `nvl72063-T02`, so it was added to the run-time
exclusion list; no compute-path fallback was added.

At 13:38 PDT, the four slots advanced to AFD YaRN b6 retry `5452813` in
`afd_irregular_qwen3_32768-131072_b6_n8_20260717_133733`, AFD native b16 job
`5452814` in `afd_irregular_qwen3_1024-4096_b16_n8_20260717_133736`, vLLM
native b24 job `5452815` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b24_20260717_133736`, and vLLM
4K--8K/b2 job `5452816` in
`vllm_irregular_qwen3_4096-8192_dp16_ep16_b2_20260717_133736`. Both AFD jobs
exclude the original four heads plus `nvl72063-T02`; exactly four jobs are
queued/running.

The two vLLM jobs completed `0:0` in 4:15 (native b24) and 4:20 (4K--8K/b2),
and the collector advanced to 24 audited rows. Native b24 used all 24 exact
lengths, required 63,152/799,920 KV tokens, and measured 27.9899 ms / 857.4512
request tok/s/GPU and 16.6372 ms / 1,442.5520 CUDA tok/s/GPU (SHA
`9441c882...`). 4K--8K/b2 used exact endpoints `[4096,8192]`, required
12,416/799,920 KV tokens, and measured 23.7957 ms / 84.0488 request tok/s/GPU
and 13.1045 ms / 152.6198 CUDA tok/s/GPU (SHA `daac005d...`).

At 13:46 PDT, those slots advanced to AFD native b24 job `5452937` in
`afd_irregular_qwen3_1024-4096_b24_n8_20260717_134548` (five head exclusions)
and vLLM native b32 job `5452938` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b32_20260717_134548`. Together with
AFD YaRN b6 retry `5452813` and AFD native b16 `5452814`, exactly four jobs are
active.

AFD native b16 attempt `5452814` failed after 8:35 on
`nvl72125-T[01,04-05,12-16]` with the same isolated artifact-finalization
signature as b6 attempt `5452373`: all 448 prompts generated 16 tokens, all 28
attention workers recorded identical 30 target graph replays, a valid
coordinator `.nsys-rep` existed, but `coordinator_nvtx_cpu.json` did not, so the
post-preset file gate exited 1 before result assembly. Prior uniform AFD b16
job `5439476` succeeded with the same profile path on head `nvl72022-T01`,
excluding batch size as the cause. `nvl72125-T01` was therefore added as the
sixth artifact-head exclusion, and only the missing AFD native b16 point was
resubmitted at 13:49 PDT as job `5452974` in
`afd_irregular_qwen3_1024-4096_b16_n8_20260717_134909`. No workload setting or
compute path changed; with the other three live jobs, four jobs are active.

vLLM native b32 job `5452938` completed `0:0` in 4:22 and advanced the
collector to 25 audited rows. All 32 exact lengths passed; KV is
84,208/799,920 and the result is 29.0993 ms / 1,099.6821 request tok/s/GPU and
17.4852 ms / 1,830.1189 CUDA tok/s/GPU (SHA `0ca06526...`). Its slot advanced
at 13:52 PDT to vLLM native b48 job `5452994` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b48_20260717_135209`, restoring four
active jobs.

AFD native b24 job `5452937` completed `0:0` in 9:06, and corrected AFD native
b16 job `5452974` produced complete valid artifacts during Slurm teardown. The
collector accepted both and advanced to 27 audited rows. AFD b16 used
42,368/839,424 KV tokens and measured 17.1497 ms / 816.3425 tok/s/GPU (SHA
`1354dd29...`); AFD b24 used 63,680/839,424 KV tokens and measured 16.8657 ms /
1,245.1321 tok/s/GPU (SHA `29c6db74...`). The head-node-only b16 rerun therefore
resolved the missing artifact without altering the workload.

At 13:58 PDT, the confirmed b24 slot advanced to AFD native b32 job `5453054`
in `afd_irregular_qwen3_1024-4096_b32_n8_20260717_135809`, using all six
artifact-head exclusions. AFD b16 was still briefly listed during teardown, so
no second replacement was submitted; the live/teardown set remained at the
four-job cap.

AFD batch semantics were clarified at the user's request. The positional AFD
batch is the total real input batch per attention-DP lane before
microbatching—not a per-microbatch value. With fixed `AFD_NUM_MB=2`, even bB
uses real halves B/2+B/2; odd bB uses ceil(B/2)+floor(B/2), and both graph
replays use the ceil bucket with dummy padding only in the smaller microbatch.
Thus global real prompts are `28*B`, never `28*B*2`. `run_afd.sh` now labels
the argument `input-batch/attention-GPU`, reports `microbatch_real_sizes`, and
adds `input_batch_per_attention_gpu` plus
`microbatch_real_sizes_per_attention_gpu` to new result JSONs while preserving
the prior `batch_per_attention_gpu` compatibility field. Local and remote
syntax/dry runs prove b6 -> 3+3, b3 -> 2+1 with padded graph input 4, and
uniform b6 -> 3+3. The sole installed AFD launcher is byte-synced at SHA-256
`431bc0ad...`; pinned source remains clean. Jobs already running had entered
their spooled script before the sync and retain identical numerical semantics.

vLLM native b48 job `5452994` completed `0:0` in 6:00 and advanced the
collector to 28 audited rows. All 48 exact lengths passed; KV is
126,304/799,920 and the result is 32.6609 ms / 1,469.6490 request tok/s/GPU and
20.8030 ms / 2,307.3624 CUDA tok/s/GPU (SHA `43f63df6...`). At 14:03 PDT the
two free slots advanced to AFD native b48 job `5453145` in
`afd_irregular_qwen3_1024-4096_b48_n8_20260717_140253` (six head exclusions)
and vLLM native b64 job `5453146` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b64_20260717_140254`. Together with
AFD YaRN b6 retry `5452813` and AFD native b32 `5453054`, four jobs are active.

The collector/report schema now makes the same batch semantics unambiguous:
it preserves `batch_per_lane` but adds identical `input_batch_per_lane`, plus
AFD-only `afd_real_microbatch_sizes` and
`afd_graph_batch_per_microbatch`. It conditionally audits the new launcher
result fields so earlier rows remain valid. In-memory compilation,
`git diff --check`, and a full partial collection passed; the refreshed CSV
shows, for example, AFD b3 as input b3 / real 2,1 / graph b2 and AFD b32 as
input b32 / real 16,16 / graph b16.

AFD native b32 job `5453054` completed `0:0` in 5:26 and advanced the collector
to 29 audited rows. It used 84,928/839,424 KV tokens and measured 18.1064 ms /
1,546.4132 tok/s/GPU (SHA `a5040783...`); its explicit input b32 maps to real
microbatches 16+16 and 896 global prompts.

AFD YaRN b6 retry `5452813` completed the workload but reproduced the missing
coordinator JSON failure on sibling head `nvl72063-T01`; its first attempt used
`nvl72063-T02`. Both failed attempts had zero outstanding `.qdstrm` files,
three valid coordinator `.nsys-rep` files, all request outputs, and all graph
proofs, but no `coordinator_nvtx_cpu.json` or result. Successful YaRN b4/b8
runs also have three coordinator reports and do have the JSON, isolating this
to the `nvl72063` head group rather than b6/report multiplicity. The next b6
retry will exclude the full `nvl72063-T[01-17]` group and wait for the failed
job's own Slurm slot to clear.

At 14:06 PDT, the already-cleared b32 slot advanced to vLLM native b96 job
`5453209` in `vllm_irregular_qwen3_1024-4096_dp16_ep16_b96_20260717_140549`.
Together with the tearing-down b6 attempt and active AFD b48/vLLM b64, the
queue remains at four jobs.

After failed attempt `5452813` cleared, the unchanged AFD YaRN b6 point was
resubmitted at 14:09 PDT as job `5453244` in
`afd_irregular_qwen3_32768-131072_b6_n8_20260717_140834`, excluding the six
known individual bad heads and the full `nvl72063-T[01-17]` group. It queued
normally and restores the fourth job without changing the workload.

AFD native b48 job `5453145` completed `0:0` in 6:02 and vLLM native b64 job
`5453146` completed `0:0` in 5:51; the collector accepted both and advanced to
31 audited rows. AFD b48 explicitly records input b48 -> real 24+24 / graph
b24, uses 127,424/839,424 KV tokens, and measures 21.0158 ms / 1,998.4967
tok/s/GPU (SHA `85032ac2...`). vLLM b64 uses 168,416/799,920 KV tokens and
measures 33.8773 ms / 1,889.1730 request tok/s/GPU and 22.7227 ms / 2,816.5728
CUDA tok/s/GPU (SHA `2f829c73...`).

At 14:11 PDT, those slots advanced to AFD native b64 job `5453270` in
`afd_irregular_qwen3_1024-4096_b64_n8_20260717_141114` (full bad-group
exclusions) and vLLM native b128 job `5453271` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b128_20260717_141115`. Together with
vLLM b96 `5453209` and pending AFD YaRN b6 retry `5453244`, four jobs are
queued/running.

vLLM native b96 job `5453209` completed `0:0` in 5:05 and advanced the
collector to 32 audited rows. All 96 exact lengths passed; KV is
252,624/799,920 and the result is 41.1475 ms / 2,333.0678 request tok/s/GPU and
28.6274 ms / 3,353.4261 CUDA tok/s/GPU (SHA `e2b8eebf...`). Its slot advanced
at 14:14 PDT to vLLM native b192 job `5453309` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b192_20260717_141405`. AFD YaRN b6
retry `5453244` is running on the known-good `nvl72094` group; AFD b64/vLLM
b128 are queued, preserving four active jobs.

AFD YaRN b6 attempt `5453244` failed after 7:54 on known-good head group
`nvl72094`, disproving the earlier head-group hypothesis. Like the first two
attempts it completed all 168x16 outputs and every graph proof, then aborted
before coordinator JSON/result assembly. The exact cause is now confirmed in
`afd-coordinator.log`: ZeroMQ asserts `msg.flags() & msg_t::more` in
`src/pipe.cpp:242` and delivers `SIGABRT` during coordinator shutdown, after
`wait_worker_hot_loops` times out but before `shutdown_nvtx_cpu_trace()` runs.
Successful b4/b8 and native b6 reach the explicit `cpu_trace flushed` line;
all three long b6 attempts abort first. Extending the external shutdown timeout
cannot fix this internal ordering race. The clean fix is to flush the complete
coordinator CPU trace before the risky queue/worker teardown, which is post-
measurement but changes the pinned source commit. No such source change or
fallback extraction was made without user direction; AFD YaRN b6 is the sole
deferred point while all unaffected sweep work continues. The full-group
exclusion remains on subsequent AFD submissions for the independently observed
artifact-head failures, but it is not considered the b6 fix.

At 14:23 PDT, the released b6 slot advanced to AFD native b96 job `5453427` in
`afd_irregular_qwen3_1024-4096_b96_n8_20260717_142318`, using the clarified
input-batch launcher and current artifact-head exclusions. Together with queued
AFD b64 and vLLM b128/b192, four jobs are active.

AFD native b96 attempt `5453427` failed before engine launch in prompt
construction: source row 64 could not hit exact 1,089 chat tokens within the
old 64-step prefix adjustment. A tokenizer probe proved this is a byte-pair
merge gap for that source: adjacent prefix choices jump from 1,088 to 1,090;
the same target had succeeded at b48 because it mapped to different source
rows. A fixed ordered suffix probe found an exact deterministic construction
at 1,079 source tokens plus suffix `" 0"`.

`run_afd.sh` now retains its existing adaptive exact construction first, then
for only an unreachable merge gap searches +/-64 source-token prefixes with a
fixed ordered one-line suffix set. It still rejects approximate lengths and
records every correction as source index, target tokens, and suffix in
`exact_length_merge_gap_repairs`; the aggregate token-ID hash continues to
cover the exact final prompts. Shell syntax, all three embedded Python blocks,
b96 dry run, and `git diff --check` passed. The sole remote launcher is synced
at SHA-256 `43cd5e3f...`, remote b96 dry run passed, and pinned source remained
clean. End-to-end b96 retry `5453537` was submitted at 14:34 PDT in
`afd_irregular_qwen3_1024-4096_b96_n8_20260717_143349`; its manifest must show
the repair before the collector can accept it.

AFD native b64 job `5453270` produced complete valid artifacts during Slurm
teardown, while vLLM native b128/b192 jobs `5453271`/`5453309` completed `0:0`
in 6:55/6:16. The collector accepted all three and advanced to 35 audited
rows. AFD input b64 -> real 32+32 / graph b32 used 169,856/839,424 KV tokens
and measured 24.3012 ms / 2,304.4166 tok/s/GPU (SHA `b888d0d4...`). vLLM b128
used 336,832/799,920 KV tokens and measured 43.1572 ms / 2,965.8986 request
tok/s/GPU and 31.9383 ms / 4,007.7244 CUDA tok/s/GPU (SHA `9882eb50...`). vLLM
b192 used 505,248/799,920 KV tokens and measured 56.0968 ms / 3,422.6571
request tok/s/GPU and 42.2642 ms / 4,542.8532 CUDA tok/s/GPU (SHA
`b20a00c5...`).

At 14:35 PDT, the two free slots advanced to AFD native b128 job `5453549` in
`afd_irregular_qwen3_1024-4096_b128_n8_20260717_143507` and terminal vLLM
native b256 job `5453550` in
`vllm_irregular_qwen3_1024-4096_dp16_ep16_b256_20260717_143507`. Together with
the repaired AFD b96 retry and brief b64 teardown, the set remains capped at
four jobs.

After b64 teardown cleared, the fourth slot advanced at 14:38 PDT to AFD
4K--8K/b2 job `5453569` in
`afd_irregular_qwen3_4096-8192_b2_n8_20260717_143803`, using exact endpoints,
input b2 -> real 1+1, the updated exact prompt constructor, and current
artifact-head exclusions. Jobs `5453537`, `5453549`, `5453550`, and `5453569`
then remained pending solely by scheduler priority through the 14:43 check;
the four-job cap includes pending jobs, so nothing additional was submitted.

The user clarified that range coverage should now advance beyond the 1K lower
bound. Current status is not uniform 1K: the active native range is irregular
1K--4K. After the already queued native points, submission order changes to
breadth-first coverage across unopened ranges while preserving every remaining
native batch in the plan. At this point 32K--128K is complete except deferred
AFD b6, vLLM 1K--4K is complete through terminal b256, AFD 1K--4K is complete
through b64 with b96/b128 in flight, and 4K--8K has vLLM b2 complete plus AFD
b2 queued.

Repaired AFD native b96 job `5453537` completed `0:0` in 8:31 and advanced the
collector to 36 audited rows. Its manifest records exactly one fail-closed
repair—source index 64, target 1,089, suffix `" a"`—and all other prompts use
the original adaptive construction. The result explicitly records input b96
-> real 48+48 / graph b48, 2,688 global prompts, 254,912/839,424 KV tokens,
and 30.2477 ms / 2,777.0728 tok/s/GPU (SHA `a6ae35e4...`). This validates the
merge-gap correction end to end.

At 15:04 PDT, the released slot opened the 8K--32K range with vLLM b2 job
`5453825` in
`vllm_irregular_qwen3_8192-32768_dp16_ep16_b2_20260717_150410`. Together with
running AFD native b128 and pending vLLM native b256/AFD 4K--8K b2, four jobs
are active or queued.

AFD native b128 job `5453549` completed `0:0` in 7:05 and advanced the
collector to 37 audited rows. It explicitly records input b128 -> real 64+64 /
graph b64, 3,584 global prompts, 339,904/839,424 KV tokens, and 37.1229 ms /
3,017.0057 tok/s/GPU (SHA `da285bfe...`).

At 15:15 PDT, that slot opened the matching AFD side of 8K--32K with b2 job
`5454004` in `afd_irregular_qwen3_8192-32768_b2_n8_20260717_151526`, using
exact endpoints, input b2 -> real 1+1, and current artifact-head exclusions.
Together with running vLLM native b256 and vLLM 8K--32K/b2 plus pending AFD
4K--8K/b2, four jobs are active or queued.

vLLM 8K--32K/b2 job `5453825` completed `0:0` in 4:15 and advanced the
collector to 38 audited rows. Its exact per-lane prompt list is `[8192,
32768]`; it used 41,088/799,920 KV tokens and measured 27.2266 ms /
73.4576 request tok/s/GPU and 15.4470 ms / 129.4752 CUDA tok/s/GPU (SHA
`7585d78f...`). This is the first audited point in that range.

At 15:18 PDT, the released slot opened 1K--16K with vLLM b2 job `5454040` in
`vllm_irregular_qwen3_1024-16384_dp16_ep16_b2_20260717_151836`. The active or
queued set is still exactly four: running vLLM native b256, pending AFD
4K--8K/b2 and AFD 8K--32K/b2, and this new breadth-first vLLM point.

Terminal vLLM 1K--4K/b256 job `5453550` completed `0:0` in 6:33, finishing the
vLLM side of that range and advancing the collector to 39 audited rows. It
used the exact 256-value inclusive linspace, 673,664/799,920 KV tokens, and
measured 64.5160 ms / 3,968.0057 request tok/s/GPU and 50.5585 ms /
5,063.4387 CUDA tok/s/GPU (SHA `e7f4b358...`).

At 15:21 PDT, its released slot opened 4K--64K with vLLM b2 job `5454067` in
`vllm_irregular_qwen3_4096-65536_dp16_ep16_b2_20260717_152119`. The launcher
revalidated the existing 65,600-token YaRN profile before submission. Together
with the two pending AFD b2 jobs and pending vLLM 1K--16K/b2, this preserves the
four-job cap and continues breadth-first range coverage.

AFD 4K--8K/b2 job `5453569` completed `0:0` in 7:46 and advanced the collector
to 40 audited rows. Its result and supporting rank artifacts confirm exact
prompts `[4096, 8192]`, input b2 -> real 1+1 / graph b1, 56 global prompts,
12,416/839,424 KV tokens, and 16.1851 ms / 108.1244 tok/s/GPU (SHA
`ddf4b30b...`). Thus this point is demonstrably a two-length irregular batch,
not a repeated single ISL.

At 15:31 PDT, its released slot opened 8K--128K with vLLM b2 job `5454177` in
`vllm_irregular_qwen3_8192-131072_dp16_ep16_b2_20260717_153049`. The launcher
revalidated the existing 131,136-token YaRN profile. With AFD 8K--32K/b2,
vLLM 1K--16K/b2, and vLLM 4K--64K/b2 still queued, four breadth-first jobs are
again active or pending.

AFD 8K--32K/b2 job `5454004` completed `0:0` in 7:56. Its artifacts passed
the collector before Slurm teardown finished and confirm exact prompts `[8192,
32768]`, input b2 -> real 1+1 / graph b1, 56 global prompts,
41,088/839,424 KV tokens, and 17.4384 ms / 100.3534 tok/s/GPU (SHA
`d3ba56d3...`). vLLM jobs `5454040`, `5454067`, and `5454177` completed `0:0`
in 5:35, 5:31, and 5:56, respectively. They add audited b2 points for
1K--16K (24.8105 ms request / 13.5861 ms CUDA; SHA `721a11fc...`), 4K--64K
(27.3532 / 14.8919 ms; SHA `ccd70881...`), and 8K--128K (28.4784 / 16.8104
ms; SHA `1a20203e...`). The partial collector advanced from 40 to 44 rows.

At 15:48--15:49 PDT, the four goal-scoped slots advanced to vLLM
1K--128K/b2 `5454404`, AFD 1K--16K/b2 `5454405`, AFD 4K--64K/b2 `5454407`,
and AFD 8K--128K/b2 `5454418`. All AFD launches retain the known artifact-head
exclusions. This means every requested range now has an audited or active b2
point on both modes except AFD 1K--128K, which is next after a slot releases.

The user clarified that the maximum of four applies to this probing goal even
while another session submits unrelated jobs. Both launchers now accept
`FASTAFD_ACTIVE_JOB_REGEX`; it defaults to the previous account-wide
`^fastafd:` behavior, remains nonempty/fail-fast, and restricts only the
submission-time count when explicitly set. This sweep will use
`^fastafd:(afd|vllm)-irregular-qwen3-`, while exact active job IDs remain the
authoritative monitor set. Local syntax, a synthetic two-match/one-nonmatch
count, and diff checks passed. Remote scripts passed syntax at SHA-256
`776b6460...` (AFD) and `a6e73a2a...` (vLLM); pinned source remained clean.

vLLM 1K--128K/b2 job `5454404` and AFD 1K--16K/b2 job `5454405` completed
`0:0` in 4:18 and 4:25, advancing the collector to 46 audited rows. The vLLM
point confirms exact endpoints `[1024, 131072]`, 132,224/799,792 KV tokens,
and 26.9873 ms request / 16.5441 ms CUDA (SHA `6db8cc92...`). The AFD point
confirms exact endpoints `[1024, 16384]`, input b2 -> real 1+1 / graph b1,
17,536/839,424 KV tokens, and 17.2776 ms / 101.2871 tok/s/GPU (SHA
`fb64c7b7...`).

At 15:56 PDT, their two released goal slots submitted AFD 1K--128K/b2 job
`5454527` and vLLM 4K--8K/b3 job `5454530`. The former is the final missing b2
mode/range pair; the latter begins the breadth-first b3 pass. Both submissions
used the explicit requested-range job regex, so unrelated account jobs do not
consume this goal's four-slot budget.

AFD 4K--64K/b2 job `5454407` completed `0:0` in 8:07 and advanced the
collector to 47 audited rows. It confirms exact endpoints `[4096, 65536]`,
input b2 -> real 1+1 / graph b1, 69,760/839,424 KV tokens, and 18.6123 ms /
94.0239 tok/s/GPU (SHA `2a1bc08f...`). At 15:59 PDT, its released goal slot
submitted matching AFD 4K--8K/b3 job `5454563`; the odd input batch will be
audited as real microbatches 2+1 with graph b2 and one padded graph request.

AFD 8K--128K/b2 job `5454418` completed `0:0` in 9:08 and advanced the
collector to 48 audited rows. It confirms exact endpoints `[8192, 131072]`,
input b2 -> real 1+1 / graph b1, 139,392/839,424 KV tokens, and 20.2566 ms /
86.3917 tok/s/GPU (SHA `e80b177d...`). At 16:02 PDT, its released goal slot
submitted vLLM 8K--32K/b3 job `5454614`, continuing the breadth-first b3 pass.

vLLM 4K--8K/b3 job `5454530` completed `0:0` in 5:09 and advanced the
collector to 49 audited rows. It confirms the exact inclusive list `[4096,
6144, 8192]`, 18,624/799,920 KV tokens, and 24.5517 ms request / 13.5199 ms
CUDA (SHA `70a2920d...`). At 16:07 PDT, its released goal slot submitted
matching AFD 8K--32K/b3 job `5454672`, whose odd input batch must audit as
real 2+1 / graph b2.

AFD 4K--8K/b3 job `5454563` completed `0:0` in 7:46 and vLLM 8K--32K/b3 job
`5454614` completed `0:0` in 5:12, advancing the collector to 51 audited rows.
AFD explicitly records the required input b3 -> real 2+1 / graph b2 split,
exact prompts `[4096, 6144, 8192]`, 18,624/839,424 KV tokens, and 17.3518 ms /
151.2811 tok/s/GPU (SHA `32d2895f...`). vLLM records exact prompts `[8192,
20480, 32768]`, 61,632/799,920 KV tokens, and 25.8589 ms request / 15.1819 ms
CUDA (SHA `97ebdb13...`).

At 16:10 PDT, the first released slot submitted vLLM 1K--16K/b3 job `5454730`.
The second was intentionally held until vLLM job `5454614` left Slurm
`COMPLETING`, then submitted matching AFD 1K--16K/b3 job `5454733`. External
`ratioep:*` arrays were present but correctly excluded from this goal's four
slots.

vLLM 1K--16K/b3 job `5454730` completed `0:0` in 5:14 and advanced the
collector to 52 audited rows. It confirms exact prompts `[1024, 8704, 16384]`,
26,304/799,920 KV tokens, and 26.9072 ms request / 14.8767 ms CUDA (SHA
`6910a236...`). At 16:21 PDT, its released goal slot submitted vLLM
4K--64K/b3 job `5454874` after revalidating the 65,600-token YaRN profile.

AFD 1K--128K/b2 remained live beyond 17 minutes rather than hanging: all 28
attention CPU traces were written around 16:13--16:14 and Nsight qdstrm files
were still updating at 16:19. The authoritative result gate had not yet
completed, so its slot remained occupied.

AFD 1K--16K/b3 job `5454733` completed `0:0` in 7:46 and advanced the
collector to 53 audited rows. It confirms input b3 -> real 2+1 / graph b2,
exact prompts `[1024, 8704, 16384]`, 26,304/839,424 KV tokens, and 17.2166 ms /
152.4695 tok/s/GPU (SHA `f243ff6d...`). At 16:27 PDT, its released goal slot
submitted matching AFD 4K--64K/b3 job `5454948`, after revalidating the same
65,600-token YaRN profile used by the running vLLM side.

AFD 8K--32K/b3 job `5454672` completed `0:0` in 8:04 and confirms input b3 ->
real 2+1 / graph b2, exact prompts `[8192, 20480, 32768]`,
61,632/839,424 KV tokens, and 17.1550 ms / 153.0163 tok/s/GPU (SHA
`8fa18333...`). vLLM 4K--64K/b3 wrote an artifact-valid result before leaving
Slurm, with exact prompts `[4096, 34816, 65536]`, 104,640/799,888 KV tokens,
and 26.9283 ms request / 16.5799 ms CUDA (SHA `594ffb06...`). Together they
advanced the artifact collector to 55 rows, but the vLLM slot remained occupied
until terminal teardown.

After AFD job `5454672` left Slurm, its released goal slot submitted vLLM
8K--128K/b3 job `5454985` at 16:30 PDT. The 131,136-token YaRN profile was
revalidated, and the still-running vLLM 4K--64K job continued to count toward
the four-job goal limit.

vLLM 4K--64K/b3 job `5454874` then completed `0:0` in 5:36. Only after it left
the queue did its goal slot submit matching AFD 8K--128K/b3 job `5454992` at
16:31 PDT. The latter uses the 131,136-token YaRN profile and must audit input
b3 as real 2+1 / graph b2.

AFD 4K--64K/b3 job `5454948` completed `0:0` in 7:30 and advanced the
collector to 56 audited rows. It confirms input b3 -> real 2+1 / graph b2,
exact prompts `[4096, 34816, 65536]`, 104,640/839,424 KV tokens, and 18.0920
ms / 145.0916 tok/s/GPU (SHA `438be3a6...`). At 16:38 PDT, its released goal
slot submitted vLLM 1K--128K/b3 job `5455097`, opening the final new range in
the b3 breadth pass after revalidating the 131,136-token YaRN profile.

vLLM 8K--128K/b3 job `5454985` completed `0:0` in 5:58 and advanced the
collector to 57 audited rows. It confirms exact prompts `[8192, 69632,
131072]`, 209,088/799,792 KV tokens, and 29.9253 ms request / 19.4203 ms CUDA
(SHA `085c5442...`). At 16:42 PDT, its released goal slot submitted matching
AFD 1K--128K/b3 job `5455169`; it uses the 131,136-token YaRN profile and must
audit input b3 as real 2+1 / graph b2.

AFD 8K--128K/b3 job `5454992` completed `0:0` in 9:35 and vLLM
1K--128K/b3 job `5455097` completed `0:0` in 4:31. They advanced the gated
collector to 59 audited rows. AFD confirms input b3 -> real 2+1 / graph b2,
exact prompts `[8192, 69632, 131072]`, 209,088/839,424 KV tokens, and 20.1527
ms / 130.2555 tok/s/GPU (SHA `bd06d927...`). vLLM confirms exact prompts
`[1024, 66048, 131072]`, 198,336/799,792 KV tokens, and 30.4389 ms request /
19.5432 ms CUDA (SHA `732062d1...`).

The first AFD 1K--128K/b2 attempt `5454527` was cancelled by the OCI occupied-
idle job reaper at 41:15, not by the workload: `scontrol` recorded "28/32 GPUs
idle for 30m, 88% waste". All 28 attention CPU traces and active Nsight qdstrm
export proved the workload had completed, but no coordinator CPU trace/result
or `SUCCESS` gate was produced. The 128K graph-node-expanded Nsight reports
kept 28 GPUs idle during post-workload export long enough to trigger the site
policy.

The AFD launcher now validates `NSYS_CUDA_GRAPH_TRACE=node|none`, defaults to
the unchanged `node`, carries the selected value through Slurm/container/dry-
run provenance, and records it in new result JSON. CUDA graphs remain enabled
and every worker's identical replay window remains mandatory when trace mode is
`none`; only Nsight's graph-node expansion is removed. Local shell syntax,
three embedded Python blocks, `none` dry run, invalid-value rejection, and diff
checks passed. Remote syntax/dry run passed at SHA-256 `84bd2160...`; pinned
source remained clean. Retry `5455276` was submitted at 16:51 PDT for only AFD
1K--128K/b2 with `NSYS_CUDA_GRAPH_TRACE=none`.

The partial collector now indexes only records with their authoritative
`SUCCESS` gate. This ignores result JSON written by live jobs before final
artifacts, while final mode still fails closed because an ungated expected key
is missing. In-memory syntax, diff, and live partial collection passed; the
usual macOS `py_compile` cache write outside the workspace was the only
irrelevant failed check.

The two remaining free goal slots then began the b4 breadth pass with vLLM
4K--8K/b4 job `5455300` and matching AFD 4K--8K/b4 job `5455321`. Together
with running AFD 1K--128K/b3 and the retry, this restored exactly four goal
jobs despite the external `ratioep:*` arrays.

AFD 1K--128K/b3 job `5455169` completed `0:0` in 9:18 and advanced the
collector to 60 `SUCCESS`-gated rows. It confirms input b3 -> real 2+1 /
graph b2, exact prompts `[1024, 66048, 131072]`, 198,336/839,424 KV tokens,
and 19.8376 ms / 132.3244 tok/s/GPU (SHA `c05e485a...`). At 16:55 PDT, its
released goal slot submitted vLLM 8K--32K/b4 job `5455363`, continuing the b4
breadth pass.

vLLM 4K--8K/b4 job `5455300` completed `0:0` in 4:26 and advanced the
collector to 61 gated rows. It confirms exact prompts `[4096, 5461, 6827,
8192]`, 24,848/799,920 KV tokens, and 24.5256 ms request / 13.7627 ms CUDA
(SHA `4ae672de...`). At 16:58 PDT, its released goal slot submitted matching
AFD 8K--32K/b4 job `5455441`.

The mitigated AFD 1K--128K/b2 retry `5455276` completed `0:0` in 5:07,
confirming the idle-reaper fix. Its gated result explicitly records CUDA graph
`enabled=true`, `nsys_cuda_graph_trace=none`, and identical replay proof across
every attention-worker log. It confirms exact endpoints `[1024, 131072]`,
input b2 -> real 1+1 / graph b1, 132,224/839,424 KV tokens, and 20.4755 ms /
85.4682 tok/s/GPU (SHA `a466e6a3...`).

AFD 4K--8K/b4 job `5455321` completed `0:0` in 7:35 and confirms input b4 ->
real 2+2 / graph b2, exact prompts `[4096, 5461, 6827, 8192]`,
24,896/839,424 KV tokens, and 16.4569 ms / 212.6771 tok/s/GPU (SHA
`1fee0b19...`). vLLM 8K--32K/b4 job `5455363` produced a gated result with
exact prompts `[8192, 16384, 24576, 32768]`, 82,176/799,920 KV tokens, and
27.3124 ms request / 15.6270 ms CUDA (SHA `5ea6c8c5...`). These three points
advanced the collector from 61 to 64 gated rows.

At 17:02 PDT, the two released goal slots submitted vLLM 1K--16K/b4 job
`5455509` and matching AFD job `5455513`, continuing the breadth-first b4 pass.
After vLLM 8K--32K/b4 left Slurm with terminal `0:0` in 5:32, its released
goal slot submitted vLLM 4K--64K/b4 job `5455533` at 17:03 PDT.

vLLM 1K--16K/b4 job `5455509` and vLLM 4K--64K/b4 job `5455533` completed
`0:0` in 5:20 and 4:08, advancing the collector to 66 gated rows. The former
confirms exact prompts `[1024, 6144, 11264, 16384]`, 35,072/799,920 KV tokens,
and 25.6866 ms request / 14.0890 ms CUDA (SHA `c0db6756...`). The latter
confirms `[4096, 24576, 45056, 65536]`, 139,520/799,888 KV tokens, and 28.3206
ms request / 17.2658 ms CUDA (SHA `80fedc08...`).

Their goal slots submitted matching AFD 4K--64K/b4 job `5455725` and, after
the second teardown cleared, vLLM 8K--128K/b4 job `5455730` at 17:19 PDT.

AFD 8K--32K/b4 job `5455441` completed `0:0` in 8:02 and advanced the
collector to 67 gated rows. It confirms input b4 -> real 2+2 / graph b2,
exact prompts `[8192, 16384, 24576, 32768]`, 82,176/839,424 KV tokens, and
17.1187 ms / 204.4545 tok/s/GPU (SHA `4b890ec9...`). At 17:23 PDT, its
released goal slot submitted matching AFD 8K--128K/b4 job `5455747`.

vLLM 8K--128K/b4 job `5455730` completed `0:0` in 4:45 and advanced the
collector to 68 gated rows. It confirms exact prompts `[8192, 49152, 90112,
131072]`, 278,784/799,792 KV tokens, and 32.3201 ms request / 21.5135 ms CUDA
(SHA `3f146e25...`). After teardown cleared, its goal slot submitted vLLM
1K--128K/b4 job `5455774` at 17:26 PDT, opening the final new range in the b4
pass.

AFD 1K--16K/b4 job `5455513` completed `0:0` in 4:30 and advanced the
collector to 69 gated rows. It confirms input b4 -> real 2+2 / graph b2,
exact prompts `[1024, 6144, 11264, 16384]`, 35,072/839,424 KV tokens, and
16.9039 ms / 207.0531 tok/s/GPU (SHA `ce65b3fb...`). At 17:29 PDT, its
released goal slot submitted matching AFD 1K--128K/b4 job `5455814`, so both
modes have now been submitted for every b4 range.

vLLM 1K--128K/b4 job `5455774` completed `0:0` in 4:44 and advanced the
collector to 70 gated rows. It confirms exact prompts `[1024, 44373, 87723,
131072]`, 264,464/799,792 KV tokens, and 32.6494 ms request / 21.3868 ms CUDA
(SHA `4f0a2c14...`). At 17:35 PDT, its released goal slot began the b6 breadth
pass with vLLM 4K--8K/b6 job `5455863`.

AFD 4K--64K/b4 job `5455725` completed `0:0` in 8:36 and advanced the
collector to 71 gated rows. It confirms input b4 -> real 2+2 / graph b2,
exact prompts `[4096, 24576, 45056, 65536]`, 139,520/839,424 KV tokens, and
18.8399 ms / 185.7757 tok/s/GPU (SHA `97f92d90...`). At 17:38 PDT, its
released goal slot submitted matching AFD 4K--8K/b6 job `5455898`.

AFD 8K--128K/b4 `5455747`, AFD 1K--128K/b4 `5455814`, and vLLM
4K--8K/b6 `5455863` completed `0:0` in 7:24, 6:30, and 5:14, advancing the
collector to 74 gated rows. AFD 8K--128K confirms input b4 -> real 2+2 /
graph b2, exact prompts `[8192, 49152, 90112, 131072]`,
278,784/839,424 KV tokens, and 19.9469 ms / 175.4656 tok/s/GPU (SHA
`51c3241e...`). AFD 1K--128K confirms `[1024, 44373, 87723, 131072]`,
264,512/839,424 KV tokens, and 20.3014 ms / 172.4021 tok/s/GPU (SHA
`622821d8...`). vLLM 4K--8K/b6 confirms `[4096, 4915, 5734, 6554, 7373,
8192]`, 37,280/799,920 KV tokens, and 25.7680 ms request / 14.4637 ms CUDA
(SHA `034b9457...`).

At 17:45 PDT, their three goal slots submitted the 8K--32K/b6 pair—vLLM
`5455964` and AFD `5455965`—plus vLLM 1K--16K/b6 `5455967`. Together with
pending AFD 4K--8K/b6, this restored exactly four goal jobs.

vLLM 1K--16K/b6 `5455967` and vLLM 8K--32K/b6 `5455964` completed `0:0` in
4:12 and 5:24, advancing the collector to 76 gated rows. The former confirms
exact prompts `[1024, 4096, 7168, 10240, 13312, 16384]`, 52,608/799,920 KV
tokens, and 26.1954 ms request / 14.7001 ms CUDA (SHA `4928eaf4...`). The
latter confirms `[8192, 13107, 18022, 22938, 27853, 32768]`,
123,296/799,920 KV tokens, and 27.7752 ms request / 17.1014 ms CUDA (SHA
`e97a9da8...`).

At 17:56--17:57 PDT, their goal slots submitted matching AFD 1K--16K/b6 job
`5456046` and, after the second teardown cleared, vLLM 4K--64K/b6 job
`5456053`.

vLLM 4K--64K/b6 job `5456053` completed `0:0` in 4:14 and advanced the
collector to 77 gated rows. It confirms exact prompts `[4096, 16384, 28672,
40960, 53248, 65536]`, 209,280/799,888 KV tokens, and 31.4571 ms request /
20.3859 ms CUDA (SHA `40ce528d...`). After teardown cleared, its goal slot
submitted matching AFD 4K--64K/b6 job `5456118` at 18:03 PDT.

AFD 4K--8K/b6 job `5455898` completed `0:0` in 8:23 and advanced the
collector to 78 gated rows (partial CSV SHA `38e815ec...`). It confirms input
b6 -> real microbatches 3+3 / graph b3, exact prompts `[4096, 4915, 5734,
6554, 7373, 8192]`, 37,376/839,424 KV tokens, and 16.4932 ms median /
318.3136 tok/s/GPU (result SHA `4441fdd7...`). At 18:56 PDT, its released
goal slot submitted vLLM 8K--128K/b6 job `5456454`; together with pending AFD
jobs `5455965`, `5456046`, and `5456118`, this restored exactly four
goal-scoped jobs. The other session's `ratioep:*` arrays remain outside the
goal-specific active-job regex and count.

vLLM 8K--128K/b6 job `5456454` completed `0:0` in 6:48 and advanced the
collector to 79 gated rows (partial CSV SHA `2ca28594...`). It confirms exact
prompts `[8192, 32768, 57344, 81920, 106496, 131072]`, 418,176/799,792 KV
tokens, and 37.6015 ms request / 26.2928 ms CUDA (result SHA
`464f58b1...`). At 19:07 PDT, its released goal slot submitted matching AFD
8K--128K/b6 job `5456585` with CUDA graphs still enabled and
`NSYS_CUDA_GRAPH_TRACE=none` to avoid trace-export graph-node expansion.

AFD 8K--32K/b6 job `5455965` and AFD 1K--16K/b6 job `5456046` completed
`0:0` in 9:10 and 8:41, advancing the collector to 81 gated rows (partial CSV
SHA `ca34b1ea...`). The former confirms input b6 -> 3+3 / graph b3, exact
prompts `[8192, 13107, 18022, 22938, 27853, 32768]`, 123,392/839,424 KV
tokens, and 17.6101 ms / 298.1241 tok/s/GPU (SHA `cd274295...`). The latter
confirms `[1024, 4096, 7168, 10240, 13312, 16384]`, 52,608/839,424 KV tokens,
and 18.0370 ms / 291.0690 tok/s/GPU (SHA `faed5eb6...`). At 19:17 PDT, their
released goal slots submitted the 1K--128K/b6 pair: vLLM job `5456676` and
AFD job `5456677`; the AFD job uses `NSYS_CUDA_GRAPH_TRACE=none` while
retaining mandatory graph replay.

AFD 4K--64K/b6 job `5456118` completed `0:0` in 8:55 and advanced the
collector to 82 gated rows (partial CSV SHA `4a958cd2...`). It confirms input
b6 -> 3+3 / graph b3, exact prompts `[4096, 16384, 28672, 40960, 53248,
65536]`, 209,280/839,424 KV tokens, and 18.8611 ms / 278.3508 tok/s/GPU
(result SHA `ea9415e6...`). At 19:18 PDT, its released goal slot began the b8
breadth pass with vLLM 4K--8K/b8 job `5456686`.

vLLM 1K--128K/b6 job `5456676` and vLLM 4K--8K/b8 job `5456686` completed
`0:0` in 7:04 and 6:17, advancing the collector to 84 gated rows (partial CSV
SHA `77859b24...`). The b6 long-range point confirms exact prompts `[1024,
27034, 53043, 79053, 105062, 131072]`, 396,704/799,792 KV tokens, and
37.1199 ms request / 26.3943 ms CUDA (SHA `01efe1ce...`). The b8 short-range
point confirms `[4096, 4681, 5266, 5851, 6437, 7022, 7607, 8192]`,
49,712/799,920 KV tokens, and 25.0176 ms request / 14.4439 ms CUDA (SHA
`625f1999...`). At 19:36 PDT, their released slots submitted AFD 4K--8K/b8
job `5456947` and vLLM 8K--32K/b8 job `5456951`.

AFD 1K--128K/b6 job `5456677` completed `0:0` in 11:34 and advanced the
collector to 85 gated rows (partial CSV SHA `763fdf0c...`). It confirms input
b6 -> 3+3 / graph b3, exact prompts `[1024, 27034, 53043, 79053, 105062,
131072]`, 396,800/839,424 KV tokens, and 20.9104 ms / 251.0717 tok/s/GPU
(result SHA `bc7371a6...`). At 19:39 PDT, its released slot submitted AFD
8K--32K/b8 job `5457000`.

At 19:47 PDT, AFD 8K--128K/b6 job `5456585` remained running at 19:47
elapsed, so a read-only progress diagnostic checked for a stall. It found all
28 attention CPU traces present, qdstrm files actively updated at 19:47, and
live Slurm-step I/O of about 19.4 GB read / 18.5 GB written. This is active
artifact export, not an idle-reaper recurrence; leave the job running and
continue the two-minute cadence.

AFD 4K--8K/b8 job `5456947` and vLLM 8K--32K/b8 job `5456951` completed
`0:0` in 4:35 and 5:50, advancing the collector to 87 gated rows (partial CSV
SHA `947b226c...`). AFD confirms input b8 -> 4+4 / graph b4, exact prompts
`[4096, 4681, 5266, 5851, 6437, 7022, 7607, 8192]`, 49,856/839,424 KV
tokens, and 17.4814 ms / 400.4264 tok/s/GPU (SHA `403bdaf4...`). vLLM
confirms `[8192, 11703, 15214, 18725, 22235, 25746, 29257, 32768]`,
164,400/799,920 KV tokens, and 30.0362 ms request / 18.9120 ms CUDA (SHA
`89a4abbf...`). At 19:50 PDT, their released slots submitted the 1K--16K/b8
pair: vLLM `5457124` and AFD `5457127`.

AFD 8K--32K/b8 job `5457000` completed `0:0` in 5:13 and advanced the
collector to 88 gated rows (partial CSV SHA `6d673e98...`). It confirms input
b8 -> 4+4 / graph b4, exact prompts `[8192, 11703, 15214, 18725, 22235,
25746, 29257, 32768]`, 164,544/839,424 KV tokens, and 17.8959 ms /
391.1505 tok/s/GPU (result SHA `959081f0...`). At 19:56 PDT, its released
slot submitted vLLM 4K--64K/b8 job `5457181` using the audited 64K YaRN
profile.

vLLM 1K--16K/b8 job `5457124` completed `0:0` in 4:12 and advanced the
collector to 89 gated rows (partial CSV SHA `a46f41e9...`). It confirms exact
prompts `[1024, 3218, 5413, 7607, 9801, 11995, 14190, 16384]`,
70,192/799,920 KV tokens, and 26.2762 ms request / 15.2532 ms CUDA (result SHA
`da37378a...`). At 19:59 PDT, its released slot submitted matching AFD
4K--64K/b8 job `5457211` using the audited 64K YaRN profile.

AFD 1K--16K/b8 job `5457127` completed `0:0` in 6:15 and advanced the
collector to 90 gated rows (partial CSV SHA `25646d8b...`). It confirms input
b8 -> 4+4 / graph b4, exact prompts `[1024, 3218, 5413, 7607, 9801, 11995,
14190, 16384]`, 70,336/839,424 KV tokens, and 17.7547 ms / 394.2622
tok/s/GPU (result SHA `0abde817...`). At 20:02 PDT, its released slot
submitted vLLM 8K--128K/b8 job `5457230`, the baseline capacity ceiling for
that range.

vLLM 4K--64K/b8 job `5457181` completed `0:0` in 6:10 and advanced the
collector to 91 gated rows (partial CSV SHA `61f23905...`). It confirms exact
prompts `[4096, 12873, 21650, 30427, 39205, 47982, 56759, 65536]`,
279,088/799,888 KV tokens, and 34.4926 ms request / 23.5019 ms CUDA (result
SHA `cfb3f55a...`). At 20:05 PDT, its released slot submitted matching AFD
8K--128K/b8 job `5457256` with graph replay enabled and
`NSYS_CUDA_GRAPH_TRACE=none`.

vLLM 8K--128K/b8 job `5457230` completed `0:0` in 6:21 and advanced the
collector to 92 gated rows (partial CSV SHA `e891576c...`). It confirms exact
prompts `[8192, 25746, 43301, 60855, 78409, 95963, 113518, 131072]`,
557,616/799,792 KV tokens, and 42.5857 ms request / 31.4660 ms CUDA (result
SHA `1e62548b...`). At 20:11 PDT, its released slot submitted vLLM
1K--128K/b8 job `5457309`.

AFD 8K--128K/b6 job `5456585` remained actively exporting qdstrm through
20:08 PDT, but the OCI occupied-idle reaper cancelled it at 45:14 with
`17/32 GPUs idle for 30m, 53% waste`. Thus `NSYS_CUDA_GRAPH_TRACE=none`
removed graph-node expansion but was insufficient for this larger b6 trace;
do not blindly resubmit the same configuration. The attempt has no gated row
and remains missing pending a clean export-path fix.

AFD 4K--64K/b8 job `5457211` completed `0:0` in 10:04 and advanced the
collector to 93 gated rows (partial CSV SHA `b5910760...`). It confirms input
b8 -> 4+4 / graph b4, exact prompts `[4096, 12873, 21650, 30427, 39205,
47982, 56759, 65536]`, 279,232/839,424 KV tokens, and 19.9078 ms /
351.6217 tok/s/GPU (result SHA `b8863ddd...`).

At 20:15 PDT, cancelled job `5456585` was still `COMPLETING`, so it continued
to count against the goal-scoped cap. The one available slot submitted vLLM
4K--8K/b12 job `5457347`. Another 128K AFD point was intentionally not
submitted while AFD 8K--128K/b8 `5457256` is already testing whether the
same export/reaper problem repeats.

Cancelled job `5456585` left `squeue` at 20:15 PDT, and its released slot
submitted matching AFD 4K--8K/b12 job `5457355`, restoring four active
goal-scoped jobs without risking a second unvalidated 128K AFD export.

vLLM 1K--128K/b8 job `5457309` completed `0:0` in 7:03 and advanced the
collector to 94 gated rows (partial CSV SHA `86464378...`). It confirms exact
prompts `[1024, 19602, 38181, 56759, 75337, 93915, 112494, 131072]`,
528,944/799,792 KV tokens, and 41.6965 ms request / 30.7296 ms CUDA (result
SHA `5930ed6b...`).

The first AFD 8K--128K/b8 attempt `5457256` failed `1:0` in 13:26 on head
`nvl72026-T03`. The workload completed all 224x16 outputs, every attention
worker proved the identical 30 graph replays, all 28 CPU traces and two
coordinator `.nsys-rep` files existed, but validation reported
`alignment_json_missing` / zero schedule steps. This exactly matches the six
documented head-node artifact-finalization failures that succeeded unchanged
on retry elsewhere, so `nvl72026-T03` became the seventh infrastructure-only
head exclusion. At 20:25 PDT, the two open slots submitted identical AFD
8K--128K/b8 retry `5457434` and remaining AFD 1K--128K/b8 `5457435`, both
with the seven exclusions and `NSYS_CUDA_GRAPH_TRACE=none`; workload inputs,
topology, graph replay, and pinned source are unchanged.

Jobs `5457347`, `5457355`, `5457434`, and `5457435` remained continuously
pending with scheduler reason `Priority` from 20:26 through at least 21:24
PDT. This is sustained external cluster pressure, not a launcher or placement
validation error; retain the four exact jobs and the five-minute pending-only
monitor cadence.

Scheduler capacity opened at 22:11 PDT: all four pending jobs started, with
the two exclusion-only AFD long-range reruns placed on non-excluded heads
`nvl72037-T01` and `nvl72105-T01`. vLLM 4K--8K/b12 job `5457347` completed
`0:0` in 5:22 and advanced the collector to 95 gated rows (partial CSV SHA
`2865cd85...`). It confirms exact prompts `[4096, 4468, 4841, 5213, 5585,
5958, 6330, 6703, 7075, 7447, 7820, 8192]`, 74,576/799,920 KV tokens, and
26.4301 ms request / 16.3406 ms CUDA (result SHA `3fcfaae6...`). At 22:14
PDT, its released slot submitted vLLM 8K--32K/b12 job `5458734`.

AFD 4K--8K/b12 job `5457355` and vLLM 8K--32K/b12 job `5458734` completed
`0:0` in 9:17 and 6:04, advancing the collector to 97 gated rows (partial CSV
SHA `30ac85f3...`). AFD confirms input b12 -> 6+6 / graph b6, exact prompts
`[4096, 4468, 4841, 5213, 5585, 5958, 6330, 6703, 7075, 7447, 7820,
8192]`, 74,816/839,424 KV tokens, and 17.4442 ms / 601.9208 tok/s/GPU (SHA
`eacca98f...`). vLLM confirms `[8192, 10426, 12660, 14895, 17129, 19363,
21597, 23831, 26065, 28300, 30534, 32768]`, 246,608/799,920 KV tokens, and
32.4308 ms request / 21.4147 ms CUDA (SHA `01bee708...`). At 22:22 PDT, their
released slots submitted AFD 8K--32K/b12 job `5458822` with all seven head
exclusions and vLLM 1K--16K/b12 job `5458825`.

The head-excluded AFD 8K--128K/b8 retry `5457434` and AFD 1K--128K/b8 job
`5457435` both completed `0:0` in 12:49, advancing the collector to 99 gated
rows (partial CSV SHA `130ddbf0...`). This confirms the first b8 failure was
placement-specific without changing workload or source. The 8K--128K point
confirms input b8 -> 4+4 / graph b4, exact prompts `[8192, 25746, 43301,
60855, 78409, 95963, 113518, 131072]`, 557,760/839,424 KV tokens, and
22.6514 ms / 309.0319 tok/s/GPU (SHA `afdada76...`). The 1K--128K point
confirms `[1024, 19602, 38181, 56759, 75337, 93915, 112494, 131072]`,
529,088/839,424 KV tokens, and 22.3492 ms / 313.2105 tok/s/GPU (SHA
`f4aa3778...`). At 22:26 PDT, their released slots submitted AFD 1K--16K/b12
job `5458843` with seven head exclusions and vLLM 4K--64K/b12 job `5458845`.

vLLM 1K--16K/b12 job `5458825` completed `0:0` in 5:35 and advanced the
collector to 100 gated rows (partial CSV SHA `c20a9318...`). It confirms exact
prompts `[1024, 2420, 3817, 5213, 6609, 8006, 9402, 10799, 12195, 13591,
14988, 16384]`, 105,296/799,920 KV tokens, and 28.0700 ms request / 16.9728 ms
CUDA (result SHA `d692cb5a...`). After teardown cleared at 22:32 PDT, its
released slot submitted AFD 4K--64K/b12 job `5458887` with all seven head
exclusions and the audited 64K YaRN profile.

AFD 8K--32K/b12 job `5458822` and vLLM 4K--64K/b12 job `5458845` completed
`0:0` in 9:52 and 6:26, advancing the collector to 102 gated rows (partial CSV
SHA `f8997bcd...`). AFD confirms input b12 -> 6+6 / graph b6, exact prompts
`[8192, 10426, 12660, 14895, 17129, 19363, 21597, 23831, 26065, 28300,
30534, 32768]`, 246,848/839,424 KV tokens, and 18.5128 ms / 567.1744
tok/s/GPU (SHA `e5373b00...`). vLLM confirms `[4096, 9681, 15267, 20852,
26438, 32023, 37609, 43194, 48780, 54365, 59951, 65536]`,
418,640/799,888 KV tokens, and 38.2975 ms request / 26.6753 ms CUDA (SHA
`24639db9...`). At 22:36--22:37 PDT, their released slots submitted AFD
8K--128K/b12 job `5458955` with seven head exclusions and graph-node tracing
disabled, plus vLLM 1K--128K/b12 job `5458970`.

AFD 1K--16K/b12 job `5458843` completed `0:0` in 8:56 and advanced the
collector to 103 gated rows (partial CSV SHA `636279fa...`). It confirms input
b12 -> 6+6 / graph b6, exact prompts `[1024, 2420, 3817, 5213, 6609, 8006,
9402, 10799, 12195, 13591, 14988, 16384]`, 105,536/839,424 KV tokens, and
17.6329 ms / 595.4793 tok/s/GPU (result SHA `17b9a3ad...`). At 22:38 PDT,
its released slot submitted AFD 1K--128K/b12 job `5458992` with all seven head
exclusions and graph-node tracing disabled.

AFD 4K--64K/b12 job `5458887` completed `0:0` in 11:01 and advanced the
collector to 104 gated rows (partial CSV SHA `b2136a75...`). It confirms input
b12 -> 6+6 / graph b6, exact prompts `[4096, 9681, 15267, 20852, 26438,
32023, 37609, 43194, 48780, 54365, 59951, 65536]`, 418,880/839,424 KV
tokens, and 20.6159 ms / 509.3159 tok/s/GPU (result SHA `5a1b14de...`). At
22:48 PDT, its released slot began the b16 breadth pass with vLLM 4K--8K/b16
job `5459081`.

vLLM 4K--8K/b16 job `5459081` completed `0:0` in 5:56 and advanced the
collector to 105 gated rows (partial CSV SHA `640d9af0...`). It confirms exact
prompts `[4096, 4369, 4642, 4915, 5188, 5461, 5734, 6007, 6281, 6554, 6827,
7100, 7373, 7646, 7919, 8192]`, 99,440/799,920 KV tokens, and 27.3174 ms
request / 16.4463 ms CUDA (result SHA `07cb99e5...`). At 00:16 PDT on July
18, its released slot submitted matching AFD 4K--8K/b16 job `5459629` with
all seven head exclusions.

vLLM 1K--128K/b12 job `5458970` completed `0:0` in 9:34 and advanced the
collector to 106 gated rows (partial CSV SHA `8a20b36e...`). It confirms exact
prompts `[1024, 12847, 24669, 36492, 48314, 60137, 71959, 83782, 95604,
107427, 119249, 131072]`, 793,424/799,792 KV tokens, and 49.5958 ms request /
38.4292 ms CUDA (result SHA `419c5c82...`). At 00:19 PDT, its released slot
submitted vLLM 8K--32K/b16 job `5459635`, keeping the goal-scoped population
at four independently of the other session's jobs.

AFD 8K--128K/b12 job `5458955` and AFD 1K--128K/b12 job `5458992` completed
`0:0` in 15:06 and 14:24, advancing the collector to 108 gated rows (partial
CSV SHA `7c8cda6d...`). Both confirm input b12 -> 6+6 / graph b6 and identical
per-rank length grids to their vLLM counterparts. The 8K--128K point uses
836,672/839,424 KV tokens and reaches 25.7671 ms / 407.4960 tok/s/GPU (SHA
`c14536a1...`); the 1K--128K point uses 793,664/839,424 KV tokens and reaches
25.5162 ms / 411.5029 tok/s/GPU (SHA `a12e50d2...`). At 00:24 PDT, their
released slots submitted AFD 8K--32K/b16 job `5459681` with all seven head
exclusions and vLLM 1K--16K/b16 job `5459682`.

AFD 4K--8K/b16 job `5459629` completed `0:0` in 9:14. Its gated result
confirms input b16 -> 8+8 / graph b8, the exact 16-length grid shared with the
baseline, 99,776/839,424 KV tokens, and 16.5984 ms / 843.4563 tok/s/GPU (SHA
`15aabcc4...`). The same collector pass also found the complete gated result
for still-tearing-down vLLM 8K--32K/b16 job `5459635`: 328,800/799,920 KV
tokens, 35.3949 ms request / 25.3642 ms CUDA (SHA `badbb244...`). This moved
the partial report to 110 rows (SHA `eb7b53a9...`), but the vLLM job continues
to count against the four-job cap until it leaves `squeue`. At 00:29 PDT, the
one actually released slot submitted AFD 1K--16K/b16 job `5459752` with all
seven head exclusions.

vLLM 8K--32K/b16 job `5459635` left `squeue` as `COMPLETED 0:0` after 6:39.
At 00:30 PDT, its now-confirmed free slot submitted the b16 baseline ceiling,
vLLM 4K--64K job `5459757`, using the audited 64K YaRN profile.

AFD 8K--32K/b16 job `5459681` completed `0:0` in 7:52 and vLLM
1K--16K/b16 job `5459682` completed `0:0` in 6:19, advancing the collector to
112 gated rows (partial CSV SHA `8066d836...`). AFD confirms input b16 -> 8+8
/ graph b8, the exact shared grid, 329,088/839,424 KV tokens, and 20.3736 ms /
687.1642 tok/s/GPU (SHA `7fa231f6...`). vLLM confirms the exact 1K-spaced
grid, 140,288/799,920 KV tokens, and 30.1741 ms request / 18.6740 ms CUDA (SHA
`e928000c...`). While AFD remained in scheduler teardown, only the baseline
slot was reused; at 00:35 PDT it submitted final b16 point AFD 4K--64K job
`5459777` with all seven exclusions and the audited 64K YaRN profile.

AFD 8K--32K/b16 job `5459681` cleared scheduler teardown at 00:36 PDT. Its
released goal slot began the b24 breadth pass with vLLM 4K--8K/b24 job
`5459789`.

AFD 1K--16K/b16 job `5459752` and vLLM 4K--64K/b16 job `5459757` completed
`0:0` in 9:10 and 6:52, completing the b16 pass and advancing the collector
to 114 gated rows (partial CSV SHA `1e008d67...`). AFD confirms input b16 ->
8+8 / graph b8, the exact shared 1K-spaced grid, 140,288/839,424 KV tokens,
and 19.7020 ms / 710.5895 tok/s/GPU (SHA `8882f57f...`). vLLM confirms the
exact 4K-spaced grid, 558,080/799,888 KV tokens, and 42.7347 ms request /
31.8231 ms CUDA (SHA `f687047f...`). At 00:41 PDT, their released slots
submitted AFD 4K--8K/b24 job `5459851` with all seven exclusions and vLLM
8K--32K/b24 job `5459853`.

AFD 4K--64K/b16 job `5459777` completed `0:0` in 8:14 and vLLM
4K--8K/b24 job `5459789` completed `0:0` in 5:58, advancing the collector to
116 gated rows (partial CSV SHA `f8eec603...`). AFD confirms input b16 -> 8+8
/ graph b8, exact 4K spacing, 558,080/839,424 KV tokens, and 23.7671 ms /
589.0497 tok/s/GPU (SHA `26cafdb2...`). vLLM b24 confirms the exact symmetric
integer grid, 149,168/799,920 KV tokens, and 30.1559 ms request / 19.2416 ms
CUDA (SHA `49257911...`). At 00:46 PDT, their released slots submitted AFD
8K--32K/b24 job `5459907` with all seven exclusions and vLLM 1K--16K/b24 job
`5459908`.

vLLM 8K--32K/b24 job `5459853` completed `0:0` in 7:00 and advanced the
collector to 117 gated rows (partial CSV SHA `cc7e3d42...`). It confirms the
exact 24-length symmetric grid, 493,232/799,920 KV tokens, and 42.6538 ms
request / 31.1310 ms CUDA (SHA `e764f915...`). At 00:51 PDT, its released
slot submitted AFD 1K--16K/b24 job `5459932` with all seven head exclusions.

AFD 4K--8K/b24 job `5459851` completed `0:0` in 9:35 and vLLM
1K--16K/b24 job `5459908` completed `0:0` in 4:52, advancing the collector to
119 gated rows (partial CSV SHA `c252f566...`). AFD confirms input b24 ->
12+12 / graph b12, the exact shared grid, 149,696/839,424 KV tokens, and
17.1082 ms / 1,227.4819 tok/s/GPU (SHA `0cf0b052...`). vLLM confirms its
exact 24-length grid, 210,608/799,920 KV tokens, and 33.2876 ms request /
22.4679 ms CUDA (SHA `83c720af...`). At 00:54 PDT, their released slots
submitted final b24 point AFD 4K--64K job `5459952` with all seven exclusions
and the audited 64K YaRN profile, plus first b32 point vLLM 4K--8K job
`5459953`.

AFD 8K--32K/b24 job `5459907` completed `0:0` in 11:53 and advanced the
collector to 120 gated rows (partial CSV SHA `8feac15b...`). It confirms input
b24 -> 12+12 / graph b12, the exact shared 24-length grid,
493,760/839,424 KV tokens, and 20.1111 ms / 1,044.2000 tok/s/GPU (SHA
`41f736f3...`). The job was still in scheduler teardown at 01:01 PDT, so its
slot was not yet reused.

AFD 1K--16K/b24 job `5459932` completed `0:0` in 9:40 and vLLM 4K--8K/b32
job `5459953` completed `0:0` in 6:03, advancing the collector to 122 gated
rows (partial CSV SHA `030dabdb...`). AFD confirms input b24 -> 12+12 / graph
b12, the exact shared grid, 211,136/839,424 KV tokens, and 18.5111 ms /
1,134.4569 tok/s/GPU (SHA `479c6205...`). vLLM confirms its exact 32-length
grid, 198,896/799,920 KV tokens, and 32.0553 ms request / 20.5625 ms CUDA (SHA
`b27ea8ba...`). At 01:04 PDT, the two slots that had actually cleared
submitted AFD 4K--8K/b32 job `5460107` with all seven exclusions and vLLM
8K--32K/b32 job `5460108`; AFD 1K--16K/b24 remained in teardown and continued
to count against the cap.

AFD 1K--16K/b24 job `5459932` cleared scheduler teardown at 01:07 PDT. Its
released slot submitted matching AFD 8K--32K/b32 job `5460144` with all seven
head exclusions.

Final b24 point AFD 4K--64K job `5459952` completed `0:0` in 11:00 and
advanced the collector to 123 gated rows (partial CSV SHA `da3cdc9f...`). It
confirms input b24 -> 12+12 / graph b12, the exact 24-length grid,
837,824/839,424 KV tokens, and 24.4549 ms / 858.7242 tok/s/GPU (SHA
`7cf30f37...`). This closes the planned b24 pass. At 01:10 PDT, its released
slot submitted vLLM 1K--16K/b32 job `5460165`.

AFD 4K--8K/b32 job `5460107` completed `0:0` in 6:05 and vLLM
8K--32K/b32 job `5460108` completed `0:0` in 7:40, advancing the collector to
125 gated rows (partial CSV SHA `34eb8cf5...`). AFD confirms input b32 ->
16+16 / graph b16, the exact shared 32-length grid, 199,616/839,424 KV tokens,
and 18.7383 ms / 1,494.2693 tok/s/GPU (SHA `9c9acd49...`). vLLM confirms its
exact grid, 657,648/799,920 KV tokens, and 46.5032 ms request / 35.8222 ms CUDA
(SHA `47eec115...`). At 01:15 PDT only the cleared AFD slot was reused, for
AFD 1K--16K/b32 job `5460214` with all seven exclusions; vLLM remained in
scheduler teardown and continued to count against the cap.

vLLM 8K--32K/b32 job `5460108` cleared scheduler teardown at 01:15 PDT. Its
released slot began the b48 pass with vLLM 4K--8K/b48 job `5460221`.

vLLM 1K--16K/b32 job `5460165` completed `0:0` in 4:41. Its collector pass
also found the complete gated result for still-running/exporting AFD
8K--32K/b32 job `5460144`, advancing the report to 127 rows (partial CSV SHA
`e151b1c0...`). vLLM confirms the exact 32-length grid, 280,816/799,920 KV
tokens, and 34.8751 ms request / 23.9042 ms CUDA (SHA `8fe9899e...`). AFD
confirms input b32 -> 16+16 / graph b16, the exact shared grid,
658,368/839,424 KV tokens, and 22.7740 ms / 1,229.4712 tok/s/GPU (SHA
`fd543386...`), but its slot remains reserved until scheduler exit. At 01:18
PDT, only the cleared vLLM slot submitted AFD 4K--8K/b48 job `5460253` with
all seven exclusions.

AFD 8K--32K/b32 job `5460144` left the scheduler as `COMPLETED 0:0` after
9:28. At 01:19 PDT, its released slot submitted vLLM 1K--16K/b48 job
`5460265`.

AFD 1K--16K/b32 job `5460214` completed `0:0` in 6:25, closing the b32 pass,
and vLLM 4K--8K/b48 job `5460221` completed `0:0` in 5:24. The collector
advanced to 129 gated rows (partial CSV SHA `0db6bfad...`). AFD confirms input
b32 -> 16+16 / graph b16, the exact shared grid, 281,536/839,424 KV tokens,
and 19.9124 ms / 1,406.1601 tok/s/GPU (SHA `bb227d54...`). vLLM confirms its
exact 48-length grid, 298,336/799,920 KV tokens, and 36.9860 ms request /
25.4284 ms CUDA (SHA `85d521c4...`). At 01:24 PDT, their released slots
submitted final b48 point AFD 1K--16K job `5460320` with all seven exclusions
and first b64 point vLLM 4K--8K job `5460321`.

AFD 4K--8K/b48 job `5460253` completed `0:0` in 6:48 and advanced the
collector to 130 gated rows (partial CSV SHA `189d9a3d...`). It confirms input
b48 -> 24+24 / graph b24, the exact shared 48-length grid,
299,456/839,424 KV tokens, and 21.6007 ms / 1,944.3809 tok/s/GPU (SHA
`5e931067...`). At 01:27 PDT, its released slot submitted matching AFD
4K--8K/b64 job `5460339` with all seven head exclusions.

vLLM 1K--16K/b48 job `5460265` completed `0:0` in 6:26 and advanced the
collector to 131 gated rows (partial CSV SHA `bdb08ade...`). It confirms the
exact 48-length grid, 421,216/799,920 KV tokens, and 42.3791 ms request /
71.2645 ms CUDA (SHA `cf1e8ed1...`); the CUDA aggregate is higher than the
request median but passed the existing per-device trace-consistency gates and
is retained as measured. At 01:30 PDT, its released slot submitted vLLM
1K--16K/b64 job `5460351`.

vLLM 4K--8K/b64 job `5460321` completed `0:0` in 5:16 and advanced the
collector to 132 gated rows (partial CSV SHA `b2866b5f...`). It confirms the
exact 64-length grid, 397,792/799,920 KV tokens, and 40.2221 ms request /
29.1189 ms CUDA (SHA `ae93973d...`). At 01:33 PDT, its released slot
submitted final b64 point AFD 1K--16K job `5460381` with all seven head
exclusions.

AFD 4K--8K/b64 job `5460339` completed `0:0` in 7:37 and advanced the
collector to 133 gated rows (partial CSV SHA `5be6540b...`). It confirms input
b64 -> 32+32 / graph b32, the exact shared 64-length grid,
399,296/839,424 KV tokens, and 24.0809 ms / 2,325.4959 tok/s/GPU (SHA
`b6b78d00...`). The job remained in scheduler teardown at 01:35 PDT, so its
slot was not yet reused.

AFD 4K--8K/b64 job `5460339` cleared scheduler teardown at 01:36 PDT. Its
released slot began the b96 pass with vLLM 4K--8K/b96 job `5460404`.

AFD 1K--16K/b48 job `5460320` completed `0:0` in 11:16 and vLLM
1K--16K/b64 job `5460351` completed `0:0` in 7:09, advancing the collector to
135 gated rows (partial CSV SHA `204250b5...`). AFD confirms input b48 ->
24+24 / graph b24, the exact shared grid, 422,336/839,424 KV tokens, and
24.5359 ms / 1,711.7751 tok/s/GPU (SHA `df80bed9...`). vLLM confirms its exact
64-length grid, 561,632/799,920 KV tokens, and 45.2104 ms request / 34.3431 ms
CUDA (SHA `0a8eecce...`). At 01:39 PDT, only the cleared AFD slot submitted
AFD 4K--8K/b96 job `5460426` with all seven exclusions; vLLM remained in
scheduler teardown and continued to count against the cap.

vLLM 1K--16K/b64 job `5460351` cleared scheduler teardown at 01:40 PDT. Its
released slot began the b128 pass with vLLM 4K--8K/b128 job `5460429`.

vLLM 4K--8K/b96 job `5460404` completed `0:0` in 6:33. Its collector pass
also found the complete gated result for still-running/exporting final b64
point AFD 1K--16K job `5460381`, advancing the report to 137 rows (partial CSV
SHA `95425425...`). vLLM confirms the exact 96-length grid,
596,688/799,920 KV tokens, and 49.5861 ms request / 37.3975 ms CUDA (SHA
`62bcc678...`). AFD confirms input b64 -> 32+32 / graph b32, the exact shared
grid, 563,072/839,424 KV tokens, and 26.1813 ms / 2,138.9288 tok/s/GPU (SHA
`c012e7d9...`), but its slot remains reserved pending scheduler exit. At
01:47 PDT, only the cleared vLLM slot submitted AFD 4K--8K/b128 job `5460470`
with all seven exclusions.

AFD 1K--16K/b64 job `5460381`, AFD 4K--8K/b96 job `5460426`, and vLLM
4K--8K/b128 job `5460429` all cleared cleanly by 01:50 PDT; the first was
already gated, while the latter two advanced the collector to 139 rows
(partial CSV SHA `4e5d5abc...`). The exact collector-derived missing set was
then AFD 1K--4K/b192, AFD 1K--4K/b256, active AFD 4K--8K/b128, and the two
known deferred long-range b6 points (8K--128K and 32K--128K). At 01:51 PDT,
the safe missing 1K--4K points submitted as jobs `5460490` (b192) and
`5460491` (b256), both with all seven head exclusions. The fourth goal slot
was intentionally left unused rather than blindly repeat a known failing
long-range b6 configuration.

AFD 4K--8K/b128 job `5460470` completed `0:0` in 10:32. Its collector pass
also found the complete gated result for still-running/exporting AFD
1K--4K/b192 job `5460490`, advancing the report to 141 rows (partial CSV SHA
`404415c5...`). The b128 point confirms input b128 -> 64+64 / graph b64,
798,656/839,424 KV tokens, and 36.7906 ms / 3,044.2551 tok/s/GPU (SHA
`f64358aa...`). The b192 point confirms input b192 -> 96+96 / graph b96,
509,824/839,424 KV tokens, and 48.4239 ms / 3,469.3602 tok/s/GPU (SHA
`b278ef2b...`); its slot remains reserved pending scheduler exit.

AFD 1K--4K/b256 job `5460491` completed the workload and produced all
attention CPU traces, but CUDA graph-node trace export remained active through
02:38:26 PDT and the occupied-idle controller cancelled the job at 44:41 before
a gated result could be written. This is the same export-growth class already
mitigated on long-context AFD runs. After scheduler teardown clears, retry this
exact workload with only `NSYS_CUDA_GRAPH_TRACE=none`; CUDA graph replay stays
enabled and required, the pinned source and prompts remain unchanged, and no
fallback execution path is introduced.

Cancelled b256 job `5460491` cleared scheduler teardown at 02:41 PDT. The
workload-identical retry submitted as job `5460841` with all seven head
exclusions and only `NSYS_CUDA_GRAPH_TRACE=none`; it retains the pinned source,
exact prompt grid, two microbatches, and mandatory CUDA graph replay.

The b256 retry `5460841` completed `0:0` in 14:44, proving the profiler-only
mitigation removed the 30-minute export tail without changing the workload.
It advanced the collector to 142/144 gated rows (partial CSV SHA
`bccdd568...`) and confirms input b256 -> 128+128 / graph b128, the exact
256-length grid, 679,808/839,424 KV tokens, and 62.4171 ms /
3,588.7588 tok/s/GPU (SHA `151c2a34...`). The only missing keys are now AFD
8K--128K/b6 and AFD 32K--128K/b6, both already known to require the
coordinator trace-flush-before-ZeroMQ-teardown source fix rather than another
identical retry.

To finish the two deferred b6 rows without contaminating the original pinned
checkout, an isolated source revision `c58f30407098dda85e343daa6489cec637e66b37`
was created directly on parent `3c7161949310b6d59d6b4cf9bf997a4935c8113b`.
Its sole changed file is `python/minisgl/afd_coordinator.py`: shutdown now
atomically flushes the complete coordinator CPU trace after stopping the CUDA
profiler but before any ZeroMQ/worker teardown, then flushes again on the clean
final path. The original `FastAFD-3c716194` checkout remains clean. The AFD
launcher also gained fail-fast `NSYS_CAPTURE_RADIUS_STEPS` provenance; radius
zero captures the single boundary step while leaving all scheduling, prompts,
two-way microbatching, and mandatory CUDA graph replay unchanged. Local/remote
shell and Python compilation, a mocked execution of the real shutdown method,
Git diff checks, and both final-shape dry-runs passed. The clean remote source
is `/home/shengjiel/scratch/fastafd_reproduce/source/FastAFD-c58f3040`; reuse
the original `minisgl-3c716194-cuda130-vllm-ep` environment because only Python
shutdown ordering changed.

At 03:15 PDT, the two final rows submitted against clean revision `c58f3040`:
AFD 8K--128K/b6 job `5461108` and AFD 32K--128K/b6 job `5461109`. Both use
the original CUDA 13 venv, audited 128K YaRN profile, exact six-length grids,
3+3 real microbatches with graph bucket b3, all seven bad-head exclusions,
`NSYS_CUDA_GRAPH_TRACE=none`, and `NSYS_CAPTURE_RADIUS_STEPS=0`. The immediate
scheduler state was pending-only for `Priority`, so monitoring moved to the
requested five-minute cadence; these are the only two goal-scoped jobs.

Both final jobs started on nonexcluded heads and completed `0:0`: `5461108` in
10:35 and `5461109` in 11:39. The early atomic trace flush produced complete
15,523,116-byte and 18,776,852-byte coordinator CPU traces before scheduler
exit, directly resolving the prior missing-artifact failure. All 28 attention
workers in each job logged `target_replay=30 all_graph_done=1`; the result JSONs
record graph batch b3 and single-step Nsight capture radius zero. AFD
8K--128K/b6 used exact lengths
`8192,32768,57344,81920,106496,131072`, required 418,176/839,424 KV tokens,
and measured 21.2044 ms / 247.5896 tok/s/GPU (result SHA
`18730527570ae3e05e4efc68f622f92be675c50e7a2614e847c25f14648673f2`).
AFD 32K--128K/b6 used exact lengths
`32768,52429,72090,91750,111411,131072`, required 492,032/839,424 KV tokens,
and measured 21.1320 ms / 248.4385 tok/s/GPU (result SHA
`e3e2e4d133bf709978606e11682e4e2daa84090ba4e109e307164091e6b1d15d`).

The strict collector now passes with all 144 expected rows (73 AFD + 71 vLLM)
and wrote `scratch/qwen3_irregular_final.csv`, SHA-256
`dfe431793dc70fe8469c263507602f3f1eda37a4f010fa53253caa37bc015b4d`.
Coverage by range is AFD/vLLM: 1K--4K 15/15, 4K--8K 13/13, 8K--32K 9/9,
32K--128K 5/5, 1K--16K 11/11, 4K--64K 8/7, 8K--128K 6/5, and 1K--128K
6/6; the differing last feasible batches are the preflighted KV-capacity
limits. Every audited row has a full set of distinct prompt lengths and exact
per-rank/per-lane manifests, ruling out a silently uniform single-ISL input.
As an independent performance sanity check, prior uniform vLLM 32K versus 128K
CUDA times at b2/b3/b4/b6 were 15.1799/15.6464/16.6005/18.5019 ms versus
19.6062/23.2063/27.3891/34.9984 ms. The matching mixed 32K--128K rows measured
17.6363/20.0799/23.0269/27.7170 ms, 55.5--59.6% through each endpoint span and
consistent with their exact 81,920-token mean ISL. No goal-scoped jobs remain
active, and both the original `3c716194` source checkout and isolated
`c58f3040` fix checkout are clean.
