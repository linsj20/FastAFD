# MegaMoE throughput optimization ideas

Source: `/Users/shengjiel/Workspace/github/cutedsl_megamoe`, commit `1667b47`.
Review date: 2026-09-23. Source inspection only; no GPU experiments performed.

The strongest direction is to preserve immediate, deduplicated pushes while making compute readiness and weight reuse scale with expert load. Increasing pipeline depth alone will not address the structural costs below. These are source-grounded hypotheses; no recorded large-batch measurements were found that establish which dominates.

## Scope and existing strengths

The all-push path is Rubin's `BlockScaledSwapAbGenphaseMoeKernel`. It currently accepts at most **1,024 tokens per rank**; larger problems select the general kernel. Confirm the compiled kernel name when attributing a regression. See [selection logic](../cutedsl_megamoe/tester/solvers/inference_solver.py).

The specialization already implements one activation send per token/destination rank, direct multicast FC1 gathers, dynamic MMA N, load-ordered experts, separate weight/token pipeline depths for N=256, and direct FC2 return. Preserve these benefits. The repository README describes an older pull-based prototype, not this path.

## 1. Make routing proportional to useful routes

Every source broadcasts its entire top-k plane to every rank. Each receiver classifies all those routes, then performs a shared-memory atomic even for nonlocal routes—all targeting one trash bucket. Under balanced routing, approximately `1 - 1/EP` of these updates serve no useful output. The helper waits for every sorter, so this serialization delays scheduling and weight prefetch.

**First change:** skip trash atomics; use warp-aggregated reservations or privatized histograms for hot valid experts.

**Fundamental extension:** push destination-specific route records into predetermined source-owned inbox segments, with length/completion headers. This retains one-way dispatch without a preceding count exchange while removing metadata broadcast amplification. Keep the existing broadcast protocol for tiny batches, where packing overhead may dominate.

**Evidence to seek:** shorter `sort.count` and earlier `sizes_ready`, especially at high EP or skew.

Source: [token_comm_gen_specialized.py](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py), `_run_exchange`, `_run_sort` (around lines 1105 and 1476).

## 2. Replace whole-rank and whole-expert readiness with exact tile dependencies

Pushers publish SF and activation completion together after draining both passes. Refine waits for every pusher on every rank; FC1 gathers also wait globally. Although refine publishes block completions, FC1 waits for **all SF blocks belonging to its expert**. A hot expert therefore delays its first tile until its entire SF pool is ready.

**First change:** publish readiness per 128-row SF block and wait only for blocks intersecting the current FC1 tile. Use block flags or a bitmap: an aggregate completion count cannot identify which out-of-order blocks finished.

**Fundamental extension:** publish bounded source-token chunks and release FC1 descriptors once their metadata, activation chunks, and SF blocks are ready. Schedule ready work while preserving FC1 producer progress; avoid parking all persistent consumers behind unavailable work.

Retain dense expert packing across chunks—independently padding every chunk would sacrifice GEMM efficiency. Do not simply restore an early SF-wide drain: the source records that this delayed activation transfer.

**Evidence to seek:** earlier first MMA and sustained overlap of dispatch/refine with useful FC1.

Sources: [communication](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py), `_run_pusher`, `_publish_payload_flags`, `_run_refine`; [mainloop](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py), `run_tma_b` (around line 1009).

## 3. Increase weight reuse across token tiles

The fixed `(4,1)` cluster shares activations across output features but provides no A-weight multicast across token tiles. This suits sparse experts. Once an expert spans many token tiles, repeated weight requests become a throughput concern. Current work ordering traverses output blocks within each token block.

**Proposal:** introduce a dense-expert specialization with a token-axis cluster dimension—such as `(2,2)`, subject to MMA/layout constraints—and a bounded tile ordering that reuses each weight region across adjacent token blocks. Trade some activation multicast for weight multicast. This requires changing gather ownership, SF addressing, and readiness accounting; it is not merely a cluster knob.

For a GEMM tile, weight-only arithmetic intensity is approximately `2N / bytes_per_weight`, ignoring scales. Larger effective token reuse changes that ratio; deeper buffering does not. Actual DRAM savings depend on existing L2 reuse.

Preserve `(4,1)` and dynamic N for sparse experts and tails. Consider one-time activation packing only if saved repeated gather cost exceeds packing traffic.

**Evidence to seek:** fewer measured weight bytes per useful FLOP, followed by higher tensor-core utilization—not just higher HBM utilization.

Sources: [mainloop geometry](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py), around line 197; [tile mapping](../cutedsl_megamoe/next/sources/kernel_src/schedulers/fc12_mapping.py), `_decode_inside_expert`.

## 4. Make communication's SM allocation temporary

The specialization fixes 27 pusher CTAs and documents a 184-SM compute configuration leaving room for communication. After communication exits, the fixed persistent compute grid cannot grow to use the released capacity.

**First change:** derive the pusher/compute split from the minimum communication concurrency that sustains required bandwidth, rather than retaining a decode-specific constant.

**Fundamental extension:** let completed communication workers transition into compute, or redesign completion so additional compute clusters can enter later. This requires changing the resident-grid barrier protocol; simply enlarging the current grid risks preventing progress. Programmatic dependent launch permits overlap but does not guarantee it.

**Evidence to seek:** idle capacity after refine while compute remains busy. If memory bandwidth is already saturated, extra compute workers may provide little benefit.

Sources: [kernel resource split](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py), around line 83; [CUDA dependent-launch semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## 5. Amortize transfer issue overhead without stalling compute

Input pushers use two staging slots, but each local row load is followed immediately by its wait. FC2 returns typically issue a separate bulk operation for each token's 128-feature slice—256 bytes for BF16. These can become instruction/transaction-rate limits before reaching wire bandwidth.

**Proposal:** prefetch local rows ahead of consumption and batch compatible destination transfers. If FC2 backpressure is measured, evaluate a bounded, destination-grouped output queue serviced independently of accumulator draining. Compare the saved issue/stall cost against added staging and scatter traffic.

Preserve direct epilogue return for small batches. Keep BF16 payload and reduction semantics unchanged; compression or rank-local pre-reduction introduces a separate numerical tradeoff.

**Evidence to seek:** transfer issue or epilogue stalls while achieved link bandwidth remains below its attainable rate for the traffic pattern.

Sources: [pusher loop](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py), `_push_one_plane`; [FC2 return](../cutedsl_megamoe/next/sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py), `fc2_ublk_store_function_impl`.

## Validation and latency preservation

Keep today's small-batch specialization intact and select throughput variants explicitly at measured crossovers. Use tokens per expert, active experts, destination fanout, and tiles per expert—not batch size alone.

Start with trash-atomic removal and SF-block readiness, then choose weight-reuse or transfer work from a representative profile. Compare identical routing and numerics across small batches, the crossover, and 1,024 tokens/rank, including balanced and skewed loads. Measure complete-call latency, slowest-rank behavior, and actual memory/link traffic.

The tester's [theoretical bandwidth model](../cutedsl_megamoe/tester/problem.py), `theoretical_cost`, assumes per-route dispatch and one weight read per expert. It does not represent this path's deduplication or repeated weight traffic; reported theoretical GB/s alone cannot identify the bottleneck. Keep cache conditions identical between candidates and separate warm-cache from cache-flushed results.

Use the existing [generation-phase runner](../cutedsl_megamoe/next/repo_internal_only/test_genphase_megamoe_rubin.py) and [tester](../cutedsl_megamoe/tester/tester.py). Verify repeated-launch workspace reuse and output correctness for protocol changes. Accept changes only when large-batch throughput improves and small-batch median/tail latency remains within measurement noise.
