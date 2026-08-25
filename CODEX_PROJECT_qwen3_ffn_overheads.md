# Qwen3 FMHA-Only Attention Worker and FFN Pipeline

## Active FFN latency iteration (2026-08-24 PDT)

The fixed evaluation target remains A1:F1, FEP4, batch 6, 128K ISL, ATP1,
MB2.  It is an acceptance gate, not an optimization specialization surface;
retain general kernel/runtime changes with no Qwen-shape or batch-specific
branches.  The previous aligned acceptance point was job `6493748` at
25.6410675 ms E2E and 25.6254446 ms strict CUDA mean.

The accepted rank-5 alignment trace has 16 graph launches.  Per replay it
contains 752 `sm100_blockscaled_gemm_1d1d_impl` calls with 11.551 ms summed
GPU time.  The 376 two-CTA grouped expert calls execute three cluster
rendezvous apiece: before TMEM allocation, after barrier/TMEM initialization,
and before TMEM deallocation.  Upstream DeepGEMM retains all three because the
teardown barrier fixes a real TMEM lifetime race, but uses relaxed-arrive
cluster synchronization for these rendezvous.

Accepted branch `codex/qwen3-deepgemm-relaxed-sync-20260824`, worktree
`worktrees/deepgemm_relaxed_sync`, commit `1c4b8ec` adopts that existing
relaxed-arrive helper only for the 2-CTA path.  Dense one-CTA O/QKV kernels,
the alignment standard, graph schedule, and transport are unchanged.  First
CPU job `6495610` failed before tests because the wrapper's home-directory
image default is not shared on OCI CPU nodes.  Replacement `6495630` used the
canonical Lustre image/venv and passed 60/60 in 3:06.  Four-rank GPU smoke
`6495716` is submitted to `short` for at most 15 minutes with a fresh
DeepGEMM cache, BF16 plus FP8 expert paths, nine repeated staged iterations,
and the production 128-compute/24-communication SM split.  It failed before
kernel execution because the new communication-header include preceded the
vendored definition of `DG_STATIC_ASSERT`.  The amended commit orders
`common/math.cuh` first, matching existing MegaMoE callers.  Replacement
smoke `6495802` reused the successfully built extension objects and a new
isolated DeepGEMM JIT cache.  It compiled the corrected header, repeatedly
passed staged BF16 execution, and passed FP8 once, then failed only because
the smoke's toy 256 hidden size violates the fused router's existing 512
alignment.  Exact-shape replacement `6495854` used six rows, hidden 4096,
intermediate 1536, 32 local experts, top-k 8, and nine iterations.  It passed
BF16, FP8, and fused-router checks in 2:02.

Exact target job `6495978` completed with 15/15 measured replays, zero
outliers, 25.6057515 ms coordinator E2E median, 25.5912937 ms strict CUDA
mean, and 117.227368 TPS/GPU.  Strict CUDA is 0.0341509 ms / 0.133 percent
faster than `6493748`.  The changed 2-CTA expert
GEMMs improve by 0.1296413 ms summed GPU time per replay: gate/up improves
4.2066478 -> 4.1585471 ms and down improves 2.5429399 -> 2.4613993 ms.  Both
general grouped shapes win while their call counts remain identical; unchanged
one-CTA O/QKV variation masks most of the kernel saving in the end-to-end
mean.  Official unchanged-scorer alignment job `6496358` completed in 11:04
with 24 prompts / 408 tokens and top-1, top-10, and top-100 agreement all 1.0;
average and maximum rank are both 1.0.  This makes `6495978` the current
aligned acceptance point.  Logs and reports are under
`deepgemm_relaxed_sync_validation/a1_fep4_b6_128k/`.

Candidate branch `codex/qwen3-deepgemm-overlap-init-sync-20260824`, worktree
`worktrees/deepgemm_overlap_init_sync`, commit `5827d42` starts the existing
post-initialization two-CTA cluster rendezvous, performs the independent
`cudaGridDependencySynchronize()` wait, and only then waits for the cluster.
This preserves both ordering conditions while overlapping their wait time and
contains no target-shape branch.  CPU job `6496563` passed 60/60.  Exact-shape
four-rank smoke `6496616` passed BF16, FP8, and fused-router execution for six
rows, hidden 4096, intermediate 1536, 32 local experts, top-k 8, and nine
iterations.  Exact unchanged-contract performance job `6496834` completed at
25.5289545 ms E2E median, 25.4867808 ms strict CUDA mean, and 117.708079
TPS/GPU.  Trace attribution rejects the candidate despite the headline result:
the affected grouped GEMMs improve by only 0.009235 ms/replay in aggregate,
while their per-replay standard deviations are 0.077 ms for down and 0.152 ms
for gate/up.  Unchanged dense O/QKV kernels account for nearly the entire
0.104513-ms strict difference.  Keep the simpler accepted relaxed-arrival
checkpoint and do not run official alignment for `5827d42`.

Communication-backend audit for the ATP1/TP1 target: PyTorch initializes its
process group with Gloo, and the TP PyNCCL enablement returns immediately at
TP1, so no PyTorch ProcessGroupNCCL collective is on the measured path.  QKV/O
movement uses the custom FMHA transport kernels.  EP4 uses DeepEP custom
dispatch/combine kernels; DeepEP directly links NCCL to initialize its host and
device communicator, discover topology, and establish symmetric/Gin transport.
Thus NCCL is an underlying DeepEP dependency, but not PyTorch's NCCL collective
data path.  Multi-rank TP normally uses the project's direct PyNCCL wrapper;
only the explicit non-PyNCCL TP mode initializes PyTorch's NCCL backend.

`comm_ref_table.txt` is the user-provided official pure-communication latency
reference.  The accepted rank-5 trace averages 14.844 us in `dispatch_impl`
plus 3.839 us in its copy epilogue, and 19.549 us in `combine_impl` plus
4.333 us in its reduction epilogue.  Conservative traffic ceilings are 192
KiB for FP8 dispatch and 384 KiB for BF16 combine, before excluding local
routes.  Against the table's next-higher EP4 rows, core/full-stage latency
efficiency is about 100/79 percent versus NCCL A2A for dispatch and 85/70
percent for combine.  Versus UBX it is only about 20/16 and 28/23 percent.
Treat this as a signal rather than a pure-communication attribution because
the DeepEP stages also perform routing, local copy, and reduction; lower actual
remote payload makes the conservative efficiencies upper bounds.  Combine is
the clearer general communication opportunity.

For protocol work, use an explicit practical envelope rather than treating the
pure table as directly equivalent to routed MoE.  Linear interpolation at the
conservative payload ceilings gives about 14.50 us NCCL A2A at 198 KiB and
15.77 us at 384 KiB.  Dispatch core is already within about 2.4 percent of its
NCCL reference, but full dispatch is about 28.9 percent slower.  Combine core
is about 24.0 percent slower and full combine about 51.4 percent slower.  Aim
to bring each full stage within 20 percent of the interpolated NCCL reference:
roughly <=17.4 us dispatch and <=18.9 us combine, while retaining unchanged
end-to-end correctness and performance.  UB-X remains the stretch signal, not
the acceptance floor, because its pure A2A does no MoE routing/reduction.

Official NCCL `contrib/nccl_ubx` source provides a general design direction,
not a reason to use PyTorch ProcessGroupNCCL.  UB-X uses the same NCCL device
API and symmetric-window foundation as the current DeepEP path.  Its pure A2A
auto policy uses barrier-free Lamport write/poll below 0.25 MiB and UC above;
the target's conservative FP8 dispatch ceiling is about 198 KiB while BF16
combine is about 384 KiB.  It also fuses BF16-to-MXFP8 quantization into token
dispatch.  Do not transfer pure-A2A launch settings blindly: the pure-A2A
launcher defaults to 32 SMs, but UB-X's fused MoE dispatch and combine
launchers default to 128 SMs because their line/sub-warp layouts continue to
scale with bandwidth.
The accepted trace pays one 1.576-us standalone dispatch quantization kernel
per dispatch, about 0.296 ms/replay, and a 4.333-us combine reduction epilogue,
about 0.815 ms/replay.  A future general path should therefore fuse dispatch
quantization while retaining the current 128-element packed UE8M0 layout, and
replace steady-state global barriers with generation-tagged/Lamport arrival
where correctness permits.  Current DeepEP combine is already PUSH-like:
expert owners write origin-rank buffers, then a receiver-local reduction
epilogue runs after the final barrier.  The opportunity is eliminating the
two global barriers and reducing/fusing the arrival-plus-local-reduction tail,
not merely toggling combine direction.  UB-X's scale-per-32 format is not
directly compatible with current DeepGEMM.  Its benchmark-only recommendation
`CUDA_DEVICE_MAX_CONNECTIONS=1` must not be applied globally without
same-workload overlap proof.  Current DeepEP already loops top-k selections
directly and deduplicates destination ranks, so UB-X's top-k-LUT improvement
over a full expert scan is not missing from this implementation.

The official UB-X MoE benchmark also compares NCCL-EP low-latency and
high-throughput algorithms.  For the installed NCCL 2.30.4 single-node setup,
its source explicitly omits HT from the safe default because HT dispatch can
abort; `nccl_ep_ll` is the relevant alternate backend.  NCCL-EP is not present
in the current runtime artifact, so adopting it requires a deliberate build
and integration rather than a configuration toggle.  The measured target is
an all-LSA/NVLink group; its data and barriers use symmetric LSA pointers, so
Gin QP-count sweeps do not address the active fast path.
UB-X itself is likewise not a drop-in DeepEP backend: its public fused dispatch
uses a different MXFP8 scale granularity than the packed UE8M0/128 layout
consumed by the current DeepGEMM path, and it does not expose a compatible
public dispatch-plus-combine replacement.  The user therefore chose to keep
the NCCL-backed DeepEP path rather than start a separate UB-X integration.

User approved changing the SM split and expects the memory-bound expert GEMMs
to retain efficiency with fewer SMs.  Candidate branch
`codex/qwen3-deepep-32sm-20260824`, worktree `worktrees/deepep_32sm`, commit
`cfa8660` tests 32 DeepEP / 120
DeepGEMM SMs on a 152-SM device, replacing the low-fan-in 24/128 split while
leaving the existing high-fan-in policy separate.  This matches UB-X's
pure-A2A small-message default but not its MoE launcher default, and is only an
isolated SM-budget probe rather than a protocol choice.  Supporting it requires adding one
complete 8-CTA cluster to the Python and JIT launch allowlists and updating the
overlap contract tests.  CPU contract job `6497505` and exact-shape four-rank
staged smoke `6497506` passed independently from the accepted relaxed-arrival
parent; the fused FP8 staged total was 0.380779 ms versus 0.384818 ms in the
clean accepted smoke, while BF16 was 0.404804 versus 0.395275 ms.  Exact
unchanged-contract job `6497628` completed successfully with 15/15 samples and
zero outliers, but is rejected: its 25.6273520 ms E2E median and 25.5953893 ms
strict CUDA mean are respectively 0.0216005 and 0.0040957 ms slower than the
accepted 24/128 result.  This confirms that copying the pure-A2A 32-SM default
does not improve this token-indexed MoE kernel at three tokens per microbatch.
The first rank-5 export attempt `6497899` failed because the container tried to
re-enter a transient Slurm spool path; durable-script replacement `6497922`
completed.  Same-query attribution versus accepted 24/128 shows full dispatch
worsening 18.634 -> 20.809 us/call (+11.7 percent), while full combine improves
only 23.864 -> 23.431 us/call (-1.8 percent).  Core dispatch is the regression
(14.799 -> 16.792 us); core combine improves 19.533 -> 19.135 us.  The mixed
component result and worse E2E make rejection causal, not just noise.

The next protocol candidate is branch
`codex/qwen3-deepep-nccl-lsa-barrier-20260824`, worktree
`worktrees/deepep_nccl_lsa_barrier`, commit `20da38d`.  It preserves the
accepted 24/128 SM split and all payload/routing/reduction logic, but requests
NCCL multimem support plus exactly ten persistent LSA barrier slots for the
declared tags 0 through 9.  It replaces DeepEP's per-peer system-scope
atomic arrival loop with `ncclLsaBarrierSession<ncclCoopCta>` using
acquire-release ordering and the existing timeout.  On NVLink multicast
hardware this changes each arrival from one release atomic per peer to one
multimem reduction, while NCCL owns the graph-safe epoch.  This is a general
barrier-protocol/configuration probe with no target-shape branch.  Tightened
source contract job `6497835` passed 61/61.  After the isolated 32-SM control
completed, exact-shape four-rank build/correctness smoke `6497900` compiled the
NCCL API change but failed safely: an incorrectly tightened two-slot resource
reservation timed out at combine tag 5 on all four ranks.  The source audit
then identified all ten explicit constants (`kDeviceBarrierTag=0` through
`kHybridCombineTag1=9`), restored `lsaBarrierCount=10`, and extended the
contract test to cover the whole tag table.  Amended commit `20da38d` is under
replacement source test `6497991`, which passed 61/61 in 3:39.  Exact-shape
four-rank replacement smoke `6498034` passed BF16, FP8, and fused-router paths
for nine repeated staged iterations on the accepted 24/128 split.  Fused FP8
staged time was 0.363890 ms versus 0.384818 ms in the accepted smoke (-5.4
percent), while BF16 was effectively flat at 0.396501 versus 0.395275 ms.
Exact unchanged-contract performance job `6498084` completed and passed the
E2E plus trace-attributed communication acceptance gate.  It retained
15/15 with zero outliers at 25.425188 ms coordinator median, 25.3775303 ms
strict CUDA mean, and 118.214813 strict TPS/GPU.  Versus aligned relaxed-
arrival job `6495978`, E2E improves 0.1805635 ms / 0.705 percent and strict
CUDA improves 0.2137633 ms / 0.835 percent.  Rank-5 export job `6498331`
attributes the gain directly: full dispatch improves 18.634451 -> 16.535447
us/call (-11.264 percent) and full combine improves 23.863784 -> 23.190299
us/call (-2.822 percent), saving about 0.521228 ms of summed communication
GPU time per replay.  Dispatch now clears the practical 17.4-us reference
envelope; combine remains about 4.29 us above its 18.9-us envelope.  Official
unchanged-scorer job `6498364` passed 24 prompts / 408 tokens with top-1,
top-10, and top-100 agreement all 1.0 and average/maximum rank 1.0.  This makes
commit `20da38d` the current aligned acceptance checkpoint.

Follow-up branch `codex/qwen3-deepep-nccl-lsa-thread-20260824`, worktree
`worktrees/deepep_nccl_lsa_thread`, commit `75e81d5` retained the accepted
multimem protocol but uses `ncclCoopThread` for its single-leader multicast
arrival and epoch poll.  NCCL 2.30.4's CTA session adds four CTA-wide
synchronizations across arrival, timed wait, and session destruction; the
enclosing DeepEP grid synchronization or final kernel completion already
appeared to order the other threads.  This was a general barrier-overhead probe with no
shape branch.  CPU job `6498499` passed 61/61.  Exact four-rank smoke
`6498503` passed repeated BF16, FP8, and fused-router checks; its staged times
were 0.393884/0.386610/0.374642 ms.  Exact target job `6498587` passed 15/15
with zero outliers but regressed to 25.539801 ms E2E median and 25.5255538 ms
strict CUDA mean, respectively +0.114613/+0.1480235 ms versus the CTA-group
checkpoint.  Export `6498857` makes the rejection causal: full dispatch
regresses 16.535447 -> 20.155926 us/call (+21.9 percent) and full combine
regresses 23.190299 -> 24.335339 us/call (+4.9 percent).  The CTA cooperative
session therefore provides useful quiescence/ordering around the multicast
epoch; retain `20da38d` and do not run alignment for `75e81d5`.

Phase-specific SM follow-up branch `codex/qwen3-deepep-phase-sms-20260824`,
worktree `worktrees/deepep_phase_sms`, commit `aba94c0` retains the accepted
NCCL barrier and uses 24 SMs for dispatch but 32 for combine at low fan-in;
high fan-in retains 8/8.  It reserves DeepGEMM against the maximum phase grid,
so GB200 compute uses 120 SMs.  This is a general per-operation policy motivated
by the isolated 32-SM control: dispatch regressed sharply while combine
improved.  The reusable smoke now accepts separate dispatch/combine budgets
and validates the max-phase device partition.  Initial CPU job `6498922`
caught one stale single-budget assertion; its dependent smoke `6498924` was
canceled pending with zero elapsed GPU time.  Amended source job `6498946`
passed 61/61 and exact 24/32 smoke `6498947` passed all repeated BF16, FP8,
and fused-router checks, reporting the intended 120/24/32 budgets.  Exact
target job `6498982` passed 15/15 with zero outliers but is rejected at
25.636775 ms E2E median, 25.6111874 ms strict CUDA mean, and 117.136311 strict
TPS/GPU.  Versus accepted job `6498084`, it regresses E2E by 0.211587 ms /
0.832 percent and strict CUDA by 0.2336571 ms / 0.921 percent.  Export job
`6499337` completed, and its local rank-5 trace is under
`scratch/qwen3_ffn_overheads_20260820/deepep_phase_sms_job_6498982/`.
Same-query attribution shows full dispatch at 16.615905 us/call, essentially
flat at +0.49 percent, while full combine regresses 23.190299 -> 24.028359
us/call (+3.61 percent).  Across 188 calls of each stage, the extra
communication time is about 0.172681 ms/replay and explains most of the
headline regression.  The phase-specific code was removed locally; retain the
accepted single 24-SM low-fan-in communication budget and 128-SM DeepGEMM
budget from `20da38d`.

Candidate branch `codex/qwen3-deepep-combine-no-pdl-20260824`, worktree
`worktrees/deepep_combine_no_pdl`, commit `6c020a9` retains all accepted NCCL
barriers, payload, reduction, and SM budgets but disables Programmatic
Dependent Launch for the combine reduction epilogue.  The epilogue has no
independent work before its immediate `cudaGridDependencySynchronize()`, while
early residency oversubscribes the intended 24-communication/128-expert SM
partition.  Normal same-stream ordering already provides the required
completion and memory visibility.  This is a general launch-policy change with
no target-shape branch.  Local static contract passes 16/16; full remote source
job `6499491` passed 61/61 in 1:28.  Fresh exact-shape four-rank smoke job
`6499513` passed BF16, FP8, and fused-router checks for nine iterations with
six rows, hidden 4096, intermediate 1536, 32 local experts, top-k 8, and the
accepted 128/24 SM split.  Its fused-router staged time is 0.357330 ms versus
0.363890 ms for accepted smoke `6498034` (-1.8 percent), while BF16 is flat at
0.395737 versus 0.396501 ms.  Exact unchanged-contract performance job
`6499595` completed successfully with 15/15 samples, zero outliers, 25.498309
ms coordinator median, 25.4775682 ms strict CUDA mean (25.442550-ms median),
and 117.750642 strict TPS/GPU.  It regresses accepted job `6498084` by
0.073121 ms / 0.288 percent E2E and 0.1000379 ms / 0.394 percent strict CUDA,
so it cannot advance regardless of the promising smoke.  Accepted trace
inspection also shows all 3,008 combine epilogues already start after their
producer ends, with a 0.279-us average gap and zero PDL overlap; the original
early-residency hypothesis was false for the full graph.  Rank-5 export job
`6500265` completed in 34 seconds and its SQLite is copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepep_combine_no_pdl_job_6499595/`.
Same-query stage spans make the rejection causal: dispatch regresses
16.704987 -> 17.019188 us/call and combine regresses 23.469765 -> 23.704444
us/call.  Across 188 calls each, their combined ~0.103190-ms/replay loss
matches the +0.100038-ms strict regression.  The epilogue kernel alone appears
0.350513 us faster, but `combine_impl` slows 0.349232 us and the inter-kernel
gap grows 0.235960 us; this is timing relocation, not useful work reduction.
The accepted PDL launch policy is restored locally.  No official alignment
will run.

Replacement branch `codex/qwen3-deepep-combine-direct-output-20260824`,
worktree `worktrees/deepep_combine_direct_output`, commit `1877495` retains
the accepted PDL, NCCL barriers, routing, reduction arithmetic, bias handling,
top-k weights, and token-bounded grid.  It instead writes each token warp's
disjoint BF16 result directly to the output row, eliminating the intermediate
shared-memory staging writes and TMA copy.  The epilogue consequently launches
with zero dynamic shared memory.  This is a general reduction data-path change
with no target-shape branch.  Local static contract passes 16/16; full remote
source job `6500366` passed 61/61 in 1:44.  Fresh exact-shape four-rank build
and correctness smoke job `6500421` passed the accepted six-row, 4096-hidden,
1536-intermediate, top-k-8, nine-iteration, 128/24-SM contract for BF16, FP8,
and fused-router paths.  BF16 staged time improves 0.396501 -> 0.386663 ms
(-2.5 percent) and fused-router improves 0.363890 -> 0.360085 ms (-1.0
percent) versus accepted smoke `6498034`.  Exact unchanged-contract two-tray
performance job `6500517` nevertheless regresses: coordinator E2E median is
25.563346 ms and strict CUDA mean is 25.527330533 ms, respectively
+0.138158 ms / +0.543 percent and +0.149800233 ms / +0.590 percent versus
accepted `6498084`; all 15 measured replays are valid with zero outliers.
Trace export `6500876`, excluding replay zero, makes the rejection causal.
Full dispatch span is 17.447410 us/call versus accepted 16.704987
(+0.742423 us), and full combine span is 24.349885 us/call versus accepted
23.469765 (+0.880120 us).  The direct-store combine epilogue itself slows
4.183321 -> 4.297382 us, so eliminating TMA does not eliminate useful work:
the warp-issued global stores are slower than shared staging plus TMA.  Across
188 dispatch/combine pairs, the trace adds about 0.305 ms of summed GPU stage
time per replay; graph overlap masks part of it in the strict headline.  The
candidate is rejected without official alignment, and the accepted
shared-memory/TMA implementation is restored locally with the 16/16 static
contract passing.

The next general NCCL candidate specializes only barrier memory ordering by
phase on branch `codex/qwen3-deepep-nccl-lsa-phase-order-20260824`, worktree
`worktrees/deepep_nccl_lsa_phase_order`, commit `d80900b`.  NCCL's official
device-API LSA collective pattern uses acquire at the entry readiness barrier
and release at the exit data-publication barrier; accepted `20da38d`
conservatively used acquire-release for both.  The candidate propagates
`gpu_barrier`'s existing `kFlushStores` phase bit into the LSA session,
selecting acquire when there are no stores to publish and release when the
phase has flushed its stores.  It keeps the same NCCL communicator, multimem
protocol, ten tag slots, CTA cooperative group, 24-SM payload grid,
payload/routing arithmetic, and timeouts.  This removes only a fence with no
consumer in that phase and applies to direct and hybrid dispatch/combine; the
local static contract passes 16/16.  Remote source job `6501029` passed 61/61
in 1:26.  Exact-shape four-rank smoke job `6501046` compiled the candidate and
passed BF16, FP8, and fused-router checks for nine iterations in 4:31.  Its
BF16 staged time is 0.401401 ms and fused-router time is 0.368587 ms, about
1.2--1.3 percent slower than accepted smoke `6498034`, so the smoke is not
positive but remains too short to decide the overlapped graph result.  Exact
unchanged-contract two-tray target job `6501150` completed successfully with
15/15 retained samples, 1.7248-percent dominant range, 25.112056-ms coordinator
median, 25.0827552-ms strict CUDA mean, and 119.604086 strict TPS/GPU.  Versus
accepted `6498084`, this improves coordinator E2E by 0.313132 ms / 1.232
percent and strict CUDA by 0.2947751 ms / 1.162 percent.  Rank-5 trace export
job `6501689` completed in 44 seconds and is copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepep_nccl_lsa_phase_order_job_6501150/`.
Excluding replay zero, full dispatch improves accepted 16.704987 -> 14.883132
us/call (-10.91 percent) and full combine improves 23.469765 -> 22.567851
us/call (-3.84 percent).  Their main kernels improve 13.340338 -> 11.548215
and 19.006978 -> 17.945612 us/call respectively; epilogues and gaps are
essentially flat except combine reduction rises 4.183321 -> 4.339266 us.
Across 188 stage pairs this removes about 0.512069 ms of summed communication
GPU time per replay, causally supporting the overlapped headline improvement.
Dispatch is comfortably within the practical reference envelope and combine
transport is below 18.9 us, while the separate reduction epilogue remains the
full-stage gap.  Official unchanged-scorer job `6501708` completed in 10:57
and passed 24 prompts / 408 generated tokens with top-1, top-10, and top-100
agreement all 1.0 and average/maximum rank 1.0.  Commit `d80900b` is the new
aligned acceptance checkpoint.

The next general compute candidate targets the final 2-CTA DeepGEMM TMEM
teardown.  Rank-5 trace `6501689` attributes 6.612450 ms per replay to the two
grouped expert GEMMs, and their implementation performs a full-cluster barrier
immediately before `tcgen05.dealloc.cta_group::2.sync`.  The candidate replaces
only that final cluster rendezvous with `__syncthreads()`: all local warps still
finish before deallocation, while the synchronous 2-SM deallocation remains the
peer-CTA rendezvous for matching warp 0.  Allocation barriers, NCCL transport,
GEMM tiling, scheduling, numerical work, and the one-CTA path are unchanged.
This is shape-independent and applies to every 2-CTA grouped GEMM using this
implementation.  Remote branch
`codex/qwen3-deepgemm-local-teardown-20260824`, worktree
`worktrees/deepgemm_local_teardown`, commit `e2e57ec` is based exactly on
aligned `d80900b`; source job `6502143` passed 62/62 in 55 seconds.  Fresh-cache
exact-shape four-rank smoke job `6502167` passed BF16, FP8, and fused-router
checks for nine iterations in 4:33.  Versus aligned-checkpoint smoke `6501046`,
its staged times improve 0.401401 -> 0.385479 ms for BF16 (-3.97 percent),
0.373657 -> 0.371598 ms for FP8 (-0.55 percent), and 0.368587 -> 0.361575 ms
for fused routing (-1.90 percent).  The smoke is directionally positive and
clears the exact target gate.  Unchanged-contract two-tray A1:F1 target job
`6502261` used a candidate-specific `DG_JIT_CACHE_DIR`, with fresh candidate
`.cu` and `.cubin` artifacts observed for the grouped and dense SM100 kernels.
It completed 15/15 samples with 1.7650-percent dominant range, 25.072530-ms
coordinator median, 25.0539479-ms strict CUDA mean, and 119.741608 strict
TPS/GPU.  Versus aligned `6501150`, coordinator E2E improves 0.039526 ms /
0.157 percent and strict CUDA improves 0.0288073 ms / 0.115 percent.  The first
CPU-only trace export `6502697` failed because its nested container resolved the
transient Slurm spool script; durable-path retry `6502715` completed in 30
seconds.  Excluding replay zero, gate/up improves 22.082394 -> 21.935163
us/call and down projection improves 13.090212 -> 12.955440 us/call.  Across
188 calls of each per replay, the affected grouped GEMMs remove 0.053016 ms /
0.80 percent, directly supporting the strict end-to-end gain.  The SQLite is
copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_job_6502261/`.

Wrap-up reduced the implementation to one local-completion statement and one
concise lifetime comment; final remote commit `bc9ab7f` has no executable delta
from `e2e57ec`.  Requested final exact performance job `6502925` completed
15/15 samples with 1.3950-percent dominant range, 25.0087105-ms coordinator
median, 24.9813376-ms strict CUDA mean, and 120.089646 strict TPS/GPU.  Versus
aligned `6501150`, it improves coordinator E2E by 0.1033455 ms / 0.412 percent
and strict CUDA by 0.1014176 ms / 0.404 percent; it is the best exact FEP4
result.  Official unchanged-scorer job `6503374` completed in 11:37 and passed
24 prompts / 408 generated tokens with top-1, top-10, and top-100 agreement all
1.0 and average/maximum rank 1.0.  Final commit `bc9ab7f` is the aligned
acceptance checkpoint.

## Governing FFN ping-pong model

Treat `FFN(L-1) + QKV/QK-norm/RoPE(L)` as one indivisible model-side unit, `compute(L-1)`.

- Only one compute- or memory-intensive model lane may own SM resources at a time.
- Batch B may start only after batch A finishes its full compute unit and hands off `q_done`.
- `q_done` follows FFN, O projection, normalization, routing, expert work, QKV, Q/K normalization, and RoPE.
- QKV publication follows `q_done`; it is a communication tail that may overlap peer-lane compute.
- Each lane also waits for its own attention O payload before starting.
- Double buffers prepost receiver readiness and overlap communication with compute. They do not authorize two heavy model batches to run together.
- A wait placed only before QKV is too late because the earlier FFN work would already be enqueued.
- Deferred cross-batch combine overlap is forbidden.

The mirror attention pipeline is `attention compute -> O quantization -> O publication -> peer attention compute`. Attention quantization completes before that lane releases attention ownership.

## Current stop state

- Date: 2026-08-24 PDT.
- Current aligned remote worktree: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/worktrees/deepgemm_local_teardown`.
- Current aligned branch: `codex/qwen3-deepgemm-local-teardown-20260824`, commit `bc9ab7f`.
- Local subproject progress is staged for review; no local commit was created.
- Exact 32/120 SM control job `6497628` is rejected as neutral-to-worse; trace
  export `6497922` confirms a large dispatch regression and only a small combine
  improvement.
- NCCL multimem LSA-barrier commit `20da38d` was aligned and accepted after
  source `6497991`, smoke `6498034`, full target `6498084`, trace export
  `6498331`, and official scorer `6498364` all passed.  It improved strict CUDA
  by 0.835 percent and reduced both dispatch and combine communication; it is
  now superseded by aligned phase-order checkpoint `d80900b`.
- Leader-only NCCL follow-up commit `75e81d5` is rejected after source
  `6498499`, smoke `6498503`, full target `6498587`, and trace export `6498857`:
  both communication stages and both headline latency metrics regress.
- Phase-specific 24-dispatch/32-combine commit `aba94c0` is rejected after
  source `6498946`, smoke `6498947`, full target `6498982`, and export
  `6499337`: strict CUDA regresses 0.921 percent and full combine regresses
  3.61 percent.  Its local source changes were removed.
- Combine no-PDL commit `6c020a9` passes local static, remote source `6499491`,
  and exact-shape smoke `6499513`, but exact target `6499595` regresses strict
  CUDA by 0.394 percent and is rejected.  Trace export `6500265` attributes
  the full loss to slower dispatch/combine stage spans; local PDL is restored.
- Direct-output combine commit `1877495` passes local static and remote source
  job `6500366` at 61/61 plus exact-shape smoke `6500421`; exact target job
  `6500517` regresses strict CUDA by 0.590 percent and is rejected.  Trace
  export `6500876` shows +0.880 us/call full combine span and a slower direct
  global-store epilogue.  The accepted TMA path is restored locally; no
  official alignment was run.
- Phase-order NCCL commit `d80900b` passes local static, source `6501029`, smoke
  `6501046`, and exact target `6501150`.  Exact strict CUDA improves 1.162
  percent; trace export `6501689` attributes the gain to 10.91-percent faster
  dispatch and 3.84-percent faster combine spans.  Official alignment
  `6501708` passed perfectly; it is superseded by `bc9ab7f`.
- Minimal DeepGEMM teardown commit `bc9ab7f` passes source `6502143`, exact-shape
  smoke `6502167`, causal trace export `6502715`, candidate target `6502261`,
  final target `6502925`, and official alignment `6503374`.  Final strict CUDA
  is 24.9813376 ms, 0.404 percent faster than `d80900b`; alignment is perfect.
- A7:F1/FEP8 job `6494764` was canceled while pending with zero elapsed time and no allocation.
- Broad A8/A15 campaigns remain stopped unless explicitly reopened.

Rollback lineage:

- `8b93a9c`: enforce the exclusive model compute-unit schedule.
- `5c7ecb3`: keep final normalization on the last compute-owner stream.
- `89b594a`: transport post-attention O in FP8.
- `247e12e`: import Torch at runtime for prequantized O projection.
- Later commits are documentation-only staging checkpoints.

## Current ownership and data path

Attention workers own:

- Physical KV cache, FMHA backend, workspace, and FMHA metadata.
- Post-FMHA BF16-to-FP8 activation quantization.
- Publication of FP8 O plus packed UE8M0 scales.

Model or FFN workers own:

- Embedding, all checkpoint weights, input and post-attention norms.
- O projection, MoE routing, dispatch, expert GEMMs, combine, and final norm.
- QKV projection, Q/K normalization, RoPE, LM head, sampling, and token writeback.

Decode path:

1. Model publishes already-FP8 Q/K/V to attention.
2. Attention runs FMHA.
3. Attention quantizes O to FP8 with packed UE8M0 scales.
4. Attention publishes both payloads and releases one O-ready word only after both remote writes are system-fenced.
5. Model consumes the transported packed-scale layout directly.
6. Model runs the FP8-input O projection locally, then completes the exclusive FFN plus next-QKV unit.
7. Model publishes next-layer Q/K/V after `q_done`.

Dynamic prefill alone repacks O scales around fan-in padding. Decode has no scale repack. Four-head-aligned TP slices are required and fail fast otherwise. No fallback path is present.

## Receiver-ready signaling

Both directions use receiver-owned double buffers.

- A receiver posts readiness as soon as its slot is clean.
- A sender that later becomes ready can publish immediately without a request or credit round trip.
- Readiness signaling is off the payload critical path.
- A slot is reused only after local causal cleanup.
- QKV and O publications each issue one remote readiness release after their complete payload.
- Direct fabric transport must stay inside one NVL72 base system; submissions use `--segment` and a fabric-block preflight.

Exact A1 uses compile-time single-source and single-ready publication variants. The generic fan-in path retains flattened task mapping, bounded publication grids, the two-slot epoch protocol, and fail-fast validation; no fallback was added.

## Accepted performance and correctness

All rows use Qwen3 235B-A22B FP8, 128K context, batch 6 per attention-DP lane, ATP1, MB2, exact warmup plus 15 measured decode steps, and zero outliers unless noted.

| State | FEP | Job | E2E median ms | Strict CUDA mean ms | Strict TPS/GPU | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy control | 4 | 6470598 | 30.6157295 | 30.4497429 | 97.988846 E2E | control |
| Initial FMHA-only | 4 | 6472328 | 25.9524935 | 25.9164299 | 115.595829 E2E | aligned |
| Preposted signal/data | 4 | 6474505 | 25.7321745 | 25.6797822 | 116.585561 E2E | aligned |
| Corrected exclusive FFN | 4 | 6490963 | 25.7428600 | 25.7122698 | - | user accepted |
| Current FP8-O | 4 | 6493748 | 25.6410675 | 25.6254446 | 117.071140 strict | aligned |
| Relaxed 2-CTA cluster arrival | 4 | 6495978 | 25.6057515 | 25.5912937 | 117.227368 strict | aligned |
| NCCL multimem LSA barriers | 4 | 6498084 | 25.4251880 | 25.3775303 | 118.214813 strict | aligned |
| Phase-ordered NCCL LSA barriers | 4 | 6501150 | 25.1120560 | 25.0827552 | 119.604086 strict | aligned |
| Minimal DeepGEMM teardown | 4 | 6502925 | 25.0087105 | 24.9813376 | 120.089646 strict | aligned final |
| Earlier FMHA-only | 8 | 6473036 | 25.3913920 | 25.3729383 | 118.236207 strict | historical |
| Current FP8-O | 8 | 6494759 | 25.2301460 | 25.1990735 | 119.051996 strict | profiled |

Final FEP4 improvement:

- Versus phase-ordered NCCL job `6501150`: coordinator E2E improves 0.412
  percent and strict CUDA improves 0.404 percent.
- Versus corrected exclusive-FFN job `6490963`: coordinator E2E improves 2.852
  percent and strict CUDA improves 2.843 percent.
- Job `6502925` retained 15 of 15 samples with 1.395-percent dominant range.

Current FEP8 evidence:

- Job `6494759` ran unchanged source in `fmha-only` mode on four trays and 16 active GPUs.
- It completed in 13:46 with exit 0, 15 of 15 samples, zero outliers, and all 16 reports.
- Versus earlier same-shape job `6473036`, E2E improves 0.635 percent and strict CUDA improves 0.685 percent.
- It is a performance and trace result; no new alignment run was requested.

Final correctness gates:

- Current remote source suite: 62 of 62 tests (`6502143`).
- Exact-shape BF16/FP8/fused-router smoke: job `6502167`, nine iterations.
- Transport job `6493119`: fresh SM100 build plus bit-exact FP8 O and packed-scale transport on two GPU pairs, 257 rows, nine iterations, and both slots.
- Corrected scheduling export `6491211`: zero cross-stream heavy-kernel intersections in the representative replay and all 15 measured replays.
- Official final alignment job `6503374`: 24 prompts, 408 generated tokens,
  top-1/top-10/top-100 all 1.0, average and maximum rank 1.0.
- First current A1 attempt `6493412` failed before measured work because Torch was imported only under `TYPE_CHECKING`; `247e12e` fixed it. It is not a performance result.

## Trace and result locations

Current aligned FEP4 remote root:

`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_final_validation/a1_fep4_b6_128k`

Current FEP4 local reports:

`scratch/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_job_6502261/trace_sqlite/`

Current FEP8 remote root:

`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/serialized_ffn_validation/a1_ep8_fp8_o`

Current FEP8 local reports:

`scratch/qwen3_ffn_overheads_20260820/fp8_o_a1_ep8_job_6494759/nsys/`

Older corrected-pipeline reports:

`scratch/qwen3_ffn_overheads_20260820/corrected_a1_job_6490963/nsys/`

Older preposted-ready reports:

`scratch/qwen3_ffn_overheads_20260820/preposted_buffer_ready_validation/trace/nsys/`

Older best A15 FMHA-only reports:

`scratch/qwen3_ffn_overheads_20260820/best_a15_fmha_job_6485295/nsys/`

## 128K batch-capacity boundary

Under ATP1 and memory ratio 0.82:

- Observed KV capacity is 839,424 tokens per attention GPU.
- Batch 6 requires 786,844 scheduler-reserved tokens and is admissible.
- Batch 7 requires 917,995 tokens and is not admissible.
- Batch 8 requires 1,049,146 tokens and is not admissible.
- The FP8 O optimization reduces transport volume, not KV-cache size.
- Batch 8 requires a shorter context, additional attention capacity such as ATP2, or a smaller KV representation. Raising the memory ratio alone cannot supply the required 25 percent.

## Measurement and acceptance contract

Performance runs must prove:

- Exact target shape and placement recorded in `afd-result.json`.
- One exact target-batch warmup followed by 15 measured decode replays.
- Representative attention and model Nsight reports with identical step mapping. Legacy audits inspect rank 1; FMHA-only audits inspect representative attention rank 1 and model rank 5.
- Final latency is the arithmetic mean of the per-step maximum complete CUDA Graph span across profiled attention and model roles.
- At least 10 retained steps, at most five outliers, dominant range at most 10 percent, and maximum-to-median difference at most 10 percent.
- `SUCCESS`, strict metric output, expected report count, and no runtime exception.
- Official unchanged-scorer alignment before a candidate becomes an accepted correctness checkpoint.
- Trace proof of no overlapping heavy model kernels across the two FFN lanes.

Use the coordinator median only as a same-field E2E diagnostic. Never compare it with an older metric that used a different latency basis.

## Grouped fan-in and publication invariants

One logical FFN worker may serve many attention sources, but it must still run one model round per microbatch, not one full graph per source.

Retained requirements:

- Group all source commands for a model step.
- Concatenate each microbatch once.
- Publish Q/K/V to all destination edges under the existing publication phase.
- Wait for all required O-ready words once.
- Run exactly two FFN rounds per layer for MB2.
- Preserve one O publication per source edge.
- Keep source tensors alive until the existing completion event is synchronized.
- Bound large publication grids while retaining grid-stride payload coverage.
- Do not add cross-GPU barriers, acknowledgements, source-switch drains, or extra readiness phases.

Representative grouped-fan-in gates:

- Jobs `6459665` and `6459664`: A1/A2 grouped model steps at 26.190541 and 27.850899 ms with correct 564-node attention and 3,676-node model graphs.
- Jobs `6462898` and `6462899`: retained-batch lifetime correction at 26.099038 and 27.619945 ms, 15 of 15, zero outliers.
- Alignment job `6463085`: perfect A2 agreement and exact per-replay expert counts.
- High-fan-in campaign jobs exposed publication and per-source service scaling; those broad campaigns are closed.

## Placement and high-fan-in conclusions

- Exact no-mode A15 control `6263396`: 30.9095675 ms E2E and 30.7524840 ms strict CUDA.
- Best retained FMHA-only A15 diagnostic `6485295`: 31.7143450 ms E2E and 31.6887659 ms strict CUDA. It remains slower than no-mode.
- A8 FMHA-only job `6488053`: 31.2774395 ms E2E and 31.2384243 ms strict CUDA.
- A8 legacy control `6265476`: 30.8973810 ms E2E and 30.7433556 ms strict CUDA.
- The measured placement crossover is A8. The Qwen3 128K/b6/ATP1/FEP4/MB2 adaptive policy selects FMHA-only below ratio 8 and legacy at ratio 8 or above.
- Eight communication SMs help the isolated 45-row high-fan-in MoE shape, but hurt A1 and do not make A15 FMHA-only competitive. Retain 24 communication SMs for low fan-in.
- QKV publication CTA reduction and system-atomic completion both regressed A15. Do not reduce publication parallelism further.

## Closed FFN scheduling directions

The following families are rejected unless a new mechanism changes the causal model:

- Standalone per-phase ready kernels: correct but launch overhead erased the data-kernel savings. Jobs `6473693` and `6474094`.
- One wait kernel per lane handoff: exactly two model streams but 187 added nodes and regression. Job `6475180`.
- Captured stream-memory waits: exactly two streams but slower. Job `6475460`.
- Event-free two-stream scheduling: correct topology but cadence regression. Job `6475991`.
- Stream or graph-node priority variants: increased contention or synchronization. Jobs `6476272`, `6476535`, `6476970`, and `6477252`.
- Startup-ticket and waiter-order variants: two-stream topology but no acceptable bidirectional stagger. Jobs `6477716`, `6478604`, `6479079`, `6479509`, and `6479731`.
- Fused per-round admission and deferred combine variants: reduced graph nodes but still regressed. Job `6480443`.
- Fused input RMSNorm plus QKV quantization: job `6481196` regressed about 1.66 percent.
- Reduced publication grids and system-atomic completion: jobs `6486377` and `6487343` regressed.
- Deferred combine from one batch overlapping heavy work from another violates the governing model even if latency appears better.

Git history preserves the detailed experiment chronology removed from this file.

## Retained engineering lessons

- Prepost readiness, then send payload later; never add a receiver-credit round trip to the critical path.
- Use ready words and local causal clearing instead of payload sentinels or complete-buffer scans.
- Keep communication tails separate from exclusive heavy compute ownership.
- A cooperative K/V-to-Q grid barrier is unnecessary; one final publication boundary is sufficient.
- More measured overlap can mean resource contention, not useful pipelining.
- One fixed post-case sleep wastes allocation time and does not prove exit; use condition-driven process completion.
- Slurm spool paths are transient; use durable source and control paths.
- Keep materialized batch tensors alive until asynchronous consumers finish.
- Never resume a stopped campaign in place after its topology or metric contract changes. Create fresh state.
- Do not use legacy tail-capture AFD rows as target-batch performance evidence.

## Reusable workflow

Reusable remote controls live under:

`scripts/experiments/afd/oci_hsg/`

No active `codex_scripts/` directory exists in this worktree.

For future GPU work:

1. Keep source clean and committed on the remote branch.
2. Submit to `short` with an explicit one-hour limit unless the user changes the policy.
3. Set `FASTAFD_ALLOW_DIRTY_SOURCE=0`.
4. Keep FMHA-only allocations within one NVL72 fabric block using `--segment`.
5. Run the CPU/source contract first.
6. Run a fresh transport smoke only when transport or synchronization changes.
7. Gate on exact A1 performance and trace structure.
8. Run official alignment.
9. Advance to larger ratios only after the A1 gate passes and only on explicit user direction.
10. Copy requested raw Nsight reports into the local workspace without touching local Git.

## Resume checklist

- Read this file before changing the pipeline.
- Confirm remote branch and worktree are clean.
- Confirm no stale GPU jobs or claims exist.
- Preserve O projection on the model or FFN side.
- Preserve attention-side O activation quantization and FP8 O transport.
- Preserve exclusive model compute-unit ownership.
- Treat current FEP4 job `6498084` as the aligned acceptance point.
- Treat FEP8 job `6494759` as the latest scaling profile.
- Do not infer that current ATP1 supports batch 7 or batch 8 at 128K.
- Do not restart A7, A8, A15, or a broad campaign without explicit user direction.
