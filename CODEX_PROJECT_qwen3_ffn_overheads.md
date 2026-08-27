# Qwen3 FMHA-Only Attention Worker and FFN Pipeline

## Active surgical pre-FFN restoration with FP8 validation (2026-08-27 PDT)

The user clarified that pre-FFN optimization denotes the FFN component, not a
specific whole commit. The target FFN code is still historical `24b90c0`,
whose A7:F1/EP4 job `6522967` measured 29.358190933-ms strict CUDA and
178.825733 TPS per active GPU. The same compute tree produced A1 job `6520395`;
unchanged alignment job `6520958` scored all 408 generated tokens at top-1
with average and maximum vLLM rank 1.0. The earlier whole-range candidate
`d0a7eaf` was superseded, and queued smoke `6575467` was canceled before
allocation with zero elapsed time.

Remote branch `codex/fmha-megamoe-fp4-preopt-ffn-20260826` and worktree
`worktrees/fmha_megamoe_fp4_preopt_ffn_20260826` start from current FP8xFP4
and per-case-memory head `a61e5d4`. Clean commit `1b7038e` restores only the
MegaMoE FFN kernel/routing/instrumentation surface. Git proves that its C++
API, JIT host wrapper, heuristics, CUDA implementation, TMA/layout/PTX, and
top-k files are byte-identical to `24b90c0`; the runtime/wrapper/adoption files
are byte-identical to minimal FP8xFP4 commit `47a0672`; and the memory contract
test is byte-identical to `54055a4`. Route-stat, fused-router,
`num_concurrent_lanes`, and scheduler-tuning paths are absent. Current
non-FFN launcher validation, explicit DeepGEMM build-dir override,
final-snapshot cleanup, reusable bundle execution, and batch-7 memory
propagation remain intact.

Focused contracts pass 20/20, launcher syntax and `git diff --check` pass.
Initial validation mistakenly selected optional FP8xFP4. Smoke `6576460`
passed; one launch failed closed before `sbatch` due a missing derived venv,
and job `6576931` failed container bootstrap in 23 seconds due the default
missing image path. With both production paths pinned, FP4 A7 job `6577407`
completed at 25.8833942 ms / 202.832749 TPS per active GPU. The user clarified
that this is not an acceptance comparison: target `6522967` used the FP8xFP8
MegaMoE kernel. Its 32 reports were nevertheless copied locally with checksum
parity (33,627,949 bytes). FP4 A1 job `6579084` was canceled at 8:35 before a
sample was produced.

FP8xFP4 stays available as an optional code path, while acceptance now
explicitly sets `MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp8`. The corrected A7
dry run and full non-submitting validation select FP8xFP8, FMHA-only MegaMoE,
EP4, MB2 3+3, 0.82 / 839,424 KV tokens, exact batch 6, eight trays/32 GPUs,
168 prompts, and one warmup plus 15 measured steps. Corrected short-QoS job
`6579568` completed `0:0` in 21:52. The unchanged strict metric retained 15/15
steps with zero outliers at 28.738912667-ms mean, 28.676994-ms median,
29.552031-ms maximum, 3.314598% dominant range, and 182.679145 TPS per active
GPU. This is 0.619278266 ms / 2.1094% faster than historical pre-FFN job
`6522967`, so the low-cost proxy passes. Its 32 Nsight reports are copied
locally with 33,045,638 total bytes. Two-tray FP8 A1 alignment-source job
`6581273` completed `0:0` in 19:15 and produced 24 prompts / 408 generated
tokens with explicit FP8xFP8 provenance. Its eight Nsight reports are copied
locally with 10,286,425 total bytes. Unchanged one-node alignment job `6582345`
completed `0:0` in 12:51: 24 prompts / 408 tokens, top-1/top-10/top-100 all
1.0, and average/maximum rank both 1.0. The corrected FP8 gates therefore pass.

The user clarified that A7 EP4 is only a low-cost FP8 non-regression proxy and
need not reproduce `6522967` exactly. Restored A7 EP8 and A8 EP8 FP8
performance is the final criterion, but their large allocations take too long
to queue and must not be submitted directly before the smaller evidence is
complete.

The final campaign is exactly eight EP8 measurements produced by two jobs.
One 16-tray A7 allocation and one 18-tray A8 allocation each run b6-FP8,
b7-FP8, b6-FP4, and b7-FP4. Remote clean branch
`codex/fmha-megamoe-preopt-matrix-20260827` at `7c1491e` is based on surgical
FFN head `1b7038e` and adds only per-case MegaMoE precision,
precision-qualified case IDs, and fail-fast memory validation. Focused tests
pass 21/21; shell syntax and diff checks pass. All eight exact clean-head dry
runs and full non-GPU preflights pass. Batch 6 is 0.82 / no page override /
839,424 KV tokens / MB2 3+3. Batch 7 is 0.90 / 14,344 pages / 918,016 tokens /
MB2 3+4, leaving 64 tokens over the 917,952-token requirement. The A7 and A8
pools each have exactly four untouched rows. Short QoS supports the required
two-hour allocation and two concurrent jobs; every case retains a 30-minute
watchdog. Exactly two final jobs are submitted: `6583163` requests 16 trays
for the four A7 rows and `6583164` requests 18 trays for the four A8 rows. A7
started at 01:19:29 PDT on one contiguous NVL72 block and first claimed b7
FP8xFP8 with the correct 0.90 / 14,344-page / 918,016-token contract. A8
remains pending for resources with a current scheduler estimate near 03:05
PDT. No additional EP8 job exists.

A7 b7 FP8xFP8 completed first with a valid strict metric: 28.373346933-ms
mean, 28.301796-ms median, 29.368612-ms maximum, 15/15 samples, zero outliers,
and 215.871607 TPS/active GPU. This is 0.0065032 ms / 0.0229% faster than
historical `24b90c0` job `6531588`, so the intended EP8 FP8 performance is
recovered. Its 64 Nsight reports are copied locally (65,118,185 bytes). A7
then completed b7 FP8xFP4 at 27.499724800-ms mean, 27.437888-ms median,
28.423936-ms maximum, 15/15 samples, zero outliers, and 222.729502 TPS/active
GPU. Its 64 Nsight reports are copied locally with checksum parity (66,373,359
bytes). A7 b6 FP8xFP8 then completed at 25.354510667-ms mean, 25.303519-ms
median, 26.139264-ms maximum, 15/15 samples, zero outliers, and 207.063748
TPS/active GPU. It is 0.106900333 ms / 0.419852% faster than historical
`24b90c0` job `6531588`; its 64 Nsight reports are copied locally with checksum
parity (63,811,452 bytes). The allocation finally claimed b6 FP8xFP4 with the
correct 0.82 / no-page-override / 839,424-token contract and completed at
24.037447667-ms mean, 23.977343-ms median, 24.846303-ms maximum, 15/15 samples,
zero outliers, and 218.409212 TPS/active GPU. Its 64 Nsight reports are copied
locally with checksum parity (62,343,765 bytes). A7 job `6583163` completed
`0:0` in 1:22:47 with all four cases and zero failures. The complete A7 trace
set is 256 reports / 257,646,761 bytes. A8 job `6583164` started at 03:25:47
PDT on contiguous block `nvl72110-T[01-18]` and first claimed b7 FP8xFP8 with
the corrected 0.90 / 14,344-page / 918,016-token contract. It completed at
28.744260067-ms mean, 28.676160-ms median, 29.787231-ms maximum, 15/15 samples,
zero outliers, and 216.468339 TPS/active GPU: 0.014036933 ms / 0.048810% faster
than historical `24b90c0` job `6531587`. Its 72 Nsight reports are copied
locally with checksum parity (71,862,352 bytes). A8 next claimed b7 FP8xFP4
with the same corrected memory contract and completed at 27.950053667-ms mean,
27.868192-ms median, 29.048536-ms maximum, 15/15 samples, zero outliers, and
222.619330 TPS/active GPU. Its 72 Nsight reports are copied locally with
checksum parity (73,696,333 bytes). A8 then claimed b6 FP8xFP8 with the correct
0.82 / no-page-override / 839,424-token contract and completed at 25.694954600-
ms mean, 25.635168-ms median, 26.638844-ms maximum, 15/15 samples, zero
outliers, and 207.563446 TPS/active GPU. It is 0.000613400 ms / 0.002387%
faster than historical `24b90c0` job `6531587`; its 72 Nsight reports are
copied locally with checksum parity (72,910,726 bytes). The eighth and final
case, A8 b6 FP8xFP4, completed with the same corrected batch-6 contract at
24.316745067-ms mean, 24.250368-ms median, 25.236064-ms maximum, 15/15 samples,
zero outliers, and 219.327600 TPS/active GPU. Its 72 Nsight reports are copied
locally with checksum parity (73,813,231 bytes). A8 job `6583164` completed
`0:0` in 1:27:06 with all four cases and zero failures. The complete final
campaign therefore used exactly jobs `6583163` and `6583164`, produced all
eight required EP8 results, and copied all 544 Nsight reports / 549,929,403
bytes locally with checksum parity; all eight metric JSON files also match the
remote copies. A second consolidated copy is under
`scratch/qwen3_ffn_overheads_20260820/final_ep8_8case_nsys_20260827/`, with
one readable config-named directory per case (for example,
`A7_F1_EP8_B6_MegaMoE_FP8xFP8`) and no job-ID directory names. Remote checksum
verification passes for all 544 reports / 549,929,403 bytes.

The `fmha_only` handoff is one linear commit atop `linsj20/fmha_only` commit
`393bc98`, with no merge commit and no inherited OCI experimental history.
Its source, scripts, and tests are byte-identical to validated commit
`7c1491e` (tree `022e3994867f6eabd3692d7dbdaeea06207b7958`), which produced
jobs `6583163` and `6583164`; only project-memory Markdown differs. The commit
records the surgical FFN restoration, mixed FP8xFP8/FP8xFP4 bundle support,
correct batch-7 memory contract, all eight results, and consolidated Nsight
trace provenance. None of the older mixed-worktree code is retained.

## Active pre-tuning FP8xFP4 A7/A8 rerun (2026-08-26 PDT)

The user rejected carrying the final A7-specific K256/128-SM/single-wave stack
into the FP8xFP4 EP8 experiment. Pending A8 job `6562900` was canceled before
allocation. Remote branch `codex/fmha-megamoe-fp4-ep8-baseline-20260826` and
worktree `worktrees/fmha_megamoe_fp4_ep8_24b90c_20260826` start at historical
checkpoint `24b90c0`, which produced the reference A7/A8 EP8 jobs
`6531588`/`6531587`. Head `bfda796` adds only FP4 selection/conversion,
production-weight smoke support, exact bundle memory propagation, and an
FP4-only two-lane launch budget. FP8 stays on the checkpoint's original
150-SM kernel; route-stat, fused-router, K256, 128-SM single-lane, wave, and
cache-policy tuning are absent.

Remote contracts pass 20/20. Initial four-GPU smoke `6572843` reached fresh
host compilation and caught asymmetric FP4/FP8 launch-pointer signatures; it
was canceled at 3:11. Commit `bfda796` fixes only that signature. Retry smoke
`6572969` completed `0:0` in 2:29 with empty stderr, FP4 cosine 0.975472, and
finite exact H4096/I1536/E128/EP4 outputs for 3- and 64-row cases.

The user added an A7:F1/EP4 performance gate before EP8. Exact 128K/b6/MB2
3+3 job `6573176` ran clean `bfda796` on eight trays, `short`, with a 30-minute
limit and completed `0:0` in 22:35. Preflight pinned FMHA-only MegaMoE
FP8xFP4, ratio 0.82, 839,424-token KV capacity, and exact-max-batch 6. The
unchanged strict metric retained 15/15 steps with zero outliers and measured
33.968444467-ms mean, 33.889990-ms median, 34.886594-ms maximum, and
154.555208 TPS/active GPU. Historical pre-tuning FP8xFP8 job `6522967` is
29.358190933 ms / 178.825733 TPS/GPU, so the baseline FP4 candidate regresses
by 4.610253533 ms / 15.70% and 13.57% throughput. Attention and model means
are 33.968444467 and 33.817956 ms, respectively. The retained FP4 two-lane
safety contract budgets 64 SMs per lane, but captured A7 graphs serialize the
two MegaMoE calls; this recreates the previously rejected lane-sharing cost
without overlap. The performance gate failed, so do not submit EP8 from this
head.

Fresh A7/A8 EP8 plans live under
`fmha_megamoe_fp4_ep8_baseline_20260826/a7_a8_ep8_b6_b7_bundle_bfda796/`;
their isolated pools validate as two pending 16-tray rows and two pending
18-tray rows. All four dry runs select FMHA-only MegaMoE FP8xFP4. Batch 6 uses
0.82 / 839,424 tokens / MB2 3+3. Batch 7 explicitly uses 0.90 / 14,344 pages /
918,016 tokens / MB2 3+4; required 917,952 leaves 64 tokens headroom. Both
enforce exact capacity maxima. The two large EP8 jobs remain unsubmitted until
the A7 EP4 gate passes.

## Accepted same-rank FP8xFP4 MegaMoE iteration (2026-08-26 PDT)

Remote branch `codex/fmha-megamoe-fp4-20260826` in worktree
`worktrees/fmha_megamoe_fp4_20260826` adds the explicit
`MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4` option while retaining `fp8` as the
default. The minimal path converts each Qwen 128x128 block-FP8 expert weight
once into packed per-32 MXFP4, transforms its scale-factor layout, dispatches
the existing same-rank `fp8_fp4_mega_moe` kernel, and records precision in
worker logs and result metadata. Executable commit
`6c210cf1fa474dbc4da4b7a7f4abe1f1269d04b6` also forwards the production
two-lane count through the host/API boundary; FP8xFP8 keeps its accepted
single-concurrent-lane behavior.

The first exact job `6559673` exposed an asynchronous FP4 launch failure at
the first eager prefill. A global `CUDA_LAUNCH_BLOCKING=1` diagnostic job
`6560987` was inconclusive because it delayed initialization until symmetric-
memory rendezvous timed out. Focused tests then isolated the missing lane
budget: production owns two lane streams, but the FP4 API hardcoded one
concurrent lane. After the fix, four-GPU job `6561772` passed the exact
production converter, numerical gate (cosine 0.975472), and concurrent
two-lane equality checks; concurrent replay was 1.53x faster than sequential
at the representative 64-row case.

Exact A1:F1/EP4/128K/b6/MB2-3+3 job `6561851` ran clean commit `6c210cf` on
`batch`/`short` with a 30-minute limit and completed `0:0` in 18:14. All four
FFN ranks reported `precision=fp8_fp4`, EP4, and two lanes; all graphs captured
and 24 samples generated 17 tokens each. Unchanged strict extractor job
`6562231` retained all 15 target steps with zero outliers and measured
22.689653667-ms mean, 22.683360-ms median, 22.798753-ms maximum,
0.594061% dominant range, and 132.218854 tokens/s/active GPU. Against accepted
FP8xFP8 job `6520395` at 23.3908480 ms and 128.255290 tokens/s/GPU, FP8xFP4
improves strict latency by 0.701194333 ms / 2.997729% and throughput by
3.090371%.

Official unchanged alignment job `6562471` completed `0:0` in 10:58 under its
30-minute short-queue limit. It scored 24 prompts / 408 generated tokens with
top-1 agreement, top-10 hit rate, top-100 hit rate, and average/maximum vLLM
rank all 1.0; the trace-inspection artifact is present. Local focused contracts
pass 20/20, both launchers pass shell syntax, and `git diff --check` passes.
All eight raw job-`6561851` Nsight reports (ranks 1--8, 9,822,812 bytes) were
copied locally with file-by-file SHA-256 parity under
`scratch/qwen3_ffn_overheads_20260820/fmha_megamoe_fp4_20260826/`
`exact_a1_6c210cf_lanes2/trace/nsys/`. This precision goal is accepted and
closed.

### Active FP8xFP4 EP8 A7/A8 scaling observation

Clean remote commit `a61e5d4` promotes the proven per-case memory contract
into `run_afd_bundle.sh` and explicitly forwards MegaMoE precision. All four
dry runs select FMHA-only MegaMoE FP8xFP4 with exact-capacity gating. Batch 6
uses memory ratio 0.82, 839,424 KV tokens, and MB2 3+3. Batch 7 uses memory
ratio 0.90, 14,344 pages / 918,016 KV tokens, and MB2 3+4; its required
917,952 tokens leave 64 tokens of configured headroom. This closes the
propagation defect from job `6555417`, where batch 7 incorrectly inherited
the batch-6 capacity.

Two same-allocation bundles are submitted on `batch`/`short`: job `6562899`
requests 16 contiguous trays / 64 GPUs for A7:F1/EP8 b7 then b6, and job
`6562900` requests 18 contiguous trays / 72 GPUs for A8:F1/EP8 b7 then b6.
Historical paired runs took about 50 minutes, so each allocation has a
one-hour cap while each individual case retains a 30-minute watchdog. Task
root is `fmha_megamoe_fp4_20260826/a7_a8_ep8_b6_b7_bundle_a61e5d4/`.

A7 job `6562899` completed `0:0` in 46:11 on
`nvl72096-T[01-15,17]`. Batch 7 passed the corrected memory gate on all 56
attention ranks: configured ratio 0.90 / 14,344 pages / 918,016 tokens,
observed 14,396 pages / 921,344 tokens, exact max batch 7, and MB2 3+4. Its
strict CUDA result is 28.942748933-ms mean, 28.890207-ms median,
29.875679-ms maximum, 3.694539% dominant range, 15/15 samples, zero outliers,
34.550968 TPS/user, and 211.624681 TPS/active GPU. Attention is critical with
a 28.942748933-ms mean versus 28.803287467 ms for model rank 57. Against
historical FP8xFP8 job `6531588` at 28.379850133 ms / 215.822140 TPS/GPU,
FP8xFP4 regresses by 0.562898800 ms / 1.983445% and 1.944870% throughput.
This is not a same-head precision isolation because later FP8 job `6555417`
did not produce a batch-7 metric.

Batch 6 received its intended ratio 0.82 / 839,424-token contract and passed
the observed exact-max-batch-6 memory gate, so the earlier propagation defect
did not recur. It then failed during first eager prefill with an asynchronous
CUDA launch failure surfaced by `gate_topk`; no batch-6 performance metric is
valid. A8 job `6562900` remains pending for resources.

## Same-rank FP8 MegaMoE iteration (2026-08-25 PDT)

The user superseded the earlier split-MegaMoE closure for one specific new
design: reuse the original AG/EG Megakernel's communication/compute overlap on
FMHA-only FFN workers, where source, expert, and destination ranks are the same
EP group.  This requires one collective MegaMoE kernel per FFN GPU/layer rather
than separate AG and EG kernels.  Precision must match the original split
AG/EG system: FP8 activations and per-32 UE8M0 FP8 expert weights.  Preserve
the existing `fp8_fp4_mega_moe`; add a distinct `fp8_fp8_mega_moe` path.

Local candidate work adds the FP8-weight specialization to the original
same-rank MegaMoE implementation without changing the default FP4
specialization, plus a narrow `MINISGL_AFD_MOE_BACKEND=megamoe` FMHA adapter.
The adapter uses the split EG path's
`requant_qwen_fp8_weights_per32`/`transform_weights_for_mega_moe`, one
graph-stable symmetric buffer per physical FFN lane, group-local TP-column
routing, one dispatch+expert-FFN+combine MegaMoE call, and the existing dense
TP reduction when TP > 1.  Router/top-k plus FP8 input staging remains a
pre-step; “one kernel” refers to replacing the split AG+EG communication and
expert kernels with one same-rank collective Megakernel.

Remote isolated worktree:
`worktrees/fmha_megamoe_fp8_20260825`, branch
`codex/fmha-megamoe-fp8-20260825`.  CPU source job `6518324` passed 63/63
tests; `6518232` was an infrastructure-only failure before tests because the
old `/home` image path no longer existed.  After correcting two smoke-harness
issues that occurred before the candidate kernel (`6518376` GPU visibility and
`6518466` scale staging), four-GPU exact-Qwen smoke `6518551` passed with
small-reference cosine 0.9998625/max-absolute error 0.672459 and finite exact
H4096/I1536/E128/EP4/top-k8 output.  This directly exercises the new FP8xFP8
same-rank dispatch, both expert GEMMs/SwiGLU, combine, and cross-rank reduction.

The first exact job `6518767` was canceled when its runtime provenance exposed
that the legacy reproduction wrapper silently dropped `fmha-only`/`megamoe` at
the container boundary; it never exercised the candidate.  Commits `fea2f31`
and `4eb0535` now propagate, validate, log, and record both settings.  Correct
exact job `6518950` loaded and transformed all 94 layers, reported
`precision=fp8_fp8`, captured both MB2 graphs, and generated all 24 samples.
The driver exited only because its target-window marker was not written, but
the coordinator CPU trace proves all 15 target decode steps 1537--1551 ran.
Their E2E intervals have 40.579301-ms median and 40.611376-ms mean, so this is
a valid diagnostic and a large performance rejection, not an accepted metric
or alignment checkpoint.  The marker loss was a reproduction-wrapper defect:
it requested only 16 output tokens, although the established contract needs
one token from prefill, one full-batch decode warmup, and 15 measured decode
steps (17 total).  Execution commit `90f5ce3` now derives that count from the
capture contract and uses it consistently for sampling, KV-capacity accounting,
and result validation; the production coordinator/profiler logic was correct.

CPU trace-stat job `6519549` attributes the regression: representative model
rank 5 has 2,820 MegaMoE calls (94 layers x MB2 x 15 steps), averaging
165.645 us and totaling 467.120 ms.  Thus the Mega kernels alone contribute
about 31.14 ms per decode step.  The original DeepEP plus two grouped GEMMs is
roughly 86 us per model round, so launching the current same-rank MegaMoE once
per microbatch cannot meet the sub-25-ms target.  Full MB2 packet merging is
not assumed safe because prior job `6508956` showed that delaying MB0 QKV until
MB1 destroys the attention wavefront.

Execution commit `ead5c08` adds a production-like fused-router plus FP8xFP8
MegaMoE CUDA-graph benchmark while retaining the independent numerical gate.
Four-GPU `short` job `6519759` reached the new fused-router build, then failed
before timing because the wrapper left TVM-FFI on the unavailable container
home cache.  Execution commit `b26e539` routes that build into the experiment
workspace, and replacement `6519790` sweeps rows 3/6 and 96/128/152 MegaMoE
SMs with 20 warmups and 200 replays.  It is the next decision gate: first test
launch/grid efficiency and the merged-token kernel ceiling, then change
production scheduling only if the measured ceiling can preserve or recover
the MB2 wavefront.  OCI-HSG has neither `crun` nor `cdb`, so this run uses the
checked-in Slurm wrapper and the canonical Lustre image and environment.

Job `6519790` completed `0:0` in 1:30 and retained the numerical result above.
Production-like fused-router plus MegaMoE graph replay at rows 3 measured
91.395/91.681/96.655 us for 96/128/152 SMs.  Rows 6 measured
121.773/113.085/113.738 us.  Thus a six-row call is 38 percent cheaper than
two best three-row calls, but not enough to override the prior exact proof
that full MB2 merging destroys the attention wavefront.  Reducing the current
grid from 152 to 96 SMs saves only about 5.3 us per model round in isolation.
All four full-run model reports independently show the same approximately
165.6--165.9-us MegaMoE average across 2,820 calls (CPU export jobs
`6519549`, `6519871`, `6519872`, `6519873`), so the 74-us gap from the smoke
is systematic rather than one slow rank.  The smoke used a 384-aligned buffer
while production used 768 (`max_rows=512`); execution commit `1a5f8e1` and
four-GPU job `6519989` now isolate that buffer-layout variable at rows 3 and
96/128/152 SMs before any production change.

Job `6519989` completed `0:0` in 1:21.  The production 512-requested /
768-aligned buffer measured 94.751/94.503/98.632 us at 96/128/152 SMs, only
about 3 us above the small buffer; capacity/scale-pool stride is not the
74-us full-run gap.  The rank-5 duration histogram is strongly modal across
the 2,820 calls: 2,725 are 155--175 us, while only 15 are 50--70 us, pointing
to layer routing density rather than random launch skew.  Execution commit
`32b851d` adds active-expert reporting plus deterministic kernel-only bounds
for 24 active experts per destination and 32 active experts on one destination.
Four-GPU job `6520067` runs those bounds at the same production bucket and
96/128/152 SMs; use it to decide whether the next change belongs in routing-
density scheduling or the communication/barrier pipeline.

Job `6520067` completed `0:0` in 1:23.  The fused random route activates at
most 20 local experts and measures 94.52/94.82/98.55 us at 96/128/152 SMs.
Balanced 24-active kernel-only time is 96.57/98.04/99.31 us; the deliberately
imbalanced 32-active destination is 116.58/139.72/119.23 us.  Route density
matters but still does not reach the full graph's 166 us.  SQLite interval
joins close the remaining cause: 2,805 of 2,820 Mega calls overlap a resident
one-CTA O-ready wait, while the only 15 calls without that wait are exactly the
50--70-us final calls.  Execution commit `272d814` adds a finite high-priority
one-CTA contention probe.  Submission `6520164` was canceled with zero elapsed
time before allocation after detecting a mistyped output root; corrected
four-GPU job `6520168` sweeps 96/128/136/144/148/150/152 SMs with a
180,000-cycle resident CTA to select the production reservation.

Job `6520168` completed `0:0` in 1:26 and isolated a 2-CTA cluster-tail
pathology.  Without the resident wait CTA, balanced-24 routing measures
96.44--101.31 us across 96--152 SMs; with the resident CTA it stays within
about 0.6--1.8 us through 150 SMs, but jumps from 98.96 to 180.33 us at the
full 152-SM grid.  The imbalanced one-destination bound behaves the same:
roughly 116.8--122.5 us through 150 SMs and 200.39 us at 152.  MegaMoE uses
2-CTA clusters, so a full-device launch strands its final cluster while the
lane's one-CTA O-ready waiter is resident.  Execution commit `db9b3f8`
therefore applies the general production policy of reserving one complete
2-SM cluster: the same-rank MegaMoE backend configures DeepGEMM for 150 of
152 SMs, validates the even positive grid, and logs both compute and reserved
SM counts.  This retains nearly all dense-compute capacity while avoiding the
measured contention cliff; the next gate is the full CPU source suite followed
by a fresh exact A1:F1 run with the corrected 17-token capture window.

CPU jobs `6520292` and `6520338` failed before executing a test because two
ad-hoc `unittest discover` invocations did not reproduce the established
container test-path setup.  Reusing the existing general
`control/cpu_unit_tests.sbatch` workflow and its CPU venv fixed the harness;
job `6520373` passed all 64 tests in 1.003 s on clean execution commit
`db9b3f8`.  The exact-launcher dry run then proved `fmha-only`/`megamoe`,
A1:F1, 128K, batch 6, MB2 3+3, capacity-max enforcement, and 15 measured
decode steps; source code derives 17 total output tokens.  Exact short-queue
job `6520395` was submitted from the same clean commit with results under
`fmha_megamoe_fp8_20260825/exact_a1_reserve2/`.

Exact job `6520395` exercised the intended candidate on all four FFN ranks:
`precision=fp8_fp8`, EP4/128 experts, two graph-stable lanes, 150 compute SMs
plus one reserved 2-SM cluster, and captured MB2 3+3 graphs.  It generated all
24 samples with 17 tokens each and synchronously captured warmup step 1537 plus
measured steps 1538--1552 in all eight Nsight reports.  The Slurm job returned
`1:0` only after the valid capture because the reproduction validator still
parsed two obsolete log contracts: it did not accept the profiler's new
`warmup_step_id`/`trace_count` fields and expected per-replay worker log lines
that production intentionally no longer emits.  Execution commits `61b9d7d`
and `ee80614` update the general validator to require the authoritative
`capture-complete.json`, exact warmup/15-step IDs, and one synchronous profiler
start/stop in every GPU-worker log.  They do not change the measured model or
kernel path.  CPU-only recovery of the checked-in validation block produced a
23.3986365-ms coordinator median; two host intervals include synchronous
profiler-stop distortion and are not used as the acceptance metric.

Unchanged strict extractor job `6520772` completed `0:0` in 30 seconds from the
eight preserved reports.  Representative attention rank 1 plus model rank 5
give 23.3908480-ms strict CUDA mean, 23.360160-ms median, 23.630528-ms maximum,
1.571633-percent dominant range, all 15 samples retained, zero outliers, and
128.255290 tokens/s/active GPU.  Versus aligned original-system checkpoint job
`6502925` at 24.9813376 ms and 120.089646 tokens/s/GPU, latency improves
1.5904896 ms / 6.3667 percent and throughput improves 6.7996 percent.  Final
clean-head source job `6520791` passed all 64 tests on `ee80614`.

Official unchanged-scorer alignment job `6520958` completed `0:0` in 11:06.
It scored the candidate's actual pinned benchmark prompt set: 24 exact 128K
Qwen chat prompts deterministically extended by cut/repeat from the established
512-by-8K real-text corpus, not dummy prompts.  Across all 408 generated tokens,
top-1 agreement, top-10 hit rate, and top-100 hit rate are each 1.0; average and
maximum vLLM rank are both 1.0.  The accepted same-rank FP8 MegaMoE checkpoint
is measured executable commit `db9b3f8` with validator-only clean handoff head
`ee80614`; it satisfies the sub-25-ms and unchanged-alignment goal.

Cleanup/default promotion on 2026-08-25 makes the accepted same-rank `megamoe`
backend the default whenever model placement resolves to `fmha-only`.  Backend
resolution is centralized and fail-fast: explicit `deepep` remains available,
same-rank `megamoe` is rejected outside FMHA-only, and split `megamoe_m2n` is
rejected outside legacy placement.  The comprehensive campaign preserves its
legacy `deepep` default; the exact reproduction wrapper preserves its legacy
`megamoe_m2n` default.  Both launchers now print the resolved backend, and the
comprehensive result records it.  Local launcher syntax/diff checks and 20
focused contracts passed.  Dry runs proved FMHA-only defaults to `megamoe`, an
explicit FMHA-only `deepep` override is retained, and each legacy default is
unchanged in both launchers.  The first comprehensive dry-run used ratio 1 and
was correctly rejected by the established 128K pmap ratio-15 invariant; the
valid ratio-15 rerun passed all three backend-selection cases.  Clean remote
handoff commit `24b90c0` records the default-policy delta, and CPU source-gate
job `6522493` completed `0:0` with all 65 tests passing.  No additional GPU run
was submitted because this delta does not alter the already validated kernel,
runtime compute path, exact 128K metric, or prompt-alignment result.

The next requested scaling observation copies all eight raw Nsight reports
from accepted A1 job `6520395` back to workspace-local
`scratch/qwen3_ffn_overheads_20260820/fmha_megamoe_fp8_20260825/`
`exact_a1_reserve2/trace/nsys/`.  A clean A7:F1 dry run on `24b90c0` proved
FMHA-only default `megamoe`, 28 attention plus four EP4 FFN GPUs on eight
trays, ATP1/MB2 3+3, batch 6 at the exact scheduler-admissible 128K ceiling,
168 pinned real-text prompts, and one warmup plus 15 measured Nsight graph
launches.  Short-queue job `6522967` was submitted with a one-hour limit to
`fmha_megamoe_fp8_20260825/a7_f1_ep4_128k_b6_default_24b90c0/`.  OCI-HSG still
lacks `crun`, so the checked-in comprehensive Slurm launcher is the
reproducible execution path.  Two-minute monitoring through completion found
no hang: all eight tray progress files advanced and all 32 GPUs remained
active throughout the otherwise quiet 128K prefill.  The job completed `0:0`
in 22:19 with all 168 samples at 17 tokens.  Strict extraction retained all 15
steps and measured 29.3581909333 ms CUDA mean, 3.591555-percent dominant range,
and 178.825733 tokens/s/active GPU from representative ranks 1 and 29.  Versus
the prior compliant A7 job `6504134` at 32.6938360 ms / 160.580728 TPS/GPU,
latency improves 3.335645067 ms / 10.202673 percent and throughput improves
11.361889 percent.  All 32 raw reports are copied locally under
`scratch/qwen3_ffn_overheads_20260820/fmha_megamoe_fp8_20260825/`
`a7_f1_ep4_128k_b6_default_24b90c0/trace/nsys/`.

The follow-up clean EP-scaling case uses normalized A4:F1 with EP8.  The
comprehensive dry run resolved this to 32 attention GPUs plus eight exclusive
FFN GPUs on ten trays, ATP1/MB2 3+3, batch 6 at the exact 128K capacity ceiling,
192 pinned real-text prompts, FMHA-only MegaMoE, and one warmup plus 15 measured
Nsight graph launches.  Short-queue job `6530508` was submitted from detached
clean production head `24b90c0683a6bdb3820ae2b7f35a360d30951826`; route
statistics are disabled.  Its run root is
`fmha_megamoe_fp8_20260825/a4_f1_ep8_128k_b6_default_24b90c0/`, with the exact
allocation under
`afd_qwen3_131072_r4_ag32_fg8_atp1_fep8_adp32_b6_n10_20260825_163031_manual_na/`.
The pre-run hypothesis is approximately 26 ms strict CUDA E2E, between clean
A1/EP4 job `6520395` at 23.390848 ms and A7/EP4 job `6522967` at
29.3581909333 ms, because each EP8 FFN rank receives four attention source
ranks instead of seven for A7/EP4 and owns 16 rather than 32 local experts.
The GPU workload and capture completed, but job `6530508` ended `1:0` after
23:34 because the newly written plan row used an unrecognized status label;
the strict extractor requires `needs_corrected_target_batch_rerun`.  This was a
post-capture bookkeeping failure: all 192 samples generated 17 tokens, the
sample/coordinator/capture records were durable, and all 40 Nsight reports were
present.  After correcting only the plan status, first CPU-only recovery job
`6531010` failed immediately because its result-manifest path omitted the
container's `experiment/` directory.  Corrected CPU-only job `6531018`
completed `0:0` in 31 seconds without another GPU run.  Strict extraction
retained all 15 steps with zero outliers and measured 24.9519134667 ms mean,
24.919811 ms median, 25.402659 ms maximum, 2.066019-percent dominant range,
and 192.370016 tokens/s/active GPU from attention rank 1 and model rank 33.
This is 1.561065467 ms / 6.673830 percent slower than A1/EP4, but
4.406277467 ms / 15.008682 percent faster than A7/EP4; it beats the 26-ms
hypothesis by 1.048086533 ms and is 0.048086533 ms below 25 ms.  The profiled
attention and model-role graph means are 24.951913467 and 24.860606733 ms,
respectively, so EP8 has reduced the FFN path enough that attention is now the
strict E2E limiter.  The strict metric, run metadata, and all 40 raw reports
are copied locally under
`scratch/qwen3_ffn_overheads_20260820/fmha_megamoe_fp8_20260825/`
`a4_f1_ep8_128k_b6_default_24b90c0/`.

The next requested EP8 scaling comparison reuses clean production head
`24b90c0683a6bdb3820ae2b7f35a360d30951826` and submits exactly two
same-allocation bundles for four cases: A7:F1 batch 6+7 and A8:F1 batch 6+7,
all at 128K, ATP1, MB2, FMHA-only MegaMoE.  A task-local copy of the reusable
bundle runner was generalized to read an explicit per-case memory contract;
the executable model/kernel source is unchanged.  Batch 6 retains memory ratio
0.82, the 839,424-token capacity, exact-capacity gating, and MB2 3+3.  Batch 7
uses the previously validated memory ratio 0.90, 14,344 pages / 918,016 tokens,
exact-capacity gating, and MB2 3+4.  All four dry runs passed.  A7 bundle job
`6531588` requests 16 trays / 64 GPUs for 56 attention plus eight FFN ranks;
A8 bundle job `6531587` requests the full 18-tray / 72-GPU NVL72 fabric for 64
attention plus eight FFN ranks.  Both use `short`, two hours, one-hour per-case
watchdogs, no requested or excluded nodes, and were pending for resources
immediately after submission.  The maximum-batch-first pool policy runs batch
7 before batch 6 in each allocation.  Local and remote task root:
`fmha_megamoe_fp8_20260825/a7_a8_ep8_b6_b7_bundle_24b90c0/`.

A7 bundle `6531588` subsequently ran on contiguous `nvl72072-T[01-16]` and
completed both cases with exit `0:0` in 50:05.  Batch 7 passed the exact
capacity ceiling on all 56 attention ranks with 14,396 observed pages / 921,344
tokens per rank (the configured override was 14,344 pages), then produced a
strict 28.3798501333-ms CUDA mean from 15/15 eligible samples, zero outliers,
35.236268 tokens/s/user, and 215.822140 tokens/s/active GPU.  Batch 6 passed
the exact ceiling with 13,116 observed pages / 839,424 tokens on every rank and
produced 25.4614114 ms, 15/15 samples, zero outliers, 39.275120 tokens/s/user,
and 206.194382 tokens/s/active GPU.  The batch-7 latency is 11.462203 percent
above batch 6, while its throughput per active GPU is higher because it serves
392 rather than 336 users on the same 64 active GPUs.  Compact metrics and run
metadata plus all 64 Nsight reports per case are copied locally under the
shared task root (`trace/nsys/i131072-fep8-r7-atp1-b{6,7}/`, 62 MB each);
local and remote report counts match exactly.  A8 bundle `6531587` remains
pending with zero elapsed time under `QOSGrpNodeLimit`.

The active A7 kernel diagnosis uses the same-rank routing invariant directly:
under EP4/TP1, all routes for one expert from all four source ranks are pooled
on its unique owner GPU, so MegaMoE already reuses that expert's weights across
tokens within a call.  A7 nevertheless activates nearly all 32 local experts,
whereas A1 activates materially fewer; the observed kernel increase from about
50--60 us to 80--100 us is therefore plausibly distinct-expert weight traffic,
not duplicate same-expert launches.  CPU-only SQLite export job `6524448`
confirms the exact scale: accepted A7 representative rank 29 has 3,008
MegaMoE calls averaging 97.4933 us (96.864-us median, 74.880-us minimum,
293.260 ms total), while accepted A1 representative rank 5 has the same 3,008
calls averaging 61.371 us (48.543-us minimum, 184.604 ms total).  The
36.122-us call delta contributes 6.791 ms across 188 calls per decode step and
fully explains the observed 5.967-ms E2E increase.  A7 also has a strong lane
asymmetry: its two model streams average 87.349 and 107.856 us, versus balanced
60.858/61.895-us A1 streams.  The exact A7 compiled specialization uses a
3,840-token aligned capacity, 16 experts per wave, two epilogue warpgroups,
and 150 SMs; A1 uses a 768-token capacity and 32 experts per wave.

One suspicious heuristic transition was tested: rows 3 schedules all 32
experts in one wave, while rows 21 schedules two 16-expert waves even though
each expert remains within one 16-row M tile.  Remote branch
`codex/fmha-megamoe-a7-wave-tuning-20260825`, commit `0462468`, adds explicit
fail-fast tuning controls plus a paired exact-Qwen smoke for expert-wave and
epilogue width.  Four-GPU `short` job `6524399` compares rows 3/21, 150 SMs,
one/two epilogue warpgroups, and 16/32 experts per wave at the production
512-token A1 bucket with 20 warmups and 200 graph replays.  The first shell
attempt created no job because local variable expansion produced an invalid
`/logs` path; the explicit-path retry is the authoritative submission.  Job
`6524399` completed `0:0` in 4:11.  At rows 21, one epilogue warpgroup with the
retained 16-expert wave is best at 130.813 us fused router-plus-MegaMoE versus
132.220 us for the production two-warpgroup configuration; forcing 32 experts
per wave regresses to 135.415/137.187 us.  The 1.407-us / 1.06-percent best
gain is far below the approximately 23.2 us/call needed to recover 4.358 ms
from exact A7, so it is rejected without an E2E run.  The user explicitly
requires this promotion rule: run E2E inference only after a substantial
isolated MegaMoE speedup.  The next benchmark revision reports MegaMoE-only
and fused timings separately, sweeps balanced 16/20/24/28/32 active-expert
loads, and uses the exact A7 3,840-token aligned capacity before any further
full-model job.

The expert-weight byte model is now exact, but its A1/A7 application remains
gated on real routing counts.  One local Qwen expert has 18,874,368 bytes of
FP8 matrix data (`2*1536*4096 + 4096*1536`) and 589,824 bytes of packed per-32
UE8M0 scales, or 19,464,192 bytes total and a 2.433-us weight-only floor at
NVIDIA's published 8-TB/s theoretical per-Blackwell-GPU bandwidth (the GB200
superchip publishes 16 TB/s for two GPUs).  A kernel reloads that full expert
for every 16-row M block, so the correct real-work byte numerator is
`19,464,192 * sum_expert ceil(real_recv_tokens / 16)`, not a guessed active
expert count.

Production-capacity isolated job `6524881` completed `0:0` in 2:51 on clean
commit `19c2d9f`, with numerical cosine 0.99986249/max-absolute 0.672459.  At
128 SMs its balanced MegaMoE-only curve is 78.370/91.695/101.572/112.518/
119.915 us for 16/20/24/28/32 active experts.  The fit is
`38.466 + 2.5978 * active_experts` us with R-squared 0.9913; 32 active experts
stream 622,854,144 algorithmic bytes at 5.194 TB/s, 64.93 percent of
theoretical.  Random 32-active routing measures 118.994 us MegaMoE-only plus
11.030 us router at the best tested 128-SM/one-epilogue-warpgroup point.  The
controlled slope proves expert density strongly affects time, but it also
invalidates the earlier assumption that exact A7's 97.493-us average means all
32 experts are active: timing interpolation would suggest roughly 22.7 at A7
and 8.8 at A1, but those are not real counts and must not be used for a
roofline conclusion.  No retained accepted-run artifact records per-call
active experts or 16-row M-block occurrences; the current Mega adapter passes
the optional expert counter as null.  Instrument the profiler window to
accumulate real token, active-expert, and M-block occurrences on each FFN GPU.
Only then compare A1 and A7 achieved bandwidth and decide whether the slowdown
is routing-driven or an A7 implementation-efficiency loss.  Do not call the
target bandwidth-impossible from synthetic 32-active data.  Official hardware
source: `https://www.nvidia.com/en-us/data-center/gb200-nvl72/`.

Remote commit `c18b4b0` adds optional production-window counters for received
routes, active-local-expert occurrences, 16-row expert-M-block occurrences,
and call count; normal calls retain null counters.  The isolated harness now
derives exact per-EP-rank route counts from generated top-k-8 IDs and reports
`19,464,192 * max_rank_M_blocks / kernel_time` as effective weight bandwidth.
The source-count scaling axis is A1/A7/A17, not an arbitrary row grid.  One A
tray contains four ATP1 attention GPUs; batch 6 with MB2 contributes three
tokens per GPU to each kernel lane.  Therefore the EP4 harness uses 3/21/51
rows per source rank, representing 12/84/204 global tokens and 96/672/1,632
global expert-token routes, respectively.  It sweeps 4/8/12/16/20/24/28/32
validated active local experts at 128 and 150 SMs with the same 3,840-aligned
A7 capacity.  Invalid submission `6526864` was canceled before use after
local shell expansion corrupted its output/source paths.  Explicit-path
replacement `6526892` is the authoritative four-GPU isolated job; no E2E
inference is authorized for this sweep.  Job `6526892` completed `0:0` in
2:39 and its timings are valid, but its provisional roofline fields are not:
the A17/51-row heuristic selects `BLOCK_M=32`, while the host model assumed
16 and consequently reported impossible greater-than-8-TB/s values.  Remote
commit `805545b` removes the block-size assumption: an untimed diagnostic
kernel now returns the actual received-route, active-expert, and M-block
counters for every route pattern, while timing still uses null counters.
Cache-reusing corrected sweep `6527151` supersedes the roofline output of
`6526892`.  Corrected job `6527151` completed `0:0` in 1:24 with numerical
cosine 0.99986249/max-absolute 0.672459 and empty stderr.  At fixed 128 SMs,
the active-expert/MegaMoE-us curves are A1: `4:43.192, 8:58.575, 12:68.955,
16:76.796, 20:87.484, 24:99.188`; A7: `4:59.923, 8:65.120, 12:70.836,
16:79.036, 20:91.956, 24:102.059, 28:113.341, 32:121.452`; A17:
`4:64.280, 8:67.099, 12:83.814, 16:82.204, 20:95.000, 24:105.010,
28:116.040, 32:122.357`.  Kernel-reported M-block counts are respectively
A1 `4/8/12/16/20/24`, A7 `12/16/12/16/20/24/28/32`, and A17
`16/16/24/16/20/24/28/32`; the A17 specialization uses `BLOCK_M=32` while
A1/A7 use 16.  A descriptive same-128-SM fit across all 22 points is
`time_us = 35.746 + 1.648*active + 0.987*M_blocks + 0.01249*received_routes`
with R-squared 0.9936 and 1.50-us mean absolute error.  At matched active and
M-block counts, A7 versus A1 and A17 versus A7 usually differ by only 1--4 us;
the large scaling steps follow extra active experts and expert tiles, not raw
token count alone.  The best controlled weight-only bandwidth is 5.779 TB/s
(72.2 percent of 8 TB/s) at A17/12 active/24 blocks/150 SMs; A7/32 active/32
blocks reaches 5.128 TB/s (64.1 percent) at 128 SMs.  This supports a real
workload-scaling cause while still leaving meaningful kernel headroom.  It
does not substitute for real accepted-run A1/A7 counts, which remain
unavailable without a newly instrumented E2E profiler window; do not infer
them from timing, and do not run E2E under the current user constraint.

The user subsequently authorized a diagnostic A7 E2E rerun to collect real
routing counts while retaining MB2.  The clean-head dry run
on `805545b` proves A7:F1, ATP1/FEP4, batch 6, MB2 `3+3`, 28 attention plus
four FFN GPUs on eight trays, 168 exact 128K prompts, and the standard one
warmup plus 15 measured Nsight launches.  Job `6527459` is submitted to
`short` with `MINISGL_MEGAMOE_ROUTE_STATS=1` under
`fmha_megamoe_fp8_20260825/a7_route_stats_805545b/`.  Counter atomics make
this a routing diagnostic rather than a performance acceptance run.  It
completed `0:0` in 21:53.  Every FFN rank reports the expected 3,008 calls (94
layers times two MB lanes times one warmup plus 15 measured launches), and the
four ranks sum to exactly 672 received routes per call (`84` tokens times
top-k 8), validating the counters.  Per-rank `received routes / active experts
/ M-blocks` are DP0 `160.644 / 19.942 / 24.082`, DP1
`166.091 / 20.716 / 24.917`, DP2 `176.266 / 20.821 / 25.461`, and DP3
`169.000 / 20.554 / 24.900`.  Across EP4 the means are 20.508 active experts
and 24.840 expert M-blocks per rank/call.  Hot experts therefore create about
4.33 extra 16-row weight tiles beyond one tile per active expert; active count
alone underestimates traffic.  Applying the controlled 128-SM fit to these
real rank triples predicts `94.386/96.554/97.391/96.305 us`, consistent with
accepted representative-rank A7's 97.493 us but not by itself proof of the
A1-to-A7 cause.  DP0, corresponding to accepted global model rank 29, streams
468,732,427 modeled expert bytes per call; its uninstrumented 97.493-us timing
is 4.808 TB/s, 60.1 percent of the GB200 GPU's 8-TB/s theoretical bandwidth.
Reaching sub-25-ms E2E from accepted 29.358 ms through MegaMoE alone requires
roughly 23.2 us/call, or about 6.31 TB/s effective DP0 weight bandwidth if the
other fused work is unchanged.

The user then required the missing real A1 control before accepting the causal
comparison.  Exact clean-head A1 dry run on the same `805545b` proves two
trays/eight GPUs, four attention lanes plus EP4 FFN, ATP1, batch 6, MB2 `3+3`,
24 exact 128K prompts, and the identical warmup-plus-15 window.  Route-stat job
`6529097` completed `0:0` in 17:47 under
`fmha_megamoe_fp8_20260825/a1_route_stats_805545b/`.  Each rank again reports
3,008 calls and the four ranks sum to exactly 96 routes/call (`12` tokens times
top-k 8).  A1 per-rank `routes / active / M-blocks` are DP0
`23.904 / 9.114 / 9.114`, DP1 `22.491 / 8.672 / 8.672`, DP2
`24.792 / 9.223 / 9.223`, and DP3 `24.813 / 9.345 / 9.345`; no A1 expert
ever crosses the 16-token tile boundary.  EP4 means move from A1
`24.000 / 9.088 / 9.088` to A7 `168.000 / 20.508 / 24.840`, so average
algorithmic expert-weight bytes rise 176,895,728 -> 483,487,488 per rank/call,
2.733x.  The controlled fit predicts mean A1/A7 times of 59.993/96.159 us, a
36.166-us delta; accepted clean traces measure 61.371/97.493 us, a 36.122-us
delta.  At matched DP0, absolute effective weight bandwidth improves from
2.890 TB/s (36.1 percent of 8 TB/s) at A1 to 4.808 TB/s (60.1 percent) at A7.
DP0's extra 291,341,682 modeled bytes divided by the 36.122-us measured time
increase is 8.065 TB/s, essentially the GB200 roofline; the 0.8-percent excess
is within algorithmic-byte/L2-reuse and cross-run trace-model uncertainty.
This is the requested direct evidence: A7 is not less efficient than A1; its
latency increase is quantitatively explained by 2.26x more active experts and
2.73x more expert-weight tiles, including A7 hot-expert spill tiles.  Do not
use either instrumented job's timing and do not submit any further E2E case
without a new decision.

The MegaMoE adoption retrospective distinguishes integration work from the
core device implementation.  Initial adoption commit `858f797` did not merely
add a Python adapter: it added a narrow compile-time FP8-weight specialization
to the existing MegaMoE kernel.  The device-kernel delta selects E4M3 rather
than packed E2M1 weights and doubles the expected B-operand TMA bytes; it does
not change expert dispatch/scheduling, the fused expert pipeline, SwiGLU,
combine/reduction, synchronization, or task ordering.  The JIT/API dispatches
the FP8 specialization at compile time, so there is no per-tile precision
branch in the hot loop.  The larger FP8 weight traffic versus the original FP4
kernel is an intentional precision requirement for parity with the replaced
split AG/EG path, not adapter overhead.  The remainder of adoption is an
adapter/runtime layer: graph-stable per-MB-lane symmetric buffers, group-local
expert mapping, one-time checkpoint conversion to per-32 UE8M0 scales, direct
router output into those buffers, and backend/launch selection.  The four core
kernel/API files have no diff between adoption `858f797` and accepted default
head `24b90c0`; accepted commit `db9b3f8` changes only the DeepGEMM SM launch
policy.  Thus accepted MegaMoE performance is not confounded by a rewritten
algorithm, although required FP8 bytes and adapter/launch choices can affect
the complete stage.  Current diagnostic commit `c18b4b0` is intentionally not
performance-equivalent: enabled route counters add device atomics, so job
`6527459` supplies trustworthy routing counts but not acceptance latency.

The next first-principles kernel probe targets repeated weight reads from A7
hot experts.  Real A7 averages 20.508 active experts but 24.840 16-row expert
tiles per rank/call; a 32-row tile can collapse the 4.33 spill tiles while
leaving A1's nine-active/no-spill guard explicit.  Clean remote branch
`codex/fmha-megamoe-block-m32-20260825`, commit `6599b9d`, adds a fail-fast
diagnostic block-M override and a reusable `rows:active:spill` route generator.
Local and remote diff/shell checks plus all 20 focused contracts pass.
Four-GPU job `6535946` was submitted to `short` with a one-hour limit, exact
3,840-token A7 capacity, 150 compute SMs, a production-like resident CTA, and
ABBA block-M order `16/32/32/16` for A1-like `3:9:0` and A7-like `21:20:4`
routes.  Its root is `fmha_megamoe_fp8_20260825/block_m32_hot_spill_6599b9d/`.
Advance to an exact eight-tray A7 run only if this isolated probe shows a
substantial, repeatable MegaMoE reduction.

Jobs `6535946` and cache-reusing `6536117` completed `0:0` in 2:48 and 1:26.
For the A7-like resident-CTA route, repeated block-16 means are
100.277/100.040 us, block-32 means are 95.818/96.308 us, and the isolated
reduction is 4.459/3.731 us or 4.45/3.73 percent.  Block 32 reduces the exact
modeled tiles from 24 to 20.  A1-like nine-active/no-spill timing is flat to
slightly faster, so there is no low-fan-in guard regression.  The second sweep
also tested block 64 in ABCCBA order: its A7-like resident mean is 96.919 us,
0.611 us slower than block 32, while retaining the same 20 weight tiles.
Therefore block 32 is the isolated optimum, but its roughly 0.70--0.84-ms
per-replay projection (`188` calls) remains below the user-required
substantial-isolated-gain gate for an eight-tray E2E run.  Close this tile-size
family without changing the production heuristic; retain `6599b9d` only as a
diagnostic branch and pursue a larger fused-kernel communication/pipeline or
weight-throughput mechanism next.

Accepted A7 rank-29 Nsight trace analysis bounds the separate router-selection
opportunity.  CPU export job `6536348` failed in 30 seconds because the reusable
exporter re-entered its transient Slurm spool path inside the container; corrected
job `6536422` pinned `JOB_SCRIPT` to the stable `/lustre/.../control` path and
completed `0:0` in 30 seconds with a one-hour limit and zero stderr.  Its graph-only
3,008-call means are 4.614 us for `gate_topk_fused_tc_kernel`, 4.508 us for
`gate_topk_select_kernel`, 0.498 us between them, and 0.527 us from selection to
MegaMoE.  Perfectly eliminating selection plus both adjacent gaps therefore saves
at most 5.533 us/call or 1.040 ms per 188-call replay.  Combined with measured M32,
the optimistic projection is only about 1.74--1.88 ms versus the 2.94-ms soft
target.  Do not build the invasive router/MegaMoE fusion on that bound alone;
profile the dominant 97.493-us MegaMoE kernel for a larger weight-throughput or
pipeline mechanism first.

The next dispatch probe removes needless expert-count atomic fan-in for small
decode batches.  In A7 each CTA covers 16 input tokens, so only two of 150 CTAs
own any of the 21 rows, while the original path makes every CTA issue one atomic
per global expert.  Clean remote commit `7e030e0` adds a diagnostic active-CTA
mode: contributing CTAs still obtain disjoint source-index offsets, the grid sync
orders all writes, and SM0 synthesizes the original fixed `kNumSMs` high-word
completion target before remote publication.  Four-GPU `short` job `6536774`
used a one-hour limit, exact 3,840-token capacity, M32 A7-like `21:20:4` routes,
150 SMs, a 180,000-cycle resident CTA, and ABBA all/active/active/all order.  It
completed `0:0` in 2:46 with zero stderr and bitwise-identical output in every
trial.  Non-resident means improve from 93.623 to 92.554 us (1.069 us, 1.14
percent); resident means improve from 96.153 to 94.716 us (1.437 us, 1.50
percent).  Against the repeated M16 baseline, the combined M32 plus active-count
reduction is about 5.32--5.68 percent or 1.00--1.05 ms per replay.  Retain this
proven diagnostic path, but it is still below the substantial isolated-gain E2E
gate.  No exact eight-tray run is warranted yet.

L2 weight-prefetch probe `3c9e9bc` ports the existing M2N evict-last bulk hint
into the same-rank kernel as a byte-capped runtime diagnostic.  It scans finalized
receive counts, spends the cap only on the earliest active experts (matching
scheduler order), and distributes each selected expert slice across all SMs while
the dispatch warps enter the token-pull phase.  Four-GPU `short` job `6537140`
used one hour, active-count M32 `21:20:4`, resident and non-resident timings, and
mirrored caps `0/32/64/96/96/64/32/0` MiB.  It completed `0:0` in 2:31 with zero
stderr and bitwise-identical outputs.  The two 32-MiB non-resident trials average
91.628 us versus 91.640 us for the mirrored zero-cap trials, i.e. neutral within
order drift; 64 MiB averages 93.134 us and 96 MiB averages 96.453 us.  Resident
results have the same direction: about 93.055, 95.999, and 98.354 us for 32, 64,
and 96 MiB versus a noisy 93.386-us mirrored zero mean.  Close this prefetch family:
larger hints compete with the live weight stream and the smallest tested cap does
not provide a repeatable gain.  Leave production prefetch disabled.

Deferred cleanup-barrier probe `e0e1a86` ports the M2N split arrive/wait protocol
to same-rank MegaMoE: phase-0 is seeded for the first launch, tag-3 arrival remains
at cleanup, its wait moves to the next launch, and a 224-thread scheduler barrier
prevents TMA/MMA warps from reading stale counts.  Four-GPU `short` job `6537478`
used a one-hour limit and fresh buffers in mirrored normal/deferred/deferred/normal
process order.  It completed `0:0` in 5:07 with zero stderr and passed all numeric,
route, and finite-output checks.  The mechanism is negative here: A7-like active-
count M32 increases from a 93.013-us normal mean to 94.934 us (+2.07 percent), the
resident mean increases from 95.498 to 97.109 us (+1.69 percent), and random-route
fused router-plus-MegaMoE increases from 130.936 to 132.311 us (+1.05 percent).
The scheduler-release cost is visible while the original final wait was already
hidden under combine/kernel completion.  Reject the split-barrier path and leave
production cleanup synchronization unchanged.

Direct low-token count publication probe `4229df8` removes the remaining global
offset atomics: CTA 0 scans all 21-token routes, writes finalized expert counts
with the original completion high word, resets its shared offsets, and pushes the
168 source indices.  Four-GPU `short` job `6537762` used a one-hour limit, active/
direct/direct/active order, active-count M32 `21:20:4`, and resident plus non-
resident timing.  It completed `0:0` in 2:34 with zero stderr and bitwise-identical
outputs.  The non-resident active/direct means are 92.748/92.648 us (0.100 us or
0.11 percent); resident means are 94.887/94.842 us (0.045 us).  This is neutral,
so retain the simpler two-active-CTA atomic-offset path and close count publication
as a material bottleneck.

Deferred router-selection probe `db54b02` factors the existing exact softmax/top-k
routine into a shared device helper, leaves the gate tensor-core kernel producing
logits and FP8 activations, and lets otherwise-idle MegaMoE initialization warps
select routes before a grid publication barrier.  The path is opt-in and keeps the
normal two-kernel router unchanged by default.  Four-GPU `short` job `6538235`
used a one-hour limit, 150 compute SMs, M32 plus active-count mode, 21 rows, and
mirrored normal/deferred/deferred/normal trials with and without a 180,000-cycle
resident CTA.  It completed `0:0` in 2:26 with zero stderr; route IDs, weights,
and final outputs were bitwise identical across all four trials.  The non-resident
normal/deferred means are 131.534/131.012 us, only a 0.522-us or 0.40-percent
reduction.  Resident means are 133.209/133.002 us, only 0.206 us or 0.15 percent.
Thus the in-Mega grid-publication and scheduling cost consumes almost all of the
5.533-us trace upper bound from removing the standalone selection launch and its
adjacent gaps.  Reject this fusion, leave it disabled, and do not run exact E2E.

Expert-weight streaming probe `f231f4c` adds an opt-in TMA `EVICT_FIRST` policy
for the FP8 matrix and UE8M0 scale tiles.  With M32 every active expert has one
M tile, and each expert weight cache line is consumed once per layer call, so
retaining those lines can only displace reusable activations and metadata.
Four-GPU `short` job `6538592` used a one-hour limit and fresh-process ABBA order
normal/evict-first/evict-first/normal with M32, active-count mode, 21 rows, and
the A7-like `20`-active/`4`-spill route.  It completed `0:0` in 5:22 with zero
stderr and all four numeric/correctness checks reporting `status: ok`.  A7-like
non-resident normal/evict-first means are 92.445/88.568 us, a 3.877-us or
4.19-percent reduction.  With a 180,000-cycle resident CTA they are
94.369/89.588 us, a 4.781-us or 5.07-percent reduction.  Random-route full-stage
means improve 129.299 -> 123.838 us (4.22 percent), and resident full-stage means
improve 131.481 -> 125.160 us (4.81 percent).  Retain this general cache-policy
gain.  Relative to the earlier repeated resident M16 mean of 100.159 us, the
cumulative M32 plus active-count plus evict-first result is 10.55 percent faster,
projecting about 1.99 ms per 188-call replay; this remains below the 2.94-ms soft
target.  Test the complementary reusable-activation cache policy and an A1 guard
before changing defaults or spending an exact E2E run.

Reusable-activation retention probe `01c1927` holds weight `EVICT_FIRST` on and
adds opt-in TMA `EVICT_LAST` for activation and activation-scale tiles, which are
requested repeatedly across an expert's N blocks.  Four-GPU `short` job `6538935`
used a one-hour limit, fresh-process ABBA normal/evict-last/evict-last/normal,
and both A1-like `3:9:0` and A7-like `21:20:4` routes.  It completed `0:0` in
4:57 with zero stderr and all four records reporting `status: ok`.  A7-like
non-resident means improve only 88.056 -> 87.544 us (0.512 us, 0.58 percent),
and resident means improve only 88.882 -> 88.682 us (0.201 us, 0.23 percent).
A1-like non-resident means improve 55.911 -> 55.484 us (0.76 percent), while
resident means improve only 56.754 -> 56.589 us (0.29 percent).  This is too
small to justify preferentially retaining activation lines across other shapes;
reject activation `EVICT_LAST` and leave its default off.  Promote only M32 for
decode rows exceeding one M16 tile, active-CTA count fan-in, and expert-weight
`EVICT_FIRST`, then run a mirrored A1/A7 production-default guard before exact
E2E.

Production-policy commit `04fb32b` promotes only the validated general pieces:
the block heuristic keeps M16 for at most 16 decode rows and selects M32 when a
larger decode batch would otherwise remain in the old low-density branch;
active-CTA count fan-in and weight `EVICT_FIRST` become defaults with explicit
diagnostic overrides.  Activation retention and deferred router selection stay
off.  Four-GPU `short` guard job `6539274` used a one-hour limit, automatic
M16(A1)/M32(A7), active counting, and fresh-process weight-policy ABBA across
A1-like `3:9:0` and A7-like `21:20:4` routes.  It completed `0:0` in 5:38 with
zero stderr and four `status: ok` records.  A1-like non-resident normal/evict
means are 58.539/56.317 us (3.80 percent faster), and resident means are
60.382/57.321 us (5.07 percent faster), so the alignment guard does not regress.
A7-like non-resident means are 92.038/88.067 us (4.31 percent faster), and
resident means are 94.073/89.849 us (4.49 percent faster).  Against the original
repeated M16 resident mean of 100.159 us, the complete promoted stack is about
10.29 percent faster and projects 1.94 ms per 188-call replay.  This clears the
user's substantial-isolated-speedup promotion gate, although it remains below
the 2.94-ms soft projection; proceed to one exact A7 performance run under the
unchanged alignment and warmup-plus-15 metric contract.

Exact promotion gate job `6539640` was submitted on 2026-08-26 to Slurm
`batch`/`short` with eight nodes, 32 GPUs, and an explicit one-hour limit from
clean remote commit `04fb32b`.  It is the accepted A7:F1, ATP1/FEP4, 128K ISL,
batch-6 FMHA-only MegaMoE case with one warmup plus 15 measured replays; its
fresh result root is
`fmha_megamoe_fp8_20260825/a7_f1_ep4_128k_b6_optimized_04fb32b`.  Dry validation
passed after creating the declared workspace-local CUDA extraction temp
directory.  It completed `0:0` in 21:57 after five-minute startup checks and a
15-minute run-to-completion interval.  Exact 168-by-131072 prompt
materialization and the batch-6 capacity gate passed.  The finalized dual-role
strict CUDA metric retained all 15 post-warmup samples with zero outliers:
28.1574149333 ms mean, 28.110402 ms median, 3.641-percent dominant range, and
186.451775 TPS per active GPU.  This improves the accepted 29.3581909333-ms A7
result by 1.200776 ms or 4.090 percent, but misses the 26.42237184-ms ten-percent
soft target by 1.735043 ms.  Attention rank 1 supplies the maximum critical span
for every measured step (28.005312--29.028832 ms), while model rank 29 is already
slightly faster (27.946046--28.110240 ms).  The promoted FFN work therefore
moves exact A7 into an attention-limited regime; further FFN-only latency work
cannot deliver the remaining E2E target under this placement.  The unchanged
accepted alignment evidence remains the correctness gate; this performance run
did not request a redundant alignment pass.

Post-promotion trace decomposition uses the exact job's model-rank-29 report.
The first CPU-export shell attempt expanded remote path variables locally and
was rejected by `sbatch` before creating a job or allocation.  Explicit-path
replacement `6540458` ran on `cpu-short` with a one-hour cap, completed `0:0`
in 44 seconds, and produced an integrity-clean SQLite database under the exact
case's `trace_sqlite/` directory; a copy is retained locally under
`scratch/qwen3_ffn_overheads_20260820/optimized_a7_04fb32b/trace_sqlite/`.
Across the 15 measured graphs, rank 29 averages 28.048 ms of model kernel span:
17.244 ms of serialized MegaMoE, 10.488 ms of other productive kernel time,
and only 0.891 ms during which no productive kernel runs.  MegaMoE never
overlaps another MegaMoE call; every call overlaps only the peer lane's
one-CTA O-ready waiter.  The two model streams are sharply asymmetric at
79.080 and 104.641 us per MegaMoE call.  Relative to the accepted pre-promotion
trace, model span improves 29.248 -> 28.048 ms and MegaMoE contributes 1.031 ms
of that 1.200-ms reduction.  Therefore launch-gap cleanup is bounded below the
remaining kernel opportunity, and the slower lane needs route/pipeline evidence.

Current-policy launch sweep `6540641` ran on four GPUs in `batch`/`short` with
a one-hour limit and completed `0:0` in 2:58 with empty stderr and exact
numerical parity.  It swept 96/112/128/144/150 SMs under M32, active-count
fan-in, weight `EVICT_FIRST`, and a 180,000-cycle resident waiter.  On the
A7-like `21:20:4` hot-spill route, resident means are 90.725/91.546/89.025/
90.154/90.586 us, making 128 SMs 1.72 percent faster than production 150 SMs.
On the denser random route, 128 SMs plus one epilogue warpgroup improves the
resident fused stage from 128.071 us at 150 SMs/two warpgroups to 125.237 us,
or 2.21 percent.  This directional launch-policy gain is not material enough
to promote alone.  Diagnostic commit `ee8aecc` keeps production defaults
unchanged, adds a fail-fast pipeline-depth override, and makes the reusable
smoke preserve a requested epilogue/wave configuration for controlled routes;
local and remote focused contracts pass 20/20.  Job `6540830` is the pending
ABBA depth sweep at 128 SMs/one epilogue warpgroup.

Pipeline-depth job `6540830` started immediately in `batch`/`short` with four
GPUs and an explicit one-hour cap.  Auto, 8, 6, and 4 stages all passed exact
reference equality on their valid trials.  A7-like `21:20:4` resident means
were 88.463/88.597/90.687/97.347 us respectively, so shallower 6- and 4-stage
pipelines regress 2.51 and 10.04 percent; 8 stages is effectively tied with the
deepest-fit production policy.  The requested 2-stage diagnostic reached a
compile-time `Hidden is too large` shared-memory assertion, terminating the
job `FAILED 143:0` after 6:32 before the mirrored controls.  This exposed an
over-permissive diagnostic input check rather than a production failure.  The
override and reusable smoke now reject forced depths below three before JIT;
the retained validation rerun needs only mirrored 8/auto controls.

Mirror control job `6541108` used corrected explicit output paths and completed
`0:0` in 3:10 on `batch`/`short` with a one-hour cap.  Stage 8 versus auto is
86.875 versus 86.785 us on the A7-skew nonresident case and 88.037 versus
87.884 us on the fixed resident route.  One 100.796-us auto skew-resident
sample is inconsistent with its normal paired routes and treated as injected
contention, not a depth signal.  Reduced depth is rejected and production
retains the deepest fitting pipeline.  An earlier corrected-job submission,
`6541098`, was cancelled while pending at 0:00 because its shell-expanded
output root resolved incorrectly; it consumed no allocation.  The next HBM
candidate splits the already useful weight `EVICT_FIRST` policy by L1/L2 phase
to test all four cache masks without changing the production default.

Phase-mask job `6541256` ran from clean diagnostic commit `940862f` on four
GPUs in `batch`/`short` with a one-hour limit and completed `0:0` in 5:39 with
empty stderr and exact reference equality.  On the A7-skew resident route,
masks 0/1/2/3 measure 94.931/94.000/94.214/90.151 us; the existing both-phase
evict-first mask is 4.10 percent faster than the best partial mask.  The fixed
resident route agrees at 94.625/93.464/93.774/90.077 us.  Reject phase-specific
retention and remove its machinery.  Advance the route-robust launch signal
instead: the M32 decode configuration selects one epilogue warpgroup and caps
only its MegaMoE grid at 128 SMs, leaving M16 and dense DeepGEMM launch policy
unchanged and retaining an explicit fail-fast MegaMoE SM diagnostic override.

Launch-policy validation is retained.  Initial GPU guard `6541504` failed
before kernel execution on `nvl72047-T18`, whose container lacked
`aarch64-linux-gnu-g++`; retry `6541592` excluded that node but still failed
during a four-rank forced host-extension rebuild.  A first inline CPU prebuild
submission was rejected by shell quoting before `sbatch` and created no job.
Reusable `prebuild_deepgemm_extension.sbatch` moves this CPU-only step to
`cpu-short`.  Its first job, `6541737`, isolated a real host compile error: the
new override validated against nonexistent `MegaMoEConfig::cluster_size`.
Commit `51d3a8c` uses the kernel's single `kClusterSize=2` invariant for both
validation and launch.  CPU prebuild `6541887` then completed `0:0` in 1:48
and populated a fresh workspace cache without consuming GPUs.

Final four-GPU guard `6541968` ran from the corrected clean source on
`batch`/`short` with an explicit one-hour cap and the compiler-broken node
excluded.  It completed `0:0` in 1:29 with empty stderr, `status: ok`, and
exact configuration/reference equality.  A1-like `3:9:0` nonresident/resident
means are 56.216/57.282 us versus the prior production guard's 56.317/57.321
us, confirming no M16 regression.  A7-like `21:20:4` means improve from
88.067/89.849 to 85.876/87.340 us, or 2.49/2.79 percent.  The fixed 20-active
resident route is 87.126 us.  Retain M32 one-warpgroup epilogue plus the
MegaMoE-only 128-SM cap.  Do not spend a 32-GPU exact rerun solely on this
isolated gain: accepted exact job `6539640` is already attention-rank-limited
on every replay, so the launch improvement cannot move its E2E critical path.

The next inter-kernel audit found a larger model-role opportunity: the exact
trace's two MegaMoE lanes never overlap because each M32 launch occupies 128
SMs.  Concurrent launch is safe because every production lane owns a distinct
`MegaMoESymmBuffer`, including independent grid counters and NVLink barrier
phase words.  Clean diagnostic commit `fb75ac0` added a mirrored paired-stream
probe with two isolated symmetric buffers, identical direct-launch overhead,
bitwise output checks, and an explicit per-launch SM sweep.  Four-GPU
`batch`/`short` job `6542133` completed `0:0` in 1:37 with empty stderr and
exact equality.  On A7-skew `21:20:4`, paired sequential/concurrent spans are
171.754/162.570 us at 128 SMs, 175.580/169.147 us at 96, 187.582/188.743 us at
80, 192.952/134.807 us at 72, and 202.826/110.466 us at 64.  The mirrored
routes agree: 172.964/163.231, 176.089/168.835, 188.055/188.507,
192.452/121.512, and 202.284/128.133 us.  Sixty-four SMs supplies the largest
robust overlap window, 1.58--1.84x, while retaining scheduling headroom.

Commit `a50a627` promotes this as a general per-call concurrency contract.
`fp8_fp8_mega_moe` accepts a positive `num_concurrent_lanes`; only the M32
low-token path selects the largest power-of-two share within
`min(128, runtime_sms / num_concurrent_lanes)`.  Thus a single lane keeps the
validated 128-SM path, while the production two-lane adapter derives 64 SMs;
M16 and dense DeepGEMM policies remain unchanged, and
`DG_FORCE_MEGAMOE_NUM_SMS` remains a diagnostic override.  CPU-only prebuild
job `6542270` ran on `cpu`/`cpu-short`, completed `0:0` in 4:43, and populated
the fresh extension cache without using GPUs.  Production-policy guard
`6542458` then ran on four GPUs in `batch`/`short` with a one-hour cap and
completed `0:0` in 1:28 with empty stderr, `status: ok`, and exact equality.
The derived two-lane policy measures 201.212 us sequential versus 127.878 us
concurrent, a 1.573x speedup and 36.45-percent paired-span reduction; mirrored
explicit-64 controls reach 1.638--1.674x.  The ordinary one-lane M32 route
remains about 86.048 us, proving that the accepted single-lane launch is
unchanged.  This clears the exact A7 observation gate.  The older trace rule
against any overlapping heavy model kernels is superseded only for these two
causally independent MegaMoE lanes with distinct buffers; overlap with a
dependent batch or shared workspace remains forbidden.

Exact transfer job `6542626` ran from clean documented head `5026f76` on eight
nodes / 32 GPUs in `batch`/`short` with an explicit one-hour cap and completed
`0:0` in 21:51.  Exact 168-by-131072 prompts, the batch-6 capacity ceiling,
FMHA-only placement, one warmup plus 15 measured replays, and all metric gates
passed.  The result is a decisive regression: 31.857557 ms strict CUDA mean,
31.807168 ms median, 2.892-percent dominant range, zero outliers, and
164.796064 TPS per active GPU, versus accepted `04fb32b` at 28.157415 ms.
Attention rank 1 remains critical on every measured step, but model rank 29 is
nearly tied at a 31.738-ms mean span.  CPU trace-export job `6543419` completed
`0:0` in 2:39 on `cpu-short` and proved why the isolated probe does not
transfer: none of the 16 captured graphs overlaps one MegaMoE call with
another.  The graph's lane dependencies remain serialized, so 64-SM launches
only increase measured MegaMoE work from 17.244 to 21.144 ms per replay while
other model work slightly improves from 10.804 to 10.595 ms.  Per-call stream
means regress from 79.080/104.641 to 103.908/121.207 us, accounting for the
3.690-ms model-span increase.  Reject production lane sharing and restore the
adapter's default one-lane/128-SM policy.  Retain the explicit concurrent-lane
API and paired-stream probe only as diagnostics for a future graph-scheduling
change that first demonstrates real captured overlap; the exact accepted
checkpoint remains `04fb32b` plus the later single-lane launch improvements.

Captured-graph dependency reconstruction closes forced lane pairing before a
new GPU experiment.  In the accepted rank-29 trace, lane 1's O-ready wait ends
about 93 us after lane 0 MegaMoE starts and its route selection completes about
184 us after that origin; removing the explicit prior-compute event would move
lane 1 MegaMoE to roughly 118 us, leaving only about 9 us of natural MegaMoE
overlap.  Holding lane 0 to manufacture a larger pair would delay its next QKV
wavefront and repeats the already-rejected merged-MB scheduling mechanism.
Leave the graph dependency unchanged.

External profiler collection is unavailable in the current OCI-HSG runtime.
Bounded capture commit `ed69fba` wraps only measured skew-graph replays with
the CUDA profiler API and adds a fail-fast PIC-C wrapper.  Four-GPU
`batch`/`short` job `6544129` failed in 20 seconds before target work because
`pic-c` is absent on the allocated node.  Commit `46b6e28` added an equally
bounded Nsight Compute fallback; job `6544501` failed in 66 seconds with exit
127 before Python because `/usr/local/cuda/bin/ncu` is absent from the image.
CPU-short inventory job `6544786` found no alternate `pic-c`, `ncu`, or `nsys`
binary under the image's `/opt`, `/usr/local`, or `/usr`.  Do not resubmit
either unchanged.  Clean diagnostic commit `28e8a5e` instead ports the
existing MegaMoE `clock64` technique into the same-rank kernel as an opt-in
17-slot per-rank phase report; the null production specialization compiles the
timing path away.  It covers dispatch/pull/cleanup, TMA-A arrivals and empty
waits, TMA-B empty waits, MMA full/TMEM waits, epilogue/TMEM/writeback, tag-2,
and combine.  CPU prebuild `6545188` completed `0:0` in 6:23 and populated a
fresh workspace cache without using GPUs.  Four-GPU A7-skew phase job
`6545489` is submitted to `batch`/`short` with a one-hour cap from clean commit
`28e8a5e`.  It completed `0:0` in 1:42 with exact equality and an 87.526-us
A7-skew kernel.  Rank 0's plausible core spans are roughly 132--135 thousand
cycles, but ranks 1--3 charged 22--70 million cycles of unsynchronized first-
specialization JIT skew to their communication waits.  Those ranks are not a
valid phase profile.  Commit `b986e44` adds one untimed instrumented warmup,
then zeroes counters behind a cross-rank barrier; cache-reusing replacement
job `6545619` is the authoritative phase-profile run.

Authoritative synchronized job `6545619` completed `0:0` in 1:15 with empty
stderr, exact equality, and an 89.091-us A7-skew mean.  Ranks 0/1/3 form the
stable timing group; rank 2 launched early and accumulated about 67 thousand
extra cross-rank startup cycles.  Across the stable ranks, the parallel
TMA-A/TMA-B/MMA/epilogue role loops span roughly 129--154 thousand cycles.
TMA-B spends 80.6--82.2 thousand cycles waiting for empty stages, MMA spends
63.9--65.7 thousand on full-stage waits, and epilogue spends 75.8--80.4
thousand waiting for TMEM while MMA's TMEM-empty wait is below 700 cycles.
Thus epilogue is not the limiter and the deep pipeline is mutually
backpressured rather than a simple insufficient-stage problem.  TMA-A still
spends 25.3--29.6 thousand cycles waiting for L1 token arrivals, while the
dispatch pre-pull phase costs 15.0--38.1 thousand cycles.  Two-stage token
pulling cannot help this exact shape because 168 received routes already fit
in the kernel's 512 pull warps, leaving no second token per warp.

Diagnostic commit `46814e8` instead ports the maintained M2N count-gated
startup protocol: the specialization skips the tag-1 full-rank barrier, waits
for all per-expert count completions, then performs a system acquire before
reading remote source indices and tokens.  CPU-short prebuild `6545878`
completed `0:0` in 6:02.  Fresh-process barrier/count/count/barrier A7 guard
`6546069` completed `0:0` in 3:59 with identical numeric checks and improved
the exact skew path from 86.929 to 85.928 us nonresident (1.15%) and from
88.801 to 87.834 us resident (1.09%).  Mirrored A1/A7 guard `6546266`
completed `0:0` in 4:07: A1 exact skew improved from 57.993 to 56.736 us
nonresident (2.17%) and from 59.732 to 56.853 us resident (4.82%); A7 improved
from 87.143 to 86.711 us (0.50%) and from 89.319 to 87.818 us (1.68%).  Fixed-
route controls improved by 1.43--4.72%, with no tested regression.  Promotion
commit `bb948d8` makes count-gated pull the production default and preserves
`DG_FORCE_MEGAMOE_COUNT_GATED_PULL=0` as a diagnostic rollback.  Focused
contract, sbatch syntax, and diff checks pass locally and in the clean remote
worktree.  Do not spend the eight-tray exact replay on this isolated gain: the
accepted trace already has attention rank 1 controlling every replay, so a
0.50--1.68% MegaMoE reduction cannot materially move coordinator E2E latency.

The next same-rank experiments closed the remaining expert-wave and combine-
readiness branches.  Commit `8364c06` adds a reusable expert-wave ABBA control;
four-GPU `batch`/`short` one-hour job `6546556` completed `0:0` in 6:19 with
exact equality.  Forcing 8 rather than the default 16 experts per wave
regressed fixed A7 by 4.93% nonresident and 4.60% resident, skew A7 by 3.75%
and 4.55%, and random routing by 2.77% and 2.48%; keep 16.  The first token-
gated combine probe (`aff1c98`) published one system atomic per returned route.
CPU-short prebuild `6546836` completed `0:0`, and four-GPU job `6546871`
remained exact but regressed the measured paths by roughly 27--91%; do not
retry per-route atomics unchanged.  Replacement commit `3e32449` aggregates
completion into one monotonically increasing epoch per producer rank.  Each
token derives the rank subset owning its selected experts and waits only on
those epochs, eliminating the tag-2 full-rank barrier with four release stores
per rank rather than one atomic per route.  CPU-short prebuild `6547210`
completed `0:0` in 5:11.  Fresh-process barrier/epoch/epoch/barrier job
`6547389` completed `0:0` on four GPUs in 4:30 with empty stderr and exact
equality throughout.  ABBA means improved A1 fixed by 3.50% nonresident and
1.92% resident, A1 skew by 1.79% and 2.15%, A7 fixed by 2.82% and 1.74%, and
A7 skew from 85.298 to 83.475 us (2.14%) and from 87.502 to 85.690 us (2.07%).
Random routing was neutral-to-positive at 0.17% and 0.87%.  Promotion commit
`bd363cb` makes rank-epoch combine readiness the default and retains
`DG_FORCE_MEGAMOE_TOKEN_GATED_COMBINE=0` as the full tag-2 barrier rollback.
The focused 20-test overlap contract, sbatch syntax, and diff checks pass
locally and in the clean remote worktree.  As with count-gated pull, do not run
the expensive eight-tray exact replay solely for this isolated improvement:
the accepted production trace is attention-critical on every replay.

Dispatch-pull spatial-distribution probe `d4acf15` kept the required
128-thread dispatch warpgroup but limited token pulling to 1/2/4 warps per
CTA.  A7's 168 routes therefore occupied 128/84/42 SMs respectively; two
warps retained full route concurrency while testing twice the production SM
spread.  CPU-short prebuild `6547699` completed `0:0` in 2:39.  Four-GPU
`batch`/`short` one-hour ABCCBA job `6547892` completed `0:0` in 6:17 with
exact equality in all six fresh processes.  Two versus four pull warps was
mixed: A7 skew improved 0.72% nonresident but regressed 0.21% with the
production-like resident waiter; fixed A7 regressed 0.55% nonresident and
improved only 0.16% resident; random fused routing improved 0.67% but regressed
0.08% resident.  A1 fixed regressed 0.26%/0.90%, while A1 skew improved
0.41%/0.81%.  One pull warp likewise mixed small nonresident gains with A7
skew resident regression of 1.16%.  Pull spatial distribution is not a robust
production opportunity; revert `e7a1f92` restores the simpler four-warp path.

K-stage aggregation diagnostic `4d13c57` represents a K256 pipeline stage as
two independent K128 TMA/UMMA atoms, loads both scale tiles, and rebases the
two-bit MMA scale IDs for the second atom.  It also generalizes the L1-to-L2
arrival mask and retains a fail-fast 128/256 override.  CPU-short prebuild
`6548361` completed `0:0` in 2:19.  Four-GPU `batch`/`short` one-hour ABBA job
`6548501` completed `0:0` in 4:21 with four successful processes, empty
stderr, identical numeric results, and zero mismatches across 14 common output
fingerprints.  K256 versus K128 improved A7 fixed by 0.85% nonresident and
1.56% resident, A7 skew by 0.74% and 1.41%, and A7 random fused routing by
0.99% and 0.95%.  A1 M16 fixed/skew paths regressed 0.45--0.97%, although A1
random routing improved 1.21--1.30%.  Promotion `852f4ce` therefore selects
K256 only for the measured M32 bucket when both K dimensions tile by 256;
other M buckets retain K128.  The explicit 128/256 override and reusable ABBA
smoke remain available for rollback and future shapes.

Grouped-scale TMA probe `492372e` encoded the two adjacent K128 scale columns
of each K256 stage as one unswizzled 2D TMA box, halving scale-load
instructions without changing bytes or layout.  CPU-short prebuild `6548821`
completed `0:0` in 2:14.  Four-GPU `batch`/`short` ABBA job `6549005`
completed `0:0` in 4:26 with four successful processes, empty stderr,
identical numeric results, and zero mismatches across 14 output fingerprints.
The effect was mixed and sub-percent: A7 skew improved 0.66% nonresident and
0.46% resident, but fixed A7 regressed 0.33% nonresident and random fused A7
regressed 0.28% nonresident; resident fixed/random gains were only 0.33% and
0.27%.  A1 controls were also mixed.  Scale instruction issue is not a robust
bottleneck after K256 aggregation; revert `35cdaef` removes the descriptor,
kernel, and smoke specialization completely.

Shared expert-count cache probe `014eecf` used the otherwise-idle fourth
non-epilogue warp to poll the 32 finalized local expert counts once, stage
them in existing shared memory, and release dispatch, TMA-A, TMA-B, MMA, and
epilogue roles through one full-CTA named barrier.  Scheduler iteration was
compile-time specialized so consumers skipped their redundant global polling;
the diagnostic also rejected deferred cleanup because a prior count generation
can remain visible.  CPU-short prebuild `6549392` completed `0:0` in 3:13.
Four-GPU `batch`/`short` one-hour ABBA job `6549610` ran on `nvl72151-T09`.
Its mode-0 baseline step completed `0:0` in 1:15 with valid numeric and exact-
shape output fingerprints, but the first cache-enabled step remained running
for more than 24 minutes after all four Gloo ranks connected, produced no JSON
record, and kept stderr empty.  The allocation was cancelled after 26:26 to
avoid wasting four GPUs.  This is a synchronization hang, not performance
evidence; the precise blocking role was not instrumented.  Revert `e75bc99`
removes the kernel, JIT, scheduler, and smoke specialization completely.  Do
not retry the same full-CTA barrier design without a bounded per-role progress
diagnostic or a non-blocking publication protocol.  Durable logs remain under
`fmha_megamoe_fp8_20260825/cache_expert_counts_014eecf/logs/`.

Post-K256 pipeline-depth control confirms that aggregation did not make the
deepest-fit policy obsolete.  For M32/K256, each stage uses 38,928 shared-memory
bytes and fixed state uses 21,892 bytes, so five stages is the exact maximum in
232,448 bytes.  CPU-short prebuild `6550424` completed `0:0` in 2:22.  Four-GPU
`batch`/`short` one-hour job `6550570` ran mirrored auto(5)/4/3/3/4/auto(5)
processes on `nvl72155-T05` and completed `0:0` in 5:36 with six successful
records, empty stderr, and zero fingerprint mismatches across all five common
A7 result keys.  Four stages regressed fixed A7 by 1.65% nonresident and 1.31%
resident, skew A7 by 1.44% and 1.24%, random fused routing by 1.17% in both
conditions, and random MegaMoE alone by 1.09%.  Three stages regressed the same
paths by 4.55--5.55%.  Retain five stages and the general deepest-fit heuristic;
do not trade pipeline latency hiding for fewer barriers after K256 aggregation.
Durable logs remain under `fmha_megamoe_fp8_20260825/k256_stage_sweep_ca43b5a/`.

Fresh synchronized phase profile `6550863` reused the production cache and
completed `0:0` in 1:17 with exact output equality and an 83.394-us A7-skew
kernel.  Across ranks, TMA-A/TMA-B/MMA/epilogue role loops are tightly aligned
at roughly 129--184 thousand cycles.  The remaining waits are mutually
backpressured: TMA-B empty 89--96 thousand cycles, MMA full 76--84 thousand,
epilogue TMEM 80--88 thousand, TMA-A empty 56--62 thousand, and TMA-A L1/L2
arrival 22--24/10--13 thousand.  This confirms that reducing the number of
complete L1-to-L2 wave transitions is the final bounded scheduling direction,
not another pipeline-depth or global-barrier change.

Final expert-wave ABBA job `6551125` ran default-16/32/32/default-16 from clean
commit `5701d95` on four GPUs in `batch`/`short` with a one-hour cap.  It
completed `0:0` in 4:01 with four successful records, empty stderr, and zero
output-fingerprint mismatches.  One 32-expert wave improves the target fixed
A7 path from 82.440 to 80.477 us (2.38%) and resident from 84.230 to 82.094 us
(2.54%); skew A7 improves from 82.882 to 80.446 us (2.94%) and resident from
84.584 to 82.324 us (2.67%).  Fully random routing is nearly neutral: fused
nonresident regresses 0.09%, fused resident 0.99%, and MegaMoE-only 0.65%.
Promote one full-rank wave only when the occupancy heuristic would otherwise
create exactly two M32 waves and the local expert group has at most 32 experts.
M16 A1 and larger expert groups retain their prior schedules.  Preserve the
explicit expert-wave override as a reusable diagnostic.  Durable logs remain
under `fmha_megamoe_fp8_20260825/k256_stage_sweep_ca43b5a/logs/wave-6551125.*`.

Final cleanup and A1 acceptance completed on 2026-08-26 PDT.  Remote commit
`d83dca9` promotes the bounded one-wave M32 rule and its source contract;
`b6801ea` makes the exact launcher honor an explicit CPU-prebuilt DeepGEMM
cache.  CPU-only `cpu`/`cpu-short` job `6551398` built that cache in 2:36 with
exit `0:0`; its only stderr was PyTorch's expected no-CUDA-runtime warning for
host compilation.  Local and clean-remote focused contracts pass 20/20, both
Slurm launchers pass `bash -n`, and `git diff --check` passes.

Two-tray A1:F1 exact job `6551590` used `batch`/`short`, a one-hour cap, clean
head `b6801ea`, 128K/b6/FEP4/ATP1/MB2, FMHA-only same-rank MegaMoE, and the
prebuilt cache.  All intended model work completed: 24 samples x 17 generated
tokens, capture warmup step 1537 plus measured steps 1538--1552, eight Nsight
reports, and a complete `afd-result.json`.  Slurm returned `1:0` only after
those artifacts because the reproducible launcher required its pre-EXIT GPU
snapshot to be empty even though Ray teardown occurs in the EXIT trap; the
snapshots contained the expected still-live workers.  The maintained
comprehensive launcher already treats this snapshot as cleanup diagnostics.
Commit `2075450` mirrors that policy, requiring a valid snapshot while leaving
the next allocation's initial snapshot as the fail-fast cleanup proof.  This
is a teardown-validator fix, not a model or kernel change, so the completed
sample was not regenerated.

Official unchanged-scorer A1 alignment job `6552212` then ran on one tray in
`batch`/`short` with a one-hour cap and completed `0:0` in 11:16.  It scored
all 24 prompts and all 408 candidate tokens: top-1, top-10, and top-100 are
each 1.0; average and maximum vLLM rank are both 1.0; reference perplexity is
1.010014.  No result has a non-top-1 position.  Candidate text is English and
locally readable, but the fixed 17-token completions often end abruptly and
some include word-splice artifacts from the synthetic cut/repeat 128K prompt
construction (for example `flutteredother` and `richthose`).  Exact vLLM
top-1 agreement on every token proves these artifacts are reference behavior,
not FastAFD corruption.  Durable artifacts are under
`fmha_megamoe_fp8_20260825/final_d83dca9_a1_alignment/`.

The user defined A1 alignment as the terminal gate after the final FFN round,
and it passes.  Close the optimization goal with the clean best-FFN state:
the last exact A7 strict result remains 28.1574149333 ms versus 29.3581909333
ms baseline (4.09% faster and attention-critical on every replay); the final
isolated M32 wave guard reaches 80.446--80.477 us on target skew/fixed routes,
2.38--2.94% faster than the prior 16-expert wave.  No post-wave exact A7 E2E
claim was made, and no further GPU experiment is queued.

The user subsequently requested the missing post-wave exact A7 observation.
The complete job-`6539640` result was copied locally, including all 32 raw
Nsight reports (ranks 1--32, 32,123,830 bytes), under
`scratch/qwen3_ffn_overheads_20260820/best_a7_job6539640_04fb32b/`.  That
accepted measurement used clean commit `04fb32b`: it already contained M32,
active-CTA count fan-in, and weight `EVICT_FIRST`, but predates the retained
128-SM/one-epilogue-warpgroup launch policy, count-gated token pulls,
rank-epoch combine gating, K256 staging, and final single-wave M32 policy.

Current-code exact A7 job `6553739` used clean head `202694a`, the same
A7:F1/ATP1/FEP4/FMHA-only/128K/b6 placement and prompt contract, one warmup
plus 15 measured replays, eight nodes/32 GPUs on `batch`/`short`, and a
one-hour cap.  It completed `0:0` in 21:27 with all 32 reports, the success
marker, 15/15 retained samples, zero outliers, and a 3.3958-percent dominant
range.  Strict CUDA mean/median are 27.6082989333/27.532512 ms and throughput
is 190.160213 tokens/s per active GPU.  Versus job `6539640`, mean improves by
0.549116 ms (1.950 percent), median by 0.577890 ms (2.056 percent), and
throughput by 1.989 percent.  Mean model span improves 28.0479202 ->
27.4905426 ms (0.557378 ms, 1.987 percent), while attention remains critical
on all 15 steps at a 27.6082989-ms mean.  Because attention span itself also
shifted by 0.549116 ms, do not attribute the exact E2E delta solely to FFN;
the isolated guards remain the causal FFN evidence.  Compact local result
metadata is under
`scratch/qwen3_ffn_overheads_20260820/latest_a7_job6553739_202694a/`; all 32
raw Nsight reports were subsequently copied into its `nsys/` subdirectory
(ranks 1--32, 33,603,369 bytes).
The three A7 comparison trace sets were also consolidated as physical local
copies under
`scratch/qwen3_ffn_overheads_20260820/a7_ffn_three_case_nsys_20260826/`:
jobs `6522967`, `6539640`, and `6553739` each have ranks 1--32, for 96 reports
and 99,202,685 total bytes.  The original local copies remain unchanged.
The five cases in the subsequent performance report were consolidated as
physical local copies under
`scratch/qwen3_ffn_overheads_20260820/five_reported_cases_nsys_20260826/`.
It contains job `6553739` A7/EP4/b6 plus jobs `6531588` A7/EP8/b6+b7 and
`6531587` A8/EP8/b6+b7: 304 reports, 305,534,787 bytes, complete rank coverage,
and no symlinks.  Its manifest records the job, code head, count, size, and
reported metric for each case.
Current-head A7/EP8 bundle job `6555417` ran from clean `202694a` on 16 trays /
64 GPUs and completed `0:0` in 26:32 with one successful and one failed case.
Batch 7 failed launcher preflight before model work because its 918,016-token
capacity override did not reach `run_afd.sh`; all ranks instead saw the
839,424-token batch-6 capacity against 917,952 required tokens.  The allocation
then completed batch 6 with 64/64 Nsight reports, 15/15 retained samples, zero
outliers, a 3.7997-percent dominant range, 25.5875678667-ms strict CUDA mean,
25.541538-ms median, 205.177766 TPS/active GPU, and 39.081479 TPS/user.  Against
previous-head batch-6 job `6531588` at 25.4614114 ms / 206.194382 TPS/GPU, this
is 0.1261565 ms (0.495 percent) slower and 0.493 percent lower throughput.
Batch 7 still requires a corrected-capacity rerun; do not treat its failed
preflight as performance evidence.
CPU-only trace-export jobs `6558537`/`6558538` and `6558783`/`6558843`
subsequently exported representative model rank 57 and attention rank 1 for
previous EP8/b6 job `6531588` and current job `6555417`.  The strict metric's
attention role is longer in both runs, but this identifies where the maximum
graph span is observed, not its causal bottleneck.  Across the 15 measured
graphs, the two attention streams are simultaneously inside
`wait_ready_kernel` for 2.315912 ms/step previously and 2.684408 ms currently,
respectively 9.096 and 10.491 percent of E2E; the 0.368496-ms increase proves a
real FFN/readiness-bound bubble.  Model-rank MegaMoE calls do not overlap each
other.  Their measured-window means regress 72.173173 -> 75.633107 us/call;
188 calls sum to 13.568557 -> 14.219024 ms/step, +0.650467 ms / +4.794 percent.
Kernel symbols show both runs already use M32 and 16 experts/wave; the current
EP8 shape therefore receives neither the M16-to-M32 gain nor the final 32-expert
wave rule.  The aggregate current stack changes K128/10 stages/150 SMs/two
epilogue warpgroups to K256/5 stages/128 SMs/one epilogue warpgroup while also
enabling weight eviction, active counting, count-gated pull, and rank-epoch
combine.  The combined configuration is slower for this 16-local-expert EP8
shape; the trace does not isolate which launch/staging change dominates.  Do
not describe EP8/b6 as attention-compute-bound or infer that FFN optimization
has no headroom from the representative-role critical label.  Companion
A8/EP8 b7+b6 bundle `6555431` was cancelled at the user's request on 2026-08-26
13:38:58 PDT while pending with zero elapsed time, no start time, and no nodes
assigned.  It consumed no GPU allocation and produced no performance evidence.
Historical-variable submission `6553492` failed launcher preflight in 21
seconds before benchmark work and is not performance evidence.

## Active FFN latency iteration (2026-08-24 PDT)

The current scaling goal is A15:F1, FEP4, batch 6, 128K ISL, ATP1, MB2, using
A7:F1 first as the performance-observation gate.  The alignment standard and
exact warmup-plus-15 strict metric contract remain unchanged.  Retain general
kernel/runtime changes with no Qwen-shape or batch-specific branches.  The
aligned A1:F1 acceptance checkpoint is remote commit `bc9ab7f`, job `6502925`,
at 25.0087105 ms coordinator E2E median and 24.9813376 ms strict CUDA mean.
The scaling expectation is approximately 26 ms because each additional
attention source adds only tens of memory-bound FFN tokens per model round.
On 2026-08-25 the user explicitly closed algorithm, topology, and precision
changes.  Treat all FP4 and split-MegaMoE results as diagnostics only; the
best compliant A7 case remains unchanged-FP8 job `6504134` at 32.6938360 ms.

Local Git/GitHub handoff on 2026-08-25: the coherent FMHA implementation was
committed locally as `393bc98` (`Implement FMHA-only AFD execution mode`) and
pushed only to personal branch `linsj20/fmha_only`; `linsj20/main` was left
unchanged.  The development checkout is execution-only and is not maintained
for GitHub synchronization.  A content comparison against accepted A7 commit
`bc9ab7f` proves that `393bc98` already contains the identical executable
source, scripts, tests, and assets.  The only extra path in the accepted tree
is an older handoff document, so preserve this newer project memory rather than
importing the stale remote-Git workflow text.

Exact A1 validation of `393bc98` used isolated execution commit `732eaa5`
whose tree object exactly matched local `393bc98`, with clean-source
enforcement.  Short-queue job `6516488` used a one-hour limit, two trays / eight
active GPUs, A1:F1, FEP4, batch 6, 128K ISL, ATP1, MB2, FMHA-only placement,
and unchanged FP8 precision and measurement rules.  It completed `0:0` in
22:13 after normal synchronized weight load/JIT and sustained all-rank prefill;
there was no hang or inefficient low-utilization phase.  The result is
25.1577345 ms coordinator E2E median and 25.1350228 ms strict CUDA mean, with
15/15 samples, zero outliers, a 1.6814-percent dominant range, and 119.355372
strict TPS/active-GPU.  Versus accepted A1 job `6502925`, this is 0.1490240 ms
(0.596 percent) slower by E2E median and 0.1536852 ms (0.615 percent) slower by
strict CUDA mean.  Compact local evidence is under
`scratch/qwen3_ffn_overheads_20260820/fmha_only_a1_job_6516488/`; the remote
task root is `qwen3_ffn_overheads_20260820/fmha_only_a1_20260825/`.

Exact A7:F1 observation job `6504134` was submitted on 2026-08-24 to `short`
with a one-hour limit, eight trays / 32 GPUs constrained to one eight-tray
NVL72 segment, FEP4, batch 6, 128K ISL, ATP1, MB2, FMHA-only placement, clean
commit `bc9ab7f`, and the unchanged strict metric contract.  Remote task root:
`ax_alignment_20260824/a7_fep4_b6_128k/`.  Monitor at five minutes during
startup and near completion or fifteen minutes during long prefill/measurement
phases.  It completed `0:0` in 23:57 after normal prompt generation, JIT, and
sustained-utilization prefill; every rank's GPU-progress record advanced and no
hang or inefficient low-utilization phase was observed.  The exact result is
32.6938360 ms strict CUDA mean and 160.580728 strict TPS/active-GPU, with all
15 target steps retained, zero outliers, a 3.2824-percent dominant range, and
one 752-node attention graph plus one 3,395-node grouped-model graph per step.
This clean result misses the approximately 26-ms scaling expectation.

CPU-only `cpu-short` export jobs `6504814`--`6504817` completed in 30--35
seconds and produced exact accepted-A1/current-A7 attention and model SQLite
comparators under the same task root.  The reusable graph comparison shows
that A7 attention compute is nearly flat: summed FMHA grows only 0.125 ms per
replay, while its readiness waits expose the longer model wavefront.  On the
model rank, expert gate/up grows 4.119734 -> 7.357282 ms/replay and expert down
grows 2.428783 -> 4.065531 ms/replay.  `combine_impl` adds 1.901784 ms and QKV
publication adds 0.869214 ms; model graph span grows 7.664016 ms.  There is no
unexplained graph idle gap or extra graph launch.  This attributes the miss to
real grouped-MoE expert-weight/communication work at 21 rows per microbatch,
not a prefill hang or per-source graph serialization.

Information-dense four-GPU smoke sweep `6505134` completed `0:0` in 8:46 on
`short` using clean `bc9ab7f`.  In one allocation it compared
rows/DeepGEMM-SMs/DeepEP-SMs `3/128/24`, `21/128/24`, `21/144/8`, and
`42/128/24`; every case passed BF16, FP8, and fused-router numerical checks.
Fused-router staged times were respectively 0.351262, 0.355903, 0.350601, and
0.338912 ms.  Moving row 21 to the existing high-fan-in 144/8 split is a small
1.49-percent isolated win.  Combining both model microbatches is only 4.78
percent faster than one row-21 stage and would forfeit the current overlap of
the first model stage with the second attention stage, so it is not a credible
route to removing the observed 7.7-ms replay gap.

Fresh branch `codex/qwen3-a7-high-fanin-20260825`, worktree
`worktrees/a7_high_fanin`, commit `b45d263` moves the existing general
high-fan-in policy threshold from eight sources to seven.  A1 remains on the
aligned 24-DeepEP/128-DeepGEMM split; A7 and above use 8/144.  The change is
three constant/comment/contract-test lines and has no model-, batch-, or shape-
specific branch.  CPU `cpu-short` contract job `6505350` was the source gate.
It completed `0:0` in 1:47 with all 62 tests passing.  Exact unchanged-contract
A7 job `6505450` is submitted to `short` for at most one hour on eight trays in
one eight-tray NVL72 segment; monitor at five minutes through startup/prefill
and fifteen minutes only for a demonstrably long steady phase.
It completed `0:0` in 26:03 after normal sustained-utilization prefill, but is
rejected: 33.5253775 ms strict CUDA mean and 156.597789 strict TPS/active-GPU,
versus 32.6938360 ms and 160.580728 for accepted-threshold control `6504134`.
The regression is 0.8315415 ms / 2.543 percent.  All 15 samples were retained
with zero outliers and a 3.2794-percent dominant range, and the attention/model
graphs remained 752/3,395 nodes, so this is a clean same-contract rejection.
CPU-only rank-29 export job `6506007` is the remaining attribution step.  Do
not merge `b45d263`; retain the threshold of eight sources in `bc9ab7f`.  A15
remains gated off because neither exact A7 result approaches 26 ms.
Export `6506007` completed `0:0` in 2:40.  Versus control rank 29, candidate
model span grows 0.834667 ms while readiness-wait sum is flat (-0.071037 ms).
Expert gate/up grows 7.357282 -> 7.785798 ms/replay and expert down grows
4.065531 -> 4.279097 ms/replay; dense O/QKV improve by about 0.127 ms in
aggregate, but memory-bound expert GEMMs regress 0.642082 ms and dispatch adds
0.105639 ms.  This causally confirms that 24/128 is the correct A7 split even
though the isolated four-rank smoke slightly favored 8/144.

The opposite general split was tested without spending an exact eight-tray
run.  Isolated branch `codex/qwen3-a7-deepep32-20260825`, worktree
`worktrees/a7_deepep32`, commit `07611bb` uses 32 DeepEP SMs and 120 DeepGEMM
SMs for the existing low-fan-in policy; its five-file diff is the preserved
general four-cluster implementation from `cfa8660`.  CPU job `6506147` failed
before tests because the wrapper used its obsolete home-directory image
default.  Corrected `cpu-short` job `6506220` pinned the same canonical Lustre
image/venv as prior gates and passed 62/62 in 1:51.  Focused four-GPU `short`
smoke `6506379` compared row-21 120/32 and 128/24 sequentially in one
allocation with a shared fresh build cache; it completed `0:0` in 5:50 with
identical BF16, FP8, and fused-router correctness.  Fused-router staged time
was 0.326447 ms for 120/32 versus 0.325565 ms for accepted 128/24, a 0.27%
regression.  Together with the preserved exact-A1 neutrality of this split,
the smoke rejects `07611bb` without an eight-tray A7 run.  Keep accepted
24/128 and do not merge this candidate.

The next expert-bandwidth audit rejected reverse lane/expert scheduling before
implementation: each lane executes gate/up then down, so hundreds of MiB of
other matrices evict the matching expert weights before another lane could
reuse them.  The legacy split-worker MegaMoE backend has dual-lane L2 reuse,
but it is not wired into FMHA-only placement and its historical optimized A15
result was still 30.752 ms.  A focused packed-weight benchmark therefore
tested the only in-scope byte-width change large enough to matter.  Initial
four-GPU `short` job `6506867` failed safely before timing because the FP8
reference declared the wrong scale recipe.  Corrected job `6506962` completed
`0:0` in 1:20 on four independent exact-Qwen grouped shapes.  Across ranks,
FP8-to-MXFP4 reduced gate/up from 0.06329--0.06365 to 0.03912--0.03918 ms
(1.617--1.627x) and down from 0.03534--0.03570 to 0.02277--0.02289 ms
(1.552--1.561x).  The cost is material numerical change: relative RMSE is
0.1244--0.1251, cosine similarity 0.99215--0.99223, and maximum absolute error
is about 0.80 gate/up and 0.54 down.  Projecting only measured expert kernels
suggests roughly 4.3 ms of A7 replay savings, or about 28.4 ms overall rather
than the 26-ms target, so unchanged full-model alignment remains mandatory.

Isolated branch `codex/qwen3-a7-fp4-experts-20260825`, worktree
`worktrees/a7_fp4_experts`, commits `299c8e3` and `4d6a744` add an explicit
`MINISGL_DEEPGEMM_EXPERT_WEIGHT_DTYPE=fp4` mode.  It converts loaded FP8 expert
weights once during model prewarm to packed per-32 MXFP4, releases the source
FP8 tensors, supplies explicit DeepGEMM recipes, handles packed logical K
widths, fails on mode changes, and keeps FP8 as the default.  The AFD bundle
and case launchers propagate the mode across container boundaries and record
it in `afd-result.json`, preventing a silent FP8 measurement.  CPU-short jobs
`6507343` and `6507515` passed 65/65 tests; the latter covers the pinned
launcher commit.  First four-GPU exact-Qwen row-21 DeepEP smoke `6507516`
made normal compile/execution progress and failed cleanly at 4:29 only because
packed FP4 reached 0.213595 maximum relative error versus the FP8-only 0.15
smoke limit; its 0.051457 maximum absolute error remained below 0.08.  Commit
`090358e` leaves the FP8 gate unchanged and gives explicit FP4 smokes a
separately reported 0.25 relative bound.  Warm-cache replacement `6507665`
passed in 1:44: staged/fused FP4 errors were 0.213595/0.237834 and fused staged
time was 0.336982 ms.  Identical same-node FP8 control `6507699` passed in
1:33 at 0.339872 ms fused, only 0.85% slower, while its non-fused stage was
1.44% faster.  Because that integrated result does not reproduce the isolated
GEMM projection, commit `bd1d610` adds expert-only timing on one live DeepEP
packet.  Paired same-allocation `short` job `6507808` completed `0:0` in 4:58
with FP8/FP4/FP4/FP8 and 100 iterations each.  Expert-only medians were
0.106021 ms FP8 versus 0.095182 ms FP4, only a 10.22% gain.  Fused full-stage
medians were 0.361713 versus 0.338165 ms (6.51%); non-fused staged medians were
0.355062 versus 0.341047 ms (3.95%).  The earlier 1.6x isolated result is best
explained by repeatedly timing one same-weight GEMM, which gives packed FP4
much stronger partial L2 reuse than the production gate/up -> activation ->
down sequence.  Applying the integrated expert gain to exact A7 predicts only
about 1.2 ms improvement, around 31.5 ms rather than 26 ms, while the numerical
error is substantial.  Reject the FP4 candidate without an eight-tray exact
run; do not spend official alignment or A15 allocations on it.

That rejection was based on eager host-driven timing and is superseded by a
CUDA-Graph audit.  Test-only commits `77fb673` and `0836cff` add an NVTX-scoped
expert benchmark and graph-replayed expert-stage benchmark without changing the
production FP4 path.  Four-GPU `short` profile job `6512629` completed `0:0` in
3:17 with normal startup and attributed 99.111 us of expert kernels to FP8
(61.096 us gate/up, 4.546 us quantization, 33.469 us down) versus 63.680 us to
FP4 (37.208/4.478/21.994 us), a 35.7% kernel-time reduction.  CPU-only export
jobs `6512717` and `6512718` completed `0:0`.  Exact-Qwen ABBA graph job
`6512853` then completed `0:0` in 5:08: FP8 graph replay was
0.103371/0.103408 ms and FP4 was 0.067863/0.067658 ms, a stable 34.4% reduction;
all BF16, FP8, and fused checks passed, with the previously accepted FP4 error
bounds unchanged.  The eager integrated gap was therefore CPU launch/config
overhead that production graph replay excludes.  Advance FP4 to one exact A7
performance gate, but still withhold alignment and A15: its projected A7 latency
is about 28.7 ms, above the 26-ms target, and the numerical change remains
material.

Exact A7 FP4 performance gate `6512936` was submitted from clean head `0836cff`
on eight trays with `MINISGL_DEEPGEMM_EXPERT_WEIGHT_DTYPE=fp4`, `short` QoS,
and a one-hour limit.  Its task root is
`ax_alignment_20260824/a7_fp4_exact_20260825`; monitor startup at five minutes
for compilation/load/prefill progress, then use a fifteen-minute cadence only
after progress is proven normal.  Do not launch alignment from this candidate
unless the unchanged exact latency metric is competitive.

Gate `6512936` completed `0:0` in 26:05.  Five-minute health checks showed
normal progress rather than a hang: all 28 attention workers initialized,
expert compilation and 48-shard loading advanced continuously, the packed
expert tray stabilized near 50.3 GiB/GPU, and sustained 128K prefill kept the
attention GPUs mostly at 83--100% utilization while the expert GPUs remained
active at roughly 34--100%.  The unchanged exact metric is 28.5652352 ms over
15/15 retained samples with zero outliers, 3.2265% dominant range, and
183.789840 TPS/active-GPU.  Relative to FP8 A7 `6504134`, FP4 saves 4.128601 ms
(12.63%) and raises TPS/GPU by 14.45%, matching the graph-based projection, but
it remains 2.565235 ms (9.87%) above the 26-ms expectation.  Therefore do not
run alignment or A15 from this candidate yet.  Local copies of its metric,
logs, all-rank GPU telemetry, and Nsight reports are under
`ax_alignment_20260824/a7_fp4_exact_20260825/`.

CPU-only `cpu-short` Nsight export jobs `6513520` and `6513522` completed
`0:0` in 2:38/2:34 for representative FP4 attention rank 1 and model rank 29;
the first submission attempt created no job because it omitted the required
CPU-partition account, then the corrected submissions used
`coreai_comparch_sysarch`.  Same-15-launch comparison shows FP4 reduces the A7
model span by 4.128996 ms and summed model kernels by 4.585750 ms.  The main
kernel changes are grouped GEMM 15.742349 -> 12.064436 ms/replay, combine
5.201061 -> 4.562706, and model wait 24.758319 -> 24.462048; attention waits
fall by 8.412258 ms summed across both lanes, yielding the exact 4.128601-ms
critical-span win.  Against A1, however, FP4 A7 model span remains 3.535020 ms
longer.  Its residual component gaps are grouped GEMM +1.128912 ms, combine
+1.263429, QKV publication +0.908994, and model wait +1.068850 (overlapped).
The shared grouped-GEMM kernel name masks the FFN-specific regression: exact
per-launch stream-order classification shows expert gate/up at
4.129558 -> 4.850686 ms (+0.721128, 17.5%) and expert down at
2.442216 -> 2.979889 ms (+0.537673, 22.0%), while dense O plus QKV projection
improves by 0.128933 ms.  Full combine including its reduce epilogue grows
4.113217 -> 5.429291 ms (+1.316074, 32.0%); full dispatch grows
2.856563 -> 2.950402 ms (+0.093838, 3.3%); the two routing kernels grow
1.559539 -> 1.665304 ms (+0.105764, 6.8%); and the fused SiLU/multiply/output
quant kernel grows 0.770948 -> 0.798460 ms (+0.027511, 3.6%).  Expert GEMMs
plus combine alone add 2.574875 ms of summed kernel work.  These sums are not
all exposed on the critical path, but they identify combine and both expert
GEMMs as the dominant FFN regressions versus A1.
No expert-only change can reach 26 ms.  Because faster FP4 compute changes the
resource balance, the next bounded test is an FP4-specific DeepGEMM/DeepEP SM
partition sweep with Nsight kernel attribution; an exact A7 rerun still
requires a credible roughly 2.6-ms integrated path.

The FP4-specific SM-partition audit is closed without an exact A7 run.  Clean
branch `codex/qwen3-a7-fp4-deepep32-20260825` combines FP4 with the previously
validated four-cluster DeepEP policy at production commit `d967fb4`; test-only
commit `b0c6253` adds an NVTX range around the fused-stage benchmark.  Full
CPU contracts job `6513645` passed 65/65 in 1:03 on `cpu-short`.  Four-GPU
`short` sweep `6513690` completed `0:0` in 10:48 with normal build/case
progress and identical BF16/FP4/fused numerical checks for
128/24, 144/8, 136/16, 120/32, and repeated 128/24 DeepGEMM/DeepEP SM splits.
Fused eager times were 0.522855/0.515900/0.523657/0.513840/0.519475 ms and
graph expert times were 0.069461/0.070249/0.069729/0.072679/0.069340 ms.
Representative CPU exports `6513947`--`6513950` completed `0:0` in
1:26--3:54; the first case-4 submission created no job because the `cpu-short`
submit limit was full, then replacement `6514111` completed `0:0` in 1:26.
NVTX attribution makes 120/32's ceiling explicit.  Against the mean of both
128/24 controls, one fused-stage GPU span improves only 0.5169865 ->
0.5098280 ms (-7.1585 us), corresponding to roughly 0.673 ms across one
94-stage critical lane.  Active union improves by 12.1225 us/stage (a loose
1.140-ms/replay ceiling): combine saves 13.7705 us and dispatch saves 1.7465,
but FP4 GEMM loses 3.4045.  The other partitions do not offer a stronger
kernel-time result.  This is far short of the 2.565-ms exact gap; reject the
32-SM default and do not spend eight trays on it.  Logs and all representative
SQLite exports are local under
`ax_alignment_20260824/a7_fp4_sm_partition_20260825/`.

The dense-weight FP4 audit is also closed without a production or exact A7
run.  A clean source audit found that the existing ungrouped DeepGEMM API
already accepts FP8 activations with packed MXFP4 weights, so QKV and O
projection could be bounded without implementing a new kernel.  Test-only
branch `codex/qwen3-a7-dense-fp4-probe-20260825`, commits `f0883f9`,
`d1e0586`, and `c84d7d7`, measures CUDA-graph replay for exact Qwen shapes
QKV `[M,4096] x [9216,4096]` and O `[M,8192] x [4096,8192]` at A1 `M=3`
and A7 `M=21`.  The initial one-GPU submission was rejected before job
creation by the short-QoS four-GPU minimum.  Job `6514520` was canceled while
pending with zero elapsed time after its clean-head guard was found to contain
a mistyped full hash.  Corrected job `6514522` failed fast in 2:21 after
successfully building the DeepGEMM extension because the probe omitted the
workspace-local TVM-FFI cache; it showed no GPU hang or stalled work.
Four-way sharded retry `6514603` used all four GPUs, `short`, and a 30-minute
limit, then completed `0:0` in 1:49 with normal quantization/JIT/case progress.
At A7's 21 rows, FP4 changes QKV 12.336320 -> 12.298880 us and O 14.353024 ->
14.349888 us, only 0.040576 us/stage or 0.007628 ms across 188 stages.  At
A1's three rows it saves 2.068480 us QKV and 2.085312 us O, about 0.780913
ms/replay, so the benefit disappears rather than scales at A7.  FP4 versus
FP8 output cosine is only 0.992927--0.993217 and relative RMS error is
12.26--12.55 percent.  Reject dense FP4: its A7 upper bound is negligible,
its accuracy is worse, and it cannot contribute materially to the remaining
2.565-ms gap.  Logs and all four Nsys reports are local under
`ax_alignment_20260824/a7_dense_fp4_probe_20260825/`.

The next causal mechanism is MB2 packet merging, motivated by an all-rank A7
trace audit rather than a synthetic-only result.  CPU-short export jobs
`6508074`--`6508076` materialized profiler ranks 30--32; rank 29 was already
local.  All four model ranks have the same 15-replay graph span
(32.579134--32.584058 ms mean), but their two expert streams are consistently
asymmetric: lane 0 sums 4.514694--4.755883 ms/replay while lane 1 sums
6.950167--7.209364 ms/replay.  Dispatch and combine are approximately balanced
across lanes and the same expert asymmetry appears on every rank, ruling out a
single-rank placement or hang.  Both microbatches activate almost every local
expert, so the current separate calls reload nearly the same expert weights.

Four-GPU `short` jobs `6508245` and `6508296` measured the resulting reuse
ceiling with 100 FP8 expert iterations and the accepted 128/24 SM split.  The
first submission encoded a comma inside Slurm `--export`, so Slurm safely ran
only rows 21; it completed `0:0` in 1:49 at 0.106763 ms expert-only and
0.367787 ms fused staged.  The explicit rows-42 replacement completed `0:0`
in 2:05 at 0.107986 ms expert-only and 0.365519 ms fused staged.  Thus one
merged packet handles twice the routed rows for only 1.1 percent more expert
time and essentially unchanged full-stage time; comparing two row-21 stages
with one row-42 stage exposes a credible roughly 50-percent MoE-stage saving.
Both jobs showed normal build/execution progress and passed BF16, FP8, and
fused-router numerical checks.

Isolated branch `codex/qwen3-a7-merged-mb2-moe-20260825`, worktree
`worktrees/a7_merged_mb2_moe`, starts from accepted `bc9ab7f`.  The candidate
applies only to multi-source (`fanin > 1`) MB2 model ranks, preserving the A1
baseline.  It retains independently masked routing for each logical MB, merges
the prepared packets for one lane-zero DeepEP dispatch/expert/combine, splits
the fixed rows before next-layer QKV publication, and uses explicit cross-lane
events.  One shared buffer is sized to the merged dispatch bucket and the exact
merged path is prewarmed before CUDA graph capture.  All invalid shape, empty-
packet, event-grid, and buffer contracts fail fast.  Remote CPU tests currently
pass 38/38.  Clean commit `92952f8` contains the five-file implementation and
tests.  Two attempted CPU-only `batch` submissions were rejected before
allocation (first missing the account, then because `batch` requires a GPU),
so the 0.14-second mocked suite ran directly rather than wasting a GPU.  The
exact A7 gate is job `6508956`, submitted to `short` with a one-hour hard
limit, eight trays in one NVL72 segment, clean-source enforcement, and the
unchanged warmup-plus-15 trace contract.  A15 and official alignment remain
gated on an approximately 26-ms A7 result.

Job `6508956` completed `0:0` in 24:41 after healthy synchronized startup and
sustained prefill: all model ranks loaded 48/48 shards together, all eight
minute-progress files advanced, attention trays held roughly 87--100% GPU
utilization, and the model tray held roughly 56--65%.  There was no hang or
inefficient low-utilization prefill.  The candidate is nevertheless rejected:
42.9341909 ms strict CUDA mean and 122.280166 strict TPS/active-GPU, versus
32.6938360 ms and 160.580728 for control `6504134`.  All 15 samples were
retained with zero outliers and a 2.1544-percent dominant range.  Attention and
model spans both expand to about 42.9 ms; attention remains 752 graph nodes,
while model drops from 3,395 to 3,113 nodes.  This is a real synchronized
wavefront regression, not a metric or hang artifact.  Do not run A15 or the
official scorer for `92952f8`.  CPU-short job `6509574` exports candidate model
rank 29 for causal kernel/overlap attribution.  The exported SQLite is local at
`scratch/qwen3_ffn_overheads_20260820/ax_alignment_20260824/a7_merged_rank29.sqlite`.
Per replay, all block-scaled GEMMs fall from 752 calls / 15.742349 ms to 564 /
11.441701 ms, dispatch and combine each fall from 188 to 94 calls, and those
work reductions save about 6.7 ms.  However, readiness waits grow from
24.758319 to 33.319597 ms and fan out over 95 streams instead of two; delaying
MB0 next-layer QKV until MB1 joins the merged packet destroys the attention
wavefront.  This is the direct cause of the net regression, so full packet
merging cannot be repaired by more kernel tuning alone.

The older density-tail commit `0fe859b` was audited before reuse.  Its previous
A15 job `6488914` was canceled after eight minutes of active prefill, not
hung: all 60 attention GPUs were sustaining 82--100 percent utilization and no
decode metric was produced.  At A7, changing expected M from 21 to 12 would
still select the same 32-row alignment and hard-coded two-CTA grouped layout,
so no duplicate GPU run was spent on it.

Branch `codex/qwen3-a7-grouped-cga-policy-20260825` tested whether high routing
density should replace the two-CTA grouped GEMM with one CTA.  Commit `dd4c6fa`
keeps the accepted layout below expected M 16 and selects one CTA above it;
focused CPU contracts pass 17/17.  First paired smoke `6509886` failed before
timing because its new wrapper omitted workspace-local TVM-FFI cache variables;
replacement `6510061` completed `0:0` in 9:24 with control/candidate/candidate/
control on one four-GPU allocation.  Numerical checks matched, but candidate
fused-stage medians were about 0.349 ms versus 0.328 ms control and candidate
expert-only time was about 0.219 ms, roughly twice the accepted two-CTA
evidence.  Reject the one-CTA policy without an exact A7 run.

The A7 trace proves clustered expert kernels overlap only the opposite lane's
one-warp readiness wait, not DeepEP communication.  Test-only branch
`codex/qwen3-a7-deepgemm-sm-sweep-20260825`, commit `2cec7b9`, therefore permits
an explicit independent compute/communication budget only in the smoke and
adds expert-only timing.  Misencoded comma-export job `6510429` was canceled
after eight seconds before a benchmark; replacement `short` job `6510433`
completed `0:0` in 12:43.  It used the actual Qwen dimensions H4096/I1536,
rows 21, E128/top-k8, fixed 24-SM DeepEP, 100 iterations, and DeepGEMM grids
128/96/112/120/136/144/152/128.  All numerical checks matched.  Expert-only
times were respectively 0.106246/0.107602/0.107195/0.107866/0.108683/0.109398/
0.106874/0.106784 ms: accepted 128 SMs remain best within noise.  Do not spend
an exact A7 run on independent expert-grid sizing.  Also note that the older
generic non-FP4 A7 sweep wrapper used H7168/I2048; use H4096/I1536 for future
Qwen expert microbenchmarks.  The FP4 comparison wrapper already defaulted to
the correct H4096/I1536 dimensions, so its rejection remains valid.

Branch `codex/qwen3-a7-gate-weight-l2-20260825`, commit `9f001d2`, tested an
`EVICT_LAST` TMA load hint only for the DeepEP partial-sum gate/up weights;
down weights and activation loads retained the accepted policy.  Focused CPU
contracts passed 18/18.  Paired exact-shape ABBA smoke `6510975` completed
`0:0` in 11:19 with matching numerical checks.  Control fused-stage times
were 0.363198/0.363426 ms, candidate times were 0.351662/0.366854 ms, and
candidate expert-only times were 0.107026/0.106779 ms.  This is run-to-run
noise, not a reproducible gain, so reject the hint without an exact A7 run.

The next isolated target is the single-scaleout combine launch.  The accepted
A7 launch has a 24-CTA, one-warp-per-CTA grid because its useful-warp cap uses
only `num_max_tokens_per_rank=21`, even though each rank reduces partials from
four scale-up ranks.  Its combine kernels cost about 5.204 ms per replay,
roughly 1.900 ms more than A1.  Branch `codex/qwen3-a7-combine-warps-20260825`,
commit `9d3d3e7`, sizes the cap from the aggregate
`num_max_tokens_per_rank * num_scaleup_ranks`: A1 remains one warp per CTA and
A7 becomes four.  Focused source contracts pass 17/17.  The reusable paired
wrapper now isolates DeepEP build caches by source label as well as DeepGEMM
caches.  First smoke `6511211` failed in 25 seconds before any benchmark code
because the submitted legacy `/home` image path no longer exists; it consumed
no meaningful GPU time.  Replacement `6511406` used the verified Lustre image
and EP venv and completed `0:0` in 10:40.  All four ABBA phases advanced in
lockstep and passed identical numerical checks; there was no hang or idle
prefill phase.  Control fused-stage times were 0.340973/0.344719 ms, versus
candidate 0.355473/0.365209 ms, so aggregate-warp sizing regresses the mean by
about 5.1 percent.  Candidate expert-only time remained 0.106093/0.107054 ms.
This confirms that the existing one-warp-per-output-token mapping is already
work-complete: extra warps add launch/synchronization cost rather than sharing
one token's four-rank reduction.  Reject `9d3d3e7` without an exact A7 run.
Follow-up commit `5a28fc8` tests the bounded midpoint: it amortizes at most two
worst-case reduced tokens per warp, selecting one/two/four warps per CTA for
the A1/A7/projected-A15 capacities.  Focused contracts pass 17/17.  Exact-Qwen
ABBA smoke `6511547` completed `0:0` in 10:15 on `short` with fresh labeled
build/JIT caches.  All four phases advanced normally and passed identical
numerical checks.  Control fused-stage times were 0.334013/0.348480 ms versus
candidate 0.345171/0.351115 ms; candidate mean is about 2.0 percent slower and
does not beat both controls.  Reject `5a28fc8` without an exact A7 run and
close the combine launch-width family.

The exact transport trace also closes QKV launch-width work as a primary
target.  A1 uses one 24-block single-source publication kernel averaging
8.647 us/call and 1.625634 ms/replay; A7's existing flattened generic kernel
uses one 168-block launch for all seven edges, averages 13.261 us/call, and
costs 2.493044 ms/replay.  Seven times the payload raises the call span by only
4.614 us and explains the known 0.867410-ms replay delta.  This is efficient
fan-out scaling and cannot recover the remaining multi-millisecond expert
weight cost.  The accepted FMHA-only initializer returns into
`AfdFmhaRuntime` before the legacy MegaMoE AG/EG setup, so its dual-lane weight
reuse is not a backend toggle under this placement; using it would require a
new persistent expert-service integration, not an environment-only A7 run.

Exact stream timing rejects that persistent-service integration under the
current schedule before implementation.  On accepted A7 rank 29, each replay
alternates expert stages on streams 32 and 164.  Lane 0 has 93 stages averaging
52.454 us and lane 1 has 93 averaging 81.017 us, but lane 1 begins an average
103.724 us after lane 0's expert stage ends (91.744--120.768 us), and the next
lane-0 stage begins an average 105.214 us after lane 1 ends
(94.976--130.624 us).  Hundreds of MiB of intervening dense matrices therefore
separate the same expert weights; retaining roughly 600 MiB of expert weights
through that interval is not credible, and waiting to pair lanes would recreate
the wavefront loss measured by merged-MB2 job `6508956`.  The failed
gate-weight L2 hint in job `6510975` independently supports this conclusion.
Close the persistent dual-lane weight-reuse path unless the graph schedule is
first changed without delaying either microbatch.

The remaining packed-weight audit found no maintained grouped-NVFP4 drop-in.
The local DeepGEMM grouped SM100 path hard-codes per-32 UE8M0 scales for its
MXFP4 instruction descriptor.  Vendored CUTLASS contains Blackwell
MXF4/NVF4 instructions with UE4M3 per-16 scaling, but exposing them here would
require new grouped runtime dispatch, scale packing/layout, recipe validation,
and numerical coverage.  Current upstream DeepGEMM grouped dispatch likewise
selects its SM100 FP8/FP4 grouped path from integer scale tensors, while its
published grouped tests exercise the existing recipe rather than a grouped
NVFP4 path.  Treat NVFP4 as a new invasive kernel/runtime feature, not a
configuration candidate; first measure whether reducing active routed work can
recover enough latency to justify any approximate byte-width or route change.

Four-GPU `short` job `6512021` completed `0:0` in 12:56 with a 20-minute hard
limit and normal progress: the first five minutes built DeepGEMM and DeepEP,
then all six 100-iteration cases advanced sequentially with fresh log markers,
and teardown finished with seven minutes of margin.  It swept top-k
8/6/4/2/1/8 at rows 21, H4096/I1536, E128, and the production 128/24 SM split.
All BF16, FP8, and fused-router numerical self-checks passed.  Fused FP8 times
were 0.333164/0.347407/0.339585/0.345222/0.335420/0.361829 ms and BF16 times
were 0.330098/0.312748/0.320043/0.318578/0.318545/0.329609 ms, so fewer route
entries alone do not improve the full stage.  The deterministic smoke routing
still activates all 32 local experts at top-k 8/6/4/2 and 21 at top-k 1;
therefore this closes route-entry/communication scaling but is not yet a strong
active-expert sparsity ceiling.  Logs are local under
`ax_alignment_20260824/a7_topk_work_sweep_20260825/logs/`.  Use the existing
test-only expert-timing branch for any sharper active-expert measurement; do
not implement route truncation from this result.

That sharper ABBA measurement is `short` job `6512211`, using clean test-only
commit `2cec7b9` and top-k 8/1/1/8.  It completed `0:0` in 8:02, advanced from
fresh build into all four GPU cases by the first five-minute check, and had no
hang or slow teardown.  Isolated FP8 expert times were
0.106035/0.089752/0.103693/0.106640 ms; paired means show only about 9.0
percent improvement from top-k 1 even though active local experts fall from 32
to 21.  Fused-stage times were 0.347743/0.347552/0.367824/0.357612 ms, so the
full stage has no gain.  This upper bound is smaller than the already-rejected
MXFP4 expert gain and comes with far greater model error.  Close active-route
truncation without an exact A7 or alignment run.  Logs are local under
`ax_alignment_20260824/a7_topk_expert_ceiling_20260825/logs/`.

The existing fused MegaMoE implementation is not a direct FMHA-only backend.
Its union topology assigns disjoint AG ranks (which own the dense model and
routing) and EG ranks (which own experts), whereas FMHA-only puts dense model,
routing, and EP experts together on the model ranks and leaves attention ranks
with FMHA only.  `_init_afd_runtime` therefore returns into `AfdFmhaRuntime`
before constructing `MegaMoEM2NAfdAdapter`; making model ranks act as both AG
and EG would require a new symmetric-buffer topology and runtime schedule.
Historical corrected MegaMoE evidence also does not establish the target:
A7:F1/FEP4/128K batch 5 was 28.055288 ms in job `6265474`, A8 batch 6 was
30.743356 ms in job `6265476`, and A15 batch 6 was 30.752484 ms in job
`6263396`.  Do not treat MegaMoE as an environment toggle or spend an exact
FMHA-only job on it without a new standalone topology proof.

The maintained split MegaMoE kernel already implements both FP8xFP8 and
FP8xFP4 expert GEMMs, so the historical 28AG+4EG topology provides a bounded
topology/format experiment rather than an FMHA-only backend toggle.  Branch
`codex/qwen3-a7-megamoe-fp4-20260825`, worktree
`worktrees/a7_megamoe_fp4`, clean commit `c6420a5` starts from the exact
historical source `762b477` used by corrected A7 job `6265474`.  It adds an
explicit `MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4` mode, converts loaded Qwen
block-FP8 expert weights once to packed per-32 MXFP4, passes the existing
kernel's `use_fp8_weights=false` contract through all per-layer/dual/persistent
EG paths, keeps FP8 as the default, and records the selected dtype in worker
logs.  The exact controller additionally records backend/dtype in
`afd-result.json` and has a one-hour hard limit.  Two preflight attempts
created no Slurm job: the reused historical controller first expected its old
intentionally dirty profiler and then found a sync-omitted empty extractor
directory.  Both were corrected without GPU use.  Exact A7 batch-6 job
`6514983` started after 14:53 queued and failed before model loading at 3:44:
the all-node port probe released its sockets, then the coordinator found
control port 27874 occupied on the head node.  This was a startup port race,
not a MegaMoE/FP4 or prefill failure, and the watchdog stopped the allocation
promptly.  Retry `6515294` uses the same exact source/shape, eight trays,
`short`, and one-hour limit with dedicated port block 43120--43126, but was
canceled at 1:48 of startup when the user reaffirmed that algorithm/topology
and precision must remain unchanged.  It produced no performance result.
Do not resume FP4, MegaMoE, alignment, A15, or any other GPU experiment from
this stop state.

Finally, exact lane attribution closes stream ordering and local compute
contention.  In accepted A1, the clustered grouped expert calls average
17.335 us on stream 32 and 17.732 us on stream 164, with 52.697/52.771 ms
summed over the trace.  Accepted A7 grows to 23.848/37.402 us and
72.497/111.307 ms.  An interval audit shows those calls overlap only the
opposite lane's one-warp `wait_ready_kernel`, never another local compute
kernel, so CUDA stream priority or lane inversion cannot remove local kernel
competition.  Even the impossible bound of making every slow-lane A7 call as
fast as the fast lane saves only `(111.307-72.497)/16 = 2.426 ms` per replay
and leaves about 30.27 ms.  Holding all A7 expert work to the A1 total removes
about 4.896 ms and still leaves about 27.80 ms, because the measured QKV and
combine payload deltas remain.  Reaching 26 ms from 32.694 ms would require
removing all expert scaling plus roughly 63 percent of the real communication
delta, despite the transport audits showing efficient seven-source QKV fanout
and work-complete combine.  Under unchanged FP8/top-k8/FEP4/F1 resources and
the unchanged wavefront, the 26-ms target is inconsistent with the measured
work; a topology, format, or model-routing scope change is required before
another exact A7 run can have a positive gate.

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

All raw Nsight reports for the best unchanged-FP8 cases were copied back and
SHA-256 verified on 2026-08-25: A1 job `6502925` has 8/8 reports and A7 job
`6504134` has 32/32 reports under
`scratch/qwen3_ffn_overheads_20260820/best_fp8_nsys_traces_20260825/`
(47 MiB total).

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
- Trace proof that heavy model overlap is limited to causally independent FFN
  lanes with distinct workspaces and synchronization state; dependent-batch or
  shared-workspace overlap remains forbidden.

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
