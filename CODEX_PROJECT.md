# Project Memory

## Surgical pre-FFN restoration with FP8 validation (2026-08-27 PDT)

“Pre-FFN optimization” is a component boundary, not a whole-commit boundary.
The authoritative FFN target remains historical commit `24b90c0`: A7:F1/EP4
job `6522967` measured 29.358190933 ms / 178.825733 TPS per active GPU, and
the same compute tree's A1 output passed unchanged alignment job `6520958` at
408/408 top-1 tokens. The initial whole-range reconstruction `d0a7eaf` was
therefore superseded; its queued smoke `6575467` was canceled at zero elapsed
time and used no GPU.

Remote branch `codex/fmha-megamoe-fp4-preopt-ffn-20260826`, worktree
`worktrees/fmha_megamoe_fp4_preopt_ffn_20260826`, starts from current
FP8xFP4/per-case-memory head `a61e5d4`. Clean commit `1b7038e` surgically
restores only the MegaMoE FFN kernel, routing, and FFN instrumentation surface.
The core C++ API/JIT/heuristics/CUDA/TMA/layout/PTX/top-k files are byte-exact
to `24b90c0`; the runtime/wrapper/adoption files are byte-exact to minimal
FP8xFP4 commit `47a0672`; and the memory contract test is byte-exact to
`54055a4`. Unrelated current launch validation, DeepGEMM build-dir override,
final-snapshot cleanup, bundle execution, and exact per-case batch-7 memory
propagation remain from `a61e5d4`. Route-stat, fused-router, concurrent-lane,
and A7 scheduler-tuning plumbing are absent.

Focused contracts pass 20/20, launcher syntax and `git diff --check` pass.
Initial validation mistakenly selected optional FP8xFP4. Its smoke `6576460`
passed, while one preflight and bootstrap attempt failed closed before model
execution due missing derived venv and image paths. FP4 A7 job `6577407`
completed at 25.8833942 ms / 202.832749 TPS per active GPU, but the user
clarified that it is not comparable to the target because validation must use
the same FP8xFP8 MegaMoE kernel as `6522967`. Its 32 Nsight reports are retained
locally under `scratch/qwen3_ffn_overheads_20260820/`
`fmha_megamoe_fp4_preopt_ffn_20260826/a7_ep4_b6_gate_1b7038e/trace/nsys/`
(33,627,949 bytes, checksum parity). The active FP4 A1 job `6579084` was
canceled at 8:35 before producing a sample.

FP8xFP4 remains available in the code, but all acceptance runs now explicitly
set `MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp8`. Corrected A7 dry and full
non-submitting validation select FP8xFP8, EP4, MB2 3+3, ratio 0.82, 839,424 KV
tokens, exact batch 6, eight trays/32 GPUs, and one warmup plus 15 measured
steps. Corrected short-QoS job `6579568` completed `0:0` in 21:52. Its strict
metric is 28.738912667-ms mean / 28.676994-ms median, 15/15 samples, zero
outliers, and 182.679145 TPS per active GPU. This is 0.619278266 ms / 2.1094%
faster than historical pre-FFN job `6522967`, so the low-cost proxy passes.
All 32 Nsight reports are copied locally under the matching task directory
(33,045,638 bytes). Two-tray FP8 A1 alignment-source job `6581273` completed
`0:0` in 19:15 and produced 24 prompts / 408 generated tokens with explicit
FP8xFP8 provenance. Its eight Nsight reports are copied locally (10,286,425
bytes). Unchanged one-node alignment job `6582345` completed `0:0` in 12:51:
24 prompts / 408 tokens, top-1/top-10/top-100 all 1.0, and average/maximum rank
both 1.0. The corrected FP8 gates therefore pass.

The A7 EP4 FP8 result is a low-cost non-regression proxy, not an exact-value
reproduction gate. Recovery of A7 EP8 and A8 EP8 FP8 performance is the final
goal, but do not submit those large, slow-to-queue allocations directly;
establish the smaller proxy and alignment evidence first.

The final measurement contract is exactly eight EP8 results from exactly two
jobs: one 16-tray A7 bundle and one 18-tray A8 bundle, each running batch 6/7
with both MegaMoE FP8xFP8 and FP8xFP4. Clean remote matrix commit `7c1491e`
adds only per-case precision, precision-qualified case IDs, and fail-fast
per-case memory validation to `1b7038e`. All 21 focused tests, shell syntax,
and diff checks pass. All eight clean-head dry runs and full non-GPU preflights
pass. Batch 6 is 0.82 / no page override / 839,424 tokens / MB2 3+3; batch 7
is 0.90 / 14,344 pages / 918,016 tokens / MB2 3+4 with 64 tokens headroom.
Both pools contain exactly four untouched rows. Short QoS permits two hours,
64 nodes, and two concurrent jobs, so each final bundle will use a two-hour
allocation with a 30-minute watchdog per case. Exactly two final jobs are now
submitted: `6583163` requests 16 trays for the four A7 rows and `6583164`
requests 18 trays for the four A8 rows. A7 started at 01:19:29 PDT on one
contiguous NVL72 block and first claimed b7 FP8xFP8 with the correct 0.90 /
14,344-page / 918,016-token contract. A8 remains pending for resources with a
current scheduler estimate near 03:05 PDT. No additional EP8 job exists.

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

## FP8xFP4 pre-tuning baseline rerun (2026-08-26 PDT)

The user rejected the later K256/128-SM/single-wave tuning for the A7/A8 EP8
study. Pending 18-tray A8 job `6562900` was canceled before allocation. Remote
branch `codex/fmha-megamoe-fp4-ep8-baseline-20260826` now starts from clean
pre-tuning checkpoint `24b90c0` and adds only FP4 precision plumbing,
production-weight smoke coverage, exact per-case bundle memory, and FP4-only
two-lane launch budgeting; FP8 retains the old 150-SM launch. Clean head is
`bfda796`. Contracts pass 20/20. Four-GPU smoke `6572843` caught a host
function-pointer signature mismatch during fresh compilation and was canceled
at 3:11; fix `bfda796` then passed smoke `6572969` in 2:29 with empty stderr,
cosine 0.975472, and finite exact-Qwen EP4 outputs at 3 and 64 rows.

A7:F1/EP4/128K/b6 gate `6573176` completed `0:0` in 22:35 on `short`, eight
trays, and a 30-minute cap. Its valid 15/15 strict metric is 33.968444467 ms
and 154.555208 TPS/active GPU, 4.610253533 ms / 15.70% slower than clean
pre-tuning FP8xFP8 job `6522967` at 29.358190933 ms / 178.825733 TPS/GPU.
The FP4-only two-lane safety contract divides 128 runtime SMs into 64 SMs per
lane; the captured A7 graph serializes those lanes, so this reproduces the
previously rejected lane-sharing regression instead of useful overlap. The
gate therefore failed and both EP8 submissions remain withheld. Fresh A7/A8
EP8 plans are under
`fmha_megamoe_fp4_ep8_baseline_20260826/a7_a8_ep8_b6_b7_bundle_bfda796/`.
Both pools initialize with two pending rows. All four dry runs pass: b6 is
0.82 / 839,424 KV tokens / MB2 3+3; b7 is 0.90 / 14,344 pages / 918,016 KV
tokens / MB2 3+4 with exact-max-batch enforcement. No EP8 job is submitted.

## Accepted same-rank MegaMoE FP8xFP4 precision (2026-08-26 PDT)

See `CODEX_PROJECT_qwen3_ffn_overheads.md` for the authoritative record.
Remote executable commit `6c210cf` adds an explicit FP8-activation/MXFP4-
weight option while retaining FP8xFP8 as the default. Exact
A1:F1/EP4/128K/b6/MB2 job `6561851` completed on `short` within its 30-minute
limit. Unchanged strict extraction job `6562231` retained 15/15 steps with no
outliers and measured 22.689653667 ms, 0.701194333 ms / 2.997729% faster than
accepted FP8xFP8 job `6520395` at 23.3908480 ms. Unchanged alignment job
`6562471` passed all 408 tokens with top-1/top-10/top-100 agreement and
average/maximum rank all 1.0. All eight raw job-`6561851` Nsight reports are
copied locally with verified SHA-256 parity under
`scratch/qwen3_ffn_overheads_20260820/fmha_megamoe_fp4_20260826/`
`exact_a1_6c210cf_lanes2/trace/nsys/`. The FP8xFP4 goal is complete.

The active FP8xFP4 EP8 scaling follow-up uses clean remote commit `a61e5d4`,
which restores explicit per-case memory propagation in the reusable bundle.
All four dry runs pass: batch 6 uses 0.82 / 839,424 tokens / MB2 3+3; batch 7
uses 0.90 / 14,344 pages / 918,016 tokens / MB2 3+4. Short-QoS jobs `6562899`
(A7:F1, 16 trays) and `6562900` (A8:F1, 18 trays) each run batch 7 then batch
6 with a one-hour allocation cap and 30-minute per-case watchdog. Both select
FMHA-only MegaMoE FP8xFP4. Results root is
`fmha_megamoe_fp4_20260826/a7_a8_ep8_b6_b7_bundle_a61e5d4/`.
Job `6562899` completed `0:0` in 46:11, but only A7/EP8 batch 7 produced a
valid metric: 28.942748933 ms strict CUDA mean, 28.890207-ms median, 15/15
samples, zero outliers, and 211.624681 TPS/active GPU. This is 1.983445%
slower than historical FP8xFP8 job `6531588`. Batch 6 received the correct
0.82 / 839,424-token contract but failed during first eager prefill with an
asynchronous CUDA launch failure, so it has no performance result. A8 job
`6562900` remains pending for resources.

## Active remote latency optimization (2026-08-24 PDT)

The user reopened optimization and requires ordinary remote Git commits for every rollback point; local Git remains untouched. Work is on branch codex/qwen3-ffn-latency-20260824. Commit 0a6a4a3 snapshots aligned job 6474505, whose accepted A1:F1 result remains 25.7321745 ms E2E, 25.6797822 ms strict CUDA, and perfect alignment in job 6474804. Trace export 6480445 completed; accepted-versus-fused analysis localizes the fused-turn regression to slower fused add-RMSNorm/FP8 work and added attention/model readiness delay, so that exact-two-stream path remains rejected.

The first new isolated candidate, commit 0ae36e8, fused steady-state input RMSNorm with QKV FP8 quantization while preserving the accepted CUDA-event and DeepEP cadence. Remote CPU contracts passed 60/60. Exact short-queue one-hour job 6481196 completed 15/15 with zero outliers but regressed to 26.1607085 ms E2E and 26.1054416 ms strict CUDA, respectively +0.428534 ms/+1.665% and +0.4256594 ms/+1.658% versus job 6474505. It is rejected without alignment and was restored by remote revert commit 487bddd. The second isolated candidate, commit 6d4b686, amortized each system fence across eight grid-stride 64-bit publication tasks. Remote CPU contracts passed 60/60. Exact job 6481517 completed 15/15 with zero outliers but regressed to 26.56672 ms E2E and 26.4969301 ms strict CUDA, +0.8345455 ms/+3.243% and +0.8171479 ms/+3.182% versus job 6474505. Reduced publication parallelism is rejected, the 16-tray A15 run was intentionally skipped, and remote revert commit c1c7e7a restores baseline. A CPU-only trace export was not submitted because short QoS rejected it with QOSMinGRES; no GPU was wasted on post-processing. The third isolated candidate, commit cd78af5, reduces DeepEP from three 8-CTA clusters to one and raises the complementary DeepGEMM budget from 128 to 144 SMs without changing collective protocol. Remote CPU contracts passed 60/60. Exact A1 job 6481772 completed 15/15 with zero outliers but regressed to 26.139278 ms E2E and 26.0721142 ms strict CUDA, +0.4071035 ms/+1.582% and +0.392332 ms/+1.528% versus job 6474505. Eight communication SMs are therefore rejected as a universal A1 policy. Exact A15:F1/b6 high-fan-in job 6481976 completed 15/15 with zero outliers at 33.261404 ms E2E and 33.286293333 ms strict CUDA, with a 6.945% strict dominant-range spread and 5.882% max/median separation. Against exact no-mode job 6263396 at 30.9095675 ms E2E and 30.752484 ms strict CUDA, the gaps are +2.3518365 ms/+7.609% and +2.533809333 ms/+8.239%; this candidate is not competitive and alignment is withheld. The eight-SM result is retained only as high-fan-in evidence. The production-shape four-GPU 8/16/24 DeepEP-SM sweep is complete. Jobs 6483946-6483948 failed before container start because the first submission used an incorrect image path. Retry jobs 6484042 and 6484043 then exposed a singular 45-row deterministic smoke input, and pending job 6484044 was canceled before allocation; commit 41ff981 makes that input full rank and passes 60/60 CPU contracts, with independent 16/24-SM cherry-picks 38bc6d8 and 824e295. Valid short-QoS one-hour jobs 6484104/6484105/6484106 all passed identical BF16, FP8, and fused-router correctness bounds. Their production fused-router staged means are respectively 0.337639/0.327233/0.318441 ms for 24/16/8 communication SMs, so eight SMs is 2.687% faster than sixteen and 5.686% faster than twenty-four for the exact 45-row A15 MoE shape. This isolates eight SMs as the best high-fan-in split in this family, while exact A1 job 6481772 still requires retaining the accepted 24-SM policy for low fan-in. Adaptive commit a8eb982 selects 8/144 communication/compute SMs at fan-in >=8 and 24/128 below it, supports only complete 8-CTA cluster counts, and passes 60/60 CPU contracts. Short one-hour smokes 6484308 and 6484309 pass the 45-row 8/144 and 3-row 24/128 branches with all BF16, FP8, and fused-router correctness checks. Exact A1 job 6484425 then selected 24/128 on all four model ranks and produced a durable 24-prompt/408-token sample with 25.9834295 ms raw E2E median, +0.251255 ms/+0.976% versus accepted job 6474505. Its strict post-processing alone failed because the supplied Pareto plan contains EP2 rather than this EP4 A1 row; all eight raw reports remain durable and strict recovery will be piggybacked on the next required GPU validation, so no A1 acceptance decision is made from the raw median alone. The standard /home fastafd_reproduce symlink was restored to the existing Lustre workspace so the pinned model-profile manifest resolves without bypassing validation.


The one-wave publication attempt is rejected as a production no-op. Wrapper job 6485116 failed before srun because sbatch --wrap used POSIX sh with pipefail; retry 6485203 reached fresh extension compilation and caught the out-of-scope CUDA_CHECK use before any workload. Fix commit 9be35c4 then passed combined short/one-hour job 6485226: the 8,192-row four-GPU transport smoke completed and recovered adaptive A1 strict metric 25.915859667 ms with 15/15 samples and zero outliers, +0.919% on the unchanged low-fan-in path. Exact A15:F1/b6 job 6485295 completed in 33:53 at 31.714345 ms E2E and 31.688765867 ms strict CUDA, 15/15 and zero outliers. Although this is 4.651%/4.799% faster than run 6481976, it remains 2.604%/3.045% behind no-mode job 6263396. Fresh rank-61 SQLite proves both 6481976 and 6485295 launched all 3,008 QKV publications as publish_qkv_kernel<128,false,true> with the same 360x256 geometry: ready_edges was one, so the multi-ready-only cap never executed and the observed difference is run/node variance, not a causal win. Reverts 78f6ec8 and 5b30bbe restore the adaptive baseline; the direct remote CPU contract passes 60/60 and Git is clean. Comparative SQLite is under analysis/a15_old_vs_twoline outside the source worktree.


Corrected commit 386e5af passed decode intent explicitly into QKV publication so graph decode alone used one device wave while eager prefill kept the proven 1,024-CTA bound and A1 remained naturally below the cap. The remote CPU contract passed 60/60; fresh production-shape four-GPU job 6486271 passed rows=45, nine iterations, and qkv_decode=1 on short with a one-hour limit. Exact A15:F1/b6 job 6486377 completed in 32:30 at 32.520627 ms E2E and 32.2289626 ms strict CUDA, 15/15 and zero outliers. Fresh rank-61 SQLite proves the intended causal signature: all 3,008 publish_qkv_kernel<128,false,true> launches changed from 360x256 to 152x256. However, versus the recent unchanged-path job 6485295, E2E regressed 0.806282 ms/2.542%, strict regressed 0.540197 ms/1.705%, and QKV kernel mean increased from 20.923 to 21.952 us/4.918%. The candidate remains 5.212%/4.801% behind no-mode job 6263396 and is rejected; revert baf1cdf restores the adaptive baseline and passes 60/60. Publication parallelism should not be reduced further.


System-atomic completion candidate 51b07a0 used an explicit high-fan-in decode-only specialization while preserving the accepted low-fan-in and eager-prefill paths. Four-GPU production-shape smoke job 6487241 passed 100 exact iterations at rows=45, sources=15, decode=1. Exact short-QoS one-hour A15:F1/b6 job 6487343 completed 15/15 with zero outliers at 33.4261575 ms E2E and 33.360445733 ms strict CUDA. Rank-61 SQLite confirms all 3,008 launches used publish_qkv_kernel<128,false,true,true> at the unchanged 360x256 geometry. QKV kernel total was 69.303433 ms, 10.116% slower than recent unchanged-path job 6485295; strict latency regressed 1.671679867 ms/5.275% versus that control, 2.607961733 ms/8.480% versus exact no-mode job 6263396, and 0.0741524 ms/0.223% versus earlier 8-SM job 6481976. The candidate is rejected; remote revert commit 1130048 restores the adaptive baseline and the direct remote suite passes 60/60. The QKV publication-parallelism and completion-atomic family is closed.


Placement-crossover job 6488053 pinned remote commit 5ee5ea9 and completed the missing exact A8:F1/ATP1/FEP4/b6 FMHA-only point on short with a one-hour limit. It completed 0:0 in 26:02 with 31.2774395 ms E2E, 31.238424333 ms strict CUDA, 15/15 samples, zero outliers, 3.971% dominant-range spread, and 170.729909 strict TPS/GPU. Optimized legacy A8 control job 6265476 is 30.897381 ms E2E and 30.7433556 ms strict CUDA, so FMHA-only loses 0.3800585 ms/1.230% E2E and 0.495068733 ms/1.610% strict. Together with the accepted A1 FMHA-only win and the A15 legacy win, this establishes A8 as the measured placement crossover. Remote commit 2c5df33 adds the explicit qwen3-128k-adaptive runner policy for exactly Qwen3 128K/b6/ATP1/FEP4/MB2: ratios below eight resolve to fmha-only and ratios at or above eight resolve to legacy; all other workload contracts fail fast. Results record requested placement, resolved placement, and policy. A1/A8 dry runs selected the intended modes, a batch-5 dry run failed before submission, bash syntax and diff checks pass, and the full remote suite passes 60/60.

## Current stop state (2026-08-24 PDT)

The user ended the exact-two-FFN-stream follow-up and selected the staged
accepted CUDA-event implementation as the final state. Local source/tests and
the canonical remote `source_two_lane_final` workspace were restored to that
staged code. No new performance run was submitted: the user explicitly waived
it because this is the already validated job-6474505 source. Report the retained
evidence as 25.7321745 ms E2E, 25.6797822 ms strict CUDA, 116.585561 TPS/GPU,
15/15 samples with zero outliers, and perfect official alignment from job
6474804. The trace retains 95 model CUDA Graph execution stream IDs because of
the event-created segments; this topology limitation is accepted for the stop
state. Do not resume the two-stream experiments unless the user reopens them.

Newer work completed the FMHA-only same-rank MegaMoE FFN optimization path on
2026-08-26.  The authoritative checkpoint and final validation record are in
`CODEX_PROJECT_qwen3_ffn_overheads.md`.  The retained A7 exact result is
27.6082989333 ms strict CUDA in current-code job `6553739`, versus
28.1574149333 ms at pre-final-FFN commit `04fb32b` and 29.3581909333 ms at the
earlier baseline.  The current result retains 15/15 samples, zero outliers,
190.160213 strict TPS/GPU, and attention as the critical role on all 15 steps;
mean model span improves 28.0479202 -> 27.4905426 ms versus `04fb32b`.  The
final isolated M32 target routes reach 80.446--80.477 us.  Official A1
alignment job `6552212` scored 24 prompts / 408 tokens with top-1/top-10/
top-100 and average/maximum rank all 1.0.  Remote clean head `202694a` contains
the winning policy, launcher cleanup, and final history.  The user-defined
terminal A1 gate passes, so this goal remains closed unless explicitly
reopened.  All 32 raw job-`6539640` Nsight reports are copied locally under
`scratch/qwen3_ffn_overheads_20260820/best_a7_job6539640_04fb32b/nsys/`, and
all 32 final job-`6553739` reports are under
`scratch/qwen3_ffn_overheads_20260820/latest_a7_job6553739_202694a/nsys/`.

## Active Qwen3 FMHA-only placement work

Minimum resume handoff: `CODEX_PROJECT_qwen3_ffn_handoff.md`.

The authoritative design, accepted evidence, campaign state, rejected-path
lessons, and completion boundary are in
`CODEX_PROJECT_qwen3_ffn_overheads.md`. That file is intentionally compact;
do not restore the chronological experiment transcript here.

On 2026-08-23 the user reopened the exact EP4 FFN path to separate DeepEP
buffer-ready control from payload transfer. Standalone prearming is numerically
correct (CPU 6473516, GPU 6473688) but rejected: job 6473693 regressed E2E to
26.0383275 ms and expanded the model graph by 376 nodes. Trace export 6473952
shows 27.843 ms in 6,016 control kernels versus 25.306 ms saved in the data
kernels. A full-handshake epilogue-tail candidate retained only two graph-start
bootstraps but job 6474094 still regressed to 26.071802 ms E2E because the tail
wait serialized following compute. The retained candidate posts readiness at
the epilogue tail without waiting and performs only a local ready-word check in
the later payload grid. Exact job 6474505 improves E2E/strict CUDA to
25.7321745/25.6797822 ms with the expected 564/3,678 graph shape. Official
alignment job 6474804 completed `0:0`: 24 prompts / 408 tokens,
top-1/top-10/top-100 all 1.0, and average/max rank 1.0. All eight raw Nsight
reports from this best case are copied locally under
`scratch/qwen3_ffn_overheads_20260820/preposted_buffer_ready_validation/trace/nsys/`.

The active follow-up is FFN stream reuse. SQLite inspection of the best-case
rank-1 attention trace finds 9,024 graph kernel calls, 564 nodes, and exactly
two kernel stream IDs. Rank-5 model has 58,848 graph kernel calls, 3,678 nodes,
but 95 kernel stream IDs: the two intended lanes plus 93 per-layer CUDA-event
segments. The first replacement reused one device turn and one-warp wait across
all layers. Job 6475180 plus SQLite export 6475310 proved exactly two model
kernel streams, but its 187 wait kernels raised model graph nodes to 3,865 and
regressed E2E to 26.429704 ms (+2.711%) and strict CUDA to 26.3694974 ms
(+2.686%); it is rejected. The current candidate retains the fused QKV release
but captures `cuStreamWaitValue32` operations on the two persistent lanes, so
the handoff has no standalone wait kernel. All 94 layers must reuse the same
two streams and two DeepEP buffers. Acceptance still requires an Nsight trace
proving exactly two model kernel streams and E2E no worse than the
25.7321745-ms aligned best case. See the two Qwen3 project files for current
validation state. The memory-wait candidate passes local 15/15, remote CPU
job 6475380 at 60/60, and fresh SM100 job 6475381, including a captured
two-stream wait/release replay. Exact job 6475460 and export 6475603 prove the
target 58,848/3,678 model kernel shape on exactly two streams with 1.811-ms
median productive overlap, but regress E2E/strict CUDA to
26.945252/26.8631071 ms (+4.714%/+4.608%); stream-memory waits are rejected.
The active candidate retains only same-layer lane-0-to-lane-1 event handoffs
and removes the cyclic lane-1-to-next-layer dependency responsible for extra
graph execution streams. It adds no steady-state wait kernel or stream-memory
operation. Local contract validation passes 15/15. CPU job 6475707 found one
stale negative-test setup before any GPU ran. Replacement CPU job 6475725
passes 60/60, and dependent fresh SM100 job 6475726 passes transport prebuild
plus two fabric pairs x nine 8,192-row iterations. Exact job 6475770 completes
`0:0` but regresses E2E/strict CUDA to 26.319437/26.2625085 ms,
+2.282%/+2.269% versus job 6474505. It is rejected without alignment. CPU
export 6475829 shows it still has 95 model stream IDs: lane 0 stays on one
stream, while every per-layer event splits lane 1 into a separate 19-node
execution segment. The next candidate removes all model-side events and relies
only on the existing per-lane O-ready waits, attention turn, and DeepEP cleanup
handshake; it adds no wait operation or kernel. Local contract remains 15/15
and remote CPU job 6475982 passes 60/60. Exact job 6475991 measures
26.1218645/26.0498022 ms E2E/strict CUDA, +1.514%/+1.441% versus job 6474505.
Export 6475992 proves the original 58,848/3,678 model shape now runs on exactly
streams 32/164, with both DeepEP lanes on the same streams and 1.148382-ms
median productive overlap. The topology succeeds but neutral-priority latency
does not. The model-lane-priority follow-up passes remote CPU 60/60 in job
6476267. Exact job 6476272 measures 26.015458/25.9618923 ms E2E/strict CUDA,
+1.101%/+1.099% versus job 6474505. Export 6476274 retains the original
58,848/3,678 shape on exactly streams 28/164; those same streams carry both
DeepEP lanes, whose productive overlap improves to 1.329086 ms median /
0.974784 ms minimum. Performance still rejects this candidate. The final
directional test extends the same lane-0 priority to attention while retaining
the zero-event graph. Local contract passes 15/15 and remote CPU job 6476500
passes 60/60. Exact job 6476535 measures 26.1026495/26.0252253 ms E2E/strict
CUDA, +1.440%/+1.345% versus job 6474505 and worse than model-only priority.
Export 6476537 again proves 58,848/3,678 on exactly model streams 28/164, with
both DeepEP lanes on those streams and 1.324384-ms median productive overlap.
The candidate is rejected. The accepted neutral-stream, per-round event
schedule has been restored locally and remotely; local contract passes 15/15,
and restored CPU job 6476746 passes 60/60. TODO remains: redesign the FFN handoff
so all 94 layers reuse exactly two CUDA Graph execution streams and two DeepEP
buffers while maintaining E2E at or below 25.7321745 ms. Transport code is
unchanged from fresh-passing job 6475726, so no redundant fabric smoke is
needed.

The active two-stream follow-up is trace-directed. Compared with accepted job
6474505, fastest exact-two-stream job 6476272 adds about 0.548 ms of attention
Q-ready wait and 0.191 ms of model O-ready wait per replay, but only 0.013 ms
of productive attention kernels. The first retained-graph candidate applied
priority -1 to every QKV publication and its four nearest productive ancestors
without adding a stream, node, or dependency. Local contract passes 15/15,
remote CPU job 6476950 passes 61/61, and focused raw-graph GPU smoke job
6476955 completes `0:0`. Full job 6476970 found 188 QKV anchors / 940 selected
kernels but regressed to 27.7717195 ms E2E and 27.6719843 ms strict CUDA:
symmetric signal-path priority synchronizes the lanes instead of staggering
them. It is rejected; trace export 6476971 is retained, and the next candidate
prioritizes only lane 0's QKV wavefront while leaving both FFN tails neutral.
The export proves 58,848/3,678 model kernels/nodes on exactly streams 32/164,
both used by DeepEP, with 2.168475-ms median productive overlap. That overlap
is counterproductive: attention Q-ready waits add 3.695024 ms and model summed
kernel time adds 3.761959 ms per replay versus job 6474505.
The asymmetric follow-up tags only model lane 0 at capture, resets that entire
lane to default, then reapplies priority -1 only to its QKV publication
ancestors. Local contract passes 15/15 and remote full CPU job 6477233 passes
61/61; job 6477192 was cancelled as a zero-output `cpu-0001` infrastructure
stall. GPU smoke 6477245 proves exactly one of two stream nodes carries the
tag and survives reset/reprioritization/replay. Exact job 6477252 and trace
export 6477253 are queued.
Exact job 6477252 found 94 lane-0 anchors / 470 selected kernels after resetting
all 1,889 tagged lane-0 kernels, but measured 26.223880 ms E2E and 26.1863071
ms strict CUDA. It is slower than both job 6474505 and neutral two-stream job
6475991, so per-layer node priority is rejected. Export 6477253 failed only
because its container could not reopen the transient Slurm spool path;
replacement 6477539 passes the durable script path explicitly.
Export 6477539 proves 58,848/3,678 model calls/nodes on exactly streams 28/164,
both used by DeepEP, but it adds 0.132855 ms model span and 0.136505 ms
attention span versus neutral exact-two-stream job 6475991. The active
replacement removes node priority, restores neutral streams, and seeds the
stagger once: lane 1 waits, lane 0 releases after its first QKV compute and
before publication, then lane 1 resets the ticket. Local contract passes 15/15
and remote CPU job 6477689 passes 60/60.
GPU replay smoke 6477708 completed `0:0` on a real GPU: nine replays ended
with `turn=0`, `timeout=(0,0)`, `produced=9`, and `observed=45`, allowing the
candidate to advance to exact performance and dependent durable trace export.
Exact job 6477716 completed `0:0` but is rejected: 26.222327 ms E2E,
26.1675189 ms strict CUDA, 114.406323 E2E TPS/GPU, 15/15 samples, and zero
outliers. Export 6477721 completed `0:0`; its local rank-1/rank-5 SQLite traces
prove 564 attention nodes on two streams and 3,680 model nodes on exactly two
streams (32/164), both used by DeepEP. The startup seed did not actually wait:
lane 1's 16 startup `wait_turn_kernel` calls average only 1.938 us. CUDA Graph
scheduled lane 0's release first, so the candidate added two nodes without
establishing the intended stagger. Productive model overlap rose to 1.564894 ms
median, but the attention/model spans are +0.487737/+0.489323 ms versus aligned
job 6474505 and the MB2 ping-pong gate fails.
The active directional replacement makes only persistent model lane 1
high-priority so its one-warp startup waiter becomes resident before lane 0's
release; attention and all other streams remain neutral. Remote CPU job 6478123
passes 60/60. Smoke 6478213 replayed correctly but its cross-stream CUDA-event
timing query was unsupported; replacement 6478515 uses a test-only device entry
probe and passes nine replays with `entered=9` and
`release_saw_entered=45`, proving waiter-before-release every time. Exact job
This advanced to exact job 6478604 and dependent durable export 6478605.
Exact job 6478604 completed `0:0` but is rejected at 26.1627855 ms E2E,
26.0976459 ms strict CUDA, 114.666689 E2E TPS/GPU, 15/15 samples, and zero
outliers. Export 6478605 produced intact local SQLite databases before its
signal-terminated teardown; both pass `PRAGMA integrity_check`. The model is
58,880 calls / 3,680 nodes / exactly streams 28/164, both carrying DeepEP.
Lane 1's startup wait still averages only 1.824 us: priority makes it enter just
before release, but graph root lane 0 has already executed QKV compute. The
active correction roots only the model graph on lane 1 so the waiting branch is
scheduled before lane 0 starts; attention remains rooted on lane 0.
The waiter-root candidate passes local contract 15/15 and remote CPU job
6479023 at 60/60; pending pin job 6479014 was cancelled because `cpu-0015` was
fully allocated. GPU smoke 6479044 completes `0:0` with nine clean replays and
5,007,285 waiter cycles, directly proving the wait spans lane-0 work. This
advanced to exact job 6479079 and dependent rank-1/rank-5 export 6479081.
Exact waiter-root job 6479079 completed `0:0` but is rejected at 26.158717 ms
E2E and 26.2745685 ms strict CUDA mean (26.115999-ms median), with 15/15
samples and zero outliers. Export 6479081 completed `0:0`; integrity-clean
local traces prove 58,880 calls / 3,680 nodes on exactly streams 28/164, both
carrying DeepEP. The full graph still delays the lane-1 waiter until just before
release (2.100-us mean), despite lane-1 rooting. The active correction uses an
in-kernel waiter-entry publication plus a one-time lane-0 entry gate, which
guarantees waiter residency before first QKV compute while adding only one node.
The guaranteed-entry candidate passes local contract 15/15, remote CPU job
6479470 at 60/60, and GPU smoke 6479372 with nine replay-stable iterations
(`turn=0`, `entered=0`, no timeout, `observed=45`). Exact job 6479509 is
rejected at 26.167521 ms E2E and 26.1245205 ms strict CUDA mean
(26.119584-ms median), 15/15 samples, zero outliers, and 114.645938 E2E
TPS/GPU. Export 6479521 proves rank 5 has 58,896 calls / 3,681 nodes on exactly
streams 28/164, both carrying DeepEP; productive overlap is 1.377312 ms median
but the bidirectional ping-pong gate still fails. The waiter itself now spans
30.064 us, but the first lane-0 entry gate is exposed for about 383.9 us on
every measured replay: CUDA Graph schedules lane 0's bootstrap and gate before
lane 1's bootstrap/waiter. The active correction enqueues the lane-1 waiter
before either DeepEP bootstrap, preserving the same two streams/buffers while
removing that exposed scheduler delay. If needed afterward, fold the 93
steady-state expert admission waits into their existing dispatch epilogues.

Waiter-first jobs 6479731/6479735 are rejected at 26.2094891-ms strict/E2E
CUDA mean (26.166656 median), 114.462361 TPS/GPU, 15/15 samples, zero outliers,
and +0.477315 ms versus the aligned target. Exported rank 5 is exactly streams
28/164 with both DeepEP lanes, but bidirectional FFN ping-pong fails. The active
replacement moves the accepted cadence into existing kernels: QKV publication
releases a local signal before its data transfer, steady-state lane admission
waits inside fused add-RMSNorm+QKV-FP8 quant, and every deferred combine
releases a one-CTA wait fused into the peer dispatch-copy epilogue. It therefore
targets exact two-stream/buffer reuse, 186 fewer norm/quant nodes, no standalone
cleanup waits, a 3,400-node model graph, and symmetric FFN cleanup/expert
overlap. Local static contract passes 15/15, remote CPU job 6480201 passes
60/60, and fresh SM100 extension compile job 6480387 passes. GPU smoke 6480416
failed before kernel launch because its wrapper omitted a writable FlashInfer
cache. After fixing the reusable wrapper, replacement GPU smoke 6480441 passes
all three gates in 6:59: fused admitted add-RMSNorm/FP8 is bit-exact across nine
two-stream graph replays, four-rank fabric signaling/data transfer passes, and
the prearmed DeepEP admission ticket is consumed and reset. Exact job 6480443
completes 15/15 with zero outliers and proves the intended 3,400-node model
graph, but is rejected at 26.2396121-ms E2E (26.211742-ms median), 114.330959
TPS/GPU: +0.507438 ms/+1.972% versus the aligned 25.7321745-ms target. Trace
export 6480445 is running to localize that remaining critical-path loss;
alignment is intentionally withheld from the known performance miss.

### Current terminal update (2026-08-23 PDT)

This update supersedes the older in-progress chronology below. The focused
128K/b6/A1:F1/ATP1/FEP4 goal is complete and Pareto remains frozen. Fresh
same-code E2E results are legacy job 6470598 at 30.6157295 ms,
32.6629486 TPS/user, and 97.9888459 TPS/GPU versus `fmha-only` job 6472328 at
25.9524935 ms, 38.5319430 TPS/user, and 115.5958290 TPS/GPU. The new mode is
15.2315% lower latency and 17.9684% higher throughput, with 15/15 eligible
samples and zero outliers in both runs.

Official new-mode alignment job 6472537 completed `0:0`: 24 prompts / 408
tokens, top-1/top-10/top-100 all 100%, average/max vLLM rank 1.0, and valid
attention/model trace audits with 16 graph launches each. Legacy official
scoring from job 6470699 has the same perfect summary; its scorer artifact is
valid even though an obsolete placement-blind post-auditor made that Slurm
record exit 1 after scoring. Manual legacy rank-1 trace validation passed.

The final code keeps generic flattened fan-in publication and adds exact
compile-time single-edge/single-ready QKV and O publication variants for A1.
Remote CPU job 6472263 passes 60/60; SM100 compile/fabric job 6472264 and trace
export job 6472538 complete `0:0`. The exact template variants are present in
the capture. The current source and runs use no source-hash manifest:
`control/final_candidate_source_manifest.sha256` is absent locally/remotely,
and focused jobs use `FASTAFD_ALLOW_DIRTY_SOURCE=1`. See
`CODEX_PROJECT_qwen3_ffn_handoff.md` for the concise final evidence and paths.

The corrected historical no-new-mode sweep row for this nominal case is
33.6810167 ms / 29.6903152 TPS/user / 89.0709455 TPS/GPU, but that row uses an
attention CUDA critical-range mean rather than E2E. Use job 6470598 for the
final apples-to-apples E2E control. The literal optimized target was `<26 ms`;
the historical accepted reference was 25.967383 ms.

Follow-on exact-A1 FEP8 job 6473036 completed `0:0` in 13:47 on four contiguous
trays / 16 active GPUs. Its E2E result is 25.391392 ms, 39.3834257 TPS/user,
and 118.1502771 TPS/GPU; strict CUDA mean is 25.3729383 ms with 15/15 samples,
zero outliers, and the expected 564/3,676 attention/model graph shapes. All 32
raw Nsight reports across legacy EP4, new-mode EP4, and new-mode EP8 are copied
under `scratch/qwen3_ffn_overheads_20260820/nsys_three_case_collection_20260823`.

### Older experiment chronology

The validation base is remote branch `generic-two-lane-final` at campaign head
`5d07865` and original runtime boundary `5bc6b1a` under the canonical Lustre
`qwen3_ffn_overheads_20260820/source_two_lane_final` workspace.
That tree currently contains the tracked dirty grouped-fan-in correction frozen
by the focused validation source manifest.
It accepts every `1 <= num_microbatches <= batch_size`: decode N=1 uses one
active stream, and decode N>1 uses exactly two parity-reused streams/slots for
FMHA, FFN, and legacy adapters. Eager prefill bounds the same resources with a
two-live-round wavefront. Each FMHA boundary has only Q/K/V payload+ready and O
payload+ready across GPUs; all reuse clearing and turn ordering is GPU-local.

The grouped-fan-in candidate passes 59/59 pinned remote tests; the final
harness's 15-test GB200 contract gate also passes locally. Its CUDA transport
build, four-GPU fabric smoke, and unchanged 8K prefill-reference smoke pass.
Final grouped focused jobs measure 26.1905406667 ms for A1:F1 and
27.8508985333 ms for A2:F1 at 128K/b6/MB2. Both use one attention and one
grouped model graph per trace step. Both execute exactly 188 gate/up and 188
down expert GEMMs per replay (94 layers * two FFN rounds). A2's dominant
kernel-duration increases are expert gate/up +1.334426 ms (+31.58%) and expert
down +0.665562 ms (+26.09%); dense QKV/O are effectively unchanged. The expert
psum scheduler streams weights per active expert, so merging six tokens rather
than three can broaden expert coverage even with skinny M. This is not
serialized FFN launching or host synchronization, and overlapping kernel sums
must not be equated directly to the E2E delta.
Unchanged official alignment job `6459889` completed `0:0` on the A2 run's own
48-prompt / 816-token sample: top-1/top-10/top-100 are all 100% and average/max
vLLM rank are 1.0. Its representative attention/model ranks contain 16/16
graph launches, proving grouped replay remains intact. A fresh exact 30-point
campaign root `fmha_only_optimized_attention_128k_pareto_grouped_fanin` was
started only after those gates, then stopped when its first large
grouped-prefill points exposed an unbounded transport-publication grid. Jobs
`6460176` and `6460178` failed r11 b5/b4 and r14 b4 after every attention rank
hit the intentional 300-second Q-ready timeout; model GPUs remained at 100%.
Each publication CTA executes the required system fence, so the fan-in-sized
128K prefill grid created hundreds of thousands of fenced CTAs before the sole
ready release. All eight GPU/controller jobs
`6460175/6460180/6460176/6460178/6460220/6460181/6460177/6460179` were stopped.
The bounded-publication correction caps each QKV/O launch at 1,024 CTAs while
retaining grid-stride payload coverage and the same one-ready protocol; A1/A2
decode grids are only 24/48 CTAs and therefore unchanged. Compile and direct
transport/prefill stress gates pass, including 1,600 large bidirectional
publications and 94 layers across a 128K prompt. Full-runtime A1/A2 jobs
`6461502/6461503` nevertheless failed asynchronously in prefill. The accepted
candidate retains each materialized batch in its in-flight handle
until the existing completion event is synchronized, closing a cross-stream
allocator-lifetime hazard without adding synchronization. Remote tests pass
59/59. Initial jobs `6462635/6462636` failed preflight before `srun` because
their retired `/home` EP-venv path was absent. Replacements `6462757/6462758`
exposed the migrated EP overlay's stale `/home` base-runtime `.pth` before Ray
startup. The one relocation record now names the canonical Lustre base and all
pinned packages resolve; both failed records are archived/released. Fresh
manifest-backed `short` jobs `6462898` (A1:F1) and `6462899` (A2:F1)
completed `0:0` at 26.0990376667 and 27.6199447333 ms: 0.35% and 0.83%
faster than the accepted grouped baseline. Both retain all 15 strict samples
with zero outliers and the exact one-attention/one-model graph shape. The
post-gate dead-code/chain-control cleanup passes 15/15 locally and 59/59 on
the synchronized canonical source; its updated 19-file manifest verifies in
both locations. Fresh official A2 alignment job `6463085` completed `0:0` with
top-1/top-10/top-100 all 100% and average/max rank 1.0; exact grouped graph and
188+188 expert-GEMM counts are preserved. The second fresh exact-30 root is
`fmha_only_optimized_attention_128k_pareto_grouped_fanin_bounded_publication`.
Initial heads are `6463783` (t18), `6463786` (t16), `6463793` (t12), and
`6463800` (t15), exactly four `short` jobs. D's start controller hit the known
`cpu-short` submit cap after creating `6463800`; recovered dependency
controller `6463801` and the durable t15 record reuse that head. The first
durable case, r15/b3 from `6463786`, was structurally valid but
regressed to 41.520687 ms versus 22.672863 ms prior (+83.13%). All eight
heads/controllers were canceled immediately; two interrupted claims were
recovered and the immutable pool is frozen at 1 completed, 29 pending, 0
claims, 0 failures. Nsight attributes the critical increase to Q/K/V
publication: 11.251208 ms/replay at A15 versus 2.219528 ms at A2. Although the
grid covered the total merged payload, each source edge restarted from the
same logical task zero, serializing useful copy work onto the same
low-numbered CTAs. The current candidate flattens source-edge task spaces
within the existing Q/K/V kernel; it retains the same payloads, fences,
counter, and sole ready publication. Local 15/15 and remote 59/59 tests pass,
and a fresh SM100 compile plus 8,192-row fabric job `6464689` pass. Standalone
plan attempt `6464726` failed before claim/model launch because its copied row
kept non-contiguous `rerun_index=14`; the index-only correction validated on
both workspaces. Exact r15/b3 job `6464940` then completed `0:0` with all 64
reports and the exact 564/3,676 attention/model graph shapes. It fails the
performance gate at 40.392720 ms E2E and 40.489008 ms strict CUDA, 78.15% and
79.93% slower than the prior optimized-attention fields. Flattening reduces
Q/K/V publication 17.32%, from 11.251208 to 9.302771 ms/replay, but E2E improves
only 2.72%. The residual has 240 fenced publication CTAs, 15 serial destination
ready releases, 33.50--35.01 ms model graphs, and normally 39.98--41.80 ms
attention graphs. Validation evidence is isolated under
`fanin_publication_flatten_validation`; never resume the stopped exact-30 pool
or create a new pool from this candidate. Resume from
`CODEX_PROJECT_qwen3_ffn_handoff.md`.

The exact 30-point optimized-attention 128K campaign was intentionally stopped
on 2026-08-22 after four r11 cases exposed invalid model-side scheduling. The
ratio is A11:F1: one logical FFN worker (EP4 over four ranks) serves eleven
attention workers. The base added-mode path incorrectly launched a complete
FFN graph once per attention source, producing eleven serialized ~26.7-ms
launches per decode step. The candidate now sends one grouped model command per
FFN DP, combines every assigned attention source within each microbatch, and
executes exactly N FFN rounds per layer. Focused latency and unchanged official
alignment now revalidate this schedule. All eight abandoned campaign
GPU/controller jobs were canceled;
the pool is clean at 4 diagnostic completions, 26 pending, 0 claims, 0 failed.

The initial apparent 512-token hang was cold per-rank extension compilation
followed by CUDA `INVALID_HANDLE` during high-fan-in transport setup, not
inference. Budgets 8192/4096/2048/1024 obscured a cold-node cgroup OOM during
Ray actor launch. Head `7e4c889` restores the proven 512 contract, shares
flock-protected builds per node, and
co-allocates each attention rank's Q/K/V+ready views plus each model rank's
O+ready views. Thus the single logical FFN worker's r11 ranks import 11 peer
attention arenas instead of 33 typed mappings; this does not mean there are 11
FFN workers.
Job `6454790` then proved every Ray node started before actor launch exhausted
the cgroup. Head `31d90c4` capped Ray at 16 logical CPUs but incorrectly fixed
the object store at 8 GiB: job `6454957` showed Pyxis exposes roughly 2 GiB to
Ray's memory detector, so startup rejected that value before CUDA work. Head
`4b32eab` keeps the CPU cap and restores cgroup-aware automatic object-store
sizing. Job `6456127` then exposed the underlying cause: nested submission had
carried the CPU controller's `SLURM_MEM_PER_NODE=2048` into the exclusive GPU
bundle, constraining its Pyxis step to 2 GiB despite a 900-GiB/node allocation.
Head `8088816` added `#SBATCH --mem=0`, which correctly gave the GPU allocation
920 GiB/node but did not override the stale variable for its inner Pyxis step:
job `6456229.0` still requested exactly 2 GiB/node and had 30 OOM kills. Head
`af168f3` passes `--mem=0` directly to that `srun` and pins both allocation and
step options; the full suite passes 53/53. The failed chain is archived under
`state/chain-attempts/inner-step-memory-export-6456228-6456230`, the pool is
clean at 30 pending, and controller `6456321` submitted tray-12 GPU job
`6456325` plus dependency controller `6456326`. Its `.0` step proved the
full-memory fix at 11,040 GiB, and all 48 workers passed the coalesced-arena and
ready gates. However, healthy 100%-utilized 512-token prefill did not reach
decode before the 30-minute case watchdog. The attempt is archived, the pool
is clean, and head `94a3f72` makes the chain require/propagate an explicit
prefill budget. Controller `6456735` submitted tray-12 job `6456738` with
8,192 tokens plus dependency controller `6456739`. All 48 workers passed the
transport/ready gates and both role types remained at 100% utilization without
runtime errors, but the prefill again exceeded the inherited 1,800-second case
watchdog. That attempt is archived under
`state/chain-attempts/fmha8192-prefill-watchdog-6456735-6456739`; the pool is
clean at 30 pending. Head `5d07865` keeps the general 1,800-second default,
raises the validated ceiling to 3,600 seconds, explicitly propagates the
campaign timeout through every chain edge, and derives the terminal audit's
expected job count from the 11 distinct tray groups in the plan instead of the
incorrect hard-coded 22. Its full pinned suite passes 53/53.
Controller `6457417` then launched the diagnostic tray-12 run. Four completed
r11 cases exposed the per-source model-graph serialization described above;
all GPU and successor jobs were canceled, in-flight claims were recovered, and
no campaign jobs remain. Details and archived attempts are in the topic memory.

Do not stage locally until the 30-case terminal audit passes. At final cleanup,
stage and commit only goal-related changes; leave the user's unrelated
`CODEX_PROJECT_RATIO_EP_SWEEP.md` modification untouched.

## Hardcoded attention/AG placement optimization (2026-08-17)

Implemented the two-site GB200-only policy from
`scratch/attention_sm_contention_optimization_20260817/README.md`.
`attention/trtllm.py` now hard-fails outside FlashInfer 0.6.6 or NVIDIA GB200
with 152 SMs and supplies an SM-count hint of 120 only for actual decode query
batch 8, otherwise 128. The M2N AG launcher now hard-fails unless it has 24
SMs divisible by 8 and uses cluster dimension 8; EG remains cluster 2.
OCI-HSG short-QoS job 6263396 tested A15:F1/ATP1/FEP4, uniform 128K, capacity
batch 6 (3+3 MB), page 64, graph bucket 3 on 16 nodes/64 GPUs with a 30-minute
limit. It completed 0:0 in 19:29, returned all 360x17 tokens, and every one of
60 attention ranks captured and replayed the same warmup plus 15 target steps.
Strict v14 CUDA-span extraction retained 15/15 with zero outliers: 30.752484
ms, 0.193086% range, and 182.912052 TPS/GPU. The canonical corrected old-code
row was 36.466635 ms / 154.250592 TPS/GPU, so the patch improved latency
15.6695% and TPS/GPU 18.5811%. Nsight proves page-64 FMHA grid 10x4x3 (120
CTAs), cluster 10, plus M2N AG grid 24/block 384/smem 99,456/cluster 8. All 64
reports (114,980,514 bytes), representative SQLite, strict metric, result, and
Slurm logs are local under
`scratch/attention_sm_contention_optimization_20260817/job_6263396/`. The
broader 8K/b16 and 8K/b96 regression gates were not requested/run.

The follow-on optimized-attention 128K near-Pareto stage is under
`scratch/attention_sm_contention_optimization_20260817/near_pareto_ep4_128k_sweep/`.
It selects corrected old-code points less than 5% below the linearly
interpolated 128K AFD Pareto curve at the same TPS/User, then retains only
ATP1/FEP4 and 8--18 allocated trays. The exact selection has 45 cases; the
execution pool has 44 because A15:F1/FEP4/b6 is already satisfied by job
6263396. The copied controls are only the prior online worker's direct
dependency closure, including integrated v14 corrected timing. Manual short
submissions kept at most four jobs live and interleaved tray counts in order
8,18,9,17,10,16,11,15,12,14,13. All used `short` QoS and explicit two-hour
limits. The terminal pool on 2026-08-17 has 44 completed, 0 claims, 0 failed,
and 0 pending. Productive jobs 6265474, 6265475, 6265476, 6265477, 6266233,
6267220, 6268394, 6268670, 6269816, 6271441, and 6271979 all completed 0:0;
the initial t11 job 6267948 hit `NODE_FAIL` before measurement and was
recovered/retried without a case failure. All 44 fresh v14 metrics passed an
exact warmup-plus-next-15 target-batch audit, retaining 11--15 samples with at
most four outliers; worst dominant-range and max-versus-median percentages
were 9.575258% and 9.329831%, below their 10% limits. Fresh TPS/GPU gains over
corrected old code range from 7.592974% to 19.840410%, median 10.568248%; the
combined 45-point median including job 6263396 is 10.838648%.
`OPTIMIZED_RESULTS.csv` is the complete result list. A minimum local copy keeps
all metrics, completion records, compact results, and Slurm logs. The raw
remote tree remains in place; `REMOTE_NSYS_TRACES.tsv` inventories its 2,304
Nsight reports (3,859,730,540 bytes) without copying them. Full execution and
artifact details are in the stage README.

The next optimized-attention stage is
`scratch/attention_sm_contention_optimization_20260817/near_pareto_ep4_32k_64k_sweep/`.
It applies the same interpolated-curve rule to ATP1/FEP4 at 32K and 64K and
contains 67 fresh cases: 15 at 32K and 52 at 64K. Combined tray counts are
`6:5, 7:6, 8:7, 9:5, 10:4, 11:7, 12:6, 13:7, 14:6, 15:5, 16:5, 17:4`.
All 67 local topology/capacity dry-runs passed; 32K uses the native model
profile with nominal max batch 25 and 64K uses the pinned YaRN-2.001953125
profile with nominal max batch 12. The OCI-HSG sweep completed on 2026-08-18
with 67/67 accepted, 0 claims, 0 failed, 0 pending, and no focused live jobs.
Twelve productive jobs covered trays in manual interleaved order
`6,17,7,16,8,15,9,14,10,13,11,12`; every one completed under `short` QoS
with an explicit `02:00:00` limit, and the focused roster never exceeded four.
One transient stale-port failure on t7 and one strict dominant-range rejection
on t11 were automatically released and passed on immediate retry. A typoed
t12 source-revision export in job 6284559 was caught and canceled at 0:00
before allocation; corrected job 6284562 completed all six t12 cases.

The terminal audit joined and hash-verified all 67 plan rows, v14 metrics,
atomic completion records, and compact results. Every capture proves one
warmup plus 15 target steps; retained ranges kept 13--15 samples with at most
two outliers, worst range 7.749333%, and worst max-versus-median 7.442592%.
All 52 64K points gained 4.984148%--21.238391% TPS/GPU over corrected old code
(median 9.586616%). At 32K, b2 gained 1.453689%--4.596336%, b8 gained
6.127293%--8.206497%, and all eight b25 points were essentially flat/slightly
lower at -0.554308% to -0.106850%; the 15-point 32K median is -0.106850%.
The overall 67-point median gain is 8.246076%. A minimal local copy contains
237 files/about 3.7 MB: metrics, completion/attempt records, compact results,
and logs. `REMOTE_NSYS_TRACES.tsv` indexes 3,040 remote Nsight reports totaling
5,039,799,171 bytes; those raw reports remain remote and were not copied.

The standalone Pareto dashboard at
`scratch/afd_baseline_pareto_2d_20260803/afd-baseline-user-tps-vs-tps-gpu-pareto.html`
now overlays optimized-attention Pareto frontiers for all three completed
long-context stages: 15 rows / 6 frontier points at 32K, 52 / 29 at 64K, and
45 / 20 at 128K. All 112 optimized measurements are rendered as dedicated red
FEP4 points, including points no longer on the optimized frontier; the same red
uses thicker lines and outer rings for the frontier and cannot collide with the
original AFD ratio colors or baseline EP colors. The HTML embeds the complete
data, styling, and JavaScript
in its `srcdoc`; unused CDN script tags were removed, and a resource audit
found no external file or network dependency. The editable source remains the
adjacent `.fragment` file, but recipients need only the `.html` file.

The original-attention FEP8 near-Pareto selection was derived on 2026-08-18
from the same canonical 1,981-row corrected AFD CSV, using the exact prior
rule: per-ISL AFD Pareto frontier over all FEPs/A:F ratios, linear
interpolation at the candidate's TPS/User with endpoint clamping, and strict
gap `<5%`. Filtering ATP1/FEP8 yields 57 cases: 23 at 32K, 19 at 64K, and 15
at 128K. Combined allocated-tray counts are
`6:1, 8:6, 10:8, 12:7, 14:12, 16:13, 18:10`. Per-ISL counts are 32K
`8:3, 10:3, 12:4, 14:4, 16:5, 18:4`; 64K
`6:1, 8:2, 10:3, 12:2, 14:4, 16:4, 18:3`; and 128K
`8:1, 10:2, 12:1, 14:4, 16:4, 18:3`. This is a read-only selection from
original performance and remains the source cohort for the completed
optimized-attention rerun documented below.

The focused optimized-attention FEP8 execution stage is
`scratch/attention_sm_contention_optimization_20260817/near_pareto_ep8_32k_64k_128k_sweep/`.
Restricting the original 57-case selection to 8--18 allocated trays removes
only the 64K/t6 point and leaves 56 cases: 15 at 128K, 18 at 64K, and 23 at
32K, with tray counts `8:6, 10:8, 12:7, 14:12, 16:13, 18:10`. Its copied
corrected-timing online worker has an explicit ISL-descending mode, so every
same-tray allocation exhausts 128K before 64K before 32K; focused failures are
released for one immediate retry and two failures stop the allocation. All 56
local topology/capacity dry-runs passed. The manually submitted short-QoS,
two-hour jobs were t8 `6294934`, t18 `6294935`, t10 `6294936`, t16 `6294938`,
t12 `6295552`, t14 `6295690`, the t14 continuation `6297349`, and the t16
continuation `6297462`. The final pool has 56 completed, zero claims, zero
pending, and zero failed; every accepted metric passed the corrected v14
target-batch and timing gates. Scheduler accounting proves at most four live
focused jobs, no overlapping same-tray allocations, and `02:00:00`/`short`
for every job. Per-tray completion order is uniformly 128K then 64K then 32K.
New-vs-original TPS/GPU gains are 4.521551%--14.964773% at 128K (median
5.811197%), 4.627827%--13.007518% at 64K (median 10.882705%), and
-18.836828%--8.710280% at 32K (median 6.236929%). The 3.1-MiB local evidence
copy deliberately excludes raw Nsight reports. `OPTIMIZED_RESULTS.csv` has 168
audited rows in the prior schema: 112 earlier FEP4 plus all 56 new FEP8 rows.
The standalone Pareto HTML includes all 56 new FEP8 measurements as red
optimized-attention points: 23 at 32K, 18 at 64K, and 15 at 128K. Each ISL's
red optimized frontier is recomputed across its prior FEP4 and new FEP8 points.
The file remains self-contained, has no external resource dependency, and
retains the original 2208-pixel maximum width.

The focused 128K A1:F1/ATP1/FEP4 batch-7 memory experiment is under
`scratch/attention_sm_contention_optimization_20260817/batch7_memory_tuning/`.
It raises the attention KV memory ratio from 0.82 to 0.90, uses a conservative
page-aligned preflight capacity of 921,280 tokens per TP1 attention lane, and
requires the observed-capacity gate to prove batch 7 is the exact
scheduler-admissible maximum before sampling. Local and remote preflight
passed for two trays/eight active GPUs, 4+3 microbatches, graph bucket 4, and
the standard warmup plus 15 target steps. OCI-HSG job `6304727` is submitted
to `short` with a two-hour limit and the known bad node families excluded; its
run root is recorded in the stage README.

Job `6304727` completed `0:0` in 12:01 and proves the 0.90 setting supports
128K batch 7. Every attention worker exposed 14,396 pages / 921,344 tokens;
the scheduler needed 917,995 tokens and independently classified batch 7 as
the exact maximum. All 28 requests returned 17 tokens, and every attention
rank replayed warmup step 1793 plus measured steps 1794--1808 with 4+3 real
microbatches in padded graph bucket 4. Strict v14 timing retained 15/15 with
zero outliers: 34.347090 ms, 0.559860% range, and 101.900918 TPS/GPU. The
minute-sampled attention-memory high was 182,829 MiB. Eight compact evidence
files/about 42 KiB are local; raw Nsight reports remain remote.

The same tuned-memory contract was expanded to 128K A16:F1 and A17:F1 through
`CASE_A16_F1_EP4.csv` and `CASE_A17_F1_EP4.csv`. Original A16 job `6323330`
failed after 08:52 during attention-rank-0 initialization: all 64 attention
workers had already proved 14,396 pages / 921,344 tokens and exact batch-7
capacity, but the first worker exhausted device memory while allocating its
two MegaMoE M2N symmetric buffers. The layout consumes about 2.400 GiB per
buffer at A16 and 2.562 GiB at A17. No samples or metric were produced; compact
failure evidence is local in `batch7_memory_tuning/job_6323330/`.

The reusable runner now supports a validated opt-in exact-page override. The
fixed A16/A17 retries pin 14,344 pages / 918,016 KV tokens, only 21 tokens above
the observed 917,995 batch-7 scheduler requirement. This releases about 611
MiB per attention GPU versus 14,396 pages while retaining the exact batch-7
gate. Original pending A17 job `6322906` was canceled before allocation. Fixed
short-QoS jobs `6327216` (A16, 17 trays) and `6327228` (A17, 18 trays) were
submitted with two-hour limits after local and remote validation; their run
roots are recorded in the stage README.

Source staging on 2026-08-19 removed the active `codex_scripts/` tree. Reusable
OCI-HSG controls are now under `scripts/experiments/afd/oci_hsg/`: the latest
Qwen3 case runner with exact-page support, its parallel preset, the same-tray
sweep worker and atomic pool, v14 CUDA extraction helpers, long-context model
profile builder, generalized vLLM baseline runner, and the older pinned
Qwen3/MiniMax reproduction interface. Case plans, manifests, submit wrappers,
node exclusions, logs, traces, and results remain task-local and are excluded
from Git with `scratch/`; `.DS_Store` is also ignored. The promoted pool and
extractor no longer contain a case-ID-specific outlier exception and instead
enforce each plan row's retained-sample/outlier contract. Local validation
covered shell syntax, AST parsing for all 17 changed/promoted Python files,
A16/A17 exact-page dry runs, a 56-case pool initialize/summary/claim smoke
test, and `git diff --check`. The promoted controls were synced as untracked
files to the OCI-HSG source checkout; remote shell parsing and source-manifest-
backed A16/A17 validate-only runs passed, as did a non-divisible AFD-DP topology
probe. The remote checkout was deliberately not advanced while pending jobs
`6327216`/`6327228` still depend on its existing head and manifest.

## Critical AFD profiling validity warning

**Do not use any legacy AFD latency or TPS value in
`QWEN3_ALL_SWEEP_RESULTS.csv` as a target-batch performance measurement.** The
previous AFD Nsight profiling is wrong for that purpose: it did not prove that
the captured decode steps were running the requested full-resident target
batch and commonly captured the end-of-decoding tail instead. This invalidates
all 1,943 legacy AFD rows for corrected Pareto/performance decisions; it does
not invalidate the separately measured vLLM baseline rows.

The corrected Pareto campaign finished 116/118 selected cases; the final
reduced correction campaign completed all 1,981/1,981 cases in
`FIRST_SWEEP_CASES.csv` successfully. Every short and normal worker claimed from
this same canonical pool with no global mode or FFN-EP boundary: retained EP2,
regular EP4, EP8, EP16, and EP32 work may run concurrently. Keep three short
and six normal jobs live with tray counts unique across both QoS classes,
preferring the smallest available tray count. At the terminal 2026-08-15 02:30
PDT audit, the pool had 1,981 completed, 0 claims, 0 failed, and 0 pending. The
accepted per-EP totals are EP2=235, EP4=1,059, EP8=448, EP16=178, and EP32=61;
the final dashboard now uses those 1,981 accepted v14 metrics as its sole AFD
dataset, with no legacy/corrected comparison labels, point highlighting, or
duplicate AFD Pareto. The standalone report is
`scratch/afd_baseline_pareto_2d_20260803/afd-baseline-user-tps-vs-tps-gpu-pareto.html`
(SHA-256 `5bbea6af83830cdc02f59f63f49f1bd2b1561404484f4645fea6599c3b581b65`).
In the combined panels, unchanged small AFD points are filled A:F-colored FEP
shapes while baseline points are hollow EP-colored circles, making the two
systems distinct without enlarging non-Pareto markers.
Its same-schema 17-column CSV companion is
`scratch/afd_baseline_pareto_2d_20260803/afd-regular-all-1981-cases.csv`
(1,981 data rows; SHA-256
`a487e8309e95c9d7883d66f5a51e9f471903765e382b735ebe2f241fc749acaa`).
Regenerate it with
`next_corrected_afd_sweep_20260804/control/export_afd_results.py
--require-complete`;
the exporter rejects incomplete, non-regular, hash-drifted, or identity-drifted
input. On 2026-08-17, selected corrected-run AFD Nsight reports were copied
back without checksums, preserving their original filenames: 128K/b6
A8:F1/EP8 has 72 reports (121,721,200 bytes) under
`scratch/afd_baseline_pareto_2d_20260803/128k_b6_nsys_correction_sweep_20260806/afd_a8_f1_ep8_b6/`;
1K/b761 A1:F1/EP4 and A2:F1/EP4 have respectively 8 reports (10,040,285 bytes)
and 12 reports (18,032,239 bytes) under
`scratch/afd_baseline_pareto_2d_20260803/1k_max_batch_nsys_correction_sweep_20260806/`.
The A2:F1 directory was already complete, so plain rsync transferred no files
for that case. A 2026-08-17 configuration audit joined the canonical 1,981
completed records to every `afd-result.json`, `afd.log`, `driver.stdout`, and
input manifest; it excluded the 39 successful irregular cases removed from
final scope. All 1,981 used AFD serve over Ray, attention TP1 and MLP TP1,
full-world FEP2/4/8/16/32, DeepEP plus DeepGEMM, the non-default
`MINISGL_AFD_MOE_BACKEND=megamoe_m2n`, two microbatches, exact
`ceil(batch_per_lane/2)` decode-graph buckets, TensorRT-LLM attention, naive
cache, page size 64, memory ratio 0.82, AFD prefill budget 512, device-comm
SMS=4, max-new-tokens=17, and explicit max sequence length `ISL+64`. Every
case kept both attention/model decode graphs and overlap enabled. The 1,567
cases through 32K used the native 40,960-position model config; 199 64K cases
used YaRN factor 2.001953125/max 65,600 and 215 128K cases used factor
4.001953125/max 131,136. All cases used per-worker Nsight node graph tracing,
one exact target-batch warmup plus 15 measured decode steps, and capture-exit.
Nine runner hashes occurred, but parsed performance-system settings were
constant; the meaningful runner variation was post-measurement teardown/prompt
caching (`POST_SHUTDOWN_SLEEP=20` for 1,790 cases and 0 for 191). Defaults
retained across all cases included BF16 auto-resolution, generic MoE `auto`,
PyNCCL enabled, no explicit KV-page count, 300-second distributed timeout,
8,192 max-extend tokens, and overlap enabled. A deeper 2026-08-17 source-level
auto-resolution audit used the exact tracked campaign sources. On GB200/SM100,
automatic attention selection resolves to TRT-LLM and normalizes page size 1
to 64, exactly matching the campaign. `afd_moe_runner_backend=auto` resolves
to DeepGEMM in `afd-serve`, BF16 auto matches the model/worker dtype,
automatic maximum running requests resolves to the same global prompt count
for every row, and automatic communication sizing resolves to a base 512
tokens for every row. The generic MoE `auto` field does not select the AFD
transport path: AFD workers force the generic field to `fused`, while the
separately selected `MINISGL_AFD_MOE_BACKEND=megamoe_m2n` differs materially
from its `deepep` fallback. CUDA graph sizing is the meaningful auto
difference. With the campaign topology, batch, naive/TRT-LLM cache, and
two-microbatch policy held fixed but both the graph bucket list and graph
maximum left automatic, 1,248/1,981 target batches select the same exact
per-MB bucket and 733/1,981 round upward. The 733 padded cases add 1--7 slots
per MB (median 3, mean 3.06), averaging 34.7% more per-MB graph slots and
reaching 60%; auto captures 1--51 buckets per case (median 19) instead of one.
Leaving only `cuda_graph_max_bs` automatic while retaining the explicit exact
bucket changes nothing because an explicit bucket bypasses that cap.
Conversely, leaving the bucket list automatic while retaining the launcher's
explicit cap of 256 leaves all 29 batch-761 cases without a covering target
graph, so their target decode falls back to eager; the campaign's explicit
bucket 381 remains valid despite the parsed 256 maximum. With each
long-context model profile retained, automatic max sequence length equals the
campaign's `ISL+64` only for the 199 64K and 215 128K rows; it would stay at
the native 40,960 positions for the other 1,567 rows. A full stock-default AFD
invocation is not a comparable counterfactual because the default radix cache
is rejected by `afd-serve` before launch, in addition to defaulting to a
different topology, one microbatch, memory ratio 0.9, and other workload
limits. The source-derived graph differences have not been benchmarked as a
separate performance experiment. A README/published-preset alignment audit
confirmed that the sweep retains the official Qwen3 AFD runtime backbone:
Ray, ATP1/MLP-TP1, DeepEP/DeepGEMM plus MegaMoE M2N, MB2, TRT-LLM/page64,
naive cache, memory ratio 0.82, prefill budget 512, device-comm SMS4, and
`ISL+64`. KV storage is BF16: the AFD worker fixes `torch.bfloat16` and passes
that dtype into the KV pool. There was no explicit page-count override; the
runtime derived pages from free memory at ratio 0.82. Canonical coordinator
logs show 13,116 pages x 64 = 839,424 KV tokens per TP1 attention lane in
1,978/1,981 cases (150.50 GiB of K+V storage per attention GPU), and 12,605
pages = 806,720 tokens in three cases: 32K/FEP4/A16:F1/b4, 32K/FEP4/A16:F1/b5,
and 64K/FEP4/A16:F1/b4. The sweep planner/admission contract used the nominal
839,424-token TP1 capacity for every row. The exact published anchors are
8K/b96/A7:F1/FEP4 and
16K/b48/A11:F1/FEP4. The 1,981-case campaign is not an exact README preset for
every row: it expands A:F and FEP2/4/8/16/32, captures only the exact target
graph bucket rather than each preset's multi-bucket list, generates 17 rather
than 16 tokens for its warmup-plus-15-step capture contract, and adds 64K/128K
YaRN profiles. The 2026-08-17 final cleanup archive is
`scratch/afd_baseline_pareto_2d_20260803/archives/legacy_wrong_afd_data_scripts_20260817.tar.zst`
(46,283,773 bytes; SHA-256
`b2f8481a3e98712a7cb056181d21ffa791399aa45e6c4cc5aff1f787d5537d15`).
Cleanup removed 1,943 invalid legacy result JSONs and 64,703 legacy AFD Nsight
reports from remote source trees after verifying the canonical compressed
archives. It preserved all 639 legacy baseline cases and treats the corrected
1,981-case task as an excluded root; its pre-cleanup inventory contains 85,749
Nsight reports totaling 133,864,451,070 bytes. Direct set
inclusion checks prove all 519 fixed-ISL 1K--16K focus IDs and all 60
maximum-batch IDs are in the canonical completed set. The submission and
bundle scripts pass `bash -n`; handoff/dashboard local and remote hashes match;
and no campaign job remains queued or running. A t9
EP2 case hit one attention-rank-0 CUDA unspecified-launch failure on
`nvl72132`; it was archived/released with no metric, leaving three unclaimed
t9 cases plus sixty t18 cases; every t9 case then passed, including the retry,
so EP2 is complete at 235/235. EP4 is complete at 1,059/1,059 and EP8 is
complete at 448/448. The
two stale t9 normals were canceled and five excluded t18 jobs restored the
final all-t18 roster at three short plus six normal. All nine remained pending
at the 20:00 checkpoint, with the lead short waiting on resources and the
others on priority; no alternate tray exists. The first t18 short began at
20:11 and the second at 20:32. The first allocation later failed after
`i2048-fep4-r17-atp1-b256` produced two strict 9/15 stability rejections; both
attempts were automatically archived/released with no accepted metric or
stranded claim. The third short took its slot, and excluded replacement
`6181506` took the next slot. The next two allocations also ended after two
strict retries each: EP4 b320 retained 7/15 then 8/15, and EP4 b256 retained
9/15 then 7/15. All four attempts were automatically archived/released without
metrics or stranded claims; excluded replacements `6182274`/`6182275` restore
the 3+6 roster. Both repeatedly unstable EP4 b256/b320 cases then passed their
strict retry. `6181506` later ended after EP4 b1 produced two strict 9/15
rejections; both attempts were archived/released and b1 then passed strictly
under `6182275`. Excluded replacement `6183260` restores 3+6 alongside running
shorts `6182274`/`6182275`. EP8 128K-r8-b4 then failed strict stability with a
7/15 dominant cluster; its evidence was archived and the case explicitly
released, leaving zero failed state for retry. That released case then passed
strictly under `6182274`. That allocation completed cleanly, and excluded
replacement `6184987` restored 3+6 behind standby `6183260`. Short `6182275`
then completed cleanly after 16 total successes; excluded `6185501` restored
3+6, with all nine jobs scheduler-pending at the 00:20 checkpoint. Both
failures released at 15:42 passed on retry, and EP16 is complete at 178/178.
Short t7 later hit one pre-capture attention-rank-0 CUDA unspecified-launch
failure on `nvl72169`; it was archived/released with no metric, and excluded
standby t7 `6173684` restored 3+6; that retry then passed and tray 7 exhausted.
The useless pending t7 short/normal were canceled and replaced by excluded t8
short `6174347` and t18 normal `6174348`. Active t7 completed cleanly and t8
started seven seconds later; excluded t9 short `6174953` now gives the final
even 3/3/3 distribution across t8/t9/t18. First t8 then completed cleanly, t9
started three seconds later, and excluded standby t8 `6175711` preserved that
distribution. Tray 8 then exhausted after nine more successes; its standby
exited claimless, the stale t8 normal was canceled, and excluded replacements
two t9 shorts plus one t18 normal now give the final smaller-first 5/4 split
over the only remaining tray counts t9/t18. The
live roster is running optimized shorts t7/t8 with t7 pending, and pending
optimized normals t7/t8/t9/t9/t18/t18. Short t6 completed cleanly after seven
successes and exhausted tray 6; t8 started 10 seconds later, its now-useless
pending normal was canceled, and excluded t7-short/t18-normal replacements
restored the balanced 3/2/2/2 allocation over the four remaining tray counts.
Short t8 previously ended after a strict 3/15
stability rejection followed by a stale ZMQ control-port collision; both cases
were archived/released with no accepted metric, t6 took the execution slot two
seconds later, and replacement short t8 `6170415` is queued. Short t6 completed cleanly and handed
the execution slot to t7 in 92 seconds; replacement short t6 `6168090` is
already queued. Short t7 previously completed cleanly and handed
the execution slot to t8 within one minute; replacement short t7 `6166312` is
already queued. Short t5 completed cleanly after six
successes and exhausted tray 5; its now-useless pending normal was canceled and
replaced by the balanced t8-short/t9-normal pair. A t7 attention-rank-0 CUDA
unspecified-launch failure was diagnosed, archived, and released without an
accepted metric; it is the first event on `nvl72168`, so no new family exclusion
was added. Trays 4/10/12/14/16 are exhausted; because only six pending
tray counts remain for nine slots, duplicate t5, t6, and t7 are the minimum
no-alternative exceptions. This no-boundary policy applies identically to all
short and normal jobs, not only t1 or a particular tray. The compact live state is
`scratch/afd_baseline_pareto_2d_20260803/next_corrected_afd_sweep_20260804/HANDOFF_REGULAR_EP8_EP16_EP32_20260810.md`.
The full main pool retains its phased regular-before-irregular ordering. The
isolated regular EP8/16/32 pool has a strict first tier containing the 60
maximum-batch points (one per ISL x EP x A:F topology), then continues all
other cases; within a tier it orders fixed ISL and batch descending with EP
only as a tie-break. Short workers use two hours. The required contract traces
the first
exact target-batch decode as warmup, measures the next exact 15 target-batch
decodes, retains a position-independent dominant range of at least 12 steps,
and enforces both 10% stability limits.

Every Slurm/Ray tray exposes four GPUs, but comprehensive EP2 cases with an
even A:F ratio use `2*ratio+2` active GPUs and therefore leave two padding GPUs
idle on the last exclusive tray. Exact total-GPU requests are accepted by
Slurm, but an 18-GPU dry run still reserved five physical segment nodes and
720 CPUs; it is not a shareable 4.5-tray allocation under the exclusive job
contract. EP4/8/16/32 and odd-ratio EP2 placements fill all allocated GPUs.
The larger measured idle source was teardown: all 17 audited live-job cases
spent 210--214 seconds after durable capture/sample before result generation,
and all hit the 180-second forced-shutdown path, while strict extraction took
only 2--6 seconds. The task runner now uses a hard maximum 10-second
condition-driven process-tree shutdown, no post-shutdown sleep mechanism, and
a 30+10-second outer fail-safe armed only after capture, sample, and coordinator
trace are durable. The current deployed
local/remote runner SHA-256 is
`53fc1ef6b8986dc588c89f267d2308cfa8dc3205cae130ebfc4a022127de03e0`;
the optimization is now validated using the user's full idle boundary, from
one case's durable decode/capture completion to creation of the next case's
`experiment/afd.log` (AFD/vLLM initialization start). Sixteen clean old-runner
transitions took 277--345 seconds (median 301 seconds); the first two optimized
transitions took 155 seconds on t16 and 141 seconds on t5. Both preceding cases
passed strict metrics, so the optimized runner cut the typical full handoff by
about 51% without accepting a failed result.

Prompt generation is shared across cases under `results/prompt-cache/`. The
cache key covers the pinned model/tokenizer revision, source hash, context or
irregular length layout, and total prompt count. Regular uniform cases with the
same context/count reuse one payload even when attention-DP and batch factor it
differently. Publication is atomic and lock-protected; each case hard-links the
immutable cached payload and records its own mapping/provenance manifest. A
remote cross-factor test generated once, then hit in 127 ms with identical
SHA-256 and inode. Pending Slurm jobs resolve the shared runner at case start,
so they inherit this change without cancellation or lost queue age. All seven
pending jobs were re-verified on 2026-08-14 to point to that shared bundle;
none needed cancellation or resubmission.

Standing execution rule from 2026-08-14: keep the campaign tight and add no
optional waiting or sleep. Use only short, bounded, condition-driven waits that
are required for artifact durability, process termination, or scheduler state
propagation. Do not hold a vacant slot, completed-job replacement, released
failure, or ready next case for a monitoring cadence; act immediately. Keep
the post-shutdown sleep mechanism absent, retain fail-fast validation, and do
not reintroduce a fixed post-capture delay after the required artifacts are
durable. After the
full inter-case handoff fell from a 301-second old-runner median to 112--139
seconds on current production transitions, the user set operator monitoring to
a 20-minute cadence; this changes polling only, never runner behavior.

The OCI HSG reproduction of the public FastAFD GB200 NVL72 performance claims
is complete for Qwen3 and MiniMax-M2.5 at 8K/16K. All four vLLM baselines now
have DP=EP4/8/16/32/64 tables using exact EP8+ KV-capacity ceilings and sparse
corresponding CUDA graph captures. EP16 is the best wide-EP point in every
case. See
[`CODEX_PROJECT_REPRODUCE.md`](CODEX_PROJECT_REPRODUCE.md) for the two retained
launchers, integrated wide-EP baseline support, pinned provenance, measurement
contract, and signed result evidence.

The OCI-HSG Qwen3 ISL/batch sweep is complete using the same two generalized
launchers: AFD is fixed at eight nodes and the wide baseline at DP=EP16, with
ISL spanning 1K--128K and batch following the requested 2^x / 1.5*2^x
sequence. Both modes successfully ran every planned point. Request decode-step
latency and TPS/GPU are the primary metrics; the full contract, long-context
YaRN profiles, result tables, failures/retries, hashes, and artifact evidence
are recorded in `CODEX_PROJECT_REPRODUCE.md` under "Qwen3 ISL / batch sweep".
The historical copy/paste report generator is retained only in the legacy
cleanup archive; it is no longer active reusable tooling.
At the user's request, that report's headline comparison follows the public
README-compatible convention: AFD coordinator TPS/GPU versus vLLM summed CUDA-
kernel TPS/GPU. Synchronized vLLM request latency/TPS remains alongside it as a
secondary column; the underlying sweep's predeclared request-metric contract is
unchanged.
The superseded standalone CSV/Google Sheets exports were removed on 2026-08-03;
their retained logical cases are consolidated in `QWEN3_ALL_SWEEP_RESULTS.csv`.

## Irregular request lengths in AFD decode

Code audit and a coordinator-side probe on 2026-07-17 indicate that the current
AFD runtime is designed to support mixed request lengths in one decode batch.
Requests retain independent prompt/output limits and KV positions; decode
plans carry a separate table index/start position per request; both FlashInfer
and TRT-LLM attention metadata carry per-request sequence lengths; and finished
requests are removed individually while graph padding uses dummy requests.

A CPU probe in the pinned remote `3c716194` environment used prompt lengths
5/9/13 and output limits 2/4/3. The first decode plan contained distinct start
positions 5/9/13, and the real batch shrank 3 -> 2 -> 1 as requests finished;
all scheduler/plan assertions passed. This validates the coordinator, request
lifecycle, and plan construction but is not a full GPU regression. Existing
AFD performance jobs intentionally used uniform prompt and output lengths, and
no checked-in irregular-batch end-to-end test was found. Treat current support
as strong code-level support pending one focused GPU correctness run covering
mixed prompt lengths, mixed `max_tokens`, and early EOS under graph replay.
The `/v1/chat/completions/batch` extension is explicitly homogeneous and has
one shared `max_tokens`; mixed output limits require separate concurrent
completion requests (which the benchmark client already supports).

The requested irregular-ISL performance sweep is complete with all 144/144
rows strictly audited (73 AFD + 71 vLLM) using the same two retained launchers.
Both expose explicit `uniform` and `irregular` modes; irregular mode supports
the eight requested inclusive ranges, constructs the same symmetric integer-
linspace prompt lengths on every AFD attention rank and vLLM DP=EP16 lane, and
fails before submission when the page-aware measured KV-cache requirement is
too large. Every AFD row proves that its positional batch is the total input
batch per attention lane, split into real microbatches `ceil(batch/2)` and
`floor(batch/2)` with graph bucket `ceil(batch/2)`. CUDA graph replay remained
enabled throughout. Exact semantics, sparse batch plans, regular-versus-
irregular sanity evidence, failures/retries, the isolated trace-flush-order fix
revision `c58f3040`, and result hashes are maintained in
`CODEX_PROJECT_REPRODUCE.md` under "Qwen3 irregular-ISL / batch sweep". The
irregular rows are consolidated in `QWEN3_ALL_SWEEP_RESULTS.csv`; the removed
standalone report had SHA-256
`dfe431793dc70fe8469c263507602f3f1eda37a4f010fa53253caa37bc015b4d`.
Every irregular row has distinct per-lane prompt lengths. For the 32K--128K
vLLM sanity comparison at b2/b3/b4/b6, irregular CUDA time is 55.5--59.6% of
the span from the prior uniform 32K result to the uniform 128K result,
consistent with the exact 80K mean ISL and inconsistent with silently running
either single endpoint. The sweep-scoped four-job ceiling excludes unrelated
probes submitted by other sessions via `FASTAFD_ACTIVE_JOB_REGEX`.

## Qwen3 A:F-ratio / baseline-EP ISL sweep

The follow-on exact-maximum-batch topology sweep is active. The temporary task
root has been renamed to `scratch/ratio_ep_isl_sweep_20260717_1548`, and its
reusable A:F-ratio and baseline-EP behavior is consolidated into the two major
launchers under `scripts/experiments/afd/oci_hsg/`. Baseline EP is complete at
40/40; the full matrix currently has 115/176 fail-closed accepted results. The
four-tray lane is gated on an exact 32K b25/n4 retry with an isolated,
hash-validated 300-second worker-hot-loop shutdown wait. See
[`CODEX_PROJECT_RATIO_EP_SWEEP.md`](CODEX_PROJECT_RATIO_EP_SWEEP.md) for the
current jobs, paths, invariants, hashes, and required continuation order.

## Qwen3 vLLM ISL x EP x batch sweep

The comprehensive uniform vLLM 3D sweep is complete under
`scratch/vllm_isl_ep_batch_3d_sweep_20260729`: 568/568 strict rows, comprising
448 new profiles and all 120 eligible prior overlaps, with zero errors or
pending cases. It spans ISL 1K--128K, DP=EP2--64, batch size 1, the
`2^x` / `2^x+2^(x-1)` batch grid, and exact maxima. Every new row uses the
aligned sparse CUDA-graph policy, exactly 15 profiled replays per GPU, and
first-to-last CUDA-graph wall latency/TPS. An independent full-artifact audit
hashed all 568 result files and reconstructed a historical maximum of three
queued/running goal jobs. See the task-local `CODEX_PROJECT.md`,
`report/completion_audit.json`, and `report/completion_result_hashes.csv`.

The baseline sweep also has an eight-facet 2x4 Pareto dashboard at
`scratch/vllm_isl_ep_batch_3d_sweep_20260729/report/baseline-user-tps-vs-tps-gpu-pareto.html`.
Each ISL facet contains every EP/batch point, with `User TPS = 1000 /
cuda_wall_ms` on x, audited `tps_per_gpu` on y, EP encoded by color, and the
nondominated upper-right frontier marked and connected. The editable fragment
is adjacent as `baseline-user-tps-vs-tps-gpu-pareto.fragment.html`.

The archived `QWEN3_ALL_SWEEP_RESULTS.csv` was the legacy reference-data CSV. It is the
deduplicated master table across the prior 487-row collection, the 568-row
baseline sweep, and the completed 1,824-row AFD comprehensive sweep. It retains
2,582 unique logical experiments after removing 297 repeated rows, including
all 144 irregular-ISL rows; its SHA-256 is
`1c98809da3889f81eb4d50c427c77ef45550674a693333881c172d824e6fa665`.
On 2026-08-03 all 47 superseded/temporary local CSVs and local raw/profiler
result copies were removed after remote source validation. The historical
builder and its removed input catalogs are retained only in the 2026-08-17
legacy cleanup archive; they are not active reusable tooling.

## Canonical result archive and retrieval policy

The 2,582 master rows map one-to-one to 2,582 unique remote raw-case
directories. The compact canonical-result archive completed and verified on
2026-08-03 at
`/home/shengjiel/scratch/fastafd_reproduce/archives/qwen3_all_sweep_results_2582_cases_20260803.tar.zst`;
it is 4,228,449 bytes with SHA-256
`544f3135602b6c94f975e42d59ff51d3a6e6a3108662db4dd158c6f95a3fc465`.
Its embedded `MANIFEST.tsv` maps every master row to its exact case folder,
`RESULT_SHA256.tsv` hashes all 2,582 archived canonical result JSONs, and its
embedded `QWEN3_ALL_SWEEP_RESULTS.csv` matches the sole local reference table
with SHA-256
`1c98809da3889f81eb4d50c427c77ef45550674a693333881c172d824e6fa665`.
All 2,582 source directories/result files existed at archive creation, and all 2,330 source hashes
available from the earlier catalogs passed. The archive contains one result
folder per case inside this single `.tar.zst`; it does not duplicate the very
large Ray/NVTX/Nsight logs. After the 2026-08-17 cleanup, it is the retained
copy of the 1,943 invalid AFD result JSONs; their ancillary remote logs remain.
The 639 baseline source results remain untouched. Sidecars are adjacent as
`.sha256` and `.summary.txt`.

An initial attempt to duplicate complete case folders was abandoned after
confirming that individual high-parallelism cases could exceed 8 GiB; its
partials/staging were removed without modifying sources. The compact design
therefore archives the canonical result payloads and uses `source_case_dir` in
the manifest to locate full profiler evidence.

Whenever files for one case are needed, first find that row in the archive's
`MANIFEST.tsv`, then extract **only** its `archive_case_dir`. Do not unpack the
complete archive for single-case work. This selective-extraction rule is the
default for all future result-file use.

The separate single-file Nsight archive completed successfully on 2026-08-03
at
`/home/shengjiel/scratch/fastafd_reproduce/archives/qwen3_all_sweep_nsys_reports_20260803.tar.zst`.
It is 595,937,885,445 bytes and contains 70,259 `.nsys-rep` files from all
2,582 canonical cases; their uncompressed source size was 854,000,037,577
bytes. Per user direction, no checksum or additional full verification pass
was performed. The archive embeds `CASE_INDEX.tsv` (one row per case) and
`NSYS_FILES.tsv` (one row per report). Paths are organized beneath
`qwen3_all_sweep_nsys_reports_20260803/cases/<source-case-relative-path>/` so a
single case is directly addressable. To retrieve a case, find its
`archive_case_prefix` in `CASE_INDEX.tsv`, then extract only that prefix with
`tar --use-compress-program=unzstd --wildcards -xf <archive>.tar.zst -C
<destination> "<archive_case_prefix>/*"`; never unpack the complete Nsight
archive for one-case work. After the 2026-08-17 cleanup, this archive is the
retained copy of the 64,703 legacy AFD reports removed from their source
trees. All baseline reports and all 85,749 reports in the corrected 1,981-case
campaign remain untouched.

The separate Pareto-only Nsight archive completed on 2026-08-03 at
`/home/shengjiel/scratch/fastafd_reproduce/archives/qwen3_afd_baseline_pareto_nsys_reports_20260803.tar.zst`.
It selects the current independent upper-right frontier for AFD and baseline
at each of the eight uniform ISLs from the comprehensive sweeps: 213 cases
(118 AFD and 95 baseline), 2,088 `.nsys-rep` files, and 24,077,637,150 source
bytes. The archive is 15,844,410,403 bytes. Reports are organized as
`cases/afd|baseline/isl_<tokens>/row_<master-CSV-line>/`. Discovery sidecars
are adjacent as `.pareto_cases.tsv`, `.case_index.tsv`, `.nsys_files.tsv`, and
`.summary.json`; the same metadata and a README are embedded before the report
payload. The full 595,937,885,445-byte Nsight archive was treated as immutable:
its size, mtime, and inode were identical before and after this build. To avoid
unpacking that full archive, the new archive streamed the same reports from
the preserved source-case directories. The one-off build logic is retained in
the legacy cleanup archive, not the active source scripts. No
full-archive checksum pass was performed. Current AFD frontier membership
inherits the known selected-cohort metric caveat documented in
`scratch/afd_baseline_pareto_2d_20260803/CODEX_PROJECT.md`.

The user subsequently narrowed the deliverable to vLLM baseline only. The
remote baseline-only archive is
`/home/shengjiel/scratch/fastafd_reproduce/archives/qwen3_vllm_baseline_pareto_nsys_reports_20260803.tar.zst`:
95 baseline Pareto cases, 436 reports, 6,509,793,573 source bytes, zero AFD
cases/files, and 3,346,936,865 compressed bytes. The full Nsight archive's
size/mtime/inode again remained unchanged. The final local sharing archive was
then repacked from that downloaded file as
`scratch/afd_baseline_pareto_2d_20260803/pareto_nsys_baseline/qwen3_vllm_baseline_pareto_nsys_by_config_20260803.tar.zst`.
It is 3,340,109,350 bytes and exposes configuration in every case prefix:
`cases/isl_<tokens>/ep_<degree>/batch_<size>/row_<master-CSV-line>/`. A full
member-list validation found all 436 reports under valid config paths and all
95 `CASE_INDEX.tsv` rows matched their ISL/EP/batch values. The sharing folder
contains only this archive plus its prefixed README and four discovery
sidecars; the superseded local archive, duplicate bare TSV/JSON files, and the
interrupted 5.8-GB combined-archive partial were removed. The one-off repacker
is retained in the legacy cleanup archive. The
earlier combined AFD+baseline derived archive remains remote but is not the
current deliverable. No checksum pass was performed.

On 2026-08-18, optimized-attention correctness job `6292512` was submitted on
OCI-HSG using the README's official FastAFD deterministic-sample to vLLM
`prompt_logprobs` alignment process. It is A1:F1 / ATP1 / FEP4, 8K/b16, two
GB200 trays (eight GPUs), `short` QoS, and an explicit `02:00:00` limit; the
strict acceptance threshold is 100% overall top-10 hit rate. Results will be
written under
`/home/shengjiel/scratch/attention_sm_contention_optimization_20260817/correctness_a1_f1_ep4`.
The existing optimized 32K/64K runner now honors an explicitly supplied
`RUN_VLLM_ALIGNMENT` while retaining its default-off behavior; the new stage
adds only the one-row `CASE.csv` contract. Immediate Slurm inspection showed
the job pending with the requested QoS, time, node/GPU count, and bad-node
exclusions.

Job `6292512` later failed `2:0` after `00:07:44` before reference scoring.
FastAFD completed 64/64 samples with 17 tokens each and all four attention
ranks recorded the identical warmup-257 plus measured-258--272 graph-replay
window. The vLLM phase never launched because `vllm` was not found inside the
Slurm container; no `alignment.json` exists. This is a launcher-environment
failure, not a correctness pass or failure. A retry must make the pinned vLLM
entry point available inside the container rather than relying on the login
shell PATH.

The container PATH defect was fixed without changing model/runtime code: the
optimized stage now contains a `control/vllm` launcher backed by the pinned
artifact Python, and `run_afd.sh` prepends its existing control directory to
the in-container PATH. The identical A1:F1 / ATP1 / FEP4 8K/b16 official
alignment job was resubmitted as `6293256` on two GB200 trays, short QoS, with
an explicit two-hour limit. Its result root is
`/home/shengjiel/scratch/attention_sm_contention_optimization_20260817/correctness_a1_f1_ep4/results/afd_qwen3_8192_r1_ag4_fg4_atp1_fep4_adp4_b16_n2_20260818_102214_manual_na`.
