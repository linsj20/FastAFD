# Project Memory

## A2 fan-in correctness repair (2026-08-31 PDT)

A2 output corruption is localized to the FMHA O single-edge publisher, not the
FP8xFP4 MegaMoE math.  One outgoing edge does not imply destination source
partition zero: in A2 the second attention source still has one edge but owns
partition one.  The old BF16/FP8/fused-FP8 publisher specializations wrote it
to partition zero, and the FP8 paths also used the wrong packed-scale stride.
Remote fix `b769743` always honors descriptor source/source-count metadata and
adds no copy/materialization, synchronization, fallback, launch, or math.
Direct pre-fix job `6741945` reproduced complete source-1 payload corruption
and scale-stride corruption; generalized post-fix job `6742728` passes 72 exact
publications for every source partition at A=1 through A=8.  Kernel-only job
`6743049` passes all rows 3/6 x buckets 518/1036 x ordinary/prepared
combinations at the unchanged
0.96 gate (actual cosine at least 0.999891400), and 41/41 focused runtime tests
pass.  Tests are committed at `bd1e3ca`.  Full unchanged A2 model job `6743115`
completed `0:0` in 18:08 with 22.938265067-ms strict CUDA and 174.381104603
TPS/active GPU.  Relative to pre-fix job `6691133`, latency is 0.145% lower and
throughput is 0.146% higher (noise-level, no measured regression).  It produced
48 coherent 17-token samples and twelve Nsight reports with the unchanged
capture shape.  Unchanged official scorer job `6743442` completed `0:0` with
816/816 top-1 decisions, top-10/top-100 also 100%, and average/max rank 1.
After that pass, requested A3:F1/EP4 full-model job `6743740` completed `0:0`
in 19:50 on four trays (12 attention + 4 EP4 model GPUs) with the same
128K/b6/MB2/capture contract.  It retained 15/15 steps at 23.188910933-ms
strict CUDA and 194.058272635 TPS/active GPU, produced 72 coherent samples /
1,224 tokens and all sixteen traces, and its first 48 token arrays exactly equal
the perfectly aligned A2 sample.  Unchanged A3 scorer job `6744025` completed
`0:0` in 21:00: 1,224/1,224 top-1 decisions, top-10/top-100 also 100%,
average/max rank 1, and inspected attention/model ranks each retain exactly 16
graph launches.  This validates the repair at source count three as well as two.
See `CODEX_PROJECT_qwen3_ffn_overheads.md` for exact evidence.

## Completed FP8xFP4 MegaMoE cleanup (2026-08-29 PDT)

The accumulated candidate `9a84a5a` restores the participant cleanup ticket
that the rejected `8b73c39` had removed, retains the proven dead stats/timing/
debug-zero and buffer-size cleanup, and adds host-only simplification with
fail-fast lane/process-group validation.  Its focused contracts pass 22/22,
EP4 and EP8 kernel gates are neutral-to-faster, and candidate-only multi-rank
smoke job `6688503` passes.  Same-allocation eight-tray full job `6688914`
then ran candidate first and production `0bee9f3` second for both required
128K/batch-6 rows.  A7:F1/EP4 strict latency improves 24.628219733 ->
24.589484600 ms (-0.157279%; TPS/GPU +0.157527%), and A3:F1/EP8 improves
22.942197333 -> 22.937496467 ms (-0.020490%; TPS/GPU +0.020494%).  Every row
has 15/15 samples, zero outliers, and acceptable spread; both pools completed
2/2 with no failures and the logs contain no runtime or bundle error.  The
candidate therefore clears the full performance gate.  Run the unchanged
official A1/EP4 alignment workflow next: first generate the exact 128K/b6
candidate sample on two trays, then score that sample on one tray.  Monitor
both GPU jobs every five minutes.

The terminal A1 workflow passes.  Exact-sample job `6690293` completed `0:0`
in 17:32 on contiguous trays `nvl72129-T06-T07`; its pool is 1/1 complete with
no failures.  It produced the required 24 samples x 17 tokens, a 15/15 strict
trace at 22.748623667 ms with 1.212985% dominant-range spread, and durable
`sample.json`/`afd-result.json`.  The launcher intentionally killed its driver
only after those artifacts became durable; the enclosing job and pool both
accepted that allocation-scoped cleanup.  Official unchanged-scorer job
`6690783` then completed `0:0` in 11:08 on one tray.  Across 24 prompts and
408 generated tokens, top-1/top-10/top-100 agreement are all 1.0, average and
maximum vLLM rank are both 1.0, and reference perplexity is 1.010741077.  The
scorer logs have zero error-signature matches and its trace audit validates
ranks 1 and 5.  Compact evidence is local under
`scratch/megamoe_cleanup_20260828/alignment/final_9a84a5a/`; the full 57-MB
alignment report remains in the remote `a1_alignment_9a84a5a/alignment` root.
Remote candidate `9a84a5a` is clean and is the fully performance-gated and
officially aligned cleanup checkpoint.  Do not modify it after alignment;
start any further cleanup on a new descendant and repeat the gates.

The persisted terminal requirement is complete: rerun the authoritative
42-case 128K near-Pareto matrix from preserved aligned `9a84a5a`, using
FP8-activation/FP4-weight MegaMoE throughout.  Fresh task root
`afd_128k_optimized_attention_near_pareto_cleanup_9a84a5a_20260829` copies the
prior executable manifest byte-for-byte.  Independent structural validation
proves 42 unique cases, six cases in each 2/3/4/8/12/16/18-tray group, EP8
A1/A3/A5/A7/A8 plus EP4 A1/A2, batches 2--7, correct batch-specific capacity
contracts, and `megamoe_expert_weight_dtype=fp4` for every row.  Its new pool
initialized at 42 pending with zero claims/completed/failed.  Seven `short`
jobs reproduce the proven at-most-four-concurrent topology: leads `6691128`
(18 trays), `6691129` (16), `6691130` (12), and `6691131` (8), followed by
dependency-gated `6691132` (4 after 18), `6691133` (3 after 16), and `6691134`
(2 after 12).  All pin clean `9a84a5a`, FMHA-only, MegaMoE, two microbatches,
one-hour per-case watchdog, exact tray-sized fabric segments, and the unchanged
warmup-plus-15 contract.  The seven jobs were monitored every five minutes and
the goal remained open until the strict 42-case completion audit passed.

Current short QoS admits only two concurrent jobs.  Initial 8-tray `6691131`
and 12-tray `6691130` completed their 12 cases `0:0` in 1:17:04/1:26:10; all
are FP4 MegaMoE with 15/15 samples and zero outliers.  Two-tray `6691134`
started next and passed the batch-7 capacity gate, while eligible 18/16-tray
leads waited for large blocks.  The original 4/3-tray dependencies were only a
four-job concurrency device; after observing the stricter QoS cap and an idle
slot for more than five minutes, release them so small jobs can backfill.
`6691132`/`6691133` remain the same two jobs with the same tray-specific cases;
only their scheduler holds changed.  Provenance is in
`submission/dependency-adjustments.tsv`.  Two-tray `6691134` subsequently
completed all six A1/EP4 rows `0:0` in 1:02:09 with 15/15 samples and zero
outliers; its batch-2--7 latency range is 16.696566--26.161676 ms.  Three-tray
`6691133` likewise completed all six A2/EP4 rows `0:0` in 1:12:20; its final
batch-2 row is 17.328143867 ms with 15/15 samples and zero outliers.  Four-tray
`6691132` completed its batch-7 row after passing the capacity gate, and
18-tray lead `6691128` immediately backfilled the free QoS slot.  Continue
five-minute monitoring from 25 completed, two claims, 15 pending, zero failed.
At 10:11 PDT, 18-tray A8/EP8 batch 6 failed fast during coordinator creation
because ZMQ control port `21042` was already in use.  This was not a capacity,
model, CUDA, or metric failure: batch 7 had already completed at 26.785187267
ms with 15/15 samples and zero outliers, its control block `21032--21038` was
disjoint, and the worker continued on block `21048--21054`.  A read-only check
on the allocation confirmed failed block `21040--21046` was free after cleanup.
The pool tool archived the failed attempt and explicitly released only batch 6
for an unchanged retry.  Continue from 29 completed, two claims, 11 pending,
zero failed; the final audit must accept the retry and retain the archived
attempt as provenance.
The unchanged retry subsequently passed on a fresh control block and completed
at 23.255804867 ms with 15/15 samples, zero outliers, 4.591079% dominant spread,
and 4.505375% max/median separation.  Its terminal metric/result checksums are
durable, so the transient port collision is fully recovered.
Four-tray `6691132` then completed all six A1/EP8 rows `0:0` in 1:04:06.
Every row retains 15/15 samples with zero outliers; batch-2--7 latency spans
16.357555133--26.212167667 ms.  Pool state is 31 completed, one claim, ten
pending, zero failed.  Sixteen-tray `6691129` is eligible but waits on large
block resources while 18-tray `6691128` continues.
Eighteen-tray `6691128` completed all six A8/EP8 rows `0:0` in 1:44:14,
including the recovered batch-6 attempt.  All six terminal metrics have 15/15
samples and zero outliers; batch-2--7 latency spans
17.733821867--26.785187267 ms.  Pool state is 38 completed, one claim, three
pending, zero failed; only 16-tray `6691129` remains active.

Sixteen-tray `6691129` completed all six A7/EP8 rows `0:0` in 1:36:49.  Every
row has 15/15 samples and zero outliers; batch-2--7 latency spans
17.810513067--26.677761867 ms.  All seven jobs `6691128`--`6691134` are
terminal `COMPLETED 0:0`, and no job remains queued.  The final pool is 42
completed, zero claims, zero pending, and zero failed.

The reusable strict audit passes.  It validates all 42 manifest rows and
terminal metric/result checksums; exact tray/job mapping; 128K uniform input;
FMHA-only placement; MegaMoE FP8 activations with FP4 expert weights; two
microbatches; batch/capacity contracts; 15 retained samples; zero outliers;
and exactly 1,512 Nsight reports.  Global mean latency spans
16.357555133--26.785187267 ms, TPS per active GPU spans
59.892553956--232.300866904, maximum dominant spread is 4.591079%, and maximum
median difference is 4.505375%, both below the 10% limits.  The only error
signature in the final logs is the archived, explicitly recovered `21042`
port collision.  Compact local evidence (1.9 MB) is under
`scratch/afd_128k_optimized_attention_near_pareto_cleanup_9a84a5a_20260829/`:
`report/completion-audit.json`, `report/case-metrics.tsv`, 42 metric JSONs, 42
terminal records, 42 `afd-result.json` files, 14 job logs, the archived failed
attempt, dependency-adjustment provenance, and `submission/sacct-final.tsv`.
The same scheduler record is mirrored to the remote task root.  Preserved
aligned commit `9a84a5a` was not modified after alignment; the cleanup goal is
complete.

Paired comparison against the previous byte-identical 42-case manifest at
`f14d8005` shows no cleanup regression.  Cleanup `9a84a5a` lowers paired
geometric-mean latency by 0.313646% and raises geometric-mean TPS per active
GPU by 0.314632%; 35/42 cases and all seven tray/EP/A:F groups improve.  Batch
2 is effectively flat at -0.008292%, while batches 5--7 improve in every case.
The two largest regressions are isolated batch-2 rows at +0.902550% and
+0.607033%; every other regression is at most +0.181323%.  Both sweeps retain
15/15 samples with zero outliers and all 1,512 expected reports.  Maximum
dominant spread changes only from 4.571679% to 4.591079%, so interpret the
small aggregate gain as positive cross-allocation consistency rather than a
proven intrinsic kernel speedup.

## Active FP8xFP4 MegaMoE cleanup (2026-08-28 PDT)

The adopted FMHA-only baseline is clean remote commit `0bee9f3`.  Cleanup is
gated on A3:F1/EP8 and A7:F1/EP4 at 128K/batch 6 in one eight-tray bundle;
EP4 A1 retains the unchanged official alignment standard.  Kernel ablations
precede full performance runs.  The first isolated candidate, `f302ce9`,
removes the participant-only cleanup ticket and its otherwise-unused
acquire-release GPU atomic, restoring the all-CTA grid rendezvous while
preserving later rank-ready publication and rank-count caching.  This tests
whether a historical 0.23--0.34% proxy optimization remains useful after the
later dispatch changes.  Focused source contracts pass 21/21.  ABBA jobs
`6671502` (EP4/A7-like, four GPUs, rows 21) and `6671503` (EP8/A3-like, eight
GPUs, rows 9) compare `0bee9f3` with `f302ce9`, FP8xFP4 only, using 20 warmups
and 400 graph replays.  Both completed `0:0` in 6:13/6:08 with every trial
`status=ok`.  EP4 control/candidate ABBA means are 84.187560/83.959723 us
(-0.227838 us / -0.270626%); EP8 means are 60.616519/60.395560 us
(-0.220959 us / -0.364517%).  Later dispatch work made the extra cleanup
ticket counter/poll/reset path counterproductive, so `f302ce9` is the
cumulative cleanup base.  Isolated candidate `0077604` removes unreachable
same-rank route-stat plumbing from the Python wrapper, C++ API, JIT arguments,
and hot cleanup loop (4 files, 4 insertions/35 deletions), while preserving the
separate M2N/DeepEP statistics path.  `git diff --check` is clean and focused
source contracts pass 21/21.  Initial jobs `6672215`/`6672216` are not evidence:
they failed after control compilation because the launch encoded one production
specialization as comma-separated entries instead of the required
`1:1:1:1` tuple.  Corrected FP8xFP4 ABBA jobs `6672461` (EP8) and `6672460`
(EP4) completed `0:0` in 5:35 with every trial `status=ok`.  EP8 improves
60.572281 -> 60.337882 us (-0.234399 us / -0.386966%), but EP4 is initially
84.957438 -> 85.081720 us (+0.124283 us / +0.146289%), driven by one slower
candidate trial.  Warm-cache EP4 confirmation `6672876` completed `0:0` in
3:40 and reverses that small loss: 84.632959 -> 84.484439 us (-0.148520 us /
-0.175487%).  Across both independent EP4 allocations, the four controls and
four candidates average 84.795198/84.783080 us (-0.012119 us / -0.014292%),
effectively neutral and non-regressive.  Accept `0077604` as the cumulative
cleanup base.  The next isolated target is same-rank debug-timing plumbing:
production always compiles `kEnableDebugTiming=false`, but seven files retain
an unused optional API/JIT argument, a second kernel specialization, and about
190 lines of device-only timing instrumentation.  Candidate `923e4ea` removes
that same-rank-only path (7 files, 6 insertions/337 deletions), including stale
smoke collection and contracts, while retaining all four accepted route/
dispatch template controls and the independent M2N timing implementation.
`git diff --check`, sbatch syntax, and 21/21 source contracts pass.  FP8xFP4
ABBA jobs `6673293` (EP4) and `6673294` (EP8) compare it with cumulative
`0077604`; both completed `0:0` with all trials correct.  EP4 improves
84.940200 -> 84.867358 us (-0.072842 us / -0.085754%).  EP8 has tightly
clustered 60.936241/60.942078-us controls and candidate trials 60.715518/
68.965840 us; the isolated 68.966-us sample makes the result inadmissible.
Warm-cache EP8 confirmation `6673525` completed `0:0` in 4:14: controls/
candidates average 60.722721/60.593920 us (-0.128801 us / -0.212119%), with
tight candidate trials 60.646639/60.541201 us.  This isolates the earlier slow
sample as allocation noise.  Accept `923e4ea` as the cumulative cleanup base.
Candidate `5ce1767` removes the now-unreachable same-rank FP8-weight API/JIT/
device specialization and the smoke's FP8-name-to-FP4 monkey patch.  The
FMHA-only adapter and launchers already fail fast unless expert weights are
FP4; M2N retains its independent FP8/FP4 switch.  The isolated diff is 6 files,
46 insertions/226 deletions; `git diff --check` and 21/21 contracts pass.
FP8xFP4 ABBA jobs `6673766` (EP4) and `6673767` (EP8) completed `0:0` with
all correctness guards unchanged.  Their control/candidate means are
84.458241/84.719763 us (+0.309646%) and 60.527000/60.568280 us (+0.068201%),
respectively.  These small first-allocation losses are inconclusive rather
than acceptable non-regression evidence.  Warm-cache EP4 confirmation job
`6674114` completed `0:0` in 3:33 and improves 85.650120 -> 85.567241 us
(-0.096765%).  Across both EP4 allocations, however, control/candidate means
remain 85.054181/85.143502 us (+0.105017%).  Final cached confirmations
`6674210` (EP4) completed `0:0` in 4:00 and improves 84.777079 -> 84.541678 us
(-0.277670%).  Across three EP4 allocations, six control/candidate samples
average 84.961813/84.942894 us (-0.022268%), clearing non-regression.  EP8 job
`6674209` is invalid infrastructure evidence: its two trays landed in different
NVL72 blocks with `SegmentSize=1`, so symmetric-memory rendezvous failed before
timing.  Corrected `--segment=2` retry `6674268` completed `0:0` in 3:47 and
regresses 60.295081 -> 60.507240 us (+0.351868%).  Across two valid EP8
allocations, four controls/candidates average 60.411041/60.537760 us
(+0.209762%).  Reject `5ce1767`: required A3/EP8 worsens despite neutral/
slightly favorable EP4, so the cumulative accepted head remains `923e4ea` and
no full run is warranted.
Reusable `codex_scripts/megamoe_source_abba.sbatch` preserves the ABBA order,
validates clean exact source heads, and can reuse an existing extension cache
so GPU allocations do not repeat CPU-only builds.  It also fails fast on
cross-NVL72 allocations; multi-tray submissions must request a matching Slurm
segment size.
Candidate `0c24f08` starts from accepted `923e4ea` and preserves the FP8/FP4
precision specialization rejected for removal in `5ce1767`.  It removes only
fixed same-rank recipe/SwiGLU/no-clamp/fast-math API and JIT parameters, unused
buffer-size arguments, compiled-out clamp/slow-math branches, and the
unreferenced `DG_COMM_KERNEL_DEBUG` post-launch zeroing hook.  M2N and every
route/dispatch specialization remain unchanged.  The five-file diff is 15
insertions/75 deletions; `git diff --check` and 21/21 source contracts pass.
FP8xFP4 EP4/A7-like ABBA job `6674529` completed `0:0` in 6:18 with unchanged
correctness guards.  Control/candidate means are 84.971957/85.095520 us, a
0.123563-us / 0.145416% candidate regression.  Warm-cache EP4 confirmation
`6674687` completed `0:0` in 3:56 and regresses 84.958882 -> 85.018759 us
(+0.070477%).  Across both allocations, four controls/candidates average
84.965420/85.057139 us (+0.107949%).  Reject `0c24f08`; skip EP8 and full
performance for it.  The accepted cleanup frontier remains `923e4ea`; retain
the precision and fixed-knob paths because removing them failed non-regression,
and retain route/dispatch modes for their functional reference coverage and
accepted performance mechanisms.  Validated
`scratch/megamoe_cleanup_20260828/full_exact/CASES.csv` contains exactly the
two required eight-tray/32-GPU FP4-weight rows (A3/EP8 and A7/EP4 at 128K/b6).
Run one adopted-`0bee9f3` control bundle and one cumulative-`923e4ea` final
bundle, with both cases in each allocation and five-minute monitoring.
Control job `6674894` completed `0:0` in 26:54 with `SegmentSize=8` on eight
trays in NVL72 block `nvl72005`; both pool rows completed with zero failures.
A7:F1/EP4 completed first with 15/15 retained samples, zero
outliers, 24.647035-ms strict CUDA mean, 24.595240-ms median, 3.645588%
dominant range, 3.483345% max/median separation, and 213.007365 TPS/GPU.
A3:F1/EP8 also retained 15/15 samples with zero outliers: 23.000511-ms strict
CUDA mean, 22.959649-ms median, 23.378945-ms maximum, 1.897455% dominant
range, 1.826230% max/median separation, and 195.647829 TPS/GPU.  These are the
exact baselines for the cumulative `923e4ea` final bundle.  Final job `6676158`
was submitted as one eight-tray `SegmentSize=8` allocation with the identical
two-row plan, source head and cleanliness guards, and runtime contract; monitor
it every five minutes.  Its A7:F1/EP4 row retained 15/15 samples with zero
outliers but measured 24.709721 ms versus the 24.647035-ms control, a
0.062686-ms / 0.254335% regression.  Do not accept the cleanup or launch final
alignment from this preliminary result.  Finish A3:F1/EP8 for evidence; because
the jobs landed on different NVL72 blocks, use the existing general paired-source
runner in one allocation if the second result indicates a common block shift.
A3:F1/EP8 measured 23.000511 -> 22.990296 ms, a 0.010215-ms / 0.044413%
improvement, also with 15/15 samples and zero outliers.  The mixed result is not
acceptable under the non-regression rule.  Paired confirmation job `6677619`
was therefore submitted with the same two-row plan and `SegmentSize=8`, running
candidate first and control second on one allocation; final alignment remains
withheld.  First paired job `6677619` produced valid candidate metrics of
24.605921 ms (A7/EP4) and 22.983244 ms (A3/EP8), both 15/15 with zero outliers,
but its first control row failed before measurement: the candidate-built
reduced-signature MegaMoE extension was reused by old-signature control Python,
raising a binding `TypeError`; Gloo disconnects were secondary.  Cancel the
remaining invalid work (`6677619` cancelled at 40:19) rather than waste the
allocation.  Harden the existing general `run_afd_paired_sources.sbatch` with
per-label DeepGEMM and TVM-FFI caches.  Fresh retry job `6679260` uses those
isolated caches, the same candidate-first/control-second two-row plan, and
`SegmentSize=8`.  The user then directed that expansive full performance be
deferred until more cleanup changes accumulate.  Job `6679260` was still
pending and was cancelled at 0:00, consuming no GPU time.  Keep alignment
withheld and continue individual cleanup iterations with small kernel ABBA
tests; only return to the two-case full bundle after a larger proven batch has
accumulated.  Next isolated candidate `b9fcd44` on
`codex/fmha-megamoe-cleanup-no-debug-zero-20260829` removes only the dormant
post-launch `DG_COMM_KERNEL_DEBUG` symmetric-buffer zeroing hook (one file,
five deletions).  This must be isolated because the earlier multi-knob removal
regressed and could not attribute the cause.  `git diff --check` and 21/21
source contracts pass.  One-tray FP8xFP4 EP4 ABBA job `6679570` is the first
small gate; run EP8 only if it is non-regressing, and do not run full perf yet.
The launch used the wrong dtype variable, so `6679570` actually exercised the
default FP8-weight path and completed `0:0` in 6:25.  Its three control/
candidate means are 100.531521/100.419040 us (balanced-24),
119.554720/119.744959 us (one-rank-32), and 97.860160/97.639041 us (rows-3):
mixed supplementary coverage, not the adopted-path gate.  Malformed retry
`6679779` was cancelled pending at 0:00.  FP4 retry `6679793` was then stopped
at 2:50 after discovering that its remaining smoke defaults did not match the
established production proxy.  Exact EP4 replacement `6679940` uses bucket
3840, rows 21, 150 SMs, 20 warmups/400 iterations, and specialization
`1:1:1:1`.  The reusable ABBA wrapper now requires and logs all of those
benchmark-defining values plus expert dtype, preventing silent fallback to a
different path or shape.  First exact launch `6679940` still failed before any
measurement because `1/1/1/1` was parsed as four separate entries rather than
the required single `1:1:1:1` tuple.  The wrapper now validates tuple syntax
before source verification/launch, and corrected job `6680100` reuses the
already-built control cache.  In parallel CPU-only preparation, independent
candidate `04ae569` on
`codex/fmha-megamoe-cleanup-no-buffer-knobs-20260829` removes only the unused
`use_fp8_dispatch` and `activation` arguments from symmetric-buffer sizing and
its two callers (3 files, 3 insertions/6 deletions).  Those constants never
participated in layout calculation or device code; `git diff --check` and
21/21 source contracts pass.  Do not combine it until its own small gates pass.
Second independent CPU-ready candidate `7a3d873` on
`codex/fmha-megamoe-cleanup-no-recipe-20260829` removes only the fixed
`(1,1,32)` recipe argument, its host assertion, and the two constant Python
arguments (3 files, 3 insertions/9 deletions).  The recipe never reached JIT
selection or device code; `git diff --check` and 21/21 source contracts pass.
Keep it separate until the earlier candidates finish their small gates.
Corrected exact EP4 job `6680100` completed `0:0` in 5:23 with all four
FP8xFP4 trials correct.  Control/candidate means are
84.888515/84.870319 us (-0.018196 us / -0.021435%), so the dormant debug-zero
removal clears EP4 non-regression.  EP8/A3-like job `6680334` uses the same
bucket 3840, 150-SM, 20-warmup/400-iteration, `1:1:1:1` standard on two nodes
with `SegmentSize=2`; full perf remains deferred.  General
`codex_scripts/summarize_megamoe_abba.py` now validates the completed ABBA
order/status/dtype and reports every common exact-Qwen metric, samples, means,
absolute delta, and percentage delta.
Follow-on small tests are scheduler-serialized to avoid concurrent GPU use:
`6680334` (debug-zero EP8) -> `6680445` (buffer-size API EP4) -> `6680450`
(fixed-recipe EP4).  Both EP4 jobs reuse the clean `923e4ea` control extension
cache from `6680100`, keeping CPU-only extension compilation off their GPU
allocations; candidate caches remain isolated because their Python bindings
differ.  Superseded dependency-pending submissions were cancelled at 0:00.
Exact EP8 job `6680334` produced four correct trials: controls
60.765438/60.334640 us and candidates 60.457282/60.345922 us.  Means are
60.550039/60.401602 us (-0.148437 us / -0.245148%).  Together with the EP4
result, this accepts `b9fcd44` as the new cumulative cleanup frontier.  It
removes a dormant host debug hook without an adopted-path regression; continue
with the already-serialized API cleanups, not full performance.
Buffer-sizing API EP4 job `6680445` produced controls
84.936476/85.185356 us and candidates 84.914484/84.808483 us.  Means are
85.060916/84.861484 us (-0.199432 us / -0.234458%), so independent candidate
`04ae569` clears EP4.  Fixed-recipe EP4 `6680450` remains next; buffer-sizing
EP8 job `6680766` is dependency-held behind it and will reuse both clean
per-source extension caches.  Temporary `.patch` artifacts were removed at the
user's direction; future candidate edits must be made directly in worktrees.
Fixed-recipe EP4 job `6680450` completed `0:0` in 5:24.  Controls are tightly
grouped at 84.925919/84.915276 us, but candidates are
85.023441/86.053362 us; means regress 84.920597 -> 85.538402 us
(+0.617805 us / +0.727508%).  The second candidate is an isolated slow repeat,
so this is not accepted or advanced to EP8.  One warm-cache EP4 confirmation,
`6681002`, is dependency-held behind buffer-sizing EP8 `6680766`; repeated loss
rejects `7a3d873`, while reversal will be judged on the aggregate.  No full
performance run is authorized yet.
The +0.727508% loss is too large for a neutral repeat to rescue on an aggregate
basis, while a large favorable reversal would be equally unreliable.  Reject
`7a3d873` on the completed EP4 evidence, cancel confirmation `6681002` at 0:00,
and do not run recipe EP8.
Buffer-sizing EP8 job `6680766` completed `0:0` in 4:09.  Candidate repeats
are tight at 60.274239/60.288482 us, while controls bracket them at
60.097280/60.307999 us; means regress 60.202639 -> 60.281360 us
(+0.078721 us / +0.130760%).  This is not acceptable yet, but the 0.210719-us
control spread warrants one warm-cache confirmation.  Job `6681178` was
released by cancelling recipe confirmation `6681002`; accept `04ae569` only if
the two-allocation aggregate clears non-regression, otherwise reject it.
Buffer-sizing EP8 confirmation `6681178` produced controls
60.839758/60.812001 us and candidates 60.515442/60.767918 us, improving
60.825880 -> 60.641680 us (-0.184200 us / -0.302831%).  Across both EP8
allocations, four controls/candidates average 60.514259/60.461520 us
(-0.052739 us / -0.087152%).  Together with favorable EP4, accept `04ae569`.
The next direct-edit candidate removes the now-host-only fixed `"swiglu"`
activation argument after buffer sizing stopped consuming it; the device
kernel remains unconditionally SwiGLU.  Keep it isolated and small-gated.
Direct-edit candidate `47e7702` on
`codex/fmha-megamoe-cleanup-no-activation-20260829` is based on accepted
`04ae569` and removes the fixed activation string from the same-rank C++ API,
both Python calls, and its host assertion (3 files, 3 insertions/8 deletions).
`git diff --check` and 21/21 source contracts pass.  Exact EP4 job `6681373`
compares it directly with `04ae569` and reuses only the accepted control cache.
Job `6681373` completed `0:0` in 4:51 with controls
85.174637/85.069122 us and candidates 85.119362/85.076084 us.  Means improve
85.121880 -> 85.097723 us (-0.024157 us / -0.028379%), clearing EP4.  Exact
EP8 job `6681534` is next and reuses the clean per-source extension caches.
Activation EP8 job `6681534` produced tight controls
60.722561/60.656800 us and slower candidates 61.163201/60.803199 us.  Means
regress 60.689681 -> 60.983200 us (+0.293519 us / +0.483640%).  Reject
`47e7702` without confirmation or combination despite neutral EP4.  The next
direct candidate returns to accepted `04ae569` and removes only the always-
disabled activation-clamp API/JIT parameter and compile-time-dead device
branch; both callers currently pass `None`/infinity.
Job `6681534` completed `0:0` in 4:24.  Direct-edit clamp candidate `a54809a`
on `codex/fmha-megamoe-cleanup-no-clamp-20260829` removes that parameter,
host infinity conversion/check, JIT template argument, dead device branch,
and both Python `None` arguments (5 files, 8 insertions/30 deletions).
`git diff --check` and 21/21 source contracts pass.  Exact EP4 job `6681862`
completed `0:0` in 5:21 with controls 85.147276/85.139036 us and candidates
85.132399/85.186005 us; means are 85.143156/85.159202 us
(+0.016046 us / +0.018845%), which is neutral enough to advance to the EP8
gate.  EP8 job `6682156` uses rows 9, 150 SMs, bucket 3840, 20 warmups/400
iterations, specialization `1:1:1:1`, two nodes in `SegmentSize=2`, and warm
per-source caches.  Independent direct-edit candidate `ccfcfd7` on
`codex/fmha-megamoe-cleanup-fast-math-fixed-20260829` fixes the production
activation math policy to the existing always-true fast path, removing the
runtime/JIT parameter and unreachable precise-math branch (5 files,
11 insertions/25 deletions); 21/21 contracts and `git diff --check` pass.  Do
not submit its GPU gate until the clamp decision completes, and do not run
full performance until more cleanups accumulate.
Clamp EP8 job `6682156` completed `0:0` in 4:11 with controls
60.394559/60.295839 us and candidates 60.322480/60.496802 us; means regress
60.345199 -> 60.409641 us (+0.064442 us / +0.106789%).  Because the candidate
pair straddled the controls, warm-cache confirmation `6682465` ran and
completed `0:0` in 4:00.  Across both EP8 allocations, four controls and four
candidates average 60.452120/60.478920 us (+0.026799 us / +0.044332%).  EP4
also regressed by 0.018845%, so reject `a54809a` under the no-slowdown rule
and retain the activation clamp plumbing.  Independent fast-math EP4 job
`6682683` now compares accepted `04ae569` with `ccfcfd7`; the candidate does
not contain the rejected clamp removal.  Full performance remains deferred.
Fast-math EP4 job `6682683` produced correct controls
85.196962/85.123358 us and candidates 84.930000/84.897604 us; means improve
85.160160 -> 84.913802 us (-0.246358 us / -0.289288%).  EP8 job `6682992`
is dependency-serialized behind the finishing EP4 job, uses two nodes in
`SegmentSize=2`, and reuses the clean control/candidate extension caches.
Job `6682683` completed `0:0` in 4:48.  Direct cumulative branch
`codex/fmha-megamoe-cleanup-cumulative-20260829` starts from accepted
debug-zero `b9fcd44` and directly reapplies accepted buffer-size API cleanup as
`8b73c39`, without patch application.  An initial compound `sed` command
malformed one call line; the 21-test contract gate caught it before commit or
GPU use.  The two affected lines were repaired directly, `git diff --check`
and 21/21 contracts pass, and the worktree is clean.  Keep pending fast-math
out of this cumulative head until EP8 completes.
Fast-math EP8 job `6682992` completed `0:0` in 4:05.  Controls are
59.706879/60.110402 us and candidates 60.152478/59.629922 us; means improve
59.908640 -> 59.891200 us (-0.017440 us / -0.029112%).  Together with the
0.289288% EP4 improvement, accept `ccfcfd7`.  Cumulative head `6169030`
directly integrates its tested files with accepted debug-zero and buffer-size
cleanups; non-overlapping files are byte-identical to `ccfcfd7`, the API keeps
the debug-zero deletion, and `git diff --check` plus 21/21 contracts pass.
Combined EP4 job `6683433` compares `923e4ea` with `6169030` before EP8 or any
full performance.  It uses a fresh cumulative cache so the tested extension
contains the exact combined source.
Combined EP4 job `6683433` completed `0:0` in 5:14 with controls
84.808083/84.763355 us and candidates 84.833679/85.038795 us; means regress
84.785719 -> 84.936237 us (+0.150518 us / +0.177528%).  This blocks EP8/full
for now.  Warm-cache confirmation `6683588` repeats the same exact sources and
shape; judge the four-sample aggregate, and if loss repeats, split the combined
head at intermediate `8b73c39` to identify whether fixed fast math interacts
with the accepted debug-zero + buffer-size pair.
Confirmation `6683588` completed `0:0` in 3:53 with controls
84.994240/84.811440 us and candidates 84.984722/84.961281 us.  Across both
combined EP4 allocations, four controls/candidates average
84.844279/84.954619 us (+0.110340 us / +0.130050%), so stop `6169030` before
EP8/full despite each component's isolated result.  Clean split worktree
`codex/fmha-megamoe-cleanup-cumulative-no-fast-math-20260829` points at
`8b73c39`, passes 21/21 contracts, and contains only accepted debug-zero plus
buffer-sizing changes.  EP4 isolation job `6683788` compares it with
`923e4ea` using a fresh exact candidate extension cache.
Split EP4 job `6683788` completed `0:0` in 5:08.  Controls are tightly grouped
at 85.024080/85.045996 us and candidates are 84.943676/84.769363 us; means
improve 85.035038 -> 84.856520 us (-0.178518 us / -0.209935%).  Therefore the
debug-zero + buffer-size pair is healthy, and the cumulative regression was
introduced only after stacking fixed fast math.  Exclude `ccfcfd7` from the
cumulative candidate despite its standalone gates.  EP8 split job `6684202`
now tests `8b73c39` with warm exact caches; full performance remains gated.
EP8 split job `6684202` completed `0:0` in 4:00.  Controls
60.294962/60.252719 us and candidates 60.297441/60.156798 us give means
60.273840/60.227120 us (-0.046721 us / -0.077514%).  With combined EP4 also
improving, `8b73c39` clears the small gates.  Fresh paired task root
`full_exact/paired_final_8b73c39` contains the unchanged two-row A3:F1/EP8 +
A7:F1/EP4 128K/batch-6 FP8xFP4 plan for both candidate and adopted `0bee9f3`
control.  Hardened per-source-cache runner syntax, two-row plans, clean source
heads, and empty pools validate.  Full job `6684519` uses one eight-tray
`SegmentSize=8` allocation, candidate first/control second, and must be
monitored every five minutes.  Final A1 alignment remains withheld until this
paired full comparison passes.
Initial full submission `6684519` failed `1:0` after 20 seconds before any
case launch: outer paired-wrapper validation requires both pool directories to
exist, while preparation had intentionally left them absent for inner-bundle
initialization.  Initialize each pool with exactly two pending eight-tray
cases, zero claims/completed/failed, then resubmit unchanged as job `6684906`.
The first failure produced no performance data; monitor the retry every five
minutes.
Paired retry `6684906` completed `0:0` in 56:17 on one `SegmentSize=8`
allocation in NVL72 block `nvl72069`.  Both candidate and control pools are
2/2 completed with zero claims/failures/pending; each source has two strict
metrics and 64 Nsight reports, and the paired log has zero traceback, CUDA,
MegaMoE-timeout, assertion, or bundle-error matches.  Every row retained 15/15
steps with zero outliers and passed unchanged spread limits.  Same-allocation
means are: A7:F1/EP4 candidate/control 24.658220/24.560235 ms, a
+0.097985-ms / +0.398958% latency regression and -0.397373% TPS; A3:F1/EP8
candidate/control 22.965270/22.967053 ms, a neutral -0.001783-ms / -0.007762%
latency change and +0.007763% TPS.  Reject `8b73c39` because required A7
worsens, keep A1 alignment withheld, and do not spend another full run until a
smaller subset accumulates enough new evidence.  CPU-only A7 paired trace
summary job `6687527` uses the validated extractor worktree to attribute the
loss from the four existing attention/model Nsight reports; the first submit
attempt did not create a job because `8b73c39` predates that general script.
Job `6687527` completed `0:0` in 34 seconds and wrote all four requested
kernel summaries.  Candidate/control A7 attention FMHA totals are effectively
identical (340.784845/340.687755 ms across 3,008 calls, +0.032 us/call), and
attention `wait_ready` is lower by 13.075099 ms in aggregate despite a
+1.056-us median.  In the model trace, candidate MegaMoE is faster by
10.781585 ms total, -3.584 us/call on average and -0.320 us at the median.
The end-to-end loss is instead concentrated in model `wait_ready`: candidate
is +28.962326 ms total, +9.629 us/call on average and +0.384 us at the median,
with a 28-ms maximum versus 15.5 ms for control.  Therefore the A7 failure is
not a slower MegaMoE compute kernel; it is synchronization/cadence movement.
Strict end-to-end non-regression still rejects the cleanup.  The most plausible
device-side cause inside cumulative `8b73c39` is the earliest `f302ce9` removal
of the participant cleanup ticket in favor of an all-CTA grid rendezvous;
later stats/timing/debug-zero/buffer-size removals are compile-time-dead or
host/API-only.  Build the next smaller candidate directly from `8b73c39` by
restoring that local cleanup-ticket path while retaining the four dead-path
cleanups.  Compare the resulting source directly with adopted `0bee9f3` using
small EP4 and EP8 kernels only.  Do not run another full bundle until further
non-regressing cleanup accumulates, and do not run A1 alignment.
Direct-edit branch `codex/fmha-megamoe-cleanup-keep-ticket-20260829` restores
the exact inverse of `f302ce9` atop `8b73c39` as commit `d9b648a`: 50
insertions/16 deletions across the device implementation and atomic helper.
The helper is byte-identical to `0bee9f3`, and the remaining production diff is
seven files containing only same-rank route-stat/debug-timing deletion plus
the dormant post-launch debug zero and unused buffer-size API arguments.
`git diff --check` is clean and the focused unittest command passes 21/21 when
the worktree Python path is explicit.  Host Python and the pinned artifact do
not provide pytest; the first unittest invocation also resolved the installed
source until `PYTHONPATH` was set, so neither failed preparation attempt is a
code result.  Exact EP4 small-kernel job `6687863` completed `0:0` in 5:03
with four valid FP8xFP4 records.  Production/candidate means are
84.308081/84.320679 us, +0.012598 us / +0.014943%, neutral enough to advance
to EP8.  EP8 job `6688035` uses rows 9 with the same 150-SM/bucket-3840/
20-warmup/400-replay contract, two nodes in `SegmentSize=2`, and isolated
source caches.  Monitor at five-minute cadence; full perf remains deferred.
EP8 job `6688035` completed `0:0` in 4:19 with production/candidate means
60.666838/60.713961 us (+0.077676%), but its 0.364-us control spread brackets
the candidate pair.  One warm-cache confirmation, job `6688225`, completed
`0:0` in 4:15.  Across both allocations the four production/candidate samples
average 60.824119/60.807421 us, -0.016698 us / -0.027452%.  Together with the
neutral +0.014943% EP4 result, `d9b648a` clears both small gates and is the
refined cleanup frontier; it is not production-promoted without a later full
paired pass.  Continue accumulating cleanup and do not run full performance
or A1 yet.

The next host-only direct cleanup is commit `3c630fc` on
`codex/fmha-megamoe-cleanup-stable-buffer-ptrs-20260829`.  The symmetric
pointer addresses are fixed for the allocation lifetime, but the wrapper kept
an unused process-group field, fabricated a single-rank `SimpleNamespace`, and
copied the stable pointer list on every MegaMoE call.  It now materializes the
list once, preserves the real multi-rank rendezvous handle lifetime, removes
the unused field/shim/property, and adds contracts for both single/multi-rank
paths (2 files, 15 insertions/14 deletions).  Generated CUDA, JIT arguments,
and launch code are byte-identical to `d9b648a`; `git diff --check` and 21/21
focused tests pass.  Accumulate more host-only cleanup before spending a small
functional smoke, then keep the full bundle deferred.
Second host-only commit `67d83dc` localizes constructor-only adapter inputs:
it removes an unused distributed-rank query and stops retaining real/local
expert counts, TP size/rank, and the validated precision environment value as
mutable fields.  Expert-map construction, coverage validation, logged group
size/derived expert count/precision label, and every later runtime input are
unchanged.  Source contracts explicitly reject the removed fields; the
cumulative host branch remains clean and passes 21/21 tests.  The cumulative
diff from `d9b648a` touches only `megamoe_mega.py`, `megamoe_afd.py`, and one
contract file; all C++/CUDA/JIT files are byte-identical.  Candidate-only EP4
functional smoke `6688503` uses the exact rows-21/150-SM/bucket-3840 FP4 path
with 10 warmups/100 replays to exercise multi-rank pointer caching without
spending a second ABBA.  Monitor every five minutes; no full run follows.
Job `6688503` completed `0:0` in 2:27 with finite output, unchanged numeric
bounds, `max_active_experts=32`, and `status=ok`; its 84.341764-us timing is a
single functional record, not a comparative gate.  The source-specific host
extension rebuilt despite byte-identical C++ because the source path changed;
use the existing general CPU `prebuild_deepgemm_extension.sbatch` before the
next fresh-worktree GPU smoke rather than spending allocation time on that
host-only build.  Follow-on host commit `82cae29` removes the adapter's fixed
`_run_mega_moe` function field and directly calls the already-adopted/fail-fast
FP8xFP4 implementation.  Commit `d27539a` removes two silent lane fallbacks:
non-positive lane counts and out-of-range lane IDs now raise clear errors
instead of coercing to one or modulo-wrapping to a different buffer.  Valid
production scheduling is unchanged.  Both commits change only Python/contracts,
leave generated CUDA byte-identical, pass `git diff --check`, and pass 21/21
focused tests.  Continue CPU/source cleanup; do not schedule full performance.
Executable CPU coverage commit `9a84a5a` adds a real zero-lane constructor test
and negative/upper-bound forward-lane tests; the focused suite is now 22/22.
With the synchronization restoration, five host/runtime cleanups, small EP4/
EP8 gates, and multi-rank functional smoke accumulated, one new full paired
gate is now warranted.  Fresh task root
`full_exact/paired_keep_ticket_host_cleanup_9a84a5a` contains exactly the same
A3:F1/EP8 and A7:F1/EP4 128K/batch-6 rows for candidate `9a84a5a` and control
`0bee9f3`.  The first remote pool-init command expanded remote-only variables
locally and ran no initialization; explicit absolute-path retries initialized
both pools correctly.  The combined pre-submit shell repeated that quoting
mistake in read-only substitutions, so its checks printed false local-path
errors while the literal remote `sbatch` still submitted job `6688914`.
Immediate independent audit proves the live job valid: two six-field pair rows,
exact clean source heads, and 2 pending/0 claims/0 completed/0 failed in each
pool.  Job `6688914` requests one eight-tray `SegmentSize=8` allocation and runs
candidate first/control second.  Monitor every five minutes; A1 remains gated
on both full rows passing.

## Active causal retest of attention-only fused O publication (2026-08-28 PDT)

CPU-short Nsight summary job `6653623` proved the combined regression belongs
to its FFN half: current MegaMoE quantization was already router-fused, model
nodes stayed at 2,079, norm+router grew 7.184 -> 10.023 us/round, and MegaMoE
median grew 43.680 -> 56.480 us after its FP8 buffers were produced earlier.
Restore accepted FFN unchanged.  Attention cast+publish instead improves
9.524 -> 7.498 us/round and removes 188 graph nodes.  Clean isolated head
`0032e1c` therefore keeps only fused attention O quantize/publish atop adopted
FP8xFP4-only `7f5c273`; 21/21 contracts pass and short four-GPU smoke `6654019`
completed `0:0`.

The initial exact run and cross-allocation trace were confounded by large
movement in byte-identical FFN kernels.  General same-allocation harness
`abb04c3` therefore ran control `7f5c273` then candidate `0032e1c` on the same
four trays in short job `6654903`; it completed `0:0` in 35:19.  Both sides
retained 15/15 samples with zero outliers.  Control/candidate strict means are
22.745179600/23.243643400 ms and TPS/GPU is 131.896079/129.067545, still a
0.498463800-ms / 2.19% candidate regression.  Paired trace extraction job
`6655853` completed `0:0` and confirms the user's expected decomposition:
attention cast+publish itself improves 9.312 -> 8.928 us/layer median
(9.295 -> 8.960 us average), while unchanged model kernels are normally
identical.  The headline loss instead coincides with attention wait median
115.072 -> 119.520 us, model wait 140.192 -> 143.280 us, and byte-identical
FP8xFP4 MegaMoE 35.424 -> 37.216 us.  Because all three shifts disadvantage
the second run, a reverse-order replay was required.  Initial reverse-order
job `6655984` failed before case claim
when Slurm spread four trays across two fabric blocks; the fail-fast preflight
worked and produced no benchmark evidence.  `6656164` added `--segment=4` and
received one correct base system, but the pair wrapper then failed before pool
initialization because its required empty pool directories had not been
created.  No case was claimed.  Corrected `short` retry `6656285` completed
`0:0` on contiguous `nvl72082-T[07-10]` in 34:51.  Candidate-first/control-
second strict means are 23.224607267/22.744459733 ms with 15/15 samples and
zero outliers, reproducing a 0.480147533-ms / about 2.11% regression and ruling
out run order.

Local Git inspection found that `0032e1c` changed A-to-F synchronization and
store shape in addition to launch count.  Control releases `attention_turn` at
publisher entry after its local cast; the direct fused kernel released only
after issuing its remote payload, matching the observed added attention wait.
Control also publishes with 16-byte vector stores while direct fusion emitted
four-byte remote stores from quantizing lanes, a plausible communication-
efficiency loss.  Clean corrective branch
`codex/fmha-attention-fused-staged-publish-20260828` at `f229ad9` stages the
FP8 result locally, releases after all CTAs finish BF16 reads, and then uses
16-byte remote stores in the same fused kernel.  No FFN/MegaMoE production
file, alignment, or tuning parameter changes.  The source contract passes
21/21; four-GPU short smoke `6657389` completed `0:0` in 3:14 with fresh JIT
prebuild plus byte-exact repeated fabric/turn validation.  Exact
candidate-first/control-second same-allocation job `6657525` completed `0:0`
in 36:56 on `nvl72004-T[02-03,15-16]`.  Candidate/control strict means are
22.966924800/22.756537933 ms with 15/15 samples and zero outliers.  Restored
handoff/store behavior recovers 0.257682467 ms versus direct fusion's
reverse-order result, but remains 0.210386867 ms / 0.924512% slower than its
same-allocation control and lowers TPS/GPU by 0.916043%.  CPU-short trace job
`6658813` completed `0:0` in 2:40 from four named reports.  It proves the
residual is local to fused transport: control cast+publish is 8.096 us/layer
median while staged fusion is 9.344 us, a 1.248-us penalty that predicts about
0.235 ms across 188 layer-rounds.  FMHA is unchanged at 112.256/112.288 us;
the longer A-to-F tail also coincides with byte-identical FP8xFP4 MegaMoE
moving 52.160 -> 53.584 us while router/model-cast timings stay effectively
identical.

Source inspection shows `f229ad9` issued one head with eight active lanes and
looped four times per warp.  Clean commit `1276ce6` instead groups four tasks
already owned by that warp, restoring all 32 lanes and 16-byte stores without
a grid barrier, parameter change, alignment change, or FFN edit.  Contracts
pass 21/21; fresh four-GPU short smoke `6659029` completed `0:0` in 2:33 with
JIT prebuild plus the byte-exact repeated fabric check.  Exact candidate-first/
control-second short job `6659188` completed `0:0` in 35:54 on one four-tray
segment without a hang.  The full-warp candidate/control strict means are
22.947676357/22.699267867 ms, with 14/15 retained samples and 1/0 outliers;
TPS/GPU is 130.732191/132.162853.  The candidate remains 0.248408490 ms /
1.094346% slower and full-warp publication recovers only about 0.019 ms versus
the prior staged candidate, so it is not promotable.  CPU-short four-report
trace job `6660922` completed `0:0` in 2:52.  Full-warp fused transport is now
8.320 us/layer median versus control cast+publish 1.664+6.464=8.128 us, only a
0.192-us local penalty (~0.036 ms/188 rounds).  The headline gap instead
coincides with attention wait 115.936 -> 118.336 us and the byte-identical
FP8xFP4 MegaMoE kernel 51.968 -> 54.944 us; dense GEMMs, router, QKV publish,
norm, RoPE, and model cast are unchanged.  This confirms a residual A-to-F
cadence/communication effect rather than FFN code.  Clean remote commit
`dbb5655` quantizes four heads concurrently in four eight-lane subgroups while
preserving the launch, payload, scale math, early handoff, and 16-byte stores;
21/21 contracts pass.  Four-GPU short smoke `6661262` completed `0:0` in 3:13
with fresh JIT prebuild and repeated two-pair/three-row/nine-generation
byte-exact transport validation.  Exact candidate-first/control-second short
job `6661613` is submitted on one four-tray segment with explicit FP8xFP4 and
a 30-minute per-case timeout.  Adopted mainstream remains FP8xFP4-only
`7f5c273`.  A targeted control/fused publisher comparison found one further
unfixed A-to-F difference: control maps each warp to four adjacent heads and
publishes their scales as one packed 32-bit remote store, whereas current
fusion groups four grid-stride-separated heads and emits four byte scale
stores.  If the isolated subgroup run remains regressed, restore that exact
spatial/store shape next without changing launch or FFN parameters.

Exact pair metrics from `6661613` are now durable.  Candidate/control strict
means are 22.690983467/22.735555867 ms, medians are 22.685218/22.728607 ms,
and TPS/GPU is 132.211105/131.951909.  Both retain 15/15 samples with zero
outliers and similar 0.571/0.588% dominant ranges.  The subgroup candidate is
0.044572400 ms / 0.196047% faster and improves TPS/GPU by 0.196432%.  Its
representative attention/model spans each improve about 0.0446 ms, consistent
with an attention-side cadence improvement propagating through the pipeline.
CPU-short four-report trace job `6663587` completed `0:0` in 32 seconds.  The
subgroup fused attention tail is 6.816 us/layer median versus control
cast+publish 1.664+6.432=8.096 us, a 1.280-us / 15.81% improvement; FMHA is
identical at 112.160 us.  The representative FP8xFP4 MegaMoE median nevertheless
moves 52.864 -> 54.208 us while dense GEMMs, routers, QKV publish, norms, RoPE,
and model cast remain effectively unchanged.  Thus A1 end-to-end improves, but
its strict no-worse-FFN trace gate is not yet met and the evidence strengthens
the suspected attention-to-FFN cadence/communication interaction.

The requested final scaling gate is now active at A4:F1/EP8/ATP1/128K/batch 6.
Short job `6663641` uses clean candidate `dbb5655`, ten contiguous trays, 32
attention plus eight exclusive FFN GPUs, MB2 3+3, 192 real prompts, exact
839,424-token capacity gating, and FP8xFP4-only case
`i131072-fep8-r4-atp1-b6-fp4`.  It compares against accepted sweep jobs
`6530508`/`6531018` at 24.951913467 ms and 192.370016 TPS/GPU.  CPU extraction
job `6663725` confirmed that baseline is also FP8xFP4 and established its
attention cast+publish median at 1.664+7.552=9.216 us and MegaMoE median at
70.080 us.  Job `6663641` was pending on short-queue priority at the first
five-minute check; no GPU was held and no hang signal existed.

Final A4 job `6663641` completed `0:0` in 22:30 on contiguous
`nvl72009-T[01-10]`.  Five-minute fixed-path checks showed all ten tray progress
files advancing through initialization and 128K prefill; no hang occurred.
The strict result retains 15/15 steps with zero outliers and a 2.234% dominant
range: 22.961559467-ms mean, 22.929184-ms median, and 209.045035 TPS/GPU.
Versus the accepted sweep baseline, latency improves 1.990354000 ms / 7.976759%
and throughput improves 8.668201%.  Attention/model role means are
22.961559467/22.871352867 ms.  CPU trace job `6665325` completed `0:0` in 30
seconds: candidate fused attention publication is 6.752 us median versus the
baseline 9.216-us cast+publish path, improving 2.464 us / 26.736%; candidate
FP8xFP4 MegaMoE is 48.064 us versus baseline 70.080 us, improving 22.016 us /
31.416%.  The result explicitly records MegaMoE backend `megamoe`, FP4 expert
weights, all 40 GPUs active, and FMHA-only placement.  This passes the requested
comparison to the previous A4 sweep at the precision/kernel-family level, but
it is not a causal no-worse-FFN proof: the sweep used old source `24b90c`, while
the candidate contains the subsequently accepted MegaMoE route-readiness,
rank-epoch/count caching, and scheduling work.  The kernel signatures differ,
and the old-to-current tree changes FFN production sources.  By contrast, the
exact-current-code A1 control/candidate pair has no FFN source diff and moves
MegaMoE 52.864 -> 54.208 us; that small regression remains consistent with an
attention-to-FFN cadence/communication effect.

The clean adopted remote FP8xFP4 worktree was promoted after a dry merge-tree
proved its result exactly matched validated candidate tree
`1289829288c325e92554962d021232e6c933b17b`.  Merge commit `0bee9f3` now heads
`codex/fmha-megamoe-fp4-mainstream-20260827`; it is clean and contains the
validated `dbb5655` attention fusion.  The final diff changes attention runtime,
transport, smoke, and contracts only; no FFN/MegaMoE production file is changed.
The adopted tree has bit-exact transport smoke.  One-tray short job `6665732`
completed the remaining unchanged-scorer full-model alignment in 16:05 with
exit `0:0`, using all 48 samples from exact A1 source tree `dbb5655` (identical
to adopted `0bee9f3`).  Top-1/top-10/top-100 are all 1.0; average and maximum
vLLM rank are both 1.0.  Representative attention rank 1 and model rank 9 each
contain 16 graph launches, 9,024/33,264 graph kernels, and the expected AFD
NVTX ranges.  Compact summaries and Slurm logs are local under
`scratch/fmha_attention_fused_staged_publish_20260828/`
`final_a1_ep8_alignment_0bee9f3_20260828/`; the full alignment stays remote.
This is the final alignment proof for retained tree `0bee9f3`; its validated
candidate tree is byte-identical, so no redundant alignment rerun is needed.

The next attention-only causal iteration targets the remaining publication
difference instead of changing FFN.  Remote branch
`codex/fmha-attention-packed-scale-20260828`, commit `b3338fe`, assigns each
warp four adjacent heads during both fused O quantization and staged
publication.  This preserves same-warp ownership and the existing early turn
release while replacing four remote UE8M0 byte stores with one aligned 32-bit
store for every complete descriptor group; partial or unaligned groups retain
correct byte ownership.  No FFN source or launch parameter changes.  Remote
source contracts pass 21/21 and `git diff --check` is clean.  The hypothesis is
that fewer/coalesced scale transactions preserve the fused attention saving
without perturbing the concurrent A-to-F/MegaMoE cadence.
Four-GPU short-queue smoke job `6666759` uses the checked-in reusable fused-cast
smoke with a fresh transport JIT cache and a 30-minute cap; output root is
`fmha_attention_packed_scale_20260828/smoke_b3338fe/`.
It completed `0:0` in 3:26: fresh transport prebuild passed and two fabric
pairs completed nine repeated turn/counter replays with bit-exact FP8 payloads
and UE8M0 scales.  Same-allocation four-tray A1 pair job `6667041` completed
`0:0` in 36:48 on `short`; candidate `b3338fe` and exact control `7f5c273`
each retained 15/15 samples with zero outliers.  Strict CUDA is
22.693617533/22.724339200 ms, so the candidate improves 0.030721667 ms /
0.135193% and TPS/GPU improves 0.135376%.

CPU-only trace extraction job `6668636` completed `0:0` in 33 seconds.  The
candidate fused attention tail is 6.784 us versus the control's
1.664+6.496=8.160-us cast/publish path, a 1.376-us / 16.863% improvement; FMHA
itself remains 112.320 versus 112.288 us.  Raw MegaMoE is 54.272 versus 52.496
us, but this is not slower FP8xFP4 math: model `wait_ready` simultaneously
moves 124.160 -> 122.240 us.  The fused handoff starts MegaMoE 1.920 us earlier
and shifts 1.776 us of A-to-F readiness wait inside that communication-bearing
kernel.  `wait_ready + MegaMoE` is therefore 176.512 versus 176.656 us, 0.144
us faster, and the MegaMoE signatures are identical.  This resolves the
apparent FFN regression as accounting migration caused by handoff timing.

Retain adopted `0bee9f3`, not packed-scale experiment `b3338fe`: `0bee9f3`
has the larger exact-pair E2E gain (0.196047% versus 0.135193%) and the larger
combined model-path gain (0.448 us versus 0.144 us), while its fused attention
tail is still 15.81% faster than exact control.  The adopted remote worktree
already remains clean at `0bee9f3`; the local transport source was restored to
the same SHA-256.  Together with alignment job `6665732`, the final goal is
closed: attention improves, the FFN compute implementation is unchanged, the
proper communication-bearing FFN subsystem is no worse, end-to-end improves,
and official alignment is unchanged.  Task root and compact trace summaries:
`fmha_attention_packed_scale_20260828/`
`paired_exact_packed_scale_control_20260828/` and
`scratch/fmha_attention_packed_scale_20260828/`
`paired_exact_packed_scale_control_20260828/paired_trace_analysis/`.
See the subproject memory and local
`scratch/fmha_attention_fused_fp8_20260828/` plus
`scratch/fmha_fused_fp8_casts_20260828/trace_comparison/` for provenance.

## Rejected fused FP8 cast iteration (2026-08-28 PDT)

The active exact target is A1:F1/EP8/ATP1/128K/batch 6, with clean matrix job
`6624199` as control at 22.718287733-ms strict CUDA mean and 132.052205 TPS/GPU.
The user requires FP8-activation/FP4-weight MegaMoE exclusively; the candidate
fails closed without lane-local FP8 input and every GPU gate must prove
`precision=fp8_fp4`.  Clean isolated remote commit `f43a2fe` fuses attention O
FP8 casting with remote publication and fuses FFN add-RMSNorm with MegaMoE's
per-32 FP8 cast, without changing alignment, routing, memory, or launch-policy
parameters.  Focused contracts pass 21/21.  CPU prebuild job `6651632`
completed `0:0`.  Remote commit `4868f08` additionally makes FP4 the default
and only accepted MegaMoE weight precision in the adapter, launchers, bundles,
and new online plans.  Initial short smoke `6651845` failed before kernel work
because its four ranks raced on the default home cache; `11efeb0` moves and
prebuilds that extension in task-local Lustre storage.  Retry `6652065` passed
the repeated fused transport test, then stopped before FFN validation on the
same class of missing FlashInfer home cache.  Commit `dc4a044` isolates that
cache and makes the smoke itself FP4-only.  Short-QoS retry `6652285` completed
`0:0`: repeated transport passed, fused norm/residual/FP8/scales were bitwise,
and production MegaMoE reported FP4 and status ok.  This released exact clean-
head four-tray short job `6652521`, which is running from `dc4a044` with
explicit MegaMoE FP4 and the unchanged warmup-plus-15 contract.  It completed
`0:0` in 18:18 with one allowed external outlier and 14 tightly clustered
retained samples: 23.237812571-ms strict mean and 129.099931 TPS/GPU.  Versus
clean control `6624199`, latency regresses 0.519524838 ms / 2.286813% and TPS
falls 2.235687%.  Attention nodes drop 752 -> 564 but both attention and model
spans regress about 0.52 ms, so the combined fused-cast candidate is rejected
and must not be promoted.  The FP8xFP4-only MegaMoE invariant remains required
for future work.  Its precision-only commit was cherry-picked onto adopted
mainstream `f14d800` as clean `7f5c273`, with no fused-cast code; 21/21 focused
contracts and launcher/diff checks pass there.  Compact terminal artifacts are
local under
`scratch/fmha_fused_fp8_casts_20260828/exact_a1_f1_ep8_128k_b6`; see
`CODEX_PROJECT_qwen3_ffn_overheads.md` for full provenance.

## Completed 42-case optimized-attention sweep (2026-08-28 PDT)

The 128K FP8xFP4 FMHA-only matrix documented at the top of
`CODEX_PROJECT_qwen3_ffn_overheads.md` is complete.  Seven `short` allocations,
one per tray count, used four lead jobs (`6624181`, `6624184`, `6624183`,
`6624182`) and three dependency-gated jobs (`6624199`, `6624197`, `6624198`) to
enforce at most four concurrently runnable jobs.  All completed `0:0`; the pool
has 42 completed and zero claimed/pending/failed.  A strict full-plan audit
passes all contracts and counts 1,512 Nsight reports; all cases retained 15/15
samples with zero outliers.  Compact reports, results, logs, and state are local
under `scratch/afd_128k_optimized_attention_near_pareto_20260827` and
checksum-identical to the remote task root.  Source was clean OCI mainstream
`f14d8005`; see the subproject memory and `report/completion-audit.json` for
job, metric, capacity, and provenance details.  Raw Nsight reports for the
highest-TPS/GPU case (72 files) and highest-User-TPS case (16 files) were later
copied into their local result trees with full SHA-256 parity; other raw
captures remain on Lustre.

## FP8xFP4 MegaMoE communication optimization (2026-08-27 PDT)

The active work keeps clean pre-FFN baseline `7c1491e` and the unchanged
alignment standard, targeting general FP8xFP4 same-rank MegaMoE improvements
for A7:F1/EP4/128K/b6 without parameter tuning.  Remote branch
`codex/fmha-megamoe-fp4-comm-profile-20260827` adds opt-in intrinsic timing at
`d049f33`; CPU prebuild `6596879` and four-GPU A7-shape job `6596931` pass.
Fused gate+Mega is 94.202 us at rows 21/capacity 3,840/150 SMs.  First L1
readiness is 21.5--22.2k cycles, first MMA-full wait 22.2--23.4k, dispatch
pre-pull 16.8--49.2k, pull 10.6--17.2k, and overlapped cleanup 133--140k.

General candidate `bfc9794` makes top-k selection group routes, publish remote
source indices, and finalize expert counts with a last-block acquire/release
ticket; prepared Mega skips route scan/count/reservation/publication and begins
at its rank barrier.  CPU prebuild `6597093` and 21/21 focused contracts pass.
AB job `6597132` is bitwise exact but rejects the first form: 94.369-us control
versus 98.279-us fused, because redundant system fences erase the shorter Mega
startup.  Commit `fba56ed` removes those fences while retaining ticket and
next-kernel rank-barrier ordering.  AB job `6597285` is then positive at
94.476-us control versus 92.413-us fused (-2.18%).  Timing/key follow-up
`974520b` passes bitwise mirrored trials: A7-shape ABBA job `6597377` averages
95.922-us control versus 93.359-us fused (-2.563 us / -2.672%), and A1 guard
`6597409` averages 76.005-us control versus 73.917-us fused (-2.089 us /
-2.748%).  The cross-capacity agreement makes route publication a retained
general mechanism, not an A7 parameter choice.

Remote candidate `ce48f79` independently replaces the full pre-combine rank
barrier with one release-published completion epoch per producer rank; token
warps wait only for ranks named by their top-k experts.  It adds one `uint64_t`
per rank to the workspace and no threshold or scheduling parameter.  The
control remains compile-time clean.  Local and remote diff/shell checks and
21/21 contracts pass.  CPU prebuild `6597490` completed `0:0` in 2:17.
Four-GPU FP8xFP4 job `6597585` is submitted as the single A7 gate, holding
route publication fused while alternating combine modes
`control,epoch,epoch,control`.  It completed `0:0` and bitwise exact: controls
average 93.377 us, epochs 91.371 us (-2.006 us / -2.149%).  Against original
job `6597377` controls, route fusion plus epochs improve 95.922 -> 91.371 us,
or 4.551 us / 4.745%.
Full A7/EP8 is the final performance criterion, but its 16-tray/64-GPU run is
withheld until a significant gain exists.  The intermediate gate is a
two-node/eight-rank FP8xFP4 MegaMoE proxy at A7's 21 rows, capacity 3,840, and
16 local experts per rank, reproducing EP8 communication/kernel topology
without allocating the 56 attention GPUs.  Remote harness commit `e4adccd`
passes 21/21 contracts.  Initial proxy `6597660` was canceled after 42 seconds
because `--segment=1` allowed its two nodes onto different NVL72 systems and
produced no valid result.  Corrected job `6597693` completed `0:0` in 1:48 on
one fabric with bitwise equality: clean controls average 74.475 us and both
mechanisms average 70.262 us (-4.212 us / -5.656%) at 128 total/16 local
experts.  This clears the full-run gate.  The exact one-row A7/EP8/b6 FP8xFP4
plan retains 16 trays/64 GPUs, 128K, MB2 3+3, ratio 0.82, no page override,
839,424 KV tokens, and one warmup plus 15 measured steps.  Remote dry and full
non-submitting validation pass from clean `e4adccd`.  Acceptance job `6597800`
ran on contiguous `nvl72098-T[01-16]` and completed `0:0` in 27:43.  Its
unchanged strict metric retains 15/15 steps with zero outliers:
23.819227600-ms mean, 23.767363-ms median, 24.597347-ms maximum, 3.633769%
dominant range, and 220.410170 TPS/active GPU.  Versus clean job `6583163` at
24.037447667 ms / 218.409212 TPS, latency improves 0.218220067 ms / 0.907834%
and throughput improves 0.916151%.  All 64 Nsight reports (64,987,911 bytes)
and compact results are local with checksum parity.  The final A7/EP8 surface
therefore inflects positively and no further 16-tray run is warranted for
this checkpoint.  The user requested continued general optimization before
terminal alignment, especially dispatch/combine and ideas from the local
TensorRT-LLM MegaMoE implementation.  Read-only inspection of TensorRT-LLM
head `fa778397` finds its fused prepare already subsumed by FastAFD's router
quantization and top-k route publication.  Its transferable mechanism is
per-expert FC2 completion.  The implementation is a separate compile-time
specialization with no launch, threshold, scheduler knob, or arithmetic
change; full-barrier and rank-epoch controls remain.  Pending `6598831` was
canceled before allocation to correct environment export, and `6598856` was
canceled after 13 seconds when review found a prior-generation reset race.
Commit `04ea787` added epoch-selected banks; `6598896` exposed and closed only
a device-`constexpr` ODR compile issue.  Corrected job `6598940` is bitwise but
rejects system scope on every FC2 tile: rank trials average 92.112 us versus
125.860 us expert-gated, with writeback expanding from roughly 3--5k to
45--75k cycles.  GPU-scope intermediate counts at `18b7730` avoid that cost
but job `6599060` proves mixed-scope updates on one word unsafe (`ready=30` or
`0` timeout).  Commit `e631808` separated a producer-local 32-bit GPU counter
from a monotonic system-scope readiness epoch, but repeated-graph job `6600341`
also failed after 7:32: expert 31 remained at epoch 830 for target 832.  Thus
the only correct per-expert form is the 36.6%-slower system-atomic version;
the mechanism and its workspace/API plumbing are rejected and removed.
Remote commits `a0189a2`/`6e06b54` briefly prepared an isolated route-ready
dispatch experiment, but the user chose to return to the last measured-positive
checkpoint before spending more GPU time.  Job `6600718` was canceled during
JIT at 1:49 and produced no performance result.  The local and remote source,
wrapper, smoke harness, and contracts are now byte-identical to retained commit
`e4adccd`: fused route preparation plus rank-gated combine, with no per-expert
or route-ready experimental plumbing.  Local diff checks and 21/21 contracts
pass.  Requested four-GPU FP8xFP4 FFN-only control-versus-retained validation
job `6600859` completed `0:0` in 2:33 with bitwise equality and the normal
numeric guard (`cosine=0.975472`, finite output).  At rows 21/capacity 3,840/
150 SMs, clean route-scan/full-barrier control is 94.906 us and the retained
fused-route/rank-gated path is 91.861 us, a 3.044 us / 3.208% reduction.  This
confirms the revert retained the measured speedup.  Stage the local checkpoint
before testing the isolated route-ready commit or examining TensorRT-LLM's NCCL
one-sided communication support.

## Accepted release-published route readiness (2026-08-27 PDT)

The mirrored OCI mainstream is branch
`codex/fmha-megamoe-fp4-mainstream-20260827`; accepted code commit `61ec1a8`
makes MegaMoE consume release-published prepared routes while retaining fused
route preparation and producer-rank-gated combine.  The production target and
all acceptance evidence below use FP8 activations times FP4 expert weights.
The change is topology-derived rather than specialized for EP4 or EP8.

Correctness and indirect performance gates passed before the scarce exact run:
A7 proxy job `6601169` was bitwise correct and improved 94.575682 to 89.313679
us (5.563801%); A1 proxy `6601275` was bitwise correct and improved 75.428400
to 70.281761 us (6.8239%); explicit eight-rank EP8 FP8xFP4 proxy `6601691`
was bitwise correct and improved 71.620162 to 66.333759 us (7.381166%).

Exact 16-node/64-GPU A7:F1/EP8 job `6603099` completed `0:0` on contiguous
`nvl72083-T[01-16]`.  Every FFN rank reported `precision=fp8_fp4`, eight group
ranks, 128 group experts, 150 compute SMs, and two reserved SMs.  All 56
attention and eight FFN/model graphs captured; logs contain no traceback,
timeout, CUDA error, or assertion.  The strict 128K/batch-6 metric retained
15/15 samples with no exclusions or outliers: 23.613828733 ms mean,
23.551799 ms median, 24.454495 ms maximum, and 222.327351455 TPS per active
GPU.  Versus accepted retained job `6597800`, latency fell 0.862324% and TPS
rose 0.869824%; versus clean baseline `6583163`, latency fell 1.762329% and
TPS rose 1.793944%.  The route-ready series is therefore fully accepted.

TensorRT-LLM inspection found that production MegaMoE `NVLINK_ONE_SIDED` is
not NCCL one-sided communication: it uses CUDA VMM fabric/FD handles, MPI
handle exchange, direct peer loads/stores, push dispatch, direct receive, and
pull combine.  FastAFD already uses the same class of direct one-sided
transport through Torch symmetric memory.  TensorRT-LLM's NCCL 2.28 symmetric
windows serve dense all-reduce, not MegaMoE put/get; its counted-write path
also needs CUDA 13.4+/IMEX while the current image is CUDA 13.0.  No NCCL
one-sided MegaMoE change is justified from this inspection.

Follow-up disposition after the exact route-ready acceptance:

- Match-any dedup job 6601728 was correct but neutral (-0.0047%); reject.
- Last-SM publisher job 6602233 timed out; designated-CTA ticket job
  6604682 later stalled at replay generation 113 and hit rank-gated combine
  timeouts; reject both resettable publication-counter designs.
- Grid-completion early-publication job 6605172 was correct but regressed the
  A7 proxy from 89.785919 to 90.334396 us (+0.6109%); reject.
- Cleanup-participant job 6602592 was correct and improved the A7 proxy from
  89.490080 to 89.187679 us (0.302401 us / 0.3379%).  A1/EP4 guard job
  6605708 was also correct and improved 70.078721 to 69.917440 us (0.2301%).
  The participant count is derived as min(kNumSMs, kNumExpertsPerRank + 1);
  there is no EP-degree literal or branch.  A two-GPU submission was rejected by
  QOSMinGRES before allocation, and no EP2-specific run is required.  Commit
  25e6f5b is therefore promoted to the mirrored mainstream provisionally,
  pending inclusion in a later accumulated exact A7/EP8 acceptance series.
- Release-only ticket refinement 0029779 remained correct in job 6606148
  but regressed the A7 ABBA mean from 89.443679 to 89.584479 us (+0.1574%);
  reject it and retain the acquire-release publisher in 25e6f5b.
- Expert-major receive-count layout b0c0c8f made dispatch warp reads
  contiguous and job 6606610 stayed correct, but the A7 ABBA mean regressed
  from 89.517121 to 90.489759 us (+1.0865%).  Sampled pull and cleanup cycles
  fell while full-kernel time rose in both candidate trials, indicating that
  shifted remote-publication/cache cost outweighed the coalesced reads; reject.
- One-vote combine rank mask 047e74f replaced the loop over ranks with a warp
  OR reduction.  Correct job 6607004 reduced sampled combine cycles but the A7
  ABBA mean regressed from 89.720483 to 89.874082 us (+0.1712%); reject.  Job
  6606959 was only a 21-second clean-head preflight failure caused by a wrong
  expected hash and launched no workload.
- Rank-major dispatch pool b3eaa39 replaced iterative min-peeling with a warp
  prefix scan while preserving workspace layout.  Job 6607158 was correct and
  reduced mean dispatch-pull cycles 16145.75 to 15671.25 (-2.94%), but the A7
  ABBA kernel mean regressed from 89.092002 to 89.347038 us (+0.2863%).  The
  reordered pool shifts cost downstream, so reject this TRT-inspired mapping.
- Order-preserving dispatch active-mask commit 4324734 reused ballots and removed
  the warp sum.  A7 job 6607349 was correct and improved 89.572001 to
  89.135842 us (-0.4869%), but A1 guard 6607492 was also correct and regressed
  69.996800 to 70.250239 us (+0.3621%).  The cross-shape sign flip fails the
  generality gate; reject and do not promote.

## Provisional rank-granular route publication (2026-08-27 PDT)

Nsight evidence from exact accepted job 6603099 localized the remaining FFN
startup cost: across 3,008 calls, gate GEMM/quant averaged 4.586 us, fused
top-k/route preparation averaged 9.437 us, and MegaMoE averaged 51.723 us.
The mean gate-to-select and select-to-Mega launch gaps were only 0.506 and
0.514 us, so eliminating another launch could save at most about 1 us.  The
next attempt therefore reduced completion polling on the dispatch critical
path instead of fusing kernels.

Rank-granular publication commit f8ea0e2 retains every exact per-source,
per-expert count and source index, but replaces one system release atomic per
global expert with one per destination rank.  One warp per CTA acquires that
rank completion, aggregates local expert totals once into shared memory, and
all scheduler roles reuse those totals.  Expert/token alignment, pool order,
capacity, math, and EP behavior remain topology-derived and unchanged.

An initial CTA-sharing implementation accidentally left the scheduler's
prefetched-count branch at zero.  Jobs 6609486 and 6609662 consequently
reported false 30-31 us timings by reusing stale internal expert outputs;
their zero MMA/TMA counters exposed the error.  Commit b6a09c4 fixed the
source and strengthened the smoke harness to require output overwrite and
bitwise equality for two distinct inputs.  These false results are rejected.

The repaired FP8xFP4 ABBA proxies all passed the two-input bitwise check and
showed real MMA/TMA execution: A7 job 6610245 improved 90.964479 to
85.866079 us (5.604825%); eight-rank EP8 job 6610356 improved 69.103360 to
64.762399 us (6.281838%); A1 job 6610364 improved 71.435680 to 65.439041 us
(8.394460%).  Focused contracts passed 21/21.  The series is promoted to the
OCI mainstream.  Exact acceptance job 6610567 then ran clean head 63f9c10 on
16 contiguous trays and completed `0:0` in 24:53.  All eight FFN ranks proved
FP8xFP4, EP8/128 experts, two lanes, 150 compute SMs, and two reserved SMs;
56/56 attention and 8/8 model graphs captured with no runtime error.  The
unchanged metric retained 15/15 samples and zero outliers: 23.127204867-ms
mean, 23.070943-ms median, 23.935711-ms maximum, and 227.005383066 TPS/GPU.
Against accepted route-ready job 6603099, latency improves 0.486623866 ms /
2.060758% and throughput improves 2.104119%.  The accumulated series through
rank-granular publication is fully accepted on the target A7/EP8 surface.

### Rejected single-lane rank acquire

Isolated commit 0ed38ac replaced the 16/32 identical rank-ready system
acquires in each CTA with one lane-0 acquire followed by warp synchronization.
It preserved exact counts, routes, alignment, capacity, math, and topology;
21/21 focused contracts and all two-input FP8xFP4 bitwise checks passed.
A7 job 6610948 improved the rank-ready-versus-per-expert delta by 0.230875 us
relative to mainstream job 6610245, and A1 job 6611072 improved it by
0.639203 us relative to job 6610364.  Eight-rank EP8 job 6611073 instead
reduced the delta from 4.340961 to 3.471203 us, an incremental 0.869758-us
regression versus mainstream job 6610356.  The EP-dependent sign flip fails
the generality gate; reject the attempt and do not promote it.

### Rejected per-cluster route-count aggregation

Isolated commit 6803bb2 made each fixed two-CTA cluster leader acquire rank
readiness and aggregate per-rank counts, then shared the totals through DSM.
Routing, alignment, capacity, math, and EP policy stayed unchanged; 21/21
contracts and the two-input FP8xFP4 bitwise checks passed.  A7 job 6611444
improved the proxy delta by only 0.123363 us, while eight-rank EP8 job 6611589
regressed incrementally by 1.620002 us.  Reject the cluster barrier/DSM handoff.

### Provisional dispatch rank-count cache

Commit b316180 reuses the existing expert-count shared-memory allocation to
cache all but the final per-rank count during the accepted rank-ready
aggregation.  Dispatch warps consume coalesced shared counts; the final rank is
reconstructed from the exact expert total, so there is no extra shared memory,
route-order change, threshold, EP literal, capacity change, or math change.
A7 job 6612346 passed the two-input FP8xFP4 check and improved the current
mainstream candidate from 85.866079 to 85.403519 us.  Eight-rank EP8 job
6613394 also passed and improved 64.762399 to 63.244481 us.  Promote this
general proxy winner provisionally; it awaits a future accumulated exact A7/EP8
series rather than another standalone 64-GPU run.

Wrap-up attempted to replace the unrolled final-rank ownership scan with one
direct indexed store.  EP8 job 6613604 remained correct with empty stderr but
the candidate average regressed to 64.574399 us.  The proven unrolled form is
therefore intentional GPU code-generation control, not removable redundancy;
the cleanup commit was reverted before mainstream promotion.

Final A1/EP4 alignment used the same clean mainstream source.  Combined source
and inline-score job 6613964 generated a valid fresh sample but its vLLM shard
used a missing home-Conda path; scorer retry 6614843 proved perfect agreement
but failed its post-check because the interrupted combined run lacked adjacent
`afd-result.json`.  No numerical failure occurred.  Clean source-only job
6615896 then completed `0:0` in 17:25 with 24 samples x 17 tokens, FP8xFP4 on
all four FFN ranks, eight Nsight reports, `SUCCESS`, and complete result
metadata.  Unchanged official scorer job 6617291 completed `0:0` in 11:26:
top-1/top-10/top-100 agreement, average rank, and maximum rank are all 1.0;
attention-rank-1 and model-rank-5 trace inspection also passed.  The remote
mainstream is ready for exact local synchronization and fresh staging.

## Accepted same-rank MegaMoE FP8xFP4 precision (2026-08-26 PDT)

See `CODEX_PROJECT_qwen3_ffn_overheads.md` for the authoritative record.
Remote executable commit `6c210cf` adds an explicit FP8-activation/MXFP4-
weight option while retaining FP8xFP8 as the default. Exact
A1:F1/EP4/128K/b6/MB2 job `6561851` completed on `short` within its 30-minute
limit. Unchanged strict extraction job `6562231` retained 15/15 steps with no
outliers and measured 22.689653667 ms, 0.701194333 ms / 2.997729% faster than
accepted FP8xFP8 job `6520395` at 23.3908480 ms. Unchanged alignment job
`6562471` passed all 408 tokens with top-1/top-10/top-100 agreement and
average/maximum rank all 1.0. The FP8xFP4 goal is complete.

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
28.1574149333 ms strict CUDA versus 29.3581909333 ms baseline (4.09% faster),
and the final isolated M32 target routes reach 80.446--80.477 us.  Official A1
alignment job `6552212` scored 24 prompts / 408 tokens with top-1/top-10/
top-100 and average/maximum rank all 1.0.  Remote clean head `2075450` contains
the winning policy and final launcher cleanup; the user-defined terminal A1
gate passes, so this goal is closed unless explicitly reopened.

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
