# FastAFD project memory

September28 checkpoint request: preserve all current dirty MegaMoE code/workflows/
results documentation on branch megamoe_throughput and push to linsj20/FastAFD.
Then restore local fmha_only and remote working code to September22 commit
2e9186cdaf27e5121c5aef9c5b7cc0f6ed6f8b4b. Preserve experiment artifacts and frozen
sources; no new GPU work. Restoration receipts live in ignored scratch/megamoe/
commit_megamoe_throughput_20260928 and corresponding HSG artifacts. This request
supersedes the earlier no-commit constraint; historical notes below remain evidence.

September22 baseline provenance verified September28: HEAD is2e9186cdaf27e5121c5aef9c5b7cc0f6ed6f8b4b
(September22 19:01:58PDT). Commit message and committed project memory explicitly
record these completed all-push/pull campaigns and their results. All-push source.patch
post-images for runtime.hpp(4c581ca) and sm100_fp8_fp4_mega_moe.hpp(d1b84db)
match HEAD blobs exactly; these are the only recorded kernel-side sweep changes
(1024→8192 host limits), device kernel unchanged per campaign/commit. Captures ran
before the commit was made, then results/source were checkpointed in it. This is
the last-commit kernel-performance baseline before current dirty optimization code,
not a fresh benchmark launched from a clean checkout of that commit. Saved interim
planner/precompiler/plot script patch blobs differ from final committed scripts;
no claim of whole-repository byte identity or capture-embedded commit proof.


Reporting correction September28: user requested latest sweep baseline from the
September22 commit checkpoint, replacing the September26 frozen-reference label.
Use ordinary_sweep_8k_20260922/{qwen3,qwen35} and
pull_original_sweep_8k_20260922/pull/{qwen3,qwen35}, recorded in2e9186c.
Compare current control using average forward/reverse rank0 kernel medians against
historical rank0 medians (1000samples); separate allocations, descriptive only.
B/september22_comparison.{json,csv}:113matched power-of-two cases. Extra192/912
controls absent from historical sweep. Fixed91 geomean latency change vsoldallpush/
oldpull: combined-3.406831%/+0.203328%; Q3-6.341038%/+0.551772%;
Q35+0.467982%/-0.239186%. Context22:-38.204497%/+19.437252%.
Report total batch size, not input sequence length; current means existing-policy
control, not resource variant. No new GPU run or change to completed goal.


Latest status September28 19:55UTC: COMPLETE under the user's instruction to finish
this sweep, report perISL results, and close the goal. All six HSGshort GPU jobs and
six CPU analyses completed0:0; no active monitors or further GPU submissions.
EP64 CPU7516325 completed1h11m45s;16cases/128captures/8,192,000rankcalls PASS.
Full B:115unique cases/920captures/18,000,000rankcalls; correctness2250record,
11250compare,18000replay. All ranks/orders/timestamp audits and five pair reviews PASS.

Final fixed91 kernel/span speedups versus retained all-push:
- Cleaned existing-policy control:1.013803710/1.012155868.
- Resource policy:1.024409970/1.020959510. Both pass agreed >=.99 fullgeomean.
- Separate8smallcontrols control:1.002563405/1.002302229; resource:.976377005/.981644190.
  Resource smallcontrols regress2.42% kernel latency; retain caveat, no universal
  preservation claim. Q3EP16ISL192+17.7988%, Q35EP16ISL912+5.7437%.
- Context22(4K/8K) resource:1.239923317/1.199363444; informational peruser.
- Fixed91 resource/pull:.996183900/.991611188, diagnostic only.
- Historical rank0median fixed91 control1.006179548/resource1.016479697;
  separate-allocation descriptive comparison, not paired allrank attribution.
No defaults changed, no further optimization runs, no commit. Source/helper cleanup
and CPU codegen gates remain as documented below. All perISL losses are preserved.

Authoritative final artifacts under local/remote
D/acceptance_pass4_distinct_orders_20260928/: final_report.md (permodel/EP/ISL
tables for both policies), isl_comparison.{json,csv}, full_comparison.json,
historical_comparison.{json,csv}, selection_review.json, campaign.json,
and ep*/selection_review.json plus five allrank reviews perEP.

The earlier progress snapshots below are historical; pending-job and open-goal
statements there are superseded by this completed status.



## Completed MegaMoE cleanup — historical progress and evidence

Goal: deeply clean all-push kernels/workflows and validate ordinary Qwen3 EP2–64 /
Qwen3.5 EP4–64, totals EP..8192 against retained performance. HSG short, one GPU
job per EP with models combined; CPU preparation/analysis separately. Monitor
active jobs every300s. No subagents, commit, new project docs, /tmp, py_check or
hash audits. Root AGENTS.md absent; creation/location already asked about.

Read [optimization memory](CODEX_PROJECT_megamoe_optimization.md) for evidence,
rejected paths and remaining work. R=/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD;
D=R/artifacts/deep_cleanup_20260927; local D=scratch/megamoe/deep_cleanup_20260927.
Frozen measured source R/source/megamoe_cleanup_pass3_20260928 has44suffix args;
retained sender_middle_20260926 has59. Final-named intermediate has48. Preserve
frozen sources/caches/results. Lower-EP experiments keep pass3. New cleanup pass4
below has43suffix args and a fresh CPU gate; no default change.

User clarification September28 ~16:28–16:32UTC: accept about<1% geomean
performance variation versus retained all-push; do not require beating pull.
Report ISL by ISL (total tokens in these sweeps), separately permodel/EP.
Slower4K/8K is explicitly allowed; retain correctness coverage there but do not
reject performance for those sizes. Keep the existing91sub4K equal-case geomean
(minimum.99speedup) and report all per-ISL losses; no new per-case/per-EP win gate.
Finish current B validation; do not launch additional GPU sweeps merely to beat pull.
The earlier F full sweep already passes this clarified cleanup performance criterion:
rows/reference1.010910kernel/1.010094span; separate smallcontrols.995597/.996196.
Previous miss was rows/pull.977835/.976284, a different implementation comparison.
Historical rank0-median ratio1.008374 is descriptive, separate from pairedallrank means.
Assistant acknowledged it overextended cleanup acceptance by requiring a pull win.

Current acceptance B=D/acceptance_pass4_distinct_orders_20260928:115unique
cases/920captures;91fixed,22context and8retained controls scored separately.
Reference/control/resource/pull, both orders,1000calls; pass4_native kernel unchanged.
All six CPU gates in A=D/acceptance_pass4_20260928 remain valid: EP2/4/8/16/32/64
jobs7513968/7513969/7513970/7514112/7514227/7514228 completed0:0. Strict same-code
specializations6/13/11/10/9/18, scheduling memberships26/48/44/44/36/32.
B explicitly reuses those validated native/JIT/TVM caches; sources/contracts unchanged.
Remote plan validation PASS all6; new launcher rejects the old plan. B has920distinct
root/case/phase outputs, forward/reverse roots separate; same full scope and orders.

A GPU campaign REJECTED: plan transformation removed _repeat from output roots,
so forward/reverse arms alias. Earlier104/104 progress double-counted existing
files and is invalid; overwritten captures cannot prove paired performance.
Cancelled GPU7514114/7514234/7514235/7514247/7514339/7514340
and analyses7514238/7514375/7514381. Preserve A evidence/caches; no results accepted.
A/output_alias_review.json proves only460unique destinations for920planned captures;
prior F full1150 and lowerEP296 outputs are distinct and remain valid evidence.
Maintained run_megamoe_sweep_group.py now fails before launch on duplicate resolved
root/case/phase, including symlink aliases and both-phase overlaps. Focused regression
test PASS (valid split stages and normal/profile phases remain allowed). New frozen
B/workflow includes this guard. This is a corrected new plan, not an unchanged retry.
All nine rejected GPU/analysis jobs confirmed CANCELLED; monitors terminal.
Corrected B GPU EP2/4/8/16/32/64:7514495/7514498/7514500/7514502/7514504/7514506.
EP2 GPU7514495 completed25m27s0:0; CPUanalysis7514509 completed2m58s0:0,
observed15:48:18Z. All104captures/13cases/208000rankcalls PASS correctness,
timing/graph/rank audits;26records/130compares/208replays. EP2selection_review.json
retains all13cases, five rank comparisons, both orders and all losses.
Fixed11case resource/control geomeans1.048513873kernel/1.041971420span;
resource/reference1.096288413/1.085654417; resource/pull0.855144022/0.865414191.
Context2 resource/control1.351704495/1.295913671; resource/pull.917641403/.940707697.
All4full-TMA cases improve everyrank mean/median/p95 bothorders versus control,
kernel+span. Nine unchanged small targets retain variation. Resource2K kernel
mean beats pull bothorders but p95 loses and reverse spanmean loses; no broad
pull win. EP2 passes clarified retained all-push tolerance; full-matrix acceptance remains open.
EP4 GPU7514498 completed59m07s0:0; CPUanalysis7514511 completed11m00s0:0,
observed16:28:27Z. All192captures/24cases/768000rankcalls PASS correctness/timing/
graph/rank audits;96records/480compares/768replays. Fixed20case resource/reference
1.034059178kernel/1.029180700span; resource/control1.013624638/1.011339064;
resource/pull.934531316/.941731853. Context4 resource/reference1.467457395/1.370621061;
resource/pull.951083714/.959356457. EP4 passes clarified retained-reference tolerance.
EP2/4 selection_review.json files now reflect this criterion and retain all losses.
B/isl_comparison.{json,csv} reports both control/resource perISL kernel+span versus
reference; EP2/4 findings below, EP8 added after its completed audit. Resource small2..512 varies -0.05%..+0.10% kernel;
1K/2K gains depend on model,4K/8K informational. Add remainingEPs onlyafteraudits.
EP8 GPU7514500 COMPLETED58m14s0:0; CPUaudit7514520 COMPLETED16m46s0:0.
All176captures/22cases/1408000rankcalls PASS;176records/880compares/1408replays.
Fixed18case resource/reference1.009103644kernel/1.007007759span;
resource/control1.001183597/1.000646756; resource/pull1.017664461/1.014873204.
Context4 resource/reference1.069647234/1.055881589; resource/control.998582519/.999023643.
Smallcontrols6 resource/reference1.004745116/1.004114769. Q35ISL64 kernel+1.866%
vsreference is retained; tolerance is aggregate, not a new per-case gate.
EP8selection_review.json preserves all22cases/fivecomparisons/ranks/stats/orders.
B/isl_comparison.{json,csv} now includes auditedEP2/4/8; remainingEP16/32/64 pending.
EP16 GPU7514502 COMPLETED1h02m46s0:0; CPUanalysis7515082 COMPLETED27m17s0:0,
observed17:49:54Z. All176captures/22cases/2816000rankcalls PASS correctness/timing/
graph/rank audits;352records/1760compares/2816replays. Fixed16case resource/reference
.993055100kernel/.994871472span (about0.70%kernel latency regression);
control/reference1.006667225/1.005475596, resource/control.986478029/.989453623.
Context4 resource/reference1.103480510/1.085507382. Extra2controls resource/reference
.895988549/.917204839, separately retained. Q3ISL256+10.9373% and controlISL192
+17.7988% kernel regressions remain visible. No noise subtraction or causal claim;
no further experiments under latest user stop condition. EP16selection_review.json
preserves everycase/rank/stat/order; perEP geomeans diagnostic, full91score pending.
B/isl_comparison.{json,csv} includes auditedEP2/4/8/16, EP32/64 pending.
EP32 GPU7514504 COMPLETED1h00m52s0:0; CPUanalysis7515800 COMPLETED42m03s0:0,
observed18:26:29Z. All144captures/18cases/4608000rankcalls PASS correctness/timing/
graph/rank audits;576records/2880compares/4608replays. Fixed14case resource/reference
1.009009867kernel/1.007194207span; control/reference1.000330102/1.001002314;
resource/control1.008676901/1.006185694; resource/pull1.064709267/1.047416159.
Context4 resource/reference1.144284650/1.118323897. Largest sub4K loss Q3ISL1024
+1.0230% kernel retained. Everycase/rank/stat/order preserved in EP32selection_review.
B/isl_comparison.{json,csv} now includes EP2/4/8/16/32; only EP64 pending.
EP64 GPU7514506 COMPLETED1h17m31s0:0, observed18:39:07Z. All six GPU jobs
now completed0:0. CPUanalysis7516325 RUNNING elapsed00:00:37,
observed2026-09-28T18:40:00.268898+00:00. Only this audit and full report remain.
No further GPU submissions. Twohour HSGshort allocations, onejob/EP.
EP16/32 analyses7515082/7515800 complete; EP64analysis7516325 RUNNING. EP64 analysis submitted
16:59:08Z after EP8 completion freedCPUslot. Live300s monitor74188(analysis64);72236/92868/93787/74850terminal0. Old86640/29380/12223/87318/45616 terminal.
B/aggregate_command.json records explicit control,resource candidates versus
pull,reference; execute only after all six passing summaries. Full performance goal
remains open, no default promotion. Next consume GPU/analysis monitors, submit the
remaining CPU analyses as slots open, audit all cases/ranks/orders and all losses.

Slow-audit diagnosis September28 19:49UTC: EP64 CPU7516325 processes128captures
×64ranks=8192rank traces. Ordinary analysis exports nsys reports serially; timeline
summary launches each rank serially and retains timestamp JSON. Five review-ranks
commands then reread overlapping arm traces. Eight CPU cores allocated but observed
review process uses94.3% of one core. Slurm step counters at~19:48:62m30sCPU,
54798592KiB peakRSS,97197928511read/73760428301write bytes (accounting counters,
not measured storage throughput). Main extraction completed; process observed in
second of five rank reviews (control/pull), no stall. Leave this nearly finished job
running. Future workflow opportunity: bounded parallel trace processing and compute
rank stats once/reuse across comparisons; no implementation/restart under stop scope.

Cleanup CPU7513604 completed5m39s0:0, observed14:35:29Z;26specializations/48case
memberships across EP4bothmodels control/resource. Strict config/instruction
comparison PASS with source/helper-symbol metadata normalization only. All26
instruction streams match pass3. This is CPU proof, not full GPU acceptance.
First7513273 failed5m03s (binding symlink chose pass3); second7513386 failed2m54s
(native C++ symlinks kept44host args against43device args). Both rejected/preserved.
General --own-native copies binding/native tree (60owned files here); precompiler
rejects unexpected binding roots/native symlinks. Focused ownership/base/cycle
checks PASS; maintained native tree has no symlinks. No further kernel edit.
Pass4 removes rejected ROW_RELEASE (61net C++
lines) plus the earlier six-brace-line cleanup. Nonzero retired setting fails;
ordinary word polling remains. Historical7379931/7379932 regress both models/
orders/everyrank by252–349us; archived evidence and before-files retained.
Fresh native/JIT/TVM required because config/template ABI changed. Four EP4
model/policy preparations cover control/resource bothmodels; strict source/helper
symbol-normalized comparison to pass3 passed. Full GPU gate pending; use only the
compatible native from successful7513604, never either failed native cache.

Completed: D/concurrency_lower_ep_20260928. EP2 GPU7512239 completed28m38s0:0,
CPUanalysis7512247 completed2m45s0:0. EP4 GPU7512241 completed1h02m47s0:0;
CPUanalysis7512253 completed10m08s0:0, observed14:31:34Z. Monitor19645 exited0;
live phase monitor_snapshot.json files own status. Q3EP2 powers-of-two totals2..8192
(13cases/104captures); bothmodelsEP4 totals4..8192 (24cases/192captures). Current
rows, worker-only, prospective resource rule, pull; bothorders/1000calls perrank.
One HSGshort GPUjob perEP,90min limits. CPU7511453/7511460 completed4m28s/6m10s0:0;
39+72resource/geometry checks PASS across37cases, all small targets preserved.
Resource rule derives workers/partitions from CTA capacity and local experts,
not measured winners: EP2Q3W64/P1; EP4Q3W64/P2,Q35W64/P1. FreshJIT/TVM caches;
native copied from same pass3 ABI. Prior small8-token regression remains covered.
Lower-EP reviews complete; resource rule is useful. Next: pass4 CPU gate then
full91/context/controls acceptance. No default promotion.

EP2 result: all104captures/13cases/208000rankcalls PASS;26records/130compares/
208replays. All4large cases improve everyrankmean/median/p95 bothorders vsrows/
workers. Fixed11case resource/rows geomeans1.065897kernel/1.058273span; resource/
pull0.855113/0.865398. Context2 ratios1.445459/1.383683vsrows and.906307/.930596vspull.
Large gains are useful but small-path deficits persist; no EP2 pull win overall.
L/ep2/selection_review.json preserves all losses and2K reverse spanp95+.064us
versus pull. Source small workers remain4for64experts and ignore positive overrides;
archived-3/-4 review confirms useful EP2 gains but EP4tiny8 and EP8normal448
regressions, plus EP2tiny2 order sensitivity. Neither unchanged policy is a valid
successor; preserve these holdouts for any new mechanism. Details in topic memory.
No new small policy implemented. Retain full91/context/control acceptance.

EP4 complete:192captures/24cases/768000rankcalls;96records/480compares/768replays,
all correctness/timing/graph/rank audits PASS. Seven changed full-TMA cases improve
everyrankmean/median/p95 bothorders versus rows and workers, kernel/span. Q3total2K
also beats pull by all those measures; other large cases retain pull gaps. Fixed20
resource/rows geomeans1.014787kernel/1.012275span, resource/pull.935575/.942147.
Context4 resource/rows1.171265/1.134138, resource/pull.943131/.955222. All17unchanged
targets and their variation retained. L/ep4/selection_review.json is authoritative.
Useful source-derived rule may advance to full sweep after cleanup CPU validation;
no per-case winner selection, default promotion or extrapolated full score.

Completed EP8 validation D/concurrency_validation_ep8_20260928:
GPU7511254 completed37m22s0:0, CPUanalysis7511258 completed7m09s0:0; monitor50540
exited0 at13:12:49Z.64nearby ordinary captures,48normal-routing runs,12AA/AB/BB
interleaved runs;1.088million rank-call samples. All correctness/timing/graph audits
PASS. Nearby64records/320compares/512replays; normal432/2160/384; calibration96each
compare/replay/interleaved. Full compact evidence: selection_review.json.
Q35normal2K saves6.87–7.01us versus rows and2.73–2.87us versus workers, allrank
mean/median/p95 bothorders. Balanced interleaved2K saves1.431/1.523us versus workers,
allrankmean/median/p95 bothorders; AA/BB/1Ksame-target controls remain separate.
Nearby3K only saves.167/.221us kernel and.160/.225us span versus rows, mixedrank/tail
signs; vsworkers saves1.855/2.079us kernel across everyrank. Still8.94–9.02us behind
pull. Eightcase resource/rows geomeans0.999815kernel/0.999547span; resource/pull
0.944153/0.952604. Losses retained. Evidence supports lower-EP testing, not acceptance.
Normal/interleaved graph times include preparation/reset and stay separate from
ordinary kernel/span. No noise subtraction, per-case selector or synthetic score.

Prior focused D/concurrency_ep8_20260928/run: GPU7510408/analysis7510412 completed;
48captures/384000rankcalls exact/timing PASS. Q35total2048 is the only changedtarget:
W64/P1 saves4.737/4.757us kernel vsrows and1.588/1.658us vsworkers, allrankmean/
median/p95 bothorders; still7.380/7.355us slower than pull. Seven/newer nearby
unchanged targets keep their measured variation. Source-derived hypothesis in
D/metadata_concurrency_hypothesis.json; do not revive rejected negative worker rules.

Completed earlier full sweep D/final_sweep:115unique cases/1150captures, all6GPU /
6analysis jobs0:0,36rank reviews complete;2250records/15750compares/22500replays.
Fixed91sub4K,22context4K/8K,8retained controls scored separately, every loss retained.
Rows trails pull:0.977835x kernel/0.976284x span; versus retained push1.010910x/
1.010094x. Historical rank0-median comparison across allocations/tuning is descriptive:
1.008374fixed/1.123726context,30/8losses. Goal remains open. Earlier full report:
[paired-sweep-report.html](scratch/megamoe/deep_cleanup_20260927/final_sweep/plots/paired-sweep-report.html).
Do not replace the full score with focused subsets or a model/token winner table.

Cleanup:18rejected options/705net CUDA lines removed; six helpers consolidated.
Nonzero retired flags fail; useful worker/row/SF/pipeline/middle-first/return paths
remain explicit. Maintained R/source/megamoe_cleanup_maintained_20260928 has37-file
overlay, six retired helpers absent; receipt D/maintained_source_receipt.json.
Common preparer/analyzer now handle normal routing; actual EP8 follow-up validates
capacity, exactness, timing and combine-flag preservation. Scheduling reviewer
accepts source-proven relocated terminal-vprintf diagnostics while cleanup mode
remains strict; saved-pair/negative tests and real CPU gates pass. No new helper.

Scope-only cleanup D/scope_cleanup_20260928 removes three redundant brace pairs
(6lines) around count stores, startup barriers and destination copies. Archived
before-header/patch preserved. Included in pass4_native with row-release removal;
all six CPU codegen gates above pass. Full runtime acceptance is the active B sweep.

Older diagnostics remain in topic memory:76small target/config identities,
24normalized old/cleaned SASS pairs,60capture timing-window review (cause unproven),
EP64calibration with Q3bothorder gain but Q35order reversal, and common reviewer
report equality. Earlier monitors are terminal;72236/74850/93787/92868 monitor corrected B; see current scope above.

## Historical MegaMoE handoff — September 25, 2026

The September25 cleanup checkpoint was complete. The user subsequently requested a handoff so the next agent
can continue the throughput goal. The original all-sub4096-win criterion was
superseded by the user's geometric-mean criterion above; historical full sweep
57/91 wins,34 losses remains diagnostic. The goal
tracker's completed state records the earlier stopping point, not performance
success. This turn updates documentation only; all submitted jobs are terminal.

Explicit metadata row partitions and positive worker counts are preserved; the
rejected expert-group prototype and automatic negative worker policies are removed. Defaults
remain unchanged, without model/token exceptions. Frozen sources/results remain
reproducible. Full pre-cleanup notes and headers are archived locally under
`scratch/megamoe/closeout_20260925/before_cleanup`, mirrored at
`R/artifacts/closeout_20260925/before_cleanup`.

Best recent robust path: Q3 row partitions save2.153/5.318us kernel at EP8 1K/2K
versus topology control; nearby/normal/interleaved gains persist. EP16 follow-up
7451910/7451911 completed:2.213/5.678us kernel gains across all ranks/both orders,
but still0.472/4.500us behind pull. Q35 evidence is mixed; no general promotion.
EP8 ordinary, normal/interleaved and EP16 exactness/replay checks passed.

Cleanup CPU7452492 completed5m06s: fresh native/JIT build, 27 specializations
with identical configurations/instructions versus validated EP8/EP16 artifacts.
No new GPU job requested.
Final dispatch comparison7452173/analysis7452177 completed and passed exactness/
replay. Dedicated dispatch gains5.793us kernel at Q35 2K but regresses Q3 1K/2K;
no general promotion. All submitted jobs are terminal. Closure follows the user’s
revised stopping point, not achievement of the original all-sub4K performance test.

Next agent: start with [prioritized TODOs](CODEX_PROJECT_megamoe_optimization.md#prioritized-todos-for-the-next-agent).
Use cleaned `R/source/megamoe_closeout_20260925` as the frozen reference. Focus
EP8 both models1K/2K plus8/32/256 controls; isolate a mechanism, pass CPU resource/
correctness gates, then nearby/nonuniform/interleaved validation before EP16.
The row+unified interaction has since been tested and rejected as a general
successor (active section above). Do not revive rejected policies or fit model/token exceptions.
After focused success, repeat the full matrix and update HTML/model-by-model
comparisons. No dormant experiment plan should be submitted blindly.

Latest full sweep HTML:
[sweep-report.html](scratch/megamoe/ordinary_latest_8k_20260924/plots/sweep-report.html).
See [optimization memory](CODEX_PROJECT_megamoe_optimization.md) for results,
measurement boundaries, failed attempts, cleanup scope and retained evidence.

## Topic index

| Work | Memory |
|---|---|
| MegaMoE kernel optimization and decomposition | [optimization](CODEX_PROJECT_megamoe_optimization.md) |
| Qwen3.5 model support | [support](CODEX_PROJECT_qwen35_support.md) |
| Qwen3.5 attention/GDN transport | [graph edge](CODEX_PROJECT_qwen35_graph_edge.md), [full attention](CODEX_PROJECT_qwen35_full_attention.md) |
| Qwen3 FFN/attention integration and earlier evidence | [handoff](CODEX_PROJECT_qwen3_ffn_handoff.md), [overheads](CODEX_PROJECT_qwen3_ffn_overheads.md) |
| Attention scheduling | [graph edge](CODEX_PROJECT_attn_graph_edge.md), [release performance](CODEX_PROJECT_attn_release_perf.md) |
| Reproduction and ratio/EP sweeps | [reproduce](CODEX_PROJECT_REPRODUCE.md), [ratio sweep](CODEX_PROJECT_RATIO_EP_SWEEP.md) |
| Cluster operations and AGA bring-up | [clusters](CODEX_CLUSTERS.md), [AGA](CODEX_PROJECT_aga_bringup.md) |
| Trace storage and interpretation | [Nsight contract](CODEX_PROJECT_nsys_traces.md) |
| Performance presentation | [Pareto](CODEX_PROJECT_pareto_perf.md), [publisher entry](CODEX_PROJECT_publish_entry_perf.md) |
| Prior runtime cleanup | [cleanup](CODEX_PROJECT_cleanup.md) |
| Reusable commands | [workflow inventory](codex_scripts/README.md) |

Model integration notes retain historical kernel evidence; the consolidated
optimization file supersedes their older MegaMoE selection/pending checkpoints.
Full-model alignment and fixed-input kernel correctness remain separate gates.

## Execution and continuity

- Local checkout is authoritative. Use plain `ssh oci-hsg` for HSG execution.
  Root R is `/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD`; frozen sources
  belong under `source/`, plans/logs/results under `artifacts/<campaign>/`.
- Current model-tested source: `R/source/local-mirror`. Historical frozen sources
  include `dispatch_overlap_u32_final` and `all_push_peer_ready_rne_20260920`.
  Preserve archived controls; do not edit source/cache while a job consumes it.
- User-requested monitoring cadence: check running jobs every5minutes (300s).
- Prepare/analyze on CPU allocations; GPUs perform complete workloads. A complete
  kernel or producer-consumer benchmark qualifies, with model E2E for promising
  candidates. No tiny GPU debugging jobs, manual waiting, padding or auto-retries.
- Preserve frozen controls and unrelated staged/dirty changes. No subagents unless
  requested, no automatic fallback, py_check, manual hash/version checks or /tmp.
- Keep project memories concise and record meaningful decisions/results in the
  relevant topic. Historical plans use frozen workflow copies; recreate current
  plans with maintained tools rather than rewriting completed evidence.

September28 16:43UTC monitor update: EP8 GPU7514500 COMPLETED58m14s0:0;
analysis7514520 RUNNING1m49s, audit pending. EP16 GPU7514502 RUNNING24m53s;
EP32 GPU7514504 now RUNNING; EP64 GPU7514506 pendingQOSMaxJobsPerUserLimit.
Existing300s monitors continue; CPU analysis64 waits for a submit slot.
No additional GPU sweep or performance rejection based on pull/4K/8K.
