# MegaMoE optimization

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



## Active cleanup — corrected full acceptance running, performance goal open

September28, 2026. Active unbounded goal: deeply clean all-push kernels/workflows,
retain useful optimizations, and gate ordinary Qwen3 EP2–64 / Qwen3.5 EP4–64,
total batches EP..8192 against retained performance. User requires HSG short,
one GPU job per EP, both models combined; CPU preparation/analysis separately.
Monitor active jobs every300s. No subagents, commit, automatic retry, /tmp,
py_check or hash/version audit. No default promotion has been made.

Earlier F sweep: all6GPU and all6analysis jobs completed0:0. Final analysis7503592 completed
1h34m58s; observed10:05:07Z. Monitor session43860 exited0 after its last job.
**Earlier full and focused sweeps are terminal; pass4 full acceptance below is active.** For future jobs,
use maintained monitor_afd_jobs.py at300s; live phase snapshots own cadence,
while campaign.json summaries are checkpoints.

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

Paths: R=/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD;
D=R/artifacts/deep_cleanup_20260927; F=D/final_sweep;
C=R/artifacts/throughput_20260922. Local D=scratch/megamoe/deep_cleanup_20260927.
Authoritative completed evidence: F/{campaign.json,measurement_matrix_review.json,
capture_matrix_review.json,full_comparison.json,small_policy_timing_review.json}.
Full report: [paired sweep](scratch/megamoe/deep_cleanup_20260927/final_sweep/plots/paired-sweep-report.html).
Prior active notes are archived in D/active_notes_before_complete_measurement_review.md;
historical sections below remain unchanged.

### Source and cleanup

Frozen measured source R/source/megamoe_cleanup_pass3_20260928 has44all-push
suffix arguments; matched reference R/source/sender_middle_20260926 has59.
The earlier final-named snapshot is an intermediate48suffix, not the measured
final candidate. Preserve frozen sources, caches, profiles and output references.
Separate R/source/megamoe_cleanup_maintained_20260928 is synced from pass3 plus
the37-file D/maintained_source_files.txt overlay. D/maintained_source_receipt.json
records37files present, two retired helpers removed/four already absent, and safe
non-symlink overlay paths. No policy/default change; frozen inputs preserved.

18rejected options/705net kernel lines removed versus before_cleanup.tar.gz:
READY_SCHEDULE, ROTATE_PEERS, SFA_MULTICAST, METADATA_REGISTER_SCAN,
METADATA_SINGLE_JOIN, SKIP_IDLE_DISPATCH, METADATA_WARP_GROUP_LIMIT,
SKIP_EMPTY_PARTITIONS, SF_EARLY_RETURN, SINGLE_GEMM_EPILOGUE, FC1_WG_READY,
FC1_ASYNC_READY, METADATA_COOPERATIVE_GROUPS, METADATA_AFTER_REGISTERS,
EARLY_SF, DIRECT_SHARED, WINDOW_WIDE, LOCAL_SOURCE. Retired nonzero flags fail.
Useful worker/row/SF/pipeline/middle-first/return paths remain explicit.
Archives, before-headers, removed_options.json, pass3_review.json and
pass3_kernel.patch retain evidence. Pass3 removed200lines, including unused
encoding/direct-gather helpers and SF-only barriers. Early-SF/direct-shared lost
all10matched cases; window-wide regressed small+large; local-source had Q35
regressions and Q3 order reversals. Do not revive these without new evidence.

Six standalone helpers were consolidated, originals archived:
router preparation→megamoe_header_smoke.py prepare-routers;
AA/AB/BB plans→prepare_megamoe_throughput.py calibrate;
rank reviews→analyze_megamoe_throughput.py review-ranks;
MMA probes→prepare_header_overlay.py --mma-wait-probe;
row resources→review_megamoe_dispatch_sass.py review-rows.
interleaved reviews→analyze_megamoe_throughput.py review-interleaved.
Remove their old paths only from the new maintained copy, never frozen controls.

Post-freeze workflow improvements are staged D/workflow_updates: exact-scope
aggregation and correctness-only audit; paired/historical report support; MMA
anchor repair; narrow terminal-vprintf resource checks; submit.sh regeneration
from authoritative JSON;300s monitor with batched accounting and explicit log paths;
optional strict validation-helper symbol normalization. These do not change the
measured ordinary harness/kernel. Focused semantic tests passed, including case
weighting/omitted losers, correctness event order, source-metadata rejection,
changed instructions/call targets, unknown symbols and submission regeneration.

September28 scope-only cleanup D/scope_cleanup_20260928: three redundant blocks
left after rejected-option removal were flattened around the shared expert-count
store, startup barrier/fences and destination assignment. Six brace lines removed;
no variable declarations within those scopes, no operation/order/flag/ABI change.
Archived before.cuh from pass3 and exact scope_cleanup.patch; reviewed delta only
braces/indentation. Local and maintained source updated, plus workflow_updates.
GPU7512239/7512241 still use immutable pass3. No extra GPU job or tests for this
syntax-only edit. Include it in the next candidate freeze and CPU codegen gate;
compiled/runtime equivalence not newly claimed. Other explicit useful paths retained
after reviewing historical evidence; no speculative feature removal.

September28 additional cleanup D/row_release_cleanup_20260928: removed ROW_RELEASE
configuration/template plumbing, row_pending helper, final-word release stores and
receiver acquire branches; ordinary per-word polling/store path remains. Net61C++
lines removed, suffix44→43. Reject nonzero stale setting in the existing retired
option guard. Prefetch remains because separate results were mixed. Historical
7379931/7379932 passed correctness but release/release+prefetch regressed Q3
+252.403134/+253.586871us and Q35+349.143190/+349.088395us kernel, every rank/both
orders. Before-files, exact diff, evidence and replacement files are archived.
Frozen R/source/megamoe_cleanup_pass4_20260928 includes this and the six-brace
cleanup, based on immutable pass3. CPU7513273 submitted14:09:15Z: fresh native/JIT/
TVM; EP4 control/resource bothmodels, four full-grid preparations, then strict
configuration/SASS comparison against pass3 with only removed inactive field and
source/helper-symbol metadata normalization. No resource-only bypass or GPU yet.
Fresh native mandatory after changed configuration ABI. Pending CPU and full
ordinary performance/correctness gates; no compiled/runtime equivalence claimed.

CPU7513273 failed5m03s, observed14:14:49Z: strict removed-field checker found
all_push_row_release still present. Root cause is source setup: symlinked
deepgemm.py uses Path(__file__).resolve() and therefore built pass3 native/device
code while metadata named pass4. Fresh caches alone are insufficient. Preserve
that failed source/cache/report; it provides no pass4 codegen validation.
Corrected immutable R/source/megamoe_cleanup_pass4_binding_20260928 owns the
binding module plus all eight prior replacement leaves. Native/JIT/TVM caches
are new cache_binding_fix; preparation_binding_fix and code_review_binding_fix
are separate. CPU7513386 running14:17:02Z, monitor10504 every300s. Maintained
precompiler now fails before build if binding _source_root differs from requested
source; explicit --header-root keeps its existing configure workflow. No kernel
change after row-release removal, no checker relaxation or reused failed native.
Monitor6616 terminal. EP4 GPU7512241 remains pass3,188/192 at14:16:30Z;
analysis7512253 still pendingafterok, monitor19645 remains live at300s.

CPU7513386 failed2m54s, observed14:22:03Z. Binding ownership fixed the Python
root, but quoted includes through symlinked native C++ still generated44host
suffix arguments against the43argument device header; CUDA assertions rejected
shifted window/geometry values. Do not reuse this native cache. General overlay
tool now has explicit --own-native: copy the52file host tree and binding, preserving
explicit replacements; reject source cycles. Focused symlink-tree/replacement/base/
existing-output/cycle checks PASS (initial fixture had broken relative links,
corrected before the successful check). Precompiler rejects native source symlinks
as well as binding-root mismatch unless explicit header-root is configured.
New immutable R/source/megamoe_cleanup_pass4_native_20260928 owns60files including
native sources and replacement leaves. CPU7513604 submitted14:25:18Z, new
cache_native_fix/preparation_native_fix/code_review_native_fix; monitor51984/300s.
Both previous failures/sources/caches preserved; no checker relaxation or GPU.
EP4 GPU7512241 completed1h02m47s0:0; analysis7512253 running2m35s at14:21:31Z,
monitor19645 active. Complete analysis pending.

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

Cleanup CPU7513604 completed5m39s0:0, observed14:35:29Z; monitor51984 exited0.
Strict preservation_review.json PASS_RETAINED_INSTRUCTIONS_EXCEPT_SOURCE_AND_LOCAL_SYMBOLS:
26specializations/48case memberships across EP4bothmodels control/resource; all26
instruction streams and retained configurations match pass3 after removed inactive
all_push_row_release/source metadata/helper-name normalization. No resource-only
bypass. Candidate has43suffix args,61net C++line removal plus earlier6brace lines.
Both failed source-ownership attempts preserved. Common --own-native and
precompiler guards prevent those errors; maintained C++ tree has no symlinks.
Advance this source to full-matrix CPU/GPU acceptance above; no runtime or full
performance equivalence follows from this CPU subset.

### Completed jobs and gates

| EP | Final CPU | GPU short | CPU analysis |
|---|---|---|---|
|2|7501321;18specializations PASS|7501627;35m14s|7502108;3m34s|
|4|7501472;39spec/72memberships PASS|7501715;1h13m58s|7503019;9m31s|
|8|7501238;33spec/66memberships PASS|7501471;1h10m26s|7501728;17m13s|
|16|7501323;30specializations PASS|7501540;1h18m42s|7501731;32m54s|
|32|7501541;27spec/54memberships PASS|7501716;1h17m01s|7503519;52m35s|
|64|7501621 built; review recovery7501947 PASS54spec/96memberships|7502092;1h37m13s|7503592;1h34m58s|

All GPU/analysis jobs above completed0:0. EP64 first CPU resource review rejected
unchanged vprintf timeout diagnostics. Recovery7501947 reused complete caches and
passed after the checker required sole-vprintf syscall metadata and exact terminal
caller blocks through BPT.TRAP, with source-proven assertion line normalization.
No kernel edit, rebuild, GPU retry or gate bypass was used. EP4 analysis had two
quota rejections with no job ID at04:54/04:58, then succeeded after an analysis
completed. EP2's first GPU submission rejected an expired completed dependency;
verified CPU0:0/resourcePASS, removed that expired dependency, submitted once.
Earlier pass1/pass2 compile/check failures and recoveries remain in the archived notes.

115unique cases=113fixed/context+2EP16 controls;1150ordinary captures. Five arms
reference/control/workers/rows/pull, both orders,1000calls/rank/pass. Same pass3
harness, matched collective reset/barrier, required source-scoped caches, weight
export0, GPU metrics0. Cases perEP13/24/22/22/18/16. Workers use2*min(localExperts,
2*EP) for full-TMA shapes; rows additionally uses2metadata row partitions. Small
shapes retain identical cleaned targets/configurations across all three policies.
All36rank reviews cover their exact case/rank/order sets and match summary scores.
All planned exact-output checks passed:2250records/15750source comparisons/
22500replays. Push compares retained-source outputs; pull has exact self-replay.
Full timestamp/graph/aggregation audits and matching GPU IDs passed for all115cases.

EP32/64 CPU exports were initially quiet. Container-step CPU/I/O counters confirmed
work; batch-wrapper CPU0 was not the workload. EP64 first timing output appeared
at48m51s. Activity/allocation observations are saved in each analysis directory;
no quiet-stage restart occurred. EP64 requested cpu8/mem184G/4h; completed normally.

### Full performance result — target not met

Fixed91sub4K cases use equal-case geometric baseline/candidate ratios, all ranks
and both orders, kernel and producer-start→main-end separately.22context4K/8K
and8original small controls are separate. Extra controls never reweight the91score.
No losing case/sample is omitted; no per-model/token winner selector is permitted.
The original all-case-win rule was superseded by this geometric criterion.

| Policy | Kernel / pull | Span / pull | Kernel / retained push | Span / retained push |
|---|---|---|---|---|
|control|0.962082|0.962625|0.994625|0.995962|
|workers|0.977202|0.975739|1.010256|1.009531|
|rows|0.977835|0.976284|1.010910|1.010094|

Rows has the highest observed fixed scores, but remains below1x versus pull;
37kernel/38span cases are slower. Its advantage over workers is only about0.065%
kernel/0.056%span. No performance success/default promotion follows from correctness
or the small observed policy lead. Rows' perEP kernel/pull scores are
.797900/.923225/1.016375/1.031118/1.067555/1.029042 forEP2/4/8/16/32/64.
For22context cases rows/pull=.840162/.862942 and rows/reference=1.173081/1.145492;
all22context cases beat matched retained push in both metrics, but21trail pull.
All regressions and by-model/EP scores are in full_comparison.json.

Historical comparison is descriptive: September24 single rank0 kernel median
versus average current forward/reverse rank0 medians, separate allocations and
potentially different tuning. No historical pipeline equivalence is claimed.
Rows/history scores1.008374 over91fixed and1.123726 over22context cases, with
30fixed/8context slower cases retained. Historical EP8/16 large cases used specific
return/combine/tile choices (ordinary_latest_8k_20260924/prepare.py); do not attribute
all such gaps to cleanup or restore a per-case winner table. Matched reference is
cleanup's primary performance comparison. The full report includes both mean
metrics, a separate historical row, all115cases,1150absolute arm/metric CSV rows,
339historical arm/case CSV rows, and provenance. Both model figures were inspected;
HTML1136616bytes; report scores match strict aggregation.

### Small-path preservation evidence and remaining timing concern

small_policy_path_review.json covers76geometry-small cases (full-TMA0), including
all8retained controls. Counts byEP=9/17/15/15/11/9; scope includes Q3total512 and
Q35total512/1024, not just totals<=256. All three cleaned policies share the exact
compiled target/configuration in each small case. Across115cases, only worker
count and metadata row-partition fields vary among policies.

Original CPU resource reports mark all76source59→44 comparisons non-identical.
Saved-SASS inspection found validation-helper signature renames from removing
unused parameters. Optional --normalize-validation-helper-symbols in the existing
checker requires source normalization and a bijection between same-ordinal,
entry-scoped validate_window_rows symbols. Only exact CALL.REL.NOINC symbol operands
are rewritten; unknown renames/changed coverage fail. Instruction addresses,
operands, function order and mapped call targets must all match. Three focused
tests passed, including changed bodies/targets, non-call/unknown symbols and swapped
symbol names. The maintained checker reproduced all24unique pairs covering76cases.
small_helper_symbol_review.json records every map and source. Frozen reviews remain
unchanged. This proves normalized SASS preservation, not identical observed tails.

The complete76case timing review retains every sample and excursion. Geometric
kernel/span ratios vs retained push: control .997776/.998369, workers .996192/
.997265, rows .995234/.996541. Worst individual rank/order rows p95 increases:
+19.648us kernel/+19.680us span atQ35 EP64 total512 reverse. Control reaches
+20.096/+20.960us atQ35 EP32 total128; workers +12.768/+12.928us atQ3 EP32 total128.
Do not substitute the SASS proof for measured data or discard these observations.
EP64 rows/reference fixed score .969947/.978055 includes large order excursions
at Q35total128/512 despite small-path SASS preservation, plus Q35total2048 with a
changed large-path policy. Detailed order means are in ep64/performance_review.json.

Further diagnostic partition (performance_diagnosis.json), with unchanged91case
weights:74small cases rows/pull=1.000449kernel/.994434span;17full-TMA cases
=.885202/.901059. The large group is the main kernel-score deficit. EP64Q35total
2048 illustrates order variation: control35.082/52.784us, rows34.061/52.221us
forward/reverse; workers34.964/34.867us, reference34.896/34.576us. Q35total512
rows21.366/35.000us despite small-path SASS preservation. Do not discard these
measurements or attribute them automatically to the policy; review calibration
and earlier rejected work before a targeted follow-up. No new job submitted.

### Next work

September28 completed calibration D/calibration_ep64_20260928/run: frozen pass3
source/cache reused, no kernel change or native/JIT compilation. New reusable
prepare_megamoe_throughput.py calibrate-sweep consumes completed ordinary
plan/summary and prepared manifests. Requires exact source/cache/shape/buffer/
routing contract, selected+combine cubins and all rank references; only positive
worker/metadata-row scheduling differences are supported. run-calibration uses
one shared container step, fail-fast rank processes and workspace temp paths.
Two focused tests passed, including source/cache/layout rejection and preserving
AB2x score despite synthetic AA/BB variation; missing pair/order fails.

EP64 bothmodels totals512/2048, workers vs rows, AA/AB/BB×bothorders×fourcases,
1000calls/arm/rank, synchronized entry,24processes/3.072million samples. Both512
cases have identical compiled targets;2048 differs only in row partitioning.
Joined-graph CUDA-event measurements include preparation; they cannot replace
ordinary kernel/span scores. CPU7509806 4s0:0 validated16prepared memberships/
256distinct rank references. GPU7509835 HSGshort oneEP64job completed11m56s0:0,
analysis7509843 completed16s0:0. PASS1536markers each compare/replay/interleaved.
Monitor26028/300s exited0 after11:06Z terminal observation; no active job/monitor.

Q3total2048 AB rows-workers means -6.243061/-4.822829us forward/reverse;
everyrankmean/median/p95 improves bothorders. Reverse windows include+3.4206us,
retained. Q35total2048 -1.710679/+1.149417us: forward allrankstats improve,
reverse allrankmeans regress+.6807..1.5740us and p95 max+1.9840us. Pooled ratio
1.003657x does not establish a robust Q35 effect. Shared-code512 AB deltas:
Q3-.148806/+.139829us, Q35-.001446/+.221938us; Q35reverse allrankmeans regress.
Identical-policy AA/BB controls also vary: Q3512 AA reverse -1.583219us pooled,
-16.4110us100-call window. No universal noise bound or subtraction. AB four-case
geomean1.019894x; AA1.011957/BB.998870 separately. selection_review.json keeps all
observations: retain usefulQ3 row path, no general/default promotion or corrected
full-sweep score. No unchanged calibration rerun planned.

Sixth helper consolidation: interleaved reviewer moved into common analyzer with
explicit arm/pair/entry support and optional separate output. CPU7510033 completed
13s0:0, entire report JSON exactly matches frozen reviewer across24processes/
3.072million samples. comparison.json is the proof; before_interleaved_merge
archives both pre-merge local helper and former maintained remote version.
Standalone helper removed only from local/maintained/staged paths, frozen
experiment/source copies preserved. Calibration planner now freezes the common
analyzer and writes analysis/commands.json; no new standalone workflow script.

Independent source-derived hypothesis (not implemented/measured): metadata
dispatch uses worker=physical_worker/P and serial expert stride when W/P<E.
Current rows W=2min(E,2EP),P2 gives only4of64 concurrent experts forQ3EP2,
8of32 forQ3EP4,8of128 forQ35EP4,16of64 forQ35EP8. A prospective general
resource rule chooses maximum W<=min(2E,floor((SMs-pushers-1)/2)) with W dividingE
or vice versa, then P2 only ifW>=2E, otherwiseP1. For current150SM/16pusher
geometry this yieldsW64 in all four groups; P1/P2/P1/P1 respectively.
Would change14full-TMA cases (6fixed+8context); existing small geometry and all
other topology settings remain. More metadata workers can delay compute by
occupying epilogue warps, so legality/concurrency alone does not prove a gain.
Earlier rejected -3 policy used /4 budget and affected small shapes; this is a
different full-TMA-only proposal using retained positive overrides. Evidence and
prospective EP8 rows/worker/proposed/pull experiment are in
D/metadata_concurrency_hypothesis.json.

September28 11:19Z: prepared D/concurrency_ep8_20260928/run from frozen44suffix
source with a fresh isolated cache. CPU7510172 compiled all kernels/routers/barrier, then failed at5m37s on reviewer
requiring identical diagnostic addresses across changed scheduling. Monitor76537
exited0 at11:29Z. Saved SASS showed the known terminal vprintf/assert block moved
with otherwise identical instructions. Scheduling review now independently checks
known relocated terminal calls with equal diagnostic opcode coverage; cleanup
comparison stays strict. Saved-pair resources pass, removing the diagnostic trap
fails. Original failed workflow/report retained. CPU-only review7510346 completed1m06s0:0, reusing all prepared caches/no rebuild.
All18scheduling checks plus32group plan PASS. Resource targets unchanged for
all Q3cases and Q35total32/1024; onlyQ35total2048 differsW64/P1. CPU monitors
76537/2410 exited0; original failure and fixed review remain separately saved.
GPU7510408 HSGshort singleEP8job completed17m55s0:0 on nvl72127-T[11-12];
CPU analysis7510412 completed4m25s0:0. Monitor38799/300s exited0 at12:02:32Z;
all jobs terminal. All48ordinary captures/384000rank-call samples audited PASS:
48records/240source compares/384replays; matched collective preparation and
complete graph/timestamp/aggregation checks. Three resource-vs-rows/workers/pull
rank reviews include every sample, order, rank and six cases.

Q35total2048 resource W64/P1 versus current rows W32/P2: kernel -4.737463/
-4.757373us forward/reverse, producer-to-main span -4.756631/-4.772220us.
Versus workers W32/P1: kernel -1.587613/-1.657600us, span -1.600581/-1.663311us.
Every rank mean/median/p95 improves for bothmetrics and bothorders against both
push controls. Still slower than pull: kernel +7.379562/+7.355451us, span
+7.765745/+7.742900us; every rank remains slower. This is promising focused
concurrency evidence, not performance-goal completion or default promotion.

The five resource/control identical-target cases remain in the report. Most mean
changes are within.064us; Q3total2048 forward kernel -.364081/span -.812396us is
same-code variation, reverse -.023412/-.024663us. Largest positive rank p95 among
unchanged cases+.224us Q35total1024forward. No noise subtraction/sample exclusion.
Sixcase resource/rows equal-case geomeans1.006906kernel/1.005928span;
resource/workers1.019752/1.016093 includes already-retained Q3 row-partition gains;
resource/pull.961226/.967533. These scores do not replace full91/context/controls.

run/selection_review.json preserves complete compact outcome and next gates:
EP8 bothmodels ordinary32/768/1536/3072, normal1K/2K and small control, plus
Q35 workers/resource AA/AB/BB interleaving at1K/2K to isolate new worker gain.
Q3 resource equals rows: do not fabricate a flag difference to force calibration.
CPU geometry/resources/capacity first, then HSGshort oneEPjob with models combined.
If robust, validate affected lowerEP geometries and full91+22context+8controls.
Preserve this completed source/cache/results. Sixcases bothmodels32/1024/2048,
current rows (`control`), worker-only, resource rule, pull;48ordinary captures/
1000calls perrank/bothorders. Only Q35full-TMA changes W32/P2 toW64/P1; Q3
resource matches rows. Existing checker gains review-scheduling for explicit
worker/row arms, validates resource rule, unchanged non-scheduling geometry,
small targets, register budgets and spills. CPU7510346 passed its real-data gate.
No kernel/default change, score replacement or acceptance narrowing.

September28 completed validation D/concurrency_validation_ep8_20260928:
nearby CPU7510946 passed24checks; normal CPU7511077 passed18checks including
worst-case routing capacity. GPU7511254 completed37m22s0:0 on nvl72113-T[02-03];
CPUanalysis7511258 completed7m09s0:0. Monitor50540 exited0 at13:12:49Z.64nearby
ordinary captures/8cases,48normal runs/6cases,12interleaved AA/AB/BB runs/2cases;
1.088million rank-call samples. All correctness/timestamp/graph/aggregation audits
PASS. Exactcounts: nearby64records/320compares/512replays; normal432/2160/384;
interleaved96each sourcecompare/replay/interleaved. Full reports collected locally;
selection_review.json preserves all cases/orders/rank deltas and the decision.

Only Q35nearby3072 resource target differs from rows. Kernel saves.167371/.220752us
forward/reverse, span.160427/.225004us, but individual ranks/tails mixed (largest
positive rank kernel mean+.266016us, spanp95+.864us). Versus workers, kernel saves
1.855284/2.079277us andspan1.851317/2.081461us, everyrankmean/median/p95 bothorders.
Still8.939–9.020us kernel/9.752–9.825us span behind pull. Nearby eightcase geomeans:
resource/rows0.999815055kernel/0.999546626span; resource/workers1.020507144/1.016488153;
resource/pull0.944152999/0.952604341. All same-target losses remain included.

Normal Q35total2048 graph saves7.011110/6.869610us vsrows and2.866188/2.729076us
vsworkers; everyrankmean/median/p95 improves bothorders. Still19.965/20.024us behind
pull. Sixcase graph geomeans1.006529vsrows,1.009020vsworkers,.898837vspull.
Balanced interleaved workers/resource2K saves1.431083/1.522611us, everyrankmean/
median/p95 in bothABBA/BAAB. At1Ksame-target ABmean-.067948/+.127620us; AA/BB
variation separately retained, never subtracted. ABtwo-case geomean1.004023212;
2K1.008270582,1K.999793735. Graph timing includes preparation/reset and never
substitutes for ordinary kernel/span. No universal nearby win or pull win claimed.

Decision: source-derived concurrency mechanism survives normal/interleaved gates;
proceed to affected lowerEP topologies, retain near-flat nearby and all measured
losses. No default promotion, source change, synthetic full score or acceptance
narrowing. Full91fixed/22context/8control gate remains required.

Common prepare_megamoe_throughput.py normal/run-graphs and common analyzer
review-normal replace reliance on historical ad-hoc normal text rewriting. Preserve
measured settings; conservative capacity all global routes+192padding/localexpert,
actual compiled tile/capacity checked. Exact per-rank input/alternate/normal/padding
sequence and all raw ranks audited. Focused test plus actual384000-sample review
PASS. Calibration preserves measured COMPARE_COMBINE: EP8selected-only flag0;
EP64selected+reference flag1, unchanged. No new build or reduced EP64 gate.
Nearby analyzer resolves shared GPUjob via nearby/gpu/job.json and capture log
symlinks; graph child logs remain separate. Combined wrapper quoting corrected
before submission. Frozen inputs/results preserved.

Active lowerEP campaign D/concurrency_lower_ep_20260928:
EP2 CPU7511453 completed4m28s0:0,39checks/13cases PASS; EP4 CPU7511460 completed
6m10s0:0,72checks/24cases PASS. All78small-case arm targets match controls; full-TMA
EP2Q3W64/P1, EP4Q3W64/P2 andQ35W64/P1 verified. CPUmonitor12555 exited0.
GPU7512239(EP2) completed28m38s0:0; CPUanalysis7512247 completed2m45s0:0.
GPU7512241(EP4) running107/192captures at13:51:24Z; CPUanalysis7512253 pendingafterok. Monitor19645 every300s; phase snapshots own currentstatus. One HSGshort
GPUjob perEP,90min limits. Fullpowers-of-two Q3EP2 totals2..8192 (13cases/104captures);
bothmodelsEP4 totals4..8192 (24cases/192captures); rows/worker/resource/pull,
bothorders/1000calls perrank, retains small8-token prior regression case. New
artifact-local workflow and fresh JIT/TVM; native copied from samepass3 ABI. No new
maintained helper. Review results and remaining fullscope before promotion.


EP2 complete result: GPU7512239 28m38s/analysis7512247 2m45s, both0:0;
104captures/13cases/208000rankcalls,26records/130compares/208replays and full timing/
graph/rank audits PASS. Selection: L/ep2/selection_review.json, where
L=D/concurrency_lower_ep_20260928. Resource improves all4full-TMA cases versus
rows and worker-only, everyrankmean/median/p95 bothorders and both metrics.
Kernel savings versus rows:1K57.823–58.363us,2K109.210–110.176us,
4K192.376–192.670us,8K362.978–366.891us. Small9targets unchanged; largest positive
same-target mean+.084508us kernel/+.089196us span, all retained.
Fixed11sub4K resource/rows geomeans1.065896589kernel/1.058273487span;
resource/pull0.855113336/0.865398037. Context2: resource/rows1.445459284/1.383682699,
resource/pull0.906306978/0.930595606.2K beats pull everyrank kernel mean/median/p95
bothorders; reverse spanrankp95 still+.064us.1K/4K/8K remain behind pull.
Retain useful concurrency path, await EP4 before final policy/default decision;
never extrapolate this EP to full91score. Unchanged EP2small gaps remain large:
512tokens resource187.073us kernel versuspull129.876us. Source at heuristics
mega_moe.hpp:390–395 fixes small workers to min(localExperts,2EP)=4 for64local
experts and ignores positive overrides at block_m<=32. Small concurrency has useful
historical evidence and explicit counterexamples; no new small-worker option or
experiment has been implemented.

September28 archived small-worker review: source archive
scratch/megamoe/closeout_20260925/before_cleanup/CODEX_PROJECT_megamoe_optimization.md,
lines479,1294,1420. Worker-3 EP2 GPU7437483/analysis7437484 passed30captures;
W4→32 improved allfive8/32/256/1K/2K cases versus control across every rank/order/
mean/median/p95, both kernel/span. At32/256 it also beat pull by those measures.
Normal7438075/7438077 confirmed broad gains, but tiny2 queued ABBA regressed
pooled+.099536us, rank1mean+.224608/p95+1.056001us; BAAB improved. EP4
7437609/7437610 Q35total8 W8→16 regressed+.524019us kernel/+.504831us span
on every rank/order/mean/median/p95 without spills. Worker-4 later capped routes
per metadata group, but EP8 normal/queued7448032/7448034 Q35total448 regressed
.349754us normal and+.341080/+.320876us ABBA/BAAB, every rank/statistic/window.
LowerEP worker-4 passed CPU gates but was never submitted after that rejection.
Do not revive either unchanged, fit token exceptions, or call small concurrency
wholly untested. Any new mechanism must explain these failures and retain tiny8,
tiny2, and normal448 holdouts. Future steady interleaving should explicitly use
queued entry; existing synchronized EP8 evidence remains labeled and uncorrected.

EP4 complete, observed14:31:34Z: GPU7512241 1h02m47s/analysis7512253 10m08s,
both0:0; monitor19645 exited0.192captures/24cases/768000rankcalls,96records/
480compares/768replays; all correctness/timing/graph/rank audits PASS.
L/ep4/selection_review.json retains every case/rank/order. All7changed full-TMA
cases improve everyrankmean/median/p95, bothorders and both metrics, versus rows
and worker-only. Q3kernel gains vsrows:1K3.940–3.965us,2K3.517–3.838us,
4K30.575–30.580us,8K55.453–56.682us. Q35:2K50.959–51.050us,
4K59.243–61.654us,8K84.028–84.086us. Q3total2K also beats pull everyrank/stat/
order kernel+span; kernel1.213/1.970us. Other large cases remain behind pull:
Q31K4.962/4.984us,4K1.284/1.999us,8K48.003/47.985us;
Q352K10.646/10.564us,4K16.752/18.596us,8K17.073/16.327us.
Fixed20case resource/rows geomeans1.014786986kernel/1.012275183span; resource/pull
.935574880/.942147240. Context4 resource/rows1.171265031/1.134138387; resource/pull
.943131466/.955221759. All17small targets unchanged; pooled ordermean variation
vsrows spans-.177986..+.204741us kernel and-.374386..+.303270us span, retained.
Decision: resource rule retains useful gains and can enter full-matrix acceptance
after pass4 CPU gate. This is ordinary pass3 evidence, not a pass4 proof or a
full91score. No default promotion, measured winner table or dropped losses.

September28 10:28Z diagnostic follow-up: broader sender-middle and SF3 evidence
already rejects robust general promotion; do not revive unchanged. EP32/64 slow
captures have median shifts and positive mean deltas on every rank (e.g. Q35
EP64rows512reverse median35.872vsreference21.312us), not only a few outliers.
Added review-windows to the existing analyzer: contiguous call windows, all
recorded arms/orders/ranks, every sample and partial tail retained; producer,
selector, kernel and following gap statistics, exact saved-mean validation.
One focused test passed, covering partial tails, absent ranks/samples/gaps,
nonfinite values and inconsistent saved means. CPU-only7509483 completed1m36s0:0,
observed10:33:48Z cpu-0004; monitor99562/300s exited0. No active job or monitor.
Independent D/timing_windows_20260928 retains60captures/3.2million rank-call samples:
Q35EP32totals128/2048, EP64totals64/128/512/2048, allfivearms/bothorders. Eight
selected slow captures have elevated kernel means throughout all ten100-call
windows; following gaps also increase, producer means nearly unchanged.
rank_stage_means.json preserves7raw-trace perrank means/launch metadata: Q35EP64
512rows-reverse longest selectors ranks28/30/31 on nvl72096-T09 are15.434/13.849/
13.270us; baseline max8.511us. Allrank kernel window means32.58..35.83us versus
reference overall21.347us. EP64total2048control-reverse also concentrates long
selectors onT09; rows-reverse onT18; total128rows-forward onT04/T09. This localizes
observations, not a proven node/hardware/policy cause or permission to blacklist.
gpu_health_before_capture.json covers same64UUIDs for512rows forward/reverse,
allP0/memory3996MHz, only idle reasons active. Snapshots precede measured work and
cannot rule out dynamic throttling. device_metadata.json reads saved rank0SQLite:
four visibleGB200devices have152SMs versus150CTA grid; resident cluster capacity
is unmeasured, so no scheduling explanation is established.
Prior C/baseline_worker_expansion_calibration_v2/selection_review.json demonstrates
identical-policy EP8small variation (Q3532 AA reverse+.107332us; allrankmean
+.033856..+.161152us), not a universal noise bound for these much larger excursions.
selection_review.json records all findings and unchanged acceptance. No GPU rerun,
kernel/default change or score replacement. Next design a source/cache-aware
same-process comparison separating worker/row effects from node-concentrated
variation, with matched diagnostic telemetry if needed; no blind full sweep retry.

1. Use the maintained source for further work; freeze a new experiment copy before
   measurements, preserving all completed source/cache/result snapshots.
2. Review the full report and rank/order evidence to separate configuration changes
   from timing variation; especially EP64 Q35total128/512/2048 and EP32Q35total2048.
   Small-path SASS equality does not justify dropping slower observations.
3. Address the fixed-score shortfall against pull and historical large-case losses
   with a defensible general policy or kernel change. Use existing calibration/
   interleaving workflows if needed to isolate order variation; no blind sweep retry.
4. Preserve useful paths and all comparisons. Any future default must be supported
   by matched evidence and CPU proof that its configurations match the measured arm
   across the declared scope. No model/token winner table. Keep the goal active
   until the actual performance requirement is achieved; correctness alone is not enough.

## Resumed focused EP8 iteration — September26

Latest user direction supersedes historical all-case-win acceptance below: choose
a small representative set for each seq first; use geometric mean speedup versus
pull for final comparison. Equal-weight per-case latency ratios over the fixed
full comparison set: exp(mean(log(pull/candidate))), >1 faster. Report kernel and
pipeline separately and show per-case regressions; do not select only winning
cases. Small-batch performance must remain unchanged against retained control.
Promote to EP16 only after promising EP8 gains. Prior rejection observations remain
valid evidence, but non-small individual losses alone no longer reject a candidate.

User resumed and requested a small target set for fast iteration, with EP16 promotion
only after promising EP8 gains. Active set: Q3/Q35 EP8 total32 small controls plus
1024/2048 targets. Compare row-only, candidate and pull in both orders;1000calls/rank.
Broader8/256 controls, nearby/nonuniform/interleaved EP8 and original EP16 small
controls remain promotion gates, not every initial experiment. Final scoring follows
the updated geometric-mean criterion above.

Previous goal turn classified progress: candidate56 measured/rejected; scan57 CPU
proof completed. Fresh scheduler audit confirms7473067/7473072/7473615 terminal0:0,
no live prior consumers. Re-read scan57 patch, ownership, resource/default/row/SASS
gates. No recompilation or source/cache changes required. New campaign
C/metadata_register_scan_focus derives only the six cases/three arms from the
unsubmitted full plan. Its `control` explicitly means parent `rows`, not topology
control. Parent full plan remains unsubmitted; source57/cache reused unchanged.
Remote plan validation PASS24groups/12roots/EP8.36captures instead of80; GPU7473998
submitted21:15:52Z, observed RUNNING nvl72078-T[07,15],30min limit. Analysis7474001 scheduled
afterok on CPU. GPU7473998 now COMPLETED14m23s0:0; all36captures, stderr empty.
Analysis7474001 COMPLETED3m33s0:0. PASS36captures/48records/144exact comparisons/
288replays, matched collective preparation; both rank reviews and selection saved.
Scan-minus-rows kernel/pipeline(us): Q332+.036098/+.043580 (pipeline every rank mean
regresses both orders); Q3532-.153702/-.163068 (mixed ranks/tails). Q31K-.148182/
-.250282 improves every statistic/order. Q32K-.109643/-.113704 has mixed reverse
ranks/tails; Q351K-.049493/-.048324 mixed ranks/tails. Q352K+.200243/+.198252:
EVERY kernel rank/pooled statistic regresses both orders (pipeline reverse p95 ties).
Reject general scan57; no cutoff/default/holdout/EP16 promotion. All samples retained.
Candidate kernel gaps to pull Q31K3.674976,Q32K4.496660,Q351K9.353999,Q352K12.839686us;
all four regress every statistic/order versus pull. Startup58 uses scan0 independently.

Independent source-only prototype58: default-off METADATA_SINGLE_JOIN removes the
first full-worker metadata startup join. Lane0 still performs SYS route acquire;
warp0 sync propagates it before lane-owned count loads. Retained full-worker join
after prefix/total publication orders all later metadata readers. SizesReady, shared
footprint, geometry, per-tile certificates/joins and retirement remain unchanged.
Four-header105-line patch/order review saved under C/metadata_startup_join_*.
Register scan is OFF in this independent comparison. No cutoff/composition/default
promotion. Uses existing R/source/local-mirror,25explicit local files synced; frozen57
GPU source/cache untouched. Fresh58-argument CPU cache; six cases32/1K/2K, expected
18resources/6geometry/6default gates. SASS/row-preservation still required before GPU.
CPU7474037 submitted21:23:01Z, observed RUNNING cpu-0013. Found dormant original
resource-checker small.append(ep8_tokens256) lookup, unused by checks and incompatible
with fast32 control. Submitted file kept frozen; generic generator now removes it.
CPU7474051 afterany7474037 reviews the same compiled artifacts with corrected separate
workflow/resources_fast.py then default instructions, no rebuild. If first job fails
at that lookup, do not rerun compilation; inspect recovery. No GPU58 submitted.
Both jobs now terminal:7474037 FAILED7m40s at unused256 lookup after compilation;
7474051 FAILED52s at real Q35total32 rows_join spill. SASS STL03f0 stores a shared
count-lane address; five LDLs restore it after role register changes. Reject initial
candidate58, no GPU. Four failed-source headers retained in C/metadata_startup_join/
source_headers, original patch/SASS/gates preserved. Local-mirror reused after both
consumers terminal. Existing count-lane recomputation now also applies when single_join
is enabled, at producer count load and scheduler fetch; default control unchanged.
Separate C/metadata_startup_join_late fresh58 cache; revised CPU7474172 submitted.
Six cases,18resource/6geometry/6default/6row-preservation gates; then startup SASS review.
Generator removes dormant256 lookup before submitting this plan. No new GPU/EP16.
CPU7474172 COMPLETED7m46s0:0 on cpu-0012; monitor39147 exited0. PASS18resources/
6geometry/6default/6row-preservation checks. Max initial/caller/callee/epilogue
125/39/125/120, no new spills. All6 saved-SASS startup reviews PASS: original five
full256-worker barrier3 sites become four; SYS acquire/equality8 poll retained,
full-mask BRA.DIV→WARPSYNC.COLLECTIVE before count LDG; scan5/10 shuffles,
carry1/2, shared3/4 stores and one SizesReady release retained. Remaining full-role
prefix handoff/tile/retirement joins stay. This is static evidence, not device proof.
C/metadata_startup_join_focus passed24group/12root plan validation. GPU7474315
submitted21:47:47Z, observed RUNNING nvl72005-T[01,11]; CPU7474318 afterok.
36captures row-only(`control`)/rows_join/pull, Q3/Q35 EP8 totals32/1K/2K bothorders,
1000calls/rank; source local-mirror and late58 cache frozen. Parent48capture plan
remains unsubmitted. Monitor47769 at300s; await correctness/all-rank performance
before any broader EP8/EP16/default promotion. Scan57 remains off and rejected.
Generic preparer now accepts --control-total so CPU preparation shares the fast set.
GPU7474315 COMPLETED14m13s0:0; analysis7474318 COMPLETED4m11s0:0; monitor47769
exited0. PASS36captures/48records/144exact comparisons/288replays and matched
collective preparation. Reject58: Q3/Q35total32 kernel+1.011980/+3.337232us,
pipeline+1.033219/+3.461911us; EVERY rank/pooled statistic regresses both orders.
Large kernel/pipeline(us): Q31K+.310047/+.220311; Q32K+.073359/-.140131 (both
metrics flip order); Q351K+.347157/+.357462; Q352K+.182998/+.188802. Q31K kernel
allstats/orders regress; other large cases have some mixed tails. No broader EP8,
EP16 or default promotion. Complete selection_review.json and rank reviews saved.
Keep single_join=0 and register_scan=0; row-only remains reference.

Next diagnostic C/mma_wait_probe: isolated1.1MB header copy adds six named IKET
probes to first FC1 and first FC2 tiles per leader CTA. Measures TMEM-empty wait
and each K-block operand-full wait. No ordinary kernel edit or native ABI change;
all probes compile away in ordinary mode. Prepared32captures (24ordinary+8profile),
six EP8 cases row-only/pull with four large profiles,10calls/rank. Reuse unchanged
validated58 native library. CPU7474739 submitted22:10:00Z, observed RUNNING cpu-0007;
monitor20755 at300s. Before GPU:8 profiler resource/register gates (MMA40), required
six-event coverage in each cubin,12 matched ordinary instruction-preservation gates,
geometry requiring FC1/FC2 work on every CTA. No early-SF/split-weight composition.
Runtime decoder requires every leader, both phases and exact ordered K-wait pairs;
rejects missing/extra/out-of-scope events. One targeted decoder test passes, including
negative cases; overlay generation on both current headers passes; diff-check passes.
These are instrumented intervals, not hardware stall counters; ordinary/profile bias
and existing core-marker coverage remain required. Source local-mirror, new cache,
headers and CPU plan frozen; no GPU profiling job submitted. New reusable helper
codex_scripts/prepare_megamoe_mma_probe.py; general phase preparation accepts
--control-total and --mma-probe-script. Campaign workflow copies carry new decoder
and reviewer; no changes to previously running source were needed.
CPU checkpoint22:15Z:7474739 RUNNING5m03s, reference ordinary compilation complete,
probe compilation started, stderr empty. Added control-vs-pull all-rank review to
unsubmitted analysis commands only; CPU source/cache/commands remain frozen.
CPU7474739 FAILED6m39s1:0 at the first IKET compile: existing32 PMTrig5 declarations
plus six new names exceed the event-ID space. All12 ordinary reference builds
completed; preserve failed headers/cache/logs. Monitor20755 terminal0. Extra compiled
low-role review7474884 submitted afterok, then CANCELLED0s when failure was observed.
No GPU. This was an instrumentation capacity failure, not a kernel-correctness result.
Revised helper deliberately keeps only6 core Dispatch/MMA/Compute/Combine markers,
removes non-core marker declarations/invocations in isolated dispatch/main headers,
and adds6 wait markers. Both pull/allpush statically have12 declarations. Maintains
PMTrig5 encoding, with detailed metadata/readiness/remainder/local-reduction analysis
explicitly disabled for this diagnostic; no detailed-phase claims. New headers_core
preserve original failed headers. Probe span ends at final operand-ready, before its
MMA; decoder label corrected accordingly. No production source or default changes.
CPU recovery7474913 submitted22:25:44Z, skips completed reference compilation/native
seed. Still requires8 profiler/resource checks and12 ordinary instruction checks,
then8 compiled low-role register/spill reviews using existing CFG reviewer. All data
and source/cache frozen during the retry; no automatic fallback or GPU submission.
CPU7474913 FAILED3m43s1:0: allpush ordinary/IKET compilation completed; pull
ordinary completed but IKET referenced MegaPayloadLoadStart, declared in dispatch
and called in main. Core filter previously collected declarations per file. Fix
collects the shared declaration set across both headers; static checks confirm12
core+probe events and all call names declared for both paths. Separate headers_core_v2
preserves failed sources/logs. CPU7475025 submitted22:34:46Z in cpu_recovery_v2;
completed reference/native builds reused. All8 resource/12 ordinary/8 low-role
compiled gates still required before any GPU. No source/default/performance change.
CPU7475025 COMPLETED5m31s0:0. Saved artifacts inspected: all8 profiler resource
checks PASS with exact12event sets; all12 ordinary instruction comparisons equal;
all8 compiled role-register reviews complete/PASS with no local-memory instructions.
GPU7475066 submitted22:41:08Z afterok CPU, observed RUNNING nvl72034-T[10,12].
32captures:24 ordinary row-only/pull mirrored plus8 profiles, allEP8, sixcases.
CPU7475072 submitted afterok GPU for correctness, all-rank control-vs-pull review,
core-marker coverage/profile bias and exact perCTA/phase/K-wait coverage. Monitor46868
at300s; source/cache/headers_core_v2 frozen. No optimization/default/EP16 promotion.
Checkpoint22:46Z:GPU7475066 RUNNING5m21s,10/24 ordinary captures observed complete
(including all8 small-control captures), stderr empty; analysis remains afterok.
Probe interpretation: markers start inside scheduler callbacks, so startup count
acquire and block-coordinate traversal are outside sampled spans. FC2-ready polling
in the activation producer may surface as operand waiting, but these intervals do
not separate readiness polling from TMA transfer time. No bottleneck claim yet.
GPU7475066 COMPLETED14m06s0:0;24 ordinary and8 profiled files verified, stderr
empty. CPU7475072 RUNNING cpu-0028 (observed50s at22:56Z); full correctness,
all-rank timing, core/probe coverage and instrumentation bias reviews pending.
CPU7475072 COMPLETED7m33s0:0. PASS32captures/48records/112exact comparisons/
320replays, matched collective preparation. Exact coverage:640calls,48,000 CTA
pairs,96,000 first-FC1/first-FC2 sampled tiles,2,016,000 operand-wait intervals.
Ordinary control-minus-pull kernel/pipeline(us): Q332-1.819504/-2.051162,
Q3532-.440655/-.598921; Q31K3.866128/3.976074,Q32K4.815584/5.357471,
Q351K9.478394/9.769273,Q352K12.926221/13.278251. No new candidate gain.
Mean sampled TMEM-reuse waits are.043–.125us. Largest later FC1 excess occurs
at K8=1024B: control/pull Q31K1.750/.178,Q32K3.181/.274,Q351K.667/.039,
Q352K1.533/.172us. Excess is positive on every rank (ranges1.292–1.881,
2.366–3.386,.359–.983,1.048–1.638us respectively). This matches prefix/middle
handoff; not proof of readiness versus TMA cause or an additive speedup budget.
Other FC1 waits mostly.04–.1us, with smaller periodic later spikes. First FC2
operand Q352K1.187 vs pull.062us; other FC2 differences smaller. Biases
profile-minus-ordinary(control/pull): Q31K-1.563/-1.357,Q32K-1.236/-.756,
Q351K-3.730/+.034,Q352K-5.022/-4.714us. Q351K relative bias is substantial.
Summaries, all-rank comparisons and wait_diagnostic_summary.json saved locally;
raw traces remain remote. Keep diagnostic/ordinary outcomes distinct.

Next hypothesis C/rows_sf_split: combine existing NARROW_SF3 with row partitions,
shifting SF/middle from4/4 to3/5 with fixed total workers/barrier identities and
matching participant counts; payload protocol unchanged. Earlier1/2-SF tests used8K/different layouts; count3 had ownership proof
but no GPU timing. New K8 evidence and row reference motivate this bounded test,
not an unchanged retry or a new token/model rule. Existing host block-M>32 gate
keeps both32 controls and Q351K compiled unchanged. First-pass sixcases,36captures,
rows(`control`)/rows_sf3/pull mirrored1000calls/rank. Generic prepare_pipeline.py
now validates SF-count geometry and can compare a named same-source reference arm.
Unchanged ABI58 native/JIT cache reused from metadata_startup_join_late after all
consumers terminal; no production edits.12 resource/6 geometry/6 rows-reference
SASS checks required before GPU; equal tuning fields require identical targets.
Plan validation PASS24groups/12roots/EP8. CPU7475525 submitted23:08:25Z, monitor86098
at300s. Source/local-mirror and shared late58 cache frozen; GPU remains unsubmitted.
CPU7475525 COMPLETED3m33s0:0:12 resources/6 geometry/6 reference-SASS checks
PASS. Maximum initial/caller/callee/epilogue registers123/39/125/120; no new spills.
Small controls and Q351K use identical targets/configs. Saved SASS for all3 changed
cases has SF barrier4=96 threads and middle barrier5=160, full256 retirement retained.
GPU7475615 submitted23:14:55Z afterok, observed RUNNING nvl72146-T[11,13];30min
limit for36 ordinary captures. CPU7475617 submitted afterok for correctness and
candidate-vs-rows/pull all-rank/order review. Monitor19806 at300s; source/cache
frozen, no default/EP16 promotion. Resource JSON files mirrored locally.
Source-only transport check while SF3 runs: rank-ready router's remote count atomic
returns an expert-local slot, used to publish receiver descriptors but not retained
by the payload sender. Senders use a destination-rank bitmask and one source-indexed
row copy per distinct peer. Direct final-row push would need retained slot plus
receiver padded-prefix/base readiness and a publication protocol; per-expert writes
would lose peer deduplication. This is a distinct transport redesign, not a pointer
substitution or justification to reread the validated source inbox with TMA. Deferred;
no code/job. Current capture observed12/36 complete (all small controls), stderr empty.
GPU7475615 COMPLETED13m25s0:0; CPU7475617 RUNNING cpu-0013 (1m18s observed
23:30Z), monitor19806 continues. Await full SF3 correctness/selection before promotion.

Independent source prototype59 C/sender_middle: profile K8 excess increases toward
later ranks, consistent with (not proof of) ascending-peer store order. Current
sender sends prefix1KB to all peers, then each peer's entire remainder. New default-off
DG_MEGAMOE_ALL_PUSH_SEND_MIDDLE_FIRST adds a middle-to-all-peers pass before tails.
The option uses existing block-M>32 geometry gate; small/M32 remains off. Four-header
patch118lines inclcontext,59th allpush template argument, no new storage, ownership,
values, peer deduplication, barriers or retirement change. Store issue order changes;
no cross-peer completion ordering is claimed. Wide sender vectors fail fast unless
the middle boundary is whole1KB. Prefix-sweep/producer-prefix intersections retained.
Host coverage696 configurations passes exact vector/word coverage with clipped/empty
windows and16/32B modes; not device proof. Patch/diff check passes. Before headers
preserved locally in C/sender_middle_source_before. Source cloned from frozen58
local-mirror to R/source/sender_middle_20260926 before four local59 header sync.
Running SF3 source/cache untouched. Local headers now59; never seed native58 or sync
those headers into active local-mirror58. Fresh C/sender_middle/cache will build59.
Generic prepare_pipeline tracks the new config and compares the named row reference
using its recorded source for assertion-line normalization.12 resources/6 geometry/
6 reference-SASS gates, then explicit sender-loop codegen review before GPU.
Pre-build plan validate correctly rejected absent fresh cache/native; CPU commands
compile/create it BEFORE their plan validation. No cache placeholder or fallback.
CPU7475759 submitted23:30:06Z, RUNNING cpu-0027, monitor73427 at300s. Six-case EP8,
36ordinary rows/middle_first/pull captures prepared but GPU unsubmitted. Independent
of SF3: NARROW_SF0 in both sender arms. No default/EP16 promotion.
SF3 analysis7475617 COMPLETED4m13s0:0. PASS36captures/48records/144exact comparisons/
288replays, matched collective preparation. PROMISING_Q3_AMBIGUOUS_Q35_NO_GENERAL_PROMOTION.
Versus rows Q31K kernel-.177630us,pipeline-.174202;Q32K-.726699/-1.161931us.
Every rank/pooled mean,median,p95,slowest and both orders improve for Q3 kernel/end.
Q352K kernel forward+.178855,reverse-.075955us; pipeline/end also flip sharply:
end+.451642/-.366903, forward every statistic regresses,reverse every statistic
improves. Paired kernel+.051450,pipeline+.042474us. Q351K unchanged code shows
-.019253/-.020759us paired kernel/pipeline, order flips. Shared-code32 controls
Q3+.001703/+.013379,Q35-.008337/-.008619us variation retained, not a feature effect.
Candidate kernel gaps to pull Q31K3.714537,Q32K3.888187,Q351K9.639251,Q352K13.174602us.
No default/EP16/case-winner policy. Broader8/256/nearby/nonuniform/interleaved EP8
and repeated Q352K required next for SF3; not yet prepared/submitted. Keep sender59
independent (SF0), do not compose its unmeasured effect with the SF3 gains.
selection_review.json,selection_summary.json,summary.json and both rank reviews
saved locally/remotely. Original full sub4K goal remains unmet.
Sender59 CPU7475759 COMPLETED6m19s0:0;12 resource/6 geometry/6 reference SASS
checks PASS, max initial/caller/callee/epilogue123/39/125/120. Explicit saved-SASS
review confirms three changed cases have prefix/middle/tail peer loops2/2/4 stores
versus2/6; same offsets cover4096B/warp. Small32 and Q351K instructions identical.
GPU7475899 submitted23:42:49Z, RUNNING nvl72172-T[12,15]; analysis7475908 afterok.
Monitor12913/300s. Source59/cache frozen; no code edits/default/EP16 promotion.
SF3 broader C/rows_sf_split_normal prepared96 normal-routing runs, both models
8/32/256/768/1024/1536/2048/3072, exact ordinary rows/SF3/pull settings mirrored.
Dropped old worker-specific384/448 holdouts to bound resources; nearby768/1536/3072
retained. General prepare_normal_followup now filters totals and takes the sole
native donor path from the ordinary plan (SF3 uses external late58 cache).
Initial plan generation used wrong arm label sf3; failed before submission, partial
folder archived *_incomplete_arm_label. Correct rows_sf3 succeeds; preflight now
rejects missing arms before writes.96-run/settings/source58 plan checks PASS.
CPU7475884 submitted23:41:38Z, RUNNING cpu-0007, monitor7635/300s.32 resource/
16 geometry/32 capacity gates required before GPU. Generic interleaved followup
preparer added; planned20 queued ABBA/BAAB processes on8/32/256/1K/2K, appends to
same allocation after normal references. Not generated until CPU gates pass.
SF3 normal CPU7475884 COMPLETED6m26s0:0;32 resource/16 geometry/32 capacity
checks PASS (max arena34,893,649,664B). All six8/32/256 controls share exact targets;
Q35 768/1K/1536 also unchanged. Only narrow_sf changes on the seven wider cases.
Generic interleaved prep initially failed because frozen source58 lacks runner;
partial folder archived *_incomplete_runner. Added explicit --runner, pre-read it
before writes, copied maintained root runner into the new workflow (no source edit).
20 queued processes now prepared, exact ordinary0/3 SF settings, both models
8/32/256/1K/2K, ABBA/BAAB,1000calls/arm/rank.96 normal runs precede references;
all samples retained. Template GPU env lacked OUTPUT_ROOT: submission rejected
locally before any scheduler call; explicit phase output now written by preparer.
GPU7476111 submitted23:53:53Z, PENDING afterany7475899 to avoid overlapping our
GPU comparisons.45min limit, two4-GPU nodes;96 normal +20 interleaved processes.
CPU analysis7476118 afterok. Monitor70068/300s. Sender7475899 last observed
RUNNING10m36s,27/36 captures complete inclall12 small captures, stderr empty;
its monitor12913/300s and CPU analysis7475908 remain active. No performance claim.
Previous goal turn: progress (sender CPU/codegen gates and GPU launch, SF3 CPU
launch). This turn: progress (SF3 broader CPU gates and normal/interleaved GPU
submission). Original all-sub4K objective remains unmet; no default/EP16 promotion.
Independent September27 sender_rotate60: all source ranks currently choose ascending
peer IDs. Saved K8 wait excess increases toward later ranks; rotation may distribute
issue pressure, but traffic contention is a hypothesis, not measured causality.
Default-off ROTATE_PEERS rotates32-bit peer masks by source rank modulo32 and
inverts bit positions on selection; mask words start at source rank/32. For EP<=32
this is cyclic peer order; larger EP uses cyclic words and within-word rotation.
No new payload/storage/barrier/retirement or per-peer writer changes. Prefix-wide
is explicitly incompatible; ordinary sender only. Existing M>32 geometry gate
leaves small/M32 off. Four-header122-line patch including context,60th template arg;
local before headers at C/sender_rotate_source_before. Host371,082 mask/source cases
PASS across EP1/2/4/8/16/32/48/64, exhaustive masks through8 plus empty/all/single/
pair/random larger masks; no device/performance proof. git diff check PASS.
Isolated R/source/sender_rotate_20260927 cloned from frozen59; four local60 headers
synced only there. Fresh C/sender_rotate/cache, never seed59. Six-case36-capture
ordinary plan rows/rotation/pull,1000calls mirrored. SF0 and SEND_MIDDLE_FIRST0;
no composition with unmeasured sender59. Generic pipeline tracks rotate config and
compares control against sender_middle/control with its recorded source.
CPU7476260 submitted00:01:14Z, RUNNING cpu-0013; monitor9267/300s.12 resource/
6 geometry/6 reference SASS gates then explicit sender-loop review before GPU.
GPU plan unsubmitted. Local headers now60; sources58/59 and their caches frozen.
Sender59 GPU7475899 COMPLETED14m08s0:0; analysis7475908 observed RUNNING2m13s,
monitor96603/300s. SF3 normal7476111 observed RUNNING1m52s nvl72172-T[12,15],
monitor70068/300s; analysis7476118 afterok. No performance conclusions yet.
Sender59 analysis7475908 COMPLETED4m07s0:0. PASS36captures/48records/144exact
comparisons/288replays, matched preparation. Both rank reviews/selection/summary
saved locally/remotely. Middle-first-minus-rows kernel/end paired means(us):
Q31K-.925543/-1.140917,Q32K-.601937/-.838837; every rank/pooled statistic
and bothorders improves for both metrics. Q352K-.209502/-.346583; kernel means
forward-.246367,reverse-.172637, every rank mean/median improves; reverse rank p95
max ties0, all other kernel statistics improve. End every statistic/order improves.
Small/M32 instructions/targets identical: Q332 kernel-.008601/end+.002183 with
orderflip; Q3532+.026045/+.022928, allrank kernelmeans worse bothorders, tails mixed.
Q351K+.052653/+.133727 paired, orderflip (end forward+.315482 every statistic
regresses, reverse-.048028). Retain all shared-code variation; no claimed small
preservation or causalfeature regression. Decision PROMISING_Q3_REQUIRES_BROADER_EP8_SMALL_CONTROL_VALIDATION.
Candidate still trails pull Q31K2.838119,Q32K4.389075,Q351K8.820583,Q352K12.715939us
kernel; no default/EP16 promotion and do not add gains to SF3 across allocations.
C/sender_middle_normal generated96 normal runs using exact source59 settings,
8/32/256/768/1024/1536/2048/3072 bothmodels; CPU7476355 submitted00:07:25Z.
Fresh normal JIT with same59 native donor; geometry must remain unchanged except
send_middle_first. Require32resource/16geometry/32capacity gates,
then append20queued interleaved (8/32/256/1K/2K) before GPU. GPU unsubmitted.
Post-GPU result sync initially included raw qwen* captures; stopped rsync session18261
via its own stdin interrupt and fetched compact rank_review/ explicitly. Raw evidence
remains remote; local partially copied capture trees are not a completeness claim.
Rotation60 CPU7476260 COMPLETED7m10s0:0.12resources/6geometry/6reference SASS
checks PASS; max initial/caller/callee/epilogue123/39/125/120. Saved sender codegen
review passes three changed specializations:2/6 stores with same eight offsets,
SHF.R.W mask rotation, add/mask destination inverse; static BAR/MEMBAR signatures
match control. Both32 and Q351K instructions identical. This is CPU evidence;
GPU exactness/performance still required. GPU7476416 submitted00:13:26Z,
PENDING afterany7476111 so own GPU comparisons do not overlap.36ordinary captures,
50min limit, source60/cache frozen. Analysis7476420 afterok; monitor97628/300s.
SF3 normal7476111 last observed RUNNING16m55s, stderr empty, monitor70068/300s.
Middle-normal CPU7476355 last observed RUNNING5m36s, monitor83440/300s; await
terminal gates before interleaved append/GPU. Previous goal turn was progress:
new rotation source/host proof/CPU launch and sender59 measured gains/broader prep.
This turn progress: rotation CPU/codegen gates reviewed and focused GPU queued.
Middle-normal CPU7476355 COMPLETED6m49s0:0;32resource/16geometry/32capacity
checks PASS, all six8/32/256 targets shared. Changed shapes: Q3 768/1K/1536/2K/3072,
Q35 2K/3072; other Q35 M32 unchanged. Appended20queued interleaved processes via
generic explicit-runner preparer; settings0/1 send_middle_first only,1000calls/arm/
rank both capture orders.96normal+20interleaved plan check PASS. GPU7476521
submitted00:20:09Z, PENDING afterany7476416 to serialize own GPU comparisons;
CPU analysis7476526 afterok. Monitor19625/300s. Same frozen source59 and its
separate normal cache; no SF3/rotation composition. SF3 GPU7476111 observed
RUNNING21m56s, stderr empty; bounded direct result-path audit at00:21Z finds all
96normal and13/20interleaved completed. Full analysis7476118 remains afterok.
Previous goal turn progress: rotation CPU/codegen proof and GPU queue. This turn
progress: middle-normal CPU gates passed, interleaved prepared, full EP8 GPU queued.
SF3 broader GPU7476111 COMPLETED25m12s0:0, analysis7476118 COMPLETED11s0:0.
PASS96normal runs:1152records/3456candidate-reference comparisons/768replays;
20interleaved processes:1440comparisons/160replays/160interleaved checks,320k rank
samples. Normal exact cross-arm comparisons are allpush candidate/reference;
pull self-replay passes but this is not a new cross-protocol byte comparison.
Decision HOLD_SF3_NO_ROBUST_BROADER_GAIN_NO_EP16. Q3 normal joined paired deltas
768/1K/1536/2K/3072=-.086652/-.142570/+.051346/-.093822/-.442622us;1536 forward
allrankmeans regress+.096124us, reverse+.006568 mixed ranks/tails. Q35 2K/3072
-.113484/-.118852us with mixed ranks/order magnitudes. Interleaved Q3 1K ABBA
+.006872/BAAB-.086244us,2K-.042868/-.171252us; only2K BAAB every rankmean improves,
rankp95 worst ties0. Q352K-.015292/-.112520us, rankp95 max+.512004/+.671998us.
Small/M32 targets identical; all variation retained. Normal Q3532/256 paired
+.057110/+.066972us, interleaved Q358+.089716/+.038628,Q3532+.013856/+.033156us,
ranks/tails mixed. Small preservation and robust common gain remain unproven;
not a causal SF3 small regression claim. No EP16/default/case-winner policy or
composition with middle-first. All rows/ranks saved in both analysis/review.json
files and compact normal/selection_review.json, mirrored locally/remotely.
Rotation7476416 now RUNNING5m50s nvl72172-T[12,15], all12small captures completed;
monitor97628/300s. Middle-normal7476521 PENDING behind rotation, monitor19625/300s.
Previous goal turn progress: middle-normal CPU proof/interleaved/GPU queue. This
turn progress: SF3 broader result reviewed and promotion withheld based on evidence.
September27 00:39Z: rotation GPU7476416 COMPLETED13m54s0:0; analysis7476420
RUNNING1m23s on cpu-0028, monitor78916/300s next~00:44Z. Middle-normal GPU7476521
last observed pending dependency, monitor19625 next~00:41Z. No new GPU submission.
Bounded source audit of Q351K M32 gap: sender changes leave this geometry unchanged;
previous FULL_TMA2/extent, WAVES2 and M64 policy tests already failed controls and
must not be revived via token/model cutoffs. Static FC1/FC2 interleaving is not a
scheduler-only edit: allpush single-wave full barriers use4 rather than6 credits;
activation issuers skip FC2 empty waits/arrivals, so a later FC1 could lap its ring
phase. Independent role schedulers also cannot choose ready tasks separately.
CuTeDSL phase-interleaving publishes common work descriptors and forces an FC1
claim ahead of a held unready FC2 consumer to preserve producer progress. A future
port needs common task publication, full producer-credit participation and a
ring-wrap/dependency proof. No kernel change made from this audit. Goal remains
unmet; defaults unchanged, small-control evidence and EP8 gains required before EP16.
September27 00:44Z: rotation analysis7476420 COMPLETED3m20s0:0. PASS36captures,
48records/144candidate-reference comparisons/288replays, matched preparation.
All compact summary/selection/rank reviews fetched explicitly; no raw capture sync.
Decision REJECT_GENERAL_PEER_ROTATION. Candidate-minus-control paired kernel/end(us):
Q31K-.060466/-.162439 (forward+.041247/+.047471, reverse-.162178/-.372350),
Q32K-.363270/-.831133 (every rank/statistic/bothorders improves),
Q352K+.168592/+.273190 (kernel bothorders+.184214/+.152970, end+.186259/+.360122;
reverse end every rank/statistic regresses). Shared-code Q332-.013152/-.013426,
Q3532+.014577/+.025415, Q351K+.038625/+.042821; retain variation, no small
preservation claim. Kernel still behind pull3.697712/4.672962/8.876369/12.983319us
for Q31K/Q32K/Q351K/Q352K. No broader GPU, composition, default or EP16 promotion;
no model/token winner policy. Selection decision saved sender_rotate/selection_review.json.
Middle-normal7476521 RUNNING3m23s nvl72172-T[12,15], monitor19625/300s next~00:46Z;
analysis7476526 remains afterok. This turn progress: audited rotation and rejected
common promotion based on repeatable Q35 regression. Full objective remains unmet.
September27 00:50Z new independent sf_multicast61: existing SFA and SFB TMA sites
issue identical full scale boxes in both CTAs; no CTA-specific SF coordinates.
Leader-only SM100_TMA_2SM_LOAD_MULTICAST_2D mask3 replaces each duplicate pair,
retaining both shared-memory copies, cache hints, all producer arrivals and 2x SF
transaction bytes. PTX9.0 states each destination sends completion to its pair's
barrier CTA selected by mbar parity; both destinations credit CTA0 here:
https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
Leader's existing empty wait follows pair-wide MMA completion; follower retains
its arrival credit so no producer can be lapped. No extra storage/barrier, scheduling,
math or format change. Source proof only; GPU byte/replay correctness remains required.
DG_MEGAMOE_ALL_PUSH_SF_MULTICAST0/1 defaults0 and applies to every geometry,
including M32/Q351K. No geometry/token/model cutoff; candidate32 code changes and
must pass perf controls. Three header changes archived in sf_multicast.patch and
sf_multicast_source_before; source60 frozen, new source sf_multicast_20260927 cloned
then three local authoritative headers synced only there. Fresh native61/JIT; no
cross-signature seed. Generic prepare_pipeline.py tracks sf_multicast in geometry
and same-target checks. Six cases Q3/Q35 EP8 32/1K/2K, control/sf_multicast/pull,
mirrored1000calls/rank,36captures prepared. SF0/middle0/rotation0 both arms.
CPU7476868 submitted00:50:14Z:12resource/6geometry/6default-SASS checks vs rotation60
control required before GPU. No GPU/analysis submitted yet. Explicit codegen review
must verify multicast destinations/leader guard and unchanged arrival accounting.
Middle-normal7476521 remains running, monitor19625/300s. Previous turn progress:
rotation measured and rejected. This turn progress: new mechanism source proof,
minimal default-off implementation, isolated source/cache and CPU job submission.
September27 01:07Z: multicast CPU7476868 COMPLETED11m01s0:0;12resource/6geometry/
6default-instruction comparisons PASS. Max initial/caller/callee/epilogue123/39/125/
120, no new local memory. Default matches60 except verified assertion-line metadata.
Saved-SASS reviewer workflow/review_multicast.py passes6specializations/48SFsites:
UTMALDG.2D.MULTICAST.2CTA, encoded mask3, nonleader skip guards, unchanged static
synchronization opcode counts and payload TMA site count. Source addresses/expected
bytes unchanged; counts are not full CFG/device proof. GPU7477048 submitted01:05:16Z
with afterok7476868 AND afterany7476521, now RUNNING on nvl72172-T[12,15];36captures,
monitor51189/300s next~01:10:50. CPU analysis7477054 afterok, not monitored yet.
Source61/cache frozen. Small candidate code changes, no default/performance claim.
Middle broader GPU7476521 COMPLETED25m19s0:0; analysis7476526 COMPLETED10s0:0.
PASS96normal:1152records/3456candidate-reference compares/768replays. Pull only
self-replay here, not a new cross-protocol byte reference. PASS20interleaved:
1440compares/160replays/160interleavedexact,320k rank samples. All compact reports
fetched explicitly. Decision HOLD_MIDDLE_FIRST_BROADER_Q3_REGRESSION_NO_EP16.
Normal joined paired deltas(us) Q3 768/1K/1536/2K/3072:
-.027814/-.127946/+.163840/+.154408/+.038250. Q31536 forward+.231792/reverse+.095888,
every rankmean regresses bothorders;2K forward+.340988 everyrankmean worse/reverse
-.032172 mixed. Interleaved Q31K ABBA+.054520/BAAB-.078908, mixedranks;
Q32K-.149684/-.217564 everyrankmean improves, worst rankp95+.224009/+.192001.
Q352K normal-.290272, everyrankmean improves bothorders (forward rankp95max+.031993,
reverse0); interleaved-.289720/-.494888, everyrankmean improves, all10windows/order
negative, rankp95max0/-.064000. Q353072 normal-.315918, mixedtails. Shared-code Q351K
normal-.232258 orderflip, interleaved-.022980/-.039620 mixedranks; retain variation.
Small controls same targets: normalQ38+.059022/Q3532+.020356, interleavedQ38
+.048592/+.004220,Q332+.088256/-.068800,Q3532+.027652/+.034248, ranks/tailsmixed.
No proof of small preservation and no causal feature-small regression claim.
No case/model winner policy, composition, default or EP16 promotion. Detailed
all-row ranges in sender_middle_normal/selection_review.json and both analysis/
review.json files. Previous turn verifiedwait; this turn progress: CPU/codegen
reviewed, focused multicast GPU launched, broader middle result reviewed/held.
September27 01:11Z multicast device failure: GPU7477048 FAILED3m02s1:0. Two
control32 captures complete, first Q3candidate32 fails exact output on all8ranks:
22664–22927 output bytes differ. Kernel completed; this is an output mismatch,
not a completion timeout. Dependent analysis7477054 CANCELLED before allocation,
confirmed terminal. Root capture logs/failure_review.json retained locally/remotely.
Correction to the earlier SFA/SFB source claim: expressions look similar across
CTAs, but the scheduler explicitly assigns adjacent N tiles (n_block_idx differs1).
SFB coordinate n_block_idx*BLOCK_N therefore differs128; leader SFB multicast was
incorrect. SFA pool row/K coordinates do match. CPU/static opcode checks cannot
validate source-coordinate equality and their earlier success is not device proof.
Do not reuse both-SF candidate or claim its partial run measures performance.
Corrected local source restores original per-CTA SFB and retains only leader SFA
multicast. Rename option/field/template to SFA_MULTICAST/all_push_sfa_multicast/
kSfaMulticast for clarity, default0 allgeometries; still61arguments, new native
needed because host field/emitter changed. Source `sfa_multicast_20260927` cloned
from failed source then3authoritative headers synced only there; failed source/cache
frozen. New C/sfa_multicast separate native/JIT, six EP8 Q3/Q35 32/1K/2K,36captures
control/sfa_multicast/pull mirrored1000calls/rank. No GPU submitted. CPU7477164
submitted01:15:30Z,12resource/6geometry/6default-SASS gates vs rotation60 control.
Independent flattened pair-coordinate model passes324configs/200448pairs with
empty/ragged experts, M16..256 and8..64 even N tiles: paired expert/M/SFA pool/K
match, SFB N differs128. Existing even grid/phase N tile counts preserve pair
adjacency. Not device/scheduler CFG proof. Ownership JSON, before headers, correction
patch saved. New codegen reviewer expects4SFA multicast sites/specialization;
SFB remains regular TMA. Prior both-SF reviewer/report retained as failed-audit
history, not silently rewritten. Default paths preserved; small timing controls
still required after exactness. This turn progress: failure diagnosed/corrected,
dependent analysis cancelled, isolated recovery source/cache and CPU job submitted.
September27 01:29Z corrected SFA-only CPU7477164 COMPLETED6m46s0:0.12resource/
6geometry/6default-instruction gates PASS; max initial/caller/callee/epilogue
123/39/125/120, no new spills. Default SASS matches rotation60 except verified
assert-line metadata. Saved reviewer passes24SFA multicast sites across6shapes,
mask3/leader guards/static sync opcode counts. Only alternating sites in the final
TMA role use multicast; earlier weight/SFB sites stay regular. Source coordinates
and200448paired-tile model distinguish SFA from nonshareable SFB; device proof
still required. Reports/reviewer mirrored. First GPU sbatch rejected obsolete
completed dependency afterok7477164, no job created. CPU completion0:0 and allgates
already verified; removed only obsolete dependency and resubmitted. Failed request/
log archived gpu/rejected_completed_dependency.*; not a GPU retry or weakened gate.
GPU7477331 submitted01:28:58Z, RUNNING nvl72165-T[07,11], monitor28243/300s next
~01:34:45Z.36focused EP8 captures with32controls first, source/cache frozen.
Analysis7477337 submitted afterok; unmonitored until useful. No GPU performance
conclusion/default/EP16 promotion. Previous turn verifiedwait; this turn progress:
corrected CPU/codegen gates reviewed and full focused device validation launched.
The user reiterated preserving small-batch performance: retain validated baseline,
reject small regressions, broaden to8/256 controls after focused progress, then EP16.
September27 01:35Z verifiedwait: monitor28243 confirms7477331 RUNNING5m36s,
11/36captures complete including corrected32 candidate runs; captured stderr empty.
This is progress past the earlier failure, not full correctness or performance proof.
Next scheduled observation~01:39:45Z; source/cache unchanged, analysis remains afterok.
September27 01:40Z monitor28243 confirms RUNNING10m37s,27/36captures complete,
including all12 small controls. Next~01:44:45Z. No timing selection yet. This turn
progress: independent scheduler host-model prerequisite completed (Remaining ideas).
September27 01:44Z GPU7477331 COMPLETED13m51s0:0, captured stderr empty and final
Q352K control-repeat capture reached. Monitor28243 terminal0. CPU7477337 RUNNING
1m29s oncpu-0010; monitor1042/300s, next~01:49:25Z. Full correctness/performance
audit pending; no broader/EP16/default promotion. This turn progress from completed
device workload; scheduler host model remains separate with no kernel changes.
September27 01:50Z analysis7477337 COMPLETED3m37s0:0; monitor1042 terminal0.
PASS36captures/48records/144comparisons/288replays, matched collective preparation.
All-rank/order reports and selection_review.json fetched and preserved. REJECT
SFA_MULTICAST_SMALL_AND_Q35_1K_REGRESSION. Candidate-minus-control kernel/pipeline(us):
Q332-.011769/-.005424 with order/rank/tail variation; Q3532+.183486/+.179028,
every rank mean/median worse both orders, forward every statistic regresses and
reverse p95 minimum ties0. Q31K-.098915/-.101041 every mean/median/order improves
(kernel reverse worst p95 ties0); Q32K-.019386/-.015846 mixed ranks/tails.
Q351K+.909318/+.912195 EVERY rank/pooled statistic and order regresses.
Q352K+.067076/+.173777, forward all rank means regress; reverse mixed.
Candidate kernel still behind pull3.719123/4.791244/9.812048/13.697256us on four
large targets, every rank/stat/order worse. Pull comparison uses exact self-replay,
not cross-protocol byte comparison. No broader GPU/EP16/default/composition or
model/token cutoff rescue. Retain SFA_MULTICAST=0 baseline and frozen candidate
for evidence. All submitted jobs terminal. This turn progress: complete audited
performance result rejects the candidate under the user's small-preservation gate.
### Rejected ready-work queue family — archived September27

All ten queue variants failed the retained-control small-batch gate or broader
performance. Their implementations were removed in cleanup pass1; defaults never
changed. Every completed comparison used Q3/Q35 EP8 totals32/1024/2048, mirrored
1000calls/rank, matched collective preparation,36captures/48records/144exact
comparisons/288replays. All ranks and samples retained. CPU/codegen gates passed;
that static evidence did not predict acceptable device performance.

Paired candidate-minus-control kernel deltas in microseconds; column order is
Q3/Q35 small32, Q3 1K/2K, Q35 1K/2K. Full pipeline/rank/order/tail evidence remains
in C/<campaign>/selection_review.json and rank_review/. Each source has its own
frozen native/JIT cache; equal62-argument ABI is not compatibility proof.

| Campaign | GPU / final analysis | Kernel deltas (six cases) | Rejection |
|---|---|---|---|
| ready_scheduler |7477991 /7477996|+5.960,+14.523,+4.381,+2.868,+9.483,+17.762|All six means regress.|
| ready_local_ack |7478615 /7478897|+3.193,+9.777,+1.686,-.323,+7.086,+2.740|Five cases regress every statistic/order.|
| ready_local_tiles |7479154 /7479157|+2.808,+7.513,+1.532,-.342,+5.628,+2.853|Five regress; Q3 2K pipeline order flips.|
| ready_phase_skip |7479616 /7479625|+2.222,+9.038,+1.609,-.742,+6.035,+3.117|Only Q3 2K robustly improves.|
| ready_joined_ack |7480072 /7480076|+2.186,+8.696,+1.539,-.665,+6.334,+2.822|Only Q3 2K robustly improves.|
| ready_local_publish |7480501 /7480539|+1.674,+1.492,+2.197,+.001,+3.920,+1.748|Small and large regressions; Q3 2K tails worsen.|
| ready_shared_ack |7481200 /7481208|+1.511,-.128,+.130,+.274,+3.916,+1.949|Q3 small regression; Q35 small p95 worsens3.456..4.704us.|
| ready_async_publish |7481531 /7481543|+.964,-2.468,-.222,-.456,-.127,+.836|Q3 small and Q35 2K every-statistic regressions.|
| ready_active_readers |7481953 /7482525|+.921,-1.982,-.129,-.213,+.126,+.618|Q3 small and Q35 2K regress.|
| ready_shared_wait |7482557 /7482570|+.799,-2.217,-2.398,-.790,-.041,+4.484|Q3 small/Q35 2K regress every statistic/order.|

All GPU/final analyses above completed0:0. Local-ACK original analysis7478629 failed on missing gpu/job.json;
7478897 recovered the saved captures without a GPU rerun. Active-reader original7481966 failed writing missing rank_review/;
7482525 recovered existing captures without GPU rerun. Submission helper now mirrors
job.json remotely; reviewers must create output directories before writing.

Mechanism progression: leader publishes full-ready FC2 else FC1, preserving carried
CTA ownership and adjacent N coordinates; bounded four-slot descriptors require six
operand credits and drain before reuse. Local ACK/descriptor replication moved DSM
reads locally; phase skipping removed unwanted-range iteration; joined ACK merged
follower relay; local/shared/async publication changed the handoff. Active readers
reduced29 acknowledgers to23 (13/11 init including relay) while retaining cleanup.
Shared-acquire wait removed18global invalidations but retained independent payload
readiness. None repaired the small regression. Do not revive these mechanisms
without materially new evidence. Retirement-based reuse was never implemented;
CTA-scoped ordering remains unproved.

Latest shared-wait six-case geometric kernel/pipeline speedups:1.006729/1.005624
versus control,.953209/.961187 versus pull; four-large versus pull.913227/.927455.
Strong order effects Q3 1K -4.394/-.402us and Q35 2K+8.033/+.935us remain visible.
Aggregate improvement over control does not waive small preservation. Host scheduler
models prove ownership/read accounting only, not real bounded-queue progress.
Full pre-consolidation detail is in deep_cleanup_20260927/before_cleanup.tar.gz,
with all original patches, source snapshots, codegen and captures unchanged.

Completed metadata_row_partitions_q35_ep16: source unchanged frozen
R/source/metadata_row_partitions_20260925; fresh native/JIT/TVM, no seed. Extend
retained mechanism following revised aggregate criterion and EP8 Q35 2K mean gains
(ordinary1.928us, normal.512us, interleaved.438/.378us); individual non-small tails
remain visible but are no longer automatic vetoes. No final/default acceptance.
Four cases Q3192/Q35912 small controls +Q351024/2048, four arms control/workers/rows/
pull mirrored1000calls/rank,32captures. Q35 EP16 base=min(512/16,2*16)=32, so
preserve logical32 with physical64/P2; W64/P1 isolates worker count. Do not copy Q3's
W16/P2, which would change logical parallelism. Q3 base8→W16; small specialization
ignores overrides. Geometry formula applies uniformly, no runtime winner rules.
W64 fits128metadata+16senders+1helper=145<=150. Both older small controls remain.
Remote plan PASS24groups/16roots/EP16. CPU resources.py adapted totals and derives
requested workers=2*base;12resource/4geometry triple checks. Added six exact ordinary
instruction comparisons against original measured small control SASS (no source/hash
check). CPU gates completed; Q3 large need not be rerun unchanged. Current local
CUDA headers still rejected shared_wait/defaultoff62; no frozen source overwritten.
CPU7483022 COMPLETED5m10s0:0. All12resources/4geometry/6original-small instruction
comparisons PASS. Q3192 small3808instructions/Q35912 small3992instructions identical
for all three push arms versus original measured controls. No new spills. Q352K
control/worker/rows initial registers115/112/112, epilogue100/100/99; caller max39,
callee max116. Actual Q351K remainsM32/control layout; changed case Q352K M96,
controlW32/P1, workersW64/P1, rowsW64/P2. GPU7483103 submitted06:55:13Z,32captures.
CPU monitor31185terminal. Previous goal turn progress(plan/submission); current
progress CPU gates reviewed and GPU follow-up submitted. Full goal unmet.
GPU7483103 COMPLETED13m34s0:0; analysis7483116 COMPLETED5m27s0:0.
PASS32captures/64records/320compares/512replays,16ranks1000calls/bothorders.
PROMISING_ROWS_EP16_HOLD_FOR_INTERLEAVED_SMALL_CHECK. Rows minuscontrol kernel/pipeline
(us): Q3192 -.001195/+.000805; Q35912 +.026466/+.024785;
Q351K -.000246/+.000134; Q352K -1.367633/-1.455773. Q352K every rank/stat/order
improves; rows versus workers-only -1.182518/-1.268171us, also all improve.
Four-case geometric mean vs control1.004733/1.003879, vs pull1.000124/1.000377;
large two-case vs pull.954913/.964951. These subset scores do not establish final
acceptance. Q35 small variation remains despite identical instructions; preparing
balanced same-process ABBA/BAAB for these four cases, reusing validated source/cache/
references. No rebuild or default promotion. Full selection_review saved.
Saved row_partition_geometry_review.json: frozen dispatcher423–429 partitions actual
ready rows, not paddedM96. Balanced Q352K gives40rows→P2 32+8; first-window rounds
3→max(2,1) at256threads/16lanes per row. Hypothetical4-row alignment20+20 still
max2 rounds but raises total rounds3→4; no new finer-grain variant justified from
balance alone. Source arithmetic is not a measured critical-path explanation.

Completed `metadata_row_partitions_ep16_interleaved`: GPU7483864 COMPLETED3m34s0:0;
CPU analysis7483868 COMPLETED3m51s0:0. PASS8processes/256000rank samples,
128exact compares/128replays/128interleaved checks. RETAIN_ROWS_FOR_BROADER_COMPARISON.
Joined rows-minuscontrol means(us), ABBA/BAAB: Q3192 +.211348/-.170838;
Q35912 +.117078/-.109530; Q351K +.091832/-.104642;
Q352K -1.293732/-1.317536. Q352K every rank mean/median/p95 and every100-call window
improves bothorders; paired-1.305634us. Four-case joined geometric mean vs control
1.002715x (Q352K1.011298x); no pull comparison. Small paired Q3192+.020255us,
Q35912+.003774us with mixed rank stats and graph-order reversal; original compiled
instructions identical. No stable small slowdown established; retain variation,
no full-set/default claim. Complete selection_review saved and mirrored.
Same Q3192/Q35912/Q351024/Q352048 cases; balanced within-process control/rows,
both ABBA/BAAB capture/schedule orders,1000calls/arm/rank,8processes256000samples.
Reuses validated frozen row source and completed EP16 native/JIT/TVM/reference;
no rebuild or source mutation, require-cache enabled. Remote plan/input checks PASS,
64 original rank references present. Existing interleaved harness retains exact
outputs and all samples; reusable review_megamoe_interleaved.py adds explicit
routing/count/order coverage and equal-case control geometric mean, rank statistics,
100-call windows and paired cycles. Joined graph includes preparation; no pull
comparison in this follow-up. The follow-up strengthens the retained row path; final small preservation and
full-set success remain open.

First `metadata_row_partitions_ep4` GPU7484248 FAILED6m46s1:0 on nvl72113-T03
while saving weights to a full filesystem; analysis7484280 cancelled unallocated.
Source/cache and partial captures preserved; recovery below. Six cases Q3/Q35 totals
32/1024/2048; control/workers/rows/pull mirrored1000calls/rank,48captures. Same
retained frozen row source, fresh native/JIT/TVM originally prepared, no seed.
Both models base=min(experts/4,8)=8; W16/P2 preserves logical8 while W16/P1 isolates
worker count. Small paths and Q351K share unchanged control targets. This extends
promising EP8/EP16 evidence to low EP before full91-case geometric comparison.
CPU7484039 FAILED7m21s1:0 after all compilation/barrier and32group/16root plan
validation passed. EP8/16 resource checker incorrectly assumed every EP4 baseline
had zero local accesses: Q3 M16 already carries a shared address via[R1+0x18]
across register changes. All three arms resolve to the exact same target/config.
CPU-only review7484200 COMPLETED52s0:0 using existing cubins; no rebuild/GPU retry.
All18resource/sixgeometry checks PASS. Existing local accesses accepted only when
all three target/configs are identical and full_tma=0; changed targets retain strict
no-spill gates and every target retains register budgets. Full SASS/memory evidence
and initial failure saved; this does not relax new-candidate spill checks.
Changed row maxima initial118/caller39/callee121/epilogue121; no new spills.
Actual changed cases Q3 1K M96/2K M128 and Q35 2K M96, W8/P1→W16/P2; Q351K M32
unchanged. Both smalls M16 unchanged. Added variants to scope for selection reviewer
before GPU. GPU submitted07:49:01Z, analysis07:49:34Z; defaults unchanged.

Prepared focused `metadata_row_partitions_ep2/ep32/ep64` CPU comparisons while EP4
GPU ran. CPU7484383(EP32) FAILED1m03s and7484384(EP64) FAILED55s, both compiler
disk-quota-exceeded. CPU7484382(EP2) was cancelled4m28s after full storage confirmed.
Partial caches preserved; clean-output/cache recovery below. EP2 Q3 only32/1K/2K; EP32 bothmodels
32/1K/2K; EP64 both64/1K/2K. Four arms, bothorders;24/48/48captures planned.
Original hold: review EP4 timing and CPU gates before additional GPUs; superseded
by September27 explicit submit-all request below, with CPU gates enforced by afterok.
Same rule physicalW=2*min(experts/EP,2*EP),P2 preserves logical workers:
EP2 Q3W8; EP32 Q3W8/Q35W32; EP64 Q3W4/Q35W16. Per-model experiment settings
are derived from topology, not measured winners. GPU plan and CPU compile arguments
agree. EP4 donor native copied into independent caches only after same frozen source,
image and venv agreement; JIT/TVM fresh, no donor/cache/source mutation.
Added reusable review_megamoe_row_resources.py ROOT, derived from passed EP4 review:
read explicit scope EP/totals, disassemble every target before review, preserve role
register budgets, require no new spills on changed targets, identical-small-target
proof and topology geometry. Copies frozen in each workflow;9/18/18resource gates
remain pending. The new reviewer itself is exercised by these CPU jobs; no GPU claim.
Previous turn progress: EP4 recovery and GPU launch. Current progress: prepared and
submitted remaining focused topology CPU checks. EP4 last monitor5m44/12of48;
analysis7484280 pending. Full91-case score remains unmeasured.

Storage recovery September27 18:05UTC: Lustre reported35TiB used/0available, causing
EP4 torch.save weight-cache writes and EP32/64 compiler writes to fail. No kernel
correctness rejection. Approved cleanup manifest/receipt C/profile_weight_cleanup_
20260927 removed exactly2880 unused profile-rank-weights.pt files from10completed
rejected ordinary-only queue campaigns:847,088,913,600bytes. All timings/traces/
exact references/source/native/JIT retained. Capacity also grew35→50TiB during
approval wait; post-cleanup16TiB available is not all attributable to deletion.
Maintained profile_megamoe_iket.sbatch now supports explicit
FASTAFD_IKET_SAVE_WEIGHTS=0 only with capture_phase=normal; default1 retains staged
profiling behavior and unsupported modes fail. Bash syntax PASS. Retry workflows
copy the exact frozen source harness plus only this shell change; CUDA source and
kernel/cache compatibility unchanged. Cleanup helper defaults to manifest dry run,
revalidates completed GPU jobs and file inode/size/mtime before explicit deletion.

September27 user explicitly requested: submit all sweep jobs and stop. All prepared
focused topology runs are submitted, superseding the manual EP4 timing hold:
EP2 CPU7492877 -> GPU7492935 -> analysis7492936;
EP4 validated cache -> GPU7492875 -> analysis7492881;
EP32 CPU7492878 -> GPU7492937 -> analysis7492938;
EP64 CPU7492880 -> GPU7492939 -> analysis7492942.
GPU/analysis links use afterok dependencies; failed CPU gates prevent GPU allocation.
At18:12UTC EP2 CPU completed0:0, EP2/EP4 GPUs running, EP32/EP64 CPUs running,
remaining GPUs and analyses pending dependencies. Same-source native copies and
fresh JIT/TVM preserve storage recovery; EP4 reuses validated original cache.
Monitor53980 stopped at user request; no further polling, optimization or submissions.
Goal remains paused, submitted Slurm jobs continue. Prior failed artifacts preserved.
No defaults changed; full91-case geometric result and final small preservation unmet.

September27 result check requested by user: all11 submitted CPU/GPU/analysis jobs
COMPLETED0:0; all4 audits PASS, matched preparation,168captures,606records/
3030candidate compares/4848replays,21focused cases,1000calls/rank/both orders.
Rows equal-case geometric speedups, kernel/pipeline vs control then vs pull:
EP2(3cases):1.173497/1.158397;0.723660/0.748560.
EP4(6):1.068948/1.059053;0.876646/0.894021.
EP32(6):1.034791/1.027309;1.029711/1.019997.
EP64(6):1.043661/1.034100;1.095612/1.066702.
Combined21:1.065972/1.056199 vs control,0.951834/0.951921 vs pull.
This focused set excludes EP8/16 and is NOT the final91-case acceptance score.
Q3 rows kernel gains at1K/2K(us):EP2 59.730/96.704,EP4 12.034/25.935,
EP32 2.621/4.637,EP64 2.138/5.446. EP64Q352K gain1.241us, every
rank/statistic/bothorders improves; EP32Q352K gain.394us but reverse tails mixed.
Workers-only beats rows in low-EP aggregate: kernel/control1.203796EP2,1.086268EP4;
rows adds7.878/13.822us to EP2Q31K/2K and27.938us to EP4Q352K vs workers-only.
Thus worker count explains much of low-EP gain; preserve both paths, no winner table.
Small unchanged targets retained: pooled rows-control kernel deltas range-.0542
through+.0118us, with rank/tail/order variation. No material small mean regression
observed; final strict small preservation is not established. No default promotion.
Detailed paired latencies and equal-case scores: C/topology_sweep_progress_20260927.json;
original per-rank/order evidence: each retry/selection_summary.json and rank_review/.
No new jobs submitted or optimization resumed; user requested results/report only.

Fixed final coverage saved in C/final_comparison_scope.json:91sub4K cases
(Q3:51/Q35:40),22context4K/8K cases, explicit retained EP8/EP16 small controls.
Equal-case mean log(pull/candidate) with matched all-rank/both-order means, kernel
and producer-to-main reported separately. No losers dropped, no extra-control
reweighting. Historical113-case report supplies IDs only, not matched final proof.
Maintained throughput preparer now accepts EP32/64 with explicit control-total
outside EP8/16; actual EP32/64 focused plans prepared and CPU-submitted above.
Previous turn progress: completed EP16 audit and fixed full comparison coverage.
Current turn progress: diagnosed/recovered CPU review and launched qualified EP4
GPU comparison. No kernel/default change; final geometric result remains unproven.

Geometric rescoring of retained experiments (C/geometric_reassessment.json):
focused six-case kernel speedup vs control/pull: sender_middle1.004487/.958529,
rows_sf_split1.002134/.955415, sender_rotate1.000716/.954838,
metadata_startup_join.983513/.942089. Row partitions ten-case score1.012082/.992418;
its case set/baseline differs, so no cross-campaign ranking by absolute score.
Broader16-case normal joined graph speedups vs control/pull: sender_middle
1.000259/.898113; rows_sf_split1.000526/.898269. Includes reset/preparation;
not interchangeable with kernel/pipeline timestamps. Individual large-case losses
no longer automatically reject these paths, but tiny aggregate improvements and
mixed small controls do not establish promotion. Small deltas/rank evidence retained.
No new GPU run for rescoring; completed shared-wait evidence and rescoring motivated
the retained row-path EP16 follow-up above.
Extended row-path scoring: broader20-case normal aggregate1.010682x vs topology
control (Q3 ten cases1.018121x;Q35 ten cases1.003297x). Historical EP16 four-case
kernel1.041944x vs control/.995671x vs pull, pipeline1.034135/.991945x. Q35 EP16
large was subsequently measured above. Q35 normal gain is mainly3072;1K/2K modest/mixed, small
shared-code controls retain variation. Revised criterion removes isolated large
losses as automatic vetoes. The shared-wait small regression led to the retained row-path EP16 follow-up;
no further queue variant is currently prepared.

Local-ack follow-up source audit: do not remove fetch_expert_recv_count from
all queue readers. Dispatch uses cached counts and pool offsets after its tile
loop for workspace cleanup/stats (get_num_tokens/get_pool_block_offset). Saved
Q351K candidate SASS retains the two dispatch LDS reads45f0/4630 for that use.
Other roles may dead-eliminate unused counts; no new experiment justified here.
Scheduler follow-up (source + saved58 row SASS): five of six fast cases have a
single expert wave (Q3 all16 experts; Q35 32/1K all64); only Q35 2K uses32 of64.
Wave-start caching therefore offers few repeated reductions in this target set.
Saved SASS does retain REDUX.SUM of RZ at scheduler initialization/rewind (e.g.
Q31K6130/64d0, Q351K4450/48a0), so a zero-offset fast path is a possible small
cleanup, not evidence it closes the multi-us pull gap. Do not add a cached-offset
register or a new GPU experiment on the repeated-wave assumption alone.

Last full matrix57/91 wins,34 losses is diagnostic; final acceptance now uses the
user's geometric-mean criterion above, with small-batch preservation required.
Defaults unchanged, no model/token winner rules, no EP16 promotion from these runs.
Keep row-only as the strongest validated Q3 path. Paths R/C and full criteria are
in the handoff section below. Evidence directories C/<campaign> contain summary.json,
selection_review.json and rank_review; local mirrors under scratch/megamoe/throughput_20260922.
Detailed superseded continuation notes are archived there as
continuation_notes_before_compaction.txt; use the current state here to resume.

Readiness polling C/readiness_poll finished: GPU7466010 completed33m54s0:0,
CPU7466013 completed7m58s0:0. PASS100 captures/80 records/560 exact comparisons/
800 replays, matched collective preparation. Source/cache frozen48 arguments.
CPU7465811 passed40 resources/10 geometry/10 default checks;30 cold SASS loops
retain initial/confirming acquire but omit repeated CCTL, same role register maxima.
Both polling kernel/pipeline(us) minuscontrol: Q3 1K-0.445937/-0.455238,
Q3 2K-0.398586/-0.392948, Q35 2K-0.112511/-0.103362. Q3 every rank mean/median/
p95/slowest/both orders improves. Q35 2K every rank mean/both orders improves, but
forward pooled kernel p95+0.032us, per-rank p95 up to+0.096kernel/+0.064pipeline.
L1 alone Q3 1K-0.197924/-0.196008,2K-0.675378/-0.884541,allstats/orders improve;
Q35 2K-0.057506/-0.044191 with mixed rank/tails. L2 alone Q3 1K flips order.
Both-vs-L1 Q3 2K also flips order; do not choose case winners. Small8/32/256 and
Q35 1K identical compiled paths; measured variation retained. Every candidate still
behind pull on all four large cases. Selection review/rank evidence saved; no default,
EP16 or full-matrix promotion. Completed C/poll_rows compares original W16/P1,
W32/P2 rows alone, rows+L1, rows+both, and pull at8/32/256/1K/2K. CPU7466836 passed
5m53s:40resources/10geometry/10default gates plus27 cold SASS loops. Frozen48
readiness_poll source/cache reused after original consumers finished.
GPU7466953 completed32m33s0:0; CPU7466960 completed8m18s0:0. PASS100 captures/
80records/560 exact comparisons/800replays, matched preparation. All9 rank reviews,
selection_summary.json and selection_review.json saved locally/remotely.
Row-control kernel/pipeline(us): Q3 1K-2.081385/-2.083940,2K-5.361744/-5.351270,
Q35 2K-1.599034/-1.593796; every rank/pooled statistic and both orders improves.
L1-minus-rows Q3 1K+0.295140/+0.302060, EVERY statistic/order regresses. Q3 2K
-0.300169/-0.519601 allstats/orders improve but reverse magnitude larger;
Q35 2K-0.154584/-0.154265, forward pipeline max rank p95 unchanged0.
Both-minus-rows Q3 1K paired+0.144569/+0.054750: forward+0.290598/+0.294258
allstats regress; reverse kernel-0.001459 mixed ranks/tails,pipeline-0.184758
allstats improve. Q3 2K-0.299177/-0.519115, forward rank kernel p95 up to+0.096.
Q35 2K-0.096289/-0.097617 with mixed forward tails. Reject polling as a general
row-path addition; preserve row-only. No case winner rule or EP16/holdout promotion.
Small8/32/256 and Q35 1K identical compiled targets; retain measured variation.
All large candidates remain behind pull; rows_l1 gaps Q3 1K4.163/4.266,
Q3 2K4.579/4.203,Q35 1K9.027/9.249,Q35 2K14.204/14.589us. No additive cross-run gains.

Completed prototype C/warp_metadata (frozen49 extension arguments; polling
GPU source remains frozen48). METADATA_WARP_GROUP_LIMIT0/2/4/8, default0. Pick the
largest power-of-two groups<=limit that fit localExperts/physicalTileWorkers; <=1
retains original path. Per expert,8/groups warps own disjoint two-row-aligned row
partitions, each publishing its own four-bit nibble. Consumers acquire all nibbles.
Every warp performs SF then middle, or prefix then tail; per-tile subgroup barriers
become full-warp sync. Original whole-role prefix setup/final retirement joins stay.
No new allocation, barriers, ownership fallback or token threshold. Row-partition/
dedicated/recycle/narrow/coalesced/direct-shared combinations fail explicitly.
Q3 EP8 topology keeps original path; Q35 both small/large will change and need timing.
Compare group limits2/4 at identical physical workers. Independent ownership audit
1584 configurations/3,200,415 rows PASS (including empty partitions, ragged experts,
multiple tiles, E31/33); CPU7466300 completed5m52s0:0: fresh49-argument build,30 resources/10 geometry/
10 default-instruction gates PASS. SASS all5 Q35 cases: named barriers4/5 removed,
barrier3/256 static sites5→3 (setup/retirement retained); consumers use group2
0x1111/2222/4444/8888 and group4 0x11/22/44/88 masks. No new spills.
Four changed headers include dispatch helper; candidate.patch/before/ownership_review
saved. Frozen R/source/warp_metadata_20260926 with fresh native/cache. GPU7466414
completed27m05s0:0; CPU7466440 completed6m59s0:0. PASS80 captures/80 records/
400 exact comparisons/640 replays; matched preparation. Selection review saved.
Groups2 Q35 kernel/pipeline(us) minuscontrol:8 +0.455621/+0.483849,
32 +0.390521/+0.383207,256 +0.389994/+0.375446; ALL rank mean/median/p95,
pooled/slowest and both orders regress.1K -8.459324/-8.469209,2K -0.603442/
-0.604521; every statistic/order improves. Still behind pull1K0.505554/0.813835,
2K15.297888/15.677992us. Groups4 small8 +0.236050/+0.268134,
32 +0.133876/+0.131952,256 +1.001490/+0.984908;1K -5.505076/-5.521392,
2K +1.742491/+1.600139. Groups4 8/256/2K allstats/orders regress;32 mixed ranks.
Q3 is identical compiled code for all arms; measured variation retained.
Reject both as general successor; no EP16/holdout promotion or token cutoff rescue.

Completed C/warp_empty source R/source/warp_empty_20260926, fresh50-argument native/cache.
Knob SKIP_EMPTY_PARTITIONS0/1 default0, requires requested warp groups; actual0
where topology disables grouping. Empty warp-owned slices skip per-tile work;
partition0 includes their readiness nibbles in its existing release. Nonempty
partitions retain independent data/certificate ownership; consumer masks unchanged.
Four headers changed; no new storage/barriers. Full-role setup/retirement joins stay.
Ownership model768 configurations (P1/2/4, ready1..256),98,688 rows PASS: each valid
row and each readiness bit owned once. No attribution of old small regression to
this work yet; explicit comparison control/groups2/groups2_empty/pull will test it.
CPU7467196 completed7m04s0:0;30resources/10geometry/10default-instruction checks PASS.
SASS all5 Q35 cases: empty branch bypasses4 static readiness releases and rejoins
tile advancement; no per-tile CTA barrier in skipped region; partition0 mask
comparisons width/2width/3width, full consumer masks and role joins retained.
Candidate initial/callee/epilogue register maxima: small120/122/100,1K120/120/100,
2K123/124/104; all limits/spills pass. No speedup assumed from static work reduction.
Additional old49-vs-new50 groups2 comparison is NOT instruction-identical: same
instruction counts/resources but mask-constant register assignments/release operands
changed. Default control exact. Save groups2_preservation_review.json; do not ignore
these differences or transfer old group timings. Direct current-allocation comparison
remains valid; frozen49 retained. GPU7467343 submitted13:21:52Z afterok7466953,
CPU7467350 afterok7467343;80 captures(control/groups2/groups2_empty/pull,both models
8/32/256/1K/2K,mirrored1000calls/rank). Completed results follow.
Initial async source copy collided with early patch sync; repaired BEFORE submission
by completing full-source rsync then applying four changed headers sequentially.
Frozen sources49/48 untouched. Candidate.patch/before/ownership_review saved.
Warp_empty GPU7467343 completed28m29s0:0; CPU7467350 completed8m37s0:0.
PASS80 captures/80records/400 exact comparisons/640 replays, matched preparation.
Candidate-minus-control Q35 kernel/pipeline(us):8 +0.125441/+0.139007,
32 +0.175172/+0.171272,256 +0.365888/+0.351254;32/256 EVERY rank/pooled/slowest
statistic and both orders regress.8 means regress both orders, allstats reverse.
1K -8.581890/-8.584613 and2K -1.197628/-1.200915, every statistic/order improves.
Versus current groups2:8 -0.362403/-0.357176 allstats/orders;32 -0.207620/-0.203964,
one reverse kernel rank p95 unchanged;256 +0.013664/+0.006473 mixed ranks/tails.
1K +0.006338/+0.114226 (reverse pipeline+0.224916 allstats regress);2K
-0.281951/-0.275604 allstats/orders.20/40 balanced rows per expert at1K/2K withP4
produce slices6/6/6/2 or10/10/10/10: NO empty slices. Thus large changes cannot be
attributed to dynamically skipped empty work; code/register scheduling and timing
also changed. Still behind pull1K0.128693/0.423673,2K14.975628/15.342641us.
Reject general successor, no small cutoff/EP16/holdout promotion. Conditional row/
group composition deferred: its small-preservation prerequisite failed. Q3 unchanged
compiled targets; measured variation retained. Selection summaries/all5 rank reviews
saved locally/remotely. All gemm_epilogue jobs now terminal; monitor96486 exited0.

Next C/row_pull_phases CPU7467719 FAILED9m55s1:0: all compiles/plan validation and
8 allpush profile-resource selections passed, then reviewer expected per-model pull
manifest while planner compiled both models into preparation/pull/manifest.json
with duplicate case labels. No guessing by case names. CPU-only recovery7467946
submitted14:05:30Z under cpu_review, COMPLETED3m04s0:0; all12 resource gates PASS:
cached pull precompile per model generates
unambiguous manifests, then validates4 pull targets and reuses prior8 results only
when PASS and saved library path matches expected cubin. Missing/mismatched result
fails. Generic prepare_phase_followup.py now splits pull preparation by model for
future plans and supports explicit resource reuse arms. Prior failed record retained;
no GPU rerun or native rebuild. GPU7468052 submitted14:13:43Z afterok7467946,
COMPLETED20m42s0:0.48/48 captures (36ordinary+12profiles),48records/272 exact
comparisons/480replays PASS, matched preparation. CPU analysis7468163 FAILED6m11s1:0: predicted incomplete window locations
GenphaseWindowValidateStart at Q3 rows1K, after ordinary summary PASS
(submitted14:19:05Z).
Job metadata synced. Reuses validated
frozen48 R/source/readiness_poll_20260926 and exact native from readiness_poll cache;
new independent JIT/profile cache. Donor poll_rows is terminal. Plan36 ordinary
(control/rows/pull ×both models256/1K/2K×mirrored1000calls/rank),12 profiles
(control/rows/pull ×both models1K/2K×10calls/rank). Generic prepare_phase_followup.py
now accepts --pull-profile-template; recorded matched_profile legacy pull contract
and split_pipeline_phases allpush contract retained with correct208 core-event
limits. reviewer derives epilogue warp count from actual compiled config, including
pull single-WG shapes; exact marker coverage and ordinary/profile bias mandatory.
Ordinary paired row-control kernel/pipeline(us): Q3 1K-2.066303/-2.072547,
2K-5.121734/-5.111409; both orders improve. Q352K-0.567005/-0.435100 paired
but forward+1.113752/+1.368759, reverse-2.247763/-2.238959. No robust gain claim.
Row-pull large gaps Q3 1K3.933161/4.024921,2K4.910568/4.591009;
Q35 1K9.708775/10.039229,2K14.834161/15.483429. Q3/Q35 256 control/rows
share exact target paths/configs. Kernel differences forward/reverse: Q3+4.003274/
-0.006002,Q35-0.011739/+1.775448. Retain variation; no correction/filtering.
Rank reviews now PASS: Q3 1K/2K every rank mean/median/p95, pooled/slowest and both
orders improve vscontrol. Q352K reverse allstats improve; forward mean regresses
and max rank p95+7.040us kernel/+7.712us pipeline. All large cases lose to pull.
Hypothesis: locate remaining row/control-minus-pull gap in first MMA, first FC2,
final accumulator and exposed completion; overlapping spans are not additive costs.
No new kernel/source edits; no attribution from static register/role inspection.
Marker interpretation checked in both pull/allpush bodies: CombineEnd is emitted
after the local combine/reduction loop, not an isolated remote-store completion
latency. ComputeEnd observes the final FC2 accumulator after its wait returns;
allpush prefetches return metadata before that wait, pull does more offset work
afterward. First-FC2 and exposed-tail envelopes must not be called pure compute
or communication costs. No probe/kernel modification from this inspection.
CPU observer follow-up: maintained profile_megamoe_iket.sbatch now derives active
physical metadata CTAs from tileWorkers=owners/rowPartitions and expands each active
tile worker to its row partitions. Frozen48 decoder incorrectly treats W32/P2 as
32 tile workers, predicting16 instead of32 active Q3 owners. Validate requested P
against compiled unsigned parameter60.650 direct device-loop ownership cases PASS;
all8 allpush compiled marker symbols match P/W; bash syntax PASS. Saved review at
row_pull_phases/workflow/row_partition_decoder_review.json. Submitted source/plan
untouched. If original CPU analysis rejects row profiles, recover saved captures
with the maintained decoder in a separate CPU phase; no GPU repeat. Prepared
row_pull_phases/analysis_recovery CPU7468483 completed7m20s0:0:17 commands, all12 saved profile
decodes using copied maintained decoder, phase summaries, two rank reviews and core
coverage review. Requires existing ordinary summary PASS; original job/logs retained.
Recovered core review PASS960 calls/72,000 CTA pairs; all four marker sets exact.
Geometry: Q3 1K all arms M96/Ewave16/stages8/epi256; Q352K all M96/Ewave32/8/256.
Q32K allpush M128/stages7 vs pull M192/stages6. Q351K allpush M32/Ewave64/10/epi256
vs pull M64/Ewave32/9/epi128, so28us first-FC2 difference is geometry-confounded.
Rows-pull pair-mean deltas firstMMA/firstFC2/finalAccumulator/combineEnd(us):
Q31K -0.008/3.115/2.972/3.887; Q352K1.507/6.440/9.277/11.301.
Rank-local latest-accumulator deltas4.558/14.680us respectively. Much delay precedes
final accumulator; neither marker is a pure stall/cost. Bias(profile-ordinary):
Q3 rows1K-1.061,2K+0.609 vs pull-1.158/-0.623; Q35 rows1K-3.782,2K-5.814
vs pull-1.184/-4.333. No correction/promotion. interpretation_review.json saved.
CPU-only7468663 completed1m38s0:0 under cpu_handoff: generic core reviewer
--dispatch-handoff validates exact MegaDispatchEnd CTA/warp set against compiled
metadata/sender counts, then measures worker return relative to first MMA/FC2.
Separate core_handoff_review output, all saved calls retained; no new probe/GPU.
Hypothesis: borrowed epilogue warps may still handle transport when MMA starts.
Overlap is not proof of TMEM backpressure; inspect this before a role redesign.
Handoff exact coverage PASS. Rows SF/middle/prefix-tail/sender mean end-minus-first
MMA(us): Q31K2.490/3.320/3.869; Q32K7.855/9.751/10.113; Q351K9.582/20.094/1.875;
Q352K14.015/17.291/11.964. SF/middle here is one combined worker category, not two
measurements. Helpers finish before MMA. No isolated-stall attribution.

New C/sf_return / R/source/sf_return_20260926 (51 extension arguments), fresh
native/JIT cache. DG_MEGAMOE_ALL_PUSH_SF_EARLY_RETURN0/1 default0, no token cutoff.
After all SF tiles, its128 threads barrier.cta.arrive6/256 and return independently;
middle128 barrier.cta.sync6/256 then thread128 publishes the original DispatchDone
once. Per-tile readiness, initial barrier3 and other roles unchanged. Barrier6 is
not used by epilogue2/3/4; SF finishes barrier4 before arrival6, so middle cannot
reuse4 before SF retires. Dispatch scratch remains disjoint by existing staticassert.
Reject recycle/dedicated/warp-groups/narrow-SF/direct-shared/coalesced-SF combinations.
MegaDispatchEnd/GenphaseEnd move to middle warp12 on SF CTAs in this mode, retaining
whole-worker timing semantics. Four headers, before snapshot/candidate.patch saved.
PTX nonaligned arrive/sync semantics checked in official CUDA13.2 ISA section9.7.13.1:
https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar
Count/order model40,320 full-warp permutations PASS; this is not compiled barrier
or memory-order proof. SASS/actual resources and full GPU replay remain mandatory.
CPU7468835 completed6m57s0:0: fresh51 build,40resource/10geometry/10default gates
PASS.20 candidate SASS cases each have BAR.ARV6/256 and BAR.SYNC6/256, thread127
split and thread128 publisher guard. Early branch bypasses wait/publication into
existing cluster wait (host cluster2, all arrivals before dispatch). Initial/callee/
epilogue maxima123/125/120, within128/128/208; no new spills. SASS review saved.
GPU7468988 submitted15:19:37Z, COMPLETED34m44s0:0,100/100 captures;
stderr empty. CPU7468996 COMPLETED10m34s0:0
(submitted15:19:52Z afterok). Full audit/selection results below. First sbatch
rejected stale completed-CPU dependency; no GPU job existed, verified CPU0:0/empty
queue then removed dependency. Rejection log/command retained. Source/cache/plan
frozen.100 captures planned:control/sf_return/rows/rows_sf_return/pull, both
models8/32/256/1K/2K mirrored1000calls/rank. Generic prepare_pipeline.py now tracks
actual SF-return config including small shapes; equal configs require target identity.
SF-return audit/selection files PASS100 captures/80records/560 exact comparisons/
800replays, matched collective preparation. All9 rank reviews saved. Added to rows,
kernel/pipeline(us) deltas: Q31K+0.018614/+0.022643, Q32K+0.149803/+0.156844,
Q351K+0.097448/+0.098685,Q352K+0.078846/+0.088266. Every large mean regresses
both orders; Q32K pipeline every rank/pooled statistic regresses in both orders.
Q35256+0.059481/+0.044565 (both means/orders regress; forward kernel allstats).
Reject general SF-return promotion, no cutoff/EP16 rescue. Alone vscontrol Q31K
+0.080303/+0.086667,Q351K+0.261978/+0.180296;2K means flip by order both models.
Row-only control gains recur: Q31K-2.110776/-2.117244,Q32K-5.385197/-5.373306,
Q352K-1.663049/-1.658097, all rank/pooled stats and both orders improve.
Rows+SF remains behind pull at1K/2K: Q3 gaps3.857837/4.969763us kernel,
Q35 gaps8.981873/14.564329us. Small Q35 8 gain does not rescue large regressions.
Selection_review.json records rejection; all measurements retained. New width test
below keeps SF interaction isolated because it removes the first-WG dependency on
middle's TMEM participation; it does not promote this failed standalone change.
Bound on interpretation: FC1 retains full epilogue barrier2 before its FC2-ready
publication, and TMEM-empty arrivals still include both epilogue warpgroups. Early
SF return can hide only work before these joins; dispatch overlap is not a speedup
budget. Deferred source-only idea if this mechanism stalls there: separate temporary
metadata launch width from compute epilogue width (allpush currently mandates256
epilogue threads for metadata, while pull Q351K uses128). Extra metadata WG would
retire after transport; active epilogue count/empty-barrier arrivals would match actual
compute participants. Requires lifecycle, register, staging and geometry proof plus
an isolated comparison; not implemented, no case/token winner policy proposed.
Further source constraint: original pull M64 pairs STORE_M32 with one epilogue WG;
allpush keeps two WGs and smaller tile/store geometry. Launch bounds currently equal
dispatch+compute+epilogue threads, while the inband combine vector-width assertions
also depend on epilogue count. A temporary extra metadata WG therefore requires
explicit launch/staging/combine lifecycle changes, not only a heuristic thread knob.
Reference check: cutedsl Blackwell next/sources/kernel_src/blackwell/inference/mega/
block_scaled_swap_ab_fc12_kernel.py fixes eight warps (four epilogue, MMA/TMAA/TMAB/
scheduler) but sets communication_enabled=False. Its8-warp result is not evidence
that a distributed allpush kernel can simply halve threads. token_comm.py retains
a transfer-lifetime join when reusing transfer storage for token-back.
The integrated Blackwell block_scaled_swap_ab_mega_moe_kernel.py DOES enable
communication: epi warps0..3, compute/scheduler4..7, token-in8..11, optional standalone
token-back12..15 (384/512 launch threads). Its input dispatch is explicitly pull.
Separate4-warp epilogue and transfer lifetime/finalize barriers support investigating
role separation, not an allpush throughput claim. Epilogue register budget256 differs
from FastAFD208; transferring its topology is not an isolated synchronization change.
Source-only alternative (unimplemented): FC1 currently joins all epilogue threads
before one readiness OR per N tile. Per-WG readiness would need distinct bits and
both bits acquired by FC2, plus a proof that the64-bit mask fits every supported
intermediate width. It doubles release operations. FC1 two-stage FP8 shared layout
and FC2 single-stage BF16 layout alias across WG regions, so phase-transition joins
must remain in BOTH directions; TMEM deallocation also requires all consumers.
Thus deleting the existing full join is unsafe even with independent publications.
The per-WG readiness alternative remains deferred. A smaller fixed-launch role test
was subsequently prototyped below; it keeps combine width and allocation unchanged.


New prototype C/gemm_epilogue, R/source/gemm_epilogue_20260926: fresh52-argument
native/cache, three headers changed from51 (before/patch/ownership review saved).
SINGLE_GEMM_EPILOGUE0/1 default0 applies all shapes, no cutoff. Only first128
of256 epilogue threads execute GEMM scheduling; WG_BLOCK_M covers the full tile.
MMA geometry, launch512, register budgets, allocated staging/stages, dispatch and
combine256 remain unchanged. TMEM-empty expects2*128 cluster arrivals, GEMM
barrier2 uses128, unused WG parks at separate barrier7/256 and rejoins after TMEM
free before cleanup/combine reuse shared storage. Existing WG barrier3 and SF
handoff6 do not alias7. Combine consumes no skipped GEMM scheduler/iteration state;
original epilogue indices and staging strides remain valid. SF+single lets middle dispatch finish independently of
active GEMM; other metadata workers retain their original joins. No GPU proof yet.
Reject recycle/nonzero reset and dedicated dispatch. Row-ownership model9028
configurations (M32..256, stores8/16/32, one/two WGs, both row orders, partial
prefixes) PASS; this is not device-memory proof. No instruction/reg preservation
claim until CPU resource/default/SASS gates. Generic preparer tracks actual new
flag even small shapes, preserving exact-target equality when configs are equal.
Planned100 captures: control/rows/rows_single/rows_single_sf/pull, both modelsEP8
8/32/256/1K/2K, mirrored1000calls/rank. This isolates width against useful row-only
and SF-return interaction. CPU7469581 COMPLETED7m33s0:0:40resource/10geometry/
10default-instruction gates PASS. Initial/callee/epilogue maxima123/125/155, within
128/128/208; no new spills.20candidate SASS checks PASS: thread127 split bypasses
GEMM into retirement, two128-thread GEMM joins precede one256-thread barrier7,
combine uses256 afterward; two TMEM-empty barriers initialized for256 cluster
arrivals. gemm_role_sass_review.json saved; no dynamic claim. GPU7469784 submitted
16:07:46Z, COMPLETED36m14s0:0,100/100 captures; CPU7469786 COMPLETED9m30s0:0.
Audit PASS100captures/80records/560 exact comparisons/800replays, matched preparation;
all9 rank reviews/selection saved. Monitor96486 exited0, no active jobs.
Startup Slurm prolog-hung warning retained; job proceeded, completed without restart.
Width-minus-rows kernel/pipeline(us): Q31K+7.337239/+7.289714,
Q32K+17.558542/+17.899612,Q351K+1.455662/+1.562742. Every rank mean/median/p95,
pooled/slowest statistic regresses both orders. Q38+.837603/+.905695,
Q332+.112596/+.119255,Q35256+.018597/+.047889 (mixed statistics).
Q352K paired-1.747086/-1.933138 but forward+1.744845/+1.749250 allstats regress,
reverse-5.239017/-5.615526 allstats improve. Actual kernel means forward/reverse:
rows127.074707/134.078066, single128.819552/128.839049, pull113.786530/113.825300.
Baseline drift/order dependence precludes robust gain. All samples retained, no correction.
Rows+single+SF minus rows: Q31K+7.553867/+7.419279,Q32K+18.122330/+18.474975,
Q351K+1.574735/+1.659409 allstats/orders regress; Q352K also order flip.
Adding SF to single regresses all large paired means; Q32K+.563788/+.575364
allstats/orders regress. Width-alone large pull gaps Q31K11.218/11.391,
Q32K22.186/22.333,Q351K10.729/11.137,Q352K15.023/15.408us.
Reject both candidates, no cutoff/default/EP16/holdout promotion. This fixed-width
failure does not prove a different launch/register topology would fail, but provides
no support for such promotion. Next inspect FC1 independent WG readiness, retaining
phase-transition storage joins in both directions and final TMEM lifetime join.
Follow-up workflow review: prepare_row_partition_followup.py hardcodes old43 source/
native donor and is not safe to invoke unchanged for52. Generic prepare_pipeline.py
handles EP8 nearby totals; prepare_normal_followup.py derives exact source/native
from completed ordinary campaign. Do not submit holdouts before current timing gates.
Frozen source/cache/plan52 and prior51 remain untouched. Normal follow-up resource generator now replaces the single total-loop structurally
and asserts one match; current spaced lists previously escaped the legacy exact-string
substitution. Focused current/legacy formatting check PASS. No holdout submitted.
Interleaved calibration helper is AA/AB/BB on an existing validated worker-policy
plan, not a general new-flag preparer: it asserts only EXPERT_CTAS variant keys.
A qualifying new flag needs an explicit source/cache/settings-aware interleaved plan;
do not invoke that old calibration unchanged. Future generic profile handoff reviewer allows middle
retirement after firstFC2 only for the explicit SF+single combination; whole-worker
coverage remains mandatory, combine still expects all8 epilogue warps.

New prototype C/fc1_wg_ready, frozen R/source/fc1_wg_ready_20260926, fresh53-argument
native/cache. Three headers changed from52; default-off DG_MEGAMOE_ALL_PUSH_FC1_WG_READY.
Keep two GEMM WGs, launch512, staging geometry, registers, dispatch and combine fixed.
Each WG completes TMA, joins its own128 threads and releases bit2*n+wg, including
empty WG row slices. FC2 waits both bits for every tile; optional stage-ready masks
also expand. No new allocation. Existing64-bit mask bounds intermediate<=2048;
unsupported width/single-WG/dedicated/recycle requests fail, no fallback/cutoff.
Full join before FC1-to-FC2 protects cross-WG FP8/BF16 shared aliases. FC2's full
per-tile join (or existing return-WG transition join) protects opposite direction.
Explicit final full join protects TMEM retirement even for a CTA ending in FC1.
Mask model1088 checks PASS including64-bit boundary, not CUDA proof. Official PTX
observation-order and memory-synchronization rules reviewed: release OR chain to
final acquire, WG barrier orders each publisher's SF data; async fence unchanged.
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#observation-order
Generic preparer tracks new config for all shapes and exact-target equality.
Plan80captures: control/rows/rows_wg_ready/pull, bothmodels8/32/256/1K/2K,
mirrored1000calls/rank. CPU7470609 completed6m46s0:0:30resources/10geometry/10default instruction gates PASS.
Initial/callee/epilogue max registers123/125/120, budgets128/128/208; no new spills.
All10 SASS checks PASS: TMA completion, WG128 publication join, doubled bit index,
predicated phase join, FC2 full join and final full join before TMEM free.
GPU7470764 submitted17:17:40Z; CPU7470765 afterok. Source/cache/plan53 frozen.
GPU7470764 COMPLETED28m17s0:0, CPU7470765 COMPLETED6m55s0:0. Monitor90042 exited0.
Audit PASS80captures/80records/400 exact comparisons/640replays, matched preparation;
all5 rank reviews saved. WG-minus-rows kernel/pipeline(us): Q31K+.492279/+.490828,
Q32K+1.248630/+.907356,Q352K+.856739/+.861608. Every rank/pooled statistic and both
orders regresses in all three kernels and Q31K/Q352K pipelines; Q32K reverse pipeline
mean+.223722 has mixed tails (forward+1.590989 allstats regress).
Small Q332+.114777/+.113703 (both means/orders regress, reverse every statistic);
Q3256+.126879/+.105585 (both means/orders regress, pipeline every statistic bothorders).
Q38+.046946/+.049870 with pipeline order flip. Q351K-.013106/-.009793 paired,
but kernel forward-.124058/reverse+.097846 and pipeline-.120163/+.100576.
Reject general successor, no cutoff/holdout/EP16 promotion. Keep full FC1 join.
Row-control gains recur in this allocation: Q31K-2.198827/-2.206309,
Q32K-4.948770/-4.588332,Q352K-2.213920/-2.078690, allstats/orders improve.
WG candidate remains behind pull at1K/2K: Q3 kernel4.327239/5.682096,
Q35 kernel8.601686/13.969045us, allstats/orders. All samples retained; do not add
cross-run gains. selection_review.json records rejection locally/remotely.

Independent candidate C/fc1_async / R/source/fc1_async_20260926, fresh54-argument
native/cache, default-off DG_MEGAMOE_ALL_PUSH_FC1_ASYNC_READY. Three headers from53.
Replace FC1 red.release.gpu.global.or.b64 with red.async.release.gpu.global.add.u64.
Each scheduler-owned tile/WG publishes exactly one disjoint bit, so sum equals OR;
not safe under duplicate publication. Retain TMA store completion, all current joins,
mask width, consumers and cleanup. No new allocation or token cutoff. Reject recycle,
dedicated and warmup scheduling until separately proven. Original single-WG/FC1 WG
options are off in this comparison, isolating publication on the useful row path.
Existing SASS motivates test: synchronous release emits MEMBAR.ALL.CTA/GPU+REDG.
Official CUDA13.2 PTX9.2 section9.7.13.7 supports release-add u64 global onsm100;
completion is ordered with CTA end, and release remains a strong GPU operation.
https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async
Cutedsl helpers/flag_batch.py GpuAsyncReleaseFlagBatchTracker uses async add32 after
store drain/join; this candidate applies add64 to unique bits, without deferred batching.
Ownership model168 configurations/151,200 unique publications PASS: empty/skewed
experts, wave sizes, regular150-CTA strides with rotation, both bit widths up to64.
FC2 observes all releases before consumption/cleanup, so no async add can remain
pending when that generation's mask is reset. No correctness/performance claim yet.
CPU7470822 completed6m31s0:0;30resources/10geometry/10default gates PASS.
Initial/callee/epilogue maxima123/125/120, no new spills. All10 publication SASS checks
PASS: native UREDGR.E.ADD.NA.64.STRONG.GPU replaces OR64, removes MEMBAR.ALL.GPU/
ERRBAR/CGAERRBAR; MEMBAR.ALL.CTA, TMA completion and full epilogue join retained.
All10 row-only instructions match53 except verified assertion line metadata.
CPU monitor67914 exited0. Frozen53 fully audited/rejected. Plan80
captures control/rows/rows_async/pull, bothmodels8/32/256/1K/2K mirrored1000/rank.
GPU7470984 submitted17:36:43Z afterok7470764, CPU7470987 afterok7470984.
GPU7470984 completed27m59s0:0; CPU7470987 completed8m48s0:0. PASS80captures/
80records/400exact comparisons/640replays, matched collective preparation. All5 rank
reviews and selection_review.json saved locally/remotely; monitor69915 terminal0.
Rows_async-minus-rows kernel/pipeline(us): Q31K-0.593176/-0.594000,
Q32K-0.215597/-0.202204,Q352K-1.317444/-1.188537; EVERY rank/pooled statistic and
both orders improves. Q38 also-0.232636/-0.239788 allstats/orders. But Q332 paired
+1.532671/+1.655505: forward+2.974098/+3.218270 allstats regress, reverse
+0.091243/+0.092739 allmeans regress (pipeline allstats). Raw kernel forward/reverse
rows34.181710/34.180056 vs async37.155809/34.271300. Keep all observations.
Q351K+3.905662/+3.898232 paired, order flip: forward-0.066234/-0.256552,
reverse+7.877559/+8.053016 allstats regress. Raw kernel rows102.706582/102.620571,
async102.640348/110.498130. No attribution/correction for this candidate order drift.
Q35small8/32/256 kernel+0.022970/+0.011513/+0.024819 both means/orders, mixed
rank/tails; Q3256-0.020104/-0.016858 mixed ranks/tails. Reject general async promotion,
no cutoff rescue/normal/interleaved/EP16. Row-control Q32K-5.470212/-5.453405 and
Q352K-2.379775/-2.505104 allstats/orders. Q31K paired-4.504618/-4.680280, bothorders
improve but control drift: forward-2.413481/-2.588921, reverse-6.595754/-6.771638.
Rows_async still behind pull kernel/pipeline Q31K3.118953/3.203005,
Q32K4.191423/4.185623,Q351K12.511643/12.824523,Q352K11.850096/12.245554.
These are current-run measurements, not additive cross-run gains. Frozen54 retained.

Independent preparation C/metadata_coop / R/source/metadata_coop_20260926,
fresh55-argument native/cache. Four-header default-off
DG_MEGAMOE_ALL_PUSH_METADATA_COOPERATIVE_GROUPS=0/2; effective2 only when two
expert groups fit E/(physicalWorkers/rowPartitions), otherwise unchanged0. No token
cutoff/model rule. Unlike rejected one-warp groups, each128-thread expert group
retains64 SF +64 middle in parallel; prefix/tail uses its128 threads. Physical row
certificates and16-row partition alignment, consumer masks, full256 setup/retirement,
register budgets, MMA/staging and compute ownership stay fixed. Old warp-group,
SF-early-return, dedicated, direct-shared, narrow/coalesced, warmup and reset/recycle
combinations reject. No empty-slice elision or async54 composition in this comparison.
Subgroup SF/middle barrier IDs4/5 and6/7 count64; prefix4/6 count128 on separate CTAs.
Uses barrier.cta.sync without aligned because groups run conditional independent
loops (official CUDA13.2 PTX9.2 section9.7.13.1). Existing full-role join3 retires
all subgroups before epilogue3/4/7 reuse; concurrent dispatch uses0/1, compute uses
separate mbarrier objects. Static ownership/participant model385 configurations,
820,554 rows/70,140 certificates PASS, including empty/ragged experts andP1/2/4/8;
not compiled/device correctness proof. Before snapshot, patch and ownership review
saved local/remote. Frozen54 source/cache/plan untouched; copied fully before edits.
Generic prepare_pipeline.py verifies actual cooperative topology at all shapes and
exact targets when configs match. Plan80 captures control/rows/rows_coop/pull across
bothmodels8/32/256/1K/2K and mirrored1000/rank. CPU7471372 FAILED8m02s1:0: compiles and plan validation PASS, resource reviewer
rejects Q352K rows_coop LDL in dispatch48. SASS shows lane/shared address stored at
03e0 to[R1+0x18], reloaded in dispatch/load/compute/epilogue roles. Do not relax gate.
Monitor95442 terminal0. Independent metadata_coop_late source/cache (same55 ABI)
adds cooperative groups to existing producer/scheduler late-count-lane recomputation,
preserving flags/geometry/register budgets and ownership. Original failed source,
cache and resource_failure_review.json retained. Full source copy finished before
final header re-sync and preparation; an early overlapping sync was superseded before
any submission. CPU7471568 FAILED7m49s1:0 after fresh build and all30 resource
checks PASS: initial/callee/epilogue maxima125/125/121, no new spills. Failure is
reviewer-only: small target identity exemption omitted expected_coop. Generic preparer
fixed; same-config exact-target equality remains required. Separate CPU7471737
completed1m03s0:0, submitted18:38:26Z:30resources/10geometry/10default instructions
PASS on saved binaries, no rebuild/GPU repeat. Monitor62717 terminal0. All10 row-only
instruction comparisons also match frozen54 except verified assertion-line metadata.
All5 active Q35 SASS cases have two64-thread SF/middle sites,two128 prefix/tail sites,
three256 setup/retirement joins; final full join after every subgroup site. Barrier
IDs derive from normalized TID>>7,doubled plus4/5; no LDL remains. Initial/callee/
epilogue maxima125/125/121 within128/128/208; three preexisting assertion STLs only.
Resource/geometry/default/row/subgroup evidence saved local/remote. GPU7471807
submitted18:46:04Z, analysis7471849 submitted18:46:30Z afterok7471807.
GPU7471807 COMPLETED28m45s0:0 on nvl72082-T[02,08]. All80captures/640rank
reports verified,stderr0. At19:17Z analysis7471849 RUNNING2m on cpu-0004, logs empty.
Analysis7471849 completed11m52s0:0; monitor7578 terminal0. PASS80captures/
80records/400exact comparisons/640replays, matched collective preparation, all5
rank reviews and selection summary. REJECT_GENERAL_METADATA_COOPERATIVE_PROMOTION.
Rows_coop-minus-rows kernel/pipeline(us): Q358 +0.509098/+0.519013,
Q3532 +0.356061/+0.353987,Q35256 +0.398010/+0.381582,Q352K
+1.111460/+1.105293, every rank/pooled statistic in BOTH orders regresses.
Q351K -8.707631/-8.731444 every statistic/order improves, but still behind pull
0.564093/0.858368. Q3 actual cooperative groups0 at every shape, identical compiled
targets to rows; preserve observed variation (not a mechanism gain), including2K
order flips. Row-control Q31K -2.289122/-2.294132,Q32K -5.663155/-6.099692,
Q352K -2.439790/-2.428809 allstatistics/orders improve. Coop-pull other large gaps
kernel/pipeline Q31K3.806330/3.933116,Q32K4.602468/4.335831,Q352K13.676726/
14.084225; all statistics/orders lose. No cutoff rescue, normal/interleaved/EP16
or default promotion. Frozen55 retained; selection_review.json saved. Plan80captures,
control/rows/rows_coop/pull ×bothmodels8/32/256/1K/2K×mirrored1000calls/rank.
No performance/default/EP16 promotion. Source/cache/plan frozen55. Profile marker decoder requires a subgroup
ownership review if phase profiling is later needed; whole-worker retirement stays.

C/review_selection.py now summarizes complete all-pair rank reviews into
selection_summary.json: paired deltas, per-order pooled statistics, per-rank extrema,
all-statistic signs and order flips, with all samples retained. Requires PASS/matched
preparation, full case/rank/pair coverage and agreement with summary paired means.
Validated on completed warp_metadata5pairs×10cases; known small regressions/large
gains retained. Future prepare_pipeline.py plans freeze/call it after rank reviews;
active plans were not rewritten. Current analyses can be summarized locally after
terminal completion and artifact fetch. Detailed rank_review files remain authority.

Source-only dispatch/register follow-up: ordinary metadata borrows epilogue threads
256..511 before their compute-role loop; dedicated transport moves to dispatch
threads0..127 and changes budgets48/40/208→128/40/168 plus metadata256→128.
This creates a possible scheduling coupling, not proof of the measured large gap.
Recovered C/dedicated_registers summary7373771/7373772 actually covers BOTH models
256/8192 (old jobs.json label QWEN3_ONLY is stale),32records/96comparisons/192replays.
At8K dedicated-control kernel Q3 -2.881435us,Q35 +6.503063us; small256 targets
unchanged. Earlier48-register dedicated7373348 failed resource expectations and
regressed massively. These are not isolated1K/2K register-budget experiments.
Later dispatch_role_pre4k already rejects its transport/thread/register bundle;
do not infer a register-only winner or repeat the same bundle. Complete representative
phase profiling remains preferable if current composition/elision results leave the
critical path unexplained. Frozen active sources/caches not changed by this review.

Source-only future idea, no implementation/plan: ordinary metadata runs before
final role register redistribution (initial128; later dispatch48/non-epilogue40/
epilogue208). Moving the SAME48/40/208 assignments before metadata might shorten
live ranges or change metadata scheduling without the rejected dedicated mode's
128/40/168 budget and128-thread transport bundle. Requires proving every128-thread
warpgroup releases before epilogue allocation, preserving initialization/cluster
progress, removing duplicate later changes and checking actual role resources.
Current coop's spill-free repair does not establish a benefit from more registers.
Cutedsl Blackwell mega assigns budgets at its role branches (scheduler/token-in/
epilogue); it is a pull-input design, not evidence for this all-push change. Defer
selection until current cooperative metadata timing is known.

Follow-up source review: the reachable-register checker follows both sides of
conditional branches, without warp-role predicates. Early allocations followed by
common metadata code therefore need a role-aware proof; do not relax its limits to
accept unreachable high-register paths. The generated resource gate also assumes
one TRY_ALLOC and pre-allocation callee budget128. Initial pool65536 can fund final
64512 (WG0/1 release21504, two epilogue WGs need20480), but arithmetic alone proves
neither progress nor register liveness. CUDA13.2 PTX9.2 section9.7.19.5 requires an
explicit warpgroup synchronization between successive setmaxnreg instructions;
its .sync qualifier synchronizes only a warp. Thus retaining late assignments after
new early ones is not justified without checking intervening group synchronization.
Official reference: https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg
No kernel changes, prototype56, CPU preparation or GPU submission from this idea.

Saved-profile critical-role review (no GPU run):
C/row_pull_phases/core_phase_review/critical_role_review.json covers all960 calls/
72,000 pairs. For each rank/call and marker, assign its latest timestamp to the
compiled transport role; split ties equally, retain mixed-role CTA pairs. Rows Q35
1K/2K final-accumulator latest is prefix/tail in100% of80 calls each; first FC2 latest
is prefix/tail100%/97.5%. These roles occupy8/16 of75 pairs respectively. Prefix/tail
mean final-accumulator is7.583/7.432us later than SF/middle on the same cases.
Q3 1K/2K final-accumulator latest is split SF/middle44.375%/55%, prefix/tail
38.125%/17.5%, plus sender/mixed/compute roles. Latest combine often shifts to
compute-only roles, consistent with its different synchronization boundary, not
proof those roles caused the delay. Descriptive placement only: preserve known
profile bias and unequal cross-arm geometry; no causal stall or speed claim.
This strengthens evaluating current cooperative metadata before early register
assignment; it does not justify reviving rejected transport/topology bundles.

Source-only follow-up if cooperative timing fails: both focused models use H4096.
Existing first1024/middle-end2048 assigns1024B/row to middle128 threads,3072B to
prefix/tail256 threads; cooperative groups preserve this ratio at64/128 threads.
Existing endpoint2560 would move512B from prefix/tail to middle (1536/2560B),
while1536 moves512B the opposite way and releases middle readiness earlier.
These isolate byte balance versus readiness timing on the current row path; byte
counts alone do not predict the kernel critical path. Historical S26/window tests
used8K and different dedicated/narrow/other bundles (Q3 endpoint3072/Q35 endpoint
1536), EP16 follow-up4K/8K; do not transfer their model-specific selection here.
No current row-partitioned1K/2K endpoint comparison found in the recorded memory.
The existing heuristic fixes endpoint2048 for M32, including Q351K, so that case
would remain an unchanged-code control, not a claimed gain. No new kernel option
was needed. After coop rejection, C/metadata_windows prepares
control/rows/rows_early1536/rows_balanced2560/pull,100 captures across both models
8/32/256/1K/2K mirrored1000/rank. Reuses unchanged frozen55 source
R/source/metadata_coop_late_20260926 with fresh native/JIT cache; no source edits.
All rejected role/group/readiness modesOFF. Generic prepare_pipeline.py now
verifies actual middle endpoint from the existing M32 heuristic and includes it in
strict same-config target equality. All other geometry remains fixed. CPU7472484
submitted19:30:24Z and COMPLETED8m14s0:0:40resources/10geometry/10default checks
PASS. Max initial/caller/callee/epilogue123/39/125/120, no new spills. Actual endpoints
1536/2560 only Q31K/2K and Q352K; all small and Q351K retain2048/exact targets.
GPU7472668 submitted19:41:19Z, CPU7472678 afterok submitted19:41:39Z. Confirmed
GPU COMPLETED35m35s0:0 on nvl72151-T[13-14]; all100captures/800rank reports verified,
stderr0. Analysis7472678 produced PASS100captures/80records/560 exact comparisons/
800replays with matched preparation and all9 rank reviews; COMPLETED10m57s0:0.
Selection review saved: reject general window promotion.
Endpoint1536 minusrows kernel/pipeline(us): Q31K+0.208322/+0.313582,
Q32K+0.740770/+0.731348,Q352K+1.124629/+1.125271. All means regress both
orders; Q32K/Q352K every rank/pooled statistic regresses. Endpoint2560 Q31K
+0.338569/+0.446748, every statistic/order regresses; Q352K-1.531151/-1.532870,
every statistic/order improves. Q32K paired+0.145748/-0.075733 flips order:
forward+0.416060/+0.423515, reverse-0.124564/-0.574980. All small and Q351K
retain identical compiled targets; measured variation retained without attribution.
Rows alone versuscontrol saves Q31K2.223345/2.338337,Q32K5.362513/5.351915,
Q352K2.314628/2.307205, every statistic/order. Balanced remains behind pull on all
large cases: Q31K4.075131/4.230769,Q32K5.098184/4.773446,Q351K9.355900/
9.779252,Q352K11.471188/11.867453us, every statistic/order. No case-specific
endpoint selection, default/holdout/EP16 promotion. Monitors2775/79041/7578 terminal0.

Independent register-placement prototype56: C/metadata_after_registers,
R/source/metadata_after_registers_20260926, default-off
DG_MEGAMOE_ALL_PUSH_METADATA_AFTER_REGISTERS=0/1, applied to all shapes. The same
transport block is an inline lambda: default calls it before cluster_wait as before;
candidate calls it inside the existing epilogue208 register branch. No added/moved
setmaxnreg operations, budgets48/40/208, participants, certificates, barriers and
scratch unchanged. It trades initialization overlap for transport under the final
allocation; unlike earlier early-allocation idea, no repeated-allocation sync or
role-predicate exception is needed. Reject dedicated/direct-shared/reset/warmup.
Source proof: scheduler constructor only initializes references/block_idx; common
count publication acquires producer route readiness independently of transport.
Each non-epilogue role deallocates before its data waits, so epilogue allocation
does not depend on its own transport. Metadata scratch begins after local expert
counts and ends before GEMM storage; transport full-role retirement precedes that
CTA's epilogue/combine reuse. This is not compiled/device proof.
Frozen55 copy completed BEFORE final three-header sync. Current local headers56;
active windows55 source/cache untouched. Fresh56 native/JIT cache required.
CPU7472628 submitted19:39:05Z, FAILED6m38s1:0 after all compilation/plan checks.
Reviewer-only failure at first candidate: TRY_ALLOC retry head is NOP rather than
the check's assumed WARPSYNC.ALL. Both are now accepted only unpredicated, retaining
exact unconditional TRY_ALLOC/!UP0 backward retry and CFG proof. Real saved Q3256
candidate passes dominance (two calls, no unresolved branches); frozen55 Q352K rows
still correctly fails (two reachable pre-allocation calls). Review validation/failure
JSONs saved. No allocation/budget or kernel relaxation. Separate CPU7472873 submitted
19:51:19Z on saved binaries, no recompilation/GPU repeat. Recovery copies corrected
reviewer locally to its phase; frozen56 source retains original failed reviewer.
Recovery7472873 COMPLETED1m05s0:0:30resources/10geometry/10default instruction
checks and10allocation proofs PASS. All20 validator callsites dominated by208
allocation, no unresolved transfers or new spills. Candidate maxima initial25,
metadata137,epilogue136; low roles checked against unchanged48/40 budgets.
All10 row-only normalized instruction comparisons also match frozen55 except the
same verified assertion-line metadata. Evidence saved local/remote.
Monitors2335/87772/2775 terminal0. GPU7473067 submitted
19:58:22Z afterok7472668, analysis7473072 submitted19:58:50Z afterok7473067.
GPU7473067 COMPLETED28m04s0:0 on nvl72160-T[04,08]. Capture check20:45:56Z
verified80/80 captures,640rank reports,stderr0. Analysis7473072 COMPLETED8m26s0:0
on cpu-0012; PASS80records/400exact comparisons/640replays, matched preparation
and all5 rank reviews. Terminal state confirmed by final sacct after summary completion.
Reject metadata-after-registers: ALL10 cases regress EVERY rank/pooled statistic,
both orders and both kernel/pipeline metrics versus rows. Kernel/pipeline(us):
Q3 small8+2.834203/+2.843051,32+2.358536/+2.385792,256+3.230599/+3.228075;
Q35 small8+2.060735/+2.059548,32+2.008307/+2.012783,256+2.312015/+2.288399.
Large Q31K+2.963756/+2.946182,Q32K+2.624909/+2.644459,Q351K+4.324864/
+4.281693,Q352K+3.754253/+4.000521. Same-order pull gaps remain6.622/6.725,
7.035/6.787,14.112/14.368,15.179/15.593us respectively. No cutoff/default/EP16
promotion. This tests the placement tradeoff; it does not isolate allocation stall
from lost startup overlap. Selection review saved; no new profiling run.
Rows-control Q31K-2.247483/-2.258847,Q32K-5.502008/-5.708354,Q352K-2.062111/
-2.302343, every statistic/order improves. Q351K rows/control code unchanged;
retain timing variation without attributing a mechanism gain or loss.
Monitor56606 observed this pair at300s; both jobs now terminal.80 captures control/rows/rows_after_regs/
pull both models8/32/256/1K/2K mirrored1000/rank. Kernels/caches/plans frozen;
run serially after window GPU, no performance/default/EP16 promotion.
Strict CPU gates:30resources/10geometry/10default checks, no new spills; metadata
callees may use208 ONLY if the new explicit CFG check proves all validator calls
cross the existing unconditional TRY_ALLOC/retry loop. Low-role/initial limits stay
unchanged. inspect_allocation_dominance added to existing SASS reviewer; negative
test on real frozen55 Q352K rows correctly finds both pre-allocation calls and fails
dominance with no unresolved branches. Candidate positive proof passed all10 cases.
Before headers and three-header patch saved; general preparer supports flag/endpoint
geometry and exact target equality. No default/performance/EP16 promotion.
Before any future phase profile, review marker register limits/decoder for moved
transport under208; current plan is ordinary timing/correctness only.

Source-only next idea: metadata warp writes lane-owned padded counts to histogram,
then reads them back for the prefix scan; no other metadata-path histogram readers.
Saved56 row SASS confirms the round trip: Q31K STS1160/LDS11d0 before SHFL1320;
Q352K STS1140 loop/LDS11d0+1320 before two-chunk scan. Thus compiler has not
removed it. Potentially retain per-lane padded counts in registers across unchanged
SizesReady publication, preserving scan order, shared offsets/total_padded, acquire,
full-worker joins and footprint. This extends register lifetimes across release;
resource/spill gates and ragged/multi-chunk count proof required before GPU testing.
Evidence resource_review/metadata_scan_source_review.json preceded prototype57 below.
Re-read cutedsl Blackwell composition: separate communication warps apply role
register changes before work. Existing dedicated-dispatch failure changes geometry/
budgets, so source resemblance alone does not justify repeating that bundle.

Independent scan prototype57 now implemented in four headers, default-off
DG_MEGAMOE_ALL_PUSH_METADATA_REGISTER_SCAN=0/1, config/template field appended.
Warp0 loads the same counts in original per-lane order into a compile-time-sized
register array (one chunk for Q3, two Q35 EP8); invalid lanes remain zero. The
unchanged scan consumes those values instead of histogram shared stores/loads.
Keep __syncwarp, SizesReady publication, both full-worker joins, shared offsets/
total_padded outputs, scratch footprint, geometry, readiness and transport unchanged.
No token/model cutoff or composition with after-registers/cooperative modes.
Independent serial-prefix proof passes864 configurations/80,514 expert offsets,
including zero, ragged, partial32 and multi-chunk counts; not device/resource proof.
Saved before headers,127-line incremental patch and ownership review under C.
Frozen R/source/metadata_register_scan_20260926 copied completely from56 BEFORE
four-header/current-SASS-reviewer sync; active56 source/cache/plan untouched.
Fresh57-argument native/JIT cache. C/metadata_register_scan plan compares control/
rows/rows_scan/pull,80captures, both models8/32/256/1K/2K mirrored1000/rank.
CPU7473615 submitted20:39:48Z, COMPLETED8m15s0:0 on cpu-0013; monitor2374
terminal0. PASS30resource/10geometry/10default gates,10row-preservation checks
against56 and10compiled scan reviews. All shapes remove histogram LDS (Q3:1,Q35:2)
and one static STS site; preserve5/10 scan shuffles,1/2 carry broadcasts,2/3 shared
output stores and one SizesReady release. No new spills; maximum initial/caller/
callee/epilogue123/39/125/120. No GPU submitted per user stop request. Generic preparer
tracks exact scan flag/target equality and permits changed small candidate code;
register/spill/footprint limits remain unchanged. Both candidate/header and workflow
are frozen now. Scope CPU_GATES_PASS_GPU_DEFERRED_USER_STOP; GPU/analysis plans
remain unsubmitted. Candidate56 timing rejected as recorded above; no promotion.

Source-only follow-up ruled out SF-padding elision: with COALESCED_SF0, SF work is
already partition_rows*(H/512), and kSFBlock is a layout stride, not a padded-work
loop bound. The rejected coalesced mode also guards padded stores. No new change/job.
Historical late-count7432636/7432687 is not isolated count-lane evidence: default
workers+late0 were compared with adaptive workers+late1. Its small regressions and
Q35 large gains remain valid for that bundle, not a causal attribution for coop55.
Do not revive the bundle or blame subgroup regressions solely on late lane reads.

Source-only MMA geometry review: UMMA_M256 is feature width (swapped A/B),
UMMA_N=BLOCK_M is tokens, dynamically updated each block with get_valid_m<true>()
= align(min(remaining_rows,BLOCK_M),16). Thus M96 does not imply every ragged tile
executes96 token rows; fixed256 does not mean256 token rows. No new tile experiment
or frozen-source edit justified solely by padded-token hypothesis.

### Completed evidence and decisions

All following correctness audits PASS with matched collective preparation. Counts
are captures / exact cross-source comparisons / graph replays; records are80 except
phase and wave runs48. Every candidate retains all ranks, samples and mirrored orders.

| Campaign | GPU / final CPU analysis | Counts | Decision |
|---|---|---|---|
| row_unified |7459061 /7460148 recovery|120 /720 /960|Reject unified+rows as general successor.|
| split_pipeline |7459558 /7459559|80 /400 /640|Extra weight depth has opposite model effects.|
| split_pipeline_isolation |7461094 /7461095|100 /560 /800|A-depth reduction reproduces the effect; more weight depth does not explain it.|
| split_pipeline_phases |7462351 /7462352|44 /272 /416|Slow-tail redistribution; profiling bias retained.|
| split_pipeline_footprint |7462934 /7462935|80 /400 /640|Allocation explains part of Q35 gain; Q3 loss persists.|
| activation_credits |7463841 /7464630 recovery|100 /560 /800|Logical credit cap reproduces opposing effects; FC2-only does not solve them.|
| idle_dispatch |7464282 /7464283|60 /240 /480|Source/SASS work reduction regresses small timing controls.|
| credit_waves |7465035 /7465044|84 /528 /672|One-vs-multiwave does not explain credit effects; reject policy.|
| readiness_poll |7466010 /7466013|100 /560 /800|Q3 focused progress; Q35 forward tails mixed. Test row composition before promotion.|
| warp_metadata |7466414 /7466440|80 /400 /640|Groups2 gains Q35 large, regresses all small statistics/orders; reject. Investigate empty-partition work.|
| poll_rows |7466953 /7466960|100 /560 /800|L1+rows regresses Q3 1K allstats/orders; both polling order-dependent. Reject general composition.|
| warp_empty |7467343 /7467350|80 /400 /640|Partial recovery at8/32, but small32/256 allstats/orders still regress. Reject; defer row/group composition.|

Row/unified at W32/P2 versus row-only kernel/pipeline deltas(us): Q3 1K
+0.024395/+0.200069; Q3 2K-0.220724/-0.035778; Q35 2K+0.120628/+0.105495.
Q3 1K pipeline pooled/all-rank means and Q35 2K all-rank means regress both orders.
Q3 2K pipeline flips forward-0.469432/reverse+0.397877, every rank. Unified-only at
W32/P1 also loses Q35 2K+0.457037/+0.446851. All large cases behind pull; Q35 1K
shared M32 code. Do not add gains from older W16 comparisons to W32.

Initial split +1 weight: Q3 1K+0.161179/+0.161211; Q3 2K kernel-0.196834 flips by
order. +2 weight: Q3 1K+2.448903/+2.422195, every rank mean/median/p95 both orders;
Q35 2K gains~3.460us kernel. Isolation fixes weight depth to original: token5 Q35
2K-4.257402/-4.241269, allstats/orders; Q3 1K+2.235338/+2.198626, allstats/orders;
Q3 2K order flip. Extra2 versus token5 loses Q35 2K+0.833419/+0.841214 allstats/orders.
Equal-depth separate barriers give Q3 1K-0.184494/-0.164866 but Q35 2K
+0.207291/+0.214731 and Q3 2K order flip. No common/EP16 promotion.

Phase run:36 ordinary captures (control/token5/pull × both models256/1K/2K × orders)
and8 profiles (control/token5 × both models1K/2K),10 calls/rank. Exact core marker
coverage640 calls×75 CTA pairs; first MMA warp6 leader75CTAs, first FC2/last
accumulator warp8 all150CTAs, CombineEnd warps8..15 all150. Broad Genphase/LocalReduce
coverage is NOT claimed; old duplicate-marker issues remain. Ordinary/profile
kernel deltas(token5-control): Q3 1K+2.225904/+2.136; Q3 2K-0.451502/-1.0268;
Q35 1K shared-0.141173/+0.2492; Q35 2K-4.405901/-4.942. Profile absolute bias:
Q3 1K control/token5-0.868802/-0.958706; Q35 2K-2.733601/-3.2697us. No correction.
Q3 1K pair-mean firstMMA→firstFC2 envelope grows2.266837us; slowest accumulator
+1.9536us. Q35 2K slowest accumulator-3.8532us, latestCombineEnd-4.9448us, while
all-pair mean accumulator is LATER+2.347392us and only12/75 pair means improve.
Pair30 finishes last53/80control calls vs22/80token5. This supports tail redistribution,
not uniform FC2 speedup; intervals contain interleaved work. core_phase_review holds
review.json, pair_samples.json, pair_distribution_review.json; C/review_core_phases.py
and prepare_phase_followup.py retain the workflow.

Footprint: padded uses token5 instructions/rings with control's launch allocation.
Padded-token5 kernel/pipeline(us): Q3 1K+0.08055/+0.07999; Q3 2K+0.53517/+0.74876;
Q35 2K+0.98407/+0.98924. Padded-control: Q3 1K+2.47440/+2.51101 (every rank/stat/order
worse); Q3 2K-0.21515/-0.21389; Q35 2K-3.07047/-3.04961 (every rank/stat/order better).
Token5/control allocations(B): Q3 1K194964/214852; Q3 2K196996/214324;
Q35 2K195988/215876. Token5+1024B fits196KiB; controls exceed it. Blackwell's unified
L1/SMEM capacity supports196/228KiB partitions (official tuning guide); host sets
MAX_DYNAMIC_SHARED_SIZE only. Actual selected carveout/cache hits unmeasured.
Allocation sensitivity is demonstrated, an L1 mechanism is not. All large cases
still behind pull; small/Q35 1K compiled targets unchanged.

Credit-all at original physical rings/allocation minuscontrol kernel/pipeline(us):
Q3 1K+2.200978/+2.175040; Q35 2K-2.937009/-2.923198. Every rank mean/median/p95/
slowest and both orders have those signs. FC2-only: Q3 1K-0.158455/-0.149857
(allstats/orders improve), Q35 2K+0.270561/+0.280878 (allstats/orders regress).
Q3 2K FC2-only kernel forward+0.343059/reverse-0.156244, pipeline+0.812895/-0.150428;
every rank flips. Removing the FC1 restriction removes both major effects: a phase
interaction, not an isolated FC1 compute-cost claim. All large cases behind pull;
small/Q35 1K identical code and measured variation retained. Reject general use.

Idle-dispatch minuscontrol kernel/pipeline(us): Q3 8+0.354885/+0.368842,
32+0.134021/+0.125215,256+0.131384/+0.156516; Q35 8+0.201234/+0.204766,
32+0.109414/+0.100690. Every rank mean/both orders regress; Q3 8/256 and Q35 8
also every median/p95. Q3 1K-0.035643/+0.065390 flips by order; Q3 2K
-0.083402/-0.108672; Q35 1K-0.028550/-0.025050; Q35 2K-0.051091/-0.041856,
small/mixed rank changes. Reject; do not rescue with a token cutoff.

G2 follow-up source/SASS review (no new kernel/job): small Q35 grouped metadata
barriers contain BRA.DIV plus collective slow paths; their presence does not prove
dynamic execution/cost. PTX aligned barriers impose convergence conditions not
proved for expert-dependent subgroup loops. Reject blind barrier.sync→bar.sync
substitution; do not blame all regression on guards or once-per-startup late counts.
Evidence metadata_expert_groups_v3/barrier_followup_review.json. No cutoff rescue.

### Prototype contracts and reproducibility

Three changed headers: heuristics/mega_moe.hpp, jit_kernels/impls/
sm100_fp8_fp4_mega_moe.hpp, deep_gemm/impls/sm100_fp8_fp4_mega_moe_all_push.cuh.
Defaults0; no production promotion. Before snapshots/candidate.patch in each
campaign; freeze source/cache while consumed. Sources under R/source:
- megamoe_closeout_20260925: cleaned43-argument reference, used by row_unified.
- split_pipeline_20260926:44 arguments, independent A/SFA vs B/SFB rings;
  WEIGHT_STAGE_EXTRA0..4, wide-only. Extra weights consume spare SMEM and reduce
  A depth to fit232448B with1024B reserve. A/B depths Q3 1K+1=7/9,+2=5/10;
  Q3 2K7/8,5/9; Q35 2K7/9,5/10. EARLY_SF/direct-shared combinations fail.
- split_pipeline_isolation_20260926:still44, ACTIVATION_STAGE_LIMIT0 or2..32;
  nonzero selects independent rings even at equal depths. Weight depth stays
  original+extra; token5 A/B Q3 1K5/8,2K5/7,Q35 2K5/8. Each ring keeps its own
  phase, full/empty barriers and SF dependencies; MMA releases both in same order.
- split_pipeline_footprint_20260926:still44, host-only KEEP_SMEM=1 requires a split
  request and restores original allocation, failing if layout exceeds it. Identical
  token5/padded device targets. Layout accounting A=loadM*K+sfM*4+16,
  B=N*K+sfN*4+16; fixed=oldSMEM-oldStages*(A+B-16); new=fixed+Adepth*A+Bdepth*B.
- activation_credits_20260926:46 arguments, ACTIVATION_CREDIT_LIMIT0 or2..32,
  FC2_CREDIT_ONLY0/1 requires a limit. Wide-only actual config, limit<=physical depth.
  Adds empty wait for absolute t-limit after original own-slot wait at four A
  producer sites; no allocation/count/retirement changes. Negative initial stages
  use completed empty phase; indices/phases persist across FC1/FC2. Arithmetic
  invariant passed all depths2..32,limits2..depth,four wraps. Multiwave issuer
  FC2 arrivals are preserved. Source review plus100-capture exactness passes.
- idle_dispatch_20260926:47 arguments, SKIP_IDLE_DISPATCH default0. Bypasses
  scheduler only for !DirectShared && (UnifiedTMA || warp>=ActivationIssuers),
  preserving fetch_expert_recv_count for cleanup and every named barrier. Cleanup
  uses cached-count accessors, not final cursors. Applies all tiles; timing rejected.
  SASS control Q3 1K roleUR9 gate63a0/6530, empty-work loop6790..6810;
  Q35 roleUR8 gate4950/4b20, loop4e70..4ef0. Candidate bypass: Q3 guard5f10/5f30,
  count load then5f90→cleanup68c0; Q35 guard4420/4450,44d0→4fe0.
  mechanism_review.json and selected SASS retain exact evidence.

Native libraries with43/44/46/47/48/49 extension arguments are incompatible; use the
matching fresh build, never oldG2 donors. General C/prepare_pipeline.py supports
variants, totals/cache-root, role/register/spill gates, independent layout/wave
geometry and default instructions versus cleaned reference. Only verified timeout
assertion-line metadata is normalized; no other instruction changes ignored.
Rank reviewer is copied from maintained R/codex_scripts to campaign/workflow,
which avoids missing reviewers in older frozen sources. CPU gate totals:
row_unified50/10geometry; split30/10; isolation40/10; footprint30/10;
credits40/10; idle20/10; credit_waves36/6. Default comparisons pass matching case counts.

Recovery lessons: row CPU7458906 passed5m25s; GPU7459061 completed40m47s;
analysis7459062 saved audit/timings but missing reviewer failed9m28s, recovered
CPU7460148 in1m19s. Phase CPU7462005 compiled/validated then failed6m29s because
broad ROOT/SOURCE replacement corrupted environment/status names; delimited
placeholders fixed, review-only7462253 passed1m22s. Idle CPU7463975 built/passed
resources then reviewer tuple-key typo failed5m56s; targeted fix/review7464126
passed53s. Credits analysis7463842 failed44s before reading data because manual
job.json used job_id instead of canonical job; corrected both campaigns and
CPU-only recovery7464630 passed8m20s. Use maintained submit_plan.py and sync its
canonical job.json remotely before analysis. Never rerun GPU for reviewer/schema errors.

### Wave factorial and next mechanism

Wave factorial holds physical rings/allocation/tiles fixed. Control/credit5,
single/single_credit5, adaptive1grid/more_credit5 plus pull, both models256/1K/2K,
mirrored1000 calls/rank. Q3 wave counts1/1/2; Q35 2K2/1/4. Small and Q35 1K
unchanged compiled paths. PASS84 captures/48 records/528 comparisons/672 replays.
Credit-minus-matching-uncapped kernel/pipeline(us): Q3 1K control+2.401/+2.387,
single+2.248/+2.133,more+2.937/+3.031; Q3 2K+0.269/+0.504,+0.266/+0.515,
+0.794/+0.984. Q35 2K-2.844/-2.828,-5.326/-5.301,+5.692/+5.590.
Q35 single-cap improvement and four-wave-cap regression hold every rank mean/median/
p95/slowest and both orders. Q3 two-wave cap regresses all statistics/orders too.
Uncapped single Q35 2K loses3.056/2.937us vscontrol; uncapped more is mixed~0.
Q3 uncapped more loses4.364/4.316us at1K,0.273/0.303us at2K. All candidates still
behind pull on all four large cases. Full deltas/rank-order evidence selection_review.json.
Thus one-vs-multiwave cannot explain model difference. Deprioritize unimplemented
wave-boundary observer handoff; do not infer wave/model policies.

Readiness polling hypothesis: control SASS repeats LDG.STRONG.GPU+CCTL.IVALL
in unsuccessful acquire loops. Opt-in cold-path relaxed loads may reduce invalidation
work while preserving a successful acquiring observation. PTX ISA9.2 section8.8:
https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#release-and-acquire-patterns
Frozen R/source/readiness_poll_20260926 uses fresh48-argument native/cache.
SASS reviewed before GPU; focused results/next composition are in the active section.
Source protocol review saved at readiness_poll/mechanism_review.json. Current plan
does not dynamically cover unified TMA, cached-destination SF or per-stage L2 options.
CPU7465811 completed12m25s0:0; GPU7466010/CPU7466013 correctness PASS.
Cutedsl reference spin_wait retains acquire loads with nanosleep150; do not claim
our relaxed-spin variant is that implementation. Its held-FC2/forced-FC1 scheduler
requires shared task publication; FastAFD fixed role replay cannot simply reorder
one role independently. Retain as separate future redesign, not this prototype.

## September 25 handoff — continue the throughput goal

The cleanup milestone is complete; the original performance goal is **unfinished**.
After closeout, the user explicitly requested recording progress and TODOs so the
next agent can continue. The goal tracker was marked complete at the user's prior
stopping point; that administrative state is not evidence of performance success.
This handoff supersedes the earlier “no further optimization work” wording. This
update documents continuation; it does not launch experiments in the current turn.
All submitted jobs are terminal. No production defaults or model/token exceptions
were promoted. The working tree contains uncommitted progress; preserve it.

Updated success criterion: geometric mean speedup versus pull over the fixed full
comparison set, while preserving small-case performance and correctness. Individual
non-small cases need not all win. The last full sweep remains **57/91 wins,34 losses**;
its historical sub4K kernel geometric mean is0.991842x (large1K/2K0.885666x).
Separate-allocation rank0 medians are descriptive, not matched final proof. Focused gains do not establish
the full-set result. Keep the through8192 sweep for context, including4K/8K regressions.

Paths: R=`/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD`;
C=`R/artifacts/throughput_20260922`. Compact local evidence mirrors C beneath
`scratch/megamoe/throughput_20260922`. Full pre-cleanup memory and four headers are
archived at `scratch/megamoe/closeout_20260925/before_cleanup` and
`R/artifacts/closeout_20260925/before_cleanup`. Frozen sources and completed caches
remain untouched. That archive preserves detailed failed attempts and job history;
it is historical evidence, not a queue of work to resume.

Latest full report:
[sweep-report.html](scratch/megamoe/ordinary_latest_8k_20260924/plots/sweep-report.html).
113 cases through 8192 total tokens: Qwen3 EP2/4/8/16/32/64 and Qwen3.5 EP4/8/16/32/64.
Its rank0 ordinary medians compare separate allocations against September22 pull;
this is a descriptive sweep, not matched producer-to-main speedup proof. ISL in
these comparisons denotes total tokens. Original sub4K matrix: Q3 51 + Q35 40 cases.

## Prioritized TODOs for the next agent

1. **Resume from the cleaned source and saved evidence.** Local checkout is
   authoritative; `R/source/megamoe_closeout_20260925` is the frozen cleanup
   reference. Read the result/selection reviews in C/metadata_row_partitions,
   C/metadata_row_partitions_normal, C/metadata_row_partitions_interleaved,
   C/metadata_row_partitions_ep16 and C/dispatch_role_pre4k. Use
   `scratch/megamoe/ordinary_latest_8k_20260924/comparison_review.json` and the
   per-model ordinary CSVs to recover the full failure matrix before expanding
   scope. Reuse completed checks; no unchanged baseline rerun merely to resume.
2. **Keep a small representative EP8 target set.** Start with both models1K/2K,
   plus8/32/256 controls. Q3 row splitting is the strongest broadly validated
   recent path; Q35 remains the major generalization gap. Compare a candidate
   against both the topology control and the appropriate explicit-worker control,
   plus pull in the same allocation. Model selection for experiments is allowed;
   model/token lookup branches or a per-case winner table in the kernel are not.
3. **Isolate the next mechanism before implementing it.** Review existing phase,
   SASS and dispatch-role evidence first. The row/unified factorial above now
   rejects unified TMA as a general successor to row-only. Split weight/activation
   rings improve Q35 2K but regress Q3 1K; the active stage-isolation comparison
   separates independent barriers from the token/weight depth tradeoff.
   For Q35, the dedicated-dispatch bundle improves2K but hurts Q3; separate metadata
   thread allocation, transport placement and register-budget effects before any
   redesign. G2's1K gain also came with small/2K losses: do not restore it unchanged
   or rescue it with a token cutoff. If existing evidence cannot identify a useful
   mechanism, profile a complete representative workload with an ordinary timing
   control; do not infer an isolated bandwidth limit from whole-graph NCU totals.
4. **Pass CPU gates before spending GPU time.** Freeze each candidate and prepare
   native/JIT kernels on CPU. Check geometry, ownership, readiness masks, empty
   partitions, barriers/retirement and actual role-specific registers/spills.
   Removed G2 had a different native/template ABI: never seed cleaned-source builds
   from G2 caches. Clean reference has43 extension arguments; the split prototype
   has44 with a different final argument from G2. Use its fresh native build only.
   Positive explicit worker counts remain supported; negative modes and G2=2 fail.
5. **Validate generality before EP16 or defaults.** Run matched ordinary EP8 with
   exact-output/repeated-replay checks and all ranks/both orders. Preserve every
   sample; review kernel and producer-to-main mean/median/p95/slowest separately.
   Promising changes then need nearby384/448/768/1536/3072, nonuniform/skew routing
   and queued ABBA/BAAB checks, including empty owners. Q35 448 caught worker-4;
   Q35 2K rank/tail checks caught row-splitting limits. Promote to EP16 only after
   relevant EP8 progress, retaining original Q3 192/Q35 912 small controls. Existing
   Q3 EP16 row evidence passes; it does not qualify new Q35/combined candidates.
6. **Finish with broad proof and reporting.** Once the focused candidate survives
   holdouts, run the original Q3 EP2/4/8/16/32/64 and Q35 EP4/8/16/32/64 ordinary
   sweep, powers-of-two totals from EP through8192, against pull with matched
   contracts. Submit independent sweep jobs together once prepared, as requested
   previously. Recompute all91 sub4K comparisons and regenerate the existing HTML
   with model/EP/token coverage and provenance. Update the score only from that
   matrix, investigate every remaining loss, and retain full-model alignment/E2E
   as a separate gate before claiming a production replacement.

Priority gaps, ordinary matched all-rank means (rows minus pull, microseconds):

| Model / EP / total | Kernel gap | Producer-to-main gap | Interpretation |
|---|---:|---:|---|
| Q3 / 8 / 1024 |3.842|4.004|Row gain robust; still slower than pull.|
| Q3 / 8 / 2048 |4.497|4.292|Row gain robust; still slower than pull.|
| Q35 / 8 / 1024 |9.111|9.433|Row candidate shares the control target here.|
| Q35 / 8 / 2048 |13.638|14.075|Rows lose to equal unsplit workers; normal tails mixed.|
| Q3 / 16 / 1024 |0.472|0.763|Closest retained large case; EP8 remains the first gate.|
| Q3 / 16 / 2048 |4.500|5.186|Validated row gain does not close the pull gap.|

These gaps come from the row campaigns below. Do not combine them with dedicated
or G2 gains from separate allocations. Dedicated Q35 EP8 2K still trails its matched
pull by7.824us kernel/8.081us pipeline; no universal best configuration is established.

Reuse `codex_scripts/prepare_megamoe_throughput.py`,
`precompile_megamoe_sweep.py`, `review_megamoe_rank_comparison.py` and the normal/
interleaved workflows in [the inventory](codex_scripts/README.md). Scratch helpers
`C/prepare_normal_followup.py` and `C/prepare_row_partition_followup.py` preserve
prior plans, but have frozen-source assumptions: adapt paths explicitly for a new
candidate, never rewrite completed plans or silently use an old native cache.
Persist each job ID/output path in `C/jobs.json` and its local mirror. Monitor at
300s; CPU preparation/analysis must not hold GPUs. Do not blindly submit dormant
worker-policy EP2/EP4 or G2 follow-up plans: their predecessors failed promotion.
Preserve rejected evidence and record each new attempt/result here concisely.

## Retained progress and cleanup

Retain the validated metadata row-partition implementation as an explicit option,
with positive worker counts and original topology defaults. Remove the rejected
expert-group implementation and automatic worker selectors -1/-2/-3/-4. Negative
worker requests fail fast; stale `METADATA_EXPERT_GROUPS=2` also fails fast. Frozen
G2 and worker-policy sources preserve all experimental evidence. Other useful
historical paths remain available; this closeout does not select a per-case winner
or erase unrelated work. Changes are uncommitted unless separately requested.

`DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS` defaults to1; explicit1..8 divides
physical workers into logical tile workers and row owners. Full-activation layouts
only; FULL_TMA1 small tiles remain P1. Each owner has an SF/A CTA pair; row ranges
align to metadata_threads/16. Four unique bits per participant use release ADD in
the existing32-bit readiness word; consumers acquire the complete mask. Empty row
owners still publish on nonempty tiles. Existing barriers, proxy fences, route
acquires, reset, DispatchDone and all150 compute CTAs remain. Recycling,
direct-shared, coalesced-SF and compute-warmup combinations are rejected.
Host ownership model passed34816 row cases/3328 ownership cases/4080 flag subsets
and39864 empty-owner cases; this is not device correctness proof.

Frozen validated source: `R/source/metadata_row_partitions_20260925`.
Clean closeout source: `R/source/megamoe_closeout_20260925`.
Closeout CPU7452492 rebuilds native/JIT in a fresh cache and compares retained EP8
control/workers/rows plus EP16 Q3 control/rows against completed GPU-validated
configurations and all disassembled instructions. No new GPU allocation requested.
CPU7452492 completed5m06s: PASS, 27 specializations with identical configurations
and all instruction addresses/operands (whitespace only normalized). No fresh GPU
claim: the compared machine code is the code already validated on GPUs.

## Row-partition results

Ordinary paired all-rank means, microseconds. EP8 rows=W32/P2 versus topologyW16/P1;
EP16 rows=W16/P2 versus topologyW8/P1. Pipeline is producer start to main end.

| Model / EP / total | Control kernel | Rows kernel | Pull kernel | Control pipeline | Rows pipeline | Pull pipeline |
|---|---:|---:|---:|---:|---:|---:|
| Q3 / 8 / 1024 |53.850916|51.698298|47.856658|66.318545|64.154235|60.150151|
| Q3 / 8 / 2048 |82.354341|77.036168|72.539164|99.255759|93.957284|89.665616|
| Q35 / 8 / 1024 |103.252063|103.265105|94.154310|123.786773|123.799944|114.366690|
| Q35 / 8 / 2048 |128.905257|126.977092|113.338901|161.535535|159.617154|145.541705|
| Q3 / 16 / 1024 |37.501975|35.288714|34.817174|47.608382|45.330088|44.566871|
| Q3 / 16 / 2048 |57.807984|52.129928|47.629957|70.309198|64.593464|59.407394|

EP8 changed cases Q3 1K/2K and Q35 2K improve every rank/both orders/mean/median/
p95/slowest versus topology control in both metrics. Q35 1K and both models8/32/256
share compiled targets; their timing differences are retained variation. Against
W32/P1, Q3 rows gain1.996/5.345us kernel, but Q35 2K loses3.003us kernel/3.017us
pipeline. Row splitting is not universally better than whole-tile parallelism.

EP16 Q3 1K/2K gains2.213/5.678us kernel and2.278/5.716us pipeline versus topology;
versus W16/P1 gains2.314/5.327us kernel and2.409/5.319us pipeline. Every16 ranks,
both orders and all four statistics improve. Still behind pull by0.472/4.500us
kernel and0.763/5.186us pipeline. Original small Q3 192/Q35 912 use shared targets;
paired rows/control pipeline variation -0.000153/-0.028452us is not a code gain.
EP16 Q35 large cases were withheld following mixed EP8 evidence; no runtime selector.

| Campaign in C | Jobs and gates |
|---|---|
| metadata_row_partitions | CPU7449413 5m36; deferred7449457 failed on SASS spacing/assert-line metadata, corrected7449655 2m. GPU7449746 28m17; analysis7449756 6m26. PASS80 captures/80 records/400 exact comparisons/640 replays; matched collective preparation. 30 strict resources/10 geometry triples, no new spills. |
| metadata_row_partitions_normal + metadata_row_partitions_interleaved | CPU7450613 5m55:40 resources/20 geometry pairs/40 capacity checks. GPU7450789 31m36; analysis7450791 4m28. Normal120 runs/1440 records/4320 comparisons/960 replays. Interleaved28 processes/448000 rank samples/2016 comparisons/224 replays/224 interleaved exact checks. All PASS. |
| metadata_row_partitions_ep16 | CPU7451720 7m14:12 resources/4 geometry triples. GPU7451910 13m52; analysis7451911 5m09. PASS32 captures/64 records/320 comparisons/512 replays; matched collective preparation. |

Normal routing: both models8/32/256/384/448/768/1K/1536/2K/3072. Interleaving:
both models8/32/256/384/448/1K/2K. Saved executed padding inputs verify empty P2
participants on nonempty tiles for both models, after exact output comparisons.
Q3 normal768/1K/1536/2K/3072 joined gains1.529/2.559/4.701/5.726/8.238us versus
control, every rank/statistic/order. Interleaved Q3 1K ABBA/BAAB gains2.559/2.643us;
2K5.920/5.859us, all ranks/statistics/windows and500/500 cycles each order.
Q35 normal2K/3072 gains0.512/6.802us but2K forward p95 is mixed. Interleaved2K
mean gains0.438/0.378us and slowest0.214/0.056us, while individual rank means reach
+0.990/+1.227us and p95 +2.048us; one BAAB window regresses0.122us. No general
promotion. Normal joined timing includes preparation/reset and cannot be compared
directly with the ordinary kernel/pipeline table. All large cases still trail pull.

## Rejected directions and useful limits

| Direction | Decision / evidence |
|---|---|
| Expert groups G2 | Removed from maintained code. V3 CPU7451316/deferred7451321 PASS20 resources/10 instruction controls after V1 barrier issue and V2 shared-address spill. GPU7451540 21m03/analysis7451543 5m06 PASS60 captures/80 records/240 comparisons/480 replays. Q35 1K gains8.343us kernel/8.354us pipeline every rank/order/tail, but still behind pull0.962/1.266us. Q35 8/32/256 regress0.589/0.297/0.304us kernel;2K pipeline +0.194us. No size exception or further follow-up. C/metadata_expert_groups_v3; frozen source metadata_expert_groups_v3_20260925. Count-lane recomputation is once per scheduler startup, not per compute tile; cause of regressions unproven. |
| Automatic worker -4 | Removed with other negative policies. Large Q35 ordinary1K/2K gains9.518/4.691us kernel, but normal Q35 448 +0.350us joined and ABBA/BAAB +0.341/+0.321us every rank/statistic/order. GPU7448032/analysis7448034 PASS normal120/interleaved28. EP2CPU7448181/EP4CPU7448182 passed; their GPU plans never submitted. Native91-shape audit changes20 cases, leaves16 old failures unchanged; coverage, not performance. Frozen batch_capped_workers_20260925. |
| Automatic worker -3 | EP4 Q35 total8 regresses0.524us kernel/0.505us pipeline every rank/order/tail. Explicit positive counts preserve useful large configurations. |
| LOCAL_SOURCE V4 | Removed in cleanup pass3 after the rejection below. GPU7447957/analysis7447958 PASS60 captures/80 records/240 comparisons/480 replays. Q35 1K/2K +1.034/+1.411us kernel and +1.042/+1.558us pipeline all ranks/orders; several tiny controls hurt. Q3 large apparent gain flips order. V2 rejected, V3 unmeasured; frozen V4 evidence retained. |
| BALANCE_EPILOGUE | Keep explicit option/default0: modest wide EP8/EP16/normal gains; no full-matrix promotion. |
| RETURN_WIDE / RETURN_WG_SYNC | Useful selected Q3 4K/8K evidence retained; Q35 rejected. No blanket promotion. |
| EXTENT_TMA / FC2_STAGE_READY | Extent1 EP16 Q3 2K +0.312us every rank; mode2 no increment. FC2 stage-ready Q35 2K -0.309us mean but reverse p95 mixed; other cases flat/worse. |
| COMPACT_TILES2 | Already removed: normal Q3 1K +3.842497us joined every rank/order. Keep COMPACT1 and explicit diagnostic BLOCK_M64. |
| Other rejected general policies | WINDOW_WIDE, DIRECT_SHARED, EARLY_SF, startup warmup-all-tiles, activation issuers2/4, FULL_TMA2 small, WAVES2, late-count standalone, SF_VECTOR, prefetch, ROW_RELEASE. Detailed matched evidence and frozen sources in archived full memory; do not revive unchanged. |

Final comparison C/dispatch_role_pre4k: CPU7449851/deferred7450043
PASS18 resources/6 geometry triples, no new spills. GPU7452173/analysis7452177
compare original/unified/dedicated/pull, EP8 both models256/1K/2K in mirrored order,
48 captures. Common16 workers/P1/LOCAL_SOURCE0, unchanged frozen row source.
Dedicated changes transport placement, metadata256→128, sequential SF/middle work
and register128/40/168; results measure this bundle, not only freed epilogue warps.
Both256 and Q35 1K share targets. GPU7452173 completed18m49s; analysis7452177
completed4m27s. PASS48 captures/48 records/240 exact comparisons/384 replays, matched
collective preparation. Unified/control Q3 1K/2K gains0.139608/0.480719us kernel,
0.149138/0.720841us pipeline, every rank/order/mean/median/p95/slowest. Q35 2K
unified gains0.157049us kernel/0.173124us pipeline mean, but reverse pipeline p95
regresses0.032us (individual ranks up to0.064us). Dedicated/control Q35 2K gains
5.793404us kernel/5.922313us pipeline across ranks/orders/statistics, but Q3 1K/2K
regress4.450483/9.667385us kernel and4.472079/9.466750us pipeline, every rank/order/
statistic. Reject dedicated as a general replacement. Dedicated Q35 2K still trails
pull7.823993us kernel/8.080805us pipeline; all four large cases trail pull under
both alternatives. Retain useful explicit options and full shared-code variation.
No normal/tiny/EP16 successors were submitted at closeout; no default or
case-specific selection. Any continuation must satisfy the TODO gates above.
Selection review, summaries and all rank reviews saved.

## Measurement and interpretation

Compare complete ordinary workloads in the same allocation: all ranks, mirrored
orders, mean/median/p95/slowest, main kernel and producer-to-main span. Keep every
sample and shared-code variation. Pull has exact replay, while cross-source byte
comparisons are a distinct gate; match collective preparation. Profiling and normal
joined graphs have different timing boundaries. Never add gains across allocations.
No model/token lookup policies, threshold fitting or sample filtering. EP8 progress
precedes EP16; tiny/nearby/normal/interleaved cases guard against overfitting.
Full matrix and model E2E remain required before claiming a general replacement.

Whole-graph NCU7443700/7443704 passed16 captures/128 reports/32 records/32 comparisons/
128 replays. AP/pull DRAM-read difference -1.75..+0.41%; AP reset writes6.3–17.9MB.
Profiler durations were distorted; no isolated kernel-bandwidth attribution.
Q3 1K phase first MMA12.60 AP versus11.54 pull, first FC2 epilogue38.06 versus32.83,
final44.75 versus39.15us. These are overlapping envelopes with waits, not pure FC1
cost or proof of an FC1 bottleneck. Existing150 CTAs join compute; destination routes,
deduplicated push and tile-window readiness already exist. Reset clears invocation
rows in both banks, not the whole reserved arena. SF words cover128K once per stage;
no four-stage reuse. One M tile per expert limits multicast reuse along token tiles.
Output-feature width is constrained by 2CTA UMMA_M256 and needs a layout redesign.
`MEGAMOE_THROUGHPUT.md` and cutedsl designs provide ideas; Rubin constants and local
cluster barriers are not SM100 tuning results or cross-GPU completion mechanisms.

The following sections preserve earlier selected implementation/model evidence.
Their dated status descriptions are historical; the handoff above is authoritative.

## Original TRT-LLM / pull-payload ordinary sweeps — complete September22

User requested the same EP/token ranges for original installed TRT-LLM and best
metadata-push/payload-pull/combine-push. Campaign
`scratch/megamoe/pull_original_sweep_8k_20260922`, matching remote artifacts.
Six EP allocations ran pull then original; each variant covers113 cases
(Qwen3 EP2–64, Qwen3.5 EP4–64; powers-of-two totals from EP through8192).
All226 cases captured1000 ordinary calls; no IKET or filtering. Monitored every300s.
CPU7363778 prepared pull kernels in6m48s; original used the untouched pinned image.
Script-only extensions support pull preparation/collection and ordinary-only original
capture/collection. No new CUDA kernel changes.

| EP | Job | Cases, both versions | Elapsed |
|---|---|---:|---|
|2|7363782|7501627 RUNNING 00:13:26; 43 rank0 reports; stderr 0 bytes|
|4|7363783|7501715 PENDING 00:00:00|
|8|7363784|7501471 COMPLETED 01:10:26; 220 rank0 reports; stderr 0 bytes; exact-output log audit PASS|
|16|7363785|7501540 COMPLETED 01:18:42; 220 rank0 reports; stderr 0 bytes; exact-output log audit PASS|
|32|7363786|7501716 PENDING 00:00:00|
|64|7363787|7502092 COMPLETED 01:37:13;160rank0 reports; stderr0; exact-output audit PASS|

All GPU jobs completed0:0. CPU7363791 exported17 independent groups with4workers,
then audited all226,000 rank0 instances in1m04s. All16 pull EP64 exact-combine checks
pass; original checks finite output and byte-exact graph replay. One Slurm-query
timeout restarted only the analysis watcher. Original EP32/64 teardown emitted UCX
peer-reset messages; captures/replay checks and both jobs completed successfully.

Collected compact results, ordinary CSVs/audits and Nsight summaries locally;
raw traces remain HSG. Two HTMLs beside the existing all-push plot:
`ep-scaling-fixed-tokens-original-trtllm.html` and
`ep-scaling-fixed-tokens-pull-payload.html`. Each has23 labeled curves/111 points;
two single-EP totals omitted, same left grey-bold labels and both-panel y ticks.
Data/anchor checks and PNG visual review pass; `plot_validation.json` records them.
Original includes routing inside MegaMoE; optimized pull has external metadata
preparation. This boundary is in plot provenance; no causal paired speedup claim.

## Ordinary power-of-two sweep — complete September22

User requested Qwen3 EP2/4/8/16/32/64 and Qwen3.5 EP4/8/16/32/64,
total tokens EP,2×EP,...,8192; one GPU job per EP, ordinary only, five-minute
monitoring until completion. Initial CPU7360150 hit the existing1024-row limit;
all six dependent jobs cancelled before allocation. User then explicitly authorized
raising the limit to8192. Only the two host assertions changed (CPU generation and
actual launch); device kernel code is unchanged. Maximum exercised rows/rank is4096,
so this sweep does not validate8192 rows/rank despite permitting that ceiling.

Recovery campaign: `scratch/megamoe/ordinary_sweep_8k_20260922` and matching
`R/artifacts/ordinary_sweep_8k_20260922`. CPU7361586 passed5m43s (44 ordinary
specializations, all plans). GPU jobs all completed0:0:

| EP | Job | Cases | Elapsed |
|---|---|---:|---|
|2|7361595|7501627 RUNNING 00:13:26; 43 rank0 reports; stderr 0 bytes|
|4|7361596|7501715 PENDING 00:00:00|
|8|7361597|7501471 COMPLETED 01:10:26; 220 rank0 reports; stderr 0 bytes; exact-output log audit PASS|
|16|7361598|7501540 COMPLETED 01:18:42; 220 rank0 reports; stderr 0 bytes; exact-output log audit PASS|
|32|7361599|7501716 PENDING 00:00:00|
|64|7361601|7502092 COMPLETED 01:37:13;160rank0 reports; stderr0; exact-output audit PASS|

All113 baseline JSONs pass shape/mode/sample checks; all16 EP64 exact-combine
checks pass. CPU analysis7362548 passed4m22s and audited113,000 rank0 ordinary
kernel instances (1000/case), with no IKET captures or sample filtering.
An initial analysis submission referenced expired completed dependencies and was
rejected without a job; submission after verified GPU completion succeeded.
`{qwen3,qwen35}/ordinary.csv` and `ordinary_audit.json` contain63/50 audited cases.
Reusable collector and EP plot loader now explicitly support ordinary-only data.

Requested existing `all_push_joined_decomposition_20260921/plots/ep-scaling-fixed-tokens.html`
updated with23 curves/111 points, unfiltered rank0 kernel medians and anchored ideal
scaling. Two single-EP totals are excluded from curves. All point values/anchors
checked and PNG visually reviewed. Previous plot/comparison preserved in recovery
campaign `previous_plot/`. Baseline graph times include preparation and routing;
the plot uses isolated main-kernel Nsight medians. Raw traces stay on HSG.
Plot follow-up labels all23 curves to the left of their leftmost measured points
with small grey bold token counts and leader lines. Labels are spaced in display coordinates
to avoid overlap; y tick labels appear on both panels. Reduced the added left
padding to roughly one-third (x-limit factor1.4→1.22), with a4pt label gap. Regenerated
existing HTML/preview and visually checked; measured data unchanged.

## Combined normalization preload — selected September22

User requested normalization preload on and local Nsight traces. GPU7356961 passed
18m42s; CPU7356986 passed40s after GPU release. Six mirrored perf-mode captures,
90 ordinary samples at Qwen3 A3:F1 EP8 batch4/128K, synthetic KV/perfect routing.
Combined = earlyPDL1/ahead1/8workers/vector reset(TMA0)/norm preload1. Selected
control differs only in preload0. Existing compiled modules reused, source unchanged.

| Mode | Pooled ms | Saving vs pull |
|---|---:|---:|
| Pull |18.144076833|—|
| Selected, preload0 |17.891784667|0.252292167ms (1.3905%)|
| Combined, preload1 |17.842612433|0.301464400ms (1.6615%)|

Preload adds0.049172233ms (0.27483%), forward/reverse0.061229267/0.037115200ms.
Retain preload1 as **new selected best**: it improves both matched orders. Total
1.6615% does NOT exceed historical1.973% from another allocation; preload-off was
already slower relative to pull here. Cause unestablished; do not add cross-allocation
savings. Evidence `preload_model_analysis/` and `combined_selected_result.json`
under `scratch/megamoe/reset_overlap_20260921`.

Fresh actual8192/A3:F1/EP8/B4 real-input alignment passes:96x17 full prompts and
continuations match cached-score inputs;1398 top1/1631 top10,1/1632 top10 misses,
within agreed max4 (same prompt92/position11/rank17). Perf mode0/NSYS0, no GPU-held
scoring/manual waits; all jobs terminal. Only the combined forward arm's exact
metric-selected attention rank1/FFN rank25 pair copied (2,497,262bytes), both nonempty
and remote/local sizes matching, with unchanged metric/result and provenance:
`scratch/afd/cases/qwen3/isl_131072/ep_8/a3-f1/batch_4/megamoe-reset-pdl-preload-20260922-j7356961/`.
Other raw captures remain remote. Alignment summary in the matching `isl_8192` path.

Cross-allocation ordinary main-kernel follow-up (rank25, all15 measured graphs per
order,188calls/graph): old pull37.999569us → new36.695467us; old selected preload0
36.233277us → new35.446559us. Both kernels improved, pull by1.304102us versus
selected0.786718us. Pull remains slower; selected advantage narrows1.766293→
1.248908us. Shrink0.517385us x188 =0.097268ms, close to observed model-advantage
loss0.105906ms. This supports reduced relative kernel gain, not exact critical-path
attribution or a proven hardware cause. New preload1 main35.368325us and output
norm3.855517us versus preload0 norm4.178173us. Evidence:
`preload_model_analysis/cross_allocation_kernel_comparison.json`.

## Selected reset/PDL implementation and invariants

Target Qwen3 A3:F1 EP8 batch4 has six physical rows per FFN lane. Campaign root
is `scratch/megamoe/reset_overlap_20260921` locally and `R/artifacts/reset_overlap_20260921`
remotely; tested serving source is `R/source/local-mirror`. Plans, commands, exact
source patches and job records remain in each experiment directory.

Selected settings (opt-in; pull and legacy defaults remain available):
`MINISGL_MEGAMOE_PAYLOAD_MODE=all_push`, `MINISGL_MEGAMOE_RECYCLE_EARLY_PDL=1`,
`MINISGL_MEGAMOE_RECYCLE_AHEAD=1`, `MINISGL_MEGAMOE_RECYCLE_CTAS=8`,
`MINISGL_MEGAMOE_RECYCLE_TMA=0`, `MINISGL_MEGAMOE_NORM_PRELOAD_WEIGHT=1`.
Both Ray environment lists forward all settings. Invalid values fail explicitly.

- Gate-entry PDL admits independent reset workers while token selectors still
  acquire complete logits/SF from the gate. Main acquires the existing CountsReady
  aggregate (physical rows+8), then the async proxy fence. No added launch/grid join.
- Two equal-layout arenas per lane alternate by layer parity, each with two
  activation banks. Initially cleared banks certify generations0 and1. Workers
  latch epoch and monotonic local/assigned-peer readiness before current admission.
  They maintain the current next bank only if needed, then the other arena's next
  bank. Current upkeep is required for repeated layers and0,1,0 graph replay's0,0
  boundary; unconditional cross-arena swapping is incorrect.
- Clear stores precede CTA join, worker-completion acqrel ticket and system-release
  payload readiness. Metadata retirement stays necessary; per-arena counters cannot
  overlap selector calls. Peer-outer contiguous clearing changes no arithmetic.
  Worker counts1/4/8 contribute8/count to CountsReady;1 omits completion aggregation.
- Independent norm preload keeps the first weight vector across the existing
  hardware producer wait and unchanged reduction. Qwen3 H4096/512threads/vec8
  covers all weights; larger dimensions load later rounds normally. SASS confirmed
  first weight load before dependency acquire and no local spills in either mode.
- Optional/default0 TMA reset retains a4KiB shared source tile, per-thread async
  shared fence and CTA join, bulk copies, commit and **full** `wait_group 0` before
  existing completion/publication. Never substitute `.read`. References and ordering
  contract: `reset_tma/contract.json`. Model increment0.002159ms is not a material
  improvement; vector stays selected. No software-gate/main-warp prototype is live.
- Diagnostics distinguish actual clears from already-ready observations; skipped
  clear statistics are null. Worker coverage, publication/acquisition and owning
  kernel intervals remain audited. Instrumentation does not establish ordinary
  latency, useful instruction overlap or a memory-bandwidth bottleneck.

## Reset/PDL model milestones and alignment

All cohorts use128K perf mode, synthetic KV/perfect routing, A3:F1 EP8 batch4 and
2microbatches. Norm preload is off in these earlier cohorts. All measured samples
are retained; the primary metric uses the maximum complete graph span across
attention rank1 and model/FFN rank25. Kernel averages use188calls/graph (94layers x2),
15 measured graphs x2 orders =5,640 calls per variant on FFN0, not across EP ranks.

| Cohort | Pull ms | Current all-push ms | Early PDL/ahead8 ms | Total saving |
|---|---:|---:|---:|---:|
|7341284, CPU recovery7341819|18.205295|17.965749|17.881714|0.323581ms /1.777%|
|7346203, CPU7346230|18.150747|17.892048|17.792549|0.358198ms /1.973%|

First cohort:90 samples; extra reset/PDL0.084035ms. Ordinary main pull37.116470us,
current35.263322us, selected35.756654us. Pull→current main projection0.348392ms;
measured total leaves0.024811ms gap. Main entry advances0.937300us while main duration
grows0.493332us; producer→main-end improvement0.443968us x188 =0.083466ms, matching
the additional model saving. Do not cherry-pick the smaller0.255645ms selected-main
projection as proof of full realization. Evidence `model_analysis_recovered/`.
Separate22,560-call model diagnostics: all clear the other arena, no current clears;
FFN0 clock-offset width1.344us,2727/2820 clears definitely before main. No guaranteed
reset-worker overlap with gate/main; not proof of zero instruction overlap.

Second cohort:120 samples including optional TMA8,23m00s GPU and54s CPU. Selected
extra reset/PDL0.099499ms; total0.358198ms meets corresponding ordinary-main projection
`(37.999569-36.124091)us*188 =0.352590ms`. TMA graph17.790391ms, extra0.002159ms
(forward0.000136/reverse0.004181). This meets the expected saving on this allocation,
not a theoretical optimum or proof other gains are impossible. Evidence
`tma_model_analysis/`; source `tma_model_source.patch` and file list. Combined preload
cohort and cross-allocation interpretation are at the top of this document.

Alignment uses actual8192 context, real routing/prefill,96prompts x17tokens and the
agreed maximum4 top10 misses. All successful generations below exactly match full
prompt/continuation cached scoring inputs;1398 top1,1631 top10 of1632, same sole
prompt92/position11/token6149/rank17 miss. This passes the agreed gate, not strict
zero-miss alignment. No fresh reference scoring is required with identical inputs.

| Version | GPU generation | CPU audit | Evidence |
|---|---|---|---|
|Recovered early PDL/ahead8|7341912,6m29s|7341917,43s|`generation_recovered/`, `generation_audit_recovered/`|
|Selected vector checkpoint|7343632,5m51s|7343648,28s|`final_alignment/`, `final_alignment_audit/`|
|Optional TMA8|7347084,5m43s|7347098,29s|`tma_alignment/`, `tma_alignment_audit/`|
|Selected vector + norm preload|7356961,18m42s including perf|7356986,40s|`preload_model/`, `preload_model_analysis/`|

Local current trace pair is linked above. The earlier four-mode cohort has16 metric-
selected reports (18,930,677bytes) in the standard case hierarchy, labels
`megamoe-reset-pdl-<arm>-20260922-j7346203`; `local_nsys_traces.json` inventories them.
All pairs are nonempty, match remote sizes and retain original metrics/results.

## Reset experiments: accepted and rejected evidence

Ordinary adapter gains below are complete-chain maximum-rank means, forward/reverse;
positive means faster. Instrumented clear spans alone never select a candidate.
All artifacts below are under the campaign root; prototypes remain isolated.

| Experiment / artifact | GPU / CPU analysis | Exact checks | Result and decision |
|---|---|---:|---|
|Early gate PDL, `adapter/`|7340099 /7340103|10,176|+0.120570/+0.160810us; retain.192 diagnostics. CPU7340052/7340095 prepared kernels/model cache.|
|Reset workers, `workers/`|7340275 /7340498|9,384|4 workers lose0.035640/0.021380us;1 loses0.107950/0.190440us. Retain8;288 diagnostics.7340278 analysis used stale repeat inputs, fixed without GPU rerun.|
|Unused DispatchDone removal, `retirement_complete/`|7340552 /7340565|6,256|+0.019260/-0.056060us mixed; not promoted. Removed41 EP8 updates in prototype only. Earlier7340412 stopped on missing diagnostic specialization; CPU7340544 completed cache.|
|Ahead reset, `ahead/`|7340736 /7340756|9,384|Ahead8 +0.126829/+0.121650us; ahead1 slower.288 diagnostics;16 current/32 ahead clears per48-call replay. CPU7340718 prepared ABI/caches. Abstract16,380 ordered sequences also passed, not device proof.|
|Norm preload, `preload/`|7341131 /7341147|6,256|+0.253960/+0.300141us; preserve independently, now model-selected.192 diagnostics. CPU7341103 prepared4norm variants/tests/SASS.|
|Addressing/sole publisher, `reset_followup/`|7341954 /7341958|12,512|Addressing-only loses0.007430/0.057549us; sole clearer loses0.082480us then gains0.018570us. Retain ahead8.384 diagnostics; CPU7341570 prepares isolated variants.|
|Norm-embedded reset, `norm_reset/`|7342779 /7343068|12,512|Regresses1.85–4.11us despite overlap. Rejected and source restored; archived `norm_reset_rejected_source.tar.gz`. CPU7342790 output-name collision recovered from saved traces.|
|Local readiness certificate, `certificate_compare/`|7343232 /7343236|6,256|Regresses0.208820/0.261340us;192 diagnostics. Reject peer-load skipping.|
|Gate-embedded reset / barrier-only, `gate_reset_compare/`|7343977 /7343980|9,384|Gate reset loses1.65–1.79us despite1.282us measured overlap; main entry~2us later. Barrier-only mixed. Neither promoted.|
|Software gate publication, `gate_signal_compare/`|7344400 /7344408|9,384|Gate-only loses1.771/1.748us; gate reset2.636/2.005us.192 ordering audits; reject separate returned ticket/software signal.|
|Packed CountsReady gate phase, `gate_packed_compare/`|7345160 /7345162|9,384|High16 gate-CTA count/low16 rows+8; non-returning release, token acquire, main one aggregate. Gate-only loses0.784–3.056us; gate reset1.536–1.989us.288 diagnostics/192 ordering audits; reject.|
|TMA1/8 and async completion, `reset_tma_compare/`|7345298 /7345300|15,640|TMA8 +0.036890/+0.164071us; TMA1 and vector/TMA non-returning worker0 aggregation mixed.480 diagnostics. Retain only TMA8 for confirmation.|
|Main-warp reset / TMA8 confirmation, `main_reset_compare/`|7345638 /7345654|9,384|Main warp1 loses4.034/3.862us; main grows~4us with entry unchanged. Reject. TMA8 +0.074910/+0.264430us confirms adapter gain.288 diagnostics.|

Main-warp prototype borrowed warp1 of the first8 main CTAs before its scheduler
walk; warp0 remained activation TMA issuer. No extra launch/barrier was added, but
startup became slower. Deferred reset behind a stable payload-tail-ready flag and
reduced per-warp acquire duplication remain untested ideas; no approval, promised
gain or pending experiment follows from recording them. Failed placements do not
prove an unattainable model saving.

## Resource and preparation lessons

- GPU7341284 was cancelled by146504 after performance capture, during alignment;
  user reports idle-GPU detection. API became ready, then sample log stayed empty
  before cancellation. No assistant gate existed; exact cause unproven. Avoidable
  work included prompt-cache miss, alignment Nsight and eager tokenizer import.
  Sampling now imports tokenizer only for scoring/comparison and flushes its
  request marker. Prebuild caches/prompts, use NSYS0 for alignment and score on CPU
  after release. The later complete sequences have no manual idle gate.
- Native/router ABI and device cache preparation must precede GPU use. Header
  timestamps can trigger host rebuild even after restoring identical contents.
  CPU7345858 built native/four routers and passed5 protocol tests, then failed on
  a nonexistent extra device donor. CPU7346155 reused those builds, completed
  device cache/17-module TVM union from proven `ahead_cpu/jit/cache` in1m54s.
- Prepared all-push router overrides retain ordinary pull routing in the same
  adapter test; never apply changed all-push ABI to pull. Source/cache freeze applies
  to every live consumer. Node/rank cache seeding and fail-fast missing-cache checks
  remain enabled; no automatic fallback or padded GPU work.
- Reuse numeric comparisons and reject incomplete/filtered contracts. Named-arm
  comparison preserves legacy RNE results. Saved diagnostic support includes
  norm/gate/reset ownership and gate acquires without a singular publisher stamp;
  do not fabricate a publication time for packed gate readiness.

## September22 progress staging

User requested cleanup, alignment verification and a commit. Scope excludes the
separate `CODEX_PROJECT_aga_bringup.md` and `codex_scripts/plot_megamoe_ep_scaling.py`
changes. Pre-cleanup19-file snapshot/patch is in `stage_progress/`; detailed earlier
working notes remain in that archive. Cleaned Python formatting/comments and
consolidated this chronology; measured CUDA source and runtime behavior unchanged.
CPU7357776 passed1m02s against the cleaned remote mirror:5 protocol tests,
stdlib-only sampling CLI startup,48 ahead-reset +48 packed-gate diagnostic records,
all90 matched performance samples (same numeric gains), and full1,632-token8K
alignment re-audit (same1 top10 miss under agreed max4). No new GPU rerun needed
for formatting/comment-only cleanup. Existing workspace/cluster instructions cover the lessons;
no global skill or AGENTS changes proposed.

## Deep cleanup and regression validation — complete September 21

Removed 27 obsolete scripts (77→50 tools), merged 12 optimization notes and 2,550
kernel-history lines from the mixed Qwen3 FFN memory, and removed exactly the two
new reports. Fifteen removed files were untracked. The existing all-push report
remains. Principal source edits remove 1,266 net lines: unreachable all-push branches,
legacy cycle timers, duplicated router code (259 shared lines extracted), and retired
sort/retirement diagnostics. Required readiness checks, IKET and recycling markers
remain. Cleanup preserved the existing index until the user's subsequent request
to commit the accumulated progress.

Candidate: `R/source/local-mirror`; frozen best control:
`R/source/dispatch_overlap_u32_final`. Same current harness, joined collective
preparation, control/candidate/candidate_repeat/control_repeat; all ordinary arms
precede profiles. EP8 rows 32 for both models, EP16 Qwen3 rows 12/Qwen3.5 rows 57, FP8×FP4,
150 SMs, bucket 3840. Each arm has 1,000 ordinary calls and 3 profiles per rank.

Both GPU jobs completed 0:0: EP8 **7338830** (26:20), EP16 **7338944** (23:45).
CPU analyses **7339425/7339424** pass: **192,000 ordinary rank samples**, **576
profile samples**, 48 reference recordings, 528 byte-exact comparisons and 576 exact
replay checks. All rank/call coverage, graph identity, preparation-before-metadata,
event-pool and phase-marker audits pass. No sample filtering.

Ordinary mean rank-local producer-start→main-end latency (µs; lower is better).
The two order deltas retain both passes, not just their pooled mean:

| Case | Frozen best | Cleaned | Change | Order 1 / order 2 delta µs |
|---|---:|---:|---:|---:|
| Qwen3 EP8 | 45.909549 | 45.947939 | +0.0836% | +0.073616 / +0.003163 |
| Qwen3.5 EP8 | 89.301682 | 89.346215 | +0.0499% | +0.021733 / +0.067334 |
| Qwen3 EP16 | 32.664039 | 32.662118 | -0.0059% | +0.000775 / -0.004616 |
| Qwen3.5 EP16 | 66.405535 | 66.379669 | -0.0390% | -0.002766 / -0.048966 |

Performance is effectively maintained: pooled changes -0.039% to +0.084%, at most
0.045 µs absolute. The positive EP8 differences remain reported; this is not a
literal zero-difference or intrinsic-speedup claim. CPU instruction review 7339485
finds exactly one changed machine word in each of the four ordinary main kernels:
the timeout assertion's source-line immediate moved from 2,281 to 1,706. Every other encoded
instruction and scheduling word matches. All 8 ordinary/IKET resource footprints
(registers, stack, shared and local memory) also match. Router binaries were not
part of this machine-word comparison; they are covered by the matched execution.

Instrumented phase means, control / cleaned (µs; separate executions from ordinary
timing). These are rank-local phase intervals, not additive wire-latency estimates:

| Case | Precompute dispatch exposure | Compute span | Postcompute combine exposure | Post-combine tail |
|---|---:|---:|---:|---:|
| Qwen3 EP8 | 6.327 / 6.437 | 25.293 / 25.332 | 2.803 / 2.831 | 0.208 / 0.221 |
| Qwen3.5 EP8 | 7.117 / 7.101 | 66.169 / 66.026 | 2.363 / 2.280 | 2.905 / 2.977 |
| Qwen3 EP16 | 6.409 / 6.273 | 14.643 / 14.644 | 2.624 / 2.655 | 0.340 / 0.319 |
| Qwen3.5 EP16 | 9.529 / 9.477 | 37.955 / 37.959 | 3.437 / 3.410 | 0.312 / 0.345 |

Validation preparation: native/router and all 16 control/candidate main builds
succeeded; 35 remote CPU tests and four retained pull builds passed. Two stale test
assertions were corrected for the shared header and Lustre aliases. Saved decoder
replays reproduce all four historical captures exactly after fixing temporary fixture
layout/schema mistakes. Five local workflow tests pass after the ordinary parser
was extended to explicitly require collective/two int resets/collective in joined
graphs; unknown nodes, wrong streams and late preparation remain errors. GPU code
stayed frozen throughout. Failed CPU preparations/first analysis and cancelled
zero-runtime dependency plans retain their actual states in the task records.

Evidence and manifests: `scratch/megamoe/deep_cleanup_20260921/validation_results.json`
and `ep{8,16}/comparison_summary.json`; matching `R/artifacts/deep_cleanup_20260921`.
Raw traces remain remote. The CPU correction used a frozen `analysis_workflow`
artifact while GPU source was in use. The local original-file archive has 53 members;
remote touched-file backups are under `source_before_sync`. No model E2E or new
alignment claim is made for this kernel cleanup. Root AGENTS.md question remains
unanswered; the user-provided instructions governed the work.

## Base kernel selection and September21 evidence

- Maintained all-push: u32 normalization producer, peer-mask activation dispatch,
  original scheduler. Remote frozen source: `source/dispatch_overlap_u32_final`.
  Pull remains the default; select `MINISGL_MEGAMOE_PAYLOAD_MODE=all_push` or pass
  `payload_mode="all_push"` to `MegaMoESymmBuffer`.
- Latest kernel gate: September21 cleanup passes all four matched Qwen3/Qwen3.5
  EP8/EP16 cases above. Earlier u32/peer-mask gains remain the selected algorithm.
  The later reset/PDL and preload model results are recorded above.
- Historical model-tested snapshot: `source/all_push_peer_ready_rne_20260920`.
  Qwen3 A3:F1, EP8, B4, 128K decode graph improves 0.121037 ms (0.668%) against
  matched pull. Its 8K alignment has one top-10 miss in 1,632 tokens, accepted
  under the user's maximum-four-miss criterion; it is not strict zero-miss agreement.
- Current decomposition: `all_push_joined_decomposition_20260921`, 164 Qwen3
  EP2–64 cases and 300 Qwen3.5 EP4–64 cases, all completed and audited. Preparation
  and workload share a CUDA graph; full profiling probes and serving kernels
  stayed unchanged. Earlier separate-graph reports are historical.

Local evidence lives under `scratch/megamoe/`. Remote root R is
`/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD`; campaign artifacts use
`R/artifacts/`, frozen source trees use `R/source/`. Large raw captures generally
remain remote. Use plain `ssh oci-hsg` for execution and `oci-hsg-data` for explicitly
needed transfers. Consult [cluster policy](CODEX_CLUSTERS.md) before remote work.

## Implementation and invariants

### Dispatch and metadata

The all-push router publishes routes and scale factors while MegaMoE pushes
deduplicated activations. Receivers validate every required transport word and
materialize independent prefix, middle and tail K windows. Eight-row TMA consumes
materialized activations; SF readiness follows route publication. Native TMA
descriptors and device transaction bytes must agree: a header-only override does
not establish compatibility. Pull has its own router, descriptors and device path.

The CTA selector hands selected IDs to independent SF writers, issues remote
expert-slot claims, then executes the original full-32-lane exponential/XOR-sum/
division normalization. Source-indexed scores do not depend on the returned slot.
Four shape constants come from the host; the bank epoch remains a live device read.
Full-CTA join, CountsReady, PDL, last-CTA GPU acqrel ticket and system release retain
their ordering. This is not an earlier-main-release protocol.

All-push claims update the low u32 of the existing count word. Bounds
`rows <= bucket` and `ranks * bucket * top_k <= UINT32_MAX` cover every claim,
including duplicate mapped experts. Full-word reset precedes epoch reuse; high32
stays zero. Consumers acquire all-source publication before reading immutable
counts. Other packed count/completion operations remain u64. Diagnostic bounds
and assertions remain enabled. Static instruction distance is not measured latency
hiding, and not every score store precedes the first returned-slot bounds use.

Readiness certifies only the bytes explicitly covered by its protocol. One changed
word cannot certify an unordered whole payload; middle-window publication does not
certify prefix or tail. Materialization avoids TMA rereads of a concurrently written
inbox. Sender issue, destination completion and consumer observation are distinct.

### Model storage, recycling and PDL

Prefill explicitly uses pull. All-push decode requests 1,024 rows per FFN rank,
rounded to 1,152 physical rows, independent of prompt size. Source and padded
expert rows must fit; receive banks, intermediates and top-k storage are additional.
There are two allocations per lane, each with two internal activation banks.
Decode uses CUDA graph replay, with no maintained batch chunker or eager fallback.
The fixed reset prefix covers the largest captured decode bucket.

Initial setup clears both banks. Eight auxiliary selector CTAs perform recycling:

1. Each latches epoch g, acquires prior local/assigned-peer payload readiness,
   release-adds to CountsReady and releases outgoing PDL before clearing.
2. Token CTAs wait for current logits and publish rows/SF. Recycled main senders
   acquire CountsReady = physical rows + 8, joining input and destination readiness;
   non-recycled senders retain the physical-row count.
3. Auxiliaries clear activation sentinels in bank g+1 with partitioned 128-bit stores.
   The last acqrel ticket resets the join and system-release publishes readiness
   for g+1. SF is not cleared: prior readers have retired before reuse.
4. Main retains route/metadata retirement. Current main and norm use bank g and
   do not depend on the bank g+1 publication. No separate maintenance launch remains.

Explicit release/acquire provides ordering; PDL launch gating is not a memory
fence. Dependencies decrease generation, so correctness does not require concurrent
scheduling. Ordered allocation reuse and all-rank route publication establish old
activation/SF reader retirement.

Main releases outgoing PDL after MMA issue, before the final TMEM-empty wait and
remaining combine/output work. Qwen3 norm preloads residuals, then every thread
performs hardware dependency wait before consuming MegaMoE output. Norm arithmetic
is preserved. The old software output-publication stamp is unused. Useful residual
preload before the last combine store is evidence of overlap; kernel-envelope
overlap or early reset completion alone is insufficient.

### Bounds and source map

Supported all-push shapes include one expert wave, BLOCK_M <= 32, hidden > 1,024
divisible by 512, at least 256 epilogue threads, sufficient dispatch CTAs and padded
materialization capacity. The concurrent-SF CTA selectors support renormalized
128-expert/top-8 and 512-expert/top-10. The 33-row refill case uses 32-row tiles.
Unsupported shapes fail before launch; no automatic fallback is provided.

The FP4 checkpoint quantizer uses nearest-even midpoint rounding, including 0.75,
1.75 and 3.5 magnitude ties. CPU oracle comparison covered 11.2M values without
differences; this does not replace model alignment.

| Component | Maintained path under `python/minisgl/` |
|---|---|
| Producer | `kernel/csrc/jit/gate_topk_fused_all_push.cu` |
| Device dispatch/main/recycle | `kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/mega_moe_all_push_dispatch.cuh`, `sm100_fp8_fp4_mega_moe_all_push.cuh`, `mega_moe_recycle.cuh` in that directory |
| Layout and launch | `kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/layout/mega_moe.cuh`, `kernel/csrc/deepgemm/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` |
| Python buffer/adapter | `kernel/megamoe_mega.py`, `moe/megamoe_afd.py` |
| Following norm | `kernel/megamoe_output_norm.py`, `kernel/csrc/jit/megamoe_output_norm.cu` |
| Model integration | `afd_fmha_runtime.py`, `afd_support.py` |

## Accepted measurements

### Model performance and alignment, September 20

Frozen RNE peer-ready source; GPU7297998 and CPU7298001 completed successfully.
Qwen3 A3:F1/EP8/B4, 128K performance mode, 94 layers, two microbatches, six
forward/reverse arms; all 90 ordinary samples retained:

| Arm | Mean complete decode graph, ms |
|---|---:|
| Pull | 18.119223467 |
| All-push without tail PDL | 18.031638700 |
| All-push with tail PDL | 17.998186033 |

Path saving is 0.087584767 ms; pooled PDL saving is 0.033452667 ms; combined saving
is 0.121037433 ms (0.668%). Combined forward/reverse gains are 0.079858800 and
0.162216067 ms. Separate PDL effects vary by order (-0.013205733/+0.080111067 ms
saved); do not claim PDL improves both orders. The user's approximate 0.5 ms was
not a required threshold. Metric is complete decode graph span over profiled
attention/model roles, not client latency or a maximum across every GPU.

Ordinary model-rank main improves 36.729731→34.983349 µs; no-PDL main is
35.279605 µs. Main/norm overlap averages 2.366161 µs in 2,790/2,820 pairs;
15 deferred final pairs require separate critical-path interpretation.
Separate 22,560-call trace passes all combine-writer/output and eight-worker/reset
audits, with zero late readiness. Residual preload precedes last combine store
in 3,754 calls (16.640%). Minimum clear/readiness margins before norm end are
42.528/40.544 µs. A common-clock-offset bound proves reset clear before main in
all 2,820 FFN0 calls (offset width 1,359 ns); this is not token-CTA concurrency proof.
Profiler tags are union rank + 1: tag25 is FFN0/global rank24.

September21 retrospective: the recorded model case is A3:F1, not A3:F3.
From `rne_validation_analysis/evidence_summary.json`, ordinary forward FFN0 main
saves 1.746382270 µs/call. At 94 layers × 2 microbatches = 188 calls/step,
the additive main-only projection is 0.328319867 ms (1.812% of pooled pull).
Producer-start→main-end saves 1.518990780 µs/call, projecting 0.285570267 ms.
Neither projection is an E2E measurement or includes a separately established PDL
critical-path saving; do not add overlapping kernel envelopes. Actual pooled
graph gain is 0.121037433 ms, 36.866% of the main-only projection. EP8 ranks
execute concurrently, so do not multiply the projection by eight. Earlier
pre-RNE model gain was 0.197694467 ms; final corrected-source evidence supersedes it.

Final nominal-8K generation has 96 prompts × 17 tokens = 1,632. Fresh reference
7298334 scored all prompts after the performance allocation released. Coverage,
prompt/token/position identity and finite logprobs pass. Top10: 1,631/1,632
(99.9387%); top1: 1,398/1,632 (85.6618%). Sole miss: zero-based prompt92,
position11, token6149, reference rank17. Accepted with explicit allowance4;
legacy strict-zero-miss wrapper exits1, while the configured full audit passes.
Earlier pull and pre-RNE sources had different misses and must not be relabeled.

Evidence: `final_all_push_20260919/signal_pdl/{rne_validation,rne_validation_analysis,
rne_score,reset_overlap_bounds,promotion_candidate}` under local/remote artifacts.
The retained [all-push report](reports/megamoe/all_push_20260919.md) includes older
checkpoints; this section supersedes its historical pending work.

### Final dispatch selection, September 21

Ordinary mean rank-local producer-start→main-end is authoritative. EP8 original
cases use rows32 for both models; EP16 uses Q3 rows12 and Q35 rows57. FP8×FP4,
150 SMs, mirrored model-interleaved arms, 1,000 ordinary calls/rank/arm and three
separate profiles. Deltas below are candidate minus same-allocation control (µs).

| Candidate / GPU and CPU jobs | Q3 EP8 | Q35 EP8 | Decision |
|---|---:|---:|---|
| Peer masks, 7309711/7309713 | -0.047950 | -0.144554 | Retained; Q3 orders mixed |
| Score only, 7316518/7316546 | -0.029599 | -0.090148 | Q3 orders mixed |
| Host layout, 7317126/7317133 | +0.018726 | -0.057496 | No common win |
| u64 normalization, 7317641/7318019 | -0.146858 | -0.155725 | Ordinary pass; compact profile failed |
| Normalization repeat, 7318156/7318171 | +0.044620 | -0.222436 | Q3 orders mixed |
| Warp0 publication, same run | +0.116496 | -0.054670 | Rejected |
| u32 normalization, 7319282/7319303 | -0.184615 | -0.129856 | Full pass, both orders |
| u32 repeat, 7319930/7320188 | -0.147614 | -0.380609 | Ordinary pass, both orders |
| u32 + FC2 placement, same repeat | -0.011717 | -0.660797 | Reject common replacement: worse than u32 on Q3 |

u32 isolated from u64 is +0.006246 µs Q3 (mixed orders), -0.036977 µs Q35;
do not attribute the whole normalization gain to slot width. CPU7318242 covered
32 dtype/balance functions; Q35 nonbalanced registers 40→38, stack32→24B,
two hot entry stores removed. CPU7318292 checked offline IKET allocation.

Final EP16 GPU7320262/CPU7320270 passes 128k ordinary spans, 384 profiles,
352 exact comparisons, 384 replays and module coverage. Q3 delta -0.098946 µs
(orders -0.064772/-0.133119; 12/16 ranks); Q35 -0.026283 µs (orders
-0.050865/-0.001700; 10/16 ranks). Profile publication advances 0.264/0.252 µs,
but first MMA is not consistently earlier. No new model E2E claim.

FC2 placement passed its 76-case/73,440-tile ownership model but lost on Q3.
Its first profile failed because CPU and GPU IKET include paths used different
filesystem aliases, changing JIT keys. Ordinary captures are valid; no placement
decomposition exists. EP16 FC1-only pairs would require explicit marker support
before any future placement profile. Preserve the useful Q35 result without adding
a model-specific branch. Evidence: `dispatch_overlap/{router_issue,compute_placement,
peer_mask}`; final selected frozen source remains `dispatch_overlap_u32_final`.

## Decomposition and measurement contract

### Current complete sweep

Campaign: `scratch/megamoe/all_push_joined_decomposition_20260921`, matching remote
artifact basename. Q3 164 cases/455 aliases; Q35 300 cases/1,111 aliases.
All 13,920 rank0 ordinary calls and 1,392 rank0 profiles pass coverage, placement
and phase-order checks; raw all-rank captures are retained. Seven GPU and seven
CPU jobs plus collector7333581 completed0:0, with five-minute monitoring and no
GPU failures/retries. Serving source/cache and full probes remained frozen.

| Group | GPU job | CPU analysis job |
|---|---|---|
| EP2 | 7333563 | 7333565 |
| EP4 first/second groups | 7333561 / 7333559 | 7333562 / 7333560 |
| EP8 | 7333555 | 7333557 |
| EP16 | 7333575 | 7333576 |
| EP32 | 7333577 | 7333579 |
| EP64 | 7333568 | 7333572 |

Final reports, relative to campaign:

- `qwen3/results/qwen3-final-all-push-megamoe-comparisons.html`
- `qwen35/results/qwen35-final-all-push-megamoe-comparisons.html`

Earlier September22 fixed-token scaling report (superseded above; archived in
the new campaign's `previous_plot/`): `plots/ep-scaling-fixed-tokens.html`, same
format as September12 pull report, 15 curves/62 points using unfiltered rank0
30-call ordinary medians. Reusable `plot_megamoe_ep_scaling.py` now accepts
`--report-subdir results`, `--label`, `--baseline-roots` and optional `--preview`;
source-report checks honor its saved filtering audit, while plotted CSV medians
remain unfiltered. All62 points/ideal anchors checked; chart PNG visually reviewed.
Historical baseline `comm_final_20260912/sweeps/{qwen3,qwen35}` matches all164/300
cases/contracts. Equal-case average latency reduction is **-7.872380%/-5.642619%**
(regression), pooled464 cases **-6.430724%**. Mean case-median latency is
81.486052→94.402049µs Q3 and103.108800→117.698082µs Q35; reduction of those means
is -15.850562%/-14.149405%, a different weighting. Losses concentrate at Q3EP2
(-25.743% mean reduction) andQ35EP4(-24.099%); EP>=8 mean reduction is
+15.783300%/+10.291559% over36/161 cases. Every higher-EP group improves on
average. This is cross-allocation historical evidence with different preparation,
not a paired causal code gain. Full per-case/by-EP data and formulas are retained
in `plots/ep-scaling-fixed-tokens.comparison.json` and embedded HTML provenance.

Original three views, aliases and shared-scale layout are preserved. Full EP64
unfiltered case-median exposed dispatch ranges are 6.784–9.248 µs Q3 and
6.048–10.432 µs Q35. Maximum-input displayed ordinary/profile/exposed values
after the requested individual-sample filter: Q3 20.784/20.992/9.536 µs;
Q35 20.096/22.816/10.512 µs. No native speedup follows just from report correction.

Only final HTML sample filtering was explicitly authorized. Per timing series,
exclude samples whose distance from median exceeds `max(3 * 1.4826 * MAD,
0.05 * median)`. Ordinary and IKET masks are independent; one IKET mask drives
every phase statistic. The retained real launch nearest median supplies the whole
timeline (ordinal breaks ties). No per-phase filters or duration rescaling.
Across 27 unique displayed cases, 6/810 ordinary and 4/81 IKET samples are excluded.
All cases/raw samples remain; exact ordinals, thresholds and counts are in each
`results/outlier_audit.json`. Raw `decomposition.csv` is unfiltered.

Independent raw→mask→phase/table/timeline checks pass in
`collect/report_sample_validation.json`; HTML data, axes, chronology and JavaScript
checks pass in `collect/html_validation.json`. No browser-validation claim.
Raw main-duration gaps remain: median absolute 0.768 µs Q3 / 1.56 µs Q35;
159/164 and 274/300 cases are within5%, 162/164 and 293/300 within10%.
See `collect/fidelity_audit.json`; distinct executions need not agree exactly.

### Why preparation changed

EP64's old 252-scalar-kernel preparation chain inflated readiness waits. Matched
GPU7308005/CPU7308006 replaced it with two collective kernels before metadata:
exposed dispatch median62.688→17.888 µs, ordinary rank0 main70.704→19.728 µs.
Both orders improved, but control variability remained. All 120 barrier placements,
768 startup samples and exact/replay checks passed. GPU7308509/CPU7308512 then
corrected all23 EP64 cases. This was benchmark preparation, not serving speedup.

Separate preparation/workload CUDA graphs still allowed a host launch gap after
the barrier. Matched full-probe EP8/16/64 comparisons then joined both in one graph.
EP8 all-rank exposed dispatch changed 10.2823→6.3523 µs Q3 and
12.4896→7.0124 µs Q35, improving both orders; compute remained similar.
Minimal-marker probes alone left substantial startup inflation. Native source
comparisons at EP16/32/64 found no large scaling regression that explained it.
Evidence: `profiling_fidelity_20260921/{ordinary_ep16,ordinary_ep32,ordinary_ep64,
subset_ep8,subset_ep64,joined_ep8,joined_ep16,joined_ep64}`. Preserve residual
execution variation rather than asserting universal profile/native agreement.

Use `MEGAMOE_DECOMPOSITION=1`, `MEGAMOE_SMOKE_PREPARE_BARRIER=collective` and
explicit `MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT=joined` for that contract. Default
layout remains separate. Joined outer CUDA-event duration includes preparation;
Nsight workload duration and IKET phases exclude it. Require barriers to finish
before the metadata selector on the same stream/graph; record the small preceding
TC-producer envelope overlap separately. Do not invent timestamp tolerance.
Separate layout retains stricter pre-TC placement. Metadata transfer stays measured.

### Interpretation rules

- Keep ordinary producer→main latency, whole main duration, instrumented phases,
  rankmax CUDA-event timing and model graph latency separate. Do not subtract
  profiler overhead or raw timestamps across unaligned GPU clocks.
- Keep both arm orders, every rank and first-call observations. Per-launch phase
  partitions may add; independently computed medians do not. Report repeat drift
  alongside small gains. The explicit final-HTML filter above is not a general
  license to filter optimization comparisons.
- Actual `MegaMmaIssue` differs from the old operand-ready `MegaComputeStart` proxy.
  Pair-local first-MMA→compute-end includes pipeline waits. `MegaCombineStart`
  marks first FC2 epilogue after accumulator readiness, not FC1 end or FC2 MMA start.
- For current top8/top10 H4096 CTA reduction, `MegaCombineEnd` follows contribution
  waits, reduction, output stores and epilogue join for the assigned loop. Latest
  rank0 marker is local completion, not a cross-rank barrier or system-visible ACK.
  Kernel end is latest warp exit; its residual tail has no measured role breakdown.
- Initialize/record the complete CUDA event pool before warmups/profiler start;
  lazy handle creation can stall a rank's launch loop. Empty ranks may lack a
  metadata origin; report coverage rather than dropping unexplained records.
- Initialization/reset is excluded from standalone transport timing. Recycling-chain
  benchmarks include reset and norm. Detailed model markers and host synchronization
  belong in separate diagnostic runs, never ordinary performance runs.

## Historical pull-kernel checkpoints

Kernel-only sections formerly mixed into `CODEX_PROJECT_qwen3_ffn_overheads.md`
are consolidated here. Its attention, model pipeline, capacity and full-model
campaign sections remain separate. These dated checkpoints explain retained
mechanisms; they are superseded by the current selection above. Original detailed
attempts, job ledgers and source paths remain in the cleanup archive, including the
complete pre-consolidation Qwen3 FFN memory.

| Checkpoint | Accepted result and limits |
|---|---|
| August25 same-rank FP8 | Added a distinct FP8-weight specialization and graph-stable per-lane symmetric buffers. One collective kernel replaces split AG/EG communication/expert work; router/input staging remains separate. Final exact A7 strict result28.1574149333ms versus29.3581909333ms (-4.09%). The later M32 wave guard reached80.446–80.477µs on target fixed/skew routes, without a new exact A7 E2E claim. Final A1 scorer6552212: all408 tokens top1. |
| August26 FP8×FP4 | `6c210cf` added production weight conversion and propagated the real two-lane budget through the host API. Job6561772 passed numeric/concurrent-lane checks; concurrent replay was1.53× sequential for64 rows. Exact A1 job6561851:22.689653667ms versus FP8 baseline23.3908480ms (-2.998%); scorer6562471 all408 tokens top1. |
| August27 fused routes and rank-gated combine | Retained route preparation in the top-k selector with its acquire/release last-block ticket. Final post-revert job6600780:94.906→91.861µs, bitwise exact. Two redundant system fences had previously erased the gain. |
| Release-published routes | `61ec1a8` improved A7/A1/EP8 proxies by5.56/6.82/7.38%, all bitwise correct. Exact A7/EP8 job6603099:23.613828733ms,15/15 retained steps,64 reports. |
| Rank-ready aggregation/count cache | Repaired shared-count publication passed two-input checks and nonzero MMA/TMA work; stale-zero scheduler attempts are invalid. Exact job6610567:23.127204867ms (-2.061% versus6603099). Dispatch shared count cache further improved A7/EP8 proxies; its unrolled final-rank ownership scan is intentional. Final scorer6617291 all408 tokens top1. |
| August29 cleanup `9a84a5a` | Kept the participant cleanup ticket while removing dead stats/timing/debug-zero/API state and simplifying host ownership. Paired job6688914: A7/EP4 24.628219733→24.589484600ms (-0.157%); A3/EP8 22.942197333→22.937496467ms (-0.020%). All15/15 steps retained per row; scorer6690783 all408 tokens top1. |
| Cleanup42-case matrix | `9a84a5a` completed the unchanged128K near-Pareto matrix,42/42 rows and1,512 reports. Against prior `f14d8005`,35/42 latency wins and -0.314% geometric mean, but separate allocations do not prove intrinsic kernel speedup. Two batch2 rows regress+0.903/+0.607%; retain those observations. |
| September9 ordered combine/early weights | Retained early weight reads, CTA bulk-TMA combine with original ascending valid-slot PTX accumulation, and short-K producer prefill correction. Exact job7019473 passed8 ranks and independent quantization. Profile7019787: ordinary kernel median44.064µs; instrumented dispatch10.304µs, compute26.336µs, exposed combine7.296µs, post-combine tail0.512µs median/0.800µs max. |
| September9 model gate | A1:F1/EP8/b6/128K sample7019832 and scorer7020231 passed all816 tokens top1. Same-allocation perf7019851:24.098079→23.553900ms (-2.258%). Historical policy retained14/15 control steps and15/15 candidate steps; this is not the current all-samples kernel contract. |

Important rejected directions and interpretation:

- Removing participant cleanup synchronization passed small kernels but worsened
  A7 model latency+0.399% in6684906; trace6687527 found slower `wait_ready` cadence
  despite faster MegaMoE. Restoring the ticket enabled accepted `9a84a5a`.
  Stacking fixed-fast-math cleanup also regressed despite isolated small wins.
- Expert-gated combine with system atomics per tile was correct but36.6% slower;
  mixed-scope and separate-epoch GPU aggregation lost readiness on repeated graphs.
  Last-SM/resettable publisher variants timed out. Do not revive them as cleanup.
- Single-lane rank acquire and per-cluster DSM aggregation regressed EP8; layouts
  that improved dispatch alone lost full-kernel time. Active-mask optimization won
  A7 but lost A1. Preserve topology-independent correctness and compare all target
  cases. Replacing the intentional unrolled count-cache store regressed EP8.
- Pairwise top-k8 summation changed numerical results and was removed. Prefix
  lookup's0.112µs apparent gain was smaller than0.416µs control drift. Direct
  register-copy dispatch's confirming24-row comparison was exactly neutral at
  46.336µs and nearby48-row mean regressed; TMA staging/reuse remains intentional.
- The August31 `clock64` decomposition (6740199) was diagnostic: control/instrumented
  medians119.296/126.304µs, +5.874% overhead. Its overlapping phase envelopes and
  calibrated clocks are historical, and its legacy timer code is removed by this
  cleanup. Current IKET decomposition and ordinary timing supersede that workflow.
- Global allocations are not proof of HBM traffic; Linear-1 staging is not hardware
  L1. TMA avoids register staging and supports reuse across GEMM N tiles. Removing a
  hop can increase remote reads or LSU work. A phase intersection is not a causal
  stall attribution; overlapping phases cannot be summed into communication cost.

Compact evidence: `scratch/megamoe_latency_20260908/RESULTS.md` and
`results/{final_e2e,final_ordered_iket,exact_results}` beneath that task;
`scratch/megamoe_cleanup_20260828/alignment/final_9a84a5a`;
`scratch/afd_128k_optimized_attention_near_pareto_cleanup_9a84a5a_20260829`;
`scratch/megamoe_fp8_fp4_decomposition_20260831`. Historical remote worktree paths
may have been retired under the source-lifecycle policy; artifacts remain evidence.
The earlier fixed-work EP4/8/16 fit had22.128µs intercept, not a pure wire-latency floor.
The historical A8/EP8/b6/128K opportunity assessment used24 rows/rank and12 routes
per expert: perfect-routing model23.100906ms versus normal23.255805ms (-0.666%).
Its47.936→45.104µs CTA-combine comparison motivated later ordered accumulation;
it does not supersede the final numerical or model gates above. Physical-path probes
and failed marker captures are archived; measured sender/receiver readiness and
full-chain timing remain distinct from static instruction counts or bandwidth estimates.

## Historical decisions and failed attempts

These are results to preserve, not pending experiments. Complete original notes
and script versions are in the cleanup archive below; artifact paths retain raw
evidence and frozen controls.

| Campaign / idea | Decision and evidence |
|---|---|
| September12–13 communication selection | Dense metadata push + payload pull + in-band push combine retained; no meaningful model step gain. `comm_final_20260912`, remote `ffn_comm_final_20260912`. Dispatch and combine are independent axes. |
| Dense direct vector/bulk push | Same final contiguous layout, exact checks passed, but full path regressed12.15–19.08% /10.25–17.82%. Direction reversal also changes address/completion ownership; not proof all push protocols fail. `dispatch_incremental_20260914`. |
| Compact/packed-source/early-store dispatch | Dispatch-only gains did not establish producer→main benefit. Early-store failed all six dispatch targets; Q35 lost against compact. Restored dense pull; no phase-only promotion. |
| September15 cleanup retest | 24 cases, four pull sources, mirrored2000-call comparisons; all48 cleaned-vs-precleanup measurements within -0.168..+0.154%. Prior three regressions did not recur; kept cleanup. `final_cleanup_20260915/retest`. |
| September15 serving | Q3 graph18.329570067ms; Q35 EP16 22.643118733ms; EP32 22.618707933ms, near historical parity. Historical stacks/allocations differ. Q3 strict top10 1630/1632 and Q35 10878/10880 fail; causes unisolated. Actual raw prompt lengths8184/8182 versus nominal8192. `final_serving_cases_20260915`. |
| Async readiness publication | Retained `red.async.release.gpu.global.add.u32` with completion/ordering preserved; EP8 useful gains, EP16 modest/near-neutral. SASS supports cheaper publication, not increased NVLink bandwidth. `comm_alignment/async_release_ep8` and `async_release_ep16`. |
| Combine pointer prefetch | Retained earlier metadata/pointer work; EP8 M32 rows21/32 save0.1004/0.1522µs in unprofiled max-rank comparison; EP16 rows12 saves0.21898µs. Other shapes near neutral. `comm_alignment/combine_pointer_ep8` and `combine_pointer_ep16`. |
| Remote BF16 L2 reduction vs Lamport | Opt-in ablation, no common promotion. Excluding output zero-fill per user scope, EP8 rank0 chain improves1.869–2.799%; EP16 rows4/6/8 regress1.142/0.556/0.434%. BF16 order differs from FP32 sum; bounded synthetic correctness is not model accuracy. Lightweight IKET still fails to reproduce ordinary gain consistently. `comm_alignment/remote_reduce_ablation`. |
| TMA/SF padding | Exact checks passed but larger EP8 rows28 regressed5.16%; no promotion or EP16 phase run. `comm_alignment/tma_sf_ep8`. |
| Full1024 model reset | 128K ABBA regressed38.19%; prefix reset reduced regression to0.601%. One retiring reset CTA did not hide reuse; four/eight improved adapter overlap, then selector auxiliaries provided accepted model path. `final_all_push_20260919/signal_pdl`. |
| Early software-output alias | Target byte28 collided with router retirement; eager equality failed, noPDL attempt timed out. Stable target and later hardware dependency design supersede it. Failed frozen sources retained. |
| Separate maintenance / dependency wait / router ordering | Screened through complete chain/model comparisons; no general replacement. Earlier pre-RNE0.197694467ms gain is superseded by final RNE0.121037433ms result. |
| Receiver-local route sorting | Single helper, source-prefix, expert-direct and source-local-atomic variants all regress both EP8 models in both orders/on every rank. Q3 penalties+1.611/+4.130/+1.887/+1.522µs; Q35+3.531/+4.712/+3.278/+1.676µs. Earlier main entry was outweighed by longer main. `dispatch_overlap/receiver_sort`. |
| Scheduler and publication alternatives | Uniform FC2 balancing, warp0 publish and common FC2 placement rejected. Early CountsReady/PDL lost1.336µs; full-producer payload delayed metadata and lost~1.65µs. No shared improvement from host layout. |

Preparation lessons: CPU and GPU IKET include spelling must match literally;
filesystem aliases affect cache keys. Match `SLURM_CPUS_PER_TASK` with inherited
`SLURM_TRES_PER_TASK` for validation subprocesses. CPU compilation is not TMA-byte,
ABI or device-correctness proof. Reduced-role register budgets and actual local
memory instructions need matched code review, not blanket zero-text-difference
rules. Overlay replacement ancestors must be owned, not symlinks into controls.
Freeze source/cache while jobs consume them. Persist reference shards before strict
scoring gates; Qwen3.5 reference scoring requires `--language-model-only`.

Older Qwen3.5 decomposition had299 cases/1,110 aliases; current300 includes the
supplemental EP32 batch40 input. Top10 CTA combine required `max(2*warps, topk)`
barriers with matching host allocation. EP64 exact-combine checks use an independent
conventional reference arena; disabling rank-gating while retaining incompatible
in-band geometry is not a valid reference. Grouped per-EP capture avoids repeated
container teardown and unused reference construction that previously wasted GPUs.

## Remaining ideas

No experiment is pending or authorized by this list. Preserve useful paths until
same-code comparisons demonstrate a replacement.

- Admit ready FC1 work while holding a bounded unready FC2 task; prove producer
  progress, tile ownership, pair adjacency and accumulation order. FC2 placement
  already lost as a common replacement; its Q35-only result is research evidence.
  Source audit: warp(kNumDispatchWarps+3) is idle after reg-deallocation, but current
  declared budget48*128+40*128+208*256=64512 already reaches the kernel's bound.
  A common task publisher must fit the40-register role or justify rebudgeting, plus
  shared descriptor/barrier storage. Existing block_idx carries across FC1/FC2;
  independently restarting each stream from blockIdx.x changes ownership. A pair
  descriptor must retain distinct adjacent N coordinates, with common expert/M.
  All role consumers need one published order and activation credits through FC2;
  The archived ready_scheduler62 prototype implemented this common queue;
  CPU/codegen and focused device checks passed, but all six paired kernel means
  regressed; this queue is rejected. Retirement-backed replacement remains CPU-only.
  Host review C/scheduler_interleave/review_schedule.py now passes288configs/
  440640CTAtiles (empty/uniform/ragged M counts, multiple wave sizes and grids).
  Independent carried-cursor and flattened-order models agree; filtering the
  original per-CTA order into FC1/FC2 streams preserves owners and adjacent N
  pairs. Fair pair service selecting ready FC2 else FC1 completes all workloads,
  with126142FC2 selections ahead of pending FC1. Tasks complete atomically in
  this model: bounded queues, publication, role progress, ring credits, initial
  scheduling and interleave-M are not proved. Subsequent queue model adds96cases/
  50800descriptor reads; device results are in the rejected queue table above.
- Explore independent weight and activation/SF pipeline depths at fixed shared
  memory. Earlier next-stage peek failed register/codegen gates; reduce state and
  prove ring-wrap/barrier byte/phase ordering before another attempt.
- Reduce remaining returned-slot/address dependencies without changing work
  assignment. Preserve canonical wave cursors; blockIdx alone is insufficient.
- Advance complete operand windows or bound remote issue pressure only with a
  cheaper handoff and useful fanout evidence. More issuers, repeated source loads,
  divergent polling and blind pacing previously lost.
- Independent combine vectors or separate load/reduce roles may avoid head-of-line
  waits if top-k summation order and register limits survive. Last output issue→exit
  was about0.23–0.31µs; final MMA reuse wait averaged0.072µs. Neither demonstrates
  a hidden1–2µs ACK. Added retirement probes increased exposed combine0.237µs.
- Batch chunking is separate unvalidated work: stable global route IDs, equal rounds
  for empty ranks, output copies before reuse and worst-case padded capacity are
  required. Current explicit bounds remain intentional.

Reference designs reside in sibling `../cutedsl_megamoe/next/sources/` (generation
token communication, asymmetric FC12 pipelines and Blackwell FC2 epilogue).
Rubin constants and cluster-local barriers are not GB200 tuning results or
cross-GPU completion mechanisms. Do not repeat serial SF release, four-row
validation, accumulator preload, descriptor batching, deferred clear or warp-local
combine masks without a materially different explanation and comparison.

## Workflow and cleanup record

Read [workflow inventory](codex_scripts/README.md) for the retained tools.
Prepare and analyze on CPU allocations (`cpu-normal` on HSG); GPU allocations
(`short`) must execute complete validation/performance workloads. A complete kernel
or producer-consumer benchmark qualifies; model E2E is a later gate. No GPU debugging,
manual idle waiting, padding or automatic retries. Monitor recorded jobs at a useful
cadence (latest sweeps300s), and resolve failures from saved evidence before reruns.
No subagents unless explicitly requested; no py_check, manual hash/version checks,
or `/tmp` artifacts.

September21 cleanup merges12 optimization documents into this file and replaces
the repetitive root timeline with a project index. Twenty-seven superseded scripts were
removed after checking retained source/test/workflow dependencies (77→50 tools). The two new
reports (dispatch-overlap and profiling-fidelity) were removed per user correction;
the existing all-push report remains. The standalone profiling-fidelity memory and
the duplicated in-tree all-push design note were absorbed here. Mixed model/attention,
reproduction, cluster and Pareto memories retain their own scope. The mixed Qwen3
FFN memory also drops2,550 lines of kernel-history sections, condensed into the
historical pull checkpoints above; its model/attention sections remain.

Original working-tree versions, including uncommitted/untracked content, are preserved
once in `scratch/megamoe/deep_cleanup_20260921/before_cleanup.tar.gz` (repository-relative
member names). `cleanup_manifest.json` beside it records exact removals and validation.
Historical frozen campaign workflows remain unchanged; restore a specific archived
helper if reproducing an old plan that requires it.

Final mirror sync used 101 explicit maintained files and removed/confirmed absent 41
obsolete paths after all consumers completed. `tested_source.patch` records relevant
staged, unstaged and untracked contents before the progress commit.

September21 progress checkpoint, explicitly requested after cleanup: commit the
accepted all-push kernel and model integration, maintained profiling/comparison
tools, shared cluster reference and consolidated documentation. Leave the separate
AGA bring-up note uncommitted. Reuse the completed remote build, exactness, replay,
decomposition and paired-performance gates above; no executable source changed
after those gates. Sync final checkpoint documentation to the HSG mirror. Existing
project/skill instructions cover the workflow; no global skill or AGENTS changes
are needed.

September28 16:43UTC monitor update: EP8 GPU7514500 COMPLETED58m14s0:0;
analysis7514520 RUNNING1m49s, audit pending. EP16 GPU7514502 RUNNING24m53s;
EP32 GPU7514504 now RUNNING; EP64 GPU7514506 pendingQOSMaxJobsPerUserLimit.
Existing300s monitors continue; CPU analysis64 waits for a submit slot.
No additional GPU sweep or performance rejection based on pull/4K/8K.
