## Checkpoint: MegaMoE all-push latency optimization — September 18

User requested a progress commit. Default payload mode remains pull. The maintained
compact-push kernel is the earlier split-gather prototype; the measured best is
an explicit header/router/native-descriptor overlay, now retained with compact
evidence under `scratch/megamoe/lamport_payload/`. See
[dispatch memory](CODEX_PROJECT_push_dispatch.md) and the
[checkpoint source map](scratch/megamoe/lamport_payload/CHECKPOINT.md).

Confirmed Qwen3 EP8 (global256, rows32, H4096/I1536/E128/topk8): ordinary
MegaMoE-only helper37.16655 vs previous-best37.24400us (-0.208%, pure ABBA).
Across the two EP8 jobs,144 exact and144 replay checks passed. Three instrumented
captures give independent medians, metadata start=0: exposed dispatch8.928us,
first MMA→last accumulator25.280us, exposed combine4.960us, cleanup0.192us;
metadata→combine-end39.712us and metadata→kernel-end39.936us. Independent
medians need not sum. Compute is near historical pull25.024us.

The 37.167us ordinary kernel duration excludes metadata before kernel entry;
metadata-origin decomposition and producer+kernel timing include that work.
All buffer initialization/reset is outside timing. Instrumented and ordinary
measurements are separate scopes, not subtractable overhead estimates.

Unchanged-code EP16 ABBA: Qwen3 global192/rows12 helper23.90270 vs24.22975us
(-1.35%); Qwen3.5 global512/rows32 helper49.30880 vs49.56815us (-0.52%).
Both orders improve, but Qwen3.5 remains0.68% slower than historical pull48.97440us.
32 reference records,128 exact comparisons and160 replay checks passed. No EP16
phase captures. Qwen3.5 global912/rows57 is unsupported (M64/128 epilogue threads);
current overlay requires M<=32/256 epilogue threads, one expert wave, and prepared
capacity. No consistent full-chain gain or general default promotion is claimed.

CPU7256953/7259660 passed ordinary/instrumented builds and role register gates:
128 registers,24 stack bytes,zero spills; the narrow helper has no local memory
or profiling markers. GPU7257013/7257222/7259728 completed successfully. Preserve
previous-best1536 and historical pull controls. All jobs are terminal. Detailed
attempts, protocol proof, rejected paths and remaining ideas are in dispatch memory;
communication ideas remain in [MEGAMOE_COMM_EXPLORATION.md](MEGAMOE_COMM_EXPLORATION.md).

Checkpoint verification (September 18): synced the candidate tools/code and
selected source/evidence assets to HSG `source/local-mirror`; remote `bash -n`
passed for `profile_megamoe_iket.sbatch`. The current latency summarizer replayed
all four saved EP8 arms (three captures each) with ordinary-pass validation and
produced `remainder_helper/checkpoint_comparison.{json,md}` on HSG. No GPU rerun
was needed: kernel sources were unchanged from the passing jobs above. Git
whitespace review passed after documentation cleanup. No global instruction or
skill change is needed; workflow lessons are recorded in project memory.

## Checkpoint: script consolidation and accumulated progress — September 15

User requested cleaning `codex_scripts` and committing all our progress.
Included documented MegaMoE work and the earlier staged AFD attention/GDN cleanup.
Scripts reduced **53 → 44**, removing 1,031 lines; see
[codex_scripts/README.md](codex_scripts/README.md) for retained commands/migrations.
- `submit_plan.py sbatch|afd` replaces the two submission helpers; records/logs,
  AFD dry runs and duplicate-submission refusal preserved. Scheduler stderr is
  logged separately from parsing the job ID.
- `seed_native_cache.py` combines DeepGEMM and TVM seeding, preserving locked
  host-module links/copies, TVM node/rank module links and overwrite refusal.
- Removed superseded automatic validation/report runners, three experimental
  marker analyzers and the standalone experimental Lamport transport probe.
  Moved the used full-path audit unchanged into `compare_megamoe_fullchain.py`.
- Archived every pre-cleanup script under
  `scratch/megamoe/commit_cleanup_20260915/scripts_before` and the matching remote
  artifact root. Historical source/workflow snapshots remain intact.

Validation against the synced remote candidate: **65 CPU tests PASS in2.092s**,
all retained shell launchers PASS syntax, three actual saved cases from7150076
replay identical gain/component values. Initial stale `kPush` source assertion
updated to `kCompactPush`; no runtime change for that fix. Existing GPU evidence
below applies; no new GPU jobs, py_check, hash/version check or kernel edits here.
Evidence: `scratch/megamoe/commit_cleanup_20260915/{cleanup_manifest.json,checks}`.
Remote candidate: R/artifacts/commit_cleanup_20260915/workspace; canonical mirror
synced with the cleaned files and obsolete helpers removed. No global skills or
AGENTS changes needed. Commit is a progress checkpoint: **both full-model strict
alignment gates still fail by two tokens**, causes unisolated; no correctness
promotion, threshold relaxation or additional464-case sweep. No push requested.

## Complete: serving tests; both strict alignment gates fail — September 15

User scope: Qwen3 A3:F1 EP8 B4 and Qwen3.5 A1:F1 EP16 B40 in one
128K perf job (perfect balance, no prefill); standard alignment with real prefill,
changed to nominal 8K. History comparison required; user accepts missing exact
Qwen3.5 EP16 history and authorized an additional EP32 historical case, still perf mode.
Keep metadata push / payload pull / push combine; compact payload push remains an
alternative. No new kernel search, promotion, rollback, commit, or 464-case sweep.

P=`scratch/megamoe/final_serving_cases_20260915`.
R=`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD`.
Remote artifacts: R/artifacts/final_serving_cases_20260915.
Base source S=R/source/ffn_final_cleanup_20260915. CUDA/kernel math unchanged here.
Full evidence and readable report: P/{SUMMARY.md,STATE.json,perf_results.json}.

### Performance complete and audited

All 15 measured steps retained after one warmup, zero outliers. Metric: mean of
maximum complete CUDA graph span across representative attention and FFN ranks
per step, not client latency or max over every rank. FP8 activations / FP4 experts,
MB2 overlap, perf1/perfect1/no prefill throughout. Historical runs use separate
allocations/source stacks; comparisons are observational, not isolated kernel gains.

- Job7157769 COMPLETED0:0/12:11, 8nodes32GPUs, original two cases sequential;
  all16 case/rank completions audited. Q3:18.329570067ms/163.669960TPS/GPU vs best
  exact pull/push history7061094:18.670507467ms/160.681224, latency−1.826%, TPS+1.860%.
  Model/shape/metric/memory/actual KV match. Q35EP16:22.643118733ms/883.270553TPS/GPU;
  exact history absent (accepted gap).
- Additional final-kernel EP32/A1/B40/128K job7158018 COMPLETED0:0/8:35,
  16nodes64GPUs, all16 rank completions audited:22.618707933ms/884.223805TPS/GPU
  vs Sep12 pull/push history7112916 candidate22.601254333ms/884.906639;
  latency+0.077224%, effectively unchanged. Both memory.82, actualKV5260736,
  MB2=20+20. This does not supply an exact EP16 baseline. Older push-dispatch
  history excluded. Discovery inspected290 result files, matching geometry/model
  paths where historical model_key was null. Raw times and sources preserved.

### Alignment complete: both numerical FAIL

Q3:96prompts×17=1632tokens; top1=1384(84.803922%), top10=1630(99.877451%).
Full ordered prompt/token/position coverage and finite chosen/candidate logprobs
PASS; strict100% top10 FAIL. Zero-based misses: prompt48/position9/token1968/rank30,
prompt95/position2/token19801/rank11. Cause unisolated; do not attribute to MegaMoE
or claim final normal-routing correctness. Generation7157806 preserved; five
fresh reference shards7158117 plus only missing36prompts612tokens7158404
(COMPLETED0:0/8:21,3nodes) merged into original order. Complete audit:
P/alignment_final/qwen3/{alignment.json,audit.json,audit.log,context_audit.json}.
Actual raw prompt length8184, nominal8192: existing prompt builder counts chat
wrapper overhead, batch sampler sends raw strings. Generation/reference encode
identical raw prefix, so comparison valid. User informed; record actual lengths.

Q35 generation7158571 PASS640prompts×17=10880tokens, downloaded with prompt file
and generation audit to P/alignment_final/qwen35. Reference shard2 TP3 failed
TorchInductor static CUDA kernel load: CUDA driver file not found. JobFAILED143:0
in27:34. Seven completed shards were only in Ray scorer memory; no merged score.
Attempted same-allocation recovery was rejected after job expired; no scoring ran.
Failure evidence: P/alignment_qwen35_cache/{collected,failure.json}.
Reference-only7158962 FAILED143:0/7:38 before scoring: standalone adapter omitted
Q35 --language-model-only; vision warmup failed missing cutlass.cute.core.ThrMma
and blake3. Operator error acknowledged; failed plan/logs retained. Corrected
text-only job7159116 used4nodes16GPUs, same sample, and saved all four160prompt
shards plus full ordered640prompt merge before strict gate. Final FAILED1:0/14:38:
top1=10603/10880(97.454044%); top10=10878/10880(99.981618%); all chosen/candidate
logprobs finite and full ordered coverage PASS. Misses zero-based prompt127 and639,
position1/token311/rank17. These inputs are identical text; full continuations
differ. Actual raw context8182 for all640, nominal8192, matched generation/reference
prefix contract as Q3. Complete score/audits: P/alignment_final/qwen35.
Reference TP4, language-model-only, eager/triton, chunks1024/concurrency1/maxseqs1/
util.75/maxlen8256. Node-local /dev/shm TMPDIR, separate persistent node caches.
All task allocations ended; no outstanding jobs. Cause of numerical misses remains
unisolated. Do not claim final correctness, automatically rollback, or relax gate.

### Recovery fixes and remaining work

Initial7157517/19 failed seeding KeyError ATTENTION_WORKERS before model execution;
7157564 failed postprocess KeyError PROMPT_MODE,7157565 canceled to avoid same error.
Explicit seeding vars and MODEL_KEY/PROMPT_MODE exports fixed; successful perf
artifact wrapper validates fixes. Superseded128K alignment7157783 canceled pending,
zero GPU use. Q3 reference7157806 OOM required only reference memory/chunk reduction;
7158117 early numerical gate interrupted three shards, recovered above without
repeating generation or successful shards. Full negative audit saves both misses.

Q35 previous7158380 failed143:0/9:46 during real prefill with Triton metadata
OSError116 stale file handle. New source S+'_triton_cache' uses a hardlink snapshot
plus atomic replacement of only afd_support.py and server/supervisor_ray.py:
TRITON_CACHE_DIR forwarded and scoped by existing node/rank helper. All16 FFN
cache scopes evidenced; subsequent generation succeeded. No CUDA/math changes.
Local maintained run_afd.sh also supplies TASK_ROOT/cache/triton default.

Maintained sequence runner, monitor, sample auditor, sharded scorer, launcher and
two cache runtime files synced explicitly to R/local-mirror; remote bash syntax
checks PASS. Actual submitted artifact workflows stay immutable. No py_check or
manual hash/version check. Preserve unrelated dirty/staged work.

Final reports: P/SUMMARY.md and P/final_audit.json; STATE preserves every failed
attempt and completed job. Final explicit report/docs sync to remote performed.
Requested serving tests are complete; both strict alignment failures reported.
No additional kernel search/sweep/commit. Follow-up ideas require a new bounded
diagnosis: numerical misses vs full-model historical reference; explicit raw/chat
prompt-length contract; DeepEP cache preseeding; persist Ray score shards before merge.

## Completed: cleanup slowdown retest — keep cleanup (September 15)

User authorized retesting and recovering cleanup if the pattern persists.
Retest P=scratch/megamoe/final_cleanup_20260915/retest: same24 manifestcases,
4pullconfigs (cleaned, immediately-before cleanup, Sep13dense, Sep12previous),
2symmetricrepeats,2000calls/200warmups/batch100/MAXrankCUDAevents/no profiling.
Jobs7157096Q3EP8/7157099Q35EP16 COMPLETED0:0 in11:36/12:19. StrictauditPASS
16records192measurements2304rankexactchecks; allsamples retained, no partials.
All48cleanedpull measurements within -0.168..+0.154% of matched immediately-before
source. Mean per-case cleaned vs precleanup: Q3+0.009%,Q35-0.011%; vs best of
bothhistoricalpull/push reruns: Q3+0.032%,Q35-0.118%. No rollback indicated.
Three oldslowcases didnotrecur: Q3rows8 cleaned47.968/47.901us vs precleanup
47.894/47.927; Q3rows18 48.877/49.025 vs48.868/49.079; Q35rows2
47.901/48.071 vs47.944/48.151. Old evidence retained; transient cause unproven.
All11production/protocol snapshotfiles identical to frozen before source; smoke
only differs in multi-rowdriver support and allvariants use common currenttest.
No production changes, knobs, staging, commit. Existing audit generalized to
summarize historical comparisons without compact; actualretestauditPASS. Plot
adds precleanup label. Read-only snapshotfetch JSON-list mistake corrected with
line-based sync_files.txt, no GPU/source effect. Evidence SUMMARY.md, REPORT.md,
audit.json, decision.json, timings.csv. Full-modelalignments/464sweep unsubmitted;
this turn completed retest/conditionalrollback decision, retaining pull default.
Collector96267 and finalone-shot98347 EXIT0; bothGPUallocations released.

## Final cleanup: correctness passed; stopped for performance report (September 15)

User chose metadata push + payload pull + push combine as the default, retaining
compact payload push as the alternative, and explicitly resumed final cleanup and
both model sweeps. Both modes keep metadata push. Earlier STOP outcomes below
remain historical evidence; this request authorizes the bounded finalization.
No new optimization search or automatic default promotion. Preserve frozen best
pull/push baselines and compact source; compare both cleaned modes against them.
Work/evidence: `scratch/megamoe/final_cleanup_20260915`; pre-edit snapshots saved.
Implemented one per-buffer payload mode: explicit payload_mode or
MINISGL_MEGAMOE_PAYLOAD_MODE=pull|compact_push (default pull). Producer, arena
sizing, host validation and both FP4/FP8 launch paths share the pinned selection.
Rejected dispatch switches/branches remain absent; preserved dense metadata,
contiguous pull consumer, tested compact frame/gather path, combine arithmetic.
CPU7155439 benchmark26.05/CUDA13.2 PASS native+host+3protocoltests,7:48
(0.041s tests). Serving25.10/CUDA13.0 CPU7155779 PASS native+host6:14, separate
cache/serving donor+native cache wired into both alignment plans.
Correctness PASS684 byteexact comparisons: pull7155542 264/5records6:20,
compact7155543 180/3/4:24, compactlarge7155731 176/2/3:19,
compactM327155840 64/4/5:01. Compact covers actualM16/32/64/96. Initial large-
shape expectationM32 wrong (selectedM64/96); preserved expected.json/coverage_gap,
audited actualtiles separately, then closedM32gap vsfreshdensecontrol. No CUDA
implementation edits after initial cleanup. correctness_summary.json records all.
Old collectors11657/37913/7354 EXIT0, allpreviousGPU/CPUjobscompleted.
Five-config24case timing jobs7155992Q3EP8/7155993Q35EP16 COMPLETED0:0,
14:58/15:56. Strict audit PASS20records240measurements2880rankexactchecks,
actual candidate payload mode checked. Collector97884 and one-shot44231 EXIT0.
All24cases/repeats retained. Mean per-case delta vs faster matched historical
pull/push: cleaned pull Q3+0.742%, Q35+0.172%; cleaned compact Q3-1.232%,
Q35+2.536%. Three unresolved slow pull repeats: Q3rows8 47.502/52.186us
(dense47.529/47.358, mean+5.061%); Q3rows18 53.130/49.270us
(dense49.052/49.271, +4.146%); Q35rows2 56.436/48.324us
(dense48.049/48.294, +8.735%). Other repeats close to dense; historical sources
also contain slow samples, so no established consistent regression or cause.
No sample filtering, reroll, or unsupported nonregression claim. STOP/REPORT
before full-model alignment and464-case sweep, per retained user rule. Known
compact model tradeoff persists; pull default unchanged, no promotion/search.
Evidence P/SUMMARY.md, comparison/{REPORT.md,audit.json,cleanup_comparison.json}.
Original missing-wrapper sbatch rejection occurred before allocation and was fixed.
Bothnormal-routingalignmentdryrunsPASS; noactualalignments/sweepsubmissionsyet.
Default sweep11groups464cases validated/synced, report plan prepared. Historical
fullpath928observations/464cases frozen fromonlytwo pull-dispatch/push-combine
sources; previous-source timelineindex copies are derived existingdata, no capture
changes. compare_megamoe_fullchain.py now accepts repeated --reference for per-case
lowest historicalmean. No staging/commit; preserveindex/unrelatedwork.
Serving review added payload env to afd_support.runtime_env explicit allowlist;
CPU7156038 COMPLETED0:0/1:14 PASS4protocoltests (0.045s), including realRay-env
forwarding; no CUDA changes.
Driver->coordinator also uses supervisor_ray._build_ray_runtime_env whitelist;
addedpayloadmode there andextendedsame4teststofullycoverbothhops. CPU7156083
COMPLETED0:0/1:51 PASS4tests(0.050s). NoCUDAbuild/retestneededforthisPython-onlyfix.
New artifact workflow/run_afd_seeded.sh retains oldhostseeding and adds maintained
seed_tvm_ffi_cache.py beforeworkers, matching-serving donor -> distinctmodel/node/
rankcaches. Alignment env/launch plansupdated, shellsyntaxPASS. Oldfrozenworkflow
untouched. Initial wrapper-preparation heredocdelimitercollision causedparseerror,
fixedbeforeactualCPUsubmission; no source/kernel/test execution fromfailedcommand.

## Completed: crossed EP comparison, user-selected8 Q3 EP16 +12 Q35 EP8

User selected ONLY8Q3EP16manifest cases after clarification; no supplemental cases
in finalcomparison. Q35 retainedfresh12/83draw,seed1035385404524719568.
Evidence scratch/megamoe/pull_compact_cross_ep24_20260915 (namepredates20cases):
SUMMARY.md,REPORT.md,audit.json,config_audit.json,ep_comparison.json,all_config_latencies.png.
Jobs7154879Q3/7154787Q35 COMPLETED0:0 in11:45/13:12; strictauditPASS20cases,
16records160timings1792rankexactchecks; all1792runtimeconfigsrecorded and selected/
compact configsidenticalpercase/rank/repeat. No missing/filtered/partial finalcases.

Meanpercase latencydelta vsfastermatched historical pull-dispatch/push-combine:
Q3EP16 selected+0.349%,compact+0.458%; Q35EP8 selected+0.277%,compact+6.635%.
Q3compact beats history1/8(rows3), loses7/8; lowestmean previousmetadata5/dense2/
compact1. Versusselected compact+0.109%average,3/8lowermeans,1/8bothrepeats;
selectedwinsboth4/8,overlappingranges6/8: small/mixed, notclearcompactadvantage.
Q35compact losesALL12bothrepeats vshistoryANDselected; rangevshistory+3.136..+10.086%,
vsselectedaverage+6.331%. Lowestmean previousmetadata5/selected5/dense2.
Historicalbest sourcevaries: Q3previous6/dense2;Q35previous5/dense7, so keephistory.

EarlierEP8Q3compact-1.455%vshistory(all12wins) didnotcarryovertoEP16. SevenQ3
sameglobaltokencases: compactadvantagevanishes/reverses6/7; rows3/global48 stillwins.
EarlierQ35EP16compact+3.278%, newEP8+6.635%; shapesdiffer(48-1024vs32-576tokens),
noexactglobaloverlap, so noisolatedEPcausalclaim. NewQ3localexperts8,Q35local64;
both8distinctdestinationranks/token. Q35getsframecopydedup8vs10butstillloses.
Commonheuristics; Q3M16/32 stages11/10,8experts/wave;Q35M16/32/64 stages11/10/9,
64experts/wave atrows6 else32. No newkernelknobs/variants orhardwareprofile.
No universalmodel/EPpreferenceestablished. STOP/REPORTretained; no promotion,
finalcleanup/kernelsweep/staging/commit. AllGPUjobs/collectorsreleased.

OriginalQ3job7154781 cancelled4:18 onusercoveragechange. Its2completeprocessrecords
andpartialtrial preserved/excluded under localqwen3_superseded_12cases; originalplan
superseded_12case_plan.json retained. NO splicing acrossallocations orrerollbytiming.
Remote mapping: currentlocalqwen3 ->remoteqwen3_manifest8; localqwen35 ->remoteqwen35;
remoteqwen3 holds CANCELLED original andmustnotbeoverwritten. Finalartifactsync
usesexplicitrootfiles and currentmodeltargets; never uploadcomparelogs.
Commonmethod2000calls/200warmups/graphbatch100/MAXrankCUDAevents/noNSys,FP4/FP8,
150SM/bucket3840,fixedbalancedroutes,exactcombinebeforetiming; fullproducer+MegaMoE,
notwholemodel/requestE2E. Priorfullmodelalignmentnotrerunfor thisbenchmarkonlytask.

## Completed: model-difference diagnosis and historical pull/push comparison

Latest user requested cause investigation plus historical best with payload-pull
DISPATCH / push COMBINE. Completed matched rerun of all24 previous random cases,
4 frozen sources: selectedexplicit, compactlinear, September13dense, September12
previousmetadata. Evidence scratch/megamoe/pull_compact_history24_20260915:
SUMMARY.md, DIAGNOSIS.md, audit.json, historical_evidence.json, phase_evidence.json,
workload_metrics.json, all_config_latencies.png. Jobs7154497Q3/7154500Q35 both
COMPLETED0:0 in11:28/12:09. PASS16records192measurements2304rankexactchecks,
all24cases4sources2repeats, no missing/filtered/partial records. All12shapes/process
and reversed second-pass order reduce startup. Same2000/200/100 MAX-rank CUDAevents,
FP4/FP8, fixed perfect rank-interleaved routing, noNSys, exact combine before timing.

Historical dense wins vs previousmetadata all24 in matched reruns (old unpaired
rank0archive min provenance22dense/2previous remains preserved). Relative to best
historical rerun mean: selected Q3-0.043%,Q35+0.085%; compactQ3-1.455%,Q35+3.278%.
Compact wins bothrepeats12/12Q3 and loses bothrepeats12/12Q35 against selected AND
history. Selected lower mean vsdense9/12Q3,5/12Q35, bothrepeats5/12,2/12: small/mixed,
not bestproof. Q35selected vsdense range-0.065..+0.753%; don't hide worstcase.
No universalwinner/promotion/finalknobcleanup/kernelsweep/staging/commit.
STOP/REPORT retained; only these explicitly requested comparisons/diagnosis resumed.

Cause: noQ3-only knob found; selected/compact actualconfigs identical percase/rank/
repeat (1152configlines initial24); sameheuristic source,150SM,bucket3840,N/K128,
M16/32 stages11/10, threads128/128/256. Router explicitlyspecializesBOTH128/top8 and
512/top10 with256threads,4352Bcompactframe. Q3EP8/local16/top8/I1536 vsQ35EP16/
local32/top10/I1024 causes waves16vs32,25%more framecopies/token, more consumer
activationgather work. Saved matched phases: compactvsdense Q3compute+.176..+.752us;
Q35compute+1.568..+3.296us,producer+.513..+2.681us despite dispatch-1.152..-1.328us.
Prior contiguousconsumer push experiment recoversQ35compute2.208..3.584us while
producer worsens1.531..1.691us. Supports layout/producertradeoff, not isolated
hardwarecounter attribution; EP/topk/experts confounded. No new variant/knob created.
Initial2source24 evidence remains pull_compact_random24_20260914: compactQ3-1.158%,
Q35+2.969%,96measurements1152rankchecks. Don't mix oldNSys rank0 minima withnewevents.

Artifact-sync incident fully recovered: assistant uploaded experimenttree during
runtime and replaced active log pathnames withstale copies. Originalbatchfdstreams
retained complete logs; copied via /proc PIDs641960/3959619 throughbatchexit to.live
files; bothrecoveryprocesses EXIT0. Preserved stale/live originals; restoredcomplete
canonical logs; strictaudit PASS allrecords. No recordsplicing or GPUrerun. Collector
45731 stopped143; finaloneshots EXIT0. AllGPU/collector/recoverytasks ended.
Lesson: never upload local result logs into an active remote experiment; sync only
explicit plans/workflows/docs (exclude compare logs), fetch results one way. Recovery
script is incident evidence in scratch, not a new production workflow.

## Final MegaMoE evaluation stopped under the user’s rule

Read CODEX_PROJECT_final_megamoe.md and selected_explicit/REPORT.md under
scratch/megamoe/pull_metadata_restore_20260914. Fixed misplaced explicit-combine
keyword (checkpoint loader -> buffer allocation). Exact7153343 PASS264/5records;
CPUadapter7153388 PASS3tests. Q3complete rows28/21: compact push faster in both
repeats, means−0.677%/−0.390% vsselectedpull. CompleteQ35rows38 favorsselectedpull
vscompact (−3.402%); precleanupdense slightly faster. No universalwinner.
Applied explicit STOP/REPORT rule: cancelled7153461/7153462 at26:30/26:15;
42records488rankexactchecks retained,3completecases12partialrecords. Original
six-case matrix incomplete (Q3rows4,Q35rows28/22); no finalknobcleanup/sweep,
recovery, promotion, staging orcommit. No taskjobs/collectors remain. All relevant
code/evidence/handoff synced. Do not resume historicalplans withoutnewrequest.
Earlier checkpoints below are historical.

## Handoff prepared — work paused at latest user request

Read [final MegaMoE handoff](CODEX_PROJECT_final_megamoe.md) for current source,
results, risks and exact remaining commands. Explicit per-buffer combine CPU7153123
PASS native+host in3:28. No running task jobs/collectors; new exact264 and five-config
Q3/Q35 unprofiled comparison plans are prepared but UNSUBMITTED. No index/commit
changes. Resume remaining validation/selection only when instructed; earlier ACTIVE
entries below are historical checkpoints. Do not claim final version or best-config
proof complete. Current state: pull_metadata_restore_20260914/selected_explicit/STATE.json.

## Active: prove the best selected communication configuration

Latest user requires early metadata push + payload pull + push-based combine,
clean code and make it work; MUST establish best using tests. If another path is
better, STOP and report rather than silently change architecture/search further.
Initial cleanup/test complete: CPU+exact264 PASS; Q3/Q35 jobs7152479/7152485 both
complete/audited24captures120phase720ordinary. Q3 fullpath−.037..+.152percent.
Q35 rows22/28 candidate_repeat broadly slow (medians65.15/71.63us); pooled means
+5.127/+8.147percent vscontrol. Rows38+.595percent. Cause unresolved; no data
filtered, no claim of universal nonregression. Full initial table/report saved in
scratch/megamoe/pull_metadata_restore_20260914/REPORT.md and results.json.

Follow-up code: combine is an explicit per-buffer protocol, passed to allocation,
host validation and both FP8/FP4 launch paths. AFD selects inband_combine=True;
standalone/reference callers retain legacy env/default0. CUDA math/core unchanged.
Eight affected active files; previous rejected-payload deletion remains applied.
Final source ffn_selected_explicit_20260914 is separate from initial frozen source.
CPU7153123 building actual native/host; exact264 plan ready, unsubmitted. Prepared
unprofiled best-config comparisons (2jobs max): original same3Q3EP8+3Q35EP16 cases,
5configs (selected, precleanup dense, previous metadata/pull, compact push, dense/
conventional combine), symmetric repeats, 2000calls, graphbatch100, slowest-rank
CUDA-event means. Shared maintained smoke comparator avoids old reference-arena
bug; exact combine checked outside every timed trial. NoNSys for winner decision.
No new GPU submitted yet. Details selected_explicit/STATE.json and
selected_explicit/config_comparison/plan.json. Await CPU/exact before timing.

## Current request: restore selected pull, remove rejected paths, test decomposition

User explicitly resumed work: select best validated metadata with pull payload,
remove rejected optimization paths from active code, then test/report decomposition.
Prior STOP applies only to old optimization plans; no new search for variants.
Working-tree kernel/router already matched selected dense descriptors. Removed
rejected Lamport inbox dispatch, native TMA sender, in-MegaMoE sender, push arena
helpers, template options and environment forwarding from seven active files.
Selected dense metadata, contiguous payload pull, combine and legacy correctness
paths retained. Reserved metadata padding keeps validated buffer offsets stable.
Backups/evidence remain in scratch; no index/commit/unrelated GDN changes.
CPU7152139 PASS all5steps in2:56 (EP2/8 kernels, native/host, SASS); final native
refresh7152230 PASS1:42. Remote source ffn_pull_metadata_clean_20260914.
Fresh fixed seed3689699192022641276: Q3EP8 rows28/21/4, Q35EP16 rows38/28/22.
Exact GPU7152282 PASS264 saved comparisons/5records, completed8:13.
Fresh before/after cleanup ABBA Q3EP8 job7152479 and Q35EP16 job7152485
submitted22:25UTC. Q3 completed12:03, collector PASS12captures60phase360ordinary;
fullpath delta−.018..+.071us (−.037..+.152% latency). Q35 still running.
Expected total24captures120phase720ordinary.
All final source/workflow files synced before validation. Evidence/workflow:
scratch/megamoe/pull_metadata_restore_20260914.

## Previous campaign stopped after completing ongoing experiments

7150076 Q3 and 7150075 Q35 COMPLETED0:0 in24:22/25:38; collectors EXIT0.
All48captures240phase1440ordinary audited. Early-store REJECTED_AS_STANDALONE:
0/6dispatch targets; fullpathvsdense Q3−0.62..+0.09%, Q35−0.60..−0.13% gains.
Compute/consumer is largest compact phase: Q3 24.70–25.65us, Q35 37.18–37.60us;
exposeddispatch5.47–5.92us. Q35compact saves1.15–1.33us exposeddispatch vsdense
but producer finishes0.51–2.68us later and compute spans grow1.57–3.30us.
No hardware-stall profile collected; rank0 timelines/phases only. No qualifying
replacement, promotion or fresh serving alignment. Goal not achieved.
Full report: [FINAL_STATUS.md](scratch/megamoe/dispatch_incremental_20260914/FINAL_STATUS.md).
No running jobs/collectors remain for this work. STOP; do not launch or resume any
historical experiment, profiling, validation, alignment, or optimization plan
below without a new user request.

## Optimization context: retain faster transfer while removing both overheads

User corrections September14: optimize metadata/payload dispatch without moving
cost into producer preparation or scattered consumer loads. Payload method/layout
may change; preserve the measured transfer advantage, not a particular push design.
Require >5% dispatch gain on fresh random3Q3EP8+3Q35EP16, faster matched ordinary
producer→MegaMoE (including repeats/frozenhistory), and no measured producer/compute/
combine regressions. Then normal-routing/slowest-rank validation and fresh model
alignment; no phase-only promotion or claim of universally minimal latency.
Goal not achieved; work stopped per user. Current source main/index preserved.

Completed experiment context (September14):
- Directbulk REJECTED:7148045/7148053 completed,48captures240phase1440ordinary
  audited; fullpath6.78–18.35% slower thancompact,10.25–17.82% slower thandense.
  Vector→bulk alone gives−0.08..2.34%; allsixdispatchgatesFAIL. ExactPASS352 retained;
  no EP2 expansion/promotion. Rejection evidence direct_bulk_push/e2e_rejection.json.
- fixed_pool_shared_targets REJECTED_AS_STANDALONE: exactPASS180; timing7148682/
  7148922 bothcomplete48captures240phase1440ordinary. All6dispatchgatesPASS.
  Sharedhandoff savesproducer0.78–1.25us vsfixedpool butstill+1.36–2.95us vsdense.
  Contiguousconsumer helpsQ35compute2.21–3.58us vscompact; remainingproducer
  costdisqualifiesstandalone. Preserve effectfor evidence-supportedcomposition.
- fixed_pool_packed_frame REJECTED_AS_STANDALONE: exactPASS352 and CPUactual
  native/hostPASS. Combined timing7149421/7149422 fullyaudited60captures300phase
  1800ordinary across bothmethods. Expertpacked5/6dispatch gatesPASS; producer
  remains+1.336..3.506us vsdense despite0.419..1.505us savingvsfixed. Q35rows13/34
  fullpath−1.282/−4.427% vsdense. Standaloneplans remainSUPERSEDED_UNRUN.
- packed_source_pull REJECTED_AS_STANDALONE: exact7149145 PASS496/12records,
  targetH4096 stages unchanged; wideH32768 drops5→4. Same combinedaudit above:
  0/6dispatchgatesPASS, fullpath0.216..1.256%slower vsdense whileproducer roughly
  unchanged(−.081..+.105us). Packing source SF alone does not retain fastdispatch.
  Evidence packed_source_pull/combined_rejection.json; collectors35996/92478 EXIT0.
  Preserve direct output-stride mechanism for scheduling experiment below; no
  standalonepromotion/EP2 expansion. Native stride validator prepared, UNRUN.
- packed_source_early_store: one-header scheduling delta on packed_source_pull.
  Start/commit local activation store immediately after frame completion, overlap
  SF transpose/weight stores; one elected thread retains issue/commit/wait ownership.
  Original final wait/release preserved. Native/layout/consumer unchanged; inherited
  wide stage-depth loss unresolved. CPU7149735 PASS5steps3:46 actualnative/host, EP2/EP8/SASS. Explicit R41<48/
  R23<40; compiledearlystore1810 beforeSFload1890, fullwait1a90 beforearrival1b30.
  Exact7149885 PASS352/8records11:11, targetstage/smem andall6M verified.
  Freshseed6974469513186867194 Q3rows9/28/10,Q35rows51/48/20; eight symmetric arms
  compact/sourcepacked/dense/earlystore. Timing7150076Q3/7150075Q35 COMPLETED,
  collectors90292/55335 EXIT0;48/240/1440 audited,0/6dispatch,REJECTED_AS_STANDALONE.
- tiled_expert_push_design.md is design-only: contiguous physicalexpert tiles,
  one consumerdirectory lookup, no finalprefixdependency. Allocation/publication,
  generationreuse/padding/capacity costs unresolved; no source/build/GPUwork.

Rejected and preserved:
- Async lane gather7147418/7147425 complete36captures180phase1080ordinary;
  all6dispatchgatesPASS, fullpath59–91%slower thancompact, producer nearlyunchanged.
  Reject this TMA replacement. Allcaptured data/failedcollectorlogs retained;
  localanalysisrecovery fixed phasehistoryargument, noGPUreruns.
- Compact linear rejected forE2E: latestmatched6cases fail4/6 despite allphasegates.
  Priorselectedsource sweeps stopped, incomplete350cases1750phase10500ordinary
  retained; cancelled7145382/7145806 and stoppedrunners mustNOTresume. Bothmodel
  alignments pass correctness only. Tokenstride/sharedroutes also rejected.
- Previous early-PDL trials moved waiting into dispatch without E2E improvement.
  Q35two-stream serving deliberately defersPDL forsharedexpertpriority.

Corrected key-enabler attribution: metadata/payload separation predates the last
compact-push change. Relevant delta is compact activation/SF/weight frames and
local auxiliary reads; same-layout Q3 comparison gives11.7–16.5% dispatch gain.
Bulk instruction alone is not sufficient; shared staging also reduces repeated
producer loads. Compact pool footprint benefit is established, its causal latency
contribution is not. Distinct-rank balancedroutes have no fanout dedup benefit.

`audit_fullpath` now guards ordinary producercompletion/start and phasecompute/
combine means/medians as well as fullpath/repeats/history; allrecords retained.
Componentworst-repeat deltas reported separately; markers include stalls/overlap,
not isolatedcompute/network efficiency. Gate checks saved component_guard*.json.
Historical fullpath baseline frozen separately fromdispatch minima for464cases.

Details and paths: [CODEX_PROJECT_dispatch_incremental.md](CODEX_PROJECT_dispatch_incremental.md),
`scratch/megamoe/dispatch_incremental_20260914/{STATE.json,NEXT_IDEAS.md}`.
Remote root `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD`.
Max2outstandingGPUjobs; CPUjobs separate. Plain ssh oci-hsg, monitor300s (nearfinish
2–3min), act on terminalstates withoutholdingreadywork forcadence. No subagents,
/tmp, py_check/hash checks, fallback, commits/staging or unrelatedmaincodechanges.
Separate GDN work untouched. AGENTS.md absent; preference alreadyasked; supplied
globalinstructions apply. Preserve staged history and all experiment evidence.

Everything below is historical and does not override this active goal.

# Project Memory

Goal closure verified: live scheduler shows all17 gate/sweep/alignment jobs
COMPLETED0:0 and no active jobs. Re-executed the complete464-case artifact audit,
recomputed historical dispatch gains from saved samples, checked both repeats of
all six sampled cases against each control and history (>5%), and compared all14
integrated files with validated overlays. All PASS. Required direct-pool push was
implemented/tested and slower; final compact metadata plus pull payload retained.
Evidence: scratch/megamoe/dispatch_dense_descriptors_20260913/goal_closure_verification.json.
Requested experiment sequence complete; no claim of globally minimal latency or
substantial whole-pipeline improvement. No additional GPU runs or index changes.

Continuation completion audit: prior turn classified as progress (validated integration
and staging). Live Slurm confirms all 17 sampled/sweep/alignment jobs COMPLETED0:0
and no active jobs. Re-audited all 464 case contracts and recomputed six sampled
gates/full history-best gains from saved phase samples; all match reported results.
All 14 integrated files still match validated overlays. Direct-final-pool push
was implemented and tested but slower, satisfying the required experiment without
promoting it. Evidence: scratch/megamoe/dispatch_dense_descriptors_20260913/
completion_revalidation.json. No GPU reruns or index changes. AGENTS.md remains
absent; supplied global instructions apply, pending optional documentation preference.

## Completed dispatch optimization and attention-only cleanup — September 14

Combine frozen at validated in-band1. Compact descriptors and attention-only cleanup
are integrated locally. User subsequently requested git-add staging only: stage the
completed dispatch/cleanup, validated attention/model dependencies and reusable tools
(57files total), retaining the previous33-file scope. No commit. Separate GDN
profiling script and its project note remain unstaged. Scope: scratch/megamoe/stage_20260914/stage_files.txt.
All fifteen rejected timing trials retained/audited24captures/120phase/720ordinary
samples each. Gains below compare fresh matched controls (negative=slower):

| Trial | Six-case result | Evidence |
| --- | --- | --- |
| Parallel counts | REJECT: -26.4% to -11.9%; SASS disproved serial-load hypothesis | [results](scratch/megamoe/dispatch_count_parallel_20260913/README.md) |
| Ordinary acquired counts | REJECT: -2.4% to +3.0%; all behind history | [results](scratch/megamoe/dispatch_count_cached_20260913/README.md) |
| Count/TMEM-init overlap | REJECT: Q3 -5.0% to +2.3%, Q35 -4.0% to -0.2% | [results](scratch/megamoe/dispatch_count_overlap_20260913/README.md) |
| Single-writer ready flags | REJECT: Q3 -4.8% to -3.7%, Q35 -8.1% to -6.8% | [results](scratch/megamoe/dispatch_rank_ready_flags_20260913/README.md) |
| Ballot rank selection | REJECT: -1.1% to +3.0%; all six gates fail | [results](scratch/megamoe/dispatch_ballot_select_20260913/README.md) |
| Grouped selector | REJECT: Q3 -0.55% to +1.84%, Q35 -5.24% to +1.28% | [results](scratch/megamoe/dispatch_grouped_selector_20260913/README.md) |
| Copy/SF overlap | REJECT: Q3 -1.26/+0.91/-1.44%, Q35 +0.32/-0.66/0%; all behind history, kernel changes <0.5% | [results](scratch/megamoe/dispatch_copy_overlap_load_20260913/README.md) |
| Local weights | REJECT: Q3 -1.89/-0.18/+0.17%, Q35 -1.15/-8.57/-3.28%; all six fail; all kernel means regress | [results](scratch/megamoe/dispatch_local_weights_20260913/README.md) |
| Active-warp bound | REJECT: Q3 -1.69/-0.72/-1.62%, Q35 -1.64/-1.54/-4.17%; all six behind history | [results](scratch/megamoe/dispatch_active_warps_20260913/README.md) |
| Warp-major ownership | REJECT: Q3 -1.28/-1.08/-0.19%; Q35 +18.90/-14.64/-0.84%; only rows1 passes | [results](scratch/megamoe/dispatch_warp_major_20260913/README.md) |
| First-round shortcut | REJECT: Q3 -2.14/+1.08/+1.76%; Q35 0/+2.89/+2.12%; all six gates fail | [results](scratch/megamoe/dispatch_first_round_20260913/README.md) |
| Expert-boundary ownership | REJECT: Q3 +2.82/-0.54/-1.42%; Q35 +16.23/+11.39/+1.00%; only rows1 passes | [results](scratch/megamoe/dispatch_expert_warp_padding_20260913/README.md) |
| Packed input | REJECT: Q3 -5.36/-7.72/-6.82%; Q35 -7.53/-3.11/-7.72%; all six fail; full pipeline regresses | [results](scratch/megamoe/dispatch_packed_input_20260913/README.md) |
| Forward rank mapping | REJECT: Q3 +16.70/+7.60/+2.91%; Q35 +17.51/+11.97/-4.85%; 2/6 pass, all kernel means improve0.43–1.55% | [results](scratch/megamoe/dispatch_forward_rank_map_20260913/README.md) |
| Payload-only packed transfer | REJECT: Q3 +0.22/-0.75/-4.06%; Q35 -6.06/-4.06/-4.01%; all six fail; pipeline regresses0.23–1.44% | [results](scratch/megamoe/dispatch_packed_payload_only_20260913/README.md) |

Checkpoint (final sweeps complete): all11groups/464cases/1566input aliases audited;
2320phase and13920ordinary samples, including23EP64 exact-combine cases. Actual
ordinary specializations confirm frozen in-band combine and both push flags off.
Full mean per-case dispatch gains vs historical best: Q3 14.108% (141/164>5%,
161improved,2regressed,1equal; worst -1.478%); Q35 23.108% (300/300>5%, min7.965%).
Original six sampled gates and both initial alignments PASS. Final cleaned-source
alignment Q3 7131066 PASS15:03:48prompts/816tokens, all top1/top10, complete ordered
pinned continuations and finite chosen/candidate scores. Q35 7131070 PASS17:49:
168prompts/2856tokens all top10,2853top1; three rank2 first tokens match prior
references. Both jobs COMPLETED0:0. Monitored300s, at most2outstanding GPU jobs.
Cleanup overlay passes33tests, native/GPU metadata/transport/cache-reuse checks.
All14validated files integrated after both final alignments; staged patch unchanged
and all5unchanged model/worker basis files preserved. No commit. User waived HTML
inspection; both same-format HTML
reports and full PNG/SVG/PDF plot generated; scientific PNG reviewed.
Full rank-0 router→MegaMoE mean per-case gains: Q3 0.478%, Q35 0.583%; reduction of
mean latency0.467%/0.130%. There are31/120 regressing cases; all records retained.
User confirmed original dispatch acceptance target; combined latency reported
separately without a net-latency gate. Producer shifts work into metadata creation:
MegaMoE starts1.735/1.710us later on average. No substantial whole-pipeline claim.
Current path pushes compact metadata and pulls payload; no new activation reshape
or packing pass. It is not literally zero-copy: source→receiver shared→expert pool.
Current dense metadata + payload-only push is UNTESTED; existing push flag changes
metadata protocol too. Prior direct-pool push passed exact checks but was slower:
Q3EP16 47.345 vs38.600us, Q35EP32 57.939 vs48.152us. Prefix removal helped, while
address-reply bypass and bulk-transfer reuse did not explain most residual gap.
Dense ordinal alone does not provide the padded expert-pool base. Preserve the
open idea; user explicitly requested no push retries now.

Retained experiment evidence (accepted compact descriptors integrated; combine unchanged):

- [Forward rank mapping](scratch/megamoe/dispatch_forward_rank_map_20260913/README.md):
  exact7120177 PASS five records/264saved comparisons; prepared SASS R42/R43<48.
  Timing7120358/7120359 completed12:52/12:32; Q3rows2/21/15, Q35rows1/3/42.
  Only Q3rows2 and Q35rows1 pass; all six ordinary kernel means improve0.43–1.55%.
  Retain component evidence without promotion. All24/120/720 audited; monitor exited.
- [Payload-only packed transfer](scratch/megamoe/dispatch_packed_payload_only_20260913/README.md):
  exact7120548 PASS three records/180saved comparisons, SASS R39/R43<48.
  Timing7120696/7120698 completed12:03/12:47: Q3rows1/4/14, Q35rows9/12/4.
  REJECT0/6 gates; all24/120/720 audited, full pipeline regresses0.23–1.44%.
  Monitor exited. Padding-cache-miss hypothesis remains unproven.
- [Compact source descriptors](scratch/megamoe/dispatch_dense_descriptors_20260913/README.md):
  producer atomics build dense lists; receiver skips count aggregation/inverse lookup.
  Original unbanked draft had a cross-generation overwrite risk; existing metadata
  slices now bank descriptors without allocation/ABI/barrier changes. Payload/combine fixed.
  Corrected CPU/native builds PASS; exact7120850 PASS264saved comparisons, including legacy.
  Timing7120939/7120940 PASS all6 gates,24captures/120phase/720ordinary samples:
  Q3rows14/32/3 matched +25.57/+25.13/+20.99%, history +14.40/+23.91/+19.56%;
  Q35rows6/16/20 matched +30.65/+28.90/+28.11%, history +21.59/+29.84/+23.01%.
  All candidate repeats pass. Full-pipeline means Q3 +1.05/-0.14/+1.14%,
  Q35 +0.56/+6.19/-0.18%; later MegaMoE starts offset much of the dispatch gain.
  No >5% whole-pipeline claim across cases; retain all calls and unpaired-history caveat.
  Initial Q3 alignment7121261 PASS17:58:48prompts/816tokens, all top1/top10,
  complete pinned-prefix coverage and finite scores. Q35 alignment7121263 PASS21:14:
  2856/2856top10,2853top1; all ordered continuations match pinned references. Three
  first-token rank2 values match prior scores; finite chosen/candidate scores audited.
  Monitor300s. Host seeding uses
  actual Ray nodes before workers; first Q3 preflight helper-path failure consumed no GPU.
- Held candidates: [normal pull-cache priority](scratch/megamoe/dispatch_load_cache_normal_20260913/README.md)
  exact7118813 PASS, timing unrun (no cache-miss evidence);
  [prefix lookup](scratch/megamoe/dispatch_expert_prefix_20260913/README.md)
  exact7118573 PASS, timing unrun (earliest experts0–2 weaken the setup tradeoff);
  [early packed store](scratch/megamoe/dispatch_packed_store_overlap_20260913/README.md)
  CPU7119826 PASS, GPU unrun after packed-input rejection;
  [first-round/ballot composition](scratch/megamoe/dispatch_first_round_ballot_20260913/README.md)
  remains local/unvalidated. No implicit promotion or combination of these paths.

Reusable `prepare_megamoe_sweep_abba.py` writes fresh plans without submitting,
preserves actual control caches, derives source metadata from env, checks EP/sample
policy and refuses existing plans. CPU-only compile parity7119325 matched all2808
addressed SASS instructions for one recorded EP8 geometry; this is an example,
not proof for other candidates. Packed-input legacy exact rebuilt its control host
extension inside the GPU allocation: verify/seed established ABI-compatible host
extensions before future launches. Detailed attempts/results stay in trial READMEs.

[Startup diagnostics](scratch/megamoe/dispatch_startup_diagnostic_20260913/README.md):
six captures/30samples; subsequent init1.09–1.28us, route wait0.99–2.10us,
count loads0.42–0.61us, join0.064us, lookup/index1.01–1.82us, first copy/publication
3.20–4.54us. First route waits9–127us retained. Required-publication→first-compute
marker gaps0.77–1.25us are observational, not causal stalls. Preserve TMA.
Cache lesson: device-header mtimes can rebuild a hard-linked host extension.
Use --copy only with established ABI compatibility; ABI changes should use the
existing CPU prebuild_deepgemm_extension.sbatch before GPU allocation.

Integrated cleanup ([details](scratch/afd/validation/attn_only_cleanup_20260913/README.md)):
per-in-flight pinned metadata fixes queued H2D corruption; incomplete real replies
now fail before mutation. Removes unused attention turns/O-turn/direct-O/remote-GDN
and prequantized-O helpers, consolidates final normalization/MegaMoE completion,
and limits coordinator timing overhead. Active event ordering, model/shared/GDN
turns, dummy/stale-request handling and initialization synchronization are retained.
GPU7121520 reproduced baseline corruption and verified all three corrected replays
([evidence](scratch/afd/validation/attn_only_cleanup_20260913/gpu_metadata/comparison.json)).
CPU7121378 failed one obsolete device-turn test; corrected CPU7121488 passed29tests
and native builds. Final draft CPU7121631 PASS33tests and native/router builds in alignment
image25.10 (5:26). Retained transport GPU7121703 PASS1:32: Q3 FP8/Q35 gated BF16,
two rank pairs, three rows and nine slot/counter reuses, QKV/shared-RMSNorm checks.

Audited full sweeps (mean per-case reductions; unpaired histories):

| Group / job | Cases | Dispatch | Cases >5% | Router→MegaMoE |
| --- | ---: | ---: | ---: | ---: |
| Q3EP2 /7121771 |86|8.856%|63|0.269%|
| Q3EP4 /7123534 |42|19.659%|42|1.008%|
| Q3EP8 /7124365 |20|16.222%|20|0.119%|
| Q3EP16 /7125100 |8|19.738%|8|-1.043%|
| Q3EP32 /7125159 |4|22.309%|4|0.654%|
| Q3EP64 /7125257 |4|38.698%|4|4.060%|
| Q35EP4 /7121770 |139|20.991%|139|-0.156%|
| Q35EP8 /7123069 |83|23.740%|83|0.457%|
| Q35EP16 /7124181 |39|24.557%|39|1.341%|
| Q35EP32 /7124704 |20|24.990%|20|3.690%|
| Q35EP64 /7124705 |19|30.877%|19|1.712%|

Dispatch uses each case’s best of two historical full sweeps. Ordinary MegaMoE
mean per-case gains5.122%/5.757% use those same selected histories (Q3/Q35).
Router→MegaMoE uses the prior frozen-combine full sweep,30calls/case/variant;
not cross-rank maximum or full-model latency. Full comparisons, all regressions,
sample audits and actual timed flags are under the compact-descriptor sweeps root.

Cleanup review covers kernels, graph scheduling, coordinator and reply ownership:
12files/1821net lines removed. Integration applied14files after final model gates;
all saved parents matched immediately before applying, and the staged patch is byte-
identical afterward.33staged communication files/preexisting edits preserved; no commit.
Final alignment topology dry runs and CPU7121801 native cache reuse PASS.
Both final alignments passed using cleaned source and prebuilt native/DeepEP/DeepGEMM
caches. Local source now matches those validated overlays. Evidence and requirement
audit: [completion](scratch/megamoe/dispatch_dense_descriptors_20260913/completion_audit.json).

## MegaMoE communication staging — September 13

Remote staging checks PASS: six shell launchers, all464saved-case contracts and
all11launch-group plans. Candidate code was synced before checks; no new GPU jobs.
Explicit33-file staging scope: scratch/megamoe/stage_20260913/stage_files.txt.
Remote workspace: FastAFD/artifacts/comm_stage_20260913/workspace; checks/summary.json
and local remote_checks.log retain evidence. No AGENTS/skill changes are needed.


User requested cleanup and git add only, no commit. The final validated communication
source is integrated locally, along with the EP64 reference-harness correction and
reusable launch/collection/report tools. Existing staged combine cleanup is retained;
unrelated attention/GDN runtime/model work stays unstaged. No kernel algorithm
changes during staging. Remote check evidence and explicit scope are under
scratch/megamoe/stage_20260913; detailed memory and rejection reasoning:
[CODEX_PROJECT_qwen35_push_comm.md](CODEX_PROJECT_qwen35_push_comm.md).

Final selection: pull dispatch + opt-in push-based in-band combine. Both normal
EP8/128K alignments PASS with complete pinned-reference coverage. Paired full-model
step means Q3 25.3333868→25.3435156ms, Q35 22.612068267→22.601254333ms: no meaningful
step gain. Most reduced ready-wait time already overlaps useful attention work.
All464cases/1566aliases,1392IKET/13920ordinary samples and23EP64bitwise checks pass.
Both same-format HTMLs and full plots complete; no task GPU jobs remain.
Artifacts: scratch/megamoe/comm_final_20260912/{results.md,completion_summary.json},
sweeps/{qwen3,qwen35}/*-megamoe-comparisons.html. Final pull remains faster than
matched tested push variants; no universal optimality or push-impossibility claim.

User's average E2E means MegaMoE invocation across shared historical/final sweeps:
164Q3/299Q35 cases, excluding newQ35global640. Mean per-case latency gains2.97%/3.77%,
exposed dispatch reduction-13.12%/-5.47% (increases), combine18.21%/21.01%.
Reduction of absolute mean latency differs:1.54%/2.63%. Historical runs/sample
policies differ; not paired causal A/B. sweep_reduction_report.md records both.

Historical checkpoints below are superseded where their status conflicts.

## Full communication optimization resumed — September 11

User explicitly resumed push dispatch + general MegaMoE communication optimization,
and final Qwen3EP16/Qwen3.5EP32 alignment/performance/decomposition sweeps, overriding
prior stop instructions below. Work autonomously, primary agent only; running-job
monitoring every3minutes. Current scheduler verified no active jobs; all seven latest
combine correctness/performance jobs COMPLETED0:0. Previous cleanup turn was progress.
Staged minimal pull/in-band combine remains intact; new experiments use frozen scratch
sources. Direct-release candidate removes only unnecessary sender acquire/return on
existing system arrival RMW, preserving consumer acquire/proxy/reuse. Exact7096621 COMPLETED0:0 in4:25; Q3/Q35 retained suites passed. CPU nvdisasm
confirms REDG system release without immediate producer invalidation. Independent
push-layout7096733 RUNNING4GPUs/15min, frozen payload/arrival/math unchanged;
destinations send count/offset packets to sources, replacing sender remote fetches.
Both new candidates passed all Q3/Q35 exact arms; compiled broad native sender
region maxR45 (<48) for all three specializations per candidate. Four-arm target
comparisons7096816 EP16/7096817 EP32 RUNNING16/32GPUs,40min limits,8primary trials
and4phase/full captures each. All5phase/30ordinary samples required; no promotion.
Plans/results: scratch/megamoe/comm_resume_20260911. Independent two-header metadata-
generation candidate prepared to replace the final cross-rank cleanup wait with
local completion, preserving local joins and using alternating metadata banks.
Protocol/validation plan: scratch/megamoe/metadata_generations_20260911/README.md.
It is LOCAL UNVALIDATED; no GPU job submitted while both target slots are active.
[Protocol and next gates](scratch/megamoe/direct_release_20260911/README.md).


## Code cleanup staged — September 11

User requested cleanup and git add only; no commit. Ten combine/runtime/test and
ABBA/profile/summarizer files staged. Cleanup changes comments/formatting only in
CUDA and removes obsolete push-dispatch marker analysis from the profiler. Remote
bash syntax checks passed; summarizer reproduced saved Q3EP8/Q35EP32 phase and
full-kernel captures (5/30 samples per arm). Existing GPU correctness/performance
validation remains applicable; no new GPU jobs. Cleaned files synced to local-mirror.
Unrelated model/attention changes, project memory and scratch experiments unstaged.

## Pull dispatch + minimal in-band combine: final measured state

User requested git restore, concise implementation, experimental in-band push
combine with pull dispatch, and BOTH Qwen3/Qwen3.5 exposed dispatch/combine plus
ordinary full-kernel metrics. Eight communication files archived/restored; current
LOCAL code is the validated fresh in-band parent (~60net production lines, ONE
DG_MEGAMOE_LAMPORT_COMBINE boolean). Two receive generations, pending-slot vector
polling, unchanged reduction order. No new counters/credits or model-specific tuning.
Default0 preserves epoch/TMA;1 enables in-band with rank-ready/rank-gated ordered
arena invocations. Unrelated model/attention changes preserved. No commits.

Six-arm exact7095226 PASS: small/Q3/Q35, normal/perfect, negative/zero/padded/
empty-source and repeated graphs. Fresh comparisons7095647/7095484/7095648 all
COMPLETED0:0; all4primary output checks and5phase/30ordinary rank0 samples per arm
retained. Same source, only combine flag differs. Epoch/TMA -> fresh in-band (us):

| Case | Dispatch median | Combine median | Full kernel mean | Harness ABBA |
|---|---:|---:|---:|---:|
| Q3EP8,192global |8.576->8.992|7.328->5.056|43.4624->43.5605|52.7414->53.0002|
| Q3EP16,192global |8.512->8.512|7.296->5.120|32.4256->32.0384|40.0890->39.7218|
| Q35EP32,640global |9.600->9.920|7.104->5.952|38.1984->37.8421|49.2810->49.2001|

Exposed combine improves1.152–2.272us; full-kernel improvement is NOT universal.
Q3EP8 mean+0.0981us, harness+0.2588us (+0.4907%, control drift0.0347us), kernel
median-0.048us. Tails grow0.544->2.912/0.832->3.136/0.640->1.632us respectively.
Direct intervals and legacy clipped exposure metrics both retained by summarizer.
Original7095338 failed closing-control exclusive-create reference, not kernel;
all subsequent timing/profile arms compare immutable refs. Sources/captures preserved.

One bounded structural follow-up moved the EXISTING epilogue grid join into dispatch
cleanup warps after final CTA handoff; no added counter/layout/math/knob. Three-arm
exact7095831 PASS. Performance7095905 Q3EP8 COMPLETED15:39 and7095906 Q35EP32
COMPLETED17:50, six primary trials +5phase/30ordinary per arm. New vs fresh parent
kernel mean43.8549->43.1979us (Q3EP8),38.6741->39.3707us (Q35EP32). REJECTED as
generic replacement; LOCAL header restored to exact tested parent. Q35 combine
6.016->4.736us but tail1.920->3.744us and kernel worsens. Q3 new/parent primary
52.5696/53.0737us with stable parent; Q3 epoch refs drift6.8973%, Q35 parent controls
drift1.7327%, so corresponding tiny/large ABBA gains are inconclusive. Q35 new
49.5933us vs stable epoch49.1912us also regresses. No causal split from phase medians.

No active task GPU jobs, no further variants/default promotion. Best retained
minimal implementation remains opt-in because e2e improvement varies by workload.
Raw rank0 .nsys-rep/CSV/decomposition triples local; no SQLite/all-rank bulk copied.
[Fresh results](scratch/megamoe/inband_combine_fresh_20260911/results.md),
[rejected follow-up](scratch/megamoe/inband_cleanup_join_20260911/results.md).

## Active combine goal: pull dispatch fixed, dependency-mask candidate — September 11

New user goal supersedes the blocked push-dispatch goal: pull dispatch + push combine,
expecting roughly0.6–1us exposed-combine saving. Current generic default already selects
that architecture; all new trial envs explicitly set three dispatch-push flags0.
Historical raw audit7088104 matches remembered0.896us lower combine exposure, but
combine code was IDENTICAL across those dispatch variants. Pull/early push combine
6.976/6.080us; full kernel32.2965/33.0848us mean (all5phase/all30ordinary retained).
This does not establish a transferable combine gain. Audit saved locally.

Current LOCAL UNVALIDATED combine_rank_mask candidate replaces per-rank any-vote
loops with one warp OR reduction in both combine geometries, EP<=32. Invalid lanes
contribute0; duplicate producers collapse; EP>32 existing per-slot path unchanged.
No payload/math/barrier/epoch/cleanup change, no tuning knob. Parent header and exact
patch saved. Exact7094952 SUBMITTED4GPUs/15min: retained Q3/Q35 normal/perfect,
empty/padding and repeated-graph suites with pull dispatch. EP16 ABBA+5phase/30kernel
plan prepared UNSUBMITTED until correctness passes. No push-dispatch/full-model jobs.
Async clarification asks whether user intended retained producer-epoch combine or
revisiting archived in-band combine; after response window, proceeding with former
as stated to user. No in-band rewrite begun. No performance or completion claim.
[Protocol and raw audit](scratch/megamoe/combine_rank_mask_20260911/README.md).

## Current direction: pull dispatch and existing push combine — September 11

User explicitly chose to stop pursuing push dispatch and retain pull dispatch +
producer-push combine. This supersedes the old push-dispatch optimization objective
and all historical pending next steps below; do not resume them automatically.
Cancelled sole active push-dispatch comparison7094758: Slurm CANCELLED in6:58;
partial primary logs saved locally, no completed phase/full-kernel comparison.
Current default dispatch flags are all0 (pull); AFD adapter enables rank-gated
combine. Existing experimental code and separately accepted full-model snapshots
are preserved; no kernel/config changes or new GPU submissions in this review.

Combine already pushes BF16 contributions into known source-token/top-k slots,
then performs local TMA reads/reduction after required producer epochs. Evidence:
- Isolated rank-gated completion ABBA6597585:93.377 ->91.371us (-2.149%),
  bitwise exact, fused routing held fixed; harness timing, not new IKET exposure.
- Accepted top-k8 CTA local combine ABBA7014107:47.936 ->45.104us kernel wall
  (-5.91%). Comparative IKET7014576/7014768 coarse exposed combine11.392 ->7.680us,
  compute proxy essentially unchanged. Evidence in CODEX_PROJECT_qwen3_ffn_overheads.md.
- These prove improvements within producer-push combine, not a pull-versus-push
  combine transport A/B. Later in-band combine variants were rejected:7076249
  +7.40%;7076436 original+2.91%,async-cleanup+1.47% against warmed harness controls.
- Recent dispatch-only smaller exposed combine is not an independent combine gain.
Future evaluation must retain exposed dispatch, exposed combine, and ordinary full
MegaMoE kernel time (including completion tail); no new case-specific tuning.

## Source-prefix validated; performance running — latest September 11

Current LOCAL/frozen direct_source_prefix passed exact7094662 COMPLETED0:0 in4:40,
both retained Q3/Q35 suites (sparse/empty sources, normal/perfect routes, graph reuse).
Logs/summary/Q3+Q35 SASS local; max explicit R45 in reviewed ordinary region under48.
Source inclusive prefixes replace dead raw send counts, use existing layout epoch,
and support binary work lookup. No payload/math/combine/flag/allocation changes.
Matched EP8 comparison7094758 RUNNING8GPUs/35min on nvl72086-T[02,08]; same-source
pull, destination-layout parent and source-prefix candidate. Six primary trials and
3 phase/full captures required. Only7094758 active; no commits or large EP jobs.

Parent7094534 COMPLETED0:0 in17:07, all6primary/all3rank0 triples local. Pull/raw-
direct/layout-ready net15.872/50.432/44.160us, full means44.1504/80.7019/75.6085us.
Layout helps versus raw direct but loses BOTH to pull. All5phase/all30ordinary retained.
Primary51.954880/85.501534/82.208752us, control drift2.80698% limits small-gain claims.
Original7094394 complete16:41: pull/inbox/direct net16.352/20.736/48.384us, full
44.0309/48.3403/79.4528us mean; all captures local. Goal remains unachieved.
Potential later tests: release-only system arrival reduction (existing producer
acq_rel emits CCTL.IVALL; PTX ordering rationale recorded), or contiguous SF transport
with local transpose while X stays direct. Neither implemented; await current evidence.
[source-prefix context](scratch/megamoe/direct_source_prefix_20260911/README.md),
[completed layout comparison](scratch/megamoe/direct_layout_ready_20260911/qwen3_ep8/analysis.md).

## Destination-owned layout candidate — latest September 11 continuation

Current LOCAL source UNVALIDATED direct_layout_ready candidate. CTA0 metadata warp
publishes padded expert offsets once into unused legacy receive-sum slots, releases
a layout epoch in existing128B padding; senders read only their expert's offset/counts.
Removes serial prefix metadata reads, retains direct-pool payload/tile-ready/compute/
combine/reuse behavior. No allocation-size/API changes or case-specific tuning.
Exact7094482 RUNNING4GPUs/15min on nvl72076-T16, latest1:13, after both headers
synced into frozen source. Original7094394 RUNNING8GPUs, latest8:37: all6primary
complete; means pull51.545988,inbox56.566696,direct85.352123us router+kernel.
Direct loses33.806135us/+65.5844%; pull drift0.04106%. Profiles now underway.
Balanced EP8 sends1536 activation rows in BOTH push designs; old metadata prefix
averages8.5 sequential iterations(max16). Hypothesis, not stall attribution.
Max2active jobs, no layout-candidate performance submission or commits. Details:
[layout candidate](scratch/megamoe/direct_layout_ready_20260911/README.md).

## Direct-pool correctness passed; matched EP8 comparison running — September 11

Current LOCAL and frozen ffn_direct_pool_20260911 native push writes directly into
final expert pools, replacing inbox gather. Exact7094309 COMPLETED0:0 in8:03:
same-source pull and direct push BOTH pass Q3/Q35 retained exact suites, including
normal/perfect routing, zero/padding/empty-source and repeated graph invocations.
Logs/summary/SASS local. Q3 sender+cleanup highest explicit R43 within48-register
allocation; compiled global proxy fences and system arrival atomic present.
Short EP4 validation timings are slower than pull; no performance win claimed.
Matched EP8 three-arm7094394 RUNNING8GPUs/35min on nvl72153-T[17-18], latest0:18.
Arms: same-source pull, retained four-load/poll inbox, direct-pool. Six mirrored
primary trials and3 phase/full-kernel captures; profile pull first to preserve a
control if candidate instrumentation fails. All5phase/all30ordinary samples required.

DNS recovered. Previous7093619 FAILED143:0 in13:32 during candidate IKET capture
(CUDA715 illegal instruction); all8primary trials complete, candidate ordinary
kernel47.8614us mean/47.888us median with all30samples retained locally. No phase
capture or matched full-kernel control; no acceptance claim. Compiled marked kernel
inspection did not establish the cause; do not assume a register-limit fix.
Batch7093721 COMPLETED0:0 in7:00, both Q3/Q35 exact suites pass; terminal logs and
summary local. Batch performance remains unsubmitted, architecture superseded.
Only7094394 active; no larger EP/full-model jobs or commits. Goal unachieved.
Code/current memory synced to local-mirror; frozen candidate unchanged during runs.
Protocol, cost tradeoffs, parent source, patch and artifacts:
[direct-pool prototype](scratch/megamoe/direct_pool_20260911/README.md).

## Architectural retrospective — September 11, 2026

User asks to reconsider the base instead of continuing marginal optimization.
Review finds explicit-flag push still delivers to a source-indexed inbox and retains
the receiver gather/compact-pool dependency. Latest complete matched comparison
still loses BOTH exposed net and full kernel to pull. Stop further small variants
and leave batch performance plan unsubmitted. Recommended next design is direct
final expert-pool delivery with tile readiness; account for metadata/layout handshake
and loss of same-destination activation deduplication before claiming a benefit.
Keep numerical core, pull control and established correctness/reuse principles.
Detailed evidence, rewrite boundary, combine caveat and decision experiment:
[retrospective](scratch/megamoe/push_dispatch_retrospective_20260911.md).
7093619 has all8primary locally (best55.676056 vs pull51.264824us router+kernel);
phase/full captures uncollected. Last observed7093619/7093721 RUNNING13:18/4:02;
current status UNKNOWN after normal+escalated SSH DNS failure. Local batch code
remains UNVALIDATED. No new jobs, code edits or commits during review. Earlier
queued/partial-trial descriptions below are historical. Goal remains unachieved.

## Active general push-dispatch investigation — September 11, 2026

Acceptance requires BOTH lower exposed dispatch/combine and lower full MegaMoE
kernel duration. Exposed communication ends at CombineEnd; full kernel includes
the later tail. Keep all5 phase samples and all30 ordinary Nsight samples. Net
exposure is median of per-sample dispatch+combine. Q3EP8/EP16 and Q35EP32 are the
targets; no new full-model runs, commits, subagents or shape-specific tuning.

Current LOCAL source UNVALIDATED per-source batch-ready candidate; frozen
ffn_signaled_batch_ready_20260911, exact7093721 queued4GPUs/15min. Last-sender GPU
ticket gathers stores, system-release epoch per destination; explicit observers
cover zero-row sources before reset/reuse. Existing128B padding holds local count,
no allocation/layout-size change. Three-arm performance plan unsubmitted.
7093441 complete23:02, all8primary/all4rank0 captures local. Pull/two-load/four-load/
four-load+poll net16.416/21.216/19.552/19.680us, full kernel means
44.5589/49.3728/48.3733/48.3349us. Four-load helps but still loses both to pull;
polling has no useful benefit established. All5 phase/all30ordinary samples kept.
7093619 eight-load/field-prefetch comparison RUNNING8GPUs; first timings give small
gains, remaining trials/profiles pending. Parent eight-load exact7093597 passed
both models5:16. Max2active; no larger EP/full-model jobs or commits. Goal active.
Completed7093123 (21:32), all8primary/all4capture triples local. Pull/full-wait/
early-ready/paired net15.680/24.672/21.984/20.416us, full kernel means
43.7493/53.6160/51.0229/49.3589us. Paired improves both versus prior push but
STILL loses both to pull. All5 phase and30 ordinary samples retained. Goal active.

Completed receiver7092404: pull/three-TMA/four-TMA/four-vector exposed net
16.352/22.112/25.440/26.272us; full kernel means43.7963/55.6832/55.9712/55.8997us.
Completed async7092547: pull/sync/async net15.392/25.280/24.576us; full kernel
means44.1226/57.0197/55.3621us. All traces and raw samples collected locally;
no push win on either target. Earlier store/poll7092173 also no win, with weak-
only phase capture explicitly missing due to fixed launcher env leakage.
All immutable experimental paths, validations, failures, results and next actions:
[CODEX_PROJECT_qwen35_push_comm.md](CODEX_PROJECT_qwen35_push_comm.md).
Root AGENTS.md absence already raised; handover read and deleted as directed.

Recent full-attention/GDN work and accepted model-level results are separate:
[CODEX_PROJECT_qwen35_full_attention.md](CODEX_PROJECT_qwen35_full_attention.md).
The A32:F32/EP32/B40 GDN stream experiment7083416 measured22.372317/22.636225ms,
FFN execution streams92→2; both metric-selected trace pairs are local. Preserve
unrelated dirty runtime/model changes. Older checkpoints below are historical.

## Accepted Qwen3.5 full-attention placement — September 11, 2026

Attention workers now run only full attention; FFN workers own GDN/linear layers.
Accepted accumulated scheduling/stream-reuse/push-dispatch candidate confirms
A3:F1/EP16/128K maximumB40: same-allocation28.432393->26.791300ms (-5.771913%),
1119.766491TPS/GPU,15samples/0outliers, unchanged metric gates. Measured transport
uses DG_MEGAMOE_LAMPORT_DISPATCH=1; generic default0 retains TMA for other workloads.
Final normal-routing A3/EP8/B7 alignment passes2856/2856 top10 (2853 top1), exact
ordered token coverage verified. Jobs7077121/7078525/7078816 all COMPLETED0:0;
no task GPU jobs remain. Cleaned source/memory synced with R/local-mirror; no commit
or push requested. Details, source snapshots, traces and rejected experiments:
CODEX_PROJECT_qwen35_full_attention.md and CODEX_PROJECT_qwen35_push_comm.md.

## Completed cleanup — September 10, 2026

Comprehensive runtime/script/test cleanup removed905 net code lines;consolidated
scheduling and test setup,removed one-shot scripts and obsolete scratch artifacts.
Final source/docs synchronized to R/local-mirror. User requested committing this cleanup together with the already staged code; no push requested.

Both Qwen3 and Qwen3.5 pass attn-only A2:F1/EP8/b7/128K/MB2 alignment:
1904/1904 tokens top10 for each (Q3 top1=1902,Q35 top1=1904). Both same-allocation
performance comparisons pass unchanged graph-span/stability policy,15samples,
0outliers:Q3 25.259388->25.258882ms;Q35 15.943200->15.943236ms. No material
regression observed. Focused CPU suite63/63 passed;no task GPU jobs remain.
Details,evidence,failed preparation attempts:CODEX_PROJECT_cleanup.md.
Control=R/source/code_reduction_control_20260910. Source patches/checks/plans:
scratch/afd/validation/cleanup_20260910 (full original source retained remotely).

**Nsight copy-back policy (user instruction):** when retrieving traces, copy
exactly one representative attention trace and one representative FFN trace
(two `.nsys-rep` files total). Do not copy other ranks' traces or unrelated
result files. Keep the full trace set remote; retrieve only the compact results
needed for the requested analysis. Store traces by case, then optimization version
under scratch/afd/cases/; follow CODEX_PROJECT_nsys_traces.md. No trace-management
script: this agreement is the workflow for future sessions. Every retained
performance version must have its two metric-selected reports; JSON-only
performance versions are now filled (207 versions,414 reports). Unprofiled alignment has no traces.

Local scratch now has only `afd/` and `megamoe/` (562MB total, including all retained performance trace pairs). AFD case data
and traces live in `scratch/afd/cases/<model>/isl_<tokens>/ep_<EP>/a<A>-f<F>/batch_<batch>/<optimization>/`; shared
reports in `scratch/afd/reports/`; active cleanup checks/plans in
`scratch/afd/validation/cleanup_20260910/`. Completed standalone MegaMoE sweeps
are in `scratch/megamoe/{qwen3,qwen35}/`. Superseded experimental directories,
numbered v1/v2/etc snapshots, incomplete captures and outdated plots were removed.
Historical local scratch paths below are superseded by this layout; remote
artifact paths are unchanged.

## Previous checkpoint — September 10, 2026

All requested sweeps and trace deliveries are complete; no monitoring or new
GPU jobs remain. Keep the existing AFD maximum representative GPU graph-span
timing, including sampling/writeback (user confirmed after TRT comparison).

- Original, early-attention, and final graph-edge AFD sweeps: 66/66 each.
  Final vs early: TPS/GPU geometric mean +1.03183%; batch7 +2.43169%,
  batch6 +2.80366%; 50/66 faster. All final jobs7061091–7061097 completed0:0.
- Qwen3.5 decomposition: 299/299 inputs,897 samples,1110 aliases audited.
  One persistent container per node/job; one EP per job,2hr,max8 outstanding.
- Qwen3.5 A3:F1/EP16/max batch38: job7063495 passed;31.02785ms,
  918.52966TPS/GPU; all64 Nsight traces local. No matching old B38 baseline.
- Both requested Pareto HTMLs contain the final curve; `afd_optimized` retains
  its label. Max-TPS Qwen3 A8:F1/EP8/b7 all72 traces copied locally.
- User authorized promotion: final Qwen3/Qwen3.5 scheduling and K10 barrier
  fix are in main code. Six superseded remote snapshot repos and five local
  overlays removed; compact patches and results retained. Changes are for
  git-index staging only, without commit or push.

Task memory and reusable workflows:

- `CODEX_PROJECT_pareto_perf.md`: original sweep, machine cross-check,
  64-bit combine-address fix, correctness and same-GPU performance evidence.
- `CODEX_PROJECT_attn_release_perf.md`: earlier scheduling comparison.
- `CODEX_PROJECT_publish_entry_perf.md`: cancelled intermediate experiment.
- `CODEX_PROJECT_attn_graph_edge.md`: final scheduling, alignment/sweep,
  trace analysis, timing scope and current cleanup checkpoint.
- `CODEX_PROJECT_qwen35_graph_edge.md`: hybrid attention/direct GDN transport.
- `CODEX_PROJECT_qwen35_megamoe_sweep.md`: decomposition/capacity/idle fix.
- `codex_scripts/megamoe_input_sweep.py` and `.sbatch`: plan, run, collect,
  audit either model; `profile_megamoe_iket.sbatch` profiles each input.
- `codex_scripts/summarize_afd_sweep.py`, `compare_afd_sweep_metrics.py`,
  `plot_afd_sweep.py`: validate, compare and plot AFD results.
- `codex_scripts/analyze_afd_attention_gap.py`: CPU-only paired Nsight analysis.

The sections below retain earlier project history. Task documents' completed
results supersede their historical pending/running entries.

## Padding-fix amendment checkpoint (2026-09-09)

User requested cleanup, amendment into the latest MegaMoE commit, then push
fmha_only to linsj20. Cleanup is limited to formatting and stale smoke-test
names (prefixes/sources become validity/row_map); validated runtime behavior
is unchanged. Sync cleaned source, tests and scripts to R/local-mirror before
checks. Focused CPU padding-replay regression and launcher bash -n pass there;
local git diff --check passes. Existing GPU/alignment/kernel evidence below
is retained; no new GPU jobs or e2e timing. Preserve both frozen source variants.
Include code, reusable validation scripts and project memory in amendment;
exclude all generated results/reports. Existing remote fmha_only is an ancestor,
so use a normal push to linsj20, without force. No global skill/rule changes
needed. CPU check command uses PYTHONPATH=python and the pinned environment
R/../fastafd_reproduce/envs/minisgl-3c716194-cuda130.artifact-5175891/bin/python
on the login node (system python lacks torch); tests/test_afd_fmha_runtime.py
AfdFmhaRuntimeScheduleTest.test_decode_replay_preserves_each_sources_padding -v.
Local check log: scratch/megamoe_cpu_validity_20260909/amend_checks.log.

## Remove GPU padding arithmetic (2026-09-09, validated and retained)

User requested moving padding overhead off GPU and validating actual AFD
performance. Candidate CPU-prepares [total, compact_index_per_row] with -1
for padding, replacing per-source GPU division/prefix arithmetic with a direct
row load. Normal routing does not consume total; perfect routing uses the same
compact expert sequence. Scalar/no-mask API paths remain unchanged. CPU fills
the pinned map with vectorized NumPy operations and an allocation-time iota;
one H2D copy per MB remains, payload grows16->72B for A2/8 padded rows.
No new kernel/event or communication change. Old fully aligned source remains
R/source/megamoe_source_padding_fix_20260909; new candidate is
R/source/megamoe_cpu_validity_20260909. Candidate included in amendment above.
Artifacts use R/artifacts/megamoe_cpu_validity_20260909 and matching scratch.
Focused remote CPU replay regression passed1/1 in0.032s. CUDA router check
7038505 COMPLETED0:0 in1:48,4 GPUs/1tray. Same8 correctness shapes passed.
ABBA router graph means:8 rows scalar6.2832/map6.39008us (+0.10688);
32 rows scalar8.19184/map8.32320us (+0.13136);128 rows scalar8.48704/map8.58720us
(+0.10016). Source arithmetic removed from GPU; lookup overhead remains,
so do not claim zero GPU overhead. Old per-source calculation added~0.2552us.

Completed7038673 (0:0,45:05): real A2:F1/EP8/b7/128K/MB2/normal-routing source pair
using existing run_afd_paired_sources.sbatch, one6-tray allocation,short1hr.
Control=aligned per-source-prefix fix; candidate=CPU row map. Each source gets
same112 prompts/17 tokens,memory0.97,NSYS15-step timing; no new vLLM. Compare
both output bundles with saved vLLM references and extract existing strict
dual-role CUDA graph span metrics. Config normal_pair.tsv and
plans/normal_pair_env.json,plan normal_a2.csv,pools normal_{control,candidate}.
Submitted manually after router check completed; monitored every3min.
Pair logs are pair-7038673.out/.err. Both frozen source snapshots preserved.
Both cached alignments pass:1904/1904 top10,1902/1904 top1. All112 prompts
and1904 generated tokens byte-identical between sources. No new vLLM run.
Whole-step GPU mean27.59092->27.83018ms (+0.867%) initially raised concern;
user explicitly directed kernel performance acceptance instead, since e2e
is noisy. No additional e2e performance run. Both sources used same24 GPUs
nvl72106-T[01-06], and batch7 capacity gate passed (993024 KV tokens).

Saved actual AFD model:rank17 Nsight traces confirm router improvement.
Exclude first graph launch as warmup; retain all15 remaining launches,
2079 kernel nodes/launch,188 calls/family/step (2820 total per source):
- Gate GEMM mean4.418855->4.419525us; median4.416->4.416us.
- Route selector mean8.213515->8.034667us (-0.178848us,-2.177%);
  median8.288->8.064us. Improvement appears in every measured step.
- Sum of router kernel means12.632370->12.454191us (-0.178179us).
- MegaMoE mean63.126272->63.369651us (+0.386%); median61.312->61.376us
  (+0.104%); p9566.143->66.496us. Essentially stable in this pair, not proof
  of exact performance equality. All outliers retained (max1263.040/1179.008us).
Retain CPU-map candidate: real router benefit, no material MegaMoE regression
observed, exact output preservation. Source/test/scripts synchronized to
local-mirror; included in amendment above. No jobs remain. Kernel summary.json and per_step.csv
under artifact kernel_analysis/ and matching local scratch; pair summary and
cached alignment JSONs retained there too. Nsight duration=end-start; router
sum excludes launch gaps. Kernel results are for profiled rank17, not all ranks.
CPU-only trace extraction: login host lacked nsys, so unsquashfs extracted only
the existing image's NsightSystems-cli-2025.5.1 subtree into artifact cpu_tools/;
target-linux-sbsa-armv8/nsys exported kernel/StringIds tables to SQLite without
allocating GPUs. Full SQLite and reports remain remote.

## Fan-in padding fix and exact A2 rerun (2026-09-09, PASSED)

User requested fixing failed A2:F1/EP8/b7, checking perf impact, then rerunning
the SAME A2 case. Explicitly no new vLLM run: reuse saved prompts/references.
Root cause: merged MB0 rows are [RRRP][RRRP] but scalar valid_count6 masks
as [RRRRRRPP], routing a dummy and dropping source1's third real request.
This predicts all eight severe failures at prompt indices33,35,...,47.
Fix uses cumulative per-source valid counts ([3,6]) in existing pinned/device
graph buffers and H2D copies. Shared CUDA helper derives source-local validity,
compact real-token index and total; fused Qwen3/Qwen3.5 and group-local selectors
use it. Perfect routing uses compact index, preserving balance despite holes.
No extra kernels/events/synchronization; scalar/no-count behavior retained.
Runtime and router code are changed locally; candidate snapshot is
R/source/megamoe_source_padding_fix_20260909 based on de837f5 plus these edits.
Candidate retained as the validated snapshot. Artifacts/plans/logs under
R/artifacts/megamoe_source_padding_fix_20260909; matching local scratch dir.

Focused remote CPU replay test passed1/1 in0.032s (batch7,unequal,empty source).
CUDA router check7037970 COMPLETED0:0 in1:50:8 shapes, E128/top8 and E512/top10,
fused/group-local valid rows byte-exact vs unmasked routing, invalid rows -1/0.
ABBA CUDA-graph router timing on identical full8-row work: scalar6.4864/6.4643us,
per-source6.7456/6.7155us, ~0.2552us added per router call. This is router-only
overhead, not an end-to-end measurement. No MoE arithmetic or comm change.

Exact failed-case AFD rerun7038012 COMPLETED0:0 in21:10, after check ended:
attn-only A2:F1,EP8,ATP1,batch7,MB2,normal routing,FP8xFP4,128K,112 prompts,
17 output tokens,24 GPUs/6 trays,short1hr,memory0.97,capacity-max gate,NSYS0.
Run basename afd_qwen3_131072_r2_ag16_fg8_atp1_fep8_adp16_b7_n6_20260909_173113_manual_na.
No scorer job. codex_scripts/compare_cached_alignment.py passed using its
experiment/sample.json with saved refs:
- R/artifacts/megamoe_alignment_a2_ep8_b7_20260909/alignment_score/alignment.json
- R/artifacts/megamoe_final_e2e_20260909/alignment_score_retry/alignment.json
The earlier passing A1 scorer covers the exact same first48 prompts, including
all eight corrupted cases, with fully top1 continuations. Newer cached scores
cover all112 including prompt87's small rank<=3 deviations. Comparator only
reuses scores at identical prompt and generated prefix; missing coverage must
fail explicitly, never claim alignment beyond saved context. Monitored every3min.
Final cached alignment:112 prompts/1904 tokens,1904/1904 covered,1904/1904
top10 (100%),1902/1904 top1 (99.89496%),no missing reference context.
Exactly the eight corrupt prompt outputs changed (33,35,37,39,41,43,45,47);
all112 input prompts identical and all other104 continuations unchanged.
The two non-top1 tokens remain the original prompt87 positions8 and13,
both rank3, unchanged by the fix. All eight corrupt cases now match the old
fully-top1 continuations. This passes the official top10 criterion, not a
claim of100% top1. Local/remote compact summary.json and cached_alignment.json
retained with original full remote samples. No jobs remain for this task.
Fix and test scripts synchronized to local-mirror; no new commit requested.
Cached comparator reproduced the original failed A2 score exactly:1904/1904
covered,1810 top1,1820 top10,exit1 expected. No vLLM rerun or GPU involved.
Clarified to user: this modifies the existing router, not a new kernel/call.
Any multi-source AFD graph with internal padding is affected by the old scalar
mask; batch7/MB2 A>=2 is one instance. A1 has only trailing padding; exact3+3
batch6 has none. Standalone MegaMoE sweep does not use the faulty merged AFD
layout. Old affected AFD timings are not numerical validation; confirming
performance claims on corrected routing remains separate from this alignment.

## Current-code A2:F1 alignment (2026-09-09, FAILED)

User requested attn-only A2:F1, EP8, batch7 alignment on current code.
Generation job7036680 COMPLETED0:0 in20:55 on24 GPUs/6 trays,
nvl72114-T[04-05,07,09-11], short QoS/1hr. Frozen source commit de837f5
under R/source/megamoe_alignment_a2_ep8_b7_20260909; R is the consolidated
remote root below. Outputs/plans/submission.log under
R/artifacts/megamoe_alignment_a2_ep8_b7_20260909. Local matching scratch dir.
Settings: normal routing, perf0, FP8xFP4,128K,ATP1,MB2 real3+4/padded4+4,
16 attention +8 FFN GPUs,112 prompts,17 output tokens per prompt expected.
Batch7 requires memory_ratio0.97 (nominal KV979200), capacity-max gate enabled;
0.82 only fits batch6. Dry run passed. No Nsight for this alignment run.
User then explicitly requested a dependent scorer; submitted7036724 manually
with afterok:7036680 and kill-on-invalid-dep=yes,4 GPUs,short QoS/1hr.
This authorizes a queued dependent job while retaining one running allocation.
Scorer7036724 FAILED1:0 in25:11 on nvl72069-T12 after completing all112
requests. Alignment gate failed; no jobs remain for this test.
Scorer uses same frozen source/model profile, sample.json from generation run,
OUTPUT_ROOT=<artifact-root>/alignment_score. Submission/env saved beside plans.
Current-code e2e alignment FAILED:1904 tokens, top1=95.06302521008403%,
top10=95.58823529411765%,top100=96.21848739495799%;94 non-top1 tokens,
84 outside top10,max rank60564. Nine prompts contain non-top1 tokens:
zero-based indices33,35,37,39,41,43,45,47,87. The first eight account for all
top10 failures; prompt87 only has two rank2/3 deviations. Other103 prompts
have100% top1 agreement. Pattern noted, cause not established. No rerun or
code fix attempted under the monitoring request. Compact alignment_summary.json
saved in local matching scratch and remote artifact root; full alignment.json
remains under remote alignment_score. Earlier A1:F1/b6 pass below predates
EP64 fixes in de837f5 and covers a different shape; do not equate validations.

## MegaMoE progress checkpoint (2026-09-09)

User requested a git commit after the completed optimization/sweep/report.
Checkpoint includes runtime fixes, profiling/sweep/validation tooling, tests
and project memory. User explicitly excludes result files and reports from
the commit; generated CSV/HTML/audit artifacts remain under scratch and on the
remote artifact tree. Stale ordered-combine/EP-range test expectations corrected.
Four focused CPU contract tests and seven shell syntax checks passed remotely
against synchronized code; no new GPU run. Existing byte-exact, e2e alignment
and164-case evidence retained. No global skill/rule changes were needed.

## Completed MegaMoE decomposition sweep (2026-09-09)

See `CODEX_PROJECT_megamoe_sweep.md`: EP2–64, MB2, batch1–7 /1–4 tokens
per attention GPU, all deduplicated NVL72 A:F inputs, integer-balanced routes.
All164 unique cases /455 AFD aliases complete, each with three IKET samples
and its own decomposition CSV. Final7028073 COMPLETED0:0 in39:22. Artifact
audit passed492 samples and all15 successful capture jobs (serial submissions,
short QoS,one-hour limits). No sweep jobs remain. Results, comparison tables,
allocation records and audit: scratch/megamoe_input_sweep_20260909/results/.
EP64 is the explicitly labeled A1:F8 exception within NVL72; its~6us cleanup
tail and retained IKET dispatch outliers remain future study, not sweep blockers.

## Validated MegaMoE latency result (2026-09-09 UTC)

The requested optimization and final validation are complete; no GPU jobs
remain active for this task. Main code and local-mirror retain the early
routing-weight read, CTA bulk-TMA combine with original ascending valid-slot
PTX accumulation, and the validated short-K producer prefill correction.
Pairwise top-k8 summation is removed because it can change results. Prefix
lookup was rejected: its 0.112 us apparent gain was below 0.416 us control drift.
Included in the MegaMoE progress checkpoint; unrelated Qwen3.5 work is preserved.

- Byte-exact source comparison 7019473 passed on 8 ranks: original control,
  control repeat, ordered combine/early-weight and prefix each had 56 records.
  Independent producer quantization also passed. No extra numeric matrix.
- Final six-point profile 7019787: Nsight kernel median 44.064 us; dispatch
  span 10.304 us; compute span 26.336 us; exposed combine 7.296 us;
  post-CombineEnd tail 0.512 us median / 0.800 us max. One dispatch startup
  outlier of 16.224 us is retained. The 6–7 us dispatch aspiration is still
  unmet; the interim ~8 us exposed-combine/no-long-tail target is met.
- Final A1:F1/EP8/batch6, 128K, ATP1, MB2, attn-only, FP8xFP4 generation
  7019832 passed. Official scorer 7020231 passed all 48 prompts/816 tokens:
  top-1/top-10/top-100 = 1.0, average/max vLLM rank = 1.0. Individual steps
  checked. Initial scorer 7019850 failed before scoring on a too-long Unix
  socket path; fixed only transient IPC location and reused the same sample.
- Same-allocation perf-mode source pair 7019851: old 24.098079 ms versus
  retained 23.553900 ms, improving 2.258%; TPS/active GPU 124.491253 to
  127.367442 (+2.310%). Both use perfect balance and all 16 GPUs; unchanged
  policy retains 14/15 control steps (1 outlier) and 15/15 candidate steps.
  This is one paired result, not an extrapolation from standalone timings.

Full concise results: scratch/megamoe_latency_20260908/RESULTS.md.
Resume details, source/artifact paths and historical evidence:
scratch/HANDOFF_megamoe_latency_20260908.md. Local compact proof is under
scratch/megamoe_latency_20260908/results/{final_e2e,final_ordered_iket,exact_results}.
The earlier fixed-global-work EP4/8/16 test supports roughly inverse-EP compute;
its 22.128 us fitted fixed term is from earlier code and is not pure wire latency.
An unrelated E<128 quantization-coverage correction remains isolated and untested
on GPU; it was deliberately not expanded under the minimum-checking instruction.

## Required remote workspace layout

The consolidated remote root is
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD`.
Keep its three roles separate:

- `local-mirror/` is the accepted integration point and execution mirror of
  the authoritative local checkout. Any change accepted from a reference or
  candidate variant must be merged into the local checkout first and then
  synchronized here. Do not develop a divergent implementation in this tree.
- `source/<variant-name>/` is the only location for future reference branches,
  experimental source snapshots, or candidate code variants. A variant is not
  accepted merely because it exists here.
- `artifacts/<subproject>/<run-name>/` is the required location for every new
  run directory, including logs, results, metrics, traces, generated caches,
  and archive payloads. Keep archive manifests, hashes, summaries, and indexes
  beside their corresponding archives.

Do not place generated results in `local-mirror/` or `source/`, and do not use
an artifact or variant tree as the accepted development source.

## Qwen3.5 AFD support and remote workspace

See `CODEX_PROJECT_qwen35_support.md` for the complete 397B implementation,
alignment, tracing, and remote-cleanup record. The primary local `fmha_only`
checkout is authoritative. Its sole accepted execution mirror is
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD/local-mirror`;
preserved Qwen3/Qwen3.5 evidence and archives live under the sibling
`FastAFD/artifacts/` tree.

## Historical A8:F1/EP8 MegaMoE investigation (2026-09-08 PDT)

Exact standalone input:24 rows/EP rank, H4096/I1536, E128/top-k8,
FP8 activations/FP4 weights,150 SMs,bucket3840,two trays,prepared production
route metadata and perfect balance. Perfect balance is never a kernel
assumption; preserve arbitrary routes and uneven source counts. User explicitly
requires first-principles memory-path reasoning and no target overfitting.
Target6–7us per communication leg; no tile/SM/threshold tuning. Nsight kernel
wall time is the oracle; no harness cadence or new clock64 timing.

Accepted top-k8 CTA-per-token bulk-TMA/pairwise combine: source ABBA7014107
47.936 ->45.104us(-5.91%). Legacy warp-only IKET7014576/7014768 showed exposed
post-compute combine tail11.392 ->7.680us; those are not full comm durations.
M16(7015208),parallel TMA issue(7015677),CTA-token dedup(7016290),warp-token
dedup(7016530),and aggregate combine barrier(7016706) are rejected. Dedup
branches/layout/gate metadata were removed locally; accepted combine remains.

Rejected candidate `source/megamoe_direct_dispatch_20260908` replaces dispatch's
peer-global ->shared ->local-global bounce with coalesced copying through
registers, preserving scheduling/routing/publication. `l1_token_buffer` denotes
Linear-1 global staging, not L1 cache. Job7016924 passed numerical checks and
measured47.632 ->45.984us(-3.46%), but control drift2.464us prevents acceptance.
Validation7017020 completed0:0 in15:34:24 rows46.336 ->46.336us,
12 rows44.464 ->44.512us,48 rows49.296 ->49.072us (mean durations regress
49.147 ->49.440us). Reject direct copy as neutral/noisy; retain TMA dispatch.
Outputs: `artifacts/megamoe_direct_dispatch_20260908/validation/rows{24,12,48}`;
local CSVs and summary under `scratch/megamoe_latency_20260908/results/direct_dispatch/validation`.


Official IKET CUDA C++ marker APIs correct the earlier unsupported-API claim.
Profiling-only compiler/kernel support plus `profile_megamoe_iket.sbatch`
now define six explicit source points and overlapping phase envelopes. Normal
builds compile markers out. First capture7016822 exposed a two-pass peer-launch
defect and then invalid NativeDump event records; timestamps are rejected.
PMTRIG5/two-pass-peer job7016973 failed because installed IKET lacked SModel.
Official SDK documents predicated-marker ghost slots explaining the earlier
NativeDump overflow. Isolated SModel dependencies were prepared on the login
CPU and resource selection passed for all six markers in the captured binary.
Retry7017202 completed0:0 in3:02 with512 slots/warp and profiler-only
`source/iket-smodel` dependencies. Three measured launches have all six
ordered boundaries. Median dispatch11.616us, compute24.448us, combine18.976us;
combine overlaps compute and exposes8.064us afterward. These are software
envelopes, not pure NVLink latency. Local validated decomposition/trace:
`scratch/megamoe_latency_20260908/results/iket_markers/7017202`.
Control7017340 completed0:0 in4:22: exposed combine10.560us versus8.064us
accepted; source compute spans24.960/24.448us. Its one long dispatch outlier is
retained. Accepted post-CombineEnd tails0.960/1.344/1.600us show no long tail in
three measured launches. User accepts~8us for now with this condition.

User next requested EP scaling and a `T(EP)=A/EP+C` latency-floor estimate.
Active EP4/8/16 jobs7017478/7017479/7017480 use1/2/4 trays respectively (the
new scaling request expands the former exact-case two-tray constraint), all
GPUs active. Fixed192 global tokens,48/24/12 rows per rank,12 routes/expert,
model/precision/150SM/bucket3840 unchanged. No tile or SM sweep. Source:
`source/megamoe_ep_scaling_20260908`; outputs:
`artifacts/megamoe_ep_scaling_20260908/ep{4,8,16}/<job>`.
The existing profiler now collects uninstrumented Nsight in its normal pass,
then six-point IKET separately (1024 slots/warp to cover EP4's greater work).
Compiler flags participate in JIT caching, keeping the two kernels distinct.
Smoke accepts16 ranks and records EP/global tokens/iterations. Local27 focused
contracts and shell syntax pass after updating stale topology assertions.
Fit total Nsight time and compute spans separately; report ratios, residuals,
and pairwise intercepts. C includes non-scaling setup/synchronization costs,
not just NVLink latency. Do not assume exact1/EP compute scaling or a literal
infinite-EP floor. Source frozen while jobs live. Accepted task files synchronized to local-mirror; no commit.
Details, exact paths, sources and live handles:
`CODEX_PROJECT_qwen3_ffn_overheads.md` and
`scratch/HANDOFF_megamoe_latency_20260908.md`.

## Active 1K perfect-routing maximum-batch search (2026-09-04 PDT)

The user requested the real maximum 1K batch for A1:F1/EP4 using the accepted
FMHA-only FFN-overhead configuration plus cost-preserving perfect routing.  The
old corrected baseline's batch 761 was tied to the conservative 0.82 KV-memory
setting and is only a lower bound.  The first high-information probe, Slurm
job `6921834`, completed `0:0` in 19:48: two trays/eight active GB200 GPUs,
attention DP4, FFN EP4,
MB2 `450+450`, uniform ISL 1,024, 3,600 prompts, FP8-activation/FP4-weight
same-rank MegaMoE, and `MINISGL_AFD_PERFECT_ROUTING=1`.  It requests exactly
15,300 KV pages / 979,200 KV tokens per attention GPU at memory ratio 0.97,
which makes batch 900 the raw KV ceiling and is four requests above `7*128`.
The launch deliberately uses the pmap contract because the comprehensive
launcher's scheduler-admission bound is conservative at short ISL; runtime
still must prove a full-resident warmup plus 15 measured target-batch graph
steps.  No fallback backend or routing mode is permitted.

The remote task root is
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/afd_1k_max_batch_perfect_ffn_ep4_20260904`;
the run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b900_n2_20260904_175401_manual_na`.
Dry-run topology/capacity resolution and login-host validation both passed.
The task control directory records full source head
`e00afda8af6e9fc6ad240ec0bd776b960066a53c`, tracked status/diff, and hashes
of the launcher and relevant FMHA/perfect-routing files.  The remote canonical
checkout remains immutable while MiniMax validation job `6920969` is live.
Job `6921834` ran on same-fabric trays `nvl72106-T[05,18]`.  All four attention
workers reported 15,516 pages at
ratio 0.97 (993,024 KV tokens), revealing that current AFD runtime sizing does
not honor the launcher's nominal 15,300-page override even though
`ServerArgs.num_page_override=15300`; it remains memory-ratio-bound.  The
central scheduler admitted exactly 900 requests at the observed capacity.  The
run produced 3,600/3,600 samples with exactly 17 output tokens, one full-resident
warmup plus the required 15 batch-900 graph replays, eight nonempty `.nsys-rep`
files, `capture-complete.json`, and `FASTAFD_AFD_SUCCESS`, with no OOM or
traceback.  Its result distinguishes the 912-request raw page-count ceiling
from the scheduler-admissible ceiling of 900, so a batch-912 ratio-0.97 run
would be invalid rather than a useful probe.

Ratio-0.98 boundary probe `6922185`, run directory
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b910_n2_20260904_181701_manual_na`,
used the same configuration with nominal 15,683 pages / 1,003,712 KV tokens and
batch 910 (`455+455`).  It failed the intentional observed-capacity gate in
3:27, before graph build or inference: all four attention workers measured
15,676 pages / 1,003,264 tokens, whose exact scheduler ceiling is 909; batch
910 needs 1,003,668 tokens.  This seven-page estimate error consumed only model
initialization time and was not retried unchanged.  Corrected exact-boundary
job `6922262` uses the observed tuple, batch 909, and MB2 `454+455`; dry-run and
login validation passed with `FASTAFD_REQUIRE_CAPACITY_MAX=1`, and it was
submitted at 18:23 PDT.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b909_n2_20260904_182355_manual_na`.
It started at 18:24 PDT on same-fabric trays `nvl72129-T[13,16]`; all four
attention workers reproduced 15,676 pages, the observed gate admitted exactly
batch 909, all eight workers reached runtime-ready and captured the padded
910-slot graph, and all four model workers confirmed FP8xFP4 perfect routing.
Job `6922262` completed `0:0` in 21:58 with all 3,636 samples at 17 output
tokens, one warmup plus exactly 15 batch-909 decode replays, eight nonempty
Nsight reports, and durable capture/result artifacts.  Ratio-0.99 exact-boundary
job `6922514` was then submitted after dry-run and login validation: predicted
15,836 pages / 1,013,504 tokens, batch 918, and MB2 `459+459`, with every other
contract unchanged.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b918_n2_20260904_184828_manual_na`.
Job `6922514` reproduced exactly 15,836 pages on all attention workers and
completed `0:0` in 19:50: exact scheduler batch 918, MB2 `459+459`, 3,672
samples with 17 output tokens, one warmup plus 15 target replays, eight nonempty
Nsight reports, and durable result/capture artifacts.  The final legal ratio
probe is job `6922758`: memory ratio 1.0, predicted 15,996 pages / 1,023,744
tokens, exact nominal batch 928, and MB2 `464+464`.  Dry-run and validation
passed before submission at 19:09 PDT; the run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b928_n2_20260904_190950_manual_na`.
Job `6922758` reproduced exactly 15,996 pages and the batch-928 admission
ceiling, then failed in 4:11 on all attention ranks while allocating the FMHA
fabric arena: ratio 1.0 consumes the entire measured free-memory budget with KV,
but the arena also contains one extra KV page plus Q transport storage.  It did
not reach attention runtime-ready or graph build.  Midpoint job `6922844` was
submitted after dry-run and validation at ratio 0.995: predicted 15,916 pages /
1,018,624 tokens, exact scheduler batch 923, and MB2 `461+462`.  Its run
directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b923_n2_20260904_191724_manual_na`.
Use graph success/failure to narrow the 0.99--1.0 interval; only the final
boundary point needs a complete target-window run.
Job `6922844` ran on the same trays as the ratio-1.0 failure, reproduced 15,916
pages and exact batch-923 admission, reached runtime-ready on all eight workers,
and captured the padded 924-slot graph on all eight with perfect routing.  It
was intentionally canceled at 7:53 before prefill to conserve GPU time.
Upper-midpoint job `6922965` was submitted after dry-run and validation at ratio
0.9975: 15,956 predicted pages / 1,021,184 tokens, exact batch 925, and MB2
`462+463` with padded graph 926.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_192846_manual_na`.
It reproduced all 15,956 pages and passed exact admission, but failed in 3:24
on every attention rank after runtime-ready and before graph capture: each had
only 17.44 MiB free when a 20.00 MiB allocation was requested.  Because batch
925 needs only 15,941 pages, this establishes that the probe overallocated 15
KV pages rather than that batch 925 is impossible.  Batch 926 needs at least
15,959 pages and has the same padded-926 graph, so its memory demand is strictly
higher than the failed probe; batch 926 and above are impossible under this
execution contract.  Final boundary job `6923008` was submitted after dry-run
and validate-only checks at ratio 0.9965625, exactly 15,941 pages / 1,020,224
tokens and exact scheduler batch 925.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_193524_manual_na`.
It failed before container creation in 0:22 because its submitted image path
mistyped `oci-hsg_onboarding` as `oci_hsg_onboarding`; no model or GPU workload
ran.  Identical corrected replacement job `6923092` uses run directory
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_193848_manual_na`.
Replacement job `6923092` completed `0:0` in 19:58 on same-fabric trays
`nvl72093-T[08,14]`.  All four attention ranks reproduced 15,941 pages, the
coordinator admitted exactly 925 requests per lane with only 11 scheduler
tokens unused, and all eight workers captured the padded-926 graph.  All four
model ranks confirmed FP8xFP4 MegaMoE and `perfect_routing=1`.  The real
3,700-prompt prefill completed, followed by full-resident warmup step 1837 and
exactly 15 batch-925 decode steps 1838--1852.  All 3,700 samples contain 17
output tokens, all eight Nsight reports are nonempty, result/capture artifacts
are durable, and `FASTAFD_AFD_SUCCESS` is present.  Therefore batch 925 is the
proven maximum under this A1:F1/EP4 contract: batch 926 needs at least 15,959
pages, already more than the failed 15,956-page same-graph footprint.  Per the
user's decision, no lower boundary points were rerun after this full success.

## Pareto filename correction (2026-09-04 PDT)

The former seven-series `fastafd-128k-five-pareto.html` was preserved verbatim
as `fastafd-128k-seven-pareto.html`, including its editable fragment. The
five-series page was regenerated from the same data after removing only
`FastAFD_FFN_Overheads_W4` and `Baseline_TRTLLM_MegaMoE_W4`; it now contains
five sweeps, 374 measured points, and 86 sweep-local Pareto points, versus
447/109 in the seven-series page. The five-series data is an exact ordered
subset of the preserved seven-series data. Fragment JavaScript parses, all
points fit the fixed chart domains, standalone titles/root IDs are correct, and
the removed series names are absent from both five-series artifacts.

## TRT-LLM perfect-router 128K comparison refresh (2026-09-01 PDT)

The TensorRT-LLM perfect-router 31-case sweep is complete and accepted for
GPU-side step-latency comparison. Jobs EP4 `6810988`, EP8 `6810989`, EP16
`6810990`, EP32 `6810991`, and EP64 `6810992` all completed `0:0` under
`qos=short`, one job per EP, with a two-hour limit. The exact matrix is EP4
batches 1--5, EP8/EP16 batches 1--6, and EP32/EP64 batches 1--7. Every point
has 15 accepted CUDA-event `gpuForwardTimeMS` samples after one full-batch
warmup, `ENABLE_PERFECT_ROUTER=1`, validated global-expert/receiver balance,
and the unchanged `MEGAMOE_DEEPGEMM`/`FUSED_COMM` backend. Attention-DP fans
rank 0's CUDA-event duration into every rank row; this existing timing was
explicitly accepted. Host `generate()` duration is not used.

The strict source and comparison artifacts are under the TensorRT-LLM
workspace at
`scratch/trtllm_qwen3_128k_pareto_resweep/results/gpu-event-perfect-sweep-31-20260901/`.
Relative to the prior 31-point Nsight CUDA-graph-span sweep, mean per-point
latency changes are EP4 +9.206%, EP8 +5.671%, EP16 +2.727%, EP32 -0.013%, and
EP64 -0.767%; 9/31 points improve and the unweighted overall mean is +2.934%.
The best new throughput is EP32/batch 7 at 33.353011 ms and 209.876103
TPS/GPU, versus 206.481802 TPS/GPU previously. Both metrics are GPU-side, but
their mechanisms differ, so these deltas are directional rather than a clean
router-only attribution.

The seven-series Pareto page
`scratch/afd_128k_optimized_attention_near_pareto_20260827/fastafd-128k-seven-pareto.html`
preserves the prior TensorRT-LLM series and adds the new run as
`TRTLLM_MegaMoE_Perfect_Balance_W4`. It contains seven sweeps, 447
points, and 109 sweep-local Pareto points; the new TRT-LLM series has 31 points
and seven Pareto points. The separately retained
`fastafd-128k-five-pareto.html` omits `FastAFD_FFN_Overheads_W4` and
`TRTLLM_MegaMoE_W4`, leaving five sweeps, 374 points, and 86
sweep-local Pareto points. The EP comparison page
`trtllm-qwen3-128k-user-tps-vs-tps-gpu-by-ep.html` now contains only the new
31-point perfect-router run, grouped by EP4/8/16/32/64. Editable fragments are
under `visualization-source/`; `build_128k_five_pareto.py` now accepts the
strict TensorRT-LLM summary, and `build_trtllm_ep_comparison.py` reproducibly
replaces the EP chart from that summary. Both extracted JavaScript programs
pass `node --check`; structural audits prove exact rounded equality to the
accepted summary, all points fit their chart domains, and both rendered
standalone documents contain their expected data/provenance globals.

The focused perfect-balance comparison is
`afd-trtllm-perfect-balance-128k-pareto.html`, with its editable fragment under
`visualization-source/`. It retains only the accepted 31-point TRT-LLM perfect
router series and the accepted 42-point AFD perfect-routing series, displayed
as `TRTLLM` and `AFD`. The two sweep-local
frontiers contain 7 and 13 points respectively (73 measured / 20 Pareto total).
The reusable `--series KEY[=DISPLAY_NAME]` builder option selects and renames
series without forking the chart implementation. Exact-data audits pass for
all 31 TRT-LLM summary rows and all 42 AFD metric JSONs, and the extracted
JavaScript passes `node --check`. Two verified axis variants are retained for
convenience: `afd-trtllm-perfect-balance-128k-pareto-zero-start.html` uses
TPS/User `[0, 80]` and TPS/GPU `[0, 245]`, while
`afd-trtllm-perfect-balance-128k-pareto-cropped.html` uses the original
`[25, 80]` and `[25, 245]` domains. The unsuffixed page aliases the zero-start
version. Both display the short labels `TRTLLM` and `AFD` and contain identical
73-point data/frontiers. Both variants use more prominent 24px/weight-500 axis
tick labels in the foreground color (black in the light theme), explicit 9px
tick marks, and 1.75px foreground axis/tick strokes.

## Completed perfect-routing 128K sweep (2026-09-01 PDT)

The completed 42-case AFD 128K matrix was rerun with perfectly balanced
expert IDs and the original gate GEMM/softmax/top-k cost retained.  Clean
remote source `e00afda8af6e9fc6ad240ec0bd776b960066a53c` (based on repaired `bd1e3ca`) uses a compile-time final
ID substitution with no added launch, graph node, synchronization, or copy.
Fresh remote task root is `afd_128k_perfect_routing_e00afda_20260901`; its pool
started at 42 pending cases.  Focused contracts pass 55/55.  Two prelaunch
attempts ran no case: `6796806` used a stale home-directory image path and
`6796939` carried a stale full SHA despite the correct short commit ID.  Both
fail-fast checks preserved the untouched pool.  Corrected 2-tray job `6797102`
and 3/4/8/12/16/18-tray jobs `6797854`--`6797859` use the original sweep's
Lustre image and verified full source SHA, one short-queue/two-hour job per
tray group. All seven final jobs completed `0:0`; the pool is 42 complete,
zero claims/pending/failed. The strict audit passes all case/result hashes,
topology and measurement contracts, 1,512 Nsight reports, and 42 metric JSONs;
all 288 model logs contain exactly one FP8xFP4 perfect-routing marker, and all
42 prompt transforms are cache hits on the original pinned source.

The paired result contradicts the expected neutral-to-faster outcome: latency
geometric mean is **+5.867545%** and TPS/GPU is **-5.542345%**, with 14/42
cases faster and 28 slower. Topology latency geomeans are +18.570508% (2t),
+14.104891% (3t), +4.543148% (4t), +2.481296% (8t), +1.224520% (12t),
+0.926520% (16t), and +0.653234% (18t). The worst row is A1/EP4 batch 2 at
+35.846598%. Global count balance is therefore not performance-neutral; the
deterministic per-token expert ordering is the leading follow-up. Detailed
evidence is in `CODEX_PROJECT_qwen3_ffn_overheads.md` and the local verified
copy `scratch/afd_128k_perfect_routing_e00afda_20260901/`.

## Standalone 128K Pareto figures (2026-09-02 PDT)

A full one-chart comparison is under
`scratch/afd_128k_optimized_attention_near_pareto_20260827/fastafd-128k-seven-pareto.html`,
with editable fragment in `visualization-source/fastafd-128k-seven-pareto.html`.
It contains seven verified sweeps, 447 measured points, and 109 sweep-local
Pareto points, including both perfect-balance sweeps. The companion
`fastafd-128k-five-pareto.html` and its editable fragment retain the same chart
but omit `FastAFD_FFN_Overheads_W4` and `TRTLLM_MegaMoE_W4`; the
result is exactly five sweeps, 374 measured points, and 86 sweep-local Pareto
points. All vLLM and TRT-LLM display names in both variants omit the
`Baseline_` prefix. Both variants are reproducible with
`build_128k_five_pareto.py` and
its `--series` selection. The separate focused comparison
`afd-trtllm-perfect-balance-128k-pareto.html` uses exactly the same formatting
but only the renamed TRT-LLM and AFD perfect-balance series (73/20). Both use a
1600x960 SVG viewBox, exact hover/selection details, sweep-local frontier lines,
EP marker shapes, translucent measured non-Pareto points, and one combined axis
pair. Their extracted JavaScript passes `node --check`, and exact source-data
audits prove all perfect-balance rows and Pareto memberships.

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

## MegaMoE FP8xFP4 kernel-only decomposition (2026-08-31 PDT)

An opt-in diagnostic specialization now records kernel, dispatch, compute, and
combine start/end points for the fused FP8xFP4 MegaMoE kernel.  It uses private
per-warp `clock64()` rows, CTA-local normalization, per-rank CUDA-event
calibration, and a queued GPU-side launch barrier; production specialization
compiles all timing code out.  The standalone runner uses perfect deterministic
routing and excludes route preparation from the event.  Final A8:F1, EP8,
batch-6 job `6740199` passed on one NVL72 block: control/instrumented medians
are 119.296/126.304 us (5.874% diagnostic overhead), dispatch/compute/combine
durations are 22.122/70.516/102.569 us, and compute hides 70.200 us of the
124.808-us communication envelope (56.320%), leaving 53.910 us exposed.
Final remote descendant `7d59050` is clean; the prebuild passed and focused
contracts are 6/6.  Code, logs, report, interpretation, and reproduction notes
are local under `scratch/megamoe_fp8_fp4_decomposition_20260831/`.  Detailed
history is in `CODEX_PROJECT_qwen3_ffn_overheads.md`.

## Local Nsight traces copied for the latest sweep selections (2026-08-31/09-01 PDT)

The complete `.nsys-rep` sets for the latest 42-case 128K near-Pareto sweep's
highest-TPS/GPU and highest-TPS/user cases are local under
`scratch/afd_128k_optimized_attention_near_pareto_cleanup_9a84a5a_20260829/nsys_winners/`.
The TPS/GPU winner is `i131072-fep8-r8-atp1-b7-fp4` at 232.300866904
TPS/active GPU (37.334067895 TPS/user), with 72 reports totaling 66,549,491
bytes.  The TPS/user winner is `i131072-fep8-r1-atp1-b2-fp4` at
61.133830322 TPS/user, with 16 reports totaling 18,516,911 bytes.  TPS/user is
computed as `1000 / cuda_wall_time_mean_ms`, so the latter is the global
minimum-latency case at 16.357555133 ms.  Both transfers passed `rsync -acni`
content verification with exit zero and no itemized differences, and exact
local/remote counts and byte totals match.  The local macOS rsync rejects
`--info=stats2`; use its supported `--stats` flag for future copies.  Selection
and verification details are in `nsys_winners/README.md`.  The additionally
requested 128K EP8 batch-6 ratio series is also local for every ratio present
in the sweep: A:F 1:1, 3:1, 5:1, 7:1, and 8:1.  The manifest has no EP8 2:1,
4:1, or 6:1 rows.  These five sets contain respectively 16/32/48/64/72
reports, 232 total, and 225,820,541 bytes.  All five `rsync -rcni` checksum
dry runs exited zero with no report-file differences, and local/remote counts
and bytes match.  The A8 set is the successful retry, so the earlier failed
control-port attempt is not mixed into the local traces.

## Remote sweep archive cleanup completed (2026-08-31--09-04 PDT)

The remote Lustre cleanup produced a complete filtered archive set and promoted
it to the canonical root
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD/artifacts/archives/sweeps_20260830`.
CPU-data-mover job `6729950` completed `0:0` in `04:03:34` and produced exactly
14 numbered `.tar.zst` sweep files, no `.partial`, a 14-row
`SWEEP_INDEX.tsv`, and a 5,218-row `CASE_INDEX.tsv`.  Every payload archive
excludes any directory subtree named exactly `tmp`; all other case content,
including nsys reports/SQLite exports, configs, result JSON, TPS/GPU metrics,
CSV/text stats, and logs, is retained.  Sweep 07 is intentionally a
metadata-only 1,935-case tombstone.  Per explicit user direction, expensive
full `zstd -t` rereads were skipped; every tar/zstd pipeline exited zero,
zstd frame checksums were enabled, and final names were created atomically
from `.partial` files.  The lightweight final audit found 14 archives, no
partial, 15 sweep-index lines including header, 5,219 case-index lines
including header, and all 14 summaries record the `tmp` exclusion.

### Extract one archived case

Use `CASE_INDEX.tsv` to find the archive filename (column 2) and internal case
path (column 4), then extract only that case:

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD/artifacts/archives/sweeps_20260830
rg -F 'CASE_NAME_OR_FRAGMENT' CASE_INDEX.tsv
mkdir -p /path/to/extract
tar --use-compress-program=unzstd -xf ARCHIVE_FILE.tar.zst -C /path/to/extract INTERNAL_CASE_PATH
```

Only the selected case is written to the destination; it contains the retained
nsys traces/SQLite exports, configs, result JSON, TPS/GPU and other perf data.
`tmp` subtrees were deliberately excluded, and sweep 07 has no payload.  Since
`.tar.zst` is a sequential compressed stream, selective extraction still scans
and decompresses the stream internally; it does not unpack the other cases to
disk.

The user explicitly confirmed source deletion while retaining the latest 42
sweep-14 case directories uncompressed.  Promotion moved the superseded
unfiltered archives 01--05 and canceled archive-06 partial to rollback root
`.../archives/sweeps_20260830_superseded_unfiltered_20260831`.  The fixed
deletion plan contains 3,623 exact targets: archived payload cases from sweeps
01--06 and 08--13, the 94/82/163 wrong-timing AFD directories from sweeps
02/03/04, the whole sweep-07 root, and 42 named `tmp` subtrees under the kept
sweep-14 cases.  CPU data-mover job `6737940` logged 3,600 deletions before its
progress stopped advancing for one minute; per the user's instruction to stop
if the tail would take long, it was canceled at `00:17:57`. CPU data-mover job
`6919189` resumed the exact plan on 2026-09-04, deleted the one remaining target,
and completed `0:0`; all 42 protected cases remained present and the canonical
14 archives/indexes were unchanged. Job `6919354` then deleted the verified
superseded unfiltered archives and abandoned partial. The filtered-rebuild and
archive staging roots are gone; canonical metadata/logs now use visible
`metadata/` and `logs/` directories, with no custom hidden directory.

Earlier job `6729317` was canceled when the user narrowed payloads to useful
result artifacts; job `6729316` never ran because its original unlimited time
limit violated the data-mover QoS.

The reusable archiver is now only in main source as
`codex_scripts/archive_sweep_cases.py`; the old scratch and remote `.tools`
copies were removed. Filtered result mode writes payload-only tar members and
keeps manifest/SHA/summary metadata as external sidecars.

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
