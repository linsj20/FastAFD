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

# Handoff: final MegaMoE communication version

## Final outcome — STOPPED, September 14

User resumed correctness/perf, then final knob cleanup and kernel sweep; explicitly
retained STOP/REPORT if another path wins. Stop condition met on two complete Q3
cases. **Do not resume any comparison/recovery/cleanup/sweep without a new request.**
Final report: `scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/REPORT.md`.
Current production retained; no alternate promotion, commit or staging.

- Fixed real integration bug: inband_combine=True was on checkpoint loader, not
  buffer allocation. Moved keyword before numerical validation, synced candidate.
- Native/host7153123 remainsPASS. Exact7153343 PASS264/5records6:16. CPU adapter
  7153388 PASS3tests1:19: checkpoint signature, explicit mode with env0, padded
  valid-count0 participation, FP8/FP4 forwarding. GPU numerical/timing usesFP4.
- Current AFD scheduler/materializer uses nonzero padded rows for idle ranks;
  no silent conventional fallback/skip. Directshape0 not a validated adapter mode.
- Timing7153461Q3 and7153462Q35 CANCELLED deliberately at26:30/26:15 after verified
  stop condition. All42primary records retained/audited488rankexactmarkers;
  30records/3completecases have all5configs×2repeats on one allocation, eachsrun
  exit0 proven by subsequentCASE.12records are partial cases, reported separately.
- Q3EP8rows28 selected50.280us vscompact49.939us (compact−0.677%); rows21
  49.400vs49.208us (−0.390%). Compact faster in BOTH repeats of bothcases, with
  nonoverlapping repeat ranges. Small measured advantage, not universal/statistical
  optimality proof. Selectedrows28 larger repeat spread retained.
- CompleteQ35EP16rows38 selected63.567us vscompact65.806us (selected−3.402%).
  Previousmetadata64.128us; conventional64.154us; precleanupdense63.365us.
  No universal winner or selected-best proof. Original Q35profile slowrepeats kept.
- Incomplete cases: Q3rows4; Q35rows28/22. Six-case matrix NOT complete. Auditor
  explicitly PARTIAL_STOPPED. Final knob cleanup, final sweep/alignment notrun.
- Slurm denied30→45min extension; neither job reachedtimeout. Prepared final-row
  recovery plans SUPERSEDED_UNSUBMITTED_STOP_CONDITION. No recovery submitted.
- Both GPUallocations released. Collector93344 stoppedEXIT143; exactcollector81073
  exited0. No outstanding task jobs/collectors. State Q/STATE.json.

Evidence: Q/integration_review.json, validation/exact_audit.json,
adapter_check/completion.json, config_comparison/{audit.json,REPORT.md,
stop_decision.json}, all original logs. Generic audit command:
`python3 codex_scripts/audit_megamoe_config_comparison.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/config_comparison --stopped`
It checks every complete record; incomplete cases are never used for a winner claim.
Current code and relevant evidence/workflows/handoff are synced to the candidate
source and remote evidence tree. Index and unrelated GDN work preserved.

Everything below describes the original plan and historical checkpoints, not
currently authorized remaining execution.

## Objective and stop condition

User's requested final version is **early metadata push + pull-based activation
payload transfer + push-based combine**. Remove rejected active optimizations,
clean the implementation, and make it work. The user explicitly requires test
results showing this is the best configuration. A correctness pass or a cleanup
comparison alone does not establish that. If another path is better, **stop and
report it**; do not silently select an alternative or start another optimization
search. If results show a tradeoff or tie, report it accurately.

The original handoff was resumed; see the final STOPPED outcome above.
The uncompleted portions below require a new request after the reported tradeoff.
No commit or new staging was requested.

## Workspace, constraints, and authoritative paths

- Local root: `/Users/shengjiel/Workspace/github/FastAFD`.
- Remote alias: `oci-hsg`; use plain `ssh oci-hsg`.
- Remote root R:
  `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD`.
- Local evidence P: `scratch/megamoe/pull_metadata_restore_20260914`.
- Current follow-up Q: `P/selected_explicit`.
- Remote evidence follows `R/artifacts/pull_metadata_restore_20260914`.
- Current candidate source: `R/source/ffn_selected_explicit_20260914`.
- Initial cleaned source: `R/source/ffn_pull_metadata_clean_20260914` (frozen).
- Selected pre-cleanup control: `R/source/ffn_dispatch_dense_descriptors_20260913`.

Read the supplied global instructions, `CODEX_PROJECT.md`, this document, and
relevant reusable `codex_scripts` before acting. Root AGENTS.md is missing; the
user was already asked about its location/creation earlier, without an answer.
Do not repeat that question or fabricate project rules. No subagents, `/tmp`,
`py_check`, or hash/version checks. At most two outstanding GPU jobs; CPU work
belongs on CPU allocations. No manual waiting while holding idle GPU resources.
Preserve unrelated attention/GDN work and the pre-existing index. Many files are
already staged from earlier requests; the current cleanup is in the working tree.
Do not reset, discard, or commit unrelated changes. Preserve historical evidence.

## Current code

Eight active files have changes for this task; exact file list is
`Q/production_files.json`, with current copied files under `Q/overlay/`:

1. `python/minisgl/kernel/csrc/jit/gate_topk_fused.cu`
2. `python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
3. `python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/layout/mega_moe.cuh`
4. `python/minisgl/kernel/csrc/deepgemm/csrc/apis/mega.hpp`
5. `python/minisgl/kernel/csrc/deepgemm/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`
6. `python/minisgl/kernel/megamoe_mega.py`
7. `python/minisgl/afd_support.py`
8. `python/minisgl/moe/megamoe_afd.py`

Removed from active code: Lamport payload inbox polling/clearing, producer
vector/TMA payload push, in-MegaMoE payload sender, dispatch arena helpers,
unused parameters/template options, and four rejected dispatch environment
switches. Reserved metadata padding preserves validated offsets. Experiment
snapshots in scratch are retained, not active runtime alternatives.

Retained: dense source descriptors allocated/published early by the producer,
destination TMA pull through dispatch shared memory into contiguous expert rows,
unchanged GEMM/quantization, accepted in-band push combine, and supported
conventional/legacy correctness paths. The destination decodes the descriptor
(rank/token/topk) and publishes local payload completion. This preserves consumer
contiguity and removes rejected payload-preparation/gather branches.

The latest follow-up makes combine an **explicit per-buffer protocol**:

- `MegaMoESymmBuffer(..., inband_combine: bool | None = None)` records the mode.
  Standalone/reference callers using None retain the previous environment/default0
  selection; explicit bool takes precedence.
- The same bool is passed to C++ buffer sizing, host validation, and FP8/FP4 launch
  argument generation. C++ no longer re-reads the combine environment variable.
- `MegaMoEAfdAdapter.allocate_buffers()` explicitly passes `inband_combine=True`.
  Its existing nonempty path already enables early rank-ready metadata publication
  and TMA payload pull. No environment setting is needed to select AFD combine.
- Underlying CUDA arithmetic and scheduling are unchanged by this API cleanup.

The current eight-file overlay was synced to the new remote source after its copy
finished. Both earlier sources are frozen and must not be overwritten. Initial
backups are `P/before_cleanup/`; the four files before the explicit-protocol change
are in `Q/before/`. No index mutations or commits were made.

## Completed validation and measurements

| Job | Scope | Result |
|---|---|---|
|7152139|Initial cleaned EP2/EP8 CUDA cubins, actual native router/host extension, SASS|PASS all5steps,2:56|
|7152230|Final native cache refresh after comment cleanup|PASS,1:42|
|7152282|Initial cleaned exact-output suite|PASS264 comparisons/5records,8:13|
|7152479|Initial cleaned Qwen3EP8 before/after ABBA|COMPLETED0:0,12:03;12captures/60phase/360ordinary audited|
|7152485|Initial cleaned Qwen3.5EP16 before/after ABBA|COMPLETED0:0,13:12;same complete coverage|
|7153123|Current explicit-protocol actual native/host build|COMPLETED0:0,3:28; see Q/cpu_build logs and completion.json|

The exact suite covers prepared small/Q3/Q35 and unprepared small recording/
comparison, including normal/perfect routing, valid-count empty/padded inputs and
graph reuse. **It precedes the explicit-protocol API change**; rerun the prepared
exact plan for the current source before timing. CPU build success alone does not
validate the new host argument wiring numerically.

`git diff --check -- python/minisgl` passed. Removed dispatch symbols are absent
from active files. Captured initial ordinary kernels confirm intended flags and
identical repeat configurations (`P/kernel_configuration_audit.json`).

Fixed draw seed3689699192022641276, unchanged:
Q3EP8 rows28/21/4; Q35EP16 rows38/28/22. Initial selected pre-cleanup vs cleaned
ordinary rank0 producer-to-MegaMoE-end pooled means:

| Model | Rows/rank | Control us | Cleaned us | Change |
|---|---:|---:|---:|---:|
|Q3EP8|4|47.126|47.197|+0.152%|
|Q3EP8|21|49.590|49.572|−0.037%|
|Q3EP8|28|49.982|49.988|+0.013%|
|Q35EP16|22|60.114|63.196|+5.127%|
|Q35EP16|28|60.745|65.693|+8.147%|
|Q35EP16|38|62.710|63.083|+0.595%|

Q35 second cleaned repeats at rows22/28 slowed across **many calls**, not just
isolated outliers. First/second repeat means60.243/66.148us and61.036/70.351us;
medians60.240/65.152us and60.768/71.632us. Sources/configs match. The cause is
unresolved: do not dismiss as noise, delete/filter results, or claim nonregression.
The follow-up uses unprofiled slowest-rank timing to investigate this.

Initial selected instrumented phase medians across10samples/case:

| Model | Exposed dispatch us | Compute span us | Exposed combine us | Tail us |
|---|---:|---:|---:|---:|
|Q3EP8|6.26–6.82|24.27–26.13|4.46–4.91|0.19|
|Q35EP16|6.93–7.04|34.30–35.14|5.41–5.76|0.19–0.21|

Compute is the largest measured span. These instrumented spans include waits;
they are separate from ordinary timings and are not hardware-stall attribution.
Full dispatch/combine overlap compute; only exposed intervals belong in the
partition. Component medians need not sum exactly. No PIC-C/NCU profile was run.
Details: `P/REPORT.md`, `results.json`, `time_decomposition.csv`, per-model
`completion.json`, `fullchain_comparison.json`, and every original capture.

## Remaining work, in order

### 1. Finish reviewing explicit selection and integration edges

Inspect the final Python/C++ argument wiring (both weight types) and AFD's explicit
selection. The actual native/host CPU build7153123 completed. Collect/inspect its
saved step markers if completion.json is not yet present; do not resubmit it.

One integration edge deserves targeted inspection before declaring the version
finished: AFD currently sets `routes_prepared = num_tokens > 0`, whereas in-band
combine requires rank-ready metadata. Determine whether actual zero-sized tensors
can reach that call or are always represented by padded nonzero rows/valid_count0.
The existing exact empty tests exercise valid-count/padding; do not assume they
cover a true shape0 invocation. If reachable, handle it with correct collective
participation; do not introduce a silent conventional-combine fallback or skip a
rank that peers need. This is a review concern, not an observed test failure.

### 2. Run current-source exact validation

Prepared, **UNSUBMITTED**, at `Q/validation`; one node4GPUs,15minute reservation.
It uses current source `ffn_selected_explicit_20260914`, current CPU caches, and
same saved reference coverage264/5records as initial7152282. Historical control
is `ffn_dispatch_dense_descriptors_20260913`. Its candidate.env selects combine1
for prepared groups and combine0 for legacy groups.

From local workspace root:

```bash
python3 codex_scripts/submit_megamoe_plan.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/validation
python3 codex_scripts/collect_megamoe_abba.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/validation --watch-seconds 180
python3 codex_scripts/audit_megamoe_exact_outputs.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/validation --expected scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/validation/expected.json --output scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/validation/exact_audit.json
```

Do not submit a directory that already has job.json. Monitor/collect that job.
Only proceed to timing when current-source numerical validation passes. Changes
after validation require an isolated source and only the checks they invalidate.

### 3. Run the prepared best-config comparisons

Prepared, **UNSUBMITTED**, at `Q/config_comparison/{qwen3,qwen35}`. Plans are
fully recorded in `plan.json`, model env.json/submit.json, and case.env. There are
five configurations using four frozen source variants:

| Label | Source/configuration |
|---|---|
|candidate_selected|Current cleaned explicit protocol: early dense metadata + payload pull + in-band push combine|
|control_dense|Selected pre-cleanup dense metadata/pull/in-band source; cleanup equivalence control|
|reference_previous_metadata|ffn_comm_final_pull_integrated_20260912: previous metadata implementation + payload pull + in-band combine|
|alternate_compact|ffn_dispatch_dense_compact_frame_linear_index_20260914: strongest retained compact payload-push variant + in-band combine|
|control_conventional|Same pre-cleanup dense/pull source as control_dense, but conventional combine0|

Do not call the previous metadata source literally all-pull: it already pushes
some metadata. Compact source implements its experiment in the frozen code; its
legacy push environment flags being0 does not mean its payload is pull.

Per model: same original three row counts, five configs, two symmetric repeats;
30primary records. Each record times2000 invocations,200warmups,graphbatch100,
NSYS0 and profiler-range0. CUDA events are reduced with MAX across ranks. Each
trial checks combine exactness against an independent conventional arena outside
timing. Compare complete producer+MegaMoE latency, not dispatch alone.

Q3 uses2nodes8GPUs; Q35 uses4nodes16GPUs. Both30minute reservations in one NVL72
block per model. Submit at most these two jobs concurrently after exact PASS:

```bash
python3 codex_scripts/submit_megamoe_plan.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/config_comparison/qwen3
python3 codex_scripts/submit_megamoe_plan.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/config_comparison/qwen35
python3 codex_scripts/collect_megamoe_abba.py scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/config_comparison/qwen3 scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/config_comparison/qwen35 --watch-seconds 300
```

Common current smoke test is already synced to
`R/artifacts/pull_metadata_restore_20260914/selected_explicit/workflow/afd_megamoe_same_rank_smoke.py`.
`MEGAMOE_SMOKE_TEST_SCRIPT` selects it for every source. This is necessary because
the frozen previous-metadata test has the old incorrect reference-arena handling.
Do not drop this override. No arithmetic/test inputs were changed.

24 cache aliases are prepared under remote `selected_explicit/cache/matrix`;
`cache_links.json` records targets. Existing actual host/native caches were checked
and reused. No GPU-side host rebuild is intended. The wrapper and other maintained
workflow files remain under remote `P/workflow`. Both case.env files passed bash-n.

**The comparison jobs have not been exercised.** Before submission, review the
common comparator against each frozen source's supported protocol. Fail clearly
on an incompatible reference setup; do not weaken exactness or timing semantics.

### 4. Audit and decide from results

The generic collector preserves30ordered primary records/model. Because records
have different inputs, its pooled primary_us is intentionally empty. Analyze each
record's `exact_qwen` entry separately using label, model dimensions, EP, rows,
global tokens, routing, graph settings and source. Verify all30record labels in
submitted order, scheduler COMPLETED0:0, all status=ok, exact-combine checks on all
ranks, and expected2000iterations/200warmups/graphbatch100 for every record.

For each of six cases, report both repeats and means for all five configurations;
show selected versus alternatives and versus pre-cleanup control. Retain all
results from initial profiling and this unprofiled comparison. Do not average
unlike shapes, suppress slow repeats, reroll cases, or treat a tiny tie as proof of
superiority. Equal selected/pre-cleanup performance is expected; it is the same
algorithm. Old additional5percent optimization gates are irrelevant to this
cleanup/best-existing-config request.

If another path is better, stop and report the winning config, cases, advantage,
and any component tradeoff. Do not silently reintroduce rejected production code.
If no universal winner exists, say so. Only call the selected configuration best
within the tested scope when the evidence supports it. Universal optimality is
not established by six sampled cases. Earlier phase captures explain components;
new event timings are not directly interchangeable with rank0 NSys envelopes.

### 5. Finalize only after the evidence supports the conclusion

Resolve actual failures/regressions within the authorized scope, or report the
better path/remaining uncertainty as requested. Complete targeted AFD integration
checks needed by the explicit mode change (including empty-row reachability).
Use existing normal-routing/model alignment workflows if final integration changes
invalidate earlier serving alignment; do not automatically rerun464cases or full
models for unchanged CUDA math. Keep validation proportional and meaningful.

Clean code, sync final code/docs to the intended remote workspace, update root
memory and this handoff with outcomes. Report the chosen config, correctness,
per-case comparative timing, decomposition, scope and limitations. No commit,
staging, deployment, or messages to others without a user request.

## Historical context that must not be lost

- Dense descriptors improved full-sweep dispatch averages14.108%Q3/23.108%Q35,
  but average per-case fullpath gains were only0.478%/0.583%, with regressions.
  Faster dispatch alone is insufficient evidence of the best complete path.
- Direct dense push, source-owned buffers, sentinel readiness, and contiguous
  expert-pool push have been tested. Several passed numerical validation and
  improved individual phases while worsening total latency/producer work.
- Compact push has producer preparation and consumer gather costs. Metadata/
  payload disaggregation predates the latest compact-frame change; do not claim
  it was the recent causal enabler. Capacity compactness alone was not isolated.
- Earlier optimization search is stopped. New work is bounded final selection,
  cleanup, validation and comparison, with the explicit stop-if-better condition.
