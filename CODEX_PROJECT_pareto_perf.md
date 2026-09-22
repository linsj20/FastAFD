# MegaMoE Pareto perf rerun — completed 2026-09-10 PDT

## Optimized-only chart — September 17

User requested a new HTML retaining only the two optimized curves from
`scratch/visualizations/afd_128k_optimized_attention/afd-trtllm-perfect-balance-128k-pareto.html`.
Created sibling `afd-trtllm-optimized-only-128k-pareto.html` and matching editable
fragment under `visualization-source/`. Keeps AFD_Optimized66points/17Pareto and
TRTLLM_Optimized24points/7Pareto, original colors, axes and point interactions.
Removed baseline series/legend entries and unused EP2/EP64 keys. Original untouched.
Exact embedded-data comparison PASS90points/24Pareto, axis bounds PASS, JS syntax
PASS; rendered with visualize skill's standalone renderer. No benchmark rerun.

Curve/kernel provenance lookup September17: AFD_Optimized is the final66-case
attention-graph-edge sweep7061091–97, same early optimized MegaMoE lineage as
the September9 Qwen3 decomposition, with the subsequent guarded64bit combine
address correction and attention scheduling changes; not an assertion of identical
frozen source. TRTLLM_Optimized24points exactly match sibling TensorRT-LLM
`scratch/trtllm/reports/qwen3/megamoe-optimized-20260910/ep{4,8,16,32}-summary.json`,
GPU7069905/07/09/11 (latencies and exclusion counts verified). TRT ports AFD's
prepared routing/readiness, early weight reads, producer-rank gating, ordered
CTA combine and static waves into newer DeepGEMM8b1392b9's ring implementation;
it is not the same kernel binary/source. See sibling
`CODEX_PROJECT_MEGAMOE_ALIGNMENT.md`. Thus the Qwen3 decomposition is the older
AFD kernel baseline, not a direct phase breakdown of both full-model curves.
Neither plotted curve represents September12–17 communication experiments.

Current checkpoint (September 10): all requested work is complete. Final
scheduling and Qwen3.5 barrier fix are promoted into main; superseded snapshot
repos removed, source patches/results retained. See CODEX_PROJECT_attn_graph_edge.md.
The execution instructions and pending states below are historical.

## Scope and current state

User requires original42 FastAFD_Perfect_Balance_W4 cases plus EP16 sweep,
including56 attention GPUs:16 FFN GPUs; all share best perf configuration.
66 cases at128K, batches2–7: EP4 A1/A2; EP8 A1/A3/A5/A7/A8; EP16
A1/A2/A3 and explicitA56:F16. One job per tray count, monitoring5 or10min.
Original42 measured/13 Pareto points are in
scratch/visualizations/afd_128k_optimized_attention/fastafd-128k-five-pareto.html.
All42 archived full-precision metrics match its exact series to displayed precision.

Final01:46PDTSept10: ALL66/66strict,0claims,0failed. 7048005completed
0:0in11:07at01:42:23,all4remainingA56:F16accepted. Finalfullruntime/metric/
fanin/executionprovenanceauditpassesall66, preserving8valid7040264captures.
Archived42pairs TPSgeomean+3.113255%,33faster, latency-3.019258%.
Bybatch2..7: +6.991908%,+4.222154%,+4.120116%,+2.470421%,+0.721047%,
+0.305008%. Archiveprefillsetupdiffers; notisolatedkernelcausalAB.
LocalreportCSV/metrics/final66-casePNG/SVG/PDF+READMErefreshed01:47.
Candidatealso66complete: finalpairedTPS+0.05844%;seeattentionrelease task.
WholegoalstillactiveuntilQwen35decomposition299complete; currently28/299.
Original18job7040264 FAILED1:0after8captures;A56:F16b7/b6recoveredonCPU
followingmetadata-onlyactive_gpusmigration, noGPUreruns/runtimechanges.
- 2t7040270:6/6, COMPLETED0:0 in21:38
- 3t7040269:6/6, COMPLETED0:0 in20:36
- 4t7040470:6/6, COMPLETED0:0 in20:53
- 8t7040267:12/12, COMPLETED0:0 in38:00
- 12t7040469:12/12, COMPLETED0:0 in40:23
- 16t7043302:12/12 complete, COMPLETED0:0 in40:37
- 18t7040264:8acceptedcases, FAILED1:0in29:02, nvl72022-T01–18.
- 18t7048005:4/4accepted,COMPLETED0:0in11:07,72022-T01–18;samefrozenruntime/config.
- Machine check7042690 FAILED134:0 in2:11 on known-success nvl72077-T[15-16]
- Address fix validation7042863 COMPLETED0:0 in2:28; exact combine output passes
- Same-GPU perf comparison7042982 COMPLETED0:0 in7:55, byte-exact all phases
- Diagnostics7042033 failed;7042379 canceled after recording errors;7042622 canceled while pending
Short QoS allows two running jobs; pending jobs advance automatically.
All jobs2hr, --segment=trays, exclusive4 GPUs/tray, no cross-NVL72 placement.
Paired42 vs archived42:TPS/GPUgeomean+3.113255%,33/42faster.
EP16 A1 all6 pass.12t A5/b7/b6/b5 also pass after startup-only checkpoints;
model rank0 max_rows2595/fanin_capacity5,2 decode streams, expected graph40rows
atb7. At20:27 its firstcase was durable capture but awaitingmetric; no hang.
Current local report/sweep.csv and summary refreshed to54 cases (22:32). Do not
summarize full performance until all66 pass.

### Historical16-tray initialization failure (recovered7043302)

7040265 FAILED1:0 in10:13, nvl72053-T[01-08,10-17]. Its first EP8 A7/b7
and b6 failed exit3 before fmha_only_ready/capture. All8 model workers reported
CUDA errors at _prewarm_model_decode_kernels -> forward_lm_head -> cuBLAS;
b7 has6 CUBLAS_STATUS_EXECUTION_FAILED and2 CUBLAS_STATUS_ALLOC_FAILED,
followed by illegal-address and DeepEP destructor aborts. Both b7/b6 show
same pattern. No valid metric/sample. Do NOT classify as hardware failure or
performance regression; cuBLAS may only report an earlier asynchronous fault.
Standalone previous sweep DID pass exact EP8 rows21 and28 (job7023771), but
it is not the complete actual-model startup path. No redundant standalone run.

Added startup-only finish_stage(stream.synchronize) after prepare_attention,
finish_attention, post_attention_layernorm and MoE, wrapping error with stage.
This localizes producing CUDA stage before lazy cuBLAS; no timed decode/graph
changes. Runtime file synchronized local->source and local-mirror20:16 before
retry. One focused prewarm-order contract passed remotely. Keep diagnostic
until cause established; do not mistake a passing sync retry for root-cause proof.
Released two failed cases with archive reasons; submitted replacement16t7041337
at20:16 with exact common perf config and same caches, all12 cases remain.
First resubmission script exited BEFORE mutation because squeue returned invalid
terminaljobID; second used authoritative sacct FAILED and submitted once.
Current jobs.jsonl last16t record points7041337; retrieve updated local copy.
Next inspect new model error stage or successful startup. Startup may read
_graph_input_ids initialized on another stream; this is an unproven idea, not
an accepted fix. Larger LM-head shapes can also select different cuBLAS kernels.

## Paths and execution contract

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Task=R/artifacts/megamoe_pareto_perf_20260909; local scratch same basename.
Source=R/source/megamoe_pareto_perf_20260909. Copied local d5d5917 tracked
files into remote clone (git base dbf6689, allow_dirty_source1); timed runtime
frozen, startup-only fault-localization checkpoints added as described above.
General launcher/pool fixes below synced local -> source and local-mirror.
Reused run_afd_bundle.sh/run_afd.sh/afd_online_case_pool.py. No extra GPU tests.
CASES.csv66, state/pools/CASES, env.json/env_trays*.json, jobs.jsonl (last record
per tray=currentjob), logs/trays<trays>-<job>.out/.err, cache/trays<trays>/.
Original queued jobs inherit earlier env but current bundle reads final explicit
per-case settings from CASES.csv on each claim.

Every row: attn-only,ATP1,MB2,MegaMoE FP8xFP4,perf1,perfect_routing1,
afd_force_multi_mb_graph_overlap1,128K randomized resident KV/no prefill,
memory0.97,nominalKV979200,afd_num_pages=none,capacity-max gate ONLY b7.
One warmup+15 measured CUDA graph replays, original dominant-range policy
retains>=10 with<=5 outliers and both10% stability limits. No policy loosening.
All rows specify actual attention_gpus_override. A56:F16 uses historical
r7 ID/argv plus override56: ACTUAL ratio3.5:1, NOT ratio7 atEP16. Plot by actual
GPU counts. Shared config dry-run verified at EP16 A56/b7 and EP4 A4/b7.
Actual early3t/8t logs prove perf1,disable_overlapFalse,pending3,retire2,MB2.

## Corrections and evidence

User clarification expanded42 ->60 ->66 and required uniform best config.
Initial60 starts7040266(12t)/7040268(4t) canceled after~1:32; no accepted
metrics. Old b7 page pin14344 conflicted with979200 nominalKV and failed before
model work; next b6 case initializing when canceled. Replaced by7040469/7040470.
Five pending jobs retained. INITIAL_42_CASES and INITIAL_60_CASES plans/pools
and all attempts preserve history; currentpool was reset before accepted work.

General bundle now reads per-case attention_gpus_override and force-overlap
rather than relying on inherited environment. Pool now accepts non-divisible
fan-in, matching extractor's existing one-grouped-model-graph contract, and
rejects missing/mismatched requested perf/routing/overlap/GPU counts. Execution
identity includes these fields. Remote11 focused metric/fan-in/launcher checks
pass and bundle bash-n passes. Changes local/uncommitted; no commit requested.

19:50 two b6 cases failed preflight: original plan also marked b6 capacity-max
under0.82. At0.97 max is7.19:51 atomic existing-pool migration under pool lock
sets b6 gate0 for all66, releases both failedb6 into SAME live3t/8t jobs,
preserves6 completions/2claims; no GPU execution in failedb6 and no new job.
capacity_plan_migration.json and pool attempts preserve records. Subsequent
19:55/20:00 audits show0failed; corrected3t b6 passes,8t b6 also passes.

56:16 previously failed symmetric rendezvous, wrong-source resubmission,
missing56 override, serialized overlap, and invalid temporary plan status.
History: CODEX_PROJECT_qwen35_support.md around lines2550–2735. Retained runtime
separates actualfanin3/4 from uniformfanin_capacity4 and uses
max_rows=source_max_rows*fanin_capacity. Confirm all16 model logs when56:16
runs, plus dp_lockstep_fixed_lead/disable_overlapFalse. Keep canonical plan
status needs_corrected_target_batch_rerun. No new numerical/runtime change.

## Remaining work and reusable analysis

Continue scheduler/pool5min; failures get immediate targeted fixed-path diagnosis.
Do not poll large trees. Jobs choose and advance cases automatically.
Run on remote CPU:
python3 R/local-mirror/codex_scripts/summarize_afd_sweep.py <Task>
  --baseline <Task>/baseline/report/metrics --allow-partial
Omit --allow-partial at terminal. Validates pool metrics/settings, topology,
context/batch/precision/memory/throughput arithmetic, exact current trayjob.
Writes report/sweep.csv and sweep_summary.json. Final separately inspect sacct
all7 jobs, no failed/claimed cases, and66 settings/runtime artifacts including
all16 model readiness logs for each56:16 case. Existing pool validator checks
15-step strict dual-role trace timing proof. No further GPU run if artifacts pass.

Baseline42 JSONs extracted selectively from R/artifacts/archives/completed_sweeps/
afd_128k_perfect_routing_e00afda_20260901.tar.zst to Task/baseline/report/metrics,
and copied locally. baseline_series.json holds42 exact chart points. Old42 used
normal prefill; changed state mode and padding fixes mean this is overall rerun
comparison, not isolated kernel A/B. Separate24 EP16 additions from paired42.

Local plotting: MPLCONFIGDIR=scratch/megamoe_pareto_perf_20260909/matplotlib
python3 codex_scripts/plot_afd_sweep.py <localTask>/report/sweep.csv
  <localTask>/baseline/report/metrics <localTask>/report/pareto-comparison
Produces PNG/SVG/PDF + frontierJSON, labels partial counts, nativeSVG hover by
actualGPU/batch. Partial9-point figure visually checked; rebuild once complete.
Scripts are also in R/local-mirror/codex_scripts. Sync final compact artifacts,
write results conclusions, update this memory/root pointer, mark goal complete
only after full66-case audit and comparison/curve are delivered.

Root AGENTS.md missing; async create/location question unanswered. Follow user's
provided global rules. No subagents. No optional py_check/hash version checks.

## Scheduling clarification — corrected20:45 PDT

All66 plan argv use qwen3, and actual afd.log confirms Qwen3-235B-A22B.
The initial answer incorrectly cited Qwen3.5 hybrid release_turn-before-publish.
Corrected publicly: Qwen3 uses quantize_publish_o_fp8_release_turn, releasing
next lane after all local O quantization completes, while remote O publication
can still overlap it. Next Q must also be ready. It does NOT release immediately
after attention before quantization. Runtime source near3300 and CUDA fused
publisher are authoritative. Qwen3.5/GDN-specific language elsewhere is inherited
metadata, not proof that this sweep uses Qwen3.5. No shared expert executes here.

## Local Nsight copies20:40 PDT

User requested one b7 and one b4 case. Copied matching EP4 A8:F4
(3 trays, job7040269) attention rank1 and model rank9 .nsys-rep files
for i131072-fep4-r2-atp1-b{7,4}-fp4 under localTask/results/<case>/experiment/ray_logs/.
Also copied each afd-result.json and strict metric JSON. Four reports/eight
files,4,597,547 bytes total; rsync exit0 and all local sizes match remote.
trace-copy-manifest.json and trace-copy-files.txt record exact paths/roles/sizes.

## Retry failure localized20:41 PDT

7041337 EP8 A7/b7 failed again. All8 model logs explicitly report
model prewarm failed at moe with CUDA illegal memory access. Earlier stages
prepare_attention/finish_attention/post_attention_layernorm synchronized cleanly;
fault precedes LM head, which only surfaced the asynchronous error previously.
_run_model_moe includes both routed MegaMoE forward and replicated shared expert,
so culprit within those is still unknown. decode_graph_bs=[4], merged rows28,
perfect_routing1, compute_sms150, lanes2. b6 already claimed automatically.
Next useful diagnosis: distinguish routed versus shared prewarm, preferably
startup-only checkpoints. No further runtime edits have been made yet.
Do not blindly repeat same failing job; 16t has no accepted metrics so far.

User additionally requested EP8 or EP16 b4/b7 traces. Copied EP16 A16:F16
(8 trays, job7040267), case IDs i131072-fep16-r1-atp1-b{7,4}-fp4.
Both attention rank1 and model rank17 .nsys-rep plus strict metrics and
afd-result.json are local under the original relative paths. rsync exit0,
eight local file sizes verified (4,981,072 bytes total). Exact paths/roles
in trace-copy-ep16-manifest.json.

## Diagnostic reduction20:48 PDT

7041337 terminal FAILED1:0 in11:07; b6 also model prewarm failed at moe.
No shared-expert branch runs for Qwen3, so failure is routed MegaMoE.
Brief startup-only routed/shared callback was added/synced20:44 and removed
20:45 once actual model was reconfirmed. Original startup stage synchronizations
remain. Existing prewarm-order contract passed1/1 remotely; timed paths unchanged.
No new64-GPU retry. Submitted reusable megamoe_same_rank_smoke.sbatch diagnostic
7042033:2 trays/8 GPUs,15min short; exactEP8 rows28,bucket3633(aligned by wrapper),
H4096/I1536/E128/K8,FP8xFP4,150SM,production perfect routes WITHOUT expert-owner
interleaving,CUDA_LAUNCH_BLOCKING1 for error attribution,1warmup+3 iterations.
Source is same sweep source, existing16tray caches. Artifacts/submission.json:
Task/diagnostics/ep8-r7-prewarm/. No Nsight or IKET; no timing comparison intended.
Prior standalone164-case sweep passed with BALANCE_RANKS1 (interleaved owner map),
so this diagnosis tests a previously untested difference from production routes.
Pending outcome; do not resubmit automatically. 16tray12cases still outstanding.

Scheduler20:45:31/66 validated,2failed,2claims; 4tray7040470 running2:48,
12tray7040469 running28:31 (EP16 A2/b6),18traypending. Nextpoll20:50.

20:53 PDT: verified18tray7040264 PENDING then scontrol hold, to give8GPU
diagnostic7042033 nextslot before another large high-fanin allocation.
Task/diagnostics/ep8-r7-prewarm/held-sweep.json records release_required=true.
Remember scontrol release7040264 once diagnostic/fix supports resuming.

## Isolated reproduction and memory checking21:01 PDT

7042033 reproduced the illegal access on all8 ranks at the MegaMoE launch
(sm100_fp8_fp4_mega_moe.hpp:112) with CUDA_LAUNCH_BLOCKING1. Router returned
before fault, no attention workers/full-model weights. FAILED143:0 in2:28
(including teardown); individual workers CUDA driver error700 then abort.
This localizes fault to the fused MegaMoE kernel in isolation. Owner mapping
versus former passing standalone remains an unproven suspected trigger.
Added opt-in MEGAMOE_SMOKE_MEMCHECK=1 to existing reusable smoke.sbatch:
per-rank compute-sanitizer memcheck logs, error-exitcode99, failfast missing
binary or simultaneous Nsight request. Synced local/source/mirror; remote
bash-n passed. Submitted7042379 same8GPU shape with DG_JIT_WITH_LINEINFO1
and diagnostic-local JIT cache,15min. Artifacts diagnostics/ep8-r7-memcheck/.
18tray job still held;16tray remains stopped. Do not release until understood.

## User steering: machine check first, detailed micro analysis only as needed

User said stop detailed micro diagnosis, check machine issue, and do detailed
micro analysis only when needed. Do not continue speculative kernel analysis.
Only2 distinct sweep cases have runtime failures (EP8 A56:F8/b7,b6), each
failed twice.42 unique cases passed;22 not yet run. Small diagnostics are
additional attempts, not additional failed sweep cases.

Memcheck7042379 reproduced combine 8192-byte TMA invalid global read at
sm100_fp8_fp4_mega_moe.cuh:1757 (tma.cuh:75), multiple/all ranks, source
0xc0008000c15c1af0 etc. Canceled21:08 after sufficient evidence to releaseGPUs.
Extracted cuobjdump/nvdisasm from existing CUDA13 image into diagnostics/tools
for CPU-only disassembly (initial cuobjdump needed nvdisasm dependency).
SASS diagnostics/ep8-r7-memcheck/kernel.sass shows suspicious malformed high
word near2GB combine offsets (0x7f7b9af0 slot0,0x815b9af0 slot1); not proven
root cause. Brief explicit64bit-add candidate synced onlytoSource and queued
memcheck7042622; user steering caused cancelWHILEPENDING and fullrevertlocal/
Source. No GPU ever tested candidate, no runtimekernelchange retained.

Hardware evidence: failures span nvl72053,nvl72129,nvl72024,nvl72076 blocks.
All involved model/diagnostic nodes have no Slurm drain/failure reason.
Direct GPU telemetry SSH denied by pam_slurm_adopt (allocations released);
recorded diagnostics/machine-health.json. This does not prove GPUs healthy.
Successful sweep node groups:2t72173,3t72165,8t72124,12t72077,4t72078.
Unrelated72078-T16 drained for adjacent link remediation; not used by4tjob.

Submitted machine crosscheck7042690 using existing smoke on2trays/8GPU,
nvl72077-T[15-16] (part of successful12trayjob7040469),10min. Same original
EP8/rows28/bucket3633/FP8xFP4/150SM/productionperfectrouting, no sanitizer,
CUDA_LAUNCH_BLOCKING0. Added optional MEGAMOE_SMOKE_GPU_HEALTH1 to reusable
smoke: each node's localrank0 captures nvidia-smi-q-x beforeworkload. Synced
local/source/mirror, bash-n passed. Logs/submission under diagnostics/
machine-crosscheck/. Inspect GPU ECC,retired/remappedpages,fabric state,driver
and outcome.18trayjob7040264 remains held;16traystopped. Do not lose66scope.

## Hardware check result21:21 PDT; bounded fix now needed

7042690 reproduced identical illegal memory access on all8 GPUs on72077-T15/16,
which had just completed12tray sweep. nvidia-smi beforeworkload: driver580.126.20,
all8 volatile+aggregate ECC counters0,remappedrows0,pending/failureNo,recoveryNone.
No evidence of isolated machine failure; common software/driver issue not ruled
out. health-summary.json and rawXML under diagnostics/machine-crosscheck.
Told user hardware evidence and necessity of narrow fix to recoverremainingcases.

Reapplied minimal explicit64bit integer combine-address addition in CTA topk
TMA read; local and Source only, not yet accepted into mirror. Submitted
7042863,diagnostics/address64-validation,8GPUs/10min,original failingrows28,
bucket3633->3840,allperfformatsunchanged,productioncontiguousperfectroutes.
No sanitizer/profiling/CUDA blocking;1warmup+3 graph iterations, finiteoutput,
expectedrouteIDs and bytewise existingfull-barriercombine reference. GPU health
capture retained. If passes, next focused fullcase validation/resume; avoid more
broad microanalysis. All42 successful sweepmetrics precede this candidate;
assess commonconfig/perf impact before final comparisons.18traystillheld.

## Candidate correctness passed; perf must not meaningfully change

7042863 COMPLETED0:0 in2:28; all8 MEGAMOE_COMBINE_EXACT_OK, expectedrouteIDs,
finiteoutputs. PreviouslyfailingEP8rows28,bucket3633->3840 nowruns with explicit
64bit addressaddition. result.json statusok,3-iteration route+kernel0.136629ms;
not a performance conclusion. No further microscopic analysis needed currently.
User explicitly says fix must not affect perf meaningfully. Replied same-GPU
before/after comparison including already-successfulshape; require meaningful
regression investigation before sweeping.

This historical run used the retired source-comparison launcher, preserved in the
cleanup archive. Current comparisons use the manifest/group workflow documented in
[codex_scripts/README.md](codex_scripts/README.md). At the time, the launcher added
explicit profile-only selection and per-phase result JSONs.
Remote bash-n passed, mirrored. Created independentcopy of candidate/python
under diagnostics/perf-control-source/python BEFORE GPUallocation, replacedonly
kernelheaderwith gitHEADoriginal (saved local diagnostics/control-kernel.cuh).
No hash/version checks. Same harness from candidate; measuredcode differsonly
addresscalculation. Submitted7042982,8GPUs15min,EP8A3b7-equivalentrows12,
bucket1557->1920,150SM,FP8xFP4,perfectproductionroutes,10warmups100iterations.
Sequence control,control_repeat,candidate1,original_again(candidate2label), all
sameGPUs and bytewise referencecheck. No sanitizer/profiler/launchblocking.
FASTAFD_COMPARE_SOURCES control:candidate:control. Artifacts diagnostics/
perf-comparison/. Candidate2 usesseparatecache despiteoriginalsource; do not
confuse label withfixedkernel. Checktimings againstcontrolvariability before
resuminglargesweeps.42completedmetrics stilloriginalkernel.18trayheld.

## Resume21:43 PDT

7042982 timingsus original56.1203,repeat55.0170,fixed55.7229,original54.9661;
allsourcebytechecks pass. Fixed+0.706us/+1.283% vscontrolmedian, insidecontrol
range54.966–56.120; do not call a definitive0% regression. User emphasized
no meaningfulperfchange. Restricted final fix to combinebuffers whoseendoffset
exceedsINT32_MAX, retaining originalpointerchain forsmallerbuffers. Compiler
canfold sizepredicate fromlayoutconstants. Largepathuses sameexplicit64bitadd
that passed7042863; this finalguard not yet E2Evalidated. Syncedlocal/Source/mirror.
No further microdiagnostics requested/needed beforeactualrecoveredcase.

User asked why not64bitrepresentation: explained pointersalready64bit; evidence
points to generatedcalculation high-word corruptionnear2GBoffset. Explicit64bit
arithmetic is correction, smallpath preservedforperf. Avoidcallingprovenplain
32bitpointeroverfloworprovencompilerrootcausebeyondcurrentevidence.

Verified7041337 FAILED and perfcheckgates; release-failed bothb7/b6 viaexisting
pool (archivespreserved), submitted16trayreplacement7043302 at21:43 withunchanged
env_trays16.json andexistingcommand. jobs.jsonl appended latest16record.
All12cases remain. Hold18tray7040264 until first recovered16trayb7validates
endtoend, then scontrol release so bothremainingtrayjobsprogress. No other
failedclaimsremainafterrelease. Scope42completed+24remaining still66.

21:48 poll: 7043302 RUNNING4:20, b7 capacity gate passed for all56 attention
workers; no strict new metric yet. 7040264 stillheld. Refreshed local report
and jobs.jsonl,42 validated/0failed/1claim. Next poll21:53; release18tray only
after first recovered16tray metric passes. User64bit question answered: promote
before offset multiplication/addition, not after overflow; pointers already64bit,
root cause remains generated-address issue rather than proven32bitpointerstorage.

21:49 local partial curve rebuilt and visually verified at42/66; PNG/SVG/PDF
and frontierJSON under report/pareto-comparison*. GeomeanTPS/GPU by completed
family(all6batches): EP4A4+3.669%,EP4A8+4.064%,EP8A8+2.238%,
EP8A24+2.908%,EP8A40+2.736%. EP16A16/A32 have no previous paired baseline.

## End-to-end fix confirmed21:53 PDT

7043302 recovered BOTH previously failing EP8 A56:F8 cases. b7 latency26.7089468ms,
TPS/GPU229.3239058; b6 latency23.1023829333ms,TPS/GPU227.2492849. Both15samples,
0outliers, strict pool validation passes with final size-guarded64bit fix. Full
audit44/66,32baselinepairs,latencygeomean-2.80617%,TPS/GPU+2.88719%,24/32faster.
No failedcases,1claim. Released18tray7040264 via scontrol release after checking
both completedrecord jobIDs7043302; scheduler confirms PENDING(None), not held.
Next poll21:58, continue bothjobs. This proves full previously-failingmodelworkload
now passes; don't claim allremainingcasesverified yet or isolate fixperf from
oldbaseline, whichpredatesotherchanges.

21:58 strict audit46/66,34paired,TPS/GPUgeomean+2.8913%,26/34faster. Enhanced
existing summarize_afd_sweep.py to inspect all model-rank startup logs plus
coordinator: ATP1, two microbatches, perf_context131072, overlap fixed-lead,
allMLPranksfanins=actualmappedDPcount with uniformceil(A/F)capacity andmax_rows,
mappedDPs cover eachattentionrankexactlyonce. This also handles56:16's3/4fanin
anduniform4capacity. Executed successfully on46completedcases (no new GPUtests);
script synced tolocal-mirror.18job schedulednode72040-T01-18/start21:58:21,
notheld/restarted. Next22:03 poll.

22:03 audit48/66,36paired,TPS/GPU+3.0304%,28/36faster. AllEP8A56:F8
batches2–7 pass;16job now progressesEP16A48:F16.18job stillPENDING(None),
squeue--start estimate22:03:27; only16jobrunning foruser, nootherQOSuse.
Keep samependingjob; noauthoritativeterminalerror or hold. Nextpoll22:08.

22:08 strict audit49/66,0failed,1claim; first EP16 A48:F16/b7 passes. Found
18traypending job repeatedlyplanned72040-T01-18 whileT10 IDLE+COMPLETING+PLANNED.
Otheruser job6958525 remainsCOMPLETING onT10 (and72115-T05), authoritative
sacct/scontrol EndTime2026-09-06T15:09:48, cancelled3daysago. This explains
continuallysliding estimatedstart, not our hold/config/QOS. Updated ONLYpending
7040264 ExcNodeList=nvl72040-T10 at22:09 to allowanotherintact18traygroup.
No resubmit, no otheruserjobmutation, same72GPUconfig and segment18.
Next scheduler poll22:13.

User expanded work22:10: secondparallel66-case scheduleoptimization sweep;
see CODEX_PROJECT_attn_release_perf.md. Original66 stillmustcomplete, preserve
controlSource. At22:13 original49pass;18rescheduled72020 afterstaleT10excluded,
estimated00:15:19. Next22:18.

## A56:F16 postprocessing recovery (2026-09-10 00:51–00:59 PDT)

7040264capturedA56:F16b7/b6successfully (72worker profiler-stoplogs,16trace
launches,15measuredsteps),but extractor rejected plan/resultactive-GPUmismatch.
CASES lackedactive_gpus; historicalr7 aliasmadeextractor infer(7+1)*16=128,
instead of existingoverride56+FEP16=72. Not a machine orruntimefailure.
Metadata-onlymigration viaexistingpoolcommand_migrate_plan addedactive_gpus
fromattention_gpus_override+ffn_ep toALL66rows in BOTHcontrol/candidate plans.
Executionidentity/alltimingpoliciesunchanged,includingcurrentcandidateclaim.
Backups CASES.before_active_gpu_metadata.csv,active_gpu_plan_migration.json.
No runtime or sourcefile changed. Existingextractor already supports explicitfield.
Localplans/jobs/migrationrecords synchronizedfromremote.

CPU-onlyexistingNsightbinary +extract_cuda_wall.py recoveredbothsavedcaptures:
- b7:26.5953792ms,TPS/GPU204.7139243062,15samples,0outliers,range2.26262%.
- b6:23.0246522667ms,TPS/GPU202.6813092601,15samples,0outliers,range2.20433%.
Existingrecover-failed-completion archivedfailedrecords andpromotedboth after
strictmetricvalidation;noGPUreruns. Fullruntime/topologyauditthenpassed62.
Artifacts/source/policyunchanged. Fourunruncasesareb5,b4,b3,b2;7048005continues
pool without repeating8acceptedcases. Oneinitialsubmissionpreflight usedsqueue
onterminalID andfailed BEFOREsubmission; correctedtoaccountqueue+sacct.
Newjobrequests1hrbecauseonlyfourcasesremain;env_trays18_continuation.json.

summarize_afd_sweep.py now accepts recorded prior same-trayjobs so continuation
preservesvalidoldercases. It additionallyrequires exactcase/job completionlog
(orfailedlog+explicitrecoveryrecord),executionidentity,andstrictmetric/runtime
proof;accepted_by_jobreports provenance. Actual62caseauditpasses after newjob
record. No optionalhash/versionchecks ornewGPUdiagnostics.
