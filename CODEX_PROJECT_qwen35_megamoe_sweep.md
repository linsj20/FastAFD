# Qwen3.5 MegaMoE decomposition sweep — complete

Current checkpoint (September 10): all requested work is complete. Final
scheduling and Qwen3.5 barrier fix are promoted into main; superseded snapshot
repos removed, source patches/results retained. See CODEX_PROJECT_attn_graph_edge.md.
The execution instructions and pending states below are historical.

## COMPLETE Sept10 ~11:20PDT
Final EP4 job7059256 COMPLETED0:0 in1:17:20, all127remaining inputs done.
EP16 job7059254 completed20:10; EP8 job7059255 completed56:20. Full final
collector/audit PASS:299/299 artifact contracts,897samples,1110equivalent
inputs, no missingcases. Scope/dedup,input/balance,3samples/6markers,phase
closure,normal/profiled status,and EP64 exactcombine allvalidated. Compact
artifacts/aggregateCSVs copied to local results. All three originally requested
sweeps now successful (AFD132/132 +decomp299/299). Activegoal continues for
new graph-edge alignment7060661 then reauthorized66AFD perf sweep. No merge.


User requested same workflow as completed Qwen3 decomposition, EP4–64,
all previous integer A:F ratios within72GPUs, calculate max batch, deduplicate.
EP4 A1–17:F1,EP8 A1–8:F1,EP16 A1–3:F1,EP32 A1:F1;
EP64 preserves previous eight-attention-GPU A1:F8 exception.
Qwen3.5-397B-A17B H4096,I1024,E512,topk10; routed experts only,
FP8xFP4,MB2,150SM,bucket3840,integer-balanced interleaved expert routing.
Frozen copy of current control source retains validated address64 fix.
Only task-local barrier-allocation correction below; no tuning or arithmetic
change. Main scheduling source stays unchanged.

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Source=R/source/qwen35_megamoe_input_sweep_20260909
Task=R/artifacts/qwen35_megamoe_input_sweep_20260909
Local=scratch/qwen35_megamoe_input_sweep_20260909
Reusable scripts megamoe_input_sweep.py/.sbatch and profile_megamoe_iket.sbatch
now accept manifest model dimensions/capacity rather than hardcoded Qwen3.
Normal Nsight and three six-marker IKET samples required per unique input.
Retain all aliases; sparse EP64 valid rows are balanced floor/ceil(tokens/64).
User authorized idle-overhead optimization and resubmission, then specified
ONE job per EP with a TWO-HOUR limit for all remaining cases. At most8 jobs
outstanding (running+queued); supersedes prior single-chunk policy. No subagents.

Latest Sept10 ~10:44PDT:241/299 captures,723samples fullauditPASS; all1110
aliases verified. EP4=81/139,EP8=83/83,EP16=39/39,EP32=19/19,EP64=19/19.
7059254 EP16 COMPLETED0:0 in20:10,27remaininginputs;7059255 EP8 COMPLETED0:0
in56:20,83inputs. Only7059256 EP4 running43:09,69/127newinputs captured,
58remaining. Job2hrshort/onecontainer. No newidle-reap cancellation.
Compact241artifacts/audit copiedlocal. ep_jobs.json records grouped jobs;
chunks.json retains originalplan/provenance. FinishEP4+finalaudit299.
Both AFD sweeps complete132/132. Newpublisher-entryfullsweep+smoke CANCELLED
byuser beforeexecution. NewDIRECTgraph-edge onecase7060293 running (prior
7059949 metadataindexfailurefixed); see CODEX_PROJECT_attn_graph_edge.md.
Goal includes299decomp and requestedonecase+tracecopy; no mainmerge/commit.
Monitor5/10min; no repeatedAFDaudits or broadmicrodiagnosis.

Idle reduction validation7059114 COMPLETED0:0 in3:49 (one GPU step.0,3:28).
Resumed only ep16_rows10..12; three captures pass full artifact audit.
Optimized weight preparation byte-identical AND CUDA RNG progression identical
on all16ranks. First case64.46s including explicit validation/cachebuild;
next two combined79.55s (~39.8s/input), versus old steady188–190s/input.
Kernel, graph timing, inputs, three samples and six markers unchanged.
Workflow now starts ONE container per node at job beginning, keeps rank workers
alive through all cases/normal+IKET modes, teardown only once at end.
Shared TVM helper cache per EP; skipped construction of discarded FP4 numeric
reference tensors in profiling path. Numeric correctness default retains them.
Modified megamoe_input_sweep.sbatch,profile_megamoe_iket.sbatch and
 tests/afd_megamoe_same_rank_smoke.py synced local,Q35Source,local-mirror.
Remote bash-n both launchers and local diff --check pass; no pycheck/hashcheck.
Original three workflow files preserved in Task/workflow_before.
Bulk jobs disable extra weight-build validation already passed by7059114.

Historical7051184 cancelled by adminUID146504 after30:01 at03:38:57,
"Cancelled by occupied idle job reaper: 4/16 GPUs idle for 30m, 25% waste".
Its9captures retained. No kernel/machine failure or unused-rank mapping bug.
Real overhead diagnosis: two25GB container filesystem launches per input,
~36–38s teardown after useful output per mode, repeated16s TVM helper build,
and large unused dequantized references. User explicitly authorized fixing and
resubmitting; old profiling-exemption question/block superseded. No dummywork
or timer-based splitting. Continuation7059114 recovered all3 remaining inputs.
Saved cancelled-7051184-accounting.txt. Compact62 artifacts/metadata local.
7049534 COMPLETED0:0in40:51,12EP32rows1–12. Firstrows1normalNsightmedian
32.256us,IKETmedian30.432us;all3samplesretained. Prior7048764completed0:0
in24:28,7remainingEP64inputs. AllEP32/EP64rangesfullycoveredandvalidated.
PriorEP64job7047368 COMPLETED0:0in44:20,12/12;EP4job7045832 COMPLETED0:0
in33:48,12/12. All3IKETsamplesretained;some231–307usenvelopes;actualnormal
NsightEP64kernels~30us. No microdiagnosis.
BOTH AFD sweepscomplete132/132strict,finalCSV/plots/READMEslocal+remote.
Finalcandidate66-pairTPS+0.05844%;b7+0.89165%. No furtherAFDauditsneeded;
no mainmerge. Wholegoalremainsactiveuntilall299decompcapturesvalidatedandjobsfinish.

Manifest299unique inputs/1110aliases,EP4/8/16/32/64=139/83/39/19/19.
Batch1–37, MB2 paddedrows1–19 per attentionGPU; largest globaltokens
EP4=1292,EP8=1216,EP16=912,EP32=608,EP64=152.
Capacity calculation and full alias/dedup audit pass. Remote bash-n pass
for both generalized launchers; original planner defaults preserve identical
164Qwen3 cases/455aliases. Actual model config confirmed H4096/I1024/E512/K10.
Profiler validates dimensions reported by harness.
Submission command/env/case IDs stored Task/jobs.jsonl. Source frozen live.

Optional question pending: previous validated Qwen35 batch37 at memory0.82
versus establishing new capacity0.97. No reply before submission; stated assumption and used previous
validated capacity (same previous Qwen35 reference), label explicitly.
Saved capacity77659pages*64=4970176tokens,128Kcontext,17outputtokens.
Scheduler floor((capacity-(131072+17))/(131072+16+63))+1=37.
This is NOT proof of maximum at0.97. Source proof is prior20260905attn-only
A12:F4/b37 capacity-preflight.json. User's overall goal now requires ALLTHREE
sweeps successfully complete; no merge or commit authorized in this task.

User23:21asked whether differing EPs need different jobs: confirmed each
job has fixed EP and minimum matching allocation. First job only12EP4cases.
Planned27boundedchunks; after initialEP4boundary, prioritize EP64,32,16,8
then remainingEP4. chunks.json is static planning data, no auto-submission.

## First compile failure and correction

7045707 FAILED143:0 in3:01 on72067-T18, before any measurement. NVCC
static assertion: CTA combine needs10 barriers but EP4large-inputgeometry
has4epiloguewarps, oldallocation2/warp=8. Not a machine fault.
Isolated task source overlay now allocatesmax(2*warps,topk) for existing
CTAshape(H4096,topk8/10), with matching hostshared-memorysize, tmem pointer
offset, and initialization. Topk8 allocation unchanged; Qwen3 controlsources
and main runtime/kernel left unchanged. Only16 extra shared bytes for4warps/K10.
Overlay files retained Local/source/python/.../sm100_fp8_fp4_mega_moe.cuh and
csrc/jit_kernels/heuristics/mega_moe.hpp. Keep the validated EP4 fix isolated until broader sweep completes/userasks.
Retry submitted with freshcache ep4/barriers10 and existing normal-only
MEGAMOE_SMOKE_COMPARE_COMBINE=1 for byte-exact selected/full-barrier check
outside timing. Same12plannedcases, none previously completed. See jobs.jsonl.

Retry7045832 submitted23:25PDT. Submission records are in Task/jobs.jsonl.

00:08PDT progress4/299,7045832running13:31,no furthererrors. Nextpoll00:35.
Both18AFDs genuinely pending, no dependency/hold: schedulerestimates original
00:33:14 on72022-T01–18, candidate02:07:54 on72047-T01–18 (estimatesnotguarantees).
NextmanualchunkEP64 includeslargestglobal152 first then8–88 (12cases),
normal-onlyexactcombinechecks required. RemainingEP64 globals96–144 (7cases).
Otherchunks EP32,16,8,remainingEP4; chunks.json staticplan, noautochaining.

Latest10:05PDT:7059254 EP16 COMPLETED0:0 in20:10,27/27; all39EP16 done. Full partial artifact audit128/299,384samples PASS and compact artifacts copiedlocal. 7059255 EP8 running25:33;7059256 EP4 running3:56. User kernel-entry release request is separate isolated source and validation7059597 queued behindthese (total3outstanding,within8). Continue5/10min monitoring.

Sept10 ~11:49PDT user asked status and similarQwen3 decompositionHTML.
Confirmed final Slurm7059254/55/56 allCOMPLETED0:0 and complete299audit.
Created results/qwen35-megamoe-comparisons.html from existingQwen3fragment;
matching editablefragmentonelevelabove, comparison_data.json +3comparisonCSVs.
Same3sections/tablesleft/fullphaselanesright/hatching/median-durationactualIKET
launch; table3samplemedians,separateNsightmarker,shared0–200us axis.
15rows: maximumB37/maxA:F acrossEP4–64; largestcommon global128tokens;
largestcommon perFFNGPU2tokens acrossALL EP4–64 (unlike Qwen3 whichcouldnot
includeEP64 inperGPUcomparison). Typical maxratio/maxbatch/middleratio aliases
retained. Capacitylabel explicitly128K atmemory0.82. ModelH4096/I1024/E512/K10.
Historical HTML-template rewriter (removed in Sept10 comprehensive cleanup) populated the template
fromaudit+CSV+selectedpercase3samples; selected15Nsightvalues checkedagainst
source duringgeneration. No browsercheck peruserpriorinstruction,nogpuruns.
Standalone renderedwithvisualizeskillrender.py; HTML+fragment+JSON+CSVs synced
remoteartifactroot. OldQwen3HTMLunchanged. AFDmonitoringcontinuesseparately.

## New requested Qwen3.5 A3:F1 EP16 perf case (Sept10 ~13:23PDT)
User asked whilewaiting to run attn-only perf case andcopyNsighttraces.
Submitted7062744,16trays/64GPU (A48:F16),128K,b37,MB2,FP8xFP4,perf1,
perfect1,explicitoverlap1,NSYSnode15steps,memory0.82,no pages,capacity4970176.
Batchunspecified; explicitlystatedB37 basedonvalidatedQwen35decompcapacity.
Task R/artifacts/qwen35_a3_ep16_perf_20260910; matchinglocal scratch.
Reused frozen Source=R/source/qwen35_megamoe_input_sweep_20260909, retaining
validatedtopk10barrierallocationand64bitaddressfix. Qwen3.5hybrid schedule
unchanged; Qwen3-onlyattentioneventoptimization doesnotapplytohybridpath.
No newcodechangesorGPUtests. Existingrun_afd_bundle onecontainerpernode/job,
short2hr,16node segment. CPUplaninitPASS1case,index1. job/env/planlocal+remote.
Only2outstandingnow including Qwen3finalsweep7061097; max8respected.
Aftercomplete validateexistingpoolmetric/runtime(actualperf1,attn-only,perfect1,
A48:F16,MB2,b37) andsummarizewithlocal-mirror/summarize_afd_sweep.py TASK
(no baseline required). Then copy ALL64rank.nsys-rep +metric/afd-result.json,
manifestwithrelativepathsandbytecounts. Goal includes thiscompletion+tracecopy
as well as remainingQwen3full66 (54/66 currentlycomplete). No mainmerge.

LatestuserrequiresQwen35schedulingoptimizationtoo: superseded7062744pending
with7062904 afterQwen35attentioneventextension. See CODEX_PROJECT_qwen35_graph_edge.md
fornewsource/tests/job/tracecopycontinuation. Decomp+HTMLremaincompleteunchanged.
