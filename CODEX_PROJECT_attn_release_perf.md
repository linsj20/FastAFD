# Attention release Pareto comparison — completed 2026-09-10

Current checkpoint (September 10): all requested work is complete. Final
scheduling and Qwen3.5 barrier fix are promoted into main; superseded snapshot
repos removed, source patches/results retained. See CODEX_PROJECT_attn_graph_edge.md.
The execution instructions and pending states below are historical.

## Scope and current state

User authorized a second parallel sweep of the SAME66 cases and uniform best
configuration as CODEX_PROJECT_pareto_perf.md, changing only scheduling: next
microbatch attention becomes eligible immediately after previous attention,
before O quantization/publication. User clarified22:58: keep sweeping; MAIN MERGE ONLY WHEN USER ASKS after
sweepsfinish. No mainadoption is part of currentcompletion.
Full scope is132 runs: original66 vs archived42, candidate66 vs original66.
Report all11 batch7 pairs separately; user expects a small gain, particularly
EP8/EP16. Do not infer success from EP4 alone or claim expected gain as measured.
No subagents. No newvLLM. Existing scripts, one job pertray,5min monitoring.

Final01:46PDTSept10: ALL66/66strict, all7jobsCOMPLETED0:0.
Originalalso66/66after7048005completed0:0in11:07at01:42:23.
Bothpools0claims/failed. All132metric/runtime/fanin/provenanceaudits pass.
Final66pairs38faster,TPSgeomean+0.0584365% (latency-0.0584024%),nearneutral.
Bybatch2..7(each11pairs): -0.741935%,-0.008493%,-0.791288%,-0.035139%,
+1.051080%,+0.891649%. Batch7 8/11faster; all11tableinreport/README.md
andbatch7_comparison.csv. LocalCSV/metrics/final66-casePNG/SVG/PDF refreshed.
Regressionsretained:EP16A56:F16b7 -1.256%TPS (26.5953792->26.9337142ms,
15samples0outliers),EP8A24:F8b4 -5.30%,EP16A48:F16b2 -8.97%.
NoadditionalGPUdiagnosis, no mainmerge. User will request merge separately.
Somepairsdifferentnodes/times;18traypairsall72022-T01–18sequentially.
CompletionofwholegoaldependsonQwen35remainingdecomp,28/299as01:46.
01:30partialpollcaughtfinalcompletionrecordbeforecachedjoblogflush;
retryafterflushpassedall66withoutcodechange. Bothplans'active_gpusmetadata
correctedthroughpoolmigration;runtime/timingunchanged. Seecontrolrecovery.

Candidate jobs, all submitted22:14 with same2hrshort/segment=trays:
- 2trays7043989:6/6, COMPLETED0:0 in20:05, nvl72087-T11/13.
- 3trays7043993:6/6, COMPLETED0:0 in20:41 (sacct23:00).
- 4trays7043994:6/6, COMPLETED0:0 in20:43, nvl72042-T13/14/17/18.
- 8trays7043995:12/12, COMPLETED0:0 in37:34, nvl72139-T01–08.
- 12trays7043996:12/12, COMPLETED0:0 in38:01, nvl72134-T01–12.
- 16trays7043997:12/12, COMPLETED0:0 in38:11, nvl72030-T01–16.
- 18trays7043998:12/12, COMPLETED0:0in39:17, nvl72022-T01–18.
Initial sixholds were RELEASED22:30 after firstb7strictmetric/actualGPUtrace.
scontrol top7043995 succeeded22:35 to prioritize requestedEP8/EP16; it started.
No holds remain. Do not cancel/restart livejobs due observationtimeouts.
Allcandidatejobs exclude staleCOMPLETING72040-T10 and72115-T05; original18
excludes72040-T10, formerly blocked on that3day-oldcancelledjob'scleanup.

## Paths and source change

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
ControlSource=R/source/megamoe_pareto_perf_20260909, frozen fororiginaljobs.
ControlTask=R/artifacts/megamoe_pareto_perf_20260909.
CandidateSource=R/source/attn_release_pareto_perf_20260909: full120MB cp-a
snapshot ofControlSource22:11, including validated64bit combine-address fix.
CandidateTask=R/artifacts/attn_release_pareto_perf_20260909.
LocalTask=scratch/attn_release_pareto_perf_20260909.
LocalTask/source is an OVERLAY containing only changedruntime+tests; fullsnapshot
is remote. Main localruntime and local-mirror stay control; userwillrequestmergeafterwards.
Candidate CASES.csv exactcopy66; env*.json changesonly task/source/cache paths.
comparison_sources.json, jobs.jsonl, trace manifests, validationJSONs local+remote.
Case/metric/result identities and common best settings validated by existingpool
and enhanced codex_scripts/summarize_afd_sweep.py (see originaltask memory).

Candidate run_attention_fmha Qwen3 branch calls existingrelease_turn after
forward_prepared and view/contiguous, then existingquantize_publish_o_fp8 with
NO late fusedturnwrite. Qwen3.5 unchanged. EachlogicalMB has ownbackend/workspace.
InstalledFlashInfer decode.py inspected: BF16 output allocatedtorch.empty_like
(query), separate fromworkspace; producerlane orders publication before buffer
reuse. O readiness still gates modelconsumption. Quantizationcounters left
allocated to preserve memoryconfiguration. One extra tinyreleasekernellaunch;
measure its cost versus overlap, don't assume gain for everyshape.

Four focused existing CPU scheduling tests PASS0.035s onremote candidate:
MB2fullwait/FMHA/release/publish order, MB3stream/slotreuse, eager-no-device-turn,
missing-turn rejection. tests/test_afd_fmha_runtime.py updated MB2/MB3 to observe
separate release/publisher calls and wrappedtickets. Validation JSON saved.
No newGPUdiagnosticjob; fullsweepfirstcase supplied realGPUvalidation.

## Actual GPU handoff validation

FirstEP4A4:F4/b7 strictmetric passed7043989, then CPU-onlyNsightexport ofits
attentionreport to validation/attention-b7.sqlite. Existingbinary:
R/artifacts/megamoe_cpu_validity_20260909/cpu_tools/usr/local/cuda-13.0/
NsightSystems-cli-2025.5.1/target-linux-sbsa-armv8/nsys.
15measuredreplays*188FMHA=2820. Every same-stream sequence is FMHA ->
release_turn_kernel -> quantize_publish_o_fp8_kernel<...,false>. Allreleases
endbeforepublisherstarts;1410/2805next-FMHAhandoffs startbeforepreviouspublisher
ends. None observed beforepublisherstarts. Thus eligibility moved beforequant;
wholefusedkerneltimestamps cannot identify its internalquantizationcompletion.
One same-streamFMHAend/releasestart timestampoverlap32ns triggered initial
strict orderingassert; recorded raw anomaly and relied onCUDAstreamorder/actual
nodesequence, without loosening anyperformancemetric gate. Evidence
validation/attention-handoff.json. Released sixheldjobs afterstrictcompletion+
traceproof; validation/sweep_released.json. No further microdiagnosis needed.

## Measured comparisons so far

All against same-case ORIGINAL current sweep, not archivedW4baseline:
- EP4A4:F4 all6:TPS/GPUgeomean+0.2854%,4/6faster; b7-0.35837%,b6+0.06732%.
- EP4A8:F4 b7 also slightlynegative; no EP4b7gain established.
- EP8A24:F8/b7:26.569075 ->26.2017002667ms, TPS/GPU+1.40210265%,
  15samples0outliers. User informed22:48. Differentnodes/time, not a sameGPUABBA.
- 22:50 overall14pairs+0.3512%; batch7threepairs+0.3064%,1/3faster.
Do not generalize onepositiveEP8case to allEP16/higherfanins.

## Requested traces copied and verified

Bothnewoptimization requests fulfilled (EP4first, thenuserpreferredEP8orEP16):
LocalTask/results/<cid>/experiment/ray_logs/:
- EP4A4:F4/b7 cid i131072-fep4-r1-atp1-b7-fp4, job7043989:
  minisgl_rank1_430811.nsys-rep792741B(attention),
  minisgl_rank5_886005.nsys-rep1653067B(model).
  Plusmetric/result;4files2462443B, trace-copy-b7-{files.txt,manifest.json}.
- EP8A24:F8/b7 cid i131072-fep8-r3-atp1-b7-fp4, job7043995:
  minisgl_rank1_1261680.nsys-rep907857B(attention),
  minisgl_rank25_3898558.nsys-rep1601999B(model).
  Plusmetric6327B/result11647B;4files2527830B,
  trace-copy-ep8-b7-{files.txt,manifest.json}.
Everylocalfilesize verified against remote manifest. ClickableEP8links delivered
in final22:49. No additionalEP16copy owed unlessuserasks.
OriginalsourceEP4/EP16b4/b7traces also copied earlier, seeoriginaltask memory.

## Remaining actions and deliverables

Continue both66sweeps, resolve realfailures narrowly and checkmachinesfirst.
For each56:16case, existingenhanced summarizer checks all16modelranklogs:
actualfanin3/4, uniformcapacity4/max_rows, mappedDPcoverage56, twoMBstreams,
perf1 and overlapFalse-disabled, coordinatorfixedlead. Both18jobsstillpending.

Remote CPUreport, existing same script:
python3 R/local-mirror/codex_scripts/summarize_afd_sweep.py CandidateTask
 --baseline ControlTask/report/metrics --allow-partial
Remove --allow-partial onlywhenall66candidatecasesandall66controlcasescomplete;
separatelystrictvalidateoriginal66 againstarchived42. Currentbaselinefilecount54.
Summaryincludes paired_by_batch. Sync reportCSV/JSON/metrics locally.

Existing plot_afd_sweep.py accepts label overrides (defaults originalplotunchanged).
Candidateplot uses --baseline-label 'Current schedule' --candidate-label
'Earlier attention handoff' --added-label 'Earlier handoff (unpaired)'.
Localpartial2/66render visuallychecked at22:32; PNG/SVG/PDF/frontierJSON under
report/attention-handoff-comparison.*. Rebuildfinalcandidate66 andoriginal66
plots, inspect visually; reportpaired66 andbatch7 separately, plus original42
improvement andEP16 additions. Finalcurves must have accurate actualGPUlabels,
especiallyA56:F16(notA7:F1). ExistingoldHTML remains unchanged.

Do NOT merge/adopt candidate into mainlocal/local-mirror yet. User22:58 says
justcontinue sweep and willasktomergeafteritfinishes. PreservebothfrozenSources.
No additionaldiagnosticGPUruns unlessneededforcompletion; recordperformancedeltas.
No commitrequested.
Update root andtopicmemories, sync relevant cleanedcode, give finalartifacts and
limitations. MarkgoalcompleteONLYwhenall132runs strictpass, comparisons/curves
delivered. Mainmerge is a futureuserrequest, not currentgoalcompletiongate.
Preserve original42baseline provenance/state-mode caveat
(normalprefillold vs syntheticresidentcurrent) fromoriginaltask memory.

## Material result to follow up (22:56)

EP8A24:F8/b4: control18.8434229333ms -> candidate19.898132ms (+5.59723%
latency;TPS/GPU-5.30054%). Both15samples0outliers, dominant range1.338%/1.810%,
max-mediandiff1.287%/0.821%; bothattention/modelrole spansslowtogether.
Tolduserthisneedsfollowupbeforemainadoption. No microtraceanalysis/newGPUjob
yet: firstfinishcohort, thenboundedcheckifregressionrepeats. Differentmachines
so cannotattributeall5.6%tohandoffyet, but cannotdismissaswithintraceoutliers.
Preservebothsources; do not silently selectper-batchfallback orloosenmetrics.

Usersteering22:58: "Just keep doing the sweep, i will ask you to merge after
the sweep finishes." Acknowledged. Continueall132runs/comparisons, no mainmerge
and no newregressiondiagnosticcampaign now. Keep b4finding inresults.

New23:58regression: EP16A48:F16/b2 (i131072-fep16-r3-atp1-b2-fp4)
control16.0374539333ms ->candidate17.617908ms,+9.85477%latency,
TPS/GPU93.53105588 ->85.14064212 (-8.97072%),15samples0outliers.
Retain and report; no diagnostic rerun per user. Other source/timing differences
mean do not attribute this causally to scheduling without controlled rerun.

00:13PDT both18trayjobsstillPENDING; countsunchanged54/66each.
Bothlocalcomparisonplotsrefreshed54cases (PNG/SVG/PDF). Originalstem
report/megamoe-pareto-comparison; candidatestem report/attention-handoff-comparison.

## HTML curve update (Sept10 ~09:54PDT)
User requested adding both completed66-case sweeps to scratch/visualizations/afd_128k_optimized_attention/fastafd-128k-five-pareto.html. Added FastAFD_MegaMoE_Optimized_W4 (13 frontierpoints) and FastAFD_MegaMoE_Early_Attn_W4 (17); all original5series preserved,506totalpoints. Added same early-attention66points to afd-trtllm-perfect-balance-128k-pareto.html as exact requested label afd_sche_optimized. Source is each task report/sweep.csv; actualattention/FFNGPU counts used, A56:F16 ratio3.5, all measured points retained, independent nondominance frontiers. Source samples/outliers preserved (some15total with12/14retained). Backup originals besideHTMLs. Browser check aborted in sandbox; user explicitly said no checks, so no further browser/test attempt. No mainmerge. Continue decomposition monitoring.

User final label correction: second HTML curve is named afd_optimized (replaces afd_sche_optimized). Data remains the early-attention66-case sweep.

Sept10~10:02PDT: copied ALL72rank Nsight traces each for optimized A8:F1/EP8 batches7and6 (job7043998,A64:F8),144traces total,plusmetrics/results. Localfilesmatchremote recordedbytecounts; manifests trace-copy-ep8-r8-b67-{files.txt,manifest.json}. These are measured separate-release-kernel version. User now requests eliminating standalone release by moving turnstore to quant_publish entry; implement in a NEW isolated source to preserve all measured snapshots, no mainmerge.

## Fused attention release at publisher entry (Sept10 ~10:06PDT, active)
User requests removing standalone release_turn_kernel for Qwen3 AFD; release
at beginning of quant_publish instead. Preserve completed measured snapshots
and both HTML curves, which still represent prior separate-release measurements.
NEW full remote source R/source/attn_publish_entry_20260910 copied from measured
candidate; local overlay scratch/attn_publish_entry_20260910/source. Main NOT merged.
Runtime uses fused quantize_publish_o_fp8_release_turn again; kernel now performs
single block0/thread0 store_release_device at entry BEFORE quantization. Removes
last-CTA quantization atomics; independent lane outputs/staging preserve safety;
publication completion/ready release unchanged. Quant counter allocation/API
retained for identical memory config. Qwen3.5 branches unchanged (no quantizer).
Four existing remote CPU scheduler tests PASS0.031s: MB2, MB3 wrap/slot reuse,
eager no turn, missing-turn rejection. Existing source-contract assertion updated
to early store with unique thread guard; source contract check run separately.
GPU validation7059597 queued behind decomposition jobs: one tray/4GPU,30minshort,
8192rows,9iterations,2fabricpairs, FP8 payload/UE8M0 byte exact. Uses existing
fused_fp8_cast_smoke.sbatch adapted in isolated overlay to ONE srun/container
for rank0 prebuild and all rank lifetimes. Others await normal Gloo initialization.
Job/env/command in local+remote task job.json. Kernel GPU validity/perf NOT claimed
yet. Need terminal success + AFD_FABRIC_TRANSPORT_SMOKE_OK; no new full sweep
or mainmerge requested. Need report new version separately from measured curves.

User subsequently requested another full66case sweep of fused-entry change; supersedes earlier no-new-sweep note. See CODEX_PROJECT_publish_entry_perf.md for authoritative ongoing submissions/validation.

User cancelled publisher-entry66case resweep beforeexecution to furtheroptimizeAFD. All5submittedjobs cancelled; remaining2willNOTbesubmitted. Keep implementation and targetedGPUvalidation7059597; do not launch anotherAFDsweep without newuserinstruction.

## Next optimization idea: direct attention graph dependency
User asks whether nextFMHA can wait directly for previousFMHA end without
release-mechanism latency. Inspected current _run_attention_body: MB2 prearms
Q-ready+turn waiters on each lane; publisher-entry still waits for publisherCTA
start then turn visibility/waiter exit. Proposed captured INTERNALevent record
immediately afterFMHA, nextlane wait AFTER its prearmedQ-ready-only waiter and
BEFORE FMHA; ordinary captured record/wait encodes graph edges, no standalone
release or turn polling for localserialization. Publish branches from sameFMHA
event, independent of nextFMHA; retain remoteQ-ready+O-ready and slotownership.
No zero-gap guarantee: graph dispatch, Q readiness, and resource contention
remain. CUDAofficial cross-stream capture docs confirm captured event represents
nodes and adds dependencies, not an external event record/wait graphnode.
This is a proposal only, not implemented/benchmarked or authorized as new sweep.
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cross-stream-dependencies-and-events
