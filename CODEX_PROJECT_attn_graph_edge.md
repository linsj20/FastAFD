# Direct attention CUDA Graph dependency — COMPLETE Sept10 2026

FINAL14:29PDT: all66/66strictPASS,0claims/failed;all7jobs7061091–97COMPLETED0:0.
Final18tray7061097 in36:58. TPSgeomean+1.03183066% vsEarlyfullyoptimized66,
latency−1.02129265%,50/66faster. b2+0.342789%,b3−0.973763%,b4+1.534211%,
b5+0.105416%,b6+2.803663%,b7+2.431688%. All11b6faster,10/11b7faster.
Final strictCSV/summary/metrics+README11batch7table/regressions and final
ParetoPNG/SVG/PDF/frontiers.json LOCAL+REMOTE under task/report. No browser
checks peruser. Source promoted into main after user approval. Q35maxB38case7063495
strictPASS,all64traceslocal,seeCODEX_PROJECT_qwen35_graph_edge.md.
Original132AFDand299Q35decomposition/897samples/1110aliasescomplete; requested
HTMLsready. Goalcompletionaudited againstfinalreports,traceproofandSlurm.

## Cleanup and staging — September 10

User approved current timing, promotion of validated Qwen3/Qwen3.5 scheduling
into main code, and removal of snapshot repos. No commit or push requested.
Main includes graph-event runtime/transport, direct GDN signaling, the K10
barrier-allocation fix, and validated64-bit combine addresses. Removed unused
runtime O-quantization counter allocation; established eager/FFN paths retained.
Cleaned repeated sweep logging, redundant planner tuple fields, and comparison
arithmetic/metric loading; decomposition defaults to one EP/job with2hr limit.

Synced main code to R/local-mirror;17 focused CPU tests passed in0.836s:
event ordering,hybrid direct output,BF16 transport,FFN,padding,eager,transport,
and pool contracts. Five shell syntax checks passed. Cleaned planner exactly
reproduces299 cases/1110 aliases; comparison helpers reproduce66-case overall
and per-batch values. No new GPU run or timing change. No skill/rule change needed.

Removed six remote source repos: megamoe_pareto_perf_20260909,
attn_release_pareto_perf_20260909,attn_publish_entry_20260910,
attn_graph_edge_20260910,qwen35_megamoe_input_sweep_20260909,
qwen35_graph_edge_20260910; also removed five corresponding local overlays.
Other older experiments, all results, traces and HTMLs remain. Compact patches,
previous notes and check logs: scratch/cleanup_20260910 and matching remote
R/artifacts/cleanup_20260910. Historical source paths below are provenance,
not current runnable workspaces. Current code lives in python/minisgl and tests.

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD.
Task artifacts: R/artifacts/attn_graph_edge_pareto_perf_20260910 and local
scratch/attn_graph_edge_pareto_perf_20260910. All requested work is complete.

## Validation and implementation

Alignment7060661 passed A2:F1/EP8/b7,attn-only,normal routing/real prefill:
112 prompts,1904 tokens,100% cached top10,99.89496% top1,full coverage.
Targeted7060293 passed25.307529ms vs26.201700ms prior (+3.53322%TPS);
different nodes/times,not isolated causal ABBA. All32 traces copied. Prior
7059949 failed before kernels from stale single-row rerun_index; index1 and
CPU pool preflight resolved it. Final sweep66 cases uses perf/no-prefill,
perfect balance,MB2,attn-only,FP8xFP4,memory0.97,no pages,explicit overlap.
Jobs7061091–97 for2/3/4/8/12/16/18 trays all completed0:0; no active jobs.

Implementation: capture allocates one internal torch.cuda.Event perFMHA,
enable_timing=False,external=False. Events retained by _FmhaDecodeGraph.
Capture Qwen3attention calls _run_attention_event_body: MB2 prearmsbothQ-ready
waiters withNOturn, eachFMHA waits on previousFMHA event AFTER Qready; event
recorded directlyafterforward_prepared BEFORE Oview/contiguous/quantpublisher.
Publication remains same-lane nonrelease quantize_publish_o_fp8; next-layer
Qwait armed afterpublication asbefore. No localturnpoll/release forgraphmode.
Events keeponegraph andtwoexplicitstreams; GPU replay MAY materialize more
executionstreams perhandoff. Existingcode records that observed onQwen35model;
user specifically concerned. Need inspectactualNsightstreamcount/kernel gaps.
Remote Q/O readiness andslotownership preserved. Hybrid attention now uses the same event path; established eager paths remain. Existing device-turn paths remain available outside the event graph.


## Delivered artifacts and trace findings

Final HTML delivery: added completed66-point graph-edge sweep as eighth curve
FastAFD_MegaMoE_Attn_Graph_Edges_W4 to local scratch/visualizations/
afd_128k_optimized_attention/fastafd-128k-five-pareto.html.17independentPareto
points, all sevenpriorseries preserved exactly; source finalreport/sweep.csv,
actualA:F ratios retained includingA56:F16=3.5. Existingaxes containnewpoints.
Backup fastafd-128k-five-pareto.before-graph-edges.html. Embeddeddata-only
source/preservation check passed; no browsercheck peruser. OtherHTMLsuntouched.

User-requested replacement: afd-trtllm-perfect-balance-128k-pareto.html now
uses the same finalgraph-edge66points/17frontierpoints for existingafd_optimized
series. Label/key/style preserved;sourceupdatedtofinalsweepCSV;TRTLLM+AFD
series unchanged. Backup adjacent *.before-graph-edges.html. Embeddeddata
match/preservation check only; no browsercheck.

Requested maxTPS/GPU trace copy complete: A8:F1EP8b7 (A64:F8),
i131072-fep8-r8-atp1-b7-fp4/job7061097.240.1589188TPS/GPU,25.9087701ms,
15samples0outliers; selected as maximum acrossfinal66CSV. All72rankreports
copied to local scratch/attn_graph_edge_pareto_perf_20260910/results/CASE/
experiment/ray_logs;66,259,341bytes,allmanifestsizechecksPASS. Taskroot
max-tps-gpu-trace-copy-manifest.json pluscaseREADME/result preserved.

Max-TPS case trace analysis requested by user (no source optimization change):
Added reusable codex_scripts/analyze_afd_attention_gap.py. CPU Nsight export
attention rank1 + model rank65, same representatives as metric; 16graphlaunches
each, all15 measured spans reproduce metric exactly. Analysis local+remote
task/analysis/max_tps_gpu/{README.md,summary.json,per_step.csv,
attention_transitions.csv,boundary_envelope.csv,producer_envelope.csv};SQLite
also local+remote. Mean25.908770ms =24.798884FMHAsum +0.541869initialwait
+0.560177netinterFMHAgaps +0.007840tail. Steps3–16 initialwait0.460352;
firstmeasuredstep2 initialwait1.683104, all15 retained. Inputembedding7.443us,
initialmodelQpath46.908us; LMhead208.433us, finalMoE→sampletail254.921us.
Outputtail belongs largely to NEXT attentiongraph initialwait; not additive.
Batch3FMHA115.945us vsbatch4 147.872us; same-MB returnpath toQready~122us.
3→4 gap6.429us, Qwaitlate1395/1410(allnonbootstrap), ready→FMHA0.225us;
4→3 Qreadyearly1395/1395. No attentionrelease/waitturnkernels. Thus input
readiness, not wrongquant/eventdependency;999/1410nextbatch4startsbefore
previousOpublicationends. FFN MegaMoE usuallyalreadydone: conservative
cross-trace handshake alignment ±2usmargin gives completion12.7–39.2us
before priorattnend onaverage,1394/1395certainlybefore. QKVpublisher14.508us.
Cross-host timestamps NOT directlysubtracted; causal26.432usenvelope and
limitations documented. Negative~0.48usFMHAboundary timestamps: use netgaps
for sum identity;positivegaps0.606739ms,FMHAunion24.752322ms. No claimall72
ranks/nohostrequestlatency/noGPUrerun. Mainmerge stilldeferred.

Follow-up initialwait GPU/stall attribution: corrected earlier vague
"coordination" attribution. For typicalsteps3–16 initialwait460.352us,
modelGPU non-waitkernels/copies362.765–363.750us; gaps96.602–97.587us
(~79%GPUwork/21%gaps, causalclockenvelope retained). Betweenmodelgraphs
122.034us =7setupkernels14.153us +17copies31.705us, union45.303us,
no-recorded-GPU-work76.731us. Copies6384B total:1D2Htoken224B,8H2D,8D2D.
Mappedpositions/requestindices,token/KVgathers,sampling/writebackmetadata,
validitymaps,graphbuffercopies,4paddingfills. NEXTcudaGraphLaunchalready
returned>=43.240ms beforepreviousGPUgraphend forALL14intervals: notCPU
launch/RPCstall. Small-operation/devicequeue/event gaps;cannotattributeeach
toexacthardwaremechanism. Extendedsameanalysis script+README, newCSV
boundary_gpu_work/initial_wait_gpu_work andstep3_boundary_events.json.
No sourceoptimization orGPUrerun.

Sept10 TRT CUDA-event scope inspection (local sibling source, read-only):
accepted31-point Qwen3 sweep gpuForwardTimeMS events wrap PyExecutor
_forward_step, including inputprep, embed/layers/finalnorm/LMhead, stopping
before _sample_async. AFD graph includes argmax/GPUtokenwriteback; prep/D2H
outside modelgraph but may expose time in next attention Q-ready wait.
Broad corecompute alignment, unequal step boundaries; do not subtract whole
initialwait (previous LMhead there is legitimate modelcompute). TRT timing
is rank0 replicated into ADP rows, NOT independent all-rank max; AFD maxof2
role representatives. Evidence: analysis/max_tps_gpu/trtllm_timing_scope.md.
No metric/source changes or runs.

Sampling exclusion follow-up: existing model SQLite all15 measured graphs
have exactly4 post-LMhead kernels: argmax19.741867us, int32cast1.427200us,
writebackgather2.208000us, scatter3.061333us; sum26.438400us. LMheadend
to modelgraphend35.936000us including gaps (previousstep tail35.968us).
Accounting-only exclusion from25.908770ms gives25.882332ms/240.404237TPS
(kernels,+0.10215%) or25.872834ms/240.492487TPS (elapsed,+0.13889%).
Not a runtime speedup or exact forward-event remeasurement. Merely trimming
current modelgraph at logits changes max-role metric byZERO: attentiongraph
dominates and previoussampling remains in its initialwait. Need matched
forward endpoints to actually exclude sampling from published timing.
Saved analysis/max_tps_gpu/sampling_exclusion.{csv,json}; unchanged curves,
runtime, originalmetrics; no GPUrun.
