# Qwen3.5 attention graph-edge perf capture — COMPLETE Sept10

FINAL14:29PDT:7063495 COMPLETED0:0 in9:44; strict1/1PASS,15samples0outliers.
B38/128K,A48:F16,EP16,attn-only,perf/no-prefill/perfect1,MB2=19+19,FP8xFP4,
memory0.82,nopages; observedmax38PASS (minKV5013568,admission4983676).
31.0278494667ms,32.22911085TPS/user,918.5296593TPS/GPU. No exactoldQ35B38
baseline, so no Q35speedupclaim. All64runtime logs validate modes; attention
48/48graph_edges,model16/16device_turn+MegaMoEFP4perfect1. All64NSYSreports
local,79,806,982bytes,manifestsizechecksPASS. README.md+strictreportlocal/remote.
CPUrepresentativetrace exports: attentionrank1 has no release_turn_kernel,
uses signal_direct_o_kernel<bool0>,fullFMHA+GDN; graphId2 uses5executionstreams
(twoexplicitlanestreams is NOT twoexecutionstreams). FFNrank49 uses2streams,
sm100_fp8_fp4_mega_moe_impl(E512,K10),1920calls acrosscapture. PreservedFFN
finalization still has48release_turn_kernel calls in16capturedreplays. This
is existingFFNpath, not an attentionrelease. validation/runtime_modes.json and
kernel_presence.jsonlocal/remote. No GPUexportjob or detailedmicrodiagnosis.
No newoutputalignment/nozero-latencyclaim. Promoted into main code after user approval.
Qwen3full66strictPASS/all7terminal0:0, finalcomparisonready. Allgoalworkcomplete.

## Promotion and cleanup

Validated source is now in main runtime, transport Python/CUDA and scheduler
tests; K10 barrier allocation is merged too. Snapshot repo and local overlay
removed, with source patches retained in scratch/cleanup_20260910. Shared
cleanup/check status: CODEX_PROJECT_attn_graph_edge.md. Results and64 traces
remain in scratch/qwen35_a3_ep16_perf_20260910 and R/artifacts/same-task.
R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD.
Historical snapshot path R/source/qwen35_graph_edge_20260910 is no longer live.

## Material attempt history

7062744 cancelled pending to apply hybrid optimization. 7062904/7063076 failed
before kernels due missing default FP4/base checkpoints; configure both model
and expert paths. Dry-run alone missed the later in-container base-path check.
Both94-shard indices,config/tokenizer and required paths were then checked.
7063234 reached capacity validation but max-only gate rejected B37: actual max38.
Pending7063444 cancelled after user explicitly required B38; pool migrated to
B38/require_capacity_max1 and final7063495 passed. No machine fault evidence.
All submission/env/failure records remain in task artifacts.

Base and expert checkpoints are under R/artifacts/qwen35_afd_overheads_20260904/
models: base huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/
ea5b4f81096f3901c91dea97f81324302495781d; experts
Qwen3.5-397B-A17B-FP8-MXFP4-MoE/snapshots/same-id.

Initial focused CPU4/4 passed; promoted combined suite17/17 passed. No new GPU
rerun or output alignment. No matched oldB38 baseline, so no Qwen3.5 speedup
claim. Attention uses5 execution streams despite two explicit lane streams;
FFN uses2. Existing hybrid FFN event alternative had226 streams, so preserve
the accepted fused FFN release/shared-expert path.

Implementation:
- Internalattentionevents nowcaptured forQwen3.5 too, retainedpergraphlifetime.
- Q-ready prearm staysindependent oflocalcompute turn; everyattentionroundwaits
  onpredecessorcompletionevent. Fullattentionrecordsrightafterforward_prepared,
  beforecontiguous/gatedBF16publication. Linear/GDNrecordsrightafterforward_linear.
- PreserveGDN direct symmetricO output; eventpath stillpassesdirectdestination
  toproducer, then new signal_direct_o publishesremotereadiness only. No
  payloadrecopy,localturnpoll/store or standaloneattentionrelease kernel.
- CUDA signal_direct_o_kernel<false> retains __threadfence_system +ready release;
  <true> retainsoldturnvariant foraccepted eager/device-turn paths. Python+FFI
  addedsignal_direct_o method. This readykernel isnecessaryremoteordering,
  separatefromlocalgraphcomputehand-off.
- Qwen3.5 FFN alreadyreleasesatbeginningfusedQKV/linear-inputpublisher via
  release_peer_lane (block0/thread0); preservedthisandsharedexpert/MegaMoE
  overlap. PriorQ35FFNevents produced226executionstreams vs2accepted; donot
  regressitbyblindlyconvertingFFN. Userexplicitlytoldthischoice/evidence.
