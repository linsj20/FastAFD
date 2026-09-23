# MegaMoE optimization

Consolidated September 21, 2026. This is the current kernel optimization memory:
implementation, accepted measurements, rejected experiments, measurement contracts,
and remaining ideas. Current work is tracked first; historical queued/running
checkpoints do not authorize new work. The completed cleanup validation below covers
the requested source, documentation and workflow scope.

September22 sweep checkpoint: ordinary-only planning/preparation/collection for
all-push, pull and original TRT-LLM; host row-limit extension; fixed-token plots
and throughput TODO. Completed remote validation covers339 cases/339,000 ordinary
rank0 calls and32 EP64 exact-combine checks across all-push/pull. All three plots
passed data/anchor and visual checks. Generated evidence stays in ignored scratch;
tracked sources/docs are synced to HSG. No new skill or AGENTS guidance is needed.

## TODO — fix all_push kernel throughput at high total tokens

September22 ordinary sweeps show all-push slower than metadata-push/payload-pull/
combine-push at every tested EP for Qwen3 totals≥1K and Qwen3.5 totals≥4K.
At8K, latency ratios are2.00–3.69× (Qwen3) and1.33–2.61× (Qwen3.5).
Investigate the32-row tile cap and single expert wave versus pull's192-row tiles
and adaptive expert grouping at8K; these are hypotheses, not isolated causes.
Confirm with same-allocation comparisons and preserve small-decode gains/exactness.
Evidence: `scratch/megamoe/{ordinary_sweep_8k_20260922,pull_original_sweep_8k_20260922}`;
current comparisons use1000-call unfiltered rank0 medians from separate allocations.

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
|2|7363782|26|11m40s|
|4|7363783|48|25m27s|
|8|7363784|44|23m02s|
|16|7363785|40|21m16s|
|32|7363786|36|21m09s|
|64|7363787|32|21m57s|

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
|2|7361595|13|5m13s|
|4|7361596|24|10m24s|
|8|7361597|22|9m29s|
|16|7361598|20|9m22s|
|32|7361599|18|10m45s|
|64|7361601|16|12m07s|

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
