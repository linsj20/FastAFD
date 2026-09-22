# Qwen3.5 full-attention-only workers — accepted September 11, 2026

## Active follow-up: device-signaled GDN, September11

September11 side request: refreshed the completed A32:F32/B40 device-GDN trace
pair from7083416 into the existing case folder. Attentionrank1=552358B,FFNrank33=
2113081B; both nonempty, sizes added to manifest. Event-GDN pair already local.
No new full-model run or extra-rank trace retrieval.

7083416 COMPLETED0:0 in13:59. A1/EP32/B40 paired means event22.372316867ms →
device22.636225267ms (+1.1796%);15samples/0outliers each. TPS/GPU893.961949→
883.539537; TPS/user44.698097→44.176977. Device signaling has no latency win in
this pair. Both metric-selected attentionrank1/FFNrank33 traces plus compact
metrics/manifests copied to scratch/afd/cases/qwen35/isl_131072/ep_32/a1-f1/batch_40/
{event-gdn-push-20260911,device-gdn-push-20260911}. CPU-only extraction7085227
submitted cpu-normal/5min for FFN stream count verification; pending. Device
graph has2924nodes versus2564events source, from added wait/release kernels.
7085227 COMPLETED0:0 in1min. All16graph replays confirm92→2 FFN execution streams.
CPU-only SQLite stays remote; compact stream_summary.json copied under local
scratch/afd/validation/qwen35_gdn_device_signals_20260911/a1_ep32_b40.
Same exports resolve the earlier GDN timing question at actual batch20:
_qwen35_gdn_decode_kernel grid(4,20,64),1350retained calls, event median27.136us,
device27.200us (versus~85.536us at oldbatch60). Conv remains separate. No per-GDN
math slowdown from device dependencies; full-model~1.18% cost includes extra
wait/release launches and changed scheduling, not isolated causal attribution.

LATEST user steering: cancel A3/EP16 and run A1:F1/EP32/B40, still perf mode and
Nsight traces.7083154 CANCELLED before start (0GPU time). Replacement7083416
submitted short/64GPUs/16trays/30min. Task subdirectory=a1_ep32_b40, with separate
plans, pools, logs and job.json. Same source pair, BOTH dispatch1,128K,MB2=20+20,
A32:F32/TP1/global1280 prompts. Each FFN GDN now has20rows. Fixedbatch40;
memory.82/nominalKV5260736, regular admission checks retained, max-batch assertion0.
Dry run confirms shape/perf1/perfect1/NSYS1/15steps. Preparation failed twice on
CPU before submission: pool requires canonical status string; default pmap only
allowsEP4/8/16. Retained canonical pool status and selected existing comprehensive
contract forEP32; no launcher/kernel change needed. Check7083036 remains applicable.

User requested implementing device dependencies, rerunning perf mode and copying
back representative traces. Candidate=R/source/qwen35_gdn_device_signals_20260911;
task=R/artifacts/qwen35_gdn_device_signals_20260911. GdnProjectionSync reuses two
ready/done cells per owning lane, using existing acquire/release device kernels
with consumer resets and existing timeout reporting. Publish after normalization;
peer waits before BA/gates, publishes after gates; owner waits after convolution
before recurrence. No per-GDN graph event edges; graph-boundary joins remain.
Ordinary non-AFD event path preserved. Native transport/math unchanged.
Remote runtime26/26 and overlap26/26 CPU tests pass. First overlap run exposed two
stale preexisting source assertions (push template and refactored normalization
location); corrected their expected locations, then passed.
Small CUDA check7083036 submitted short/4GPUs/15min, initially priority-pending;
rows19/60, exact outputs/conv/recurrent state, six chained calls across both owners,
eight graph replays plus event/device ABBA. Prepared same64GPU A3/EP16/B40/128K
pair in task/plans, not submitted until numeric check passes. Control=accepted
accumulated source, candidate=device GDN; BOTH push dispatch1, same capacity and
strict metric gates. Reuse accepted native cache (no native source edits).
Keep two reports per retained variant; don't claim stream count until measured.
7083036 COMPLETED0:0: both19/60 rows exact output/state and8 bidirectional graph
replays pass. Component event/device ABBA ms: rows19 .074693/.076003/.074763/.074225;
rows60 .137730/.139031/.138778/.136096. Device signals add roughly1–2us here;
full-model impact unmeasured. This check used the old wrapper's default home
DeepGEMM cache; fixed run_cuda_check.sbatch defaults to OUTPUT_ROOT for subsequent
checks (bash -n passes), retaining explicit cache overrides.
Perf7083154 submitted short/64GPUs/1hr, initially resource-pending. Both native
cache paths reuse accepted dispatch binaries; task/job.json records exact command.
Pending reservation shortened30min to improve backfill (previous pair13:30).
September11 GDN latency question resolved without new GPU work: user's FlashInfer
`scratch/visualizations/attention-token-latency/editable/qwen35-122b-linear-attention-hot-l2.html`
times conv+packed recurrence, not recurrence alone; 1K ISL B48=71.200us,
B64=91.904us. Current B40/MB2/A3 fan-in gives60 rows per FFN GDN, not20.
Nsight candidate grid=(4,60,64),256threads,40regs;1350 retained GDN calls median
85.536us (min84.512,max89.184), conv median6.816us, gate median2.240us (overlaps).
Both122B chart and397B model have16key/64value heads,128dims,FP32 recurrent state;
each row is4MiB, so60rows=240MiB. Hot-L2 means no explicit flush, not guaranteed
whole-state L2 residency. Different kernels/timing scope prevent an exact matched
speed claim, but batch mismatch explains the apparent30–35vs85us discrepancy.
Evidence=old task/b40/stream_analysis/gdn_latency_summary.json, chart/profiles and
FlashInfer CODEX_PROJECT_MLA_LINEAR_HSG_SWEEP.md. Perf pair remains unchanged.

Goal complete: attention workers execute only full attention; FFN workers own
GDN/linear attention and all intervening model work. General scheduling and
communication optimizations pass A3:F1/EP16 maximum-batch performance and final
normal-routing alignment. No task GPU jobs remain. Commit/push not requested.
User constraints: concise logs, accumulate changes per expensive64-GPU run,
monitor every5min, no subagents, py_check, source-hash/version audits, or /tmp.
Root AGENTS.md is absent; its creation/location question was asked earlier and
left unanswered. User-supplied global instructions apply.

## Source and execution

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Remote host=oci-hsg; task=R/artifacts/qwen35_full_attention_20260910.
Local worktree authoritative; cleaned27 source/workflow/memory files synchronized
to R/local-mirror. Manifest=validation/current_source_files.txt. Frozen math/runtime:
- Old placement: R/source/qwen35_full_attention_control_20260910.
- First new placement: R/source/qwen35_full_attention_candidate_20260910.
- Accepted accumulated: R/source/qwen35_full_attention_accumulated_20260911.
Archives/overlays are authoritative; inherited remote Git metadata is stale.
Do not reset or audit source versions. Common mirror launchers supply archive-aware
execution; later launcher hooks do not change the frozen candidate's math/runtime.

## Implemented behavior

September11 follow-up: user observed fragmented FFN streams in the delivered B40
trace. CPU-only exports of the existing representative rank49 reports confirm
control4 versus accumulated92 execution streams in EVERY one of16 graph replays;
candidate2564 kernels/replay,90 GDN calls and120 MegaMoE calls (MegaMoE spans90
streams). Two explicit Python streams did not preserve two execution streams.
`Qwen3_5LinearAttention._project_inputs` still records/waits start/done CUDA events
at each GDN; caller-supplied peer stream reuse retained those repeated graph
fork/joins. Outer FFN QKV publication/shared-expert handoffs already use device
turns; their release mechanism was not reverted to CUDA events. The GDN event
edges are the identified fragmentation source; restoring two main lanes needs
device-signaled projection readiness/completion while preserving BA/gate overlap,
ordering and buffer reuse. No runtime change or GPU run for this diagnosis;
two-stream correction remains unimplemented. Compact evidence:
validation/b40/stream_analysis/summary.json; full SQLite exports remain remote
under task/b40/stream_analysis. Current accepted performance remains
1119.766491 TPS/GPU and37.325550 TPS/user (1000/26.7913ms), perfect routing,
A48:F16/EP16/B40/128K. Stream count alone does not establish a latency penalty.

- Reuse standard Qwen3.5 model on FFN workers; remove duplicate stage classes and
  GDN transport arenas/publications/direct writes. Attention workers allocate no
  GDN state and load no GDN stage weights; FFN state owns/randomizes the pool.
- Remote rounds, readiness tickets and event grids contain full-attention layer
  IDs only. Each lane executes its local layer interval before publishing the
  next full-attention request; peer lanes overlap remote attention with local
  work. Leading/trailing GDN intervals and irregular layer patterns are supported.
- Shared-expert/MegaMoE overlap extends across local GDN layers; completion is
  joined before its output is consumed. GDN projections reuse the peer lane's
  stream. Preserve existing arithmetic and BF16 rounding boundaries.
- Request namespaces and GDN slots cover assigned attention fan-in lanes; preserve
  state_indices through eager merged batches, decode subbatches and graph replay.
- Full push dispatch sends FP8 activations, packed UE8M0 scales and selected FP32
  route weights. Publish routes before fanout, deduplicate rows per destination,
  poll every32-bit payload word, clear actual rows after all dispatch readers
  finish while GEMM continues. Two rotating slots use prior route publications
  to guarantee safe reuse, without a new payload-completion flag. Exact reserved
  negative-zero words are canonicalized only in activation/weight payloads;
  packed scales remain exact. Details/proofs/trials: CODEX_PROJECT_megamoe_optimization.md.
- Combine retains its existing producer-push BF16 contributions, completion epoch
  and local TMA reduction. In-band combine variants increased actual latency and
  were removed from active code; rejected implementations remain archived.
- Fixed fused quantization coverage when fewer than16 expert blocks launch;
  fold the two per-warp owners modulo available blocks, preserving original
  ownership for128+ experts. Exact FP8/SF coverage passes8/16/24/64/128/512experts.

Reported and aligned configuration uses DG_MEGAMOE_LAMPORT_DISPATCH=1, fixed for
allocation lifetime; generic API default0 retains TMA for other workloads.
Keep this explicit selectable transport rather than extrapolating one target's
performance to all shapes. Push requires fused quantization, rank-ready routes,
rank-gated combine, H divisible by1024 and topk<=32; unsupported use fails fast.
No model/shape-specific runtime performance heuristic was added.

## Performance and maximum batch

Both comparisons use A48:F16/EP16/TP1/128K/MB2, memory ratio0.82, denseFP8/MXFP4
experts, perfect routing, and the unchanged maximum representative-role CUDA
graph-span metric (including sampling/writeback). Do not add kernel-family sums
or multiply gains from different batches/allocations.

| Pair | Batch | Control ms | Candidate ms | Latency change | Job |
| --- | ---: | ---: | ---: | ---: | --- |
| Old placement -> first full-attention placement | 38 | 30.916114643 | 27.995547533 | -9.45% | 7073181 |
| First placement -> accumulated scheduling/push | 40 | 28.432392933 | 26.791300000 | -5.771913% | 7077121 |

7073181 COMPLETED0:0 in14:19 on nvl72055-T[02-17]. TPS/GPU921.849344 ->1018.019025.
Control14retained/1excluded (step2=43.5576ms), candidate15/0; policy unchanged.
Attention graph nodes1170->90, FFN1606->2654. First-placement trace median kernel
sums: full attention20.121ms, GDN7.304ms, MegaMoE7.877ms, denseFP8 GEMMs5.021ms,
selector1.225ms. These overlap and cannot be summed into step latency.
CPU-only trace extraction7074753 completed0:0 in29s; cpu-short had rejected two
submissions due unrelated quota occupancy, so cpu-normal was used, without GPUs.

7077121 COMPLETED0:0 in13:30,10:59:24–11:12:54UTC, nvl72170-T[01-04,06-16,18].
TPS/GPU1055.134545669 ->1119.766491361 (+6.125470%). Both15retained/0outliers;
unchanged10% stability limits pass: dominant ranges8.165599%/9.591452%,
max-median differences8.069155%/9.521382%. No samples silently discarded.
BOTH sources observed82199pages=5260736 tokens per attention rank; admission
requires5245978 tokens atB40; exact/raw maximum40 is confirmed. Old placement's
78337pages=5013568 tokens admittedB38. B40 gain is for the accumulated candidate,
not isolated attribution to push dispatch. Compact comparison=b40/summary.json.

## Final normal-routing alignment

Generation7078525 COMPLETED0:0 in21:51,11:15:23–11:37:14UTC on
nvl72145-T[04-09,12,18],32GPUs. A24:F8/EP8/TP1/B7/128K/MB2(3+4), real prefill,
normal routing,NSYS0, accepted accumulated source, dispatch1. Produced168prompts
and2856tokens (17each); actual result confirms perf_mode0 and perfect_routing0.

Reference7078816 COMPLETED0:0 in39:22,11:40:20–12:19:42UTC,4GPUs on nvl72058-T10.
Unchanged sharded-vLLM DP4/EP4/TP1, DeepEP low latency, DeepGEMM, no prefix cache,
language-model-only, maxseq4, concurrency4, pinned reference venv artifact5179989.
Reference source remains the prior clean qwen35_accum_direct_cta_selector_fix
snapshot; it is independent of the AFD candidate being scored.
- Top10:2856/2856 (100%, unchanged required gate).
- Top1:2853/2856; three remaining tokens rank2 (sample119/123/143, position0).
- Coverage independently checked: all168 ordered prompts match the generation,
  all2856 generated IDs and positions match, exactly17 steps per prompt, finite
  logprobs, consistent top1 flags, no missing/extra steps. Summary rates agree
  with recomputed ranks. Compact tokens/ranks retained; no raw prompts copied.
Evidence=qwen35_score/alignment_summary.json and qwen35_generation/summary.json.

## Other validation and rejected attempts

- Remote CPU runtime/ownership/overlap suite53/53 in2.276s; focused27/27 in1.804s
  after shared overlap/stream reuse. Tests cover irregular full-layer positions,
  trailing local layers, remote epochs, every MoE invocation and numeric state.
-7073425,4GPUs,2:48: synthetic complete GDN output/conv/recurrent state exact at
  rows19/57/60. Owned/peer stream ABBA ms .074448/.073731, .132873/.132080,
  .137129/.136354 (~0.6–1% faster); component evidence only.
-7073656,4GPUs,1:14: recurrent+norm/FP8 fusion exact but74–89% slower; removed.
-7073783,4GPUs,1:21: shared-add/RMS+FP8 input fusion exact,25% faster atrows19,
  only0.1–0.2% at target57/60; removed. Patches/logs retained. Do not repeat a
  tile/warp sweep or replace FlashInfer post-attention norm without numeric proof
  (prior V6 fusion failed alignment).
- Communication protocol/component attempts and rejected combine variants are
  fully recorded in CODEX_PROJECT_megamoe_optimization.md. Full dispatch was only
  provisional after noisy EP16 microbenchmarks; accumulated model result above
  supplies end-to-end latency evidence. Cleaned native validation7077011 passed
  on4GPUs (3:32): exact normal/negative/zero inputs, empty-source/prefix/interleaved
  padding and repeated graphs. No further algorithm variants required.

## Artifacts and reusable workflows

Local validation root=scratch/afd/validation/qwen35_full_attention_20260910.
Exactly two metric-selected Nsight reports per retained performance version:
- cases/qwen35/isl_131072/ep_16/a3-f1/batch_38/
  full-attention-placement-control-20260910 and full-attention-on-attention-workers-20260910.
- cases/qwen35/isl_131072/ep_16/a3-f1/batch_40/
  full-attention-placement-control-20260911 and
  full-attention-accumulated-push-dispatch-20260911.
Accepted alignment (no traces): cases/qwen35/isl_131072/ep_8/a3-f1/batch_7/
full-attention-accumulated-push-dispatch-alignment-20260911/{alignment,manifest}.json.
All case paths above are under scratch/afd/. Full raw reports/samples remain remote;
do not copy100MB sample logs, prompts, SQLite exports or extra rank traces.

B40 plans=task/b40/plans/{perf_b40.csv,perf_env.json,pair.tsv,control.env,candidate.env}.
Use common run_afd_paired_sources.sbatch with FASTAFD_PAIRED_CONTROL_DIR and
FASTAFD_PAIRED_ENV_DIR; separate variant caches, FASTAFD_VERIFY_SOURCE=0.
Normal generation uses common run_afd.sh qwen35 131072 7 3:1 1 8 and
planned_alignment_env.json; no case pool required. Initial host preparation
failed before submission because cuda_extract/tmp was absent; created it, no
failed GPU allocation. Goal-scoped active-job throttle; known bad nodes excluded.
Scorer=task/jobs/score_vllm_sharded.sbatch, copied from the prior successful wrapper
with only optional source verification (set0). bash -n passed; scorer/math/gate
unchanged. planned_score_env.json points to this generation's actual sample.
Do not use the old generic scorer launcher: its vLLM console entry was missing.

Native caches use ray-node-<IP> tags. Matching binaries were hard-linked under
build locks before compilation:32 actual macro paths and8 alignment paths, no
busy locks. Records=b40/native_cache_seed.json and qwen35_generation/native_cache_seed.json.
Control donor=prior task/cache/candidate/deepgemm/ray-node-*/lib; candidate donor=
dispatch_clean_validation/cache/deepgemm/lib. Proposed-node pre-seeding initially
covered nvl72013, but actual allocation changed; always check actual node tags.
No source hash/version audits, extra GPU tests, or additional64-GPU run required.
