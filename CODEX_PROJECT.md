# FastAFD project memory

FastAFD contains the MiniSGL attention/FFN disaggregation runtime and local
DeepGEMM kernels. The user-supplied global instructions apply. Root AGENTS.md is
absent; its creation/location has been asked about. Do not create it without the
user's answer. Read the relevant memory and `codex_scripts/README.md` before work.

## Current MegaMoE state — September22, 2026

Completed: real-model A17:F1 EP8 batches4–7 on8racks, one FFN/rack,
with Nsight verification. Opt-in QKV LL128 (no receiver ACK),144 active GPUs on40
four-GPU trays. Exact integrated transport GPU40681 and CPU40691 placement/arena
alignment passed; GPU40692 completed all4cases in14m56s. CPU40693 analyzedB4–6;
40695 completedB7 after allowing <=100ns clock conversion disagreement (observed25ns).
Publisher medians2.66–3.55µs across8FFNs; same-host sender-start→receiver-end medians
4.946/6.516/6.096/7.741µs forB4–7, including unpack,2820 samples/case. Thus4–6µs
completion target is not met across all batches. No gradual comm slowdown observed.
Fullgraph B5–7 passed stability (21.041/21.807/26.267ms means); B4 rejected:
one67.96ms step among14 roughly17–18ms. EP8 batch7 all-push capacity1920.
User then authorized excluding outliers: only B4 step14 dropped (>10% from median).
Filtered graph latency B4–7=17.447/21.041/21.807/26.267ms;
TPS/user=57.32/47.53/45.86/38.07; TPS/144activeGPU=216.53/224.43/259.85/251.69.
Trace retention corrected per user's reminder and Nsight agreement: exactly8reports,
one attention+oneFFN percase under `scratch/afd/cases/qwen3/isl_131072/ep_8/a17-f1/`.
The576-report collection folder and extra payloads were removed; shared results
in `scratch/afd/reports/qwen3/aga-a17-lamport-20260922/`.
See [A17 Lamport run](CODEX_PROJECT_aga_bringup.md#a17-lamport-model-run).
Session staging: all36 selected CPU checks pass after a targeted generic-Ray
environment forwarding fix (CPU40697/40698); scorer CLI and3remote shell syntax
checks pass. Code/docs synced to AGA. Session-only commit excludes unrelated
MegaMoE sweep/compiler/plot edits and all scratch artifacts. Batch7's exposed
publication is Q-readiness imbalance:4-token return125.72µs exceeds intervening
3-token attention116.00µs; both lanes already have early attention graph edges.

Completed AGA rack scaling: user approved512GPUs,A496:F16,batches4–7 after no complete
576GPU domain was available. One FFN per each of16racks,31local attention senders.
Reusable rack mapper and sequence domain guard added; CPU preparation passed,
GPU39892 completed batch4 but failed batch5 control-port allocation. GPU39894
completed5–7 using explicit low control ports. CPU39896 verified all8Nsight traces
and4rack maps; all4timings rejected by unchanged10%stability gate. Alljobs terminal.
User then authorized downloading the8traces: retained locally with provenance.
Timings show an early fast stretch then a step-up around decode5–8; late-window
means30–47%higher, with partial B7recovery. Plot/trace links in the rack-scaling doc.
Follow-up CPU/GPU trace correlation identifies coordinator throughput/serial
dispatch as the main long-wait cause: B6late dispatch41.36ms versusFFNgraph31.88ms;
launch arrives8.15ms after previousFFNend (5.43ms before worker prep+2.72ms prep).
Initial command lead drains; all496attention commands precede16FFNcommands.
No evidence of broken ready tickets; first-MegaMoE EP skew remains partly open.
Reusable `analyze_afd_cpu_dispatch.py`; this diagnosis used CPU jobs.
Subsequent one-rack QKV resource study and fix completed: one reusable
`bench_qkv_fanout.py` covers CPU build, resource sweeps and model comparison.
Direct per-destination CTAs reduce A31 publication36/60/60→17.5/17.5/18.9µs
for2/3/4rows, with unchanged payload/readiness semantics. Registers52→32;
34,560checked publications passed across A3/A7/A15/A31. Default optimization
applies to grouped Qwen3 small-row shapes with≥7destinations; A3 keeps original.
Full-model A7:F1EP4B4 comparison40561/40562 passed15/15samples in both runs:
QKV12.947→10.453µs,graph21.333→19.895ms (one control/candidate pair,6.74%lower).
The entire graph gain is not isolated to QKV; coordinator scaling remains open.
Code/docs synced to AGA only; no bulk scratch transfer or512GPU rerun.
The subsequent Lamport LL128 study prioritized preservingbest
latency while minimizingparticipatingSMs, fullyusingeachSM's resources; stopbroad
parametersearches. Pollingcontention/prepackingwere ruledout asdominantcosts.
FixedgeometryLL128 measured4.15/4.60/5.59µs (2/3/4rows), but matchingevent-control
runfailed onnewmeasurementprobe's remote-device streamselection; corrected.
40650/40651/40652 confirmedCUDAevents inflatekernel timings; validLL128
reference4.18/4.64/5.68µs. Initial16/8/4SMpools aretoo slow.40658/40659/40660
and40665/40666/40667 passed aligned packing and balanced-row comparisons;
16SM push still slower than reference, so no reduced-SM winner selected.
Final unprofiled timing40677/40678 then warmed-graph40679 passed all checks.
Publisher-clock start before push/end after all receiver ACKs; parallel ACK
observer replaces serial scanning. Pooled warmed completion bounds9.44/10.18/11.33µs
(2/3/4rows),16SM9.47/10.50/11.81. Sender-only4–6µs does not establish goal.
Warming the captured graph removes first-replay0.3–0.4ms event discrepancy;
outer event/device spans differ only12–28µs over8–13ms. All jobs terminal.
Default production transport remains the measured direct-edge version. The LL128
path is now opt-in in the real model, validated by the A17 run above; other
protocol/resource variants remain benchmark-only.
See [rack scaling](CODEX_PROJECT_aga_bringup.md#rack-scaling-and-communication).

AGA promotion complete: checkpoint `2be271e` plus scorer import fix; code/docs only,
no scratch transfer. One-rack Qwen3 A3:F1 EP8 batch4 perf128K:17.848781ms,
168.078709tokens/s/GPU,15/15samples. Actual8K alignment:1629/1632top10,3misses
passes agreed≤4 gate,strict zero-miss fails. [AGA results](CODEX_PROJECT_aga_bringup.md#qwen3-promotion)
record placement,jobs,full-score audit and the required `--propagate=NONE` setting.

Selected target: **Qwen3 A3:F1 EP8 batch4, early PDL + ahead8 vector reset +
normalization weight preload**. Pull remains default; reproduction settings and
invariants are in [MegaMoE optimization](CODEX_PROJECT_megamoe_optimization.md).

- Latest matched run7356961/analysis7356986: pull18.144077ms, preload-off17.891785ms,
  preload-on17.842612ms. Preload saves0.049172ms in both orders; total0.301464ms
  (1.6615%). All90 ordinary perf-mode samples retained,128K synthetic KV/perfect routing.
- Earlier allocation7346203: preload-off saves0.358198ms (1.973%), meeting its
  corresponding ordinary-main projection0.352590ms. Do not add cross-allocation gains.
  Pull's main improved more between allocations than selected (1.304 vs0.787us),
  narrowing the advantage; hardware cause and exact critical-path attribution unproven.
- Combined actual8K/real-routing alignment passes1/1632 top10 misses, within the
  agreed maximum4, with exact full-prompt/continuation cached-score coverage.
- Current attention/FFN trace pair, metrics and manifest:
  `scratch/afd/cases/qwen3/isl_131072/ep_8/a3-f1/batch_4/megamoe-reset-pdl-preload-20260922-j7356961/`.
  Evidence: `scratch/megamoe/reset_overlap_20260921/combined_selected_result.json`.
- Optional TMA reset is validated/default0, but its0.002159ms model increment is
  not material; vector remains selected. Rejected placements and sync variants
  are recorded in the topic memory, with isolated source and raw evidence retained.

User requested a cleanup/progress commit. Python readability and documentation
cleanup preserve measured CUDA paths. Remote CPU7357776 passed1m02s:5 protocol
tests,96 diagnostic records,90-sample comparison and full1,632-token alignment
re-audit (same1 top10 miss). The AGA note and EP-scaling plot edit are unrelated
and remain untouched.
No new global skill/AGENTS guidance is needed.

Previous September21 checkpoint consolidated kernel history/workflows and passed
Qwen3/Qwen3.5 EP8/EP16 exactness, replay and decomposition. Current sweep coverage
is164 Qwen3 +300 Qwen3.5 cases. The optimization memory is authoritative for those
results, historical model checkpoints and measurement limits.

## Topic index

| Work | Memory |
|---|---|
| MegaMoE kernel optimization and decomposition | [optimization](CODEX_PROJECT_megamoe_optimization.md) |
| Qwen3.5 model support | [support](CODEX_PROJECT_qwen35_support.md) |
| Qwen3.5 attention/GDN transport | [graph edge](CODEX_PROJECT_qwen35_graph_edge.md), [full attention](CODEX_PROJECT_qwen35_full_attention.md) |
| Qwen3 FFN/attention integration and earlier evidence | [handoff](CODEX_PROJECT_qwen3_ffn_handoff.md), [overheads](CODEX_PROJECT_qwen3_ffn_overheads.md) |
| Attention scheduling | [graph edge](CODEX_PROJECT_attn_graph_edge.md), [release performance](CODEX_PROJECT_attn_release_perf.md) |
| Reproduction and ratio/EP sweeps | [reproduce](CODEX_PROJECT_REPRODUCE.md), [ratio sweep](CODEX_PROJECT_RATIO_EP_SWEEP.md) |
| Cluster operations and AGA bring-up | [clusters](CODEX_CLUSTERS.md), [AGA](CODEX_PROJECT_aga_bringup.md) |
| Trace storage and interpretation | [Nsight contract](CODEX_PROJECT_nsys_traces.md) |
| Performance presentation | [Pareto](CODEX_PROJECT_pareto_perf.md), [publisher entry](CODEX_PROJECT_publish_entry_perf.md) |
| Prior runtime cleanup | [cleanup](CODEX_PROJECT_cleanup.md) |
| Reusable commands | [workflow inventory](codex_scripts/README.md) |

Model integration notes retain historical kernel evidence; the consolidated
optimization file supersedes their older MegaMoE selection/pending checkpoints.
Full-model alignment and fixed-input kernel correctness remain separate gates.

## Execution and continuity

- Local checkout is authoritative. Use plain `ssh oci-hsg` for HSG execution.
  Root R is `/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD`; frozen sources
  belong under `source/`, plans/logs/results under `artifacts/<campaign>/`.
- Current model-tested source: `R/source/local-mirror`. Historical frozen sources
  include `dispatch_overlap_u32_final` and `all_push_peer_ready_rne_20260920`.
  Preserve archived controls; do not edit source/cache while a job consumes it.
- Prepare/analyze on CPU allocations; GPUs perform complete workloads. A complete
  kernel or producer-consumer benchmark qualifies, with model E2E for promising
  candidates. No tiny GPU debugging jobs, manual waiting, padding or auto-retries.
- Preserve frozen controls and unrelated staged/dirty changes. No subagents unless
  requested, no automatic fallback, py_check, manual hash/version checks or /tmp.
- Keep project memories concise and record meaningful decisions/results in the
  relevant topic. Historical plans use frozen workflow copies; recreate current
  plans with maintained tools rather than rewriting completed evidence.
