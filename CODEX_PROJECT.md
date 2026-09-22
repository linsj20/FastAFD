# FastAFD project memory

FastAFD contains the MiniSGL attention/FFN disaggregation runtime and local
DeepGEMM kernels. The user-supplied global instructions apply. Root AGENTS.md is
absent; its creation/location has been asked about. Do not create it without the
user's answer. Read the relevant memory and `codex_scripts/README.md` before work.

## Current MegaMoE state — September22, 2026

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
