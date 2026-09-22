# FastAFD project memory

FastAFD contains the MiniSGL attention/FFN disaggregation runtime and local
DeepGEMM kernels. The user-supplied global instructions apply. Root AGENTS.md is
absent; its creation/location has been asked about. Do not create it without the
user's answer. Read the relevant memory and `codex_scripts/README.md` before work.

## Current MegaMoE state — September 21, 2026

[The consolidated MegaMoE optimization memory](CODEX_PROJECT_megamoe_optimization.md)
is authoritative for kernel design, measured results, rejected approaches, sweeps,
measurement limits and future ideas. The completed selection is u32 normalization
plus peer-mask all-push dispatch and the original scheduler. Pull remains default.
Latest kernel gates passed; later dispatch gains do not establish a new model gain.

The last model-tested RNE peer-ready source improves Qwen3 128K decode graph latency
by0.121037ms (0.668%); its 8K alignment has1/1632 top10 misses, accepted under the
explicit maximum-four-miss criterion. Current joined-graph decomposition covers
164 Qwen3 +300 Qwen3.5 cases; all13,920 rank0 ordinary calls and1,392 profiles pass.
The deep cleanup is complete. Qwen3/Qwen3.5 EP8/EP16 exactness, replay and kernel
decomposition pass (GPU 7338830/7338944, CPU 7339425/7339424): 192,000 ordinary
rank samples and 576 profiles, all retained. Pooled ordinary latency changes range
from -0.039% to +0.084%; performance is effectively maintained. Main-kernel machine
code changes only in a timeout assertion's source-line immediate. See the optimization
memory for the four-case tables and validation limits. The September 21 progress
commit checkpoints the optimization, integration, reusable workflows and cleanup;
the separate AGA bring-up note remains outside its scope.

Twelve optimization notes and the kernel-only history from the mixed Qwen3 FFN
memory are consolidated; 27 obsolete scripts and the two new dispatch-overlap/
profiling-fidelity reports are removed (15 removals were untracked). The existing
all-push report remains. Original dirty/untracked contents are archived under
`scratch/megamoe/deep_cleanup_20260921/`; the manifest records scope and checks.

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
- Final kernel snapshot: `R/source/dispatch_overlap_u32_final`. Last model-tested
  snapshot: `R/source/all_push_peer_ready_rne_20260920`. Do not confuse them with
  earlier prototypes or edit source/cache while a job consumes it.
- Prepare/analyze on CPU allocations; GPUs perform complete workloads. A complete
  kernel or producer-consumer benchmark qualifies, with model E2E for promising
  candidates. No tiny GPU debugging jobs, manual waiting, padding or auto-retries.
- Preserve frozen controls and unrelated staged/dirty changes. No subagents unless
  requested, no automatic fallback, py_check, manual hash/version checks or /tmp.
- Keep project memories concise and record meaningful decisions/results in the
  relevant topic. Historical plans use frozen workflow copies; recreate current
  plans with maintained tools rather than rewriting completed evidence.
