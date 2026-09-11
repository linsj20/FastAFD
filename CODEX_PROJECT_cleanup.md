# Comprehensive reduction — completed September 10, 2026

Goal: reduce changed runtime, tests and general scripts; remove one-shot scripts;
clean local scratch. Preserve standard execution, top-10=100% alignment and the
existing strict maximum representative-role CUDA graph-span timing policy.
User added alignment for BOTH Qwen3 and Qwen3.5: A2:F1/EP8/b7, attn-only.
User requested committing this cleanup together with the already staged code; no push requested. No subagents. Root AGENTS.md is missing; asked user
whether to create it or use another location, no answer yet. Supplied global
instructions apply. Global skill/rules were not changed.

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Candidate=R/local-mirror (local checkout remains authoritative).
Frozen original=R/source/code_reduction_control_20260910, cloned then overlaid
with ALL original tracked worktree files, including previously staged edits.
Artifacts=R/artifacts/code_reduction_20260910. Local artifacts now live at
scratch/afd/validation/cleanup_20260910; remote paths are unchanged.
Original before.patch remains local. Remote control includes before.tar.gz.
Do not overwrite the control. No CUDA kernel arithmetic or tuning changed.

## Completed reduction passes

- Unified graph-event/eager/device-turn attention scheduling in one loop while
  preserving Q-ready prearming, wait-before-compute and record-before-publication.
  Removed duplicate BF16 publication branches and the now-resolved address-bug
  startup fault-localization synchronizations (final prewarm sync remains).
- JIT export table now maps a single list of names to bindings; same exports.
- Consolidated MB2 attention test setup; full ordered device-turn trace remains,
  along with graph-event and direct GDN behavior checks. Pruned redundant symbol
  presence inventories and unused source reads from historical source tests.
- Removed finished idle-diagnosis timestamps and opt-in duplicated weight-build
  verification; retained fast weight construction, numeric-reference path and
  cache validation. The exact fast-build/RNG comparison passed job7059114 before
  cleanup. No numeric test/timing window change.
- General run_cuda_check.sbatch forwards arguments and replaces the dedicated
  run_qwen35_gdn_fusion_smoke.sbatch. For GDN set FASTAFD_CHECK_SCRIPT to
  codex_scripts/validate_qwen35_gdn_fusion.py and pass --rows 19 --bench-iters 500.
  Removed a return-None check from benchmark functions that always return tensors.
- Removed build_megamoe_comparison.py, a literal-string rewriter for a delivered
  HTML template. Delivered HTML/CSV data remain; general sweep/plot tools remain.

## Final validation and results

Completed all requirements. Net905 code/test/script lines removed versus the
starting staged worktree. No CUDA arithmetic/tuning changes. Final candidate
source and docs synchronized to R/local-mirror;full candidate.patch saved locally
and under remote task artifacts. New case manifests identify base/source patch.
Candidate patch base is local parent d5d5917cb4bb2bb8acc199f93e11ecc3797a0233.
The remote control has a different Git HEAD with the original worktree overlaid;
use its frozen source directory as authority, not its HEAD alone.
Original before.patch and frozen remote control remain. User requested committing this cleanup together with the already staged code; no push requested.

Common shape:BOTH models,A16:F8(A2:F1),EP8,b7,128K,ATP1,MB2(3+4 real tokens),
attn-only,FP8xFP4. Alignment=real prefill/normal routing/NSYS0. Perf=randomized
resident KV/GDN/perfect routing/NSYS1. Q3memory.97/KV979200/max-capacity gate;
Q35memory.82/KV5019776,max-only gateOFF because b7 is below max38.

| Model | Alignment | Control latency | Cleanup latency | Latency change |
| --- | --- | --- | --- | --- |
| Qwen3 | 1904/1904 top10;1902 top1 | 25.259387667ms | 25.258882133ms | -0.002001% |
| Qwen3.5 | 1904/1904 top10 AND top1 | 15.943200067ms | 15.943236000ms | +0.000225% |

Both alignment results cover112prompts/1904tokens,zero missing score contexts.
Q3 uses cached scores only at identical prompt+prefix;Q35 uses official vLLM.
Q35 compact extraction additionally verifies112x17 ordered score steps,generated
IDs and ranks for every token. The unchanged100%top10 standard passes both.

Both performance pairs ran control then candidate on the same24GPUs/6trays;
all15samples retained,0outliers,18execution fields match. Q3 prompt metadata
only differs in output/cache directory paths. No material performance regression.
TPS/GPU:Q3 184.749794->184.753492;Q35 292.705771->292.705111.
Policy unchanged:maximum representative-role CUDA graph span,trace first exact
then next15,min10retained,max5outliers,10%dominant-range and10%max-median limits.
These are same-allocation comparisons,not claims of exact speed equality.

All validation jobs COMPLETED0:0;no task GPU jobs remain:
- Q3 generation7065359 25:00;paired perf7065971 15:33.
- Q35 generation7065649 21:53;official score7066164 26:56;paired perf7066251 14:11.
Each performance/generation job used24GPUs/6trays,short1hr;scorer4GPUs/1tray.
The two independent Q35 checks ran concurrently,with at most2task jobs active.

CPU validation:runtime28/28 initially;final runtime/overlap-contract/online-pool
suite63/63 in0.659s. Log=validation/cleanup_cpu.log. Four remote bash syntax
checks passed (profile_megamoe_iket,megamoe_input_sweep,run_cuda_check,score_afd_samples).
Final git diff --check passed. No py_check/hash-version audit or added test matrix.
Command:PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tests <pinned-python> -m unittest
 test_afd_fmha_runtime test_gb200_overlap_contract test_afd_online_case_pool -q.
Pinned Python=R/../fastafd_reproduce/envs/minisgl-3c716194-cuda130.artifact-5175891/bin/python.

## Evidence and reproducibility

Case artifacts:scratch/afd/cases/{qwen3,qwen35}/isl_131072/ep_8/a2-f1/batch_7/
code-cleanup-{alignment,control,candidate}-20260910. Each perf version has exactly
attention.nsys-rep,ffn.nsys-rep,metric.json,result.json,manifest.json. Alignment
versions have compact alignment.json+manifest.json;profiling was disabled.
Validation contains cleanup_summary.json,per-model performance comparisons,
Q3 per-token cached result,CPU log,plans,before.patch,candidate.patch.

Remote task artifacts=R/artifacts/code_reduction_20260910. Q3 sample:
generation/results/afd_qwen3_131072_r2_ag16_fg8_atp1_fep8_adp16_b7_n6_20260910_154335_manual_na/experiment/sample.json.
Q35 sample=qwen35_generation/results/afd_qwen35_131072_r2_ag16_fg8_atp1_fep8_adp16_b7_n6_20260910_160043_manual_na/experiment/sample.json.
Q35 full score=qwen35_score_retry/alignment.json;compact=qwen35_alignment_summary.json.
Q3 cached references=R/artifacts/{megamoe_alignment_a2_ep8_b7_20260909/alignment_score,
megamoe_final_e2e_20260909/alignment_score_retry}/alignment.json.

Reuse existing run_afd_paired_sources.sbatch and plans/{qwen3,qwen35}_{perf.csv,
perf_env.json,pair.tsv}. Q35 official scorer uses the existing remote
R/artifacts/qwen35_afd_overheads_20260904/jobs/score_vllm_sharded.sbatch and
plans/qwen35_score_retry_env.json. Reference settings match successful6983296:
DP4/EP4/TP1,DeepEP low latency,DeepGEMM,no prefix caching,language-model-only,
maxseq4,score concurrency4,pinned reference venv artifact5179989. Its clean
reference source is R/artifacts/qwen35_attn_only_optimization_20260906/
qwen35_accum_direct_cta_selector_fix/source/candidate;AFD samples use NEW cleanup
candidate. The reference source path does not select the AFD runtime being tested.

Resolved preparation failures:pool rejected alignment status=pending;corrected
perf status to needs_corrected_target_batch_rerun. Perf7065847 failed1:0 in1:16
before runtime because perf_state_source was empty;corrected BOTH plans to
randomized_resident_kv_and_gdn and passed actual launch parser. Unoccupied old
pools archived as pool-incomplete-plan;fresh pools used corrected plans.
Q35 scorer7066083 failed2:0 in1:07 because the generic launcher expects absent
vllm console entry point. Existing module-based sharded launcher resolved this;
no dependency installation/new script. Trial generic-scorer edits reverted.
No correctness/performance standard was relaxed for any retry.

## Scratch and future-session agreement

Deep cleanup leaves ONLY scratch/afd and scratch/megamoe. Initially2.4GB->56MB;
then user requested traces for every retained perf version:final~562MB,71configs,
207performance versions/414reports. Removed obsolete/wrong metrics and plots,
numbered v1/v2/etc snapshots,rejected v6,old A56 intermediates,incomplete captures,
raw samples/logs/NVTX/SQLite,one-shot scripts,overlays and old snapshot-patch backups.
Current before/candidate patches and full remote control retained. No remote
result files deleted. Accepted history includes three66-point Q3 sweeps plus
Q35 original/shared-fusion/final-two-stream/graph-edge milestones. Old v9 is
named ffn-two-streams-final-20260907.

Hierarchy:scratch/afd/cases/<model>/isl_<tokens>/ep_<EP>/a<A>-f<F>/batch_<batch>/<optimization>/.
Ratios reduced from exact GPU counts:A56:F16=>a7-f2. Full settings/provenance in
manifests. Shared reports=afd/reports/model;validation=afd/validation/cleanup_20260910;
standalone164/299-input MegaMoE sweeps=megamoe/{qwen3,qwen35}/results.
CODEX_PROJECT_nsys_traces.md is the future-session agreement. NO management script;
temporary manage_nsys_traces.py deleted. Every retained perf version must contain
exactly ONE attention and ONE FFN trace from its metric,not a different run.

Copied376reports(455635492B) to fill188 JSON-only historical versions. Corrected
one stale diagnostic pair beside final A3:F1/EP8/b7 graph-edge metrics. Final audit
passed all207versions:exact pair count/names,nonzero sizes,recorded sizes,metric
role paths/ranks. Temporary transfer plans/staging directory removed. Migration
records in validation/{scratch_cleanup,deep_scratch_cleanup,trace_migration,
case_hierarchy_migration,cleanup_summary}.json. Intermediate local paths are
historical;the latest hierarchy supersedes them. Remote paths unchanged.
