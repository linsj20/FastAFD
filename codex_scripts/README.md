# Agent workflows

Keep reusable launch, audit and analysis tools here. Experiment-specific plans,
frozen helper versions, logs and generated reports belong under `scratch/` locally
and the corresponding remote artifact directory. Read the relevant
`CODEX_PROJECT_<topic>.md` before resuming a campaign.

## Main commands

| Task | Command |
|---|---|
| Submit `env.json` + `submit.json` | `python codex_scripts/submit_plan.py sbatch PLAN` |
| Submit `env.json` + `launch.json` | `python codex_scripts/submit_plan.py afd PLAN` |
| Preview an AFD launcher plan | Append `--dry-run` to the preceding command |
| Monitor recorded jobs | `python codex_scripts/monitor_afd_jobs.py PLAN... --watch-seconds 180` |
| Run several AFD plans in one allocation | `run_afd_plan_sequence.sbatch` |
| Score saved model samples | `score_afd_samples.sbatch` |
| Audit complete generation and fresh reference scores | `audit_afd_sample.py` |
| Compare only matching saved reference prefixes | `compare_cached_alignment.py` |
| Seed a compatible DeepGEMM host module | `seed_native_cache.py DONOR --cache-root CACHE --output RECORD` |
| Seed completed TVM modules | Same command with `--kind tvm`; optional `--node-ip IP --rank-count N` |
| Plan, collect and audit kernel sweeps | `megamoe_input_sweep.py` |
| Execute an explicitly authorized sweep order | `run_megamoe_sweeps.py` |

Submission requires a dedicated output directory and refuses an existing
`job.json`. Meet the campaign's correctness gates and outstanding-job limit before
submission. There is no automatic retry or promotion workflow. Scoring preserves
all shards before enforcing the strict top-10 gate; Qwen3.5 text-only reference
plans must include `--language-model-only` in `VLLM_EXTRA_ARGS`.

Cache seeding requires matching source/ABI, compiler target and execution image.
It does not seed device kernels or establish binary compatibility. Existing
unrelated native modules are never replaced.

## Retained specialized tools

| Purpose | Files |
|---|---|
| Repeated source comparisons and exact checks | `megamoe_source_abba.sbatch`, `compare_megamoe_results.sbatch`, `collect_megamoe_abba.py`, `summarize_megamoe_abba.py`, `audit_megamoe_exact_outputs.py`, `audit_megamoe_config_comparison.py` |
| Repeated sweep arms | `prepare_megamoe_sweep_abba.py`, `megamoe_sweep_abba.sbatch`, `collect_megamoe_sweep_arms.py`, `compare_megamoe_sweep_arms.py` |
| Sweep execution and profiling | `megamoe_input_sweep.sbatch`, `profile_megamoe_iket.sbatch`, `iket_event_register_limits.py`, `review_megamoe_dispatch_sass.py` |
| Historical comparisons | `compare_megamoe_sweeps.py`, `compare_megamoe_frozen_history.py`, `compare_megamoe_fullchain.py`, `freeze_megamoe_fullpath_history.py` |
| Timing and routing analysis | `summarize_megamoe_router_timeline.py`, `compare_megamoe_router_timelines.py`, `summarize_megamoe_routing.py`, `summarize_megamoe_scaling.py`, `summarize_afd_kernel_families.py`, `analyze_afd_attention_gap.py` |
| Reports and plots | `build_megamoe_comparison.py`, `plot_megamoe_config_comparison.py`, `plot_megamoe_decomposition.py`, `summarize_afd_sweep.py`, `compare_afd_sweep_metrics.py`, `plot_afd_sweep.py` |
| Focused CUDA validation | `run_cuda_check.sbatch`, `validate_afd_graph_metadata.py`, `validate_fused_quant_outputs.py`, `validate_qwen35_gdn_fusion.py` |
| Evidence archival | `archive_sweep_cases.py` |

The timing tools distinguish ordinary GPU execution from instrumented phase spans
and client latency. Preserve all observations and historical provenance; a failed
performance or alignment gate is a result, not permission to change the threshold.

## September 15 consolidation

`submit_plan.py` replaces `submit_megamoe_plan.py` and `launch_afd_plan.py`.
`seed_native_cache.py` replaces both native cache-seeding scripts.
The still-used `audit_fullpath` function moved from the retired validation runner
to `compare_megamoe_fullchain.py` without changing its gates.

Removed the superseded automatic validation/report runners, three experimental
per-warp marker analyzers, and the standalone experimental Lamport transport
probe. Pre-cleanup versions are preserved under
`scratch/megamoe/commit_cleanup_20260915/scripts_before` and in the corresponding
remote artifact directory. Historical plans continue to use their frozen workflow
copies; recreate a current plan using the commands above rather than rewriting
completed experiment evidence.
