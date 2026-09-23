# Agent workflows

Keep reusable launch, correctness and analysis tools here. Plans, frozen workflows,
logs and generated outputs belong under `scratch/` locally and the matching remote
artifact directory. Read [MegaMoE optimization](../CODEX_PROJECT_megamoe_optimization.md)
for current source selection, protocol, results and measurement contracts.

## Plan execution and CPU preparation

| Task | Tool |
|---|---|
| Submit `env.json` + `submit.json` | `python codex_scripts/submit_plan.py sbatch PLAN` |
| Submit `env.json` + `launch.json` | `python codex_scripts/submit_plan.py afd PLAN` (add `--dry-run` to preview) |
| Monitor recorded jobs | `monitor_afd_jobs.py PLAN... --watch-seconds 300` |
| Run several model plans in one allocation | `run_afd_plan_sequence.sbatch` |
| Map TP1 attention ranks to rack-local FFN GPUs | `map_afd_racks.py` |
| Correlate coordinator CPU work with FFN graph submission | `analyze_afd_cpu_dispatch.py COORDINATOR_CPU_JSON WORKER_CPU_JSON WORKER_SQLITE --output REPORT_JSON` |
| Inspect kernel resources and concurrent GPU work | `analyze_nsys_kernel_overlap.py TRACE_SQLITE --kernel NAME_SUBSTRING --output REPORT_JSON` |
| Explain attention microbatch gaps and publication overlap | `analyze_afd_attention_gap.py ATTENTION_SQLITE FFN_SQLITE METRIC_JSON OUTPUT`; Lamport uses `--q-wait-kernel receive_qkv_lamport_kernel --q-publish-kernel publish_qkv_lamport_kernel`; `--same-host` requires verified placement |
| Compare QKV communication mechanisms in one rack | `bench_qkv_fanout.py` (embedded CUDA, CPU preparation, checked GPU comparisons and analysis) |
| Prepare and analyze rack-mapped Qwen3 model cases | `prepare_afd_rack_run.py prepare/validate/analyze --root RUN` with an explicit `mapping.json` |
| Seed compatible host/device/TVM modules | `seed_native_cache.py` |
| CPU-compile saved DeepGEMM specializations | `precompile_deepgemm_cache.py` |
| CPU-prepare a sweep's ordinary/profiling kernels | `precompile_megamoe_sweep.py` |
| Own replacement paths in a fresh header overlay | `prepare_header_overlay.py` |
| Run standard smoke with explicit native/router/header sources | `megamoe_header_smoke.py` |
| Check saved IKET discovery resource allocation | `validate_iket_module_resources.py` |
| Extract selected CUDA functions / inspect dispatch registers | `extract_cuda_sass.py`, `review_megamoe_dispatch_sass.py` |

Submission requires a dedicated output directory and refuses an existing `job.json`.
CPU-dispatch analysis requires coordinator/worker traces from the same host. It
aligns matching worker CPU/NVTX materialization ranges, reports alignment spread,
and measures command cadence, coordinator stages and worker preparation against
actual CUDA graph-launch API timestamps. It uses existing captures, with no GPU run.
QKV fan-out uses real fabric allocations and exact Q/K/V checks before slot reuse.
`submit --experiment protocol --profile --iterations 96` compares LL64, LL128 and
flag-based controls with fixed launch resources. `--experiment polling` isolates
whole-packet versus tag-only polling and a quiescent-payload diagnostic control.
`--experiment measurement` compares CUDA-event instrumentation on/off in the same
binary. These are mechanism comparisons, not parameter searches. No polling-delay
or CTA-grid tuning. LL64 uses atomic32-bit data+32-bit epoch pairs; LL128 uses
NVLink ordered128B lines,120Bdata plus8Bepoch. LL128 applies to the tested Blackwell
NVLink setting; vector stores are not generally atomic. Receivers unpack into
model-like94-layer/128K cache and Q slots. `trace` separates push time, sampled
receiver arrival→store span and all-receiver-ready roundtrip upper bound. The
latter includes returning ACKs and graph overhead, excludes the payload checker,
and uses matched ready observations for the control and Lamport paths.
`--experiment final` measures completion without Nsight: a publisher-clock marker
precedes the push, and a warp observes all receiver ACKs before recording the end.
This is a completion upper bound including the return ACK and launch boundaries;
input generation and exact payload checking lie outside each measured interval.
It compares the reference and16SM sender in both orders, plus an explicit
CUDA-event instrumentation control whose stop event follows receiver completion.
One CUDA-event pair outside capture surrounds the entire checked graph; a checked
warm replay excludes first-replay setup from the final result. Device
timestamps inside its boundaries provide an independent cross-check. The whole
graph includes fills/checks and is not communication latency. Raw samples and
event boundary overhead are retained. Nsight supplies kernel breakdowns separately.
`--experiment production` compares original and integrated direct-edge publication.
`--group-sizes 4,8,16,32` exercises A3/A7/A15/A31 using all32GPUs. CPU preparation
is separate; `--prepared PATH` reuses an explicitly unchanged compiled module.
Each submission freezes this same script in its result root; rejected experiments
remain reproducible there without cluttering the maintained script.
`model --model-template PLAN --model-ratio 7` reuses the matching promoted32GPU
Qwen3 environment for the original direct-edge control/candidate comparison.
`analyze` runs saved strict metric extraction and verifies the executed kernel.
Keep each attempt in a fresh output root. `MINISGL_AFD_QKV_EDGE=0` selects the
original transport; default direct-edge dispatch is restricted to validated grouped
decode. Opt-in `MINISGL_AFD_QKV_LAMPORT=1` selects the integrated small-decode
LL128 path for Qwen3 TP1/MB2; validation status is in the A17 Lamport project doc.
`--experiment integration` checks the exact production sender/receiver module.
Model inboxes require128B fabric-view alignment; no return ACK is added to the
model path. The existing transport remains the default.
Optional `--data-host oci-hsg-data` separates transfer from the submission host.
There is no automatic retry/promotion. CPU preparation/analysis uses cpu-normal on
HSG; GPU work uses short. Run complete validation/performance workloads, including
kernel/chain comparisons where appropriate; never allocate GPUs for debugging,
manual waiting or padded runtime. Check a timed-out observation before resubmitting.

Rack mapping uses an explicit JSON inventory of `{host,address,rack,gpus}` nodes;
`address` must match Ray's node address. Run `map_afd_racks.py inventory.json
--ffn-racks R01,R02,... --output mapping.json` on CPU before submission, then merge
its `environment` object into each plan's `env.json`. Keep the manifest with the
plan and verify the allocation and observed rank placement against it. Each named
rack gets one FFN GPU, and every remaining GPU there sends to that FFN. Each
attention-only rack distributes its GPUs equally across all FFNs. Ordering the
attention ranks by destination preserves existing integer AFD routing, with no
transport or model change. TP1 and equal positive fan-in are required; invalid or
uneven layouts fail. Slots describe node capacity, not pinned CUDA ordinals.
Fabric connectivity and allocation size remain the caller's responsibility.
For 18×32 GPUs with16FFNs this produces31 local+2+2=35 attention GPUs per FFN;
16×36 produces35 local attention GPUs per FFN. For an AGA multi-rack sequence,
set `FASTAFD_SEQUENCE_FABRIC_SCOPE=domain`, `FASTAFD_SEQUENCE_RACK_MAP` to the
manifest path, and `FASTAFD_CONTAINER_MOUNTS=/lustre:/lustre,/scratch:/scratch`.
The wrapper checks one NVL576 domain, exact hosts, and each plan's mapping.

Cache donors must match source/ABI, compiler target and execution image. Device
seeding uses `--kind device` with a CPU-precompiled donor; `DG_JIT_REQUIRE_CACHE=1`
fails on missing specializations. Native compatibility remains the caller's
responsibility. DeepEP seeding uses `--kind deepep --copy` and preserves build.signature;
TVM supports node/rank scoping. Fresh CPU DeepEP builds need an explicit architecture
(GB200: `TORCH_CUDA_ARCH_LIST=10.0`). Never overwrite unrelated caches.

CPU/GPU IKET include-path spelling must match literally: compiler flags enter cache
keys even when filesystem aliases resolve identically. Header overlays own every
replacement ancestor and reject existing destinations. Freeze base/overlay/cache
while jobs consume them. `MEGAMOE_NATIVE_SOURCE_ROOT` pins native checks to the
intended host source; a device-header override alone cannot validate descriptor ABI.
Offline IKET allocation checks are not numerical correctness or ABI proof.

## Kernel comparisons and complete sweeps

| Task | Tool |
|---|---|
| Plan, collect and audit deduplicated AFD-equivalent inputs | `megamoe_input_sweep.py` |
| Capture ordinary/IKET sweeps | `megamoe_input_sweep.sbatch`, `profile_megamoe_iket.sbatch` |
| Validate/run/analyze explicit same-EP model groups | `run_megamoe_sweep_group.py validate\|capture\|analyze PLAN` |
| Execute authorized sweep order | `run_megamoe_sweeps.py` |
| Analyze mirrored comparisons, preserving ranks/calls | `analyze_megamoe_sweep_comparison.py` (`--ordinary-only` recovers a complete ordinary prefix) |
| Trace producer-to-main intervals and repeated comparisons | `summarize_megamoe_router_timeline.py`, `compare_megamoe_rank_timelines.py` |
| Phase/pair and rank-local startup summaries | `summarize_megamoe_latency.py`, `summarize_megamoe_startup.py` |
| Three-window marker decoding | `decode_megamoe_readiness_windows.py` |
| Decomposition-only collective preparation | `decomposition_barrier.py`, `decomposition_barrier.cu` |
| All-rank CUDA event collection/comparison | `collect_cuda_rank_timings.py`, `compare_cuda_rank_timings.py` |

`precompile_megamoe_sweep.py MANIFEST... --source SOURCE --cache-root CACHE
--iket-include INCLUDE` validates all-push capacity and prepares kernels without
initializing CUDA. `--preparation-inputs` includes saved weight/input preparation
specializations. Manifests can explicitly pin per-case bucket/source contracts;
launch, preparation and analysis must use those values. All-push uses one expert
wave and at most32-row tiles, retaining actual input counts for larger problems.

The group runner does not submit/retry jobs. Plan validation checks every manifest,
shape and required path before GPU use. The profiler captures every rank, exports
`rank_decomposition/rank-R/decomposition.json`, and preserves rank0 report schema.
`MEGAMOE_IKET_ROLE_CONTRACT=all_push_sweep` validates actual active marker roles,
including sparse workers. Missing required markers remain errors. Explicit capture
phases `normal|profile` allow all ordinary arms before profiling; profiling requires
that run's ordinary baseline.

Current decomposition uses `MEGAMOE_DECOMPOSITION=1`,
`MEGAMOE_SMOKE_PREPARE_BARRIER=collective` and explicitly
`MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT=joined`. Preparation and workload share a
CUDA graph; outer event span includes preparation, while Nsight workload duration
and IKET phases exclude it. Barrier-to-metadata placement must pass. The default
separate layout retains its stricter pre-producer placement audit.

Initialize/record the entire CUDA event pool before warmups via
`collect_cuda_rank_timings.make_timing_event_pairs`; use
`summarize_megamoe_router_timeline.py --require-preinitialized-events` to audit it.
For complete ordinary graph comparisons use `--require-graph-identity
--retain-timestamps`, then `compare_megamoe_rank_timelines.py` with explicit ranks,
samples, control and candidates, each with a matching `_repeat` arm.
Joined collective captures additionally require `--joined-collective-preparation`:
validate both barriers and both resets in the same graph/stream before metadata.
The paired analyzer selects this audit from the plan's preparation contract.

Latency analysis preserves actual MMA issue, metadata-origin coverage, pair-local
pipeline spans and issue/publication/observation distinctions. Three-window decoder
retains raw wire names and window ordinals; H<=2048 has no tail. Optional
`decode_mma_retirement` requires the exact active even CTA set. Publication of one
window does not certify other windows. Use explicit metadata materialization,
SF-work-warp, publisher-count and activation-load-semantics settings matching the
captured source; do not silently reinterpret historical tags.

## Model correctness and overlap

| Task | Tool |
|---|---|
| Score generated samples against fresh reference | `score_afd_samples.sbatch` |
| Audit ordered prompt/token coverage and finite scores | `audit_afd_sample.py` |
| Compare saved scores only at identical prefixes | `compare_cached_alignment.py` |
| Separate path and PDL gains in paired orders | `compare_afd_ablations.py` |
| Kernel families/complete graph spans | `summarize_afd_kernel_families.py` |
| Explain attention/step differences | `analyze_afd_attention_gap.py` |
| Account for overlapping trace regions | `summarize_cuda_coverage.py` |
| Reset/output/consumer ordering | `summarize_megamoe_recycling.py` |
| Bound reset overlap with one common clock-offset interval | `bound_megamoe_reset_overlap.py` |
| Focused CUDA launch/metadata/quantization/GDN checks | `run_cuda_check.sbatch`, `validate_afd_graph_metadata.py`, `validate_fused_quant_outputs.py`, `validate_qwen35_gdn_fusion.py` |

`audit_afd_sample.py` defaults to zero top10 misses. `--max-top10-misses N` applies
only an explicitly agreed allowance; final all-push used4 across1,632 tokens.
The legacy scoring wrapper still has a zero-miss exit gate. Preserve shards before
that gate; Qwen3.5 reference plans need `--language-model-only` in VLLM_EXTRA_ARGS.

Selected Qwen3 A3:F1 EP8 batch4 configuration:

```bash
export MINISGL_MEGAMOE_PAYLOAD_MODE=all_push
export MINISGL_MEGAMOE_RECYCLE_EARLY_PDL=1
export MINISGL_MEGAMOE_RECYCLE_AHEAD=1
export MINISGL_MEGAMOE_RECYCLE_CTAS=8
export MINISGL_MEGAMOE_RECYCLE_TMA=0
export MINISGL_MEGAMOE_NORM_PRELOAD_WEIGHT=1
```

Use matched preload0/pull controls to attribute gains. Performance uses128K perf
mode (synthetic KV/perfect routing); correctness uses actual8K real prefill/routing,
NSYS0 and prepared prompt/kernel caches. Run analysis and scoring after GPU release.

`MINISGL_MEGAMOE_RECYCLE_TMA=1` opts into TMA clearing for ahead reset; default0
retains vector clearing. Prepare both router variants and fresh native/device caches
before a paired model run; source changes can invalidate cached build timestamps.

Prepared all-push router modules use `MEGAMOE_ROUTER_WRAPPER=GateTopkAllPushKernel`
with `MEGAMOE_ROUTER_MODULE`. The smoke wrapper then retains the ordinary router
for pull calls, allowing both paths in the same adapter correctness comparison.

Adapter correctness lives in `tests/afd_megamoe_adapter_smoke.py`; optional chain
benchmark uses `MEGAMOE_ADAPTER_BENCH_ITERATIONS` and rows (default6). Model-matched
Q3 B4 uses `MEGAMOE_SMOKE_BUCKET=1548` for pull; all-push remains1152. Keep reset and
norm inside chain timing; exclude only norm fixture setup. Input seeds are fixed
independently of preceding fixture coverage. Rank timing pools each rank before maxima.

For selector-fused recycling use `--independent-readiness
--hardware-output-dependency`; every combine writer must precede output use and all
eight clears precede reuse readiness. Model diagnostics use
`MINISGL_MEGAMOE_TRACE_ROOT` with measured capture metadata, explicit ranks/layers/
lanes, and complete coverage. Instrumented latency is not ordinary performance.
Profiler worker tags are global rank +1; Q3 A3:F1 EP8 tag25 is FFN0. Deferred final
norm pairs remain in audits and require care for per-call critical-path claims.

`FASTAFD_DEFER_CUDA_EXTRACTION=1` saves the exact CPU extraction argument list at
`CASE_RUN_DIR/experiment/cuda-extract-command.json`; run it after GPU release.
GPU capture completion and validated metric extraction are separate gates.

## Reports and reference backend

| Task | Tool |
|---|---|
| Full decomposition plots | `plot_megamoe_decomposition.py` |
| Refresh same-format comparison HTML | `build_megamoe_comparison.py` |
| Fixed-total-token EP scaling HTML | `plot_megamoe_ep_scaling.py`, template `megamoe_ep_scaling.html` |
| AFD metric sweeps/comparisons/plots | `summarize_afd_sweep.py`, `compare_afd_sweep_metrics.py`, `plot_afd_sweep.py` |
| Explicit installed TRT-LLM DeepGEMM reference sweep | `trtllm_megamoe_sweep.py`, `trtllm_megamoe_sweep.sbatch`, `trtllm_megamoe_iket_nvcc.sh`, `iket_event_register_limits.py` |
| Archive complete cases/selected evidence | `archive_sweep_cases.py` |

Load the visualization skill before producing/updating visualizations. Reference
TRT-LLM sweeps use the original installed backend, not optimized AFD or TRT CuTe;
ordinary timing uses the untouched image. Its sequence command runs one EP job and
CPU analysis at a time, stopping on failure. `collect` exports the shared schema;
build reports with `--source-format trtllm-original`.

`build_megamoe_comparison.py --exclude-outliers` is an explicit report option:
per-case distance from median must exceed max(3 ×1.4826×MAD,5% of median), separately
for ordinary/IKET duration. One retained IKET set supplies all phase statistics.
All cases/raw observations remain; `outlier_audit.json` records exact exclusions.
Do not apply this option to optimization evidence without authorization.

## September21 cleanup

Removed27 obsolete helpers: the legacy ABBA and multi-arm sweep/history stacks, temporary
marker-subset/profiling comparisons, rejected receiver/scheduler models and finished
probe summaries. Current sweep/group runner, rank comparison, full profiler and
correctness contracts remain. No retained code imports or invokes removed helpers.
Original versions and the old workflow guide are archived once at
`scratch/megamoe/deep_cleanup_20260921/before_cleanup.tar.gz`; the adjacent manifest
lists exact filenames/reasons and additional code archives. Completed plans should use their frozen workflows;
do not repoint them to a different current tool and silently change their contract.
