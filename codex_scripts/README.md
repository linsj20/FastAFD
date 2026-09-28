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
| Monitor recorded jobs | `monitor_afd_jobs.py PLAN... --watch-seconds 300 [--log-count-pattern REGEX]`; batches scheduler accounting, reads log paths from submission commands, and saves bounded tails/counts in each plan's monitor_snapshot.json |
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
| Check IKET resource allocation from discovery or a standalone cubin | `validate_iket_module_resources.py` |
| Extract selected CUDA functions / inspect dispatch registers | `extract_cuda_sass.py`, `review_megamoe_dispatch_sass.py` |
| Verify cleanup preserves compiled code | `review_megamoe_dispatch_sass.py --compare-preparations PAIRS.json --removed-config FIELD --nvdisasm NVDISASM --output DIR`; pairs name before/after preparation directories; removed fields must be inactive, all remaining configurations and instructions must match |
| Review row-partition resources across EP sizes | `review_megamoe_dispatch_sass.py review-rows ROOT`; uses scope totals and compiled control/workers/rows manifests, saves all SASS before checking register budgets, new spills, topology and identical small targets |
| Review explicit worker/row scheduling arms | `review_megamoe_dispatch_sass.py review-scheduling ROOT [--control ARM] [--resource-policy ARM]`; checks compiled geometry against the execution plan, preserved small targets and non-scheduling settings, register budgets and new spills; optional resource policy maximizes balanced workers within the metadata CTA budget |
| Review a manifested cleanup of profiling weight caches | `cleanup_megamoe_profile_weights.py MANIFEST`; dry run by default, `--delete` rechecks completed jobs, file identity and exact paths before removal; writes a receipt |

Cleanup comparison is strict by default. `--normalize-source-metadata` permits
only entry-symbol renaming and proven assertion source-line arguments; raw SASS
is saved. If cleanup changes generated code, `--resource-gate` checks retained
configurations, register budgets and new local-memory instructions, and explicitly
requires a subsequent matched GPU comparison against the frozen implementation.
It does not assert identical code or performance preservation.
With source normalization enabled, `--normalize-validation-helper-symbols` also
permits a bijection between same-ordinal, entry-scoped `validate_window_rows`
symbols. It rewrites only exact relative-call symbol operands; function order,
addresses, every body instruction and other operands must still match. Unknown
renames or changed function coverage fail. Reports retain the explicit symbol map;
this structural evidence does not remove measured timing excursions.

For ordinary-only MegaMoE capture, set `FASTAFD_IKET_SAVE_WEIGHTS=0` to avoid
writing transformed-weight caches that only instrumented replay consumes. The
default remains 1 for staged profiling. Disabling caches rejects profile/all modes;
timings and exact-output references are still saved.

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

Cache donors must match source/ABI, compiler target and execution image. Use
`prepare_header_overlay.py --own-native` for host/configuration changes. It copies
the binding and native C++ tree while preserving explicit replacements. Both the
binding and quoted native includes can otherwise resolve symlinks to base sources,
even with fresh caches. The precompiler rejects these symlinks before building;
explicit `--header-root` uses the separate configured-overlay workflow.
Host configuration ABI changes require a fresh native extension as well as JIT code.
Device seeding uses `--kind device` with a CPU-precompiled donor; `DG_JIT_REQUIRE_CACHE=1`
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

For all-rank Nsight Compute diagnostics, the ordinary capture-only wrapper accepts
`MEGAMOE_SMOKE_NCU_METRICS` and an explicit executable in `MEGAMOE_SMOKE_NCU`.
It uses the TCP communicator with all Slurm ranks, synchronized launches inside
the `megamoe_ncu` NVTX range, and each rank's assigned GPU. Initialization and
warmups remain outside that range. One MegaMoE kernel per rank is profiled;
clock/cache controls are disabled and reports stay under each rank's `ncu/`.
The wrapper excludes simultaneous Nsight Systems or sanitizer collection and
skips weight exports needed only by IKET. Profiled timings are diagnostic, never
ordinary latency evidence. Verify tool flags/metrics on CPU, then require report
coverage, replay correctness and collection-pass review from the first live run.
The first live TCP startup failed because peer discovery did not complete.
CPU socket probes found one elected listener per node, even with the head hostname
specified. `ncu_tcp_launcher.py` reserves the port on client nodes and starts the
head listener first; all eight CPU peers then initialized one communicator before
the expected missing-CUDA-driver error. The opt-in wrapper uses this launcher with
a fresh shared coordination directory per capture. No automatic retry occurs.
The GPU run also initialized all eight peers, then exposed an incomplete workflow
snapshot: collective preparation imports `decomposition_barrier.py`, which builds
the adjacent `decomposition_barrier.cu`. Include both files alongside
`collect_cuda_rank_timings.py` when snapshotting the smoke test. Import the test
and its deferred helpers and prepare the barrier on CPU before GPU submission;
use a separate TVM cache when the helper's source path changes. GPU capture/replay
support with this startup fix remains unvalidated. A later run entered router JIT
after NCU attached. `megamoe_header_smoke.py prepare-routers --source SOURCE --cache-root
CACHE --output MAP.json` builds the ordinary pull and all-push routers on CPU and
records the actual loaded libraries. `MEGAMOE_ROUTER_MODULES=MAP.json` makes
`megamoe_header_smoke.py` load both explicitly; recycling/ahead modes are rejected.
Check source identity, both bindings with JIT disabled, and collective-barrier
preparation before GPU use. `review_megamoe_ncu.py ROOT` audits planned correctness
and per-rank report coverage and exports raw duration/DRAM counters. Review replay
passes before interpreting these counters; they are not weight-only traffic.
Prepared-router GPU7442963 reached profiling on all ranks but hung at50% with
the NCU graph-node replay warning. `MEGAMOE_SMOKE_NCU_GRAPH_PROFILING=graph`
selects whole-graph profiling, requires one joined preparation/workload graph,
and removes the individual-kernel filter. It is opt-in; its counters include
reset/barrier, routing and MegaMoE and must never be labeled kernel-only.
Whole-graph GPU7443080 completed13 captures before a transient permission error
while inspecting a child fd stopped listener discovery. The launcher now skips
inaccessible/exited fd entries while still requiring an owned listening socket;
a CPU regression test also rejects foreign listeners. `review_megamoe_ncu.py
ROOT --completed-prefix` explicitly labels analysis of completed leading captures
as partial and retains the original planned count. Corrected CPU analysis passed13 captures/104 one-pass reports and exact-source/replay
checks. NCU raw CSV uses metric columns and a units row; export with --print-units
base. Whole-graph durations were heavily distorted, and all-push reset writes must
be distinguished from kernel traffic when comparing with barrier-only pull.

## Kernel comparisons and complete sweeps

Current cleanup removes the rejected `READY_SCHEDULE`, `ROTATE_PEERS`,
`SFA_MULTICAST`, `METADATA_REGISTER_SCAN`, `METADATA_SINGLE_JOIN`,
`SKIP_IDLE_DISPATCH`, `METADATA_WARP_GROUP_LIMIT`, `SKIP_EMPTY_PARTITIONS`,
`SF_EARLY_RETURN`, `SINGLE_GEMM_EPILOGUE`, `FC1_WG_READY`, `FC1_ASYNC_READY`,
`METADATA_COOPERATIVE_GROUPS`, `METADATA_AFTER_REGISTERS`, `EARLY_SF`,
`DIRECT_SHARED`, `WINDOW_WIDE` and `LOCAL_SOURCE` options.
Nonzero stale settings fail fast. Reproduce those experiments only in their frozen
sources/caches. Router preparation now uses `megamoe_header_smoke.py prepare-routers`;
AA/AB/BB planning uses `prepare_megamoe_throughput.py calibrate`.
The requested final sweep uses HSG `short` QoS, one GPU job per EP, with both models
in each supported EP job. Keep CPU compilation and analysis outside GPU allocations.

| Task | Tool |
|---|---|
| Plan, collect and audit deduplicated AFD-equivalent inputs | `megamoe_input_sweep.py` |
| Capture ordinary/IKET sweeps | `megamoe_input_sweep.sbatch`, `profile_megamoe_iket.sbatch` |
| Validate/run/analyze explicit same-EP model groups | `run_megamoe_sweep_group.py validate\|capture\|analyze PLAN` |
| Execute authorized sweep order | `run_megamoe_sweeps.py` |
| Analyze mirrored comparisons, preserving ranks/calls | `analyze_megamoe_sweep_comparison.py` (`--ordinary-only` recovers a complete ordinary prefix) |
| Trace producer-to-main intervals and repeated comparisons | `summarize_megamoe_router_timeline.py`, `compare_megamoe_rank_timelines.py` |
| Review per-rank mean/median/p95, pooled tails and equal-case geometric mean speedup in both orders after a passing audit | `analyze_megamoe_throughput.py review-ranks ROOT --candidate ARM [--control BASELINE] --output REVIEW.json`; aggregate covers every campaign case, with small-batch preservation judged separately |
| Audit a completed capture log before timeline extraction | `analyze_megamoe_throughput.py ROOT --correctness-only`; requires every planned start/completion and exact per-rank event sequence, writes capture_correctness_review.json; timing/graph and performance gates remain separate |
| Aggregate a complete EP matrix against fixed case/control/context scopes | `analyze_megamoe_throughput.py aggregate ROOT --scope SCOPE.json --output REPORT.json`; requires passing matched audits and matching GPU job IDs; reports every case ratio/regression, equal-case geometric scores by model/EP, and separate small controls; performance review remains required |
| Prepare isolated first-FC1/first-FC2 operand and accumulator wait probes | `prepare_header_overlay.py --base HEADER_ROOT --output OUTPUT --mma-wait-probe`; requires CPU resource and ordinary-instruction preservation gates, exact coverage decoder, and matched ordinary/profile timings |
| Compare two all-push graph variants within one smoke process | `interleave_megamoe_smoke.py`; set `MEGAMOE_INTERLEAVED_VARIANTS` to two ordered environment dictionaries and `MEGAMOE_INTERLEAVED_OUTPUT` to a new JSON path; joined ordinary graphs only, repeated ABBA, exact output checks before/after timing |
| Audit balanced or normal interleaved runs | `analyze_megamoe_throughput.py review-interleaved ROOT [--output REVIEW.json]`; scope declares arm labels, entry protocol, EP, iterations, warmups, routing and exact-check counts; retain ranks/windows/both orders; AA/BB calibration ratios stay separate from the AB policy score |
| Phase/pair and rank-local startup summaries | `summarize_megamoe_latency.py`, `summarize_megamoe_startup.py` |
| Three-window marker decoding | `decode_megamoe_readiness_windows.py` |
| Decomposition-only collective preparation | `decomposition_barrier.py`, `decomposition_barrier.cu` |
| All-rank CUDA event collection/comparison | `collect_cuda_rank_timings.py`, `compare_cuda_rank_timings.py` |

Sweep forward and reverse arms require separate output roots. The group launcher
rejects repeated resolved root/case/phase destinations before capture, including
symlink aliases. Disjoint case stages and separate ordinary/profile phases may
share a root; `both` reserves both phases.

Interleaved timing defaults to `MEGAMOE_INTERLEAVED_ENTRY=synchronized`, retaining
the device synchronization and rank barrier after warmups. Experimental `queued`
moves the rank barrier before warmups and enqueues timing directly afterward on
the same stream. It changes no kernel, warmup count, or sample count; both entry
protocol and warmup count are recorded. Compare protocols on shared-code controls
before interpreting small timing differences, and retain all original samples.
An eight-process EP8 comparison passed exact checks and reduced the shared-code
first-call difference from47–67us to0.2–1.3us. New interleaved steady-replay runs
should select `queued` explicitly; remaining order/rank variation is still measured.

`precompile_megamoe_sweep.py MANIFEST... --source SOURCE --cache-root CACHE
--iket-include INCLUDE` validates all-push capacity and prepares kernels without
initializing CUDA. `--preparation-inputs` includes saved weight/input preparation
specializations. Manifests can explicitly pin per-case bucket/source contracts;
launch, preparation and analysis must use those values. Opt-in wide/compact all-push
paths change tile sizes and expert waves; use the generated native configuration
and recorded environment to interpret each run.
`DG_MEGAMOE_ALL_PUSH_BLOCK_M=64` opts into M64/STORE16 with two epilogue
warpgroups above16.5 expected rows per expert. WIDE=0, pull and smaller cases
retain their original selection. This targets the 1K/2K crossover; CPU resource
review and matched correctness/performance checks are required before selection.
Automatic M64 selection from expected load (`COMPACT_TILES=2`) was rejected:
it improved balanced holdouts but regressed Qwen3 EP8 1K normal routing by
3.8425us in the joined graph, on every rank in both orders. `COMPACT_TILES`
accepts only0/1; explicit `BLOCK_M=64` remains available for diagnostics.
No model/case-ID selection policy was promoted. The phase analyzer accepts M64;
instrumentation resource and timing-bias checks remain required.
`DG_MEGAMOE_ALL_PUSH_BALANCE_EPILOGUE=1` experimentally distributes store blocks
round-robin across epilogue warpgroups without changing physical tile size, waves,
or shared-memory staging ownership. Default0; small tiles (M<=32) and pull force0.
CPU resource review and exact-output/replay comparisons across balanced and normal
routing are required before performance promotion. This is not a model/case rule.
`DG_MEGAMOE_ALL_PUSH_WAVES=2` experimentally applies the existing adaptive expert-wave
rule to small tiles as well. Mode0 keeps one wave; mode1 retains the established
small-tile schedule and adapts wide tiles. The wave-only comparison uses the same
tiles, staging and other settings, with both models and small controls. Register,
barrier-progress and exact-output validation are required before performance use.
Mode2 failed the small-control performance requirement: Qwen3.5 EP8 256 tokens
regressed 3.3358us while 1K improved 4.6326us, consistently across ranks and both
orders. It remains an explicit diagnostic mode; no general or case-specific policy
was selected, and its planned normal-routing GPU follow-up was withheld.
`DG_MEGAMOE_ALL_PUSH_FULL_TMA=2` experimentally extends full activation boxes to
small tiles. Mode1 retains wide-only behavior; mode0 retains eight-row FC1 loads.
At M32 with 32 aligned valid rows this halves activation TMA issue count at equal
bytes; shorter extents read extra padding. Compare with WAVES1 and unchanged tiles,
including both models and small controls. Runtime correctness passed, but Qwen3
EP8 256 regressed 0.1963us across all ranks/both orders and Qwen3.5 1K regressed
0.1675us on average. Mode2 is rejected for general use; normal/EP16 GPU follow-ups
were withheld. No case exception or default change.
`DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS` accepts positive explicit worker counts.
The default remains `min(local_experts, 2*EP)`; small tiles retain that topology
allocation. Automatic policies -1 through -4 were retired at closeout: large-case
wins did not generalize to all small/nearby controls. Negative values fail fast.
Frozen sources and results remain under `artifacts/throughput_20260922`; useful
large-count configurations remain available through positive explicit counts.
`DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS` defaults to1 and experimentally
partitions rows inside full-activation materialization tiles (1..8, dividing the
metadata worker count). It changes ownership, independently of the worker-count
selector: each tile worker has one SF/A CTA pair per row partition. Row ranges
align to metadata_threads/16; empty partitions still publish. Four distinct bits
per participant preserve the32-bit readiness word; every consumer waits for the
complete mask. Existing barriers/proxy fences/retirement remain. Full-TMA-off
layouts retain1 partition. Recycling, coalesced-SF and compute
warmup combinations fail validation. Current experiment compares explicit32
workers with1/2 partitions against topology16 and pull, both models EP8 with
8/32/256 controls and1K/2K targets. No automatic policy/default promotion.
Host ownership/flag and30 strict CPU resource checks pass. Eighty ordinary captures
pass400 exact comparisons/640 replays. Versus topology16, row partitions improve
Q3 1K/2K by2.153/5.318us and Q35 2K by1.928us kernel time in both orders/all ranks.
Versus equal32 unsplit workers, Q35 2K instead regresses3.003us; this is not a
universal improvement over whole-tile parallelism. All four large cases still
trail pull. Nearby/nonuniform validation passed correctness; Q3 gains persist across
768..3072 and interleaved1K/2K. Q35 2K mean improves slightly but interleaved rank
and tail deltas are mixed. Keep experimental; defaults unchanged. The Q3 EP16
follow-up preserves eight logical expert workers (16 physical workers/P2), with
an unsplit16-worker arm and both original EP16 small controls;12 CPU resource
checks passed. EP16 ordinary validation passed 320 exact comparisons/512 replays;
Q3 1K/2K gain 2.213/5.678us kernel time versus control across all ranks and both
orders, but remain 0.472/4.500us behind pull. No default promotion.
The expert-group prototype was removed at closeout. Stale
`DG_MEGAMOE_ALL_PUSH_METADATA_EXPERT_GROUPS=2` requests fail fast. Its frozen V3
source/results preserve the Q35 1K gain and the small-control/2K regressions;
no model or token exceptions were added to rescue it.
`DG_MEGAMOE_ALL_PUSH_COMPUTE_WARMUP_ALL_TILES=1` opts small tiles into the existing
bounded startup schedule selected by `COMPUTE_WARMUP_ROUNDS` (which must be nonzero).
Free compute CTA pairs get the initial work; metadata pairs join the regular
schedule after the prefix. Worker counts, ownership rules and kernel ABI stay
unchanged. Default0 preserves the wide-only schedule. This tests scheduling as a
cause of worker-expansion regressions without adding a model/token exception;
Thirty focused resource checks and ten configuration checks passed. EP8 exact
comparison/replay checks passed, but the uniform two-round policy regressed all
four 1K/2K targets versus workers alone by1.30–6.96us, with every rank mean,
median and p95 worse in both orders. Several small controls also regressed.
General use is rejected; no round retuning or case-specific exception is selected.
`DG_MEGAMOE_ALL_PUSH_EXTENT_TMA=1` experimentally trims the existing full-box
FC1 activation loads to the runtime MMA extent. Each CTA selects an eight-row-multiple
descriptor for half the aligned valid rows; the coordinator expects exactly those
bytes. Physical tiles, FC2 loads and readiness ownership stay unchanged. The default
is0; the option is inactive on the existing eight-row path, including small tiles
with FULL_TMA1. Host/device descriptor ABI, resource use, correctness and performance
need validation before use. Saved routing predicts fewer padded transfer rows, not
measured DRAM savings or a speedup.
Combining FULL_TMA2 with EXTENT_TMA1 also failed general performance validation:
Q35 768/1K normal-routing means regressed on every rank in both orders, and
interleaving confirmed mean regressions at Q35 256/1K. Tiny-case gains do not
justify a model/token exception. Keep this combination diagnostic; no EP16 promotion.
`DG_MEGAMOE_ALL_PUSH_EXTENT_TMA=2` uses the same extent descriptors and byte
counts as mode1 but skips their startup prefetches. Descriptors are fetched on
demand by TMA; other descriptor prefetches stay unchanged. This isolates unused
descriptor startup work and requires the same resource/ABI/runtime checks.
Normal-routing and interleaved validation passed correctness, but mode2 showed
no stable incremental gain over mode1 at the focused targets. Keep it diagnostic;
no default or EP16 promotion follows from its gains over the original full boxes.
`DG_MEGAMOE_ALL_PUSH_LATE_COUNT_LANE=1` experimentally re-reads the lane ID at
expert-count publication and scheduler fetch. This separates count addresses from
barrier-initialization expressions that the compiler can otherwise keep live
across startup and register-budget changes. Default0; pull is unchanged. It does
not alter worker counts, routing, synchronization or tile selection. CPU spill
checks and matched runtime validation are required before selection.
`DG_MEGAMOE_ALL_PUSH_FC2_STAGE_READY=1` tests FC2 operand loads after acquiring
the two FC1 arrival bits needed by each K slice, instead of waiting for the whole
FC1 output. Acquired mask bits are cached until that tile finishes; publication,
stage credits, TMA shapes and retirement remain unchanged. Default0; pull and
tiles with M<=32 retain the existing wait. EP8 resource and exact-output checks
passed, but matched timing showed little broad benefit: Qwen3.5 2K gained0.309us
with mixed tail results; Qwen3 means barely changed and some tails worsened.
All four large targets still trail pull. Keep diagnostic; broader runs and default
promotion are withheld. The upstream whole-mask wait replaced a slower per-block
approach; narrower dependencies alone do not establish a performance benefit.
`DG_MEGAMOE_ALL_PUSH_ACTIVATION_ISSUERS=2|4` assigns each physical activation
pipeline slot to one of the existing dispatch warps (`stage % issuers`). Each
owner acquires payload windows before its first load in that window and supplies
the existing FC2 observer credit for its slots in multiwave runs. Barrier counts,
TMA shapes, transaction count, threads and shared memory are unchanged. Default1;
pull and M<=32 retain1. Multiple issuers require separate full-box loads; IKET
profiling fails explicitly until its marker contract supports the extra issuers.
Eighteen focused CPU resource checks passed with no new spills; all48 ordinary
captures passed exact source/replay checks. General performance promotion is
rejected: two issuers regress Qwen3 2K, and four regress Qwen3 1K/Qwen3.5 2K
across both orders. Gains elsewhere do not justify case-specific issuer selection.
All large targets still trail pull. Default1 remains; broader runs are withheld.
`prepare_megamoe_throughput.py calibrate NORMAL INTERLEAVED OUTPUT --cases
qwen3:16,qwen35:32` prepares AA/AB/BB graph comparisons from validated normal and
queued interleaved campaigns. It reuses frozen kernels, exact references and the
rank reviewer; both graph orders and all samples are retained. AA/BB measure
identical-policy variation alongside the AB policy comparison. It only prepares
jobs, with no compilation, submission, automatic noise subtraction or promotion.
For a completed ordinary balanced sweep, use
`prepare_megamoe_throughput.py calibrate-sweep SWEEP OUTPUT --preparations PREPARED_ROOT
--control workers --candidate rows --cases qwen3:512,qwen35:512,qwen3:2048,qwen35:2048`.
It requires matching frozen source/cache/shape contracts, prepared cubins and every
rank's exact reference. It preserves the measured `COMPARE_COMBINE` setting;
EP64's enabled conventional-combine check requires both prepared kernels, while
EP8's disabled check requires only the selected kernel. Only worker-count/metadata-row scheduling differences are
accepted; other changes fail. The generated24process AA/AB/BB plan uses both graph
orders,1000calls/arm/rank and synchronized entry (`--entry queued` is explicit).
`run-calibration OUTPUT` executes it in one matching NVL72 allocation/container
step; submit the generated GPU plan through `submit_plan.py`, then run the frozen
reviewer on CPU. These joined-graph measurements include preparation and do not
replace ordinary kernel/span scores. No new kernel build or automatic submission.
`prepare_megamoe_throughput.py normal SWEEP OUTPUT --preparations PREPARED_ROOT
--candidate ARM --totals 32,1024,2048` builds a normal-routing follow-up from a
passing ordinary sweep. Totals must include its small control first. It preserves
the measured source/settings, copies the compatible native module into a fresh
cache, and reserves capacity for arbitrary expert skew. CPU checks enforce tiles
at most192 and sufficient materialization capacity. `run-graphs OUTPUT` uses the
same shared-container runner as calibration. The generated
`analyze_megamoe_throughput.py review-normal OUTPUT` analysis validates all rank
samples, both orders and the exact per-rank sequence of nine input/output checks
plus replay, including empty/padded routes. Normal joined-graph times include
preparation/reset and remain separate from ordinary kernel/span scores.
For combined allocations, `job.json` may list `graph_plans` containing these
scope.json paths; the300s monitor reports completed/expected graph processes
alongside ordinary sweep progress. A completed result file is progress, not a
substitute for the later correctness/timing audit.
`analyze_megamoe_throughput.py review-windows ROOT --cases
qwen35/ep64_tokens128,qwen35/ep64_tokens2048 --output REVIEW.json` inspects saved
ordinary traces in contiguous100-call windows (`--window-size` overrides this).
It retains every arm, order, rank and sample in the selected cases, including the
partial final window and inter-call gaps. Means must match the completed trace
audit. Producer, selector, kernel and gap co-variation is diagnostic, not causal
proof or a replacement score. Run on CPU; it does not launch GPU work.
`prepare_megamoe_throughput.py --totals` accepts distinct
positive multiples of EP, including non-power-of-two holdouts; it constructs the
requested cases and reserves common capacity across arms.
The planner supports EP2/4/8/16/32/64. Use `--models` to select an explicit subset
present in the baseline EP plan; unsupported models fail before output creation.
`--model-variants FILE.json` accepts arm → model → environment overrides for
explicit geometry-derived settings. The same resolved settings govern capacity,
CPU compilation and GPU capture. It does not add a runtime winner selector.
EP2/4/32/64 requires `--control-total`, a positive multiple of EP shared by the selected
models. EP8/16 retain their previous controls unless explicitly overridden.
These options select experimental coverage, not kernel policy. GPU submissions
retain the baseline topology (EP2: two ranks on one four-GPU node; EP4: four ranks).
`DG_MEGAMOE_ALL_PUSH_BLOCK_M=80|96` selects M80/STORE8 or M96/STORE16 for
the wide all-push path above32.5 expected rows per expert. Small batches,
WIDE=0 and pull retain their existing selection. This is an explicit experiment
for shapes such as80 rows per expert; no automatic crossover changes.
`DG_MEGAMOE_ALL_PUSH_BLOCK_M=128|160|192|256` overrides the tile only above
96.5 expected rows per expert; zero retains selection. M256 requires
`DG_MEGAMOE_ALL_PUSH_ACCUM_STAGES=1` to fit tensor memory. Existing workspace
capacity must cover worst-case route padding; insufficient capacity fails.
Small kernels and the workspace layout retain their existing configuration.
`DG_MEGAMOE_ALL_PUSH_COMPACT_STORE=1` is an experimental M128-only output
staging choice (16 rows instead of32), allowing the host to select a deeper
operand pipeline. It leaves small controls and other tile shapes unchanged;
check ordinary paired timings because it also increases store iterations.
`DG_MEGAMOE_ALL_PUSH_LINEAR_SOURCE=1` compares direct descriptor/top-k source-row
decoding with the existing rank/token calculation on large all-push tiles.
It preserves the generation-bank offset and the payload readiness protocol.
`DG_MEGAMOE_ALL_PUSH_CACHE_DESTINATIONS=1` caches each materialized row's
return address during SF packing and reuses it across FC2 output tiles. Large
all-push/in-band-combine only; FC2 acquires SF-ready before reading the cache.
The cache reuses the unused source-metadata allocation without changing its size.
`DG_MEGAMOE_ALL_PUSH_INTERLEAVE_M=1` alternates two M tiles per N-pair
inside each expert. Adjacent two-CTA clusters remain on one M tile, and a final
unpaired M tile keeps its original order. Large all-push only; experimental.
`DG_MEGAMOE_ALL_PUSH_NARROW_SF=1|2|3` assigns that many metadata warps to scale
packing and the remaining warps to middle-window validation. Zero preserves the
original split:4/4 in original dispatch, or four dedicated warps doing both tasks
sequentially. Nonzero overlaps the two tasks in dedicated dispatch too. Small/pull
keep their original roles. The `all_push_pipeline` profile contract checks the
compiled count and corresponding middle-validator warp. Device marker validation
remains required.

`DG_MEGAMOE_ALL_PUSH_COALESCED_SF=1|2` visits physical scale rows contiguously
when packing large all-push tiles. Mode1 assigns one chunk per task;mode2 handles
all chunks of a row with one descriptor/source lookup. Both preserve layout and publication,
but trades contiguous output stores against scattered input loads and inactive
padded-row work. Small controls retain the original ordering; experimental.

`DG_MEGAMOE_ALL_PUSH_PREFETCH_SEND=1` refills each sender warp's shared input
slot after its current row has been read into registers, overlapping the next
row's TMA with current remote stores. It keeps the existing slot, barrier phases,
payload encoding and readiness protocol. Small controls force this option off.

The rejected `DG_MEGAMOE_ALL_PUSH_ROW_RELEASE` experiment is removed; nonzero
settings fail explicitly. Frozen source and matched performance evidence remain
in the September23 row-publication campaign. Ordinary per-word polling remains.

`DG_MEGAMOE_ALL_PUSH_MIDDLE_WINDOW_END=1536|2048|2560|3072` moves the
middle/tail validation boundary for large all-push tiles. Default2048 and small
specializations are retained. Both validators and TMA issuers use the same endpoint;
profile contracts verify the recorded byte windows against the compiled value.

`DG_MEGAMOE_ALL_PUSH_HALF_WARP=1` assigns16 lanes to each row in the
middle/tail validators, replacing two row addresses per lane with one and twice
the vectors. Full-warp polling still covers two rows and stores exact validated
values. Prefix ownership/publication and small kernels remain unchanged.
Experimental: compiled register bounds and device correctness/performance required.

`DG_MEGAMOE_ALL_PUSH_FIRST_WINDOW_END=512|1024` changes the first readiness
boundary for large all-push tiles (default1024). Prefix/middle validators and
both TMA issuer paths share the setting; sender ordering retains its1024-byte
first window. Small/pull force1024. Experimental; profiling records byte windows.

`DG_MEGAMOE_ALL_PUSH_STICKY_VALIDATION=1` retains each fully validated
activation vector while polling the remaining vectors. Default0; small/pull force0.
`DG_MEGAMOE_ALL_PUSH_FULL_ROW_COMBINE=0|1|2|3|4` selects chunked, full-row serial,
or full-row paired-vector combine. Mode2 polls two vectors per thread with
independent completion bits, preserving each vector's reduction order. It requires
in-band combine, exactly two vectors per epilogue thread and top-k<=16.
Mode3 pairs adjacent vectors for 32-byte loads and retirement stores, validating
all eight words before retaining and clearing a pair. It uses the same capacity
constraints as mode2 and preserves each output element's reduction order.
Mode4 additionally writes each final adjacent output pair with one aligned
32-byte store, retaining mode3's arrival checks and reduction arithmetic.
Its host entry rejects output pointers that are not 32-byte aligned.
Default0; small/pull force0. These options require CPU register/spill review and
ordinary device correctness, replay and mirrored timing comparisons before selection.

`DG_MEGAMOE_ALL_PUSH_WEIGHT_CACHE=1` applies an L2 evict-last hint to weights
and weight scales loaded by TMA. Activations retain normal eviction priority.
Default0; small/pull force0. This is a performance hint, not a residency guarantee;
use matched timings and traffic evidence to judge the experiment.

`DG_MEGAMOE_ALL_PUSH_WARP_COMBINE=1` gives each warp its own read-only route
mask and removes the two per-row CTA combine barriers. Inbox/output vector
ownership, payload polling, clearing and reduction order stay unchanged.
Requires in-band combine; default0 and small/pull force0. Compare saved barrier
cost against duplicated mask loads, with register bounds and exact replay gates.
Mode2 additionally assigns a whole row to each warp (requires full-row combine).
Its 32 lanes own disjoint vectors with the same per-element reduction order;
fewer repeated route masks trade against more serial vector work per row.
Modes3/4 assign two/four warps per row to retain more combine parallelism when
few local rows would leave most CTAs idle in mode2. Each warp still owns disjoint
vectors and its read-only mask; no new shared state or inter-warp barrier.
Default0 and the small/pull specialization remain unchanged.

`DG_MEGAMOE_ALL_PUSH_PREFIX_SWEEP=1` sends each warp's 1KB prefixes for all
assigned rows before its 3KB remainders (for hidden4096). The two disjoint TMA
loads keep the same shared slot and carry its parity across passes, including
partial CTA tails. Deduplicated peers, encoded bytes and final retirement stay
unchanged; there is no inter-warp pass barrier. Earlier prefix availability trades
against a second route lookup/TMA setup per row. Default0; small/pull force0.
Requires sender-prefetch0/row-release0. Validate register bounds and mirrored
ordinary correctness/replay/timing before selecting this experiment.

`DG_MEGAMOE_ALL_PUSH_PRODUCER_PREFIX=1` moves the first 1KB into the selector's
existing deduplicated SF sends; native senders load/send only the 3KB tail.
The selector's CTA join and route publication cover prefix retirement. Recycled
banks acquire each destination epoch before writing; nonrecycled calls retain
external preparation. Never wait aggregate CountsReady inside selector token CTAs.
Both producer and native selection require wide all-push with expected routed
tokens/expert >32.5; small/pull select the original paths. Requires hidden4096,
first-window1024 and prefix-sweep/prefetch/row-release0. Default0, experimental;
CPU resource/control-code checks and complete pipeline correctness/timing gate use.

`DG_MEGAMOE_ALL_PUSH_SENDER_LDG=1` replaces native senders' one-row TMA/shared
staging with coalesced L2 vector loads into the same payload registers. The
CountsReady acquire, route priority, peer deduplication, sentinel encoding,
stores and retirement remain. Sender transaction barriers are unused; existing
CTA/warp joins stay. Default0, small/pull force0. Requires hidden4096 and both
prefix options, prefetch and row-release disabled. Compare saved TMA setup/wait
against ordinary global-load issue/latency in complete pipeline measurements.
Mode2 uses aligned 256-bit loads/stores with the same 32 payload registers per
lane. It requires a capacity divisible by8 and first-window1024. Each warp still
stores the first1KB before the remainder; byte encoding and retirement stay the
same. Mode1 retains 128-bit accesses; mode0 retains TMA staging. Validate generated
instructions/registers and complete-call timing before selecting either mode.

`DG_MEGAMOE_ALL_PUSH_PREFIX_WIDE=1` validates and materializes the receiver's
first1KB with two32B vectors per lane instead of four16B vectors. The same16lanes
own each row; every word must pass the sentinel check before publication. Sticky
validation retains complete32B groups. Middle/tail ownership, warp votes, proxy
fences and retirement remain. Requires hidden4096, capacity divisible by8,
first-window1024 and row-release0. Default0; small/pull force0. This reduces
instruction count, not payload bytes; gate use on resources and paired timings.

`DG_MEGAMOE_ALL_PUSH_CACHE_PRIORITY=1` retains the helper's first-pass priority
keys for its queue scatter pass. Two disjoint key banks follow the two queue banks
inside the unused gather-index arena; existing generation retirement protects reuse.
Mode1 requires pool capacity at least four times input-row capacity. Mode2 keeps
keys in the helper CTA's existing shared scratch, after the aligned histogram;
the host rejects actual row counts that exceed this space (4224 rows for16 local
experts,4160 for64, at hidden4096). Both modes require prioritization. Default0;
small/pull force0. Mode1 adds one local global-memory key store/load per token;
mode2 adds shared accesses without increasing shared allocation. Measure complete
pipeline timing before selecting either mode. Mode3 prepares the same keys in
the selector from mapped local-expert IDs, publishes them through CountsReady,
then copies them into helper shared scratch. Invalid physical rows write the
sentinel key. It requires both global-bank and shared-scratch capacity and
producer-prefix0; no new allocation or barrier. Compare complete pipeline timing
against mode2 to include the added selector cost.
Mode4 retains mode2's shared cache and reads int64 route IDs in aligned vectors
in the helper: four IDs per32-byte load. Full-width validity and priority
computation are unchanged. Requires top-k divisible by4 and hidden divisible by1024; no selector
key preparation, additional allocation or barrier. Test issue savings against
register use and complete pipeline timing; default and small/pull paths are unchanged.

`DG_MEGAMOE_ALL_PUSH_RETURN_WIDE=1` uses eight lanes per FC2 output row and
one 32-byte remote store per lane. Each lane reads two existing swizzled 16-byte
fragments; destination prefetch uses the same revised row/column ownership.
BF16 packing, sentinel normalization, staging barriers and payload bytes are
unchanged. Requires in-band combine, STORE_BLOCK_M divisible by16, hidden
divisible by1024, intermediate divisible by128, padded SF-pool rows divisible
by32, capacity divisible by8, and 32-byte peer-buffer alignment. Small/pull
force0; unsupported large geometry fails.
Mode1 repeatedly improves Qwen3 EP8 but regresses Qwen3.5; it remains opt-in.
The explicit-global 16-byte mode2 experiment did not produce a repeatable gain
and has been removed. Its frozen source and results remain in the return-store
campaign. Mode0 preserves the original 16-byte return path.

`DG_MEGAMOE_ALL_PUSH_RETURN_WG_SYNC=1` joins each epilogue warpgroup separately
between FC2 tiles, whose staging regions are disjoint. Full epilogue joins remain
at FC2-to-FC1 transitions and before retirement, protecting the aliased FC1 layout
and shared cleanup. Payload stores and TMEM arrivals are unchanged. Default0 and
small/pull force0. Qwen3 EP8 4K improves across all ranks and both orders;
an independent repeat and EP16 4K validation are pending. The Qwen3 8K repeat
reverses pipeline gains by order, and Qwen3.5 8K regresses. Retain mode0 for
those sizes/models; do not promote the first 8K mean gain.

`DG_MEGAMOE_ALL_PUSH_SF_VECTOR=1` copies selector scale factors in 64-bit pairs
instead of 32-bit words. Destination deduplication, bytes, slot allocation,
normalization and publication barriers remain unchanged. This may reduce SF loop
iterations for hidden8192; it does not reduce transferred bytes. The opt-in path
requires aligned buffers and supported rank-ready routing, excludes producer-prefix
and prepared-key modes, and is forced off for small/pull. CPU and GPU validation
are pending; default0.

`DG_MEGAMOE_ALL_PUSH_GROUP_PRIORITY=1` combines equal-priority histogram and
scatter claims within each helper warp. Warp-uniform rounds ballot valid rows,
and matching lanes reserve one contiguous span with a single shared atomic.
The queue retains priority order and exact row coverage; ties within a bucket
may be reordered. Works with recomputed or cached keys. Requires prioritization;
default0 and small/pull force0. Publication and retirement remain unchanged.
Measure benefit against match/shuffle overhead; routing determines group sizes.
Mode2 instead gives each helper warp a private aligned histogram, with ordinary
atomics and expert-major/warp-major prefix spans. It adds no match/shuffle work;
the longer prefix walk must earn its cost. Histogram storage precedes shared
cached keys in existing scratch, with matching host row bounds (4128 rows for16
local experts,3872 for64, at hidden4096). No additional allocation or barrier.

The group runner does not submit/retry jobs. Plan validation checks every manifest,
shape and required path before GPU use. The profiler captures every rank, exports
`rank_decomposition/rank-R/decomposition.json`, and preserves rank0 report schema.
For a separate ordinary hardware-metric capture, set
`MEGAMOE_SMOKE_NSYS_GPU_METRICS=1` with the existing Nsight output prefix.
Only node-local rank0 samples all GPUs at20kHz; every rank keeps its CUDA timeline.
The collector records Nsight version, supported metric sets and sampling arguments.
Use an exclusive allocation and verify nonempty per-device metric data after CPU
export. These device-wide samples diagnose utilization/traffic; they are not
weight-specific byte counts or replacements for uninstrumented paired timings.
`MEGAMOE_SMOKE_NSYS_GPU_METRICS_SET` selects an explicit supported set, such as
`gb10x-ct` for GB200 cache/warp-stall triage. Unsupported sets fail in Nsight.
After CPU export, `summarize_nsys_gpu_metrics.py REPORT.sqlite...
--kernel KERNEL_SUBSTRING --samples 1000 --edge-margin-ns 60000 --output SUMMARY.json`
checks device identity and sample coverage, then summarizes whole-kernel and
interior samples in the same report's clock domain. The margin must cover the
largest sampling interval; no cross-report clock alignment is inferred. Clock
values remain raw export integers; display labels can have unapplied multipliers.
`MEGAMOE_IKET_ROLE_CONTRACT=all_push_sweep` validates actual active marker roles,
including sparse workers. Missing required markers remain errors. Explicit capture
phases `normal|profile` allow all ordinary arms before profiling; profiling requires
that run's ordinary baseline.

Use `MEGAMOE_IKET_ROLE_CONTRACT=all_push_pipeline` for the current unified-TMA/
dedicated-dispatch profiles. It validates the compiled issuer, receiver and
accumulator settings, including M160 tiles. The planner applies the corresponding
per-event register limits. `validate_iket_module_resources.py --cubin ...` without
`--tracker` checks allocation using an explicitly labeled CPU fixture; this does
not establish launch ABI, numerical correctness or performance.

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
retains raw wire names and window ordinals; H at or below the configured middle
endpoint has no tail. `DG_MEGAMOE_ALL_PUSH_MIDDLE_WINDOW_END=1536|2048|2560|3072`
changes the middle/tail ownership boundary only for large all-push tiles; default
2048 and small kernels remain unchanged. Nondefault endpoints must fit H. The
planner records the matching profile byte windows and the decoder verifies the
compiled endpoint. This preserves the existing per-window readiness bits. Optional
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
| Paired all-rank sweep report, both timing metrics | `plot_megamoe_ep_scaling.py throughput ROOT --scope SCOPE --output REPORT.html --render-script RENDER.py [--preview]` |
| AFD metric sweeps/comparisons/plots | `summarize_afd_sweep.py`, `compare_afd_sweep_metrics.py`, `plot_afd_sweep.py` |
| Explicit installed TRT-LLM DeepGEMM reference sweep | `trtllm_megamoe_sweep.py`, `trtllm_megamoe_sweep.sbatch`, `trtllm_megamoe_iket_nvcc.sh`, `iket_event_register_limits.py` |
| Archive complete cases/selected evidence | `archive_sweep_cases.py` |

Load the visualization skill before producing/updating visualizations. Reference
TRT-LLM sweeps use the original installed backend, not optimized AFD or TRT CuTe;
ordinary timing uses the untouched image. Its sequence command runs one EP job and
CPU analysis at a time, stopping on failure. `collect` exports the shared schema;
build reports with `--source-format trtllm-original`.

The `throughput` report mode requires passing summaries for every declared EP/case,
matching recorded GPU jobs, equal sample counts and matched preparation. It plots
both all-rank/both-order mean metrics, exports absolute latencies to CSV, and keeps
fixed-score, context and retained-control aggregates separate with regression counts.
It clears stale HTML/aggregate status before validation; coverage does not imply
performance acceptance. Historical rank-0 medians remain the default mode's metric.
For a separate historical row in the paired report, repeat `--history MODEL=ROOT`
for each model's audited ordinary sweep. It compares the historical rank-0 median
with the average of current forward/reverse rank-0 medians from the rank reviews.
Every declared comparison/context case is required; extra small controls remain
outside this historical score. The report labels the separate allocations/tuning,
retains every slower case, and exports absolute medians and ratios to `.history.csv`.

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
