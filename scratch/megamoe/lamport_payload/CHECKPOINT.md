# September 18 MegaMoE checkpoint

The measured best is the **remainder_helper** overlay. The maintained opt-in
compact-push implementation is an earlier split-gather prototype; it is not the
37.167us result. Pull remains the default. No experiments are launched by this
checkpoint. Paths below are relative to this directory.

| Component | Frozen source |
|---|---|
| Device main, regular eight-row TMA | `concurrent_dispatch/receiver_validate/contiguous_tma/compact_main.cuh` |
| Matching native host descriptor | `concurrent_dispatch/receiver_validate/contiguous_tma/compact_host.hpp` |
| Native include entry points | `concurrent_dispatch/receiver_validate/contiguous_tma/{api_mega.hpp,python_api.cpp}` |
| Best dispatch, narrow unmarked remainder helper | `concurrent_dispatch/receiver_validate/remainder_helper/dispatch.cuh` |
| Router with concurrent SF pushes | `concurrent_dispatch/router_sf/router.cu` |
| Previous-best1536 dispatch | `concurrent_dispatch/receiver_validate/parallel_remainder/best1536_dispatch.cuh` |
| Parallel1024 control dispatch | `concurrent_dispatch/receiver_validate/parallel_remainder/dispatch.cuh` |
| Historical pull main with actual MMA marker | `history_actual/main.cuh` |
| Exact prepared-buffer smoke harness | `latency_overlap/afd_megamoe_same_rank_smoke.py` |

Overlay `headers/include/deep_gemm/impls/` maps the frozen main to
`sm100_fp8_fp4_mega_moe.cuh` and dispatch to `mega_moe_dispatch.cuh`; unchanged
headers come from the maintained DeepGEMM include tree. Native overlays must use
real `csrc/apis` entry-point files so their relative includes reach the matching
host header. `contiguous_tma/build_compact_commands.json` records native build
commands. Reuse the existing matching native cache or rebuild this descriptor;
the maintained boxM1 descriptor is not compatible with the best boxM8 path.

`remainder_helper/{build_commands,capture_cases,capture_plan,confirmation_plan}.json`
and `ep16_validation/{build_commands,capture_plan}.json` preserve exact executed
commands, shapes, overlays and environment. These are provenance manifests with
absolute HSG workspace paths, not portable launch defaults. Reusable launch and
analysis tools are in the repository's `codex_scripts/`; frozen profiler/wrapper
copies preserve the measured run. Large traces, caches and binaries stay remote.
Historical native/source controls remain in `artifacts/push_dispatch/pull_baseline`
and `source/comm_alignment`, as recorded in capture manifests. Raw traces reside at
`artifacts/push_dispatch/lamport_payload/concurrent_dispatch/receiver_validate/`
`remainder_helper/iket/<arm>/7257013` under the HSG FastAFD workspace.

The receiver validates every SYS-loaded payload word and materializes the exact
observed values before regular TMA. Prefix1024 and remainder3072 have independent
workers/readiness; acquiring the remainder does not certify the prefix. SF has
its own coordinator. Main/prefix work stays inline; only the batched two-row
remainder validator is noinline and has no profiling markers. This recovers
compute throughput while overlapping the independent work.

Current shape limits: M<=32,256 epilogue threads, one expert wave, and prepared
capacity (`rows + aligned pool <= 3840`). Qwen3.5 EP16 global912/rows57 is unsupported;
we explicitly tested global512/rows32 instead. Exact and replay checks passed,
but fixed balanced routing and one batch per model do not establish broad support.

Results and all three detailed EP8 captures are in
`remainder_helper/RESULT.md`, `comparison_7257013.md` and `ep16_validation/RESULT.md`.
Metadata-origin traces include dispatch metadata before MegaMoE entry through
combine completion. Ordinary MegaMoE kernel times exclude the preceding metadata.
Initialization/reset is outside both timed workloads. Independent phase medians
need not sum, and instrumented spans are separate from ordinary measurements.

No AGENTS.md or skill change is needed for this checkpoint. Task-specific protocol,
timing and source-map lessons belong in the project memory and these manifests.

Checkpoint verification (September 18): synced the candidate tools/code and
selected source/evidence assets to HSG `source/local-mirror`; remote `bash -n`
passed for `profile_megamoe_iket.sbatch`. The current latency summarizer replayed
all four saved EP8 arms (three captures each) with ordinary-pass validation and
produced `remainder_helper/checkpoint_comparison.{json,md}` on HSG. No GPU rerun
was needed: kernel sources were unchanged from the passing jobs above. Git
whitespace review passed after documentation cleanup. No global instruction or
skill change is needed; workflow lessons are recorded in project memory.
