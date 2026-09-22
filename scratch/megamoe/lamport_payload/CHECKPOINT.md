# All-push checkpoint — September 19, 2026

The retained SF-units implementation is being consolidated into maintained
**all-push**, selected with `MINISGL_MEGAMOE_PAYLOAD_MODE=all_push`. Pull remains
the default. The September18 remainder-helper source below this directory is
historical evidence, not the current winner.

Latest handoff: no commit; next agent continues. All new jobs use short.
Current model candidate is `source/all_push_signal_vector_20260919` on HSG.
Adapter checks pass; full-model8K alignment and final128K perf remain pending.
One-CTA reset overlaps norm work but outlasts it; compare4–8 reset CTAs next.
See the exploration handoff for exact evidence and remaining gates.

Current source map, limits, timing semantics and final validation status:
[project memory](../../../CODEX_PROJECT_push_dispatch.md).
Remaining communication/compute ideas:
[exploration](../../../MEGAMOE_COMM_EXPLORATION.md).
Final evidence: `scratch/megamoe/final_all_push_20260919` and HSG
`FastAFD/artifacts/final_all_push_20260919`.

The old remainder-helper comparison/confirmation/EP16 artifacts remain intact.
SF-units/latest full audit is in
`concurrent_dispatch/receiver_validate/fc2_work_balance/gpu_validation_repaired/RESULT.md`.
Frozen controls must retain their matching native descriptors and headers.
Do not rebuild a boxM1 host source into a boxM8 control cache.
