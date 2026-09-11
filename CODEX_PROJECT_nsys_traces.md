# AFD artifact and Nsight agreement — applies to future sessions

Use this hierarchy for ALL retained AFD cases:
`scratch/afd/cases/<model>/isl_<tokens>/ep_<EP>/a<A>-f<F>/batch_<batch>/<optimization>/`.
Group by model, input length, FFN EP, reduced A:F GPU ratio, batch, then system
optimization. Example: A56:F16 is `a7-f2`; record exact GPU counts in the manifest.
For irregular input use `isl_<min>-<max>`. Batch is per attention DP lane.

```
scratch/
  afd/
    cases/
      qwen3/isl_131072/ep_8/a8-f1/batch_7/
        megamoe-optimized-20260909/
        attention-early-release-20260909/
        attention-graph-edges-20260910/
          attention.nsys-rep
          ffn.nsys-rep
          metric.json
          result.json
          manifest.json
      qwen35/isl_131072/ep_16/a3-f1/batch_38/
        attention-graph-edges-20260910/
    reports/<model>/
    validation/cleanup_20260910/
  megamoe/
    qwen3/
    qwen35/
```

Use meaningful optimization labels, with dates/job suffixes for repeated runs.
Do not silently overwrite an existing version. Record full execution settings
in its manifest: exact GPU counts, attention TP, microbatch count, placement,
weight dtype, real-prefill vs synthetic-KV perf mode, normal vs perfect routing,
memory ratio/pages and model/source revision. Use distinct version labels for
runs with different settings; compare performance only under matching contracts.

A trace-bearing version contains exactly TWO trace files: one representative
attention report and one FFN/model report. Prefer ranks selected by the strict
metric; read placement rather than assuming an FFN rank. Copy only these two
explicit files and retain full raw trace sets remotely. Never recursively copy
ray_logs/results directories. Do not copy extra rank traces, prompts, full sample
logs/scores, NVTX dumps, SQLite exports or unrelated files. Retrieve compact
results only when needed for the requested analysis.

Every retained performance version must contain its matching attention/FFN trace
pair alongside `metric.json`, optional canonical `result.json`, and `manifest.json`.
The user explicitly requested filling all JSON-only performance versions. Select
the exact two reports named in that version's metric, not reports from another
run of the same configuration. Alignment-only versions retain a compact alignment
result; unprofiled alignment runs have no Nsight traces. Shared reports go in afd/reports, validation plans
and check logs in afd/validation. Standalone MegaMoE sweeps go in megamoe/model.

After copying the two reports, verify they are nonempty and save a small manifest
with original remote host/report paths, roles/ranks/sizes, source revision/patch
or snapshot, job ID, original metric/result paths, settings and existing strict
latency/TPS/sample/outlier values. Do not change the measurement standard while
organizing files. Do not invent missing settings or passing performance status.

Keep accepted optimization milestones. Remove rejected/superseded v1/v2/etc
snapshot folders, invalid metrics, incomplete captures, stale plot backups,
one-shot scripts and copied raw payloads. Keep active validation intact. Do not
recreate dated experiment roots at scratch top level, and do not create a trace-
management script: this documented agreement is the future-session workflow.

September10 consolidation:71 performance configurations,207 performance versions,
207 trace pairs/414 reports;562MB total scratch after filling all JSON-only
performance versions. Each pair's path/rank matched its metric and its local size
matched the manifest; all reports are nonempty. Alignment-only versions
have no traces because profiling was disabled. Accepted final Q35
v9 is named ffn-two-streams-final-20260907. No transfer staging directory remains.
Migration/deletion records: scratch/afd/validation/cleanup_20260910/{deep_scratch_cleanup,
case_hierarchy_migration,cleanup_summary}.json. Historical docs' old local paths
are superseded; remote paths are unchanged. Current shared reports:
- scratch/afd/reports/qwen3/optimization-comparison.html
- scratch/afd/reports/qwen3/afd-vs-trtllm.html
- scratch/afd/reports/qwen3/{trtllm-baseline,vllm-baseline}.html
