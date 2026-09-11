# Publisher-entry AFD Pareto sweep — cancelled by user Sept10 2026

Current checkpoint (September 10): all requested work is complete. Final
scheduling and Qwen3.5 barrier fix are promoted into main; superseded snapshot
repos removed, source patches/results retained. See CODEX_PROJECT_attn_graph_edge.md.
The execution instructions and pending states below are historical.

LATEST: User cancelled this resweep to further optimize AFD. Five queuedjobs
7059772–7059776 cancelled beforeexecution; DO NOT submit prepared16/18jobs.
Remote submission_plan.json marks all7 cancelled; cancelled_by_user.json records
reason. This66case resweep is REMOVED from currentcompletion requirements.
Continue Qwen35 decomposition and targeted GPU validation7059597; mainmerge
still deferred. Historical preparation below retained solely for reproducibility.

User requested replacing standalone attention release kernel with release at
quant_publish entry, then explicitly requested another AFD sweep. Same66cases,
perf/no prefill/perfect balance, MB2, FP8xFP4,memory0.97,no pageoverride, explicit
overlap; one job per tray2/3/4/8/12/16/18,2hrshort. Preserve measured snapshots,
main merge deferred. Need compare all66 against previous separate-release66,
plus batch7 separately; existing HTMLs remain prior measured version until new
results exist. Whole active goal now includes this additional66-case sweep,
GPU validation, and finishing Qwen35299; original and earlier-attention132done.

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Source=R/source/attn_publish_entry_20260910 (full copy of prior measuredcandidate).
Task=R/artifacts/attn_publish_entry_pareto_perf_20260910
Local=scratch/attn_publish_entry_pareto_perf_20260910
Source overlay=scratch/attn_publish_entry_20260910/source
SmokeTask=R/artifacts/attn_publish_entry_20260910 (separate from fullsweep).
Control=R/artifacts/attn_release_pareto_perf_20260909

Runtime reuses fused quantize_publish_o_fp8_release_turn; CUDA block0/thread0
stores device-release turn before any quantization. No lastCTAquantization
counter atomics or standalone attention release kernel for Qwen3. Independent
lane O/staging buffers, producer stream ordering and publication-ready signal
preserve safety. Quantization counter allocation/API retained for memory parity.
Qwen35 branches unchanged. Main unchanged. Four remote CPU scheduler tests
PASS0.031s; existing transport source-contract updated and PASS0.020s.
GPU smoke7059597 PENDING behind EP8+EP4 decomposition:4GPU,8192rows,9iterations,
2fabricpairs,byte-exactFP8/UE8M0. Existing smoke launcher adapted ONLY isolated
source to onecontainer/srun for rank0prebuild and allranklifetimes. Job/env local
in scratch/attn_publish_entry_20260910/job.json. GPU success/perf not yet claimed.

At10:15PDT first5sweep jobs SUBMITTED with afterok:7059597 and
kill-on-invalid-dep=yes; cannot consume GPU before validation success:
2trays7059772(6cases),3trays7059773(6),4trays7059774(6),
8trays7059775(12),12trays7059776(12). Remaining16/18trays12cases each PREPARED
but not yet submitted because user max8outstanding (running+queued), currently
2decomp+1smoke+5AFD=8. Submit remaining as slots open. Account permits2RUNNING.
Allcommands/envfiles in submission_plan.json; append each acceptedjob jobs.jsonl
and set its planstatus/jobid immediately. Recheck squeue beforeeachsubmission.
No need change sourcelive; allrequired source synced beforeGPUjobs.
CASES.csv exactcopy66 including corrected A56:F16 active_gpus72 metadata.
Pool initialized empty; environment substitutions only task/source/cache paths.
Per-case uniform config checked. Existing run_afd_bundle already ONE live
container pernode/job. No holds, only validationdependency. No array/chaining.
Monitor5min nearcompletion,10min steady. Never restart activejobs for logtimeouts.

After GPU smoke terminal0:0 require AFD_FABRIC_TRANSPORT_SMOKE_OK marker and
record evidence. On failure, pendingdependentjobs auto-cancel; diagnosebefore
newsubmission. For remaining16/18jobs use prepared command/env, afterok stillvalid.
After each AFD completion collect strict metric/result with existing
codex_scripts/summarize_afd_sweep.py --baseline Control/report/metrics, retain
allregressions. Do not attribute node/time noise to code without evidence.
No extra microanalysis unlessneeded. Finish all299decomp and66newAFD,save
reports/plots, then completionaudit; no mainmerge or commit requested.

Publisher-entryGPUsmoke7059597 also cancelled as superseded by user-requested directgraph-edge experiment7059949. Do not resubmit. Publisher-entrysource remains unvalidatedonGPU and preservedonly as alternative.
