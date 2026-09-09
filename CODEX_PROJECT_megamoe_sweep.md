# MegaMoE decomposition sweep — complete 2026-09-09

## Final state

All164 unique inputs /455 AFD aliases complete: EP2/4/8/16/32/64 counts
86/42/20/8/4/4. Every case has one capture, three IKET samples, six marker
families, decomposition.json/.csv, and successful normal/profiled reports.
Final job7028073 COMPLETED0:0 in39:22 on nvl72056-T17; ended10:48:49 PDT.
Final poll10:52 PDT confirmed completion. No sweep jobs remain; unrelated
qwen35-full-smoke7028920 was running and was not modified.

CPU artifact audit passed164 contracts/492 samples, scope and deduplication,
all455 aliases, integer balance, six markers and per-sample phase closure.
Existing EP64 exact-combine reports passed for all four inputs. Final Slurm
records prove all15 successful capture jobs COMPLETED0:0,short QoS,1hr and
serial submissions. No extra GPU checks or new tuning during the sweep.
First eight7020835 captures have the older fixed-row metadata and no explicit
gap fields; audit validates their fixed rows and derives gaps from boundaries.

Results are local under scratch/megamoe_input_sweep_20260909/results/ and
remote under R/artifacts/megamoe_input_sweep_20260909. Key files:
- decomposition.csv: all164 aggregate rows; per-case CSVs retain all samples.
- equivalent_inputs.csv:455 AFD settings mapped to deduplicated cases.
- ep_scaling.csv: equal-global-work comparisons across all available EPs.
- max_inputs_ep4_ep64.csv: maximum batch7/maximum A:F table, EP4–64.
- fixed_global_tokens32.csv: largest shared global input across EP2–64.
- fixed_tokens_per_ffn_gpu4.csv: largest shared source rows across EP2–32.
- coverage.json, audit_summary.json, jobs.csv, README.md: final proof/context.
- megamoe-comparisons.html: self-contained report with three rows, tables left
  and full dispatch/compute/combine/tail lanes right; shared0–120us axis. Each
  chart uses the actual median-duration IKET launch, with communication/compute
  intersections hatched, dashed IKET end and solid separate Nsight median.
  Tables retain phase medians and add full/exposed/overlap communication values;
  overlap medians are calculated from each sample's interval intersections.
  Initial/gap time stays as blank timeline space. Selected-launch boundaries
  may differ from table medians. No new GPU run, browser check or git commit.
  Uses the three requested CSVs plus their per-case captures. Editable fragment is one level
  above results. Browser preview was blocked by sandbox; user explicitly
  declined further checking. No GPU jobs or new measurements for this report.
  First column now shows up to three representative AFD aliases per input:
  max A:F, max batch, and an interior middle ratio where available. Aliases
  come from equivalent_inputs.csv; same examples merge, ratio ties use the
  larger valid batch. Maxima are within that input's equivalent configurations.
  All16 displayed inputs annotated; timing values unchanged. Charts align to
  the expanded table rows; no browser checks per the user's instruction.
User explicitly excludes results and reports from the progress commit.
Generated CSVs, HTML, manifest, coverage and audit remain at the artifact paths
above. The commit contains source, scripts, tests and project memory only.
All16 rows across the three requested comparison tables match source metrics.
EP64 cannot match four source rows/rank: maximum valid count is0.5/rank.
EP2 at four source rows/rank has global8 and only64/128 active experts,
explaining EP2/4 compute49.184/47.840us in that growing-global-work comparison.

Largest EP2/global280: Nsight121.536us, exposed dispatch8.192, compute106.368,
exposed combine9.184, post-combine tail0.224us (component medians not additive).
EP64 median post-combine tails5.984–6.400us remain a future study issue.
IKET dispatch outliers remain intact, e.g. EP32/global32 maximum341.472us;
three samples do not establish frequency or cause. No pure-wire interpretation.

## Execution lessons and recovered attempts

Progress checkpoint: corrected stale source-text assertions to require the
retained ascending accumulation and EP2–64 harness range. Four focused remote
contract tests passed in0.068s; all seven changed/new shell launchers passed
remote bash -n. Relevant code/scripts synchronized before checks. Existing GPU
results/artifact audit remain unchanged; no extra GPU run or browser preview.
Retrospective: keep setup-amortization and snapshot/stdin lessons in project
memory; no global skills or AGENTS.md changes needed. Commit explicitly
authorized after sweep completion; unrelated Qwen3.5 progress preserved.

-7020713 canceled40s before measurement to fix sparse expert-rank balancing.
-7021305 failed143:0 in3:33 before kernel: candidate cloned HEAD had a20-arg
 Python wrapper against21-arg C++. Copying the actual mirrored runtime tree
 and overlaying local fixes repaired source consistency.
-7021751 failed143:0 in8:22: high ranks waited on ready=0. The one-row gate
 launched32 threads and notified only ranks0–31. Fixed both selector variants
 with CTA-strided destination publication; arithmetic/counts/IDs unchanged.
-EP64 additionally needed per-topk producer acquires above32 ranks; original
 uint32-mask combine path retained for EP<=32. Four EP64 cases passed byte-
 exact selected versus full-barrier combine on all64 ranks outside timing.
-7021990 completed tokens8 only in4:14: nested profiler/srun consumed the
 case-loop stdin. Redirecting profiler stdin from /dev/null fixed the launcher;
7022378 completed the remaining three in10:48. Successful tokens8 was reused.
-7023062 initially pinned trays later occupied by other users' long jobs.
 Cleared pending ReqNodeList, retaining segment2/8GPUs; it completed on another
 NVL72. Subsequently use scheduler-selected segments and record allocations.
-Short QoS rejected2GPU allocation with QOSMinGRES (no job created). EP2 uses
 one4GPU tray with2 active ranks. One manually submitted sweep job at a time;
 no automatic submissions, no changes to unrelated jobs, five-minute polling.
-Per-case profiler/model setup took roughly3min across EPs; EP2 occupied most
 sweep time because it has86 cases, not because kernel execution took minutes.
 Setup amortization is a future workflow improvement; no mid-sweep change.

Frozen runtime: R/source/megamoe_ep64_support_20260909; IKET dependencies
R/source/iket-smodel. Accepted source fixes and scripts are in the local
checkout and R/local-mirror. Raw traces/weights remain remote.

## Scope and user constraints

EP2/4/8/16/32/64; Qwen3-235B-A22B H4096/I1536,E128,topk8,FP8xFP4,
150 SM,bucket3840,prepared production routing. AFD MB2,batch1–7 (user
corrected6), equivalently1–4 tokens per attention GPU per microbatch.
Deduplicate inputs; distribute tokens across both ranks and experts as evenly
as integer counts permit. No tile/SM optimization or arithmetic changes.

At most one running job, one manually submitted sweep job at a time,
short QoS,one-hour limit,monitor every five minutes. No arrays, automatic
submission chains or subagents. Minimum
correctness checks; no py_check/hash/version audit. Plain ssh oci-hsg.
Preserve unrelated Qwen3.5 changes. AGENTS.md absent; earlier creation question
was unanswered, do not repeat. User-supplied global instructions still apply.

NVL72 integer A:F limits: attention GPUs=A*EP,total=(A+1)*EP<=72.
EP2/4/8/16/32 max A=35/17/8/3/1. Rows/rank=A*ceil(batch/2).
Unique cases=86/42/20/8/4. EP64 has no feasible A>=1 ratio (A1:F1 needs128
GPUs). Stated assumption after optional question: eight attention GPUs,
A1:F8, giving8/16/24/32 global tokens. Clearly label this exception.
Four EP64 cases make164 total,455 AFD aliases. Preserve this full scope.

EP64 uses one padded row per source with0/1 valid rows, retaining the real
8–32 global tokens. expert_map rotates the production rank*valid_rows
sequence to source-prefix offsets, then interleaves owner ranks:
expert=(sequence%EP)*(128/EP)+(sequence/EP), sequence modulo128.
CPU checks proved <=1 count spread across experts/ranks and unique topk IDs.
GPU harness checks finite output and exact generated IDs before timing.

## EP64 code and validation

Both original rank-gated combine paths used uint32 masks/32 waiter lanes,
which cannot represent ranks32–63. Local/candidate kernel now has per-valid-
topk-lane producer acquires for EP>32; the existing mask path remains for
EP<=32. CTA barrier / __syncwarp follows acquires. Slot arithmetic unchanged.
All four EP64 cases GPU-validated; accepted into local-mirror.

MEGAMOE_SMOKE_COMPARE_COMBINE=1 byte-compares selected output to existing
full-barrier combine outside timing. All64-rank checks and all four case artifacts passed. Native EP64 kernel did not run in7021305.
Accepted files synchronized, preserving user changes. All planned cases are now complete.
Do not rerun successful cases; original captures and per-case CSVs are retained.

## Files and interpretation

R=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD
Local accepted checkout / remote local-mirror; variants under R/source;
all remote output under R/artifacts/megamoe_input_sweep_20260909.
Local manifest and results: scratch/megamoe_input_sweep_20260909/.
Scripts: codex_scripts/megamoe_input_sweep.py (plan/collect/audit),
megamoe_input_sweep.sbatch (explicit case chunk; no self-submission),
profile_megamoe_iket.sbatch (normal Nsight plus six-point IKET,per-case CSV).
Scripts/harness and accepted fixes are synchronized local,candidate,local-mirror. Candidate syntax and CUTLASS/fmt headers checked remotely.

Collect only decomposition.json/.csv,baseline.json,profiled.json,and Nsight
kernel summary CSVs with filtered rsync; raw traces/weights remain remote.
Collector writes aggregate decomposition.csv,equivalent_inputs.csv,coverage.json.
Nsight is separate uninstrumented rank0 kernel wall time; IKET gives rank0
phase envelopes. Compute includes internal stalls. Exposed comm includes
readiness/local work. Do not add full overlapping spans or component medians
and claim an exact representative total. Preserve all three samples/outliers.

Job history:7020713 canceled after40s before measurement to balance sparse
expert ownership across ranks;7020835 eight cases complete;7021305 source
wrapper mismatch fixed as above. Detailed historical preparation retained in
scratch/megamoe_input_sweep_20260909/preparation_history.md.
Snapshot lesson: git clone copies HEAD, not the mirrored working tree; overlay
actual runtime files before candidate runs. Copying only local git-status
changes misses committed local changes absent from remote mirror HEAD.

## Fixed-work scaling collection

Collector now also writes ep_scaling.csv for every global-token count with
at least2 completed EPs, using lowest available EP as baseline. Local/mirror
collector executed on all164 cases; no GPU reruns for postprocessing.
Global128 compute medians EP2/4/8/16/32=98.816/49.920/24.352/13.568/8.864us;
Nsight113.504/66.464/41.824/31.808/28.160us. Near inverse-EP compute at low
EP, then diminishing benefit. No pure-wire/infinite-EP intercept claim.

User asked why still EP2 at09:21 PDT. Explained86/164 unique cases arise from
A1:F1–A35:F1, all higher EP complete, ~3min/case profiler startup/capture.
Those questions did not cancel or narrow the sweep; all86 EP2 inputs completed.

Final allocation metadata for all15 successful capture jobs is in
results/jobs.psv and jobs.csv. All COMPLETED0:0,short QoS,1hr; each successful
job was submitted after the previous successful job ended. Final audit passed.

User requested max-batch/max-A:F EP4–64 decomposition table while EP2 runs.
Used existing captures, no new GPU job. Saved results/max_inputs_ep4_ep64.csv
locally/remotely and showed table: batch7/MB2, EP4 A17:F1/global272,EP8
A8:F1/256,EP16 A3:F1/192,EP32 A1:F1/128,EP64 A1:F8/32 exception.
Table uses phase medians and separate Nsight total (not additive); highlighted
EP64 post-combine tail6.24us and EP4 exposed combine10.30us.
