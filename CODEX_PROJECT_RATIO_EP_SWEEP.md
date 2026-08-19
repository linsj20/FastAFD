# Qwen3 A:F-ratio / baseline-EP ISL sweep

## Goal and locations

Measure Qwen3 uniform ISL at 1K/2K/4K/8K/16K/32K/64K/128K for every
AFD A:F tray ratio 1:1 through 17:1, with one FFN tray/four FFN GPUs fixed,
and for vLLM DP=EP4/8/16/32/64. Every topology uses the exact maximum batch
admitted by its observed post-init KV capacity. Preserve every raw run tree,
including rejected attempts.

- Local task root: `scratch/ratio_ep_isl_sweep_20260717_1548`.
- Remote task root:
  `/home/shengjiel/scratch/fastafd_reproduce/ratio_ep_isl_sweep_20260717_1548`.
- Detailed chronological evidence: the task root's `CODEX_PROJECT.md`,
  `CODEX_PROJECT_REPRODUCE.md`, and `HANDOFF.md`.
- Raw results stay under the remote task root. Do not delete or compress them
  without explicit user approval.

## Current status (live; updated 2026-07-21 22:14 PDT)

- Baseline EP sweep: 40/40 accepted and complete; do not rerun it.
- Full matrix: 171/176 accepted after two stable fail-closed collection passes.
  Stable hashes are `d745e722...` for `report/audit.json` and `ff308dcb...`
  for `report/all_results.csv`.
- Fourteen-node A13:F1 16K/32K retries `5499401`/`5499444` and 8K A17:F1 job
  `5485585` passed complete audits as results 155--157/176.
- The old-path compatibility symlink was removed after both legacy topology
  jobs left the queue. The canonical renamed task tree and every raw run remain
  preserved. Profile-incomplete 16K attempt `5499083` is retained under
  `results/rejected_profile_incomplete/` so it cannot duplicate the accepted
  key.
- Final A13:F1 64K job `5502013` passed strict artifact audit at 486.8901
  TPS/GPU and 22.885774 ms and is accepted as result 159/176, closing A13.
- A14:F1 16K job `5504114` passed strict audit at 1,257.7252 TPS/GPU and
  37.104024 ms and is accepted as result 160/176.
- A14:F1 32K job `5504115` passed strict audit at 804.7840 TPS/GPU and
  28.993288 ms and is accepted as result 161/176.
- A14:F1 64K job `5504698` passed strict audit at 459.5422 TPS/GPU and
  24.372079 ms and is accepted as result 162/176.
- A14:F1 128K job `5504699` passed strict audit at 247.7096 TPS/GPU and
  22.607114 ms and is accepted as result 163/176, completing all A14 points.
- A15:F1 16K/32K jobs `5506276`/`5506277` passed strict audits at
  1,238.9189/825.3282 TPS/GPU and are accepted as results 164--165/176.
- A15:F1 64K job `5506451` passed strict audit at 476.8816 TPS/GPU and
  23.590759 ms and is accepted as result 166/176.
- A15:F1 128K job `5515660` passed its exact b6 capacity gate, then all sixty
  attention ranks deadlocked at DeepGEMM barrier counter 186272 and hit the
  300-second NVLink timeout. It had no Xid, result, or SUCCESS artifact and no
  log progress after the failure; it was cancelled at 47:39 and its raw tree
  remains preserved. Its later unchanged retry succeeded, showing this was a
  transient/runtime-specific incident rather than a reproducible capacity bug.
- A15:F1 128K retry `5539628` completed `0:0` in 18:38 and passed strict audit
  at 250.8939 TPS/GPU and 22.419840 ms, becoming result 171/176. It has exact
  b6 capacity, two complete windows, 1,800 replays, 65 CPU traces, all 64 rank
  plus three coordinator nsys reports, sixteen clean snapshot pairs, verified
  inputs, and zero temporary or scoped-error artifacts.
- A16:F1 16K/32K/64K jobs `5515709`/`5529093`/`5530349` passed strict audits
  at 1,172.0696/783.1286/467.9003 TPS/GPU and are accepted as results
  167--169/176. Each has two complete windows, 1,920 replays, 69 CPU traces,
  68 rank plus three coordinator nsys reports, 17 clean snapshot pairs,
  verified inputs, and zero scoped errors or temporary artifacts.
- A16:F1 128K job `5539085` passed strict audit at 248.3381 TPS/GPU and
  22.739394 ms and is accepted as result 170/176. It has exact b6 capacity,
  two complete windows, 1,920 replays, 69 CPU traces, 68 rank plus three
  coordinator nsys reports, 17 clean snapshot pairs, verified inputs, and no
  temporary or scoped-error artifacts.
- The user lifted the four-job ceiling for final closure. Pending unstarted
  A17:F1 64K job `5539595` was cancelled to prioritize A15:F1 128K retry
  `5539628`, which is now accepted. The five remaining A17:F1
  4K/16K/32K/64K/128K jobs `5502011`/`5539086`/`5539087`/`5539629`/`5539630`
  are all submitted and pending as of 22:14 PDT. They use the merged launcher,
  isolated v3 source, one hour, null requested nodelists, and unchanged
  exclusions.
- Seven-tray 128K 6:1 retry `5488825` passed as result 129/176 at 243.5799
  TPS/GPU and 21.113632 ms. Seven-tray 64K attempt `5488676` completed all
  sampling/replay and all per-rank trace flushes but hung indefinitely in the
  unbounded worker-shutdown wait; it was cancelled after 30:21 and retained.
- Eight-tray 16K b50/n8 job `5489732` and 32K b25/n8 job `5489734` passed as
  results 131--132/176. Exact 128K b6/n8 job `5490110` passed as result
  133/176 at 247.0877 TPS/GPU and 21.247514 ms. Parallel exact 64K b12/n8 job
  `5490082` encountered fresh Xid 43 events on all four GPUs of
  `nvl72051-T08`, followed by an all-rank NVLink barrier timeout before any
  replay; it was cancelled and retained. Exact retry `5490414` passed on the
  identical allocation as result 134/176 at 496.7173 TPS/GPU and 21.138786 ms,
  proving the first incident transient.

## Consolidated controls and shutdown fix

The reusable ratio/EP behavior is merged into the two major launchers:

- `scripts/experiments/afd/oci_hsg/run_afd_reproduce.sh` supports A:F topology metadata,
  nominal and observed scheduler-admissible capacity maxima, a pre-sample
  all-attention-worker capacity gate, task-scoped result roots, and precise
  validation of a single patched coordinator file.
- `scripts/experiments/afd/oci_hsg/run_vllm.sh` supports capacity-max batches
  for EP4/8/16/32/64 without the isolated array/recovery special cases.

Local and remote validation passed all 176 dry-run points, all 48 one-past
rejections, eight explicit below-nominal AFD probes, shell parsing, and seven
embedded Python blocks. Remote launcher hashes are `60e5e92c...` (AFD) and
`867b5b89...` (vLLM).

The pinned source remains untouched at commit `3c716194`. Isolated source
copies under the task root have only `python/minisgl/afd_coordinator.py`
modified by `control/afd_worker_hot_loop_timeout.patch`: the hard-coded five-second Ray
worker-hot-loop wait is controlled by
`MINISGL_AFD_WORKER_HOT_LOOP_SHUTDOWN_TIMEOUT_S` with default 300 seconds, and
the previously unbounded `worker.shutdown()` wait is controlled by
`MINISGL_AFD_WORKER_SHUTDOWN_TIMEOUT_S` with default 120 seconds. The v3 patch
makes the ZeroMQ stop broadcast idempotent in both supervisor/`ExitMsg` race
orders. The current coordinator SHA-256 is `782d5287...`; the launcher validates that hash and
rejects any other dirty or untracked source state. The outer AFD stop allowance
remains 600 seconds and every Slurm job remains exactly one hour.

## Execution invariants and next action

- The historical four-live-job ceiling was explicitly lifted by the user for
  final closure on 2026-07-21; all five remaining points may be live together.
  No arrays or controller submissions.
- Exactly `01:00:00`; no requested nodelist; only evidence-backed exclusions.
- Review every terminal run before accepting it. All remaining points are now
  submitted, so no refill is needed unless a retry becomes necessary.
- Monitor after two minutes while running or transitioning; after five minutes
  only when every live job is steadily queued.
- Nine-tray 16K A8:F1 b50 job `5490717` and guarded 32K b25 retry `5491243`
  passed full review and stable collection as results 135--136/176. Final
  nine-tray 64K/128K jobs `5491574`/`5491576` passed and closed the stage as
  results 137--138/176. Ten-tray 16K/32K jobs `5492038`/`5492039` passed as
  results 139--140/176, and 128K b6 job `5492965` passed as result 141/176.
  Initial 64K job `5492964` exposed the reverse shutdown race and is retained;
  exact v3 retry `5493369` passed as result 142/176. Eleven-tray 16K job
  `5493742`/`5493743` passed as results 143--144/176. Final eleven-tray
  64K/128K jobs `5494214`/`5494215` passed as results 145--146/176. Twelve-tray
  16K/32K jobs `5494517`/`5494518` passed as results 147--148/176. Final
  twelve-tray 64K/128K jobs `5494868`/`5494870` are queued.

Exact retry job `5486348` was submitted at 12:43 PDT through the merged major
launcher. Slurm proves four nodes, `TimeLimit=01:00:00`, `ReqNodeList=(null)`,
the exact exclusions, task result root, isolated source path/hash, b25,
A3:F1, and the two 600/300-second shutdown allowances. It was initially
pending as the third live job; keep the fourth slot empty until review.

Job `5486348` started on freely selected `nvl72002-T[01-02,05,10]`. At 4:35,
all 12 attention workers had reported capacity with a minimum of 839,424
tokens; the pre-sample gate proved requested b25 equals both the observed
scheduler-admissible maximum and raw KV maximum. The run had zero error
signatures and remained in progress.

At 11:48 elapsed, the retry had generated all 300 samples, every attention
worker had 30 target replays (360 total), the coordinator trace was a nonempty
18.6 MB JSON, no trace `.tmp` files remained, and both `afd-result.json` and
`SUCCESS` existed. Slurm/final artifact and two-pass collector review were
still pending; do not fill the spare slot before they pass.

Job `5486348` completed `0:0` in 12:10 and measures 850.3801 TPS/GPU at
22.048963 ms median. It has two complete 15-step waves, no partial wave,
all 12 attention plus four MLP plus coordinator CPU traces finalized, four
clean final GPU snapshots, and recorded hashes `fb055bb0...` for the patched
coordinator and `19effcb6...` for the merged launcher. Two collector passes
were byte-identical at 116/176: `01bcdc28...` (`audit.json`) and
`c3a8ba11...` (`all_results.csv`). The shutdown-fix gate is cleared.

Four-tray 64K b12 job `5486625` and 128K b6 job `5486626` were submitted at
13:01 PDT. Both use A3:F1, four nodes, one hour, null requested nodelists,
exact exclusions, pinned YaRN profiles, the isolated source/hash, and the
merged launcher. Together with queued 17:1 jobs `5479092`/`5485585`, the live
count is exactly four. Both new jobs were initially pending.

Both final four-tray jobs started on freely selected nodes. Job `5486625` uses
`nvl72069-T[04-06,09]`; job `5486626` uses `nvl72009-T[15-18]`. Their
post-init gates each observed 839,424 tokens on every one of the 12 attention
workers and proved exact scheduler-admissible/raw-KV maxima of b12 and b6,
respectively. There were no runtime error signatures at the gate; keep
monitoring to completion before opening the five-tray stage.

Both completed `0:0`: 64K job `5486625` measured 460.7455 TPS/GPU at
19.533560 ms, and 128K job `5486626` measured 225.8190 TPS/GPU at 19.927468
ms. Each has two complete waves, no partial wave, 12 attention logs with 30
target replays apiece, all 17 CPU traces finalized, four clean final GPU
snapshots, exact source/launcher hashes, and only the known Ray warning. Two
collector passes were byte-identical at 118/176: audit hash `8e79265d...` and
all-results hash `db9781fe...`.

The five-tray stage opened with exact 16K A4:F1 b50 job `5486905` and 32K
A4:F1 b25 job `5486906`. Both are manual non-array five-node jobs with one
hour, null requested nodelist, exact exclusions, isolated source/hash, and the
merged launcher. They were initially pending; with the two retained 17:1
waiters, live count remains exactly four.

Both started together on freely selected nodes (`nvl72042-T[04,06-08,10]`
and `nvl72078-T[02-05,14]`). All 16 attention workers in each allocation
reported the expected 839,424-token minimum; the gates proved exact b50 and
b25, respectively, with no startup error signature.

Both completed `0:0`: 16K job `5486905` measured 1,615.9730 TPS/GPU at
24.752889 ms; 32K job `5486906` measured 871.0218 TPS/GPU at 22.961539 ms.
Each has two complete waves, no partial wave, 16 attention logs with 30 target
replays each, all 21 CPU traces finalized, five clean final GPU snapshots,
exact source/launcher hashes, and only the known Ray warning. Two collector
passes were byte-identical at 120/176: audit `cbc89103...`, all-results
`9598cdec...`.

The remaining five-tray points are exact 64K b12 job `5487113` and 128K b6
job `5487114`, both A4:F1. They were manually submitted on five nodes with one
hour, null requested nodelist, exact exclusions, isolated source/hash, and the
merged launcher; both were initially pending.

Both started together on freely selected nodes (`nvl72042-T[06-08,10-11]`
and `nvl72009-T[03-04,15-17]`). All 16 attention workers in each allocation
reported the expected 839,424-token minimum; the gates proved exact b12 and
b6 with no startup error signature.

Both completed `0:0`: 64K job `5487113` measured 471.5886 TPS/GPU at
20.356726 ms; 128K job `5487114` measured 239.8768 TPS/GPU at 20.010273 ms.
Each has two complete waves, no partial wave, 16 attention logs with 30 target
replays each, all 21 CPU traces finalized, five clean final GPU snapshots,
exact hashes, and only the known Ray warning. Two collector passes were
byte-identical at 122/176: audit `b28802d3...`, all-results `48890026...`.

The six-tray stage opened with exact 16K A5:F1 b50 job `5487390` and 32K
A5:F1 b25 job `5487391`. Both are manual six-node jobs with one hour, null
requested nodelist, exact exclusions, isolated source/hash, and the merged
launcher. Both were initially pending; live count is four with the 17:1 jobs.

Both started together on freely selected nodes (`nvl72137-T[01-06]` and
`nvl72095-T[05,08-10,12-13]`). All 20 attention workers in each allocation
reported 839,424 tokens minimum; the exact b50/b25 runtime gates passed with
no startup error signature.

Both completed `0:0`: 16K job `5487390` measured 1,683.5058 TPS/GPU at
24.749939 ms; 32K job `5487391` measured 880.7014 TPS/GPU at 23.655387 ms.
Each has two complete waves, no partial wave, 20 attention logs with 30 target
replays each, all 25 CPU traces finalized, six clean final GPU snapshots,
exact hashes, and only the known Ray warning. Two collector passes were
byte-identical at 124/176: audit `0c72218c...`, all-results `aa01c139...`.

The remaining six-tray points are exact 64K A5:F1 b12 job `5487712` and 128K
A5:F1 b6 job `5487713`. Both are manual six-node jobs with one hour, null
requested nodelist, exact exclusions, isolated source/hash, and the merged
launcher; both were initially pending.

Both started together on freely selected nodes (`nvl72097-T[04-07,09,11]`
and `nvl72005-T[04-05,07-09,11]`). All 20 attention workers in each
allocation reported 839,424 tokens minimum; exact b12/b6 gates passed with no
startup error signature.

The 64K job completed both replay waves and produced a result, but the 128K
job failed before any replay: all 20 attention workers report
`torch.AcceleratorError: CUDA error: unspecified launch failure` during the
first eager attention forward, and allocated-node kernel logs record fresh
Xid 43 events at the same time. This is not a capacity-gate or shutdown-wait
failure. Preserve the run, keep the stage stopped, await its final Slurm exit,
then retry exact 128K A5:F1 b6 on a new free allocation before advancing.

Job `5487712` completed `0:0` and measures 487.7125 TPS/GPU at 20.503881 ms;
it has two complete waves, 20 attention logs with 30 replays each, 25 final
CPU traces, six clean snapshots, and exact hashes. The dead 128K job remained
hung after every attention actor exited, so Codex canceled only `5487713`
after 19:48; its failed tree is retained. Two collector passes accepted only
the valid 64K point at 125/176, with audit `6f75bdec...` and all-results
`bf445dfb...`. Exact 128K A5:F1 b6 retry `5488171` was submitted alone on six
nodes and initially pending; seven trays remains blocked.

Slurm reassigned the retry to the same freely selected six trays
`nvl72005-T[04-05,07-09,11]`; no requested nodelist or unsupported exclusion
was added. All 20 attention workers again report 839,424 tokens minimum and
exact b6, and the service is ready with zero tracebacks/launch failures at the
gate. This is therefore a strict same-allocation reproducibility test.

The retry completed `0:0` on the identical allocation without any Xid,
traceback, or launch failure, proving the first event transient. It measures
243.4174 TPS/GPU at 20.540851 ms, with two complete waves, all 20 attention
logs at 30 replays, 25 final CPU traces, six clean snapshots, and exact
hashes. Two collector passes were byte-identical at 126/176: audit
`c7315891...`, all-results `d6ba16a8...`. This closes six trays.

Seven trays opened with exact 16K A6:F1 b50 job `5488419` and 32K A6:F1 b25
job `5488421`. Both are manual seven-node, one-hour jobs with null requested
nodelists, exact exclusions, isolated source/hash, and merged launcher. Both
were initially pending; live count remains four with the retained 17:1 jobs.

Both started on freely selected nodes (`nvl72005-T[04-05,07-11]` and
`nvl72116-T[01-03,05,07,09-10]`). All 24 attention workers in each allocation
reported 839,424 tokens minimum; exact b50/b25 gates passed with zero
tracebacks or launch failures.

Both completed `0:0`: 16K job `5488419` measured 1,670.8875 TPS/GPU at
25.649330 ms; 32K job `5488421` measured 911.5154 TPS/GPU at 23.508732 ms.
Each has two complete waves, no partial wave, 24 attention logs with 30 target
replays each, all 29 CPU traces finalized, seven clean final GPU snapshots,
exact hashes, and only the known Ray warning. Two collector passes were
byte-identical at 128/176: audit `00707c66...`, all-results `c6220063...`.

The remaining seven-tray points are exact 64K A6:F1 b12 job `5488676` and
128K A6:F1 b6 job `5488677`. Both are manual seven-node, one-hour jobs with
null requested nodelists, exact exclusions, isolated source/hash, and merged
launcher. Both were initially pending; live count remains four.

Both started on freely selected nodes (`nvl72005-T[04-05,07-11]` and
`nvl72040-T[01-07]`). All 24 attention workers in each allocation reported
839,424 tokens minimum; exact b12/b6 gates passed with zero tracebacks or
launch failures.

Job `5488677` then failed fast `2:0` before sampling because its head-node API
server could not bind validated port 25416 (`EADDRINUSE`). It has no worker
traceback or result; preserve the run. This is a control-port collision, not a
capacity/model failure. Exact 128K A6:F1 b6 retry `5488825` was submitted
alone in that lane; its different job ID chooses a different deterministic
seven-port block. Job `5488676` continues independently; eight trays remains
blocked until the retry passes.

Retry `5488825` started on freely selected `nvl72018-T[05-09,11-12]`. Its
new control block is `26600-26606` with no bind error. All 24 attention
workers passed the post-init gate; observed minimum capacity is 837,696 tokens
(slightly below nominal but still exact scheduler-admissible b6), with zero
tracebacks. Continue the exact retry; do not open eight trays yet.

Retry `5488825` completed `0:0` in 13:25 and measures 243.5799 TPS/GPU at
21.113632 ms. It has two complete waves, no partials, all 24 attention logs at
30 target replays, 29 final CPU traces including the coordinator, seven clean
snapshots, exact source/launcher hashes, and no runtime error signature. Two
collector passes were byte-identical at 129/176: audit `f2777d4d...` and
all-results `f841804d...`.

The parallel 64K job `5488676` completed sampling and all 24 attention plus
four MLP trace flushes, then remained inside `AfdCoordinator.shutdown` at the
unbounded `ray.get(shutdown_refs)` call. It never produced a coordinator trace
or result and made no progress for more than 22 minutes, so Codex cancelled it
at 30:21 while preserving the full raw tree. This is an intermittent worker
resource-teardown defect, not a capacity or measurement failure.

The isolated coordinator now bounds that worker-shutdown wait at 120 seconds
via validated `MINISGL_AFD_WORKER_SHUTDOWN_TIMEOUT_S`, after which the existing
explicit actor-kill and final coordinator-trace path runs. The source hash is
`71f5a578...` and merged AFD launcher hash is `60e5e92c...`. Local and remote
syntax/compile/reverse-patch checks and the complete 176 dry-run/48 rejection
suite pass. Exact 64K A6:F1 b12/n7 retry `5489361` was submitted with the new
hash and exact exclusions; it was initially pending with one hour and a null
requested nodelist. Seven trays and all later stages remain gated on it.

Retry `5489361` started on freely selected `nvl72040-T[01-07]`, passed exact
b12 at 839,424 tokens on all 24 attention workers, and completed `0:0` in
10:14. It measures 493.0125 TPS/GPU at 20.862990 ms, with two complete waves,
all 29 CPU traces, complete nsys artifacts, seven clean snapshots, exact new
hashes, and no error or shutdown-timeout signature. Two collector passes were
stable at 130/176: audit `d36df567...`, all-results `f26ae867...`. Seven trays
is closed.

Eight trays opened with exact 16K A7:F1 b50 job `5489732` and 32K A7:F1 b25
job `5489734`. Both are manual eight-node jobs with one hour, null requested
nodelists, exact exclusions, patched source/hash, and the merged launcher.
Both were initially pending; together with 5479092/5485585 the live count is
exactly four.

Jobs `5489732`/`5489734` started together on freely selected
`nvl72040-T[01-08]` and `nvl72056-T[02-09]`; all 28 attention workers in each
reported 839,424 tokens and the b50/b25 exact gates passed. Both completed
`0:0`: 16K measures 1,640.3677 TPS/GPU at 26.670850 ms, while 32K measures
903.4614 TPS/GPU at 24.212435 ms. Each has two complete waves, all 33 CPU
traces, complete nsys artifacts, eight clean snapshots, exact hashes, and no
error/timeout signature. Two stable collector passes reached 132/176 with
audit `5e60a5db...` and all-results `73f3b7b3...`.

The final eight-tray pair is exact 64K A7:F1 b12 job `5490082` and 128K
A7:F1 b6 job `5490110`. Both are manual eight-node jobs with one hour, null
requested nodelists, exact exclusions, patched source/hash, and the merged
launcher. Both were initially pending; with 5479092/5485585 the live count is
exactly four.

Job `5490110` completed `0:0` in 10:54 after passing exact b6 with an
839,424-token minimum. It measures 247.0877 TPS/GPU at 21.247514 ms, with 840
target replays, all 33 CPU traces, complete nsys artifacts, eight clean
snapshots, and no error/timeout signature. Two stable collector passes
accepted it as 133/176 with audit `b961b6e6...` and all-results
`de96f47f...`.

Job `5490082` never reached replay: all ranks timed out in the 300-second
DeepGEMM NVLink barrier at counter 118484 after all four GPUs on allocated
`nvl72051-T08` logged fresh Xid 43 events and workers reported CUDA launch
failure/fatal abort. It was cancelled after the failure was conclusive and
its full raw tree is retained. Since an earlier same-allocation Xid retry
succeeded, T08 is not yet excluded. Exact b12/n8 retry `5490414` was submitted
at 16:46 PDT through the merged launcher with the unchanged exclusions,
one-hour limit, null requested nodelist, and patched source/hash. It is pending;
nine trays remains blocked.

Retry `5490414` ran on the identical `nvl72051-T[03-10]` allocation and
completed `0:0` in 10:49. All 28 attention workers passed exact b12 at a
minimum 839,424-token capacity and recorded exactly 30 target replays (840
total). The result is 496.7173 TPS/GPU at 21.138786 ms, with all 33 nonempty
CPU traces, 35 nsys reports, eight clean final snapshots, exact input hashes,
no partial files, and no Xid/CUDA/NVLink/runtime error. Two stable collectors
reached 134/176 with audit `48d0530b...` and all-results `0421e3ac...`.

Nine trays opened with exact 16K A8:F1 b50 job `5490717` and 32K A8:F1 b25
job `5490719`. Both are manual nine-node jobs with one hour, null requested
nodelists, exact unchanged exclusions, merged launcher, and patched
source/hash. They were initially pending; together with 5479092/5485585 the
live count is exactly four.

Nine-tray 16K job `5490717` completed `0:0` in 13:56 at 1,558.7679 TPS/GPU
and 28.512548 ms. It passed exact b50 at 839,424 tokens and has 960 target
replays, all 37 nonempty CPU traces, 39 nsys reports, nine clean final
snapshots, exact hashes, zero partial files, and clean scoped logs.

Parallel 32K job `5490719` passed exact b25, completed its batch request, and
recorded all 960 replays plus all 36 worker traces. Its POST returned `200`
and every worker logged `hot_rpc_loop:stop`, but coordinator shutdown never
started: no coordinator trace or final snapshots appeared. The prior patch's
worker waits were therefore downstream of the hang. A live process inspection
also showed the outer 600-second guard blocked in its own `ps` probe. The job
was cancelled at 27:17 and its raw tree retained.

The full coordinator patch now records shutdown entry and skips the second
ZeroMQ `AfdStopCmd` broadcast when `ExitMsg` already requested shutdown. Fresh
isolated source `FastAFD-3c716194-shutdown-v2` has only that coordinator
modified, hash `e5555f77...`, and passes apply/reverse, diff, compile, all 176
dry runs, all 48 one-past rejections, and all probe checks. Exact 32K b25/n9
retry `5491243` was submitted at 17:41 PDT; nine-tray 64K/128K remains blocked.

Guarded retry `5491243` completed `0:0` in 13:38 and proves the v2 shutdown
guard: exact b25 passed at 839,424 tokens, performance is 891.8103 TPS/GPU at
24.918104 ms, all 960 replays and 37 nonempty CPU traces finalized, 39 nsys
reports and nine clean snapshots exist, and the coordinator logged shutdown
entry before its trace flush. Hashes match and scoped logs are clean. Two
stable collectors accepted `5490717` and `5491243` as results 135--136/176:
audit `2d60ca58...`, all-results `e3dd22c5...`. Exact final nine-tray 64K b12
job `5491574` and 128K b6 job `5491576` were submitted at 18:01 PDT with the
v2 source/hash, one hour, null requested nodelists, and unchanged exclusions.

Nine-tray 64K job `5491574` passed exact b12 at 839,424 tokens and completed
`0:0` in 10:36 at 492.5434 TPS/GPU and 21.656297 ms. It has two complete
15-step windows, 960 replays, all 37 nonempty CPU traces, 39 nsys reports,
nine clean final snapshots, exact hashes, zero partial files, and no scoped
error. Parallel 128K b6 job `5491576` then started on the same freely selected
nine-tray allocation; stage collection and ten trays remain blocked.

Nine-tray 128K job `5491576` passed exact b6 at 839,424 tokens and completed
`0:0` in 11:45 at 246.4901 TPS/GPU and 21.637113 ms. It has two complete
15-step windows, 960 replays, all 37 CPU traces, 39 nsys reports, nine clean
snapshots, matching hashes, no partial files, and no scoped error. Two stable
collectors accepted `5491574`/`5491576` as results 137--138/176: audit
`31006d25...`, all-results `0375c42d...`. Exact ten-tray 16K A9:F1 b50 job
`5492038` and 32K b25 job `5492039` were submitted at 18:33 PDT with the v2
source/hash, one hour, null requested nodelists, and unchanged exclusions.

Ten-tray 16K job `5492038` passed exact b50 with an observed 839,104-token
minimum and completed `0:0` in 14:08 at 1,558.1271 TPS/GPU and 28.880828 ms.
It has two complete 15-step windows, 1,080 replays, all 41 CPU traces, 43 nsys
reports, ten clean snapshots, exact hashes, zero partial files, and no scoped
error. The parallel 32K job `5492039` remains queued; the final pair is gated.

Ten-tray 32K job `5492039` passed exact b25 at 839,424 tokens and completed
`0:0` in 14:37 at 881.0327 TPS/GPU and 25.538212 ms. It has two complete
windows, 1,080 replays, all 41 CPU traces, 43 nsys reports, ten clean snapshots,
matching hashes, no partial files, and no scoped error. Two stable collectors
accepted `5492038`/`5492039` as results 139--140/176: audit `d46d72f8...`,
all-results `4947123a...`. Exact final ten-tray 64K b12 job `5492964` and 128K
b6 job `5492965` were submitted at 19:22 PDT with all standard invariants.

Ten-tray 128K job `5492965` passed exact b6 at an observed 839,040-token
minimum and completed `0:0` in 15:00 at 248.9972 TPS/GPU and 21.686991 ms. It
has two complete windows, all 1,080 replays, 41 CPU traces, all 40 worker nsys
reports plus two coordinator reports, ten clean snapshots, matching hashes,
and no partial/error evidence. Parallel 64K job `5492964` completed all compute
and 40 worker traces but failed `3:0` before the coordinator trace. Evidence
shows supervisor shutdown won the race (`already_requested=0`), then the event
loop consumed `ExitMsg` and issued a second blocking stop broadcast. V3 makes
the `ExitMsg` branch idempotent as well; fresh source hash `782d5287...` passes
compile, patch/reverse checks, and the full validation suite. Two collectors
accepted 128K as result 141/176 (`96c83c9e...`, `91bd5a75...`). Exact v3 64K
retry `5493369` was submitted at 19:46 PDT; eleven trays remains blocked.

Ten-tray 64K v3 retry `5493369` passed exact b12 at an observed minimum
838,912-token capacity and completed `0:0` in 14:15 at 481.7767 TPS/GPU and
22.417023 ms. It has two complete windows, all 1,080 replays, 41 CPU traces,
40 worker plus three coordinator nsys reports, ten clean snapshots, exact
hashes, zero partial files, and no scoped error. This proves the v3 idempotent
stop fix under the reverse shutdown race order. Two stable collectors accepted
it as result 142/176: audit `605a9546...`, all-results `6b066a92...`.

Exact merged-launcher dry runs passed for eleven-tray 16K b50 (scheduler max;
raw max 51) and 32K b25 (scheduler/raw max). An initial invocation of the
archived task-local launcher exited before submission because it correctly
rejects the patched source; no job was created. The merged major launcher then
submitted `5493742`/`5493743` at 20:10 PDT. Slurm proves eleven nodes, one
hour, null requested nodelists, unchanged exclusions, v3 source/hash, and exact
b50/b25. Including retained topology jobs, exactly four jobs are live.

Eleven-tray 16K job `5493742` passed exact b50 at an observed minimum 839,424
tokens and completed `0:0` in 14:18 at 1,480.6445 TPS/GPU and 30.699163 ms.
It has two complete windows, 1,200 replays, all 45 CPU traces, 47 nsys reports,
eleven clean snapshots, matching inputs, zero partial files, and clean scoped
runtime logs. Two stable collectors accepted it as result 143/176: audit
`76e7b99b...`, all-results `98a23fd...`. Parallel 32K job `5493743` then
started on the same freely selected eleven-node allocation; the final pair
remains gated on its review and stable collection.

Eleven-tray 32K job `5493743` passed exact b25 at an observed minimum 839,424
tokens and completed `0:0` in 10:53 at 867.0179 TPS/GPU and 26.213152 ms. It
has two complete windows, 1,200 replays, all 45 CPU traces, 47 nsys reports,
eleven clean snapshots, matching inputs, zero partial files, and zero scoped
errors. Two stable collectors accepted it as result 144/176: audit
`3ab2ce2d...`, all-results `f247e459...`.

Exact merged-launcher dry runs then passed for 64K b12 and 128K b6, both equal
to scheduler/raw capacity maxima. Jobs `5494214`/`5494215` were submitted at
20:43 PDT. Slurm proves eleven nodes, one hour, null requested nodelists,
unchanged exclusions, exact batches, and v3 source/hash. Including retained
topology jobs, exactly four jobs are live; twelve trays remains blocked.

Eleven-tray 64K job `5494214` passed exact b12 at an observed minimum 839,424
tokens and completed `0:0` in 11:28 at 479.0109 TPS/GPU and 22.774200 ms.
Parallel 128K job `5494215` passed exact b6 at 838,912 tokens and completed
`0:0` in 15:32 at 247.5118 TPS/GPU and 22.037516 ms. Each has two complete
windows, 1,200 replays, 45 CPU traces, 47 nsys reports, eleven clean snapshots,
matching inputs, zero partial files, and zero scoped errors. Two stable
collectors accepted them as results 145--146/176: audit `6c6e21f9...`,
all-results `f128a3fa...`.

Exact merged-launcher dry runs then passed for twelve-tray 16K b50 (scheduler
max; raw max 51) and 32K b25 (scheduler/raw max). Jobs `5494517`/`5494518`
were submitted at 21:03 PDT. Slurm proves twelve nodes, one hour, null requested
nodelists, unchanged exclusions, exact batches, and v3 source/hash. Including
retained topology jobs, exactly four jobs are live.

Twelve-tray 16K job `5494517` passed exact b50 at an observed minimum 838,912
tokens and completed `0:0` in 14:53 at 1,434.0752 TPS/GPU and 31.960202 ms.
Parallel 32K job `5494518` passed exact b25 at 838,848 tokens and completed
`0:0` in 15:41 at 852.0580 TPS/GPU and 26.895666 ms. Each has two complete
windows, 1,320 replays, 49 CPU traces, 51 nsys reports, twelve clean snapshots,
matching inputs, zero partial files, and zero scoped errors. Two stable
collectors accepted them as results 147--148/176: audit `4cbd702c...`,
all-results `e483efcd...`.

Exact merged-launcher dry runs then passed for twelve-tray 64K b12 and 128K
b6, both equal to scheduler/raw capacity maxima. Jobs `5494868`/`5494870`
were submitted at 21:28 PDT. Slurm proves twelve nodes, one hour, null requested
nodelists, unchanged exclusions, exact batches, and v3 source/hash. Including
retained topology jobs, exactly four jobs are live.

Twelve-tray 64K job `5494868` passed exact b12 at an observed minimum 838,912
tokens and completed `0:0` in 12:31 at 480.9453 TPS/GPU and 22.871626 ms.
Parallel 128K job `5494870` passed exact b6 at 838,976 tokens and completed
`0:0` in 16:29 at 241.7346 TPS/GPU and 22.752221 ms. Each has two complete
windows, 1,320 replays, 49 nonempty CPU traces, 51 nonempty nsys reports,
twelve clean final snapshots, matching inputs, zero partial files, and zero
scoped errors. Two stable collectors accepted them as results 149--150/176:
audit `ce65da7b...`, all-results `f3671a0c...`.

Exact merged-launcher dry runs then passed for thirteen-node 16K b50
(scheduler max; raw max 51) and 32K b25 (scheduler/raw max), both resolving to
12:1. Jobs `5495038`/`5495039` were submitted at 21:49 PDT. Slurm proves
thirteen nodes, one hour, null requested nodelists, unchanged exclusions, and
the exact job names/batches. Including retained topology jobs, exactly four
jobs are live; review both before the thirteen-node 64K/128K pair.

Job `5495038` started on a freely selected thirteen-node allocation and its
live gate proved all 48 attention workers, an observed minimum 839,424 tokens,
raw max 51, scheduler max 50, and exact requested b50. Job `5495039` remains
pending; continue the two-minute cadence while either job runs or transitions.

Thirteen-node 64K job `5495377` completed `0:0` in 16:56 at 490.4714
TPS/GPU and 22.584240 ms. It has two complete windows, 1,440 replays, 53
nonempty CPU traces, 55 nonempty nsys reports, thirteen clean snapshots,
matching inputs, zero partial files, and zero scoped errors. Hold collector
acceptance until paired 128K job `5495378` completes and passes audit.

Job `5495378` started at 02:21 PDT on a freely selected thirteen-node
allocation after a long scheduler wait. Its live gate proved all 48 attention
workers, observed minimum 838,976 tokens, raw max 6, scheduler max 6, and exact
requested b6. Continue the two-minute cadence and audit the terminal result.

Job `5495378` failed `2:0` after 8:16 before sampling. All capacity checks and
worker/model loading passed, but the head-node API server could not bind the
previously validated base port 23024 (`EADDRINUSE`); there is no result or
partial artifact. This matches two earlier transient control-port takeovers and
is not hardware evidence, so the failed raw tree is preserved and
`nvl72095-T01` is not excluded. An exact unchanged b6 dry run passed, then retry
job `5498437` was submitted alone in that lane at 02:32 PDT. It requests
thirteen nodes, one hour, a null requested nodelist, unchanged exclusions, and
the same v3 source/profile; the new job ID selects a distinct control block.
Hold fourteen nodes until the retry completes and the pair collects twice.

Retry `5498437` started on a freely selected thirteen-node allocation with
distinct block 23496--23502. Its live gate proved all 48 attention workers,
observed minimum 839,424 tokens, raw max 6, scheduler max 6, and exact b6; no
bind or worker error is present after 6:24. Continue the two-minute cadence.

Retry `5498437` cleared the prior collision point and completed `0:0` in 16:41
at 248.3813 TPS/GPU and 22.298224 ms. It has two complete windows, 1,440
replays, 53 nonempty CPU traces, 55 nonempty nsys reports, thirteen clean
snapshots, matching inputs, zero partial files, and zero scoped errors. Two
stable collectors accepted the 64K/retry pair as results 153--154/176: audit
`342c598e...`, all-results `6eb228e3...`; the failed attempt remains excluded.

Exact merged-launcher dry runs then passed for fourteen-node 16K b50 and 32K
b25, both resolving to 13:1. Jobs `5499083`/`5499084` were submitted at 03:31
PDT. Slurm proves fourteen nodes, one hour, null requested nodelists, unchanged
exclusions, exact names/batches, and merged launcher. Including the two retained
topology jobs, exactly four jobs are live; review both before the 64K/128K pair.

Job `5495039` then started on a separate freely selected thirteen-node
allocation. Its live gate proved all 48 attention workers, observed minimum
838,912 tokens, raw max 25, scheduler max 25, and exact requested b25. Both
first-pair jobs are independently capacity-valid; audit each terminal result.

Thirteen-node 16K job `5495038` completed `0:0` in 18:16 at 1,393.5664
TPS/GPU and 33.119231 ms. It has two complete windows, 1,440 replays, 53
nonempty CPU traces, 55 nonempty nsys reports, thirteen clean snapshots,
matching inputs, zero partial files, and zero scoped errors. Hold its collector
acceptance until paired 32K job `5495039` completes and passes the same audit.

Thirteen-node 32K job `5495039` completed `0:0` in 15:52 at 832.0121
TPS/GPU and 27.736284 ms. It has two complete windows and the same complete
1,440-replay, 53-CPU-trace, 55-nsys-report, thirteen-snapshot evidence set as
the paired 16K run, with matching inputs, zero partials, and zero scoped errors.
Two stable collectors accepted the pair as results 151--152/176: audit
`d5d584cd...`, all-results `f35cdf99...`.

Exact merged-launcher dry runs then passed for thirteen-node 64K b12 and 128K
b6, both scheduler/raw capacity maxima. Jobs `5495377`/`5495378` were submitted
at 22:24 PDT. Slurm proves thirteen nodes, one hour, null requested nodelists,
unchanged exclusions, exact names/batches, and merged launcher. Including the
two retained topology jobs, exactly four jobs are live; review both before
fourteen nodes.

Job `5495377` started on a freely selected thirteen-node allocation and its
live gate proved all 48 attention workers, observed minimum 839,424 tokens,
raw max 12, scheduler max 12, and exact requested b12. Job `5495378` remains
pending; continue the two-minute cadence while either job runs or transitions.

At 04:00 PDT, fourteen-node 16K job `5499083` completed `0:0` in 15:23.
Its numerical result is internally valid: A13:F1, exact b50 (observed minimum
838,976 tokens; raw max 51), 1,344.6651 TPS/GPU at 34.527981 ms, 2,600
samples, two full windows, 1,560 replays, 57 CPU traces, fourteen clean
snapshots, matching inputs, no partials, and no scoped runtime error. Strict
profile review found only 58 nonempty nsys reports: all 56 rank reports but
only two coordinator reports, versus the established 59-report contract. No
zero-length report or residual qdstrm exists anywhere in the raw tree. Preserve
the full run but do not accept or collect it. Exact unchanged retry `5499401`
was dry-run and submitted at 04:06 PDT with the merged launcher hash
`60e5e92c...`, fourteen nodes, one hour, null requested nodelist, and unchanged
exclusions. Paired 32K job `5499084` is running; accepted count remains
154/176. Topology jobs `5479092`/`5485585` remain queued, so retain the old-name
compatibility symlink.

Paired 32K job `5499084` then failed `2:0` after 8:24 before sampling. Its
exact 52-worker b25 capacity gate and all model loading passed, but Uvicorn
could not bind base port 20672 (`EADDRINUSE`). No result or partial artifact
exists. This is another transient control-plane collision, not hardware
evidence; preserve the raw tree and leave exclusions unchanged. Exact unchanged
retry `5499444` passed dry-run and was submitted at 04:12 PDT with fourteen
nodes, one hour, null requested nodelist, and the merged launcher. Both retries
and both topology jobs are pending; use the five-minute cadence until one runs.

After the delayed scheduler interval, topology job `5479092` failed `1:0` in
32:41 before sampling: no sample/result/SUCCESS, zero replays, no final
snapshots, and multiple CUDA launch failures followed by DeepGEMM NVLink
barrier timeouts. The same full `nvl72138-T[01-18]` allocation then completed
8K job `5485585` cleanly, so no exclusion is supported. Job `5485585` measures
1,402.5528 TPS/GPU at 68.010910 ms with 2,040 replays, 73 CPU traces, 75 nsys
reports, and eighteen clean snapshots. Retries `5499401`/`5499444` measure
1,360.2607/852.2628 TPS/GPU at 34.132113/27.238411 ms; each has 1,560
replays, 57 CPU traces, 59 nsys reports, fourteen snapshots, matching inputs,
and no partial/error artifact. The rejected `5499083` tree was moved intact to
`results/rejected_profile_incomplete/`, allowing two stable collectors at
157/176 (`22c39903...`, `78861ec5...`). Both legacy jobs left the queue, so the
old-name symlink was removed while the canonical tree remained intact. Exact
dry runs then passed and jobs `5502011` (4K A17:F1 retry), `5502013` (64K
A13:F1), and `5502012` (128K A13:F1) were submitted with merged launcher,
v3 source, exact exclusions, null requested nodelists, and one-hour limits.

Collector hash `8b3dc24e...` now fail-closes AFD profile artifacts: exactly one
nonempty CPU trace per GPU plus coordinator, every numbered GPU-rank nsys
report, and the evidence-backed two-or-three coordinator nsys reports. An
initial fixed-three check correctly exposed 17 historical runs with two
coordinator reports but all GPU ranks; the final rank-explicit contract preserves
those valid results without weakening GPU coverage. In-memory compilation and
two remote full passes succeeded with the unchanged 157/176 report hashes.

Fourteen-node 128K A13:F1 job `5502012` completed `0:0` in 17:33 at
249.0501 TPS/GPU and 22.370717 ms. It has exact b6 capacity at an observed
839,424-token minimum, two full windows, 1,560 replays, 57 CPU traces, all 56
rank plus three coordinator nsys reports, fourteen clean snapshots, matching
inputs, and no partial. The sole broad grep hit is a benign tokenizer warning
for chat-template length 131,087 versus tokenizer metadata 131,072; the pinned
profile/runtime limit is 131,136 and the run completed cleanly. Two stable
trace-aware collectors accepted it as result 158/176 (`e380712c...`,
`aca58247...`). Paired 64K job `5502013` and 4K A17 retry `5502011` remain
pending; do not open A14:F1 before 64K completes and the A13:F1 stage closes.

## 2026-07-24: attention-CUDA latency profile recovery

The July 22 consolidated CSV has 487 rows: 197 vLLM rows and 290 AFD rows.
The fail-closed attention-CUDA extractor version
`20260724-attention-cuda-execution-span-v11` produced 478 records and rejected
nine AFD rows. Six irregular rows have no attention-worker Nsight reports;
two other irregular rows and uniform 4K A2:F1 have worker reports that omit
both child graph kernels and the CUPTI graph-execution interval. Existing
evidence cannot recover the requested attention CUDA execution span for those
nine rows.

The user authorized exact recovery runs and requested monitoring to terminal
state. The retained launcher
`/home/shengjiel/scratch/fastafd_reproduce/control/run_afd.sh` is byte-identical
to local SHA-256 `60e5e92c...`; recovery runs set
`NSYS_CUDA_GRAPH_TRACE=node` and `NSYS_CAPTURE_RADIUS_STEPS=15`, retain the
original topology, batch, prompt distribution, model/profile, one-hour limit,
null requested nodelist, and evidence-backed exclusions. Outputs are isolated
under remote
`ratio_ep_isl_sweep_20260717_1548/cuda_latency_recovery_20260724/results`.

The first probe wave covers every failure class:

- `5591131`: irregular 1K--4K, b96, A7:F1/n8; prior run had no worker reports.
- `5591133`: irregular 32K--128K, b6, A7:F1/n8; prior report omitted graph execution.
- `5591134`: uniform 4K, b201, A2:F1/n3; prior report omitted graph execution.

All three passed exact dry-run validation and were submitted at 20:17 PDT.
Initial Slurm state is eligible/pending with no dependency or hold,
`TimeLimit=01:00:00`, `ReqNodeList=(null)`, and the established exclusions.
Keep at most four recovery jobs live. Validate an extractable attention CUDA
span from this probe wave before submitting the remaining six rows.

At 22:59 PDT the three probe jobs were still pending and unallocated. The user
then redirected work to an available-data update; the Slurm jobs were left
untouched, but the interactive monitor was stopped.

The new non-overwriting available-data artifacts are:

- `analysis_n17_20260722/all_sweep_results_cuda_available_20260724.csv`,
  SHA-256 `833caeff...`: all 487 logical rows, 478 rows updated to v11 CUDA
  timing, and nine AFD rows explicitly blank with
  `latency_basis=unavailable_attention_cuda_span`.
- `analysis_n17_20260722/cuda-available-four-3d-diagrams-20260724.html`,
  SHA-256 `c1624da0...`: four updated 3D diagrams for AFD ratio, baseline EP,
  AFD A7:F1 batch, and baseline EP16 batch. The unavailable 4K A2:F1 ratio
  point is a surface gap; the eight unavailable irregular points are not among
  these four diagrams.

Validation proved 487 unique row keys, 478 valid metric formulas, nine exact
blank error rows, four diagram blocks/four Plotly calls, valid inline
JavaScript syntax, and no `NaN`/`Infinity`. The July 22 CSV and prior HTML
remain byte-identical at SHA-256 `075d5563...` and `463e1dbe...`.

## 2026-07-25: corrected AFD recovery capture windows

The first recovery probes reached terminal state. Job `5591134` (uniform 4K
A2:F1/b201) completed with all attention-rank reports. Strict v11 extraction
from its isolated result succeeded on five captured selected-wave steps:
attention CUDA latency `21.430016 ms`, steps 1621--1625, with the inter-step
CUDA-idle gap excluded. Job `5591131` (irregular 1K--4K/b96) completed but
produced only coordinator reports. Its configured worker capture was steps
512--542 while the run's final full decode wave was 495--509; the worker
`cudaProfilerStart` trigger was never reached. Job `5591133` (irregular
32K--128K/b6) failed before profiling because API port 21064 was already in
use.

The original result step IDs make the corrected windows deterministic. For
the six high-batch rows, use radii 48/64/96/128 for 1K--4K b96/b128/b192/b256
and 48/64 for 4K--8K b96/b128. These begin 14--16 coordinator steps before the
final selected wave and extend past its end. For the two long-range b6 rows
and the uniform A2:F1 row, radius 32 covers the complete final selected wave.
All recovery runs retain `NSYS_CUDA_GRAPH_TRACE=node`; extraction remains the
strict attention-CUDA v11 contract and does not use eager or scheduler-wall
fallbacks.

Four corrected jobs were submitted under the existing isolated recovery root,
with the established node exclusions and at most four scoped recovery jobs:

- `5602393`: 1K--4K b96, radius 48; validation probe for high-batch capture.
- `5602394`: 32K--128K b6, radius 32; replaces the transient port failure.
- `5602396`: uniform 4K A2:F1/b201, radius 32; obtains the full selected wave.
- `5602398`: 8K--128K b6, radius 32.

Do not submit the other five high-batch rows until `5602393` proves that
attention-worker reports contain extractable selected-wave CUDA spans. Refill
slots as terminal results are validated, preserving every prior run tree.

`5602393` completed in 10:33 and proved the corrected window: all 28 worker
reports plus the coordinator reports were present. Strict v11 extraction used
TP-rank zero from all seven attention trays and all 15 final-wave steps
(495--509), yielding `30.301984 ms`. `5602398` completed in 10:59, `5602396`
in 11:56, and `5602394` in 12:03, all with exit code 0. The freed slots were
refilled with:

- `5602523`: 1K--4K b128, radius 64.
- `5602526`: 1K--4K b192, radius 96.
- `5602535`: 1K--4K b256, radius 128.
- `5602536`: 4K--8K b96, radius 48.

The only recovery row not yet submitted is 4K--8K b128 (radius 64). Submit it
when one of these four jobs reaches terminal state.

All remaining jobs completed with exit code 0. `5602523` ran 11:27,
`5602526` 12:57, `5602535` 11:34, `5602536` 13:06, and final job `5602746`
(4K--8K b128, radius 64) ran 14:56. No scoped recovery jobs remain active.
Every recovered point passed strict v11 extraction using all 15 steps from the
run-selected final wave and TP-rank zero from every attention tray:

- irregular 1K--4K b96/b128/b192/b256:
  `30.301984`, `37.135264`, `48.382208`, `62.037824 ms`;
- irregular 4K--8K b96/b128: `30.215488`, `36.997984 ms`;
- irregular 8K--128K b6 and 32K--128K b6:
  `18.944576`, `19.028128 ms`;
- uniform 4K A2:F1/b201: `21.483105 ms`.

The final non-overwriting manifest is local
`analysis_n17_20260722/recovery_validation_20260725/cuda_latency_manifest_recovery_final9.json`
(SHA-256 `9ea4e57b...`) and points exactly nine logical rows to isolated recovery
trees. The corresponding complete metrics file is
`cuda_latency_metrics_recovery_complete_20260725_v11.json`
(SHA-256 `ea82446f...`). Strict full-manifest validation reports 487 records,
487 unique row keys, zero errors, metric version
`20260724-attention-cuda-execution-span-v11`. The earlier July 22 CSV/HTML and
the July 24 available-data derivatives remain untouched; use this complete
metrics file for the next non-overwriting CSV/HTML update.

## 2026-07-25: final complete CUDA-timing CSV and four-diagram HTML

The final v11 metrics were applied with the retained scripted workflow, never
by manual row editing. New non-overwriting outputs are:

- `analysis_n17_20260722/all_sweep_results_cuda_complete_20260725.csv`
  (82,051 bytes, SHA-256 `c617fbf3...`): all 487 logical sweep rows use
  attention CUDA execution-span latency for AFD and CUDA execution-span
  latency for vLLM; no row is unavailable.
- `analysis_n17_20260722/cuda-complete-four-3d-diagrams-20260725.fragment.html`
  (35,139 bytes, SHA-256 `f974cea9...`): editable four-diagram source.
- `analysis_n17_20260722/cuda-complete-four-3d-diagrams-20260725.html`
  (66,694 bytes, SHA-256 `82c3a859...`): rendered standalone HTML containing
  the AFD ratio, baseline EP, AFD A7:F1 batch, and baseline EP16 batch plots.

The final validator
`analysis_n17_20260722/validate_cuda_complete_artifacts.mjs` recomputed every
CSV latency and throughput value from the 487-record metrics file. It proved
487 unique row keys, zero metric errors, all nine recovered rows with 15
samples, four diagram blocks/four Plotly calls, valid inline JavaScript, a
complete numeric 8-by-16 AFD-ratio grid, and the recovered 4K A2:F1 point at
`21.483105 ms` and `6237.459622 TPS/GPU`. The row breakdown is AFD
131 ratio + 86 batch + 73 irregular-batch and vLLM 40 EP + 86 batch + 71
irregular-batch.

The reusable HTML builder now derives its accessibility description from
whether chart points are actually missing; it reports complete CUDA
execution-span measurements for this final dataset. Both builder and validator
pass `node --check`. The original July 22 CSV (`075d5563...`), July 24 CSV
(`833caeff...`), July 24 fragment (`3a422926...`), July 24 standalone HTML
(`c1624da0...`), and original diagram fragment (`9c03adcc...`) remain
byte-identical.
