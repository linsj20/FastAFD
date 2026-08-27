# Qwen3 FMHA/FFN Minimum Handoff

## Active remote latency optimization (2026-08-24 PDT)

The user reopened optimization and requires ordinary remote Git commits for every rollback point; local Git remains untouched. Work is on branch codex/qwen3-ffn-latency-20260824. Commit 0a6a4a3 snapshots aligned job 6474505, whose accepted A1:F1 result remains 25.7321745 ms E2E, 25.6797822 ms strict CUDA, and perfect alignment in job 6474804. Trace export 6480445 completed; accepted-versus-fused analysis localizes the fused-turn regression to slower fused add-RMSNorm/FP8 work and added attention/model readiness delay, so that exact-two-stream path remains rejected.

The first new isolated candidate, commit 0ae36e8, fused steady-state input RMSNorm with QKV FP8 quantization while preserving the accepted CUDA-event and DeepEP cadence. Remote CPU contracts passed 60/60. Exact short-queue one-hour job 6481196 completed 15/15 with zero outliers but regressed to 26.1607085 ms E2E and 26.1054416 ms strict CUDA, respectively +0.428534 ms/+1.665% and +0.4256594 ms/+1.658% versus job 6474505. It is rejected without alignment and was restored by remote revert commit 487bddd. The second isolated candidate, commit 6d4b686, amortized each system fence across eight grid-stride 64-bit publication tasks. Remote CPU contracts passed 60/60. Exact job 6481517 completed 15/15 with zero outliers but regressed to 26.56672 ms E2E and 26.4969301 ms strict CUDA, +0.8345455 ms/+3.243% and +0.8171479 ms/+3.182% versus job 6474505. Reduced publication parallelism is rejected, the 16-tray A15 run was intentionally skipped, and remote revert commit c1c7e7a restores baseline. A CPU-only trace export was not submitted because short QoS rejected it with QOSMinGRES; no GPU was wasted on post-processing. The third isolated candidate, commit cd78af5, reduces DeepEP from three 8-CTA clusters to one and raises the complementary DeepGEMM budget from 128 to 144 SMs without changing collective protocol. Remote CPU contracts passed 60/60. Exact A1 job 6481772 completed 15/15 with zero outliers but regressed to 26.139278 ms E2E and 26.0721142 ms strict CUDA, +0.4071035 ms/+1.582% and +0.392332 ms/+1.528% versus job 6474505. Eight communication SMs are therefore rejected as a universal A1 policy. Exact A15:F1/b6 high-fan-in job 6481976 completed 15/15 with zero outliers at 33.261404 ms E2E and 33.286293333 ms strict CUDA, with a 6.945% strict dominant-range spread and 5.882% max/median separation. Against exact no-mode job 6263396 at 30.9095675 ms E2E and 30.752484 ms strict CUDA, the gaps are +2.3518365 ms/+7.609% and +2.533809333 ms/+8.239%; this candidate is not competitive and alignment is withheld. The eight-SM result is retained only as high-fan-in evidence. The production-shape four-GPU 8/16/24 DeepEP-SM sweep is complete. Jobs 6483946-6483948 failed before container start because the first submission used an incorrect image path. Retry jobs 6484042 and 6484043 then exposed a singular 45-row deterministic smoke input, and pending job 6484044 was canceled before allocation; commit 41ff981 makes that input full rank and passes 60/60 CPU contracts, with independent 16/24-SM cherry-picks 38bc6d8 and 824e295. Valid short-QoS one-hour jobs 6484104/6484105/6484106 all passed identical BF16, FP8, and fused-router correctness bounds. Their production fused-router staged means are respectively 0.337639/0.327233/0.318441 ms for 24/16/8 communication SMs, so eight SMs is 2.687% faster than sixteen and 5.686% faster than twenty-four for the exact 45-row A15 MoE shape. This isolates eight SMs as the best high-fan-in split in this family, while exact A1 job 6481772 still requires retaining the accepted 24-SM policy for low fan-in. Adaptive commit a8eb982 selects 8/144 communication/compute SMs at fan-in >=8 and 24/128 below it, supports only complete 8-CTA cluster counts, and passes 60/60 CPU contracts. Short one-hour smokes 6484308 and 6484309 pass the 45-row 8/144 and 3-row 24/128 branches with all BF16, FP8, and fused-router correctness checks. Exact A1 job 6484425 then selected 24/128 on all four model ranks and produced a durable 24-prompt/408-token sample with 25.9834295 ms raw E2E median, +0.251255 ms/+0.976% versus accepted job 6474505. Its strict post-processing alone failed because the supplied Pareto plan contains EP2 rather than this EP4 A1 row; all eight raw reports remain durable and strict recovery will be piggybacked on the next required GPU validation, so no A1 acceptance decision is made from the raw median alone. The standard /home fastafd_reproduce symlink was restored to the existing Lustre workspace so the pinned model-profile manifest resolves without bypassing validation.


The one-wave publication attempt is rejected as a production no-op. Wrapper job 6485116 failed before srun because sbatch --wrap used POSIX sh with pipefail; retry 6485203 reached fresh extension compilation and caught the out-of-scope CUDA_CHECK use before any workload. Fix commit 9be35c4 then passed combined short/one-hour job 6485226: the 8,192-row four-GPU transport smoke completed and recovered adaptive A1 strict metric 25.915859667 ms with 15/15 samples and zero outliers, +0.919% on the unchanged low-fan-in path. Exact A15:F1/b6 job 6485295 completed in 33:53 at 31.714345 ms E2E and 31.688765867 ms strict CUDA, 15/15 and zero outliers. Although this is 4.651%/4.799% faster than run 6481976, it remains 2.604%/3.045% behind no-mode job 6263396. Fresh rank-61 SQLite proves both 6481976 and 6485295 launched all 3,008 QKV publications as publish_qkv_kernel<128,false,true> with the same 360x256 geometry: ready_edges was one, so the multi-ready-only cap never executed and the observed difference is run/node variance, not a causal win. Reverts 78f6ec8 and 5b30bbe restore the adaptive baseline; the direct remote CPU contract passes 60/60 and Git is clean. Comparative SQLite is under analysis/a15_old_vs_twoline outside the source worktree.


Corrected commit 386e5af passed decode intent explicitly into QKV publication so graph decode alone used one device wave while eager prefill kept the proven 1,024-CTA bound and A1 remained naturally below the cap. The remote CPU contract passed 60/60; fresh production-shape four-GPU job 6486271 passed rows=45, nine iterations, and qkv_decode=1 on short with a one-hour limit. Exact A15:F1/b6 job 6486377 completed in 32:30 at 32.520627 ms E2E and 32.2289626 ms strict CUDA, 15/15 and zero outliers. Fresh rank-61 SQLite proves the intended causal signature: all 3,008 publish_qkv_kernel<128,false,true> launches changed from 360x256 to 152x256. However, versus the recent unchanged-path job 6485295, E2E regressed 0.806282 ms/2.542%, strict regressed 0.540197 ms/1.705%, and QKV kernel mean increased from 20.923 to 21.952 us/4.918%. The candidate remains 5.212%/4.801% behind no-mode job 6263396 and is rejected; revert baf1cdf restores the adaptive baseline and passes 60/60. Publication parallelism should not be reduced further.


System-atomic completion candidate 51b07a0 used an explicit high-fan-in decode-only specialization while preserving the accepted low-fan-in and eager-prefill paths. Four-GPU production-shape smoke job 6487241 passed 100 exact iterations at rows=45, sources=15, decode=1. Exact short-QoS one-hour A15:F1/b6 job 6487343 completed 15/15 with zero outliers at 33.4261575 ms E2E and 33.360445733 ms strict CUDA. Rank-61 SQLite confirms all 3,008 launches used publish_qkv_kernel<128,false,true,true> at the unchanged 360x256 geometry. QKV kernel total was 69.303433 ms, 10.116% slower than recent unchanged-path job 6485295; strict latency regressed 1.671679867 ms/5.275% versus that control, 2.607961733 ms/8.480% versus exact no-mode job 6263396, and 0.0741524 ms/0.223% versus earlier 8-SM job 6481976. The candidate is rejected; remote revert commit 1130048 restores the adaptive baseline and the direct remote suite passes 60/60. The QKV publication-parallelism and completion-atomic family is closed.


Placement-crossover job 6488053 pinned remote commit 5ee5ea9 and completed the missing exact A8:F1/ATP1/FEP4/b6 FMHA-only point on short with a one-hour limit. It completed 0:0 in 26:02 with 31.2774395 ms E2E, 31.238424333 ms strict CUDA, 15/15 samples, zero outliers, 3.971% dominant-range spread, and 170.729909 strict TPS/GPU. Optimized legacy A8 control job 6265476 is 30.897381 ms E2E and 30.7433556 ms strict CUDA, so FMHA-only loses 0.3800585 ms/1.230% E2E and 0.495068733 ms/1.610% strict. Together with the accepted A1 FMHA-only win and the A15 legacy win, this establishes A8 as the measured placement crossover. Remote commit 2c5df33 adds the explicit qwen3-128k-adaptive runner policy for exactly Qwen3 128K/b6/ATP1/FEP4/MB2: ratios below eight resolve to fmha-only and ratios at or above eight resolve to legacy; all other workload contracts fail fast. Results record requested placement, resolved placement, and policy. A1/A8 dry runs selected the intended modes, a batch-5 dry run failed before submission, bash syntax and diff checks pass, and the full remote suite passes 60/60.

## Final stop decision (2026-08-24 PDT)

The user chose the staged job-6474505 CUDA-event handoff and stopped the
exact-two-stream follow-up. Local source/tests and the canonical remote
`source_two_lane_final` workspace are restored to that accepted implementation.
No redundant performance rerun was submitted. Retained evidence is
25.7321745-ms E2E, 25.6797822-ms strict CUDA, 116.585561 TPS/GPU, 15/15 samples,
zero outliers, and perfect official alignment in job 6474804. Its model trace
has 3,678 nodes and 95 CUDA Graph execution stream IDs; the two physical
PyTorch streams and two DeepEP buffers are reused, but CUDA events create the
extra graph execution segments. All later exact-two-stream candidates remain
rejected. Resume only on explicit user request.

## Active stream-reuse follow-up (2026-08-23 PDT)

The focused 128K, batch-6, A1:F1, ATP1/FEP4 performance/alignment comparison is
complete, but its FFN execution topology is not. The opt-in `fmha-only` mode
passes the literal `<26 ms` E2E gate and the unchanged official alignment
scorer. Pareto is frozen and is not part of this completion boundary.

| placement | job | E2E median (ms) | TPS/user | TPS/GPU | strict CUDA mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 6470598 | 30.6157295 | 32.6629486 | 97.9888459 | 30.4497429 |
| `fmha-only` exact A1 | 6472328 | 25.9524935 | 38.5319430 | 115.5958290 | 25.9164299 |
| separated signal/data | 6474505 | 25.7321745 | 38.8618537 | 116.585561 | 25.6797822 |
| full device-turn kernels (rejected) | 6475180 | 26.429704 | 37.8362164 | 113.5086492 | 26.3694974 |
| captured stream-memory waits (rejected) | 6475460 | 26.945252 | 37.1122898 | 111.3368693 | 26.8631071 |

Job 6474505 is the current best: 15/15 eligible samples, zero outliers, and
1.882% spread. Its official alignment job 6474804 completed `0:0` on 24 prompts
/ 408 tokens: top-1/top-10/top-100 are all 1.0 and average/max rank are 1.0.
E2E throughput uses 24 global tokens per step over eight GPUs:
`TPS/user = 1000 / latency_ms` and `TPS/GPU = 3000 / latency_ms`.

Best-case trace inspection exposed the active defect. Attention rank 1 has
9,024 graph kernel calls, 564 nodes, and exactly two kernel stream IDs. Model
rank 5 has 58,848 calls, 3,678 nodes, and 95 kernel stream IDs: two intended
lanes plus 93 CUDA-event-created layer segments. The implementation already
owns only two streams and two DeepEP buffers; the per-round event handoff is
what expands the captured topology. A first replacement used one device turn
for every layer, a one-warp wait on the idle lane, and a next-lane release fused
at QKV publication entry. Job 6475180 plus export 6475310 proved exactly two
model kernel streams, but added exactly 187 wait-kernel nodes (3,865 total) and
regressed E2E to 26.429704 ms and strict CUDA to 26.3694974 ms. It is rejected.
The second replacement captured CUDA stream-memory waits instead of launching
those kernels while retaining the fused release. It passed the local contract 15/15 and
`git diff --check`. Remote CPU job 6475380 passes 60/60, and fresh SM100 job
6475381 passes the transport build, two fabric pairs x nine 8,192-row
iterations, plus the captured two-stream wait/fused-release replay. Exact
A1:F1 EP4 job 6475460 completed at 26.945252 ms
E2E and 26.8631071 ms strict CUDA, +4.714%/+4.608% versus job 6474505, so it is
rejected without alignment. Export 6475603 proves exactly two rank-5 model
kernel streams and the original 58,848/3,678 kernel shape, with 1.811453-ms
median productive lane overlap. The next candidate keeps only acyclic
same-layer lane-0-to-lane-1 event handoffs; cyclic cross-layer handoffs are the
part that generated 93 extra CUDA Graph execution streams. It has no extra
steady-state wait kernel or stream-memory operation. Local contract validation
passes 15/15. CPU job 6475707 found one stale negative-test setup before any
GPU ran. Replacement CPU job 6475725 passes 60/60, and dependent fresh SM100
job 6475726 passes transport prebuild plus two fabric pairs x nine 8,192-row
iterations. Exact job 6475770 passes execution but regresses E2E/strict CUDA to
26.319437/26.2625085 ms, +2.282%/+2.269% versus job 6474505. It is rejected
without alignment. CPU export 6475829 shows the original 58,848/3,678 model
shape still uses 95 stream IDs: lane 0 remains on stream 32, while every
same-layer event splits lane 1 into a separate 19-node execution segment. This
proves any per-layer captured event fork, not only a cyclic edge, expands the
trace. The next candidate removes all model-side event handoffs and relies on
the existing per-lane O-ready waits, attention device turn, and DeepEP cleanup
handshake. Local contract remains 15/15 and remote CPU job 6475982 passes
60/60. Exact job 6475991 completes `0:0` at 26.1218645 ms E2E and
26.0498022 ms strict CUDA, +1.514%/+1.441% versus job 6474505. Export 6475992
proves the target 58,848/3,678 model shape on exactly streams 32 and 164; those
same streams carry both DeepEP lanes, with 1.148382-ms median productive
overlap. This topology succeeds but performance rejects the neutral-priority
schedule. The model-only lane-0-priority candidate passes CPU 60/60 in job
6476267. Exact job 6476272 measures 26.015458/25.9618923 ms E2E/strict CUDA,
+1.101%/+1.099% versus job 6474505. Export 6476274 retains the original
58,848/3,678 graph shape on exactly model streams 28/164, with both DeepEP
lanes on those streams and 1.329086-ms median productive overlap. Performance
still rejects it. The final directional test applies lane-0 priority to both
roles because attention remains the longer profiled role in every retained
sample; it still adds no graph dependency, wait, or node. Local contract passes
15/15 and remote CPU job 6476500 passes 60/60. Exact job 6476535 measures
26.1026495/26.0252253 ms E2E/strict CUDA, +1.440%/+1.345% versus job 6474505.
Export 6476537 retains the original 58,848/3,678 shape on exactly model streams
28/164, both used by DeepEP, with 1.324384-ms median productive overlap. This
direction is rejected. The accepted neutral-stream, per-round event schedule
is restored locally and remotely; local contract passes 15/15, and restored
CPU job 6476746 passes 60/60. TODO remains to reuse exactly two CUDA Graph
execution streams and two DeepEP buffers across all 94 FFN layers while
maintaining E2E at or below 25.7321745 ms and preserving attention-like
ping-pong overlap. A redundant fabric smoke remains omitted because
transport/DeepEP code is unchanged from fresh-passing job 6475726. Earlier CPU submissions
6475038/6475079 were
infrastructure-only failures from retired default `/home` image/venv paths;
6475090 found the final obsolete unit-test expectation.

The active follow-up attributes the fastest exact-two-stream regression mainly
to readiness cadence: job 6476272 adds 0.548 ms of attention Q-ready waits and
0.191 ms of model O-ready waits versus job 6474505, while productive attention
kernels add only about 0.013 ms. It retains the raw event-free graph and sets
priority -1 only on each QKV publication plus its four nearest productive
ancestors before graph instantiation; all FFN/DeepEP tails remain default
priority. This should create the desired compute-versus-wait stagger with the
same two streams and no added node or dependency. Local contract passes 15/15,
remote CPU job 6476950 passes 61/61. Raw-graph GPU smoke 6476955 completed
`0:0`, reprioritizing two named nodes and replaying the explicitly instantiated
graph three times. Exact job 6476970 found the expected 188 QKV publication
anchors / 940 selected kernels, but regressed to 27.7717195 ms E2E and
27.6719843 ms strict CUDA. Giving both lanes the same signal-path priority
makes them advance in phase rather than ping-pong, so the candidate is
rejected. Dependent trace export 6476971 is retained. The next direction gives
high priority only to lane 0's publication ancestors, while resetting lane 0's
FFN/DeepEP tail and all of lane 1 to default priority.
Export 6476971 confirms 58,848 model graph kernel calls / 3,678 nodes on exactly
streams 32 and 164, with both DeepEP lanes on the same pair. Model productive
overlap is 2.168475 ms median / 1.694047 ms minimum, but attention Q-ready wait
time grows by 3.695024 ms and model summed kernel time by 3.761959 ms per
replay versus job 6474505. The extra overlap is resource contention, not a
latency win.
The asymmetric implementation now uses lane 0's captured stream priority only
as an identity tag, resets that full lane to default, and reapplies priority -1
to its publication ancestors. Local contract passes 15/15, focused remote CPU
job 6477184 passes, and replacement full-suite job 6477233 passes 61/61. Job
6477192 was cancelled as an infrastructure-only `cpu-0001` pre-`srun` stall.
GPU smoke 6477245 identifies exactly one high-priority node among two captured
streams, resets/reprioritizes it, and replays correctly. Exact job 6477252 and
trace export 6477253 are queued.
Job 6477252 found the exact expected 94 lane-0 anchors / 470 selected kernels
after resetting 1,889 lane-0 kernels, but measured 26.223880 ms E2E and
26.1863071 ms strict CUDA. It is +1.911%/+1.972% slower than job 6474505 and
slower than neutral exact-two-stream job 6475991, so per-layer node priority is
rejected. Export 6477253 hit an infrastructure-only missing Slurm spool-script
path inside the container; replacement export 6477539 uses the durable control
script path.
Export 6477539 proves exactly streams 28/164 for all 58,848 model graph calls /
3,678 nodes, both used by DeepEP, and 1.407709-ms median productive overlap.
It nevertheless adds 0.132855 ms model span and 0.136505 ms attention span
versus neutral job 6475991, confirming per-layer priority itself is harmful.
The active replacement uses two neutral streams and only one startup ticket:
lane 1 waits once, lane 0 releases after its first QKV compute and before data
publication, and lane 1 resets the ticket for replay. Local contract passes
15/15 and remote CPU job 6477689 passes 60/60. Jobs 6477571/6477625 were
cancelled after container-prolog stalls, while pending 6477677 was replaced on
known-good `cpu-0015`.
GPU replay smoke 6477708 completed `0:0`: nine replays ended with `turn=0`,
`timeout=(0,0)`, `produced=9`, and `observed=45`, allowing the candidate to
advance to exact performance and dependent durable trace export.
Exact job 6477716 completed `0:0` but is rejected at 26.222327 ms E2E and
26.1675189 ms strict CUDA, with 15/15 samples and zero outliers. Export 6477721
completed `0:0`: rank 5 has 58,880 calls / 3,680 nodes on exactly streams
32/164, both carrying DeepEP. Lane 1's one startup wait averages only 1.938 us
per replay, proving the graph schedules lane 0's release first rather than
forming a real stagger. The next candidate must force waiter-before-release on
the same two streams; periodic reseeding is premature until that first seed is
real.
The active directional candidate makes only persistent model lane 1
high-priority. Remote CPU job 6478123 passes 60/60. Smoke 6478213 replayed
correctly but its captured-event elapsed-time query is unsupported; replacement
6478515 passes with `entered=9` and `release_saw_entered=45`, proving the
high-priority waiter entered before every neutral lane-0 release. This advanced
to exact job 6478604 and dependent durable export 6478605.
Exact job 6478604 completed `0:0` but is rejected at 26.1627855 ms E2E and
26.0976459 ms strict CUDA, with 15/15 samples and zero outliers. Export 6478605
produced integrity-clean local traces: rank 5 is 58,880 calls / 3,680 nodes on
exactly streams 28/164, both carrying DeepEP. The lane-1 startup wait still
averages only 1.824 us because lane-0 graph rooting lets QKV compute execute
first. The active correction roots only model capture/replay on lane 1;
attention remains rooted on lane 0.
The waiter-root candidate passes local contract 15/15 and remote CPU job
6479023 at 60/60. GPU smoke 6479044 completes `0:0` with nine clean replays and
5,007,285 waiter cycles, proving the wait spans lane-0 work. This advanced to
exact job 6479079 and dependent durable export 6479081.
Exact waiter-root job 6479079 completed `0:0` but is rejected at 26.158717 ms
E2E and 26.2745685 ms strict CUDA mean (26.115999-ms median). Export 6479081
completed `0:0`; rank 5 is 58,880 calls / 3,680 nodes on exactly streams
28/164, both carrying DeepEP. Its lane-1 waiter still averages only 2.100 us,
showing full-graph scheduling delays it until just before release. The active
correction publishes waiter entry inside the waiting kernel and gates only lane
0's first compute on that flag, adding one node while retaining two streams.
The guaranteed-entry candidate passes local contract 15/15, remote CPU job
6479470 at 60/60, and GPU smoke 6479372 (`turn=0`, `entered=0`, no timeout,
`observed=45` across nine replays). Exact job 6479509 is rejected at
26.167521 ms E2E and 26.1245205 ms strict CUDA mean (26.119584-ms median),
with 15/15 samples and zero outliers. Export 6479521 proves rank 5 at 58,896
calls / 3,681 nodes / exactly streams 28/164, both carrying DeepEP. Productive
overlap is 1.377312 ms median but the bidirectional ping-pong gate fails. The
waiter now spans 30.064 us; the exposed cost is instead the first lane-0 entry
gate at about 383.9 us because both DeepEP bootstraps precede the lane-1 waiter
in graph insertion order. The active candidate moves the waiter before either
bootstrap. If that does not close the remaining gap, fuse the 93 steady-state
expert admission waits into the existing dispatch-copy epilogues.

Waiter-first jobs 6479731/6479735 are also rejected: strict/E2E CUDA mean is
26.2094891 ms (median 26.166656), 15/15 with zero outliers, and the local rank-5
trace has exactly streams 28/164 plus 3,681 nodes, but bidirectional FFN
ping-pong still fails. The active replacement fuses the per-round turn into the
existing path: QKV publication releases the scheduling signal before its data
push, and steady-state input admission waits inside fused add-RMSNorm+FP8
quant. Every combine is deferred symmetrically, while one CTA in the peer
dispatch-copy epilogue waits for combine residency before admitting the expert.
The intended result is exact two-stream cadence, 186 fewer norm/quant nodes
(93 steady-state layers times two microbatches), no standalone cleanup wait
nodes, and bidirectional FFN cleanup/MoE overlap. The expected model graph is
3,400 nodes. Local static contract is 15/15, remote CPU job 6480201 passes
60/60, and fresh SM100 extension compile job 6480387 passes. GPU smoke 6480416
failed before kernel launch only because its wrapper omitted the writable
FlashInfer cache path; dependent jobs were canceled. The wrapper is fixed, and
replacement smoke 6480441 passes all three focused GPU gates in 6:59:
bit-exact fused add-RMSNorm/FP8 admission over nine two-stream graph replays,
four-rank fabric signal/data transfer, and prearmed DeepEP admission-ticket
consumption/reset. Exact job 6480443 completes 15/15 with zero outliers and
proves the intended 3,400-node model graph, but is rejected at 26.2396121-ms
E2E (26.211742-ms median), 114.330959 TPS/GPU: +0.507438 ms/+1.972% versus the
aligned 25.7321745-ms target. Trace export 6480445 is running to localize the
remaining critical-path loss; do not run alignment on this candidate.

The older corrected no-new-mode sweep row for this same nominal case reports
33.6810167 ms, 29.6903152 TPS/user, and 89.0709455 TPS/GPU in
`regular_ep4_t01_t17_current.csv`. Its latency basis is the attention CUDA
critical-range mean, not E2E, so use fresh job 6470598 for an apples-to-apples
E2E comparison. The prior optimized target was the literal `<26 ms` boundary;
the accepted historical reference was 25.967383 ms and prior current-code job
6438212 measured 25.9576875 ms.

## Alignment and structural evidence

- New-mode official scorer job 6472537 completed `0:0` in 11:15 on the exact
  job-6472328 sample: 24 prompts, 408 generated tokens, top-1/top-10/top-100
  all 1.0, and average/max vLLM rank both 1.0.
- Its post-score trace audit passed on attention rank 1 and model rank 5. Each
  has 16 CUDA graph launches; graph-kernel counts are 9,024 and 58,816.
- Legacy official scorer output from job 6470699 is also perfect on 24 prompts
  and 408 tokens: top-1/top-10/top-100 all 1.0 and average/max rank 1.0. The
  scorer completed before that Slurm record exited 1 only because the old
  placement-blind post-auditor requested model rank 5 for a legacy capture.
  Manual rank-1 SQLite validation found 16 graph launches, 33,136 graph
  kernels, and 100 AFD ranges. The wrapper now inspects only rank 1 for legacy
  and ranks 1/5 for `fmha-only`.

## Final candidate

The retained implementation specializes only the exact single-source,
single-destination publication shape while preserving the generic fan-in
implementation:

- `finish_payload_publication<true>` releases the sole ready word directly
  from lane 0; generic fan-out keeps the last-warp release loop.
- QKV and O publication kernels dispatch compile-time
  `<kSingleEdge=true, kSingleReady=true>` variants when the exact A1 contracts
  hold. These variants compile out source-offset lookup, source-stride
  arithmetic, and generic edge-base updates.
- High-fan-in flattened task mapping, bounded publication grids, the two-slot
  epoch protocol, retained-batch lifetime, and the fail-fast validation path
  remain intact. No fallback was added.

Optimization progression was 26.002127 ms before specialization, then
26.173696/26.192970 ms for ready-only candidates, 26.0772875 ms after the
single-source indexing branch, and 25.9524935 ms after exact single-edge plus
single-ready compile-time dispatch. The last result is the only claimed final
performance result.

Validation is complete:

- local focused contract: 15/15; `git diff --check` clean;
- remote CPU contract job 6472263: 60/60;
- fresh SM100 compile/fabric job 6472264: `0:0`, transport prebuild and two-pair
  8,192-row fabric smoke passed;
- trace export job 6472538: `0:0`; exact QKV `<128,true,true>` ran 3,008 times
  at 8.628617 us/call with 38 registers/thread, and exact O release-turn ran
  3,008 times at 6.368160 us/call with 38 registers/thread.

## Source, artifacts, and constraints

Canonical remote source:

`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/source_two_lane_final`

Final remote roots:

- legacy performance/alignment:
  `fanin_single_source_fastpath_validation/legacy_mode`
- new-mode performance/alignment/trace:
  `single_edge_publication_fastpath_validation`

Current focused runs use `FASTAFD_ALLOW_DIRTY_SOURCE=1` without a source hash
manifest. `control/final_candidate_source_manifest.sha256` is absent locally
and remotely. Historical experiment manifests remain only with their immutable
old roots. No source sync occurred while a GPU allocation was active.

Preserve the unrelated user-owned `CODEX_PROJECT_RATIO_EP_SWEEP.md` change.
Nothing from this goal has been staged or committed. Do not resume either
frozen Pareto pool unless the user explicitly changes scope.

## Active separated-control DeepEP experiment

The standalone-per-phase implementation is correct but rejected on latency.
Remote CPU job 6473516 passed 60/60 and focused GPU smoke 6473688 passed BF16
and FP8 normal/prearmed comparisons. Exact EP4 job 6473693 measured
26.0383275 ms E2E and 25.9808789333 ms strict CUDA, respectively 0.331% and
0.249% slower than accepted job 6472328. It added 376 model graph nodes.
Export job 6473952 attributes 27.843 ms over 6,016 standalone barrier kernels,
more than the 25.306-ms aggregate dispatch/combine data-kernel reduction.

The full-handshake epilogue-tail follow-up is also correct but rejected. Remote
CPU job 6474052 passed 60/60, fresh-build four-rank job 6474087 passed normal
and prearmed BF16/FP8, and exact EP4 job 6474094 completed `0:0` with the exact
564/3,678 attention/model graph shape. It measured 26.071802 ms E2E and
26.0584953333 ms strict CUDA, +0.460%/+0.548% versus accepted job 6472328.
Trace export 6474347 shows dispatch-copy at 7.768 us and combine-reduce at
8.880 us, versus 3.811/4.173 us without their tail waits; both role graph means
regress by about 0.075 ms.

The active refinement therefore posts readiness without waiting from the last
epilogue CTA. Intervening compute absorbs signal propagation; the later data
grid sends no control traffic and validates only the local preposted word before
its cooperative CTA release. The path fails fast outside direct NVLink scale-up
and retains the 3,678-node target. CPU job 6474402 passes 60/60 and dependent
fresh GPU smoke 6474403 passes normal/preposted BF16 and FP8. Decisive exact
EP4 job 6474505 completed `0:0` at 25.7321745 ms E2E and 25.6797822 ms strict
CUDA, 0.849%/0.913% faster than accepted job 6472328, with 15/15 samples, zero
outliers, and 564/3,678 graph nodes. Official scorer job 6474804 completed
`0:0` on the exact sample with 24 prompts / 408 tokens,
top-1/top-10/top-100 all 1.0, and average/max rank 1.0.

## Follow-on exact-A1 EP8 run

Job 6473036 completed `0:0` in 13:47 for uniform 128K, batch six per
attention-DP lane, A1:F1, ATP1, FEP8, and MB2 with the final `fmha-only`
source. EP8 preserves the 1:1 active-GPU ratio with eight attention ranks plus
eight EP ranks, so the job used four contiguous NVL72 trays / 16 active GPUs.
It used the unchanged general bundle runner, `FASTAFD_ALLOW_DIRTY_SOURCE=1`, no
source manifest, a 3,600-second case watchdog, 8,192-token prefill chunks, and
the standard warmup-plus-15 measured graph capture.

E2E median is 25.391392 ms, 39.3834257 TPS/user, and 118.1502771 TPS/GPU.
Strict CUDA mean is 25.3729383 ms / 118.2362074 TPS/GPU with 15/15 samples,
zero outliers, 1.4492% dominant spread, and one 564-node attention plus one
3,676-node model graph per measured step. The accepted pool is one completed,
zero pending/claims/failures. Task root:
`qwen3_ffn_overheads_20260820/single_edge_ep8_validation`.

All raw Nsight reports for the three focused cases were copied locally under
`scratch/qwen3_ffn_overheads_20260820/nsys_three_case_collection_20260823`:
8 legacy-EP4 reports, 8 new-mode-EP4 reports, and 16 new-mode-EP8 reports.
