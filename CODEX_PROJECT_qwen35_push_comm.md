# MegaMoE communication optimization

## Staging checkpoint — September 13

Remote staging checks PASS: six shell launchers, all464saved-case contracts and
all11launch-group plans. Candidate code was synced before checks; no new GPU jobs.
Explicit33-file staging scope: scratch/megamoe/stage_20260913/stage_files.txt.
Remote workspace: FastAFD/artifacts/comm_stage_20260913/workspace; checks/summary.json
and local remote_checks.log retain evidence. No AGENTS/skill changes are needed.


User requested cleanup and git add only; no commit. The validated seven-file
communication selection is now integrated into the local working tree, preserving
the existing staged combine cleanup. The EP64 independent-reference-arena fix,
archive-aware launchers and reusable collection/report/profiling tools are included.
Unrelated attention/GDN model/runtime edits remain outside this staging scope.
Remote staging checks and the final explicit file list are recorded under
scratch/megamoe/stage_20260913. No kernel algorithm changed during this staging.
Skill retrospective: existing skills remain appropriate; reusable fixes belong in
codex_scripts, so no AGENTS.md or skill changes are proposed.

The previous 2,836-line chronological memory is preserved at
scratch/megamoe/stage_20260913/CODEX_PROJECT_qwen35_push_comm_history.md and in the
remote staging artifact directory. The compact result/protocol/rejection records
below replace repetitive running-job checkpoints. Historical next-step statements
in the experiment ledger are superseded by the final completed selection above them.

## Communication optimization and final validation complete — September 13

Campaign validation and requested artifacts COMPLETE; no task GPU jobs remain.
Authoritative artifact root: scratch/megamoe/comm_final_20260912, especially
completion_summary.json, candidate_selection.json and results.md. No meaningful
full-model decode-step gain; no universal latency-optimality or push-impossibility
claim. Selected frozen source ffn_comm_final_pull_integrated_20260912 shares the
validated GDN runtime with ffn_comm_final_control_20260912. Dispatch/combine are
independent: final0/0/0/1 flags = pull dispatch + push-based in-band combine.
Control already includes in-band combine. Selected7files and review patch remain
under review/selected and review/selected_communication.patch. No production/index
edits or commits during finalization; preserve ten pre-staged cleanup files and
unrelated model/attention work. Disabled useful push research paths remain frozen.

Requested SAME-FORMAT standalone HTMLs generated and validated:
- sweeps/qwen3/qwen3-megamoe-comparisons.html:16 displayed rows, three sections,
  maximum batch7/EP4–64, fixedglobal32/EP2–64, fixedrows4/EP2–32, shared120us axis.
- sweeps/qwen35/qwen35-megamoe-comparisons.html:15rows, maximum batch37/EP4–64,
  fixedglobal128/EP4–64, fixedrows2/EP4–64, shared200us axis. Supplemental EP32
  batch40/global640 remains in the full matrix, not substituted into the old view.
All displayed medians independently recomputed from final raw captures; actual
median-duration IKET launch supplies every boundary, overlap computed per launch
before median; aliases/selections preserved. Each HTML's6JSblocks node--check PASS.
No browser rendering inspection; original responsive CSS/JS preserved. Full-curve
plots/decomposition.png/.svg/.pdf regenerated for all464cases; PNG visually checked.
Generic build_megamoe_comparison.py accepts --template/--output/--standalone/
--render-script; per-model html_plan.json records exact command, html_validation.json
proof. It uses the visualize renderer then unwraps the frame for original full-page
layout. Reload full visualize SKILL.md before future visualization edits.

Full11EP sweep audit COMPLETE: Q3 164cases/455aliases, Q35 300cases/1111aliases;
total464cases/1566aliases,1392IKET samples and13920ordinary instances. Historical
bucket3840 retained; actual-target microbuckets1920/768 stay separate. Both audits
complete=true; source/input/route/bucket/six-marker/sample/alias contracts pass.
All23EP64cases pass pre-timing bitwise selected-vs-full-barrier combine. Actual
ordinary specialization flags checked for all464 in sweeps/timed_kernel_flags.json:
in-band1, both push-dispatch flags0, prepared/ready/gated1, debug0. All11groups have
router_timeline.json/router_intervals.csv with13920same-stream ordered calls.
Completed Q3 EP2/4/8/16/32/64 jobs7113515/7114300/7114713/7115033/7115147/7115254;
Q35 EP4/8/16/32/64 jobs7113514/7113891/7114606/7114878/7115214. Final EP64 elapsed
6:22 Q3 and21:37 Q35. Runner11846 exited0 at04:49UTC,180s cadence/max2jobs.

Q35 originalEP647114927 failed2:56 before timing: comparator disabled rank-gated
combine while retaining in-band1, rejected by CUDA static assertion. Workflow-only
exact_combine/ smoke override allocates independent conventional-combine0 reference
arena, then restores selected arena/flag1 before timing. Both EP64 plans use it;
frozen kernel/source untouched. Full recovery all23cases PASS. Failed batchlogs,
failure.json and reference_arm.patch retained. Earlier initial sweeps7113453/7113454
failed archive Git lookup; exact-image CPU7113485 proved Torch imports and archive-
aware profiler override fixed workflow only. Neither is kernel latency evidence.
One-hour EP64 limits justified by historical24:28 runtime; original requests and
scheduler adjustments preserved. Do not attribute a queue gain without evidence.

Fresh normal EP8/128K real-prefill alignment PASS: Q3 7112597,48prompts/816tokens,
alltop1; Q35 7112542,168prompts/2856tokens,2853top1/alltop10. Every ordered prompt/
token prefix/model profile/position/finite score audited against pinned same-model
vLLM references. Reference inference reused, not recomputed. EP8 correctness is
separate from targetEP16/32 performance; alignment_summary.json records proof.
Both same-allocation full-model source comparisons pass unchanged15-step stability/
coverage gates,0outliers: Q3 7113289,25.3333868→25.3435156ms (+0.039982%);
Q35 7112916,22.612068267→22.601254333ms (-0.047824%). Performance uses perfect
routing/randomized resident KV/GDN, no real prefill; normal alignment above separate.
Q3 recovery plan directory qwen3_perf_recovery_01; original7112992 setupindex60→1
failure retained. Representative attention/FFN captures per arm and metrics local.

Same-stream routed completion improves Q3 53.828941→53.453807us and Q35
80.717102→77.314289us across2820/1800calls. Q3 earlier launch/greater PDL overlap
explains longer kernel duration but earlier routed completion. CPU attention unions
from all15graphs/arm exactly match step metrics: Q3 wait union decreases80.567us,
wait-only just0.693us; Q35 wait union decreases282.901us, wait-only grows2.055us.
Most reduced waits already overlap useful attention; FMHA stable. These explain why
adding routed interval savings predicts the wrong step gain. Not individual-wait
causal attribution or cross-rank clock alignment. attention_trace_analysis records.
Q35 dense N4096/K1024 grows5.350542→36.892050us; all1800calls overlap MegaMoE,
only0.623609us average positive extension past it (721calls,max4.16). Not31.54us
additive step cost or isolated contention attribution. CPU exports7113376/7113469/
7114802 used saved captures, no extra GPU inference or bulk SQLite copy-back.

Latest matched MAX-rank micro means, combine fixed1: Q3 pull36.570332,parent
push37.164459,epilogue-sender push37.184706us; Q35 pull46.384352,parent47.824064,
new47.570887us. Q35 sender improvement supported but still1.186535us behind pull;
Q3 parent/new difference below variation. Full rejection ledger in
comm_resume_20260911/rejection_reasoning.md distinguishes intervention evidence,
wrong assumptions, unresolved attribution and unmeasured open ideas. Do not reopen
unmeasured variants without a concrete hypothesis. No additional microvariant selected.
Final candidate bucket3840 target medians Q3EP16/global192:29.600us ordinary,
8.736dispatch exposure/13.792compute/4.736combine exposure/.192tail; Q35EP32/global640:
35.872us,10.368/17.344/5.664/.192. Separate medians are not additive, nor matched A/B.

Reusable collect/audit: megamoe_input_sweep.py fetch LOCAL_MODEL --remote-root
REMOTE_MODEL --remote-script REMOTE_HELPER (plain ssh oci-hsg). Explicit completed
files only, no cache/SQLite/other-rank traversal. Final script/compact artifact sync
is recorded in completion_summary.json and completed_records_files.txt.

User clarified average E2E means MegaMoE invocation latency across old/new sweeps,
not decode-step latency. Equal shared-case scope164Q3/299Q35 (exclude new global640).
Mean per-case percentage gains: kernel2.9691%/3.7738%; dispatch reduction
-13.1193%/-5.4676% (increases); combine reduction18.2137%/21.0082%.
Reduction of mean latencies differs: kernel1.5449%/2.6337%, dispatch-12.6099%/
-5.0394%, combine20.3210%/20.8731%. Historical-vs-final, not paired causal A/B;
old3/final30 ordinary samples. Artifact sweep_reduction_report.md and
sweeps/historical_reduction_summary.json record both definitions and raw means.


## Protocol

- Alternate two payload generations per microbatch lane. Destination clears a
  consumed generation and grants its next reuse credit EARLY, while intervening
  compute/other generation is active. Sender acquires that destination credit
  before writes. No post-payload source-ready flag on the latency-critical edge.
- Consumer polls payload words until they differ from reserved -0.0 bit pattern,
  using the values returned by those loads. Do not poll one word then assume the
  rest of a vector/row arrived. Preserve existing arithmetic/reduction order.
- Prefer 32-bit transport-word sentinel0x80000000 (not floating-point equality).
  For packed BF16/FP8 it represents only signed/unsigned zeros; normalize that
  exact payload word to0 before sending. Valid negative-zero words otherwise
  collide with the sentinel. Scale/metadata representation also needs proof.
- Credits must follow completed consumption and clearing; double buffering alone
  does not prove safe reuse. Adversarial delayed producers/consumers, generation
  wrap/replay, changing routes, empty destinations, padding and true-zero payloads
  must pass on arbitrary normal routes. Avoid global critical-path reset/barrier.
- Dispatch: decouple route/count publication from payload-push tail. Existing
  consumer still needs route metadata/counts. Publish metadata before fanout, then
  detect arrival from each data word; retaining a count epoch that waits for all
  payload pushes would defeat the latency objective.
- Combine: retain producer direct remote stores, replace rank-wide payload-ready
  waiting with local in-band loads; fuse clear/early next-credit into consumer
  work. Assess whether local grid/barrier placement still prevents useful early
  consumption. Current TMA path must remain available until same-code wins.

First gate should be a small multi-GPU protocol/component ABBA with real producer-
consumer dependency, concurrent work and delayed arrivals; measure input-ready to
usable output, startup/tail/credit waits and steady-state latency. No64GPU run for
an isolated speculative change. Integrate only after exact numeric/state and
repeated-generation correctness, then include it in the next accumulated candidate.

## Primary-source references inspected

NVIDIA TRT-LLM moeAllReduceFusionKernels.cu has Lamport-style transport: 32-bit
negative-zero detection/canonicalization, volatile128-bit loads checking every
word, and THREE rotating buffers with separate clearing bookkeeping. It is a
reference for the primitive, not proof our asymmetric sparse two-buffer protocol
is correct. Its numeric reduction changes are unrelated; preserve ours.
https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.cu
TRT-LLM ops.py allocates3 Lamport buffers and initializes them to-0.0:
https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/distributed/ops.py
vLLM MiniMax workspace documents the same sentinel:
https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/minimax_rms_norm/lamport_workspace/



## Rejected optimizations: expectations and evidence

Current conclusion: the micro campaign is complete; integrated pull is selected
for final model validation. Later entries below supersede earlier next-step or
running-job statements. [Consolidated results](scratch/megamoe/comm_final_20260912/results.md)
separate useful push improvements from push-over-pull claims and list remaining
final gates. No universal push impossibility or optimality claim is established.


For every rejection record the expected dependency saving, actual comparison,
wrong or incomplete assumption, causal confidence, and next discriminating test.
A measured regression establishes rejection, not its precise cause. Code/SASS can
establish a dependency or resource fact; phase medians cannot assign its cost.
An improved intermediate path that still loses to pull is not a failed experiment.
Instrumentation failure and noisy/incomplete evidence are separate outcomes.

## Causal completion requirement (latest user direction)

A losing push benchmark is not sufficient completion. A concrete explanation must
include a controlled intervention and its measured effect on the predicted critical
path, corroborated by generated instructions or protocol dependencies. Label a
remaining unexplained gap explicitly. Parallel publication is the next clean
intervention for the full-packet regression. Early-inbox versus direct-pool is a
useful architectural comparison but changes several operations; by itself it cannot
assign the whole gap to the destination-address reply. Any static-routing oracle
must be labeled diagnostic work moved outside timing, never a general speedup.

## Resumed target comparisons (7096816/7096817, completed)

**Release-only row arrival, standalone: no general gain established.** Expected:
removing the sender acquire and unused atomic return would speed row publication.
Actual direct dispatch23.808→23.840us (Q3EP16),31.296→31.360us (Q35EP32); ordinary
kernel48.3317→48.5387 and60.5930→59.9765us. Q35 harness is essentially unchanged;
Q3 original-direct repeats drift57.0078→54.7335us while release stays54.76us.
Wrong expectation: acquire completion follows the row's publication. These inputs
have96/200 routes per rank and600 available sender warps, so each active warp sends
one row; there is no next row on that warp to accelerate. Release ordering, payload
writes and consumer acquire remain. SASS confirms the intended instruction change;
its location in the dependency graph explains why lower instruction cost need not
lower first-consumer latency. It does not prove every possible cleanup/throughput
benefit is zero. Do not promote on the small inconsistent ordinary-kernel change.
A deeper multi-row workload could test throughput, but is not a reason to spend
more GPUs before addressing the actual target critical path.

**Serialized full-layout packets: rejected as implemented.** Expected: replace
sender remote count/offset reads with local reads, saving remote request latency.
Actual primary55.8707→109.9664us /69.5145→138.0836us and ordinary kernel
48.3317→103.7461us /60.5930→129.1669us versus direct parent. Direct dispatch is
64.800/87.808us; old clipped dispatch58.624/81.280us is a different metric.
Wrong expectation: local consumer loads are not free of upstream dependencies.
The destination publishes12–13 packets sequentially per expert; each packet copies
a full rank-count vector then releases a ready epoch. We exchanged parallel remote
fetches for replicated data and serialized publication on few metadata warps.
Code and actual routing enumeration prove that serialization exists; they do NOT
prove how much of the regression it causes. Parallel full-layout publication now
changes only packet assignment, preserving packet contents/order/consumer/payload.
Its matched comparison will test serialization; compact packed-base publication
separately tests the unnecessary vector and ready-word protocol.

**Direct-pool push as replacement for pull: still rejected, useful experimental
parent.** Expected: remove inbox polling and shared-to-pool copy. Actual fresh
target pull/direct kernel32.2699/48.3317us and39.3248/60.5930us; original EP8 raw
pool79.4528us vs inbox48.3403 and pull44.0309us. Wrong expectation: destination
addresses are not available to the sender for free. Expert packing requires global
counts, offsets and source row placement before the first useful write. Source-
prefix/layout publication improved this parent but did not eliminate the handshake,
scattered scale placement or per-row readiness. The data dependencies are proven;
their separate latency contributions remain unresolved. Rank-segment and packed-
base experiments isolate address-description work before judging push direction.

## Earlier experiments: corrected expectations and evidence limits

| Experiment / disposition | Expected saving | What went wrong with the expectation; evidence and remaining uncertainty |
|---|---|---|
| Router fanout / rejected on current Q3EP8 | Push avoids a remote request and destination dedup reduces copies. | Actual EP8 harness53.0641→66.0861us with stable controls (7083888). Historical top10 routing averaged1.25 destinations; current top8/top10 sends to8/10 distinct ranks, eliminating that dedup advantage. Q3 selector has only2CTAs and serializes fanout per token. Nsight shows launch overlap already works and no reported spills; neither missing overlap nor more bytes alone explains the measured gap. |
| Native-grid sender / rejected as pull replacement | More sender CTAs should cure router serialization. | All four current geometries still lose (e.g. Q3EP16 38.6908→40.6567us, Q35EP32 48.1254→51.6089us for1KB). Launched150CTAs is not150active senders; equal grid also leaves source staging, inbox loads, polling and clearing. Corrected expectation: parallel placement removes one constraint, not all local work. These components were not independently timed. |
| Early sender / useful intermediate, not replacement | Overlap metadata/setup with transport. | Recovers about1us versus late native but stays roughly2.2% above pull. The gain supports earlier work placement; assuming it removes the whole metadata dependency was wrong. Retain as experimental parent. |
| Metadata helper warp / no useful gain | Independent metadata work enables compute earlier. | EP8 dispatch11.072→11.040us, harness52.7602→52.7861us (7089358). A remaining sender-dependent CTA join still gates progress. Independence in one producer does not remove a later join. This is a code-supported explanation, not an isolated cost measurement. |
| Direct fenced receiver / rejected | Remove shared-to-pool TMA stage. | EP8/16 harness52.4634→53.0548 /39.7976→40.4169us. Generic stores require cross-proxy and publication ordering that TMA previously supplied. Removing a copy also removed its implicit ordering benefit. Conservative fences/scattered writes are plausible costs; no isolated fence cost was measured. Unfenced numeric pass did not prove ordering and its timings were excluded. |
| Packed payload / no general gain | Fewer separate metadata/SF stores. | EP8 dispatch improves0.512us but EP16 worsens0.224us. A4256B row takes five1KB packets instead of four activation packets plus small scalar stores. Assuming packing necessarily reduces transaction/staging work was wrong. Still loses to pull; exact packet counts are known, transaction latency is not. |
| Full-row pipeline / not accepted, profile failure | Receive while outgoing sends continue; avoid partial packets. | Primary pipeline vs pull+4.77% EP8/+1.61% EP16. Optional instrumentation used R49/R51 after48-register deallocation, a concrete capture defect. Ordinary exact passes do not validate instrumentation; missing phases cannot explain the performance. Neither failure proves pipeline correctness is bad nor overlap is beneficial. |
| Signaled inbox / rejected | One ready poll plus local TMA beats polling every payload word. | EP8/16 exposed net24.128/18.176us vs pull16.800/15.392us (7091511/7091510). Separate publication adds an ordered dependency before reception. Fewer receiver polls can be outweighed by later readiness and staging. Actual split remains unmeasured. |
| Weaker stores / relaxed repeated polling / not promoted | Avoid unnecessarily strong instructions. | Strong/weak/relaxed primary65.7377/65.0058/65.1749us vs pull51.5746us. Final release/acquire and payload dependency remain; relaxing repeated checks cannot remove them. Weak-only profile missing due environment override; do not claim its phase gain. |
| Four receivers or vector receive / rejected | More workers or cheaper local loads improve consumption. | Full kernel three-TMA55.6832, four-TMA55.9712, four-vector55.8997us vs pull43.7963us. More receivers did not advance readiness and changed overlap. Assuming receiver capacity was the bottleneck was unsupported. No direct contention/stall attribution. |
| Async ready and early source-ready / useful intermediate | Remove synchronous completion and unnecessary full-selector wait. | Kernel57.0197→55.3621us for async; independent comparison53.6160→51.0229us for early source-ready. Both help parent but remain slower than pull. Dependency removal was useful; expecting that single dependency to explain the whole gap was wrong. |
| C++ stores/restrict / compiler hypothesis rejected | Compiler should issue independent loads earlier. | Exact7092750 passed, SASS still alternates LDG/store; only store form changed. Source appearance did not imply generated scheduling. No performance run needed for that proposed mechanism. |
| Paired loads / useful intermediate | Expose two independent memory operations. | SASS confirms LDG/LDG/STG/STG within48 registers; kernel51.0229→49.3589us, pull43.7493us. Mechanism works, remaining gap disproves only sufficiency. |
| Eight-load instrumentation / incomplete, not a latency rejection | More memory-level parallelism. |7093619 ordinary candidate47.8614us but IKET CUDA715 and remaining profiles absent. Reviewed registers do not establish this crash's cause. Do not invent numerical or spill diagnosis; missing phase evidence prevents acceptance. |
| Batch-ready / exact only, unmeasured | Amortize per-row publication. |7093721 exact passes. No performance run; cannot call it rejected or slow. A source-wide completion ticket may delay first rows and requires observing empty-source signals before reuse. This is an open risk, not a result. |
| Destination offset and source-prefix publication / useful intermediate | Avoid repeated remote preceding-expert scans and long local source scans. | EP8 layout kernel80.7019→75.6085us, still far above pull44.1504us. Source-prefix search removes up to128 local scan steps but cannot remove the remote layout dependency. Arithmetic work counts do not directly measure stall time. |
| Old early in-band combine / rejected | Reduce as soon as individual contributions arrive. |7076249 primary+7.40%, stable controls. Moving cleanup's cross-rank barrier after reduction makes the slowest consumer a completion dependency. Early local progress is not early global completion. This dependency exists in code; counters overlap and cannot quantify a causal split. |
| Two-vector in-band dispatch / rejected | Eliminate spills by reducing register demand. | Spills removed but7076249 primary+1.48%. Independent loads also decreased; spill-free is necessary resource hygiene, not sufficient latency evidence. Serial load cost is plausible, not independently measured. |
| Register-slack/async-cleanup old variants / not promoted | More load parallelism and preserve cleanup overlap. |7076436 dispatch+1.35%, async combine+1.47%, both+3.14% vs control means; variability limits tiny conclusions. Available registers and fewer joins do not ensure useful overlap at actual arrival times. |
| Fresh minimal in-band combine / retained opt-in, default promotion rejected | Earlier contribution consumption should lower entire kernel. | Combine saves1.152–2.272us, but Q3EP8 kernel43.4624→43.5605us and primary+0.49%; Q3EP16/Q35EP32 improve modestly. Post-combine tails grow. Assuming exposure gain equals full-kernel gain was wrong; phase movement is not automatically speedup. |
| Earlier cleanup join / generic replacement rejected | Move an existing wait into dispatch cleanup and overlap it. | Exact passes; Q3EP8 kernel improves, Q35EP32 38.6741→39.3707us worsens despite combine6.016→4.736us. Tail1.920→3.744us. The wait's location changes when progress becomes globally visible; moving it is not deleting it. No causal subtraction of separate captures. |

Historical launch/configuration failures (wrong source field, missing env propagation,
exclusive-create references, stale profile variant export) are workflow errors,
not optimization failures. Their logs remain with each experiment. Older combine
mask and other pending labels in chronological memory need terminal evidence before
an acceptance/rejection classification; do not infer a result from an abandoned plan.

Current raw evidence: [completed results](scratch/megamoe/comm_resume_20260911/results.md). Historical links and job IDs
in CODEX_PROJECT_qwen35_push_comm.md point to the frozen experiments above. No old
failure is a proof that every push design must lose. Next work tests explicit
critical-path assumptions rather than accumulating instruction-level changes.

## What the successful cleanup experiment changes in the diagnosis

Full metadata generations7097652/7097720 improve in-band ordinary kernels
32.9888→30.3488us /38.9589→37.1797us and primary means40.7672→38.7614 /
49.1440→47.7849us. All exact trials and full captures pass. Earlier join movement
kept the global dependency; this protocol proves it unnecessary and removes it.
The initial expectation that all saved tail time would remain a saving in every
phase is too strong: EP32 dispatch9.856→10.592us and net16.032→16.448us worsen,
while repeated total latency improves. Inter-rank skew can migrate to next routes.
Do not causally subtract instrumented phase medians from ordinary kernels.
Full results: ../metadata_generations_20260911/results.md. Compact banking is an
unvalidated footprint refinement, not a rejected latency path.

## Completed rank-segment / compact-packet step (EP16)

7106007 all8exact and all4profiles complete. Rank-segment dispatch23.552→23.520us
but kernel46.7296→47.0037us: no latency gain established. The balanced target has
<=1row/source/expert, so physical row order does not change. This was a compact-
address enabler; assuming its cheaper arithmetic would matter was unsupported.
Compact packets DO improve that parent: dispatch23.520→19.936us, kernel47.0037→
44.2091us, primary53.5919→50.9838us. Still reject as a pull replacement (pull
kernel32.2805us, primary38.7330us). A local address load still waits for a pushed
address after destination packing. Do not equate removing remote lookup with
removing that readiness dependency; residual phase costs are not isolated.
Raw result: ../packed_layout_20260911/qwen3_ep16/results.md. EP32 remains due.

Compact metadata-banking EP16 primary repeats38.5155/38.5445us trail the full-bank
parent38.0348/37.9780us while beating original39.8616/39.8750us; final profiles
pending. A confound was identified before choosing a cause: payload arena offset
mod32 changes0→16 on both target layouts. New alignment-only exact7106452 and
planned 2×2 alignment/cleanup comparisons test that hypothesis. Smaller footprint
was not a guarantee of equal latency; physical placement also changed.

## Metadata compaction expectation tested by alignment intervention

The initial compact candidate saved most extra metadata but lagged full-bank by
0.524us primary/0.792us ordinary kernel on Q3EP16. The assumption that a smaller
allocation preserves address behavior was wrong: compact shifted arena mod32 from
0 to16. Factorial7106636 now completes a controlled intervention at fixed protocol:
16→128padding changes ordinary kernels32.1216→31.5115us (original) and30.3296→
29.7429us (compact), about0.6us each. Cleanup/banking at fixed alignment separately
saves1.7920/1.7686us. Thus padding materially affects latency; footprint alone was
an incomplete expectation. It is still not proof of a specific extra NVLink/TMA
transaction count, nor a same-allocation equivalence test of full-bank and compact128.
Q35 full factorial evidence pending. These Q3 measurements use historical3840;
actual deployed1920 target evidence is included in newly prepared dispatch tests.

## Serial publication now has controlled causal evidence

7107044 Q3EP16 complete: unchanged packet/payload/protocol, only assignment changed.
Serial8metadata warps×12packets → parallel96warps×1packet. Primary repeats110.2933/
110.9214→56.8314/56.8714us; dispatch64.864→24.928us; ordinary kernel105.0709→
50.6432us. This establishes the publication assignment/serialization cause for the
full-layout regression. Local consumer reads were incorrectly assumed to imply
early availability; their producer created a much longer sequential dependency.
No12-RTT claim or additive phase cost decomposition. It does not explain the entire
remaining push/pull gap. Both raw triples and deterministic per-rank work inventory
are retained under push_layout_parallel_20260911. Q35 counterpart still pending.

## Early-inbox with local cleanup and in-band combine (Q3 complete)

Expected earlier predetermined-address sends plus removed cross-rank cleanup to
make push competitive.7107319 matched integratedpull/inboxinband primary36.2656/
37.3304us, kernel29.7259/30.0416us, dispatch8.480/10.432us. All8exact/4captures
complete, actualcapacity1920. Wrong assumption: early outgoing payload does not
imply independent destination scheduling. The retained metadata phase follows
all native sends and a CTA join. This dependency is code-proven; its isolated
cost is NOT measured. The comparison alone does not explain the entire gap.
Minimalcontrol primary37.6029us uses slower warp router8.0384us versus integrated
CTArouter6.8043us; purepull MegaMoE kernels essentiallyequal29.6608/29.7259us.
Do not credit the apparent win against minimalcontrol to push. Router-only
minimal port remains useful. Fullresults: ../early_inbox_generations_20260912/
qwen3_ep16/results.md. Q35 matching job7107761 active.

## Address-reply dependency removed: insufficient, Q3 controlled evidence

Static diagnostic7107607 COMPLETED all8exact/all4triples5phase30ordinary. Identical
retained packet, producer, payload, SF, readiness and clearing policy; only current
epoch→anynonzero predicate changes. Fixed routes keep old/current addresses equal.
Retained-current/cached-any kernel44.7701/44.7435us, directdispatch20.576/20.704us.
Cached still exceeds pull32.2581us by12.4854us. Primary0.2238us difference is below
retained-repeat span0.4651us; no robust saving. Expected reply latency was not a
measured exposed wait: local global-count acquire/prefix/readiness/search still
precede the check. Those operations may hide the reply. This rules out the reply
as a sufficient explanation in this workload, not all layouts or a zero hardware
RTT. Remaining gap is explicitly unexplained; next direct-ordinal and source-prefix
contrasts test local work. Q35 counterpart7108030 submitted. Full explanation/limits
in ../static_layout_diagnostic_20260912/qwen3_ep16/results.md.

Q35 early-inbox counterpart7107761 now COMPLETED0:0 all8exact/all4triples. Matched
pull/inboxinband kernels35.7568/38.2037us andprimary46.3845/48.4568us; dispatch
10.208/12.384us. Both repeats lose, so the current architecture is rejected on
both targets. Native inband versus epoch saves0.3414us kernel but combine exposure
5.536→5.696us does not improve; avoid a phase-only success claim. Purepull router
cooperative9.5040 versus warp-local8.2208us supports minimal CTA port testing.
Exact router job7108166 now active. Remaining push gap is still a causal task:
helper join, prefix work and clearance assignment are prepared interventions.

## Instrumentation failure separated from optimization evidence

Static Q35 7108030 passed all8ordinary primary checks before packed-parent IKET
capture failedCUDA715. CPU allocator replay reports126 available registers at
DispatchEnd reached by48-register dispatch warps; conditional deallocation is not
a dominator after a helper-role merge. Correcting only the exclusive bound to46
with the original liveness selector changesR47→R39 and no other marker resources.
GPU recovery7108794 tests this hypothesis using unchanged kernel sources. Until
that passes, do not classify it as a push correctness failure or claim a complete
Q35 phase comparison. Original ordinary captures are retained.

The source-ordinal/prefix, actual outgoing CTA join, and clear-assignment candidates
all passed their general both-model exact gates7108456/7108414. Their target latency
contrasts are pending; none is accepted or rejected on correctness timing noise.

## An apparent kernel-duration regression is not an end-to-end regression

Clean CTA router7108685 saves1.297888us paired primary atfixedcompact, but ordinary
MegaMoE duration29.1744→29.8400us increases. A30-call Nsight timeline, aligned to
each call's producer start, shows MegaMoE starts1.0144us earlier and completes
0.3488us earlier. Its measurement interval grew because its start moved earlier.
Wrong expectation: independently shortening upstream work must also shorten a
downstream kernel's duration. With PDL overlap, start times move and kernels can
include different amounts of dependency wait. The timeline proves that interval
shift; it does not assign all waits or explain1.2979us from a different capture.
Do not reject this router on kernel duration alone, or credit it to push dispatch.

Q35 fixed reply bypass recovery7108794 COMPLETE all4triples5phase30ordinary.
Retained-current/cached kernel56.7392/56.4373us, pull39.1029; cached still+17.3344us.
Direct dispatch28.416→27.392us (pull9.344). Original7108030 primary65.619598→
65.461414us. Small effect possible, but fresh reply is not a sufficient explanation
of the large residual. Allocator bound46 made the instrument-only failure disappear;
original sources and ordinary workloads unchanged. Full provenance/results retained.

## Source ordinals: earlier local compute is not completion

7109073 Q3 all8primary/all4triples complete. Packed→ordinal+retainedprefix dispatch
20.480→16.736us, but primary50.359852→50.046787us and kernel44.4341→44.1600us only
slightly improve. Per-sample DispatchStart-relative medians: ComputeEnd35.200→
31.424us, CombineEnd41.536→41.280us. Exposed combine6.048→9.664us accounts for
most of the apparent early-phase gain moving to a later wait. Wrong expectation:
earlier local compute was treated as a proxy for final output readiness. The trace
locates the compensation; it does not identify late remote contributors or prove
slower transport. Ordinal enumeration also changes route-to-warp order; causal
attribution solely to removed binary search is not warranted. Primary is MAX over
ranks; these profiles are rank0. Fullresult/phase_timeline saved percase.

Atfixedordinalmapping, removing the unused source prefix saves2.701788us primary/
2.4320us kernel. SM0 no longer scans128entries/4prefix groups before publishing
layout-ready to responder/sender warps. Same17STL/noLDL in both target dispatch
spans, maxR43/45; no spill-removal signature. This is controlled evidence for the
producer-prefix dependency, not an instruction-by-instruction latency assignment.
Still loses to pull by8.745251us primary/9.5979us kernel. Q35 counterpart7109459
running; candidate remains experimental. Replicated destination-prefix and SF
transport counterfactuals are prepared to test the residual, not assumed wins.

## A removed join was not the last join

Q3 helper7109468 completed all8primary/all4triples. Removing fullCTAjoin saves
only.084480us primary/.0288us kernel. The expectation of early receive overlap
was wrong: a later unconditional four-dispatch-warp join still sits before the
receive loop. Existing PushEnd markers show2–3sendingwarps in each of150CTAs,
leaving the fourth idle but blocked. Median send-finish skew .192–.224us; PushEnd
means shared-source release, not remote arrival. New candidate removes only that
remaining initial join for prepared native routes, retaining metadata acquire,
own source-release and later cleanup joins. Its exact checks run in7109891.
The evidence explains why the earlier intervention could not create the intended
overlap; it does not yet prove the remaining barrier's cost or whole push gap.

Q35sourceprefix7109459 completed all8primary/all4triples. Removing the unused pass
saves6.869343us primary/6.1984us kernel; still9.786360/11.4091us behindpull.
Ordinal-only ComputeEnd moves5.792us earlier but CombineEnd only1.056us earlier,
again showing compensation after localcompute. Same17STL/noLDL broad ordinary
span, withR70/R68 confined to helper branch that skips dispatchDEALLOC48.
Detailed results and all phase samples retained; no final candidate selected.

The follow-up source review found a third startup dependency:384-thread dispatch+
epilogue join still follows the dispatch-only join. Thus skipping only the latter
also cannot create receive overlap. Do not time that as a complete decoupling.
Fullchain candidate removes both startup join sides when their legacy NVLink
barrier exclusion is already unnecessary, retains all latercleanupjoins, and adds
IKET-only ReceiveStart to test overlap directly. This corrects the expectation
before target GPU timing; general exact tests are still required.

## Dispatch and combine are independent optimization axes

Keep the combine implementation fixed for every dispatch contrast (native inbox
targets use in-band; direct-pool factorials use epoch). An earlier local ComputeEnd
can expose more subsequent waiting without changing combine work. Neither a smaller
nor a larger exposed-combine interval alone demonstrates a combine transport change.
Current full-startup candidate7110377 has eight primary timings: pull36.2444,
pushparent37.2855, dispatch-only37.6397, full-removal37.1768us. Full-removal's .497us
repeat spread exceeds its .109us mean gain; profiles/overlap proof pending. The
opening .35us gain was provisional and must not be presented as an established win.

Selector thread-bound7110445 failed CUDA compilation (undefined host kExperts),
not candidate numeric correctness or runtime behavior. Derive the constant from
experts_tag; recovery7110688 uses a separate source/cache/plan. No performance
conclusion can be drawn from this setup failure.

## Complete startup removal: overlap works, full win still absent

Q3 recovery7110749 all4triples5/30 completes. Full-removal ordinarykernel30.2677→
29.9029us versus parent, but pull29.5424 remains faster. Primary gain .1087us
below candidate .497us repeatspread. ReceiveStart nowprecedeslastPushEnd in17/24
receivingCTAs in each of4nonspiking samples; retained anddispatch-only0/24. First
longspike includedseparately, nooverlap. This proves dependency removal produced
the intended overlap, not a totalpushwin. Wrong expectation would be treating
one earlyreceiver as a readycompute tile: only1/4active receiverwarps was freed,
and the12-rowtile waitsall12arrivals. Concrete source predicates and mapping
supportthat structural constraint; particularcriticalrows remainunmeasured.
Same-source4KBchunks afteralljoinremoval are a justifiednewtest, withcombinefixed.
See early_inbox_startup_joins.../qwen3_profiles_recovery/results.md and rawtimelines.

Q35 counterpart7110856 completesall8primary/all4triples. Fullstartupremoval saves
.583395us primary and.6549us ordinarykernel; candidatevariation.0082us muchsmaller
than gain. Still1.5330us primary/1.4944us kernel behindpull. Receiveoverlap40–46
of50CTAs ineachof4nonspiking samples,0withjoins; all4warpssend, so earlyfinishers
nowreceivewhile othersfinishremainingjobs. Parent/candidate combineexposure5.728us
andtail.192us unchanged. This establishes an independent dispatch improvement,
not a combine improvement or a push-over-pull win. FourKB specializations passed
all4generalmodel/combinesuites7110975; same-source chunk targets7111806/7111831
nowrunning, with usefulpullcontrols.

## Clean CTA versus integrated pull is not a router-only contrast

Currentfour-arm targetprimaries: Q3CTA256/12836.689676/36.680630us (negligible);
Q35cleanCTA/integratedpull46.664244/46.522736us (.141508us gap, controlsstable).
A targetedsource diff identifies a non-router difference: integratedworkspace
reserves two native receive-row descriptors perrank in EACH metadata bank, even
when push isdisabled. Expertcount pointers shift256B(Q3)/512B(Q35) perbank and
whole workspace size shifts512B/1024B, while128B alignment remains. Python also
constructs inputviews earlier (outsidetiming). Therefore the full-source delta
cannot be assignedsolely to selectorport/compiler quality. Cache/layout effects
are plausible, NOT measuredcauses. A padding-only clean-source intervention would
isolate this ifneeded; do not infer that smallerfootprint must be faster. Preserve
the integratedpull path and keep finalsource selectionopen.

## Larger chunks after full startup removal: scheduling benefit, mixed latency

7111806 Q3 and7111831 Q35 complete all8primary/all4triples. Same native source,
combine fixed1, chunks1024/4096. Q3 primary37.534477→37.669184us regresses despite
more receivingCTAs overlapping sends (17–19/24→24/24). Q35primary47.963997→
47.756216 improves, overlap37–44/50→50/50, but remains1.233480us behindmatchedpull.
Wrong expectation: freeing some dispatch warps and reducing sender-job count should
necessarily make whole tiles ready sooner and reduce full latency. Own senders
still own receiver rows; fulltile readiness is stronger than firstreceiver entry.
This is a concrete dependency constraint, not proof of which exact row is critical.
Firstspikes retained. PushEnd marks sharedsource release, not remote arrival.
Rank0 short kernels change opposite to primary on BOTHmodels (.1888Q3improvement,
.2325Q35regression); producer-relative finishes agree with their shortkernels.
The capture/rankscope mismatch is unresolved; do not invent rank imbalance or a
constant launch delay. Reject4KB as a universalreplacement, retainresearchpath.

Follow-up implemented: ~20line epilogue-sender1KB prototype stages in initiallyidle
C/Dscratch, frees ALLdispatchwarps, joinsallepiloguewarps beforeC/Dreuse. Eligible
senderwarps alsoincrease(8at targets), so notpure role-onlycontrast. Exact7112011
all4model/combinearmsPASS;6entryspans maxR39–44withininitial128,noLDL/STL. Q35target
7112115nowrunning withpull/parentsender/candidatesender andcombinefixed. Fullmodel
validation stillpending; no win or general4KBsupport claimed.

## Kernel boundaries and repeated invocation cycles are different metrics

7112115Q35sender completedall6/all3: primaryparent47.824064→47.570887us improves
.253178, rank0kernel37.6736→37.9979 worsens.3243. SameSQLite29completeproducer
cycles51.406345→51.154759 improve.251586, because gapafterMMfalls4.290207→3.766069.
This resolves the apparent signcontradiction for thiscase withoutnewGPUwork.
Wrongexpectation: kernelend (evenproducer-relative) includesallinvocation cost.
It omitsthegap before nextproducer. This is observedcostlocation, notproofofdriver,
clock orrankcause. Actualprofilewrapperforcesgraph1/30warmups, primarygraph20/
5000warmups; absolute levels shouldnotmatch. Updatedroutertimelinehelperretains
29intercallgaps/producercycles andoriginal30kernelrecords withoutchangingmetrics.
Q3chunkcompletecycles40.902621→40.905931 eliminateapparentkernelgain; primary
+.134706stilldiffers. Q35chunkcycles51.581793→51.845517stillopposeprimary-.207781:
NOT universallyresolved. Fullmodelcomparison remainsessential; no extrapolation.

Q3directreuse7112086completedall8/all4: full/SF/payload/both primary50.792288/
50.850401/50.649193/50.335333; kernels44.6357/44.5536/44.4544/44.0853. Bothgain
.456955primary/.5504kernel, noSF-onlyusefulgain. PayloadreuseadvancesComputeStart
3.232us but expandscompute span3.040us, littleearlierfinish. Mostdirectpoolgap
survives removalofbulktransferwork, contraryto bandwidth/scatter-dominance guess.
Compilerbranchesactuallyskipwork, allactualspans17STL/noLDL,R43–45. Notpurewirecost
or productionoptimization. Q35counterpart7112381running; source/layout/readiness
remainuntouched, so no completewhole-gapcause claim.

## Epilogue sender: Q35 gain, no general replacement

Q37112378 all6/all3complete: parent37.164459/new37.184706us primarydifferencebelow
.118/.122repeatranges. Pull36.570332mean,bothtrialsfasterthannewpush. Ordinarykernel
30.0181→29.8901 improves, butsamecapturefullcycles40.505379→40.632276 regress as
intercallgapexpands.269241us. Receiveoverlap22/18/23/18→24/21/23/23of24; outgoing
finish~4.8–5.2usunchanged. Wrongexpectation: freeingreceivers andaddingeligible
senderwarps mustyieldagenericgain. Q3jobs384alreadyfit600warps; Q35jobs800hadasecond
wave, whichthisdesigneliminates. Exactjobmapping andmarkers supportthisdifference,
notanisolatedquantificationofeveryremainingcost. KeepQ35.253178gain asrealwithin
itscase, rejectuniversalreplacement. Selectretainedintegratedpullforfinalgates.

Q35transfer7112381 completesall8/all4. Full/SF/payload/both primary64.801403/
64.541115/64.123447/63.730728us; kernels55.8400/55.2213/55.2149/54.3136. Bothreuse
saves1.070675primary/1.5264kernel, mostcostremains. SFonlygain.260288belowcontrol
.285225span. All30samplesretainedincludingpayload67.136usoutlier. Fourbroadspans
17STL/noLDL,R70inwarp7helperthatskipsDEALLOC48, notillegaldispatchuse. Bulkwork
contributesbutdoesnotexplainthewholegap on either model. No purewirebound, no
productiongain, no newmicrovariant; proceedwithintegratedpullfinalgates.

Classificationclarification: epiloguesender isretained as a usefulQ35push improvement,
withQ3primaryneutral/inconclusive(.020usdifferencebelowvariation). This isnot a
provenregression warrantingdiscardingthepushcandidate. Whatitdoesnotreplace is
pull: bothmatchedmodelsstillfavorpull. Preservebothparentandepilogueresearchpaths;
do notmislabelone-modelgain/other-modelneutralas a failedoptimization.


## Final integrated pull: scoped gain, no full-model step gain (September13)

Expectation: general metadata/router improvements measured in isolation might
reduce full-model decode latency. Both normal EP8/128K alignments pass; paired
EP16 Q3/EP32 Q35 final step changes are +0.039982%/-0.047824%, all15steps and
zero outliers per arm, so a meaningful full-model speedup is not established.
This does not reject every component or justify calling push impossible.

The wrong expectation would be to add kernel duration savings as if they all
lay serially on the global step critical path. Existing full-model representative
FFN traces show Q3 router-to-MegaMoE end53.828941→53.453807us, despite standalone
MegaMoE duration43.938672→44.517308us. Producer-relative launch occurs0.953771us
earlier and selector/MM overlap grows3.085765→3.442882us. Q35 routed interval
80.717102→77.314289us, with shorter launch delay and MegaMoE duration. All2820/1800
measured calls retained, same-stream pairing and launch order validated. These
intervals establish scoped in-model gains, not additive step savings.

Q35 same dense N4096/K1024 specialization5.350542→36.892050us; all1800calls overlap
MegaMoE on another stream, with overlap3.696053→36.088459us. Other dense shapes
and GDN stable, waits shorter. This observation suggests changed overlap/resource
interaction, but does not isolate contention or prove this GEMM delays the join.
Causal follow-up would need controlled concurrency or dependency/critical-path
measurements; no new GPU variant or recapture is selected during final sweeps.
Evidence: ../comm_final_20260912/{qwen3_perf_recovery_01,qwen35_perf}/comparison.json
and trace_analysis/{control,candidate}_families.json; Q35 dense_overlap.json.


Additional CPU interval check: the overlapping Q35 N4096/K1024 dense GEMM ends
before MegaMoE in all1800 control calls, but after it in721/1800 candidate calls.
Its positive extension beyond MegaMoE averages0.623609us over all calls (max4.16us).
Thus the dense duration increase is largely overlapped, with some changed completion
ordering; it is not a31.54us additive step penalty. These intervals alone still do
not establish actual join/step critical-path delay. Evidence:
../comm_final_20260912/qwen35_perf/trace_analysis/dense_completion_overlap.json.

## Final full-model speedup expectation: attention overlap evidence

CPU7114802 exports all four saved attention traces; all15 graph spans/arm match
reported step metrics exactly. Same-graph useful FMHA/publication union plus
wait-only plus no-kernel intervals partitions the observed step. Q3 wait union
decreases80.567us, but wait-only just0.693us; Q35 wait union decreases282.901us
while wait-only grows2.055us. Most reduced waiting already overlaps useful work.
FMHA per-call durations remain essentially identical. The assumption that summing
shorter FFN/router intervals predicts step savings was wrong. This directly shows
why the measured wait reduction has little exposed effect; it does not assign
every wait change to one FFN operation or prove cross-rank critical-path identity.
Evidence: ../comm_final_20260912/attention_trace_analysis/comparison.json and
per-arm family JSONs. No new GPU run.

### Final EP64 reference harness correction — September13

Q35 job7114927 failed before timing because the comparator set rank-gated
combine=false without disabling in-band combine, violating the explicit kernel
contract. The expectation that changing only the gate produced a conventional
reference was wrong: in-band combine also changes the combine protocol and arena.
Workflow-only recovery7115214 uses an independent conventional-combine0 arena,
then restores the selected arena/flag1 before timing. First two completed inputs
(global8/16) pass bitwise comparison and the3phase/30ordinary/source/input audit.
This is harness setup evidence, not a selected-kernel correctness failure or
performance rejection. Frozen source and measured kernel are unchanged. Final
remaining EP64 cases still require their own gate. Patch and failed logs are in
`comm_final_20260912/sweeps/workflow/exact_combine` and `sweeps/qwen35/ep64`.

EP64 correction final result: recovery7115214 completed0:0 in21:37; all19Q35
inputs plus all4Q3 inputs (7115254,6:22) pass bitwise comparison and final contracts.
Reference-only fix confirmed across the full intended EP64scope; no kernel changes.
