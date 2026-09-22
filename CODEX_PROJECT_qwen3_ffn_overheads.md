# Qwen3 FMHA-Only Attention Worker and FFN Pipeline

Historical attention-worker, FFN pipeline, capacity and model-performance records.
The kernel-only optimization history (August25–September9), accepted checkpoints,
measurement caveats and rejected directions are consolidated in
[MegaMoE optimization](CODEX_PROJECT_megamoe_optimization.md#historical-pull-kernel-checkpoints).
That document owns the current kernel selection and validation status. Dated active/
pending statements below describe their original model campaigns, not new work.

## Active 1K A1:F1/EP4 capacity boundary (2026-09-04 PDT)

Slurm job `6921834` completed `0:0` in 19:48 as the first real maximum-batch
probe for the current
FMHA-only FP8xFP4 FFN path with cost-preserving perfect routing.  It uses two
trays/eight active GPUs, A1:F1, attention DP4, EP4, MB2 `450+450`, uniform
1,024-token prompts, and 3,600 total requests.  The exact-page pair is memory
ratio 0.97 plus 15,300 pages / 979,200 KV tokens per attention GPU, so batch
900 is the raw KV ceiling and exceeds `7*128=896`.  The pmap launch contract
avoids treating the launcher's conservative short-ISL scheduler estimate
(887) as a physical ceiling; acceptance still requires the real runtime to
reach one full-resident warmup and 15 consecutive batch-900 graph steps with
same-rank MegaMoE, FP4 expert weights, and perfect-routing markers.

Remote task root:
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/afd_1k_max_batch_perfect_ffn_ep4_20260904`.
Run directory:
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b900_n2_20260904_175401_manual_na`.
The exact source head is `e00afda8af6e9fc6ad240ec0bd776b960066a53c`;
the task control directory preserves the dirty-checkout status/diff and
relevant file hashes because the canonical checkout also serves live MiniMax
job `6920969` and must not be changed.  Dry-run and validate-only preflights
passed.  Job `6921834` ran on same-fabric trays `nvl72106-T[05,18]`.  All
attention workers measured 15,516 pages / 993,024
KV tokens at ratio 0.97 rather than the nominal 15,300-page override, proving
that current AFD runtime sizing remains memory-ratio-bound despite the parsed
server argument.  The coordinator admitted local batch 900; all eight workers
reached runtime-ready with graph bucket 450, then produced one full-resident
warmup and exactly 15 target graph replays.  All 3,600 samples contain 17 output
tokens, all eight Nsight reports and the capture marker are nonempty, and the
launcher emitted `FASTAFD_AFD_SUCCESS` without OOM or traceback.  Observed
capacity has a raw ceiling of 912 but a scheduler-admissible ceiling of 900;
the latter is the real batch boundary at ratio 0.97.

Ratio-0.98 probe `6922185` used nominal 15,683 pages / 1,003,712 tokens, batch
910, and MB2 `455+455`, with all other contracts unchanged.  Its observed-
capacity gate stopped before graph build or inference after all four attention
workers measured 15,676 pages / 1,003,264 tokens: exact scheduler maximum 909,
while batch 910 needs 1,003,668 tokens.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b910_n2_20260904_181701_manual_na`.
An earlier dry-run of the 0.98 point specified pages without the required
matching token-capacity variable and failed before allocation; no GPU time was
consumed.  Corrected job `6922262` uses batch 909, the exact observed tuple,
and MB2 `454+455`; dry-run and login validation passed with
`FASTAFD_REQUIRE_CAPACITY_MAX=1`.  It was submitted at 18:23 PDT with run
directory
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b909_n2_20260904_182355_manual_na`.
It started at 18:24 PDT on same-fabric trays `nvl72129-T[13,16]`.  All attention
workers reproduced 15,676 pages and passed exact batch-909 admission; all eight
workers reached runtime-ready and captured the padded 910-slot graph, with all
four model logs confirming FP8xFP4 and `perfect_routing=1`.
Job `6922262` completed `0:0` in 21:58 with all 3,636 samples at 17 output
tokens, one warmup plus exactly 15 batch-909 target replays, eight nonempty
Nsight reports, and durable capture/result artifacts.  Ratio-0.99 boundary job
`6922514` was then submitted after dry-run and login validation with predicted
15,836 pages / 1,013,504 tokens, exact nominal batch 918, and MB2 `459+459`.
Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b918_n2_20260904_184828_manual_na`.
Job `6922514` reproduced exactly 15,836 pages on all attention workers and
completed `0:0` in 19:50: exact scheduler batch 918, MB2 `459+459`, all 3,672
samples at 17 output tokens, one warmup plus 15 target replays, eight nonempty
Nsight reports, and durable result/capture artifacts.  The final legal ratio
probe is job `6922758`: ratio 1.0, predicted 15,996 pages / 1,023,744 tokens,
exact nominal batch 928, and MB2 `464+464`.  Dry-run and validation passed; it
was submitted at 19:09 PDT with run directory
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b928_n2_20260904_190950_manual_na`.
Job `6922758` reproduced 15,996 pages and the exact batch-928 admission ceiling,
then failed in 4:11 on every attention rank during FMHA fabric allocation.  At
ratio 1.0 the KV budget consumes all measured free memory, while the fabric
arena also requires one extra KV page and Q transport storage; the run never
reached attention runtime-ready or graph construction.  Midpoint job `6922844`
was submitted after dry-run and validation at ratio 0.995: predicted 15,916
pages / 1,018,624 tokens, exact scheduler batch 923, and MB2 `461+462`.  Its run
directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b923_n2_20260904_191724_manual_na`.
Job `6922844` ran on the same trays as the ratio-1.0 failure, reproduced 15,916
pages and exact batch-923 admission, reached runtime-ready on every worker, and
captured the padded 924-slot graph on all eight with perfect routing.  It was
intentionally canceled at 7:53 before prefill to conserve GPU time.  Upper-
midpoint job `6922965` was submitted after dry-run and validation at ratio
0.9975: 15,956 predicted pages / 1,021,184 tokens, exact batch 925, and MB2
`462+463` with padded graph 926.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_192846_manual_na`.
It reproduced 15,956 pages and exact batch-925 admission, then failed in 3:24
on every attention rank after runtime-ready and before graph capture: only
17.44 MiB remained when another 20.00 MiB allocation was required.  This point
overallocated 15 pages relative to batch 925's true minimum.  Batch 926 needs
at least 15,959 pages and shares padded graph 926, so it is strictly larger than
this failed memory footprint and cannot run under the same contract.  Exact-
minimum boundary job `6923008` was submitted after both preflights at ratio
0.9965625: 15,941 pages / 1,020,224 tokens, exact scheduler batch 925, MB2
`462+463`, and padded graph 926.  Its run directory is
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_193524_manual_na`.
It failed before container creation in 0:22 because the submission mistyped
the image path (`oci_hsg_onboarding` instead of `oci-hsg_onboarding`), so no
workload ran.  Corrected, otherwise identical job `6923092` uses run directory
`results/afd_qwen3_1024_r1_ag4_fg4_atp1_fep4_adp4_b925_n2_20260904_193848_manual_na`.
Corrected job `6923092` completed `0:0` in 19:58 on same-fabric trays
`nvl72093-T[08,14]`.  Every attention rank reproduced 15,941 pages, exact
scheduler admission was batch 925 with 11 tokens unused, and every worker
captured the padded-926 graph; all four model logs confirm FP8xFP4 MegaMoE with
`perfect_routing=1`.  Its real 3,700-prompt prefill finished, then warmup step
1837 and exactly 15 full-resident batch-925 decode steps 1838--1852 completed.
All 3,700 samples contain 17 output tokens, all eight Nsight reports are
nonempty, result/capture artifacts are durable, and `FASTAFD_AFD_SUCCESS` is
present.  This proves 925 is the maximum under the A1:F1/EP4 contract because
batch 926 needs at least 15,959 pages, exceeding the failed 15,956-page
same-padded-graph footprint.  No lower candidate was rerun after this full
success, per the user's decision.

## Workspace/archive update (2026-09-04 PDT)

The uncompressed perfect-routing task root and its local result copy were
removed after verified archival. The authoritative payload is now
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD/artifacts/archives/completed_sweeps/afd_128k_perfect_routing_e00afda_20260901.tar.zst`
with external manifest/SHA/summary sidecars. It has exactly 1,599 payload-only
members (1,512 NSYS, 84 per-case result/metric JSONs, and three audit/comparison
files), full integrity/member audits passed, and archive SHA-256 is
`3dc1842ba90410dc921103450ac45ad5eb1cbe6e2d6b4d13cbc6322350f52089`.
Current rendered figures moved to
`scratch/visualizations/afd_128k_optimized_attention/`; all other local scratch
content was deleted. `run_afd.sh` is the sole canonical launcher; see the root
project memory and OCI-HSG README for merger/test status and the teardown-hang
explanation.

## Perfect-routing 42-case 128K rerun (2026-09-01 PDT, complete)

The authoritative 42-case 128K matrix was rerun with globally perfect
expert balance while preserving the original router cost.  Remote clean branch
`codex/perfect-routing-42-20260901` at
`e00afda8af6e9fc6ad240ec0bd776b960066a53c` descends from repaired A2/A3
head `bd1e3ca`.  Opt-in `MINISGL_AFD_PERFECT_ROUTING=1` retains the normal gate
GEMM, softmax, top-k comparisons, and top-k weight calculation and substitutes
only the final expert ID through a compile-time specialization.  It adds no
kernel launch, CUDA graph node, synchronization, or copy; default routing is
unchanged.  Across each microbatch the global deterministic sequence gives
every expert either floor(N/128) or ceil(N/128) routes, hence exact equality
whenever divisible and the mathematically minimal one-route difference
otherwise.  The source also supports reusing the prior sweep's exact pinned
prompt source/cache before GPU work.

The now-removed remote task root was
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/afd_128k_perfect_routing_e00afda_20260901`.
Its `CASES.csv` is byte-identical to the completed `9a84a5a` sweep; the fresh
pool validated at 42 pending, zero claims/completed/failed, with six cases in
each 2/3/4/8/12/16/18-tray group.  Focused FMHA/MegaMoE contracts pass 55/55
in the remote venv.  The first two-tray submission `6796806` failed in Pyxis
before process startup because it used the obsolete home-directory image path;
the next submission `6796939` failed source preflight because its pinned full
SHA was stale even though its short `e00afda` ID matched.  Neither attempt ran
a case, and both left the pool untouched.  Corrected 2-tray job `6797102` and
3/4/8/12/16/18-tray jobs `6797854`--`6797859` use the successful original
sweep's Lustre image and verified full SHA; each is a short-queue/two-hour job.
All seven final jobs completed `0:0`, including `6797859` for the 18-tray
group; pool state is 42 complete, zero claims/pending/failed. The strict
completed-plan audit passes all hashes, topology and measurement contracts,
42 metric JSONs, and 1,512 Nsight reports. Every paired row has 15 retained
samples and zero outliers. All 288 model logs contain exactly one FP8xFP4
`fmha_megamoe_buffers` marker with `perfect_routing=1`, and all 42 prompt
transforms are exact cache hits on the original pinned source.

Performance contradicts the expected neutral-to-faster outcome: 42-row latency
geometric mean is +5.867545% and TPS/GPU is -5.542345%, with 14 rows faster
and 28 slower. Topology latency/TPS geomeans are +18.570508%/-15.661996% (2t),
+14.104891%/-12.361338% (3t), +4.543148%/-4.345716% (4t),
+2.481296%/-2.421218% (8t), +1.224520%/-1.209707% (12t),
+0.926520%/-0.918015% (16t), and +0.653234%/-0.648995% (18t). The worst row
is A1/EP4 batch 2 at +35.846598%. Global-count balance is therefore not
performance-neutral; investigate the deterministic per-token expert ordering.
The exact comparison is the archived `report/perfect-vs-original.txt`; there is
no longer an uncompressed local result copy.

The verified AFD sweep is included in the seven-series Pareto page as
`FastAFD_Perfect_Balance_W4` (42 points / 13 sweep-local Pareto points). A
focused two-series page,
`scratch/visualizations/afd_128k_optimized_attention/afd-trtllm-perfect-balance-128k-pareto.html`,
compares it with the accepted 31-point TRT-LLM perfect-router sweep under the
display names `AFD` and `TRTLLM`. Exact-data and
Pareto audits cover all 73 points, and the extracted JavaScript passes
`node --check`. Two versions retain identical 73-point data: the `-zero-start`
page uses TPS/User/TPS-GPU domains `[0, 80]`/`[0, 245]`, and the `-cropped`
page uses `[25, 80]`/`[25, 245]`. The unsuffixed page aliases the zero-start
version; all three display the short labels `TRTLLM` and `AFD`.
Both named variants use prominent 24px/weight-500 foreground tick labels,
explicit 9px tick marks, and 1.75px foreground axis/tick strokes; this renders
black in the light theme while remaining legible in dark mode.

## A2 fan-in correctness repair (2026-08-31 PDT)

The A2 failure is localized outside the FP8xFP4 MegaMoE math.  Each A2 attention
rank has one outgoing O edge, so the JIT selects the single-edge publisher, but
the second attention rank's sole edge owns destination source partition 1.  The
BF16, FP8, and fused-quantized FP8 O publishers incorrectly equated one edge
with source index zero; the FP8 paths also substituted source count one when
computing the packed-scale stride.  That overwrote source partition zero and
left partition one invalid.  A1 has only one source per model rank, which is
why the same defect is invisible there.

The minimal repair always reads source index and source count from the existing
descriptor while retaining the one-edge loop specialization.  It is zero-copy
and adds no tensor materialization, barrier, synchronization, fallback, launch,
or numerical operation.  The fix is remote commit `b769743`, descended from
aligned checkpoint `9a84a5a`; completed test coverage is commit `bd1e3ca` on
branch `codex/a2-fp8fp4-fix-20260831`.

Four-GPU pre-fix control job `6741945` exercised two one-edge attention/model
pairs with source indices 0/1 and source count 2.  It failed exactly at the
predicted boundary: source-1 payload mismatched all 24,576 bytes and the
source-count-dependent scale view mismatched 24/48 words.  The unchanged test
passed after the repair in job `6742095` for all nine iterations.  Generalized
job `6742728` then passed 72 exact publications covering every source partition
for logical fan-in A=1 through A=8, validating the entire FP8 payload arena,
packed-scale arena, and readiness state.

The repaired kernel-only numerical check compares exact deployed FP8
activations and requantized FP4 weights at the unchanged cosine threshold 0.96.
Job `6743049` passed all eight combinations of rows 3/6, graph buckets
518/1036, and ordinary/production-prepared routes; observed cosine is
0.999891400--0.999959111.  Each case owns a scoped symmetric buffer, matching
the separate A1/A2 processes and avoiding invalid cross-case route state.
Focused runtime contracts pass 41/41.

Full unchanged A2:F1/EP4 model validation is job `6743115`: 3 trays/12 GPUs,
attention DP 8, EP4, batch 6 per lane, two microbatches, 128K context,
FMHA-only placement, and MegaMoE FP8xFP4.  It completed `0:0` in 18:08,
retaining all 15 measured decode steps and producing 48 coherent samples / 816
tokens plus all twelve Nsight reports.  Strict CUDA is 22.938265067 ms and
throughput is 174.381104603 TPS/active GPU, versus 22.971659867 ms and
174.127599974 TPS/active GPU in pre-fix job `6691133`: respectively 0.145%
lower latency and 0.146% higher throughput, which is noise-level evidence of no
performance regression.  Unchanged official scorer job `6743442` completed
`0:0` in 15:48: all 816 token decisions are top-1 matches, top-10/top-100 are
also 100%, average/max vLLM rank are 1, and representative attention/model
ranks retain exactly 16 graph launches.  After this pass, requested A3:F1/EP4
full-model job `6743740` completed `0:0` in 19:50 with twelve attention lanes
plus four EP4 model GPUs, batch 6 per attention lane, and the same
128K/MB2/FP8xFP4/capture contract.  All four model ranks receive exactly three
mapped sources (`q_edges=3`, max rows 1554), while every attention publisher
retains one O edge, directly exercising source indices 0/1/2.  It retained
15/15 measured steps at 23.188910933-ms strict CUDA and 194.058272635
TPS/active GPU, produced 72 coherent samples / 1,224 tokens and all sixteen
Nsight reports, and its first 48 token arrays exactly equal the perfectly
aligned A2 sample.  Unchanged official A3 scorer job `6744025` completed `0:0`
in 21:00: all 1,224 token decisions are top-1 matches, top-10/top-100 are also
100%, average/max vLLM rank are 1, and representative attention/model ranks
each retain exactly 16 graph launches.  A2 and A3 therefore both pass the
unchanged end-to-end correctness standard, while the generalized kernel test
already covers every source partition for A1 through A8.  A representative
model-trace audit finds no reshape/contiguous kernel names; A2 and A3 have the
same three PyTorch copy-kernel classes at the same 16-launch counts.  Together
with the source diff (descriptor loads/address arithmetic only), this excludes
a fix-specific copy launch or synchronization.  Earlier jobs `6742458` and
`6742681` did not execute the model: the first failed campaign
ordinal validation and the second was canceled pending before kernel-first
validation, respectively.

## A1 control rerun and A2:F1/EP4 alignment failure (2026-08-31 PDT)

The requested A2:F1/EP4 correctness test and A1 control rerun are complete on
preserved clean source `9a84a5a`.  A2 reused the exact successful production
sweep sample from job `6691133`: 48 prompts x 17 generated tokens at 128K,
attention DP 8, four FFN GPUs, EP4, batch 6 per lane, two microbatches,
FMHA-only placement, and MegaMoE FP8xFP4.  Every FFN-rank log independently
records `fmha_megamoe_buffers precision=fp8_fp4 ... group_ranks=4`.  The
unchanged one-tray official scorer job `6740882` completed all 48 HTTP
comparisons but correctly exited `1:0` on strict acceptance after 16:05.
Across 816 generated-token decisions, A2 has 314 top-1, 442 top-10, and 540
top-100 hits: rates 0.384803922/0.541666667/0.661764706, average vLLM rank
10100.873775, maximum rank 151798, and perplexity 631.597994.  Forty-seven
prompts first disagree at generated position zero and one at position one.
All eight attention lanes are affected.  Accuracy also falls sharply by the
six per-lane batch slots: top-1 is 0.625000/0.551471/0.477941/0.514706 for
slots 0--3, then 0.110294/0.029412 for slots 4--5.  This localizes a strong
batch-row effect but does not yet prove the root cause.

To exclude scorer drift, one-tray control job `6741120` rescored the exact A1
sample from job `6690293` with the identical current script, model, image,
environment, and thresholds.  It completed `0:0` in 10:57 and reproduced
perfect alignment across 24 prompts and 408 generated tokens: top-1/top-10/
top-100 all 1.0, average/max rank both 1.0, and perplexity 1.009447258.  The
current scorer is therefore repeatable; A1 remains validated, while the tested
A2 production path is not alignment-correct.  Compact evidence is local under
`scratch/megamoe_alignment_a1_a2_20260831/`; full A1/A2 alignment JSONs remain
remote under `results/fmha_megamoe_cleanup_20260829/a1_rerun_9a84a5a_20260831/`
and `.../a2_alignment_9a84a5a_20260831/`.

The same-sweep A1:F1/EP8/batch-6 sample on `9a84a5a` was then compared directly
with the previously official-scored A1:F1/EP8 sample on tree-identical production
base `0bee9f3`.  All 48 prompts, all 48 generated-token arrays, and all 48 decoded
texts match exactly.  The earlier unchanged scorer reported 816/816 top-1 tokens,
so current A1/EP8 is also alignment-correct without spending another scorer run.
Together with current A1/EP4 correctness, this excludes EP4 or EP8 size alone:
the failing structural delta is A2 fan-in (two attention sources and six rows per
microbatch per FFN rank) or an interaction between that fan-in and EP4.

## Active causal retest of attention-only fused O publication (2026-08-28 PDT)

Existing exact control/candidate reports were compared without further GPU
work using general CPU-short extractor commit `e65220b`; job `6653623`
completed `0:0` in 33 seconds.  The trace disproves the FFN-fusion premise for
current production: MegaMoE input quantization was already fused into the
router, so fused norm+per32 quant removed no model graph node.  Norm plus fused
router grew 7.184 -> 10.023 us/round, and moving FP8 buffer production earlier
made MegaMoE median grow 43.680 -> 56.480 us.  The model graph stayed at 2,079
nodes.  Restore the entire accepted FFN path.

Attention evidence has the opposite sign.  Accepted per-token cast plus O
publication costs 1.633 + 7.892 = 9.524 us/round, while direct fused
quantize/publish costs 7.498 us, a 2.026-us local saving.  It removes exactly
188 graph nodes (one per 94 layers x MB2), from 752 to 564.  This selects a
general attention-only candidate rather than another combined experiment.
Isolated remote branch `codex/fmha-attention-fused-fp8-20260828` is clean at
`0032e1c`; its tree differs from adopted FP8xFP4-only mainstream `7f5c273` in
only seven attention transport/runtime/test files, and 21/21 focused contracts
pass.  One-tray/four-GPU short transport smoke `6654019` completed `0:0` in
3:16.  Its repeated fused transport check passed two pairs, three rows, and
nine turn/counter generations.

The first exact submission, `6654187`, used a nonexistent image path and
failed before container start in 23 seconds; it never claimed the case.  The
corrected four-tray short job `6654241` used the established Lustre image and
completed `0:0` in 20:18.  The terminal pool has one completed case and zero
pending/claimed/failed; there are 16 Nsight reports and all eight FFN logs
prove `precision=fp8_fp4`.  The unchanged strict metric retains all 15 samples
with zero outliers: 23.215000067-ms mean, 23.208192-ms median, 23.323425-ms
maximum, 0.585035% dominant range, and 129.226793 TPS/GPU.

Against exact clean control `6624199`, strict mean/median/maximum regress by
0.496712333/0.496799/0.497696 ms (2.186399%/2.187444%/2.180417%), and
throughput falls 2.825413 TPS/GPU (2.139618%).  Attention graph nodes fall
752 -> 564, but mean attention span grows 22.718288 -> 23.215000 ms and model
span grows 22.652951 -> 23.150948 ms.  The local 2.026-us cast/publication
the first cross-allocation result therefore appears adverse.  Do not promote
`0032e1c` from that evidence; adopted mainstream remains clean FP8xFP4-only
`7f5c273`.  Compact exact
artifacts are local under
`scratch/fmha_attention_fused_fp8_20260828/exact_a1_f1_ep8_128k_b6`, and the
kernel comparison remains under
`scratch/fmha_fused_fp8_casts_20260828/trace_comparison/`.

Follow-up CPU-short job `6654815` extracted the exact attention-only reports
alongside the exact control.  The fused attention sequence is locally faster:
FMHA median falls 113.024 -> 112.096 us and cast+publish falls
1.664+7.872 -> 7.584 us, about 2.9 us/layer total.  The headline loss instead
coincides with byte-identical production MegaMoE changing 43.680 -> 60.832 us
median across allocations; other unchanged model kernels move in both
directions.  This makes the historical cross-node comparison non-causal.
General same-allocation harness commit `abb04c3` runs two isolated one-case
roots sequentially without touching either source.  Four-tray short job
`6654903` ran clean control `7f5c273` first and attention-only `0032e1c` second
on the same `nvl72157-T[02-03,05,13]` allocation and completed `0:0` in 35:19.
Both exact metrics retain 15/15 samples with zero outliers.  Control is
22.745179600-ms mean, 22.736992-ms median, and 131.896079 TPS/GPU; candidate is
23.243643400-ms mean, 23.229607-ms median, and 129.067545 TPS/GPU.  Thus the
same-allocation candidate is 0.498463800 ms / about 2.19% slower end to end.

CPU-short job `6655853` extracted exact paired kernel summaries.  They confirm
the user's expected role-local behavior: control attention cast+publish is
1.600+7.712 = 9.312 us/layer median while candidate fused quantize+publish is
8.928 us, a 0.384-us saving; averages are 9.295 -> 8.960 us.  FMHA itself is
effectively identical at 113.120/113.152 us median.  Unchanged model kernels
including O projection, QKV projection, publish, add-RMSNorm, fused router,
rope, and model-side per-token cast are identical or within 0.208 us median.
The end-to-end loss instead coincides with attention wait 115.072 -> 119.520
us, model wait 140.192 -> 143.280 us, and byte-identical FP8xFP4 MegaMoE
35.424 -> 37.216 us median.  Since all of those shifts disadvantage the second
run, the A/B order is not yet a causal verdict.  Reverse-order four-tray short
job `6655984` was submitted on fresh roots with candidate first and control
second, but failed before either case was claimed because Slurm allocated its
four trays across two NVL72 fabric blocks.  The fabric preflight rejected the
allocation in 21 seconds and no benchmark ran.  `6656164` added the required
`--segment=4` and received one base system, but the pair wrapper then failed in
21 seconds before pool initialization because its required empty pool
directories did not yet exist.  No case was claimed.  Corrected `short` retry
`6656285` completed `0:0` on contiguous `nvl72082-T[07-10]` in 34:51.  With
candidate first and control second, both retain 15/15 samples and zero
outliers: candidate/control strict means are 23.224607267/22.744459733 ms,
medians are 23.213887/22.737922 ms, and TPS/GPU is 129.173336/131.900253.
The candidate therefore remains 0.480147533 ms / about 2.11% slower in reverse
order; run order is ruled out.

Local Git inspection identified two coupled A-to-F cadence changes, not a
mathematical cost in FMHA.  The accepted path finishes local
`per_token_cast_fp8`, launches the publisher, and releases `attention_turn` at
publisher entry before issuing remote stores.  Candidate `0032e1c` instead
released only after its fused quantize/publish payload loop, delaying the peer
lane handoff until after fabric instructions were issued.  It also replaced
the accepted publisher's 16-byte vector stores with four-byte stores emitted
directly by quantizing lanes, which can reduce A-to-F communication efficiency
despite a shorter fused kernel duration.  This explains why local
cast+publication improved while per-layer attention/model waits grew.

Clean corrective worktree
`codex/fmha-attention-fused-staged-publish-20260828` stages FP8 O and unpacked
UE8M0 scales in persistent slot-local buffers, releases attention ownership
after every CTA stops reading BF16 O, then performs vectorized 16-byte remote
publication in the same kernel.  Commits `3d623a6` and `f229ad9` preserve one
fused launch, use no cooperative grid barrier, change no alignment or tuning
parameter, and leave all FFN/MegaMoE production files unchanged.  The 21/21
source contracts pass.  Four-GPU `short` transport smoke `6657389` completed
`0:0` in 3:14: fresh JIT prebuild passed and two fabric pairs passed three-row,
nine-generation byte-exact FP8/scales and turn/counter reuse.  Exact
candidate-first/control-second same-allocation job `6657525` completed `0:0`
in 36:56 on `nvl72004-T[02-03,15-16]`, with one four-tray segment and FP8xFP4
explicit.  Candidate/control strict means are 22.966924800/22.756537933 ms,
medians are 22.960352/22.755520 ms, and TPS/GPU is
130.622625/131.830246.  Both retain 15/15 samples with zero outliers.  The
corrective design recovers 0.257682467 ms versus direct fusion's reverse-order
23.224607267 ms, supporting the A-to-F handoff/store diagnosis, but remains
0.210386867 ms / 0.924512% slower than its same-allocation control; TPS/GPU is
0.916043% lower.  CPU-short paired trace job `6658813` completed `0:0` in 2:40
from four explicitly named reports.  It localizes the residual: control local
cast+publish is 1.664+6.432 = 8.096 us/layer median (8.197 us average), while
staged fusion is 9.344 us median (9.455 us average), a 1.248/1.258-us penalty.
Across 188 layer-rounds this predicts about 0.235/0.237 ms, matching the
0.210-ms headline gap.  FMHA is unchanged at 112.256/112.288 us; attention
and model waits each differ by only 0.768 us median.  Byte-identical FP8xFP4
MegaMoE moved 52.160 -> 53.584 us while router and model cast remain
effectively identical, consistent with the longer overlapping A-to-F tail.

The remaining source-level inefficiency is publication lane occupancy:
`f229ad9` used eight lanes for one 128-wide head and iterated four times per
warp, whereas accepted publication keeps all 32 lanes issuing 16-byte stores.
Commit `1276ce6` groups four tasks already produced by the same warp, so all
32 lanes publish concurrently without reading another CTA's staging or adding
a grid barrier.  It changes no parameter, alignment, or FFN path; 21/21
contracts pass.  Fresh four-GPU short smoke `6659029` completed `0:0` in 2:33
with JIT prebuild plus the nine-generation byte-exact fabric check.  Exact
candidate-first/control-second short job `6659188` completed `0:0` in 35:54
from the clean commit on one four-tray segment; fixed progress files advanced
through both cases and the run did not hang.  Candidate/control strict means
are 22.947676357/22.699267867 ms, medians are 22.939089/22.698023 ms, and
TPS/GPU is 130.732191/132.162853.  Candidate/control retain 14/15 samples with
1/0 allowed outliers.  Full-warp publication is still 0.248408490 ms /
1.094346% slower and improves only about 0.019 ms over the prior staged
candidate, so the publication loop was not the main residual.  CPU-short job
`6660922` completed `0:0` in 2:52 from the four explicitly named reports.  The
full-warp fused kernel is 8.320 us/layer median versus control cast+publish
1.664+6.464=8.128 us, a residual local penalty of only 0.192 us/layer or about
0.036 ms over 188 rounds.  FMHA is unchanged at 112.192/112.128 us.  The
headline regression instead coincides with attention wait 115.936 -> 118.336
us and byte-identical FP8xFP4 MegaMoE 51.968 -> 54.944 us; dense GEMMs, both
router kernels, QKV publish, norms, RoPE, and model FP8 cast are effectively
identical.  Candidate/control representative attention spans differ by
0.248408 ms and model spans by about 0.247225 ms, confirming propagation of
an attention-side A-to-F cadence/communication cost rather than an FFN code
change.

Clean remote commit `dbb5655` attacks the remaining pre-handoff serialization:
each warp now quantizes four heads concurrently in four eight-lane subgroups,
using two 16-byte BF16 loads and one 16-byte staged FP8 store per lane.  Its
task ownership exactly matches full-warp publication and changes no grid,
alignment, payload, scale definition, remote store, or FFN file.  The isolated
remote two-file diff passes 21/21 contracts.  Four-GPU short smoke `6661262`
completed `0:0` in 3:13 with fresh JIT build and the repeated two-pair,
three-row, nine-generation byte-exact transport check.  Exact candidate-first/
control-second short job `6661613` is submitted on one four-tray segment with
explicit `MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4`, FP4-only case rows, and a
30-minute per-case timeout.  Narrow source comparison found another remaining
communication mismatch independent of the subgroup experiment: accepted
publication assigns each warp four adjacent heads and copies their four packed
UE8M0 scale bytes with one 32-bit remote store.  Current fusion fills the warp
with four heads separated by `warp_stride` and issues four byte scale stores.
If `6661613` remains slower, the next causal iteration should make quantize and
publish own adjacent four-head groups and restore the packed scale store while
leaving grid, alignment, FP8 math, and FFN untouched.

Exact pair metrics from `6661613` are durable.  Candidate/control strict means
are 22.690983467/22.735555867 ms, medians are 22.685218/22.728607 ms, and
TPS/GPU is 132.211105/131.951909.  Both retain 15/15 samples with zero outliers
and similar 0.571/0.588% dominant ranges.  The subgroup candidate improves
latency by 0.044572400 ms / 0.196047% and TPS/GPU by 0.196432%.  Representative
attention spans improve 0.044572 ms and model spans improve about 0.044592 ms,
consistent with an attention-side cadence saving propagating through the
pipeline.  CPU-short four-report job `6663587` completed `0:0` in 32 seconds.
Subgroup fusion reduces the attention publication tail from control
cast+publish 1.664+6.432=8.096 us/layer median to 6.816 us, a 1.280-us / 15.81%
improvement, while FMHA remains 112.160 us in both reports.  Representative
FP8xFP4 MegaMoE moves 52.864 -> 54.208 us even though dense GEMMs, router,
QKV publish, norm, RoPE, and model-cast timings are effectively unchanged.
The A1 headline result is positive, but the strict no-worse-FFN trace gate is
therefore not met; the remaining effect is consistent with attention-to-FFN
handoff cadence or communication rather than changed FFN production code.

The user requested the stronger A4:F1/EP8/ATP1/128K/batch-6 gate against the
previous accepted sweep.  Fresh task root
`fmha_attention_fused_staged_publish_20260828/`
`a4_f1_ep8_b6_quant_subgroup_20260828/` contains a single FP4-qualified plan.
Dry-run topology is 32 attention plus eight exclusive FFN GPUs on ten
contiguous trays, MB2 3+3, 192 real prompts, memory ratio 0.82, and exact
839,424-token capacity gating.  Short job `6663641` was submitted from clean
`dbb5655` with a one-hour case watchdog.  Its accepted comparison is sweep
jobs `6530508`/`6531018`: 24.951913467-ms strict mean, 192.370016 TPS/GPU,
15/15 samples, and zero outliers.  CPU-only job `6663725` re-extracted the
baseline representative reports and confirmed FP8xFP4 MegaMoE; baseline
attention cast+publish is 1.664+7.552=9.216 us median and MegaMoE is 70.080 us
median.  At the first five-minute sample `6663641` remained pending for short
priority, holding no GPUs and showing no hang condition.  Acceptance requires
the candidate strict mean to beat 24.951913467 ms, its fused attention tail to
beat 9.216 us, and its FP8xFP4 MegaMoE median not to exceed 70.080 us.

Job `6663641` completed `0:0` in 22:30 on contiguous
`nvl72009-T[01-10]`.  Five-minute fixed scheduler/progress/log checks showed all
ten tray progress files advancing through initialization and 128K prefill; no
hang occurred.  Its strict metric retains 15/15 measured launches, zero
outliers, a 2.233660% dominant range, and reports 22.961559467-ms mean,
22.929184-ms median, 23.430752-ms maximum, and 209.045035 TPS/GPU.  Attention
and model-role means are 22.961559467 and 22.871352867 ms.  Relative to
`6530508`/`6531018`, strict latency improves 1.990354000 ms / 7.976759% and
TPS/GPU improves 8.668201%.  The durable result explicitly records `megamoe`,
FP4 expert weights, FMHA-only placement, and all 40 GPUs active.

Candidate role extraction job `6665325` completed `0:0` in 30 seconds.  The
subgroup fused attention publisher is 6.752 us median versus the baseline
1.664+7.552=9.216-us cast+publish path, improving 2.464 us / 26.736%.
FP8xFP4 MegaMoE is 48.064 us median versus baseline 70.080 us, improving
22.016 us / 31.416%.  Thus the requested comparison against the prior A4 sweep
passes at the FP8xFP4 kernel-family level.  It is not a causal no-worse-FFN
proof, however: baseline source `24b90c` predates the accepted MegaMoE
route-readiness, rank-epoch/count caching, and scheduling work in the candidate.
The traced kernel signatures differ, and the old-to-current tree changes FFN
production sources.  The exact-current-code A1 control/candidate pair has no
FFN source diff and instead moves MegaMoE 52.864 -> 54.208 us, consistent with
a small attention-to-FFN cadence/communication regression.  Compact metrics,
Slurm logs, manifests, and baseline/candidate kernel summaries
are local under
`scratch/fmha_attention_fused_staged_publish_20260828/`
`a4_f1_ep8_b6_quant_subgroup_20260828/`; raw reports remain remote.

Promotion first attempted a fast-forward and correctly stopped because the
mainstream and candidate branches carry equivalent FP8xFP4 prerequisite patches
under different commit IDs.  A read-only merge-tree then proved the clean
three-way result exactly equals validated candidate tree
`1289829288c325e92554962d021232e6c933b17b`.  Clean merge commit `0bee9f3` now
heads remote adopted branch `codex/fmha-megamoe-fp4-mainstream-20260827` and
contains validated attention commit `dbb5655`.  The final production diff is
limited to attention runtime/transport plus validation assets; no FFN/MegaMoE
production file changes.  Existing exact A1:F1/EP8 sample source `dbb5655`
contains 48 prompts, 16 Nsight reports, and explicit FP8xFP4; its tree is
identical to adopted merge `0bee9f3`, so rerunning AFD was unnecessary.
One-tray short job `6665732` completed the unchanged official scorer in 16:05
with exit `0:0`.  All 48 prompts pass: overall top-1, top-10, and top-100 are
1.0, with average and maximum vLLM rank both 1.0.  Representative attention
rank 1 has 16 graph launches/9,024 graph kernels and model rank 9 has 16 graph
launches/33,264 graph kernels; both contain the expected AFD NVTX ranges.
Compact summaries and Slurm logs are local under
`scratch/fmha_attention_fused_staged_publish_20260828/`
`final_a1_ep8_alignment_0bee9f3_20260828/`; full `alignment.json` remains under
the corresponding remote task root.  This is the final alignment proof for
retained tree `0bee9f3`; the scored candidate tree is byte-identical.

The next attention-only causal iteration is remote commit `b3338fe` on branch
`codex/fmha-attention-packed-scale-20260828`.  It removes the remaining
publication-shape mismatch: each warp now owns four adjacent heads in both
fused O quantization and staged publication, enabling one aligned 32-bit remote
UE8M0 store instead of four byte stores for a complete descriptor group.
Same-warp staging ownership, the early attention-turn release, FP8 math,
alignment, launch geometry, and all FFN production code remain unchanged;
partial or unaligned groups retain exact byte ownership.  Remote source
contracts pass 21/21 and `git diff --check` is clean.  Validate fabric
correctness first, then use a same-allocation A1 control/candidate run to test
whether the reduced transaction count keeps attention faster without the
observed MegaMoE regression.
Four-GPU short-queue job `6666759` runs the reusable fabric/FP8xFP4 smoke from
commit `b3338fe` with a fresh transport JIT cache and a 30-minute cap.  Output
root: `fmha_attention_packed_scale_20260828/smoke_b3338fe/`.
It completed `0:0` in 3:26: transport prebuild passed and two fabric pairs
completed nine repeated turn/counter replays with bit-exact FP8 payloads and
UE8M0 scales.  Four-tray same-allocation A1 pair job `6667041` completed `0:0`
in 36:48 on `short`.  Candidate `b3338fe` and exact control `7f5c273` each
retain 15/15 samples with zero outliers.  Strict CUDA mean is
22.693617533/22.724339200 ms: 0.030721667 ms / 0.135193% lower latency and
0.135376% higher TPS/GPU.

CPU-only Nsight extraction job `6668636` completed `0:0` in 33 seconds.  The
packed-scale fused attention tail is 6.784 us versus exact control
1.664+6.496=8.160 us, improving 1.376 us / 16.863%; FMHA remains essentially
identical at 112.320/112.288 us.  The apparent raw MegaMoE regression
(54.272/52.496 us) is handoff accounting rather than slower FP8xFP4 compute:
model `wait_ready` improves 124.160 -> 122.240 us, so MegaMoE starts 1.920 us
earlier and absorbs 1.776 us of A-to-F readiness wait.  The combined
communication-bearing path is 176.512 versus 176.656 us, 0.144 us faster, and
the MegaMoE kernel signatures are identical.

The final retained version remains clean adopted commit `0bee9f3`; reject
experimental `b3338fe` because its exact-pair E2E gain is smaller than
`0bee9f3` (0.135193% versus 0.196047%) and its combined model-path gain is also
smaller (0.144 versus 0.448 us).  `0bee9f3` still improves the fused attention
tail by 15.81%, changes no FFN/MegaMoE production file, and has the completed
official alignment proof from job `6665732`.  The local transport source was
restored to the adopted SHA-256.  The final goal is achieved under the correct
subsystem accounting: improved attention and E2E, unchanged FP8xFP4 FFN math,
no worse `wait_ready + MegaMoE`, and unchanged alignment.  Task root:
`fmha_attention_packed_scale_20260828/`
`paired_exact_packed_scale_control_20260828/`; compact trace summaries are
local under the matching `scratch/.../paired_trace_analysis/` directory.

## Rejected fused FP8 cast iteration (2026-08-28 PDT)

The active target is the exact A1:F1/EP8/ATP1/128K/batch-6 FMHA-only row, and
the user requires every same-rank MegaMoE execution to use FP8 activations and
FP4 weights.  The implementation therefore fails closed if MegaMoE does not
receive its lane-local FP8 input, and validation must show
`precision=fp8_fp4`; there is no alternate MegaMoE precision fallback.  Clean
matrix job `6624199`, case `i131072-fep8-r1-atp1-b6-fp4`, is the unchanged
control at 22.718287733-ms strict CUDA mean, 22.711393-ms median, 15/15 samples,
zero outliers, and 132.052205 TPS/GPU.

Isolated remote branch `codex/fmha-fused-fp8-casts-20260828`, worktree
`worktrees/fmha_fused_fp8_casts_20260828`, contains the fused candidate at
`f43a2fe`.  The candidate
fuses attention BF16-to-FP8 per-head casting into the existing O remote-publish
kernel and separately tracks local quantization completion so attention-turn
ownership is not released before quantization completes.  It also generalizes
the existing fused add-RMSNorm kernel to produce MegaMoE's per-32 FP8 payload
and scale factors while retaining the BF16 normalized tensor for routing.
MegaMoE consumes those exact precomputed buffers instead of launching its
standalone per-token cast.  These are general launch-elimination changes; head
dimension, four-head packing, alignment, memory, routing, and launch-policy
parameters are unchanged.

Focused local contracts pass 21/21 and both fused operations have GPU bitwise
checks against their prior component kernels.  CPU DeepGEMM prebuild job
`6651632` completed `0:0` in 2:39.  The user then made FP8xFP4 execution an
unconditional MegaMoE invariant: remote commit `4868f08` defaults to FP4,
rejects any FP8-weight MegaMoE request in both adapter and launcher, and limits
new online plans to `fp4`.  The exact one-row plan was corrected to rerun index
one and initializes as one pending four-tray case.

Initial short-QoS smoke `6651845` failed in 2:21 before either fused check when
four ranks raced to create the default home-directory DeepEP cache.  This is a
harness failure and no CUDA result was produced.  Commit `11efeb0` moves that
extension into a precreated task-local Lustre cache and prebuilds it on one
rank.  Retry `6652065` then passed the repeated fused transport gate for two
pairs, three rows, and nine turn/counter generations.  It failed before the
FFN fused check because FlashInfer independently selected a missing home cache;
again no numerical mismatch occurred.  Commit `dc4a044` gives FlashInfer a
task-local cache and makes the standalone MegaMoE smoke itself accept only FP4
weights.  One-tray/four-GPU short job `6652285` completed `0:0` in 3:52.  The
transport check passed two pairs, three rows, and nine generations; fused
add-RMSNorm matched hidden/residual/FP8/scale bitwise; the production smoke
reported `expert_weight_dtype=fp4`, numeric cosine 0.979040, and `status=ok`.
This released exact four-tray short job `6652521` from clean `dc4a044` with one
NVL72 segment, two-hour limit, explicit MegaMoE/FP4 environment, and unchanged
one-warmup-plus-15 performance contract.  It completed `0:0` in 18:18.  The
terminal audit has one completed case, zero pending/failed/claimed, 16 Nsight
reports, eight FFN logs proving `precision=fp8_fp4`, no runtime-error matches,
and clean source.  One attention sample at decode step 1545 was a 1,479.5984-ms
external outlier and was excluded by the unchanged policy; the 14 retained
samples are exceptionally tight at 23.237812571-ms mean, 23.239632-ms median,
23.250816-ms maximum, 0.137558% dominant range, and 129.099931 TPS/GPU.

Against exact clean control `6624199`, strict mean/median regress by
0.519524838/0.528239 ms (2.286813%/2.325877%) and throughput falls 2.952274
TPS/GPU (2.235687%).  Attention graph nodes fall 752 -> 564, but retained
attention span regresses 22.718288 -> 23.237813 ms and model span regresses
22.652951 -> 23.176690 ms; eliminating the attention cast launch therefore did
not improve the governing cadence.  Reject the combined fused-cast candidate
and do not promote `dc4a044`.  The user's independent invariant remains: any
future MegaMoE execution must be FP8xFP4 and fail closed otherwise.  Precision-
only commit `4868f08` was therefore cherry-picked onto clean adopted mainstream
`f14d800` as `7f5c273`; it carries no fused-cast code and passes 21/21 focused
contracts plus launcher syntax and diff checks.  Compact terminal artifacts
are local under
`scratch/fmha_fused_fp8_casts_20260828/exact_a1_f1_ep8_128k_b6`; raw reports
remain on Lustre.

## Completed optimized-attention 128K near-Pareto sweep (2026-08-28 PDT)

The requested matrix is
`scratch/afd_128k_optimized_attention_near_pareto_20260827/afd_ep8_ep4_42_cases.tsv`:
42 FP8-activation/FP4-weight FMHA-only cases at 128K, ATP1, batches 2--7,
covering EP8 ratios A1/A3/A5/A7/A8 and EP4 ratios A1/A2.  The executable
`CASES.csv` has six cases in each of the seven allocation groups
2/3/4/8/12/16/18 trays and validates with the reusable online case-pool tool.
Batch 2--5 rows use the established 0.82 memory ratio and 839,424-token KV
capacity without claiming a capacity maximum; batch 6 requires that exact
maximum.  Batch 7 uses the already supported explicit 0.90 ratio, 14,344
pages, and 918,016-token exact capacity rather than incorrectly attempting it
with the batch-6 ceiling.

Remote task root is
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/afd_128k_optimized_attention_near_pareto_20260827`.
The clean source is the OCI mainstream worktree at commit
`f14d8005ae312ae808a41e6514aba975d78ece98`; plan SHA-256 is
`a2b1e1ec81db138c88a05240b9c7216a74982e9279b26b286a754523dd3cd782`.
All jobs used `batch`/`short`, a two-hour allocation limit, a one-hour per-case
watchdog, one job per tray count, two microbatches where batch permits, and the
strict warmup-plus-15 metric.  Lead jobs were 18 trays `6624181`, 16 trays
`6624184`, 12 trays `6624183`, and 8 trays `6624182`.  Dependency-gated jobs
were 4 trays `6624199` after `6624181`, 3 trays `6624197` after `6624184`, and
2 trays `6624198` after `6624183`, permitting at most four concurrently
runnable GPU jobs.  All seven completed `0:0`: 18/16/12/8/4/3/2-tray runtimes
were respectively `01:44:40`, `01:29:31`, `01:26:30`, `01:19:12`, `01:05:32`,
`01:06:20`, and `01:04:10`.

Final pool state is 42 completed, zero claimed, zero pending, and zero failed.
The strict completion audit passed every metric/result hash, tray-to-job
mapping, topology, capacity gate, FMHA-only MegaMoE FP4 contract, prompt and
microbatch contract, warmup-plus-15 sequence, `SUCCESS` marker, and report
count.  All 42 cases retained 15/15 samples with zero outliers.  Across the
matrix, mean CUDA wall time spans 16.327949--26.922733 ms and throughput spans
59.520477--231.114065 TPS per active GPU; the maximum dominant-range spread is
4.571679% and maximum median difference is 4.423164%, both below the unchanged
10% limits.  The 1,512 expected Nsight reports are present remotely.  Compact
artifacts are checksum-identical locally and remotely under the task root:
`report/completion-audit.json`, `report/case-metrics.tsv`, 42 strict metrics,
42 `afd-result.json` files, terminal pool state, logs, and submission metadata.
Most raw captures remain only on Lustre.  On 2026-08-28 PDT the user-selected
extrema were copied locally with complete SHA-256 parity: highest TPS/GPU
`i131072-fep8-r8-atp1-b7-fp4` at 231.114065 TPS/GPU and 37.143332 User TPS has
72 reports / 72,439,281 bytes; highest User TPS
`i131072-fep8-r1-atp1-b2-fp4` at 61.244680 User TPS and 61.244680 TPS/GPU has
16 reports / 19,354,395 bytes.  Both live under their respective
`results/<case>/experiment/ray_logs/` directories in the local task root.

## Active FFN latency iteration (2026-08-24 PDT)

The current scaling goal is A15:F1, FEP4, batch 6, 128K ISL, ATP1, MB2, using
A7:F1 first as the performance-observation gate.  The alignment standard and
exact warmup-plus-15 strict metric contract remain unchanged.  Retain general
kernel/runtime changes with no Qwen-shape or batch-specific branches.  The
aligned A1:F1 acceptance checkpoint is remote commit `bc9ab7f`, job `6502925`,
at 25.0087105 ms coordinator E2E median and 24.9813376 ms strict CUDA mean.
The scaling expectation is approximately 26 ms because each additional
attention source adds only tens of memory-bound FFN tokens per model round.
On 2026-08-25 the user explicitly closed algorithm, topology, and precision
changes.  Treat all FP4 and split-MegaMoE results as diagnostics only; the
best compliant A7 case remains unchanged-FP8 job `6504134` at 32.6938360 ms.

Local Git/GitHub handoff on 2026-08-25: the coherent FMHA implementation was
committed locally as `393bc98` (`Implement FMHA-only AFD execution mode`) and
pushed only to personal branch `linsj20/fmha_only`; `linsj20/main` was left
unchanged.  The development checkout is execution-only and is not maintained
for GitHub synchronization.  A content comparison against accepted A7 commit
`bc9ab7f` proves that `393bc98` already contains the identical executable
source, scripts, tests, and assets.  The only extra path in the accepted tree
is an older handoff document, so preserve this newer project memory rather than
importing the stale remote-Git workflow text.

Exact A1 validation of `393bc98` used isolated execution commit `732eaa5`
whose tree object exactly matched local `393bc98`, with clean-source
enforcement.  Short-queue job `6516488` used a one-hour limit, two trays / eight
active GPUs, A1:F1, FEP4, batch 6, 128K ISL, ATP1, MB2, FMHA-only placement,
and unchanged FP8 precision and measurement rules.  It completed `0:0` in
22:13 after normal synchronized weight load/JIT and sustained all-rank prefill;
there was no hang or inefficient low-utilization phase.  The result is
25.1577345 ms coordinator E2E median and 25.1350228 ms strict CUDA mean, with
15/15 samples, zero outliers, a 1.6814-percent dominant range, and 119.355372
strict TPS/active-GPU.  Versus accepted A1 job `6502925`, this is 0.1490240 ms
(0.596 percent) slower by E2E median and 0.1536852 ms (0.615 percent) slower by
strict CUDA mean.  Compact local evidence is under
`scratch/qwen3_ffn_overheads_20260820/fmha_only_a1_job_6516488/`; the remote
task root is `qwen3_ffn_overheads_20260820/fmha_only_a1_20260825/`.

Exact A7:F1 observation job `6504134` was submitted on 2026-08-24 to `short`
with a one-hour limit, eight trays / 32 GPUs constrained to one eight-tray
NVL72 segment, FEP4, batch 6, 128K ISL, ATP1, MB2, FMHA-only placement, clean
commit `bc9ab7f`, and the unchanged strict metric contract.  Remote task root:
`ax_alignment_20260824/a7_fep4_b6_128k/`.  Monitor at five minutes during
startup and near completion or fifteen minutes during long prefill/measurement
phases.  It completed `0:0` in 23:57 after normal prompt generation, JIT, and
sustained-utilization prefill; every rank's GPU-progress record advanced and no
hang or inefficient low-utilization phase was observed.  The exact result is
32.6938360 ms strict CUDA mean and 160.580728 strict TPS/active-GPU, with all
15 target steps retained, zero outliers, a 3.2824-percent dominant range, and
one 752-node attention graph plus one 3,395-node grouped-model graph per step.
This clean result misses the approximately 26-ms scaling expectation.

CPU-only `cpu-short` export jobs `6504814`--`6504817` completed in 30--35
seconds and produced exact accepted-A1/current-A7 attention and model SQLite
comparators under the same task root.  The reusable graph comparison shows
that A7 attention compute is nearly flat: summed FMHA grows only 0.125 ms per
replay, while its readiness waits expose the longer model wavefront.  On the
model rank, expert gate/up grows 4.119734 -> 7.357282 ms/replay and expert down
grows 2.428783 -> 4.065531 ms/replay.  `combine_impl` adds 1.901784 ms and QKV
publication adds 0.869214 ms; model graph span grows 7.664016 ms.  There is no
unexplained graph idle gap or extra graph launch.  This attributes the miss to
real grouped-MoE expert-weight/communication work at 21 rows per microbatch,
not a prefill hang or per-source graph serialization.

Information-dense four-GPU smoke sweep `6505134` completed `0:0` in 8:46 on
`short` using clean `bc9ab7f`.  In one allocation it compared
rows/DeepGEMM-SMs/DeepEP-SMs `3/128/24`, `21/128/24`, `21/144/8`, and
`42/128/24`; every case passed BF16, FP8, and fused-router numerical checks.
Fused-router staged times were respectively 0.351262, 0.355903, 0.350601, and
0.338912 ms.  Moving row 21 to the existing high-fan-in 144/8 split is a small
1.49-percent isolated win.  Combining both model microbatches is only 4.78
percent faster than one row-21 stage and would forfeit the current overlap of
the first model stage with the second attention stage, so it is not a credible
route to removing the observed 7.7-ms replay gap.

Fresh branch `codex/qwen3-a7-high-fanin-20260825`, worktree
`worktrees/a7_high_fanin`, commit `b45d263` moves the existing general
high-fan-in policy threshold from eight sources to seven.  A1 remains on the
aligned 24-DeepEP/128-DeepGEMM split; A7 and above use 8/144.  The change is
three constant/comment/contract-test lines and has no model-, batch-, or shape-
specific branch.  CPU `cpu-short` contract job `6505350` was the source gate.
It completed `0:0` in 1:47 with all 62 tests passing.  Exact unchanged-contract
A7 job `6505450` is submitted to `short` for at most one hour on eight trays in
one eight-tray NVL72 segment; monitor at five minutes through startup/prefill
and fifteen minutes only for a demonstrably long steady phase.
It completed `0:0` in 26:03 after normal sustained-utilization prefill, but is
rejected: 33.5253775 ms strict CUDA mean and 156.597789 strict TPS/active-GPU,
versus 32.6938360 ms and 160.580728 for accepted-threshold control `6504134`.
The regression is 0.8315415 ms / 2.543 percent.  All 15 samples were retained
with zero outliers and a 3.2794-percent dominant range, and the attention/model
graphs remained 752/3,395 nodes, so this is a clean same-contract rejection.
CPU-only rank-29 export job `6506007` is the remaining attribution step.  Do
not merge `b45d263`; retain the threshold of eight sources in `bc9ab7f`.  A15
remains gated off because neither exact A7 result approaches 26 ms.
Export `6506007` completed `0:0` in 2:40.  Versus control rank 29, candidate
model span grows 0.834667 ms while readiness-wait sum is flat (-0.071037 ms).
Expert gate/up grows 7.357282 -> 7.785798 ms/replay and expert down grows
4.065531 -> 4.279097 ms/replay; dense O/QKV improve by about 0.127 ms in
aggregate, but memory-bound expert GEMMs regress 0.642082 ms and dispatch adds
0.105639 ms.  This causally confirms that 24/128 is the correct A7 split even
though the isolated four-rank smoke slightly favored 8/144.

The opposite general split was tested without spending an exact eight-tray
run.  Isolated branch `codex/qwen3-a7-deepep32-20260825`, worktree
`worktrees/a7_deepep32`, commit `07611bb` uses 32 DeepEP SMs and 120 DeepGEMM
SMs for the existing low-fan-in policy; its five-file diff is the preserved
general four-cluster implementation from `cfa8660`.  CPU job `6506147` failed
before tests because the wrapper used its obsolete home-directory image
default.  Corrected `cpu-short` job `6506220` pinned the same canonical Lustre
image/venv as prior gates and passed 62/62 in 1:51.  Focused four-GPU `short`
smoke `6506379` compared row-21 120/32 and 128/24 sequentially in one
allocation with a shared fresh build cache; it completed `0:0` in 5:50 with
identical BF16, FP8, and fused-router correctness.  Fused-router staged time
was 0.326447 ms for 120/32 versus 0.325565 ms for accepted 128/24, a 0.27%
regression.  Together with the preserved exact-A1 neutrality of this split,
the smoke rejects `07611bb` without an eight-tray A7 run.  Keep accepted
24/128 and do not merge this candidate.

The next expert-bandwidth audit rejected reverse lane/expert scheduling before
implementation: each lane executes gate/up then down, so hundreds of MiB of
other matrices evict the matching expert weights before another lane could
reuse them.  The legacy split-worker MegaMoE backend has dual-lane L2 reuse,
but it is not wired into FMHA-only placement and its historical optimized A15
result was still 30.752 ms.  A focused packed-weight benchmark therefore
tested the only in-scope byte-width change large enough to matter.  Initial
four-GPU `short` job `6506867` failed safely before timing because the FP8
reference declared the wrong scale recipe.  Corrected job `6506962` completed
`0:0` in 1:20 on four independent exact-Qwen grouped shapes.  Across ranks,
FP8-to-MXFP4 reduced gate/up from 0.06329--0.06365 to 0.03912--0.03918 ms
(1.617--1.627x) and down from 0.03534--0.03570 to 0.02277--0.02289 ms
(1.552--1.561x).  The cost is material numerical change: relative RMSE is
0.1244--0.1251, cosine similarity 0.99215--0.99223, and maximum absolute error
is about 0.80 gate/up and 0.54 down.  Projecting only measured expert kernels
suggests roughly 4.3 ms of A7 replay savings, or about 28.4 ms overall rather
than the 26-ms target, so unchanged full-model alignment remains mandatory.

Isolated branch `codex/qwen3-a7-fp4-experts-20260825`, worktree
`worktrees/a7_fp4_experts`, commits `299c8e3` and `4d6a744` add an explicit
`MINISGL_DEEPGEMM_EXPERT_WEIGHT_DTYPE=fp4` mode.  It converts loaded FP8 expert
weights once during model prewarm to packed per-32 MXFP4, releases the source
FP8 tensors, supplies explicit DeepGEMM recipes, handles packed logical K
widths, fails on mode changes, and keeps FP8 as the default.  The AFD bundle
and case launchers propagate the mode across container boundaries and record
it in `afd-result.json`, preventing a silent FP8 measurement.  CPU-short jobs
`6507343` and `6507515` passed 65/65 tests; the latter covers the pinned
launcher commit.  First four-GPU exact-Qwen row-21 DeepEP smoke `6507516`
made normal compile/execution progress and failed cleanly at 4:29 only because
packed FP4 reached 0.213595 maximum relative error versus the FP8-only 0.15
smoke limit; its 0.051457 maximum absolute error remained below 0.08.  Commit
`090358e` leaves the FP8 gate unchanged and gives explicit FP4 smokes a
separately reported 0.25 relative bound.  Warm-cache replacement `6507665`
passed in 1:44: staged/fused FP4 errors were 0.213595/0.237834 and fused staged
time was 0.336982 ms.  Identical same-node FP8 control `6507699` passed in
1:33 at 0.339872 ms fused, only 0.85% slower, while its non-fused stage was
1.44% faster.  Because that integrated result does not reproduce the isolated
GEMM projection, commit `bd1d610` adds expert-only timing on one live DeepEP
packet.  Paired same-allocation `short` job `6507808` completed `0:0` in 4:58
with FP8/FP4/FP4/FP8 and 100 iterations each.  Expert-only medians were
0.106021 ms FP8 versus 0.095182 ms FP4, only a 10.22% gain.  Fused full-stage
medians were 0.361713 versus 0.338165 ms (6.51%); non-fused staged medians were
0.355062 versus 0.341047 ms (3.95%).  The earlier 1.6x isolated result is best
explained by repeatedly timing one same-weight GEMM, which gives packed FP4
much stronger partial L2 reuse than the production gate/up -> activation ->
down sequence.  Applying the integrated expert gain to exact A7 predicts only
about 1.2 ms improvement, around 31.5 ms rather than 26 ms, while the numerical
error is substantial.  Reject the FP4 candidate without an eight-tray exact
run; do not spend official alignment or A15 allocations on it.

That rejection was based on eager host-driven timing and is superseded by a
CUDA-Graph audit.  Test-only commits `77fb673` and `0836cff` add an NVTX-scoped
expert benchmark and graph-replayed expert-stage benchmark without changing the
production FP4 path.  Four-GPU `short` profile job `6512629` completed `0:0` in
3:17 with normal startup and attributed 99.111 us of expert kernels to FP8
(61.096 us gate/up, 4.546 us quantization, 33.469 us down) versus 63.680 us to
FP4 (37.208/4.478/21.994 us), a 35.7% kernel-time reduction.  CPU-only export
jobs `6512717` and `6512718` completed `0:0`.  Exact-Qwen ABBA graph job
`6512853` then completed `0:0` in 5:08: FP8 graph replay was
0.103371/0.103408 ms and FP4 was 0.067863/0.067658 ms, a stable 34.4% reduction;
all BF16, FP8, and fused checks passed, with the previously accepted FP4 error
bounds unchanged.  The eager integrated gap was therefore CPU launch/config
overhead that production graph replay excludes.  Advance FP4 to one exact A7
performance gate, but still withhold alignment and A15: its projected A7 latency
is about 28.7 ms, above the 26-ms target, and the numerical change remains
material.

Exact A7 FP4 performance gate `6512936` was submitted from clean head `0836cff`
on eight trays with `MINISGL_DEEPGEMM_EXPERT_WEIGHT_DTYPE=fp4`, `short` QoS,
and a one-hour limit.  Its task root is
`ax_alignment_20260824/a7_fp4_exact_20260825`; monitor startup at five minutes
for compilation/load/prefill progress, then use a fifteen-minute cadence only
after progress is proven normal.  Do not launch alignment from this candidate
unless the unchanged exact latency metric is competitive.

Gate `6512936` completed `0:0` in 26:05.  Five-minute health checks showed
normal progress rather than a hang: all 28 attention workers initialized,
expert compilation and 48-shard loading advanced continuously, the packed
expert tray stabilized near 50.3 GiB/GPU, and sustained 128K prefill kept the
attention GPUs mostly at 83--100% utilization while the expert GPUs remained
active at roughly 34--100%.  The unchanged exact metric is 28.5652352 ms over
15/15 retained samples with zero outliers, 3.2265% dominant range, and
183.789840 TPS/active-GPU.  Relative to FP8 A7 `6504134`, FP4 saves 4.128601 ms
(12.63%) and raises TPS/GPU by 14.45%, matching the graph-based projection, but
it remains 2.565235 ms (9.87%) above the 26-ms expectation.  Therefore do not
run alignment or A15 from this candidate yet.  Local copies of its metric,
logs, all-rank GPU telemetry, and Nsight reports are under
`ax_alignment_20260824/a7_fp4_exact_20260825/`.

CPU-only `cpu-short` Nsight export jobs `6513520` and `6513522` completed
`0:0` in 2:38/2:34 for representative FP4 attention rank 1 and model rank 29;
the first submission attempt created no job because it omitted the required
CPU-partition account, then the corrected submissions used
`coreai_comparch_sysarch`.  Same-15-launch comparison shows FP4 reduces the A7
model span by 4.128996 ms and summed model kernels by 4.585750 ms.  The main
kernel changes are grouped GEMM 15.742349 -> 12.064436 ms/replay, combine
5.201061 -> 4.562706, and model wait 24.758319 -> 24.462048; attention waits
fall by 8.412258 ms summed across both lanes, yielding the exact 4.128601-ms
critical-span win.  Against A1, however, FP4 A7 model span remains 3.535020 ms
longer.  Its residual component gaps are grouped GEMM +1.128912 ms, combine
+1.263429, QKV publication +0.908994, and model wait +1.068850 (overlapped).
The shared grouped-GEMM kernel name masks the FFN-specific regression: exact
per-launch stream-order classification shows expert gate/up at
4.129558 -> 4.850686 ms (+0.721128, 17.5%) and expert down at
2.442216 -> 2.979889 ms (+0.537673, 22.0%), while dense O plus QKV projection
improves by 0.128933 ms.  Full combine including its reduce epilogue grows
4.113217 -> 5.429291 ms (+1.316074, 32.0%); full dispatch grows
2.856563 -> 2.950402 ms (+0.093838, 3.3%); the two routing kernels grow
1.559539 -> 1.665304 ms (+0.105764, 6.8%); and the fused SiLU/multiply/output
quant kernel grows 0.770948 -> 0.798460 ms (+0.027511, 3.6%).  Expert GEMMs
plus combine alone add 2.574875 ms of summed kernel work.  These sums are not
all exposed on the critical path, but they identify combine and both expert
GEMMs as the dominant FFN regressions versus A1.
No expert-only change can reach 26 ms.  Because faster FP4 compute changes the
resource balance, the next bounded test is an FP4-specific DeepGEMM/DeepEP SM
partition sweep with Nsight kernel attribution; an exact A7 rerun still
requires a credible roughly 2.6-ms integrated path.

The FP4-specific SM-partition audit is closed without an exact A7 run.  Clean
branch `codex/qwen3-a7-fp4-deepep32-20260825` combines FP4 with the previously
validated four-cluster DeepEP policy at production commit `d967fb4`; test-only
commit `b0c6253` adds an NVTX range around the fused-stage benchmark.  Full
CPU contracts job `6513645` passed 65/65 in 1:03 on `cpu-short`.  Four-GPU
`short` sweep `6513690` completed `0:0` in 10:48 with normal build/case
progress and identical BF16/FP4/fused numerical checks for
128/24, 144/8, 136/16, 120/32, and repeated 128/24 DeepGEMM/DeepEP SM splits.
Fused eager times were 0.522855/0.515900/0.523657/0.513840/0.519475 ms and
graph expert times were 0.069461/0.070249/0.069729/0.072679/0.069340 ms.
Representative CPU exports `6513947`--`6513950` completed `0:0` in
1:26--3:54; the first case-4 submission created no job because the `cpu-short`
submit limit was full, then replacement `6514111` completed `0:0` in 1:26.
NVTX attribution makes 120/32's ceiling explicit.  Against the mean of both
128/24 controls, one fused-stage GPU span improves only 0.5169865 ->
0.5098280 ms (-7.1585 us), corresponding to roughly 0.673 ms across one
94-stage critical lane.  Active union improves by 12.1225 us/stage (a loose
1.140-ms/replay ceiling): combine saves 13.7705 us and dispatch saves 1.7465,
but FP4 GEMM loses 3.4045.  The other partitions do not offer a stronger
kernel-time result.  This is far short of the 2.565-ms exact gap; reject the
32-SM default and do not spend eight trays on it.  Logs and all representative
SQLite exports are local under
`ax_alignment_20260824/a7_fp4_sm_partition_20260825/`.

The dense-weight FP4 audit is also closed without a production or exact A7
run.  A clean source audit found that the existing ungrouped DeepGEMM API
already accepts FP8 activations with packed MXFP4 weights, so QKV and O
projection could be bounded without implementing a new kernel.  Test-only
branch `codex/qwen3-a7-dense-fp4-probe-20260825`, commits `f0883f9`,
`d1e0586`, and `c84d7d7`, measures CUDA-graph replay for exact Qwen shapes
QKV `[M,4096] x [9216,4096]` and O `[M,8192] x [4096,8192]` at A1 `M=3`
and A7 `M=21`.  The initial one-GPU submission was rejected before job
creation by the short-QoS four-GPU minimum.  Job `6514520` was canceled while
pending with zero elapsed time after its clean-head guard was found to contain
a mistyped full hash.  Corrected job `6514522` failed fast in 2:21 after
successfully building the DeepGEMM extension because the probe omitted the
workspace-local TVM-FFI cache; it showed no GPU hang or stalled work.
Four-way sharded retry `6514603` used all four GPUs, `short`, and a 30-minute
limit, then completed `0:0` in 1:49 with normal quantization/JIT/case progress.
At A7's 21 rows, FP4 changes QKV 12.336320 -> 12.298880 us and O 14.353024 ->
14.349888 us, only 0.040576 us/stage or 0.007628 ms across 188 stages.  At
A1's three rows it saves 2.068480 us QKV and 2.085312 us O, about 0.780913
ms/replay, so the benefit disappears rather than scales at A7.  FP4 versus
FP8 output cosine is only 0.992927--0.993217 and relative RMS error is
12.26--12.55 percent.  Reject dense FP4: its A7 upper bound is negligible,
its accuracy is worse, and it cannot contribute materially to the remaining
2.565-ms gap.  Logs and all four Nsys reports are local under
`ax_alignment_20260824/a7_dense_fp4_probe_20260825/`.

The next causal mechanism is MB2 packet merging, motivated by an all-rank A7
trace audit rather than a synthetic-only result.  CPU-short export jobs
`6508074`--`6508076` materialized profiler ranks 30--32; rank 29 was already
local.  All four model ranks have the same 15-replay graph span
(32.579134--32.584058 ms mean), but their two expert streams are consistently
asymmetric: lane 0 sums 4.514694--4.755883 ms/replay while lane 1 sums
6.950167--7.209364 ms/replay.  Dispatch and combine are approximately balanced
across lanes and the same expert asymmetry appears on every rank, ruling out a
single-rank placement or hang.  Both microbatches activate almost every local
expert, so the current separate calls reload nearly the same expert weights.

Four-GPU `short` jobs `6508245` and `6508296` measured the resulting reuse
ceiling with 100 FP8 expert iterations and the accepted 128/24 SM split.  The
first submission encoded a comma inside Slurm `--export`, so Slurm safely ran
only rows 21; it completed `0:0` in 1:49 at 0.106763 ms expert-only and
0.367787 ms fused staged.  The explicit rows-42 replacement completed `0:0`
in 2:05 at 0.107986 ms expert-only and 0.365519 ms fused staged.  Thus one
merged packet handles twice the routed rows for only 1.1 percent more expert
time and essentially unchanged full-stage time; comparing two row-21 stages
with one row-42 stage exposes a credible roughly 50-percent MoE-stage saving.
Both jobs showed normal build/execution progress and passed BF16, FP8, and
fused-router numerical checks.

Isolated branch `codex/qwen3-a7-merged-mb2-moe-20260825`, worktree
`worktrees/a7_merged_mb2_moe`, starts from accepted `bc9ab7f`.  The candidate
applies only to multi-source (`fanin > 1`) MB2 model ranks, preserving the A1
baseline.  It retains independently masked routing for each logical MB, merges
the prepared packets for one lane-zero DeepEP dispatch/expert/combine, splits
the fixed rows before next-layer QKV publication, and uses explicit cross-lane
events.  One shared buffer is sized to the merged dispatch bucket and the exact
merged path is prewarmed before CUDA graph capture.  All invalid shape, empty-
packet, event-grid, and buffer contracts fail fast.  Remote CPU tests currently
pass 38/38.  Clean commit `92952f8` contains the five-file implementation and
tests.  Two attempted CPU-only `batch` submissions were rejected before
allocation (first missing the account, then because `batch` requires a GPU),
so the 0.14-second mocked suite ran directly rather than wasting a GPU.  The
exact A7 gate is job `6508956`, submitted to `short` with a one-hour hard
limit, eight trays in one NVL72 segment, clean-source enforcement, and the
unchanged warmup-plus-15 trace contract.  A15 and official alignment remain
gated on an approximately 26-ms A7 result.

Job `6508956` completed `0:0` in 24:41 after healthy synchronized startup and
sustained prefill: all model ranks loaded 48/48 shards together, all eight
minute-progress files advanced, attention trays held roughly 87--100% GPU
utilization, and the model tray held roughly 56--65%.  There was no hang or
inefficient low-utilization prefill.  The candidate is nevertheless rejected:
42.9341909 ms strict CUDA mean and 122.280166 strict TPS/active-GPU, versus
32.6938360 ms and 160.580728 for control `6504134`.  All 15 samples were
retained with zero outliers and a 2.1544-percent dominant range.  Attention and
model spans both expand to about 42.9 ms; attention remains 752 graph nodes,
while model drops from 3,395 to 3,113 nodes.  This is a real synchronized
wavefront regression, not a metric or hang artifact.  Do not run A15 or the
official scorer for `92952f8`.  CPU-short job `6509574` exports candidate model
rank 29 for causal kernel/overlap attribution.  The exported SQLite is local at
`scratch/qwen3_ffn_overheads_20260820/ax_alignment_20260824/a7_merged_rank29.sqlite`.
Per replay, all block-scaled GEMMs fall from 752 calls / 15.742349 ms to 564 /
11.441701 ms, dispatch and combine each fall from 188 to 94 calls, and those
work reductions save about 6.7 ms.  However, readiness waits grow from
24.758319 to 33.319597 ms and fan out over 95 streams instead of two; delaying
MB0 next-layer QKV until MB1 joins the merged packet destroys the attention
wavefront.  This is the direct cause of the net regression, so full packet
merging cannot be repaired by more kernel tuning alone.

The older density-tail commit `0fe859b` was audited before reuse.  Its previous
A15 job `6488914` was canceled after eight minutes of active prefill, not
hung: all 60 attention GPUs were sustaining 82--100 percent utilization and no
decode metric was produced.  At A7, changing expected M from 21 to 12 would
still select the same 32-row alignment and hard-coded two-CTA grouped layout,
so no duplicate GPU run was spent on it.

Branch `codex/qwen3-a7-grouped-cga-policy-20260825` tested whether high routing
density should replace the two-CTA grouped GEMM with one CTA.  Commit `dd4c6fa`
keeps the accepted layout below expected M 16 and selects one CTA above it;
focused CPU contracts pass 17/17.  First paired smoke `6509886` failed before
timing because its new wrapper omitted workspace-local TVM-FFI cache variables;
replacement `6510061` completed `0:0` in 9:24 with control/candidate/candidate/
control on one four-GPU allocation.  Numerical checks matched, but candidate
fused-stage medians were about 0.349 ms versus 0.328 ms control and candidate
expert-only time was about 0.219 ms, roughly twice the accepted two-CTA
evidence.  Reject the one-CTA policy without an exact A7 run.

The A7 trace proves clustered expert kernels overlap only the opposite lane's
one-warp readiness wait, not DeepEP communication.  Test-only branch
`codex/qwen3-a7-deepgemm-sm-sweep-20260825`, commit `2cec7b9`, therefore permits
an explicit independent compute/communication budget only in the smoke and
adds expert-only timing.  Misencoded comma-export job `6510429` was canceled
after eight seconds before a benchmark; replacement `short` job `6510433`
completed `0:0` in 12:43.  It used the actual Qwen dimensions H4096/I1536,
rows 21, E128/top-k8, fixed 24-SM DeepEP, 100 iterations, and DeepGEMM grids
128/96/112/120/136/144/152/128.  All numerical checks matched.  Expert-only
times were respectively 0.106246/0.107602/0.107195/0.107866/0.108683/0.109398/
0.106874/0.106784 ms: accepted 128 SMs remain best within noise.  Do not spend
an exact A7 run on independent expert-grid sizing.  Also note that the older
generic non-FP4 A7 sweep wrapper used H7168/I2048; use H4096/I1536 for future
Qwen expert microbenchmarks.  The FP4 comparison wrapper already defaulted to
the correct H4096/I1536 dimensions, so its rejection remains valid.

Branch `codex/qwen3-a7-gate-weight-l2-20260825`, commit `9f001d2`, tested an
`EVICT_LAST` TMA load hint only for the DeepEP partial-sum gate/up weights;
down weights and activation loads retained the accepted policy.  Focused CPU
contracts passed 18/18.  Paired exact-shape ABBA smoke `6510975` completed
`0:0` in 11:19 with matching numerical checks.  Control fused-stage times
were 0.363198/0.363426 ms, candidate times were 0.351662/0.366854 ms, and
candidate expert-only times were 0.107026/0.106779 ms.  This is run-to-run
noise, not a reproducible gain, so reject the hint without an exact A7 run.

The next isolated target is the single-scaleout combine launch.  The accepted
A7 launch has a 24-CTA, one-warp-per-CTA grid because its useful-warp cap uses
only `num_max_tokens_per_rank=21`, even though each rank reduces partials from
four scale-up ranks.  Its combine kernels cost about 5.204 ms per replay,
roughly 1.900 ms more than A1.  Branch `codex/qwen3-a7-combine-warps-20260825`,
commit `9d3d3e7`, sizes the cap from the aggregate
`num_max_tokens_per_rank * num_scaleup_ranks`: A1 remains one warp per CTA and
A7 becomes four.  Focused source contracts pass 17/17.  The reusable paired
wrapper now isolates DeepEP build caches by source label as well as DeepGEMM
caches.  First smoke `6511211` failed in 25 seconds before any benchmark code
because the submitted legacy `/home` image path no longer exists; it consumed
no meaningful GPU time.  Replacement `6511406` used the verified Lustre image
and EP venv and completed `0:0` in 10:40.  All four ABBA phases advanced in
lockstep and passed identical numerical checks; there was no hang or idle
prefill phase.  Control fused-stage times were 0.340973/0.344719 ms, versus
candidate 0.355473/0.365209 ms, so aggregate-warp sizing regresses the mean by
about 5.1 percent.  Candidate expert-only time remained 0.106093/0.107054 ms.
This confirms that the existing one-warp-per-output-token mapping is already
work-complete: extra warps add launch/synchronization cost rather than sharing
one token's four-rank reduction.  Reject `9d3d3e7` without an exact A7 run.
Follow-up commit `5a28fc8` tests the bounded midpoint: it amortizes at most two
worst-case reduced tokens per warp, selecting one/two/four warps per CTA for
the A1/A7/projected-A15 capacities.  Focused contracts pass 17/17.  Exact-Qwen
ABBA smoke `6511547` completed `0:0` in 10:15 on `short` with fresh labeled
build/JIT caches.  All four phases advanced normally and passed identical
numerical checks.  Control fused-stage times were 0.334013/0.348480 ms versus
candidate 0.345171/0.351115 ms; candidate mean is about 2.0 percent slower and
does not beat both controls.  Reject `5a28fc8` without an exact A7 run and
close the combine launch-width family.

The exact transport trace also closes QKV launch-width work as a primary
target.  A1 uses one 24-block single-source publication kernel averaging
8.647 us/call and 1.625634 ms/replay; A7's existing flattened generic kernel
uses one 168-block launch for all seven edges, averages 13.261 us/call, and
costs 2.493044 ms/replay.  Seven times the payload raises the call span by only
4.614 us and explains the known 0.867410-ms replay delta.  This is efficient
fan-out scaling and cannot recover the remaining multi-millisecond expert
weight cost.  The accepted FMHA-only initializer returns into
`AfdFmhaRuntime` before the legacy MegaMoE AG/EG setup, so its dual-lane weight
reuse is not a backend toggle under this placement; using it would require a
new persistent expert-service integration, not an environment-only A7 run.

Exact stream timing rejects that persistent-service integration under the
current schedule before implementation.  On accepted A7 rank 29, each replay
alternates expert stages on streams 32 and 164.  Lane 0 has 93 stages averaging
52.454 us and lane 1 has 93 averaging 81.017 us, but lane 1 begins an average
103.724 us after lane 0's expert stage ends (91.744--120.768 us), and the next
lane-0 stage begins an average 105.214 us after lane 1 ends
(94.976--130.624 us).  Hundreds of MiB of intervening dense matrices therefore
separate the same expert weights; retaining roughly 600 MiB of expert weights
through that interval is not credible, and waiting to pair lanes would recreate
the wavefront loss measured by merged-MB2 job `6508956`.  The failed
gate-weight L2 hint in job `6510975` independently supports this conclusion.
Close the persistent dual-lane weight-reuse path unless the graph schedule is
first changed without delaying either microbatch.

The remaining packed-weight audit found no maintained grouped-NVFP4 drop-in.
The local DeepGEMM grouped SM100 path hard-codes per-32 UE8M0 scales for its
MXFP4 instruction descriptor.  Vendored CUTLASS contains Blackwell
MXF4/NVF4 instructions with UE4M3 per-16 scaling, but exposing them here would
require new grouped runtime dispatch, scale packing/layout, recipe validation,
and numerical coverage.  Current upstream DeepGEMM grouped dispatch likewise
selects its SM100 FP8/FP4 grouped path from integer scale tensors, while its
published grouped tests exercise the existing recipe rather than a grouped
NVFP4 path.  Treat NVFP4 as a new invasive kernel/runtime feature, not a
configuration candidate; first measure whether reducing active routed work can
recover enough latency to justify any approximate byte-width or route change.

Four-GPU `short` job `6512021` completed `0:0` in 12:56 with a 20-minute hard
limit and normal progress: the first five minutes built DeepGEMM and DeepEP,
then all six 100-iteration cases advanced sequentially with fresh log markers,
and teardown finished with seven minutes of margin.  It swept top-k
8/6/4/2/1/8 at rows 21, H4096/I1536, E128, and the production 128/24 SM split.
All BF16, FP8, and fused-router numerical self-checks passed.  Fused FP8 times
were 0.333164/0.347407/0.339585/0.345222/0.335420/0.361829 ms and BF16 times
were 0.330098/0.312748/0.320043/0.318578/0.318545/0.329609 ms, so fewer route
entries alone do not improve the full stage.  The deterministic smoke routing
still activates all 32 local experts at top-k 8/6/4/2 and 21 at top-k 1;
therefore this closes route-entry/communication scaling but is not yet a strong
active-expert sparsity ceiling.  Logs are local under
`ax_alignment_20260824/a7_topk_work_sweep_20260825/logs/`.  Use the existing
test-only expert-timing branch for any sharper active-expert measurement; do
not implement route truncation from this result.

That sharper ABBA measurement is `short` job `6512211`, using clean test-only
commit `2cec7b9` and top-k 8/1/1/8.  It completed `0:0` in 8:02, advanced from
fresh build into all four GPU cases by the first five-minute check, and had no
hang or slow teardown.  Isolated FP8 expert times were
0.106035/0.089752/0.103693/0.106640 ms; paired means show only about 9.0
percent improvement from top-k 1 even though active local experts fall from 32
to 21.  Fused-stage times were 0.347743/0.347552/0.367824/0.357612 ms, so the
full stage has no gain.  This upper bound is smaller than the already-rejected
MXFP4 expert gain and comes with far greater model error.  Close active-route
truncation without an exact A7 or alignment run.  Logs are local under
`ax_alignment_20260824/a7_topk_expert_ceiling_20260825/logs/`.

The existing fused MegaMoE implementation is not a direct FMHA-only backend.
Its union topology assigns disjoint AG ranks (which own the dense model and
routing) and EG ranks (which own experts), whereas FMHA-only puts dense model,
routing, and EP experts together on the model ranks and leaves attention ranks
with FMHA only.  `_init_afd_runtime` therefore returns into `AfdFmhaRuntime`
before constructing `MegaMoEM2NAfdAdapter`; making model ranks act as both AG
and EG would require a new symmetric-buffer topology and runtime schedule.
Historical corrected MegaMoE evidence also does not establish the target:
A7:F1/FEP4/128K batch 5 was 28.055288 ms in job `6265474`, A8 batch 6 was
30.743356 ms in job `6265476`, and A15 batch 6 was 30.752484 ms in job
`6263396`.  Do not treat MegaMoE as an environment toggle or spend an exact
FMHA-only job on it without a new standalone topology proof.

The maintained split MegaMoE kernel already implements both FP8xFP8 and
FP8xFP4 expert GEMMs, so the historical 28AG+4EG topology provides a bounded
topology/format experiment rather than an FMHA-only backend toggle.  Branch
`codex/qwen3-a7-megamoe-fp4-20260825`, worktree
`worktrees/a7_megamoe_fp4`, clean commit `c6420a5` starts from the exact
historical source `762b477` used by corrected A7 job `6265474`.  It adds an
explicit `MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4` mode, converts loaded Qwen
block-FP8 expert weights once to packed per-32 MXFP4, passes the existing
kernel's `use_fp8_weights=false` contract through all per-layer/dual/persistent
EG paths, keeps FP8 as the default, and records the selected dtype in worker
logs.  The exact controller additionally records backend/dtype in
`afd-result.json` and has a one-hour hard limit.  Two preflight attempts
created no Slurm job: the reused historical controller first expected its old
intentionally dirty profiler and then found a sync-omitted empty extractor
directory.  Both were corrected without GPU use.  Exact A7 batch-6 job
`6514983` started after 14:53 queued and failed before model loading at 3:44:
the all-node port probe released its sockets, then the coordinator found
control port 27874 occupied on the head node.  This was a startup port race,
not a MegaMoE/FP4 or prefill failure, and the watchdog stopped the allocation
promptly.  Retry `6515294` uses the same exact source/shape, eight trays,
`short`, and one-hour limit with dedicated port block 43120--43126, but was
canceled at 1:48 of startup when the user reaffirmed that algorithm/topology
and precision must remain unchanged.  It produced no performance result.
Do not resume FP4, MegaMoE, alignment, A15, or any other GPU experiment from
this stop state.

Finally, exact lane attribution closes stream ordering and local compute
contention.  In accepted A1, the clustered grouped expert calls average
17.335 us on stream 32 and 17.732 us on stream 164, with 52.697/52.771 ms
summed over the trace.  Accepted A7 grows to 23.848/37.402 us and
72.497/111.307 ms.  An interval audit shows those calls overlap only the
opposite lane's one-warp `wait_ready_kernel`, never another local compute
kernel, so CUDA stream priority or lane inversion cannot remove local kernel
competition.  Even the impossible bound of making every slow-lane A7 call as
fast as the fast lane saves only `(111.307-72.497)/16 = 2.426 ms` per replay
and leaves about 30.27 ms.  Holding all A7 expert work to the A1 total removes
about 4.896 ms and still leaves about 27.80 ms, because the measured QKV and
combine payload deltas remain.  Reaching 26 ms from 32.694 ms would require
removing all expert scaling plus roughly 63 percent of the real communication
delta, despite the transport audits showing efficient seven-source QKV fanout
and work-complete combine.  Under unchanged FP8/top-k8/FEP4/F1 resources and
the unchanged wavefront, the 26-ms target is inconsistent with the measured
work; a topology, format, or model-routing scope change is required before
another exact A7 run can have a positive gate.

The accepted rank-5 alignment trace has 16 graph launches.  Per replay it
contains 752 `sm100_blockscaled_gemm_1d1d_impl` calls with 11.551 ms summed
GPU time.  The 376 two-CTA grouped expert calls execute three cluster
rendezvous apiece: before TMEM allocation, after barrier/TMEM initialization,
and before TMEM deallocation.  Upstream DeepGEMM retains all three because the
teardown barrier fixes a real TMEM lifetime race, but uses relaxed-arrive
cluster synchronization for these rendezvous.

Accepted branch `codex/qwen3-deepgemm-relaxed-sync-20260824`, worktree
`worktrees/deepgemm_relaxed_sync`, commit `1c4b8ec` adopts that existing
relaxed-arrive helper only for the 2-CTA path.  Dense one-CTA O/QKV kernels,
the alignment standard, graph schedule, and transport are unchanged.  First
CPU job `6495610` failed before tests because the wrapper's home-directory
image default is not shared on OCI CPU nodes.  Replacement `6495630` used the
canonical Lustre image/venv and passed 60/60 in 3:06.  Four-rank GPU smoke
`6495716` is submitted to `short` for at most 15 minutes with a fresh
DeepGEMM cache, BF16 plus FP8 expert paths, nine repeated staged iterations,
and the production 128-compute/24-communication SM split.  It failed before
kernel execution because the new communication-header include preceded the
vendored definition of `DG_STATIC_ASSERT`.  The amended commit orders
`common/math.cuh` first, matching existing MegaMoE callers.  Replacement
smoke `6495802` reused the successfully built extension objects and a new
isolated DeepGEMM JIT cache.  It compiled the corrected header, repeatedly
passed staged BF16 execution, and passed FP8 once, then failed only because
the smoke's toy 256 hidden size violates the fused router's existing 512
alignment.  Exact-shape replacement `6495854` used six rows, hidden 4096,
intermediate 1536, 32 local experts, top-k 8, and nine iterations.  It passed
BF16, FP8, and fused-router checks in 2:02.

Exact target job `6495978` completed with 15/15 measured replays, zero
outliers, 25.6057515 ms coordinator E2E median, 25.5912937 ms strict CUDA
mean, and 117.227368 TPS/GPU.  Strict CUDA is 0.0341509 ms / 0.133 percent
faster than `6493748`.  The changed 2-CTA expert
GEMMs improve by 0.1296413 ms summed GPU time per replay: gate/up improves
4.2066478 -> 4.1585471 ms and down improves 2.5429399 -> 2.4613993 ms.  Both
general grouped shapes win while their call counts remain identical; unchanged
one-CTA O/QKV variation masks most of the kernel saving in the end-to-end
mean.  Official unchanged-scorer alignment job `6496358` completed in 11:04
with 24 prompts / 408 tokens and top-1, top-10, and top-100 agreement all 1.0;
average and maximum rank are both 1.0.  This makes `6495978` the current
aligned acceptance point.  Logs and reports are under
`deepgemm_relaxed_sync_validation/a1_fep4_b6_128k/`.

Candidate branch `codex/qwen3-deepgemm-overlap-init-sync-20260824`, worktree
`worktrees/deepgemm_overlap_init_sync`, commit `5827d42` starts the existing
post-initialization two-CTA cluster rendezvous, performs the independent
`cudaGridDependencySynchronize()` wait, and only then waits for the cluster.
This preserves both ordering conditions while overlapping their wait time and
contains no target-shape branch.  CPU job `6496563` passed 60/60.  Exact-shape
four-rank smoke `6496616` passed BF16, FP8, and fused-router execution for six
rows, hidden 4096, intermediate 1536, 32 local experts, top-k 8, and nine
iterations.  Exact unchanged-contract performance job `6496834` completed at
25.5289545 ms E2E median, 25.4867808 ms strict CUDA mean, and 117.708079
TPS/GPU.  Trace attribution rejects the candidate despite the headline result:
the affected grouped GEMMs improve by only 0.009235 ms/replay in aggregate,
while their per-replay standard deviations are 0.077 ms for down and 0.152 ms
for gate/up.  Unchanged dense O/QKV kernels account for nearly the entire
0.104513-ms strict difference.  Keep the simpler accepted relaxed-arrival
checkpoint and do not run official alignment for `5827d42`.

Communication-backend audit for the ATP1/TP1 target: PyTorch initializes its
process group with Gloo, and the TP PyNCCL enablement returns immediately at
TP1, so no PyTorch ProcessGroupNCCL collective is on the measured path.  QKV/O
movement uses the custom FMHA transport kernels.  EP4 uses DeepEP custom
dispatch/combine kernels; DeepEP directly links NCCL to initialize its host and
device communicator, discover topology, and establish symmetric/Gin transport.
Thus NCCL is an underlying DeepEP dependency, but not PyTorch's NCCL collective
data path.  Multi-rank TP normally uses the project's direct PyNCCL wrapper;
only the explicit non-PyNCCL TP mode initializes PyTorch's NCCL backend.

`comm_ref_table.txt` is the user-provided official pure-communication latency
reference.  The accepted rank-5 trace averages 14.844 us in `dispatch_impl`
plus 3.839 us in its copy epilogue, and 19.549 us in `combine_impl` plus
4.333 us in its reduction epilogue.  Conservative traffic ceilings are 192
KiB for FP8 dispatch and 384 KiB for BF16 combine, before excluding local
routes.  Against the table's next-higher EP4 rows, core/full-stage latency
efficiency is about 100/79 percent versus NCCL A2A for dispatch and 85/70
percent for combine.  Versus UBX it is only about 20/16 and 28/23 percent.
Treat this as a signal rather than a pure-communication attribution because
the DeepEP stages also perform routing, local copy, and reduction; lower actual
remote payload makes the conservative efficiencies upper bounds.  Combine is
the clearer general communication opportunity.

For protocol work, use an explicit practical envelope rather than treating the
pure table as directly equivalent to routed MoE.  Linear interpolation at the
conservative payload ceilings gives about 14.50 us NCCL A2A at 198 KiB and
15.77 us at 384 KiB.  Dispatch core is already within about 2.4 percent of its
NCCL reference, but full dispatch is about 28.9 percent slower.  Combine core
is about 24.0 percent slower and full combine about 51.4 percent slower.  Aim
to bring each full stage within 20 percent of the interpolated NCCL reference:
roughly <=17.4 us dispatch and <=18.9 us combine, while retaining unchanged
end-to-end correctness and performance.  UB-X remains the stretch signal, not
the acceptance floor, because its pure A2A does no MoE routing/reduction.

Official NCCL `contrib/nccl_ubx` source provides a general design direction,
not a reason to use PyTorch ProcessGroupNCCL.  UB-X uses the same NCCL device
API and symmetric-window foundation as the current DeepEP path.  Its pure A2A
auto policy uses barrier-free Lamport write/poll below 0.25 MiB and UC above;
the target's conservative FP8 dispatch ceiling is about 198 KiB while BF16
combine is about 384 KiB.  It also fuses BF16-to-MXFP8 quantization into token
dispatch.  Do not transfer pure-A2A launch settings blindly: the pure-A2A
launcher defaults to 32 SMs, but UB-X's fused MoE dispatch and combine
launchers default to 128 SMs because their line/sub-warp layouts continue to
scale with bandwidth.
The accepted trace pays one 1.576-us standalone dispatch quantization kernel
per dispatch, about 0.296 ms/replay, and a 4.333-us combine reduction epilogue,
about 0.815 ms/replay.  A future general path should therefore fuse dispatch
quantization while retaining the current 128-element packed UE8M0 layout, and
replace steady-state global barriers with generation-tagged/Lamport arrival
where correctness permits.  Current DeepEP combine is already PUSH-like:
expert owners write origin-rank buffers, then a receiver-local reduction
epilogue runs after the final barrier.  The opportunity is eliminating the
two global barriers and reducing/fusing the arrival-plus-local-reduction tail,
not merely toggling combine direction.  UB-X's scale-per-32 format is not
directly compatible with current DeepGEMM.  Its benchmark-only recommendation
`CUDA_DEVICE_MAX_CONNECTIONS=1` must not be applied globally without
same-workload overlap proof.  Current DeepEP already loops top-k selections
directly and deduplicates destination ranks, so UB-X's top-k-LUT improvement
over a full expert scan is not missing from this implementation.

The official UB-X MoE benchmark also compares NCCL-EP low-latency and
high-throughput algorithms.  For the installed NCCL 2.30.4 single-node setup,
its source explicitly omits HT from the safe default because HT dispatch can
abort; `nccl_ep_ll` is the relevant alternate backend.  NCCL-EP is not present
in the current runtime artifact, so adopting it requires a deliberate build
and integration rather than a configuration toggle.  The measured target is
an all-LSA/NVLink group; its data and barriers use symmetric LSA pointers, so
Gin QP-count sweeps do not address the active fast path.
UB-X itself is likewise not a drop-in DeepEP backend: its public fused dispatch
uses a different MXFP8 scale granularity than the packed UE8M0/128 layout
consumed by the current DeepGEMM path, and it does not expose a compatible
public dispatch-plus-combine replacement.  The user therefore chose to keep
the NCCL-backed DeepEP path rather than start a separate UB-X integration.

User approved changing the SM split and expects the memory-bound expert GEMMs
to retain efficiency with fewer SMs.  Candidate branch
`codex/qwen3-deepep-32sm-20260824`, worktree `worktrees/deepep_32sm`, commit
`cfa8660` tests 32 DeepEP / 120
DeepGEMM SMs on a 152-SM device, replacing the low-fan-in 24/128 split while
leaving the existing high-fan-in policy separate.  This matches UB-X's
pure-A2A small-message default but not its MoE launcher default, and is only an
isolated SM-budget probe rather than a protocol choice.  Supporting it requires adding one
complete 8-CTA cluster to the Python and JIT launch allowlists and updating the
overlap contract tests.  CPU contract job `6497505` and exact-shape four-rank
staged smoke `6497506` passed independently from the accepted relaxed-arrival
parent; the fused FP8 staged total was 0.380779 ms versus 0.384818 ms in the
clean accepted smoke, while BF16 was 0.404804 versus 0.395275 ms.  Exact
unchanged-contract job `6497628` completed successfully with 15/15 samples and
zero outliers, but is rejected: its 25.6273520 ms E2E median and 25.5953893 ms
strict CUDA mean are respectively 0.0216005 and 0.0040957 ms slower than the
accepted 24/128 result.  This confirms that copying the pure-A2A 32-SM default
does not improve this token-indexed MoE kernel at three tokens per microbatch.
The first rank-5 export attempt `6497899` failed because the container tried to
re-enter a transient Slurm spool path; durable-script replacement `6497922`
completed.  Same-query attribution versus accepted 24/128 shows full dispatch
worsening 18.634 -> 20.809 us/call (+11.7 percent), while full combine improves
only 23.864 -> 23.431 us/call (-1.8 percent).  Core dispatch is the regression
(14.799 -> 16.792 us); core combine improves 19.533 -> 19.135 us.  The mixed
component result and worse E2E make rejection causal, not just noise.

The next protocol candidate is branch
`codex/qwen3-deepep-nccl-lsa-barrier-20260824`, worktree
`worktrees/deepep_nccl_lsa_barrier`, commit `20da38d`.  It preserves the
accepted 24/128 SM split and all payload/routing/reduction logic, but requests
NCCL multimem support plus exactly ten persistent LSA barrier slots for the
declared tags 0 through 9.  It replaces DeepEP's per-peer system-scope
atomic arrival loop with `ncclLsaBarrierSession<ncclCoopCta>` using
acquire-release ordering and the existing timeout.  On NVLink multicast
hardware this changes each arrival from one release atomic per peer to one
multimem reduction, while NCCL owns the graph-safe epoch.  This is a general
barrier-protocol/configuration probe with no target-shape branch.  Tightened
source contract job `6497835` passed 61/61.  After the isolated 32-SM control
completed, exact-shape four-rank build/correctness smoke `6497900` compiled the
NCCL API change but failed safely: an incorrectly tightened two-slot resource
reservation timed out at combine tag 5 on all four ranks.  The source audit
then identified all ten explicit constants (`kDeviceBarrierTag=0` through
`kHybridCombineTag1=9`), restored `lsaBarrierCount=10`, and extended the
contract test to cover the whole tag table.  Amended commit `20da38d` is under
replacement source test `6497991`, which passed 61/61 in 3:39.  Exact-shape
four-rank replacement smoke `6498034` passed BF16, FP8, and fused-router paths
for nine repeated staged iterations on the accepted 24/128 split.  Fused FP8
staged time was 0.363890 ms versus 0.384818 ms in the accepted smoke (-5.4
percent), while BF16 was effectively flat at 0.396501 versus 0.395275 ms.
Exact unchanged-contract performance job `6498084` completed and passed the
E2E plus trace-attributed communication acceptance gate.  It retained
15/15 with zero outliers at 25.425188 ms coordinator median, 25.3775303 ms
strict CUDA mean, and 118.214813 strict TPS/GPU.  Versus aligned relaxed-
arrival job `6495978`, E2E improves 0.1805635 ms / 0.705 percent and strict
CUDA improves 0.2137633 ms / 0.835 percent.  Rank-5 export job `6498331`
attributes the gain directly: full dispatch improves 18.634451 -> 16.535447
us/call (-11.264 percent) and full combine improves 23.863784 -> 23.190299
us/call (-2.822 percent), saving about 0.521228 ms of summed communication
GPU time per replay.  Dispatch now clears the practical 17.4-us reference
envelope; combine remains about 4.29 us above its 18.9-us envelope.  Official
unchanged-scorer job `6498364` passed 24 prompts / 408 tokens with top-1,
top-10, and top-100 agreement all 1.0 and average/maximum rank 1.0.  This makes
commit `20da38d` the current aligned acceptance checkpoint.

Follow-up branch `codex/qwen3-deepep-nccl-lsa-thread-20260824`, worktree
`worktrees/deepep_nccl_lsa_thread`, commit `75e81d5` retained the accepted
multimem protocol but uses `ncclCoopThread` for its single-leader multicast
arrival and epoch poll.  NCCL 2.30.4's CTA session adds four CTA-wide
synchronizations across arrival, timed wait, and session destruction; the
enclosing DeepEP grid synchronization or final kernel completion already
appeared to order the other threads.  This was a general barrier-overhead probe with no
shape branch.  CPU job `6498499` passed 61/61.  Exact four-rank smoke
`6498503` passed repeated BF16, FP8, and fused-router checks; its staged times
were 0.393884/0.386610/0.374642 ms.  Exact target job `6498587` passed 15/15
with zero outliers but regressed to 25.539801 ms E2E median and 25.5255538 ms
strict CUDA mean, respectively +0.114613/+0.1480235 ms versus the CTA-group
checkpoint.  Export `6498857` makes the rejection causal: full dispatch
regresses 16.535447 -> 20.155926 us/call (+21.9 percent) and full combine
regresses 23.190299 -> 24.335339 us/call (+4.9 percent).  The CTA cooperative
session therefore provides useful quiescence/ordering around the multicast
epoch; retain `20da38d` and do not run alignment for `75e81d5`.

Phase-specific SM follow-up branch `codex/qwen3-deepep-phase-sms-20260824`,
worktree `worktrees/deepep_phase_sms`, commit `aba94c0` retains the accepted
NCCL barrier and uses 24 SMs for dispatch but 32 for combine at low fan-in;
high fan-in retains 8/8.  It reserves DeepGEMM against the maximum phase grid,
so GB200 compute uses 120 SMs.  This is a general per-operation policy motivated
by the isolated 32-SM control: dispatch regressed sharply while combine
improved.  The reusable smoke now accepts separate dispatch/combine budgets
and validates the max-phase device partition.  Initial CPU job `6498922`
caught one stale single-budget assertion; its dependent smoke `6498924` was
canceled pending with zero elapsed GPU time.  Amended source job `6498946`
passed 61/61 and exact 24/32 smoke `6498947` passed all repeated BF16, FP8,
and fused-router checks, reporting the intended 120/24/32 budgets.  Exact
target job `6498982` passed 15/15 with zero outliers but is rejected at
25.636775 ms E2E median, 25.6111874 ms strict CUDA mean, and 117.136311 strict
TPS/GPU.  Versus accepted job `6498084`, it regresses E2E by 0.211587 ms /
0.832 percent and strict CUDA by 0.2336571 ms / 0.921 percent.  Export job
`6499337` completed, and its local rank-5 trace is under
`scratch/qwen3_ffn_overheads_20260820/deepep_phase_sms_job_6498982/`.
Same-query attribution shows full dispatch at 16.615905 us/call, essentially
flat at +0.49 percent, while full combine regresses 23.190299 -> 24.028359
us/call (+3.61 percent).  Across 188 calls of each stage, the extra
communication time is about 0.172681 ms/replay and explains most of the
headline regression.  The phase-specific code was removed locally; retain the
accepted single 24-SM low-fan-in communication budget and 128-SM DeepGEMM
budget from `20da38d`.

Candidate branch `codex/qwen3-deepep-combine-no-pdl-20260824`, worktree
`worktrees/deepep_combine_no_pdl`, commit `6c020a9` retains all accepted NCCL
barriers, payload, reduction, and SM budgets but disables Programmatic
Dependent Launch for the combine reduction epilogue.  The epilogue has no
independent work before its immediate `cudaGridDependencySynchronize()`, while
early residency oversubscribes the intended 24-communication/128-expert SM
partition.  Normal same-stream ordering already provides the required
completion and memory visibility.  This is a general launch-policy change with
no target-shape branch.  Local static contract passes 16/16; full remote source
job `6499491` passed 61/61 in 1:28.  Fresh exact-shape four-rank smoke job
`6499513` passed BF16, FP8, and fused-router checks for nine iterations with
six rows, hidden 4096, intermediate 1536, 32 local experts, top-k 8, and the
accepted 128/24 SM split.  Its fused-router staged time is 0.357330 ms versus
0.363890 ms for accepted smoke `6498034` (-1.8 percent), while BF16 is flat at
0.395737 versus 0.396501 ms.  Exact unchanged-contract performance job
`6499595` completed successfully with 15/15 samples, zero outliers, 25.498309
ms coordinator median, 25.4775682 ms strict CUDA mean (25.442550-ms median),
and 117.750642 strict TPS/GPU.  It regresses accepted job `6498084` by
0.073121 ms / 0.288 percent E2E and 0.1000379 ms / 0.394 percent strict CUDA,
so it cannot advance regardless of the promising smoke.  Accepted trace
inspection also shows all 3,008 combine epilogues already start after their
producer ends, with a 0.279-us average gap and zero PDL overlap; the original
early-residency hypothesis was false for the full graph.  Rank-5 export job
`6500265` completed in 34 seconds and its SQLite is copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepep_combine_no_pdl_job_6499595/`.
Same-query stage spans make the rejection causal: dispatch regresses
16.704987 -> 17.019188 us/call and combine regresses 23.469765 -> 23.704444
us/call.  Across 188 calls each, their combined ~0.103190-ms/replay loss
matches the +0.100038-ms strict regression.  The epilogue kernel alone appears
0.350513 us faster, but `combine_impl` slows 0.349232 us and the inter-kernel
gap grows 0.235960 us; this is timing relocation, not useful work reduction.
The accepted PDL launch policy is restored locally.  No official alignment
will run.

Replacement branch `codex/qwen3-deepep-combine-direct-output-20260824`,
worktree `worktrees/deepep_combine_direct_output`, commit `1877495` retains
the accepted PDL, NCCL barriers, routing, reduction arithmetic, bias handling,
top-k weights, and token-bounded grid.  It instead writes each token warp's
disjoint BF16 result directly to the output row, eliminating the intermediate
shared-memory staging writes and TMA copy.  The epilogue consequently launches
with zero dynamic shared memory.  This is a general reduction data-path change
with no target-shape branch.  Local static contract passes 16/16; full remote
source job `6500366` passed 61/61 in 1:44.  Fresh exact-shape four-rank build
and correctness smoke job `6500421` passed the accepted six-row, 4096-hidden,
1536-intermediate, top-k-8, nine-iteration, 128/24-SM contract for BF16, FP8,
and fused-router paths.  BF16 staged time improves 0.396501 -> 0.386663 ms
(-2.5 percent) and fused-router improves 0.363890 -> 0.360085 ms (-1.0
percent) versus accepted smoke `6498034`.  Exact unchanged-contract two-tray
performance job `6500517` nevertheless regresses: coordinator E2E median is
25.563346 ms and strict CUDA mean is 25.527330533 ms, respectively
+0.138158 ms / +0.543 percent and +0.149800233 ms / +0.590 percent versus
accepted `6498084`; all 15 measured replays are valid with zero outliers.
Trace export `6500876`, excluding replay zero, makes the rejection causal.
Full dispatch span is 17.447410 us/call versus accepted 16.704987
(+0.742423 us), and full combine span is 24.349885 us/call versus accepted
23.469765 (+0.880120 us).  The direct-store combine epilogue itself slows
4.183321 -> 4.297382 us, so eliminating TMA does not eliminate useful work:
the warp-issued global stores are slower than shared staging plus TMA.  Across
188 dispatch/combine pairs, the trace adds about 0.305 ms of summed GPU stage
time per replay; graph overlap masks part of it in the strict headline.  The
candidate is rejected without official alignment, and the accepted
shared-memory/TMA implementation is restored locally with the 16/16 static
contract passing.

The next general NCCL candidate specializes only barrier memory ordering by
phase on branch `codex/qwen3-deepep-nccl-lsa-phase-order-20260824`, worktree
`worktrees/deepep_nccl_lsa_phase_order`, commit `d80900b`.  NCCL's official
device-API LSA collective pattern uses acquire at the entry readiness barrier
and release at the exit data-publication barrier; accepted `20da38d`
conservatively used acquire-release for both.  The candidate propagates
`gpu_barrier`'s existing `kFlushStores` phase bit into the LSA session,
selecting acquire when there are no stores to publish and release when the
phase has flushed its stores.  It keeps the same NCCL communicator, multimem
protocol, ten tag slots, CTA cooperative group, 24-SM payload grid,
payload/routing arithmetic, and timeouts.  This removes only a fence with no
consumer in that phase and applies to direct and hybrid dispatch/combine; the
local static contract passes 16/16.  Remote source job `6501029` passed 61/61
in 1:26.  Exact-shape four-rank smoke job `6501046` compiled the candidate and
passed BF16, FP8, and fused-router checks for nine iterations in 4:31.  Its
BF16 staged time is 0.401401 ms and fused-router time is 0.368587 ms, about
1.2--1.3 percent slower than accepted smoke `6498034`, so the smoke is not
positive but remains too short to decide the overlapped graph result.  Exact
unchanged-contract two-tray target job `6501150` completed successfully with
15/15 retained samples, 1.7248-percent dominant range, 25.112056-ms coordinator
median, 25.0827552-ms strict CUDA mean, and 119.604086 strict TPS/GPU.  Versus
accepted `6498084`, this improves coordinator E2E by 0.313132 ms / 1.232
percent and strict CUDA by 0.2947751 ms / 1.162 percent.  Rank-5 trace export
job `6501689` completed in 44 seconds and is copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepep_nccl_lsa_phase_order_job_6501150/`.
Excluding replay zero, full dispatch improves accepted 16.704987 -> 14.883132
us/call (-10.91 percent) and full combine improves 23.469765 -> 22.567851
us/call (-3.84 percent).  Their main kernels improve 13.340338 -> 11.548215
and 19.006978 -> 17.945612 us/call respectively; epilogues and gaps are
essentially flat except combine reduction rises 4.183321 -> 4.339266 us.
Across 188 stage pairs this removes about 0.512069 ms of summed communication
GPU time per replay, causally supporting the overlapped headline improvement.
Dispatch is comfortably within the practical reference envelope and combine
transport is below 18.9 us, while the separate reduction epilogue remains the
full-stage gap.  Official unchanged-scorer job `6501708` completed in 10:57
and passed 24 prompts / 408 generated tokens with top-1, top-10, and top-100
agreement all 1.0 and average/maximum rank 1.0.  Commit `d80900b` is the new
aligned acceptance checkpoint.

The next general compute candidate targets the final 2-CTA DeepGEMM TMEM
teardown.  Rank-5 trace `6501689` attributes 6.612450 ms per replay to the two
grouped expert GEMMs, and their implementation performs a full-cluster barrier
immediately before `tcgen05.dealloc.cta_group::2.sync`.  The candidate replaces
only that final cluster rendezvous with `__syncthreads()`: all local warps still
finish before deallocation, while the synchronous 2-SM deallocation remains the
peer-CTA rendezvous for matching warp 0.  Allocation barriers, NCCL transport,
GEMM tiling, scheduling, numerical work, and the one-CTA path are unchanged.
This is shape-independent and applies to every 2-CTA grouped GEMM using this
implementation.  Remote branch
`codex/qwen3-deepgemm-local-teardown-20260824`, worktree
`worktrees/deepgemm_local_teardown`, commit `e2e57ec` is based exactly on
aligned `d80900b`; source job `6502143` passed 62/62 in 55 seconds.  Fresh-cache
exact-shape four-rank smoke job `6502167` passed BF16, FP8, and fused-router
checks for nine iterations in 4:33.  Versus aligned-checkpoint smoke `6501046`,
its staged times improve 0.401401 -> 0.385479 ms for BF16 (-3.97 percent),
0.373657 -> 0.371598 ms for FP8 (-0.55 percent), and 0.368587 -> 0.361575 ms
for fused routing (-1.90 percent).  The smoke is directionally positive and
clears the exact target gate.  Unchanged-contract two-tray A1:F1 target job
`6502261` used a candidate-specific `DG_JIT_CACHE_DIR`, with fresh candidate
`.cu` and `.cubin` artifacts observed for the grouped and dense SM100 kernels.
It completed 15/15 samples with 1.7650-percent dominant range, 25.072530-ms
coordinator median, 25.0539479-ms strict CUDA mean, and 119.741608 strict
TPS/GPU.  Versus aligned `6501150`, coordinator E2E improves 0.039526 ms /
0.157 percent and strict CUDA improves 0.0288073 ms / 0.115 percent.  The first
CPU-only trace export `6502697` failed because its nested container resolved the
transient Slurm spool script; durable-path retry `6502715` completed in 30
seconds.  Excluding replay zero, gate/up improves 22.082394 -> 21.935163
us/call and down projection improves 13.090212 -> 12.955440 us/call.  Across
188 calls of each per replay, the affected grouped GEMMs remove 0.053016 ms /
0.80 percent, directly supporting the strict end-to-end gain.  The SQLite is
copied locally under
`scratch/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_job_6502261/`.

Wrap-up reduced the implementation to one local-completion statement and one
concise lifetime comment; final remote commit `bc9ab7f` has no executable delta
from `e2e57ec`.  Requested final exact performance job `6502925` completed
15/15 samples with 1.3950-percent dominant range, 25.0087105-ms coordinator
median, 24.9813376-ms strict CUDA mean, and 120.089646 strict TPS/GPU.  Versus
aligned `6501150`, it improves coordinator E2E by 0.1033455 ms / 0.412 percent
and strict CUDA by 0.1014176 ms / 0.404 percent; it is the best exact FEP4
result.  Official unchanged-scorer job `6503374` completed in 11:37 and passed
24 prompts / 408 generated tokens with top-1, top-10, and top-100 agreement all
1.0 and average/maximum rank 1.0.  Final commit `bc9ab7f` is the aligned
acceptance checkpoint.

## Governing FFN ping-pong model

Treat `FFN(L-1) + QKV/QK-norm/RoPE(L)` as one indivisible model-side unit, `compute(L-1)`.

- Only one compute- or memory-intensive model lane may own SM resources at a time.
- Batch B may start only after batch A finishes its full compute unit and hands off `q_done`.
- `q_done` follows FFN, O projection, normalization, routing, expert work, QKV, Q/K normalization, and RoPE.
- QKV publication follows `q_done`; it is a communication tail that may overlap peer-lane compute.
- Each lane also waits for its own attention O payload before starting.
- Double buffers prepost receiver readiness and overlap communication with compute. They do not authorize two heavy model batches to run together.
- A wait placed only before QKV is too late because the earlier FFN work would already be enqueued.
- Deferred cross-batch combine overlap is forbidden.

The mirror attention pipeline is `attention compute -> O quantization -> O publication -> peer attention compute`. Attention quantization completes before that lane releases attention ownership.

## Current stop state

- Date: 2026-08-24 PDT.
- Current aligned remote worktree: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/worktrees/deepgemm_local_teardown`.
- Current aligned branch: `codex/qwen3-deepgemm-local-teardown-20260824`, commit `bc9ab7f`.
- Local subproject progress is staged for review; no local commit was created.
- Exact 32/120 SM control job `6497628` is rejected as neutral-to-worse; trace
  export `6497922` confirms a large dispatch regression and only a small combine
  improvement.
- NCCL multimem LSA-barrier commit `20da38d` was aligned and accepted after
  source `6497991`, smoke `6498034`, full target `6498084`, trace export
  `6498331`, and official scorer `6498364` all passed.  It improved strict CUDA
  by 0.835 percent and reduced both dispatch and combine communication; it is
  now superseded by aligned phase-order checkpoint `d80900b`.
- Leader-only NCCL follow-up commit `75e81d5` is rejected after source
  `6498499`, smoke `6498503`, full target `6498587`, and trace export `6498857`:
  both communication stages and both headline latency metrics regress.
- Phase-specific 24-dispatch/32-combine commit `aba94c0` is rejected after
  source `6498946`, smoke `6498947`, full target `6498982`, and export
  `6499337`: strict CUDA regresses 0.921 percent and full combine regresses
  3.61 percent.  Its local source changes were removed.
- Combine no-PDL commit `6c020a9` passes local static, remote source `6499491`,
  and exact-shape smoke `6499513`, but exact target `6499595` regresses strict
  CUDA by 0.394 percent and is rejected.  Trace export `6500265` attributes
  the full loss to slower dispatch/combine stage spans; local PDL is restored.
- Direct-output combine commit `1877495` passes local static and remote source
  job `6500366` at 61/61 plus exact-shape smoke `6500421`; exact target job
  `6500517` regresses strict CUDA by 0.590 percent and is rejected.  Trace
  export `6500876` shows +0.880 us/call full combine span and a slower direct
  global-store epilogue.  The accepted TMA path is restored locally; no
  official alignment was run.
- Phase-order NCCL commit `d80900b` passes local static, source `6501029`, smoke
  `6501046`, and exact target `6501150`.  Exact strict CUDA improves 1.162
  percent; trace export `6501689` attributes the gain to 10.91-percent faster
  dispatch and 3.84-percent faster combine spans.  Official alignment
  `6501708` passed perfectly; it is superseded by `bc9ab7f`.
- Minimal DeepGEMM teardown commit `bc9ab7f` passes source `6502143`, exact-shape
  smoke `6502167`, causal trace export `6502715`, candidate target `6502261`,
  final target `6502925`, and official alignment `6503374`.  Final strict CUDA
  is 24.9813376 ms, 0.404 percent faster than `d80900b`; alignment is perfect.
- A7:F1/FEP8 job `6494764` was canceled while pending with zero elapsed time and no allocation.
- Broad A8/A15 campaigns remain stopped unless explicitly reopened.

Rollback lineage:

- `8b93a9c`: enforce the exclusive model compute-unit schedule.
- `5c7ecb3`: keep final normalization on the last compute-owner stream.
- `89b594a`: transport post-attention O in FP8.
- `247e12e`: import Torch at runtime for prequantized O projection.
- Later commits are documentation-only staging checkpoints.

## Current ownership and data path

Attention workers own:

- Physical KV cache, FMHA backend, workspace, and FMHA metadata.
- Post-FMHA BF16-to-FP8 activation quantization.
- Publication of FP8 O plus packed UE8M0 scales.

Model or FFN workers own:

- Embedding, all checkpoint weights, input and post-attention norms.
- O projection, MoE routing, dispatch, expert GEMMs, combine, and final norm.
- QKV projection, Q/K normalization, RoPE, LM head, sampling, and token writeback.

Decode path:

1. Model publishes already-FP8 Q/K/V to attention.
2. Attention runs FMHA.
3. Attention quantizes O to FP8 with packed UE8M0 scales.
4. Attention publishes both payloads and releases one O-ready word only after both remote writes are system-fenced.
5. Model consumes the transported packed-scale layout directly.
6. Model runs the FP8-input O projection locally, then completes the exclusive FFN plus next-QKV unit.
7. Model publishes next-layer Q/K/V after `q_done`.

Dynamic prefill alone repacks O scales around fan-in padding. Decode has no scale repack. Four-head-aligned TP slices are required and fail fast otherwise. No fallback path is present.

## Receiver-ready signaling

Both directions use receiver-owned double buffers.

- A receiver posts readiness as soon as its slot is clean.
- A sender that later becomes ready can publish immediately without a request or credit round trip.
- Readiness signaling is off the payload critical path.
- A slot is reused only after local causal cleanup.
- QKV and O publications each issue one remote readiness release after their complete payload.
- Direct fabric transport must stay inside one NVL72 base system; submissions use `--segment` and a fabric-block preflight.

Exact A1 uses compile-time single-source and single-ready publication variants. The generic fan-in path retains flattened task mapping, bounded publication grids, the two-slot epoch protocol, and fail-fast validation; no fallback was added.

## Accepted performance and correctness

All rows use Qwen3 235B-A22B FP8, 128K context, batch 6 per attention-DP lane, ATP1, MB2, exact warmup plus 15 measured decode steps, and zero outliers unless noted.

| State | FEP | Job | E2E median ms | Strict CUDA mean ms | Strict TPS/GPU | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Legacy control | 4 | 6470598 | 30.6157295 | 30.4497429 | 97.988846 E2E | control |
| Initial FMHA-only | 4 | 6472328 | 25.9524935 | 25.9164299 | 115.595829 E2E | aligned |
| Preposted signal/data | 4 | 6474505 | 25.7321745 | 25.6797822 | 116.585561 E2E | aligned |
| Corrected exclusive FFN | 4 | 6490963 | 25.7428600 | 25.7122698 | - | user accepted |
| Current FP8-O | 4 | 6493748 | 25.6410675 | 25.6254446 | 117.071140 strict | aligned |
| Relaxed 2-CTA cluster arrival | 4 | 6495978 | 25.6057515 | 25.5912937 | 117.227368 strict | aligned |
| NCCL multimem LSA barriers | 4 | 6498084 | 25.4251880 | 25.3775303 | 118.214813 strict | aligned |
| Phase-ordered NCCL LSA barriers | 4 | 6501150 | 25.1120560 | 25.0827552 | 119.604086 strict | aligned |
| Minimal DeepGEMM teardown | 4 | 6502925 | 25.0087105 | 24.9813376 | 120.089646 strict | aligned final |
| Earlier FMHA-only | 8 | 6473036 | 25.3913920 | 25.3729383 | 118.236207 strict | historical |
| Current FP8-O | 8 | 6494759 | 25.2301460 | 25.1990735 | 119.051996 strict | profiled |

Final FEP4 improvement:

- Versus phase-ordered NCCL job `6501150`: coordinator E2E improves 0.412
  percent and strict CUDA improves 0.404 percent.
- Versus corrected exclusive-FFN job `6490963`: coordinator E2E improves 2.852
  percent and strict CUDA improves 2.843 percent.
- Job `6502925` retained 15 of 15 samples with 1.395-percent dominant range.

Current FEP8 evidence:

- Job `6494759` ran unchanged source in `fmha-only` mode on four trays and 16 active GPUs.
- It completed in 13:46 with exit 0, 15 of 15 samples, zero outliers, and all 16 reports.
- Versus earlier same-shape job `6473036`, E2E improves 0.635 percent and strict CUDA improves 0.685 percent.
- It is a performance and trace result; no new alignment run was requested.

Final correctness gates:

- Current remote source suite: 62 of 62 tests (`6502143`).
- Exact-shape BF16/FP8/fused-router smoke: job `6502167`, nine iterations.
- Transport job `6493119`: fresh SM100 build plus bit-exact FP8 O and packed-scale transport on two GPU pairs, 257 rows, nine iterations, and both slots.
- Corrected scheduling export `6491211`: zero cross-stream heavy-kernel intersections in the representative replay and all 15 measured replays.
- Official final alignment job `6503374`: 24 prompts, 408 generated tokens,
  top-1/top-10/top-100 all 1.0, average and maximum rank 1.0.
- First current A1 attempt `6493412` failed before measured work because Torch was imported only under `TYPE_CHECKING`; `247e12e` fixed it. It is not a performance result.

## Trace and result locations

All raw Nsight reports for the best unchanged-FP8 cases were copied back and
SHA-256 verified on 2026-08-25: A1 job `6502925` has 8/8 reports and A7 job
`6504134` has 32/32 reports under
`scratch/qwen3_ffn_overheads_20260820/best_fp8_nsys_traces_20260825/`
(47 MiB total).

Current aligned FEP4 remote root:

`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_final_validation/a1_fep4_b6_128k`

Current FEP4 local reports:

`scratch/qwen3_ffn_overheads_20260820/deepgemm_local_teardown_job_6502261/trace_sqlite/`

Current FEP8 remote root:

`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen3_ffn_overheads_20260820/serialized_ffn_validation/a1_ep8_fp8_o`

Current FEP8 local reports:

`scratch/qwen3_ffn_overheads_20260820/fp8_o_a1_ep8_job_6494759/nsys/`

Older corrected-pipeline reports:

`scratch/qwen3_ffn_overheads_20260820/corrected_a1_job_6490963/nsys/`

Older preposted-ready reports:

`scratch/qwen3_ffn_overheads_20260820/preposted_buffer_ready_validation/trace/nsys/`

Older best A15 FMHA-only reports:

`scratch/qwen3_ffn_overheads_20260820/best_a15_fmha_job_6485295/nsys/`

## 128K batch-capacity boundary

Under ATP1 and memory ratio 0.82:

- Observed KV capacity is 839,424 tokens per attention GPU.
- Batch 6 requires 786,844 scheduler-reserved tokens and is admissible.
- Batch 7 requires 917,995 tokens and is not admissible.
- Batch 8 requires 1,049,146 tokens and is not admissible.
- The FP8 O optimization reduces transport volume, not KV-cache size.
- Batch 8 requires a shorter context, additional attention capacity such as ATP2, or a smaller KV representation. Raising the memory ratio alone cannot supply the required 25 percent.

## Measurement and acceptance contract

Performance runs must prove:

- Exact target shape and placement recorded in `afd-result.json`.
- One exact target-batch warmup followed by 15 measured decode replays.
- Representative attention and model Nsight reports with identical step mapping. Legacy audits inspect rank 1; FMHA-only audits inspect representative attention rank 1 and model rank 5.
- Final latency is the arithmetic mean of the per-step maximum complete CUDA Graph span across profiled attention and model roles.
- At least 10 retained steps, at most five outliers, dominant range at most 10 percent, and maximum-to-median difference at most 10 percent.
- `SUCCESS`, strict metric output, expected report count, and no runtime exception.
- Official unchanged-scorer alignment before a candidate becomes an accepted correctness checkpoint.
- Trace proof that heavy model overlap is limited to causally independent FFN
  lanes with distinct workspaces and synchronization state; dependent-batch or
  shared-workspace overlap remains forbidden.

Use the coordinator median only as a same-field E2E diagnostic. Never compare it with an older metric that used a different latency basis.

## Grouped fan-in and publication invariants

One logical FFN worker may serve many attention sources, but it must still run one model round per microbatch, not one full graph per source.

Retained requirements:

- Group all source commands for a model step.
- Concatenate each microbatch once.
- Publish Q/K/V to all destination edges under the existing publication phase.
- Wait for all required O-ready words once.
- Run exactly two FFN rounds per layer for MB2.
- Preserve one O publication per source edge.
- Keep source tensors alive until the existing completion event is synchronized.
- Bound large publication grids while retaining grid-stride payload coverage.
- Do not add cross-GPU barriers, acknowledgements, source-switch drains, or extra readiness phases.

Representative grouped-fan-in gates:

- Jobs `6459665` and `6459664`: A1/A2 grouped model steps at 26.190541 and 27.850899 ms with correct 564-node attention and 3,676-node model graphs.
- Jobs `6462898` and `6462899`: retained-batch lifetime correction at 26.099038 and 27.619945 ms, 15 of 15, zero outliers.
- Alignment job `6463085`: perfect A2 agreement and exact per-replay expert counts.
- High-fan-in campaign jobs exposed publication and per-source service scaling; those broad campaigns are closed.

## Placement and high-fan-in conclusions

- Exact no-mode A15 control `6263396`: 30.9095675 ms E2E and 30.7524840 ms strict CUDA.
- Best retained FMHA-only A15 diagnostic `6485295`: 31.7143450 ms E2E and 31.6887659 ms strict CUDA. It remains slower than no-mode.
- A8 FMHA-only job `6488053`: 31.2774395 ms E2E and 31.2384243 ms strict CUDA.
- A8 legacy control `6265476`: 30.8973810 ms E2E and 30.7433556 ms strict CUDA.
- The measured placement crossover is A8. The Qwen3 128K/b6/ATP1/FEP4/MB2 adaptive policy selects FMHA-only below ratio 8 and legacy at ratio 8 or above.
- Eight communication SMs help the isolated 45-row high-fan-in MoE shape, but hurt A1 and do not make A15 FMHA-only competitive. Retain 24 communication SMs for low fan-in.
- QKV publication CTA reduction and system-atomic completion both regressed A15. Do not reduce publication parallelism further.

## Closed FFN scheduling directions

The following families are rejected unless a new mechanism changes the causal model:

- Standalone per-phase ready kernels: correct but launch overhead erased the data-kernel savings. Jobs `6473693` and `6474094`.
- One wait kernel per lane handoff: exactly two model streams but 187 added nodes and regression. Job `6475180`.
- Captured stream-memory waits: exactly two streams but slower. Job `6475460`.
- Event-free two-stream scheduling: correct topology but cadence regression. Job `6475991`.
- Stream or graph-node priority variants: increased contention or synchronization. Jobs `6476272`, `6476535`, `6476970`, and `6477252`.
- Startup-ticket and waiter-order variants: two-stream topology but no acceptable bidirectional stagger. Jobs `6477716`, `6478604`, `6479079`, `6479509`, and `6479731`.
- Fused per-round admission and deferred combine variants: reduced graph nodes but still regressed. Job `6480443`.
- Fused input RMSNorm plus QKV quantization: job `6481196` regressed about 1.66 percent.
- Reduced publication grids and system-atomic completion: jobs `6486377` and `6487343` regressed.
- Deferred combine from one batch overlapping heavy work from another violates the governing model even if latency appears better.

Git history preserves the detailed experiment chronology removed from this file.

## Retained engineering lessons

- Prepost readiness, then send payload later; never add a receiver-credit round trip to the critical path.
- Use ready words and local causal clearing instead of payload sentinels or complete-buffer scans.
- Keep communication tails separate from exclusive heavy compute ownership.
- A cooperative K/V-to-Q grid barrier is unnecessary; one final publication boundary is sufficient.
- More measured overlap can mean resource contention, not useful pipelining.
- One fixed post-case sleep wastes allocation time and does not prove exit; use condition-driven process completion.
- Slurm spool paths are transient; use durable source and control paths.
- Keep materialized batch tensors alive until asynchronous consumers finish.
- Never resume a stopped campaign in place after its topology or metric contract changes. Create fresh state.
- Do not use legacy tail-capture AFD rows as target-batch performance evidence.

## Reusable workflow

Reusable remote controls live under:

`scripts/experiments/afd/oci_hsg/`

No active `codex_scripts/` directory exists in this worktree.

For future GPU work:

1. Keep source clean and committed on the remote branch.
2. Submit to `short` with an explicit one-hour limit unless the user changes the policy.
3. Set `FASTAFD_ALLOW_DIRTY_SOURCE=0`.
4. Keep FMHA-only allocations within one NVL72 fabric block using `--segment`.
5. Run the CPU/source contract first.
6. Run a fresh transport smoke only when transport or synchronization changes.
7. Gate on exact A1 performance and trace structure.
8. Run official alignment.
9. Advance to larger ratios only after the A1 gate passes and only on explicit user direction.
10. Copy requested raw Nsight reports into the local workspace without touching local Git.

## Resume checklist

- Read this file before changing the pipeline.
- Confirm remote branch and worktree are clean.
- Confirm no stale GPU jobs or claims exist.
- Preserve O projection on the model or FFN side.
- Preserve attention-side O activation quantization and FP8 O transport.
- Preserve exclusive model compute-unit ownership.
- Treat current FEP4 job `6498084` as the aligned acceptance point.
- Treat FEP8 job `6494759` as the latest scaling profile.
- Do not infer that current ATP1 supports batch 7 or batch 8 at 128K.
- Do not restart A7, A8, A15, or a broad campaign without explicit user direction.

User reaffirmed non-overfitting on this continuation: use the exact case as a
latency microscope, not as a kernel assumption. Future accepted changes must
support arbitrary route IDs and uneven source counts, and a promising normal
kernel-wall win needs a nearby workload non-regression check before promotion.
No tile/SM/threshold sweeps or perfect-pattern-only dispatch. Both rejected
candidates remain isolated; normal accepted code keeps the established paths.
