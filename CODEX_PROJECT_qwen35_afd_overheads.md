# Qwen3.5 AFD overhead placement

## Goal and constraints

Support `Qwen/Qwen3.5-397B-A17B-FP8` in AFD using
`hao-ai-lab/FastAFD:feat/qwen35-integration` only as a reference. Preserve the
current `fmha_only` branch's two-lane schedule and optimizations. Linear
attention, its convolution state, and its recurrent GDN state must remain on
attention workers in both placements. The canonical split-placement name is
`attn-only`; accept and immediately normalize the old `fmha-only` spelling.
First prove normal-routing A1:F1/EP4 alignment at the real 128K maximum batch,
then prove normal-routing A3:F1/EP4 alignment. Run each alignment topology in
both `legacy` and `attn-only`. After correctness is established, capture
perfect-balanced `attn-only` Nsight Systems reports for A1:F1 and A3:F1 and
retain both report sets locally. Use HSG `short`, one-hour jobs, with
5/15-minute monitoring. Keep plans, results, metrics, and traces isolated by
purpose and placement so repeated shape IDs cannot overwrite one another.

The primary local checkout remains untouched while the candidate is under
validation; once both requested traces pass, copy the final validated source
back to that current branch without switching branches. The editable source is
`scratch/qwen35_afd_overheads_20260904/source_mirror`. The isolated remote task root is
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/qwen35_afd_overheads_20260904`;
its `source` checkout is a test/launch copy synchronized from the local mirror.
Do not modify the canonical remote FastAFD checkout.

## Implementation state

Reference head `1a29bc7` was fetched into the mirror as `reference/qwen35`.
The core Qwen3.5 support from reference commit `b4d4082` is resolved against the
current branch without switching it; the current coordinator shutdown, strict
15-step trace, optimized DeepEP completion, and model-loading behavior win the
six merge conflicts. The TRT-LLM page-table ceiling correction from `e0d44c0`
is also applied.

In `attn-only`, model workers retain embeddings/final norm/LM head, every FFN,
and full-attention input norm, QKV/QK-norm/RoPE, output projection, and
post-attention norm. They publish normalized hidden states for linear layers
and Q/K/V plus output gate for full layers. Attention workers own all GDN
weights/state and execute linear layers; they return BF16 linear output. For
full layers they own only KV state and FMHA execution, apply the output gate,
and return the existing packed-UE8M0 FP8 payload. All 60 layers remain rounds
in the existing MB2/two-slot wavefront; only the 15 full-attention layers occupy
KV storage. Qwen3.5 currently fails fast unless attention TP and model TP are
both one, matching A1/A3 with FFN EP4.

The fabric transport now accepts head dimensions divisible by 128 and uses one
UE8M0 scale per 128 values, preserving Qwen3 head-dim-128 behavior and adding
Qwen3.5 head-dim-256. A separate coallocated BF16 O view carries linear-layer
outputs. Decode graph metadata includes mutable GDN state indices. The VLLM
reference command appends `--language-model-only` for Qwen3.5. Result extraction,
campaign validation, launch examples, and new metric records use the canonical
`attn-only` name while retaining the old spelling only as an input/result
compatibility alias.

The sharded vLLM scorer accepts the pinned environment as an absolute prefix.
It then launches that prefix's Python via `-m vllm.entrypoints.cli.main` and
exports its `bin`, `CONDA_PREFIX`, and `VIRTUAL_ENV`, avoiding the nonexistent
home-Conda activation path and the absent `vllm` console script. Alignment-only
runs may disable Nsight; comprehensive postprocessing now runs only when Nsight
was enabled.

## Validation and active work

- Local Python AST parsing covers all changed Python sources; shell syntax and
  `git diff --check` pass. Local source-contract suite: 22/22 pass.
- Remote pinned-environment focused suite: 63/63 AFD runtime, protocol,
  overlap, metric-validation, and Qwen3.5 ownership tests pass. A regression
  test covers the hybrid MB2 graph schedule publishing normalized linear-layer
  hidden states without entering the full-attention QKV splitter.
- The selected 397B FP8 config is pinned at revision `ea5b4f8` and config SHA256
  `9a142fc6...d6e5db8`. It parses as 60 layers = 45 linear + 15 full, hidden
  4096, head dim 256, Q/KV heads 32/2, 512 experts, top-k 10, MoE/shared
  intermediate 1024, and FP8 block 128x128. EP4 therefore owns 128 routed
  experts per model worker.
- The same-rank MegaMoE fused gate/router used by `attn-only` previously fixed
  one token-tile grid to 128 experts. It now scales that grid to 64 eight-expert
  blocks for 512 experts while retaining the same two kernel launches and
  in-kernel perfect-ID specialization. Only the first 16 expert blocks own the
  existing disjoint FP8 hidden quantization slices, preventing duplicate work.
- HSG job `6925193` (`short`, one tray, one-hour limit) compiled transport
  specializations for head dimensions 128 and 256 and passed three repeated
  fabric iterations with Qwen3.5 shape 32Q/2KV/256 across two independent
  pairs. Output is `logs/transport-smoke-6925193.out` under the task root.
- The pinned environment parses the actual 397B config, and targeted meta
  construction of the `attn-only` model stage succeeds with 813 tensors and
  routed expert shape `[128, 2048, 4096]` in FP8. A first diagnostic that kept
  all four full stage state dictionaries live together exhausted login-host
  thread-local memory; the isolated construction with one BLAS thread passed,
  so subsequent stage checks must remain isolated.
- Target-shape router smoke job `6925717` compiled the 512-expert kernel and
  passed its preliminary route-prepared numeric checks, then failed because
  the new deterministic-ID test called `torch.equal` inside CUDA graph capture.
  The assertion is moved to the synchronized eager validation immediately
  before capture; the candidate kernel itself was not implicated.
- Corrected HSG job `6925762` (`short`, one tray, one-hour limit) completed
  `0:0` in 1:24. It exercised hidden 4096, intermediate 1024, 512 experts,
  top-k 10, EP4, FP8 activations/FP4 weights, route-prepared/rank-ready
  dispatch, graph replay, and perfect routing. All preliminary numeric cases
  passed; the exact target case reported 40 active experts, finite output, and
  0.127328 ms for its single timing iteration. The failed parent allocation
  from `6925717` lingered after its step failure and was explicitly canceled
  before the corrected job ran.
- Normal-routing HSG job `6925879` completed `0:0` in 1:49 with the same
  target shape. Its controlled monotonic gate first selected expert IDs
  511--502, proving the fused selector covers the full 512-expert range, then
  the route-prepared FP8xFP4 graph case completed with finite output.
- Normal-routing A1 capacity probes `6927848` (legacy) and `6927850`
  (`attn-only`) independently measured the same conservative allocation
  minimum: 4,970,176 KV tokens per attention-DP worker and exact 128K batch 37.
  The first exact reruns `6927904`/`6927907` exposed that runtime sizing dropped
  the explicit 77,659-page override, allowing a different allocation to create
  78,440 pages and demand batch 38. `build_runtime_sizing` now carries
  `ServerArgs.num_page_override` into each attention worker; a pinned-environment
  targeted check confirms that 77,659 is retained. This is capacity plumbing
  only and does not alter prompts, model computation, or alignment scoring.
- Fixed-capacity reruns `6928019`/`6928022` both passed the gate at exactly
  77,659 pages, 4,970,176 tokens, and batch 37. Legacy then reached prefill but
  its DeepEP group-local softmax rejected 512 experts because the CUDA launcher
  stopped at 256 even though the Python contract already allowed 512; the
  existing warp-per-row kernel is now instantiated for 512. Attn-only exposed
  BF16 `weight_scale_inv` tensors from the 397B checkpoint loading into FP32
  runtime scale parameters; selected weights now convert to the expected
  parameter dtype, with a focused regression test.
- Reruns `6928176`/`6928177` passed both earlier blockers. Legacy reached its
  DeepGEMM expert call, where the older M2N dispatch record lacked the same
  graph-static routing-density hint already carried by the ordinary DeepEP
  dispatcher. M2N now computes/carries `expected_m_per_expert` and derives its
  contiguous-layout alignment from it. Attn-only reached linear-attention Q
  publication, where zero-width K/V tensors exposed `SymbolicSize`'s zero-as-
  unset sentinel. The publisher now derives zero KV edge/width counts from the
  empty KV descriptor without unwrapping those unset symbols.
- A subsequent normal-routing A1 pair advanced both modes further. Legacy job
  `6928449` passed its exact-capacity gate and captured the batch-19 decode
  graph, served all 148 long-prefill requests, and wrote its sample bundle.
  Its unchanged vLLM scorer could not start one baseline shard because the
  inherited long run-directory temp path made vLLM's Unix-domain RPC address
  exceed `sockaddr_un`'s 107-byte limit. The shard launcher now explicitly uses
  a short relative RPC base under the ignored source `cache/` directory; this
  changes only baseline process plumbing, not scoring. Attn-only job `6928456` passed the
  corrected 397B weight loading and zero-KV publication but failed during
  decode-graph capture because the gated RMSNorm's lazy `torch.compile` path
  attempted an Inductor code-cache load and CUDA synchronization while its
  stream was being captured. The fused gate is now an explicit Triton kernel;
  its exact decode bucket is exercised by the existing attention prewarm, so
  capture no longer depends on lazy compiler activity. The prior eager path is
  retained only as the explicit A/B environment selection.
- Legacy A1's completed sample from job `6928449` passed the unchanged scorer
  in score-only job `6930405`: 148 prompts, 2,516 generated tokens, top-1
  0.9996025, top-10/top-100 1.0, mean rank 1.0003975, and maximum rank 2.
  This is the first strict normal-routing 397B alignment pass at batch 37.
  The score-only wrapper now re-enters the container through an explicit script
  path and uses 8,192-token prompt chunks, concurrency four, and memory ratio
  0.82. These settings avoid the baseline's Unix-socket limit and its full-vocab
  prompt-logprob OOM without changing the vLLM model, scorer, compared tokens,
  or top-10=1.0 requirement.
- Attn-only A1 sample job `6929543` captured every attention/model graph and
  produced all 148x17 tokens. Its post-sample exit 3 was only a missing optional
  coordinator CPU trace after graceful-shutdown timeout. Strict score job
  `6930412` failed alignment: top-1 0.15938, top-10 0.54412, top-100 0.72655,
  mean rank 7,802.58, maximum rank 248,310. Per-position analysis shows the
  mismatch at generated token zero, so this is a prefill-path discrepancy.
- The discrepancy is explained by a precision contract mismatch rather than
  routing mode: attn-only selected same-rank MegaMoE, whose adapter always
  converted the FP8 checkpoint experts to FP4, while the unchanged vLLM
  reference used the checkpoint FP8 weights. The MegaMoE extension and existing
  weight utilities already support FP8xFP8. The attn-only adapter and launch
  contracts now accept explicit `fp8|fp4`: FP8 uses the existing per-32 UE8M0
  conversion plus `fp8_fp8_mega_moe`; FP4 retains the optimized existing path
  and remains the default. Alignment plans select FP8 with normal routing;
  perfect-balanced performance trace plans remain FP4.
- Normal-routing FP8 target-shape smoke job `6930893` completed `0:0` in 2:58
  on one four-GPU tray. It exercised 512 experts, top-k 10, hidden 4,096,
  intermediate 1,024, route-prepared/rank-ready graph execution, and the
  FP8xFP8 kernel. All row/bucket numeric cases passed with cosine
  0.999747--0.999896; the exact target case produced finite output with 14
  active experts. This validates the selectable FP8 kernel and weight layout
  before another full 397B run.
- Full FP8 attn-only A1 sample job `6931009` captured its graphs and produced
  all 148x17 tokens at exact batch 37; its exit 3 again followed sample
  publication and reflects only the absent optional coordinator CPU trace.
  Unchanged strict score job `6931595` failed at top-1 0.15382, top-10
  0.50437, and top-100 0.68005. This is close to the FP4 failure and disproves
  expert precision as the primary cause. FP4 and FP8 attn-only streams agree
  on 22.06% of generated IDs but neither agrees with legacy at token zero.
  The reusable same-rank smoke is therefore parameterized to reference-check
  the actual batch-19/512-expert/top-k-10 production specialization next;
  its earlier target-shape check only established routing reach and finiteness.
- Production-shape numeric job `6931995` isolated the failure: the ordinary
  batch-19/512-expert/top-k-10 FP8 path matches its explicit distributed
  reference at cosine 0.999914, but the route-prepared/rank-ready path falls to
  0.385559--0.653640. In rank-ready initialization, one warp indexed local
  experts only by its 32-lane `lane_idx`; this covered Qwen3's 32 experts per
  EP4 rank but only the first 32 of Qwen3.5's 128. The warp now strides by 32
  across every local expert while retaining the same shared-memory layout,
  kernel specialization, and launch count.
- Corrected production-shape job `6932099` completed `0:0` in 2:39. Ordinary
  and route-prepared/rank-ready batch-19/512-expert/top-k-10 FP8 paths now both
  match the explicit reference at cosine 0.999913871 and maximum absolute
  error 22.537384; the exact graph case exercised 103 active experts and
  finite output. This is the required kernel-level proof before repeating A1.
- Corrected full-model FP8 MegaMoE sample job `6932154` produced all 148x17
  tokens; 137/148 sequences exactly match the passing legacy sample and first-
  token agreement improved from zero to 95.95%. Strict score job `6932885`
  nevertheless fails the unchanged gate at top-1 0.977742, top-10 0.994436,
  top-100 0.998808, mean rank 2.9523, and maximum rank 3,170. The residual
  difference is consistent with MegaMoE's required checkpoint block-FP8 to
  per-32-UE8M0 weight requantization. Do not weaken the top-10=1.0 criterion.
  Strict attn-only alignment therefore uses normal-routing DeepEP with the
  checkpoint FP8 arithmetic, while requested perfect-balanced Nsight traces
  retain same-rank FP4 MegaMoE.
- Normal-routing DeepEP attn-only sample job `6933383` produced all 148x17
  tokens and improved exact sequence agreement with the passing legacy sample
  to 139/148. Its unchanged strict scorer job `6934407` still failed: top-1
  0.9888712, top-10 0.9968203, top-100 0.9992051, mean rank 1.28537, and
  maximum rank 315 across all 2,516 generated tokens. This leaves eight top-10
  misses and proves the residual is backend-independent; the top-10=1.0 gate
  remains unchanged.
- The passing legacy sample predates the explicit Qwen3.5 gated-RMSNorm Triton
  kernel. The retained Inductor artifact from failed graph-capture job
  `6928456` shows its old fused kernel used a two-row reduction tile with two
  warps and formed the SiLU gate separately. The replacement used one row and
  four warps, changing FP32 reduction and multiply order in all 45 linear
  layers. The explicit graph-safe kernel now mirrors the generated production
  specialization. The existing gate graph smoke compares it bitwise against
  `torch.compile` at the actual decode shape `[19,64,128]` before and after
  CUDA-graph capture. HSG job `6934956` completed `0:0` in 1:26 and passed that
  exact check, including graph replay. This validates the arithmetic-preserving
  graph-safe replacement before the current-source full-model reruns.
- Same-source normal-routing DeepEP A1 jobs `6935143` (legacy) and `6935160`
  (`attn-only`) both completed all 148x17 tokens at 128K before their outer job
  statuses became non-successful. Legacy made its 88 MB sample durable at 58
  minutes, then failed only because `NSYS_CUDA_GRAPH_TRACE=none` still left
  Nsight enabled and the postprocessor demanded a deliberately absent graph
  trace. Attn-only made its sample, result, and `SUCCESS` marker durable by 59
  minutes, then crossed the one-hour Slurm teardown boundary. The new legacy
  sample matches 146/148 sequences from the previously passing legacy sample;
  the new attn-only sample matches 139/148 sequences from both legacy samples.
  Do not repeat these expensive 128K inference runs.
- The user selected 8K with a small batch for the remaining alignment proof.
  Keep the 397B model, normal routing, all generated tokens, unchanged vLLM
  scorer, and top-10=1.0 gate. `NSYS=0` must be a real alignment-only path:
  sample shape/token count remain validated, while coordinator/worker trace
  artifacts, trace-window auditing, and performance fields are not required.
  The general launcher now terminates after the alignment sample is durable and
  records null/empty trace metrics rather than collecting hundreds of MB of
  trace data. Perfect-balanced `NSYS=1` runs retain the complete strict trace
  contract unchanged.
- Normal-routing, trace-free 397B A1 inference jobs `6936281` (legacy) and
  `6936282` (`attn-only`) completed `0:0` in 9:11 and 10:05 respectively at
  8K, batch four per attention-DP lane, MB2, and 17 generated tokens. Each
  wrote a validated 16-prompt sample and `SUCCESS`; 11/16 complete sequences,
  14/16 first tokens, and 204/272 generated token IDs agree between modes.
  The one-node score-only wrapper now accepts `SCORE_NODES=1|2`, retaining the
  same vLLM DP4/EP4 backend and unchanged scorer while avoiding a second tray
  for these 16-prompt samples.
- Attn-only score job `6936518` completed baseline evaluation in 9:20 and
  correctly failed the unchanged top-10=1.0 gate: top-1 0.9080882, top-10
  0.9926471, top-100 1.0, mean rank 1.46691, and maximum rank 50 over all 272
  generated tokens. The only top-10 misses are prompt 1 position 7 (rank 19)
  and prompt 14 position 4 (rank 50). Prompt 1 is identical to legacy through
  position 6 and diverges exactly at the miss; prompt 14 first chooses a
  plausible non-top-1 token and the miss occurs after that alternate prefix.
  No further full-model job was launched before isolating the remaining
  arithmetic asymmetry.
- Legacy feeds the gated full-attention result directly into its BF16
  O-projection, whereas Qwen3.5 attn-only was quantizing that result to FP8 for
  inter-worker transport. Linear-attention output already uses a BF16 arena.
  The current candidate uses that existing BF16 arena for both full and linear
  layers in hybrid Qwen3.5, then runs the model-side BF16 O-projection; the
  established FP8 O transport remains unchanged for non-hybrid models. Four
  focused remote unit tests cover BF16 full-attention publication, model-side
  O-projection finish, linear-layer behavior, and the unchanged FP8 MB2 path;
  the Qwen3.5 placement structure test also passes. The reusable fabric smoke
  now selects BF16 or FP8 O payloads. BF16 job `6936921` completed `0:0` in
  3:06 and verified nine bitwise-exact publication/ready-counter/turn-reuse
  cycles across two attention/model rank pairs. This is the final cheap
  transport proof before repeating 8K attn-only inference.
- First BF16 full-model retry `6937050` failed fast during decode-graph capture
  in 9:16, before serving any prompt. The new O-projection correctly rejected
  a K-width mismatch: Qwen3.5 GDN output is the 4,096-wide hidden size, but
  full attention has 32 heads x 256 = 8,192 values. The initial candidate had
  reused a GDN-sized arena and therefore truncated full-attention output. The
  shared BF16 allocation now uses the larger width and distinct descriptors
  and receive views preserve 4,096-wide linear versus 8,192-wide full payloads.
  Five focused runtime tests pass, including both widths and unchanged FP8 MB2.
  Exact 32Q/2KV/head-dim-256 fabric job `6937235` completed `0:0` in 3:32 and
  passed nine bitwise-exact BF16 publication/ready/turn-reuse cycles across two
  attention/model rank pairs. Do not count `6937050` as an inference attempt:
  it produced no sample and consumed no alignment scorer.
- The trace-free alignment launcher is committed as mirror commit `b677252`
  and isolated remote commit `f9dc7e1`; the primary checkout remains clean at
  `3c5315f`. The validated small A1 point deliberately overrides the available
  capacity to four requests per attention-DP lane (16 total prompts), using
  2+2 MB2 on four attention plus four EP4 model GPUs across two trays.
- The earlier 122B snapshot completed but was superseded by the user's 397B
  selection and remains intact. The pinned 397B snapshot is complete at
  revision `ea5b4f8`: all 107 repository files and 94 tensor shards are present,
  no blob is incomplete or symlink broken, and the index declares
  406,125,181,280 tensor bytes. Six concurrent standard-HTTP workers initially
  accelerated the transfer but triggered Hugging Face 429/queue errors; a
  single-worker `HF_HUB_DISABLE_XET=1` resume completed reliably. Avoid both
  that concurrency and the earlier failed high-worker Xet mode.
- Corrected normal-routing 397B attn-only A1 job `6937369` completed `0:0` in
  11:44 at 8K, batch four per attention-DP lane, MB2, and 17 generated tokens.
  It produced a validated 16-prompt sample and `SUCCESS`. All 272 generated
  token IDs are exactly equal to the durable same-source legacy A1 sample from
  job `6936281`; 16/16 complete sequences and first tokens match. Unchanged
  score job `6937561` completed `0:0` in 9:58: top-1/top-10/top-100 are all
  1.0, mean and maximum rank are 1.0, and all 272 generated tokens were scored.
  This is the accepted small normal-routing A1 attn-only alignment proof.
- The successful reference branch retains Qwen3.5's legacy algorithm but has
  no split attn-only placement. Current GDN/convolution/state math remains
  inherited from that implementation. Full attention preserves its ordering:
  attention, output-gate multiply, then BF16 O projection. Attn-only splits
  that exact sequence only at the worker boundary: the attention worker runs
  attention and gate, transports BF16, and the model worker runs O projection.
  The exact legacy/attn-only generated stream is the end-to-end confirmation.
- Final normal-routing 397B attn-only A3:F1/EP4 inference job `6937742`
  completed `0:0` in 11:45 at 8K, batch four per each of 12 attention-DP
  lanes, MB2, and 17 generated tokens. It produced a validated 48-prompt,
  816-token sample and `SUCCESS`. Unchanged scorer job `6937912` completed
  `0:0` in 10:17: top-10/top-100 are 1.0, top-1 is 0.9791667, average rank is
  1.0318627, and maximum rank is 5. A1 and A3 attn-only therefore both pass
  the unchanged all-generated-token top-10=1.0 alignment standard.
- The first real-prefill 128K A1 trace attempt, job `6938071`, was canceled
  before inference after the user rejected spending the trace budget on slow
  128K prefill. No trace result from that job is accepted. Profiling now has an
  explicit opt-in perf mode which bypasses the prefill scheduler, atomically
  allocates the exact logical 128K request/page state, starts every request at
  the final resident-context token, and fills MHA KV plus convolution/recurrent
  GDN state with random GPU values before graph warmup/capture. Normal serving
  and both alignment paths are unchanged; perf mode is rejected unless Nsight,
  uniform context, `attn-only`, and same-rank MegaMoE are selected.
- Perf mode always forces `MINISGL_AFD_PERFECT_ROUTING=1` through the campaign,
  Slurm, container, coordinator, and Ray worker environment. Perfect routing
  retains the normal fused gate GEMM, softmax, top-k comparisons/weights, route
  preparation, dispatch, expert, and combine work. It changes only final IDs
  through the existing compile-time kernel specialization, with no extra
  kernel, graph node, synchronization, copy, or device runtime branch. The
  397B 512-expert mapping now uses 32-bit multiply/add/power-of-two-mask
  arithmetic instead of the prior 64-bit remainder; unsupported expert counts
  and route ranges fail before execution so the routing mechanism itself does
  not add a division to the measured path.
- Remote focused perf/protocol/perfect-routing contracts pass 19/19, including
  duplicate-UID and KV-capacity atomicity; shell syntax and `git diff --check`
  pass. Zero-GPU A1 and A3 launcher rehearsals independently report
  `afd_perf_mode=1 perfect_routing=1` even with an incoming explicit routing
  value of zero. Both retain batch 37 per attention lane, MB2 `18+19`, exact
  128K logical context, capacity 4,970,176, and one warmup plus exactly 15
  measured decode launches.

Next: rerun the focused contracts including duplicate-UID atomicity, then run
zero-GPU source/model validation. Submit only the FP4 perf-mode A1 trace first;
monitor manually at 5/15-minute intervals and treat a 50-minute execution as
suspicious. Submit A3 only after A1 completes the exact 15-step trace contract.
Copy both report sets into the local workspace scratch tree, then propagate the
validated candidate source to the untouched primary current branch.
