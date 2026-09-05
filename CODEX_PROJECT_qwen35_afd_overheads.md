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

The primary local checkout must remain untouched. The editable source is
`scratch/qwen35_afd_overheads_20260904/source_mirror`. The isolated remote task
root is
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
- The implementation is committed as mirror commit `2d8eb5e` and isolated
  remote commit `80380b3`; the primary checkout remains clean at `3c5315f`.
  Direct-launch dry runs resolve the provisional A1 point to 45 requests per
  attention-DP lane, 180 prompts, four attention plus four EP4 model GPUs on
  two trays, normal routing, and the unchanged strict vLLM scorer. Legacy uses
  its established 23+22 MB2 order; attn-only uses 22+23. This is only the
  planning estimate; the runtime page-count gate remains the placement-specific
  maximum-batch authority.
- The earlier 122B snapshot completed but was superseded by the user's 397B
  selection and remains intact. The pinned 397B snapshot is complete at
  revision `ea5b4f8`: all 107 repository files and 94 tensor shards are present,
  no blob is incomplete or symlink broken, and the index declares
  406,125,181,280 tensor bytes. Six concurrent standard-HTTP workers initially
  accelerated the transfer but triggered Hugging Face 429/queue errors; a
  single-worker `HF_HUB_DISABLE_XET=1` resume completed reliably. Avoid both
  that concurrency and the earlier failed high-worker Xet mode.

Next: complete normal-routing alignment for legacy and attn-only at exact A1
batch 37 and then A3, capture perfect-balanced attn-only A1/A3 Nsight reports,
and copy both report sets into the local workspace scratch tree.
