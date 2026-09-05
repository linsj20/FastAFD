# Qwen3.5 AFD overhead placement

## Goal and constraints

Support `Qwen/Qwen3.5-122B-A10B-FP8` in AFD using
`hao-ai-lab/FastAFD:feat/qwen35-integration` only as a reference. Preserve the
current `fmha_only` branch's two-lane schedule and optimizations. Linear
attention, its convolution state, and its recurrent GDN state must remain on
attention workers in both placements. The canonical split-placement name is
`attn-only`; accept and immediately normalize the old `fmha-only` spelling.
First prove A1:F1/EP4 alignment at the real 128K maximum batch, then prove
A3:F1/EP4 alignment and retain local Nsight Systems reports for both. Use HSG
`short`, one-hour jobs, with 5/15-minute monitoring.

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
and return the existing packed-UE8M0 FP8 payload. All 48 layers remain rounds
in the existing MB2/two-slot wavefront; only the 12 full-attention layers occupy
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

## Validation and active work

- Local Python AST parsing covers all changed Python sources; shell syntax and
  `git diff --check` pass. Local source-contract suite: 22/22 pass.
- Remote pinned-environment focused suite: 63/63 AFD runtime, protocol,
  overlap, metric-validation, and Qwen3.5 ownership tests pass. A regression
  test covers the hybrid MB2 graph schedule publishing normalized linear-layer
  hidden states without entering the full-attention QKV splitter.
- Actual pinned config parses as 48 layers = 36 linear + 12 full, hidden 3072,
  head dim 256, Q/KV heads 32/2, 256 experts, top-k 8, FP8 block 128x128.
  Meta materialization of that exact config gives 64 local experts per EP4
  model worker in FP8 and an attention stage containing 432 GDN tensors, with
  recurrent `A_log` retained as FP32.
- HSG job `6925193` (`short`, one tray, one-hour limit) compiled transport
  specializations for head dimensions 128 and 256 and passed three repeated
  fabric iterations with Qwen3.5 shape 32Q/2KV/256 across two independent
  pairs. Output is `logs/transport-smoke-6925193.out` under the task root.
- The 127,195,718,583-byte pinned model snapshot is downloading into the task
  model cache. The first 8-worker Xet attempt failed a 67 MB allocation on the
  login host; the resumed standard HTTP path uses `HF_HUB_DISABLE_XET=1` and
  two workers and is progressing. Do not repeat the failed mode.

Next: finish the snapshot, reconcile and commit the isolated remote source,
run a minimal real Qwen3.5 startup/alignment probe, discover the exact A1 128K
capacity from post-init page counts, rerun that exact maximum through alignment
and Nsight capture, then run the final A3 alignment/capture and copy both report
sets into the local workspace scratch tree.
