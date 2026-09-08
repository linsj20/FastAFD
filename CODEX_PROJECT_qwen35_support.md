# Qwen3.5 AFD support

## Remote workspace consolidation (2026-09-05 PDT)

The primary local `fmha_only` checkout remains the authoritative development
source. The sole accepted remote mirror for future execution is now
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD/local-mirror`.
Immediately after the remote move, its clean tree object
`106a8cb4ecdc16a69058fe6249fbba9249c355cd` exactly matched the local committed
tree. The project-memory relocation recorded here is synchronized into the
amended accepted checkpoint in both checkouts.

Qwen3 and Qwen3.5 experiment evidence is consolidated without deletion under
`FastAFD/artifacts/`. The preserved roots are
`qwen3_ffn_overheads_20260820`, `qwen3_ffn_overheads_20260828`,
`qwen3_ffn_overheads_20260831`, and `qwen35_afd_overheads_20260904`.
The former remote `scratch/qwen3_ffn_overheads_20260820` tree is preserved as
`FastAFD/artifacts/qwen3_ffn_overheads_20260820/auxiliary_scratch`.

All 105 files from `fastafd_reproduce/archives` were moved intact to
`FastAFD/artifacts/archives`, including compressed payloads, SHA files,
manifests, summaries, case indexes, metadata, and logs. In particular, the
perfect-routing archive and its three verification sidecars remain together at
`FastAFD/artifacts/archives/completed_sweeps/afd_128k_perfect_routing_e00afda_20260901.tar.zst*`.
The old Qwen3/Qwen3.5 task-root paths and old archive path no longer exist.
No result, archive, cache, log, or source snapshot was deleted during this
consolidation.

## Previous workspace consolidation and canonical AFD launcher (2026-09-04 PDT)

Local `scratch/` is now visualization-only: 25 standalone HTML/PNG files,
3.2 MiB total, organized under `scratch/visualizations/` by study. All local
result trees, NSYS reports, source snapshots, scripts, docs, caches, temporary
review directories, `.DS_Store`, and custom hidden directories were removed.
Older memory entries retain their historical paths as provenance; they are not
claims that those deleted scratch files still exist. Current visualization
paths start at `scratch/visualizations/`.

The complete perfect-routing sweep was archived remotely as
`FastAFD/artifacts/archives/completed_sweeps/afd_128k_perfect_routing_e00afda_20260901.tar.zst`.
It contains exactly 1,599 result payloads / 1,473,202,820 uncompressed bytes:
1,512 `.nsys-rep`, 42 `afd-result.json`, 42 metric JSON, and three audit/
comparison data files. No scripts, docs, logs, or caches are members. Full
`zstd -t`, exact member audit, external 1,599-row SHA manifest, and archive
SHA-256 `3dc1842ba90410dc921103450ac45ad5eb1cbe6e2d6b4d13cbc6322350f52089`
passed. The uncompressed sweep and completed archive staging were then removed;
the `.manifest.tsv`, `.sha256`, and `.summary.json` sidecars remain beside the
607 MiB archive.

The older remote archive workspace is also consolidated. CPU data-mover job
`6919189` completed `0:0`, deleted the last of 3,623 confirmed source targets,
and revalidated all 42 protected sweep-14 cases. The canonical 14-archive set
remains under `archives/sweeps_20260830/` with 5,219-row case and 15-row sweep
indexes; its former `.work` is now visible `metadata/`, logs are under `logs/`,
and the duplicate `.tools` scripts and empty rebuild/staging roots were removed.
Job `6919354` then removed the superseded unfiltered rollback set: six payloads
totaling 520,203,971,236 bytes, including the abandoned 9,763,782,656-byte
`.partial`. The archive tree now has no custom hidden directories.

`scripts/experiments/afd/oci_hsg/run_afd.sh` is now the only AFD launcher in
main source. It absorbs the removed `run_afd_reproduce.sh` and remote
`codex_scripts/reproduce_oci_hsg/run_afd.sh`: Qwen3 pmap/comprehensive and
MiniMax 8K/16K published layouts, uniform/irregular prompts, explicit topology,
legacy/FMHA/adaptive placement, DeepEP/MegaMoE/MegaMoE-M2N, production/perfect
routing, direct/bundle execution, and strict NSYS extraction. Legacy pmap
defaults to `megamoe_m2n`, legacy comprehensive to `deepep`, and FMHA-only to
same-rank `megamoe`; incompatible combinations fail rather than fall back.
Persistent reproduction/source/image/venv paths are required, eliminating the
stale home-image default. The canonical launcher and M2N fix match locally and
remotely; historical worktree copies are not launch sources. The remote was
physically consolidated
on 2026-09-04: the exact used `e00afda8af6e9fc6ad240ec0bd776b960066a53c`
checkout, plus the validated M2N template fix and merged launcher, was promoted
to `users/shengjiel/FastAFD/local-mirror`. All 81 linked experiment worktrees, their
common Git repository, one standalone FMHA checkout, and the superseded former
main checkout were removed after verification; the experiment root now has
zero remaining Git roots. Nonempty `scratch/` trees were moved first, without
compression, to visible
`fastafd_reproduce/legacy_workspace_scratch_20260904/`. The temporary staging
and rollback paths were removed. Standard `.git` in the sole canonical checkout
is the only hidden repository directory intentionally retained.

Real merged-launcher validation is under remote visible task root
`run_afd_canonical_validation_20260904`. Qwen job `6918707` correctly exposed
that a pmap `deepep` default was invalid on source `e00afda`: it failed before
measurement with `DeepEPM2NDispatchOutput` lacking `expected_m_per_expert`.
No fallback was attempted; the backend policy was corrected explicitly.
Replacement Qwen M2N job `6918910` then failed before inference during JIT
warmup, while pending MiniMax job `6918902` was canceled before allocation to
avoid repeating the failure. CPU-only NVCC job `6919340` reproduced the exact
cause: scheduler additions had shifted `layout::WorkspaceM2N` into the new
boolean `kRankReadyRecvCounts` template slot. The M2N instantiation now passes
explicit `false` values for both scheduler booleans; CPU recheck `6919396`
compiled the same generated Qwen kernel to a 150 KiB cubin in 1:29. A focused
source contract guards the template argument order. Real Qwen replacement job
`6919438` then completed inference and made the 2,688-sample output, coordinator
trace, capture marker, and all 32 Nsight reports durable, but failed in outer
postprocessing because `run_afd.sh` was replaced in that worktree while Bash
was still reading it. Reusing the stable launcher's exact embedded postprocessor
against those artifacts passed: 17 tokens/sample, warmup step 1536 plus 15
measured steps, 16 traced steps, 33.369306-ms median coordinator interval, and
2517.283398103 tokens/s/active GPU (-0.028459% from the published reference).
This was an operational live-file replacement failure, not a kernel or teardown
hang. Immutable-file rerun `6919706` then completed `0:0` end to end in 12:52
with the strict two-file source manifest: 2,688 samples, 17 tokens/sample,
warmup 1536 plus 15 measured/16 traced steps, 32 Nsight reports, a
33.403805-ms median coordinator interval, and 2514.683581706
tokens/s/active GPU (-0.131708% from reference). MiniMax job `6919520` remained
pending and was canceled before allocation at 16:49 PDT so its source could be
consolidated safely. The identical published 8K/b72/17:1 real case was
resubmitted as job `6920969` from the sole canonical checkout with an 18-node,
two-hour request. It was pending on priority at 17:02 PDT. The exact source
head, tracked status, and hashes of all existing tracked changes at submission
are under `run_afd_canonical_validation_20260904/control/`. This submission
uses explicit dirty-source mode because the removed duplicate launcher is a
tracked deletion, which the current two-column SHA manifest cannot represent.
Do not edit or resync the remote canonical checkout until `6920969` is terminal.

After the Qwen evidence and result SHA were recorded, five superseded
validation directories and the accepted Qwen payload were deleted. This
reduced the temporary validation root from 15 GiB to 95 MiB. Only the shared
prompt cache and `6920969`'s MiniMax run directory remain; remove that task root
after the MiniMax result is recorded. The deleted validation payloads were
temporary and are not recoverable; the completed 42-case sweep archive was not
touched.

The earlier apparent hang is now documented in the OCI-HSG README. In 17
audited cases, capture/sample/coordinator artifacts were already durable and
only Ray/process-tree teardown consumed another 210--214 seconds. The current
runner terminates its allocation-scoped driver after those three durability
gates; status 137 plus `post-capture-driver-termination.txt` is expected cleanup,
not failure. An NSYS export is progressing while report/stream sizes or progress
records advance; a ZMQ port collision is a separate exact-case retry condition.

## Goal and constraints

Support `Qwen/Qwen3.5-397B-A17B-FP8` in AFD using
`hao-ai-lab/FastAFD:feat/qwen35-integration` only as a reference. Preserve the
current `fmha_only` branch's two-lane schedule and optimizations. Linear
attention, its convolution state, and its recurrent GDN state must remain on
attention workers in both placements. The canonical split-placement name is
`attn-only`; accept and immediately normalize the old `fmha-only` spelling.
Alignment uses normal routing and the unchanged all-generated-token scorer with
top-10 exactly 1.0. Legacy A1 at the real 128K maximum batch established the
accepted baseline; the user then directed work to focus on attn-only and
explicitly allowed 8K/small-batch A1 and A3 alignment. After correctness,
capture perfect-balanced `attn-only` Nsight Systems reports for A1:F1 and A3:F1
using direct-decode perf mode with random resident 128K state, and retain both
report sets locally. Use HSG `short`, one-hour jobs, with 5/15-minute
monitoring. Keep plans, results, metrics, and traces isolated by purpose and
placement so repeated shape IDs cannot overwrite one another.

The validated source has been propagated to the primary local `fmha_only`
checkout without switching branches; that checkout is now the authoritative
source. The workspace reference copy remains under
`scratch/qwen35_afd_overheads_20260904/source_mirror`. The isolated remote
execution checkout is now `FastAFD/local-mirror`, synchronized from the primary
local checkout, and its associated task evidence is under
`FastAFD/artifacts/qwen35_afd_overheads_20260904`. Do not develop directly in
the remote checkout.

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
and return BF16 O so the model worker can run the gated output projection in
the original algorithmic order. The legacy non-hybrid transport remains the
existing packed-UE8M0 FP8 path. All 60 layers remain rounds
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
- One-tray router smoke `6938500` compiled the modified CUDA extension, then
  failed before its exact case because the submission over-reduced an unrelated
  numeric helper to hidden size 512 although fused quantization requires a
  multiple of 1,024. Corrected job `6938580` used the exact 397B router shape
  (hidden 4,096, intermediate 1,024, 512 experts, top-k 10, EP4), perfect
  routing, FP4 expert weights, and prepared/rank-ready execution. It completed
  `0:0` in 1:21: both compact numeric controls passed at cosine
  0.999939--0.999986, and the exact batch-19 case produced finite output with
  all 128 local experts active. The CUDA/JIT and balanced-ID paths are therefore
  validated before any multi-node full-model trace.
- Perf-mode A1 trace job `6938654` ran on two trays/eight GPUs and completed
  the GPU workload in 9:49. It allocated the exact 77,659-page capacity,
  randomized 76,342,886,400 KV elements plus GDN state on every attention
  rank, bypassed prefill, and produced all eight nonempty Nsight reports plus
  the coordinator trace/sample. The exact trace contract passed: warmup step 1
  and measured steps 2--16. The outer job exited 1 only because its external
  plan used an obsolete `-fp4` case-ID suffix/status label; no GPU rerun was
  made. The plan was corrected to the launcher's canonical case ID and required
  extractor status, and the unchanged postprocessor was recovered on the login
  CPU using the Nsight CLI extracted from the pinned container. A1 retains all
  15 samples with zero outliers: mean/median/max critical spans are
  41.409710/41.398816/41.740449 ms, dominant range 1.7198%, maximum/median gap
  0.8252%, and throughput 446.755124 tokens/s per active GPU. This is an
  accepted trace despite the metadata-only Slurm exit.
- Perf-mode A3 trace job `6938898` then ran on four trays/16 GPUs and completed
  cleanly `0:0` in 9:04 with `SUCCESS`. All 12 attention ranks randomized the
  same 76,342,886,400-element KV allocation plus GDN state, and all 16 workers
  emitted nonempty Nsight reports. It captured warmup step 1 plus the required
  steps 2--16. The unchanged stability policy excluded only step 2 and retained
  14/15 samples: mean/median/max critical spans are
  42.706790/42.694548/43.589412 ms, dominant range 3.3230%, maximum/median gap
  2.0960%, one outlier, and throughput 649.779572 tokens/s per active GPU. Both
  A1 and A3 report directories and strict metric JSONs are copied into the
  local task scratch tree.

Completion audit: the primary checkout, workspace mirror, and isolated HSG
source contain the same runtime implementation. Remote focused tests pass
65/65 across Qwen3.5 ownership, FMHA protocol/runtime, perf mode, perfect
routing, transport, and trace-window contracts. Two stale test fixtures were
updated to initialize Qwen3.5 fields when bypassing the runtime constructor,
and the transport text contract now recognizes hybrid BF16 O plus zero-KV
linear-attention layers; production code and the alignment standard did not
change. Local shell syntax and `git diff --check` pass. The local primary
checkout remains the continuing main version.

## 2026-09-05 A3 perfect-routing alignment sanity check

- Latest accepted remote `local-mirror` source (`dbf66891`, tree
  `729bf903`) ran the 397B Qwen3.5 attn-only A3:F1/EP4 configuration at 8K,
  attention DP12, batch four per attention lane, MB2, and 17 generated tokens,
  with perfect routing and trace/perf mode disabled. Inference job `6940773`
  completed `0:0` in 12:47 on four trays and produced the expected 48-prompt,
  816-token sample plus durable `SUCCESS`; every FFN lane reported the
  512-expert FP8/FP4 MegaMoE perfect-routing path, and logs contain no runtime,
  CUDA, or OOM error.
- The unchanged vLLM teacher-forced scorer and all-generated-token
  `top10_hit_rate == 1.0` gate were retained. Initial scorer job `6940955`
  stopped before model loading because it was given the inference venv without
  `deep_ep`; it produced no alignment result. Replacement job `6941045` used
  the preserved vLLM-EP environment, served all requests, wrote
  `vllm-score-vllm-ep/alignment.json`, and exited 1 solely because the strict
  gate correctly rejected top-10 `0.9509803922`. Across 816 tokens, top-1 is
  `0.8210784314`, top-100 is `0.9901960784`, average/max vLLM ranks are
  `12.2463235294`/`5790`, mean vLLM log probability is `-0.9208419023`, and
  perplexity is `2.5114038567`.
- This diagnostic therefore **does not pass alignment**. That is expected when
  comparing deterministic balanced expert IDs against the normal-routing vLLM
  reference: perfect routing intentionally changes model math. Relative to the
  prior passing normal-routing A3 sample, 30/48 complete generated sequences,
  35/48 first tokens, and 529/816 token IDs remain identical. Normal-routing A3
  remains the correctness/alignment result; perfect routing remains a tracing
  and performance control, not an alignment configuration.
- Artifacts are under
  `FastAFD/artifacts/qwen35_support/a3_perfect_balance_alignment_20260905/`;
  the accepted inference result is the
  `afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260905_171909_manual_na/`
  subdirectory.

## 2026-09-05 repeated A3 normal-routing sanity run

- Repeated the latest accepted 397B Qwen3.5 attn-only A3:F1/EP4 case with the
  same 8K, attention-DP12, batch-four-per-lane, MB2, and 17-token contract,
  explicitly setting perfect routing, perf mode, and Nsight off. Exact dry-run
  and login-host validation passed before submission. Job `6941987` completed
  `0:0` in 11:41 on four trays with 48 samples, durable `SUCCESS`, and no
  runtime, CUDA, or OOM error. The result records `attn-only`, same-rank
  MegaMoE FP8/FP4, and `perfect_routing=false`.
- Per user direction, no new vLLM GPU scorer was launched because the prior A3
  vLLM results are preserved. The new prompts exactly equal the previously
  scored A3 prompts, but generation is not bitwise repeatable: 42/48 complete
  sequences and 759/816 token IDs equal the prior normal-routing A3 sample.
  Thus the prior vLLM report directly covers the 42 identical sequences (714
  tokens), all of which passed top-10, but cannot validly score the six changed
  continuations. The unchanged scorer teacher-forces each AFD-generated prefix;
  after a prefix diverges, old prompt-logprob results do not represent the new
  path. This run is accepted as a successful normal-routing inference/repeat
  sanity check, not as a new complete alignment verdict. The earlier fully
  scored normal-routing A3 result remains the alignment proof.
- Artifacts are under
  `FastAFD/artifacts/qwen35_support/a3_normal_alignment_repeat_20260905/`;
  the result directory is
  `afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260905_180518_manual_na/`.

## 2026-09-05 MegaMoE W4 adoption for both placements

- The next acceptance target is Qwen3.5 A3:F1/EP4 with MegaMoE W4 in both
  placements: normal-routing 8K alignment first, followed only after both
  unchanged top-10 gates pass by maximum-batch 128K perf-mode Nsight capture.
- The adjacent preconverted checkpoint is
  `models/Qwen3.5-397B-A17B-FP8-MXFP4-MoE/snapshots/ea5b4f81096f3901c91dea97f81324302495781d`.
  Its manifest reports 92,160 routed projection weights as packed MXFP4 E2M1
  `uint8` with group-32 UE8M0 `uint8` scales. Header-only inspection confirmed
  Qwen3.5 shapes: gate/up weights `[1024,2048]`, scales `[1024,128]`, down
  weights `[4096,512]`, and scales `[4096,32]`.
- Work in progress adds one shared manifest/index-driven loader which reads
  only each rank's `local_expert_offset` shard, reconstructs exponent-only
  scale floats, and reuses the existing DeepGEMM/MegaMoE layout transforms.
  Same-rank `megamoe` selects FP8xFP4 directly; split `megamoe_m2n` passes the
  W4 selector through its per-layer, dual-lane, and persistent all-layer EG
  launches. Existing fused routing, rank-ready dispatch/combine, full-device
  EG, expected-token, and prefetch optimizations are unchanged. Qwen3.5 now
  requires the W4 checkpoint rather than silently converting the FP8 experts.
- The canonical launcher derives the adjacent W4 path, requires FP4 for
  Qwen3.5 MegaMoE, defaults Qwen3.5 legacy to `megamoe_m2n`, and permits perf
  mode for both valid placement/backend pairs. Attn-only perf retains its
  perfect-balanced route substitution; legacy perf retains normal routing
  because split M2N has no equivalent fused substitution. Local AST checks,
  launcher `bash -n`, `git diff --check`, and all 23 overlap source contracts
  pass. The Qwen3.5 structural runtime test cannot run in the local system
  Python because `msgpack` is absent; run it in the pinned remote environment.

## 2026-09-05 MegaMoE W4 A3 alignment and 128K traces

- Focused validation before GPU submission passed 72/72 checks: 23 local
  overlap/source contracts plus 49 remote FMHA protocol, runtime, case-pool,
  and Qwen3.5 structure tests in the pinned environment. The submitted source
  manifest covers the 11 modified implementation, launcher, test, and project
  files. Both placements use the adjacent preconverted FP4 checkpoint; runtime
  logs report `fp8_fp4`, with all four legacy expert groups registering 60/60
  layers. Existing MegaMoE scheduling, fused routing, and communication paths
  remain enabled.
- Legacy normal-routing A3:F1/EP4 inference job `6945990` completed `0:0` in
  8:12 at 8K, attention DP12, batch four per lane, MB2, and 17 output tokens.
  Per user direction, its independent vLLM retry scored all prompts rather than
  merging old results. Stable scorer job `6946912` completed `0:0` in 10:08
  with 48/48 prompts and all 816 generated tokens: top-1 `0.9852941176`,
  top-10/top-100 `1.0`, average/max rank `1.0196078431`/`4`, and perplexity
  `1.0995432551`. This is the accepted legacy alignment result.
- Attn-only normal-routing A3:F1/EP4 inference job `6946121` produced a valid,
  runtime-clean 48-prompt/816-token sample; only its integrated high-concurrency
  vLLM scorer failed from an avoidable scorer OOM. Independent full scorer job
  `6947313` then completed `0:0` in 10:20 using the stable four-sequence/8,192-
  token batching configuration. It scored all 48 prompts and 816 tokens:
  top-1 `0.9852941176`, top-10/top-100 `1.0`, average/max rank
  `1.0171568627`/`3`, and perplexity `1.0985010656`. This is the accepted
  attn-only alignment result. Legacy retry `6946559` and the integrated scorer
  are retained as failed configuration evidence; both used excessive vLLM
  batching and produced no accepted report.
- Only after both top-10 gates passed, legacy normal-routing 128K trace job
  `6947699` ran at the scheduler-admissible maximum batch 37 per each of 12
  attention lanes, MB2, EP4, and 16 active GPUs. It produced 444 samples x 17
  tokens, all 16 nonempty Nsight reports, and the exact warmup step 1 plus
  measured steps 2--16. GPU execution and capture were clean. Slurm exited 1
  only in the post-capture validator: the legacy log parser did not accept the
  newer `step_id ... real ... bs` format and did not exclude unprofiled step
  17. Offline validation with the corrected rule proves every attention worker
  has the exact trace window; no GPU rerun was made. The launcher now accepts
  the optional `real` field and filters worker evidence to the declared trace
  step set. `experiment/trace-validation.json` records this transparent
  recovery alongside the trace.
- Attn-only perfect-routing 128K trace job `6948469` used the same maximum
  batch/capacity and completed cleanly `0:0` in 8:46 with canonical `SUCCESS`.
  Its result records placement `attn-only`, backend `megamoe`, FP4 checkpoint,
  `perfect_routing=true`, 128K context, batch/capacity ceiling 37, and trace
  steps 1--16. All four expert groups report `fp8_fp4`, 512 experts,
  perfect-routing specialization, and the preserved 150-compute-SM/2-reserved-
  SM configuration. All 16 Nsight reports are nonempty and runtime logs contain
  no CUDA, OOM, traceback, or runtime error.
- Complete trace run directories are copied locally under
  `scratch/qwen35_megamoe_w4_both_placements_20260905/trace_legacy/results/`
  and `scratch/qwen35_megamoe_w4_both_placements_20260905/trace_attn_only/results/`.
  Local integrity validation found 16 reports totaling 37,494,361 bytes for
  legacy and 16 totaling 27,464,741 bytes for attn-only; the smallest reports
  are 623,316 and 1,589,560 bytes respectively. The two accepted all-prompt
  alignment JSON reports are also copied under the corresponding local
  `alignment_legacy/` and `alignment_attn_only/` directories.

## 2026-09-06 Qwen3.5 attn-only optimization baseline

- The active target is Qwen3.5 128K A1:F1, attention-DP4, EP4, batch 37 per
  attention lane, MB2 `18+19`, same-rank FP8xFP4 MegaMoE, perf mode, and
  compile-time perfect routing. Optimization scope includes communication,
  cross-kernel scheduling, and kernel fusion while preserving the accepted A3
  8K normal-routing top-10 alignment gate.
- Local launcher syntax, staged-diff integrity, and 23 focused overlap/source
  contracts pass. Exact remote dry-run and login-host validation resolve two
  trays/eight active GPUs, 77,659 pages / 4,970,176 KV tokens per attention
  rank, the exact batch-37 scheduler ceiling, padded graph batch 38, one warmup,
  and 15 measured decode steps.
- Baseline Nsight job `6949312` was submitted at 2026-09-06 00:04 PDT to HSG
  `short` with a one-hour limit. Its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/baseline_a1_128k/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260906_000434_manual_na`.
  The five-minute check found it completing; Slurm recorded `COMPLETED 0:0` in
  8:54 with canonical success, all eight nonempty reports, and no runtime/CUDA/
  OOM error. Offline dual-role extraction accepted all 15 measured steps with
  zero outliers: mean/median/max critical spans are
  41.247700/41.305479/41.364486 ms, dominant range 1.012880%, and throughput
  448.509856 tokens/s per active GPU. The durable metric is
  `analysis/baseline_a1_128k_metric.json` under the task root.
- The pmap target exposed an extractor-only restriction to the comprehensive
  contract. The extractor now accepts both supported launcher contracts while
  retaining all topology, graph, window, and plan checks; its focused unit
  suite passes 6/6. No GPU rerun was needed.
- Nsight graph-kernel accounting shows 29.677 ms/step of non-wait attention-role
  work and 27.896 ms/step of non-wait model-role work, yet the critical graph
  spans 41.248 ms. The ~11.6-ms excess over the larger compute role is pipeline
  dependency bubble. Attention FMHA is 19.609 ms for 37 queries versus
  TensorRT-LLM EP4's 13.257 ms for 25, exactly the same 0.530 ms/query. Model
  MegaMoE is 18.202 ms for MB2 versus TensorRT-LLM's 10.315 ms for one batch;
  the remaining AFD model kernels total 9.694 ms versus TensorRT-LLM's 7.426 ms
  despite 48% more tokens. MB2 is a hard constraint, so optimization must
  retain the accepted `18+19` wavefront while reducing its duplicated kernel
  cost or hiding more of it behind peer-role work.
- One-microbatch control job `6949793` passed dry-run/login validation and was
  submitted at 00:19 PDT, but was canceled before allocation when MB2 was made
  a hard constraint. Slurm records `CANCELLED by 147730`, zero elapsed time, and
  no assigned nodes, so it consumed no GPU resources and must not be used as a
  candidate or comparison result.
- Optimization evidence is organized around three coupled objectives: kernel
  efficiency, maximum MB2 overlap, and communication efficiency. TensorRT-LLM
  is the expert-optimized WideEP reference for implementation and trace ideas;
  FastAFD remains an AFD implementation and should adopt only mechanisms that
  improve its measured MB2 critical path while preserving its contracts.
- Stream-memory-wait candidate `6950437` replaced model-side readiness polling
  CTAs with exact 64-bit CUDA stream memory waits plus remote-write flush and
  restored MegaMoE from 150 to all 152 GB200 SMs. Its focused local source
  contracts passed 23/23 and the pinned remote runtime/source suites passed
  46/46. The first HSG `short`/one-hour run reached JIT loading but failed
  before graph capture because the generated transport object did not link the
  CUDA driver library (`undefined symbol: cuDeviceGetAttribute`). It was
  canceled at 12:47 to release both trays; this is build-integration evidence,
  not a performance result. The transport JIT now explicitly links `-lcuda`;
  the otherwise unchanged exact MB2 candidate was resubmitted as job `6950888`
  at 00:54 PDT to HSG `short`, two trays/eight GPUs, with a one-hour limit.
- Linked rerun `6950888` reached MegaMoE setup (`compute_sms=152`) but GB200
  reported `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES=0`; the explicit
  capability gate failed before graph capture, and Slurm recorded `FAILED 3:0`
  in 7:49. The safe revision keeps the exact nonresident 64-bit stream wait
  without the unsupported flush flag, then launches the existing one-CTA
  system-acquire readiness check. The producer already publishes the ticket
  with a system-release store, so this preserves payload ordering while the CTA
  becomes runnable only after readiness and exits immediately instead of
  occupying an SM during peer compute.
- The next stacked candidate also follows TensorRT-LLM's independent shared-
  expert/routed-expert streams. Baseline shared-expert dense, gate, activation,
  sigmoid, multiply, and add kernels account for about 2.97 ms/step. Each
  150-SM MegaMoE CTA consumes 233,472 bytes of shared memory, preventing normal
  co-residency once the persistent kernel fills the device. Two graph-captured
  auxiliary streams therefore start the replicated shared branch before routed
  `gate_topk`/MegaMoE and join only before the residual add; this targets
  overlap with the routing prefix while retaining MB2 and the full 152-SM
  MegaMoE grid. Cross-stream tensor lifetime is recorded on the consuming main
  stream. It is initially isolated behind
  `MINISGL_AFD_QWEN_SHARED_EXPERT_OVERLAP=1` (default off), allowing a corrected
  stream-wait-only trace before the stacked overlap trace from the same source.
- Corrected stream-wait-only job `6951272` was submitted at 01:08 PDT to HSG
  `short`, two trays/eight GPUs, with a one-hour limit and shared-expert overlap
  explicitly disabled. This isolates the nonresident wait + post-wait acquire
  + 152-SM change before evaluating the prepared overlap stack.
- Job `6951272` reached the candidate's 152-SM setup, then one worker encountered
  a stale node/rank-local FlashInfer `silu_and_mul/build.ninja` whose absolute
  source path omitted `FastAFD/artifacts`; this is unrelated to the candidate.
  The job was canceled at 5:44, and only the broken rank-7 `silu_and_mul` cache
  directory and orphaned lock were moved to recoverable `quarantine_6951272`
  under that cache root so FlashInfer can regenerate them on reuse.
- Corrected stream-wait-only retry `6951447` was submitted at 01:14 PDT with
  the same exact MB2/perf contract, HSG `short`, one-hour limit, and the
  affected `nvl72104-T15` tray excluded.
- A local-only experiment control now gates the model-side mechanism with
  `MINISGL_AFD_MODEL_STREAM_WAIT` (now default off). Setting it to zero restores the
  baseline's early resident O waiter, event order, 150 compute SMs, and exact
  wait kernel. This permits a shared-overlap-only run and a final stacked run
  without changing source between comparisons. It was intentionally not synced
  into job `6951447`, whose run used the isolated stream-wait candidate.
- Retry `6951447` completed cleanly in 9:09 with canonical success, eight
  nonempty reports, no runtime/CUDA errors, and all 15 measured steps retained.
  Its mean/median/max critical spans were
  43.215494/43.192909/43.799712 ms (2.2333% range), or 428.087204
  tokens/s/active GPU. This is 4.77% slower than the 41.247700-ms baseline:
  model-role mean rose from 40.946286 to 43.215494 ms and attention-role mean
  rose from 41.236752 to 41.965208 ms. Although readiness-kernel execution fell
  from 46.156 to 0.144 ms/step of resident duration, moving the wait behind the
  local event added dependency latency; the 152-SM MegaMoE specialization also
  regressed from 18.202 to 18.532 ms/step. Reject this path, retain the resident
  early waiter and 150-SM kernel, and do not stack the losing mechanism with
  shared-expert overlap. Durable metric and representative SQLite exports are
  `analysis/stream_memop_acquire_sms152_metric.json`,
  `analysis/stream_memop_attention.sqlite`, and
  `analysis/stream_memop_model.sqlite`.
- Shared-overlap-only job `6952176` was submitted at 01:38 PDT to HSG `short`,
  two trays/eight GPUs, with a one-hour limit and the exact 128K batch-37 MB2
  contract. It explicitly restores the baseline model waiter/150-SM kernel and
  enables only the two auxiliary shared-expert streams.
- Job `6952176` completed cleanly in 8:49 with canonical success, eight
  nonempty reports, and no runtime/CUDA errors. All 15 post-warmup steps were
  retained: mean/median/max critical spans are
  38.803495/38.789697/39.200447 ms (1.6225% range), or 476.761176
  tokens/s/active GPU. This is 5.93% lower latency and 6.30% higher throughput
  than the 41.247700-ms / 448.509856-token/s baseline. Model-role mean falls
  from 40.946286 to 38.441580 ms and attention-role mean from 41.236752 to
  38.764172 ms as the shorter model dependency chain propagates through the
  pipeline. Retain the shared-expert auxiliary-stream path.
- The representative model timeline proves the intended overlap: in a final
  layer pair, the auxiliary shared gate-up starts about 19.7 us before the
  routed MegaMoE launch; shared down finishes at about +1.7 us, its gated
  elementwise tail finishes at +22.7 us, and the 149.8-us MegaMoE remains the
  enclosing operation. The residual add begins immediately after MegaMoE at
  +150.0 us, so the shared branch adds no join stall. MegaMoE itself changes
  only from 151.680 to 152.364 us average despite overlap. Durable evidence is
  `analysis/shared_expert_overlap_metric.json`,
  `analysis/shared_overlap_attention`, `analysis/shared_overlap_model`, and
  `analysis/shared_overlap_model_cuda_gpu_kern_sum.csv` under the task root.
- The losing stream-memory-wait implementation and its experiment flag were
  removed completely after the controlled result; the retained path again has
  one early resident readiness CTA and a 150-SM MegaMoE specialization.
- The next isolated kernel candidate folds Qwen3.5 full-attention
  `sigmoid(gate) * O` into the existing BF16 O publication kernel. The baseline
  trace contains 480 sigmoid and 480 multiply launches (about 0.212 ms per
  measured step) in addition to 480 publishers. The fused publisher preserves
  the two BF16 rounding points—sigmoid to BF16, then product to BF16—while
  performing the remote payload store and readiness release. Focused local
  source/extractor contracts pass 29/29 and pinned remote source/runtime suites
  pass 46/46. Its exact dry-run retains batch 37, MB2 `18+19`, 77,659 pages,
  4,970,176 KV tokens, perfect routing, two trays/eight active GPUs, HSG
  `short`, and the one-hour limit.
- Fused gated-O publisher job `6952897` was submitted at 02:04 PDT with that
  exact contract. Its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/candidate_gated_o_publish_mb2/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260906_020425_manual_na`.
- Job `6952897` completed cleanly `0:0` in 8:51 with canonical success, eight
  nonempty reports, and no runtime/CUDA error. All 15 measured steps remain:
  mean/median/max critical spans are 38.666928/38.634336/39.104576 ms
  (2.0086% range), or 478.445039 tokens/s/active GPU. This is another
  0.136567 ms (0.35%) below shared-overlap-only and 6.26% below the original
  baseline. Attention/model role means are 38.628206/38.315811 ms.
- Kernel accounting matches the end-to-end delta: the attention graph loses
  exactly 60 nodes (two launches for each of 30 full-attention MBs). Each
  gated publisher averages 9.300 us, versus 4.956-us sigmoid + 1.661-us
  multiply + 7.256-us publisher before fusion, saving about 4.57 us x 30 =
  0.137 ms/step. The 90 ungated linear-attention publishers remain a separate
  compile-time specialization at 7.201 us, unchanged from baseline. Retain the
  fused gated publisher. Durable evidence is
  `analysis/gated_o_publish_metric.json` and
  `analysis/gated_o_attention_cuda_gpu_kern_sum.csv` under the task root.
- With both measured wins accepted, the shared-overlap experiment variable was
  removed: Qwen hybrid-attention MB2 with same-rank MegaMoE now takes the one
  optimized auxiliary-stream path automatically; there is no retained
  fallback/control branch.
- A final isolated kernel candidate generalizes the existing TensorRT/vLLM-
  derived fused Q/K RMSNorm+RoPE kernel from full 128-wide rotation to
  128/256-wide heads with an independently templated partial rotary dimension.
  Qwen3.5 uses head dimension 256 and rotary dimension 64. Each warp now loads
  8 BF16 elements per lane, normalizes the full head, rotates only the leading
  eight lanes, and leaves the rest normalized. This targets the current two
  QK-RMSNorm launches plus one partial-RoPE launch per full-attention MB.
  Local source/extractor contracts pass 30/30 before remote validation.
- Pinned remote source/runtime suites pass 47/47. Exact QK-fusion candidate
  job `6953511` was submitted at 02:26 PDT to HSG `short`, two trays/eight
  GPUs, with the one-hour limit and unchanged batch-37 MB2 `18+19` contract.
  Its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/candidate_qk_norm_rope_hd256_mb2/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260906_022650_manual_na`.
- Job `6953511` completed cleanly `0:0` in 8:11 with canonical success and all
  15 measured steps, but regressed the accepted gated-publisher candidate:
  mean/median/max critical spans were 38.764529/38.773439/39.073249 ms
  (1.8052% range), or 477.240423 tokens/s/active GPU. This is 0.097600 ms
  slower than the retained 38.666928-ms mean. The model graph did lose the
  expected 60 nodes; the fused QK kernel averaged 3.319 us, versus about
  6.706 us for the two prior RMSNorm launches plus RoPE per full-attention MB,
  for roughly 0.101 ms/step of isolated kernel savings. That small saving did
  not improve the distributed critical path, so the head-256/partial-RoPE
  extension and its source-contract test were removed rather than spending
  additional GPU time on a high-variance ABBA study. Preserve the existing
  Qwen3.5 QK path and the faster measured end-to-end candidate. Durable
  evidence is `analysis/qk_norm_rope_hd256_metric.json` and
  `analysis/qk_fused_model_cuda_gpu_kern_sum.csv` under the task root.
- The final accepted source (shared-expert auxiliary-stream overlap plus fused
  gated-O publication, with the QK experiment removed) passes 29/29 focused
  local source/extractor tests and 29/29 pinned remote source/extractor tests;
  its pinned remote runtime suite passes 23/23. A3/8K normal-routing dry-run
  validation reports 48 prompts, attention DP 12, four FFN ranks, MB2 `2+2`,
  four trays/16 active GPUs, and no perfect-routing or perf-mode shortcuts.
  Final alignment inference job `6953946` was submitted to HSG `short` with a
  one-hour limit at 02:43 PDT. Its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/final_a3_normal_alignment/results/afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260906_024341_manual_na`.
- Job `6953946` completed cleanly `0:0` in 9:54. The normal-routing run wrote
  exactly 48 samples, every sample generated 17 tokens, and the total is the
  expected 816 tokens; its AFD log contains no traceback, CUDA error, runtime
  error, assertion error, or failure marker. The unchanged scorer source is
  clean at pinned revision `dbf66891a872e81d5f6ee0cf923fb31bf0d64a6b`.
  One-tray scorer job `6954382` was submitted at 02:57 PDT to HSG `short` with
  a one-hour limit; output is under the run's `vllm-score-final/` directory.
- TensorRT-LLM's custom MoE routing path motivated one last isolated kernel
  candidate. Qwen3.5 has `norm_topk_prob=True`, so its final weights are a
  softmax over only the chosen top-10 logits. The existing AFD selector still
  computes max and denominator over all 512 experts before discarding them.
  A compile-time `kRenormalize` specialization removes that unused 512-way
  exponential/reduction work while leaving non-renormalized routing in the
  original operation order. The local source contracts pass 29/29. Keep this
  only if an exact MB2 end-to-end trace beats the accepted gated-O result.
- Scorer job `6954382` completed cleanly `0:0` in 10:39. It evaluated all 48
  prompts and 816 generated tokens: overall top-10 and top-100 hit rates are
  both 1.0, top-1 agreement is 0.984069, average/max vLLM rank is 1.019608/4,
  and mean log-probability/perplexity is -0.094037/1.098600. The strict
  unchanged correctness gate therefore passes for the retained optimization
  stack before the isolated selector experiment.
- Pinned remote source/runtime suites pass 29/29 and 23/23 with the selector
  specialization. Exact dry-run validation retains the target's batch 37,
  MB2 `18+19`, 77,659 pages / 4,970,176 KV tokens, perfect routing, two
  trays/eight active GPUs, HSG `short`, and one-hour limit. Performance job
  `6954952` was submitted at 03:15 PDT; its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/candidate_renorm_selector_mb2/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260906_031530_manual_na`.
- Job `6954952` completed cleanly `0:0` in 8:35 with canonical success, eight
  nonempty reports, and all 15 measured steps. Mean/median/max critical spans
  are 38.454877/38.461471/38.851680 ms (1.7174% range), or 481.083317
  tokens/s/active GPU. This beats the prior accepted 38.666928-ms candidate by
  0.212051 ms (0.55%) and raises throughput by 0.55%. Attention/model role
  means are 38.448832/38.055466 ms; graph node counts remain 1890/2531.
- The specialized selector averages 12.831 us versus 14.356 us before, saving
  1.525 us x 120 MB2 launches = 0.183 ms/step, consistent with the end-to-end
  gain. Retain it. Durable evidence is `analysis/renorm_selector_metric.json`
  and `analysis/renorm_selector_model_cuda_gpu_kern_sum.csv` under the task
  root. Relative to the original baseline, the final stack is 2.792823 ms
  (6.77%) lower latency and 32.573461 tokens/s/GPU (7.26%) higher throughput.
- Final post-selector A3/8K normal-routing dry-run validation again reports 48
  prompts, attention DP 12, four FFN ranks, MB2 `2+2`, four trays/16 active
  GPUs, and no perfect-routing or perf-mode shortcuts. Alignment inference job
  `6955286` was submitted at 03:27 PDT to HSG `short` with a one-hour limit;
  its run directory is
  `FastAFD/artifacts/qwen35_attn_only_optimization_20260906/final_a3_normal_alignment_renorm/results/afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260906_032743_manual_na`.
- Job `6955286` completed cleanly `0:0` in 9:42. It wrote exactly 48 samples,
  every sample has 17 generated tokens, and the total is 816; the AFD log has
  no traceback, CUDA error, runtime error, assertion error, or failure marker.
  Final one-tray vLLM scorer job `6955552` was submitted at 03:43 PDT to HSG
  `short` with a one-hour limit using the same clean pinned scorer source. Its
  output is under the run's `vllm-score-final/` directory.
- Final scorer job `6955552` completed cleanly `0:0` in 10:23. It evaluated
  all 48 prompts and 816 generated tokens: overall top-10/top-100 hit rates are
  exactly 1.0, top-1 agreement is 0.984069, average/max vLLM rank is
  1.018382/3, and mean log-probability/perplexity is -0.096089/1.100857. The
  required post-selector A3/8K normal-routing correctness gate therefore
  passes unchanged.
- Final retained implementation combines three complementary optimizations:
  (1) TensorRT-style auxiliary-stream overlap of replicated shared-expert and
  routed-expert work, (2) fusion of full-attention sigmoid gating into remote
  O publication, and (3) compile-time removal of unused all-expert softmax
  normalization in Qwen's top-k-renormalized selector. Exact 128K A1:F1 EP4
  MB2 performance improves from 41.247700 ms / 448.509856 tokens/s/GPU to
  38.454877 ms / 481.083317 tokens/s/GPU: 6.77% lower latency and 7.26% higher
  throughput. The nonresident stream-wait/152-SM experiment and head-256
  partial-RoPE fusion were rejected on measured end-to-end regressions and are
  absent from the final source.
- A completion audit then found one remaining kernel-version question rather
  than closing the task prematurely. The TensorRT-LLM EP4 trace's routed
  kernel is DeepGEMM's newer ring-scheduled implementation: its specialization
  uses M16/K256, a 152-SM grid, and 207,476 bytes of dynamic shared memory,
  while the retained AFD pool-scheduled specialization uses M16/K128, 150 SMs,
  and 233,472 bytes. This is a complete scheduler/communication redesign, not
  the isolated K256 change that previously regressed AFD's M16 path.
- Upstream DeepGEMM public-release commit
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` was captured under the task's
  `upstream_ring_probe/` artifacts. A standalone deterministic perfect-routing
  probe was first submitted as HSG `short`/one-hour job `6956557` on one
  tray/four GPUs. It failed before kernel compilation in 1:15 because the
  container did not inherit Slurm's requested working directory and
  `setup.py` could not resolve `csrc/python_api.cpp`; this is harness evidence,
  not a GPU result. Identical retry jobs `6956723` and `6956781` also failed
  before compilation in 1:14/1:09: changing the parent shell's directory and
  Slurm's explicit `--chdir` were both reset at container entry. Read-only
  preflight confirms that the declared remote root does contain both
  `setup.py` and `csrc/python_api.cpp`. The launcher now changes directory and
  asserts the source file from inside the container command itself. Retry job
  `6956920` used the same one-tray HSG `short`/one-hour contract and measured
  18, 19, and 37 local tokens with Qwen's exact 4096/1024,
  512-expert, top-10, EP4 shape before any ring-kernel integration touches the
  accepted AFD source. Retain the current source unless the 18/19 measurements
  justify the integration; a full-batch-only win is irrelevant under hard MB2.
- Ring probe job `6956920` completed cleanly `0:0` in 5:20. Per-rank
  `mega_moe` time is 172--174 us at 18 tokens, 172--174 us at 19 tokens, and
  172--176 us at 37 tokens. This confirms the exact newer specialization seen
  in the TensorRT-LLM trace, but it is 12--15% slower than AFD's retained
  route-prepared kernel (~153.7 us averaged over hard-MB2 launches) despite
  using all 152 SMs rather than AFD's overlap-safe 150. Even the impossible
  best case of deleting AFD's entire 12.831-us selector would not offset the
  ring-kernel gap, and AFD still requires that selector to produce quantized
  inputs, IDs, and weights. Reject the ring port; no production source was
  changed for it. Its fixed scheduling/synchronization cost is not amortized
  at MB2, which explains why TRT-LLM's optimized wide-EP design is not the
  right same-part kernel for AFD's imbalanced 18+19 pipeline. The retained
  pooled scheduler plus fused route publication remains the better design for
  this target.
- A post-completion challenge reopened the full post-attention-to-MegaMoE
  boundary instead of treating the routed kernel alone as the complete audit.
  In a representative full-attention MB from final job `6954952`, remote O
  readiness to routed-MegaMoE launch is about 41.1 us: per-token FP8 cast
  1.79 us, O projection 12.83 us, fused residual/RMSNorm 3.20 us, fused gate
  GEMM 7.58 us, and the route-preparing selector 12.03 us, with only about
  1.5 us of launch gaps on the direct path. The replicated shared-expert
  branch is already completely enclosed by the routed 150-SM MegaMoE kernel.
  Linear-attention MBs have no O projection; their larger apparent handoff
  delay is the intentional MB2 wait for the preceding exclusive routed work,
  not uncovered local prefix work.
- TensorRT-LLM's comparable full-attention prefix is about 30.7 us, but the
  boundary is not work-equivalent: its 4.48-us routing plus 2.30-us prepare
  does not publish AFD's remote per-route grouping/count metadata, and that
  work remains inside TRT-LLM's slower 172--174-us ring MegaMoE. AFD's gate,
  route-preparing selector, and retained MegaMoE total about 172.6 us versus
  roughly 186 us for TRT-LLM's corresponding routed sequence. The credible
  remaining same-part target is therefore the 12.831-us AFD selector itself:
  it rescans each lane's 16 logits for every top-10 winner, whereas TRT-LLM
  sorts those 16 candidates once and consumes the per-lane heads. Evaluate an
  exact-output, same-node MB2 source A/B before retaining any such change.
- The first narrow candidate applies the sort-once epilogue only to the
  compile-time-renormalized Qwen path; the non-renormalized selector and its
  arithmetic order remain untouched. It also preserves AFD's 512-thread block
  because the last selector block must finalize all 512 remote expert counts,
  work absent from TRT-LLM's 256-thread router. Local diff/source contract
  checks pass. Same-node control/candidate/candidate/control job `6966077` was
  submitted to HSG `short` with an explicit one-hour limit, one tray/four GPUs,
  exact hard-MB2 rows `18,19`, 150 MegaMoE SMs, FP4 experts, 512 experts,
  top-10, hidden/intermediate 4096/1024, deterministic perfect routing, 30
  warmups, and 1,000 measured graph replays. Clean source snapshots are
  `selector_sort_abba/source/{control,candidate}` at heads `8214c08` and
  `3bd9f3e`; output is under `selector_sort_abba/output/`.
- Job `6966077` failed before the exact-shape test in 3:04, after its control
  build and all small normal/prepared numeric checks passed. The reusable ABBA
  wrapper accepted slash encoding for specialization tuples but forwarded the
  similarly slash-encoded `18/19` row list verbatim; the Python smoke correctly
  rejected it as a non-integer. This is harness evidence only. Normalize the
  sbatch-safe row encoding in the general wrapper and retry the identical A/B;
  reuse only the clean control compile cache, not any timing result.
- The generalized ABBA wrapper now normalizes slash-encoded row lists and
  passes `bash -n`. Identical HSG `short`/one-hour retry job `6966216` uses
  clean control/candidate heads `6536e1c`/`322c7c2`, a fresh result root
  `selector_sort_abba/output_retry/`, and only the prior clean control
  DeepGEMM build cache. All exact MB2 and precision/routing parameters remain
  unchanged.
- The broader boundary audit now covers every traced kernel, not only the
  selector. On a representative full-attention MB, AFD's FP8 O cast is
  1.792 us versus TRT-LLM's aggregate 2.177 us, O projection is 12.832 us
  versus 12.941 us, fused residual/RMSNorm is 3.200 us versus 3.286 us, and
  routed gate GEMM is 7.584 us in this instance (6.127-us aggregate) versus
  TRT-LLM's 7.816-us aggregate. Direct-path launch gaps from O readiness
  through MegaMoE are only about 1.5 us total. These parts are already at or
  ahead of the TRT-LLM reference; the selector is the only direct-prefix
  kernel with a credible algorithmic delta.
- The same trace also closes the replicated shared-expert side path. Its FP8
  cast/L1/SwiGLU/cast/L2/gate-reduction/sigmoid/multiply sequence runs on the
  auxiliary stream from 303598.645 to 303645.653 us, entirely inside routed
  MegaMoE's 303620.533--303771.381-us interval. The post-Mega shared add plus
  next input fused-add/RMSNorm is 1.632+2.848=4.480 us, already below
  TRT-LLM's fused shared combine plus fused-add/RMSNorm aggregate
  2.172+3.286=5.458 us. AFD's isolated broadcast multiply is inefficient
  (16--20 us) but is intentionally hidden on the two reserved SMs; moving it
  into a TRT-style post-Mega fused combine would lengthen rather than shorten
  the critical path. Do not trade away the measured auxiliary-stream overlap
  merely to reduce non-critical aggregate kernel time.
- The remaining full-attention cast cannot be removed by transporting gated O
  as FP8 without violating the established correctness contract. Earlier
  Qwen3.5 attn-only FP8 O transport diverged from the legacy BF16 ordering and
  failed the unchanged top-10 gate (`6936518`, 0.992647); restoring BF16 O
  transport made every token identical at A1 and restored top-10=1.0
  (`6937369`/`6937561`). Current AFD therefore performs the same necessary
  pre-O-projection FP8 quantization that TRT-LLM traces as a separate
  ~2.177-us kernel; AFD's representative kernel is 1.792 us. Moving the
  12.8-us O projection and roughly 1 GB of weights to the already-slower
  attention stage would also worsen the measured pipeline imbalance. Preserve
  the BF16 split and model-side cast/O projection.
- Sort-once retry `6966216` is rejected. Control trial 0 passed every small
  normal/prepared numeric check and measured 0.169783/0.169698 ms at exact
  rows 18/19. Candidate trial 1 passed those checks but measured
  0.172005/0.172102 ms, regressions of 2.222/2.404 us (1.31/1.42%) for the
  gate+Mega stage. Candidate trial 2 then produced NaN output in the repeated
  normal-routing row-6 check, so the job correctly terminated before the
  fourth leg and Slurm reports `FAILED 143:0`; the successful first candidate
  timing does not mitigate that correctness failure. Remove the sort and its
  extra 16 indices/thread from production. The lower-register follow-up keeps
  the original repeated scan and exact tie order but consumes a winner by
  overwriting its existing register value with `-FLT_MAX`, eliminating the
  used-mask bit shift/test on every local candidate. Local diff/source
  contract checks pass; validate it independently against the original
  accepted control.
- Same-node ABBA job `6966386` was submitted to HSG `short` with an explicit
  one-hour limit for the in-place candidate. It compares accepted control
  `6536e1c` against clean candidate `bfe6341` in control/candidate/candidate/
  control order, reusing only the prior clean control DeepGEMM cache. Exact
  FP4, rows `18,19`, 150 SMs, bucket 768, 512 experts, top-10, 4096/1024,
  perfect routing, and 1,000-replay settings remain unchanged; output is
  `selector_inplace_abba/output/`.
- In-place selector ABBA `6966386` completed cleanly in 5:24 but is rejected as
  noise/regression. Control placements averaged 0.169480/0.169481 ms for rows
  18/19; candidate placements averaged 0.169567/0.169668 ms, regressions of
  0.087/0.187 us. Every small prepared/normal numeric check passed, so this is
  a performance rejection rather than a correctness failure. Restore the
  original used-mask scan.
- Iteration policy clarified by the user: accumulate a meaningful optimization
  batch before running another alignment test. During optimization discovery,
  use the faster exact-shape perf-mode A/B as the gate; retain only inexpensive
  kernel/numeric smoke needed to catch malformed candidates. Run alignment once
  at the end of the accumulated batch, not after each micro-optimization.
- A proposed perfect-sequence-only direct route-count publication was removed
  before remote testing after the user clarified the optimization contract.
  Perfect routing is only a repeatable perf input, not a production assumption:
  every retained optimization must improve the general arbitrary-routing path.
  Use TRT-LLM as an expert reference for optimization structure, but because its
  inputs and work boundaries differ, accept or reject changes from same-input
  AFD-before versus AFD-after perf-mode measurements rather than absolute
  TRT-LLM latency. For routing changes, include a representative non-perfect
  performance case in addition to the deterministic exact-shape perf input.
- The next routing-independent selector candidate replaces each five-stage
  shuffle-tree warp argmax with Blackwell `redux.sync.max.u32`. A monotonic
  float-to-unsigned transform plus a second lane-key reduction preserves the
  original lower-lane tie policy, including canonicalized signed zero, while
  removing ten shuffles and their comparisons per top-k round. Same-node ABBA
  job `6966691` completed cleanly in 6:19 on HSG `short` with a one-hour limit.
  For deterministic perfect routing, control placements averaged
  169.516/169.614 us at rows 18/19 and candidate placements averaged
  168.115/168.134 us, wins of 1.401/1.479 us. Every small normal/prepared
  numeric check passed.
- Non-perfect-routing ABBA job `6966880` then completed cleanly in 5:14 under
  the same HSG resource/time contract. Control placements averaged
  147.175/149.024 us at rows 18/19; candidate placements averaged
  145.765/147.537 us, consistent wins of 1.410/1.487 us. This is a general
  selector optimization, not a perfect-routing shortcut, so retain it for the
  next full AFD perf-mode comparison.
- The remaining clear TensorRT-LLM fusion/overlap delta is Qwen3.5 GDN input
  projection. The accepted trace runs separate QKV and Z FP8 GEMMs plus two
  separate BF16 B/A GEMMs for each of 45 linear-attention layers and both MB2
  lanes. The prepared candidate packs QKVZ into one block-scaled FP8 projection
  and BA into one BF16 projection, preserving the checkpoint's actual quant
  exclusions. BA runs on one persistent auxiliary stream per AFD decode lane;
  its join is deferred until after the independent QKVZ projection and causal
  convolution. This follows TRT-LLM's fused projection structure while
  extending the overlap window. Checkpoint merge order is dense `[Q|K|V|Z]`
  and `[b|a]`; the loader now also requires a module-name boundary when
  recognizing split projections so already-fused names cannot be re-merged.
  A focused source contract passes locally. Measure this increment against a
  clean selector-reduction control in exact batch-37 MB2 perf mode; defer the
  next alignment test until the optimization batch is complete.
- Clean GDN control/candidate sources are
  `selector_redux_abba/source/candidate` at `f16a205` and
  `gdn_projection_fusion_overlap_ab/source/candidate` at `ad46487`. Exact
  login-host dry-run validation reports Qwen3.5 128K A1:F1/EP4/ATP1, batch 37,
  MB2 `18+19`, 77,659 pages / 4,970,176 KV tokens, FP4 experts, perfect-routing
  perf mode, two trays/eight active GPUs, and 15 measured steps. Same-allocation
  AFD-before/AFD-after job `6967061` was submitted to HSG `short` with a
  one-hour limit; it runs control then candidate with isolated build caches and
  writes under `gdn_projection_fusion_overlap_ab/perf/`.
- Job `6967061` exposed a paired-harness issue before model loading or GPU
  execution: the reusable online bundle still hardcoded `qwen3` even though
  each plan row already carries `run_afd_argv`. It was canceled after 1:14 to
  release both trays. The bundle now validates and forwards the row's supported
  model key (`qwen3`, `qwen35`, or `minimax`), and the one-case plan uses the
  current required `-fp4` case-id suffix. Local plan initialization, shell
  syntax, and two focused source contracts pass. Harness-identical clean
  control/candidate heads are now `ea958db`/`b85f590`; unchanged paired perf
  retry `6967240` was submitted to HSG `short`, two trays/eight GPUs, explicit
  one-hour limit.
- Pre-launch fused-layout review found that QKVZ makes the Z slice contiguous
  within each token/head but leaves a wider token stride. The old gated
  RMSNorm required global contiguity and would have repacked or rejected this
  view, unlike TRT-LLM's direct strided-Z consumption. Job `6967240` was
  canceled after 0:09, before useful GPU work. The gated RMSNorm now indexes Z
  with its actual token stride while preserving contiguous head blocks; clean
  candidate head `0dac533` is paired with unchanged control `ea958db` in retry
  `6967394`, again HSG `short`, two trays/eight GPUs, one-hour limit.
- The user clarified that the remaining optimization target is specifically
  fusion and overlap. TRT-LLM source review confirms the relevant GDN pattern:
  parallel packed QKVZ/BA projections, strided Z consumption, fused post-conv
  preparation, and gated RMSNorm fused with the downstream quantization. A
  local next-batch candidate now extends the isolated remote projection change:
  causal convolution writes globally token-contiguous Q/K/V directly; it also
  computes one Q/K normalization multiplier per produced BF16 head so all
  recurrent value tiles reuse it instead of repeating reductions. BA is read
  in place, and fused FP32 gate preparation runs after BA on the same auxiliary
  stream while the main stream performs QKVZ plus convolution; the join remains
  immediately before recurrent-state update. The recurrent kernel consumes the
  precomputed gates. Finally, gated per-head RMSNorm produces dynamic FP8 plus
  packed UE8M0 scales directly for the FP8 output projection, removing the
  intervening BF16 materialization/reload and quantization launch. These are
  general decode transformations with no routing assumption. A reusable focused
  CUDA numeric smoke is `codex_scripts/validate_qwen35_gdn_fusion.py`; do not
  sync or launch this stacked source until active paired job `6967394` exits.
- Isolated paired job `6967394` completed cleanly `0:0` in 14:07. The accepted
  selector-reduction control measured 39.212974 ms and 471.782628 TPS/GPU; the
  packed QKVZ/BA plus auxiliary-stream BA candidate measured 39.108414 ms and
  473.043987 TPS/GPU, a 0.104560-ms / 0.267% latency win and 0.267% throughput
  win. It also reduced the profiled attention graph from 1,890 to 1,620 nodes,
  exactly 270 launches (three per GDN layer across 45 layers x hard MB2), while
  the model graph stayed at 2,531 nodes. Retain the projection fusion/overlap;
  its isolated E2E benefit is partially hidden by the imbalanced pipeline.
  The stacked source is clean at remote head `e06c79e` under
  `gdn_fusion_overlap_stack_ab/source/candidate`. One-tray CUDA smoke job
  `6967752` was submitted to HSG `short` with an explicit one-hour limit before
  spending the next two-tray allocation.
- Smoke `6967752` compiled and passed the packed-convolution, Q/K-scale,
  precomputed-gate, recurrent-output, and recurrent-state comparisons, then
  failed in 1:20 while compiling only the final fused FP8 scale store. Triton
  preserved a length-one block after reducing a `[4,1]` expression, which
  cannot be stored through a scalar pointer. The pack now reduces a 1D four-row
  exponent vector to a true scalar. Clean candidate head `08d9e00` contains
  the fix; otherwise-identical short/one-hour smoke retry is `6967830`.
- Corrected smoke `6967830` completed cleanly `0:0` in 1:16. At the exact
  Qwen3.5 16-key-head/64-value-head/128-wide specialization it reports zero
  difference for convolution output/state, recurrent output/state, and fused
  gated-RMSNorm dynamic-FP8 values plus packed UE8M0 scales. Stacked paired
  perf job `6967895` now compares accepted projection control `0dac533` with
  candidate `08d9e00` on HSG `short`, two trays in one segment, eight GPUs,
  explicit one-hour limit, exact 128K batch-37 MB2 `18+19`, FP4 experts, and
  the same cached input/perf protocol.
- First stacked submission `6967895` was canceled after 1:09 when the control
  preflight correctly rejected a missing MXFP4 checkpoint under the supplied
  reproduction root; no model loaded and no measurement exists. The paired
  launcher needs the campaign artifact root for model/checkpoint discovery,
  while the EP virtual environment remains under `fastafd_reproduce`. Fresh
  retry `6967996` supplies that exact accepted root plus explicit dense and
  MXFP4 checkpoint paths. It otherwise retains HSG `short`, two trays/segment,
  eight GPUs, hard MB2, and the explicit one-hour limit.
- Live inspection of retry `6967996` after the user challenged an inaccurate
  prefill explanation found `afd_perf_mode: 0` in the inner driver despite the
  plan's explicit `afd_perf_mode=1`. The paired bundle parsed memory and
  precision fields but silently dropped the per-case perf contract at the
  container boundary, so the run was canceled at 14:00 before any measurement.
  This is harness evidence only; it must not be used as a performance result.
  `run_afd_bundle.sh` now requires and validates `afd_perf_mode`,
  `afd_perf_context_len`, `perfect_routing`, and `perf_state_source` from every
  claimed plan row, then explicitly forwards perf mode and routing to the case
  runner. Focused shell/source checks pass in both clean snapshots. Updated
  control/candidate heads are `f61ab77`/`5d3f787`. Corrected job `6968363` was
  submitted to HSG `short`, two trays/eight GPUs, with an explicit one-hour
  limit; verify the inner driver reports perf mode before allowing it to run.
- Corrected job `6968363` completed cleanly in 15:11. Both halves report
  `afd_perf_mode: 1`, use the cached profiling stub, and initialize randomized
  resident KV+GDN state on every attention worker; no 128K prefill occurs.
  Same-tray control is 38.369092 ms / 482.158927 TPS/GPU with 1,620 attention
  graph nodes. The full stacked candidate reaches the expected 1,170 nodes but
  measures 38.426063 ms / 481.444064 TPS/GPU: a 0.056972-ms / 0.148% latency
  regression and 0.148% throughput regression. Reject the full stack as-is.
- Focused captured-graph microbenchmark `6969027` passes exact rows-19 numeric
  checks. Packed convolution/layout plus shared Q/K scales saves 8.194 us per
  invocation, and gated RMSNorm plus the production packed-UE8M0 FP8 quantizer
  saves 2.055 us. Precomputing gates separately loses 2.258 us versus leaving
  gate arithmetic inside the recurrent kernel, before accounting for its
  extension of the auxiliary BA-stream join. Therefore keep the two measured
  fusion wins but remove decode gate hoisting. The first microbenchmark attempt
  `6968898` failed only its clean-head preflight due an incorrectly expanded
  hash and contains no timing evidence.
- Clean no-gate candidate `3431f38` retains packed QKVZ/BA, auxiliary BA
  projection, direct packed convolution output, Q/K scale reuse, and fused
  gated-norm/FP8 output. Exact MB2 perf-mode job `6969097` compares it directly
  with packed-projection control `f61ab77` on one same-fabric two-tray HSG
  `short` allocation with an explicit one-hour limit. Its control inner driver
  reports perf mode 1; retain only on an end-to-end win.
- No-gate job `6969097` completed cleanly in 15:04. Same-tray packed-projection
  control is 38.505249 ms / 480.453982 TPS/GPU with 1,620 attention nodes; the
  candidate is 38.799618 ms / 476.808816 TPS/GPU with 1,080 nodes, a
  0.294370-ms / 0.764% latency regression. Reject it. Compared with the full
  stack's much smaller regression, gate precomputation recovers roughly
  0.24 ms despite adding launches; avoiding repeated strided BA/gate work in
  the recurrent kernel matters more than raw graph-node count.
- The next binary ablation restores auxiliary-stream gate preparation and
  disables only gated-RMSNorm-to-FP8 fusion, retaining packed convolution,
  direct Q/K/V layout, Q/K scale reuse, and the previously accepted packed
  projections/overlap. Clean candidate `a4d7461` is compared directly with
  packed-projection control `f61ab77` in exact perf-mode job `6969400`, HSG
  `short`, two same-fabric trays/eight GPUs, explicit one-hour limit. This
  isolates whether norm/quant fusion is harmful under the full attention graph
  even though its standalone captured graph is faster.
- Job `6969400` completed cleanly `0:0` in 14:54. Its same-allocation control
  measured 38.455017 ms / 481.081568 TPS/GPU with 1,620 attention graph nodes;
  the no-norm candidate measured 38.960319 ms / 474.842106 TPS/GPU with 1,260
  nodes, a 0.505302-ms / 1.314% latency regression and 1.297% throughput loss.
  This rejects the no-norm composition. Together with the standalone captured
  norm+production-quant win and TRT-LLM's same fused post-GDN design, retain
  gated-RMSNorm-to-FP8 fusion. Across separate exact-pair allocations it closes
  roughly 0.448 ms of the no-norm regression in the full stack; treat that
  magnitude as directional rather than a same-allocation subtraction.
- Optimization acceptance is now explicitly two-axis. Exact MB2 A1 AFD-before
  versus AFD-after perf mode remains the primary system measurement, but an
  optimization may be retained when A1 is pipeline-masked/noise-level if it has
  a sound general algorithmic case: less launched work or global traffic,
  independently faster production-equivalent component timing, and no narrower
  routing assumption. This matters at larger A:F ratios where the exposed
  attention-side balance differs. Clear A1 regressions without compensating
  component evidence remain rejected. The no-gate and no-norm compositions are
  examples of rejection; the full stack's +0.057-ms A1 result remains under
  component isolation rather than being discarded wholesale.
- The final binary isolation keeps packed QKVZ/BA projection, auxiliary-stream
  BA plus gate preparation, strided packed BA/Z consumption, and fused gated
  RMSNorm-to-FP8 output, but restores the established convolution output plus
  recurrent Q/K normalization. Its clean remote source is `d114353` under
  `gdn_fusion_no_conv_ab/source/candidate`; the focused source contract passes.
  Exact same-allocation perf-mode job `6969823` was submitted to HSG `short`
  with two same-fabric trays/eight GPUs and an explicit one-hour limit. It uses
  hard MB2 `18+19`, 128K logical randomized-resident state without prefill, FP4
  experts, and packed-projection control `f61ab77`; results write under
  `gdn_fusion_no_conv_ab/perf/`.
- A proposed recurrent tile/warp parameter sweep was stopped when the user
  clarified that structural fusion, overlap, and communication alignment must
  come first. Job `6969913` was canceled after 0:40 before any benchmark result;
  do not use it as evidence. Its tuning-only local edits were removed. The
  Qwen3.5 GDN component validator remains because it measures the relevant
  structural fusions rather than launch-parameter search.
- Qwen3 performance is a hard non-regression contract. Qwen3.5 GDN,
  gated-output, and hybrid shared-expert paths are architecture-guarded and do
  not execute for Qwen3. Generic selector/renormalization changes do affect the
  common MegaMoE router and therefore require a same-shape Qwen3 before/after
  perf check against the pre-optimization implementation before the accumulated
  batch is finally accepted.
- No-conv isolation job `6969823` completed cleanly. Its exact same-allocation
  packed-projection control measured 38.515220 ms / 480.329596 TPS/GPU; the
  candidate that retained projection/gate/norm fusion but restored the old
  convolution layout and recurrent Q/K normalization measured 38.561088 ms /
  479.758247 TPS/GPU, only +0.045868 ms / +0.119%. This is indistinguishable in
  direction and magnitude from the full stack's +0.056972-ms A1 result, while
  the direct packed convolution plus scale reuse independently saves 8.194 us
  and removes materializations/launches. Retain the full structural GDN stack
  under the two-axis acceptance rule; neither no-conv nor full stack has a
  meaningful A1 advantage, and full stack is the stronger general/large-A:F
  implementation.
- The next structural communication candidate is a strictly Qwen3.5-only full
  attention publisher. One CUDA kernel reads the interleaved Q/gate plus K/V
  projection views, applies per-head Q/K RMSNorm with the established BF16
  rounding boundary, applies 64-of-256 NeoX RoPE, writes prepared Q plus raw
  gate directly to remote Q slots, writes prepared K plus raw V directly to
  the remote cache, executes the system fence, and releases readiness. This
  removes split-query/gate materialization, separate Q norm, K norm, RoPE,
  concatenation, and the reread-only generic transport launch. In MB2 graph
  scheduling the handoff event is recorded after the producer QKV GEMM and
  before this fused compute/communication tail, preserving overlap with the
  peer lane's next compute unit. Qwen3 retains the existing generic preparation
  and publisher byte-for-byte at runtime; the dedicated JIT is built only on a
  hybrid model worker. Clean remote candidate `9cdd8b3` lives under
  `qwen35_qkv_publish_ab/source/candidate`; source contracts pass. Initial smoke
  `6970249` was canceled after 1:04 when source review found the eight-lane
  partial-RoPE shuffle incorrectly declared a full-warp participation mask; no
  result from it is accepted. The exact `0x000000ff` mask fix is remote head
  `2b316ee`. Retry `6970269` then failed during JIT compilation before runtime:
  this repository's host tensor matcher has no direct `__nv_bfloat16` dtype
  trait. The wrapper now uses the established symbolic dtype plus explicit BF16
  runtime check at clean head `4844f89`. Numeric, repeated counter/slot, and
  remote-fabric retry `6970331` completed successfully on HSG `short`, one
  tray/four GPUs, with an explicit one-hour limit: all two remote attention
  pairs passed three-row/nine-iteration BF16-O validation, including normalized
  Q/K, partial RoPE, raw gate/V layout, readiness, and counter/slot reuse.
- The user explicitly accepted an overlap invariant for both pipeline sides:
  make the next microbatch compute-runnable before launching the current
  microbatch's publication, so the whole communication tail can overlap. The
  model-side MB2 handoff event already follows that order for QKV publication.
  Qwen3.5 attention-side O publication now uses a one-thread device turn release
  ordered before the BF16 O publisher instead of releasing inside that
  publisher. Qwen3 retains its existing quantize/publish/release path.
- The same integrated Qwen3.5-only communication stack fuses each linear
  attention input's residual add plus RMSNorm with direct remote publication.
  It eliminates the local normalized activation materialization and its later
  reread/copy, writes the BF16 residual locally for following layers, and
  records the model-side handoff before the fused norm/publish launch. Thus the
  next FFN microbatch can begin before this communication tail. The full-QKV
  path likewise records handoff after QKV GEMM and before its fused publisher.
  Width/dtype/architecture validation is fail-fast and Qwen3 never enters these
  paths. Runtime/source tests pass in the isolated remote snapshot; clean head
  `2bdfc92` is under `qwen35_qkv_publish_ab/source/candidate`.
- Integrated fabric smoke `6970562` completed cleanly `0:0` in 3:38 on HSG
  `short`, one tray/four GPUs, explicit one-hour limit. It passed the expanded
  QKV publisher, linear residual+norm publisher, early O turn release, BF16 O
  publication, remote readiness, and repeated counter/slot reuse for both
  remote pairs across three rows and nine iterations.
- Initial paired submission `6970652` failed in its CPU-side case-ID validator
  before model loading because the descriptive `-normal-routing` suffix is not
  part of the reusable case-ID grammar. It was canceled at 0:56 to release both
  trays; it contains no performance evidence. The known-valid case ID now
  leaves routing to the explicit perf/alignment execution policy rather than
  encoding it in the name.
- Exact same-allocation before/after retry `6970726` compares full-GDN control
  `5d3f787` with integrated communication/overlap candidate `2bdfc92` on HSG
  `short`, two same-fabric trays/eight GPUs, and an explicit one-hour limit.
  It retains logical 128K randomized-resident perf state without prefill, hard
  MB2 `18+19`, and FP4 experts. It completed cleanly `0:0` in 16:17 with 15/15
  samples for both sides. Control measured 38.680765733 ms strict CUDA and
  478.273882 TPS/GPU; candidate measured 38.356154733 ms and 482.321550
  TPS/GPU, a 0.324611-ms / 0.839% latency win and 0.846% throughput win.
  The result correctly records perfect routing: the user clarified that perf
  mode must force the deterministic perfect router, while normal routing is
  reserved for correctness alignment. Retain this accepted performance win;
  the transformations themselves contain no perfect-route-only assumption.
  A briefly attempted harness change to permit normal routing in perf mode was
  reverted in all clean remote sources after that clarification. Final candidate
  head `b7ffae9` has a tree identical to measured head `2bdfc92`. Corrected-run
  job `6971147` failed in 0:22 before either source ran because fresh empty pool
  directories were absent; dependent `6971152` never allocated. Normal-perf
  retry `6971171` completed only its unnecessary control at 38.774573 ms and
  was canceled at 11:25 during candidate startup; dependent `6971175` never
  allocated. None of those jobs changes the accepted perfect-perf result.
- Qwen3 non-regression is isolated to the general Blackwell selector-reduction
  change. A clean control was cloned from integrated head `2bdfc92` and reverts
  only commit `f16a205`; its original remote head is `7934dfd`. Thus control and
  candidate have identical current harnesses and identical Qwen3.5-only code,
  differing for Qwen3 in one CUDA selector file. Job `6970920` was prepared as
  an exact same-allocation Qwen3 128K A1:F1 EP4/ATP1 batch-6 MB2 perf-mode pair
  on HSG `short`, two trays/
  eight GPUs, explicit one-hour limit. Logical 128K state is resident/random
  without prefill; this prevents overlapping GPU allocations and provides the
  required Qwen3 before/after evidence before final acceptance. Job `6970920`
  reached a two-tray allocation but stopped in 0:22 before model startup because
  the newly named prompt-cache root did not exist; it contains no performance
  evidence. The root was created and retry `6971028` reached model startup but
  was canceled at 3:03 during the temporary routing-contract misunderstanding;
  it contains no measurement. After restoring forced-perfect perf behavior,
  clean control/candidate heads `d276203`/`b7ffae9` again differ for Qwen3 only
  in the selector CUDA file. Job `6971374` used an unsupported FP4 Qwen3 weight
  configuration and both sides failed identically during model load because
  `model.layers.*.mlp.experts.gate_up_proj_scale` is absent; it contains no
  performance sample. Supported-FP8 retry `6971659` then failed before model
  startup because the reusable online-pool validator still restricted its
  already-generic FP8/FP4 launch contract to FP4. The validator now accepts
  either supported MegaMoE expert-weight dtype. Independent experiment source
  snapshots apply that same one-line harness correction to both sides, heads
  `3f6172e`/`32e96b5`, so the active alignment source remains untouched. Job
  `6971742` then exposed a substantive Qwen3 loader regression before graph
  capture: per-expert split FP8 scales were flattened to `gate_proj_scale` and
  `up_proj_scale` before projection merging, but `_get_merge_info` recognized
  only dotted projection suffixes, leaving required `gate_up_proj_scale`
  tensors absent. The general merge mapper now fuses flattened expert scale
  names too; focused control and candidate checks both pass. Identically fixed
  source heads are `88f080f`/`f2c1c3d` and still differ in Qwen3 performance
  code only by the selector CUDA reduction. Job `6971981` retries the same
  perfect-routing 128K-resident/no-prefill A1:F1, EP4/ATP1, batch-6 MB2 `3+3`
  pair on two trays/eight GPUs, HSG `short`, and an explicit one-hour limit.
  It completed cleanly `0:0` in 13:32. Both sides retained 15/15 strict samples
  with zero outliers and recorded perf-mode-forced perfect routing, randomized
  resident 128K state/no prefill, FP8 expert weights, and MB2 `3+3`. Control
  measured 31.267700400 ms and 95.945655 TPS/GPU; selector candidate measured
  31.038431267 ms and 96.654369 TPS/GPU. Thus Qwen3 improves by 0.229269133 ms /
  0.733246% latency and 0.708714 TPS/GPU / 0.738662% throughput; the required
  non-regression gate passes. The full-model run also proves the restored FP8
  expert-scale loader path through load, conversion, graph capture, and replay.
  With all jobs stopped, the Qwen3 loader repair and generic FP8/FP4 plan
  validation were folded into the clean primary remote candidate as head
  `147e284`. Relative to aligned/measured `b7ffae9`, these final additions do
  not enter Qwen3.5's preconverted-routed-expert runtime path; relative to the
  Qwen3 A/B candidate, they are exactly the already-exercised loader/harness
  corrections.
- Final normal-routing alignment job `6971380` reached graph capture but failed
  before sampling with `OSError: [Errno 116] Stale file handle` while reading
  Triton cache metadata on `nvl72115-T01`; dependent scorer `6971397` was
  canceled automatically. This is a node/filesystem failure, not an alignment
  mismatch. Retry `6971635` excludes `nvl72115-T01` and `nvl72104-T15` and uses
  the same A3/8K, EP4/ATP1, batch-four-per-attention-lane, MB2 `2+2`, 48-prompt/
  816-generated-token configuration on four trays/16 GPUs, HSG `short`, and an
  explicit one-hour limit. Dependent scorer `6971667` will run only after a
  durable sample. Retry `6971635` completed cleanly `0:0` in 9:10. Its normal-
  routing artifact contains exactly 48 samples, 17 generated tokens for every
  sample, 816 tokens total, and no empty output. Dependent unchanged vLLM
  scorer `6971667` became eligible afterward, but its submission accidentally
  passed the Python scorer itself as the container re-entry shell script and
  failed in 1:01 before model startup. Corrected one-tray scorer `6971875` uses
  the established `score_vllm_sharded.sbatch` for both host and container entry
  and writes to `vllm-score-final-r2/alignment.json`. The Qwen3 FP8 A/B retry
  remains independent, preserving the two-job concurrency cap.
- Corrected scorer `6971875` completed cleanly `0:0` in 11:08 and produced the
  complete 48-prompt/816-token report:
  overall top-10 and top-100 hit rates are exactly 1.0, top-1 agreement is
  0.980392, average/maximum vLLM rank is 1.035539/6, and mean log-probability/
  perplexity is -0.106862/1.112781. The accumulated Qwen3.5 communication,
  fusion, and early-publication overlap stack therefore passes the required
  A3/8K normal-routing correctness gate.

## Post-alignment TensorRT-LLM optimization-gap target set

This is the ranked non-parameter exploration set after the accepted MB2 stack.
Measure AFD before/after on identical inputs; TensorRT-LLM is the structural
reference, not an absolute latency gate. Perf mode must use forced-perfect
routing and randomized resident 128K state without prefill; correctness uses
the accumulated 8K normal-routing alignment gate. Preserve arbitrary routing
in implementation and keep Qwen3 guarded or explicitly non-regressed.

1. Fuse routing into a single-pass, multi-CTA producer with programmatic
   dependent launch across gate logits, top-k/renormalization, route counts and
   MegaMoE metadata. Avoid materializing all 512 logits and the second exposed
   launch where possible while preserving exact tie and renormalization rules.
   This attacks the remaining AFD selector/route-preparation gap and repeats on
   every routed layer and MB.
2. Redesign one-sided dispatch around rank-major symmetric receive buffers and
   destination-rank deduplication: send each source token once to a destination
   rank even when several of its top-10 experts live there, then let MegaMoE
   consume the rank-major raw receive buffer directly. AFD already has FC2
   write directly into symmetric per-route return slots and pulls/reduces them
   inside the same MegaMoE kernel, so direct symmetric FC2 output is closed;
   the real remaining structural gap is destination deduplication and its
   coordinated scheduler/return layout. This should scale especially with EP,
   top-k, and larger A:F ratios.
3. Replace the current serial top-k accumulation in the otherwise zero-copy
   combine path with a vectorized pairwise/tree reduction into a stable final
   token slot. Keep that final layout directly consumable by the following
   fused epilogue and avoid adding another exposed launch.
4. Move producer output directly into symmetric transport storage where a GEMM
   or attention epilogue can do so safely. The current fused QKV and O
   publishers eliminate preparation kernels but still reread locally
   materialized producer output; direct epilogues remove that last launch and
   memory pass and become more valuable with communication fan-out.
5. Re-profile the retained Qwen3.5 GDN stack and close only an observed
   recurrent-core gap: fuse/vectorize convolution-to-state handoff or recurrent
   update/layout if the post-stack kernel still exceeds TensorRT-LLM's roughly
   17-us wide-vector implementation. Do not repeat packed QKVZ/BA projections,
   direct packed convolution, scale reuse, gate overlap, or fused gated
   RMSNorm+FP8 work already implemented and component-tested.
6. Defer the final, already-ready shared-expert add into the next layer's
   high-parallelism residual/RMSNorm consumer (or its fused linear-input
   publisher). Retain shared-expert computation and gate multiplication on the
   hidden auxiliary stream, preserve the exposed BF16 rounding boundary, and
   remove the standalone read/add/write without extending MegaMoE's
   low-parallelism combine epilogue.
7. Replace exposed workspace reset, memset, route-swizzle, and epoch barrier
   kernels with epoch tags or fuse them into existing producers, but only where
   a fresh trace proves they remain on the critical path.

Primary technique references: TensorRT-LLM's EP scaling write-up documents PDL,
multi-field A2A fusion, shared-add/local-reduction fusion, and removal/fusion of
memset, routing-finalize, and swizzle work:
`docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md`.
Its one-sided NVLink write-up documents rank-major receive layout, one send per
source/destination rank, raw-buffer MoE consumption, stable output slots,
pairwise zero-copy combine, vectorized transactions, epoch barriers, and
optional FP8 combine:
`docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md`.

Closed/non-targets: MB1, parameter sweeps, perfect-route-only algorithms, 128K
prefill, the slower TRT ring-kernel port, FP8 attention-O transport (alignment
failure), moving O projection to attention, and already-parity post-attention
prefix work. Retain the shared-expert overlap, fused gate/O publication,
selector renorm specialization, fused QKV/linear-input publishers, packed GDN
stack, refined PDL, and early next-MB releases unless a same-code comparison
disproves them. At the user's direction, remove the former exploration-list
items for low-precision combine, quantized LM head, and EPLB; those are not
system changes being pursued.

- First post-alignment router experiment: trace/source inspection shows a
  comparable TRT-LLM gate-GEMM + custom route + MegaMoE-prepare span of about
  15.1 us (8.0 + 4.7 + 2.4 us), versus AFD's fused gate producer + route-
  publishing selector at about 17.9 us (5.5 + 12.5 us). AFD's selector was
  unconditionally publishing 512 per-expert remote count stores per source,
  although rows 18/19 top-10 can activate at most 180 experts. The candidate
  skips count==0 stores. This is routing-independent: the symmetric buffer is
  zero-initialized, MegaMoE clears every per-source receive-count slot before
  reuse, and the existing cleanup barrier plus per-destination release epoch
  remain unchanged. Clean remote control/candidate heads are `147e284` and
  `58fe73f` under `router_sparse_count_publish_ab/source/`; the focused source
  contract passes. Parallel HSG `short` submissions use explicit one-hour
  limits: job `6972329` is the exact Qwen3.5 128K resident/no-prefill, A1:F1,
  EP4/ATP1, hard-MB2 18+19 paired perf gate (perf forces perfect routing), and
  job `6972328` is rows-18/19 normal-routing MegaMoE ABBA. Outputs are under
  `router_sparse_count_publish_ab/perf/` and `component_normal/` respectively.
  Both jobs completed cleanly, but this candidate is rejected and removed
  locally. Normal-routing ABBA is the decisive general-path gate: row 18 moved
  from 145.995697 to 146.432831 us (+0.437134 us / +0.2994%) and row 19 from
  147.782806 to 148.472496 us (+0.689690 us / +0.4667%). The full perfect-route
  pair moved from 38.477900600 to 38.386355733 ms (-0.091545 ms / -0.2379%),
  too small and contradictory to accept over the isolated regression. Dense,
  coalesced 4-KiB count publication is more efficient than divergent sparse
  stores at these shapes despite writing more bytes.
- Second post-alignment router experiment implements a complete, scoped PDL
  chain modeled on TRT-LLM rather than adding a launch attribute alone. The
  gate producer releases the dependent selector after every block has written
  logits; the selector waits at entry, publishes all route indices/count
  increments, then releases MegaMoE before the final rank-ready publication.
  Only the MegaMoE DeepGEMM launch is forced PDL-enabled. Its persistent kernel
  prefetches TMA descriptors and initializes cluster/shared-memory barriers
  before `cudaGridDependencySynchronize()`, placed immediately before its first
  route-data dependency. Non-route-prepared calls retain ordinary completion
  ordering and Qwen3 uses the same general mechanism rather than a Qwen3.5 or
  perfect-route special case. Clean isolated control/candidate heads are
  `147e284`/`d2f5a53` under `router_pdl_chain_ab/source/`; the focused source
  contract passes locally and remotely. Parallel HSG `short` jobs use explicit
  one-hour limits: `6972554` runs rows-18/19 normal-routing MegaMoE ABBA and
  `6972553` runs the exact Qwen3.5 128K resident/no-prefill, A1:F1, EP4/ATP1,
  hard-MB2 18+19 paired perf gate (perf forces perfect routing).
  Normal-routing job `6972554` completed cleanly `0:0` in 7:09 and all
  prepared/unprepared-route numeric and repeated-buffer cases passed. Its ABBA
  timing is mixed: row 18 control/candidate means are 146.949837/148.070465 us
  (+1.120628 us / +0.7626%), while row 19 is 148.885040/146.624512 us
  (-2.260529 us / -1.5183%). Candidate row 18 itself spans 151.004959 and
  145.135971 us while the controls are tight, so this component sample is too
  noisy to decide retention. Use the full repeated 60-layer MB2 pair plus the
  structural overlap argument; do not claim an isolated-kernel win from this
  ABBA.
  Full paired job `6972553` completed cleanly `0:0` in 15:12 with 15/15
  samples, but rejects this first PDL placement: control measured 38.274382667
  ms and 483.352015 TPS/GPU; candidate measured 38.452594600 ms and 481.111878
  TPS/GPU, a +0.178211933-ms / +0.4656% latency regression and -0.4634%
  throughput regression. Do not fold remote head `d2f5a53`. The first version
  unnecessarily added a producer block barrier and then waited for full
  selector-grid completion inside MegaMoE despite the existing release/acquire
  route-ready epoch. A refined candidate removes both waits while retaining
  scoped PDL launch attributes and explicit route-ready ordering.
- Next router candidate is prepared locally but not yet GPU-compiled or
  measured: port TRT-LLM's local-candidate strategy into AFD's selector. Each
  lane bitonically sorts its 16 experts once using a packed monotonic float/
  lower-index key, then only the winning lane shifts its list after each of the
  ten fast `redux` warp maxima. This replaces ten repeated 16-way local scans
  while preserving AFD's existing lower-local-index then lower-lane tie rule.
  CPU simulations covering 1,000 random/tied/signed-zero local lists and 1,000
  full 512-expert/top-10 warps exactly match the old selected expert sequence.
  It is now layered with the refined PDL placement so the next same-source
  comparison tests the useful overlap mechanism without the two redundant
  waits rejected above.
  Clean isolated control/candidate heads for that cumulative comparison are
  `147e284`/`fc2bdf5` under `router_topk_pdl_refined_ab/source/`. The focused
  source contract passes locally and remotely. Parallel HSG `short` jobs use
  explicit one-hour limits: `6972805` runs rows-18/19 normal-routing MegaMoE
  ABBA and `6972804` runs the exact 128K resident/no-prefill, A1:F1, EP4/ATP1,
  hard-MB2 18+19 perf pair with perf-mode-forced perfect routing.
- Component job `6972805` terminated in 4:50 during candidate trial 1. Its
  control passed every numeric case and measured 148.236/149.859 us at rows
  18/19; the candidate compiled but produced NaNs in the first unprepared-route
  row-3 case. The cause is scoped, not a sort arithmetic mismatch: the candidate
  forced MegaMoE's PDL attribute for every API call while removing the consumer
  grid-dependency wait, but an unprepared-route call has no route-ready epoch to
  replace ordinary stream ordering. The local repair enables the MegaMoE PDL
  launch only when `route_ready_dispatch` is true; all other calls again wait
  normally in stream order. The focused contract passes. Job `6972804` uses
  the production prepared/rank-ready path and remains valid to measure while a
  corrected normal-routing component rerun is prepared after its result. Clean
  corrected candidate head `d4ea8c1` is isolated under
  `router_topk_pdl_refined_v2_ab/source/candidate`; normal-routing ABBA job
  `6972988` reuses only the completed control build cache, otherwise retains the
  same rows-18/19, FP4, 150-SM, 1,000-replay, HSG `short`/one-hour contract.
- Corrected component job `6972988` proved that the scoped launch repair fixes
  the first unprepared-route failure: candidate trial 1 passed five successive
  prepared/unprepared cases. It then produced NaNs on the later bucket-1036,
  rows-3 prepared reuse and exited `FAILED 143:0`. This repeats the earlier
  sort-once candidate's intermittent reuse corruption, so reject the packed
  bitonic sort again and restore the retained `redux` winner reduction with the
  original local scan. The perfect-routing two-node job `6972804` was canceled
  at 15:24 because a fast result cannot make this general implementation safe.
  Preserve the conditional PDL repair and isolate refined PDL without the sort.
- Refined-PDL-only candidate head `7ed26e4` is clean and isolated under
  `router_pdl_refined_only_ab/source/candidate`. It differs from accepted
  control `147e284` only by gate-to-selector PDL, selector-to-route-ready
  MegaMoE PDL, the scoped DeepGEMM launch control, and the removal of the
  redundant consumer grid wait; the accepted `redux` local scan is unchanged.
  Parallel HSG `short`/one-hour jobs `6973080` (normal-routing rows-18/19 ABBA)
  and `6973079` (exact 128K resident/no-prefill hard-MB2 perfect-routing pair)
  were submitted as the acceptance runs. Job `6973079` failed before entering
  its GPU worker because the two paired online-case pools had not been
  initialized (`run_afd_paired_sources.sbatch` preflight line 43); it produced
  no benchmark result. Both one-case pools were then initialized from the
  unchanged plan and the identical target pair was resubmitted as `6973167`.
  Component job `6973080` completed cleanly. Every prepared/unprepared
  bucket-518/1036 numeric case passed in all four ABBA trials (unlike either
  sort experiment). At exact Qwen rows 18/19, candidate means are
  143.585541/145.478149 us versus control 145.563446/147.603203 us: wins of
  1.977905 us (1.3588%) and 2.125053 us (1.4397%). Exact target job `6973167`
  then completed cleanly with 15 retained samples/source: control
  38.528663 ms/480.161997 tokens/s/GPU and candidate 38.520603 ms/480.262474.
  The end-to-end change is a small 0.008061-ms (0.0209%) win because the
  1.98--2.13-us routed-span saving is largely hidden by the imbalanced
  attention/model pipeline, but it is non-regressing, structurally removes a
  producer/consumer launch bubble, and has clean ordinary-routing coverage.
  Retain refined PDL head `7ed26e4`.
- The next independent local candidate implements target-set item 6 without
  disturbing the accepted auxiliary-stream overlap. Each MegaMoE lane owns a
  graph-stable int64 readiness vector with one element per layer. Layer 0
  resets the entire vector once per decode, avoiding the original draft's one
  reset launch per layer; the auxiliary stream computes each existing BF16
  gated shared output and performs a GPU-release store into that layer's flag.
  Only the Qwen3.5 TP1 call passes the tensor and flag into a
  `kFuseSharedOutput` MegaMoE specialization, whose combine warps
  acquire the flag and add shared BF16 values while forming the final store.
  The routed FP32 sum is rounded to BF16 before the shared add, preserving the
  current `routed_bf16 + shared_bf16 -> bf16` ordering. The main stream retains
  the done-event wait after MegaMoE solely to close the CUDA-graph stream fork;
  it cannot stall because MegaMoE already consumed the release. Qwen3 and TP>1
  stay on the non-fused specialization/current add path. A reusable same-rank
  smoke now checks bitwise equality against the exposed BF16 add. The two
  focused source contracts and `git diff --check` pass locally. Clean isolated
  candidate head `27fecfb` is based on refined-PDL head `7ed26e4` under
  `qwen35_shared_combine_pdl_ab/source/candidate`. One-tray HSG `short`,
  one-hour-limit job `6973304` completed cleanly in 3:07: FP4 compiled, ordinary
  and prepared-route numeric cases passed, and the fused shared store is
  bitwise equal to the exposed BF16 add (`shared_max_abs=0`). Exact target pair
  `6974330` compares refined PDL control `7ed26e4` against shared-fusion
  candidate `27fecfb` with 128K resident/no-prefill hard MB2 18+19 perf mode,
  two trays, HSG `short`, and an explicit one-hour limit. Concurrent Qwen3
  non-regression pair `6974514` compares accepted pre-PDL head `147e284`
  against the complete `27fecfb` stack at logical 128K, batch 6, hard MB2 3+3,
  EP4/ATP1, FP8 experts, perf-mode perfect routing, two trays, `short`, one
  hour. This covers both PDL and the unchanged non-fused shared-output API path.
- The next communication candidate should implement target-set item 2 as a
  coordinated layout change, not as a perfect-routing shortcut. After normal
  top-k is known, each selector token warp can use `__match_any_sync` on routed
  destination ranks and vector-write the already-quantized FP8 token plus UE8M0
  scales once into each unique destination's stable `[source_rank, token]`
  symmetric receive slot. Route metadata can carry its FP32 weight alongside
  the source token/top-k index. The route-ready system release then orders all
  fields, and MegaMoE's existing expert scheduler locally expands references
  from the rank-major buffer instead of issuing one remote TMA payload pull per
  expert route. At top-10/EP4 this reduces payload fanout from ten copies to at
  most four for arbitrary routing and usually one in the deterministic perf
  distribution, while preserving the current direct symmetric FC2 return and
  combine path. It requires matching host/device buffer layouts and dedicated
  prepared-route numeric coverage before any end-to-end claim.
- Exact-target job `6974330` rejects the first shared-add fusion placement even
  though its smoke/numeric coverage was clean. Refined-PDL control measured
  38.502850 ms/480.483916 tokens/s/GPU, while head `27fecfb` measured
  43.311469 ms/427.138598: a 4.808619-ms (12.49%) latency regression. The
  readiness acquire was at entry to every persistent MegaMoE epilogue warp;
  with the shared branch still taking roughly 47 us on its two reserved SMs,
  150 CTA epilogue warps spin and consume issue capacity before routed MMA.
  The approximately 40-us/layer delay over 120 layers accounts for the full
  end-to-end loss. Reject this placement, not the fusion objective.
- Clean follow-up head `1bcb98f` under
  `qwen35_shared_combine_late_acquire_ab/source/candidate` moves the readiness
  acquire after the first routed top-k chunk's loads/reduction and immediately
  before the first shared value is read for the final combine store. This keeps
  the bitwise-preserving BF16 add but prevents epilogue warps from spinning
  while routed work is ready. One-tray FP4 smoke/numeric job `6975343` is on
  HSG `short` with an explicit one-hour limit; run the exact hard-MB2 18+19
  resident-128K perf pair only after that gate passes.
- Qwen3 non-regression pair `6974514` completed cleanly for the unchanged
  logical-128K hard-MB2 3+3 EP4/ATP1 FP8 perf case. The accepted pre-PDL
  control measured 30.898669 ms/97.091561 tokens/s/GPU; the complete
  PDL/optional-shared-API candidate measured 30.814163 ms/97.357828, a
  0.084506-ms (0.274%) latency win. The Qwen3 false-specialization path is
  therefore non-regressing and should remain guarded from Qwen3.5-only shared
  fusion behavior.
- Late-acquire shared-fusion smoke job `6975343` completed cleanly in 2:49:
  FP4 compiled, ordinary and prepared-route numeric cases passed, and the
  fused shared store remains bitwise identical to the exposed BF16 add
  (`shared_max_abs=0`). Exact hard-MB2 18+19 job `6975405` now compares
  refined-PDL control `7ed26e4` with late-acquire head `1bcb98f` using the
  same resident/no-prefill logical-128K perf case, two trays, HSG `short`, and
  a one-hour limit.
- Exact late-acquire job `6975405` completed cleanly but rejects head
  `1bcb98f`: refined-PDL control measured 38.554436 ms/479.841024
  tokens/s/GPU versus candidate 39.082489 ms/473.357777, a 0.528053-ms
  (1.37%) regression. Nsight Systems kernel sums from the model rank identify
  the cause: the shared specialization grows MegaMoE from 156.701 to
  169.774 us/layer (+13.073 us), while the removed standalone BF16 add costs
  only 1.757 us/layer and the new release kernel only 1.390 us/layer. Thus
  fusing the add into MegaMoE's low-parallelism one-warp-per-token combine is
  structurally wrong even when its acquire is late and ready. Preserve both
  failed heads as evidence and move the same semantics to the next layer's
  high-parallelism residual/RMSNorm consumer (including the existing fused
  linear-attention publisher), where `bf16(routed + shared)` can be formed
  before the residual update without slowing MegaMoE. SQLite exports are under
  `qwen35_shared_combine_late_acquire_ab/perf/analysis/`.
- The rank-major communication draft was corrected after comparison with
  TensorRT-LLM's actual `vectorized_dispatch_impl`: each 128-bit FP8 or scale
  vector is now loaded from local memory once and the register value is fanned
  out to every unique destination rank. The earlier loop deduplicated NVLink
  traffic but unnecessarily reloaded the source once per destination. Clean
  isolated head `1829398` under `qwen35_rank_major_dispatch_ab/source/candidate`
  is based on late-acquire head `1bcb98f`. It adds a shared host/device
  rank-major buffer layout, destination-rank deduplication in the selector,
  co-published FP32 route weight, and direct local consumption by MegaMoE;
  arbitrary normal routing remains supported and the existing ordinary pull
  path remains unchanged. Focused source contract and `git diff --check` pass.
  One-tray FP4 compile/prepared/unprepared numeric job `6975417` is on HSG
  `short` with a one-hour limit. Do not spend a full pair until this first GPU
  gate is clean. At hidden 4096 the FP8 row plus UE8M0 scales is 4,224 bytes:
  top-10 expert-major pulling moves 42,240 payload bytes/token, whereas the
  rank-major path moves 4,224 bytes in the exact perfect distribution. Under
  independent uniform EP4 routing, the expected unique destinations are
  `4*(1-(3/4)^10)=3.775`, or about 15,946 bytes/token, a theoretical 62.25%
  arbitrary-routing payload reduction before accounting for metadata.
- Rank-major v1 smoke `6975417` passed the ordinary-route numeric case, then
  made no progress for more than six minutes in its first prepared-route case
  and was canceled at 9:17 to conserve the tray. The defect was local and
  deterministic: the scale payload has only eight 128-bit vectors, so lanes
  8--31 skipped the scale loop while lanes 0--7 called a full-mask warp
  shuffle inside it. That prevents selector completion and therefore the
  route-ready publication. Preserve head `1829398` as the failed-attempt
  source. Clean follow-up head `33f9851` under
  `qwen35_rank_major_dispatch_v2_ab/source/candidate` iterates vector tiles
  warp-convergently and predicates only each lane's load/store; every lane now
  participates in destination-pointer shuffles. Focused source contract and
  diff checks pass. V2 smoke `6975609` reuses only the identical DeepGEMM build
  and recompiles the changed selector in a fresh TVM cache on HSG `short`, one
  tray, one-hour limit.
- Rank-major v2 smoke `6975609` completed cleanly in 1:59: ordinary and
  prepared normal-routing numeric cases passed and the formerly divergent
  scale fanout no longer stalls. Same-fabric normal-routing ABBA job `6975709`
  also passed every prepared/unprepared bucket-518/1036 numeric case in all
  four trials, but the integrated stage regresses at the exact rows: candidate
  means are 146.924/147.412 us for rows 18/19 versus control
  143.510/143.848 us, losses of 3.414 us (2.38%) and 3.564 us (2.48%). The
  payload-volume reduction is valid, but one selector warp per token exposes
  push latency that the prior persistent MegaMoE path overlaps with remote TMA
  pulls. Do not accept this layout yet. Rebase it away from rejected shared
  fusion before any exact full run, measure the one-destination target case,
  and investigate dispatch geometry/producer overlap rather than routing
  assumptions or parameter tuning.
- Because shared-Mega heads `27fecfb`/`1bcb98f` are rejected, the communication
  candidate was rebased onto retained refined-PDL head `7ed26e4`. The rank
  commits cherry-picked cleanly in production sources; only their mixed
  shared-fusion test context was intentionally omitted. Clean isolated head
  `210f5c0` is under `qwen35_rank_major_pdl_rebase_ab/source/candidate` and has
  no shared-output MegaMoE specialization. One-tray normal-routing FP4 smoke
  `6975870` uses a fresh DeepGEMM ABI cache and the identical validated selector
  cache on HSG `short`, one-hour limit. Submit the exact full pair only after
  this rebased-source compile/numeric gate passes.
- Rebased rank-major smoke `6975870` completed cleanly in 2:43: both ordinary
  and prepared normal-routing FP4 numeric cases passed from retained-PDL base
  head `7ed26e4`. Exact hard-MB2 18+19 job `6975983` now compares that control
  against rebased rank-major head `210f5c0` at logical 128K, randomized
  resident state/no prefill, perf-mode routing, EP4/ATP1, two trays, HSG
  `short`, and an explicit one-hour limit. This is the decisive one-destination
  target measurement; normal-routing component evidence remains a 2.4%
  regression and prevents general acceptance without further refinement.
- Shared-output consumer-fusion candidate head `a342988` is isolated under
  `qwen35_shared_consumer_fusion_v2_ab/source/candidate` from retained-PDL head
  `7ed26e4`. MegaMoE now returns routed and auxiliary shared outputs separately.
  The next Qwen3.5 layer preserves `bf16(routed + shared)` and folds it into
  either the existing fused linear-attention residual/RMSNorm publisher or a
  new 256-thread, one-CTA-per-token full-attention residual/RMSNorm kernel;
  final norm consumes the last deferred output the same way. Qwen3 and
  non-Mega paths retain their established semantics. The source contract suite
  passes locally (24 tests), `git diff --check` is clean, and one-tray HSG
  `short`/one-hour compile plus nine-iteration BF16 transport/numeric smoke is
  job `6976198`. Initial job `6976163` was canceled after roughly one minute
  because its source was amended with a final-stream lifetime fix just after
  launch; no result from that mutable-source interval will be used. Do not
  submit the exact target pair until the immutable v2 GPU gate passes.
- TensorRT-LLM confirms the consumer-side placement directly in
  `moeAllReduceFusionKernels.cu`/`allreduceOp.cpp`: its MoE finalize path folds
  routed top-k reduction, shared-expert output, residual add, and RMSNorm in one
  high-parallelism consumer kernel. This validates the direction of
  `a342988`; AFD initially preserves its current intermediate BF16 rounding for
  alignment, while target-set item 3 can later move the routed top-k reduction
  into the same consumer boundary.
- Prepared rank-major follow-up head `63d6b68` is isolated under
  `qwen35_rank_major_early_pdl_ab/source/candidate`. The current rank-major
  selector performs its payload pushes before triggering the dependent
  MegaMoE launch, exposing the whole push tail despite MegaMoE's explicit
  route-ready acquire. The follow-up triggers MegaMoE immediately after the
  selector grid becomes resident; MegaMoE can overlap descriptor prefetch,
  barriers, cluster sync, and TMEM allocation with all top-k/metadata/payload
  work, then still cannot consume anything until the unchanged system-release
  route-ready epoch. The 24-test source contract and diff check pass. Normal-
  routing FP4 rows-18/19 ABBA job `6976399` completed cleanly in 6:39 but
  rejected this placement: control head `210f5c0` averaged 86.328/86.423 us,
  while early-PDL head `63d6b68` averaged 86.413/86.545 us, regressions of
  about 0.10%/0.14%. Starting the dependent grid before the selector payload
  pushes therefore does not recover the exposed rank-major selector tail.
- Exact rank-major job `6975983` completed cleanly but rejects head `210f5c0`
  even for perf-mode routing: retained-PDL control measured 38.500876 ms and
  480.508552 tokens/s/GPU, while rank-major measured 38.643666 ms and
  478.733051, a 0.142790-ms (0.371%) latency regression. This confirms that
  reduced NVLink payload alone is insufficient when the push tail is exposed
  before MegaMoE launch. Model-rank Nsight sums sharpen the diagnosis: the
  selector grows from 11.724 to 16.429 us/layer (+4.704 us), while rank-local
  MegaMoE falls from 159.129 to 154.188 us/layer (-4.942 us). The summed kernels
  are nominally 0.238 us/layer faster, but the selector shift is exposed at the
  attention/model overlap boundary and loses end-to-end. Early-PDL head
  `63d6b68` also loses in the component test above; retain accepted PDL head
  `7ed26e4` rather than either rank-major head.
- Immutable consumer-fusion head `a342988` passed its one-tray GPU gate in
  3:57: the changed CUDA transport compiled and all nine repeated Qwen3.5
  QKV, linear-input, shared-add/RMSNorm, and BF16 O numeric/publication rounds
  completed successfully across four ranks. Exact hard-MB2 18+19 logical-128K
  perf job `6976389` completed cleanly in 13:43 and accepts the candidate:
  retained-PDL control measured 39.171955 ms and 472.276654 tokens/s/GPU;
  consumer fusion measured 38.577778 ms and 479.550684 tokens/s/GPU. This is a
  0.594177-ms (1.517%) latency reduction and 1.540% throughput gain. Perf mode
  reports `perfect_routing=true` and `perfect_routing_forced_by_perf_mode=true`;
  it still executes the normal gate GEMM, softmax, top-k, dispatch, expert, and
  combine work. Accept `a342988` as the new accumulated Qwen3.5 source. Qwen3
  remains on the unchanged non-hybrid consumer path. Alignment remains deferred
  until another useful optimization is accumulated.
- The next local-only prototype targets combine-kernel geometry directly. For
  Qwen3.5's top-10/hidden-4096 specialization, it uses one CTA per token,
  issues all ten 16-byte contribution loads before conversion, performs a
  shallow pairwise FP32 reduction, and stores directly to the stable token
  output. The retained path uses one warp per token with serialized staged TMA
  accumulation and exposes only three CTAs at rows 18/19. Every other model
  shape retains the existing path, so Qwen3 is structurally guarded. This
  prototype still requires source-contract, compile, and numeric gates before
  any performance submission. Direct-load head `542f0db` passed all 24 source
  contracts and every compile/numeric case in one-tray job `6976638`, but is
  rejected on component performance: accepted-PDL control averaged
  83.110/83.022 us at rows 18/19, while direct-load CTA pairwise averaged
  83.638/83.720 us, regressions of 0.528/0.698 us (0.636%/0.840%). Preserve
  one-CTA/token and the pairwise tree in a v2 local prototype, but restore TMA:
  issue all ten top-k transfers back-to-back into shared memory, wait on one
  transaction barrier per valid slot, then reduce locally and direct-store the
  final token. This isolates remote direct loads as the v1 loss mechanism and
  keeps every other model shape on the retained path. The v2 is layered on
  accepted consumer-fusion head `a342988` and isolated at clean remote head
  `180f3d8`; all 24 source contracts pass. One-tray HSG `short`/one-hour ABBA
  job `6976808` provides its compile/numeric/component gate against the
  byte-identical retained MegaMoE kernel at PDL head `7ed26e4`. Job `6976808`
  completed cleanly in 5:22 and every numeric/reuse case passed. Control
  averaged 84.242/84.235 us at rows 18/19; TMA pairwise averaged
  84.202/84.160 us, small wins of 0.040/0.076 us (0.047%/0.090%). Treat this as
  a structurally sound, non-regressing component candidate rather than a large
  measured win: it restores bulk TMA efficiency, increases rows-18/19 token
  parallelism, and shortens top-10 arithmetic depth, so its value should be
  greater when the FFN side is exposed by a larger A:F ratio. Accumulate it
  with any successful route publisher before the next exact E2E/alignment gate.
  Because the route-publisher first trial already trends slower, exact hard-MB2
  logical-128K pair `6977015` independently compares accepted consumer fusion
  `a342988` with consumer+TMA-pairwise `180f3d8` on two HSG `short` trays. It
  uses the same resident randomized state, no prefill, and perf-mode-forced
  perfect routing under the explicit one-hour limit.
- A lower-risk route follow-up is now preferred before another rank-major
  payload attempt: keep the gate producer and selector math unchanged, but
  move route atomics/metadata plus the dense count finalization into a PDL-
  chained one-CTA-per-token publisher. The current selector has only two large
  CTAs at rows 18/19 and measures about 11.7 us because top-k and remote route
  publication share that low-parallelism tail; TRT-LLM separates its roughly
  4.7-us custom route and roughly 2.4-us MegaMoE preparation. The publisher can
  release MegaMoE immediately and let descriptor/barrier/TMEM initialization
  overlap its arbitrary-routing publication. If this general split wins, add
  rank-major payload fanout inside the already-parallel publisher as a second
  isolated increment; this directly addresses why rejected rank-major head
  `210f5c0` saved about 4.94 us in MegaMoE but exposed about 4.70 us in the
  selector.
- The general route-publisher split is implemented on accepted consumer-fusion
  head `a342988` at clean remote head `9c6d612`. The selector still writes the
  exact ids, weights, and symmetric weight stage, then PDL-releases a new
  256-thread, one-CTA-per-token publisher. That publisher waits on selector
  completion, immediately PDL-releases MegaMoE for initialization overlap,
  publishes the unchanged per-route source indices, and uses the established
  release/acquire last-block ticket to emit dense per-expert counts plus the
  unchanged rank-ready epoch. Prepared routing with both rank-ready modes is
  supported; normal/perfect routing math is unchanged. All 24 source contracts
  and `git diff --check` pass. Concurrent one-tray HSG `short`/one-hour ABBA job
  `6976864` completed cleanly in 5:38 and every numeric/reuse case passed, but
  rejects the split. Control averaged 83.896/83.755 us at rows 18/19; the
  dependent publisher averaged 84.907/84.629 us, regressions of 1.011/0.874 us
  (1.205%/1.043%). The added launch costs more than one-CTA-per-token route
  parallelism saves, so keep route publication fused into the selector and do
  not layer rank-major payload fanout onto this split.
- Exact hard-MB2 logical-128K job `6977015` completed cleanly in 14:19 on HSG
  `short`. Both sides used resident randomized state/no prefill, perf-mode-
  forced perfect routing, and the required 15 retained steps. Accepted
  consumer fusion measured 38.299076 ms and 483.040371 tokens/s/GPU; consumer
  fusion plus TMA pairwise measured 38.299239 ms and 483.038314 tokens/s/GPU,
  only +0.000163 ms/+0.000426%. Together with the isolated combine wins at
  both hard-MB2 row counts, accept head `180f3d8` as a structural non-
  regression: one CTA/token, ten back-to-back bulk TMA loads, and a shallow
  reduction tree are more favorable when the FFN side is exposed at larger
  A:F ratios. Qwen3 exact non-regression job `6977381` and the unchanged
  Qwen3.5 A3/8K, EP4/ATP1, MB2 `2+2`, 48-prompt normal-routing alignment job
  `6977382` were submitted concurrently to HSG `short`, each with an explicit
  one-hour limit. Both initially queued with the intended 2-/4-tray shapes.
- Source inspection for target-set item 4 identified a concrete general direct
  producer opportunity on Qwen3.5 linear-attention layers. FlashInfer's
  TRT-LLM decode API already accepts caller-owned output, but full-attention O
  cannot use a remote output directly without turning the required local gate
  into a remote read/modify/write. In contrast, the GDN `out_proj` is already
  the final linear-attention producer and each attention DP rank owns one
  contiguous source slice of the model rank's symmetric BF16 `linear_o` slot.
  Letting the FP8 DeepGEMM write that slice directly, followed by a system
  visibility/readiness signal, can remove the standalone BF16 O-copy launch
  without changing arithmetic or assuming perfect routing. Prototype this
  only after the current immutable candidate finishes its acceptance gates.
- The isolated direct-linear-O prototype now exists at clean remote head
  `42fe7e2` under `qwen35_direct_linear_o_ab/source/candidate`, based on accepted
  `180f3d8`. Qwen3.5 decode GDN passes a caller-owned contiguous BF16 output to
  its existing FP8 DeepGEMM `out_proj`, which writes the imported symmetric
  slot directly; a 32-thread signal kernel releases the next MB and publishes
  remote readiness with no payload-copy loop. Full-attention O retains its
  required local gate-and-publish path, prefill retains the old path, and Qwen3
  never enters the optional-output branch. Because Qwen3.5 has 7,168 hidden
  values but 64x256 full-attention O values, the model arena now has distinct
  co-allocated linear/full BF16 slot views so both producer layouts are truly
  contiguous rather than silently relying on padded row strides. The same
  fabric allocation/handle and ready protocol remain. Remote source checks
  pass: 24/24 overlap contracts plus three focused runtime tests for direct
  source partitioning, copy-free signal dispatch, and distinct receive widths.
  The first local runtime-test attempt was invalid because macOS Python 3.9
  lacks this project's `typing.TypeAlias`; the pinned remote Python 3.12 run is
  authoritative. A one-tray fabric compile/numeric smoke is prepared but must
  wait for one of jobs `6977381`/`6977382` to release the two-job concurrency
  cap.
- Accumulated-head alignment generation job `6977382` completed cleanly in
  9:11. It used A3/8K, EP4/ATP1, hard MB2 `2+2`, ordinary routing, no perf
  mode, and produced exactly 48 nonempty samples with 17 generated tokens each.
  Unchanged one-tray vLLM scoring job `6977672` is queued with the exact prior
  top-10 acceptance rule. Qwen3 exact non-regression job `6977381` continues in
  parallel; hold the direct-linear-O GPU smoke until either job releases a
  slot.
- Qwen3 exact non-regression job `6977381` completed cleanly `0:0` in 13:15.
  On identical hard-MB2 logical-128K perf-mode input, the retained Qwen3
  control measured 30.338281 ms / 98.884969 tokens/s/GPU and accumulated
  Qwen3.5-guarded head `180f3d8` measured 29.903162 ms / 100.323840
  tokens/s/GPU: 0.435120 ms (1.434%) lower latency, so the accepted stack does
  not regress Qwen3. This released one of the two allowed job slots.
- Strengthened the direct-linear-O fabric smoke before spending that slot:
  instead of copying an already-produced BF16 tensor into the imported peer
  mapping, a deterministic FP8 DeepGEMM now writes its 7,168-wide BF16 output
  directly into the imported symmetric slot for three alternating-slot
  iterations, signals system visibility/readiness, and is checked bitwise on
  the owning model rank. Remote source tests pass 28/28 (all overlap contracts
  plus four focused direct/hybrid runtime tests) and `git diff --check` passes.
  The clean remote prototype is now head `ccbadb2`. One-tray HSG `short` smoke
  job `6977797` was submitted with an explicit one-hour limit while scorer
  `6977672` continues, preserving the two-job cap.
- Accumulated consumer-fusion plus TMA-pairwise alignment scorer `6977672`
  completed cleanly `0:0` in 10:52. The unchanged A3/8K normal-routing gate
  covers 48 prompts and 816 generated tokens; top-10/top-100 hit rates are
  exactly 1.0, top-1 agreement is 0.982843, average/maximum vLLM rank is
  1.017157/2, and mean log-probability/perplexity is -0.091461/1.095774. The
  accumulated accepted head therefore remains aligned under the fixed
  standard.
- Direct-output smoke `6977797` reached the new numeric path and failed before
  launching the peer-output GEMM because its new validation incorrectly
  required the imported tensor's PyTorch device label to match the producer.
  Imported fabric storage retains its owner label (`cuda:2/3`) while producer
  ranks run on `cuda:0/1`; existing one-sided kernels intentionally address it
  by peer pointer. The correction still requires a contiguous CUDA BF16 tensor
  but permits an owner-device label, matching the intended fabric contract.
  The same 28/28 source/runtime checks and diff check pass at clean remote head
  `844f372`. Cached one-tray HSG `short` smoke retry `6977931` uses the same
  explicit one-hour limit. Per the user's batching direction, keep the
  prepared full A/B idle and accumulate several focused-gated optimizations
  before the next 128K perf and 8K alignment pair.
- Corrected direct-output smoke `6977931` completed cleanly `0:0` in 1:52 using
  the cached builds. Across two attention/model pairs, three alternating-slot
  iterations of a 7,168-wide FP8 DeepGEMM wrote BF16 directly into owner-
  labeled imported fabric storage, the copy-free system-ready signal completed,
  and the model owner matched every value bitwise. This clears the direct GDN
  producer's focused compile/numeric/fabric gate; retain it for the next
  accumulated stack rather than spending a standalone full A/B.
- Next focused routing experiment is isolated from accepted head `180f3d8` at
  clean remote head `ac3620f` under `qwen35_cta_selector_ab/source/candidate`.
  For Qwen3.5's ordinary 512-expert/top-10 renormalized route, the selector now
  launches one 256-thread CTA per token instead of one warp/token in only two
  CTAs at rows 18/19. Eight warps each scan two of the original lane's 16
  candidates; warp 0 reconstructs the same strict-`>` local-index order and
  lower-lane tie rule, then the first ten threads publish unchanged arbitrary
  routes and weights. Perfect mode only performs its established final ID
  substitution after the same top-k work. Other shapes—including Qwen3—retain
  the old kernel. All 25 source contracts and diff checks pass. One-tray HSG
  `short`/one-hour normal-routing ABBA job `6978126` runs compile/numeric and
  exact rows-18/19 component timing against accepted control; no full E2E gate
  is scheduled for this individual item.
- Per the user's updated validation cadence, individual structural candidates
  now receive only focused source/compile/numeric/component gates. Accumulate
  several clean, theoretically justified changes before spending one exact
  logical-128K perf-mode comparison and one unchanged A3/8K normal-routing
  alignment run.
- CTA-selector ABBA job `6978126` completed cleanly `0:0` in 5:25, with every
  prepared/unprepared normal-routing numeric case passing. At exact rows 18,
  control trials averaged 138.727 us and candidate trials 138.160 us, a
  0.567-us / 0.409% reduction. At rows 19, control averaged 140.164 us and
  candidate 139.642 us, a 0.523-us / 0.373% reduction. Accept `ac3620f` for
  accumulation: it improves the exposed route/compute component without an
  extra launch, keeps ordinary routing, and leaves all Qwen3 shapes on the old
  selector.
- Side trace requested during optimization is prepared under
  `qwen3_128k_b4_a56_ep16_perfect_nsys_20260906`. HSG job `6978327` requests
  one full 18-tray/72-GPU segment for exactly 56 ATP1 attention workers plus
  16 FFN ranks (EP16), Qwen3 logical 128K batch 4, hard MB2 `2+2`, FP4
  MegaMoE, perf-mode-forced perfect routing, randomized resident state/no
  128K prefill, and the standard Nsight Systems warmup plus 15 retained decode
  steps. It uses accepted runtime head `180f3d8`, `short`, and a one-hour time
  limit. A task-local runner override only expresses the otherwise unsupported
  56+16 topology; the output metadata records exact `ag56/fg16`, 14+4 trays,
  and 224 prompts. Job `6978327` is pending for resources.
- The direct-linear-O smoke was extended with an exact-shape captured producer
  comparison: Qwen3.5 GDN rows 19, K=4096, N=7168 compares the accepted local
  FP8 DeepGEMM + early turn release + BF16 fabric publish against FP8 DeepGEMM
  writing the imported owner slot + the copy-free readiness/turn signal. Four
  impacted runtime contracts pass at clean remote head `2df9f79` (production
  remains identical to passing head `844f372`). One-tray HSG `short` timing job
  `6978361` is queued with 30 warmups and 1,000 captured replays per ABBA trial;
  it is the second allowed optimization/side-trace job alongside `6978327`.
- Direct-producer ABBA job `6978361` completed cleanly `0:0` in 4:38. The
  accepted local-GEMM + turn release + BF16 publish controls averaged 18.454 us;
  direct peer-fabric GEMM + copy-free signal candidates averaged 14.349 us.
  Eliminating the standalone 19x7168 BF16 publication saves 4.104 us / 22.24%
  on the captured producer sequence, in addition to the prior bitwise peer-
  fabric proof. Accept the direct GDN producer for the accumulated stack.
- Clean accumulated head `a01266f` combines accepted pairwise/consumer fusion,
  direct GDN output, and the accepted one-CTA-per-token selector. All 29 focused
  overlap/direct contracts pass. The next item-2 refinement is clean remote head
  `5fdc048` under `qwen35_rank_major_cta_ab/source/candidate`: only Qwen3.5's
  512-expert/top-10/hidden-4096 prepared path appends rank-major receive slots.
  Each token CTA deduplicates arbitrary selected routes by destination rank;
  its 256 threads load each 16-byte FP8 vector once and fan it plus eight scale
  vectors directly to those destinations. MegaMoE then loads token/scales from
  its local `[source_rank, token]` slot instead of pulling the same source once
  per expert route. Qwen3 retains both its original workspace size and remote-
  pull code path. Source contracts pass 26/26. One-tray normal-routing ABBA job
  `6978444` compares rows 18/19 against `a01266f`; separate control/candidate
  DeepGEMM caches prevent the changed workspace layout from being mixed.
- Early-PDL rank-major follow-up head `724a211` is prepared immutably under
  `qwen35_rank_major_cta_pdl_ab/source/candidate`. It is byte-identical to
  rank-major CTA head `5fdc048` in the MegaMoE API, layout, and kernel, but
  triggers the dependent MegaMoE grid immediately after route metadata becomes
  CTA-visible and before the vectorized rank-major payload fanout. The existing
  system-release route-ready epoch still prevents payload consumption, so only
  initialization overlaps the push. All 26 focused overlap contracts pass;
  exact hashes confirm the shared DeepGEMM build is safe to reuse in the
  pending normal-routing rows-18/19 ABBA comparison once job `6978444` frees a
  submission slot.
- Rank-major CTA job `6978444` completed cleanly `0:0` in 7:00 and every
  normal-routing numeric case passed. Control `a01266f` averaged
  138.373/139.957 us at rows 18/19, while rank-major head `5fdc048` averaged
  138.678/140.247 us: regressions of 0.305/0.290 us (0.221%/0.207%). Do not
  accept the late-PDL variant. Focused early-PDL job `6978588` now isolates
  only the launch placement against `5fdc048`, reusing the byte-identical
  DeepGEMM build but compiling separate selector caches. It requests one HSG
  tray on `short` for one hour and is the sole optimization job beside the
  pending 18-tray Qwen3 side trace `6978327`.
- Early-PDL rank-major job `6978588` completed cleanly `0:0` in 5:29 and all
  ordinary-routing numeric cases passed. Against byte-identical late-PDL
  rank-major control, early PDL changed rows 18/19 from 138.050/140.186 us to
  138.223/140.006 us: +0.173/-0.181 us, effectively zero over MB2. Because
  rank-major itself was already +0.305/+0.290 us versus accumulated base, this
  launch move does not recover it; reject both `5fdc048` and `724a211`.
- Final accumulated candidate for the next system gate is clean head
  `a01266f`: retained consumer-side shared fusion and TMA pairwise combine,
  direct Qwen3.5 GDN output into symmetric fabric storage, and the general
  one-CTA-per-token Qwen3.5 selector. Exact dry-run and login validation pass
  for logical 128K, A1:F1, EP4/ATP1, batch 37, hard MB2 `18+19`, 77,659 pages,
  randomized resident state/no prefill, FP4 MegaMoE, perf-mode-forced perfect
  routing, two trays/eight GPUs, 15 measured decode steps, HSG `short`, and a
  one-hour limit. Full Nsight/perf job `6978678` is submitted beside pending
  side trace `6978327`; compare its strict metric with the preoptimization AFD
  baseline 41.247700 ms / 448.509856 TPS/GPU.
- Full gate `6978678` failed before graph capture in 6:16, so it contains no
  performance evidence. Production prewarm exposed missing composition
  plumbing: `Qwen3_5LinearAttention` passed `output=` to
  `LinearOProj.forward_fp8_prequant`, but that override had not threaded the
  keyword into `_LinearTPImpl` even though the low-level direct-output smoke
  had validated the actual peer-fabric GEMM. Clean corrective head `b0c7efc`
  adds only the missing override argument/forward plus a source contract; all
  25 focused contracts pass and the exact diff from `a01266f` is eight lines.
  Same-config replacement job `6978868` uses this immutable source on two HSG
  `short` trays with the one-hour limit. The failed run's prompt cache and
  exact-capacity checks had both passed; no alignment job was launched.
- Corrected accumulated job `6978868` completed cleanly `0:0` in 8:44 with
  canonical success: exact batch-37 capacity, randomized resident 128K KV/GDN
  state and no prefill, perf-mode-forced perfect routing, all 148 samples,
  warmup plus 15 measured steps, and eight durable Nsight reports. The expected
  allocation-scoped driver status 137 occurred only after required capture
  artifacts were durable. First CPU metric extraction `6979068` failed in 31
  seconds without touching the capture because its reusable wrapper omitted
  the extractor's required `--case-id`; corrected task-local wrapper job
  `6979149` supplies `i131072-fep4-r1-atp1-b37`. Keep alignment idle until the
  strict metric and fresh-trace kernel audit complete.
- Corrected CPU extractor `6979149` completed cleanly in 0:30. The strict
  dual-role metric accepts all 15 steps with zero outliers: mean/median/max
  critical span 37.951664/37.785281/40.546978 ms, dominant range 8.7673%
  under the 10% limit, max/median difference 7.3089% under the 10% limit, and
  487.462159 TPS/active GPU. Relative to the preoptimization same-input AFD
  baseline 41.247700 ms / 448.509856 TPS/GPU, accumulated head `b0c7efc`
  improves latency by 3.296036 ms / 7.990836% and throughput by 38.952303 /
  8.684827%. Metric is `qwen35_accumulated_final_perf/analysis/metric.json`.
  CPU-only representative attention-rank1/model-rank5 Nsight export and kernel
  summary job `6979213` is the final trace-driven audit before alignment.
- Kernel-audit job `6979213` completed cleanly in 0:31. The fresh accumulated
  trace has no standalone route-reset, route-swizzle, or epoch-reset kernel.
  Route readiness and cleanup remain inside existing producer/MegaMoE kernels.
  The sole CUDA memset is one 152-byte zero per model graph (~0.96 us), between
  the final LM-head GEMM and argmax/sampling; it is not MoE workspace work.
  Four tiny generic long fills per step total ~4.45 us and are likewise graph
  bookkeeping. Do not create a target-7 candidate from these non-route tails.
  Durable representative SQLite and CUDA kernel summaries are under
  `qwen35_accumulated_final_perf/analysis/kernel_audit/`.
- Final normal-routing correctness job `6979299` is submitted from immutable
  head `b0c7efc` after an exact login validation: A3/8K, EP4/ATP1, batch 4,
  hard MB2 `2+2`, 48 prompts, ordinary routing, no perf mode, no Nsight, four
  trays/16 GPUs, HSG `short`, one-hour limit. Run the unchanged vLLM scorer
  only after a durable 48x17 sample; side trace `6978327` remains the other
  queued job.
- Alignment generation `6979299` completed cleanly `0:0` in 9:20 with
  canonical success. Its normal-routing artifact contains exactly 48 nonempty
  samples, 17 generated tokens each, and 816 tokens total. Unchanged one-tray
  vLLM scorer job `6979442` is submitted from the same immutable `b0c7efc`
  source with HSG `short` and a one-hour limit; top-10 must equal 1.0.
- Final scorer `6979442` completed cleanly `0:0` in 9:56. Normal-routing
  A3/8K alignment passes all acceptance gates across 816 generated tokens:
  top-1 agreement 0.987745, top-10/top-100 hit rates exactly 1.0, average rank
  1.012255, and maximum rank 2. The durable result is
  `final_a3_normal_alignment_accumulated/results/afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260906_215602_manual_na/vllm-score-final/alignment.json`.
- Reconciled the local workspace to the exact measured/aligned `b0c7efc`
  production and contract-test files, removing both rejected rank-major CTA
  variants. Byte comparisons against the immutable final source snapshot and
  `git diff --check` pass; the 25 focused overlap contracts pass locally in
  0.203 s. Keep the locally improved reusable metric wrapper (which supplies
  mandatory `--case-id`) and the expanded GDN validation helper. The pending
  Qwen3 56-attention-worker/EP16 perfect-balancing Nsight side job `6978327`
  remains the only outstanding run.
- Final status of the seven-item non-parameter TRT-LLM-inspired exploration
  set: (1) routing/PDL is accepted via refined scoped PDL and Qwen3.5's
  one-CTA-per-token selector; sparse count stores, packed sort, and a separate
  publisher were measured and rejected. (2) rank-major destination-deduplicated
  dispatch is theoretically traffic-efficient but every safe arbitrary-routing
  variant exposed enough selector push latency to regress component/E2E, so it
  is rejected for the current geometry. (3) one-CTA/token bulk-TMA pairwise
  combine is accepted as a component non-regression with better large-A:F
  structure. (4) direct GDN producer output is accepted (4.104 us / 22.24%
  producer-sequence saving); QKV publishers were already fused, while direct
  full-attention O is unsafe because its local gate still consumes O. (5) the
  observed GDN projection/conv/gate/norm opportunities are already packed,
  vectorized, overlapped, or fused; re-profile found no additional safe general
  fusion beyond direct output. (6) shared-expert add is accepted in the next
  high-parallelism residual/RMSNorm consumer (1.517% isolated E2E win); both
  low-parallelism MegaMoE placements were measured and rejected. (7) the fresh
  final trace contains no standalone route reset/swizzle/epoch kernel, and its
  only memset/fills are sampling or graph bookkeeping outside the retained
  target, so no candidate is warranted. This closes the requested list for the
  current pass without parameter tuning or perfect-route-only assumptions.
- Final cleanup/goal closure: both staged and unstaged diff checks pass, the
  25 focused overlap contracts pass again in 0.123 s, and no rejected
  rank-major or low-parallelism shared-output symbols remain in production
  source. The exact measured/aligned accepted implementation is integrated
  locally and its relevant files were hash-matched to the accepted remote
  mirror. Per the user's direction, leave the independent 18-tray Qwen3 trace
  job `6978327` alone; its queued/completion state is not a blocker for closing
  the Qwen3.5 optimization goal.
- Copied the final optimized Qwen3.5 Nsight capture from job `6978868` back to
  local `scratch/qwen35_accumulated_final_perf/final_optimized_capture/`.
  `nsys/` contains all eight nonempty `.nsys-rep` files and every SHA-256
  matches its remote original. `metadata/` contains `capture-complete.json`,
  `SUCCESS`, `afd-result.json`, `sample.json`, and the strict `metric.json`;
  `sqlite/` contains the representative attention-rank1 and model-rank5
  exports. The complete local bundle is 19 MiB.
- FFN stream-topology follow-up: representative final model-rank5 SQLite has
  31,376 graph kernel executions on 226 CUDA Graph execution stream IDs for
  graph 2, although the runtime owns only two decode lanes plus two Qwen3.5
  shared-expert auxiliaries. Source comparison isolates captured CUDA events
  as the cause: the common per-round compute handoff segments both models,
  while Qwen3.5 additionally records/waits shared-expert events each layer.
  Attention/GDN is explicitly out of scope. Candidate
  `source/qwen35_ffn_two_stream_v1` leaves Qwen3's accepted event schedule
  unchanged; Qwen3.5 instead uses a model device turn acquired by the existing
  O-ready waiter and released at the fused QKV/linear publication boundary.
  It queues each shared expert on the peer one of the same two FFN lanes with
  reusable device flags, including an explicit final shared-output join, and
  creates no shared auxiliary stream or steady-state FFN CUDA event. Local
  static overlap contracts pass; full CUDA/PyTorch validation, A1/128K perf,
  and node-level Nsight stream audit are combined in two-tray HSG short/1h job
  `6980732`. Compare latency with accepted job `6978868` at 37.951664 ms and
  require the model graph to collapse to the intended two execution streams.
- Two-stream v1 job `6980732` completed cleanly `0:0` with canonical success
  and all eight node-level Nsight reports. Corrected CPU postprocessing jobs
  `6980870`/`6980871` prove graph 2 is exactly two model execution streams
  (32/145), down from 226, with 2,206 kernel nodes/replay. All 15 strict steps
  are retained: 38.498336-ms mean, 38.333924-ms median, 39.649856-ms max, and
  480.540251 TPS/active GPU. This is 1.4404% slower than accepted accumulated
  job `6978868`, so do not accept v1. Jobs `6980830`/`6980831` were harmless
  30-second CPU wrapper failures because the inner container inherited a
  transient Slurm spool path; the corrected jobs passed a durable script path.
  Local analysis is `scratch/qwen35_ffn_two_stream_v1/analysis/`.
- The v1 trace identifies the regression precisely. Its shared-expert gate/up
  GEMM still starts before MegaMoE, but the activation/cast tail misses the
  MegaMoE boundary: shared down starts about 5.4 us after MegaMoE and stretches
  to 139 us under resource contention, versus starting 1.3 us before MegaMoE
  and taking 4.8 us in the accepted auxiliary-stream graph. The 120 shared
  release kernels plus 120 redundant input casts account for the lost launch
  margin. Two-stream v2 fuses post-attention add/RMSNorm, BF16 writeback,
  packed-UE8M0 FP8 quantization, and last-CTA device-turn release; the shared
  gate/up consumes that prequantized view. This removes 240 steady-state nodes
  while preserving the two-lane device-wait schedule and arbitrary routing.
  Local overlap contracts pass 25/25 and diff checks pass. The first nominal
  v2 job `6981015` completed cleanly but is not candidate evidence: a long
  remote v1-to-v2 source clone finished after the incremental file sync and
  overwrote the fused files. Its unchanged 2,206-node/kernel-name trace proves
  it reran v1 (38.401558 ms), and it is discarded. The clone is no longer
  running; the fused source sizes and symbols were explicitly verified after a
  fresh sync. Replacement exact A1/128K MB2 perf plus node-level trace job
  `6981122` completed cleanly `0:0` in 7:34 from the corrected
  `source/qwen35_ffn_two_stream_v2`. Corrected CPU jobs `6981247`/`6981248`
  extracted the trace and strict metric. Graph 2 has exactly two execution
  streams and 1,966 kernels/replay, proving the fused kernel is present and
  removing 240 nodes versus v1. All 15 strict steps are retained with
  38.420000-ms mean, 38.304543-ms median, 39.119008-ms max, and 481.520051
  TPS/active GPU. This remains 0.468335 ms / 1.234% slower than accepted job
  `6978868`, so v2 is not accepted. Local analysis is
  `scratch/qwen35_ffn_two_stream_v2/analysis/fused/`.
- Cross-capture model-rank5 analysis localizes v2's remaining FFN cost. Its
  shared gate/up starts 10.9 us before MegaMoE but takes 9.44 us median; the
  following cast starts after MegaMoE and shared down consequently starts
  5.63 us after MegaMoE, stretching to 141.1 us under contention. In the
  accepted graph, shared down starts 1.38 us before MegaMoE and takes 4.90 us.
  The key cause is MegaMoE's early programmatic-dependent launch from the
  route selector, not attention: it becomes resident while the peer's shared
  cast/down tail is still queued. Candidate v3 retains the route-ready
  protocol and PDL-capable MegaMoE but, only for Qwen3.5 exact-two-lane FFN,
  defers the selector's explicit dependent-launch signal until normal selector
  completion. This should give the already-queued shared down a roughly 5-us
  head start without adding a graph node, stream, event, parameter tuning, or
  perfect-routing assumption. Qwen3 and other routing callers retain early
  PDL. The 25 focused overlap contracts and `git diff --check` pass locally
  and in the immutable remote candidate. Combined exact A1/128K MB2 perf plus
  node-level Nsight job `6981417` is submitted from
  `source/qwen35_ffn_two_stream_v3` on two HSG `short` trays with a one-hour
  limit. Require exactly two FFN execution streams and compare strict latency
  against accepted 37.951664 ms; the trace must also show shared down launching
  before MegaMoE rather than merely moving wall time.
- V3 job `6981417` completed cleanly `0:0` in 8:47 with canonical success and
  eight durable reports. CPU jobs `6981511`/`6981512` completed strict metric
  and representative export. Graph 2 remains exactly two execution streams
  and 1,966 kernels/replay. The intended ordering is restored: shared down now
  starts 0.86 us before MegaMoE and takes 4.77 us median, versus v2's +5.63 us
  start and 141.1-us duration. All 15 steps pass stability checks at
  38.289638-ms mean, 38.224128-ms median, 38.919167-ms max, and 483.159436
  TPS/active GPU. This is still +0.337974 ms / +0.8905% versus accepted
  `6978868`, so do not accept v3 or run alignment yet. Local trace evidence is
  `scratch/qwen35_ffn_two_stream_v3/analysis/nsys/`.
- V3 leaves 120 shared-branch `wait_turn` CTAs per replay (plus bootstrap and
  final waits). They add a kernel node per layer and remain resident for a
  summed 15.733 ms while the peer computes. Candidate v4 keeps v3's proven
  PDL ordering but replaces only those Qwen3.5-local shared waits/resets with
  captured `cuStreamWaitValue32`/`cuStreamWriteValue32` operations. The fused
  norm publishes the same binary turn with a device-release u32 store. This
  targets 120 fewer kernels and no waiter-SM residency; model-compute and final
  timeout-bearing waits remain unchanged. The path is isolated to Qwen3.5's
  shared expert, links `libcuda` explicitly, and does not change Qwen3. The 25
  focused contracts and diff checks pass locally and in the immutable remote
  candidate. Combined exact A1/128K MB2 perf plus node-level Nsight job
  `6981702` is submitted from `source/qwen35_ffn_two_stream_v4` on two HSG
  `short` trays with a one-hour limit. Require exactly two FFN execution
  streams, 120 fewer wait kernels, retained pre-MegaMoE shared-down launch,
  and strict latency no worse than accepted 37.951664 ms.
- V4 job `6981702` completed cleanly `0:0` in 7:43; expected driver status 137
  happened only after canonical success and eight durable reports. CPU jobs
  `6981822`/`6981823` completed strict metric and representative export. The
  model graph is exactly streams 32/145 and drops to the predicted 1,846
  kernels/replay; only two kernel wait-turns remain. Nevertheless, stream
  memory operations delayed the shared gate/up start from v3's -12.96 us to
  -7.28 us relative to MegaMoE. Shared down consequently starts +7.52 us and
  stretches to 135.8 us median. Strict mean is 38.398342 ms / 481.791631
  TPS/active GPU, +0.446679 ms / +1.177% versus accepted. Reject v4: removing
  waiter residency is not useful when CUDA Graph schedules the nonresident
  wait too late. No alignment is warranted.
- Candidate v5 restores v3's early resident shared waiter and correct
  pre-MegaMoE down ordering. It instead specializes the fused add/RMSNorm/FP8
  publisher by compile-time groups-per-warp. At hidden 4096, each thread's
  scratch shrinks from the runtime-indexed maximum 32 floats to exactly 16,
  and both deliberately non-unrolled group loops become fully unrolled. This
  targets the fused kernel's 7.55-us cost (roughly 2.3 us/layer above the
  accepted separate norm+shared cast) without adding nodes, synchronization,
  streams, parameters, or a routing assumption. Hidden 1024/2048/3072/4096
  remain explicit fail-fast specializations. The Qwen3 path is untouched and
  25 focused contracts plus diff checks pass locally and in the immutable
  remote candidate. Combined exact A1/128K MB2 perf plus node-level Nsight job
  `6981897` is submitted from `source/qwen35_ffn_two_stream_v5` on two HSG
  `short` trays with a one-hour limit. Require v3's exactly-two-stream/1,966-
  kernel graph and pre-MegaMoE shared down, plus fused-kernel and strict E2E
  improvement sufficient to reach the accepted 37.951664-ms ceiling.
- V5 job `6981897` completed cleanly `0:0` in 8:48 with canonical success and
  eight durable reports. CPU jobs `6982004`/`6982005` completed strict metric
  and export. Graph 2 remains exactly two streams and 1,966 kernels/replay.
  Compile-time specialization reduces fused norm/quant from v3's 7.55 to 5.96
  us/layer and MegaMoE from 153.01 to 151.70 us average. Shared down remains
  pre-MegaMoE at a -0.48-us median, but the faster start increases the fraction
  of gate/up/down tails that race the routed launch, so 24.3% of down launches
  still start late. Strict mean/median/max are 38.218047/38.132067/38.855427
  ms and throughput is 484.064500 TPS/active GPU. V5 improves v3 by 0.071591
  ms but remains +0.266384 ms / +0.702% versus accepted; keep the real kernel
  specialization, but do not accept the system result or align it yet.
- Candidate v6 adds a second must-have fusion to v5: the shared-expert scalar
  gate dot product is computed inside the same fused post-attention norm/FP8
  publisher while each normalized BF16 value is already live. A block
  reduction emits the `[M,1]` BF16 gate logits before the existing last-CTA
  release. The peer shared branch consumes those logits and no longer launches
  120 separate cuBLAS dot kernels, removing one node and a contended 30.9-us
  average kernel per layer while allowing the peer lane to finish immediately
  after shared down/output gating. The standard non-overlap fallback and Qwen3
  path are unchanged. Focused contracts and diff checks pass locally and in
  the immutable remote candidate. Combined exact A1/128K MB2 perf plus
  node-level Nsight job `6982117` is submitted from
  `source/qwen35_ffn_two_stream_v6` on two HSG `short` trays with a one-hour
  limit. Require exactly two streams, 1,846 kernels/replay (120 fused-away dot
  kernels), pre-MegaMoE shared down, and strict latency no worse than the
  accepted 37.951664-ms result.
- V6 job `6982117` completed cleanly `0:0` in 8:29 with canonical success and
  all eight durable Nsight reports. CPU jobs `6982172`/`6982173` completed the
  strict metric and representative export. All 15 steps are retained with
  mean/median/max critical spans of 37.861737/37.820896/38.323104 ms,
  dominant-range/max-median spreads of 2.0795%/1.3279%, and 488.619955
  TPS/active GPU. This is 0.089927 ms / 0.237% lower latency than accepted
  `6978868`, so V6 passes the system performance gate. Graph 2 still has
  exactly two execution streams (32/145). Its 1,726 kernel nodes/replay are
  correct rather than the predicted 1,846: the old cuBLAS scalar gate dot used
  both one `dot_kernel` and one `reduce_1Block_kernel` per layer, so fusing the
  dot removes 240 nodes/replay while adding no launch beyond the changed fused
  publisher. Shared gate/up starts 12.90 us before MegaMoE; shared down starts
  0.48 us before it with a 4.96-us median, and the late-down fraction improves
  from V5's 24.27% to 13.96%. Local evidence is
  `scratch/qwen35_ffn_two_stream_v6/analysis/`.
- First V6 alignment submission `6982255` failed fast in 23 seconds before
  runtime startup because the reconstructed submission inherited `NSYS=1`
  while correctly requesting graph-trace mode `none`; it is not correctness or
  performance evidence. Second setup attempt `6982286` passed that gate but
  failed after 79 seconds because the intentionally dirty immutable V6 snapshot
  was not accompanied by the same explicit allow-dirty contract as its passing
  perf job. Corrected job `6982320` explicitly exports `NSYS=0`, allow-dirty,
  attn-only placement, MB2, MegaMoE, and memory-ratio settings from immutable
  `source/qwen35_ffn_two_stream_v6`: A3/8K, EP4/ATP1, batch four, hard MB2
  `2+2`, 48 prompts, ordinary routing, no perf mode, no Nsight, four HSG
  `short` trays, and an explicit one-hour limit. Run the unchanged vLLM scorer
  only after the durable 48x17 sample is present. The independent 18-tray side
  job remains intentionally unmonitored.
- Corrected V6 alignment generation `6982320` completed cleanly `0:0` in 9:34.
  It produced canonical success and exactly 48 nonempty samples, each with 17
  generated tokens (816 total), under ordinary routing with perf mode and
  Nsight disabled. Unchanged one-tray vLLM scorer job `6982503` is submitted
  on HSG `short` with a one-hour limit; it uses the same fixed reference model
  and top-10=1.0 acceptance rule as scorer `6979442`.
- V6 scorer `6982503` completed evaluation but correctly failed the unchanged
  gate: top-1 0.177696, top-10 0.405637, top-100 0.654412, average rank
  2,175.13, and maximum rank 164,642 across all 816 tokens. V6 is therefore
  rejected despite its perf win. All 48 token-zero outputs exactly match the
  previously passing accumulated sample, but zero complete sequences match and
  agreement collapses beginning at generated token one (70/816 total IDs).
  This localizes the defect to the decode-graph FFN path; eager prefill is
  unchanged and correct. The likely semantic risk is the custom fused
  post-attention RMSNorm/scalar-gate arithmetic introduced only inside the
  two-stream decode graph, not ordinary routing itself. Preserve the exact
  FlashInfer post-attention normalization and established gate arithmetic in
  the next candidate unless a focused numeric comparison proves a fused form
  equivalent. No additional full alignment should run before that cheap gate.
- V7 removes the unsafe arithmetic change rather than weakening alignment.
  Qwen3.5 decode again calls the established FlashInfer residual-add/RMSNorm;
  only its already-rounded BF16 output is read by a new one-CTA-per-token
  kernel that jointly produces packed-UE8M0 shared-expert input, the scalar
  shared gate, and the peer-lane release. This retains V6's eliminated cuBLAS
  dot/reduction and release launches without changing the routed FFN input.
  The standalone quant scale floor now also matches the established packed-FP8
  kernel. Local and remote focused contracts pass 25/25 and both diff checks
  pass. Combined exact A1/logical-128K MB2 perf plus node-level Nsight job
  `6982728` is submitted from immutable
  `source/qwen35_ffn_two_stream_v7` on two HSG `short` trays with a one-hour
  limit. Require exactly two streams, the shared down launch before MegaMoE,
  and strict latency no worse than 37.951664 ms before another alignment run.
- V7 job `6982728` completed cleanly `0:0` in 7:53 with canonical success.
  CPU jobs `6982798`/`6982799` produced the strict metric and representative
  node export. The exact FlashInfer normalization restores the intended safe
  arithmetic boundary, and graph 2 remains exactly streams 32/145 with 1,846
  kernels/replay. All 15 strict steps are retained at 38.123919-ms mean,
  38.024096-ms median, 39.345442-ms max, and 485.259657 TPS/active GPU.
  This is +0.172255 ms / +0.454% versus accepted `6978868`, so V7 is not a
  performance acceptance candidate and no full alignment was launched. Trace
  decomposition explains the gap: exact FlashInfer add/RMSNorm plus the new
  FP8/gate publisher costs 8.387 us/layer versus V6's unsafe 6.309-us fused
  operation. The peer shared branch also still materializes separate BF16
  sigmoid and multiply kernels (240 nodes/replay). Local evidence is
  `scratch/qwen35_ffn_two_stream_v7/analysis/nsys/`.
- Candidate V8 preserves V7's exact post-attention normalization and removes
  only the shared branch's exposed sigmoid/multiply tail. `_run_model_moe`
  now returns raw BF16 shared-expert output plus BF16 gate logits. The next
  layer's existing shared-add/RMSNorm or linear-input publication consumer
  applies the scalar gate while reading the shared output, explicitly keeping
  both established BF16 rounding boundaries (sigmoid, then multiply). The
  ordinary non-overlap fallback still calls the unchanged
  `qwen35_shared_output`, and Qwen3 cannot enter any of the hybrid-only code.
  This should remove 240 graph nodes/replay without adding a stream or changing
  routing. Local focused overlap contracts pass 25/25 and `git diff --check`
  passes. The performance gate must prove exactly two FFN graph streams,
  approximately 1,606 kernels/replay, and mean critical latency no worse than
  37.951664 ms before normal-routing A3/8K alignment is submitted.
- V8 was synchronized to immutable remote snapshot
  `source/qwen35_ffn_two_stream_v8`; its remote diff check and all 25 focused
  contracts pass. Combined exact A1/logical-128K MB2 perfect-routing perf plus
  node-level Nsight job `6982929` is submitted on two HSG `short` trays with a
  one-hour limit. Its durable run directory is
  `qwen35_ffn_two_stream_v8/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260907_041000_manual_na`.
- V8 job `6982929` completed cleanly `0:0` in 9:25 with canonical success and
  all eight durable reports. Metric job `6982997` passed; the first Nsight
  summary attempt `6982998` failed only because its copied manifest retained
  V7's per-process report suffixes, and corrected job `6983046` passed after
  selecting V8 rank1/rank5 reports. Graph 2 has exactly two execution streams
  (32/145) and exactly 1,606 kernels/replay, the predicted 240-node reduction.
  Shared down starts 0.544 us before MegaMoE at median, is late for only 5.89%
  of layers, and takes 5.056 us at the median; consumer fusion also reduces
  average shared-down contention from V7's 22.200 to 12.904 us and MegaMoE
  from 151.555 to 149.571 us. All 15 strict steps pass stability at
  37.980069-ms mean, 37.863103-ms median, 38.687647-ms max, and 487.097582
  TPS/active GPU. V8 recovers 0.143850 ms versus V7 and is 7.922% below the
  original 41.247700-ms preoptimization AFD result, but remains +0.028406 ms /
  +0.075% versus accepted multi-stream `6978868`; retain V8 but do not align
  yet. The trace shows V8 adds only about 0.378 us to each of the 32/replay
  full-attention/final shared-add consumers; its eight warps redundantly
  evaluate the same scalar sigmoid. The next safe candidate should compute and
  BF16-round sigmoid once when the fused producer already owns the scalar dot,
  then have both consumers load the ready BF16 scale.
- Candidate V9 implements that trace-directed refinement. The fused FP8/gate
  producer rounds its dot sum to the established BF16 gate-logit boundary,
  publishes the peer turn as soon as FP8/scales are ready, and then computes
  and BF16-rounds sigmoid while the peer begins shared gate/up. Consumers load
  that BF16 scale and retain the separately rounded BF16 shared multiply. This
  eliminates eight redundant `expf` evaluations per consumer without adding a
  graph node or delaying the peer release. The hybrid-only runtime variables
  now name the value as a scale; ordinary routing, non-overlap fallback, and
  Qwen3 remain unchanged. Local diff checks and all 25 focused contracts pass.
- V9 is synchronized to immutable remote snapshot
  `source/qwen35_ffn_two_stream_v9`, where the same diff check and 25 contracts
  pass. Exact A1/logical-128K MB2 perfect-routing perf plus node-level Nsight
  job `6983101` is submitted on two HSG `short` trays with a one-hour limit.
  Durable run directory:
  `qwen35_ffn_two_stream_v9/results/afd_qwen35_131072_r1_ag4_fg4_atp1_fep4_adp4_b37_n2_20260907_043200_manual_na`.
- V9 job `6983101` completed cleanly `0:0` in 5:52 with canonical success and
  all eight reports. CPU jobs `6983174`/`6983175` passed strict extraction and
  Nsight export. All 15 steps pass stability with 37.800180-ms mean,
  37.737760-ms median, 38.309440-ms max, and 489.415659 TPS/active GPU. This
  beats accepted multi-stream job `6978868` by 0.151484 ms / 0.399% latency
  and 0.401% throughput, and beats the original 41.247700-ms preoptimization
  AFD baseline by 3.447520 ms / 8.358% latency and 9.12% throughput. Graph 2
  is exactly streams 32/145 and 1,606 kernels/replay. Shared down starts 0.448
  us before MegaMoE at median and takes 4.848 us for the median-paired layer;
  the fused producer remains only 4.785 us despite adding the once-per-row
  sigmoid. V9 therefore passes the structural and performance gates. Proceed
  to one accumulated ordinary-routing A3/8K alignment, not another perf
  variant.
- Final accumulated alignment generation job `6983202` is submitted from V9
  on four HSG `short` trays with a one-hour limit: ordinary routing (perfect
  routing and perf mode both disabled), A3/8K, EP4/ATP1, batch four, hard MB2
  `2+2`, 48 prompts, no Nsight, exact dirty-source authorization. Durable run:
  `final_a3_normal_alignment_two_stream_v9/results/afd_qwen35_8192_r3_ag12_fg4_atp1_fep4_adp12_b4_n4_20260907_044000_manual_na`.
  Require exactly 48x17 generated tokens, then run the unchanged vLLM scorer
  with the existing top-10=1.0 acceptance rule.
- Copied V9's complete final profiling evidence to local
  `scratch/qwen35_ffn_two_stream_v9/final_optimized_capture/`: all eight
  nonempty raw `.nsys-rep` files, capture/SUCCESS/result/sample metadata, the
  strict metric, representative attention/model SQLite exports and summaries,
  postprocessor logs, and report manifest (17 MiB total). A checksum dry-run
  against the remote raw-report directory is empty, proving byte identity.
- V9 alignment generation job `6983202` completed cleanly `0:0` in 9:39 with
  canonical success. Its durable sample contains exactly 48 prompts, every one
  with 17 generated tokens (816 total). Unchanged one-tray vLLM scorer job
  `6983296` is submitted on HSG `short` with a one-hour limit, the fixed
  reference model, and the existing top-10=1.0 gate. Its output is
  `final_a3_normal_alignment_two_stream_v9/.../vllm-score-final/alignment.json`.
- V8's first Nsight summary retry `6982998` failed only because its copied
  two-report manifest retained V7's per-process filename suffixes. The raw V8
  reports were complete; correcting the two manifest paths and resubmitting as
  `6983046` passed in 29 seconds. Metric job `6982997` also passed. This was a
  postprocessing-input error, not GPU/runtime evidence.
- Final scorer `6983296` completed cleanly `0:0` in 11:24. Across all 816
  ordinary-routing A3/8K tokens, top-1 agreement is 0.986519608, top-10 and
  top-100 are both 1.0, average vLLM rank is 1.014705882, maximum rank is 3,
  and reference perplexity is 1.096093738. V9 therefore passes the unchanged
  correctness gate. The complete sample and alignment JSON are copied locally
  under `scratch/qwen35_ffn_two_stream_v9/alignment/` with an empty checksum
  dry-run against the remote alignment report.
- Final Qwen3 nonregression disposition: accepted Qwen3 same-code A/B job
  `6977381` was already favorable (29.903162 ms candidate versus 30.338281 ms
  control). Every subsequent FFN stream/gate change is gated by
  `is_hybrid_attention` or lives in Qwen3.5-only JIT kernels; Qwen3 retains its
  established CUDA-event handoff and never allocates the Qwen3.5 device turns,
  shared gate scale, or shared consumer kernels. The final focused contract
  explicitly protects that split. Do not spend another multi-tray Qwen3 run
  for changes it cannot execute.
- Goal achieved: V9 is the cleaned final implementation. It reduces the
  Qwen3.5 model CUDA Graph from 226 execution streams to exactly two MB2
  ping-pong streams while improving the accepted accumulated baseline, passes
  normal-routing alignment, preserves Qwen3 structurally, and has complete raw
  final Nsight evidence copied locally. Leave independent 18-tray job
  `6978327` unmonitored as directed.
- 2026-09-07 explicit follow-up check of the previously unmonitored Qwen3 side
  job `6978327`: Slurm state is `FAILED`, elapsed 6:49, exit `3:0` on all 18
  trays. It failed before decode/capture in MegaMoE symmetric-buffer
  rendezvous with `Expected reqs[r].block_size == reqs[0].block_size`. The
  56-attention-worker to EP16 mapping is uneven: eight FFN ranks import four
  attention-DP sources and eight import three, so the current per-rank fan-in
  sizing requests different symmetric allocation sizes inside one EP group.
  All 16 MLP logs report the mismatch. There is no canonical `SUCCESS`, result,
  sample, capture-complete marker, or nonempty `.nsys-rep`; therefore this job
  produced no usable Qwen3 trace or performance result. Perfect balancing is
  unrelated because initialization failed before routing. Do not resubmit the
  unchanged case; a future retry needs uniform group-wide MegaMoE allocation
  sizing (for example, the maximum fan-in across the EP group) or a topology
  with divisible attention/FFN DP counts.
- 2026-09-07 Qwen3 56:16 follow-up fix: `AfdFmhaRuntime` now keeps `fanin` as
  each MLP rank's actual source count for routing, graph shape, and source
  offsets, while sizing transport and MoE communication buffers with the
  topology-wide `attn_fanin_per_mlp_dp` capacity. For 56 attention DP lanes
  over 16 MLP DP/EP ranks, execution therefore retains the legitimate 3/4
  source mapping but every symmetric rendezvous uses four-source capacity.
  The change is general to non-divisible A:F layouts and does not assume
  perfect routing. A new runtime contract protects the separation, and the
  topology test now covers 56:16 explicitly. Local diff/static checks pass;
  the local combined protocol test was dependency-blocked only by absent
  `msgpack`, while the production HSG environment passed all 44 focused
  protocol and overlap-contract tests. Candidate source is preserved at
  `source/qwen3_a56_ep16_uniform_fanin_v1` (base HEAD
  `dbf66891a872e81d5f6ee0cf923fb31bf0d64a6b`). Identical perf-mode,
  perfect-routing, logical-128K batch-4, A56:F16/EP16, MB2, node-level Nsight
  rerun `6986545` was submitted on 18 HSG `short` trays with a one-hour limit;
  durable run timestamp is `20260907_101725`. It was pending for resources at
  submission and must prove symmetric-buffer initialization plus capture.
- Rerun `6986545` exited `2:0` after only 23 seconds, before container or GPU
  work. This was a submission reconstruction error, not a runtime-fix result:
  replaying the prior `sacct` SubmitLine omitted
  `FASTAFD_ATTENTION_GPUS_OVERRIDE=56`, which the original command had
  inherited invisibly through `--export=ALL`. The worker-side launcher then
  recomputed ratio-7/EP16 as 112 attention GPUs / 32 trays and correctly
  rejected it against the 18-tray limit. No model work or useful trace was
  produced.
- The general comprehensive launcher now restores the explicit attention-GPU
  override and propagates both it and `FASTAFD_MAX_TOTAL_TRAYS` explicitly at
  login-to-Slurm and Slurm-to-container boundaries; it no longer depends on an
  ambient `--export=ALL` variable for this topology. Local `bash -n`, diff, and
  27 static contracts pass; production HSG passes all 45 focused protocol and
  overlap contracts. The exact clean pre-submit gate then passed both dry-run
  assertions and full login validation: Qwen3 logical 128K, batch 4, MB2
  `2+2`, A56:F16, EP16, 14+4=18 trays, 72/72 active/allocated GPUs, perfect
  routing, perf mode, and node-level 15-step Nsight. Only after those checks,
  corrected job `6987184` was submitted from the same shell/environment with
  a one-hour HSG `short` limit. `sacct` verification confirms the submitted
  contract explicitly contains the 56-GPU override, 18-tray bound, A56:F16
  derived topology, perfect routing, and candidate source. Durable run:
  `qwen3_128k_b4_a56_ep16_perfect_nsys_20260906/results/afd_qwen3_131072_r7_ag56_fg16_atp1_fep16_adp56_b4_n18_20260907_110738_manual_na`.
- Status audit of corrected job `6987184`: Slurm reports `FAILED`, exit `1:0`,
  elapsed 9:35, but the GPU workload and requested trace completed. All 16 MLP
  ranks reached `fmha_only_ready`; their actual fan-in alternates between
  three and four while every rank reports `fanin_capacity=4` and identical
  `max_rows=2064`, proving the symmetric-buffer fix on the real 56:16
  topology. There is no block-size rendezvous or runtime traceback. The run
  produced `afd-result.json`, `capture-complete.json`, exactly 224 samples with
  17 generated tokens each (3,808 total), and all 72 nonempty raw `.nsys-rep`
  files. Slurm failure occurred only in post-capture CPU metric extraction:
  `extract_cuda_wall.py` synthesizes plan `active_gpus` as
  `(normalized_ratio + 1) * ffn_ep` = 128 for r7/EP16, while the explicit
  attention override correctly records 56+16=72 active GPUs. The resulting
  `plan/result active-GPU mismatch` prevented the canonical success marker.
  Do not rerun the expensive GPU case; update the extractor/plan contract for
  explicit attention-GPU counts and postprocess these existing 72 reports.
- Qwen3 A56:F16 postprocessing and copy-back completed without another GPU
  run. The plan parser now preserves an explicit positive `active_gpus` field
  when supplied and retains ratio-derived behavior for legacy plans. The FMHA
  extractor also accepts any supported positive `attention_dp >= mlp_dp`
  topology because grouped execution emits one graph launch per role per step
  even when fan-in is uneven. Tests use only generic synthetic layouts, per
  user direction; local diff plus 35 relevant contracts pass, and the HSG
  production environment passes the same 35. First CPU extraction job
  `6989619` exposed the obsolete integral-fan-in check and failed in 30 seconds;
  corrected CPU-only job `6989683` completed `0:0` in 29 seconds. The strict
  metric retained 14/15 measured steps (step 2 excluded), with critical CUDA
  mean/median/max 21.136990/21.070766/21.492894 ms, dominant range 2.798944%,
  maximum/median difference 2.003382%, 147.187990 tokens/s per active GPU, and
  10,597.535303 aggregate tokens/s across 72 active GPUs. The critical path is
  attention rank 1 (same 21.136990-ms retained mean); model rank 57 averages
  18.566269 ms, median 18.568866 ms, max 18.692258 ms. There is no same-shape
  pre-fix performance result because the old run failed during initialization.
  All 72 nonempty raw reports plus result/capture/sample metadata, strict
  metric, plans, and logs are local under
  `scratch/qwen3_128k_b4_a56_ep16_perfect_nsys_20260906/final_capture_20260907_110738/`.
  Raw reports total 74,887,357 bytes, and an rsync checksum dry-run reports no
  remote/local difference.
- 2026-09-07 layer-local diagnosis corrects the initial interpretation of the
  A56:F16/EP16 result. The trace is not attention-bound: both EP8 and EP16 model
  graphs have exactly two CUDA streams and 188 wait/MegaMoE pairs (MB2 across
  94 layers), while EP16 shortens model span 20.448277 -> 18.566269 ms,
  attention steady core 20.071898 -> 18.202846 ms, median within-layer
  microbatch FMHA gap 27.472 -> 15.200 us, and mean layer cadence 213.711 ->
  193.754 us. The regression is a launch-contract error for a maximum-overlap
  comparison: current `server/args.py` auto-sets `afd_disable_overlap=True` for
  MB2 graphs unless diagnostic force-overlap is requested. Job `6987184`
  consequently logs `dp_lockstep_normal`, versus the EP8 reference's
  `dp_lockstep_fixed_lead ... max_pending=3`. The attention first-FMHA bubble
  grows 0.475661 -> 2.934144 ms; median post-graph idle grows 0.037632 ->
  4.288800 ms on attention and 0.093184 -> 6.813600 ms on model. Attention graph
  start-to-start is therefore 25.434949 ms versus 20.588316 ms, even though the
  inner EP16 pipeline is faster. Do not use 21.136990 ms as overlapped EP16
  acceptance evidence: the extractor omits its 4.287-ms post-graph gap, and the
  older EP8 result also lacks an explicit perfect-routing record. Full evidence
  is under `scratch/qwen3_128k_b4_a56_ep16_perfect_nsys_20260906/latency_investigation/`.
  No GPU retry was submitted. The next implementation must preserve exactly two
  MB streams without serializing decode steps, then compare EP8/EP16 under one
  identical perf-routing contract.
- 2026-09-07 controlled maximum-overlap rerun submitted as HSG job `6990299`.
  It keeps the corrected Qwen3 A56:F16/EP16 configuration unchanged: logical
  128K resident state without prefill, batch 4 per attention-DP lane, MB2
  `2+2`, attention TP1, 56 attention plus 16 FFN GPUs, 18 trays/72 active
  GPUs, attn-only MegaMoE FP4, perf-mode perfect routing, and node-level
  15-step Nsight. The sole execution-policy change is explicit
  `--afd-force-multi-mb-graph-overlap`, restoring cross-step overlap instead
  of the auto-serialized `dp_lockstep_normal` path. The general launcher now
  validates, propagates, prints, and records this setting across login,
  Slurm, container, server CLI, and result JSON. Local shell/diff checks,
  remote dry-run topology/config resolution, and full remote source/model/
  prompt validation passed before submission. `scontrol` confirms the job's
  explicit export contains `AFD_FORCE_MULTI_MB_GRAPH_OVERLAP=1`, perfect
  routing, the 56-GPU attention override, 72 active GPUs, `short` QoS, and a
  `01:00:00` limit. It is pending for resources. Durable run directory:
  `qwen3_128k_b4_a56_ep16_perfect_overlap_nsys_20260907/results/afd_qwen3_131072_r7_ag56_fg16_atp1_fep16_adp56_b4_n18_20260907_144615_manual_na`.
- Job `6990299` failed before decode/capture after 6:28 (`FAILED`, exit `3:0`)
  with the old 56:16 MegaMoE symmetric-memory mismatch:
  `Expected reqs[r].block_size == reqs[0].block_size`. No usable Nsight report
  or performance result was produced. This was a source-selection regression
  in the rerun wrapper: it pinned clean HEAD `180f3d85` under
  `artifacts/.../qwen35_cta_pairwise_tma_combine_ab/source/candidate`, whose
  `AfdFmhaRuntime.max_rows` still scales by each rank's actual 3/4 `fanin`.
  The proven job `6987184` instead ran
  `source/qwen3_a56_ep16_uniform_fanin_v1`, whose input manifest and logs show
  the required topology-wide `fanin_capacity=4` sizing. The pre-submit source
  validation proved that the selected checkout was internally clean, but did
  not compare it with the prior successful run's recorded input paths; that
  missing comparison allowed the wrong clean snapshot through. Do not
  resubmit until a clean immutable source snapshot is built from the proven
  implementation and its manifest is compared directly with job `6987184`
  plus the intended overlap-only delta.
- Corrected overlap resubmission is HSG job `6994474`. A detached clean source
  snapshot was built at
  `artifacts/.../qwen3_128k_b4_a56_ep16_perfect_overlap_nsys_resubmit_20260907/source`
  with HEAD `2a7346e0a2c549f4c89496c3e254290861f74b4b`. Its recorded inputs were
  checked byte-for-byte against job `6987184`'s `inputs.sha256`; every source,
  prompt, and model-profile entry passed. The snapshot additionally passes the
  focused non-divisible-fan-in contract and contains both topology-wide
  `fanin_capacity` sizing and the multi-MB force-overlap parser path. Local
  shell/diff checks, exact remote dry-run, and full clean-source/model/prompt
  validation passed before submission. `scontrol` confirms source HEAD/path,
  A56:F16/EP16, MB2, batch 4, logical 128K perf state, perfect routing,
  `AFD_FORCE_MULTI_MB_GRAPH_OVERLAP=1`, 72 GPUs/18 trays, node-level Nsight,
  `short` QoS, and a one-hour limit. The job is pending for resources. Durable
  run directory:
  `qwen3_128k_b4_a56_ep16_perfect_overlap_nsys_resubmit_20260907/results/afd_qwen3_131072_r7_ag56_fg16_atp1_fep16_adp56_b4_n18_20260907_200023_manual_na`.
- Job `6994474` completed the GPU workload and exact Nsight capture but Slurm
  ended `FAILED` (`1:0`, 9:19) because the in-job CPU extractor rejected the
  noncanonical temporary plan status `needs_overlap_rerun`. This happened only
  after `afd-result.json`, `sample.json`, `capture-complete.json`, and all 72
  nonempty raw reports were durable. Runtime evidence confirms
  `dp_lockstep_fixed_lead`, `disable_overlap=False`, `max_pending=3`, two model
  decode streams, and uniform `fanin_capacity=4` for both actual fan-in 3 and
  fan-in 4 ranks. The plan status was corrected to the extractor's supported
  `needs_corrected_target_batch_rerun`; CPU-only recovery job `7003138`
  completed `0:0` in 2:41 without another GPU run. The strict 15/15-sample
  Nsight metric has mean/median/max latency
  `18.529548867/18.510272/18.861247 ms`, zero outliers, TPS/user `53.967855`,
  TPS/active-GPU `167.899992`, and aggregate TPS `12088.799442` across 224
  users. Attention/model mean graph spans are
  `18.529548867/18.443435800 ms`, and coordinator median cadence is
  `18.551657500 ms`, confirming removal of the old step-boundary bubble.
  Versus serialized EP16 job `6987184`, latency improves `12.335916%` and
  throughput `14.071802%`. Versus the older overlapped A64:F8/EP8 reference,
  latency improves `9.821168%` and TPS/user `10.890768%`; TPS/GPU is
  `2.970578%` lower because EP16 runs 224 rather than 256 users on the same 72
  GPUs, and the older reference lacks explicit perfect-routing provenance.
  All 72 raw reports and supporting metadata/logs are copied locally under
  `scratch/qwen3_128k_b4_a56_ep16_perfect_overlap_nsys_resubmit_20260907/final_capture_20260907_200023/`;
  zero reports are empty and an rsync checksum dry-run shows no remote/local
  difference.
- 2026-09-08 Qwen3.5 optimization checkpoint preparation completed. The
  checkpoint includes the accepted two-physical-stream MB2 FFN schedule,
  device-turn synchronization, fused Qwen3.5 QKV/linear-input and shared-gate
  paths, direct BF16 hybrid-output transport, uniform uneven-fan-in capacity,
  MegaMoE overlap/communication changes, general launch and Nsight extraction
  support, smoke tooling, and focused contracts. The exact candidate files
  were synced to the authoritative `local-mirror`; the deleted obsolete
  `add_rmsnorm_fp8_quant.cu` was moved to the recoverable sibling directory
  `checkpoint-backup-20260908/`. Remote shell syntax checks passed for the five
  touched launch/extraction scripts. In the pinned production environment, 54
  protocol/overlap/extractor/Qwen3.5-structure tests passed, then all 25 FMHA
  runtime schedule tests passed after their mocks were updated for the new
  `shared`, `shared_gate`, `turn`, and `expected_turn` contracts. No GPU job was
  submitted for this checkpoint; the accepted performance/alignment and final
  Nsight evidence above remain the execution validation.
