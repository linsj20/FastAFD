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
