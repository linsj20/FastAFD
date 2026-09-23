# AGA FastAFD — September 2026

**Complete:** environment independently provisioned; both exact alignment tests audited; exact128GPU performance capture, CPU metric and selected trace pair validated. Qwen3 strict numerical failure is retained, not resolved or relaxed. See [final report](scratch/cluster/oci-aga/bringup_20260915/SUMMARY.md) and [machine-readable state](scratch/cluster/oci-aga/bringup_20260915/STATE.json).

Read [CODEX_CLUSTERS.md](CODEX_CLUSTERS.md) for operations/storage policy and [trace contract](CODEX_PROJECT_nsys_traces.md) for retained artifact layout.

## FFN rack placement verified — September 17

User asked whether the previous A3:F1/EP32 run kept all32FFN GPUs in one rack.
**No:** job25922 placed FFN ranks across four racks in domain `nvl576d002`:
R01 T04/T05/T09=12GPUs; R04 T03=4; R11 T06/T09=8; R12 T02/T08=8.
Each tray contributes4GPUs. Verified the remote canonical `afd-result.json`
`placement.mlp_dp_rank_nodes` against the same plan's `cache-ready.json` host/IP
map; local retained result agrees. Slurm confirms COMPLETED on September15
23:31:42–23:38:40 (scheduler timestamps). The eight allocated R12 trays do not
imply FFN colocation: Ray orders the driver first, then nodes lexicographically
by IP, and assigns ranks from that order (`supervisor_ray._build_ray_rank_plan`).
No rack-aware FFN ordering exists in that path. Evidence is under
`perf_qwen35_available` in the task root below and the retained local perf case.
Read-only inspection; no GPU submission or runtime change.

## Requested and completed work

| Test | A:F | EP | Context | B/lane | GPUs | Result |
|---|---|---|---|---|---|---|
| Qwen3 alignment |2:1|4|nominal8192|16|12|2174/2176 top10 (99.908088%), strict FAIL|
| Qwen3.5 alignment |2:1|8|nominal8192|16|24|4352/4352 top10, strict PASS|
| Qwen3.5 perf + Nsight |3:1|32|131072|40|128|26.955806ms;1112.932790tokens/s/GPU|

TP1,attn-only,MB2,MegaMoE FP8 activation/FP4 experts,payload pull. Alignments use real prefill/normal routing and fresh vLLM scoring; ordered full coverage and finite logprobs verified. Q3 GPU25066/CPU25248:128prompts×17tokens, top1 87.040441%; two misses at zero-based58/5/token33693/rank13 and110/16/token279/rank12, cause unisolated. Actual minimum KV838080tokens (13095pages all8workers). Q35 GPU25133/CPU25249:256prompts×17tokens, top1 97.587316%; minimum5251968KVtokens (82062pages), memory.82. Alignment context follows existing nominal8K prompt-builder contract; raw prefix length was not newly audited.

Perf25922 COMPLETED0:0/6m58s; CPU26230 COMPLETED0:0/34s.96attention+32FFN,32nodes,TP1,EP32,MB2(20+20),3840globalprompts,perf/perfect routing. Runtime capacity PASS all96workers:82048pages/5251072tokens, scheduler required5245978tokens, admissibleB40. Explicit memory budget.95; not equivalent to historical.82. One warmup+15measured graph launches, steps2–16,all15retained,zerooutliers,dominant range6.661875%<10%. Metric averages per-step maximum complete CUDA graph spans across representative attentionrank1 and FFNrank97; does not measure client latency/maxall128ranks. Canonical result producer placeholder remains pending_postprocess; separate metric.json is authoritative completed CPU analysis.

Retained perf case: scratch/afd/cases/qwen35/isl_131072/ep_32/a3-f1/batch_40/oci-aga-20260915/{result,metric,manifest}.json and exactly attention.nsys-rep(531165bytes),ffn.nsys-rep(2170764bytes). Paths/roles/ranks, local/remote nonempty sizes, geometry and metric arithmetic PASS. Alignment cases in matching hierarchy. Raw scores/traces/SQLite remain on AGA. All task jobs terminal; no remaining required work.

## Storage and provisioning

P=scratch/cluster/oci-aga/bringup_20260915.
R=/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD on AGA.
Task R/artifacts/aga_bringup/20260915; source R/source/local-mirror (no Git database); dependencies R/dependencies/{images,envs,models,cache}. Source revision2ee78e96004c5ba79fd434614a7c576baad90e71 plus P/source.patch. No model/kernel math changes for this task. Plain oci-aga for scheduler; oci-aga-data for explicit artifact transfers. /lustre resolves through /scratch: container mounts both.

User requires direct external downloads on AGA, no inter-cluster copying/connections/trust. CPU24862 downloaded Q3 revision39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35 and Q35 revisionea5b4f81096f3901c91dea97f81324302495781d. CPU24991 (11m39s) downloaded ARM64 pytorch25.10 image and built pinned cu130 environment; CPU24992 (2m16s) DeepEPsm100/native build and converter import; CPU25065 TVM15modules.

Unchanged local TensorRT-LLM codex_scripts/oci_hsg_trtllm/convert_fp8_moe_to_mxfp4.py uses ModelOpt MXFP4QTensor group32, main routed experts, preserves other tensors/MTP scope. GPU25015 (4GPUs/2m24s) converted on AGA; probe meanabs3.5767502e-05,max.00390625. CPU25016 finalized94shards/92160expert weights,total_size224884062560 and fp4.ready. models.ready/environment markers also verified by successful model runs.

CPU-only tasks always use CPU allocations: downloads, install/native builds, header finalization, node-cache seeding, alignment audits and Nsight exports/analysis. run_plan.py --prepare records cache-ready.json host/IP/cache paths; GPU path validates it and never seeds caches. prepare_plan.sbatch CPU4cores8GB; postprocess.sbatch CPU16cores96GB, accepts optional command JSON. GPU only for actual CUDA quantization/inference/profiling. Do not reserve GPUs for manual waiting or repeat standalone hardware probes.

## Retained attempts and lessons

- Rejected HSG relay exec2373 stoppedexit130; removed852292608-byte partial and copied index/manifest plus local metadata. No SSH key added; temporary-key request withdrawn. copy-removal.json retained. No further HSG connections.
- Image24861 failed Lustre overlay extraction;24991 reused downloaded layers with node-local /dev/shm ENROOT_TEMP_PATH. Dependent24935 cancelled; earlier pending24915 replaced for CUDA stub linker path. CPU native build avoids GPU dependency.
- Standalone hardware24914 consumed6s: GB200/sm100,driver610.31,189471MiB. Optional unsupported FABRIC flag causedexit2 after query; do not repeat.
- Q35 vLLM first warmup compiled Triton for several minutes. Bounded3s strace showed compilation/cache progress; no restart. Future idea: node-local compiler temp/offline prebuild investigation, without scoring math edits.
- GPU25339 cancelled8m17s after required81960pages>73807; four R11T01 GPUs had73807/75037/75038/74064, other92>=82050. Cause of reduced free memory unisolated; compute-process snapshot empty. CPU25340 cancelled; no validcapture.
- GPU25438 failed143:0/2m50s before model: TVM seeder refuses populatedtargets even matching links. Launcher error. CPU25439 cancelled. CPU25568 (2m48s) seeded fresh per-plan TVM; compatible DeepGEMM/DeepEP/device caches retained. Original wrapper saved run_plan_before_cpu_seeding.py. No fallback added.
- GPU25603 delayed by cleanup of preempted25529 on R01T02/T08, eventually started then failed3:0/3m21s at capacitygate. HeadR01T01 pages71885/73402/71824/73406, minimum4596736tokens supportsB35; other92>=82052. CPU25604 cancelled. Failure evidence preserved separately.
- CPU25839 prepared alternate32hosts/2m55s: R01T[01,03-07,09],R04T03,R07T[01-08],R11T[02-09],R12T[02-09],same nvl576d002,segment1/consolidate. Explicit82048page pool and.95budget preserved fullB40/128K and succeeded25922. Initial head free163096/167440/167440/163262MiB recorded inside actual run; lowfreecause unisolated, no unknown process killed.
- CPU25923 failed47s before import: original case.csv had rerun_index/statusprepared, incompatible with extractor future-plan schema. Created standard analysis-plan.csv (plan_index,statuspending,context131072,active128,actualmemory.95/pages82048); preserved originalcase/capture. postprocess-retry.json changes only --plan; CPU26230 succeeded on same traces. No GPU rerun or threshold change.

No subagents, commits, py_check, manual hash/version audit or unrelated change edits. AGENTS.md absent; user-supplied instructions apply. Optional creation question unanswered. Future work, only if requested: isolate Q3's two numerical misses and investigate reduced free memory without wasting GPUs.


## Qwen3 promotion

September22: optimized source promoted from checkpoint `2be271e` plus scorer import
fix. A3:F1 EP8 B4 on32GPUs in d002-R01:24attention+8FFN,TP1,MB2,FP4 all-push,
earlyPDL,ahead8 resetCTAs,TMA0,norm weight preload,memory.97.

| Check | Result | Jobs |
|---|---|---|
|128K synthetic-KV/perfect-routing perf|17.848781ms;168.078709TPS/GPU;15/15 stable|39751,39753|
|Actual8192 real-prefill/routing alignment|1629/1632 top10;3misses pass agreed≤4; strict zero-miss fails|39752,39886,39887|

Perf is mean maximum graph span across A1/F25, not client latency/all-rank max.
Alignment covers96×17 positions and finite scores. Scorer imports `load_tokenizer`
from `minisgl.hf_support`; generation was not repeated after the scorer fix.
Task=`R/artifacts/aga_promotion/20260922`; `SUMMARY.json`, `jobs.json`,
`placement-audit.json`, `alignment/` retain evidence. Code/docs sync only, no HSG.
Use `--propagate=NONE`: inherited login limits caused Ray fork failures. CPU
preparation must use actual384-rounded capacity and serving reset-row specializations.

## Rack scaling and communication

September22: user selected512GPUs after no complete576GPU domain was available.
A496:F16:16racks×32GPUs in d002, one FFN/rack,31local attention GPUs/FFN;
TP1,EP16,MB2,FP4,128K synthetic KV/perfect routing. GPU39892 completed B4;
39894 completed B5–7. All captures/maps pass; all original timing sets fail stability.
Raw means B4–7:31.700/36.341/38.032/40.102ms, not accepted performance results.

Late-step slowdown is primarily CPU dispatch starvation:496attention commands
precede16FFN commands and drain the initial three-step lead. B6 late dispatch
cadence41.36ms exceeds FFN graph31.88ms; worker preparation starts5.43ms after
previous GPU end, then takes2.72ms. Attention math stays flat; no broken-ticket
evidence. Instrumentation cost/first-MoE EP skew remain open. No coordinator fix
or512GPU rerun. Next: reduce CPU per-lane work and dispatch FFNs earlier/by rack.

`map_afd_racks.py` preserves integer routing with one local FFN per selected rack;
attention-only racks stripe evenly. Requires TP1/equal positive fan-in; physical
CUDA ordinals are Ray-assigned. Sequence validates hosts/domain/map. AGA topology
has16racks/domain,9four-GPU trays/rack; recheck availability before new allocations.
512GPU requests use128nodes,`--segment=8 --consolidate-segments`,QOSnormal;
use explicit low control ports8040/8048/... . Task=`R/artifacts/aga_rack_scaling/20260922_a496_f16`,
`case-plans.json`/`SUMMARY.json` identify results. Exactly A1/F497 traces percase
under `scratch/afd/cases/qwen3/isl_131072/ep_16/a31-f1/`.
[Dispatch evidence](scratch/afd/reports/qwen3/aga-rack-stability-20260922/README.md).

QKV carries18,432B/token. At assumed80% of900GB/s one-way NVLink, A31 payload floors
are1.59/2.38/3.17µs for2/3/4rows versus original34–61µs. Descriptor scans, registers
and grid waves matter; long calls persist without competing compute. Hardware
bandwidth/occupancy counters were not captured, so no proven hardware bottleneck.
Default direct-edge transport reduces registers52→32 and A31 medians
36.00/60.13/60.41→17.47/17.48/18.93µs;34,560checked publications pass. Restricted to
validated grouped Qwen3 small decode with≥7peers; `MINISGL_AFD_QKV_EDGE=0` is control.
A7:F1 EP4 B4 model40561/40562:both15/15 stable, graph21.333→19.895ms,
QKVmean12.947→10.453µs. One ordered pair; full graph gain is not isolated to QKV.

`bench_qkv_fanout.py` keeps CPU preparation, one-rack emulation, exact checks,
protocol/resource comparisons and timing controls together. LL128 sender medians
4.18/4.64/5.68µs; reduced-SM variants did not preserve best latency across sizes.
Prepacking/polling contention were not dominant. Warm replay40679 gives
9.44/10.18/11.33µs ACK-inclusive upper bounds; inference needs no return ACK.
Per-publication CUDA events perturb timing; outer graph events include fills/checks;
warm replay removes first-replay overhead. Sender time is not receiver completion.
[Benchmark evidence](scratch/qkv_fanout/20260922/README.md); raw attempt history is
under `R/artifacts/qkv_fanout/20260922*`, not repeated in project memory.

## A17 Lamport model run

GPU40692 completed B4–7 on eight d002 racks:136attention+8FFN,17local senders/FFN.
Five trays/rack reserve160GPUs for144active workers. Qwen3 TP1,MB2,FP4 all-push,
128K synthetic KV/perfect routing; no new alignment test. User requested outlier
exclusion: remove warmup then steps >10% from median. Only B4 step14=67.963392ms
is excluded; original strict rejection retained. B5–7 pass the original gate.

| Batch | Samples | Mean latency ms | TPS/user | TPS/active GPU | QKV ready median / p95 µs |
|---|---:|---:|---:|---:|---:|
|4|14/15|17.447|57.32|216.53|4.946 / 5.490|
|5|15/15|21.041|47.53|224.43|6.516 / 9.841|
|6|15/15|21.807|45.86|259.85|6.096 / 7.248|
|7|15/15|26.267|38.07|251.69|7.741 / 12.461|

Latency=max representative A1/F137 CUDA graph span, averaged, excluding CPU gaps.
TPS/user=1000/latency_ms; TPS/GPU=TPS/user×136×batch/144. QKV readiness includes
receiver unpack, same-host paired2,820samples/case, origins agree within25ns.
All receives prearmed; eight FFN publisher medians2.66–3.55µs. No progressive comm
slowdown;4–6µs completion goal not met for every batch. No all-destination bound.

Opt-in `MINISGL_AFD_QKV_LAMPORT=1`:4×256 CTAs/peer,128-byte LL128 lines with
120-byte payload+8-byte epoch, receiver poll/reload/unpack, no return ACK.
Requires Qwen3 64Q/4KV/head128,TP1,MB2,uniform1–4decode rows/peer. Default remains
direct-edge. GPU40681 passed23,808exact receiver checks. Shared fabric inboxes
require128-byte view alignment;16-byte alignment caused a caught startup failure.
EP8 B7 all-push capacity1920 after rounding, others1152. Explicit rack maps support
partially occupied trays. Use `prepare_afd_rack_run.py` and sequence wrapper;
allocation requires exact hosts,`--segment=5 --consolidate-segments`.

B7 scheduling is symmetric: both lanes release after FMHA and before publish_o.
Four-token return to Qready125.72µs exceeds intervening three-token attention116.00µs.
Thus3→4 has10.04µs mean gap/15% publication overlap;4→3 has≈zero gap/100% overlap.
MoE alone finishes early; optimize the full post-MoE/QKV return path. Sub-µs negative
trace gaps are timing uncertainty. CPU40696 reproduced all15 graph spans.

Task=`R/artifacts/aga_rack_scaling/20260922_a136_f8_lamport`; `plans.json` selects
`resume2/b*`, `resume2-jobs.json` records commands, `SUMMARY.json` retains analysis.
Local cases: `scratch/afd/cases/qwen3/isl_131072/ep_8/a17-f1/batch_{4,5,6,7}/lamport-push-oci-aga-20260922-j40692/`.
Exactly A1/F137 trace pair+metric/result/manifest percase. Extra local raw traces
and SQLite exports removed; full raw data remains remote.
[Results/scheduling analysis](scratch/afd/reports/qwen3/aga-a17-lamport-20260922/README.md).
Staging:36selected CPU checks pass across40697/40698 after generic-Ray environment
forwarding fix; scorer CLI and3remote Bash checks pass. Code/docs synced to AGA;
logs in `scratch/afd/validation/aga-a17-lamport-20260922/staging/`. All jobs terminal.
