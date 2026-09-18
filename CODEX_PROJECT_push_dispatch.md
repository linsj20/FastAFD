# MegaMoE all-push dispatch

## Checkpoint: MegaMoE all-push latency optimization — September 18

User requested a progress commit. Default payload mode remains pull. The maintained
compact-push kernel is the earlier split-gather prototype; the measured best is
an explicit header/router/native-descriptor overlay, now retained with compact
evidence under `scratch/megamoe/lamport_payload/`. See
[dispatch memory](CODEX_PROJECT_push_dispatch.md) and the
[checkpoint source map](scratch/megamoe/lamport_payload/CHECKPOINT.md).

Confirmed Qwen3 EP8 (global256, rows32, H4096/I1536/E128/topk8): ordinary
MegaMoE-only helper37.16655 vs previous-best37.24400us (-0.208%, pure ABBA).
Across the two EP8 jobs,144 exact and144 replay checks passed. Three instrumented
captures give independent medians, metadata start=0: exposed dispatch8.928us,
first MMA→last accumulator25.280us, exposed combine4.960us, cleanup0.192us;
metadata→combine-end39.712us and metadata→kernel-end39.936us. Independent
medians need not sum. Compute is near historical pull25.024us.

The 37.167us ordinary kernel duration excludes metadata before kernel entry;
metadata-origin decomposition and producer+kernel timing include that work.
All buffer initialization/reset is outside timing. Instrumented and ordinary
measurements are separate scopes, not subtractable overhead estimates.

Unchanged-code EP16 ABBA: Qwen3 global192/rows12 helper23.90270 vs24.22975us
(-1.35%); Qwen3.5 global512/rows32 helper49.30880 vs49.56815us (-0.52%).
Both orders improve, but Qwen3.5 remains0.68% slower than historical pull48.97440us.
32 reference records,128 exact comparisons and160 replay checks passed. No EP16
phase captures. Qwen3.5 global912/rows57 is unsupported (M64/128 epilogue threads);
current overlay requires M<=32/256 epilogue threads, one expert wave, and prepared
capacity. No consistent full-chain gain or general default promotion is claimed.

CPU7256953/7259660 passed ordinary/instrumented builds and role register gates:
128 registers,24 stack bytes,zero spills; the narrow helper has no local memory
or profiling markers. GPU7257013/7257222/7259728 completed successfully. Preserve
previous-best1536 and historical pull controls. All jobs are terminal. Detailed
attempts, protocol proof, rejected paths and remaining ideas are in dispatch memory;
communication ideas remain in [MEGAMOE_COMM_EXPLORATION.md](MEGAMOE_COMM_EXPLORATION.md).

Checkpoint verification (September 18): synced the candidate tools/code and
selected source/evidence assets to HSG `source/local-mirror`; remote `bash -n`
passed for `profile_megamoe_iket.sbatch`. The current latency summarizer replayed
all four saved EP8 arms (three captures each) with ordinary-pass validation and
produced `remainder_helper/checkpoint_comparison.{json,md}` on HSG. No GPU rerun
was needed: kernel sources were unchanged from the passing jobs above. Git
whitespace review passed after documentation cleanup. No global instruction or
skill change is needed; workflow lessons are recorded in project memory.

## Historical experiment journal

The entries below describe decisions at the time of each run. Earlier statements
that no commit was requested are superseded by the September 18 checkpoint.

## Current acceptance and state

User requests full latency-chain optimization: overlap independent work, release
MMA early and broadly, and get compute time at least comparable to historical
best (~24.8us previously measured with operand-ready proxy), alongside lower
MegaMoE-only kernel time. Recovery to current Lamport parent's~30us is insufficient.
Metadata may retain push + explicit ready publication; metadata Lamport NOT
required. History best: metadata remote slot-reservation atomics + route_ready
SYS release/acquire; dispatch TMA pull; combine negative-zero Lamport push/poll.

Fixed EP8,32 rows/rank,H4096,I1536,E128,topk8,8sender/150compute CTAs only. All
payload preparation/reset outside kernel timing. Preserve baselines, no promotion
or commit until measured acceptance. Local main is split_gather experiment;
local router restored to destination-list late-CountsReady/PDL parent after
rejecting early-input launch. Frozen active experiment inputs remain unchanged.

A=/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD/artifacts/push_dispatch.
Most overlays/plans/evidence: A/lamport_payload/{latency_overlap,early_gather,
early_inputs,history_actual,arrive_gather,split_gather}; local compact counterparts
under scratch/megamoe/lamport_payload. Overlays own changed main/dispatch headers
only and symlink unchanged includes; no full source copies. Original push/pull
baselines remain A/startup_detail/{push_headers,pull_headers,push_router.cu,pull_router.cu}.

## Matched six-arm validation — GPU7236911 / analysis7236914

GPU COMPLETE19m35s on nvl72022-T14/T15; analysis COMPLETE49s.144 exact output
and144 replay checks PASS. All six use actual
first-MMA marker; historical main differs only by marker placement/name.

| Arm | Ordinary kernel mean(us),10 calls |
|---|---:|
| Historical pull |37.8336|
| Original allpush |43.6512|
| Destination-list uniform Lamport parent |42.8384|
| Parent + epilogue metadata prefetch |42.5120|
| Epilogue + early local-input/PDL publication |46.0640|
| Early A gather + original five-warp rendezvous |43.1808|

Matched actual-MMA compute spans and startup samples are recorded below. Epilogue gain repeats (~0.3us), but best
candidate remains4.6784us slower than history. Early-input launch rejected for
kernel-time target, working-tree router restored from early_inputs/router.before.cu.
Early A with original join recovers previous independent-arrival70.71us regression,
but remains slightly slower than parent. Compact ordinary evidence:
early_gather/ordinary_7236911.json. No acceptance claim from these numbers.

| Arm | First actual MMA from metadata start(us),all3 | Compute span(us),all3 |
|---|---|---|
| History |115.488/10.368/9.632|24.960/25.632/25.120|
| Original |11.104/11.808/11.968|28.704/29.056/28.928|
| Parent |12.608/13.952/12.160|30.112/30.432/30.304|
| Epilogue |12.032/13.856/12.064|29.376/29.792/30.144|
| Early inputs |14.336/15.456/14.272|32.512/30.432/32.224|
| Early gather |11.776/13.504/11.840|30.592/30.848/30.144|

Historical sample0 has metadata acquisition at110.400us (vs4.992us in both other
samples); preserve this outlier. Its compute span remains24.960us. All ordinary
kernel means are separate from sparse profile envelopes. Matched compute target
is now25.120us median, with identical actual-MMA boundary; parent30.304 and early
A30.592 do not meet it. Per-pair median spans history23.168/23.904/23.392us vs
parent29.280/29.024/28.256 show a real pipeline gap independent of start skew.
Early gather advances first MMA in all3 samples vs parent, but does not improve
compute/ordinary kernel time. Next experiment therefore targets producer overlap
throughout the pipeline, retaining original full-barrier accounting.
Evidence: early_gather/{comparison_7236911.json,outcome_7236911.json}; raw traces
and six complete decomposition.json files stay remotely under iket/<arm>/7236911.

Preparation: CPU7236663 PASS6m37s for early gather and exact router/include paths;
historical actual-marker CPU7236902 PASS2m20s. CPU7236509 PASS53s original actual
marker; CPU7236547 PASS3m32s exact router paths. Pending7236801/7236802 cancelled
before GPU allocation to add historical arm. Slurm rejected dependency on aged
completed7236547; removed after successful completion verified, retained live
build dependency. No GPU time spent on cancelled pending plans.

## Prepared next compute-overlap comparison

Concurrent7239114/analysis7239115 COMPLETE12m28s/44s on nvl72086-T12/T17;
96exact+96replay PASS. Ordinary mean(us): concurrent41.3920,parallel42.7712,
prefetch45.2832(noisy40.928–49.344; median45.920),history38.1568. Concurrent
saves1.3792us against stable parallel control; best-prefetch comparison needs repeat.
All actual-MMA starts metadata-origin(us): concurrent15.936/15.712/16.256,
parallel13.632/13.248/13.600,prefetch13.632/13.472/12.000,
history9.664/10.080/9.760. Compute spans(us): concurrent24.864/24.800/24.992,
parallel27.136/27.392/27.296,prefetch31.904/32.352/31.328,
history25.248/25.216/24.576. Concurrent achieves history-like compute, but firstMMA
is later: earlier-start acceptance NOT met, no promotion.
Release spread concurrent.960/.608/.736us vs parallel4.608/4.160/4.352us.
All four metadata-worker pairs actually issueMMA before localGenphaseEnd in all
three samples, by.768–1.600us; parallel haszero overlappingpairs. Per-pair concurrent
compute medians23.872/23.744/23.872us. This proves serialization removed locally.
Current new bottleneck is payload/SF readiness: concurrent payloadsend starts~.3–.6us
later, some sender retirement~2us later, metadata publication~2–3us later.
Drained→retirement includes sender joins/fence/tail-store completion, not pureNVLink.
Evidence concurrent_dispatch/comparison_7239114.json (full samples retained).

FollowupsCPU7239386 COMPLETE2m18s, both ordering/init variants normal+IKET/register
gates PASS, init selector gates PASS. Next direct TailPublishStart marker replaces
redundant PayloadSenderDone; GenphaseEnd still records sender retirement. Marker is
immediately after CTAjoin/SYSfence/CTAjoin and BEFORE tail stores. Same30eventtypes.
Both ordered_headers/init_headers now use dispatch_timing.cuh; original headers/main
and dispatch used by7239114 preserved unchanged. profile parser accepts optional new
marker and validates sender count; summary exposes it. bash -n passed.
CPU7239590 validates updated instrumentation + both selector roots; GPU7239646/
analysis7239647 submitted afterok. Four arms ordered concurrent/init-overlap/prefetch/
history under concurrent_dispatch/followup. Inputs frozen, no newGPU shape cases.
CPU7239590 COMPLETE2m29s all gates PASS; GPU7239646 live on nvl72127-T06/T10,
analysis7239647 afterok. Live ordinary ordered42.0512us(median41.568,min40.736,
max44.928) vs init40.8800us(median40.880,min40.288,max41.664); ordinary correctness/
replay pass, decomposition pending. No claimed firstMMA gain before that capture.
Summary pairs sender issue/drained/tail-publish/retirement
by exact CTA+warp, validates ordering and retains raw counts; old captures lack
the new boundary explicitly. This avoids cross-sender extrema subtraction.
The GPU run will isolate setup overlap and locate publication delay; no speculative
sender fence removal. Before promotion fix init variant's inherited comment that
claims all initialization completes before roles split (actual splitwait is explicit).


Latest7238788/7238789 COMPLETE11m05s/44s on nvl72078-T14/T17,96exact+96replay
PASS. Ordinary prefetch41.9488/tile-SF44.6784/parallel42.8480/history37.6320us.
Parallel packing saves1.8304us vs serial tile-SF but remains0.8992us above best
prefetch; not promoted. Actual-MMA metadata-origin samples(us):
prefetch11.808/11.776/11.840, serial14.080/13.952/14.208,
parallel13.152/13.280/13.216, history9.920/10.144/9.920.
Compute spans: prefetch32.192/31.776/30.976, serial28.160/29.344/27.744,
parallel27.008/27.872/27.136, history24.704/24.768/25.600.
Parallel per-pair medians23.424/23.168/23.264 vs history23.040/23.424/23.712us;
release spreads4.480/4.416/4.544us vs history1.056/.992/1.088. All three parallel
last-finish pairs are metadataCTAs0..7. Evidence tile_sf_parallel/comparison_7238788.json.

User explicitly says remove serialization. Placement-only dispatch_tail CPU7238856
PASS3m51s, but NO GPU submitted; held/superseded. New concurrent_dispatch keeps
original17workerCTAs and runs workers in physicalwarps8..11 (first epilogue WG),
while warps0..7 gather/load/issueMMA. Initial cluster/barrier/TMEM init precedes role
split. Remove pre-GEMM genphase CTA join; shared-counts join becomes existing
compact-unused dispatch_barrier0, init32/one release-arrive per warp0 writer/all
consumers acquire phase0. GenphaseWG consumes sharedcounts only after returning.
Shared counts[0,64), scratch[128,17152), GEMM starts17408; source slotsdepth1
(fixed32rows/8senders/4warps meansone token/warp, no useful lookahead lost).
For extra rows refill occurs after prior complete bulk wait/fence; no extra GPU cases.
Namedbar3 stays owned by sameWG8..11, then reused later by its epilogue; WG12..15
usesbar4. Existing cleanup joins/retirement protect reuse and barrier invalidation.
No new shared allocation/host ABI/payload-ready flag; tile readiness unchanged.
CPU7238978 FAILED47s: sizeof result/uint32 constexpr_align type deduction; explicit
uint32_t fixes it; CPU7239077 COMPLETE2m06s, normal+IKET/selector gates PASS
(128regs/24-byte stack/0spill). No GPU for failed build. GPU7239114 RUNNING on
nvl72086-T12/T17/analysis7239115 pending; candidate-first four arms
(concurrent/parallel/prefetch/history). Concurrent ordinary exact/replay PASS,
mean41.392us,median41.376,min40.928,max42.240; matched controls/decomposition pending.
Each capture subprocess has240s timeout to bound an unexpected synchronization hang;
Slurm terminates remaining steps when batch fails.
Genphase/compactDispatchEnd moved to physicalwarp8 with128initial-register bound;
compute marker roles unchanged. Sources frozen during builds. No default promotion.
Current ordinary SASS confirms count STS0x5200→ARRIVE0x5250→TRYWAIT0x52b0→
scheduler LDS0x53e0. Before promotion add compiler-only memory clobbers around
Cutlass count-barrier arrive/wait wrappers (no clobbers in wrapper), then revalidate;
current live sources unchanged. Potential next overlap: run startup worker between
second cluster-arrive and cluster-wait, allowing it to overlap barrier/TMEM init.
Do not remove cluster initialization ordering; wait current decomposition first.
CPU followups prepared within concurrent_dispatch: main_ordered.cuh adds compiler
memory clobbers around the new count barrier; main_init_overlap.cuh additionally
runs startup between second cluster arrive/wait. Compute acquires initialization
before use; startup only touches disjoint scratch. No extra GPU submitted pending
7239114 decomposition. CPU7239258 failed43s due build-plan rename accidentally
changing the original TU path;7239310 cancelled28s before same error. Corrected
TU path preserved; consolidatedCPU7239386 compiles ordinary+IKET both variants,
register gates and init-root selectors. ordered_headers/init_headers isolate live
main. Summary now reports pair-local firstMMA minus GenphaseEnd (negative proves
MMA overlaps at least one local startup worker); raw timestamps retained.
Current ordinary7239114: concurrent41.392,parallel42.7712,prefetch45.2832,
history38.1568us. Prefetch has broad40.928–49.344us range(median45.920), so don't
claim clean best-path gain from its mean; repeat needed. Profiles still running.


GPU7238494/analysis7238495 COMPLETE9m15s/47s on nvl72038-T01/T03;
72 exact-output+72 replay checks PASS, stderr empty. Ordinary mean(us):
prefetch42.7552, tile-SF45.3920, history37.8176. Tile-SF regresses kernel by2.6368us.
Actual first MMA from metadata, all samples(us): prefetch13.504/13.280/11.904;
tile-SF16.192/14.336/16.384; history18.688/9.824/10.016 (outlier retained).
Compute spans(us): prefetch32.192/32.640/31.680; tile-SF28.096/27.936/28.192;
history24.672/25.344/24.352. The per-pair compute medians tell a stronger story:
prefetch29.024/28.768/28.608; tile-SF23.648/23.232/24.224;
history23.104/23.456/23.104. First-MMA release spreads tile-SF5.056/5.184/4.896us
vs history1.504/1.184/1.152us. Tile-SF recovers most per-pair compute throughput;
late, staggered tile release now deserves priority, not a speculative gather join.
First-MMA→first-FC2-epilogue pair medians tile-SF19.232/19.232/19.872us versus
history19.520/19.584/19.680us. This is not an FC1-only attribution. IndexReady is
last metadata-CTA publication; it includes row acquisition/SF packing for all
owned tiles. Evidence: tile_sf/comparison_7238494.json. No target win/promotion.

Parallel SF packing CPU7238630 COMPLETE2m41s; ordinary+IKET register/selector
gates PASS,128regs/24-byte stack/0spills. GPU7238788/analysis7238789 submitted:
four matched arms prefetch/tile-SF/parallel/history. Metadata keeps row ownership;
SF copies stripe uint4 chunks across all128 metadata threads instead of16 lanes
serially copying32 words each. Each active SF task derives its own source descriptor
and acquires row tail; warp-uniform masked loops, unique row/K writes, original
proxy fence/CTA join/tile release. No extra barriers, compute header unchanged.
BM32/64/128 all valid-M ownership mapping checks pass. SF chunk means16bytes
(uint4): H4096 has128SFbytes/token,8chunks across8lanes; format/value unchanged.
GPU decision is now justified by measured per-pair recovery and release spread.

Further trace/source audit identifies the late pairs as metadataCTAs0..7: e.g.
tile-SF sample1 CTA0 IndexReady16.416→dispatchEnd16.448→metadataAcquire17.216
→gatherIssued18.528→actualMMA19.392us; regularCTA18 MMA14.784us. All dispatch
participants join the whole CTA before GEMM shared-memory/cluster initialization.
The scheduler also gives first17CTAs3FC1+3FC2 blocks(132Ksteps), whereas last18
have2FC1+3/4FC2(100/112Ksteps). Thus current placement delays heavier compute.
Prepared dispatch_tail assigns identical logical17workers to physical132..148
(pair-aligned132..149) instead of0..16. Logical worker index passed explicitly;
all source/expert mappings,150computeCTAs and8senders retained. Parent parallelSF.
CPU7238856 submitted; no GPU until parallel result. Ownership mapping recorded
in dispatch_tail/ownership_check.json. Candidate uses own main/dispatch overlay;
live input snapshots unchanged. No claim this scheduling change is a win yet.
Analyzer now includes startup markers for last-starting and last-finishing pairs,
not just the first-MMA pair; rerun on7238494 passes all pairing/order checks.
This confirms exposed dispatch duties in the delayed pairs without new GPU markers.
Open small opportunity: packed-SF path now uses SF TMA but compact branch still
skips prefetch of the FC1 SF descriptor. Keep isolated until current result.

Follow-up7238237/analysis7238238 COMPLETE8m29s/44s on nvl72078-T14/T17.
72 exact-output and72 replay checks PASS; stderr empty.
Ordinary split42.7168us, SF-prefetch41.8976us, history37.7728us; gain0.8192us.
All actual-MMA starts from metadata(us): split12.128/11.584/11.936;
prefetch13.856/11.776/11.904; history58.816/10.368/10.176 (outlier retained).
Compute spans(us): split30.688/32.096/31.968; prefetch31.776/31.328/30.976;
history24.832/25.216/24.576. Prefetch's0.640us median compute gain is useful,
but first-MMA median essentially unchanged; does not meet historical compute
or earlier-MMA acceptance. Per-pair compute medians split28.704/29.056/29.344;
prefetch28.928/28.768/28.704; history23.232/23.808/23.520. Most gap remains
before first FC2 epilogue. Evidence sf_prefetch/comparison_7238237.json.
Preserve best observed all-push ordinary path; no default promotion.
CPU7238203 COMPLETE2m15s, normal+IKET and both selector paths PASS.

User asks whether gather-load MMA should be efficient when indices are cheap.
Confirmed same MMA instructions/layout and indices cached outside K loop. Existing
source_counts header indeed uses activation gather4 with canonical SF TMA.
Matched direct_sf_gather7213127 older-marker compute span25.344→29.376us when
moving SF into compute; this supports investigating SF rather than assuming an
inherent gather4 bottleneck. Different-era absolute spans are not directly matched
to current actual-MMA measurements.

Next tile_sf prototype CPU7238445 COMPLETE2m18s, normal+IKET/selector gates PASS: reuse preallocated canonical SF
layout/descriptor; metadata worker acquires ONLY its tile's source-row tails,
packs SF with indices/scores, fences+joins, then releases existing tile counter.
Activation gather4 remains; coordinator restores SF TMA. All consumers retain
stage-empty protection and original4 full arrivals; no five-warp rendezvous or
per-stage scalar SF in compute. Single-wave guard retained. No new payload-ready
flag/global payload barrier; no activation materialization into HBM. Row publication
now includes SF, so GenphaseIndexReady/MegaIndicesReady semantics include it;
MegaGatherSfReady is coordinator warp4's tile-SF acquisition. CPU scalar mapping
check confirms unique/in-bounds positions BM32/64/128, no extra GPU cases.
128regs/24-byte stack/zero spills; dispatch R39/48, other rolesR18/40; uniform
gather operands remain hoisted in inspected SASS. Frozen snapshot only;
working tree unchanged. GPU7238494 / analysis7238495 submitted: best prefetch,
tile-SF, historical pull. Same fixed EP8 only, normal arms first. Candidate
MegaGatherSfReady profiler register bound40 matches coordinator warp4; analyzer
now records explicit per-arm index/SF marker semantics in output notes.

GPU7237512/analysis7237513 COMPLETE14m20s/49s, nvl72035-T11/T12.
Ordinary means parent43.0496us/split42.9120/arrive42.9504/history37.7056.
All three actual-MMA starts, metadata-origin(us): parent12.032/12.000/14.240;
split11.680/13.152/12.032; arrive12.000/11.680/11.872; history57.760/10.816/9.760.
Compute spans(us): parent29.920/29.536/30.048; split31.200/32.736/31.296;
arrive30.144/30.624/30.560; history24.864/24.256/25.248.
Historical sample0 delayed metadata acquisition52.576us retained; others~5us.
Split releases A early but compute is worse; nonblocking arrivals roughly match
parent ordinary time. Neither meets acceptance; no promotion. Per-pair medians
in split_gather/comparison_7237512.json; first-FC2-epilogue partition still places
most parent/history gap before that marker (~24.2 vs19.7us).
Existing7236911 markers further partition each CTA pair at first FC2 epilogue:
first-MMA→first-FC2-epilogue median samples history19.616/20.224/19.872us versus
parent24.192/24.448/24.416us; remaining epilogue→compute-end history4.192/3.680/
3.808 versus parent3.776/3.776/3.680. Most gap precedes first FC2 epilogue; this
boundary is NOT FC1 end or FC2 MMA start. Evidence early_gather/pipeline_split_7236911.json.

User authorized optional Computelab PIC-C, then said it is unnecessary if a
concrete next idea exists. Setup PAUSED; no GPU/CPU job submitted there. Eight-B200
crun dry-run matched hardware but none immediately free (normal queueing).
PIC-C setup knowledge: existing attention_profile/.venv (Torch cu130), profiler
/home/scratch.svc_compute_arch/release/pic-c/latest/pic-c; no nvcc in that venv.
Created only empty FastAFD source/artifact/profile directories on Computelab;
small rsync stalled in shell startup and was stopped. No kernel source copied.
Resume only if HSG decomposition leaves a question utilization can resolve.
Next isolated sf_prefetch prototype prepared from split: scalar SF loads into
registers before stage-empty wait, shared writes remain after consumer release.
Source immutable after row acquire; one-wave guard and original stage accounting
retained, no additional registers by source design. CPU7238203 submitted for
normal/IKET/register/router gates. Normal+IKET device/register gates PASS,
128regs/24-byte stack/zero spills; SASS SF LDG0x6e50/0x6e70 precedes empty-stage
TRYWAIT0x6ec0. GPU7238237 afterok CPU / analysis7238238 submitted for matched
split/prefetch/history ordinary+decomposition. Working-tree kernel unchanged.

Four-arm parent/split32/arrive32/history32 GPU7237512 / analysis7237513 submitted
under split_gather after reviewing7236911 decomposition. Both candidates
pass ordinary+IKET builds,128regs,24-byte stack,zero spills, reachable48/40-role
register gates, and exact router/header-path selector prebuilds.

- **arrive_gather**, CPU7236952 COMPLETE2m41s: four A/SF producers signal per-stage
  named barriers5..14 without waiting; only coordinator waits then performs the
  original full arrival. Stage-empty wait protects barrier-slot reuse. Retains
  original4 full-stage arrivals and early A issue. SASS has BAR.ARV and existing
  post-TMA proxy fence. GPU7237512 validation complete above.
- **split_gather**, CPU7237041 COMPLETE5m28s (slow CPU image extraction): two A
  gather warps and two SF warps. Only SF+coordinator join (96threads), original4
  full arrivals. A issuers have no per-stage SF loads/stores/fences/join. Generated
  code confirms role branches and uniform TMA operands without stage-local R2UR
  reconstruction in inspected A blocks. Bounded evidence: sass_review.json.

Split correctness invariants: A warps poll/acquire their own source rows before
TMA. SF warps independently acquire full-tile SF rows and warp-sync before scalar
loads. Unique SF writers per valid row per CTA. Coordinator sets all expected A
bytes BEFORE SF join, then full-arrives; A/B bytes + SF completion jointly gate
MMA. MMA empty-stage release protects A/SF reuse; final dispatch/epilogue join
protects cleanup. Integer coverage check for BM32/64/128 and all valid M passed
(no extra GPU shapes). MegaPayloadReady/GatherIssued sample A warp0; SFReady
samples SF warp2. Updated analyzer wording and run contract preserve granularity.

Both prototypes fail clearly outside single-expert-wave scheduling or supported
warp/barrier geometry, with no fallback. Gather skips FC2 stages; multi-wave phase
tracking requires separate design before broader support/promotion. Current
fixed EP8 case is single-wave. GPU7237512 runtime/performance evidence is above.

## Prior four-arm result — GPU7236394 / analysis7236395

COMPLETE11m49s/46s;96 exact output and96 graph replay checks PASS. Ordinary means:
original43.504us, parent43.0304, epilogue42.7296, independent-arrival gather70.7104
(rejected). Original capture used operand-ready proxy; other arms actual MMA.

| Arm | First MMA from metadata start(us),3 samples | MMA→compute end(us),3 samples |
|---|---|---|
| Parent |12.032/12.000/12.064|30.016/30.656/30.080|
| Epilogue |12.128/12.192/12.096|30.592/30.240/29.952|
| Independent arrival |11.744/11.680/11.680|58.336/57.472/57.984|

Critical-pair gather issued: parent11.616–11.680us vs independent9.568–9.888us,
metadata-origin. Earlier A is real, but first MMA advances only~0.35us while
compute almost doubles. Per-pair first-MMA→own-compute-end medians: parent
29.056/29.440/28.864; epilogue28.352/28.224/28.480; independent54.656/54.816/55.104us.
Thus start-time skew does not explain slowdown. These are pipeline spans with
waits, not isolated tensor-core instruction durations. Summary validated on all
75CTA pairs/sample: latency_overlap/comparison_7236394.json and source decompositions.

Independent protocol had12 full arrivals (A1+B1+SF4 per CTA), SF warp proxy fence+
syncwarp+arrival, FC2 counted5 arrival for A/SF. Stage spacing46→58 SASS instructions.
**Attribution correction:** parent also has MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S;
these were not added. Early A moves fence after TMA, but retained-join43.1808us
recovers~27us with that ordering, so fence ordering alone cannot explain regression.
Evidence points to independent-arrival protocol/FC2 accounting; exact cost unisolated.

Ownership bug found before unsafe run: original SF/poll warp0 covers rows0–7,
but at valid16 companion A warp0 gathers8–11. Old join transferred readiness.
Cancelled7236299 at1m45s and analysis7236314 before gather arm ran. Corrected
same-role SF/poll quartets map to each issuing warp's four-row gather groups in
BOTH CTA halves, preserving complete SF rows/CTA with no extra polls. CPU7236393
PASS53s rebuilt corrected gather before7236394. Split candidate uses separate
explicit A/SF acquisitions instead, with its own coverage check.

## Instrumentation and failed preparation retained

Direct markers, no injection framework. PMTrig5 has30-event effective budget:
main18+dispatch12. REPLACE MegaComputeStart proxy with MegaMmaIssue immediately
before first fma; analyzer requires exactly one contract and labels actual/proxy.
All existing dispatch-start markers retained. Extra SF-start/epilogue wait markers
removed to stay in budget. ExtendedNativeDump is NOT a working substitute.

CPU7235856 extra-marker limit; CPU7235915 parent passes then epilogue32→32u
compile fix; CPU7235948 variants pass. ExtendedNativeDump GPU7235950 failed6m52s
at profiler teardown(return−11) after parent exact/replay all8ranks; no candidates
ran. Partial ordinary original44.2272,parent42.9952 retained. CPU7236210 confirms
even one added event exceeds budget; replacing proxy passes CPU7236287 in59s.
Failed-build dependent GPU/analysis jobs cancelled before allocation. Plans/patches
and failed evidence remain under latency_overlap; do not silently drop samples.

Maintained profile_megamoe_iket.sbatch supports normal-first then profile phases
with same-run baseline required; no automatic fallback. summarize_megamoe_latency.py
reports common metadata/selector origins, critical CTA pairs, MMA-release spread,
per-pair pipeline spans and last-finishing pair. New pair analysis validated on
real7236394 captures. Raw traces remain remote; retain compact summaries locally.

## Early-input router experiment (rejected kernel-time result)

CTA selectors128/8 and512/10 only: top-k lanes store local IDs/scores, masked
syncwarp, lane0 release-increments CountsReady and triggers PDL BEFORE remote slot
atomics. Invalid physical rows increment once too. Final route_ready SYS release/
acquire and generic selector unchanged; no allocation/ABI changes. CPU7236084
PASS5m33s; exact epilogue paths prebuilt by7236663. Early launch can inflate
kernel duration; compare common producer/metadata origins. Frozen router.cu and
router.before.cu retained under early_inputs. Working tree restored after46.064us
result. Matched metadata-origin firstMMA worsens to14.272–15.456us and compute
span30.432–32.512us; neither kernel time nor useful startup improves.

## Systematic latency audit — September 17

User requests auditing all latency sources, not only push, and avoiding serial
work after long waits when other warps can overlap it. Completed static audit of
router, metadata, startup/cluster joins, sender, SF/gather, weight/MMA pipelines,
FC1→FC2 dependency, epilogue, in-band combine and cleanup. No GPU jobs or kernel
edits. Full dependency/ownership table and implementation priorities:
[latency audit](scratch/megamoe/lamport_payload/LATENCY_AUDIT.md).
Key candidates: epilogue score/destination loads currently after accumulator
wait; SF loads+rendezvous before activation gather issue; communicating CTAs
finish genphase before compute initialization; full-row sender-completion-based
tail publication; all-FC1-mask FC2 wait; two combine vectors/lane processed
serially. Existing A/B/MMA specialization and cleanup overlap must be preserved.
At fixed32,8senderCTAs×4warps already cover one row/warp; deeper next-row buffering
alone has no work to overlap. Bulk waits are issuer-thread scoped, not movable
to an arbitrary helper warp; no explicit receiver-consumption ACK in this path.
Metrics caveat found: MegaComputeStart precedes SF UTCCP and actual MMA issue;
first-MMA tables use this operand-ready proxy. Gather/payload markers sample
warp0, not all gather warps. No timing benefit claimed from static findings.

## Latest completed compute-loop repair results — September 17

uniform_gather GPU7234938 COMPLETE8m19s, analysis7234939 COMPLETE43s;
broadcast_uniform GPU7235086 COMPLETE6m37s, analysis7235087 COMPLETE45s.
Warp-uniform polling fixes the demonstrated TMA operand-hoisting regression.
Destination-list matched ordinary kernel means: original44.0768us,
divergent-Lamport47.712us, uniform-Lamport42.976us. Compute spans respectively
30.048/30.272/30.432,36.352/36.544/36.224,31.264/31.744/31.456us.
Uniform candidate kernel-relative firstMMA10.208/10.272/10.048us vs original
10.688/10.752/10.624, but selector-origin firstMMA16.032/15.840/15.936 vs
13.472/13.792/13.440: metadata/PDL shift still prevents a common-origin startup win.
Original-broadcast metadata isolates payload change: baseline43.5936us vs
uniform-Lamport44.5088us (+0.9152us,+2.10%). FirstMMA baseline10.592/8.736/9.280
vs candidate9.280/10.304/10.208; compute spans29.024/29.216/29.088 vs
29.312/30.656/30.656. Thus most compute regression is repaired, but no consistent
early-start plus kernel-duration win with unchanged router. Remaining overhead
not causally isolated; per-row polling/tail-store grouping are hypotheses.
Destination-list42.976us is now a measured kernel-only improvement, not an accepted
full-chain or common-origin startup improvement. No code promotion/commit here.
All four GPU/analysis jobs finished; result paths remain under their named overlays.


## Active: tile-local Lamport payload readiness — September 17

User requests negative-zero in-band push/poll so first MMA can start when its own
payload arrives. Latest user constraint: **use the fixed EP8 rows32/rank case
for now; no random-routing/extra-shape sweeps**. Focus on **MegaMoE kernel-only
time and first-MMA latency**, not producer+kernel timing (historical all-push
kernel44.480us; current paired pipeline numbers70/130us are not kernel times). Existing arbitrary4/65 checks
already finished before this instruction; all remaining timing/captures are EP8
rows32. Future submissions omit those validation arms. **Detailed dispatch decomposition is the primary diagnostic from
now on**: compare matched operation starts, tile indices/SF/A readiness and first
MMA; expect an earlier first MMA, verify rather than infer from total runtime.
User explicitly permits preinitialized buffers and requires **all initialization
and replay resets outside kernel timing**. No payload clearing in MegaMoE.

Current local candidate restores both bulk TMA send and gather. The final FP8
word of each source row remains0x80000000 throughout activation/SF bulk transfer;
after full TMA completion, its real bits are release-stored (canonicalize only an
exact negative-zero collision). Receiver acquires each needed row's tail word
once per FC1 tile, then uses the original SF prefetch/TMA gather loop. No separate
ready flag/counter/global payload wait. **Readiness is whole-source-row completion,
not individual128-column K slices**; unrelated experts/rows need not finish.

Preparation: existing symmetric-memory GPU barrier, reset addressable source rows
in both banks, barrier; separate preparation/workload CUDA graphs. Event pairs
exclude preparation. Primary metrics are Nsight MegaMoE-only duration and IKET
startup timestamps, preserving producer→MegaMoE PDL inside the workload graph.
No dispatch-buffer clearing in MegaMoE; no extra NCCL communicator/event nodes.

Row-tail7232585 PASS5m57s on nvl72114-T[10-11], analysis7232587 PASS45s.
Kernel control44.0928us vs candidate52.6976us (10 calls); first MMA control
10.720/10.720/11.136 vs candidate10.112/11.008/10.240us. Compute spans control
30.272/30.048/30.240 vs38.688/38.400/38.752us. Restoring TMA gather recovers most
of the109us regression but still loses to baseline. First-MMA spread ends at
13.056–13.568us control vs16.832–17.120 candidate; sender tail-word release/drain
ends11.232–11.968us and sender retirement13.376–13.952. Broader startup skew
contributes to compute span; do not equate span with intrinsic MMA throughput.
All48 fixed-input and48 replay comparisons passed. CPU7232584 device builds and
selectors passed; its register-review argument accidentally targeted prior tma
artifacts. Corrected standalone audit7233196 timed out during container extraction before
checks (3min was too short); no gate claim for7232584. Sender-fence CPU7233356
passes both corrected ordinary/IKET register gates and selectors,128regs24-byte
stack/zero spills. Original row-tail headers saved as validated_*.cuh
beside results before reusing the small candidate headers overlay.

Sender-fence-only row_tail_fence7233357 PASS6m18s, analysis7233368 completed
both decompositions. CPU7233356 PASS1m24s, corrected register gates and selectors.
Control44.1088us, candidate52.9216us; firstMMA10.848/10.720/10.752 vs
10.304/10.400/10.304, compute spans30.272/30.048/30.560 vs39.136/38.848/38.336.
No gain from one issuer SYS fence +parallel relaxed stores versus divergent
release stores: serialization hypothesis not supported. Candidate last tail-store
issue/drain11.552–11.872us; sender retirement13.536–14.112us. Frozen validated_*
headers retained beside this result; candidate overlay reused only after job end.

row_tail_retire: CPU7233936 PASS1m54s; GPU7233938 kernel control44.256us vs
candidate51.312us, detailed capture/analysis7233939 still pending. Original TMA
paths kept; redundant final bulk wait/SYS fence/CTA join removed; receiver polls
relaxed SYS then acquires once (old loop emitted CCTL.IVALL per failed poll).
This improves~1.6us over sender-fence version but still misses baseline. Shared
row_tail/headers now retains this variant while its capture finishes.

Active next: row_tail_cta CPU7234220, dependent GPU7234230 (also after7233938),
analysis7234231. Single SYS fence per sender CTA per four-row group, following
all four warp drains, then parallel actual tail-word stores. This restores the
baseline's one-fence-per-CTA pattern; prior Lamport version introduced four.
Receiver retains relaxed polling+one acquire and original TMA gather. Partially
active sender warps join every group consistently. Granularity includes sender
four-row group completion, still no all-expert/all-payload gate. PayloadDrained
now marks first bulk completion BEFORE CTA join/fence/tail stores (contract
records this change); SenderDone follows tail-store issue and CTA join. Separate
small two-header artifact overlay permits CPU prebuild while prior GPU capture
uses frozen headers; no full source copy. Same EP8 fixed32 only,10 kernels+3 IKET
samples/path, no promotion or commit. Do not edit active source overlays.

Latest user steering: metadata starts earlier in historical pull; use pull's way.
Confirmed pull publishes destination descriptors/count reservations before PDL;
broadcast push signals PDL before metadata network tail. Historical~−3.3 vs−1us
metadata starts are relative to MegaMoE entry and do not prove2.3us router-start
regression. Existing Lamport/control use the SAME broadcast router and starts are
nearly identical (~−1.1us). User clarified steady buffers: "allocate slot" means
atomic index reservation in preallocated storage, NO runtime memory allocation.
User also clarified terminology: BOTH protocols push metadata. "Pull-style"
means the old pull kernel's producer-built destination expert-list/count format,
not metadata transport direction. Lamport refers only to payload readiness.

Current local candidate and active pull_metadata chain CPU7234525/GPU7234526/
analysis7234527: router reuses pull's existing remote expert count atomic and
source-descriptor write before PDL. Compact additionally pushes each selected
route score to existing source-indexed score inbox (destination only); no layout
or native ABI change. Local CountsReady publishes immutable rows before PDL for
independent payload senders. Pull's existing last-CTA SYS route_ready release
orders descriptors/counts/scores. Receiver acquires route_ready, keeps natural
padded expert prefixes, scatters only destination expert lists, and publishes
per-tile index arrivals. This is producer metadata, not old sender_descriptors
path; no all-local-index barrier added. Lamport CTA-fenced payload and original
TMA gather unchanged. Frozen control+row_tail_cta parent+candidate all run EP8
fixed32 (10 Nsight kernels+3 detailed samples each). Primary kernel timing plus
new common selector-entry/metadata-start→first-MMA/kernel-end trace metrics;
common selector origin excludes earlier gate/quant kernel and is instrumented.
All payload reset remains outside timing. No promotion/commit; active sources
frozen in small pull_metadata artifact header/router overlay.

Destination-list ordinary7234526 results: original control43.7504us, matched
CTA-Lamport parent49.3696us, destination-list candidate47.7824us (10 kernels each).
Candidate initial fixed-input+graph-replay exact checks pass all8 ranks; final
IKET samples/analysis still pending. Thus metadata change helps~1.59us but does
not yet beat original baseline. No runtime memory allocation.

Compute codegen root cause found in ordinary SASS: divergent per-lane payload
spin makes stage/phase and TMA operands use lane registers; every K stage reloads
tensor-map pointers and repeats~8 R2UR operand conversions. Baseline keeps them
in uniform registers outside K loop. Four unrolled gathers are~67 lines apart in
row_tail_cta/pull_metadata vs46 in baseline. Local uniform_gather replaces spins
with warp-uniform __any_sync loop (inactive lanes use0), then one acquire and
original proxy fence; same readiness semantics and no separate publication.
CPU7234838 both ordinary/IKET register gates pass (128regs24stack0spills); SASS
confirms hoisted uniform operands restored, gather spacing46 lines again. Source
copies for comparison in scratch/megamoe/lamport_payload/compute_sass; remote
ordinary SASS retained under each run. New GPU7234938 afterok CPU+7234526 and
analysis7234939 queued; compares original control, matched destination-list
parent, uniform-poll candidate, fixedEP8 only. Parent metadata/router and payload
protocol unchanged. **Current local main includes uniform polling**, dispatcher
and router are destination-list variant. Active overlay roots are immutable.

pull_metadata7234526 COMPLETE9m18s,72 exact+72 replay checks; analysis7234527
completed all three common-origin decompositions. Destination-list metadata has
kernel-relative transfer start~−3.6us (broadcast~−1.1), but selector-entry→metadata
start2.208us vs1.728–1.760, selector-entry→firstMMA median16.192us vs13.088 parent,
and selector-entry→MegaMoE end55.520us vs52.320 parent (instrumented). Its smaller
ordinary kernel time therefore does NOT establish full producer-chain gain.
This was explained to user; both protocols push metadata, timing origin differs.

To isolate the compute repair without that producer/PDL penalty, broadcast_uniform
CPU7235085/GPU7235086(after7234938)/analysis7235087 queued. This uses original
broadcast router+CTA-fenced row-tail payload+uniform polling. Two matched arms,
original all-push control and corrected candidate, fixedEP8 32 only;10 ordinary
kernels+3 detailed samples each. Separate small artifact overlay; maintained local
still destination-list+uniform polling until both results determine best path.



Previous TMA-send/direct-gather result (A/lamport_payload/tma): CPU7231832 PASS2m20s;
GPU7231833 PASS6m42s, CPU analysis7231835 PASS51s. Same EP8 case, reset excluded.
Control kernel mean43.8688us versus candidate109.2192us. First MMA control
10.688/10.752/10.880us versus candidate9.472/9.056/9.312us: median advances1.440us.
But compute span29.632–30.496us expands to96.096–97.312us. Thus early-start idea
works, while replacing TMA gather with serial per-K SF/activation polling/staging
loses throughput. This is the motivation for restoring the original gather loop.
Candidate critical-pair indices7.776–8.576us, SF8.544–9.120, A ready8.992–9.408,
gather rendezvous9.024–9.440. Control global payload acquire9.824–9.952, SF10.304–
10.464, gather issue10.400–10.560. Both decompositions and all samples retained;
no end-to-end improvement or promotion claimed for this intermediate candidate.

Earlier attempt retained under A/lamport_payload:
- Generic vector sender + direct gather: CPU7230480 PASS1m09s, final no-publication
  CPU7230584 PASS1m15s; all four ordinary M16/M32/M64 and M32 IKET register gates,
 128regs48stack/zero spills. Sources frozen in candidate_headers.
- GPU7230585 FAILED1m48s in control graph capture because WORLD uses Gloo, not a
  capturable collective. Candidate not run; analysis7230588 cancelled.
- Dedicated NCCL preparation fixed capture. GPU7230896 passed all exact/replay
  checks including arbitrary4/65 cases (before user's fixed-case-only constraint)
  and32-row ABBA. Generic candidate pipeline ~130us versus~70us matched control;
  these are producer+kernel, **not** MegaMoE-only times. Control Nsight kernel
 57.9904us versus historical44.480us; full-capacity reset was outside timing but
  swept~256MB and changed cache/launch conditions. No causal attribution claimed.
- That job FAILED11m31s when IKET tracker segfaulted in cuGraphInstantiateWithFlags
  on the NCCL/external-event graph. No complete candidate kernel timing or startup
  decomposition; analysis7230897 cancelled. Current separate-graph/symmetric
  barrier preparation removes those graph features and clears only active rows.
- Signed-zero fixtures verified directly from saved rank0 FP8 bytes:4096/4096
  (rows4) and66560/66560(rows65) packed words collide with the reserved sentinel;
  all were canonicalized with exact output agreement. sentinel_input_check.json.


## Complete: matched direct startup markers — September 17

User requests equal detailed startup decomposition for historical pull and latest
restored8-sender all-push, including operation starts; no injection framework.
Direct guarded markers now cover producer metadata transfer/publication, receiver
metadata/count acquisition, activation/SF load, payload send/drain/publication,
index scatter/acquisition, SF/gather and first MMA. Ordinary builds remove markers;
no protocol/barrier changes. Historical pull received only equivalent source edits
in its frozen overlay. Wrapper forwards the existing IKET include/define to the
router; ordinary/instrumented module cache keys differ. Existing profiler retains
all source markers by CTA/warp and pairs each selector with its MegaMoE launch.

EP8,Q3,32rows/rank,150CTAs,FP8/FP4,balanced rank-interleaved routing, fixed Lamport
combine. GPU7228730 COMPLETED0:0/7m05s on nvl72172-T[02-03]; CPU analysis7228731
COMPLETED0:0/47s. Three rank0 samples/path; all48 exact comparisons+48 graph replay
checks pass across ordinary, discovery and instrumented worlds. Both M32 ordinary
and instrumented register gates pass:128registers,24-byte stack,zero spills; four
selector builds pass. Shell syntax and git diff whitespace checks pass. No active
jobs, optimization change, performance promotion or commit.

Times below are microseconds relative to MegaMoE entry. Keep all three samples:

| Path / sample | Local metadata transfer starts | Local metadata published | First-compute pair counts acquired | First MMA |
|---|---:|---:|---:|---:|
| Pull0 | -3.232 | 1.280 | 52.192 | 57.312 |
| Pull1 | -3.328 | 1.056 | 1.792 | 6.688 |
| Pull2 | -3.264 | 1.280 | 1.792 | 6.368 |
| Push0 | -0.960 | 3.392 | 25.152 | 29.376 |
| Push1 | -0.960 | 3.520 | 5.472 | 10.528 |
| Push2 | -0.928 | 3.424 | 5.216 | 10.624 |

Latest all-push detail for the two shorter samples (sample0 retained above):

| Boundary | Sample1 | Sample2 |
|---|---:|---:|
| First activation/SF local load starts | 1.472 | 1.536 |
| Payload send starts, first–last sender warp0 | 2.208–2.880 | 2.304–2.880 |
| Last local sender-warp0 TMA drain | 5.120 | 5.120 |
| All incoming metadata acquired, first–last exchange CTA | 4.064–4.128 | 3.968–4.032 |
| Counts publication issued | 4.800 | 4.512 |
| Index scatter starts, first–last exchange CTA | 4.992–5.056 | 4.736–4.800 |
| Index publication issued, first–last exchange CTA | 6.528–6.912 | 6.272–7.136 |
| Critical gather pair acquires indices, latest | 8.384 | 8.512 |
| Last local payload-ready publication issued | 9.088 | 9.184 |
| Critical gather pair acquires all payload, latest | 9.600 | 9.632 |
| First-stage SF preparation complete, critical pair | 10.144 | 10.240 |
| First-stage gather issued, critical pair | 10.208 | 10.304 |
| First MMA | 10.528 | 10.624 |

Critical pairs are32/33 and92/93. Sender CTA11's warp0 drains at5.120 in both;
its ready publication follows3.968/4.064us later. This interval includes other
sender warps, CTA joins, SYS fence and counter fanout; **not isolated fence or
network latency**. Payload acquisition→SF/gather issue→MMA takes0.928/0.992us.
Indices become available earlier than all-payload acquisition in these samples.
Do not add overlapping metadata, index and payload phase envelopes.

Pull's first-compute pairs60/61 and28/29 need expert2/1. With16rows/expert,
24FC1 N blocks/expert and token=CTA*4+warp, corresponding pull owners are CTAs8–11
and4–7. Their transfers start2.208–2.400/2.304–2.496, last PullArrived4.704/4.832,
last dispatch retirement5.824/5.920, pair tile-ready6.208/6.048, MMA6.688/6.368.
PullArrived follows overlapped SF/weight reads plus activation wait, not pure RTT.
The MMA operand barrier combines activation/weight readiness; no separate B-arrival
attribution is claimed. In the first sample, local metadata is published early
but incoming acquisition is late for both paths; untraced peer cause unresolved.

PDL entry differs: local metadata-start→MMA medians10.016us pull/11.552us push,
versus kernel-relative6.688/10.624. Thus the3.936us kernel-relative gap is not an
end-to-end dispatch penalty. Rank0 Nsight kernel means37.8144/44.480us (10 calls).
Nsight-pass max-rank CUDA-event95.159/96.883ms means contain profiler-start/rank
skew: the harness synchronizes before profiler start, not after it. Preserve them
but do not treat them as ordinary performance; use the prior long unprofiled
comparisons. Instrumented spans likewise are diagnostic, not acceptance timing.

Evidence remains remote in A/startup_detail/{result_summary.json,contract.json,
compiled_review.json,ordinary_compiled_review.json,compare-7228730.out}, plus
iket/{pull32,push32}/7228730/decomposition.json and raw traces. New source markers
are maintained locally; frozen historical marker patch and capture sources remain
in that artifact directory. User reiterated both paths and operation starts during
capture; no additional GPU cases were needed.

Preparation failures retained: initial injection overlay had an ambiguous anchor;
CPU7227339 failed52s at PMTRIG5 event cap. User rejected the framework; removed
local script, kept abandoned source only with failed-build evidence. Compact direct
markers stay within PMTRIG5. CPU7228115 PASS6m43s checked instrumented kernels and
four selector modules. GPU7228483 FAILED2m13s before capture because historical
patch omitted non-IKET no-op macro definitions; dependent analysis7228484 cancelled.
Fixed that guard, added both ordinary compile/register checks in CPU7228688
(PASS51s), then successful GPU7228730/analysis7228731. No failed result relabeled.

## Complete: explicit 16 sender CTAs — September 17

CPU7225701 COMPLETED0:0/3m10s; GPU7225802 COMPLETED0:0/11m02s on
nvl72016-T[09-10]. Strict final audit PASS all8arms:240 exact comparisons,
176 graph replay checks,144 recordings,0 separate conventional-combine checks;
fresh arbitrary4/65-row normal/alternate/sparse/empty routing passed. No stderr.
Ordinary full producer+MegaMoE MAX-rank CUDAgraph means8→16 senders at4/32/64:
49.089584→49.563681us(+0.9658%),52.663967→54.400713(+3.2978%),
60.351545→60.035816(−0.5232%). Every cross-repeat sign agrees.16 samples:
4 49.527840/49.599522;32 54.392513/54.408913;64 60.027138/60.044495.
8 samples:4 49.216080/48.963089;32 52.582462/52.745472;
64 60.401730/60.301361. Pull45.138649/47.303425/52.998470 means;
16 slower9.8032/15.0038/13.2784%. All150 compute-capable CTAs retained.
Small64 gain is real in these repeats, not a general small-batch win. User agrees
more senders is not useful for now; restore local helper to1–8 parent policy.
Remote16 overlay/caches/patch remain immutable. No new profiles, runs or commit.
Evidence: senders16/{audit.json,pipeline_comparison.json,result_summary.json,
candidate.patch,contract.json,compare-7225802.out}; docs synced under docs/.
An interim audit refused incomplete arm order while the final arm was running;
final audit passed without filtering, retrying GPU work or changing its contract.

Preparation/provenance follows (historical status observations):

New user authorization: test16 sender CTAs without lowering compute CTA count.
Prior stop/CTA-search restriction does not prohibit this single requested pair.
Parent producer_metadata remains frozen. Local layout helper temporarily returns
16 instead of capacity-derived1–8; total grid150, four sender warps/CTA, metadata,
payload transport, SF gather and Lamport combine unchanged. All sender strides,
payload-ready targets and DispatchDone targets derive from the same helper.
Inline sender roles precede their CTA's compute; increasing17→25 early-dispatch
participants atEP8 may delay more compute CTAs but does not reduce the grid.
At4/32 rows only1/8 sender CTAs carry payload;64 rows exercises all16. Plan:
CPU device/register checks, fresh arbitrary-routing/replay validation, then
ordinary4/32/64 full-pipeline symmetric comparison versus8 senders and retained
pull. No IKET, additional sender counts, combine changes or automatic promotion.
Remote artifact root R/artifacts/push_dispatch/senders16. CPU7225701 submitted:
M16/M32/M64 device/register probes, fresh candidate-layout native host build and
selector cache preparation. Same frozen parent main/dispatch/router; owned layout
diff saved. GPU plan prepared but unsubmitted: fresh8-sender arbitrary4/65-row
reference then16-sender comparison (normal/alternate/sparse/empty/replay), followed
by pull/8/16/16/8/pull ordinary4/32/64 at200warmups/2000calls/graphbatch100.
GPU7225802 submitted afterok7225701,25min EP8; only this GPU job. CPU at1m17s
has THREE_REGISTER_GATES_PASS (128registers,24-byte stack,zero spills; dynamic
role budgets pass); candidate host/selector build continues before GPU release.
At17:31UTC CPU7225701 COMPLETED0:0/3m10s: all three device/register gates,
candidate native host and selector PASS. GPU7225802 pending at this observation.
At17:34UTC GPU7225802 RUNNING2m35s on nvl72016-T[09-10]; candidate4-row
arbitrary/padding comparisons and all8 replay checks pass so far,65-row gate
still in progress. stderr empty. No ordinary timing result yet.
Initial local plan-command nested-quote SyntaxError occurred before remote writes
or submissions; corrected without spending cluster resources. No profiling.

## Read-only startup decomposition follow-up — September 17

User asks whether exposed pre-compute dispatch was decomposed after16-sender test.
Re-read existing producer_metadata/result_summary.json (GPU7216016,rank0,EP8,
32rows); no new profiling. Two short instrumented samples have local InputsReady
1.472–1.824us, all metadata acquired3.872–4.384, SizesReady4.320/4.928,
IndexScatter starts4.544–5.216, IndexReady6.208–7.584, local PayloadPushed
6.976–7.392, firstMMA8.864/9.696, relative to MegaMoE entry. These overlap;
source publication is not all-incoming payload acquisition, cross-CTA envelope
is not the first tile's causal dependency, and first-stage SF/activation/weight
readiness is not separately attributed. Exposed dispatch7.552us both samples;
another0.864/1.696us precedes firstMMA after measured dispatch envelope.
Retain third sample: metadata102.752–102.784, firstMMA107.648 despite local
inputs1.472–1.632 and outgoing payload6.912–9.056. Late incoming metadata
explains that observed gate; untraced peer/producer/network cause unresolved.
Earlier narrowing payload dependencies and moving SF ownership regressed full
pipeline, so no claim that one diagnosed component universally dominates.
Same-case historical best is producer_metadata/iket/reference32 in the same
GPU7216016 comparison: EP8,Q3,32rows/rank,150CTAs,FP8/FP4,balanced rank-interleaved
routing. Three retained rank0 samples; pull medians firstMMA6.336us, dispatch
end7.520, exposed dispatch5.888, compute25.024, exposed combine4.256 versus
push9.696/8.000/7.552/27.904/4.448. Every pull sample starts compute before
dispatch ends; all three push samples start afterward. These are kernel-relative
phase markers, not additive causal costs; PDL launch offsets differ. Pull capture
has dispatch_detail=false/metadata_detail=false: no equally fine decomposition
of metadata/index/payload waits, so the first-tile causal split remains open.

## Working standard — expert kernel optimization (user direction, September 17 UTC)

This is a hard, expert-level kernel optimization problem. Be patient and tackle it
step by step. Do not cheat, lower the bar, redefine success around an easier
comparison, or promote an intermediate cleanup because it yields a small win.

- **Acceptance baseline:** our historical best **metadata push + payload pull +
  push combine**. Combine is fixed at **Lamport push + polling** for both sides;
  do not tune/change combine to manufacture a dispatch improvement.
- **Target:** adopt CuTeDSL `next` Gen-specialized **all-push dispatch** efficiently
  on GB200. The expected mechanism is less exposed dispatch by eliminating the
  payload-pull network leg. Final success is lower end-to-end megakernel latency,
  with metadata transfer, payload transfer, MoE compute and combine accounted for.
- **Routing:** arbitrary selected expert IDs are required. Run normal random,
  sparse and zero-valid correctness/replay before performance. Balanced routing
  is only the historical benchmark input; never use it in kernel assumptions.
- **Combine check:** approximately4–6us exposed combine is an expectation to test,
  not a number to force. Measure the post-compute exposed tail, not the entire
  combine interval that overlaps compute. Report contrary evidence explicitly.
- **Baseline fidelity:** retain exact historical code/configuration and identify
  timing boundaries. Metadata placement varies by implementation: the retained pull path and
  current producer variants publish it in the router; older all-push variants
  publish it inside MegaMoE. Never silently omit metadata cost,
  charge unrelated gate/quantization work to only one side, or relabel a comparison
  between two all-push versions as a win over the required historical baseline.
- **Method:** use ordinary timing for performance decisions and lightweight IKET
  to inspect metadata/payload readiness, exposed dispatch, compute and combine.
  Establish a specific bottleneck, make a concise structural change, validate
  correctness/replay, then compare against the fixed baseline under matched work.
  Preserve all samples, regressions, drift and uncertainty; do not select favorable
  cases or use instrumented spans as ordinary kernel latency.
- **Async operations (user direction):** use asynchronous operations wherever
  dependencies permit: TMA transfers and async release readiness publication;
  **The purpose is overlap**, not merely selecting async instructions. Identify
  the independent work/transfer that progresses in flight, wait at consumption
  or buffer reuse, and verify reduced exposed waits with traces/ordinary timing.
  Async publication alone is not evidence of improvement.
  For each async change, state what overlaps what, where the first true
  dependency forces a wait, and which exposed interval should shrink.
  Preserve visibility/lifetime ordering. Index allocation still requires an
  atomic return value; do not replace necessary dependencies with unproven flags.
- Intermediate all-push A/B runs are diagnostic progress only. Keep the goal open
  until the requested baseline comparison and the intended latency mechanism are
  supported by evidence. No subagents, CTA-count search or combine redesign.
- Keep artifacts/history remote; copy back only necessary code. Local
  `scratch/megamoe` retains only the eight requested decomposition HTML reports.

## Read-only comparison with Rubin Genphase — September 17

Compared restored producer_metadata source against sibling CuTeDSL `next` Rubin
GenphaseTokenComm and its specialized mainloop. Same deduplicated source-indexed
payload push, metadata broadcast, local expert gather-index construction, four-warp
gather4 and all-rank/all-pusher payload readiness; no receiver activation pull.
Rubin: standalone PDL communication kernel; peer exchange/counting-sort workers,
one count/prefix/load-order helper; pushers/helper refine SF into a canonical GMEM
pool. Gather waits for all local metadata; SF consumer waits per expert. Payload
SF/data passes share completion publication, so SF is not released early.
Current: producer broadcasts IDs/scores/source counts and releases local inputs
before network tail; inline 128-thread roles repeat local prefixes, publish sizes
from CTA0, scatter indices/scores/source metadata and publish tile intersections.
FC1 waits only its tile's indices plus global payload readiness, then gathers SF
directly into stage SMEM; no active global SF refinement. Natural expert order;
Rubin helper uses descending load. Rubin has27 256-thread pushers sized around its
184-SM compute footprint; ours has1–8 pushers from capacity, inside existing CTAs.
Both fully drain outgoing copies before peer readiness; all-push does not imply
one-way completion latency. No new tests/jobs/kernel edits; experiments stay stopped.

Payload-ready follow-up: current code already publishes a separate remote counter
after TMA drain, CTA join, SYS fence and join; every pusher signals every peer,
including peers receiving no bytes. EP8 with8pushers waits64 arrivals. Rubin
`_publish_payload_flags` (token_comm_gen_specialized.py:1451) uses the same sequence,
but increments token_data_ready and token_sf_plain_ready together after both
passes; both targets are world_size*pusher_cta_count. Activation consumption and
SF refinement acquire their respective counters; no early SF publication.
PDL launch overlap does not replace these data-visibility/readiness signals.

## Limits of historical evidence

| Historical observation | Supported conclusion; remaining question |
| --- | --- |
| SF-owner + fewer active gather warps regressed | Reject that composition. Later isolated active-SF-gather also lost every repeat; neither establishes a universal warp-count conclusion. |
| SF-owner alternate32 mean improved after active-warp guard | Actual32-row kernel uses BLOCK_M32, and both variants still have4 gather warps. The gain cannot be attributed to reduced active warps; code-generation/resource effects and variability remain unisolated. |
| Prefix change on async-publication parent regressed | No general prefix verdict. One-warp direct-expert prefix later removed M192 spills. |
| Destination-owned generic payload was slow | That ownership/transport combination lost; it does not reject all generic-store or source-reuse designs. |
| Producer payload shortened sender CTA start delays | Timing supports that local effect; later metadata arrival still outweighed it at32 rows. |
| Direct-expert metadata improved over producer-payload mean | Repeat ranges overlap and it loses every repeat to producer_metadata. No accepted improvement. |
| Source-rank descriptor array appeared unbanked | Existing rank-ready path already uses two source slices as generation banks; the earlier layout objection was incorrect. |
| A shorter dispatch/combine trace interval | Work may have moved to setup/compute or marker semantics may differ. Require matched full-pipeline improvement. |
| Earlier/later first MMA relative to MegaMoE entry | PDL launch offsets differ across protocols. Compare producer-inclusive timing; do not treat a kernel-relative gap as the full-chain startup difference or add medians from separate captures. |

## Stopped by user — September 17

**User requested return to the best all-push version and stop at16:36UTC.**
Do not resume experiments/jobs without a new explicit request. Local main,
dispatch and router restored from frozen producer_metadata; host/layout already
compatible. Recent sender-specific analyzer additions removed locally; frozen
experiment analyzer/evidence retained. No commit, new test or new job on restore.
Unsubmitted local source-partitioned follow-up (local expert slot allocation,
per-source list sections, online prefix lookup) discarded; design remains an
unvalidated idea, not a result.

GPU7222752 and dependent analysis7222753 CANCELLED at9m09s/0s; CPU7222427
COMPLETE0:0/4m56s. All7 ordinary arms did finish:144 exact+104 replay+16 recordings;
normal/sparse/zero-valid validation passed. Profiling stopped by user direction.
Remote stop_by_user.json preserves all ordinary records and stop reason.
Means4/32 us: sender candidate52.4374161/61.6165447; all-push parent48.8053608/
51.6173992; retained pull45.2933922/47.2824631. Candidate loses every cross-repeat.
Samples4: candidate52.4285774/52.4462547; control48.7867851/48.8239365;
pull45.3172150/45.2695694. Samples32: candidate61.6133614/61.6197281;
control51.6124954/51.6223030; pull47.2892151/47.2757111.
No trace-based attribution of regression, no promotion, objective unmet.

**Tracing constraint:** ordinary uninstrumented timing drives decisions. Avoid
routine broad IKET capture; use sparse targeted markers only when a specific
uncertainty or promising result justifies them. No tracing/jobs while stopped.

The following entries record the sequence before the explicit stop and are not
instructions to restart those experiments.


**User direction16:14–16:20UTC: overlap sender metadata with payload ACKs;
calculate gather coordinates online; arbitrary routing required.** Implemented
parent producer_metadata with exactly3 owned kernel/router files under
A/sender_descriptors. Local main/dispatch/router now this candidate. No new
allocation, native host ABI, packet layout, SF synchronization or Lamport combine
protocol. Separate SF/packed directions frozen and parked.

Each sender warp issues/commits its deduplicated payload stores first. Every
valid selected route then obtains a unique destination expert slot using SYS
atomic countsum and stores source descriptor plus its score. Descriptor encodes
rank*bucket*topk+token*topk+slot. Gather decodes generation*EP*bucket+descriptor/topk
at consumption, retains row registers across K tiles, and does not read a separate
gather-index array. FC1 score loads and FC2 return addresses decode the same
expert-local list; math and Lamport transport unchanged. Receiver count/prefix/
index/source-metadata/score scatter and selector metadata broadcast are removed.
Raw expert-ID routing remains arbitrary; no balanced/all-peer assumption. Invalid
routes are skipped, duplicate peer payloads deduplicated, empty senders still
publish completion. Existing count/descriptor/score generation banks reused.

Sender generic writes → CTA join → SYS fence → CTA join → per-peer metadata
completion. One helper acquires all source flags==kPushers then releases SizesReady;
existing scheduler startup acquires it. Payload full drain and original
proxy/SYS/CTA publication follow descriptor work; gather retains its payload
acquire. Original all-incoming helper/DispatchDone/final cleanup protect even
empty destinations. The removed pooled L1 arrival wait is unused in compact
TMA-A; actual gather transactions still gate MMA. Full proof/protocol.diff remote.

Unexplored if slot allocation is exposed: sparse routing bitmaps permit
non-returning remote OR and online rank/select at gather, replacing returned
remote slot atomics. Need arbitrary-row/count/generation/slot-score mapping proof
and estimate repeated scan cost before implementation; do not assume rows<=32 or
balanced routes. Keep current direct-descriptor run isolated first.

**Readiness tradeoff clarified to user:** the previous implementation waits for
global counts, then only the indices for each assigned FC1 tile (not the entire
index array). It also waits for all payload senders. The current direct-list
candidate gates SizesReady on all sender descriptors, so it may lose tile-level
index/compute overlap. Test whether descriptor work fits beneath the already
global payload wait; do not infer a win from removing index materialization.

This differs from direct_expert_metadata7218435: that variant pushed payload in
the selector and still performed receiver index/scatter. Historical rejection
therefore does not test this implementation. Remote slot round trips, later count
availability and descriptor-dependent epilogue loads remain possible costs.

**CPU7222427 submitted16:21:31UTC**, M16/M32/M192 ordinary + actual M32 critical
register gates and fresh selector build PASS128regs/48stack/no spills. Dispatch
39/48 small+critical,45/48 M192; compute18/40 small,17/40 M192; no local accesses/
unresolved transfers. GPU7222752/analysis7222753 submitted16:27:10UTC with CPU
afterok gate. Only GPU allocation;25min limit. CPU COMPLETE0:0/4m56s.
Poll16:28:38UTC GPU RUNNING1m22s on nvl72158-T[01,16], initial arbitrary normal/
sparse/zero-valid64 exact+8 replay PASS,stderr0. The6-iteration validation timing
79.392us is diagnostic, not an acceptance comparison. Poll16:30:23UTC RUNNING
3m07s, first retained pull4/32=45.31721/47.28922us. Candidate timing pending;
next routine poll~16:35UTC.
Seven-arm plan requires
normal random/sparse/zero-valid output and graph replay before full matched EP8
rows4/32 ordinary timings,3 critical captures+30 timelines. Analyzer frozen with
explicit SenderDescriptors milestone; it does not invent IndexReady/scatter
markers. Perfect route inputs serve historical performance comparison only.

SF repair CPU7221865 COMPLETE0:0/4m05s, all4 register gates PASS; no GPU repair
submitted because this user direction has priority. Packed failure remains
preserved, no unchanged retry. No target win/default promotion.

**Packed_payload7221564 FAILED1:0/2m09s; analysis7221565 CANCELLED.** Submitted
16:04:43UTC, nvl72102-T[13-14]. CPU7220822 COMPLETE0:0/3m17s: all4 device/register
gates PASS128regs/24stack/no spills; dispatch39/48 small/critical,41/48 M192,
compute18/40; no local accesses/unresolved transfers. Dedicated native build PASS,
ninja dependency confirms owned descriptor header. GPU passed8 exact-input
fixture comparisons, then all8 ranks failed first rows4 output-byte comparison
at result.2. ZERO output/replay checks, timing records or captures. Full failure
saved in failure-7221564.json; preserve sources and investigate before any retry.

Root R/artifacts/push_dispatch/packed_payload, parent producer_metadata. Exactly3
owned headers: device main, dispatch, native descriptor host header. Existing
shared slot holds X[hidden]+SF[hidden/32] after2 inbound loads; one outbound TMA
store sends the complete slot to each selected peer. Destination row pitch and
SF word indexing use hidden+hidden/32; hidden4096/EP8 full route stores16→8/token,
bytes unchanged4224/peer. Existing compact allocation covers generation banks;
canonical SF/score offsets and ABI arguments unchanged. Original4 SF/gather warps,
blocking join, drains/proxy fences/SYS publication, cleanup, math and Lamport
combine retained. Fresh native module compiled from source_counts donor plus
owned descriptor header. No performance evidence; correctness failure is not
evidence against the transport-command-count hypothesis.

**SF_arrive_pipeline7220814 FAILED134:0/2m09s; analysis7220815 CANCELLED.** All8
ranks illegal instruction at first torch.cuda.synchronize after run_stage. Only8
input-fixture comparisons, ZERO output/replay/timing checks. Verified source bug:
barrier IDs3+stage alias epilogue warpgroup barriers3/4, which stay concurrently
live during FC1. The original audit missed dynamically indexed BarrierStartIdx
uses. Earlier assertion that IDs3..15 were free was WRONG. Exact trapping PC
unobserved; alias is a source correctness defect, not proven sole hardware cause.
Failed source, protocol.diff/review, CPU records, cubins/SASS preserved in
failed_barrier_ids; full failure-7220814.json retained. Original CPU7220591 had
passed all4 register gates in4m05s; compile success did not establish barrier safety.

**CPU repair7221865 COMPLETE0:0/4m05s**, submitted16:09:52UTC, fresh cpu_repair_job.json record. All4 register gates PASS; GPU parked for sender-descriptor direction.
Repaired main-only overlay uses stage0→dispatch ID0, later stages→
kEpilogueWGBarrierStartIdx+kNumEpilogueWarpgroups-1+stage. With2 epilogue groups,
SF IDs0,5..15 preserve live epilogue IDs1..4. Initial CTA barrier0 completes before
compute; final cleanup0 follows dispatch/epilogue join1 after all MMA/SF uses.
Each same-stage empty acquisition protects named-phase reuse. SF warps acquire
empty, write/fence SF, barrier.arrive then issue gathers without blocking; TMA-A
syncs that stage ID before its original full-barrier arrival. Expected bytes,
full arrivals and FC2 unchanged. Leader A pending arrival prevents premature
completion even if gather transactions precede expect_tx. Bound ensures IDs<16;
no stage count tuning. Current M16=11/M32=10 stages. Corrected protocol diff/proof
saved. CPU gates M16/M32/M192 ordinary + actual M32 critical; GPU rerun gated on
these checks. Unchanged router cache reused, failed GPU fully audited.

**Active-SF-gather7220325/7220326 COMPLETE0:0 in15m53s/3m49s.**144 exact+104
replay+16 recordings,3 valid captures+30 timelines,stderr0; full
result_summary.json saved. Mean4/32 candidate49.52301/52.09432us versus
parent49.35015/52.01970 and pull45.47058/47.34344. Candidate loses every cross-
repeat:4 delta+0.046703..+0.299007us;32+0.008465..+0.140781us. Reject replacement.
Critical32 control/candidate/pull firstMMA10.464/10.560/6.464us;
compute28.384/29.184/25.088us; dispatch9.312/9.248/6.016us;
postcompute combine4.704/6.656/4.448us. Nsight kernel means44.2784/44.4384/
37.7024us; producer-to-end51.3984/51.6512/47.5648us. Retain firstMMA outliers
128.480control/136.384candidate, peer cause unisolated. All actual32 captures use
BLOCK_M32/stages10, so neither version reduces active warps there. Only4-row
ordinary timing tests the intended reduction. No promotion.

Further ordering review: PTX9.0 async-proxy section says completion supplies an
implicit generic/async proxy fence; full bulk wait makes writes visible to its
issuer. This supports questioning the explicit post-drain proxy fence but does
not by itself prove a relaxed SYS flag is sufficient for other ranks. Keep SYS
publication intact. Control InputsReady1.344–2.016us versus PayloadPushed6.784–
9.728us includes staging/issue/drain/fence, not pure NVLink flight time. The packed
candidate isolates outgoing command count without weakening that ordering.

**Active-SF-gather GPU7220325 / analysis7220326 submitted15:33:39UTC.** Seven arms,
144 exact+104 replay+16 recordings expected,3 critical32 captures+30 ordinary
timelines. EP8 rows4/32,150CTAs,full producer+MegaMoE; parent and fresh pull.
Immediate GPU PENDING/analysis dependency; only GPU allocation,25min limit.
Next routine poll~15:38UTC. Sources/analyzer frozen.
Poll15:36:44UTC GPU RUNNING3m00s on nvl72172-T[09-10],stderr0; candidate
normal/sparse/empty validation64 exact+8 replay PASS, first pull4/32
45.47386/47.33837us. Next routine poll~15:41UTC.
Verified runtime kernel name:32 rows uses BLOCK_M32/stages10, so this candidate
directly changes only rows4/BLOCK_M16. Rows32 is a collateral-code-generation
comparison; its captures do not measure the intended inactive-warp reduction.
Poll15:41UTC RUNNING7m49s,stderr0; candidate/control ordinary repeats complete,
final pull starting. 4-row candidate49.63298/49.41304 versus parent49.36634/
49.33397us: candidate loses all cross-repeats, no isolated active-warp win.
32-row candidate52.04221/52.14643 versus parent52.03374/52.00565us, no gain.
Full audit and captures pending; next routine poll~15:46UTC.
CPU7220146 COMPLETE0:0/4m07s: all4 device/register gates PASS128regs/24stack,
no spills/local accesses/unresolved transfers; dispatch39/48 small/critical,
41/48 M192, compute18/40. Exact unchanged selector/layout reused. Root
`R/artifacts/push_dispatch/active_sf_gather`, isolated parent producer_metadata.
Frozen main restored from that parent, then only row-owning dispatch warps
enter FC1 scheduling: M16 uses2 warps, M32+ retains4. Both sides of the original
SF/TMA-A named join use96 threads at M16 instead of160. Original SF ownership,
stage-empty acquire, proxy fence, full barrier counts/arrival ordering and all
transport remain. Inactive warps explicitly fetch scheduler counts needed for
cleanup and join the existing epilogue/dispatch retirement barrier. They cannot
reuse its named cleanup barrier until local compute has finished. Four device
build/register gates on CPU; exact router/layout unchanged, existing compiled
selector reused. Bounded main-only overlay and protocol.diff frozen. Seven-arm
comparison submitted after CPU checks and prior full audit. Tests the
unisolated effect of inactive warps, not the historical SF-owner composition.

**Independent-SF ordinary timing rejects candidate.** Poll15:21UTC found all
seven arms complete; 4/32 repeat means candidate140.94473/149.31729us versus
parent48.31838/52.15285 and pull45.27630/47.29112. Both repeats lose substantially.
GPU7219911/analysis7219912 COMPLETE0:0 in15m38s/3m49s;144 exact+104 replay+
16 recordings,3 valid critical captures+30 ordinary timelines,stderr0. Full
result_summary.json remote, including every sample/marker and cross-repeat delta.
Critical32 control/candidate/pull firstMMA10.976/11.200/6.784us;
compute27.936/123.872/24.800us; dispatch9.728/9.792/6.336us;
postcompute combine6.624/9.056/3.936us. Ordinary Nsight kernel means
44.1632/142.5120/37.7888us, producer-to-end51.3152/149.7024/47.6320us.
Major added cost is stage progress. Explicit release.cluster emits ERRBAR+
CGAERRBAR and is a plausible cause, but not isolated attribution. Do not conclude
that every independent-SF protocol is slow. Candidate firstMMA40.032us outlier
retained: incoming topk33.728–33.792 despite local payload6.848–9.728us.
Untraced peer cause unknown. Sources remain frozen; local candidate now active-SF.

Refinement now implemented in SF-arrive-pipeline above. Simply replacing the
single reused named barrier with barrier.arrive is unsafe: a fast warp could
arrive twice before phase completion. Separate ring-stage IDs preserve lifetime.

**Independent-SF GPU7219911 / analysis7219912 submitted15:11:57UTC.** Seven arms,
144 exact+104 replay+16 recordings expected;3 critical32 captures+30 ordinary
timelines. EP8 rows4/32,150CTAs,full producer+MegaMoE max-rank graph timing;
original producer_metadata control and fresh pull. Only GPU allocation,25min
limit. Sources/analyzer frozen. CPU7219592 PASS4m56s, submitted14:54:18UTC; root
`R/artifacts/push_dispatch/independent_sf_arrivals`; isolated parent
producer_metadata, whose router/dispatch were recovered locally. Frozen main
contains only this additional change; retirement overlays remain frozen. Remove
the per-K-stage160-thread SF/TMA-A rendezvous while retaining each SF writer's
stage-empty acquire, generic/async shared proxy fence and warp join. Each SF warp
then arrives with release.cluster at the leader CTA's full barrier. Four SF warps
+TMA-A+TMA-B per CTA make12 arrivals per phase. FC2 has no SF gather warps, so its
TMA-A contributes a counted5-arrival release after declaring unchanged TMA bytes;
TMA-B contributes1. Non-compact path keeps4 arrivals. No new scratch, data/layout,
CTA count, SF ownership, transport, arithmetic or combine change. PTX count and
remote release semantics checked against CUDA13.0/PTX9.0 primary documentation.
Main-only overlay owns header, other headers/router reference frozen parent;
protocol.diff preserves exact87-line patch. Four device/register gates+selector
build PASS128regs/24stack/no spills; dispatch39/48 small/critical,41/48 M192,
compute18/40, no local accesses/unresolved transfers. Seven-arm GPU plan prepared
(144 exact+104 replay+16 recordings,3 captures+30 timelines), submitted only after
retirement full audit. SASS shows explicit release.cluster emits ERRBAR+CGAERRBAR
before new arrivals; ordering cost may offset rendezvous removal. No performance
gain inferred from source-level synchronization count.
Poll15:14:05UTC RUNNING1m53s on nvl72173-T[07-08],stderr0. Initial normal/sparse/
empty validation64 exact+8 full-pipeline graph replay PASS. First reference arm
starting; ordinary timing and captures pending. Next routine filtered poll~15:19UTC.
Validation's6-iteration timing is diagnostic only, not the ordinary acceptance run.
Hypothesis: the MMA barrier is the true join, making the earlier producer join
redundant. Additional barrier arrivals may still cost more than they save.

**Retirement_overlap7219376/7219377 COMPLETED0:0 in24m57s/57s.** Audit240 exact+
144 replay+16 recordings,stderr0;4 valid critical captures+40 ordinary timelines.
Full result_summary.json remote. Mean4/32 us: control48.77971/52.14713;
cleanup48.83471/51.73474; warp-split50.75810/54.04410; pull45.15226/47.25804.
Cleanup32 saves0.412386us, every cross-repeat (range0.081795–0.742977us);4 mean
regresses0.055002us with overlapping ranges. Preserve cleanup as a structural
option; no target win/default promotion. Warp-split loses every repeat to control.
Cleanup also shifts sender CTA IDs9..16→8..15 and CTA-pair membership; do not
attribute its complete gain solely to DispatchDone polling removal.
Critical32 control/cleanup/warp-split/pull firstMMA10.496/9.760/12.160/6.720;
compute29.344/28.320/27.360/25.024; dispatch9.280/8.096/10.496/6.272;
combine7.712/3.936/4.384/4.128. Cleanup raw role-end7.648, first payload acquire
8.544,last9.664us; alternate9.568/10.944/11.968. All150 acquisition locations
validated each sample. Ordinary Nsight kernel means47.9968/44.4320/46.7392/37.6288;
control capture slowdown exaggerates gain versus full max-rank graph timing.
Selector durations7.2032/7.1904/9.9872us; alternate metadata arrives6.528–6.656us
after main start vscleanup3.968–4.064 in nonoutlier samples. Separate warp groups
did not establish earlier metadata completion; resource/peer cause unisolated.
Keep firstMMA136.064cleanup/125.408alternate/34.272pull outliers; corresponding
late incomingmeta130.880–130.912/119.616–119.680us despitecleanup local pushes
7.296–7.520us. No untraced-peer attribution. All samples saved.

Retirement plan, submitted14:42:38UTC: ten arms, candidate/alternate edge
validation, then pull/control/candidate/alternate/alternate/candidate/control/pull.
Expected240 exact+144 graph replay+16 recordings,4 critical32 captures+40 ordinary
timelines. EP8 rows4/32,150CTAs,full producer+MegaMoE max-rank graph timing.
Candidate CPU7218955 PASS1m48s; alternate7219203 PASS2m27s. Both all4 device
builds128regs/24stack/no spills; dispatch39/48 small/critical,44/48 M192,
compute18/40, no local accesses/unresolved transfers; exact selector builds PASS.
Candidate isolates cleanup reuse on producer_metadata. Alternate adds cleanup
reuse and independent selector metadata/payload warps on producer_payload; it
does not independently isolate those two changes. Compare full-chain gain first;
no target win or promotion presumed. Local sources since restored to original
producer parent plus independent-SF candidate above. Poll14:53:49UTC RUNNING11m09s
on nvl72096-T[01,18],stderr0. Both edge validation arms64 exact+8 replay PASS.
First/repeat4/32 us: control_a48.60229/52.00384; cleanup_a48.89029/51.54744,
cleanup_b48.77914/51.92205; warp-split_a50.75466/54.06723,
warp-split_b50.76154/54.02096; pull_a45.17022/47.27138. Final control/pull repeats
and captures were pending at that poll; terminal result above is authoritative.

First-principles follow-up, unimplemented: source sender CountsReady covers both
quantized X/SF and selected route IDs. Hoisting it ahead of selection is unsafe;
the sender's deduplication reads topk before payload prefetch. Earlier PDL alone
can overlap setup but cannot remove this data edge. CUDA allows early dependent
launch after all producer blocks trigger; data still needs an explicit acquire
(CUDA13.0 Programming Guide6.2.8.6.2). Current producers never wait on consumers.
Route-independent all-peer payload push could overlap top-k selection using the
seven idle warps in the supported one-token CTA selector, while warp0 selects
and publishes metadata. Target balanced EP8/topk8 already visits every peer, so
target payload byte count would be unchanged; arbitrary sparse routing would
send extra bytes. This is an untested tradeoff, not a universal improvement.
Preserve per-issuer full drain/proxy ordering, independent metadata/payload
tickets, and generation-bank retirement if investigating it after this run.

Direct-expert metadata7218435/7218436 COMPLETED0:0 in20m38s/44s;176 exact+136
graph replay+16 recordings,stderr0;4 valid critical captures+40 ordinary timelines.
Full result_summary.json remote. Mean4/32 us: control48.54235/52.70826;
direct51.95345/53.86092; producer_payload49.31403/54.80822; pull45.30674/47.30499.
Reject direct as replacement: every cross-repeat loses to control. Retain first4
sample54.04571us. At32 direct improves over producer_payload mean but their repeat
ranges overlap (alternate56.24950/53.36693), so no repeat-consistent win there.
Critical32 firstMMA control/direct/alternate/pull12.512/11.872/11.648/6.528us;
compute30.112/28.064/28.064/25.664; dispatch10.624/10.624/10.176/6.080;
ordinary Nsight kernel medians44.400/46.048/45.376/37.952. Trace envelopes and
max-rank long graph scope disagree; neither substitutes for acceptance timing.
Keep control/alternate/pull firstMMA maxima32.384/39.840/38.880us; untraced peer
causes unknown. Less metadata work did not remove slot-return/resource costs.

Retirement implementation rationale (candidate CPU7218955 submitted14:28:45UTC): root
`R/artifacts/push_dispatch/retirement_overlap`; parent producer_metadata. Direct
expert-list ordinary repeats still trail that control, so do not build acceptance
on that unproven composition. Candidate removes the early payload helper and
DispatchDone increments/polls only when existing kLastCTACleanup collects every
CTA. Source metadata/payload CTAs finish before each CTA's initial GEMM join;
last cleanup cannot run until all those CTAs retire. Active FC1 acquires payload
as before; CTAs with no FC1 acquire all incoming completion before their cleanup
ticket. Other cleanup modes retain their existing protocol. No new fallback,
CTA count, payload transport, SF or Lamport-combine change. Main early roles17→16.
New CleanupOwnsRetirement ownership marker and150 per-CTA MegaPayloadAcquired
locations. Analyzer retains raw dispatch-role end and all acquisition timestamps;
dispatch end=max(raw role end,first all-source payload acquisition). Removing a
helper marker must not falsely remove exposed payload latency. Candidate sources
frozen. Both CPU gates and direct-expert full audit completed before7219376.

Alternate implemented: parent producer_payload plus same cleanup reuse; one
selector warp owns metadata broadcast/count forwarding and other warps own TMA
payload, with separate subgroup retirement/tickets. This removes metadata's wait
for payload staging and payload's wait for metadata finalization. Generic compact
selector requires at least2 warps; token-selection work remains guarded by actual
physical rows. Quantizer acquire/proxy fence, TMA completion/proxy ordering and
reset-before-SYS-publication remain required. No performance gain presumed.
Local headers/router match this alternate; candidate code is preserved in its
bounded remote overlay. Alternate CPU7219203 submitted14:36:04UTC and passed.

**Direct-expert metadata GPU7218435 submitted14:12:51UTC; analysis7218436 dependent.**
Root `R/artifacts/push_dispatch/direct_expert_metadata`, experimental parent
producer_payload (not accepted). Local router/dispatch now implement this variant;
main/layout/SF/combine unchanged. Bounded overlay owns dispatch+router only, main
symlinked to frozen producer_payload. Selector issues deduplicated payload first,
then each valid route allocates a destination expert slot with the retained
SYS atomic, writes its descriptor into the existing generation slice and score
only into that destination's banked source inbox. Metadata last-CTA ticket/SYS
publication retained, source histograms/count forwarding removed. Payload full
drain/proxy/join/ticket/SYS publication unchanged. Receiver reads final expert
totals, computes padded prefixes, assigns experts across the same EP metadata
CTAs, and writes gather/score/combine metadata without local atomic row assignment.
One complete tile contribution per owning CTA replaces source intersections.
PayloadReady and DispatchDone retirement still cover empty destinations. New
ExpertLists marker identifies protocol, not a completion milestone. No new layout,
native ABI, totalCTA count, SF gather or Lamport combine changes. Remote slot return
adds a dependency; reduced metadata traffic/work may or may not compensate.
Diff/shell syntax pass; recovery CPU gates and prior GPU audit PASS below.
Sources/analyzers frozen. Only GPU allocation,30min limit;9 arms,176 exact+136 replay
checks expected plus16 recordings,4 critical captures and40 ordinary timelines.
Poll14:14:28UTC RUNNING1m29s on nvl72018-T[13,15], stderr0. Initial validation
64 exact+8 replay PASS across normal/sparse/empty inputs. Timings/captures pending.
Next routine filtered poll near14:19UTC; no target win or promotion.
CPU7218153 FAILED1:0/1m55s at M192 register review: dispatch45/48 but two
local loads from stack+0x18; target M16/M32/critical all39/48 and no local accesses.
Four-warps prefix attempt preserved with cubins/SASS/review/patch under
four_warp_prefix. Local correction uses one warp to scan final expert totals,
removing the zero-input warp sums/inter-warp joins for target16 local experts;
one CTA join publishes offsets. Generic prefix still loops over all experts.
Selector unchanged. Recovery7218284 submitted14:06:51UTC for all4 device/register
checks plus exact selector build after transfer completed; sources frozen.
Recovery7218284 COMPLETED0:0/1m50s: all4 builds128regs/24stack/no spills;
dispatch39/48 smallM/critical,43/48 M192; compute18/40. No local accesses or
unresolved transfers. Exact selector module passes. Frozen candidate ready.
Nine-arm plan prepared: direct candidate, producer_metadata control,
producer_payload alternate, fresh pull;176 exact+136 replay+16 recordings,
4 critical captures+40 ordinary timelines. GPU submitted only after all gates.

Producer-payload ordinary repeats complete at14:01UTC; captures still running.
4/32 us: control49.52486,49.48254 /51.88035,51.89597;
candidate49.35423,49.33834 /53.55384,53.52549;
pull45.23581,45.26288 /47.32232,47.30270. Candidate helps4 by~0.157us
but regresses32 by~1.652us in every cross-repeat; both remain behind fresh pull.
GPU7217857 COMPLETED0:0/17m23s; analysis7217858 PASS46s. Audit144 exact+104
replay+16 recordings,0 separate combine,stderr0;3 valid critical captures and30
ordinary timelines. Full result_summary.json remote includes all samples and CTA
first-compute roles. Do not promote producer-payload as a replacement. Means4/32
control49.503704/51.888161, candidate49.346281/53.539665, pull45.249344/47.312511;
candidate remains9.054%/13.162% behind pull. Critical32 control/candidate/pull:
firstMMA9.152/11.040/6.304; compute28.160/27.712/25.184; exposeddispatch
7.360/8.960/5.856; exposedcombine4.128/4.320/4.352; Nsightkernel median
44.688/45.904/37.696. Candidate incomingmeta5.344–5.664 vscontrol3.680–4.032us
in nonoutlier samples; candidate payload acquire7.712/7.840, lastindices8.864/9.376.
Former sender pairs now start within0.448/0.704us of firstMMA vscontrol1.088/2.176;
initialization penalty improves but later startup dominates. Selector duration
7.2288→11.3824us, overlap4.5184→8.416; mainstart7.1744→7.4336, ordinary gate-to-end
51.9008→53.4784us (pull47.6064). Retain firstMMA outliers101.728 control/58.112
candidate and late metadata flags; untraced-peer cause unknown. This run is terminal.

**Producer-payload GPU7217857 submitted13:50:36UTC; analysis7217858 dependent.** Root
`R/artifacts/push_dispatch/producer_payload`; parent producer_metadata. Restore
original gather-warp SF main, preserving the completed sf_tma_owner overlays. Selector
prefetches contiguous quantized X/SF into shared memory during selection, then
warp-per-token/peer sends deduplicated rows. Metadata finalizer stays separate;
full outgoing TMA drains + issuer proxy fences + CTA joins precede acqrel GPU
payload tickets. Last CTA resets the banked counter before SYS fence and one
PayloadReady contribution per destination. No producer waits on another CTA or
consumer. Main keeps EP metadata CTAs plus one incoming-payload retirement helper;
FC1 waits EP payload contributions, cleanup waits EP+1 retirements. Same150CTAs,
math, layout, gather SF and Lamport combine. Generic16-token selector needs
128+tokens*(H+H/32) dynamic shared bytes; specialized selector stages one token.
Host checks device capacity without fallback. Full wait_group0 is required before
publication; read7 only paces source reads, per primary PTX bulk-wait semantics.
New PayloadFromProducer marker follows helper destination acquisition, not send.
Hypothesis: remove selector-to-main sender handoff/duplicate staging setup. Risk:
more selector work/resources and peer fanout may outweigh savings. No gain assumed.
Local diff/shell syntax pass; bounded overlays/router/analyzer synced. CPU compiles
four ordinary/critical shapes, reviews reachable register regions and builds the
exact selector include/cache module. Candidate sources now frozen. Seven-arm
comparison submitted (144 exact+104 replay+16 recordings,3 captures+30 timeline calls).
Only GPU allocation;30min limit. Poll13:51:54UTC RUNNING1m18s on
nvl72102-T[13-14], stderr0; candidate validation64 exact+8 replay PASS for
normal/sparse/empty valid sets. Ordinary repeats/captures still pending.
Follow-up filtered poll near13:56UTC. No source/
analyzer edits while queued/running. SF comparison and dependent audit finished.
Ordering review found the selector prefetch also needs the original issuer-side
`fence.proxy.async.global` after PDL wait to order quantizer generic writes before
async reads. Added locally while CPU7217633 was frozen; synced/rebuilt only after
that job ended. GPU uses the corrected selector. PTX bulk-copy
size requires16-byte alignment/multiples and valid source/destination ranges;
H%512 and bounded dynamic shared enforce these for both selector paths.
Primary semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

SF GPU7217083/analysis7217084 COMPLETE0:0 in24m14s/43s. Audit240 exact+144
graph replay+16 recordings PASS,0 separate conventional combine,stderr0; all4
critical32 captures valid and40 ordinary timeline calls. Full result_summary.json
saved remotely with all samples, markers, timelines and cross-repeat deltas.
Pipeline4/32 us: control48.41230,48.39680 /51.86330,51.85776;
SF-owner51.43363,51.37523 /54.84904,59.08757;
SF-owner+active-warps51.50491,51.47656 /54.80990,54.74205;
pull45.26475,45.29506 /47.28082,47.31629. Both variants regress every
cross-repeat; preserve59.08757us sample. Reject both compositions. Means4/32:
control48.40455/51.86053; SF-owner51.40443/56.96830; alternate51.49074/54.77597;
pull45.27990/47.29855. FirstMMA medians control/candidate/alternate/pull
10.624/10.688/10.656/6.688us; compute29.312/32.512/32.832/24.480;
exposeddispatch9.536/9.280/9.280/6.240; exposedcombine7.072/3.584/3.616/4.672;
Nsight ordinary kernel44.256/47.840/47.568/37.760. Less exposed combine does not
repair full-chain regression. Selector/main overlap stays~4.4us and Nsight main
starts7.1393/7.1936/7.1104us; execution growth dominates launch shift.
Candidate firstMMA45.568: TopkReceived39.936–39.968 vsInputs1.920–1.952 and
PayloadPushed7.296–7.616. Alternate firstMMA20.800 also retained. No untraced-peer
cause assumed. SF-owner mainloop serialization plausible, exclusive cause unproven.
Participant reduction was combined with SF-owner; no isolated original-SF verdict.
No promotion.

CPU7217633 FAILED1:0/4m06s after four successful device compiles: review command
still expected candidate_ prefixed filenames from its two-variant parent. Fixed
explicit four-name loop; recovery7217825 submitted13:46:37UTC reuses cubins and
runs register review plus exact selector build. Proxy fence now synced only after
initial CPU terminal; protocol_initial.diff/build_commands_initial.json retained.
Recovery7217825 COMPLETED0:0/2m10s. All4 register gates pass: dispatch39/48
smallM/critical,41/48 M192; compute18/40, no local accesses/unresolved transfers.
Exact selector JIT passes, including prefetch proxy fence and dynamic shared host
setup. Sources frozen; GPU/analysis submitted after this gate and SF audit.
Known dependency cost to inspect: payload stage wait/issue currently precedes the
metadata last-CTA ticket; signals are independent but metadata may be delayed by
stage readiness. Do not assume the new ordering is optimal before measurement.
Additional hypothesis: removing payload roles lets8 former sender CTAs initialize
GEMM earlier. Compare late first-MMA locations and compute span as well as payload
readiness; any full-chain gain is not automatically pure transport latency.
Saved SF-control first-compute markers now grouped by CTA pair in remote
sf_tma_owner/first_compute_roles.json. Across3 samples, sender pairs10/12/14/16
start median2.304–2.416us after firstMMA, helper/sender pair8 +2.368–2.496us,
other pairs>=18 +0.224us. Metadata pairs0/2/4/6 start earliest. This supports an
initialization penalty but does not attribute the last-compute tail.

Corrected historical assumption: get_src_token_topk_idx_ptr is not intrinsically
banked, but the existing rank-ready dense path already uses two source-rank slices
as generation banks (rank dimension=metadata_slot for EP>1; EP1 stream ordered).
Workspace per-expert slice capacity is EP*bucket. Thus the earlier blanket claim
that direct expert lists require a new allocation/layout was too strong. A
possible experiment could reuse retained-pull remote slot allocation plus
source-indexed payload push, store scores only to their destination's banked
source inbox, then locally build gather indices from destination expert lists.
Tradeoff: remote atomic return adds a route-allocation dependency but can remove
broadcast/recount/prefix work. Not yet implemented or scheduled; no assumed win.
Avoid direct writes into unbanked local output metadata by next-invocation senders.

**SF-owner implementation/build provenance**
Root `R/artifacts/push_dispatch/sf_tma_owner`, nvl72173-T[02,09]. Frozen main
overlays implement TMA-A SF ownership and exact gather participants; original
producer_metadata parent has been restored locally for producer-payload work.
Current payload_dependency GPU sources/analyzers remain frozen and untouched.
Candidate moves SF from four gather warps to TMA-A: one lane acquires aggregate
payload readiness, warp sync; small tiles prefetch four K words/row into uint4
before stage-empty wait; warp writes/fences SF after stage acquire, then releases
activation gathers through the existing named barrier and full-barrier arrival.
Large tiles load one SF word at its store to limit live registers. SF layout/math,
producer, source TMA transport, prefix/index arrivals, CTA count/combine unchanged.
Alternate adds exact gather participants: M16 uses two gather warps plus TMA-A
(96 threads instead of160); inactive warps still fetch/advance scheduler counts
and join dispatch retirement/cleanup. M>=32 keeps four gather warps. No fallback.
New MegaSFByTMAA marker identifies ownership, validated150 locations; not a
completion milestone. Both variants frozen in bounded main-header overlays.
CPU7216950 FAILED1:18 at strict register gate after all8 builds: M192 TMA-A37/40
but11 local accesses while holding six prefetched SF words across empty wait.
Evidence under prefetch_large_tile. Current rewrite shortens that large-tile
lifetime. Recovery7217034 COMPLETED0:0/3m22s: all8 builds128regs/24stack/no spills;
reachable dispatch39/48 smallM,45/48 M192, TMA-A34/40 smallM,37/40 M192,
other compute<=23/40; no local accesses/unresolved transfers. Both selector JIT
modules compile for exact include/cache roots. GPU outcome is audited above.
Submitted10-arm full-pipeline comparison after prior audit: two sparse/empty validation arms,
reference_a/control_a/candidate_a/alternate_a/alternate_b/candidate_b/control_b/
reference_b;240 exact+144 replay+16 recordings,4 critical32 captures expected.
Four captures include fresh pull; dependent CPU analysis also records40 ordinary
producer/selector/MegaMoE timelines. No promotion or goal completion.

**GPU7216464 COMPLETED0:0/15m33s; analysis7216465 PASS42s**, root
`R/artifacts/push_dispatch/payload_dependency`. Seven full-pipeline arms:
candidate_validation, reference_a, control_a, candidate_a, candidate_b, control_b,
reference_b;144 exact+104 replay+16 recordings PASS,0 separate conventional combine,
stderr empty. Control is frozen
producer_metadata, candidate only narrows payload dependency; fresh pull retained.
Three critical32 captures and ordinary router timelines analyzed in one dependent
CPU job. Ran on nvl72173-T[07-08]. All3 captures valid,30 ordinary timeline calls;
full result_summary.json saved remotely. Candidate
4/32 repeats54.49891,58.47480/57.18099,57.21695us versus producer control
48.18575,48.12666/52.51557,52.79962 and pull45.23249,45.24485/47.28902,47.26587.
Means4/32 candidate56.48686/57.19897us vs control48.15620/52.65759 and pull
45.23867/47.27745. Regresses8.33066/4.54138us versus control in every cross-repeat;
24.864%/20.986% behind pull. Reject this composition; no active GPU.
Critical32 control/candidate/pull medians: firstMMA11.072/10.880/6.656us;
compute27.840/34.016/24.768; dispatch exposure9.792/9.664/6.176;
combine exposure9.120/3.904/4.416; Nsightordinarykernel45.008/50.544/37.776.
Slightly earlier MMA and less exposed combine do not imply whole-chain savings.
Preserve candidate first-MMA42.432us; incoming metadata36.512–36.544 despite
local inputs1.280–1.696 and outgoing payload6.976–8.960us. No untraced-peer cause.
Nsight rank0 main-start7.2032/7.1968us and end52.1280/57.7184 control/candidate
confirm execution growth despite similar start; separate scope from max-rank graph.
Repeated source/index arithmetic and SYS acquires plausibly add mainloop cost;
exclusive cause unproven. M192 spills cannot explain spill-free target regression.

CPU7216361 COMPLETED1m08s; first prototype compiled, but M192 spilled.
Moved scalar SF-quad polling before gather-index arrays become live. Rebuild
7216434 COMPLETED1m04s; selector7216379 COMPLETED2m04s (new include path required
its own cached TVM module). Strict all-tile review7216451 FAILED on M192's47
local-memory instructions; preserve failure and initial_live_indices evidence.
Target M16/M32 ordinary and M16critical have no local accesses/unresolved transfers;
dispatch43/48,43/48,42/48; compute<=19/40. All compile128regs; do not claim M192
spill-free. GPU scope explicitly targets4/32 only: large-tile performance unresolved
and required before promotion, but not a target correctness/register hazard.
Exact target_gate.json and both strict/manual decisions retained remotely.

Frozen payload_dependency prototype narrows the payload dependency without changing transport/math.
In target32 balanced routing, expert e uses source tokens e/e+16: two of eight
payload CTAs per source. The old single counter forces every tile to await all8.
New source masks occupy existing reserved double-banked metadata words (no ABI,
allocation or offset change). Same drain/proxy fence/CTA joins/SYS fence precede
parallel relaxed SYS adds of distinct pusher bits. SF quad leaders acquire only
their row owners; warp sync plus existing SF/TMA-A join orders consumption.
Helper still acquires all complete source masks, including empty destinations,
before DispatchDone; last cleanup clears masks only after all dispatch readers.
One marker renamed PayloadPartitioned; analyzer declares its narrower dependency.
Four device builds and target register review preceded GPU submission. No gain presumed: added
row/source arithmetic and repeated per-tile polling can cost more than saved wait.
Frozen producer selector/wrapper/smoke remain reusable; new headers and analyzer
uploaded to a bounded overlay, no queued sources changed. Diff/shell syntax PASS.

First-principles alternatives, unimplemented: direct source expert lists could
remove index scatter, but old get_src_token_topk_idx_ptr storage is unbanked and
unsafe against next-invocation remote writers. A correct banked compact layout
and repeated descriptor resolution costs need analysis, not historical inference.
SF loads could move before stage-empty wait because register loads do not reuse
shared staging. Empty-pusher aggregation mainly helps4 rows; all8 own32-row work.
For BLOCK_M16, only two dispatch warps own SF rows or activation gather groups;
the other two still acquire every stage and join the160-thread SF/TMA-A barrier.
An exact active-warp rendezvous could remove that work without changing CTA count,
but scheduler-count initialization/cleanup must still cover all dispatch warps.
Unimplemented; consider only after current dependency experiment is audited.
No default promotion or goal completion.

**producer_metadata GPU7216016 COMPLETED0:0/18m15s on nvl72172-T[03-04];
analysis7216017 PASS3m50s; timeline7216087 PASS1m02s.** Audit144 exact+104
replay+16 recordings,0 separate conventional-combine checks, stderr empty.
All3 critical captures valid; full result_summary.json saved remotely with all
samples/markers/first-compute locations and30 ordinary timeline records.
Pipeline4/32 control50.34539/55.02553us; producer48.27965/51.78622;
pull45.13103/47.29934. Producer saves2.06574/3.23930us in all cross-repeats,
still6.9766%/9.4862% behind fresh pull. Retain as next experimental parent only.
Critical32 control/producer/pull medians: firstMMA14.784/9.696/6.336us;
compute27.264/27.904/25.024; dispatch exposure11.968/7.552/5.888;
combine exposure4.832/4.448/4.256; Nsightordinarykernel47.392/44.784/37.664.
Preserve producer first-MMA107.648us outlier: incoming metadata102.752–102.784,
local inputs1.472–1.632, outgoing payload6.912–9.056us; no untraced-peer cause.
End18raw/17unique retained. Do not add independent medians or subtract scopes.
Ordinary Nsight confirms mean selector/MegaMoE overlap0.2048/4.464/2.512us.
Selector grows2.624→7.2192us with network metadata; gate-start-to-MegaMoE-end
means54.288/52.0096/47.5488us. These rank0 captures are separate from max-rank
CUDAgraph acceptance. Legacy timeline genphase_dispatch=false detects a separate
dispatch kernel, not inline protocol identity; use decomposition contracts/sources.

**source_reuse_compare GPU7215591 COMPLETED0:0/25m55s on nvl72063-T[01,05];
analysis7215592 COMPLETED0:0/44s.** Audit PASS16 arms,336 exact+96 replay+16
recordings,0 separate conventional-combine checks, stderr empty. Three valid32-row
captures with3 IKET samples+10 ordinary Nsight calls each. Full result_summary.json
saved remotely with all samples, first-compute roles/locations and marker counts.
Pipeline4/32 means: control50.20573/54.80010us, vector50.07604/54.76163,
cachedSF51.03317/55.28774, pull44.92000/47.28326. Vector saves0.12969us at4 in
all cross-repeats,32 mean0.03847us gain lies inside sample spread (signs disagree).
Kernel control43.34128/48.11834us, vector43.11104/47.94381,
cachedSF43.93955/48.70135. Vector kernel saves0.23024/0.17453us consistently;
does not establish32-row pipeline gain. Vector11.478%/15.816% behind pull.
Reject cachedSF (both sizes/scopes regress); preserve frozen vector's small4-row
gain without promotion. Keep original TMA/SF parent for producer experiment.
Critical32 control/vector/cachedSF medians: firstMMA15.104/14.816/15.008us;
compute27.328/28.096/28.192; exposed dispatch12.128/11.968/12.064;
combine exposure4.640/4.128/4.928; Nsightordinary48.240/48.112/48.736.
Preserve control firstMMA38.240us with TopkReceived32.576–32.640us despite
local TopkSent2.720–2.976 and PayloadPushed7.392–9.536us. No cause assigned to
untraced peers; do not sum independent medians or subtract timing scopes.
Original cached_gather_sf GPU plan was never submitted; CPU evidence retained.

**producer_metadata CPU7215833 COMPLETED0:0/3m26s; GPU7216016 submitted.**
Root `R/artifacts/push_dispatch/producer_metadata`. Four device builds128regs,
24-byte stack/no spills; dynamic dispatch39/48 smallM,41/48 M192, compute18/40;
no local memory/unresolved transfers. Selector native TVM JIT also builds using
the exact frozen router source/runtime header include and its dedicated TVM cache.
Same compatible source_counts native host guard; no ABI or allocation change.

Local source now implements producer metadata + explicit local input publication,
with original gather-warp SF/source-TMA/prefix/publication (not cached SF/vector).
Selector CTA joins its completed topk writes, release-adds its physical row count
to unused CountsReady, triggers PDL, then broadcasts normalized topk/scores and
adds source counts. A last-CTA GPU acq_rel ticket collects those writes; finalizer
forwards counts, clears local counts/ticket, joins, tid0 SYS-fences, joins and
parallel-publishes source flags. No producer waits for the dependent kernel.
Reset happens before remote publication so consumer retirement cannot race a
delayed producer reset. Invalid/empty valid-row sets still publish physical rows.

Metadata CTAs acquire producer SYS flags and use unchanged prefix/index scatter.
Payload/helper CTAs acquire the local physical-row total, warp-join and issue
generic/async global proxy fences before TMA reads, replacing whole-selector
griddepcontrol.wait. Existing all-payload helper/DispatchDone retirement and
double-bank cleanup remain. Producer input counter must complete before payload
retirement, including empty destinations. Fixed150 CTAs/math/Lamport combine.
This explicitly exposes payload/metadata-tail overlap; no gain assumed. More
producer stores/count traffic or ticket latency may regress full-pipeline time.
Primary PTX memory/proxy and CUDA PDL contracts rechecked:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html

Reusable `codex_scripts/megamoe_header_smoke.py` injects bounded header and optional
router sources; producer variants fail fast on kernel-only timing/nonproduction
routes. New smoke source compares graph outputs to validated outputs for pipeline
and kernel-only scopes, including sparse validation. No candidate kernel-only
replay: its flags require producer replay each invocation. Seven-arm plan:
candidate_validation, reference_a, control_a, candidate_a, candidate_b, control_b,
reference_b;144 exact+104 replay+16 recordings expected. Three32-row captures are
control/candidate/fresh pull, all full-pipeline replays. Updated analyzer records
MetadataFromProducer8 participants and InputsReady9 helper/payload participants;
marks producer metadata placement explicitly, preserving earlier marker contracts.
All prototype sources/scripts frozen remotely; GPU7216016 submitted after completed
7215591/7215592 audit. Local diff/shell syntax pass. No producer GPU result yet.

**Recovery GPU7215143 COMPLETED0:0/29m26s on nvl72063-T[17-18]; analysis7215144
COMPLETED0:0/47s**, root `R/artifacts/push_dispatch/warp_count_prefix/runtime_recovery`.
Audit PASS16 arms,336 exact+96 replay+16 recordings,0 separate conventional-combine
checks, empty stderr. Three valid critical32 captures,3 IKET samples+10 ordinary
Nsight calls each. Full samples/first-compute locations/counts/result summary saved.

Pipeline means4/32us: original50.84268/54.93351; async release53.30288/56.20012;
added prefix52.62896/56.41802; pull45.24739/47.27361. Every cross-repeat comparison
of either candidate against original regresses. Release17.803%/18.883% behind pull;
prefix16.314%/19.344%. Kernel original44.02010/48.54326; release46.56285/49.00681;
prefix46.03312/49.92208. Reject async release composition; prefix partly recovers4
on that parent but worsens32. Its isolated effect on original release is untested.
Retain original prefix/publication for independent vector/SF tests.

Critical32 medians original/release/prefix: first MMA13.760/13.792/15.040us;
compute27.840/28.000/27.456; dispatch exposure10.880/12.352/12.192; combine
exposure4.064/8.192/5.888; ordinary Nsight47.792/48.992/49.152. Do not add medians.
PayloadDrained issue4.6–5.1us is earlier than control PayloadPushed6.6–9.3us,
but the markers have different completion boundaries; no destination-ready gain
established. Three traces do not prove the larger combine tail's cause.
Preserve first-MMA20.512/132.704/32.256us outliers, with incoming TopkReceived
15.040–15.296/127.072–127.104/25.120–25.920us despite local TopkSent near3us.
Do not assign causes to untraced peers. Prefix TopkReceived16raw/8unique and
SizesReady2raw/1unique preserved alongside End18raw/17unique; no extra participants.

Bounded SASS inspection gives a plausible cost for async publication: original
payload tail has full TMA drain/proxy fence, one tid0 SC.SYS fence between joins,
then lane-parallel REDG SYS reductions. New tail emits a CTA MEMBAR and uniform
address broadcast/reconvergence sequence for each peer's UREDGR SYS release.
Fewer source-level fences/joins does not establish less machine work. This is
not exclusive causal proof; metadata/index also use UREDGR, so aggregate opcode
counts cannot attribute the payload cost.

**source_reuse_compare CPU review7215491 COMPLETED0:0/1m02s, PASS all8 builds:**
128regs/24-byte stack/no spills; dynamic dispatch39/48 smallM,41/48 M192,
compute18/40 (vector M19217/40), no local memory/unresolved transfers.
CPU7215404 FAILED1:0/4m06s in inspector:
four vector builds succeeded128regs/24-byte stack/no spills; review helper's
fixed artifact-depth assumption missed codex_scripts import. No kernel failure.
Recovery explicitly supplied canonical codex_scripts PYTHONPATH, reviewed existing
vector cubins, and compiled/reviewed the cached variant. No redundant vector build;
original failure and cubins retained. Independently rebase source-vector
payload and four-group SF cache onto faster gather_sf_metadata. Vector header
was composed locally, synced, then frozen: old four-warp count/prefix and original
tid0 SYS fence/join/parallel relaxed publication; per-writer async/global proxy
fences order generic payload stores for TMA consumers. Cached variant changes
only SF main and keeps original TMA dispatch. Eight builds use actual runtime
include roots, complete cutlass/cute links, no extra canonical include. New16-arm
GPU7215591/analysis7215592 submitted after7215144 audit; independent variants,
fresh pull,336 exact+96 replay+16 recordings+3 critical32 captures. Earlier CPU
records remain under their original parent; no false provenance rewrite.

**GPU7215074 FAILED1:0/1m43s; analysis7215075 CANCELLED** before first validation,
in `R/artifacts/push_dispatch/warp_count_prefix`, on nvl72106-T[04,07].
Runtime layout JIT could not find `cutlass/detail/helper_macros.hpp`: new overlays
linked only deep_gemm/, omitting top-level cutlass/ and cute/. CPU builds' secondary
canonical include had masked this packaging error. No kernel/correctness/perf result.
Both links now repaired in source_tma_release, warp_count_prefix and source_vector_payload;
kernel headers unchanged. **CPU7215129 PASS** in warp_count_prefix/runtime_include_check:
exact failed layout source plus each overlay's M16, using only the runtime include.
Distinct `warp_count_prefix/runtime_recovery` completed after CPU success;
original failure logs/records retained. Same16 arms,35min limit:
gather_sf_metadata control; source_tma_release alternate; warp_count_prefix candidate;
fresh pull. Both new variants have64-reference gates;336 exact comparisons,
96 replay checks,16 recordings and3 critical32 captures expected. Same gather-warp
SF main, isolating sender publication then receiver prefix.150 CTAs/math/combine fixed.
One GPU allocation at a time. Recovery is finished; source_reuse_compare is active. Source/native/kernel
choices frozen; no canonical mirror edits. No defaults promoted or goal completion.
Initial sbatch explicitly rejected completed-job dependencies (no job/stdout);
all four predecessors verified COMPLETED0:0, then submitted without obsolete
dependencies. Failure record retained; no observation-timeout restart.

**Local state:** producer metadata/input publication with original gather-warp SF,
source-TMA/fence/publication and four-warp prefix, CPU7215833. source_reuse_compare
cached/vector sources remain frozen; earlier CPU records remain intact.
Host is source-count guard. Live
overlays frozen. Analyzer recognizes IndexScatter/TopkWritten and
PayloadIssued/PayloadDrained, separating release issue from remote readiness;
warns that first-stage SF work can move into the pre-compute gap. Results stay remote.

**cached_gather_sf CPU7215265 COMPLETED0:0/1m04s:** all4 builds128regs/24-byte stack/no spills;
dynamic dispatch39/48 atM16/M32,41/48 atM192, compute18/40, no local memory or
unresolved transfers. Small-M SF cache holds four adjacent four-stage groups:
loads every16K stages, shifts consumed register groups every4; largeM keeps one
group within the same48-register budget. Final partial group guarded; all original
payload/index/stage acquires, proxy fence,160-thread join, layout/math/combine fixed.
Hypothesis: reduce repeated SF load dependencies that serialize activation issue;
extra initial loads/register moves may offset any gain. CPU-only evidence so far.
Prepared16-arm plan independently compares source_vector_payload and cached SF,
both versus warp_count_prefix. **Held, unsubmitted**: early7215143 data instead
favors gather_sf_metadata; rebase future candidates to that parent if final audit
confirms. Do not stack changes on a slower parent or rewrite current CPU provenance.

**source_vector_payload CPU7214998 COMPLETED0:0/4m06s:** all4 builds128regs/24-byte
stack/no spills. Dynamic dispatch39/48 atM16/M32,41/48 atM192, compute18/40 or17/40;
no local memory/unresolved transfers. SASS confirms128-bit source loads and the
same R24..R55 row vectors reused across successive peer128-bit store sequences.
Based on frozen warp_count_prefix,
one source-owned warp deduplicates destination ranks, loads a complete FP8 row into
uint4 registers (8 vectors/thread at4096 hidden) and SF words, then reuses them for
coalesced peer stores. Removes payload shared staging/input barriers/TMA full drain.
Every writer executes async/global proxy fence, CTA join precedes uniform async
SYS releases; fixed readiness threshold and helper retirement retained. Marker is
PayloadIssued, not remote visibility. Same metadata/index/SF/math/combine/native.
Tests a distinct hypothesis from rejected destination-generic sender: eliminate
its repeated local loads without relying on those loads as proven regression cause.
Risks: register pressure/code size, bulk throughput and store issue bandwidth.
GPU unsubmitted: standalone11-arm plan deferred into cached_gather_sf16-arm plan,
now held for the parent decision above. Live recovery sources unchanged.

**warp_count_prefix CPU7214893 COMPLETED0:0/1m11s:** all4 builds128regs/24-byte
stack/no spills; dynamic dispatch39/48 atM16/M32,41/48 atM192, compute18/40,
no local memory/unresolved transfers. One warp acquires all metadata flags,
warp-joins, folds immutable source counts and scans padded expert prefixes in
32-expert chunks with register carry. It writes final shared offsets and publishes
SizesReady after a warp join; one final128-thread join orders metadata acquires
and offsets before every scatter warp. Removes shared histogram/warp sums and
intermediate CTA joins. Same source-ring offsets, arbitrary valid counts, exact
tile publication, source-owned TMA release, main/math/combine/native. Larger local
expert sets may trade parallel loads for fewer joins; no performance gain presumed.
Live16-arm plan here isolates gather_sf_metadata control, source_tma_release
alternate, then this prefix candidate and fresh pull. Two sparse64-reference gates,
336 exact+96 replay+16 recordings,3 critical32 captures expected. Shell syntax PASS.
GPU7215074 failed runtime include setup; CPU7215129 PASS, recovery7215143 submitted. Standalone
source_tma_release11-arm plan deferred into this combined comparison. Source overlays
frozen; recover concrete files rather than header symlinks if changing local parent.

**source_tma_release CPU7214808 COMPLETED0:0/4m08s:** all4 builds128regs/24-byte
stack/no spills; dynamic dispatch39/48 or41/48, compute18/40, no local memory or
unresolved transfers. Preserve gather_sf_metadata's source
row reuse, route deduplication and two-slot TMA pipeline. Each of the four issuers
fully drains outgoing TMA, then128-thread join precedes warp-uniform async SYS
releases (peer=warp,stride4). Removes explicit async/global proxy fence, blocking
SYS fence and second CTA join. Full TMA completion supplies implicit generic/async
visibility; receiver acquire,64-contribution threshold and empty-pusher/helper
retirement unchanged. Same main/math/combine/native. PayloadDrained distinguishes
full TMA completion/release issue from the old completed SYS fence marker. First
GPU attempt7215074 failed before validation; no gain assumed. Primary PTX contract checked.

**gather_warp_sf CPU7214207 COMPLETED0:0/1m03s:** all4 builds128regs/24-byte stack,
no spills. Dynamic dispatch maxR39/48 atM16/M32,41/48 atM192, compute18/40;
no local memory or unresolved transfers. Distribute SF handling across the existing
four activation gather warps. A lane quad owns one SF row and caches4 adjacent
K words, sharing one selected word per stage by quad shuffle. Cache source row IDs
per FC1 block. All gather threads acquire incoming payload once and each stage's
empty barrier before shared writes, then async shared proxy fence +existing160-thread
join order them before TMA-A's original full-barrier arrival. Both CTAs fill all
valid SF rows. Removes TMA-A cp.async SF groups/waits, no extra join/arrival/scratch.
Full barrier4, activation expected bytes, FC2/math/combine/CTA count unchanged.
Empty-destination helper retirement is preserved from direct SF. Hidden%512 static
guard covers complete4-stage word batches; existing fused quant already needs%1024.
Risks: gather register pressure, extra empty-barrier polls/shuffles/fence, exposed
first fetch. Existing48/40 budgets remain sufficient. Used in audited7214400.
Primary PTX shared-memory/tcgen05 proxy ordering checked; target still missed.

**gather_sf_metadata + destination_tma_sf CPU7214370:** COMPLETED0:0/1m27s,
8 builds (M16/M32/M192 ordinary +M16 critical per composition),128regs/24-byte stack,
no spills. Both dispatch39/48 atsmallM,41/48 atM192;compute18/40, no local memory
or unresolved transfers. Former combines gather-warp SF main and previously built
direct_metadata index-only dispatch with original TMA sender. Latter retains
two TMA slots but one pusher owns each destination; warp scans for matching rows,
prefetches first2, then refills previous slot after outgoing read1 while current
send progresses. Per-slot barrier phases persist across strided peers. Full outgoing
drain +128-thread join precede one async SYS release of kPushers per destination,
including empty messages. Removes all-pusher/all-peer fanout and separate SYS fence;
full TMA completion supplies generic/async proxy ordering. PayloadDrained marks
that boundary and release issue, distinct from old PayloadPushed and generic
PayloadIssued. Helper still acquires all incoming payload before empty-rank cleanup.
No new scratch/CTA/native/math/combine change. Risks: extra route scanning, uneven
warp work, first-use SF cost. Each composition needs GPU correctness/replay.
Prelaunch plan assertion caught validation tuple ordering; corrected before submit.

**destination_payload CPU7213868:** COMPLETED0:0/1m06s; M16/M32/M192 ordinary
and M16 critical128regs/24-byte stack/no spills. Dispatch maxR39/48, compute18/40;
no local memory/unresolved transfers. Same source-count/global-SF parent. Existing
pushers own destinations, warp-copy uint4 activation and uint32 SF words directly
to peers, then per-writer async proxy fence +128-thread CTA join +one async SYS
release. Removes local TMA staging/drain and all-pusher/all-peer signal fanout.
One owner per destination contributes kPushers to unchanged incoming threshold;
empty destinations still publish, EP/pusher mismatch uses a strided peer loop.
All participants still retire DispatchDone. PayloadIssued means publication
issued, not destination visibility; consume/acquire boundary remains explicit.
No ABI/native/CTA/math/combine change. Risks: duplicated local reads, SM store
pressure and bulk bandwidth. Source TMA path retained for measured comparison.
Standalone GPU plan is deferred into the combined comparison below. Prelaunch
include assertion caught missing unchanged-header links;
added bounded symlinks, no kernel/CPU-build change. Analyzer shell syntax/diff pass.

**destination_metadata CPU7214011:** COMPLETED0:0/1m03s; all4 builds128regs,
24-byte stack/no spills. Dispatch maxR39/48, compute18/40, no local memory or
unresolved transfers. Rebase the previously compiled
direct-metadata method onto destination_payload with canonical global SF retained.
Coalesced generic topk/score stores overlap shared counting; counts stores +CTA
join +async SYS publication replace shared metadata staging/TMA/full drain.
Metadata readers use generic proxy, existing payload proxy ordering unchanged.
This is a smaller dependency change than combining with the rejected serial SF.

### Completed: destination TMA trades startup distribution against throughput

`destination_tma_sf`: GPU7214400 COMPLETED0:0/26m35s on nvl72098-T[07,12];
analysis7214401 COMPLETED0:0/47s. Audit PASS17 arms,400 exact comparisons,
96 replay checks,16 reference recordings,0 separate conventional-combine checks;
empty stderr. All3 critical32 captures pass,3 IKET samples+10 ordinary Nsight calls
each. All samples/markers/first-MMA locations saved in result_summary.json.
All three use gather-warp SF; comparison isolates metadata and sender ownership,
not this SF method against older controls on different allocations.

Pipeline4/32: control50.93693/55.17674us, directmetadata50.96703/54.97907us,
destinationTMA49.54376/56.41666us, pull45.48540/47.32657us. Metadata remains
12.051%/16.170% behind pull; destinationTMA8.922%/19.207%. Kernel control
43.99210/48.69795us, metadata44.00634/48.38948us, destination43.09456/49.99894us.
Direct metadata saves0.30848us at32 with every cross-repeat agreeing;4 overlaps
control variation. Destination ownership helps4, regresses32; no target win or promotion.

Critical32 control/metadata/destination medians: first MMA13.696/13.728/13.440us,
compute27.488/27.360/28.576us, exposed dispatch10.816/11.104/11.840us,
precompute gap2.336/2.496/1.248us, exposed combine4.512/4.128/5.600us.
Ordinary Nsight48.336/47.008/49.984us. Do not add independent medians. Destination
helper/payload CTA first-MMA maxima14.816/15.424/15.744us while compute-only maxima
14.464/14.048/14.240us: correlation with compute spread, not exclusive cause proof.
Preserve control first-MMA20.512 and metadata35.968us outliers: incomingmetadata
14.848–15.104/30.176us despite local TopkSent~3us and PayloadPushed<9.3us.
No explanation assigned to untraced peers. All raw/unique marker counts retained.

### Completed: generic payload stalls startup

`destination_metadata`: GPU7214045 COMPLETED0:0/27m50s on nvl72151-T[01-02];
analysis7214046 COMPLETED0:0/54s. Audit PASS16 arms,336 exact,96 replay,16 recorded
references,0 separate conventional-combine checks; empty stderr. All3 critical32
captures pass (3 IKET samples +10 ordinary Nsight calls each). Result summary saved.
Source-count/global-SF fixed; payload-only alternate then added metadata candidate.

At4/32 pipeline control50.57437/55.72064us, payload50.29582/72.75091us,
metadata50.27522/72.82180us, pull44.95454/47.27382us. Candidate remains
11.836%/54.043% slower than pull. Kernel control43.81917/49.00678us,
payload44.17274/73.12096us,metadata44.15318/73.16275us.4-row pipeline gain reverses
in kernel-only scope; do not subtract different preparation/cache/dependency
histories to assign producer cost. Every32-row repeat regresses strongly.

Critical32 control/payload/metadata medians: first MMA16.576/40.096/40.352us,
compute25.792/25.984/25.760us, exposed dispatch15.776/39.616/39.456us,
combine4.640/5.344/5.184us. Nsight ordinary48.720/73.136/72.880us. Generic
PayloadIssued~31–32us, receiver FusedScatter~33–34us, IndexReady~39–41us;
control TMA PayloadPushed~8.6us, FusedScatter~9us, IndexReady~15–16us. These
source marker types differ, but receiver milestones and first MMA locate the
regression before compute. Metadata_received~10–11us is hidden behind payload.
Retain control first MMA48.672us and payload78.816us outliers, with late incoming
metadata/payload; no cause assigned to untraced peers. No samples discarded.
SASS payload LDG.E.128 +ST.E.128 is not scalarized; no root-cause claim from that.
Reject generic payload, retain TMA. Direct metadata isolated next on TMA transport.

### Completed: SF overlap helps32 rows, warp SF scatter regresses

`warp_sf_scatter`: GPU7213770 COMPLETED0:0/27m59s on nvl72151-T[01-02],
analysis7213771 COMPLETED0:0/46s. Audit PASS16 arms,336 exact comparisons,
96 replay checks,16 recordings,0 separate conventional-combine checks;empty stderr.
Three valid32-row captures,3 IKET samples and10 ordinary Nsight calls per arm.
`result_summary.json` stores all timings/markers/samples and rejection decision.

Pipeline source-count control50.54470/55.72626us, overlap50.55259/54.96658us,
warpSF50.77567/56.30511us, fresh pull44.97859/47.30043us at4/32. Overlap remains
12.393%/16.207% slower than pull. Kernel source43.80253/48.84608us,
overlap43.80947/48.24566us, warp43.85069/49.51776us.32-row overlap saves0.6004us
with every cross-repeat sign agreeing;4-row differences overlap control variation.
Reject warpSF as next parent; retain overlap as diagnostic alternative only.

32-row critical medians (source/overlap/warp): first MMA17.952/13.248/18.656us,
compute25.408/27.712/25.984us, exposed dispatch17.024/10.816/17.792us,
pre-compute gap0.480/1.952/0.256us, exposed combine4.672/5.696/3.840us.
Nsight ordinary kernel49.024/48.128/49.392us. Do not sum independent medians.
Preserve source first MMA123.936us and warp142.016us outliers: incoming metadata
115.616–115.680us /133.248–133.312us versus local send~3us,payload~9us. No cause
assigned to untraced peers. Other source starts16.800/17.952,warp18.656/18.400us.
Overlap starts13.248/14.112/13.216us, IndexReady final11.104/12.352/11.200us.
Its earlier start still trades against longer compute and combine exposure.

### Completed: serial direct SF is slower despite earlier first MMA

`direct_sf_gather`: GPU7213127 COMPLETED0:0/20m20s on nvl72066-T[06-07]; CPU7213128
COMPLETED0:0/44s. It cleared the normal short-QOS wait without scheduler changes.
Audit PASS11 records,208 exact reference comparisons,16 reference records,
64 exact replay checks,0 separate conventional-combine checks,empty stderr.
Both32-row captures pass,3 IKET samples and10 ordinary Nsight calls each.
`result_summary.json` records outcomes, markers, samples and rejection decision.

At4/32, pipeline source-count control50.68149/55.70910us; serial direct SF
51.41223/56.50268us; fresh historical45.21655/47.29662us. Candidate still
13.702%/19.465% slower than pull. Kernel control43.80166/48.83027us versus
candidate44.49808/49.82976us: +1.590%/+2.047%. Every cross-repeat sign agrees.
No numerical threshold relaxed; no promotion based on smaller dispatch labels.

32-row critical medians control→direct SF:
- First MMA16.640→13.472us; compute span25.344→29.376us.
- Exposed dispatch15.680→10.816us, pre-compute gap0.512→2.112us.
- Exposed combine4.800→4.448us; ordinary Nsight kernel48.464→49.472us.

Do not add median components as exact attribution. The ~4.9us dispatch-label gain
is not a ~4.9us startup gain: first-stage SF/activation work moved into the gap.
First MMA advances3.168us while compute lengthens4.032us; ordinary time regresses.
Candidate IndexScatter starts8.896–9.216us, finalIndexReady11.104–11.328us;
payload finishes9.248–9.408us; first MMA13.216–13.472us. No outliers discarded.
The per-stage wait_group0 is a concrete throughput hypothesis, not a proven sole cause.

This draft removed canonical-SF writes/readback and the index-scatter payload gate.
TMA-A caches source-row IDs and gathers4-byte SF words into the same permuted shared
layout each stage; delayed original A arrivals cover completion in both CTAs.
Full barrier stays4 arrivals, activation-only expected tx bytes. FC1 acquires all
payload once at first use; the helper also acquires it before DispatchDone for
empty-destination cleanup. CPU7213006 passed128regs/24-byte stack/no spills,
M16/M32/M192 ordinary and M16 critical. No ABI/native/math/combine change.

### Preserved implementations from the completed SF comparison

**direct_sf_overlap7213526**, CPU COMPLETED0:0/1m02s. Two SF copy groups overlap:
issue current group/activation gather, wait_group1 then publish preceding stage's
A arrival; drain final group/arrival at every FC1 block end. Empty-stage acquire
still gates writes and reuse, both CTAs contribute one A arrival per stage; FC2
unchanged. Host already requires>=2 stages; device assertion makes it explicit.
No fallback. M16/M32/M192 ordinary +M16 critical128regs/24-byte stack/no spills;
dispatch39/48, TMA-A23/40 or31/40, no local memory/unresolved transfers. SASS
TMA-A instruction regions321→353,320→352,367→392. GPU result above.
Risk: first MMA waits for the second group's issue; exact/replay gates must cover
stage wrap and final flush. Frozen main paired with original staged metadata.

**warp_sf_scatter7213727**, CPU COMPLETED0:0/1m45s. Source-count/global-SF parent:
each route owner still allocates/writes index/score/source metadata. Ballot/shuffle
then lets the whole warp copy each selected row's SF words across K, replacing
sparse lanes' serial32-word stores. Warp-uniform route bound includes tail lanes;
unchanged proxy fence, CTA join and exact tile-count publication cover all writes.
Canonical layout and FC1 TMA retained.128regs/24-byte stack/no spills; dispatch
39/48 and all compute regions18/40, no local memory/unresolved transfers across
M16/M32/M192 ordinary and M16 critical. No native/ABI/CTA/math/combine change.
Risk: dense routing can serialize more row owners and worsen store coalescing;
large/other-model performance checks remain required after a target win.

GPU7213770 compared both to the faster global-SF control and fresh pull.
Acceptance remains the retained pull implementation.

**Deferred, no GPU submitted:** direct_metadata7213160 and composition
sf_overlap_metadata7213590 both CPU PASS (1m03s each), safe register regions/no spills.
Generic coalesced metadata stores while counting replace shared staging/TMA drain;
CTA join +async SYS release publishes counts/data. TopkWritten distinguishes the
changed milestone. Standalone11-arm and combined16-arm plans remain prepared;
SF-method evidence now takes priority. No edits to their captured sources.

### Retained source-count result

`source_counts`: GPU7212895 COMPLETED0:0/28m29s; analysis7212896 COMPLETED0:0/48s.
Audit PASS16 records,336 exact comparisons,16 recordings,96 exact replay checks,
0 separate conventional-combine checks,empty stderr. Three32-row captures pass.
Pipeline scalar51.12505/56.69243us, uniform51.58466/56.85065us, source counts
50.40684/55.58243us, pull45.38200/47.30522us at4/32. Source counts remains
11.072%/17.497% slower than pull. Kernel scalar44.67069/49.94579us, uniform
45.06058/50.10186us, source43.83654/48.75706us. All cross-repeat signs agree.
Uniform release alone regresses; source counts saves0.834/1.189us inside MegaMoE.
32-row dispatch/compute/combine medians scalar16.000/25.536/4.928us,
uniform16.448/25.792/4.032us, source15.360/25.440/5.312us. Preserve scalar first
compute44.032us outlier (other15.744/17.184): late peer metadata, not SF work.
`result_summary.json` retains full results. Prior scalar7212583/7212585 also passed
208 exact+64 replay checks but regressed32-row kernel1.273%; artifacts retained.

### Open dependency questions, not accepted conclusions

- Producer-boundary audit: prior compact selector produced only local top-k/scores;
  dispatch immediately griddepcontrol.waited, requiring producer completion/flush.
  CPU7215833 now builds the producer metadata + local GPU input-counter protocol
  above. Payload TMA remains in MegaMoE; counter acquire/proxy fence permits overlap
  with the selector network tail. Sparse/empty routes, bank retirement and exact
  full-pipeline replay still require GPU validation. Kernel-only replay is invalid
  after these producer-published flags reset. Historical producer-path failures
  bundled older indexing/SF methods and do not settle this current composition.
- Workload audit: actual target uses `MEGAMOE_SMOKE_BALANCE_RANKS=1`, mapping
  expert IDs as `(id%EP)*local_experts+id//EP`. With topk=EP=8 each token visits
  ALL8 ranks, not one rank. At32 rows each source sends~1MiB activation payload.
  Destination ownership reloads each local row8 times; source-owned sender loads
  it once then fans out. This tradeoff, not fewer flags alone, must explain timing.
  Source `tests/afd_megamoe_same_rank_smoke.py` also confirms pipeline invokes
  prepare_input each iteration whereas kernel-only prepares once outside capture.
  Cache/dependency histories differ; never subtract the two timing scopes to
  assign producer cost. The actual cause of their discrepancy remains unisolated.
- Source-owned TMA full-drain/join/async-release prototype now compiled, with
  separate one-warp prefix candidate; recovery GPU7215143 submitted (state above).
  Separately, an unimplemented hypothesis:
  gather warps owning no valid SF rows could skip SF-only empty waits/shuffles/
  fences while still joining the existing activation rendezvous. Await current
  SF results; do not edit the queued implementations to add these hypotheses.
- Required startup chain: producer inputs → concurrent metadata/payload sends;
  metadata acquire → dense counts/padded bases; max(prefix-ready,payload-ready) →
  canonical index/SF scatter → tile acquire/first MMA. SF-gather alternatives move
  this work into the consumer. Optimize these actual dependencies, not the label
  "all-push" or phase names. The2us one-way planning number is not a measured
  latency bound for release ordering plus local indexing and compute setup.
- Gather-warp SF arms and correctness/trace audit completed in7214400/7214401.
  Current compositions miss the retained pull target. Do not infer its
  isolated gain by subtracting controls from different allocations.
- Push does not prove one-way critical latency: TMA senders drain writes before a
  separate readiness publication. A read-only drain does not prove destination
  visibility; generic red.async release and async-copy proxies are distinct.
  Re-reading the Rubin genphase source confirms its own stated design target is
  one round trip, not one-way completion; that comment is not a GB200 measurement.
- At4 rows only one bucket-sized pusher group owns tokens but all eight fence and
  signal every destination. Aggregating empty contributions is unimplemented;
  it cannot explain the32-row gap where all eight groups own tokens.
- Destination-owned generic payload was rejected by7214045; destination-owned
  TMA7214400 improves4 but regresses32. Neither settles whether
  source-owned generic stores with register reuse could avoid full TMA drain
  without the prior repeated-load cost. CPU7214998 compiles that implementation;
  this remains unmeasured and not an explanation for the31us generic-sender stall.
- Historical early-producer-push regressions bundled old sort/helper/SF machinery;
  they do not isolate early sending itself. Any revisit must use the current
  measured local path and include producer work in the acceptance timing scope.

## First-principles restart — September 17, latest user direction

User explicitly warns that historical optimization conclusions may mislead.
Treat previous gains as observations, not constraints; derive dependencies from
the current code and test their actual cost. Previous continuation made progress
(batching results and adaptive CPU build), not a live wait. Fresh sacct verifies
7211923/7211924/7212154 COMPLETED0:0 and no outstanding jobs before new work.
Adaptive batching's prepared19-case GPU plan stays unsubmitted: it targets bulk
payload throughput, while the requested small-row target has serial local handoffs.

**Target correction:** actual retained `final_decomposition_20260915/qwen3/`
HTML uses EP8 rows4/32, not the recent16/31. Its ordinary kernel medians are
37.824/40.016us, exposed dispatch6.976/6.592us and exposed combine4.480/4.960us.
HTML ordinary full producer chain47.040/49.952us includes producer/metadata work;
never compare that scope to kernel-only timings. Use4/32 now, with fresh retained
pull baseline in the same allocation; keep the HTML observations as targets.

**Fused tile scatter:** dense counts/padded bases are still required. Once these
exist, a row needs its source-index, score, combine destination, payload and
canonical SF; it does not need unrelated tiles' indices. Exchange workers now
acquire payload at SF consumption, allocate final rows, and write canonical SF
alongside index/score/metadata. Async-proxy fence then per-row GPU release increments
the existing L1 tile counter; both FC1 gather and SF consumers acquire the exact
valid-row count. Removes global IndicesReady wait and separate refiner work.
All EP+pushers+1 participants retire through DispatchDone before arena cleanup;
empty workloads remain covered. CountsReady and all-pusher payload gate remain.
SF scatter/counter traffic and delayed exchange-CTA compute are explicit risks.
PTX release/acquire/proxy requirements checked against NVIDIA PTX ISA documentation.

For isolation, restored the validated one-row previous-slot sender from remote
payload_batch/control (only necessary header recovered); adaptive draft remains
preserved remotely. No batching/masks/native/layout/CTA-count/compute/combine
change in this experiment. Two device headers plus profile analyzer changed;
GenphaseFusedScatter distinguishes the new8 exchange refiners from old9 groups.
Canonical remote mirror stays frozen, candidate uses a two-header include overlay.

R/artifacts/push_dispatch/fused_tile_scatter holds source/contract/plans/evidence.
CPU7212510 COMPLETED0:0/1m08s: M16/M32/M192 ordinary and M16 critical compile,
128regs/24-byte stack/zero spills; dynamic regions R39/48 and R18/40, no local
memory or unresolved branches. GPU7212550 submitted25min/EP8: Q3 sparse/edge
exact gate, six pipeline arms (pull/control/candidate then reverse) at4/32,
four all-push kernel-only arms, then control/candidate/pull critical32 captures.
CPU7212551 afterok analyzes. This is a hypothesis under validation, not a win.
Keep sources/overlays/caches frozen while consumed; results stay remote.

7212550 FAILED1:0/1m35s before MegaMoE execution: my runtime overlay omitted the
library-root `include/` level, causing weight-layout include resolution failure.
Cancelled dependent7212551; all failure artifacts retained. Wrapped unchanged
header tree in the required include directory and checked the exact missing file.
Recovery plan `fused_tile_scatter/runtime_recovery`: GPU7212583, analysis7212585.
No source/protocol/compiler change or new CPU rebuild; no numerical/perf conclusion
from the failed setup. Read `runtime_recovery` for current execution, original root
for compiled source/CPU/SASS evidence.

Recovery7212583's Q3 arbitrary-route/empty-rank/padding gate passes64 saved-input/
output comparisons. This driver does not execute the fixed-route conventional-
combine ablation (its marker count is0); corrected the analyzer's mistaken >0
expectation to the actual64-reference gate before its CPU job starts. Do not report
these as separate combine checks. Ordinary pipeline/replay/critical phases remain.

CPU-only follow-up `fused_tile_release` (7212651 COMPLETED0:0/1m02s): scalar publication SASS contains
MEMBAR.ALL.GPU+REDG per route iteration. Source rows already form contiguous
expert intervals, so after proxy fence+CTA join each warp publishes interval/tile
intersection counts with uniform async release, removing per-row GPU fences and
atomics. The true consumer acquire is unchanged; publication overlaps remaining
expert releases and GEMM setup. Tradeoff: waits for whole source scatter before
first tile publication. No runtime gain assumed. Local dispatch is this CPU draft;
live scalar GPU source/overlay untouched. Main header shared with frozen scalar
candidate; no native/ABI/CTA/combine change. Need completed scalar timing/trace and
new compile/register results before choosing GPU follow-up. M16/M32/M192 ordinary
and M16 critical compile with128regs/24-byte stack/zero spills; inspected dynamic
regions stay R39/48 and R18/40. Generation-prefix MEMBAR.ALL.GPU sites drop1→0
in all three ordinary builds, confirming instruction removal, not a latency gain.
Its11-arm4/32 GPU plan and paired32 captures are prepared, **unsubmitted**.

First-principles `source_counts` draft now implemented locally, based on the frozen
uniform tile-release candidate. Each sender histograms its destination while
staging top-k, then generic64-bit count stores progress alongside metadata TMA.
CTA join plus TMA drain precede the existing async SYS publication. Receivers
acquire every metadata flag, then directly fold immutable counts; no receive
recount or CountsReady reduction/poll. Existing double-buffered count rows retain
epoch lifetime. Added1KiB/rank Q3EP8 count traffic and sender histogram cost must
be measured, not assumed free. Host shared bound now includes4*local_experts after
8*aligned(wire_routes); actual GEMM allocation/native ABI unchanged. TopkReceived
now means all metadata/counts acquired, so its old paired-peer timing is not an
identical phase boundary. No new instrumentation or change to compute/combine.

Artifacts R/artifacts/push_dispatch/source_counts; CPU7212728 COMPLETED0:0/1m57s,
native guard build7212729 still running at last read. Local dispatch+host header
are this next draft; scalar GPU source and uniform CPU artifacts stay immutable.
No source-count GPU plan submitted. Prefer one combined follow-up comparing
original all-push / uniform fused / source counts / historical pull over separate
GPU allocations if completed evidence supports it. Preserve exact source-specific
controls; an instruction removal is not a performance conclusion.

Scalar7212583 ordinary arms now complete (profiles pending): kernel-only control→
candidate means rows4:45.74635→44.87170us(−1.912%), rows32:49.60915→50.24045us
(+1.273%). Every cross-repeat comparison has the same respective sign (4 gain
0.791–0.958us;32 regression0.450–0.813us). Pipeline control52.83995/56.51665→
candidate51.37375/56.86220us; historical pull45.21160/47.21775us. No acceptance.
Source-count plan prepared16records: original all-push, uniform fused, source counts,
historical pull symmetric pipelines; all three all-push kernel-only pairs; separate
64-reference sparse gates for both new candidates; three32-row critical captures.
All share the new native guard except historical pull.336 ordinary reference and
96 replay checks expected; actual dimensions/4,32 keys/valid ranks audited.
Standalone uniform GPU plan superseded, remains unsubmitted. No extra allocation
for each draft. GPU7212583 time-limit extension request25→30min was denied by
Slurm (not sandbox); original25min limit remains. Last reference capture running
at22m39s, preserve completed evidence if it reaches the limit; never restart on
observation timeout alone.

## Scope clarification — September 17 UTC

User explicitly defines the measured megakernel region: metadata transfer,
payload transfer, MoE compute and combine. Keep combine fixed at Lamport push
plus polling; expected exposed tail approximately4–6us, to verify from final
accumulator-ready to combine completion (not the whole overlapping combine span).
The optimization target is CuTeDSL Gen-specialized all-push dispatch. User further clarifies
that the expected mechanism is reduced exposed dispatch versus pull, saving one
network leg; total kernel time remains the final metric. Earlier all-push-only ABBA
did not establish a speedup over pull. Matched router+kernel ABBA7209335 now shows
the candidate slower than historical pull, with fixed Lamport push+polling. Its
scope is broader than the requested megakernel interval because historical pull
pushes metadata in its router. Report that placement explicitly; never label these
pipeline timings kernel-only. Current all-push kernel-only runs and critical IKET
include its metadata/payload/compute/combine, with router/quantization outside.

## Dispatch latency model — user-provided working assumptions

Use approximately **2us one-way NVLink latency** and **4us round-trip latency**
when estimating expected dispatch ranges. These are the user's planning values,
not latency measurements established by the current benchmark.

- Payload push transport floor: about2us plus transfer-size/bandwidth cost.
- Payload pull request/response floor: about4us plus transfer-size/bandwidth cost.
- Replacing that pull round trip with a push removes approximately2us of transport
  latency for that leg. Do not claim a2us measured gain or a2x whole-dispatch gain
  from this assumption alone.
- Build the estimate from the actual dependency timeline. In Gen-specialized
  dispatch, metadata and payload pushes start concurrently: the consumer waits
  for both final indices and payload readiness. A useful model is
  `max(metadata arrival + local index preparation, payload arrival) + SF refinement
  + remaining readiness/compute-start overhead`. Do not sum overlapping transfers.
- For the historical pull baseline, account for whatever metadata/index work must
  finish before the payload request can issue, then the request/response path.
  The exposed benefit can differ from2us because dependency overlap changes too.
- Add launch/startup, serialization, contention, local metadata/SF work and
  required signaling explicitly; keep these separate from NVLink flight time.
  Compare the model with ordinary latency and lightweight trace evidence. Current
  ~17–20us exposed all-push dispatch is not explained by a2us network leg alone.

## Latest measured checkpoint — September 17 UTC

### Status clarification: small-row bottleneck and completed batching timings

Small-row decomposition evidence predates the latest payload variants: concise_push
7208884 medians at16/31 rows are exposed dispatch17.056/19.872us,
compute24.736/26.752us, exposed combine4.768/3.584us. Payload publication finishes
around9.3–9.8us; metadata/count/index/SF handoffs continue until17–20us. Later
local_prefix7209768 at16 rows measures15.584us dispatch,25.600us compute,3.520us
combine (first-sample metadata outlier retained). This supports prioritizing local
dispatch dependencies for small rows; it is not a fresh decomposition of batching.

Batching GPU7211923 COMPLETED0:0/24m24s. Ordinary audit and both decompositions
are written; CPU7211924 was still running at the status read. Audit464 exact
reference comparisons,32 combine checks,96 replay checks, empty stderr.
Kernel-only control→two-row means at16/31/1024:47.4771→48.3535(+1.846%),
50.1231→50.7108(+1.172%),297.8661→291.2337us(−2.227%). Pipeline:
54.7466→55.3615(+1.123%),56.7692→57.2028(+0.764%),317.4191→314.3226us(−0.976%).
Historical pull46.4871/47.3041/258.7903us; candidate remains19.09/20.93/21.46%
slower. Do not adopt unconditional batching for small rows.

Batching1024 critical samples: first compute162.016/89.920/90.304us,
payload82.048/83.360/83.168us, indices140.864/70.656/69.600us,
compute171.616/171.360/172.256us, exposed combine25.536/25.120/26.208us.
Retain first metadata receive100.672us outlier (others29.760/29.472us).
Control first compute124.928/95.392/94.336us, payload92.096/87.200/87.296us;
its first metadata receive57.600us is also elevated. InputsReady last1.568–1.696us
across both arms, staging last19.456–20.448us: the roughly18–19us staging interval
follows input readiness, rather than being explained by griddepcontrol waiting.
These phase observations do not establish a pure NVLink flight-time measurement.
Adaptive one-row below64/two-row otherwise plus rank masks/coalesced pushes has
passed CPU7212154 (1m08s); its prepared19-case GPU plan remains unsubmitted.

### Current: previous-slot payload overlap validated; batching next

GPU7211301 COMPLETED0:0/26m49s; CPU7211302 COMPLETED0:0/44s. All15 cases and
paired critical captures completed. Audit:464 exact reference comparisons,
32 exact-combine checks,96 exact replay checks; stderr empty. Added Q35 rows129
prefix/row-map padding covers multiple reuse phases and empty committed groups.

At16/31/1024 rows, kernel-only ABBA means (gather+old refill→previous-slot refill,
us):47.6377→47.5458(−0.1929%),49.7391→49.6958(−0.0871%),300.5468→297.8488
(−0.8977%). Small-case control drift−0.7545/−0.6054% exceeds those differences;
large drift−0.1595%. Pipeline means54.6476→54.6562(+0.0158%),56.2318→56.2472
(+0.0273%),329.0906→317.9988(−3.3704%); large control drift−0.0091%.
Historical pull46.3268/47.1851/262.5508us: candidate remains17.9795/19.2053/
21.1190% slower. Keep pipeline and kernel-only replay scopes separate; no router
cost subtraction, acceptance win or default promotion.

Critical1024: control payload93.568/93.536/95.328us, candidate89.024/86.208/88.128us.
Control first compute177.920/100.160/101.856us; candidate95.808/93.344/94.656us.
Retain control first-sample peer-arrival outlier108.384us and indices148.864us;
the other index endpoints69.984/70.112us, candidate69.856/71.328/70.336us.
Compute spans control172.736–177.024us, candidate171.808–175.840us. Exposed combine
control25.664–26.496us, candidate24.832–25.312us, still above4–6us at this shape;
combine code fixed. Nsight ordinary medians301.600→299.424us. Evidence supports
useful read/refill overlap, not a claim that all exposed dispatch is solved.

New metadata markers: candidate TopkStaged last19.904/20.000/20.384us, TopkSent
21.184/21.248/21.664us, TopkReceived29.344/29.888/30.208us. Roughly20us precedes
staging completion; local send completion is roughly1.3us later. These are local
milestones, not pure NVLink flight time. Next paired capture adds InputsReady
immediately after griddepcontrol.wait to separate input waiting from staging.
CPU7211906 COMPLETED0:0/53s: both critical M16 controls with that identical marker
pass128registers/24-byte stack/zero spills; ordinary code and completed capture
remain unchanged.

Select previous-slot overlap as the next diagnostic control. Two-token batching
has passed CPU/native preparation below. Submitted GPU7211923 (14 cases,35min
limit) and CPU analysis7211924 afterok; sources/overlays/caches remain frozen
through consumers. Both all-push arms share the separately built host module;
historical pull retains its own source/cache. Evidence root:
R/artifacts/push_dispatch/payload_read_overlap/{comparison.json,
kernel_comparison.json,audit.json,iket/*/7211301/decomposition.json}.

### Previous: four-warp gather validated

GPU7210893 COMPLETED0:0/24m00s; CPU7210894 COMPLETED0:0/44s. All13 planned
validation/timing cases and both critical captures completed. Core audit:400 exact
reference comparisons,32 exact-combine checks,96 exact replay checks; stderr empty.
At16/31/1024 rows, kernel-only ABBA means (prior prefetch→four-warp gather, us):
47.7054→47.4810(−0.4706%),50.8478→50.3524(−0.9742%),368.0419→301.7664(−18.0076%).
Matched router+kernel means:54.7767→54.4482(−0.5997%),57.0998→56.7291(−0.6492%),
401.2451→330.7533(−17.5683%). Historical pull46.3860/47.2136/257.1518us;
candidate remains17.3806/20.1542/28.6218% slower. Large-case control drift−0.1553%,
pull range0.1373%. Intermediate improvement only; no acceptance/default promotion.

Paired1024 traces: prior compute237.152/239.008/237.440us, gather173.792/173.056/
170.848us. Gather first compute98.720/97.568/99.744us follows payload92.032/90.432/
92.768us; indices68.224/69.024/69.280us finish earlier. Retain prior first-sample
metadata outlier: indices99.456us and first compute127.168us (other starts97.856/
100.160us). Ordinary Nsight medians366.640→301.744us. Exposed combine remains
26.688–28.384us for gather, not4–6us; combine code stays fixed. Sources/overlays
were frozen throughout. Evidence: R/artifacts/push_dispatch/gather_warps/
{comparison.json,kernel_comparison.json,audit.json,iket/*/7210893/decomposition.json}.

Next isolated change targets payload buffer reuse: push the current slot, then
wait only for the previous slot's outgoing reads before refilling it. Leave the
current group's reads in flight with `wait_group.read 1`; use `read 0` if no current
group was committed. Initial two-token prefetch and final remote drain remain.
This overlaps current outgoing reads with the next local load; wait at actual
slot reuse. Local draft only until CPU/GPU validation. Defer counted offsets:
indices already finish before payload on the measured critical path. Preserve its
compiled artifact and plan remotely; do not mix it into this payload experiment.
CPU7211283 COMPLETED0:0/1m05s: M16/M32/M192 ordinary and M16 critical pass
128registers/24-byte stack/zero spills; reachable dispatch R39/48 and TMA/MMA R18/40,
zero local-memory instructions/unresolved transfers in inspected regions.
Remote root: R/artifacts/push_dispatch/payload_read_overlap. Both all-push arms use
validated gather and the same new metadata markers; only slot scheduling differs.
Submitted GPU7211301 (35min limit), CPU analysis7211302 afterok. The15-case plan adds historical-pull recording and candidate exact
comparison at Q35 rows129 with prefix padding/empty ranks and row-map holes, crossing
multiple slot phases and exercising empty-mask waits. Existing Q3/Q35 checks,
matched pipeline/kernel-only timings and paired1024 captures remain. Canonical
sources, header overlays and consumer inputs remain frozen; results stay remote.

During7211301, prepared a separate payload batching draft in
R/artifacts/push_dispatch/payload_batch (CPU7211354). Local dispatch now stages
two adjacent tokens per slot: one contiguous X load and one SF load cover both,
then per-token deduplicated pushes share a committed group. Two-slot previous-read
overlap remains; partial batches load only live rows, all-invalid batches advance
zero-byte barrier parity without committing a group. Same8pushers/4warps/150SMs,
workspace, compute and combine. Host scratch bound increases8→16 rows within the
existing GEMM allocation (fail fast if insufficient); this host change is staged
but not in the live native cache. Before/candidate files preserved remotely.
CPU7211354 COMPLETED0:0/1m03s: M16/M32/M192 ordinary and M16 critical with
validated gather header,128registers/24-byte stack/zero spills; reachable dynamic
regions remain R39/48 and R18/40, no local memory/unresolved transfers.
CPU7211446 COMPLETED0:0/3m56s: separate native module with raised host scratch check.
Its52-file C++ overlay uses symlinks to unchanged headers and the one changed host
header; no source tree copy or live-cache modification. Native artifacts stay
under payload_batch/native. Full overlap result and traces above support choosing
its candidate as control; marker build7211906 passed and GPU7211923/analysis7211924
are submitted under payload_batch. Preserve all compared device variants.
Both future all-push arms use the same new host module; actual shared allocation
unchanged. Reuse7211301's exact historical-pull sparse129 reference, avoiding a
redundant recording arm. Both batching arms now include InputsReady in their
critical-only dispatch header; the analyzer validates17 distinct participants
when present and still accepts old traces without it. No ordinary-code change.
Hypothesis: fewer local-load/barrier rounds reduce exposed bulk payload latency.
Risks: more registers, loading an unused neighbor in sparse batches, fewer active
warps for tiny inputs, larger required scratch. No measured gain claimed. Current
CuTeDSL reference also waits for bulk payload before refinement; progressive
readiness would require a distinct ordering proof, not just removing that wait.
Source-review follow-up, now a local CPU-only draft: CuTeDSL `_classify_tokens` reduces
destination-rank bits into a warp-uniform mask. Our Route still retains per-lane
peer IDs, match-any/ballot deduplication and a shuffle per top-k slot on output.
A rank bitmask removes that state and repeated shuffles: each valid lane ORs its
owner bit within its mask word, and lane0 iterates only set destinations. Preserve
ceil(ranks/32) words, invalid/out-of-range/tail entries and exact deduplication.
Same two-row batches, buffer reuse/visibility, CTAs, host bound, compute and combine.
CPU7211993 builds M16/M32/M192 ordinary + M16 critical and separate dispatch-only
EP64/72 compile probes; the latter do not establish runtime/topology support.
Artifacts R/artifacts/push_dispatch/rank_masks; before header is the frozen batching
candidate. No GPU plan until batching7211923/7211924 evidence selects next step.
Inspect actual set-bit-loop lowering; fewer source operations alone are not a
performance claim. Canonical mirror and live batching inputs remain untouched.

Mask-only CPU7211993 COMPLETED0:0/1m05s, all six kernels compiled with zero spills:
four full kernels128registers/24-byte stack, standalone EP64/72 probes40/48regs,
zero stack. Reachable full-kernel dynamic regions R39/48 and R18/40, no local memory
or unresolved branches. M16 generation-prefix static SASS: indexed shuffles17→1,
MATCH.ANY6 and VOTE.ANY6 replaced by REDUX.OR6; instruction sites2790→2092.
These are compiled sites, not dynamic operation counts or measured latency.

Extended the local draft to coalesce a two-row batch per destination. Iterate the
union of the two row masks; if a peer needs both rows, send contiguous two-row X
and SF regions, otherwise send exactly its single row. Same bytes/destinations,
no padded-row transfer or readiness shortcut. This removes duplicate send commands
where adjacent rows share a peer, with no CTA/workspace/native/compute/combine
change. CPU7212053 COMPLETED0:0/4m10s; full ordinary M16/M32/M192 and critical M16
pass128registers/24-byte stack/zero spills. EP64/72 dispatch-only probes both40regs,
zero stack/spills. Full-kernel reachable dynamic regions R39/48,R18/40 with no local
memory or unresolved branches. Output root
R/artifacts/push_dispatch/rank_masks_coalesced. Code is the saved header under
rank_masks/candidate_coalesced; mask-only header/builds remain immutable.
Future GPU attribution must cover **bitmask plus coalescing together**, not claim
an isolated mask-only gain. Await batching results before choosing the next GPU run.

Batching7211923 paired pipeline timings (closing samples now available):
control16=54.7805/54.7127→candidate55.3577/55.3653us;
31=56.7831/56.7553→57.1934/57.2123us;
1024=317.3614/317.4768→314.3004/314.3448us. Historical pull closing means are about
46.4871/47.3041/258.7903us. Sparse129 and preceding Q3/Q35 checks passed;
kernel-only/traces/full audit still pending. Small regressions repeat; do not
hide them behind the large-shape gain or promote unconditional batching.
Mechanistic risk: pairing rows halves active pusher warps for tiny inputs. Next
policy should retain one-row distribution until every existing warp can receive
a full two-row batch (64 rows with8pushers×4warps); CTA counts stay fixed. This
policy is not implemented yet. Next comparison should include the validated
single-row overlap path so restoring small-case performance is measured against
that useful path as well as the batching candidate and required historical pull.

### Previous: prefetch validated

GPU7210118 COMPLETED0:0/18m16s; CPU7210138 COMPLETED0:0/47s. All11 validation/timing
cases and the candidate1024 capture completed. Ordinary/validation audit:352 exact
reference comparisons,32 exact-combine checks,48 exact replay checks; stderr empty.
Matched pipeline means at16/31/1024 (us): prior all-push55.0547/57.6787/401.1893,
prefetch54.9645/57.4186/399.4541, historical pull46.3977/47.1808/260.0608.
Prefetch versus prior all-push −0.1639/−0.4511/−0.4325%; versus required pull baseline
+18.4639/+21.6990/+53.6003%. Large-case control drift0.5131% and pull range1.0261%
exceed the small large-case improvement. No acceptance win or default promotion.
Candidate kernel-only repeats16=47.6579/47.4653,31=50.9050/50.8547,
1024=367.3965/368.3178us; do not subtract these from pipeline timings to infer a
fixed router cost, since replay/input-cache conditions differ.

Critical1024 samples: payload last92.800/91.616/91.936us; first compute100.544/
99.264/98.432us; compute spans234.048/236.512/238.976us; exposed combine25.824/
25.696/26.624us. Nsight ordinary kernel median366.480us. Earlier local-prefix trace
payload107.008–109.408us and first compute114.176–116.448us are from a different
allocation: consistent with useful prefetch overlap, not an isolated15us claim.
Compute span remains long and the all-pusher gate remains. Retain prefetch as the
working candidate for the next isolated change; no claim it solves dispatch.
Evidence: R/artifacts/push_dispatch/payload_prefetch/{comparison.json,audit.json,
iket/candidate1024/7210118/decomposition.json}; all raw artifacts remain remote.

GPU7210893 plan (completed above): Q3/Q35 correctness, same-allocation historical pull/prior-prefetch/
four-warp-gather pipeline comparison, matched prior-prefetch/gather kernel-only
ABBA16/31/1024, and paired critical1024 traces. CPU7210894 analyzes afterok7210893.
Canonical mirror stays at prefetch; candidate uses only the new main-kernel header
through an include overlay with the same native ABI/cache. All sources/caches and
overlay symlinks frozen through the consumer. Slurm rejected the first submission's
completed-job dependencies; verified both successes through sacct and no named job
in squeue before submitting without those stale dependencies. No duplicate GPU job.

### Previous: local-prefix validated, still slower than pull; lost overlap measured

GPU7209595 COMPLETED0:0/11m05s. All nine planned cases completed: Q3/Q35 edge,
Q35 boundaries15/33/65/1024, matched pull/candidate/candidate/pull timing, and two
candidate kernel-only16/31 repeats.288 exact reference comparisons,32 exact-combine
checks and32 exact post-replay checks; stderr empty. CPU7209594 previously completed
0:0/2m10s, corrected ordinary M16 zero spills and native/router preparation passed.

Matched router+MegaMoE ABBA means (pull→local-prefix candidate, us):
16=46.3714→55.0688(+18.7560%);31=47.2161→57.1815(+21.1060%);
1024=264.7199→400.0174(+51.1096%). Control first-to-last drift+0.0386/+0.0327/
−1.0790%; preserve the larger-case drift. Candidate kernel-only repeats:
16=47.8764/47.7971us;31=50.9799/50.7964us. These are separate timing scopes.
Earlier all-push kernel-only checkpoint16/31=48.9156/53.0309us is observational
across allocations; do not claim an isolated prefix gain from that subtraction.
The required historical baseline is still substantially faster. No promotion.
Evidence: R/artifacts/push_dispatch/local_prefix/{comparison.json,audit.json,
ordinary.log,compare-7209595.out}; original full log retains validation and all arms.

GPU7209768 COMPLETED0:0/7m58s; CPU7209769 COMPLETED0:0/49s, all three decompositions,
stderr empty.3 critical samples each,32timestamps/warp,10 ordinary Nsight calls. Candidate16
and candidate1024 capture kernel-only; historical pull1024 replays its producer.
Artifacts: local_prefix/iket/{candidate16,candidate1024,control1024}/7209768.

At1024 rows, candidate first compute114.176–116.448us follows payload-push
completion107.008–109.408us; indices finish68.096–68.736us. Thus the all-pusher
dependency exposes substantial bulk transfer instead of overlapping it with
compute. Pull first compute9.536/9.568us (first-sample31.776us retained), while
dispatch continues to75.264/78.368us (first107.680us). Pull metadata is outside
its kernel; these start times are diagnostic, not a matched total-latency speedup.
Candidate compute span235.104–238.208us versus pull162.976–163.552us also leaves
a separate gather/compute question. Same tile192x128x128,8experts/wave,150SM.
Ordinary Nsight MegaMoE medians382.064us candidate /205.968us pull exclude the
pull producer; use the matched pipeline ABBA above for acceptance.

At16 rows, median exposed dispatch15.584us, compute25.600us, exposed combine
3.520us; first sample contains late metadata98.656us and kernel envelope135.424us,
versus45.856/45.888us for the others. Retain the outlier, do not invent its cause.
Ordinary Nsight median47.424us. At1024 exposed combine25.312–27.104us candidate
and29.216–31.744us pull, contrary to a universal4–6us expectation. Combine stays
fixed. Instrumented envelopes are not ordinary timings or additive causal costs.
Next bounded step: validate the payload-prefetch draft and compare it with prior
all-push and true historical pull in the same allocation; it targets local-load
serialization, not the entire large-shape regression or the all-pusher barrier.

### Baseline establishment and implementation provenance

Previous note-only turn made no optimization progress; resumed from live source
and confirmed no jobs remained. CPU7209334 COMPLETED0:0/4m32s prepared the retained
pull native/router cache. GPU7209335 runs pull/current-all-push ABBA at16/31/1024,
EP8/FP4/150SM,200warmups/2000calls/batch100, exact input/output comparisons.
Remote root: R/artifacts/push_dispatch/pull_baseline. Baseline source/comm_alignment
has accepted async per-token publication, combine destination prefetch and last-CTA
cleanup; remote_reduce0/Lamport1. Timing is explicitly router+MegaMoE on both sides
because the pull router produces communication each replay. This supplementary
scope includes common gate/quantization and is not kernel-only acceptance proof.
GPU7209335 COMPLETED0:0/5m56s;24 exact reference records and72 exact comparisons,
stderr empty. Matched ABBA means (pull→current all-push, router+kernel us):
16=46.2588→56.0952(+21.2639%);31=47.1872→58.6913(+24.3797%);
1024=268.5735→402.5919(+49.9000%). Control repeat drift0.0864/0.0211/0.2690%.
Evidence: pull_baseline/{comparison.json,audit.json,contract.json,source_evidence}.
This establishes a substantial regression against the required historical path,
including at the large shape. No acceptance/promotion. Previous all-push cleanup
gains must not obscure this result. These are full route-producer+kernel times,
not pure kernel intervals or measured NVLink flight times.

Local unvalidated candidate removes the helper PrefixReady handoff: each exchange
acquires immutable source counts, computes ring prefix and padded expert base in
shared memory, then atomically allocates final gather rows from those local offsets.
Exchange CTA0 publishes SizesReady; refiners acquire it and cache totals while
exchange CTAs build indices, then acquire IndicesReady at consumption and derive
bases cumulatively. This preserves useful preparation overlap rather than merely
reducing the number of waits. Removes global pool-base storage and PrefixReady flag;
keeps participant/compute CTA counts and combine unchanged. Payload-ready fanout
retains the established TMA drain, system fence and parallel relaxed reductions;
the proposed lane-varying async substitution was withdrawn after SASS review below.
Tradeoff: duplicated small count reads versus removed global handoff
and per-route offset loads. Initial draft CPU7209390 COMPLETED0:0/4m01s: M16/M32
ordinary and M16 critical,128registers/24-byte stack/zero spills. Its artifact
`candidate/` is preserved. Final `candidate_overlap/` moves refiner count staging
before the index wait; CPU7209514 rebuilds this final revision plus native/router.
It depends on baseline7209335 and copies exactly the three staged source files
to the mirror only after that consumer completes. GPU7209515 depends on7209514:
Q3/Q35 edge and Q35 boundary validation, true pull/current-candidate pipeline
ABBA16/31/1024, then kernel-only candidate16/31 repeats. Artifacts/plans under
R/artifacts/push_dispatch/local_prefix. Lightweight capture awaits these results;
no performance/correctness claim for the new candidate yet.
Follow-up: GPU7209515 cancelled while PENDING (zero GPU runtime) to review the
async payload fanout. CuTeDSL's helper documents uniform operands. PTX specifies
per-thread async release semantics without that uniformity precondition; actual
nvcc SASS7209548 resolves the concern: the eight different lane addresses are
preserved by an ELECT/R2UR.BROADCAST/UREDGR/branch loop, one lane per iteration.
This is serialization, not a demonstrated correctness bug. Reverted only that
instruction substitution; retain single-issuer async metadata/local publications.
No claim that an async mnemonic itself improved overlap. Evidence remains remote:
local_prefix/{async_review.json,async_fanout.sass,async_fanout_context.txt}.
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async

CPU7209514 COMPLETED0:0/7m06s: final-overlap M16/M32/critical builds pass128registers/
24-byte stack/zero spills; native/router preparation passes. CPU7209594 depends on
it, copies the single corrected dispatch header from `candidate_fanout/`, then
builds ordinary M16 and prepares native/router caches. GPU7209595 depends on7209594
and repeats the full originally planned validation/ABBA/kernel-only sequence.
The corrected build and GPU validation have now completed; results are above.

Submitted lightweight trace7209768: candidate16 and candidate1024 kernel-only,
plus historical pull1024 with required producer replay;3critical samples each,
32timestamps/warp,10ordinary Nsight calls. The added large pull case distinguishes
payload delay from possible gather-compute slowdown instead of assuming all134us
regression is dispatch. `capture_cases.json` and `capture.py [JOB_ID]` drive capture
and CPU-only analysis with explicit source/cache/scope. Execution is gated on the
complete validation/timing job succeeding; analysis uses CPU7209769.

Payload-prefetch draft: local dispatch now stages SF+activation concurrently
under one transaction barrier, using two complete-token slots per warp. Prefetch
the next token while pushing the current one; refill only after TMA shared-source
read completion, leaving remote stores in flight. Empty routes advance the slot
phase with zero transaction bytes. Final remote drain/fence/readiness publication
is unchanged. This targets repeated local-load exposure; it does not remove the
global all-pusher payload dependency. No runtime correctness or speedup claim yet.
CPU7209991 COMPLETED0:0/1m11s: M16/M32 ordinary and M16 critical device builds pass,
128 registers,24-byte stack,zero spills. Only two changed headers are staged under
R/artifacts/push_dispatch/payload_prefetch/{before,candidate}; build commands,
contract and logs remain there. Trace7209768 finished before native preparation.
CPU7210072 depends on successful analysis7209769/device7209991, copies only those
two staged headers into the mirror, and prepares the native cache. GPU7210118
depends on7210072: Q3/Q35 edge and Q35 boundary validation, then same-allocation
reference/control/candidate/candidate/control/reference at16/31/1024 with
200warmups/2000calls/batch100. Reference is historical pull; control uses the
validated local-prefix dispatch header overlay with the same current native ABI;
candidate uses prefetch. All six ordinary arms measure router+MegaMoE. Two
candidate kernel-only repeats cover all three shapes. One candidate1024 critical
capture follows; CPU7210138 summarizes timing/checks and analyzes it afterok7210118.
This consumer completed; results are recorded above. Artifact-plan preparation
had one shell-quoting NameError before submission;
fixed without a duplicate GPU submission or GPU time.

Large-shape overlap hypothesis from current code: every SF refine group acquires
the all-rank/all-pusher PayloadReady counter before its first scale read. Thus no
compute block can be published until the last bulk payload pusher completes;
historical pull publishes each completed token/block incrementally. The large-case
trace must test how much dispatch is exposed by this global dependency, separately
from gather-compute throughput. If bulk completion dominates, prefetching local
loads alone may be insufficient: investigate bounded progressive payload readiness
with in-flight later transfers, proving remote completion and counter lifetimes.
Do not add per-chunk full drains that merely replace one large serialization with
many network round trips. No implementation or measured attribution yet.

Separate local gather draft (remote mirror remains prefetch for GPU7210118): saved
SASS local_prefix/async_fanout.sass shows ELECT/R2UR.BROADCAST/UTMALDG.GATHER4/
BRA.U.ANY serializing per-lane gather operands. CuTeDSL Gen-specialized mainloop
explicitly uses four gather warps to bound uniform operands across the K loop.
Reuse the four existing dispatch warps after generation for FC1 gathers, retaining
row IDs across K tiles. TMA-A acquires a free stage and publishes expected bytes;
a160-thread local join releases its four gather warps. FC2 advances their stage
state without producing data; next FC1 join prevents phase overtaking. Keep
DispatchDone retirement, block readiness and final cleanup ordering. CTA count,
register budgets, workspace, MMA and Lamport combine unchanged. Removed duplicate
count fetch for helpers. This is a throughput hypothesis, not attribution of the
entire measured compute gap. CPU7210413 COMPLETED0:0/1m05s: M16/M32/M192 ordinary
and M16 critical pass128registers/24-byte stack/zero spills. One-header overlay;
SASS shows elected uniform gather issue without the old per-lane election loop.
Reachable ordinary dispatch regions peak R39 within48, TMA/MMA regions R18 within40,
no local memory instructions/unresolved transfers in that review. GPU7210893 now
tests this draft through an include overlay after prefetch validation/analysis.
Evidence/plans: R/artifacts/push_dispatch/gather_warps.

Marker consequence: compact MegaDispatchEnd now comes from generation participants,
before GEMM initialization, since the old dispatch warps now also gather throughout
compute. Their lifetime is no longer a dispatch-work duration. Compare first
compute time and Genphase markers across versions; do not claim a dispatch win
from this marker relocation or silently compare its changed envelope. Ordinary
timing remains the performance decision metric.

Next isolated metadata draft, not in GPU7210893: use the histogram atomic's return
as the route's within-expert offset. Its exchange CTA overwrites only valid local
entries in its received top-k plane with `-2-(offset*local_experts+expert)`; invalid
`-1` and nonnegative unowned entries stay unchanged. Scatter decodes this entry
and adds the immutable source prefix, removing its second shared atomic and
reclassification. The sender has already completed that double-buffered plane;
only this exchange consumes it. Existing publication/retirement ordering remains,
and a compile-time bound prevents signed overflow. Tradeoff: one extra global
store per owned route versus the redundant allocation. No new workspace or token
copy, no numerical shortcut. Local dispatch and artifact overlay only; CPU7210973
COMPLETED0:0/4m15s with frozen gather main header: M16/M32/M192 ordinary + M16
critical128registers/24-byte stack/zero spills; ordinary inspected regions retain
maxR39/48 dispatch and R18/40 TMA/MMA, no local memory/unresolved branches. GPU plan
is prepared but deferred: gather traces show payload, not indices, delaying first
compute. Counted-offset logic removed from the local draft for the isolated payload
experiment; its remote candidate remains preserved and unsubmitted.
Artifacts: R/artifacts/push_dispatch/counted_offsets.

Added critical-only TopkStaged/TopkSent markers to that next draft to distinguish
local input dependency/staging, local TMA completion/notification issue, and the
following peer wait. They do not measure pure NVLink flight time. The maintained
analyzer accepts the complete optional pair, requires EP-sized counts, and rejects
a partial pair. CPU7211141 COMPLETED0:0/52s: updated M16 critical variant passes
128registers/24-byte stack/zero spills. Retain the two markers for the next payload
experiment. Completed gather capture/analyzer stay frozen.

Payload follow-up (now local draft): validated prefetch refills the just-pushed slot
after `wait_group.read 0`. With two slots, consider refilling the previous slot
after pushing the next slot and waiting for all but the current store group's
reads. That could leave the current TMA reads in flight while reloading the other
slot. Empty destination masks need `read 0` when no current group was committed;
never treat logical token rounds as committed groups. Prove phase/reuse and empty
routes before testing. Do not mix this change into the pending gather/offset tests.

GPU7208884 COMPLETED0:0/11m58s. All four ordinary arms plus both critical
captures finished. CPU7208885 COMPLETED0:0/44s analyzed both three-sample captures; stderr empty.
Remote evidence: R/artifacts/push_dispatch/concise_push/comparison.json and
iket/rows{16,31}/7208884/decomposition.json. No raw artifacts downloaded.
Matched all-push-only ABBA averages (control→candidate, us):16=49.5069→48.9156
(−1.1945%);31=52.6572→53.0309(+0.7097%);1024=396.8015→383.5368(−3.3429%).
96 exact references and96 replay checks. These are **not wins over the acceptance
baseline**. Both use kernel-only metadata+payload+compute+Lamport combine, same
host ABI/inputs and150SMs; control is immediately preceding direct scatter.

Critical IKET observed median exposed dispatch17.056us at16rows and19.872us at31;
compute spans24.736/26.752us; exposed combine4.768/3.584us. Combine sample ranges
4.416–5.024/3.584–4.832us. No samples discarded, only3samples/shape. Instrumented
medians are not additive ordinary timing or precision claims. Kernel envelopes
47.520/51.296us; Nsight31 median52.784us. Actual trace confirms one MegaMoE per
measured kernel-only replay and no extra kernels. Combine stays fixed.

Concrete dispatch issue: outgoing payload done by~9.3–9.8us. At16rows, last
metadata receive~8.1–8.6us, sizes10.1–10.5, indices12.7–13.3, SF-refine start14.0–14.5,
all dispatch roles end~17.1–17.3us. At31rows two samples show last receive9.4–9.5,
sizes11.5–11.6, indices15.6–15.9, refine16.5–17.0, end20.2–20.7; first sample has
later metadata and is retained. Thus removing a network leg alone has not removed
the count/helper-prefix/index/SF handoff costs. These timelines support investigating
local metadata/index dependencies, not another combine optimization.

**Historical reference:** source/comm_alignment with payload_mode=pull,
Lamport1/remote_reduce0 is the verified retained baseline used by7209335. Metadata is
sent by gate_topk's prepared-route branch before MegaMoE; naive kernel-only replay
omits that cost and cannot replay without republishing metadata. Need honest
reported timing boundaries: retain actual best baseline, show full communication
path and kernel/dispatch decomposition, do not silently weaken or rewrite baseline.
Existing pull evidence is under artifacts/push_dispatch/timing (control, rows31/
15/7/19/13/3) and artifacts/comm_alignment/combine_pointer_ep8/qwen3 (candidate).
Do not mistake historical compact-push controls for the requested pull baseline.

The local-offset experiment motivated by this trace is implemented and awaiting
GPU validation as described above. Do not reapply the removed PrefixReady handoff
or treat its standalone compilation as runtime/performance evidence.

## Active: selective recovery and concise push — September 17 UTC

User restored local tracked code and removed scratch history; remote artifacts are
authoritative. Reviewed7208427:32 exact saved-reference and32 exact replay checks
pass, rows16=49.1665/49.2026us and31=52.2304/52.2951us. Versus direct-SF,
16rows improves~1%,31rows regresses~1%; no matched ABBA claim.
Recovered only dispatch/host/layout/selector, critical-IKET support and kernel-only
replay testing. Excluded remote BF16 atomic-reduction ABI/code/tests; preserve
Lamport push combine. Removed obsolete identity expert-order metadata/scheduler
lookup. Payload warps now classify destination masks once per token, then send
SF and activation with that mask; descriptor prefetch uses a compute warp so
routing starts immediately. Retain necessary payload release/acquire, index/SF
readiness and final cleanup lifetime ordering. CuTeDSL unchanged.
Remote run R/artifacts/push_dispatch/concise_push preserves before/candidate code
and plans; selected local files synced into source/local-mirror after no live jobs.
CPU7208659 COMPLETED0:0/3m32s: M16/M32 ordinary and M16 critical all128regs,
24-byte stack/zero spills; native and selector prepared on CPU. GPU7208665
COMPLETED0:0/8m07s: Q3/Q35 edge/padding and Q35 15/33/65/1024 boundary pass;
208 saved-reference comparisons,80 exact-combine and80 exact replay checks.
Two kernel-only max-rank500-call repeats us at16/31/1024:
49.0730/52.4684/389.5303 and49.1221/52.6895/383.9480. Small-case31 slightly
slower versus prior observational run;1024 faster, allocation confounding unresolved.
GPU7208884 runs same-allocation control/candidate/candidate/control at16/31/1024,
then3 critical IKET samples each at16/31 (32timestamps/warp). Control uses four
preserved before-device headers via a symlink include overlay; common current host
ABI, local-only selector, Python, runtime cache, inputs and Lamport combine. No
source clone or mutable-source swapping. CPU7208885 afterok analyzes traces.
Source stays frozen through this chain. Local/remote heuristics and kernel runtime
have no source differences; no extra recovery needed. Analyzer now checks the
kernel-only timing contract and rejects extra Nsight kernel launches. Raw traces,
results/plans remain remote; scratch/megamoe still contains only requested HTMLs.
Goal active; no promotion/commit. Need matched timing and actual decomposition
before deciding retention or further structural changes.

## Local restore and scratch cleanup — September 16, latest user direction

User intentionally git-restored local maintained code; remote history remains.
Recover only necessary code, not the accumulated remote tree or result artifacts.
Latest explicit cleanup keeps only eight complete decomposition HTML reports at
existing paths under local `scratch/megamoe`; all other contents are removed.
Historical local scratch links below are no longer available; use remote artifacts.
No remote sources/results were changed by cleanup. Fresh scheduler check found
CPU7208426 and GPU7208427 COMPLETED0:0 (7m04s/3m30s); their result contents still
need review before accepting direct-index scatter. Kernel optimization remains open.

## Active: structural dispatch optimization resumed — September 17

User explicitly resumed kernel implementation: remove unnecessary sync, keep push,
follow Rubin specialized metadata send/receive and index/scale/weight preparation;
lightweight tests first, no CTA tuning, one MegaMoE kernel, CuTeDSL untouched.
Prior stopped campaigns below are historical; this is fresh authorization.
Current single-kernel baseline CPU7207737/GPU7207767 passed Q3/Q35 correctness:
112 exact-combine and240 reference checks; two kernel-only max-rank means at
16/31/1024 rows51.1997/53.3080/400.6354us. Trace6calls=6MegaMoE/zero other kernels.
GPU7207994 completed2m57s with56 exact post-replay and56 saved-reference checks
across seven Q3/Q35 shapes. Baseline report now TIMING_SCOPE_AND_REPLAY_PASS.
Router/quantization outside timing; all metadata/payload transfers, index/SF,
compute and push combine inside. Retained source/comm_alignment kernel-only is
not a matching control because it sends communication from the router.

First resumed attempt `async_readiness`: Rubin-style async release readiness
reductions replace synchronous atomics and redundant generic-proxy fences;
receiver histogram initializes after local send drain but before remote wait;
refinement caches metadata/initializes barriers before acquiring payload arrival.
Retain lane-fanout payload system fence and all required subgroup barriers.
CPU7208108 device-only M16/M32 and critical build; GPU7208113 afterok lightweight
16/31 exact replay/reference + two500-call repeats,50warmups,graphbatch100.
CPU7208108 completed1m,128 registers/zero spills. GPU7208113 short repeats:
16rows49.8369/49.7171us,31rows51.9206/51.8923us;32 exact replay and32 reference
checks pass. About2.6–2.8% below prior baseline, not yet matched ABBA.
Header timestamp still triggered a native rebuild on the GPU job; future CPU
preparation must run ensure_deepgemm_built in candidate_runtime directly.
Baseline dispatch saved in async_readiness. Next direct_scale_refine builds
indices/weights before payload acquire, writes only valid canonical SF entries,
and removes padded128-row TMA transpose/staging barriers. Preserve proxy fence
between generic SF writes and TMA consumption and all incoming-push retirement
ordering, including empty ranks. CPU7208184 completed4m27s including native runtime build, all three device
specializations128 registers/zero spills. GPU7208185 first16/31 short repeat
49.6488/51.7595us with exact replay/reference pass; second repeat pending at
last read. Follow-up7208286 measures1024 (same500-call timing) and validates
Q35 rows15/33/65/1024 with short replay against saved outputs. Need large-shape
evidence before retaining direct SF writes. Prepared, not applied, push_mask_reuse:
classify destination masks during SF send and reuse for activation, and let a
non-dispatch warp prefetch descriptors so metadata send starts earlier.

Latest completed: direct_scale_refine GPU7208185 completed3m08s; two16/31
repeats49.6488/51.7595us and49.6914/51.7302us,32 exact replay+32 reference checks.
Follow-up7208286 passed40 replay+40 reference checks across Q3-1024/Q35-15/33/65/1024.
Q3-1024 ordinary400.4065us versus prior400.6354us: approximately unchanged;
Q35 short timings are correctness-only. Keep direct SF path pending final validation.

User clarifies FC1 gather should eliminate sorting. New `direct_index_scatter`
removes the expert-load sort and intermediate sorted-route buffer. Exchange CTAs
histogram routes, acquire helper prefixes, then directly scatter final gather
indices/combine destinations/weights while payload pushes proceed. Natural expert
order; one IndicesReady release/acquire makes completed index lists visible to SF
refinement. No per-row source-prefix scan or sorted-route readback. Marker renamed
GenphaseIndexReady; flag allocation/cleanup follows NumGenphaseFlags. All compute
CTA counts and push combine unchanged. CPU7208426/GPU7208427 lightweight16/31 first; broad correctness
and lightweight IKET follow if it passes. Device compilation passed zero spills;
native/selector preparation still running on CPU, GPU dependency-gated.
Mask reuse remains an unapplied draft, rebased on direct scatter, also removing
now-redundant expert-order lookup/metadata and overlapping descriptor prefetch.

## BLOCKED / STOPPED — September 17, final structural comparison

Per user's stop rule, no additional tuning or profiling after repeated regressions.
GPU7207353 terminal CANCELLED0:0/9m50s. Q3 and Q35 edge/boundary checks passed;
first ordinary control/candidate pair (rows16/31/1024,router+kernel,max rank):
46.4939/48.3118/368.5628us versus57.1487/58.0219/385.2604us.
Remaining candidate/control repeats cancelled; not a complete ABBA. Whole completed
log contains104 exact-combine checks,208 exact-result comparisons,24 records.
Completed arms: candidate_validation, candidate_boundary, candidate_q35_validation,
candidate_q35_boundary, control_a, candidate_a. Remote artifact
artifacts/push_dispatch/single_kernel_timing/comparison.json preserves evidence.
Early sends + fused local index/SF preparation + removing the extra dispatch kernel
still do not beat retained compact control. No candidate promotion. Keep the code
and results for review, default pull and faster source/comm_alignment preserved.
CPU7207343 passed0:0/2m46s; native/selector/runtime seed and all three device builds
passed,128 registers,24-byte stack,zero spills. Earlier pending GPU7207285 cancelled
with zero GPU time. No task allocations remain. CuTeDSL git status clean; untouched.
Do not resume experiments without explicit user direction. No new lightweight IKET:
ordinary timing already fails the user's continuation criterion; prior traces retained.

## Implemented: restore one kernel for the original MegaMoE region

User questioned the extra dispatch kernel. It was a port implementation choice,
not a Genphase algorithm requirement, and performance has not justified it.
Cancelled pending GPU7207285 before allocation (zero GPU time). CPU7207279
completed0:0/2m52s; all early-push ordinary/critical device/native/selector builds
passed. No GPU validation of that intermediate version.
Current candidate retains immediate producer route/payload pushes, but integrates
local sort/prefix/index/SF work into the existing MegaMoE kernel. The first128
threads of the original participating CTAs run dispatch before their GEMM SMEM
initialization; other CTAs initialize concurrently. Subgroup barriers1/2/3 stay
separate from the caller CTA join; TMA barriers are invalidated before SMEM reuse.
No standalone dispatch launch, extra CTA grid or full-selector dependency wait.
150 compute CTAs remain fixed; participant count retains the previous role budget.
Host asserts scratch fits existing SMEM and participant count fits the grid.
IKET Genphase markers now live inside MegaMoE. Selector remains a separate router
kernel, so full pipeline is gate/quant → selector/early push → single MegaMoE.
CPU7207343 single_kernel_cpu completed0:0/2m46s: M16/M32 ordinary and M16
critical passed with128 registers,24-byte stack,zero spills; native/selector/seed
passed. GPU7207353 running: Q3 edge and33/65/1024 correctness passed; Q35
checks and ordinary ABBA16/31/1024 pending. No GPU correctness/performance claim yet.

## Intermediate: initiate communication in the selector

Latest steering: initiate communication as early as possible; keep CTA counts fixed.
CPU7207107 completed0:0/6m08s for intermediate fused-index candidate; its prepared
GPU test was never submitted. Superseded locally before spending GPU time.
New candidate pushes top-k/score inbox rows immediately when each token completes
selection, plus one FP8/SF copy per destination. Specialized selectors cooperate
on one shared stage; generic selectors use independent warp stages. The existing
last-CTA acquire/release ticket publishes one source completion covering both
planes. All-invalid rows still publish metadata/completion correctly.
The dependent communication kernel only sorts, builds prefixes and refines SF /
final indices. Removed its metadata-send pass, payload pass and PayloadReady
all-peer counter/wait. PrefixReady transitively acquires source completion.
Compute150 and total communication CTA count unchanged. CuTeDSL git status clean.
CPU7207279 builds early_push_cpu; early_push_timing reuses Q3/Q35 correctness
plus ordinary16/31/1024 ABBA. No GPU evidence yet; preserve faster compact control.
Current IKET schema drops PayloadPushed (traffic now belongs to producer).

## Intermediate: structural index/SF fusion

Latest user instruction: **do not tune CTA counts; focus on structural improvement**.
7206958 cancelled0:0/6m14s after control150 and candidate150/142/134/134-repeat.
Ordinary us by rows16/31/1024: control47.733/48.394/370.745;
candidate15057.064/60.220/416.778;14258.389/61.323/434.544;
13458.298/61.189/460.554;134repeat58.318/61.133/460.697.
120 exact comparisons/120 exact-combine checks; no complete ABBA claim.
Lowering compute budget did not help; no CTA tuning adopted. No task jobs remained
when source sync began. User steered directly to structural work; goal remains active.

Implemented exchange sort → helper prefixes → fused index/SF refinement:
- Exchange writes source-local sorted route ordinals into the existing unused
  legacy source-index region, publishes counts, and exits. No per-route register
  arrays or wait for helper prefixes; no later exchange scatter pass.
- Helper retains load ordering, early SizesReady and ring prefixes.
- Refiners stage counts/expert IDs/bases/source prefixes once per CTA. Each valid
  row resolves its source segment, loads the sorted route, writes final gather
  index/score/combine destination, and issues its full-row SF TMA load. The existing
  per-block L1 readiness release follows these writes and canonical SF completion.
- Removes MetadataReady counter and associated all-exchange completion wait;
  refiners acquire PrefixReady instead. Five flags, enum-sized allocation/cleanup.
- IKET GenphaseSortReady replaces GenphaseMetadataReady; the frozen old workflows
  retain the earlier marker schema. Current analyzer requires the new marker.
- Compute150/pushers8 and Lamport combine unchanged. No new workspace allocation:
  scratch was already reserved; shared refinement header includes source prefixes.

Saved before-files under fused_index_refine. Synced only changed source/analyzer
files after old job termination. CPU7207107 submitted for six ordinary/critical
specializations, native/selector rebuild and runtime-cache copy. Prepared
fused_index_timing: same Q3/Q35 edge/boundaries and three-case ordinary ABBA;
not yet submitted. This structural candidate has no GPU correctness/perf evidence yet.

## Rejected: GB200 resource-budget test

User continued the goal after clarifying GB200-specific execution tuning. Fresh
blocked audit; preserve the previous stop and its evidence below. No broad search
or further variants after this bounded test unless it supports a useful gain.
Read-only query of saved7206028 Nsight TARGET_INFO_GPU confirms NVIDIA GB200,
152 SMs (all four recorded devices). Current150 compute CTAs in2-CTA clusters
leave only2 SMs for1 helper+8 resident pushers; delayed compute cluster starts
in the saved IKET trace are consistent with contention.
Prepared gb200_sm_budget: unchanged source/native caches, candidate compute
budgets150/142/134, symmetric eight-arm order around control150, rows16/31/1024,
same200warmups/2000calls/batch100/max-rank router+kernel.142 reserves10 SMs;
134 reserves18 as a conservative allowance for cluster fragmentation. This tests
the resource hypothesis without introducing another kernel change.72 read-only
reference aliases point to completed7206476 control_a tensor files; only filename
SM labels differ. Exact inputs/outputs/combine checks remain enabled. CuTeDSL clean.
Stop additional optimization if no tested candidate beats the retained baseline;
only capture new lightweight IKET if ordinary timing supports a useful change.
GPU7206958 submitted for this eight-arm comparison,15min limit. Both source
trees stay frozen. No other new jobs or changes to CuTeDSL.

**BLOCKED / STOPPED September17 UTC, per user instruction.** Repeated genphase
variants remain slower than retained pre-genphase compact dispatch. Do not resume
optimization or submit further jobs without explicit user direction. Latest
TMA7206476 first ordinary control/candidate pair:16rows46.312/56.952us,
31rows48.233/59.939us,1024rows370.192/418.096us. All Q3/Q35 correctness arms
passed; remaining ABBA repeats cancelled, not completed. Initial port and helper
results below establish earlier repeated regressions. No baseline performance
win or default promotion. Keep experimental code/results for review; CuTeDSL
stays clean and untouched. Existing push combine and default pull remain.

User clarification after stopping: borrow Rubin GenphaseTokenComm's algorithm,
but optimize implementation details for **GB200**. Preserve deduplicated all-push,
local index construction and gather loads, plus FastAFD push combine. Rubin CTA
counts, role partitioning, tile sizes, staging depth and launch boundaries are
not requirements. The reference itself says its27 pushers follow Rubin's compute
cluster footprint and must be rederived for Blackwell. Our current up-to8 count
depends on token capacity, not the concurrent compute cluster resource budget;
that is a concrete GB200 tuning issue to address before further SF micro-tuning.
Also evaluate launch/synchronization overhead for small batches and SF staging
cost. This clarification is recorded; experiments remain stopped/goal blocked.

Goal: implement that all-push/index/gather design efficiently on GB200; keep
existing push combine; remove unnecessary synchronization and redundancy;
validate with ordinary timing and lightweight IKET.
Read [cluster policy](CODEX_CLUSTERS.md). No subagents or remote-result downloads.

Reference resolved by user's Rubin clarification: local sibling
`../cutedsl_megamoe` is clean; `git status` reports main up to date with origin/main,
nothing to commit. Keep that repo untouched. Earlier inspection missed the
generation-specialized Rubin implementation; no alternate revision is needed.
Standard Blackwell and Rubin mega kernels both instantiate
`communication/nvlink_domain/token_comm.py::TokenCommNonDeterministic` (metadata
push followed by activation pull). The intended reference is
`next/sources/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py`, used
by `block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py`:
exchange CTAs send top-k planes concurrently with deduplicated payload pushers;
each exchange CTA locally counting-sorts its received peer's routes, a helper
folds counts/prefixes, and local scatter writes gather indices/source metadata.
SF refinement is local; FC1 reads source-indexed activations via gather4 in
`block_scaled_swap_ab_fc12_mainloop_gen_specialized.py`. SF and activation passes
share one completion publication (no early-SF readiness claim). Adapt this
dispatch protocol to FastAFD while retaining its existing push combine.
Current compact_push cleanup is not yet this reference port.

Existing FastAFD compact_push already pushes one activation/SF/weight frame per
source token/destination, allocates dense expert descriptors remotely, and uses
FC1 gather4 loads. Default is still pull. No default promotion yet.

Candidate device-only cleanup (local checkout and remote source/local-mirror):
- Skip initial dispatch/epilogue rendezvous when both route-ready dispatch and
  rank-gated combine remove the NVLink barriers it orders. Keep cleanup join.
- Skip dispatch-only join already covered by rank-ready CTA synchronization.
- Dense routes overwrite all used shared counts: skip redundant initialization;
  omit unused expert-send/per-rank count clearing. One lane clears expert total.
- Keep final epilogue grid join: protects L1/L2 flags and combine metadata from
  other CTAs' cleanup. Removing it requires a replacement lifetime guarantee.

Initial baseline source/comm_alignment was frozen for GPU7204827. Both had no live consumers at
start. Remote header diff before the later count cleanup confirmed only this
turn's rendezvous change. Local baseline header and full dirty patch retained in
`scratch/megamoe/push_dispatch`; remote artifacts `FastAFD/artifacts/push_dispatch`.
Do not edit either source while CPU/GPU consumers are live.

CPU7204755 FAILED1:0/48s: command generator replaced `-o` instead of its operand.
First upload also lacked parent directory; corrected before job read the plan.
No GPU used. Preserve failed plan. Corrected CPU7204805 compiles compact-push
M16/M32 ordinary and critical IKET: PASS0:0/61s, all four zero spills. No host ABI change; existing
lightweight_iket native donor copied into three isolated caches; compatible
four-module producer TVM cache linked. No new native build on GPUs.

Prepared timing plan `scratch/megamoe/push_dispatch/timing`: fresh random EP8
rows31/15/7/19/13/3, seed5894115885702016329. Symmetric six-arm order:
pull control, original compact reference, cleaned compact candidate, then reverse.
All use Lamport push combine, rank-interleaved balanced routes, same inputs;
200warmups/2000calls/graphbatch100, max-rank producer+kernel CUDA events.
Control first records exact inputs/output; all others compare, with exact combine
checks. GPU7204827 COMPLETED0:0/9m04s on nvl72098-T[07,09]. All36 case/arm
records,48 input records+240 cross-source comparisons+288 exact-combine checks
pass; stderr empty. Remote timing/comparison.json keeps both repeats/all values.
Cleanup versus original compact: rows7/13/15/19 save0.634/0.650/0.758/0.840us;
worst cross-repeat savings0.606/0.643/0.738/0.802us. Rows3 improves every pairing
but controls drift (pooled1.214us, worst0.647us). Rows31 is inconclusive:
original compact54.577/49.791us, cleaned51.160/49.082us; pooled gain cannot be
trusted. Cleaned compact versus original pull improves five cases consistently;
rows31 pooled regression0.890us with mixed signs. No observations removed.
This validates useful initial cleanup, not exact reference matching or a uniform
all-push benefit. Raw remote results stay remote.

## Active: remove the in-band epilogue grid wait

Local candidate now uses last-CTA cleanup for dense routes/rank-gated/in-band
combine. Each CTA joins its local dispatch and epilogue, then publishes an
acquire-release ticket. Only the last CTA cleans all expert flags/counts using
one warp per expert, then advances the existing generation. Other epilogues
start unchanged Lamport combine without a grid wait. Dispatch's completed
mbarrier is invalidated before reusing its word for the CTA-local ticket result.
In-band masks move past expert counts so late idle scheduling warps can still
read counts. Shared lifetime, double-buffer protocol, combine writes and math
are preserved; legacy epoch/reduction paths retain their grid synchronization.

CPU7204961 PASS0:0/69s: M16/M32 compact ordinary+critical and pull ordinary,
all6 zero spills. Build uses captured candidate.cuh, leaving the then-live
ordinary source untouched. After7204827 ended, source/comm_alignment advanced
to the validated initial cleanup (baseline/pre_last_cta_cleanup.cuh);
source/local-mirror received last-CTA candidate. Original header retained as
artifact, not an extra maintained source tree. No source edits while consumed.

GPU7204979 FAILED1:0/1m38s in the baseline validation harness: its rows3
legacy fixed24 benchmark calls an unprepared kernel on an in-band/compact
buffer and hits the host protocol assertion. Parent's normal/zero/padding
records were saved, but no complete arm or candidate ran. No kernel conclusion.
Recovery7205037 submitted, artifact last_cta_timing_fixed: same code/cases/caches,
new reference/output directory, validation rows4 avoids the incompatible fixed24
legacy sub-benchmark while exercising normal/alternate/zero/padded/empty-rank
routing. First two arms compare parent/candidate; then compact-only ABBA on fresh rows16/17/28/15/31/27,
seed8082405836389244434. Same200warmups/2000calls/batch100 and exact-combine
checks; all inputs recorded by parent then compared. Two isolated native caches
seeded from same compatible host donor, producer caches linked.

Lightweight phase plan prepared but unsubmitted, `lightweight/qwen3`: SAME six
last-CTA cases, compact fixed, four symmetric arms;30ordinary+5criticalIKET
samples/arm/case after30warmups, timestamp reservation32. Source snapshots and
payload/combine contracts enforced. Full EP8 batch1..61 manifest has135 cases;
only selected6 are authorized, so full audit.complete=false is expected while
selected6/6 must pass. CPU analysis plan collects/audits traces and compares
ordinary producer/kernel spans. Need selected exact-combine and additive phase
attribution audit after capture. No files downloaded.

Next: validate7205037, implement the identified Rubin generation-specialized
dispatch, then run ordinary timing/lightweight attribution and decide defaults.
Do not spend an extra GPU attribution run on an interim implementation unless
needed to isolate a correctness or performance issue. No commit or default
promotion. Latest7205037 poll RUNNING5m55s; do not edit its remote sources.

## Genphase port in progress

GPU7205037 COMPLETED0:0/8m55s, all6 arms status ok, stderr empty;
112 recorded-input checks,208 comparisons,192 exact-combine checks. Ordinary
control/candidate/candidate/control us by rows:15=47.265/50.315/46.358/47.258;
16=48.634/46.340/47.834/47.219;17=47.218/46.352/46.344/51.481;
27=48.394/47.571/47.607/48.378;28=48.478/47.699/47.782/48.482;
31=49.033/48.330/48.297/49.002. Four cases improve all cross-repeat pairings;
15/16 mixed due drift. Full records in remote last_cta_timing_fixed/comparison.json.

Local candidate now implements separate `impls/mega_moe_dispatch.cuh`: exchange
CTAs push int32 top-k/FP32 scores and locally counting-sort the received peer;
helper folds counts/prefixes; pushers send separate plain SF/activation planes
with two staging slots and one completion publication; helper/pushers refine SF
locally; FC1 uses a flat gather-index pool. Selector compact mode now only selects.
Native launch releases the dependent MMA kernel via PDL. Existing Lamport combine
and last-CTA cleanup remain. Cleanup retires the new generation-banked flags only
after every communication role and local consumer is done. No remote per-route
allocation atomics in the new compact dispatch. Default remains pull pending tests.

Reference adaptations: capacity remains FastAFD's graph bucket (often3840) while
active genphase rows are limited to1024, like the reference. Top-k publication
carries the active row count, allowing unequal rank counts without transmitting
inactive capacity. Payload pusher count depends on shared bucket capacity, so
all ranks publish the same number even when active counts differ. SF padding
matches FastAFD MMA tiles. The helper now ranks experts by descending load with
stable expert-ID ties, and the scheduler translates slot indices to weight expert
indices. The obsolete compact send-counter region holds the banked permutation.
Source segments use the reference's ring order starting at the local rank.

Saved exact pre-port files under baseline/pre_genphase locally and remotely.
After7205037 ended, source/comm_alignment advanced to last-CTA baseline;
source/local-mirror received the port. CPU7205256 FAILED1:0/56s: M16 compute
compiled with zero spills, new communication PTX rejected mbarrier-init fence
scope `.cta`; corrected to `.cluster`. CPU7205285 is the corrected build, includes
dispatch M16/M32/Q35, compute M32, new native ABI and TVM selector. Its temporary
compiler files are explicitly directed into the artifact workspace.
CPU7205285 COMPLETED0:0/3m01s: four remaining device specializations zero spills;
new host extension and TVM selector built. CPU7205346 COMPLETED0:0/1m07s:
descending-load order plus light Genphase markers, all8 ordinary/critical builds
zero spills (dispatch202regs Q3,226 Q35; main128regs). Captured-header ring-order
CPU7205490 COMPLETED0:0/51s/zero spills without editing then-live remote source.

GPU7205365 COMPLETED0:0/3m45s, genphase_smoke: both candidate arms status ok,
stderr empty,112 exact-input/output comparisons and48 exact-combine checks.
Normal/alternate/padded/empty cases and six balanced inputs pass. Small50-call,
graphbatch1 diagnostic times are not comparable with the2000-call/batch100
baseline. Ring-order fix was local during this run; synced only after it ended.

GPU7205504 submitted for genphase_timing: repeat edge tests, boundary rows33/65/
1024 to exercise two-stage reuse and the row limit, then same six EP8 ABBA timing
cases. Candidate includes ring order; control is pre-genphase last-CTA compact.
New exact-input references are captured by control. Protocol host binary/cache
is reused only across device-only changes, with source/ABI correspondence recorded
in genphase/cache/candidate_runtime/seed.json. Freeze both sources until terminal.

Lightweight plan remains UNSUBMITTED; candidate caches now genphase, control
caches last-CTA, references genphase_timing/reference. Local workflow/analyzers
now include separate communication-kernel markers and ordinary kernel intervals.
Marker count checks cover exchange/helper/pusher/refine roles; negative relative
times denote communication before MMA launch. No full-dispatch claim from the
main kernel's dispatch-warp wait alone. Upload the refreshed workflow/plan before
submission. Need timing/boundary results, Q35 runtime coverage, light traces and
measured optimization/default choice. No performance or completion claim yet.

### September 16 continuation: boundary and trace pipeline

GPU7205504 FAILED1:0/4m18s after both ring-order edge-validation arms passed.
Boundary kernels never launched: per-arm `MEGAMOE_ABBA_ROWS=33/65/1024` bypassed
the launcher's earlier global slash-to-comma normalization; test parser requires
commas. Recovery7205570 (`genphase_timing_fixed`) uses33,65,1024 and reruns only
unfinished control/candidate boundary arms plus four timing arms. Same source,
caches and six timing inputs; new control references under its own reference/.

Lightweight7205538 was automatically CANCELLED with zero runtime on failed
dependency; retained job/submit/submission records carry its ID. Replacement
7205626 depends afterok:7205570, same light24-capture plan. All plans/workflows
uploaded successfully before it can start. Candidate manifest now requires
`genphase_dispatch=true`, control false; both require exact combine. Audit checks
the actual contract and per-sample Genphase marker presence, in addition to
existing input/phase/sample checks. No code edits to frozen remote sources.

Q35 EP8 validation plan `genphase_q35` is UNSUBMITTED:512 experts/topk10,
intermediate1024/hidden4096; paired edge checks plus fixed rows15/33/65/1024,
2warmups/6calls. It checks the second router specialization, partial wire-plane
padding and staging reuse. Submit after the current paired experiment unless
its evidence requires a code correction first.

### Reference helper and ordinary timing result

GPU7205570 COMPLETED0:0/8m51s: all6 arms status ok, stderr empty,240 exact-combine
checks,168 exact-input/output comparisons,72 recorded inputs. Boundary33/65/1024
PASS. Full ordinary ABBA control/candidate/candidate/control us by rows:
15=46.290/68.128/69.206/51.141;16=47.743/67.961/67.925/46.336;
17=46.290/69.175/69.118/46.293;27=47.778/72.818/72.716/47.704;
28=47.710/70.414/70.942/47.726;31=48.392/71.175/74.382/48.354.
Genphase regresses every case: no promotion. Remote comparison.json retains
the compact summary; boundary short-run times are diagnostic only.

Reference review found the helper's exact quadratic global-memory ranking was
not the reference implementation. Local candidate now uses warp shuffle ranking
for<=32 local experts and the reference descending-binade histogram above32;
counts/order stay in shared memory, source ring prefixes follow SizesReady, and
sort workers stage source bases once. Uniform early exits skip inactive rounds
of the capacity-sized unrolled sorting loops. CPU7205813 compiles captured
M16/Q35 ordinary and M16 critical headers; remote live source remains frozen.
Host shared-memory calculation includes the helper's full overlay. These local
edits are not yet GPU validated or synced into source/local-mirror.

Lightweight7205626 remains the frozen pre-helper genphase/control comparison;
CPU analysis7205814 submitted afterok:7205626. Single-case trace input RNG matches
multirow ordinary references: smoke main resets the same per-rank seed for each
row. Q35 runtime stays unsubmitted until the correction is ready. CuTeDSL sibling
again verified with full git status: clean main, up to date with origin/main.

CPU7205813 COMPLETED0:0/52s: M16 ordinary/critical use152 registers/zero stack,
but the topk10 build put route arrays on a640-byte stack despite no reported
spills. Force the known unroll count rather than relying on compiler heuristics.
CPU7205865 COMPLETED0:0/1m05s: corrected M16 ordinary/critical152regs and Q35
EP8 (64 local experts, histogram branch)184regs, all zero stack and zero spills.
Native launch now explicitly rejects topk>32, matching the warp destination
dedup implementation; public buffer docs state the limit. No GPU validation of
this helper correction yet. Captured headers/plans under genphase_helper_cpu*
preserve both attempts; live trace remains untouched.

GPU7205626 FAILED1:0/6m30s: all six control captures complete; candidate rows16
ordinary/discovery/instrumented workloads completed with saved-input matches,
then analyzer rejected `incomplete generation-specialized dispatch role markers`.
Subsequent peer sysconfig/getcwd failures are teardown effects, not the initiating
error. Saved candidate trace/profiled.json/baseline.json remain; diagnose on CPU
instead of recapturing. Cancelled dependent CPU7205814 and preserved its records.

After7205626 terminated, synced helper correction/native smem guard/selector
unused-parameter cleanup and current analyzers into source/local-mirror. CPU7205904
submitted: inspect saved marker counts then build native extension and selector.
Prepared UNSUBMITTED genphase_helper_timing: reuse passed Q3 edge/boundary
references, candidate edge/boundary followed by the same six-case2000-call ABBA.
Q35 plan still unsubmitted. CuTeDSL status again clean; no defaults promoted.

CPU7205904 COMPLETED0:0/3m37s native+selector build. Marker diagnosis found
PayloadPushed has16 records but8 unique warp locations (lane-predicated peer
publication executes the instrumented point twice); every other role matches.
Analyzer now counts unique locations and retains raw_count plus first/last
timestamps. CPU salvage7205943 COMPLETED0:0/48s on the saved rows16 capture;
all5 phase/role records validate, without another GPU capture.

Initial candidate IKET medians relative to main entry (us): GenphaseStart=-1.152;
lastTopkReceived=5.760;SizesReady=11.872;lastMetadataReady=16.480;
lastPayloadPushed=9.600;RefineStart=16.928–17.152;lastEnd=31.008.
Main compute starts23.584 versuscontrol5.344, compute span30.080 versus25.088;
main envelope57.696 versus34.368. All samples/outliers retained; these overlap
and are instrumented, not additive ordinary phase costs. Genphase SF refinement
is a large exposed delay. Code review confirms old loop stores lane-adjacent SF
words at widely separated column addresses. Local correction walks physical
SF rows for coalesced stores, inverts the unchanged128-row swizzle for gather,
and loads each gather index once across all SF columns. Inverse identity checked
for128/256 rows. CPU7205999 compiles captured corrected M16/Q35/critical headers.
No source sync of this newest SF edit until compile succeeds.

Prepared lightweight_helper/qwen3 (unsubmitted) narrows attribution to rows16/31,
covering M16/M32: four symmetric arms,30ordinary+5critical samples each (8captures).
Six-case ordinary ABBA remains genphase_helper_timing. This avoids repeating the
full24-capture diagnostic; all prior evidence remains under lightweight/.

### Current validation chain (September 17 UTC)

CPU7205999 COMPLETED0:0/55s, M16/Q35/critical all zero stack/spills (152/184regs).
After native/CPU jobs terminated, synced the final SF header, analyzer and docs.
Copied the newly rebuilt7205904 native binary into candidate_runtime (not a
hardlink), refreshed its timestamp only for the subsequent device-only edit;
seed_helper.json records compatibility and build IDs. Selector cache is shared
with the completed CPU build. Source changes are now frozen again.

Submitted serial afterok chain:
- GPU7206025 genphase_helper_timing: candidate Q3 edge/boundary versus existing
  saved references, then fresh six-case control/candidate/candidate/control ABBA.
  Last poll01:29UTC RUNNING1m05s on nvl72147-T[17-18].
- GPU7206027 genphase_q35: paired edge and15/33/65/1024 rows, pending7206025.
- GPU7206028 lightweight_helper/qwen3: rows16/31, four arms,8captures, pendingQ35.
- CPU7206035 lightweight_helper/analysis: collect/audit two selected cases per
  arm, ordinary timeline extraction and four-arm comparison, pending7206028.

No edits to source/local-mirror or source/comm_alignment until the chain is
terminal. Plans/job records are in correspondingly named local scratch folders;
remote results stay remote. Need correctness/timing outcomes, selected2/2 trace
contracts and actual final decomposition before acceptance/completion. Initial
genphase regression and failed marker-count audit remain recorded above.

User reiterates efficiency is required, not just protocol compatibility.
7206025 completed0:0/8m47s; first candidate timing59–62us improves the initial
port but still trails46–48us control. 7206027 Q35 completed0:0/5m48s, including
15/33/65/1024 boundaries. Lightweight7206028 and dependent analysis remain live.
Local-only next refinement stages whole SF rows via TMA, transposes uint4 runs
through skewed shared memory, and bulk-stores into a tile-major canonical pool.
Two independent64-thread groups use four output stages; expert tables load once
per CTA. FC1 SF descriptor/index changes accordingly; combine is unchanged.
Native guard requires hidden divisible512 for16-byte SF row alignment.
Captured-header CPU7206329 submitted for six ordinary/critical specializations;
completed0:0/1m03s, zero spills (dispatch152/184regs, zero stack; main retains
24-byte stack). Not yet GPU validated. New native descriptor requires a rebuild after the frozen
chain ends. CuTeDSL sibling git status again clean; no source changes there.

7206025 full ABBA control/candidate/candidate/control us:
15=46.255/59.299/59.348/46.250;16=46.297/59.197/59.156/46.259;
17=46.253/60.707/60.681/46.219;27=47.602/62.257/62.316/47.599;
28=47.844/61.710/61.737/47.744;31=48.365/62.040/62.075/48.342.
216 exact-combine checks,232 comparisons,48 records; stderr0. Q35 7206027:
64 exact-combine checks,96 comparisons/96 records,stderr0. Short-run boundary
timing is diagnostic only. Each remote run now has comparison.json.
All eight7206028 decomposition files exist; final job/audit status still pending.
Candidate rows16/31 medians: SizesReady9.248/9.664us, MetadataReady12.512/13.600,
RefineStart13.472/14.304, End22.784/23.872; repeats agree. Compute starts
17.824/18.784 versuscontrol5.472/5.344; instrumented envelopes overlap.
Next prepared genphase_tma_timing combines Q3/Q35 edge/boundary correctness and
three-case16/31/1024 ordinary ABBA; genphase_tma_native_cpu rebuild must precede it.

7206028 COMPLETED0:0/8m51s; CPU7206035 COMPLETED0:0/48s. All selected2/2 cases
in each of four arms pass contracts (40 IKET samples,240 ordinary records);
full135-case coverage correctly remains false. After both jobs ended, synced
the three TMA candidate headers into source/local-mirror. CPU7206471 now rebuilds
the native descriptor/launch geometry and copies the result into runtime cache.
The prepared GPU plan depends on its success; sources are frozen again.
GPU7206476 submitted afterok7206471,15min limit. Eight arms: candidate Q3/Q35
edge+boundary checks against saved references, then ordinary three-case ABBA
with fresh control inputs. Do not sync further kernel changes during this chain.
Open after result: tune SM budget only with evidence; reference27 pushers targets
Rubin212SM/184SM compute and is not a portable Blackwell count. Late MMA CTA
starts in the current trace also suggest communication occupancy may matter.
CPU7206471 COMPLETED0:0/4m48s, new host binary copied into candidate_runtime
(`TMA_NATIVE_RUNTIME_SEEDED`). GPU7206476 RUNNING at first poll30s, both sources
remain frozen. Local whitespace check on the modified tracked headers passes.
7206476 partial: Q3 edge and33/65/1024 boundaries PASS; Q35 edge PASS; stderr0.
Wait for Q35 boundary and ordinary ABBA before judging TMA performance.
Unimplemented hypothesis from reference review: reference separates shared
top-k/payload readiness from local counters, each region16-byte aligned. Our
CountsReady/SizesReady/PrefixReady/MetadataReady/PayloadReady/DispatchDone all
share one32-byte sector. Padding/separating readiness flags may reduce contention
between remote payload atomics, local helper publication and150 polling MMA CTAs.
Only test if current ordinary results still justify optimization; no claim yet.

### User stop

User explicitly requested marking blocked and stopping after repeated failures
to improve performance.7206476 reached all four correctness arms and the first
ordinary control/candidate pair; candidate is slower in all three cases above.
Cancelled remaining repeats at~9min; partial comparison.json retains six completed
arms,24 recorded inputs,208 comparisons,104 exact-combine checks,stderr0 before
cancellation. No complete ABBA claim. CPU7206471 had already completed.
The local-only128-byte counter-padding proposal was reverted without any remote
sync/submission; scratch/genphase_flag_padding preserves it as untested context.
Current local/remote kernel source is the tested TMA candidate. Retained faster
baseline remains source/comm_alignment and baseline/pre_genphase artifacts.
No further optimization, profiling, builds or GPU submissions while blocked.

## Concurrent startup follow-up7239646 and release-publication candidate

GPU7239646 COMPLETE11m31s, analysis7239647 COMPLETE44s, nvl72127-T06/T10;
96exact+96replay PASS. Ordinary ordered42.0512/init40.8800/prefetch41.9648/
history38.1344us. Metadata-origin actual first MMA samples:
ordered15.872/15.776/14.112; init13.728/15.264/15.712;
prefetch14.240/13.600/13.728; history9.792/10.336/9.856.
Compute-span samples ordered25.024/25.024/25.696; init25.088/25.120/24.736;
prefetch31.712/32.096/31.648; history24.800/25.120/25.248us.
All init/ordered samples have four metadata-worker pairs issue before their own
startup ends. Retain init as promising candidate; first MMA still later.
Evidence: concurrent_dispatch/followup/comparison_7239646.json.

Paired outgoing sender CTA/warp0 drain→TailPublishStart is2.464–4.640us(init),
2.496–4.704us(ordered); publish→retirement.256–.320us. This is other-warp wait +
firstjoin + systemfence + secondjoin, not isolated fence or network latency.
Only rank0 outgoing measured; cannot infer incoming readiness from it.
Critical receiver tile-ready→MMA ~.6–.7us. Tail marker denotes store issue only.

Next scoped candidate concurrent_dispatch/release_publish: init-overlap parent,
full bulk wait/proxy + first CTAjoin retained, replace tid0 threadfence_system +
secondjoin + relaxedtailstore with per-publisher st.release.sys tailstore.
Release pattern directly on publisher, transitive completed issuer→CTAbarrier→
publisher; receiver already acquire-loads tail. Original frozen sources retained;
corrected stale initialization comment in new candidate only. CPU normal/IKET
compile and register/SASS gate; unchanged router reuses7239590 selector gate.
Reference: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#release-and-acquire-patterns
Goal remains active; no promotion/commit.

Release-publication preparation: CPU7240881 compiled normal+IKET128regs/24stack/0spills, then review failed because cubins were written one directory above expected. No source/compiler failure. CPU7240912 review-only reuses those two cubins after relocating them; future build paths corrected. TailPublishStart precedes release-store ordering in new candidate, so compare drain→retirement, not just drain→marker. Summary now reports this combined interval.

CPU7240912 register/SASS gates PASS. Sender fence changed MEMBAR.SC.SYS→MEMBAR.ALL.SYS, CCTL.IVALL absent at publication; ERRBAR/CGAERRBAR remain. GPU7240934 and afterok analysis7240936 submitted for release32/init32/prefetch32/history32. Fixed EP8, per-command240s timeout, reset outside timing; no additional GPU jobs. Sources frozen.

7240934 ordinary phase complete: release40.7136us(mean)/40.784median, init40.9600/40.960, prefetch41.7280/41.792, history37.5360/37.504. Candidate vs init gain0.2464us(0.60%), small relative to sample ranges; wait for decomposition before judging acceptance. No promotion.

## Release-store publication result7240934: reject for startup

GPU7240934 COMPLETE12m50s, CPUanalysis7240936 COMPLETE48s, nvl72174-T02/T09;
96exact+96replay PASS, analysis stderr empty. Evidence:
concurrent_dispatch/release_publish/comparison_7240934.json, with all4 compact
decompositions under results/. Ordinary kernel mean/medianus:
release40.7136/40.784, init40.9600/40.960, prefetch41.7280/41.792,
history37.5360/37.504. Release gain over init0.2464us(0.60%), ranges overlap.
Metadata-origin first actual MMA samples(us): release15.392/15.136/15.168;
init13.696/15.360/13.888; prefetch11.808/13.536/11.936;
history174.784/10.080/9.920. Preserve history sample0 delayed outlier; kernel entry
frommetadata3.328us, so this is an in-kernel wait, not a PDL entry offset.
Compute spans: release24.800/25.408/25.056; init24.928/25.120/24.640;
prefetch32.160/32.320/32.608; history25.472/24.672/25.120.
Release4/5/4 and init4/4/4 pairs issueMMA before their own startup ends.

The sender tweak does NOT achieve earlier MMA. The marker's apparent
2.5–4.7us→0–.384us drain→publication reduction merely shifts the release ordering
past TailPublishStart. Per-sample median drain→retirementus:
release4.432/4.608/4.432 versus init2.848/4.624/2.992. Candidate maxima include
6.784us; release-store issue→retirement2.752–6.400us vs parent.256–.320us.
Do not mistake the moved boundary for reduced dependency latency. Reject this
sender variant for startup, retain init-overlap concurrent candidate and all
previous sources. First CTAjoin removal alone can save only its measured small
portion; four independent release fences still carry the expensive completion.
Potential next protocol change: receiver-verifiable payload chunks or SF push
before activation wait, requiring a separate correctness proof. No new run for
these ideas. Current validated structural fix decouples metadata workers from
compute and preserves historical-like compute; earlier firstMMA/history kernel
win still unmet. No jobs active, no default promotion, no commit.

## Router SF overlap candidate — CPU7241226

Previous goal turn is progress:7240934 ruled out publication-fence/secondjoin as
startup cure. Next follows remaining SF→activation serialization in init-overlap.
Candidate concurrent_dispatch/router_sf uses selector warps1..7 to push SF while
warp0 reserves destination route slots. One CTA handoff after topk, then independent
roles; each destination assigned exactly one warp and deduplicated by warp vote.
Copies existing source SF into existing plain-SF rows using existing generation
mapping. Existing last-CTA acqrel ticket→SYS route_ready publication covers SF.
Metadata workers acquire route_ready then pack canonical SF without activation-tail
waits. Gather warps separately acquire their four-row tails with warp-uniform polling;
SF TMA/coordinator and K-loop remain unchanged. Sender TMA moves activation only;
retain original CTA fence protocol, not rejected release-store variant. No buffer
allocation/newflags; off-timed-path resets unchanged. Compact candidate explicitly
requires the existing supported CTA-selector shapes, no silent generic alternate.

Direct router markers RouteSfTransferStart/Issued added; issue is not remote
visibility. Summary retains all producer marker bounds from metadata=0 and counts
per-token SF issue before metadata issue. Common selector-origin metrics guard
against hiding the added handoff/PDL shift. Existing trace re-summary passed.
CPU7241226 compiles ordinary+IKET MegaMoE/register gates and both selectors;
GPU plan is candidate/init/history fixedEP8, no submission until gate passes.
Sources frozen for build. No default promotion or commit.

User directed consulting MEGAMOE_COMM_EXPLORATION.md when ideas run short; read
it during CPU7241226. Current router-SF branch matches independent SF/activation
ownership (3) and idle-role overlap (4/7). Next high-upside branch if sender feedback
remains critical is receiver-side validation (2), preserving TMA gather and ~25us
compute. Early SF publication simplifies its protocol by leaving only activation
sentinel encoding/validation, but transport atomicity, cross-proxy visibility and
safe generation reuse must be proven. No such prototype yet. Independent row/peer
publishers (1) remain possible;7240934 shows merely changing fence mnemonic/join
without removing feedback did not help. Keep combine fixed. Exploration proposals
are not performance evidence.

CPU7241226 COMPLETE5m30s, MegaMoE normal/IKET128regs24stack0spills and reachable-register gates PASS, both selector builds PASS. GPU7241365 running on nvl72102-T09/T17; analysis7241367 afterok. Three arms router_sf/init/history,72 expected exact+replay checks. Sources frozen.

7241365 ordinary phase: routerSF kernel39.2992us vs init41.0080 (−4.17%), history37.6928; routerSF range38.752–39.616, init40.384–41.440. Full producer+MegaMoE max-rank CUDAgraph means67.276800/69.094401us improve1.817601us. History fullchain95911.242676us is a large outlier although rank0 kernel mean37.6928; retain, do not compare as a normal fullchain baseline. Candidate instrumented correctness passes; decompositions pending.

## Router SF overlap7241365: new best tested all-push candidate

CPU7241226 COMPLETE5m30s; GPU7241365 COMPLETE9m26s,nvl72102-T09/T17;
analysis7241367 COMPLETE52s.72exact+72replay PASS; stderr empty. FixedEP8 only.
Evidence: concurrent_dispatch/router_sf/comparison_7241365.json and results/*/
decomposition.json. Full producer-chain ordinary means retained below separately.

| metricus | routerSF | init parent | history pull |
|---|---:|---:|---:|
| ordinary kernelmean |39.2992|41.0080|37.6928|
| ordinary kernelmedian |39.248|41.152|37.648|
| firstactualMMA metadata-origin samples |12.000/11.712/12.096|13.760/15.584/13.696|16.992/9.984/10.272|
| firstactualMMA selector-origin samples |14.432/14.048/14.528|16.096/17.792/16.096|19.232/12.352/12.544|
| compute-span samples |25.248/25.312/25.312|24.544/24.704/24.544|25.152/25.280/24.576|
| latest CTA-pair firstMMA samples |12.576/12.480/12.736|14.400/16.672/14.560|19.040/11.040/11.872|

Kernel improves1.7088us(4.17%); firstMMA median13.760→12.000us, also earlier from
selector origin16.096→14.432us. Compute median25.312us near history25.152, though
~.768us above unusually fast parent24.544 in this run. All75pair start ranges
advance; not one lucky early tile. Metadata worker finishes before MMA now, so
MMA-before-local-startup count0 is expected and not lost concurrency.
Fullchain max-rank CUDAgraph67.276800us vs69.094401 parent (−1.817601us).
History fullchain95911.242676us outlier retained; ordinary rank0 kernel and its
instrumented first-sample delay16.992us are separate observations, no attribution.

New producer SF markers prove all32 tokenCTAs issueSF before own metadataissue in
all3samples. SFStart first−.352us relative to first route-metadata start; SFIssued
last.576/.416/.480us vs metadataissuedlast2.464/2.208/2.432. Negative offset is
real SF work before route atomic issue, not a PDL artifact. RouteMetadataPublished
5.024/4.736/5.024 vsparent4.960/4.832/4.928us, no major metadata-delay shift.

Critical-pair canonicalSF-ready first10.112/9.600/9.472us vsparent13.024/14.816/
12.992; gather-warp0 activation-ready10.368/10.432/10.624; gatherissued10.496/
10.560/10.752; MMA12.000/11.712/12.096. The~1.1–1.5us gatherwarp0issue→MMA gap
also includes other gather warps' unmeasured readiness, not proof of slow TMA.
Rank0 sender drain→retirement median3.168/3.040/3.008us remains; outgoing markers
cannot directly explain incoming activation timing from other ranks.

Retain router_sf as new best tested all-push candidate. No default promotion or
commit; previous init preserved. All jobs terminal. Broader objective still active:
history kernel37.693us remains1.606us faster; payload readiness still waits issuer
completion+publication. Next use communication exploration(1/2): independent
completion ownership or receiver validation, preserving this SF split and TMA
compute. Add direct first-tile activation-ready/gather-issue markers for all4
gather warps on both compared paths if needed to distinguish last-required input
from the currently sampled warp0; no source edits or jobs for that next step yet.

## Receiver-validation/materialization prototype — CPU7241899

Previous goal turn was progress: routerSF7241365 improved kernel and broad MMA
startup while preserving history-like compute. New receiver_validate candidate
keeps routerSF producer and compute TMA. Sender local TMA loads remain; remote
payload uses aligned st.relaxed.sys.v4.u32, each scalar u32 sentinel collision
canonicalized to numeric zero. Receiver metadata warps validate every scalar of
ld.relaxed.sys.v4.u32 and copy exactly those observed values into receiver-owned
HBM rows. Never reread a live source inbox through TMA after a sentinel test.
Strong same-proxy, completely overlapping u32 operations provide scalar atomicity;
one writer per word/epoch and off-timed-path initialization make non-sentinel a
complete value. Ordinary local stores→proxy fence+CTA join→local release protect
materialized payload; gather acquire+proxy fence composes TMA visibility. Source:
https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html
§8.2.4/8.7/8.10.3. This is a correctness argument for this ownership change, not a
claim that arbitrary generic polling of weak TMA writes is valid.

Uses own-source rank's inactive rows [num_tokens,num_tokens+padded_pool_rows) in
existing source bank; trap if beyond bucket capacity. Fixed32 case needs512padded
rows with3840bucket, fits; no general full-capacity support claimed. Existing
counter phase1=valid_m publishes SF/index; phase2=2*valid_m publishes copied A.
Coordinator waits>=valid_m, gather loads indices at phase1 then waits==2*valid_m.
SF remains independent of A validation; no new flag/allocation. Sender SYS fence
and join remain for retirement but do not gate receiver readiness. Cleanup/reset
and generation lifetime remain as before. Local A copy traffic is a real cost;
more contiguous gather addressing is also changed, so gain is combined protocol
plus materialization, not uniquely an ACK-latency attribution.

Candidate and best routerSF parent add direct first-tile indices/A-ready/gather
markers on all4gatherwarps. Replace old sender drain/tail marker types with
PayloadStoresIssued and PayloadValidateStart, preserving marker-type budget.
GenphaseEnd remains sender retirement; last IndexReady includes final phase2.
Summary reports first-pair eight-gather-warp timings and store-issue→retirement;
existing parent trace re-summary PASS. profile shell syntax PASS.
CPU7241899: candidate normal+IKET and matched parent IKET register gates; warm
unchanged router normal+IKET for both new include roots (cache keys differ).
No GPU submitted yet. Parent and candidate frozen. No promotion/commit.

CPU7241899 candidate normal/IKET +parent IKET reachable-register gates PASS,128regs24stack0spills. GPU7241998 afterok CPU7241899 and analysis7242000 afterok GPU submitted; awaiting final cache warms. Three fixedEP8 arms validate/routerSF/history;72 expected exact+replay checks. No speculative additional GPU jobs.

## Receiver validation7241998: serialization exposed; batched follow-up7242329

CPU7241899 COMPLETE3m30s; GPU7241998 COMPLETE9m17s,nvl72106-T09/T15;
analysis7242000 COMPLETE44s.72exact+72replay PASS. Reject unbatched prototype:
ordinary kernel59.5104us vs routerSF39.1904/history37.6256. FirstactualMMA
metadata-origin19.776/20.416/19.712us vs12.000/11.840/12.064; selector-origin
22.208/22.848/22.144 vs14.400/14.400/14.496. Compute34.400/33.632/34.304us
vs25.216/25.024/25.088 (history24.640/25.024/25.376). History startup outlier
69.312us retained alongside9.888/9.728; not discarded. Fullchain86.396800us vs
control66.358401us. Evidence: receiver_validate/comparison_7241998.json.

All-four-gather markers resolve the old gap: best routerSF firstpair's LAST
issue→MMA is only.192/.256/.192us, whereas warp0-only had suggested~1us.
Late companionCTA warp1 waits payload until11.680/11.488/11.744us. This is
operand readiness skew, not evidence that TMA itself consumes the whole gap.
Prototype validation starts8.000–9.056us, firstpair A-ready~19.3–20.2us;
last metadata-worker phase2 ends~36us. Sender store issue→retire~2.6us, but
rank0 outgoing timing is not incoming remote causality. Firstpair lastgather
issue→MMA.224/.160/.128us: validation/materialization dominates startup.

Assembly exposes one SYS128load→check/vote→localstore→nextload chain. New
batched dispatch issues8independent SYS128loads before any result check, then
8localstores; sender loads8vectors into distinctregisters and calculates peer
address once per8-store row. CPU7242229 COMPLETE2m34s; normal/IKET128regs,
24stack,0spills, reachable-register gates and both selector cachewarms PASS.
SASS confirms8consecutive SYS128loads before firstISETP and8consecutive
SYS128stores using distinctdataregisters/base+immediates. This removes actual
issue serialization, not merely source-level loop syntax. No new protocol,
allocation, marker, or TMAcompute change. Existing receiver root/control frozen.
GPU7242329 and dependent CPUanalysis7242330 submitted: batched/routerSF/history,
fixedEP8 only. No claimed gain until ordinary timing + detailedMMA/compute pass.
Keep best routerSF; no default promotion or commit. Broader goal remains active.

Prepared (unsubmitted) wide_metadata overlay, based on batched: metadata uses
both epilogue WGs256threads/eightwarps; sender/helper stay128 and8senders. Each
16-row tile has2validation rows/warp instead4. Existing bar3 uses256 participants
only onmetadataCTAs; finaljoin separates later epilogue barrier reuse. Sameprotocol,
TMAcompute, scratcharena; no flags/allocations. Local diff reviewed and sources/
include overlay synced. Wait for7242329 decomposition before job submission.

## Batched receiver validation7242329: substantial repair, still later MMA

GPU7242329 COMPLETE8m14s,nvl72106-T09/T15; CPUanalysis7242330 produced all three
decompositions;72exact+72replay PASS. Evidence: receiver_validate/batched/
comparison_7242329.json. Ordinary kernelmean/median/min/max(us):
- batched41.9808/41.984/41.536/42.656;
- routerSF42.624/42.512/39.168/47.200 (noisy, retain all samples);
- history37.936/37.872/37.472/38.368.
Batched is much faster than previous unbatched59.5104, but no accepted win over
best routerSF39.19–39.30 stable prior runs. Matched noisy control mean is not
sufficient for promotion, and startup criterion explicitly fails below.

FirstactualMMA metadata-origin(us): batched12.736/12.992/12.800 vs routerSF
11.936/12.000/11.872 vs history9.536/9.952/9.760. Selector-origin batched
15.168/15.392/15.200 vs14.304/14.368/14.336 vs11.776/12.224/12.224.
Compute25.344/25.984/26.112 vs25.312/25.504/25.056 vs24.992/25.440/25.120.
Latest75-pair firstMMA14.144/14.496/14.176 vs12.544/12.640/12.480;
more than one early tile still delayed. Firstpair lastgather→MMA.192/.192/.224us.
Batched critical A-ready12.224–12.672us, SF-ready9.760–10.208us; metadata
validation starts8.192–9.216us and last tile publication19.36–20.00us. Outgoing
sender stores allissued<=8.608us, vs unbatched<=10.560; remote source causality
still unmeasured. Batching removes repeated issue/result serialization and
recovers near-history compute, but row validation and second expert remain late.
Full producer-chain batched70.716800us vsrouterSF68.608000us; separately measured.

Next measured dependency: increase metadata participants only128→256 so each
warp validates2rows/tile instead4. Prepared wide_metadata ownership proof/diff
above; sources synced. Four-arm next comparison includes batched parent, best
routerSF and historical pull to isolate widening despite prior timing noise.
CPU gate must pass before GPU. No default promotion, no changes to best path.

Wide-metadata CPU7242511 submitted and observed RUNNING. Normal/IKET compile, reachable-register review and both selector cache warms; GPU not submitted. Planned four arms wide/batched/routerSF/history,96expected exact+replay checks. CPUanalysis7242330 confirmed COMPLETE43s.

CPU7242511 normal+IKET reachable-register gates PASS,128regs24stack0spills; selector cache warms still completing. GPU7242547 submitted afterok CPU7242511; analysis7242548 afterok GPU. Four fixedEP8 arms wide/batched/routerSF/history;96expected exact+replay checks. Sources frozen. No promotion or commit.

CPU7242511 COMPLETE3m07s; selectors/register gates PASS. GPU7242547 observed
RUNNING on nvl72106-T09/T15; first wide ordinary exact/replay checks passed,
fullchain66.960001us (not kernel-only result). SASS retains8consecutive SYS128
loads at24c0–2530 before checks. Analysis7242548 pending afterGPU.

Additional source review while capture runs: current prefix uses only16 nonzero
lanes but4/8warps write/read zero warp sums and join repeatedly. Historical
single-warp padded-prefix implementation (direct_expert_metadata CPU7218284)
removed that overhead successfully for correctness/register allocation, although
its whole older protocol was not a performance winner. Candidate follow-up is
onewarp32-lane scan directly from expert counts, sharedoffsets/total then oneCTA
join; SizesReady depends only on acquired route_ready/counts and need not wait
for sharedprefix. This keeps all existing SF/validation warps; not packing payload
work into onewarp. No new source or job yet. Also reviewed old producer_payload
7217857 and independent router warp-split7219376 negatives: moving payload into
router alone made metadata/startup later, so do not blindly repeat it. A future
router transfer would require demonstrably independent metadata/SF publication
and per-word receiver readiness, not the former all-rank PayloadReady ticket.

## Independent SF/index and activation validation — CPU7242697

Prepared receiver_validate/split_validation from wide_metadata. Source-confirmed
remaining edge: prefix→SF/index work→all-worker join→A validation; activation does
not require SF/index stores to start, only to publish the final consumable tile.
Use existing WG0(threads256–383) for SF/indices, WG1(384–511) for A validation.
Both independently traverse experts/tiles from immutable shared offsets. SF local
join is bar3/128; A local join is bar4/128; shared setup/final retirement is
bar5/256, so neither subgroup can mix phase arrivals or prematurely reuse its
epilogue barrier. Source staging, senders, math/TMAcompute unchanged.

WG0: SF/index stores→proxy fence+bar3→phase1 release(valid_m). WG1: batch SYSloads,
validate each scalar, copy observed bytes→proxy fence+bar4→leader acquire-waits
phase1==valid_m→phase2 release(addvalid_m). If A arrives first it cannot expose
an SF-ready value; SF never waits A. Thus payload validation overlaps SF packing,
and only publication joins the necessary inputs. Finalbar5 precedes DispatchDone
and epilogue reuse. Same existing counter/generations/capacity guard, no newstate.
ValidateStart moves to logicalwarp4/physicalwarp12 (same8metadataCTA markers);
IndexReady/End remain warp0 after bothgroupsfinish. Main reuses wide header.

Tradeoff: only4A-validation warps vswide8, but they start~2us earlier and SF can
prepare later experts while A validates the current one. Must measure firstMMA,
all75pair releases, compute span and ordinary kernel; no presumed win. Local diff
and ownership reviewed, small dispatch/include overlay synced. CPU7242697 observed
RUNNING: normal/IKET register gates and selectors. No GPU submitted until current
wide7242547 comparison/analysis7242548 completes. Planned split/wide/routerSF/
history four-arm comparison, not an extra shape sweep. Sources frozen for CPU.

## Wide metadata7242547: useful reduction, best routerSF still faster

GPU7242547 COMPLETE10m53s,nvl72106-T09/T15; all4 decompositions collected;
96exact+96replay PASS. receiver_validate/wide_metadata/comparison_7242547.json.
Normal kernelmean/median(us): wide40.2016/40.160, batched41.9552/41.984,
routerSF39.4784/39.376, history37.7184/37.568. Wide saves1.7536us(4.18%) vs
batched but loses.7232us(1.83%) vsbest control; no promotion. Rangeswide39.840–
40.640,batched41.536–42.560,routerSF38.784–41.088,history37.248–38.624.

| metricus(three instrumented samples) | wide | batched | routerSF | history |
|---|---|---|---|---|
| firstactualMMA metadata-origin |11.520/11.840/11.776|12.480/12.736/12.832|11.904/11.776/12.064|114.080/10.368/9.760|
| firstactualMMA selector-origin |13.952/14.176/14.144|14.880/15.072/15.136|14.304/14.112/14.496|116.320/12.640/12.000|
| compute span |25.856/25.440/25.600|25.568/25.888/25.696|25.408/25.184/25.248|24.832/24.768/25.152|
| latest75-pair firstMMA |12.960/13.248/12.896|14.080/14.336/14.656|12.672/12.512/12.672|115.296/11.456/11.200|

Wide firstMMA median11.776 vscontrol11.904 is only.128us earlier; latestpair
stilllater, so broad-start acceptance fails. History114.080us delayed sample
retained. Criticalwide SFready9.568/9.760–9.888/9.856us; Aready11.136–11.520;
lastgather→MMA.128/.224/.160us. Validation starts8.000–8.896 afterSFwork;
lasttilepublication16.896/16.960/16.768, improved frombatched19–20us. Compute
nearhistory but~.35us slower thanmatchedrouterSF median. Fullchainwide66.960001us
vsrouterSF67.356801us improves.3968us, but does not override kernel-only loss.

CPU7242697 split_validation COMPLETE2m18s: normal/IKET128regs24stack0spills,
register gates andbothselectorwarmPASS. Source-confirmed SFpacking→Avalidation
edge remains worthwhile despite widening: independent WGs can start validation
at prefix completion, rather than8–9us. GPU7242794 submitted afterCPU; analysis
7242795 afterGPU. Fourarms split/wide/routerSF/history,96expectedexact+replay.
Updated direct-marker interpretation to allow ValidateStart beforephase1 and
physicalwarp12; shellsyntaxPASS. Counters/markers unchanged; no injection.
Sources frozen, keepbest routerSF, no promotion/commit. Goal remains active.

## Two-row load batch — CPU7242916

Current split7242794 capture continues; no result/promotion inferred yet. Next
source-confirmed serialization: validation handles one4KB row's loads/checks/copy
before issuing the next row. Prepared row_batch based on split_validation: each
validation warp owns contiguous rowpairs, strides8rows across four warps, and
holds16uint4 vectors(2rows×8vectors) before inspecting either row. Every scalar
stillchecked; partial-row guards warp-uniform, no inactive pointeraccess. Two
batches cover each warp's4rows in this16-row tile. Same store/counter/publication,
SF/WG overlap, math and buffers. Only load/copy schedule changed.
CPU7242916 normal/IKET reachable-register gatesPASS128regs24stack0spills; selector
warmup in progress at first poll. SASS16consecutive predicated SYS128loads at
2a40–2b30 useR20..R83 with independent rowbases; first data check2b90, afterall
loads. Row-active predicates precede loads and do not depend on payload. Thus
actual independent memory issue survives compilation. No GPU submitted; await
split7242794 decomposition to decide. Planned row_batch/split/routerSF/history.

Rejected follow-up for this fixed case: source-row validation deduplication.
Although multiple experts can share a token generally, the harness enables
rank-interleaved expert_map=(expert%EP)*local_experts+expert//EP. EP8/topk8 assigns
one expert perrank pertoken;256unique received tokens/rank already match256local
expert routes. There is no8x validation duplication here. Source: frozen
latency_overlap/afd_megamoe_same_rank_smoke.py:791–810. Do not use generic routing
assumptions or an irrelevant dedup experiment to explain this case's timing.

## Split SF/validation7242794: overlap works, no kernel improvement

GPU7242794 COMPLETE10m54s; CPUanalysis7242795 COMPLETE46s;96exact+96replayPASS.
Evidence receiver_validate/split_validation/comparison_7242794.json. Kernel
mean/medianus: split41.5648/41.152, wide40.4704/40.368, routerSF39.2896/39.312,
history37.7600/37.728. Splitrange40.128–44.416 vswide39.936–41.248,
routerSF38.752–39.648. Rejectsplit alone; best routerSF retained.

| metricus(three samples) | split | wide | routerSF | history |
|---|---|---|---|---|
| firstactualMMA metadata-origin |11.744/11.936/11.712|11.552/12.128/11.840|11.712/11.680/11.808|9.600/10.176/9.792|
| firstactualMMA selector-origin |14.176/14.336/14.144|13.984/14.560/14.272|14.112/14.048/14.240|11.872/12.416/12.032|
| compute span |25.472/25.920/25.536|25.920/25.344/25.536|25.248/24.896/26.016|25.376/24.800/25.120|
| latest75-pair firstMMA |13.568/13.728/13.120|13.024/12.896/13.088|12.544/12.448/12.416|11.328/11.296/11.648|

Intended overlap is measured: validation starts6.528–6.656us, previously8–9us.
Critical SFready9.728–10.272, Aready11.264–11.680; lasttilepublication16.640–
17.088us, essentially unchanged fromwide~17. Four Avalidation warps still process
four rows sequentially; earlier start is offset by fewer row workers. No isolated
causality claim beyond this measured overlap/source assignment. Broad firstMMA
release is worse thanwide, compute~25.5 remainsnearhistory, ordinarykernel loses.
Fullchain split66.976000us/wide66.723202/routerSF67.014400; history94119.384766us
large outlier retained separately. These do not override kernel-only loss.

CPU7242916 row_batch COMPLETE2m18s: both selectors and normal/IKET gatesPASS,
128regs24stack0spills. SASS16independent loads before data checks verified above.
Targets remaining row serialization with unchanged SF/A ownership/readiness;
GPU7243096 and afterokanalysis7243097 submitted, fourarms row_batch/split/routerSF/
history,96expectedexact+replay. Initial sbatch withafterokCPU failed because live
Slurm no longer retained completed7242916 (scontrol invalid; sacct COMPLETE0:0,
no matching GPUqueue entries). Submitted after verified successfulpreflight with
expired CPUdependency removed; exact resolved argv saved beside plan. No rebuild
or duplicate GPU. Sources frozen; no promotion/commit; broadergoal stillactive.

## Direct source loads — CPU7243254

Prepared receiver_validate/direct_source from faster receiver wide_metadata
(parent40.2–40.5us), without stacking split/row_batch changes. Generic sender
currently does localTMA→shared→mbarrierwait→LDS→register→genericSYSstore. Replace
that redundant source staging with8batched ld.relaxed.gpu.v4.u32 loads directly
from producer-published localX. Keep source-owned destination dedup, distinct
registers/per-peer store offsets, sentinel canonicalization, genericSYS remote
stores, wide receiver validation/materialization, TMAcompute and sender retirement
SYSfence/join. No source-slot mbarriers/init/invalidation/joins remain; sharedarena
sizes/layout/CTA count stay unchanged. Independent warp tokenloops join only for
retirement. CountsReady acquire+syncwarp orders generic X/topk loads; no generic→
async sourceproxy fence is needed because this source path no longer usesTMA.

Earlier source_vector_payload7215591 removed staging while also changing remote
transport/publication, yielding only~.175us32-row kernelgain on an older parent.
This new comparison isolates staging inside an already-generic transport and
receiver-validation protocol; do not claim the old result predicts a gain here.
Direct source/vector CPU gate and selectors submitted7243254; no GPU yet. Sources
and bounded include overlay synced/frozen, main referenceswide header. Planned
fourarms direct/wide/routerSF/history after row_batch7243096 decomposition.

Marker nuance: source LoadStart remains before route dedup/inputread phase;
StoresIssued follows all peer stores. Old generic SendStart precedes LDS/encoding,
new direct SendStart follows global loads/encoding before peer stores. Do not
attribute that marker delta solely to transfer latency; compare whole sourcephase
and common-origin receiver/MMAmilestones. Metadata/SF/main markers unchanged.

Direct_source CPU7243254 normal/IKET gatesPASS128regs24stack0spills; SASS8consecutive
GPU128loads2ff0–3060 confirms no intermediate source sharedstaging. Secondselector
warm stillrunning at4m27s. Row_batch GPU7243096 COMPLETE12m21s; analysis7243097
pending. No nextGPU submitted. Deferred first512B-then-rest materialization/first
window network priority idea recorded in audit; no source/job for it. This targets
full-H-before-firstK dependency while retainingTMA, with explicit extra-phase risks.

## Two-row batch7243096: reject compute regression

GPU7243096 COMPLETE12m21s,nvl72131-T08/T09; analysis7243097 COMPLETE57s;
96exact+96replayPASS. Evidence receiver_validate/row_batch/comparison_7243096.json.
Ordinary kernelmean/medianus: rowbatch41.8976/41.888, split40.4000/40.384,
routerSF39.3056/39.280, history37.8720/37.840. Rowbatchrange41.472–42.272 exceeds
split40.000–40.800: +1.4976us/+3.71% mean. Reject; no default promotion.

| metricus(three samples) | rowbatch | split | routerSF | history |
|---|---|---|---|---|
| firstactualMMA metadata-origin |11.840/12.224/11.744|11.616/11.840/11.936|11.840/11.904/11.872|102.240/9.984/9.600|
| firstactualMMA selector-origin |14.336/14.688/14.144|14.048/14.272/14.368|14.272/14.432/14.304|104.512/12.256/11.968|
| compute span |29.984/29.088/29.408|25.888/25.856/26.016|25.568/25.472/25.280|24.736/25.056/25.088|
| latest75-pair firstMMA |13.536/13.248/13.024|13.280/13.312/13.312|12.448/12.480/12.640|104.032/11.072/12.256|

Validation starts6.528–6.656, lasttilepublication15.456–15.648 (earlier thanold
split~17), but firstMMA has no improvement and compute-span median29.408 vs25.888
regresses3.52us. No spills and16independent load instructions did not suffice for
performance. Do not claim an isolated cache/register/queue cause without evidence.
Fullchain rowbatch70.003201/split69.382400/routerSF66.134400us; history96925.616455us
large outlier retained separately, alongside instrumented firstMMA102.240us.

Next direct_source stays onfaster wide_metadata parent, not the losing rowbatch.
CPU7243254 COMPLETE5m37s includingbothselectorwarmups, normal/IKET128regs24stack
0spills/gatesPASS. Eightconsecutive GPU128source loads at2ff0–3060 confirmed.
GPU7243409 and afterokanalysis7243410 submitted: direct/wide/routerSF/history,
96expectedexact+replay. Successful completedCPU preflight alreadyverified, so
GPUplan omits obsolete CPUdependency; resolvedargv saved. This removes local
TMA/shared/register staging while retaininggenericremote stores, widevalidation
andTMAcompute. New source-phase interpretation metadata documents unequalSendStart
boundaries; comparewholephase andcommon-originreceiver/MMA. ShellsyntaxPASS.
Sources frozen, best routerSFretained, no promotion/commit. Goalstillactive.

## First-window readiness — CPU7243595 / GPU7243644

Following exploration items 2/3, `first_window` starts from `wide_metadata` and
keeps the sender unchanged. Metadata workers validate and copy the first 512 B
of every tile row using 16 lanes/row and two independent uint4 loads/lane.
Phase 2 releases the first four 128 B gather stages. The remaining seven vectors
per row retain batched validation, followed by phase 3. Gather waits for
`counter >= 2*valid_m` initially and, once at `k=512/BLOCK_K`, for
`counter >= 3*valid_m`. The comparisons cannot miss an already advanced phase.
SF/index remain phase 1. All consumed words are validated and copied; proxy
fences and group joins precede each release. Capacity guards, ownership, epochs,
CTA count and resets outside timing remain unchanged. Risks: another readiness
boundary, a stage-4 stall and memory contention. Require lower ordinary kernel
time, earlier/broader MMA and compute near 25 us; no per-K generic compute gather.

Marker budget unchanged: first-tile `FullPayloadIssue` replaces last `IndexReady`.
It marks phase-3 release issue, not receiver observation or completion; `End`
still covers final retirement. Analyzer accepts exactly one contract and checks
EP locations. Shell syntax passed. CPU7243595 COMPLETE 2m15s: normal/IKET register
gates, both router warmups, 128 registers/24-byte stack/zero spills. GPU7243644
and afterok analysis7243645 submitted only after direct-source GPU completed.
Four arms: window/wide/routerSF/history; 96 exact + 96 replay checks expected.
Sources frozen; no promotion or commit.

## Direct source7243409: no meaningful gain

GPU7243409 COMPLETE on nvl72148-T14/T16; CPUanalysis7243410 COMPLETE 44s.
96 exact + 96 replay PASS. Evidence: `receiver_validate/direct_source/comparison_7243409.json`.
Ordinary kernel mean/median us: direct40.2784/40.224, wide40.3296/40.320,
routerSF39.2992/39.264, history37.7280/37.568. Direct range39.712–41.024 overlaps
wide39.872–40.768; only 0.0512 us/0.127% mean change. No meaningful kernel/startup win.

| metric us (three samples) | direct | wide | routerSF | history |
|---|---|---|---|---|
| first actual MMA, metadata origin |12.096/12.064/11.680|12.064/11.872/11.840|11.968/13.280/11.872|9.888/10.144/9.760|
| first actual MMA, selector origin |14.496/14.496/14.112|14.464/14.304/14.176|14.496/15.648/14.304|12.128/12.384/12.000|
| compute span |25.888/25.760/25.664|25.664/26.400/25.792|25.344/25.280/25.216|24.704/24.480/24.864|
| latest of 75 pair starts |13.152/13.632/13.312|13.056/13.024/12.896|12.640/13.824/12.544|11.648/11.168/11.616|

Matched per-sender LoadStart→StoresIssued medians: direct2.704/2.736/2.704 us,
wide2.736/2.640/2.848; whole source phase also essentially unchanged. Last stores
issued: direct8.320/8.416/8.480 vs wide8.320/8.480/8.640. Validation starts
8.352–9.472 and last tile publishes16.864–17.664, without improvement.
Last first-pair gather issue→MMA is0.096–0.160 us. Rank0 outgoing markers do not
establish incoming remote arrival. RouterSF's13.280 us first-MMA sample is retained
(companion CTA warp1 Aready12.960). Fullchain direct68.441601/wide67.472001/
routerSF66.534400 us; historical105467.565918 us outlier retained separately.
No promotion; best routerSF remains about39.3 us. First-window tests a different
dependency and deliberately retains the wide parent's sender.

## First-window7243644: earlier/broader MMA, but kernel regression

GPU7243644 COMPLETE12m58s on nvl72107-T06/T14; analysis7243645 COMPLETE50s.
96 exact + 96 replay PASS. Evidence: `receiver_validate/first_window/comparison_7243644.json`.
Kernel mean/median us: window40.6496/40.448, wide40.2304/40.208,
routerSF39.2448/39.232, history37.7120/37.712. Window +0.4192 us/+1.04% vs parent;
no promotion. Fullchain69.184001/wide67.260801/routerSF68.060800 us;
historical97972.381592 us outlier retained separately.

| metric us (three samples) | window | wide | routerSF | history |
|---|---|---|---|---|
| first actual MMA, metadata origin |10.848/10.816/11.776|12.416/11.904/11.744|11.872/11.744/13.568|10.048/10.144/9.760|
| first actual MMA, selector origin |13.184/13.152/14.272|14.752/14.272/14.176|14.272/14.080/15.936|12.320/12.416/12.032|
| compute span |27.296/27.008/26.592|25.824/26.336/25.984|25.184/25.440/25.376|24.480/24.896/24.896|
| latest of 75 pair starts |11.488/12.096/12.352|13.568/13.440/12.992|12.608/12.320/14.080|11.264/11.168/11.232|

The intended overlap is real: first MMA precedes even the earliest first-tile
FullPayloadIssue in every sample (10.848<11.488,10.816<11.296,11.776<12.064).
First-MMA median advances1.056 us and latest-pair median1.344 us vs wide, but
compute median grows1.024 us. SF ready on first pair9.44–10.016 us; A first-window
acquire10.4–11.456. Full-payload release issues span11.296–13.152; metadata retirement
now19.136–19.744 vs parent's roughly17. Earlier launch is insufficient; no isolated
claim about stage4 vs other stalls without additional evidence. RouterSF13.568 us
first-MMA outlier retained. Best all-push remains routerSF~39.25 us.

Next bounded idea: remove the two activation-wide 256-thread joins. Each validation
warp can release its own completed rows and proceed independently. Encode first-window
and remainder counts in separate halves of the existing u32 counter so a fast
remainder publisher cannot falsely satisfy first-window readiness. Keep SF's existing
initial join and final retirement join; per-warp proxy fences/syncwarp order copies.
This trades more local reductions for less cross-warp waiting, with no allocations,
new CTA roles or more data in flight. Exact counter ownership and marker counts must
be updated explicitly. Not yet compiled or measured.

## Independent warp publication — CPU7244063

Prepared `warp_publish` from first_window. Remove only the two activation-wide
256-thread joins; retain initial SF/index and final retirement joins. Each warp
fences its copied bytes, syncwarps, then lane0 asynchronously adds its owned row
count. Existing u32 counter: low16=SF valid_m + first512B rows, high16=remainder
rows. Gather masks low16 for index/SF and first-window acquisition, then high16
before later K. Static BLOCK_M<32768 prevents inter-field carry. Different warps
may finish remainder first without falsely releasing missing first-window rows.
Byte ranges are disjoint; final group join preserves retirement/barrier reuse.
No additional allocation, transfer volume, outstanding load batch or sender change.

First-tile FullPayloadIssue now marks each owning warp's remainder publication:
64 locations in fixed EP8/16-row tiles, versus8 in the parent. Explicit per-case
analyzer count and descriptions preserve the distinction; marker-type budget stays
unchanged. Fewer group waits trade against more local atomic reductions. Do not
claim this fixes the compute regression until measured. CPU7244063 submitted for
normal/IKET register gates and both router warmups; sources synced/frozen. Planned
four-arm comparison warp/window/routerSF/history; no GPU submission yet.

Warp_publish CPU7244063 COMPLETE2m20s: normal/IKET register gates and both router
warmups PASS,128 registers/24-byte stack/zero spills. GPU7244130 and afterok CPU
analysis7244131 submitted: warp/window/routerSF/history,96 exact +96 replay checks
expected. Successful CPU completion verified before submission; no expired CPU
dependency. Sources frozen; results pending. Best routerSF preserved, no promotion
or commit, latency-chain goal remains active.

Further first_window evidence: absolute compute ends (metadata origin) are
38.144/37.824/38.368 us vs wide38.240/38.240/37.728. Thus the longer compute span
mostly consumes the earlier start; it does not establish slower MMA service.
Last-finishing pairs are0/6/44 for window; pairs0 and6 contain metadata workers
whose local startup ends19.136/18.592 us. Wide last pairs4/0/2 finish startup at
17.344/16.192/16.128. This suggests an epilogue admission dependency but does not
prove that accumulators were already ready. No current first-FC1 accumulator
marker exists; optional EpiL1 markers listed in register limits are absent here.

A useful next measurement is a guarded IKET observer on otherwise-idle warp7:
wait on the first tmem_full barrier without consuming it, mark its ready
observation, and separately mark the first epilogue wait return. Gate on actual
FC1 work to avoid empty-work hangs; compare same-CTA observations and startup end.
These are observation times, not exact hardware completion; small signed skew is
possible. Two redundant optional markers (TileReady duplicates SFReady, OperandsWait)
can be replaced to preserve the30-type budget. Ordinary kernel must be unchanged.
If readiness precedes worker retirement, a bounded follow-up is16 metadata CTAs
(one expert each) rather than8 (two experts each), still8 senders and150 compute
CTAs, to retire epilogue workers earlier. This changes peak validation pressure
and sender/helper placement, so it needs a matched comparison and explicit role/
DispatchDone/marker counts. Neither observer nor16-metadata change is implemented.

Accumulator observer source is now prepared locally in `receiver_validate/accumulator_probe`:
first_window_main.cuh and warp_publish_main.cuh, both guarded by IKET+CRITICAL.
Warp7 counts actual FC1 work, then observes tmem_full[0] phase0; epilogue warp0
marks its first FC1 wait return. No arrival, TMEM read or ordinary-kernel change.
Replace optional TileReady/OperandsWait markers, preserving type budget. Summary
script now pairs same-CTA observations, retains signed scheduling skew, and records
whether ready was observed before that CTA's startup retirement. Not compiled,
synced or measured yet;16-metadata-CTA experiment remains only a proposal. Current
GPU7244130 is still running; do not edit its frozen source or repeat submission.

## Warp publication7244130: earlier retirement, no best-path gain

GPU7244130 COMPLETE12m34s on nvl72066-T06/T07; analysis7244131 COMPLETE46s.
96 exact +96 replay PASS. Evidence `receiver_validate/warp_publish/comparison_7244130.json`.
Kernel mean/median us: warp40.7616/40.768, window40.9696/40.800,
routerSF39.4688/39.376, history37.7152/37.696. Only0.208 us/0.51% vs parent,
with overlapping ranges40.160–41.856 and40.224–42.080; no best-path win.

| metric us (three samples) | warp | window | routerSF | history |
|---|---|---|---|---|
| first MMA, metadata origin |10.656/11.552/11.200|10.560/11.168/10.976|12.192/12.064/12.064|9.824/9.696/9.888|
| first MMA, selector origin |13.088/13.984/13.536|12.928/13.600/13.408|14.624/14.496/14.464|12.096/11.968/12.160|
| compute span |27.840/27.072/26.496|26.880/27.104/27.680|24.992/25.792/25.600|24.704/25.248/24.896|
| latest of75 pair starts |11.680/12.288/12.256|11.232/12.064/12.032|12.768/12.608/12.704|11.104/10.816/11.328|

Metadata retirement17.504/18.176/17.440 vs18.368/19.744/19.008 advances median
1.504 us, but compute remains27.072 vs27.104; broad starts do not improve.
FullPayloadIssue count64 vs8 validates the distinct contracts. First per-warp
remainder publication10.208–10.464 is not full-tile readiness; never compare it
as such against parent after-group publication. Fullchain67.788802/window68.668800/
routerSF70.732800 us, with historical100673.126221 us outlier retained separately.
No promotion; retain simpler first_window as base for the next structural test.

Prepared `metadata_spread`:16 metadata CTAs handle one expert each, instead of8
handling two. Helper16 and senders17..24 replace helper8/senders9..16; eight
senders and150 compute CTAs remain. DispatchDone target25, metadata markers16,
route-ready source count still8. First-window three-phase protocol unchanged.
Both candidate and parent have guarded accumulator/epilogue observers; normal
kernel excludes them. This tests earlier epilogue-worker retirement and directly
measures whether ready accumulators were waiting. Peak validation pressure and
physical sender placement change; no isolated causal claim before measurement.
Source/CPU/GPU plans prepared, shell syntax PASS; no CPU or GPU submission yet.

Metadata_spread CPU7244419 submitted for candidate/control ordinary+IKET compiles,
register gates and all four router warmups. Both bounded overlays synced; sources
frozen. Plans compare spread/window-with-probe/routerSF/history. No GPU submission
yet; wait for CPU success. Ordinary probe code is preprocessor-guarded, and the
observer only waits on the existing first accumulator barrier. Goal remains active.

Metadata_spread CPU7244419 COMPLETE3m50s: candidate/control ordinary+IKET register
gates,128 registers/24-byte stack/zero spills, and all four router warmups PASS.
GPU7244470 and afterok CPUanalysis7244471 submitted: spread/window-probe/routerSF/
history. Completed CPU success verified; no stale CPU dependency. Sources frozen;
no promotion or commit. Candidate16 metadata CTAs, helper16, senders17..24;
control8 metadata CTAs, helper8, senders9..16; both150 compute CTAs/eight senders.

Additional source fact: BLOCK_M=32, WG_BLOCK_M=16 and valid_m=16 in this fixed
balanced case. The second epilogue WG handles no valid rows, but still owns TMEM
empty-barrier credit; it cannot simply be dropped or treated as a free worker.
This may inform later role placement. Current test does not change epilogue credit
ownership. The earlier thought that both WGs handle active rows was incorrect.

## Metadata spread7244470 and next gather-loop test

GPU7244470 COMPLETE13m13s, analysis7244471 COMPLETE45s;96 exact+96 replay PASS.
Kernel mean spread40.4736/window40.7200/routerSF39.3216/history37.8464 us.
Spread firstMMA10.624/10.752/10.720 us, compute26.272/27.520/26.784; parent
firstMMA10.624/10.592/10.720, compute27.008/26.592/26.272. Metadata retirement
12.800/12.928/12.960 vs18.720/18.432/19.264. No best-path win/promotion.
Accumulator probes: parent1/2/4 CTAs observed ready before local startup ended,
with ready-observation→epilogue gaps up to1.600us. Spread removes those delays
(all gaps−.032..064us), but ordinary gain only.2464us. Epilogue admission was
a real local delay, not the dominant remaining loss. Fullchain spread67.555201/
window67.369601/routerSF67.244801/history94076.403809us; retain history outlier.
Evidence: receiver_validate/metadata_spread/comparison_7244470.json.

SASS: wide_metadata has4 static GATHER4 sites; first_window and metadata_spread
have1. Uniform gather operands remain intact, so this is lost loop unrolling,
not repeated operand conversion. Prepare split_k_loop: issue first4 stages,
wait phase3 once outside steady loop, explicitly unroll steady loop4. Second
arm uses relaxed GPU polling+uniform vote then one acquire for only that wait,
testing repeated CCTL.IVALL cost separately. Same readiness/proxy/lifetime rules,
16metadata/eight senders/150compute. Five arms loop/relaxed/spread/routerSF/history;
CPU gates before22min GPU comparison,120 exact+120 replay expected. Prepared,
not yet compiled/submitted. No maintained kernel changes or commit.

Exploration ideas remain useful, but direct st.async shared-cluster staging is
parked: CUDA13/PTX9 requires destination and mbarrier in the same CTA; it is not
a drop-in for the existing peer-destination/leader-barrier gather protocol.

Split_k_loop CPU7245094 submitted for both variants normal/IKET, register gates and four router warmups. Bounded overlays inherit metadata_spread headers; only main differs. Sources frozen; GPU not submitted.

Split_k_loop CPU7245094 COMPLETE3m58s: all four normal/IKET register gates and
four router warmups PASS,128regs24stack0spills. SASS now8GATHER4 sites (4+4),
uniform operands retained. Relaxed retry has no CCTL.IVALL; one remains after
exit before proxy fence. GPU7245184 and afterok analysis7245185 submitted,
loop/relaxed/spread/routerSF/history. Sources frozen;120exact+120replay expected.
No measured gain yet, no promotion.

Split_k_loop GPU7245184 FAILED1:0 after4m21s: loop32 ordinary passed initial
exact-output checks on all8ranks, then timed out at240s before replay completion
or profiler-range start (no Nsight file). Remaining arms not run; dependent
analysis7245185 cancelled. No timing result and no promotion. CPU checks do not
prove replay correctness. Next bounded diagnostic compares unchanged spread
control then same loop with host-stack timeout to locate the blocked phase.

Summary tool now pairs first-FC1 accumulator observations with each MMA pair;
uses later CTA observation, explicitly an upper bound. Checked real spread/window
traces and legacy traces lacking probes. Previous spread first-MMA→first-FC1-ready
medians6.784/6.688/6.592us vs window6.656/6.816/6.720: earlier worker retirement
did not shorten first-FC1 completion. No additional GPU instrumentation needed.

Diagnostic GPU7245366 submitted: parent spread32 then unchanged loop32, ordinary captures only,150s per-arm timeout/7min allocation. Optional90s per-rank Python stack capture through maintained smoke wrapper, no kernel changes. First job failure retained; no speculative broader run.

Diagnostic7245366 COMPLETE3m17s on nvl72010-T04/T05,16exact+16replay PASS.
Both parent and unchanged loop passed; no nonempty stack dumps. First stall
7245184 remains unexplained, not a proven kernel or infrastructure cause.
Ordinary mean/median loop40.5824/40.560 vs spread40.5504/40.480us, ranges
40.288–40.960 vs40.096–40.992. Fullchain68.044800 vs68.400000us. No unrolling
gain; do not spend full five-arm capture to promote this branch. Relaxed-poll
arm remains compiled but unmeasured. Next: first window1536B/12K stages vs
512B/4stages in10-stage pipeline, same total bytes and three publications.
Add matched direct remainder-wait markers on active gather warps, replacing
the two accumulator probes to keep marker budget. Test hypothesis that the
first window runs dry while remaining validation is still pending; not assumed.

Window_depth CPU7245542 submitted.1536B first window covers12K stages versus
512B/4 stages in10-stage pipeline, keeping total bytes/three releases fixed.
First validator uses6 independent vectors per half-warp lane rather than2;
remaining uses5 vectors per warp lane rather than7. Coverage check proves
first/remainder disjoint and covering all256 uint4s/row in both arms. Same16meta,
eight senders,150compute CTAs. Shared main retains original K loop.

Matched direct remainder markers replace two accumulator probes. Only active
gather warps0/1 mark (valid_m16→8rows per CTA→two4-row gather warps);300
locations expected. Earlier generic first-gather markers include inactive warps
2/3 as well, so they are not all necessary A issuers in this fixed case.
WaitStart is before phase3 acquire at k4/k12; Ready follows acquire, syncwarp,
proxy fence but precedes empty-stage wait. This is issuer wait, not an exact
MMA-stall measurement. Parser validates locations/pairing/order; shell syntax
and summary real-old-trace/paired-marker checks PASS. CPU gates pending; no GPU.

Window_depth CPU7245542 COMPLETE3m52s; all four normal/IKET gates and router
warmups PASS,128regs24stack0spills. Both ordinary variants retain one static
GATHER4 site and uniform operands. GPU7245584/afterok analysis7245585 submitted:
window1536/window512/routerSF/history,96exact+96replay expected. Sources frozen.
No timing result/promotion yet; active goal unchanged. Optional host-stack90s
logging retained for timeout diagnosis; no kernel injection or reset timing.

During window_depth7245584: ordinary window1536 mean39.3824/median39.376us
(range38.976–39.680) vs window512 mean40.6464/median40.672 (40.128–40.992),
10instances each, initial exact/replay checks pass. Preliminary−1.264us/3.11%;
controls and decomposition pending. Do not promote on incomplete capture.

Static follow-up: actual valid_m16 aligns to16, so each CTA gathers8 rows and
only warps0/1 issue4-row TMA. Warps2/3 still acquire metadata/readiness and wait
every stage while issuing no A. Prior active-SF tests gated on BLOCK_M16, so
BLOCK_M32 case was unchanged (docs7220325,7217034); this dynamic-row question
is distinct. No gather barrier arrival credit belongs to inactive warps in
current packed-SF/TMA scheme; coordinator/weight roles retain four arrivals,
MMA returns empty credits. Any skip must retain scheduler/phase advancement
and final dispatch+epilogue retirement join. Beware moving idle warps into
early global DispatchDone polling; a warp0-only acquire before the common
retirement join can carry lifetime ordering to other cleanup participants.
Not implemented/submitted; keep this axis separate from window-size test.

Prepared locally only `receiver_validate/active_rows`: dynamic aligned-row
ownership skips inactive gather polls/empty-stage waits, preserving scheduler
and pipeline phase advancement. Warp0 alone observes DispatchDone before
existing384-thread dispatch+epilogue retirement barrier, preventing inactive
warps from replacing stage waits with an early global-counter spin. No full/
empty barrier arrival credit is removed; warp3 TMEM allocation remains before
this branch. Control sparse first-gather markers restricted to the same active
warps for matched instrumentation. Await window_depth decomposition before
choosing base/building; no new CPU/GPU job.

## Window depth7245584: useful overlap confirmed

GPU7245584 COMPLETE12m33s on nvl72151-T01/T02; analysis7245585 COMPLETE52s.
96exact+96replay PASS, stderr empty;300 paired active-gather remainder locations
per sample. Evidence window_depth/comparison_7245584.json and run_audit_7245584.json.

| us | window1536 | window512 | routerSF | history |
|---|---|---|---|---|
| ordinary mean/median |39.3824/39.376|40.6464/40.672|39.9296/39.584|37.7408/37.776|
| ordinary range |38.976–39.680|40.128–40.992|38.720–43.232|37.184–38.336|
| first MMA metadata origin |11.072/10.560/10.880|10.976/10.336/10.976|11.776/11.680/11.744|10.080/10.080/9.888|
| first MMA selector origin |13.472/12.928/13.280|13.376/12.704/13.312|14.176/14.048/14.080|12.352/12.352/12.160|
| compute span |25.536/26.080/25.536|26.816/26.848/26.752|25.280/24.896/25.504|24.832/24.608/25.792|
| latest75-pair start |12.192/12.224/11.936|12.032/11.648/12.128|12.256/12.320/12.384|11.392/11.040/11.360|
| compute end metadata |36.608/36.640/36.416|37.792/37.184/37.728|37.056/36.576/37.248|34.912/34.688/35.680|

Remainder-wait medians1536=.512/.544/.576us vs512=2.016/1.760/1.760.
Its start is median.992us after the pair's firstMMA vs.032/.032/.064us in512:
more issued stages overlap remainder readiness. Do not equate issuer wait with
exact MMA-stall time. First MMA is essentially preserved; ordinary−1.264us/3.11%
and median compute−1.280us establish real benefit over512. Pair compute medians
24.224/24.288/24.128 vs25.600/25.216/25.664. Compared with routerSF, startup
advances~.864us median and compute is close (+.256us median), but ordinary
mean−.5472us includes router's43.232us outlier; median advantage only.208us.
Useful receiver-validation base, not a proven default replacement of best routerSF.
Fullchain68.697600/window51269.315200/routerSF67.948801us; historical96959.704590us
outlier retained separately. History still~1.64us ahead in ordinary kernel.

Proceed with active_rows on1536 base: omit no-data gather waits, preserve active
parallelism, carry retirement through warp0 and common join. Both arms include
an empty compiler memory barrier after the dispatch retirement join and matched
active-warp first-gather markers. No added GPU instruction/arrival. Prepared
plans compare active/control1536/routerSF/history; CPU gates next, no GPU yet.

Active_rows CPU7245953 submitted: normal/IKET candidate+matched control, register gates, four router warmups. Both overlays inherit validated window1536 dispatch. Sources frozen; no GPU submitted.

Active_rows CPU7245953 COMPLETE3m46s: all four ordinary/IKET register gates and
four router warmups PASS,128regs24stack0spills. Both ordinary arms retain one
GATHER4 site with uniform operands. GPU7246043 and afterok analysis7246044
submitted: active/control1536/routerSF/history,96exact+96replay expected.
Sources frozen, no default promotion or commit. New test retains the recovered
1536B initial-window overlap and isolates removal of no-data gather waits.

Window_depth broad-start evidence: by metadata+12us,1536B starts68/72/75 of75
CTA pairs vs512B74/75/73, routerSF30/32/15, history75/75/75. Early release is
broad, not a single outlier tile. Instrumented postcompute tails overlap across
arms (~4–5us); do not subtract instrumented boundaries from ordinary timings.

Active_rows7246043 ordinary preliminary: active39.0912/median39.088us (38.752–39.360)
vs matched1536control39.4016/median39.376 (39.040–39.776). Both exact/replay passed;
−.3104us/−.79%. Controls/decomposition still pending; no promotion yet.

Concrete next dependency if startup remains exposed: current metadata CTA packs
SF/indices, fences/joins and publishes phase1 before it starts first-window A
validation. Earlier split-SF/A-WG7242794 cut A validators8→4 warps and lost;
that does not test independent CTAs with all8 A warps retained. A bounded
proposal is16 SF/index CTAs plus16 A-validation CTAs (same150compute/eight
senders), separate expert-local SF/first-A/full-A bits in the existing u32.
Each producer adds its unique1/2/4 bit once per tile via existing async release
add; disjoint bits make addition safe even when A precedes SF. Consumers acquire
SF bit1 before indices/SF, A bit2 before first window, A bit4 at remainder.
Never use the current cumulative count thresholds with reordered publishers.
Both producers derive identical offsets from final counts; A need not wait SF
to validate/materialize. Keep all256 threads on each metadata CTA initially
to avoid changing warpgroup/barrier participation; SF has only128 useful lanes
for this shape. Total workers32+helper1+senders8=41; move sender/helper roles
and update DispatchDone/marker counts explicitly. No extra buffers or bytes,
but peak memory pressure, duplicate descriptor/count reads and CTA placement
change. Incoming A arrival could still limit gain; no NVLink latency claim.
Not implemented or submitted; await active_rows decomposition before proceeding.

## Active rows7246043: small kernel gain, broader early MMA

GPU7246043 COMPLETE12m50s nvl72173-T02/T16; analysis7246044 COMPLETE52s.
96exact+96replay PASS, stderr empty. Evidence active_rows/comparison_7246043.json.
Ordinary mean/median active39.0912/39.088, control1536 39.4016/39.376, routerSF
39.4208/39.280, history37.7984/37.856us. Active−.3104us/.79% vsparent and
−.3296us/.84% vsrouterSF; modest. Ranges active38.752–39.360/control39.040–39.776/
router38.944–40.224/history37.376–38.304. No default promotion/commit.

| us, three captures | active | matched1536 | routerSF | history |
|---|---|---|---|---|
| first MMA metadata |10.848/10.592/10.784|11.328/13.088/11.584|11.904/12.000/12.032|24.288/9.760/9.792|
| first MMA selector |13.248/12.960/13.216|13.888/15.840/13.952|14.336/14.400/14.432|26.528/12.032/12.256|
| compute span |25.472/25.504/25.504|28.832/26.336/25.856|24.832/25.056/25.184|25.024/25.088/24.448|
| latest75-pair start |11.872/11.712/11.808|15.264/14.112/13.120|12.480/12.640/12.672|25.888/11.264/11.424|
| compute end metadata |36.320/36.096/36.288|40.160/39.424/37.440|36.736/37.056/37.216|49.312/34.848/34.240|

Active starts75/75/75 pairs by12us, router7/2/0. Control instrumented samples
are noisier/later than prior window_depth captures despite stable ordinary
39.40 vs39.38us; do not claim a multi-us ordinary gain from those profiles.
History24.288us startup outlier retained. Active per-pair compute medians
23.904/24.224/24.352 vsrouter23.904/24.000/24.256 andhistory~23.26us.

Active remainder wait medians.992/1.312/1.344us vscontrol1.168/.576/.800.
Its start is consistently.544us after firstMMA versus~.99–1.02 in control's
last two samples; advancing the issuer can lengthen its observed wait while
MMA proceeds. Wait duration alone is not a speed metric. Active first-gather
locations4/pair (two perCTA), same as control; remainder300/sample validated.
Fullchain65.836801/control65.148800/router66.944001us; history92240.515137us
outlier. Focus remains ordinary kernel; no fullchain gain claim vsparent.

Keep active_rows as the next experimental base. Remaining source serialization:
SF/index packing precedes activation validation. Active metadata received~5.25–
5.70us, index scatter5.856–6.432, A validation7.616–9.664. Separate SF/index
and A-validator CTAs while retaining8 A warps (prior split WG used only4).
Use independent SF/firstA/fullA bits1/2/4 in existing arrival word, one unique
release-add publisher perbit; consumer masks each bit, so either plane may
finish first. Same buffers/bytes,32metadata+helper+8sender roles within150CTAs.
Protocol/source implementation next; no new submission yet.

Metadata_roles source and bounded overlays prepared; CPU7246642 submitted.
Candidate32 metadata CTAs: SF/index0..15, A16..31, helper32, senders33..40;
150compute/eight senders unchanged. Both roles retain256 participants/barrier3,
A retains eight warps. Unique SF/firstA/fullA release-add bits1/2/4 replace
cumulative thresholds; all six publication orders preserve independent bits.
Coordinator acquiresSF; active gather acquiresSF/firstA thenfullA atK12, proxy
fences intact. No cross-plane producer join, new allocation or payload bytes.

Profiler now accepts distinct SF/index and A-validator CTA counts, each default
to total metadata CTAs for old runs. Candidate TopkReceived/MetadataFromProducer32,
IndexScatter16,ValidateStart/FullPayloadIssue16,Start/End41,InputsReady9.
Shell syntax PASS. Plans compare roles/validatedactive/routerSF/history;96exact
+96replay expected after CPU gates. Source frozen; GPU not submitted.

Metadata_roles CPU7246642 COMPLETE3m43s: all four ordinary/IKET reachable
register gates and four router warmups PASS,128regs24stack0spills. Ordinary
candidate/control retain one GATHER4 with uniform operands. GPU7246716 and
afterok analysis7246717 submitted, roles/active/routerSF/history,96exact+96replay
expected. Sources frozen, sameEP8 and preparation outside timing. No promotion.

Bounded follow-on audit while7246716 runs: same-CTA active_rows7246043
TopkReceived→IndexScatter median .800/.752/.784us (range .576–.960),
IndexScatter→PayloadValidateStart2.272/1.840/2.304us. Evidence
active_rows/prefix_dependency_7246043.json. These include loads/joins, not
isolated barrier cost. Current split targets the larger second dependency.
Prefix path has only16 experts but256 participants: route acquire→join,
histogram→join, eight warp sums→join, offsets→join. Possible later delta:
warp0 acquire/count-load and 32-lane exclusive prefix, one256-thread handoff
to readers; other warps wait only for the finished shared offsets. Keep
SizesReady early and capacity total correct; for >32 experts process32-wide
chunks with carry. This removes shared warp-sum exchange and redundant joins
without duplicating global loads or reducing A workers. Not implemented;
await current comparison before another branch.

Metadata_roles7246716 ordinary preliminary: roles38.4128/median38.288us
(range38.080–39.488), activeparent42.2272/42.224(39.296–44.416),
routerSF39.6480/39.504(39.168–40.256), history37.9776/37.968(37.632–38.720).
Ordinary ten instances/arm;32exact+32replay checks PASS across four arms.
Parent is substantially slower/noisier than prior39.0912us, so don't claim
3.8us repeatable gain vsparent. Split is1.2352us faster than same-runrouter
and .4352us behindhistory. Fullchain64.041601/68.665601/66.729601us,
history95110.418701us outlier retained. Detailed captures still running,
no compute/earlier-MMA claim or default promotion yet.

## Independent SF/A workers7246716: earlier broad MMA, ~history compute

GPU7246716 COMPLETE13m10s nvl72106-T12/T14; analysis7246717 COMPLETE47s.
96exact+96replay PASS, stderr empty. Evidence metadata_roles/comparison_7246716.json,
run_audit_7246716.json and role_overlap_7246716.json. FixedEP8/rows32, reset outside
kernel timing; all8 A-validator warps retained. No default promotion/commit.

| us, three captures | roles32 | activeparent | routerSF | history |
|---|---|---|---|---|
| ordinary mean/median |38.4128/38.288|42.2272/42.224|39.6480/39.504|37.9776/37.968|
| ordinary range |38.080–39.488|39.296–44.416|39.168–40.256|37.632–38.720|
| first MMA metadata |10.080/10.400/9.792|10.816/10.656/11.488|12.160/11.744/11.680|51.712/10.016/9.920|
| first MMA selector |12.512/12.768/12.256|13.184/13.088/13.920|14.592/14.208/14.016|53.952/12.256/12.192|
| compute span |25.280/24.896/24.864|25.088/25.728/25.152|25.088/25.056/24.768|24.608/24.448/25.504|
| latest75-pair start |10.880/11.136/10.688|11.712/11.712/12.384|12.832/12.480/12.352|53.184/10.944/11.584|
| compute end metadata |35.360/35.296/34.656|35.904/36.384/36.640|37.248/36.800/36.448|76.320/34.464/35.424|

Kernel −1.2352us/3.12% vs same-runrouterSF; .4352us/1.15% behindhistory.
Parent ordinary42.2272 is anomalously slow versus prior39.0912; do not claim
repeatable3.8144us parent gain. New38.4128 is .6784us below previous experiment's
bestactive mean, a cross-run comparison only. Historical51.712us startup outlier
and95110.418701us fullchain outlier retained. Ordinary fullchain roles64.041601,
active68.665601,router66.729601us. Kernel remains primary acceptance metric.

Direct dependency evidence: A validation starts6.496–6.592/6.144–6.720/
6.208–6.496us vsparent8.256–9.792/7.680–8.768/8.288–9.760. All16 A owners begin
before corresponding SF-owner retirement in every capture, median overlap
1.728/1.856/1.776us (interval includes SF retirement, not isolated SF instruction
time). New fullA issue9.824–11.072/10.048–11.040/9.472–10.784us vsparent
11.328–13.152/10.944–12.736/11.776–13.088. Marker counts32metadata,16SF,16A,
41retirements and300remainder intervals/sample passed.

Bymetadata+11us roles75/53/75 pairs vsparent10/24/0 androuter0/0/0; by12us
roles75/75/75,parent75/75/62,router0/40/41. Remainder wait medians.544/.384/.416us
vsparent1.312/1.120/1.056. Per-pair compute medians24.192/24.000/23.680us vsparent
24.128/24.000/23.872,router24.032/24.096/24.000,history23.712/23.008/23.200.
Earlier release is broad; ~25us aggregate compute is preserved. Keep roles as
next experimental base, with prioractive andhistory retained. Prepared-buffer
and materialization capacity contracts unchanged; not a general default switch.

Next bounded source opportunity remains warp0 count/prefix construction with
one256-thread handoff, removing the currently redundant eight-warp prefix
joins without reducing validation workers. Its measured parent route-acquire→
index-start envelope is ~.75–.80us, not entirely barrier cost. Keep32worker roles,
independent ready bits, payload volume,1536B window and150compute CTAs fixed.
Normal controls in the next comparison must revisit parent variability before
any broader speedup claim. No next source edit/job submitted yet; goal active.

Prefix_warp source prepared and CPU7247279 submitted. Parent metadata_roles7246716.
Warp0 loads padded counts and scans32-lane chunks with carry; publishes same
SizesReady after count loads, writes offsets and capacity total, then one256-thread
handoff. Initial route-acquire256join retained. Removes two prefix256joins and
cross-warp shared-sum exchange; all8 SF/A workers and32metadata roles unchanged.
Only dispatch header differs; candidate/main and control headers link to validated
metadata_roles. Same bits1/2/4,1536Bwindow,8senders/150compute, bytes and reset
outside timing. Four deterministic16-expert host prefix fixtures PASS; no extra
GPU shapes. CPU compiles ordinary+IKET candidate/control, reachable register
gates and four router warmups. Frozen source; no GPU submitted. Planned four
arms prefix/roles/active/history also recheck prior noisy active ordinary result.

Prefix_warp all four ordinary/IKET reachable register gates PASS,128regs24stack
0spills. Candidate/control ordinary GATHER4 retains uniform operands. GPU7247368
submitted afterokCPU7247279 (remaining router warmups); analysis7247369
afterokGPU. Four arms prefix/roles/active/history; fixedEP8,96exact+96replay
expected. Frozen source, no default promotion; timing results pending.

Prefix CPU7247279 COMPLETE3m43s; all gates and four router warmups passed.
GPU7247368 confirmed RUNNING nvl72145-T11/T12, analysis7247369 dependency queued.
Source frozen. Previous goal turn classified progress (implemented prefix delta,
CPU register gates passed, dependent comparison submitted).

Follow-on static audit (not implemented): metadata_roles ordinary SASS shows
SF-bit retry at0x5740 LDG.E.STRONG.GPU→0x5750 CCTL.IVALL→bit test/backedge.
SizesReady similarly0x5050→0x5060. Gather index loads follow the SF acquire
in inspected compiled code; no observed ordering failure. ptx::ld_acq source
wrapper lacks compiler memory clobber, so any new helper should explicitly
order following generic loads. Old split_k_loop relaxed remainder arm was
compiled but never measured; its unroll/control result doesn't test polling alone.

Candidate design after prefix result: only compact SF/firstA/fullA ready-bit
waits, including coordinatorSF. Initial acquire fast path; if unready, warp-
uniform relaxedGPU retries followed by a checked acquire before consumption.
Keep final acquire/proxy fence and source lifetime unchanged. Already-ready
flags pay no redundant relaxed-load round trip (important now A often precedes
SF). Matched strong-only helper with same warp vote/compiler ordering isolates
retry mode; compare frozen best andhistory too. Inspect GATHER4 uniform operands
and spills. This is a hypothesis about retry/cache pressure, not a measured
explanation of publication→observation delay; outgoing release issue/end markers
do not prove exact readiness/completion or isolated invalidate cost.

Readiness_poll source prepared and CPU7247546 submitted while prefix GPU captures
continue. Parent dispatch is frozen prefix_warp; only new main helpers differ.
Candidate and acquire-only control differ solely kRelaxedReadyRetry constexpr.
Both perform initial checked acquire with explicit compiler memory clobber and
full-warp any vote. Candidate enters relaxedGPU retries only if unready, then
returns to checked acquire; no relaxed observation authorizes payload/index use.
Changes gatherSF/firstA/fullA and coordinatorSF waits only; SizesReady, metadata,
combine, proxy fences and all worker counts remain fixed. Bareprefix andhistory
are other planned arms to separate helper overhead from retry choice. No new
GPU submission until prefix's full verification and these four register gates/
warmups pass. Sources frozen. Prefix ordinary preliminary38.3744 vsroles38.6176us
(overlapping ranges); detailed result still pending, not a robust gain claim.

Prefix7247368 ordinary preliminary all four arms: prefix38.3744/median38.416us
(38.080–38.784), roles38.6176/38.544(38.048–39.360), active39.3696/39.536
(38.560–40.096), history37.9968/38.080(37.600–38.400). Prefix−.2432us/.63% vs
roles with overlapping ranges; .3776us/.99% behindhistory. Active returned to
~39us, supporting prior42.2272 as noisy; separateworkers improvement~.752us
in this comparison, not3.8us. Detailed captures still running.

Readiness CPU7247546 COMPLETE3m37s; four register gates+four router warmups PASS.
128regs24stack0spills; one uniform GATHER4 in each ordinary variant. SASS
relaxed retry has LDG+bit test+VOTE/backedge without CCTL; initial and final
checked acquires retain CCTL.IVALL. Strong control retries retain CCTL eachtime.

Added independent fifth arm sf_warp32 beforehistory; CPU7247670 submitted for
ordinary/IKET/register gates +two router warmups. Only dispatch index/score
loop ownership rotates (tid+32)%256: fixed16 rows use warp7 instead ofwarp0,
so warp0..3 SF packing can overlap metadata loads/stores. Full cyclic mapping
proven bijective; no reduction in work parallelism for larger tiles and no
new join/buffer/bytes/readiness protocol. Same256-thread publication join.
SF arm uses bareprefix main, no polling helper; results isolate each idea.
Next allocation planned relaxed/acquire/prefix/sf_warp/history,22min,120exact+
120replay. No GPU submission until prefix verification and both CPU gates pass.

Prefix GPU7247368 COMPLETE13m17s,96exact+96replay PASS, stderr empty;
analysis7247369 pending/awaiting allocation at lastcheck. Fullchain64.544001/
66.355200/68.160000us; historical94346.453857us outlier retained.
SF CPU7247670 COMPLETE2m15s: ordinary/IKET gates+two warmups PASS,128regs24stack
0spills, uniform GATHER4. All follow-on sources frozen and compiled. Five-arm
GPU still not submitted pending prefix decomposition. SF arm's IndexScatter
remains warp0 phase-start; actual warp7 metadata start is not separately timed.
Variant interpretation explicitly records this; no false direct overlap timing.

## Prefix7247368: shorter setup, small ordinary difference, no MMA advance

GPU7247368 COMPLETE13m17s nvl72145-T11/T12; analysis7247369 COMPLETE51s.
96exact+96replay PASS, stderr empty. Evidence prefix_warp/comparison_7247368.json,
run_audit_7247368.json, setup_intervals_7247368.json. Same fixedEP8 and reset
outside kernel timing. No default promotion/commit; roles remains the stronger
validated optimization, prefix retained as a controlled experimental branch.

| us, three captures | prefix | roles parent | active | history |
|---|---|---|---|---|
| ordinary mean/median |38.3744/38.416|38.6176/38.544|39.3696/39.536|37.9968/38.080|
| ordinary range |38.080–38.784|38.048–39.360|38.560–40.096|37.600–38.400|
| first MMA metadata |10.048/10.400/10.432|10.368/10.432/10.016|10.368/10.656/10.464|81.376/9.984/9.792|
| first MMA selector |12.448/12.736/12.864|12.768/12.864/12.384|12.800/13.152/12.896|83.616/12.256/12.096|
| compute span |25.504/25.152/25.504|24.832/25.440/24.992|25.664/25.152/24.960|24.704/24.928/25.376|
| latest75-pair start |11.168/11.264/11.232|11.200/11.648/10.880|11.456/11.552/11.648|82.336/10.880/11.456|
| compute end metadata |35.552/35.552/35.936|35.200/35.872/35.008|36.032/35.808/35.424|106.080/34.912/35.168|

Same-CTA TopkReceived→IndexScatter medians prefix.688/.576/.592us vsroles
.800/.768/.800; TopkReceived→A-validation .880/.832/.848 vs1.040/1.040/1.024.
The local setup shortened ~.1–.2us, but firstMMA median10.400vs10.368 unchanged
and compute median25.504vs24.992 slightly longer, with overlapping sample ranges.
Remainder waits .384/.416/.384vs.352/.288/.416. By11us prefix58/42/36pairs vs
roles50/46/75; everypair starts by12us. No demonstrated broader/earlier-MMA win.
Ordinary−.2432us/.63% mean and−.128us median; ranges overlap. Active control
returns39.3696us, so prior42.2272 is not a repeatable parent baseline; independent
SF/A workers still save~.752us vsactive in this comparison. Historical81.376us
startup and94346.453857us fullchain outliers retained; no filtered headline gain.
Fullchain prefix64.544001/roles66.355200/active68.160000us. Do not mix instrumented
compute spans with ordinary kernel duration or claim prefix solved dispatch.

Next readiness_poll GPU7247900, afterok analysis7247901 submitted. Both CPU
preflights COMPLETE:7247546(3m37s) relaxed/strong helpers,7247670(2m15s) SF warp
rotation. Six normal/IKET gates and six router warmups PASS,128regs24stack0spills,
uniform gather retained. Five arms relaxed/acquire/bareprefix/sf_warp/history,
22min,120exact+120replay expected. Prefix held fixed across three new/control
arms so retry mode and SF/index ownership are separately testable. All source
frozen; no default promotion. Best roles path preserved if prefix's slightly
longer compute persists. This turn made progress: completed prefix comparison,
identified remaining SF-warp serialization in source and cache-invalidation
retry loop in SASS, prepared and compiled isolated alternatives, submitted
matched timed comparison. Goal active.


## Readiness comparison7247900: ordinary result and report workflow

Five ordinary arms complete,120exact+120replay PASS,stderr empty. Preliminary
kernel mean/median us: relaxed38.4832/38.464,acquire38.4032/38.608,
bareprefix38.2784/38.304,SF-warp38.1120/38.048,history37.9744/37.792.
Relaxed retry has no ordinary benefit. SF index ownership rotation is only
.1664us mean faster thanprefix with overlapping ranges; detailed acceptance
awaits analysis7247901. Fullchain67.155201/66.028801/68.214400/66.060801/
97647.613525us respectively; historical outlier retained. No default promotion.

Reusable summarize_megamoe_latency.py now supports --markdown-output for
side-by-side ordinary timing plus one detailed metadata-origin table per
capture. Adds existing source-load/metadata-acquire/pull issue/completion marker
bounds, SF-gather starts and per-pair start spread. Preserves all captures,
actual-MMA vsoperand-ready distinction, selector origin, variant ownership notes,
and outgoing/incoming scope. No injected/new GPU markers. Regenerated real
four-arm prefix7247368 JSON+Markdown as validation; report does not sum phases
or mix instrumented spans with ordinary timing. README documents optionalflag.

Re-read MEGAMOE_COMM_EXPLORATION.md peruser: independent issuer/worker ownership
and first-consumable windows already informed current SF/A split and SF-warp
arm. Avoid unchangedearlyPDL (failed7236911), genericcompute-gather, and simply
adding sender CTAs. Potential next bounded axis is A-validator row ownership
or remaining SF/index publication dependency, selected aftercurrentdecomposition.
No new protocol/experiment yet. Removing receiver materialization remains
unproven: generic Lamport observation alone is not an established cross-proxy
proof for later TMA reread. Keepexactobserved-value copies/proxyfences; PTX9.4
new relaxed bulk/fabric facilities require separate toolchain/architecture and
ordering investigation, not a drop-in optimization.


## Readiness7247900 complete: SF/index ownership advances broad MMA release

GPU7247900 COMPLETE18m08s nvl72150-T05/T17; analysis7247901 COMPLETE1m26s.
120exact+120replay PASS,stderr empty. Evidence readiness_poll/comparison_7247900.json
and .md (all five arms/three captures),run_audit_7247900.json,
sf_intervals_7247900.json. SamefixedEP8; reset excluded; no defaultpromotion.

| us,three captures | relaxed | acquire helper | prefix parent | SF warp | history |
|---|---|---|---|---|---|
| ordinary mean/median |38.4832/38.464|38.4032/38.608|38.2784/38.304|38.1120/38.048|37.9744/37.792|
| ordinary range |38.112–38.912|37.792–38.720|37.760–38.656|37.504–38.752|37.312–38.848|
| first MMA metadata |10.592/10.112/10.496|10.368/9.824/10.368|10.592/9.792/10.304|10.080/9.664/10.016|10.016/10.112/9.856|
| first MMA selector |12.896/12.480/12.832|12.704/12.160/12.832|13.056/12.160/12.736|12.448/12.096/12.352|12.288/12.384/12.128|
| latest75-pair start |11.200/11.232/10.976|11.136/11.104/11.200|11.392/11.072/11.264|10.592/10.560/10.336|11.392/11.008/11.552|
| compute span |24.768/25.312/24.608|25.120/25.344/25.664|25.056/25.536/25.120|25.056/25.504/24.864|25.376/24.608/24.896|
| remainder wait median |.448/.224/.288|.320/.384/.352|.192/.320/.256|.704/.528/.496|—|
| compute end metadata |35.360/35.424/35.104|35.488/35.168/36.032|35.648/35.328/35.424|35.136/35.168/34.880|35.392/34.720/34.752|

SF rotation removes the fixedcase warp0 index→SF serialization: SF phase-start
through same-CTA retirement medians1.568/1.536/1.520us vsparent2.144/3.040/1.936.
This includes publication/retirement; actual warp7 index-start is not timed.
All75pairs startby11us everySFcapture vsparent51/64/58; by10.5us67/60/75 vs0/43/14.
FirstMMA median10.016 vsparent10.304; lateststart median10.560 vs11.264.
Compute median25.056 vs25.120parent and24.896history; no compute regression.
Perpair compute-span medians24.064/23.776/24.000 vsparent24.128/24.064/24.320,
history23.808/23.424/23.008. Broader earlierMMA is observed, but ordinarygain
only.1664us/.43% mean (.256us median), overlappingranges. .1376us/.36%behind
historymean; do not callnoise-free e2e win ordefaultpromote. PreserveSFarm as
useful experimentalbase and keeproles/prefix controls. Relaxedretry+.080us
vsacquire and+.2048vsprefix, no benefit; do not adoptpollhelper.

EarlierSF exposes activation: firstpairSFobserved8.384–8.416/8.352–8.512/
9.280–9.344us, Aready9.536–9.568/9.088–9.216/9.664–9.696. Sequential consumer
observations do not isolate pure network wait, but identify nextdependency.
Remainder issuer waits increase asgatheradvances; compute~25us preserved.
Nextbounded investigation: release activation for eachcomputeCTA's consumed
row subset independently, or overlap remaining per-warp row validation. Must
preserve exactword validation/materialization, arbitrarypartialtile ownership,
unique readiness-bitpublishers, proxyfences, andgenerationretirement; no new
GPUrun/code yet. Avoid unprovedliveinbox TMA reread and oldgenericgather path.

Fullchain67.155201/66.028801/68.214400/66.060801/97647.613525us retained; lastis
historicaloutlier, not kernel time. No GPUjobs remain for thisgoal. Goalactive.


## Activation worker split prepared; CPU7248647

Previousgoalturn madeprogress: completed7247900 and found broadMMA advance from
SF/indexwarp separation. Now targetactivation readiness afterSF. Two CTA halves
shareoneMMA, so per-CTAhalfready alone cannotreleasepairedMMA. Instead test32A
validatorCTAs (two disjointrowowners/expert) vs16A SF-warp parent, withhistory.
16SFCTAs/eightsenders/150computeCTAs unchanged; totalstartup participants57.
Eachfixed16-row expert splits8+8: onefullwarp/row,3independentuint4loads/lane for
first1536B,5remainderloads; removes6-vectorfirstwindow and two serialremainder
rows/warp inparent. ExtraAworkers alsoaddcount/prefixreads and mayincrease
memory/schedulingpressure; comparefullkernel, all75MMAstarts, remainderwaits,
compute~25us, andsameexact/replay checks beforeacceptance.

Readinessusesexistingu32 boundedcounterfields: oneSFadd1; eachAowneradd2first,
add8full, exactlyoncepertile. Carrybits4/16 requirebothhalves; lowpartialbits
cannotauthorizeconsumption. Emptyhalfstillcontributes. Exactvalidatedword
materialization,localproxyfences/256joins,consumeracquire+proxyfence,
retirement57 andoffcriticalreset preserved. CPUrowcoverage1..32 andall120
publicationorder permutationsPASS. No additionalGPUshapes. Frozen scratch
activation_workers/main.cuh+dispatch.cuh+patches+contract/configs; defaultunchanged.
CPU7248647 submitted ordinary/IKET/registergates andtwo routerwarmups. GPUplan
three arms14min/72exact+72replay, notsubmitted untilgatespass. Goalactive.

Activation_workers CPU7248647 COMPLETE: ordinary/IKET128regs24stack0spills,
bothregistergates+two routerwarmupsPASS; singleuniformGATHER4 retained.
GPU7248773/afterokanalysis7248774 submitted, three arms split_a32/SFparent/history,
14min72exact+72replay expected. Sourcefrozen; gather masks consume bothAowners,
notpartialhalf. No defaultpromotion. Awaitordinarykernel plusall75MMAstarts,
compute/remainderwaitdecomposition beforedecision.


## Activation7248773 ordinary preliminary; elected publication CPU7248894

Priorgoalturn wasprogress: prepared/provedsplitAownership,passedCPU7248647
compile/register/router gates,submittedGPU7248773+analysis7248774. CurrentGPU
stillrunning. Ordinary means/medians us: splitA38.4256/37.952(range37.760–40.480),
SFparent38.0800/38.096(37.632–38.592),history38.0352/37.984(37.312–39.040).
Candidatemean+.3456 butmedian−.144 vsparent; retainoutliers, no winclaim before
decomposition. Firstcandidate8exact+8replayPASS; fullrunstillpending.

StaticSASSreview confirmscandidatefirst3SYSvectorloads and5remainderloads.
It also exposes remaining route descriptor reload andsourceaddressdecode after
firstwindowpublication before remainderloads (LDG1fa0→decode1fd0..2030).
Potential pointerreuse savesdependency butmayincreaseregisterlifetime/arraystate
forlargerpartialtiles; notimplemented pendingresult. Don't redo16-vectorfullrow
batchwhichlostcompute. A halfalone cannotstartpairedMMA.

Separately CPU7248894 submitted elected_publish: six localGPU asyncpublication
sites onfrozenSFparent change tid0/lane0 toexplicit elect_one_sync inwarp0.
Onecontribution,identicaladdresses/values/fences/joins/roles/protocol. Hypothesis:
removecompilerELECT/reconvergence loop aroundsinglelaneUREDGR (activationSASS
1df0..1e90), notchangecompletionsemantics. PTXred.asyncremainsperthread generic
proxyrelease; doNOTbroadcastaddtoall32lanes. CPUordinary/IKET/registerinspection
only; routerwarmups/GPU deferreduntilmachinecodebenefit established. No default
changes. Source/patch/commands/provenance inelected_publish. Goalactive.

Elected CPU7248894 COMPLETE ordinary/IKET128regs24stack0spills,registergatesPASS.
SASSall6UREDGR siteslosepostpublicationloopbackedges; totalBRA.U.ANY9→3,
ELECT25→31. Fences andsixreductionsremain, so fewerloopsisnotaloneaspeedup.
RouterwarmupCPU7248987 submitted; noGPUyet. PTXsemantics reference:
https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async
Electionstillsinglethread; broadcastingaddto32laneswouldchangeprotocol.

Summarizer nowjoins existingfirst-stageA/SF/index/gatherobservations toeachof75
MMApairs, andexports latestAminuslatestSF andlastgathermarkertoMMA intervals.
Markdownlabelsconsumerobservations explicitly; no pure-network/operand-stall
claim. Validatedagainstallreadiness7247900captures: SF-arm Aobservationlaterin
75/75pairseachcapture,mediangap.608/.544/.576us; gathertoMMAmedian.288each.
Prefix74–75pairsAlater,gap.544. Reportregenerated withallcaptures/outliers.


## Activation7248773 complete: remainder faster, no startup/kernel win

GPU7248773 COMPLETE8m59s nvl72022-T06/T14; analysis7248774 COMPLETE49s.
72exact+72replayPASS,stderr empty. Evidence activation_workers/comparison_7248773
.json/.md andrun_audit_7248773.json. Alloutliersretained; no defaultpromotion.

| us,threecaptures | splitA32 | SFparent16A | history |
|---|---|---|---|
| ordinarymean/median |38.4256/37.952|38.0800/38.096|38.0352/37.984|
| ordinaryrange |37.760–40.480|37.632–38.592|37.312–39.040|
| firstMMAmetadata |10.080/9.984/10.080|10.048/9.632/9.760|98.976/10.080/9.856|
| firstMMAselector |12.416/12.576/12.416|12.352/11.968/12.096|101.248/12.320/12.096|
| latest75pair |10.688/11.232/10.624|10.848/10.304/10.496|99.744/10.944/11.264|
| computespan |24.768/25.216/24.960|24.768/25.408/25.184|24.288/24.704/25.216|
| remainderwaitmedian |.224/.192/.192|.544/.800/.432|—|
| computeendmetadata |34.848/35.200/35.040|34.816/35.040/34.944|123.264/34.784/35.072|

FullAcontributionissue ranges8.320–9.376/7.776–9.984/8.384–9.024 vsparent
10.048–11.200/9.728–10.784/9.760–10.752us. Candidate32markers areeachhalf's
contributionissue, not fulltilecompletion. Remainderissuerwaits shortened,
but firstMMAmedian+.320us andlatestpairmedian+.192us. By11us75/71/75pairs
vsparent75/75/75. Mean kernel+.3456us whilemedian−.144us; upperoutlier40.480
retained. Notconvincinggain; keep16Aparent. Compute remains~25us so no headline
compute regression, but startup/ordinaryacceptance unmet. Perpaircomputemedian
23.744/23.968/23.808 vsparent23.744/23.744/23.776,history23.104/23.360/23.264.
AobservedlaterthanSF inall75pairsforbothpusharms. ExtraAworkers alsochange
placement/countreads/queuepressure; tracesdo notuniquelyidentifyregressioncause.
Fullchain67.408000/66.009601/97889.105225us; historicalstartup98.976us andfullchain
outliersretained. Do not discard themorcompareinstrumentedspan toordinarykernel.

Next electedpublication remainsonSFparent16A. CPU7248894 COMPLETE53s;
routerwarmup7248987 COMPLETE2m05s bothPASS. Sixpublisherbackedgesremoved,
128regs24stack0spills; samefences,protocol,workersanduniformgather. NextGPU
three-arm elected/SFparent/history planned, readytosubmit. No sourcepromotion.

Elected GPU7249167/afterokanalysis7249168 submitted afterallCPU/previousGPU
gates COMPLETE. Three arms elected32/SFparent/history14min72exact+72replay
expected. Sourcesfrozen; no source/defaultpromotion. Goalactive.


## Arithmetic indices / independent A prepared; CPU7249293

Priorgoalturn wasprogress: completed activation7248773 negativecomparison and
submitted elected7249167. CurrentelectedGPUconfirmedRUNNING on nvl72099-T03/T06;
analysis7249168 dependent. No rerun/resubmit.

Sourceaudit found an unnecessarydependency preserved fromsource-index gather:
receiverA ismaterialized contiguously, butgatherstillwaitsSF/indexbit1 andloads
HBMindices whosevalues equal materialized_base+pool_block*BLOCK_M+peerrow.
Schedulerprefix sumsceil(count/BLOCK_M), exactlydispatchalign(count,BLOCK_M)
rowprefix. Sizes/countreadiness alreadyheld beforegather schedule. Ownrank
num_tokens andmetadataslot define samematerializedbase asvalidator.

New arithmetic_indices mainonly variants on frozen16A SFparent: address32
replacesHBMindexloads byformula, retainingSFbitwait; independent32 additionally
removesAwarpSFwait. CoordinatorstillacquiresSFbit1 beforeSF TMA. Afirst/full
acquires,proxyfences,invalidrowOOBindices,Kloop,threads,senders,bytes/reset and
materialization remainunchanged. SF/indexwritersretained toisolateread-side
change. Thisneverreads liveinbox throughTMA. No needseparatehalfreadyprotocol.
Fourarms independent/address/SFparent/history isolateindexmemorycost from
SF→Aserialization, planned18min96exact+96replay. NoGPUsubmitted untilcurrent
comparisoncomplete andnewCPUgates/warmupsPASS.

11520deterministic addresscasespassed across partial/empty/multiblockexpert
counts,8ranks,2generations,matchingwrittenindices andOOBguards. CPU7249293
submittedfourordinary/IKET/registergates andfourrouterwarmups. Sourcesfrozen.

Correctedreportlabel: MegaIndicesReady is index-pathmarker, not index-load
completion. Inexistingmaterializationparent it followsSF/indexacquire and
precedesHBMloads; arithmeticarm keepsmarkerbeforeaddresscalculations, with
independentarm notclaimingSFready. Maintainedprofiler supports explicit
index_path interpretation; newfrozenprofiler/caseenv carryscope. No newGPU
marker/injection. Summarizerusesvariant-scope label; allpriorrawtimingsretained.
No defaultpromotion/commit. Goalactive.

Elected7249167 ordinarypreliminary: electedmean38.0352/median38.144us vs
SFparent37.9936/37.968 andhistory37.9840/37.936. Ranges37.664–38.368,
37.664–38.592,37.440–38.688. Noordinarygainfrompublicationloopremoval; detailed
capturespending. Arithmetic7249293 allfourregistergatesPASS,128regs24stack
0spills; bothordinaryarmsoneuniformUTMALDG.GATHER4. Routerwarmups inprogress.
No newGPUsubmitted; sourcefrozen. Indexread-sidechange leavesSF/indexstores
intact anddoesnotchangebufferownership or readliveinbox.

ArithmeticCPU7249293 COMPLETE3m25s allfourregistergates+fourrouterwarmupsPASS;
128regs24stack0spills,uniformgatherretained. ElectedGPU7249167 COMPLETE9m21s,
72exact+72replayPASS,stderr empty;analysis7249168running. Fullchain65.110401/
64.297600/103169.909668us retained, lastishistoryoutlier. NextGPUstillpending
fullanalysis. Newcaseinterpretations explicitlycorrect16SF/16Aownership and
independentA'sabsenceofSFwait; no changetorunning/completedcapturefiles.

ArithmeticGPU7249583/afterokanalysis7249584 submitted,fourarms independent32/
address32/SFparent/history18min96exact+96replay expected. CPU7249293 fullyPASS
andpreviousGPU7249167 verified72exact+72replay,stderr empty. PreviousCPUanalysis
7249168 confirmedRUNNING butslow/nooutputat1m36s; prematurecompactJSONcollection
returnedmissingfile(rsync23), notanalysisfailure. Do notrestart. Changedplan:
newcomparisonusesunchangedverifiedSFparent anddoesnotdependonelectedanalysis,
so submitindependentGPUwhileCPUanalysiscontinues ratherthanidle. No source
mutation forrunningjobs. Needcollectandinterpretelecteddecompositionwhenready.


## Elected7249167 complete: no publication-loop performance gain

GPU7249167 COMPLETE9m21s nvl72099-T03/T06; analysis7249168 COMPLETE3m49s.
72exact+72replayPASS,stderr empty. Analysiswas slowbutactive; prematurecollection
failedmissingfile, laterallfilescollected successfully. No jobrestart. Evidence
 elected_publish/comparison_7249167.json/.md andrun_audit_7249167.json.

| us,threecaptures | elected | SFparent | history |
|---|---|---|---|
| ordinarymean/median |38.0352/38.144|37.9936/37.968|37.9840/37.936|
| ordinaryrange |37.664–38.368|37.664–38.592|37.440–38.688|
| firstMMAmetadata |9.952/9.600/9.696|9.376/9.472/9.888|21.152/10.112/9.856|
| firstMMAselector |12.288/11.936/12.192|11.776/11.808/12.416|23.392/12.384/12.096|
| latest75pair |10.592/10.304/10.528|10.560/10.304/10.528|22.592/10.848/11.072|
| computespan |24.960/25.120/25.504|24.928/25.664/25.216|25.184/24.352/25.440|
| remainderwaitmedian |.480/.896/.864|.544/.640/.688|—|
| computeendmetadata |34.912/34.720/35.200|34.304/35.136/35.104|46.336/34.464/35.296|

Sixpublicationloopbackedgesremoved butordinarymean+.0416us/median+.176us,
firstMMAmedian+.224us; latestpairmedianidentical10.528. Bothpusharmsall75by11us
allcaptures,compute~25us. No gain; donotadoptjustforSASSsimplification. Retain
unchangedSFparent. Perpaircomputemedian23.936/23.936/24.064 vsparent23.808/
24.384/24.064,history23.136/23.360/23.680. All75pairs AlaterthanSF; observation
medians1.024/.480/.928vs.800/.512/.576us.
Historical21.152us startup and103169.909668us fullchainoutliersretained.
Fullchain65.110401/64.297600/103169.909668us; notkernelmetric.

Arithmetic_indices GPU7249583 nowRUNNING nvl72112-T08/T12;analysis7249584dependent.
CPU7249293 COMPLETE3m25s allfourgates/warmups128regs0spills/uniformgatherPASS.
Fourarms independent/address/SFparent/history,18min96exact+96replay expected.
Addressesderivedfromexistingcontiguousmaterializedlayout; eliminateHBMindexread
andseparatelytestremovingAwarpSFdependency whilecoordinatorSFwaitremains.
No defaultpromotion/commit. Goalactive.


## Arithmetic7249583 ordinary preliminary; direct-count acquire CPU7249798

Priorgoalturn madeprogress: completed electednegativecomparison,provedarithmetic
indices,passedCPU7249293,submittedfour-armGPU7249583. Currentjobconfirmedlive;
no rerun. Ordinarymeans/medians us: independent37.9328/37.952(range37.568–38.304),
address37.9456/37.952(37.632–38.528),SFparent38.2368/38.320(37.760–38.688),
history37.7632/37.680(37.472–38.528). Arithmetic−.2912to−.3040us mean; independent
vswait-retainingaddress only−.0128us. Suggestsindexreadcost ratherthanSFwait,
but waitfordecomposition andallcorrectness beforeadoption. Rangesoverlap.

Upstream auditfoundanotherrelay: computeCTAwarp0 waitsSizesReadyGPU fromstartup
CTA0, thenreads producer'sfinalcountarray. Producerroute_readySYS alreadycovers
allcountatomics/descriptors viaits gridrelease/lastblockpublication; SF/Aworkers
andhistoricalpull directlyacquireit. Densecountsimmutableuntil last-CTAcleanup.
New direct_counts main simplyusespull'sSYSacquirebeforeexistingcountloads and
sharedcountarrivalbarrier, eliminatinglocalrelaydependency. StartupSizesReady
publicationretained unchanged toisolatereadside. No router/sender/payloadchange.
Risk:150warp0SYSpollersmayincreasehotcounter/cachepressure; measuremetadata
acquisition andkernel/actualMMA, don'tinfergainfromone earlierobservation.

PreparedplaincountsonverifiedSFparent andcombinedcountsonarithmeticindependent
main. CPU7249798 submittedfourordinary/IKET/registergates andfourrouterwarmups;
sourcesfrozen. Compareplanselectedaftercurrentarithmeticdecomposition; no next
GPUyet. Needbothindividual/combinedcontrols toseparateearlymetadata fromindex
loadremoval ifbothpromising. SameEP8/resetoutsidekernel, no defaultpromotion.

Direct_counts CPU7249798 normal/IKETfourregistergatesPASS128regs24stack0spills;
bothordinaryvariants retainsingleuniformUTMALDG.GATHER4. Routerwarmupsongoing.
Preparednextfive-arm2x2 counts32/combined32/independent32/SFparent/history,22min
120exact+120replay,notGPUsubmitted. ThisseparateslocalSizesReadyrelayremoval
fromarithmeticindex/noSFwaitchange andtheirinteraction. Sourcepublication and
allstartuproles remainfixed. Existingarithmetic7249583 continues; decompneeded
beforeacceptinganynewbase. No defaultpromotion.

ArithmeticGPU7249583 COMPLETE11m09s96exact+96replayPASS,stderr empty;
analysis7249584 pending. Fullchainindependent65.113601/address63.833600/
SFparent64.880000/history98256.408691us, preservehistoryoutlier.
Direct_counts CPU7249798 COMPLETE3m45s fourgates+fourwarmups128regs0spills,
oneuniformgatherPASS. GeneratedcombinedSYSacquire4b10/CCTL4b20/check8/retry
precedescountload4ba0/sharedstore4bc0; existingcountbarrierretained.
Direct_counts GPU7249988/analysis7249989 submittedfive-arm2x2+history,22min
120exact+120replay expected. Thisfollow-onkeepsbothSFandverifiedarithmetic
controlsanddoesnotdependonadoptingarithmeticdecomposition. CPUanalysisandnew
GPUproceedconcurrently; no defaultpromotion. Needcollect7249584whenready.


## Arithmetic7249583 complete: modest kernel/broad-start improvement

GPU7249583 COMPLETE11m09s nvl72112-T08/T12;analysis7249584 COMPLETE46s.
96exact+96replayPASS,stderr empty. Evidence arithmetic_indices/comparison_7249583
.json/.md,run_audit_7249583.json. SamefixedEP8/resetoutsidekernel; no defaultpromotion.

| us,threecaptures | independent | address+SFwait | SFparent | history |
|---|---|---|---|---|
| ordinarymean/median |37.9328/37.952|37.9456/37.952|38.2368/38.320|37.7632/37.680|
| ordinaryrange |37.568–38.304|37.632–38.528|37.760–38.688|37.472–38.528|
| firstMMAmetadata |9.728/9.728/9.824|9.824/10.016/9.792|9.824/10.240/9.792|73.920/10.272/10.144|
| firstMMAselector |12.096/12.096/12.192|12.192/12.352/12.224|12.160/12.736/12.256|76.192/12.512/12.384|
| latest75pair |10.368/10.336/10.656|10.336/10.496/10.592|10.848/11.104/10.784|75.712/11.552/10.944|
| computespan |25.056/24.992/25.632|25.120/25.504/24.640|25.408/25.952/25.120|25.216/25.504/24.512|
| remainderwaitmedian |.672/.864/.576|.736/.480/.672|.448/.240/.448|—|
| computeendmetadata |34.784/34.720/35.456|34.944/35.520/34.432|35.232/36.192/34.912|99.136/35.776/34.656|

Independentordinary−.3040us/.795%mean,−.368median vsSFparent. Address−.2912us
mean andsamemedian37.952; additionalSFwaitremovalonly−.0128usmean, notseparate
kernelgain. Rangesoverlap; rechecknextcomparison. FirstMMAmedian−.096us,
latest75median−.480; by10us30/41/28pairs vsparent31/0/3; by11us75/75/75 vs
75/69/75. Compute~25uspreserved (median25.056vs25.408parent/25.216history).
Perpairmedians23.712/23.744/24.320 vs24.448/24.544/23.968; history23.808/
23.136/23.168. Don'tattributewholecomputespandifferencetoindices; samplevariation
andstartupskew remain. Histkernelstill.1696usfastermean.

IndependentAobservedbeforeSF in21/1/0pairs, tiedin1/0/0, laterin53/74/75;
addressandparent Alaterinall75eachcapture. Thisdemonstratesremovedorderingedge
canpermitoverlap, butobservationgapsare notpureoperand/networkstalls. Independent
AminusSFmedians.160/.768/.576us. Earlierissuershavehigherremainderwaits without
compute regression. Strictcountsverifiedfrompairdata; tiesnotcountedasbefore.
Historical73.920usstartupand98256.408691usfullchainoutliersretained. Fullchain
65.113601/63.833600/64.880000/98256.408691us, distinctfromordinarykernel.

Next direct_counts GPU7249988/analysis7249989 queuedfive-arm2x2+history:
counts32/combined32/independent32/SFparent/history. CPU7249798 COMPLETE3m45s
allgates/warmups128regs0spills/uniformgatherPASS. Tests directrouterSYSacquire
independentlyandcombinedwitharithmeticindices. Sourcesfrozen,22min120exact+
120replay expected; no defaultpromotion/commit. Goalactive.

Duringdirect_counts7249988,ordinarypreliminarycounts38.0576/median37.984us,
combined37.7248/37.664,independent38.0832/38.032,SFparent38.1824/38.176;
historynotyetcollected. Combined−.3584vsindependent and−.4576vsSFparentmean,
counts-only−.1248vsSF. Needdecomposition/lastcontrol beforegainattribution.

Boundedscheduleraudit (arithmetic_indices/initial_work_map.json) verifiesfirst75
MMApairs coveronlyexperts0..6:16rows/e,oneMblock,3072/128=24Nblocks/e,initial
block_idx=CTAindex. Experts0..5 each12pairs;expert6threepairs. Laterexperts'
validationcompletionisnotdirectfirst-MMAdependency. Do notinterpretall75initial
startsasall16expertsready. Thiscouldinformfuturepriority/pacing, butexisting
8-vs16metadatahistory cautionsagainstsimplyreducingworkers. No suchvariantyet.
Potentialmeasurementgap:firstwindowpublicationissue isnotdirectlymarked, so
validation→consumerobservation cannotbesplitexactly. Directfirst-windowrelease
marker andallwarp copy-phaseendscouldseparatework/handoff; notimplemented or
GPUsubmitted pendingcurrentresults. Existingtimestamps distinguishissuesfrom
completion; neverinferremoteincominglatencyfromrank0outgoingmarkers.


User guidance: consult MEGAMOE_COMM_EXPLORATION.md when choosing further ideas.
Re-read the proposals; prioritize a measured remaining dependency, preserving
EP8, ordinary kernel timing, actual MMA distribution, and ~25us compute.
Direct_counts7249988 ordinary collection now includes history: means counts
38.0576, combined37.7248, independent38.0832, SFparent38.1824, history37.9040us;
medians37.984/37.664/38.032/38.176/37.920. Combined is .4576us below SFparent
and .1792us below history in this run, with overlapping ranges; provisional
until decomposition and repeated evidence. GPU still running, stderr empty;
combined IKET capture finished, remaining controls in progress. No new job.


Remaining dependencies audited while direct_counts captures run:
- SF/index producers still join all 256 threads before SF bit1 publication.
  In this EP8 case, SF packing uses warps0–3 and index/score work uses warp7.
  Thus independent SF work can still wait for index/score loads at publication.
  Arithmetic consumers removed index loads, but FC1 scores and FC2 destination
  metadata still need ordered visibility. Any split must give them a separate
  acquire/publication; deleting the join or publishing SF early without this
  replacement is invalid. Need per-role completion markers before attribution.
- The 1536B A prefix uses half-warp-per-row batched loads, while the remaining
  2560B loops over two rows per warp serially and reloads descriptors. A bounded
  alternative is half-warp-per-row for the remainder too (10 uint4 registers
  per lane rather than 5). This is distinct from the failed full-16-vector
  all-payload batch and the failed 32-CTA split. It may cost registers/queues;
  only pursue if the new decomposition shows exposed remainder latency.
- Existing source ABBA script supports Nsight and source/env-specific arms,
  but frozen overlays are currently launched through capture.py. Any repeated
  confirmation must preserve the same prepared-buffer harness and kernel-only
  extraction, rather than substitute whole-pipeline timing. No new run yet.


## Direct counts7249988: earlier metadata, modest ordinary kernel gain

GPU7249988 COMPLETE15m40s on nvl72032-T13/T14; CPU analysis7249989
COMPLETE47s. 120 exact + 120 replay comparisons PASS; stderr empty.
Evidence: receiver_validate/direct_counts/comparison_7249988.json/.md and
run_audit_7249988.json. Reset remains outside kernel timing; no promotion.

| us, three captures | counts | combined | arithmetic | SF parent | history |
|---|---|---|---|---|---|
| ordinary mean/median |38.0576/37.984|37.7248/37.664|38.0832/38.032|38.1824/38.176|37.9040/37.920|
| metadata acquire median |5.664/5.248/5.632|5.664/5.440/5.696|6.720/6.240/6.848|6.816/6.368/6.624|49.584/4.896/5.184|
| first MMA from metadata |9.760/9.152/9.824|9.440/10.048/9.792|9.824/9.440/9.760|10.016/9.632/9.728|54.496/10.048/9.728|
| last initial MMA pair |10.688/10.208/10.784|10.048/11.008/10.208|10.368/10.208/10.464|11.104/10.464/10.496|55.840/11.424/11.584|
| compute span |24.896/25.344/25.088|25.184/24.992/24.864|25.024/26.112/25.536|25.024/25.344/25.024|25.376/24.672/26.080|
| remainder wait median |.448/.448/.384|.832/1.168/.576|.864/.912/.832|.512/.944/.416|—|
| compute end from metadata |34.656/34.496/34.912|34.624/35.040/34.656|34.848/35.552/35.296|35.040/34.976/34.752|79.872/34.720/35.808|

Combined ordinary mean improves .4576us/1.20% against SFparent and .3584us
against arithmetic; .1792us below history in this run. Ranges overlap; only
10 ordinary instances/arm, so not a robust claim of beating history. The
intended metadata dependency moves ~1us earlier; last initial-pair median
10.208 vs10.496 parent, but first-MMA median9.792 vs9.728 and capture1 is
slower. Preserve all three samples rather than claim uniformly earlier MMA.
Compute stays ~25us (combined median24.992, parent25.024, history25.376).
Combined by10us:66/0/64 pairs; by11us:75/73/75. Parent0/35/19 and74/75/75.
The75 initial pairs cover experts0..6, not all16.

Combined latest-A observation precedes latest-SF in38/12/59 of75 pairs,
with7/6/1 ties. Median A-minus-SF =-.032/.192/-.224us; arithmetic has
4/1/3 strictly earlier A pairs. This motivates inspecting SF publication,
not a claim of isolated SF stall. Gather-marker→MMA median .512/.352/.672
vs parent .288/.320/.288. Separate readiness issue/observation timing needed.
Fullchain counts63.472000/combined65.097600/arithmetic66.243201/parent66.204800/
history96407.678223us. Historical54.496us startup and fullchain outliers retained.

Next bounded candidate: separate SF completion from warp7 index/score work
inside each SF/index CTA, preserving explicit metadata readiness for later
epilogue consumers. Use matched direct SF-store/publication and first-A
publication markers to distinguish work from handoff. Keep sender/A protocol,
150 compute CTAs, arithmetic indices/direct counts, and buffers unchanged.


## SF completion candidate prepared; CPU preflight7250740 active

Experimental sf_completion starts from direct_counts combined32. SF warps0–3
process128-chunk batches and join independent named barrier4 (128 threads),
then publish existing SF bit1. Warp7 independently writes indices/scores/return
metadata, syncs its lanes and publishes new bit8. Epilogue lane0 explicitly
GPU-acquires bit8 with compiler memory clobber, then syncwarp, before FC1
score or FC2 destination reads. Existing A2/4,1536B window, direct counts,
arithmetic indices, sender/reset/cleanup are unchanged. BLOCK_M<=32 asserted;
CPU ownership check0..32 rows covers every SF chunk once; fixed16 rows retains
original active tid mapping. Barrier3 remains full initialization/retirement;
barrier4 is disjoint from compute barriers0/1/2. No blind join deletion.

Both candidate and matched control have direct first-tile SfStoresIssued
(all SF-role warps:4 vs8, with some parent warps inactive at16 rows),
MetadataStoresIssued(warp7), SfReadyIssue(warp0), FirstPayloadIssue(Awarp0).
Markers distinguish stores/phase ends and release issue; none is completion.
Profiler validates optional marker counts; summarizer pairs local work and
publication by CTA/warp and retains every observed time. Existing real combined/
history decompositions still summarize successfully with new optional fields.

CPU7250653 FAILED50s: ordinary candidate128regs24Bstack0spills; instrumented
build exceeded PMTrig5's30-event limit. No GPU was submitted. Existing memory
records ExtendedNativeDump failure, so do not switch instrumentation method.
Removed two unused pull events plus redundant MegaMetadataEnd/MegaPayloadStart
from both experimental main headers; metadata-detail mode already0. Preserve
MetadataStart/Acquired, sender starts, actualMMA, SF/A, remainder andcompute-end.
Marker total remains30. Corrected sources synced; CPU retry7250740 submitted
four ordinary/IKET register gates andfour router warmups. cpu_job.json retains
failed job; cpu_retry_job.json records current job. Source snapshots frozen.

Next after CPU gates: inspect generated register/barrier/gather/acquire ordering,
then same EP8 ordinary split/control/control/split/history (ABBA) plus three
profiles split/control/history. Generic capture manifest phases select repeated
ordinary-only arms, using same prepared-buffer harness. 20min plan, no GPU yet;
expect88exact+88replay across fiveordinary andthreeprofile invocations.
Do not replace maintained kernel or promote without measured benefit.

SF completion7250740 COMPLETE3m58s: all four ordinary/IKET register gates and
four router warmups PASS,128regs24stack0spills. Both ordinary kernels have one
uniform GATHER4. Candidate SASS: barrier4 count128; metadata acquire/CCTL/bit8
test precede FC1 score and FC2 destination loads.
GPU7250809/analysis7250810 submitted, then GPU held while still PENDING with
no GPU allocation: review identified new metadata acquire also ran on row-idle
epilogue warpgroups. Added uniform valid-row guard; idle groups still return
TMEM credits but do not acquire metadata they never read. CPU7251294 compiles
only changed candidate normal/IKET and rechecks register gates; unchanged
control/router builds reused. GPU dependency updated to afterok7251294 and
remains HELD until generated-code review; analysis dependency unchanged.
Pre-guard review saved as codegen_review_before_guard.json. Do not report
current candidate validated until7251294 passes; no GPU experiment ran yet.

SF completion guard CPU7251294 COMPLETE57s, both changed candidate builds
128regs24stack0spills and register gates PASS. Final codegen_review.json:
valid-row guard a720 skips metadata acquire on idle epilogue warpgroups;
active lane0 acquire a760/CCTL a770/bit8 test a790 precedes score load a8f0
and FC2 metadata loads bd60+. Single uniform gather at5930; SF128-thread
barrier4 retained. CPU7250740's control builds/fourrouter warmups reused.
Released GPU7250809 after generated-code review; dependency afterok7251294.
Analysis7250810 remains afterokGPU. Fixed EP8,20min; ordinary ABBA split/control/
control/split + history, profiles split/control/history, expected88exact+
88replay. No GPU result yet; no maintained-header promotion or commit.

Instrumentation participation correction: cancelled GPU7250809 at1m12s and
dependent analysis7250810 before any completed capture. Candidate's initial
8 exact rank checks passed; no replay/timing/decomposition result. Control SF
phase-end marker had8 locations/CTA including4 idle, candidate4; that would
bias the intended publication-delay comparison. Control now marks the same
four active SF warps as candidate; both64 SF-store locations across16 CTAs.
Before-fix control source archived; ordinary kernel unchanged. CPU7251420
rebuilds only control IKET and rechecks existing register matrix, reusing
7250740 warmups and7251294 candidate validation. No GPU currently running;
new comparison follows successful marker preflight. No measured candidate gain.

Matched marker CPU7251420 COMPLETE49s: control IKET128regs24stack0spills and
all register gates PASS. Candidate SASS review7251294 remains current;
codegen_review.json now contains latest register matrix. Both arms record64
SF-store locations (same4activewarps ×16experts). Corrected GPU7251457 and
analysis7251458 submitted (gpu_retry_job.json/analysis_retry_job.json);20min,
fixedEP8 ABBA ordinary + split/control/history profiles,88exact+88replay
expected. All required preflights complete. No measured SF candidate result
yet. Keep original cancelled-job evidence and all previous timings.

Static follow-up audit while SF run queues: fixed EP8 L1 readiness flags have
base offset0x12b0 and4B stride;16tiles occupy64bytes in one128-byte-aligned
region relative to workspace base. This is a layout fact, not measured cache
contention or proof of atomic serialization. If publication/handoff remains
exposed, compare spaced words or per-stage single-writer notifications.
Arithmetic gather no longer reads the separate gather-index array, which
could provide existing steady storage, but its current index writes must be
removed and every publisher/consumer/cleanup changed consistently; never
alias live indices or assume old values are zero. Allocation is zeroed once
by MegaMoESymmBuffer; invocation reuse still requires explicit flag cleanup.
No code change/job for this proposal. Evidence sf_completion/readiness_layout_audit.json.

SF completion7251457 ordinary ABBA complete, profiles ongoing. Means in order:
split38.0800, control37.8048, control-repeat37.6096, split-repeat38.2720us;
history37.9456. Medians38.032/37.840/37.568/38.240/37.968. Equal20-instance
means: split38.1760 vs control37.7072, +.4688us/+1.24%. Candidate loses in
both passes; do not adopt even if a profile milestone advances. Evidence
sf_completion/ordinary_7251457.json. Need new publication intervals andcompute
span to distinguish insufficient SF gain from epilogue metadata-acquire cost.
Best control is prior direct_counts combined32, now showing37.61–37.80us
means vs historical37.9456 in this allocation; do not pool medians or claim
statistical certainty from20 instances. Profiles remain diagnostic; no new
variant until current cause is clearer.


## SF completion7251457 complete: real startup gain, kernel regression

GPU7251457 COMPLETE12m09s nvl72108-T05/T08; analysis7251458 COMPLETE43s.
88 exact +88 replay PASS; stderr empty. Five normal invocations each8 checks
plus three profile invocations each16 =88. Earlier104 expectation reversed
these multiplicities; per-section audit verifies all8 invocations, no missing
checks. Evidence sf_completion/comparison_7251457.json/.md, ordinary_7251457.json,
run_audit_7251457.json. Both arms mark the same4active SF warps.

Ordinary ABBA means split38.080/38.272 vscontrol37.8048/37.6096us; equal20-
instance means38.176 vs37.7072 (+.4688us,+1.24%). History37.9456. Candidate
loses bothpasses; do not adopt. Fullchain65.817601/64.588800/64.998401/
65.190401/history95586.566162us in actual execution order; outlier retained.

| us,three captures, metadata origin0 | split | control(best combined) | history |
|---|---|---|---|
| first actual MMA |8.896/9.152/9.184|9.152/9.760/9.216|13.568/9.824/9.888|
| last75initial pairs |9.760/10.016/10.080|9.856/10.272/10.080|15.296/11.040/11.904|
| compute span |25.216/25.216/24.960|25.472/25.504/24.832|24.832/24.384/24.992|
| final accumulator observation |34.112/34.368/34.144|34.624/35.264/34.048|38.400/34.208/34.880|
| accumulator observation to kernel end |6.112/5.472/6.016|5.216/4.352/5.696|4.096/5.568/4.960|
| remainder issuer wait median |1.184/1.104/.896|.992/1.024/.960|—|
| SF store-phase-end to publication issue median |.176/.160/.192|.704/.512/.832|—|
| A validation-start to first-window publication median |1.632/1.728/1.600|1.680/1.824/1.696|—|

SF publication precedes metadata-store marker in11/12/16 of16 worker CTAs
vs0/0/0control. Signed median metadata-store minus SF-publication issue
+.128/+.288/+.352 vs-.160 eachcontrol. This verifies removed join allows
independent SF publication. Work→publication spans include fence/join/issuer
work; not isolated fence latency or publication completion. First MMA improves
.256/.608/.032us, but longer post-compute tail outweighs startup gain. Compute
stays~25us; do not claim compute slowed. Tail is an envelope, not isolated
metadata-acquire latency. New bit8 acquire is after accumulator wait, making
its latency exposed before score/return processing.

Prepared early_metadata_acquire: move same active-group acquire and independent
coordinate calculation before accumulator wait; keep score/destination loads,
producer/SF/A protocol, reset, arithmetic, and matched markers unchanged.
This isolates whether the acquire can overlap ongoing MMA. Compare early,
late split, best unsplit control, history; ordinary early/control/control/early/
late/history, four profiles,112exact+112replay expected. Risks: early FC1 polls
may add contention; ComputeEnd observation moves after pre-wait work and can
shift without physical completion change, so require ordinary kernel gain.
No next GPU yet; new candidate CPU normal/IKET/register checks +2routerwarmups
next. No maintained-header promotion or commit.

Early_metadata_acquire CPU7252041: ordinary/IKET register gates PASS128regs
24stack0spills; two router warmups in progress. Codegen_review.json confirms
active-row guard, metadata acquire a6c0/CCTL a6d0/bit8 a6f0 precede accumulator
TRYWAIT a7d0, while score load stays after wait at a910; one uniform gather.
GPU7252064 submitted afterok7252041; analysis7252065 follows. Four-way profiles
early/control/late/history with early/control ABBA ordinary and additional
late/history ordinary,20min,112exact+112replay expected (6×8+4×16). Source
frozen. No result yet; compare ordinary time, actual MMA distribution and
observation-based tail together. No maintained-header promotion/commit.

Early metadata CPU7252041 COMPLETE2m13s, bothrouterwarmupsPASS inaddition
to ordinary/IKET register gates. GPU7252064 queued, analysis7252065 dependent.


Early metadata ordinary7252064: early38.1376/38.1600us versus best-control
37.8208/37.9136us, equal20-instance means38.1488 versus37.8672 (+.2816us,
+.744%). Both passes lose; no adoption. Late split38.7456 (median38.320,
max41.280), history38.2976 (median38.208,max39.744); retain all samples.
Profiles/analysis pending; earlier acquisition helps relative to late split
but does not establish a win over best combined all-push.

Same-expert handoff audit of completed SF7251457 pairs first75 MMA clusters
with their actual expert0..6 local publishers (SF CTA=e, A CTA=e+16).
SF-publication-issue→latest-SF-observation medians split1.280/1.280/1.344us,
control.800/.992/.800; A-publication→latest-A-observation split1.056/1.056/
1.184,control.992/1.024/1.024. Latest(A,SF)observation→MMA split.352/.352/.384,
control.480/.448/.480. Earlier publication is not fully passed through to
consumers. These intervals include scheduling/acquire and marker placement;
not isolated visibility/network latency. Full75-pair evidence retained in
sf_completion/publication_handoff_7251457.json; mapping derives from verified
arithmetic_indices/initial_work_map.json, not sorted load assumptions.

Exploration reference remains MEGAMOE_COMM_EXPLORATION.md, especially independent
completion ownership (1), polling/queue pressure (4), and early independent
metadata work (7). Feasibility only: official PTX st.async documentation permits
scalar st.async.release.gpu.global.u32 on sm100+ (PTX8.7+). Separate single-writer
words could avoid shared RMW, but cannot replace shared bitmask reductions
in-place; generic release does not certify prior TMA completion. No such
variant implemented or submitted. Existing packed-readyword audit establishes
layout only, not contention. Next change must follow current decomposition.


## Early metadata7252064 complete: overlap does not beat best control

GPU7252064 COMPLETE14m38s nvl72063-T17/T18; analysis7252065 COMPLETE49s.
112exact+112replay PASS, stderr empty (6normal×8 +4profile×16). Full evidence
in receiver_validate/early_metadata_acquire/comparison_7252064.json/.md,
ordinary_7252064.json, run_audit_7252064.json, publication_handoff_7252064.json.
Markdown preserves all6ordinary passes; fourprofile arms each3captures,
metadata origin0 with selector origin retained. No active job or promotion.

Ordinary ABBA early38.1376/38.1600 vscontrol37.8208/37.9136us; equal20-instance
mean38.1488 vs37.8672 (+.2816,+.744%). Late38.7456 (median38.320,max41.280),
history38.2976 (median38.208,max39.744). Early improves over late split but
loses bothpasses against best combined all-push. Retain best control.
Fullchain in execution order66.336000/65.414401/63.065601/64.025601/66.800001/
history97494.671631us; outlier retained, not the ordinary-kernel metric.

| us,three captures;metadata origin0 | early | control | late | history |
|---|---|---|---|---|
| first actual MMA |9.440/9.536/9.504|9.152/9.472/8.992|9.120/8.928/8.928|25.312/9.728/9.760|
| latest75initialpairs |10.496/10.528/10.464|9.760/10.080/9.664|9.760/9.760/9.824|27.072/11.328/11.424|
| compute span |25.088/25.184/25.344|25.184/25.088/25.536|25.472/26.112/24.896|25.216/24.800/25.280|
| compute-end observation→kernel-end |5.728/5.056/4.992|5.472/5.312/5.536|5.760/5.056/6.624|3.488/4.960/4.384|
| remainder wait median |.352/.448/.448|.960/1.568/1.264|1.440/1.584/1.088|—|
| SF phase-end→publication issue median |.160/.160/.144|.704/.400/.592|.192/.160/.208|—|

Same-expert handoff medians earlySF1.760/1.824/1.760us versuscontrol.960/.928/
1.024 andlate1.248/1.248/1.312; earlyA1.824/1.568/1.792 versuscontrol1.024/
.928/1.056 andlate1.056/.992/.992. Last(A,SF)observation→MMA early.352/.352/
.320,control.416/.480/.416,late.352/.352/.352. Thus broad startup delay lies
before the consumer readiness observations, not a larger post-observation
MMA-issue gap. These are scheduling/acquire envelopes, not pure hardware
visibility latency. Early polls plausibly add queue/scheduling pressure;
current experiment does not isolate that mechanism. Lower remainder wait
also partly reflects later first-stage admission, not independently faster
remaining-payload arrival. Compute remains~25us. Moving work before the
accumulator wait can change ComputeEnd observation without changing hardware
completion; no gain inferred solely from tail redistribution.

MEGAMOE_COMM_EXPLORATION.md items1/4/7 remain useful. Next bounded direction:
reduce redundant early readiness acquisition or separate publication domains,
while preserving a proof of metadata visibility before epilogue reads. Packed
flags' layout alone is not proof of contention. Scalar single-writer release
slots and coordinator relay are only proposals; neither implemented, no job
submitted. Do not add another all-worker join or blindly move more blocking
work to the first-MMA path. Preserve best direct_counts combined32 and fixed
EP8/steady-reset-outside-timing acceptance.


Prepared metadata_signal after early-acquire7252064 negative result. Candidate
separates metadata bit8 publication/acquisition from original SF1/A2/A4 word,
using gather-index[poolblock*BLOCK_M] (128B stride at fixed BLOCK_M32). The
arithmetic gather path has no index-array readers. Remove obsolete index
stores in BOTH candidate and matched packed control; same red.async release,
GPU acquire, active epilogue warp lane0/syncwarp and early wait placement.
Separate signal cleanup joins existing last-CTA active-tile cleanup after all
metadata readers retire; allocation initially zeroed outside timing. No buffer
size/layout/reset changes, no source/destination/score value change. Source
layout review verifies all7776 possible BLOCK_M32 tile slots fit248832 words; see
storage_review.json; no extra allocation. Do not infer cache contention from
layout alone. Changes frozen in scratch/receiver_validate/metadata_signal.

CPU7252554 submitted: four normal/IKET builds/register gates, four router
warmups. GPU not submitted yet. Planned ordinary separate/control/control/
separate/packed/history, profiles separate/control/packed/history, same fixed
EP8,20min,112exact+112replay expected. Matched packed control isolates address
separation plus cleanup; best direct_counts control determines net gain.
Publication/consumer handoff, broad actual MMA starts, compute span and ordinary
kernel timing remain acceptance evidence. No maintained-header promotion.


Metadata_signal CPU7252554 four register gates PASS128regs24stack0spills.
SASS confirms separate metadata publisher UREDGR28f0/cleanupSTG61f0..6690/
consumerLDGa6a0 all use dedicated offset0x5f5600; packed consumer uses0x12b0.
Both ordinary acquirea6a0→bit8checka6d0→accumulatorTRYWAITa7b0, one uniform
GATHER4 each; IKET checks likewisepass. Codegen_review.json retains actual
regions. At2m24s CPUstillRUNNING with1of4routerwarmups observed; remaining
warmups gated by afterok dependency. GPU7252599 submitted afterok7252554,
analysis7252600 afterokGPU, bothPENDING dependency. No performance result.
Sources frozen; no further mutation during run. Ordinary separate/control/
control/separate/packed/history, four profiles,112exact+112replay expected.


Metadata_signal CPU7252554 COMPLETE4m26s, all4routerwarmupsPASS after four
register gates. GPU7252599 RUNNING on nvl72131-T13/T14; analysis7252600 pending.
First separate32 ordinary mean37.696us/median37.664/min37.344/max37.952;
controls/repeats pending, not a gain claim. Sources stay frozen.

While current run executes, static follow-up reviews one GPU acquire per
active epilogue warpgroup plus128-thread local handoff before accumulatorwait:
fixed16validrows implies150 potential pollers vs600 (not measured concurrent
loads). Existing WG barrier IDs3/4 are reusable only after dispatch's final
worker join retires earlier SF barrier4 use. Prefer unaligned barrier.sync
with compiler memory clobber for conditional warpgroup participation; existing
sync_aligned wrapper lacks that clobber. Row-idle WG must skip uniformly and
keep TMEMcredits. Additional local rendezvous may lose despite fewer pollers.
No implementation/job; warpgroup_acquire_review.json retains proof obligations
and official PTX barrier reference. Do not blindly use an independently reused
operand-barrier phase as an epilogue startup hint.

Latency summarizer now accepts --ordinary-input ordered timing JSON with
--markdown-output, including ordinary-only repeats in the primary table.
Validates every profiled arm against its captured ordinary timings and timing
bounds; retains execution order and per-pass medians, no median pooling.
Verified against completed early_metadata_acquire7252064 all6passes; resulting
comparison JSON unchanged, all instrumented samples retained. Evidence
comparison_all_passes_7252064.json/.md. README updated; no extra GPU test.


Metadata_signal7252599 all ordinary passes available: separate37.6960/37.7824
versus bestcontrol37.7216/37.7344us (equal20-instance means37.7392/37.7280,
+.0112us; split direction, no established gain). Matched packed/no-index37.9904
and history37.7824. Relative to packed, separation saves.2512us at mean level;
one packed pass, no causal claim beyond address/cleanup change matched in code.
GPU at9m02s RUNNING nvl72131-T13/T14, first profile captured; remaining profiles
and analysis7252600 pending. No adoption from ordinary tie.

Prepared warpgroup_metadata from frozen metadata_signal separate parent:
warp_idx_in_wg0/lane0 performs metadata GPU acquire, then all128 active group
threads execute barrier.sync(3+wg,128) with compiler memory clobber BEFORE
accumulator wait. Replaces fouracquires/syncwarps withoneacquire/localhandoff;
all other protocol/source/cleanup/markers unchanged. Row-idle groups skip
uniformly and retain TMEMcredit work. Dispatch finaljoin retires earlier
barrier3/4 ownership before epilogue uses its existing groupIDs. Uses unaligned
barrier instruction for conditional participation. No new shared storage.
CPU7252889 submitted for normal/IKET/register gates and2routerwarmups; latest
PENDING. No next GPU until current decomposition. This CPU preparation overlaps
ongoing captures without changing their frozen sources. Added join may lose;
require measured ordinarygain plus broadMMArelease/compute span before adoption.


Warpgroup_metadata CPU7252889 ordinary/IKET register gates PASS128regs24stack
0spills. SASS ordinary active-row guard a600/a620 skips both metadata acquire
andhandoff for idleWG; owneracquirea6a0/bit8checka6d0→128-thread BAR.SYNC.
DEFER_BLOCKINGa730→accumulatorTRYWAITa7c0→scoreloada910. IKETacquireb7f0→
handoffb880→TRYWAITb920; oneGATHER4 each. One of2routerwarmups observed; CPU
stillactive at last status. Prepared/synced runplan wg/control/control/wg/
separate/history, profileswg/control/separate/history,112exact+112replay.
No GPU submission yet; current7252599 decomposition is required first.

Workspace-lifetime audit: profiling launcher starts a fresh workload process
perarm; fixed test constructs MegaMoESymmBuffer, whose constructor zeroes it
and joins peers before timing. Replay keeps the same variant and last-CTA
cleanup clears active signal slots. Never hot-switch an existing index-writing
workspace to the reused-index signal scheme without off-critical reinitialization.
Current experiment does not switch variants inside a live workspace.


## Metadata signal7252599 complete: startup recovery, ordinary tie

GPU7252599 COMPLETE13m55s nvl72131-T13/T14; analysis7252600 COMPLETE43s.
112exact+112replay PASS, stderr empty. Evidence metadata_signal/comparison_7252599
.json/.md, ordinary_7252599.json, run_audit_7252599.json, publication_handoff_7252599.json.
New --ordinary-input report includes everyordinary pass in actual order.
Separate37.6960/37.7824 versus best37.7216/37.7344us: equal20-instance means
37.7392/37.7280 (+.0112us,mixed direction). Matchedpacked37.9904,history37.7824.
No established gain over best and no promotion. Separation improves matched
packed design but does not isolate a particular hardware contention mechanism.

| us,three captures;metadata origin0 | separate | bestcontrol | packed | history |
|---|---|---|---|---|
| firstactualMMA |9.120/9.280/9.056|10.080/9.312/9.248|9.568/12.000/10.432|60.608/10.208/9.728|
| latest75initialpairs |9.952/10.240/9.792|10.560/10.144/9.888|10.592/14.144/12.416|62.208/12.000/11.104|
| compute span |25.120/26.144/25.056|24.736/26.016/24.736|25.536/28.928/25.760|25.056/25.632/24.640|
| accumulatorobservation→kernelend |5.344/4.544/6.144|5.088/4.256/5.984|4.640/2.944/6.016|3.488/3.488/5.120|
| remainderwaitmedian |.896/1.056/1.168|.608/1.184/.976|.384/.480/.416|—|
| SFphaseend→publicationmedian |.224/.192/.192|.576/.512/.608|.160/.688/.480|—|

Separate has75/65/75 initial pairs issued by10us versusbest0/65/75 andpacked
39/0/0. Same-expert SFpublication→lastSFobservation medians separate1.344/1.280/
1.152us versusbest.992/.864/.864 andpacked1.664/1.440/1.760; Ahandshake separate
1.056/.992/1.088 versusbest1.024/.960/.992 andpacked1.696/1.216/1.440. Last(A,SF)
observation→MMA separate.384/.416/.416 versusbest.512/.480/.416 andpacked.352/
.448/.544. Intervals include scheduling/acquire/marker placement, not pure
visibility/network latency. Earlier admission is recovered, but compute/tail
observations offset it in total kernel time; packed profile includes startup
andcompute outliers, retained rather than discarded. Historical firstMMA60.608
andfullchain93377.874756us retained. Fullchain in execution order64.585600/
64.371200/64.896001/64.777600/65.577600/93377.874756us.

Warpgroup_metadata CPU7252889 COMPLETE4m03s: ordinary/IKET/register gates
128regs24stack0spills,2routerwarmupsPASS. Verified active-row guard; acquire
→128-thread handoff→accumulatorwait→scoreload, oneGATHER4. Current evidence
justifies testing whether fewer global metadata acquirers reduces remaining
startup/compute interference without excess local rendezvous cost. GPU7253064
submitted AFTER reviewing completed7252599; CPUcompletion revalidated via
sacct before submit. Analysis7253065 afterokGPU. Ordinarywg/control/control/
wg/separate/history, profileswg/control/separate/history,112exact+112replay
expected,20min. No new candidate performance result. Sources frozen. No
maintained-header promotion or commit; best direct_counts combined preserved.


Warpgroup GPU7253064 confirmed RUNNING2m23s nvl72077-T12/T18; analysis7253065
pending. First wg32 ordinary mean37.6544us,median37.696,min37.280,max38.016;
no comparison claim before matched controls/repeats/decomposition. Evidence
warpgroup_metadata/ordinary_7253064.json. Sources frozen; no nextvariant.


Warpgroup7253064 ordinary ABBA complete: wg37.6544/37.5936 vsbest37.8848/
37.5392us; means37.6240 vs37.7120 (-.0880us), mixed pass direction. Medians
37.696/37.632 vs37.744/37.632; mean advantage alone is weak. Parentseparate
37.6448/history37.7568; only-.0208us vs parent. Profiles ongoing at14m16s,
analysis7253065 pending. No promotion before decomposition.

Source/lifetime audit suggests a more direct metadata path: producer dense
expert descriptors and score inboxes are already covered by existing SYS
all-rank route-ready. Compute's warp0 acquires it and relays through the existing
shared-count mbarrier with compiler clobbers. Epilogue can prefetch original
scores/packed return descriptors before accumulatorwait, avoiding receiver
metadata copies and bit8 entirely. Originalhelper and proposedreader use same
expert ordinal mblock*BLOCK_M+row. Generation proof: starting g+2 requires
local completiong+1, which waits everyrank's route-readyg+1; each rank starts
g+1 only afterlocalg exits, so bankg cannot be reused while anyg reader remains.
Inspected ordinarySASS has no GRIDDEPCONTROL launch instruction; officialCUDA
PDL docs say implicit trigger waits allprimaryblocks exit. Revisit proof if
successor launch/streams/banks change. No implementation/job yet. Requirements,
source locations and officiallink in metadata_signal/direct_metadata_review.json.
Remove obsolete metadata-store marker honestly and explicitly allow absence
in validator/summarizer; no dummy marker for workremoved. Prefetch/decode may
increase queue/register pressure, so require actual samecase performance.

Separate optional SFstore-layout audit (no kernel change/job) confirms for
16validrows scalarSFstores write2048B into256 perwarp/instruction32B sectors;
16B vectors with3paddingwords would write8192B into sameaddressedsectors.
Padding corresponds torows+32/+64/+96, outsideBLOCK_M<=32; allwrites unique.
This is address arithmetic, not measured transaction/RMWcost; extra bytes and
register/setup cost may lose. Evidence sf_completion/sf_store_layout_review.json.
Directmetadata removal has a clearer dependency rationale than this padding idea.


Warpgroup metadata7253064 COMPLETE14m32s; analysis7253065 COMPLETE45s.
112exact+112replayPASS; stderr empty. Ordinary WG37.6544/37.5936 vs best
37.8848/37.5392us, equal20-instance means37.6240/37.7120 with mixed direction;
parent separate37.6448/history37.7568. No compelling gain; unpromoted.
FirstactualMMA WG9.184/9.280/9.120 vs best9.184/9.312/9.696, parent9.216/
9.184/9.504, history38.944/12.256/11.968us (all retained). Latest75initial
pairs9.952/9.984/9.984 vs best9.792/9.984/10.080, parent10.304/9.632/10.176,
history39.520/13.664/12.608. Compute WG25.440/25.120/25.312 vs best25.280/
25.088/24.544, parent25.504/24.992/25.120, history27.808/27.584/26.080.
Observation-based tail WG5.600/5.184/6.016 vs best5.504/5.760/5.600,
parent5.728/5.760/5.536, history3.136/4.544/7.136. Fewer pollers alone does
not establish useful whole-kernel gain. Fullchain in actual pass order64.438400/
63.568001/63.772799/64.614401/64.374400/95264.923096us; historical outlier retained.
Evidence: warpgroup_metadata/comparison_7253064.json/.md, ordinary_7253064.json,
run_audit_7253064.json. Best direct_counts combined32 preserved.

Next direct_epilogue_metadata follows exploration item7: remove receiver
route/score stores, metadata8 publication/acquire and signal cleanup. Epilogue
prefetches original producer descriptors/scores before accumulatorwait, then
preserves exact arithmetic/shuffles/combine addressing. Existing SYSroute-ready
plus count handoff orders reads; generation reuse proof in
metadata_signal/direct_metadata_review.json. SF128-thread join/A validation/
materialization/retirement unchanged. Frozen main/dispatch overlays implemented;
CPU preflight7253540 submitted for ordinary/IKET register gates and router
warmups. No GPU yet. Profiler/summarizer explicitly record
receiver_metadata_materialized=false and require absent metadata-store marker
while retaining SF/A detail; no no-op marker. Intended ABBA direct/control/
control/direct, plus separate parent/history, four profiles112checks. Existing
buffers/reset outside timing; fixed EP8 only. No promotion or commit.


Direct_epilogue_metadata CPU7253540 COMPLETE2m29s; ordinary/IKET128regs,
24stack,0spills,2routerwarmupsPASS. SASS ordinaryFC2descriptorloads a370/a410,
FC1descriptor a4e0/score a590 before accumulatorTRYWAITa670; IKET b3e0/b480,
b550/b600 beforeTRYWAITb6d0. OneGATHER4 each and no early-successor trigger.
Explicit metadata-materialization contract verified against all4 prior7253064
reports: serialized results unchanged; invalid absent/present markers rejected.
GPU7253594 submitted only after completepreflight; analysis7253595 afterokGPU.
6ordinary passes direct/control/control/direct/separate/history,4profiles,
112exact+112replay expected. No new performance result; source frozen.


## Direct epilogue metadata7253594 complete: modest ordinary gain, broad release unresolved

GPU7253594 COMPLETE15m15s nvl72107-T06/T08; analysis7253595 COMPLETE45s.
112exact+112replayPASS,stderr empty. Direct37.4400/37.4848 versusbest37.6896/
37.6032us; equal20-instance ABBAmeans37.4624/37.6464 (-.1840us,.489%), both
pass directions faster. Direct medians37.440/37.552 vsbest37.712/37.648.
Parentseparate37.7472/history38.0736 singleordinarypasses. Retain as faster
fixed-case candidate; previousbestcontrol preserved, no default/general-header
promotion or commit. Detailed evidence direct_epilogue_metadata/comparison_7253594
.json/.md,ordinary_7253594.json,run_audit_7253594.json,codegen_review.json.

| us,three captures;metadataorigin0 | direct | bestcontrol | separateparent | history |
|---|---|---|---|---|
| firstactualMMA |9.088/9.216/9.056|9.280/9.280/9.248|9.088/9.248/8.928|60.736/10.176/9.856|
| median75initialpairs |9.664/9.504/9.696|9.632/9.664/9.440|9.440/9.728/9.504|61.504/10.688/10.752|
| latest75initialpairs |9.952/9.984/10.016|9.824/10.112/9.696|9.856/10.080/9.984|61.824/11.136/11.040|
| compute span |24.992/25.280/25.536|26.016/24.896/25.248|24.864/25.440/25.056|24.832/24.928/24.768|
| accumulatorobservation→end |6.624/5.376/5.280|4.384/5.504/5.344|5.984/5.856/6.176|3.456/4.448/5.184|
| SFphaseend→pubmedian |.208/.160/.208|.624/.592/.848|.208/.176/.208|—|

Initialpairsby10us direct75/75/74, best75/59/75,parent75/61/75,history0/0/5.
FirstMMAearlier vsbest but whole75pairdistributionmixed; no demonstrated broad
MMArelease improvement. Computemedian25.280 vsbest25.248/history24.832: preserved
near25us, not a compute speedup. Lastgathermarker→MMA perpairmedian direct.288/
.320/.288 vsbest.320/.416/.448, but earlieroperandreadiness observations still
vary. Aobservation laterthanSF bymedian.288/.256/.320us; intervals include
scheduling and marker placement, not pure communication or operand stall.
Pair firstMMA→firstFC2epilogue medians19.872/20.032/20.160 vsbest20.192/20.384/
20.640 andparent20.864/20.672/20.512; laterpairinterval offsets somegain. Do not
attribute all ordinarygain to one instrumented interval or treat epilogue
observation as exactphysicalMMAcompletion.

Original metadata directreads remove helperroute/scorecopies andmetadata8ready
traffic; actualSASS confirmsFC1score/FC2destinationprefetchbeforeaccumulatorwait.
Existingroute-readyordering andbanklifetime preserved. Localmetadata-store marker
absent byexplicitcontract; SF/A markers retained. No newallocation/initialization.
Explorationitem7 supplied useful independentprefetch boundary; broadreleasegoal
remains open. Fullchaininpassorder67.868801/64.278401/67.894401/66.937600/66.425600/
94237.103271us retained, no fullproducer-chain win claimed. All3captures retained,
including historicalstartup60.736us. No active CPU/GPU jobs after analysis.


## Row-specific readiness prepared; CPU7254051

Previous turn made progress: direct_metadata7253594 faster ordinary kernel,
but broad75pairMMA release remains mixed. CurrentA validator still256-thread
joins before first/full publication; each gather warp waits all tile rows.
New row_ready keeps16A CTAs/eightwarps/1536Bwindow, but eachwarp publishes its
own adjacentrowpairs independently. First owner(row)=(row%16)/2; remainder
mapping changed to match. SFbit0,firstA1..8,fullA9..16; unique activeowner
adds once, mask-equality consumer acquires only live consumed row owners.
All_main control has same publishers/mapping but waits everyactive owner.
This distinguishes narrower consumer dependency from removingjoins/moreatomics;
oldwarp_publish7244130 still waitedwhole-tile and32Aworker7248773 changedCTAcount.

Exactobservedword materialization, perwarp proxyfence/syncwarp→GPUrelease,
consumeracquire/syncwarp/proxyfence→TMA retained. Strong RMW observation chain
orders earlierrelease publications; PTXofficial observation-order/red.async docs
verified. No publication fromidleowners; everyactivebit eventually consumed.
Finalworkerjoin/DispatchDone/lastconsumer cleanup remain; no newallocation/reset.
CPU ownership check256cases acrossvalid_m1..32,bothCTAhalves,4gatherwarpsPASS.
GPU remainsfixedEP8 rows32 only. Both ordinary/IKET variants registergatesPASS
128regs24stack0spills; CPU7254051 routerwarmups notyetcomplete at lastcheck.
No GPU submitted. Frozen source/patch/proof in receiver_validate/row_ready.

First/FullPayloadIssue become128locations=16CTAs*8owners. Analyzer explicitly
uses per-case publishercounts; summary retains perwarpfirstissues andmaxissue.
Earliestissue certifiesonlyitsrows. ValidateStart remainswarp0phaseentry, so
crosswarpoffsets canbesigned; onlysamewarpordering is asserted. Allprior4
7253594reports remainunchanged afternewoptionalperwarpdetail. Plannedordinary
row/direct/direct/row/all/history,fourprofiles112exact+112replay. Extraatomic
pressure maylose; no speedclaim before matched timing/decomposition.


Row_ready CPU7254051 COMPLETE2m37s:4ordinary/IKET registergatesPASS128regs,
24stack,0spills,2routerwarmupsPASS. SASS rowordinary firstmaskacquire5900/
maskcompare5930,fullmask5a00/5a30,GATHER4at5b60; allcontrolGATHER4at5af0.
Allfourvariants retainoneuniformgather and2mask-equalityGPUacquire loops;
codegen_review.json records excerpts. GPU7254075 submitted onlyafter verified
CPUcompletion;analysis7254076 afterok.6ordinaryrow/direct/direct/row/all/history,
4profiles112exact+112replay expected. No newperformance result; sources frozen.

Row_ready GPU7254075 confirmedRUNNING on nvl72150-T03/T06;analysis7254076
pendingafterok. Firstcandidate8exact+8replayPASS; comparisons/captures pending.
Ownershipaudit corrected its illustrative alignedextent from8to16 to match
scheduler get_valid_m<true>; all256partialtilecases pass withactualalignment,
including coverage ofeveryliverow. Kernel already usedscheduler actualextent;
no source mutation. Evidence ownership_review.json records correction.


Row-ready ordinary results are complete: row 37.5712/37.7408 us versus direct
parent 37.5168/37.6064 us. Equal 20-instance means are 37.6560/37.5616 us;
no gain established. Whole-tile-wait control 37.7376 us, historical pull
37.7216 us. Captures still running; 80 exact and 80 replay checks pass so far.
Do not adopt on the basis of fewer joins. The matched all-owner control and
per-owner publication-to-observation analysis will distinguish readiness from
extra publication/consumer overhead.

A separate source audit found that receiver-owned A rows are now contiguous,
while compute still uses gather4. A regular 2D TMA issuer could replace two
active gather warps per CTA while retaining independent SF issue. This needs
native descriptor boxM 1→LOAD_BLOCK_M and matching full-barrier transaction
bytes; a device mnemonic-only change would be incorrect. Full-box loads read
some duplicate/padding rows at valid_m16, so retain whole-tile first/full
readiness and prove all read addresses stay inside the owned padded block.
Existing shared allocation fits the full box. Extra A bytes may outweigh fewer
issues. Requirements and native-build route are saved in
`direct_epilogue_metadata/contiguous_tma_review.json`. No implementation,
CPU build or new GPU job for this idea; await current decomposition first.


## Row-ready7254075 complete: narrower dependency loses to added path cost

112 exact +112 replay checks PASS, stderr empty. Ordinary row37.5712/37.7408us
versus direct parent37.5168/37.6064us, equal20-instance ABBA37.6560/37.5616
(+.0944us,.251%). Whole-tile-wait matched control37.7376, history37.7216.
No established gain: reject row_ready as a replacement and retain direct_metadata.
No maintained kernel promotion or commit. Report and raw-derived evidence:
row_ready/comparison_7254075.json/.md, ordinary_7254075.json,
publication_handoff_7254075.json, run_audit_7254075.json.

| us, three captures; metadata origin0 | row subset | direct parent | all owners | history |
|---|---|---|---|---|
| firstactualMMA |9.408/9.600/9.280|8.960/9.184/8.864|9.280/9.856/9.440|56.704/9.984/9.728|
| median75initialpairs |9.792/9.824/9.664|9.344/9.472/9.344|9.696/10.048/9.632|57.312/10.688/10.368|
| latest75initialpairs |10.112/10.336/10.048|10.016/9.824/9.696|10.304/10.432/10.048|57.952/11.584/11.296|
| compute span |24.768/25.312/24.992|25.088/25.792/25.440|25.280/24.896/25.376|24.640/26.048/24.992|
| accumulatorobservation→end |5.472/4.672/5.088|6.304/4.160/5.472|5.216/4.352/4.864|3.744/3.360/4.704|
| remainderwaitmedian |.544/.832/.704|1.184/1.440/1.184|.704/.480/.992|—|

Initialpairs by10us row73/59/70, parent74/75/75, all67/4/73, history0/4/12.
Row absolute compute ends34.176/34.912/34.272 versus parent34.048/34.976/34.304:
shorter span mostly consumes the later start; no demonstrated faster MMA service.
First/broad starts regress~.45us while ordinary kernel slightly loses.

Derived mapping uses actual fixedcase: firsttile expert=CTA//24 (FC1N3072,
BLOCK_N128), source AvalidatorCTA=expert+16, SFCTA=expert. Four consumed rows
map to two adjacent validatorwarps; absolute publicationwarpIDs8..15. 300A and
150SF consumer observations per capture checked. Required Apublications lead
lastwhole-tile publication by only median.128/.080/.128us. Their issue→consumer
observations row1.536/1.344/1.408 versus parent.992/.832/.832 and all1.312/1.488/
1.344. SFissue→observation row1.344/1.472/1.216 versusparent1.248/1.248/1.104,
all1.216/1.248/1.280. Intervals include scheduling/acquire/maskprep andmarker
placement, not isolated cache/fence/network latency. Row/all instrumentation
matches; parent has one publication marker versus eight perCTA. Added path cost
outweighs the small readiness head start in this experiment; no per-warp default.

Fullchain in pass order66.140800/65.638401/65.881601/65.603200/65.440001/
98994.299316us retained. Historical firstMMA56.704 retained; no trimming.
Next: contiguous-TMA source audit in direct_epilogue_metadata/contiguous_tma_review
.json. Materialized rows are contiguous, so regularTMA may reduce issue work;
host descriptor box and completion bytes must change together. No implementation
or new job for this idea. No active GPU after completed capture/analysis.

Terminal accounting: GPU7254075 COMPLETED0:0 in15m46s; CPUanalysis7254076 COMPLETED0:0 in45s. No remaining jobs.


## Contiguous activation TMA prepared; CPU7254572

Row_ready7254075 made progress by rejecting finer certificates: ordinary loss,
~.45us later broad MMA, and no absolute compute-end gain. New candidate starts
from preserved direct_metadata and replaces gather4 by one regular2D TMA
activation issuer (dispatchwarp0) per CTA; SF coordinator remains independent.
Other dispatch warps retain scheduler/phase progression and final retirement.
Native compact-A descriptor boxM changes1→LOAD_BLOCK_M16; completion accounting
changes Abytes2048→4096 percluster/stage at fixedvalid_m16, plus1024SFbytes.
Four barrier arrivals still come only from A/SF and weight coordinators in both
CTAs; the activation issuer performs completion transactions, no extra arrival.

Address audit64cases acrossvalid_m1..32 andbothCTAhalvesPASS: full16-row boxes
remain inside ownedBLOCK_M32, cover every useful MMA row, and extra live rows
are covered by original whole-tile first/full certificate. Padding has no
currentgeneration writer. Existing materialized-capacity guard reserves padded
wholetiles. No newallocation/reset or live-inbox TMA reread. More A bytes might
lose despite fewer instructions; require ordinary ABBA andtimed decomposition.

Frozen bounded device/native C++ overlays in receiver_validate/contiguous_tma;
sourceAPI translation unit copied, native directories explicit so normalrelative
includes select modified host.hpp. Separate native cache for candidate only;
parent/history keep their native/header pairs. CPU7254572 submitted for2device
builds, register gates, nativeC++build and2routerwarmups. Device register gates
PASS128regs24stack0spills; native build active atlastlog. No GPU yet.
Plan tma/direct/direct/tma/history ordinary and3profiles =88exact+88replay.
Remainder markers150locations (oneissuer/CTA) vsparent300, explicit envcontract.
Report labels now say activation-load issue and preserve primitive/byte-count
notes, with existing marker names and numericJSONfields retained.


Before GPU submission, refined the regular-TMA candidate to avoid extra A bytes:
compact_main.cuh/compact_host.hpp use constant eight-row boxes. One box/CTA
covers alignedM16, two cover alignedM32, with matching smem offsets and original
alignedM*BLOCK_K activation completion bytes. This supports every current
BLOCK_M<=32 live extent without a fixedvalid_m16 assumption. Whole-tile readiness
and independent SF retained. Another64 actual-alignment address cases PASS.
Full16-row version remains a matched control to isolate extra read volume.
No live source changed: second device/native overlays and compact_native cache
are separate. CPU7254655 submitted for compact device/native/router preflight;
full CPU7254572 already shows device/native PASS and2routerwarmups in its log.
Revised one-GPU plan: compact/direct/direct/compact/full/history,4profiles112checks.
No GPU submitted yet. More complete experiment avoids attributing any result to
instruction count when padding byte volume differs.


Fullbox CPU7254572 COMPLETE4m39s: device/native/register gates and2routerwarmups
PASS. Compact CPU7254655 remains active; after a slow startup it now reports
COMPACT_TMA_REGISTER_GATES_PASS128regs24stack0spills and is building native C++.
No restart was made: sacct step stayedRUNNING andlaterlogs progressed.
SASS: fullbox ordinary/IKET activation UTMALDG.2D.2CTA at52b0/5f40; noGATHER4.
Compact ordinary sites5330/53c0, second guarded by rows>=9 at52a0→5350, shared
addresses differ1024B (eightrows*128B). IKETsites5f80/6030. Thus fixedrows8/CTA
execute one regular load, alignedM32 executes two. Codegen_review.json retains
contexts. Instruction form alone is not accepted as a performance win.


ContiguousTMA preflights complete: full CPU7254572 4m39s, compact CPU7254655
7m28s;4device/register gates128regs24stack0spills,bothnative C++builds and4
routerwarmupsPASS. GPU7254807 submitted onlyafter sacctverified bothcomplete;
analysis7254808 afterok. Ordinarycompact/direct/direct/compact/full/history,
4profiles112checks expected; no measuredperformance yet. Candidate native/header
pairs remain isolated fromparent/history. Sources frozen; no defaultpromotion.

ContiguousTMA7254807 confirmedRUNNING5m19s on nvl72106-T02/T05. Ordinary
compact8 ABBA37.3568/37.3088 vsdirect37.4944/37.7440us: means37.3328/37.6192
(-.2864us,.761%), both pass directions faster. Medians37.424/37.216 vs37.568/
37.712.32exact+32replayPASS sofar, no native rebuild on GPU. Fullbox/history and
all4profiles pending; no promotion/compute-start claim before decomposition.


## Compact regular TMA7254807 complete: small kernel gain, startup still unresolved

GPU7254807 COMPLETED0:0 in15m09s; analysis7254808 COMPLETED0:0 in48s.
All112 exact-output and112 replay checks PASS; stderr empty. No remaining jobs.
Fixed EP8/rows32, steady preparation outside timing. Six ordinary passes:
compact37.3568, direct37.4944, direct-repeat37.7440, compact-repeat37.3088,
full16-control37.5360, historical pull37.9168us. Equal20-instance ABBA means
compact37.3328 versus direct37.6192us (-.2864us, .761%), faster in both pass
orders; ranges overlap. Retain compact8 as a measured fixed-case candidate.
Keep direct_metadata as control; no maintained/default kernel promotion or commit.
Matching compact_native/compact_headers and prepared-capacity guard are required.

| us, three captures; metadata start0 | compact8 | direct gather parent | full16 | history pull |
|---|---|---|---|---|
| First actual MMA |9.152/9.184/9.120|9.152/9.120/8.832|9.120/9.440/9.472|101.568/10.016/9.632|
| Median75 initial pairs |9.472/9.504/9.248|9.664/9.408/9.408|9.504/9.632/9.696|102.016/10.368/10.400|
| Latest75 initial pairs |9.888/10.560/9.664|10.272/9.600/9.824|10.080/9.952/9.952|102.720/11.744/11.968|
| Compute span |25.600/25.600/25.120|25.696/25.824/26.304|25.312/25.216/25.504|24.480/25.312/24.800|
| Final accumulator observation |34.752/34.784/34.240|34.848/34.944/35.136|34.432/34.656/34.976|126.048/35.328/34.432|
| Accumulator observation to kernel end |5.024/4.512/5.856|5.440/4.096/4.608|5.536/5.120/4.864|4.672/3.936/5.184|
| Median remainder issuer wait |.448/.384/.384|.832/1.344/.960|.832/.896/.736|—|

Initial pairs by10us: compact75/72/75, direct63/75/75, full68/75/75,
history0/0/12. First and broad release are NOT consistently earlier. Compact
absolute compute completion advances in all three captures versus parent,
with roughly25us compute maintained, but no isolated MMA-service claim.
Remainder wait samples150 regular issuers versus300 gather issuers and is
not exact MMA stall. Full16 has extra activation bytes; its one ordinary pass
does not establish a robust byte-volume effect. Legacy MegaGatherIssued now
marks regular activation TMA in these two arms; report states exact semantics.

All-capture comparison: receiver_validate/contiguous_tma/comparison_7254807.md
and JSON, ordinary_7254807.json, result_summary_7254807.json, run_audit_7254807.json.
Full-chain times in pass order65.833600/65.155200/64.588800/64.684800/65.014400/
95485.852051us: no producer-chain win. Historical startup101.568us and huge
full-chain outlier retained without trimming. Instrumented intervals are not
subtracted from ordinary kernel timings.

MEGAMOE_COMM_EXPLORATION.md reread for next direction. Items2/3/4/6 remain
relevant: readiness ownership, first consumable windows, independent issue roles,
and operand-pipeline overlap. Many proposals are already implemented or rejected;
consult this audit/history before repeating. Compact8's smaller remainder wait
makes a modest first-A-window reduction a possible bounded follow-up, but prior
512B-window compute regression still applies as a rejection criterion. This is
an untested idea, not a new submitted job. Keep~25us compute and require both
ordinary improvement and broadly earlier actual MMA. Goal remains active.


## Compact first-window follow-up:1024 versus1536, CPU7255222

Previous turn completed compact TMA7254807 with measured ordinary gain but
inconsistent earlier first/broad MMA. Current authoritative accounting confirms
both7254807/7254808 terminal. One-line device experiment compact_window changes
kFirstPayloadWindowBytes1536->1024 on compact8, preserving native descriptor,
main kernel, sender, SF, metadata, combine and every fence/join/readiness bit.
Prefix validates4 rather than6 uint4s per half-warp lane; remainder6 rather than5
per warp lane. Full-A acquire moves Kstage12->8 in existing10-stage pipeline.
66 ownership cases (both windows,ready0..32) prove complete disjoint coverage.

Hypothesis: reduced prefix work advances broad MMA; risk is underfilled pipeline
and exposed remainder wait. Previous512B7245584 regressed ordinary~1.26us and
compute~1.28us and remains relevant. This tests1024 on changed compact-TMA path,
not a presumption smaller is better. Require ordinary ABBA gain, broad actual
MMA starts and~25us compute with all outliers retained. FixedEP8/rows32 only.
Plan1024/1536/1536/1024/history ordinary, three profiles,88exact+88replay.
CPU7255222 submitted for2device builds and reachable-register gates; native and
router unchanged/reused. No GPU until preflight passes; no promotion or commit.


Compact_window CPU7255222 COMPLETED0:0 in3m56s: ordinary/IKET128registers,
24stack0spills, all reachable-register gates PASS. SASS keeps regular2CTA TMA
(noGATHER4); native descriptor/router are unchanged and reused. GPU7255276
submitted only after terminal preflight verification; CPUanalysis7255277 afterok.
Three profiles and ABBA+history ordinary,88exact+88replay expected. Sources
frozen; no measured result or promotion yet. Artifact:receiver_validate/compact_window.

Next possible structural axis, conditional on this decomposition: current A
validators finish and publish prefix before beginning remainder. Existing SF
CTAs have four inactive worker warps after initialization; assigning a separately
owned remainder there might overlap validation while preserving prefix8warps.
This needs a new disjoint ownership/publication and barrier proof and may slow
SF through contention or remainder through fewer workers. Do not implement or
launch before reading current window test; untested idea only.


Compact_window7255276 ordinary ABBA complete:1024=37.7760/37.8272,
1536=37.3792/37.4880us, equal20-instance means37.8016/37.4336
(+.368us,.983%), slower both pass orders. History37.8720. Retain1536;
smaller window is not a replacement.72exact+72replayPASS,stderr empty;
both all-push profiles complete, history/analysis pending. No causal claim yet.
Additional source audit confirms first75pairs mapexperts0..5:12each and6:3;
experts7..15 are not initial compute work. A possible fixed-resource alternative
pairs16A validatorCTAs over first8experts thenexperts+8 (vs currentoneCTA/all16).
Unlike rejected32Aworkers, totalCTAs staysfixed. Requires row ownership and
bounded-counter proof, later-expert progress, and honest subset marker mapping.
Audit compact_window/initial_expert_priority_review.json; no implementation/job.


## Compact window7255276 complete: earlier startup loses overlap

GPU7255276 COMPLETED0:0 in12m17s on nvl72136-T10/T11;
analysis7255277 COMPLETED0:0 in47s.88exact+88replay PASS; stderr empty.
No remaining jobs. CPU7255222 gates passed128registers24stack0spills.
Reject1024 alone. Ordinary ABBA1024=37.8016 versus1536=37.4336us
(+.368us,.983%), slower both pass directions. Historical pull37.8720us.
Keep compact1536 as measured best path; no maintained/default promotion/commit.

| us, three captures; metadata start0 | window1024 | window1536 | history pull |
|---|---|---|---|
| First actual MMA |8.928/8.800/8.960|8.896/9.248/9.280|11.936/10.592/9.952|
| Median75 initial pairs |9.504/9.248/9.408|9.376/9.632/9.728|13.120/10.912/10.720|
| Latest75 initial pairs |9.760/9.792/9.856|9.696/9.856/10.176|13.472/11.232/11.232|
| Compute span |25.696/25.856/25.760|25.376/25.216/25.184|25.216/24.640/25.120|
| Final accumulator observation |34.624/34.656/34.720|34.272/34.464/34.464|37.152/35.232/35.072|
| Accumulator observation to kernel end |5.504/5.184/5.184|5.856/4.928/5.600|3.456/4.128/4.224|
| Median remainder issuer wait |1.008/1.072/.864|.224/.320/.384|—|
| Remainder wait starts after own pair MMA, median |.640/.544/.672|1.440/1.408/1.376|—|
| Validation start to first-A publication, median |1.392/1.616/1.456|1.536/1.872/1.648|—|

First and broad MMA starts improve in2/3captures; medians first8.928vs9.248,
broad9.408vs9.632. Initialpairs by9.5us35/58/43vs44/23/6; by10us75/75/75
vs75/75/59. Yet compute median grows25.216->25.760us and absolute completion
is later inall3captures. The issuer reaches remainder wait~.7-.9us earlier
relative to its pair's firstMMA; wait grows by.48-.784us. Both arms have150
matched issuer locations. This supports lost overlap from8 versus12 initially
ready Kstages; issuer wait is not an exact MMA-stall measurement. First/full
publications, SF, local and producer starts retained in detailed comparison.

Fullchain ordinary pass order63.536000/64.044800/64.080001/63.478401/
104348.535156us. Candidate fullchain ABBA63.507201vs64.062401us moves opposite
to kernel-only timing. Preserve it: max-rank producer+kernel scope differs from
rank0 MegaMoE-only Nsight measurement; these traces do not isolate launch/router/
inter-rank effects. Kernel criterion fails despite that separate fullchain result.
Historical outlier retained, no trimming or mixed timing subtraction.

Evidence receiver_validate/compact_window/comparison_7255276.json/.md,
result_summary_7255276.json, ordinary_7255276.json, run_audit_7255276.json.
Previous turn was progress: completed measurement and rejected shrinking alone.
Next structural candidate: overlap prefix and remainder validation using idle SF
worker warps, retaining1024 startup gain only if compute/kernel recover. Read
remainder_overlap_review.json first:4 idle versus8 current remainder workers may
create a new per-warp serial-row bottleneck; require ownership/barrier/marker proof.
Separately initial_expert_priority_review.json records a fixed16A-CTA alternative:
two row owners/expert for first8experts, thenexpert+8. Initial75pairs use only
experts0..6, so this prioritizes currently consumable work while keeping resources
fixed, unlike rejected32Aworkers. Both are unimplemented; choose bounded tests
from completed evidence, preserve best1536/control/history and exact timing scope.
Goal remains active. MEGAMOE_COMM_EXPLORATION.md items2/3/4/5 inform these axes.


## Independent remainder validation prepared; CPU7255515

Previous turn completed compact_window7255276:1024 starts earlier in2/3captures,
but its remainder wait and absolutecomputeend regress. Now implement structural
overlap from that evidence. parallel_remainder/dispatch.cuh keeps1024 prefix in
16A CTAs/eightwarps, moves remaining3072B to SF CTA workerwarps4..7 (physical
12..15), which were idle. They start after route/count acquisition concurrently
with SF and prefix. No prefix-publication join on their issue path. Four remainder
warps process4rows each at fixedcase, versus eightwarps/tworows before; this may
offset overlap or interfere with SF/prefix and must be measured.

Each word still validated and exact observedvalues materialized to disjoint
owned row intervals. Prefix uses barrier3/256 and bit2; SF barrier4/128 bit1;
remainder now barrier5/128, publisherworker4/tid128 bit4. Main namesFullAReady
remain legacy: bit4 alone certifies remainder, not prefix. Existing main acquires
bit2 beforeK0 andbit4 beforeK8; both proxyordered. Initial/final barrier3/256,
DispatchDone, common retirement/cleanup retained. Main/native descriptors/TMA
bytes,16SF+16A startupCTAs,8senders150compute and reset timing unchanged.
Ownership33cases ready0..32 and6release-order permutations PASS.

New direct GenphaseRemainderValidateStart appears in candidate and matched
serial1024/best1536 controls:30marker types. Candidate locationSFworker4,
controlsAworker0; publication paired by exactCTA/warp. Maintained profiler and
summarizer support it and reject missing/duplicate/reversed pairs. Existing
three-arm7255276 JSON summaries unchanged after serialization-normalized check;
focused pairing cases and bash syntax PASS. No code injection.

CPU7255515 submitted: candidate normal/IKET, both matched-control IKET builds,
reachable-register gates and6CPU routerwarmups. Nativecompact8 reused. Source
frozen; no GPU yet. Plan parallel1024/best1536/best_repeat/parallel_repeat/
serial1024/history ordinary plus4profiles=112exact+112replay. Require ordinary
ABBA gain vs retained1536 and restored~25us compute with broadearlierMMA;
serial1024 isolates concurrency. No defaultpromotion or commit. Expert-priority
source audit remains a separate unimplemented alternative.


Parallel_remainder CPU7255515 COMPLETED0:0 in4m54s: all4device builds and
reachable-register gates128regs24stack0spills,6routerwarmupsPASS. SASS has
independent barrier5/128 then GPU release reduction (ordinary2780/2880,
IKET2600/2700), normal activationTMA/noGATHER4.30marker types compile.
Raw repeated publication markers are permitted by existing profiler semantics;
new remainder pairing retains per-location timestampbounds/counts, rejects
missing/reversed locations. Three old7255276 summaries remain unchanged;
raw-repeat/missing/reversed checks pass. Earlier duplicate-rejection draft was
corrected before GPU submission, not a runtime fallback.
GPU7255606 submitted after terminalCPU verification; analysis7255607 afterok.
Six ordinary passes/fourprofiles112checks, sources frozen; no measured result.
No defaultpromotion/commit. Candidate retains1024 only as experiment until it
beats best1536 with broadearlyMMA and compute~25us. Goal remains active.


## Parallel remainder 7255606: earlier broad MMA, replacement rejected

GPU7255606 completed0:0 in17m07s on nvl72049-T09/T18; analysis7255607
completed0:0 in50s. All112 exact and112 replay checks pass; GPU/analysis stderr
empty. CPU7255515 previously passed four builds,128registers/24stack/zero spills,
and six router warmups. Fixed EP8 rows32/rank,H4096,I1536,E128,topk8,8senders/
150compute. Steady buffer initialization/reset excluded from kernel timing.

Ordinary ABBA parallel1024=37.5232 vs retained1536=37.3760us
(+.1472us,+.394%); slower in both orders (+.0352,+.2592us).
Single-pass serial1024=37.8656, historical pull=37.7408us. Concurrency helps
versus the smaller serial window's ordinary result but does not beat retained1536.
No promotion/default change/commit. Preserve all samples and controls.

| us, each of three captures, metadata start0 | parallel1024 | retained1536 | serial1024 | historical pull |
|---|---|---|---|---|
| First actual MMA |8.512/8.960/8.672|9.216/9.120/9.120|8.832/9.184/8.672|13.152/9.920/9.824|
| Median75 initial pairs |8.800/9.216/9.056|9.568/9.408/9.504|9.152/9.312/8.960|14.016/10.528/10.560|
| Latest75 initial pairs |9.376/9.728/9.472|9.952/10.144/9.888|12.384/9.568/9.664|14.432/11.072/10.976|
| Compute span |25.408/25.664/26.240|25.024/24.992/24.960|26.944/25.312/25.184|24.512/25.120/24.832|
| Final accumulator observation |33.920/34.624/34.912|34.240/34.112/34.080|35.776/34.496/33.856|37.664/35.040/34.656|
| Accumulator observation to kernel end |5.536/5.056/4.704|5.888/4.800/5.504|7.424/4.704/6.208|3.936/4.320/4.864|
| Median remainder issuer wait |1.344/1.472/1.408|.352/.288/.352|1.152/1.216/1.184|—|
| Remainder validation start to publication, median16 |3.728/3.824/3.712|1.872/1.808/1.824|2.176/2.112/2.208|—|
| Remainder wait begins after own pair MMA, median150 |.512/.544/.544|1.344/1.312/1.360|.576/.576/.560|—|

First/median/latest MMA all improve in all3 captures versus1536. All75pairs
start by9.5us in2captures (75/72/75 vs17/48/36). Median first/broad advance
.448us, but compute grows.672us and absolute completion worsens in2/3captures.
The earlier MMA is real; ordinary kernel criterion still fails.

New matched producer markers prove remainder begins before prefix publication:
median -1.376/-1.520/-1.488us, versus approximately0 for both serial controls.
Its four workers still finish one row before issuing the next (four rows/warp
at this case versus two with eight workers). Remainder start-to-publication
includes arrival polling, descriptor/load/copy work, fence and join; longer
interval alone does not establish that row serialization is the cause. Earlier
polling naturally includes more arrival wait. Remaining publication follows
prefix by2.384/2.272/2.432us, versus serial1024 2.176/2.128/2.224us.
No evidence of SF/prefix startup regression versus1536: prefix validation to
publication1.392/1.424/1.440 versus1.872/1.808/1.872us.

Fixed initial-consumer mapping expert=CTA//24 was used for all150 consumers.
Remainder publication-to-ready median .928/.928/.896us versus1536
1.216/1.152/1.120us; this includes when the consumer reaches its acquire, not
isolated notification latency. Prefix publication-to-observation .992/.832/.896
versus .960/.832/.896us; SF 1.088/.960/1.024 versus .960/.992/.928us.
Publication markers follow release issue, not completion; issuer waits are not
exact MMA stalls. Local producer/consumer association retained in result JSON.
Outgoing sender markers are not incoming transfer causality.

Ordinary producer+kernel max-rank times in execution order:
67.017600/65.929600/63.276801/66.777601/66.304000/116507.714844us.
Candidate ABBA66.897600 vs1536 64.603201us; no full-chain gain either. Keep the
large historical outlier; this scope differs from rank0 Nsight kernel-only.

Evidence: receiver_validate/parallel_remainder/comparison_7255606.{json,md},
result_summary_7255606.json, ordinary_7255606.json, run_audit_7255606.json.
Maintained summarizer handles remainder starts and exact producer pairing;
raw repeated publication records retained with counts/bounds. Twelve current
capture summaries pass pairing checks. Historical behavior was previously checked.

Next source-only audit: remainder_row_window_review.json proposes issuing two
independent rows before waiting, with per-row progress, retained exact values,
unchanged ownership/retirement and register/spill gate. It targets the remaining
row-to-row dependency, not another window-size-only sweep. No implementation or
new job yet. Alternative readiness_spacing_review.json records packed SF/A words
and ~1us handoff observations; cache contention is a hypothesis, requiring an
address-relocation control. initial_expert_priority_review.json remains another
fixed-resource alternative. MEGAMOE_COMM_EXPLORATION.md items1/2/4/6 inform these
choices. No remaining jobs. This turn completed measurement and rejected a
replacement; the latency-overlap goal remains active.


## Remainder two-row issue window prepared; CPU7256230

Previous turn made progress: completed7255606 decomposition and rejected its
replacement despite earlier broadMMA. New remainder_row_window/dispatch.cuh
changes only the four remainder workers: hold two row states, issue both batches
of6uint4/lane before testing either, independently store fully valid rows and stop
polling those rows. A missing row cannot delay its peer's load issue or copy.
Advance by8rows/warp after both finish. Prefix1024/SF roles, nativecompact8,
barrier5/128, bits1/2/4, generations and all retirement remain unchanged.

Ownership checks pass ready0..32: every row once; six vectors/lane span exactly
remaining3072B with no prefix overlap. Pending flags begin and remainwarp-uniform,
partial secondrow performs no reads, exact observedwords stored before register
reuse. Values48words/lane vs24; reachable startup128reg/zero-spill gate mandatory.
No performance claim until actual SASS and EP8 capture. Direct markers unchanged.

CPU7256230 submitted and confirmed RUNNING oncpu-0016: two devicebuilds/register
gates andtwo routerwarmups for newheaderkey. Existing compact_native and frozen
best1536/parallel1024/history controls reused. No GPUyet. Prepared6ordinary/
4profile schedule (112exact+112replay), row2/best/best_repeat/row2_repeat/parallel/
history. SameEP8/resetexcluded. Compare ordinaryABBA plus75pair first/broadMMA,
remainderstart/publication/wait and computeabsolutecompletion. Source frozen.
Artifact receiver_validate/remainder_row_window; contract/patch/ownershipreview
and reusable runner JSONs retained. No defaultpromotion/commit. Goal active.


Remainder_row_window CPU7256230 COMPLETED0:0 in2m20s; two device/register
gates128registers24stackzero spills andtwo routerwarmupsPASS. SASS bothbuilds
has12independentSYSuint4loads beforefirstsentinelcomparison,48wordcomparisons,
12stores and unchangedbarrier5/128 thenGPUreleasepublication. Ordinary loads
2750..2800 beforefirstsentineltest2860; independentregistersR40..63/R64..87.
IKET2630..2700 likewise. codegen_review.json retains compactevidence.
GPU7256310 submitted afterCPUterminalverification; CPUanalysis7256311 afterok.
Sixordinary/fourprofiles112exact+112replayexpected, fixedEP8 andresetexcluded.
No runtime/performance result yet; sources frozen, no defaultpromotion/commit.
This turn implemented andvalidated the next latency-overlap experiment; goalactive.


## Two-row remainder window7256310 rejected: compute code generation regresses

GPU7256310 completed0:0 in14m51s on nvl72054-T05/T06; CPUanalysis7256311
completed0:0 in48s.112exact+112replayPASS across all10sections; bothstderr empty.
CPU7256230 previously completed2m20s,128registers24stackzero spills, two router
warmups. Same fixedEP8 case and offcritical reset. No jobs remain.

Ordinary ABBA row2=40.168 vsbest1536=37.408us (+2.760us,+7.378%), worseboth
orders (39.872vs37.488 and40.464vs37.328). Single-row parallel1024 parent37.8912;
historypull37.872us. Reject replacement, retaincompact1536, no promotion/commit.

| us, allthree captures; metadata starts0 | row2 | best1536 | parallel1024 parent | historypull |
|---|---|---|---|---|
| First actual MMA |9.056/8.928/9.056|9.056/9.152/8.864|9.024/9.056/8.832|78.240/10.048/9.600|
| Median75 initial pairs |9.280/9.216/9.504|9.536/9.376/9.312|9.344/9.280/9.152|79.040/10.816/10.528|
| Latest75 initial pairs |10.048/9.792/10.080|9.952/9.856/9.792|9.728/9.472/9.792|79.680/11.040/10.944|
| Compute span |28.736/28.832/29.216|25.344/25.088/24.896|25.568/25.408/25.952|25.184/25.184/25.056|
| Final accumulator observation |37.792/37.760/38.272|34.400/34.240/33.760|34.592/34.464/34.784|103.424/35.232/34.656|
| Observation to kernel end |6.080/5.024/4.960|5.408/5.216/6.080|4.960/4.992/5.024|3.584/4.192/4.704|
| Remainder validation to publication, median16 |3.696/3.664/3.632|1.952/1.920/1.952|3.984/4.288/3.792|—|
| Remainder issuer wait, median150 |.512/.448/.416|.352/.384/.352|1.280/1.376/1.376|—|
| Wait starts after own pair MMA, median150 |1.120/1.024/1.088|1.408/1.408/1.408|.592/.608/.576|—|

No consistent first/broadMMA gain overparent; median75 unchanged9.280us and
latest75 laterall3. Compute median grows25.568->28.832us (+3.264us), best25.088.
Absolute compute completion laterall3. Newrowloads do modestly reduce receiver
validation-to-publication (median3.984->3.664us), but the large issuer-wait drop
1.376->.448us is partly later issuer arrival, not pure readiness gain. Matched150
consumer wait-start/ready median timestamps row2=10.336/10.880,
10.240/10.752,10.592/11.104 versusparent9.984/11.200,9.920/11.296,9.728/11.024.
Ready earlier2/3, laterthird. Per-expert publication association expert=CTA//24;
publication isissue, notcompletion, and issuerwait isnotexactMMAstall.

Concrete SASS difference despite unchangedcompact_main.cuh: reachable reduced-
register region instruction counts parent[366,255,325,1,296] versusrow2
[389,298,361,1,328]. In contiguousregion1, row2 adds43R2UR and replaces17
UISETP.NE with17ISETP.NE; region2 adds23R2UR. Extra general-register ALU,
reconvergence/control and register transfers show lost uniform code generation.
Static counts are not executed-instructioncounts or isolatedcycle costs, but
this is a plausible contributor supported by much longer compute and no spills.
Both ordinary/IKET startup SASS correctly issue12SYSuint4loads beforevalidity
comparisons; the requested load overlap is present. Do not blame failed overlap
or attribute every compute microsecond to compiler changes without an isolation.

Next bounded CPU check: simplify two-row polling into onecombinedwarp vote/loop,
retainingbothloadbatches beforetests, everywordvalidation andexactvalues. Current
readinessbit4 needsallrows, so independent earlyrow stores do not themselves
release a consumer. Simpler control may preserve uniform compute codegen; the
tradeoff is re-polling an alreadyvalid row. Compare SASS withparent beforeanyGPU;
requirezero spills and preserved role/register invariants. No implementation/job
yet. Noninlinehelper is only an alternative if simplercontrol doesnotfixcodegen.
Descriptor-load issue serialization also recorded, secondary to compute regression.

Ordinary max-rank producer+kernel in executionorder:
66.835201/65.990401/63.904000/67.772801/66.646400/101323.767090us.
CandidateABBA67.304001 vsbest64.947201us, also worse. Historical fullchainoutlier
and instrumented78.240us firstMMA retained; scopes notmixed withordinarykernel.

Evidence receiver_validate/remainder_row_window/comparison_7256310.{json,md},
result_summary_7256310.json, ordinary_7256310.json, run_audit_7256310.json,
compute_codegen_review.json, codegen_review.json, descriptor_issue_review.json.
Detailed maintainedreport includes operationstarts and all3captures sidebyside.
This turn completed measurement, rejectedreplacement andidentified a concrete
compiler/codegen direction. Goalactive; no defaultpromotion/commit.


## Simpler two-row polling prepared; CPU7256760

Previous turn completed7256310 and identified longercompute plusconcreteuniform
codegen loss. New remainder_batch/dispatch.cuh removesmutable per-rowpending
branches. Bothrowloadbatches stillissuebeforeanyvaliditytest; onecombinedwarp
vote covers allwords frombothactiverows, thenexact finalobservations materialize.
Same33rowownership cases/3072Bextent, partialrowguards, fourremainderwarps,
barrier5/128, bits1/2/4, prefix/SF/main/native andretirement. It mayrepoll already
validrow whilepeerpending; existingtilecertificate requiresboth regardless.

CPU7256760 submitted: two newdevicebuilds/registergates andtwo routerwarmups.
NoGPUyet. Require12loads beforefirstvaliditytest and no newcompute codegen
regression versus single-rowparallelparent; comparewithrejectedindependentrow2.
Prepared sameEP8 ordinaryABBA vsbest plusparallelparent/history andallcapture
latencydecomposition ifCPUevidence warrants. Candidate sources frozen, no
promotion/commit. Goalactive. Artifactreceiver_validate/remainder_batch.


Simplerinline vote alone failscodegen gate: CPU7256760 completed3m51s with
128regs/zero spills/two routerwarmups, butordinary/IKET compute counts remain
exactlythe regressedrow2 counts. NoGPU spent onthisvariant. Preserve dispatch.cuh
anditscompiled evidence. Stronger isolation: noinline_dispatch.cuh adds a device
function boundary aroundstartupdispatch. CPU7256778 completed54s; ordinary
reachablecounts restoreexactly toparent[366,255,325,1,296] fromrow2
[389,298,361,1,328], withsamefirstfourcontiguous opcodecounts. Calleezero stack/
zero spills, kernel128regs24stack, localmemoryops matchparent's existingones.
Bothbuilds preserve12SYSvectorloads before48wordcomparisons andbarrier5/128.
IKETcounts[405,255,337,1,303] differparent[424,260,331,1,304]; instrumentation
remainsseparatescope. Thisrestoresordinarycodegen, notyetprovenruntimecompute.

GPUcases nowselectnoinline_headers/noinline_dispatch.cuh, freshJITcache; direct
markers remainincallee, no sourceinjection. CPU7256824 warms two newheader-key
routers beforeGPU. NoGPUyet. Main/native andalllifetimeinvariantsunchanged.
Candidatechangescombinedpoll plusfunctionboundary versusrow2; no purelysingle-
variable runtimecausalclaim. Evidencecompute_codegen_review.json,
noinline_codegen_review.json,noinline_transport_review.json. Goalactive.


Noinlinecandidate CPU7256824 routerwarmups completed0:0 in2m09s, bothPASS.
Afterterminalverification ofdevice7256778 androuter7256824, GPU7256840 submitted;
CPUanalysis7256841 afterok. Sixordinary/fourprofiles112exact+112replayexpected.
SameEP8, bufferinit/resetoutsidekernel; comparebatch2/best1536 ABBA withparallel
single-rowparent/history controls anddirecttimingstarts. Sources frozen. No
runtime/performance claimyet, no defaultpromotion/commit. Thisturnmadeprogress:
simplificationfailedCPUgate, functionboundaryrestoredordinarycomputecodegen,
and stronger candidate nowawaitsmeasureddecomposition. Goalactive.


## Whole-dispatch function7256840 rejected; profiling failure; narrower helper

GPU7256840 FAILED1:0 in10m21s on nvl72118-T10/T12; analysis7256841 CANCELLED
dependency. Allsixordinarypasses completedwith48exact+48replayPASS. Candidate
profilingdryrun another8+8PASS; instrumentedrun neverstarted. IKET process_targets
func.get_instruction_by_pc raises IndexError2778>=2771 whena marker in the
noninlineddispatchcallee liesbeyond callerinstructionrange. No decomposition
exists. Do notclaimmeasuredcompute recoveryfromrestoredordinarySASS. No
profilerinjectionpatch/fallback/retry ofsamefailedsetup.

Ordinary ABBA batch2=38.136 vsbest1536=37.4064us (+.7296us,+1.9505%), losesboth
orders:38.1152vs37.4752 and38.1568vs37.3376. Single-rowparallelparent37.6832,
historypull37.8592us. Rejectreplacement; retainbest. Fullchain max-rankproducer+
kernel inexecutionorder66.828800/65.510400/64.300800/64.755199/64.694400/
94212.744141us; historicaloutlier retained. Thesearevalidordinarymeasurements,
notinstrumentedphase durations. run_audit_7256840.json records56+56 andfailure;
ordinary_7256840.json andcompare-7256840.err retained. No correctnessfailure
reported, but no newtimeddecomposition and no defaultpromotion/commit.

Sourceaudit next_latency_review.json: whole-dispatchcallee reloadsparameterbase
and repeatedlymovesdescriptors fromgeneral to uniformregisters, whereasparent
inlinepath keepsconstantdescriptors. Thismayaddstartupcost; notmeasuredcausal.
Anotherdeferredidea splitsremaining3072 into512+2560 windows, butrequireshonest
multi-phase waitmarkers; do notimplementbeforediagnosingcurrentpath.

Nextcandidate remainder_helper/dispatch.cuh keepsmega_moe_dispatch forceinline
and extracts onlytwo-rowcombinedvote validation/materialization into anunmarked
noinlinehelper. AllGenphase/timingmarkers remainincaller; metadata/SF/prefix/
sender code staysinline. Rawdescriptor/inputgeneration/outputtile pointers plus
ready/owner/lane avoidpassingWorkspaceaggregate.384representativeaddresscases
acrossbothslots/8ranks/tokens/topksPASS; same33rowownership and3072Bextent,
everywordvalidation/exactobservations, barrier5/128 andbits1/2/4/retirement.
Existingfirstremainder marker precedespointerprep/call; publication markerfollows
callerfence/join/release. No markersorpublications insidehelper.
CPU7256953 submitted: ordinary/IKET device/registergates andtwo routerwarmups.
Require restoredcompute codegen andnoPMTRIGincallee beforeGPU. NoGPUyet;
sourcesfrozen. Thisturncompletedordinarycomparison, diagnosedrealprofilerfailure,
andimplementednextisolation. Goalactive; no blocked/complete claim.


Narrowhelper CPU7256953 COMPLETED0:0 in2m17s: two device/register gates
128registers24stackzero spills andtwo routerwarmupsPASS. Helper0stack andno
localmemoryinstructions;12SYSuint4loads before48wordcomparisons. No PMTRIG in
helper. IKETcompute reachablecounts[424,260,331,1,304] exactlymatchparallelparent;
ordinary[384,260,325,1,299] vsparent[366,255,325,1,296], soordinarynotidentical.
Allreducedrole budgetsPASS. codegen_review.json retainsplacement/evidence.

InitialGPU7256991 failed1:0 in19s beforekernels: profile_megamoe_iket.sbatch
requiresbaseline.json fromanordinarycapture beforeprofile. The attemptedprofile-
firstplan wasinvalid. Analysis7256992cancelled; no correctness/performance result.
Afterterminalverification, correctedexplicitcapture_plan.json: helper2ordinary,
helper2profile, remaining5ordinary preservingABBAsequence, then3controlprofiles.
Candidateprofiling nowvalidatedbeforemostcomparisons, but interleavesafterfirst
ordinarypass; preserveexecutionorder anddo notcallit anuninterruptedABBA.
Expectedsectioncounts[8,16,8,8,8,8,8,16,16,16], total112exact+112replay.
Genericcapture.py executesexplicitplan/printsCAPTURE_STEP; analysisall4arms.
GPU7257013 submittedwithanalysis7257014afterok. Sources frozen,no resultyet.
No profilerpatch/injection workaround, no defaultpromotion/commit. Goalactive.


## Narrow remainder helper: 7257013 completed; confirmation 7257222

GPU7257013 completed 12m56s on nvl72056-T01/T11; CPU analysis7257014 completed
52s. All 112 exact-output and 112 replay checks passed, stderr empty. Explicit
10-step plan audited; candidate profile interleaves after its first ordinary
pass. No marker inside the helper, so all four detailed captures now succeed.

Ordinary kernel ABBA: helper 37.2416us vs retained best1536 37.3328us
(-0.0912us, -0.2443%). Both orders improve: 37.2768 vs37.3312 and37.2064
vs37.3344. Parallel parent37.5168, history pull37.8944. Only20 instances/arm;
per-pass standard deviations .1766–.3191us, so this tiny lead is provisional.
No default promotion. CPU codegen recovery is now supported by measured compute
recovery, although ordinary and instrumented scopes remain distinct.

All three instrumented captures, metadata-transfer start=0, us:

| Metric | helper2 | best1536 | parallel1024 | history pull |
|---|---|---|---|---|
| First MMA | 8.864/9.088/8.928 | 9.088/9.280/8.992 | 8.864/9.024/8.928 | 20.160/10.112/10.080 |
| Median of75 MMA pairs | 9.312/9.376/9.472 | 9.728/9.440/9.504 | 9.184/9.312/9.280 | 21.056/10.496/10.688 |
| Latest of75 MMA pairs | 9.824/10.048/10.016 | 10.080/10.144/9.984 | 9.600/9.728/9.856 | 21.280/11.232/11.072 |
| Compute span | 25.280/25.280/25.824 | 25.120/25.184/25.152 | 26.080/26.432/25.888 | 24.608/25.024/25.760 |
| Compute end | 34.144/34.368/34.752 | 34.208/34.464/34.144 | 34.944/35.456/34.816 | 44.768/35.136/35.840 |
| Post-compute tail | 6.176/5.152/5.184 | 5.856/5.312/6.432 | 5.472/4.448/5.120 | 4.096/4.800/3.776 |
| Remainder validation→publication median16 | 3.136/3.472/3.232 | 1.888/2.096/1.888 | 3.856/4.384/3.936 | — |
| Remainder issuer wait median150 | 0.480/0.768/0.480 | 0.224/0.352/0.320 | 1.312/1.472/1.488 | — |
| Wait start−own pair MMA median150 | 0.608/0.608/0.608 | 1.408/1.408/1.440 | 0.608/0.640/0.544 | — |
| Remainder publication→observation median150 | 0.896/0.928/0.928 | 1.120/1.120/1.152 | 0.896/0.992/0.960 | — |

Compared with the parallel parent, producer validation→publication improves
0.704–0.912us and issuer waiting improves0.704–1.008us. Wait start relative to
its own MMA remains approximately0.608us (parent0.544–0.640), unlike the rejected
row2 variant's delayed issuer. Thus wait reduction is accompanied by earlier
readiness, not mostly later admission. Publication→observation stays about0.9us;
this includes consumer scheduling/polling and is not isolated notification cost.
Initial expert mapping CTA//24; remainder publisher CTA=expert for helper/parent,
16+expert for best1536. All150 consumer waits and75 MMA pairs validated.

Against best1536, first and median MMA improve in all3 captures; latest improves
2/3. Compute is0.160/0.096/0.672us longer by paired capture,
so compute end improves2/3 only. Median compute25.280 vsbest25.152/history25.024.
Against parallel parent, first MMA is unchanged2/3 and .064us later1/3, broad
MMA later all3, but compute span/end improve all3. This is remainder latency
recovery, not an across-the-board startup win. Historic20.160us first-MMA outlier
retained. Publication issue is not completion; issuer wait is not exact MMA stall.

Max-rank producer+kernel ordinary in execution order:
65.705600/63.884800/64.502400/66.422401/65.532800/96584.802246us.
Helper ABBA66.064001 vsbest64.193600: full chain still worse. History outlier
retained. Do not mix this scope with the requested MegaMoE-only optimization.

Next: uninterrupted ordinary-only ABBA,100 timed instances/pass, same frozen
headers/native/cache and EP8 case. GPU7257222 submitted (10min limit), running
on nvl72087-T09/T10. No repeated IKET or CPU build needed; four normal phases,
32 exact+32 replay expected. capture.py now accepts --plan and explicit step
environment overrides; run.sbatch forwards arguments. Original capture_plan.json
retained, new confirmation_plan/contract/job manifests preserve provenance.

Evidence: remainder_helper/comparison_7257013.{json,md}, result_summary_7257013.json,
ordinary_7257013.json, ordinary_dispersion_7257013.json, run_audit_7257013.json.
Sources frozen while7257222 runs. Goal remains active; no commit/default promotion.

Read-only follow-up: readiness_handoff_review_7257013.json matches each of75
MMA pairs to its expert publisher. Helper median SF publication→last observation
1.152/1.216/1.184us; prefix A1.120/.800/1.152us; max(SF,A publication)→MMA
1.600/1.344/1.600us. Includes consumer scheduling and publication completion,
not isolated atomics/network cost. MEGAMOE_COMM_EXPLORATION item4 plus existing
parallel_remainder/readiness_spacing_review.json suggests a controlled flag
relocation/spacing experiment if current candidate survives confirmation. No
implementation/job yet; require original-packed, relocated-contiguous control
and128B spacing, same readiness/reuse proof and compute-codegen gate.


## Helper confirmed and retained for fixed EP8

GPU7257222 completed0:0 in5m00s on nvl72087-T09/T10. Uninterrupted ordinary
ABBA,100 instances/pass: helper37.1549/37.1782 vsbest37.2429/37.2451us.
Combined helper37.16655 vsbest37.24400us (-.07745us,-.208%),200 samples/arm;
forward-.08800us, reverse-.06690us. Both orders also won initial7257013.
All32 exact+32 replay checks PASS, stderr empty, same fixed EP8 contract and
off-critical reset. Across both jobs144 exact+144 replay PASS. No live jobs.

Retain narrow helper as best measured ordinary kernel path for this fixed case,
with best1536 preserved as control. The gain is small and repeated; no statistical
independence claim for replay samples. Decomposition shows real remainder-wait
recovery versus parallel parent while median compute25.280us remains near
best25.152/history25.024. First/median MMA earlier thanbest all3, compute end2/3.
The main successful change batches independent row loads in a narrow unmarked
helper, avoiding the inline compiler regression and whole-dispatch call cost/
profiler failure. Remaining handoff analysis is retained as an unexplored idea.

Separate producer+kernel confirmation helper56.181280 vsbest56.357280us; prior
7257013 full chain regressed, so no consistent full-chain gain claim. Longer
iteration count changes aggregate overhead; do not compare these full-chain
means directly with the earlier10-iteration job.

Evidence remainder_helper/RESULT.md, decision.json, ordinary_7257222.json,
run_audit_7257222.json; detailed all-capture table comparison_7257013.md.
Experimental header overlay requires existing prepared-capacity guard and
matching compact native descriptor. No general/default promotion or commit.
This optimization/measurement iteration is complete; future flag spacing is
explicitly unimplemented, not required evidence for this result.


## EP16 transferability validation requested September18

User notes possible EP8 overfitting and requests Qwen3 EP16 plus Qwen3.5 EP16.
Keep helper and previous best1536 unchanged, with historical pull for exact
per-rank reference and timing. Selected Q3 global192/rows12,H4096/I1536/E128/K8;
Q35 global512/rows32,H4096/I1024/E512/K10. Both choose BLOCK_M32 naturally,
150 compute CTAs,8 senders, bucket3840, off-critical reset. Capacity upper bounds
rows+local_experts*32=268/1056, below3840. Existing row/SF ownership covers
partial tiles0..32 and topk descriptors; EP16 source addresses remain64-bit.

Portability limitation found before GPU: Q35 maximum-case global912/rows57
selects BLOCK_M64 and128 epilogue threads, rejected by existing compact path's
BLOCK_M<=32 and256-worker requirements. Not retiled or silently substituted:
report unsupported912 separately and explicitly test512. This limitation alone
prevents claiming general model/batch coverage from successful small-case tests.

Run records remainder_helper/ep16_validation reuse existing headers/native caches
(no kernel source copies or tuning). CPU7259660 prebuilds six ordinary device
specializations and checks reduced-warp register budgets; GPU not submitted yet.
Prepared one EP16 four-node allocation: for each model historical pull records
reference then helper/best/best/helper,100 timed iterations each. Expected32
record+128 compare+160 replay checks. Separate model reference directories.
No profiling required for initial benefit test; normal Nsight extracts MegaMoE
only and max-rank producer+kernel remains separately reported.

CPU7259660 completed1m31s: all six ordinary device builds and reduced-role
register gates PASS,128 registers/24 stack bytes/zero spills. Both helper
specializations have0stack/0spills. GPU7259728 submitted on4nodes/16GPUs,20min
limit, executes the two-model10-step plan with unchanged kernel sources.
No runtime result yet.

GPU7259728 running nvl72167-T07/T08/T09/T10. Q3 historical reference16record+16replay PASS; runtime host configuration M32/10stages/256epilogue threads matches preflight. Comparisons pending.

Q3 EP16 portion7259728 complete: historical25.7731us; helper23.9158/23.8896,
best24.2752/24.1843. ABBA23.90270 vs24.22975us (-.32705us,1.3498%); both orders
improve.16reference records+64exact comparisons+80replay PASS. Q35 portion
starting; do not extrapolate the Q3 result.


## EP16 validation complete: benefits transfer, limited coverage

GPU7259728 COMPLETED0:0 in14m30s on nvl72167-T07/T08/T09/T10. Same unchanged
helper/best1536 and native caches; no kernel/heuristic tuning. Historical pull
records reference for each model, followed by uninterrupted helper/best/best/
helper ABBA,100 ordinary samples/pass (200 per optimized arm;100 history).
32reference records,128exact comparisons and160replay checks PASS on all16ranks;
stderr empty. CPU7259660 six device/register gates PASS,1m31s. No live jobs.

Ordinary rank0 MegaMoE-only means, us:

| EP16 case | Helper | Previous best1536 | Reduction | Historical pull |
|---|---:|---:|---:|---:|
| Q3 global192/rows12 |23.90270|24.22975|1.3498%|25.77310|
| Q35 global512/rows32 |49.30880|49.56815|0.5232%|48.97440|

Both orders improve: Q3 helper-minus-best-.3594/-.2947us; Q35-.2167/-.3020us.
The EP8 gain transfers to these two EP16 shapes without retuning, but Q35 helper
is still.6828% slower than historical pull. Do not claim general superiority or
statistical independence among graph replay samples. Fixed balanced routing,
FP8xFP4,150compute CTAs,8senders,bucket3840; initialization/reset excluded.
No fine-grained captures in this validation, so no new causal phase attribution.

Separate max-rank producer+kernel ABBA: Q3helper42.336160 vsbest42.759040us;
Q35helper70.142561 vsbest70.125921us (effectively unchanged/slightly worse).
Historic full-chain outliers9238.859863/9484.893799us retained; scopes notmixed.

Q35 global912/rows57 remains unsupported: natural M64/128epilogue threads
violates current compact path's M<=32/256worker requirements. Never launched
or retiled. Supported512 case explicitly chosen, not proof of larger-batch
support. One batch per model and fixed routing are limited coverage; preserve
previous best and pull controls, no general/default promotion or commit.

Compact evidence remainder_helper/ep16_validation/{RESULT.md,comparison.json,
ordinary_7259728.json,run_audit_7259728.json,contract.json,compiled_review.json}.
All ten pass statistics/dispersion/extrema retained; raw Nsight remainsremote.
This completes the requested two-model EP16 benefit check.
