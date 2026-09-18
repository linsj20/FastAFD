# MegaMoE latency dependency audit — 2026-09-17

Scope: current maintained destination-list / warp-uniform row-tail Lamport
candidate; compare original-broadcast variant and retained original all-push
where different. Fixed EP8,32 rows/rank,H4096,I1536,E128,topk8,8 sender CTAs,
150 compute CTAs. Static source audit only; no new GPU jobs or kernel edits.
Existing performance evidence is in CODEX_PROJECT_push_dispatch.md.

User principle: overlap memory, atomic, synchronization and completion latency
with independent work in other warps or pipeline stages. Fewer warps is not an
objective. Moving work to another warp only helps if its dependencies, storage
and synchronization actually allow independent progress.

Source locations below refer to the working tree at audit time:

- M: python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
- D: same directory, mega_moe_dispatch.cuh
- R: python/minisgl/kernel/csrc/jit/gate_topk_fused.cu
- S: python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
- T: python/minisgl/kernel/csrc/deepgemm/deep_gemm/include/deep_gemm/ptx/tma.cuh

## Warp ownership

Warps0–3 do genphase in participating CTAs, then FC1 row polling/SF/gather and
cleanup. Warp4 coordinates FC1 A arrival and loads FC2 A/SF. Warp5 loads B/SF.
Warp6 issues MMA in the leader CTA; warp7 has no steady-state assigned work.
Warps8–15 do epilogue, remote return stores, then local top-k reduction.
The companion CTA's warp6 does not issue MMA. Idle role capacity is real but
has a40-register budget and cannot simply take ownership of another thread's
bulk async group. Shared-memory and TMEM lifetimes also constrain reassignment.

## Findings and candidate changes

| Area / source | Confirmed dependency chain | Overlap opportunity and constraint |
|---|---|---|
| Router, R675–760 (generic sibling R450–520) | Remote atomic returns a destination slot; descriptor store depends on the returned value. Local CountsReady and PDL follow this work. | Independent local input readiness/launch setup can be separated from remote slot reservation. Another warp cannot eliminate the atomic→descriptor dependency itself. This explains why kernel-entry-relative gains can hide later producer completion; preserve common-origin timing. |
| Dispatch CTA setup, M304–447 | Only first4 warps communicate, but the entire participating CTA joins before shared/TMEM initialization; paired cluster synchronization can hold its companion too. | Let other roles initialize or prefetch concurrently only after assigning nonoverlapping scratch and respecting cluster/TMEM allocation. Existing tensor-map descriptor prefetch already overlaps. Not a safe barrier deletion: genphase reuses the compute shared arena. |
| Counts/scheduler, D65–99, M453–493, S190–232 | All rank metadata→counts/prefix→CTA join→all role schedulers. Broadcast also acquires all source count rows before prefix construction. | Independent descriptor/address preparation may run earlier. Current packed offsets and deterministic assignment genuinely need final counts; expert-local scheduling requires a layout/scheduler change, not skipping the acquire. |
| Sender, D181–239 | Local TMA load→send to peers→full bulk wait→four-warp join→SYS fence→join→tail store. | Tail readiness remains behind sender completion. Each row also waits for the CTA's other3 rows and all its destination peers. Separate independent work from that drain path; do not add work after a blocking send and call it overlap. A receiver-detectable, chunk-valid protocol would remove the sender-completion dependency, but one unordered final TMA word cannot certify an entire row. |
| Sender pipeline, D181–239 | Each row now fully drains before advancing. At fixed32 rows,8CTAs×4warps assign exactly one row per warp. | A deeper next-row pipeline alone has no extra row to overlap in this case. Original baseline allowed multiple bulk groups before final drain; row-tail changed that for larger inputs. The trailing wait_group.read is already subsumed by the full drain in this candidate. |
| FC1 polling, M680–725 | Tile indices→source-row tail polling→acquire/proxy fence→K loop. Each row tail certifies full H plus SF. | Readiness could be finer than a whole source row; current first K stage waits for all K data of needed rows. Keep distributed warp-uniform polling. Both CTAs retain full SF row sets, so not every repeated row check is removable without changing SF ownership. |
| SF/gather, M730–762 and M1221–1233 | Empty-stage wait→global SF loads/shuffle/shared writes→five-warp rendezvous→activation gather issue. | Activation transfer is scheduled behind SF work even though the two are data-independent once payload/stage are ready. Candidate: issue A gather early, keep SF completion and expected-byte accounting correct before MMA sees a full stage. A separate existing role can prepare SF/coordinates while issuers progress; it must not become a single-warp bottleneck. Prefetch SF into registers before a stage-reuse wait where lifetimes permit. |
| FC2 A, M1180–1195 | All FC1 output bits for the tile must arrive before any FC2 A stage loads. | Per-K or bounded-window readiness could start FC2 earlier. Source explicitly records a previous on-demand scheme as negative under expert-wave scheduling: retain that evidence and do not assume finer granularity wins. |
| MMA/SF, M1380–1462 | Accumulator slot free→A/B/SF full barrier→UTCCP SF copy→MMA issue. | A/B producers and multi-stage MMA already overlap correctly. SF preparation/descriptor arithmetic may move earlier only with correct SF TMEM reuse ordering. Do not remove operand or accumulator reuse waits. |
| Epilogue inputs, M1567–1622, M1794–1812 | Accumulator-ready wait precedes routing-score loads (FC1) and destination metadata/address work (FC2). | Prefetch independent scores/destinations before the accumulator wait, or in another existing warp. Preserve route readiness and register budget. This delays TMEM consumption/release today and can backpressure later MMA. Strong small-scope candidate unrelated to network push completion. |
| FC1 epilogue stores, M1681–1767, T92 | Buffer-reuse wait uses full bulk completion; final full drain plus all-epilogue join precede FC2 arrival-mask publication. | For intermediate shared-slot reuse, read-completion is sufficient in principle; keep full completion before publishing global FC2 data readiness. For this fixed case only one valid16-row store slice per active WG means the intermediate reuse change may have little opportunity. Separate store completion handling from subsequent independent epilogue work using bounded stages. |
| FC2 return stores, M1828–1897 | TMEM read/pack→shared staging→WG join→remote stores→all-epilogue join→next tile. | Address prefetch and registers-first preparation can overlap latency; revisit join scope only after proving shared staging lifetime. Plain remote stores have no explicit receiver-consumption acknowledgement here; do not label this join an ACK wait without generated-code/trace evidence. |
| Combine entry, M767, M996, M1916–1954 | Each CTA finishes its assigned epilogue work, then joins dispatch before local reduction starts. Dispatch also waits all genphase retirements. | A separate role could reduce a ready local token while epilogue continues independent tiles, provided scratch/metadata lifetimes are separated. No global combine-ready barrier is active in this in-band/last-CTA specialization. Existing cleanup already overlaps reduction after the local join. |
| In-band combine, M2130–2237 | Each lane batches loads of pending top-k vectors, polls missing contributions, then reduces/stores this vector; next vector follows. | Already avoids sequentially waiting each top-k slot. At H4096 BF16,512 uint4 vectors/256 epilogue threads means2 vectors/thread; a missing first vector can hold the second vector behind it. Bounded multi-vector progress or distinct loader/reducer roles are candidates, retaining exact slot-order accumulation and208-register budget. |
| Cleanup, M999–1125 | Dispatch/epilogue local join→last-CTA ticket→clear routing/arrival state; other dispatch CTAs return. | Correctness/lifetime dependency; no all-CTA cleanup barrier in the active path. Keep cleanup independent of combine reduction. Move initialization outside timing only where the invocation/ownership contract permits; payload initialization already is outside. |

## Already useful overlap to preserve

- Weight TMA has a separate warp from activation gather/polling and MMA.
- Multiple operand stages and two accumulator stages decouple producers/MMA/epilogue.
- Epilogue releases consumed TMEM before finishing remaining transform/store work.
- Combine return stores overlap later compute tiles; cleanup overlaps local reduction.
- Warp-uniform polling restored gather operand hoisting: measured compute span
  ~36us→~31us in the matched destination-list experiment. This is an established
  codegen fix; the other opportunities above are static findings, not measured wins.

## Completion semantics and measurement gaps

Bulk async groups belong to the issuing thread. A different warp cannot execute
wait_group on behalf of that issuer. Full wait includes destination writes;
`.read` completion only permits safe reuse of source storage. Neither is a
software ACK from the receiver's consuming warp. Reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group

Do not infer remote visibility, receiver use, or individual warp wait duration
from a whole phase envelope. Existing direct markers are useful but incomplete:

- GenphasePayloadDrained samples warp0 and is before CTA fence/tail store in the
  current variant; it is not the last completion among all sender warps.
- MegaPayloadReady, MegaGatherSfReady and MegaGatherIssued sample warp0 of the
  first FC1 tile, not completion of all gather warps. GatherIssued follows issue.
- MegaComputeStart is immediately after operand-barrier wait, BEFORE SF UTCCP
  and the first MMA instruction. Existing “first MMA” tables use this proxy.
- MegaComputeEnd is the final accumulator-ready event observed by epilogue.
- Separate A/B arrival waits cannot be recovered from the shared full barrier.
- CombineStart includes BF16 packing/return; it is not local reduction start.

Next targeted direct markers, only with a chosen change: independent-work start,
wait begin/end for responsible warps, SF load start, actual first MMA issue,
FC1 output-ready / FC2 first A issue, combine reduction start. Preserve old marker
semantics and compare instrumentation consistently; no injection framework.

## Implementation priority

1. Hoist epilogue routing/address prefetch across the accumulator wait; isolate
   register/lifetime effects, then compare matched EP8 kernel and compute span.
2. Decouple SF preparation from activation gather issue within the existing roles
   and stage accounting; compare payload-ready→A-issue→operand-ready→MMA-issue.
3. Rework communication/setup overlap and sender/receiver completion dependency
   with explicit scratch ownership, rather than blindly adding sender warps.
4. FC2 readiness and combine multi-vector progress only when decomposition shows
   their waits exposed; retain previous negative FC2 on-demand evidence.

No performance ranking among these unmeasured changes is established. No new
tests needed for this read-only audit; kernel source is unchanged by the audit.

## Timed follow-up — September 17

GPU7236394: epilogue prefetch43.0304→42.7296us ordinary kernel (repeat pending).
Removing five-warp rendezvous + independent SF arrivals advances first actual
MMA metadata-origin12.032→11.680us median but compute span30.080→57.984us and
kernel70.7104us. Reject this form; extra per-stage fences/arrivals are a hypothesis.
The original join also synchronized payload readiness across mismatched SF/poll
and gather ownership. Before testing its removal, SF/poll quartets were mapped
to each issuing warp's four-row gather groups in both CTA halves. This preserves
full-SF replication while guaranteeing each issuer has acquired its own A rows.
Next7236801 tests early A issue with corrected ownership AND the original join
retained after SF work. Also isolates early local-input/PDL publication before
remote slot atomics and repeats epilogue prefetch. All five arms use actual-MMA
markers (replacement of old proxy within PMTrig event-count budget).

Fence correction: parent already emits the same MEMBAR.ALL.CTA + async proxy
fence as early-A variants; only its placement relative to A issue changes.
Do not attribute the regression to adding a fence. Next CPU prototypes separate
nonblocking named arrival (per-stage IDs, only coordinator waits) and SF/A roles
(2A+2SF warps,96-thread SF/coordinator join). The latter removes SF fences from
activation-issuing warps. Both are single-wave guarded experiments, not promoted.
PTX named barrier arrival/sync ordering and reuse requirements verified at:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar

User acceptance: compute pipeline time must be comparable to historical pull,
not merely recover to the current all-push parent. Metadata may retain explicit
publication. Matched historical actual-MMA capture added to7236911. Per-pair
compute span analysis now separates start skew from each pair's pipeline time.

Matched follow-up7236911/7236914 PASS144exact+144replay. Historical actual-MMA
compute span25.120us median; parent30.304; early-A+original join30.592. Per-pair
history~23.4 vs parent~29.0 confirms steady-pipeline gap, not only startup skew.
Early-A+original join43.1808us restores independent12-arrival70.7104 regression,
so post-TMA fence ordering alone cannot explain the latter. Early-input launch
46.064us regresses both kernel and common-origin startup; restored local router.
Next7237512/7237513 compares parent, split2A+2SF, nonblocking named arrival, history.
Both keep original4 full arrivals. Historical startup outlier retained in summary.

7237512/7237513 complete: split and nonblocking-arrival ordinary42.912/42.950us
vs parent43.050/history37.706; compute medians31.296/30.560 vs29.920/24.864.
No acceptance. Pair partition at first FC2 epilogue shows most gap already
present there (parent~24.2 vs history~19.7us after first MMA). This boundary
includes FC1 plus early FC2 and is not FC1-only attribution. Next SF register
prefetch ahead of stage-empty wait compiles without spills; shared stores
remain gated. GPU7238237/analysis7238238 compare split/prefetch/history.
Optional Computelab PIC-C preparation paused, no Computelab jobs submitted.

SF focus from verified source: old source_counts activation gather4 + canonical
SF TMA achieved~25.3us (older operand-ready marker), whereas the matched
direct_sf_gather experiment increased compute to~29.4us. Gather indices remain
cached outside K and MMA/layout unchanged. Do not claim inherent gather4
inefficiency without operand-delivery evidence. New SF-prefetch7238237 saves
0.8192us ordinary and0.640us median compute, but remains~31.3 vs24.8 history.
Tile-SF7238494 tests per-tile Lamport acquire→one-time SF packing→SF TMA,
retaining activation gather4 and removing scalar SF/fences/joins per K stage.
If packing becomes startup-critical, distribute each row's uint4 SF chunks
across the existing128 metadata threads; current row-owner loop uses only
16 lanes for the fixed tile and serially stores32 words per lane. That is a
concrete independent-work opportunity, not yet a measured win or submitted run.

Tile-SF7238494/7238495 COMPLETE,72exact+72replay PASS. Ordinary45.392us regresses
vs prefetch42.755/history37.818. However per-pair first-MMA→lastaccum medians
23.648/23.232/24.224us now approach history23.104/23.456/23.104, and per-pair
first-FC2-epilogue timing also matches~19–20us. Overall compute28.096/27.936/
28.192us includes~5us release spread (history~1–1.5us). Next parallelSFpacking
GPU7238788/analysis7238789 directly targets startup. CPU7238630 all gates PASS.
Original five-warp stage join is not the first suspect given recovered per-pair
throughput without it. Remaining static opportunity: each metadata CTA serializes
two experts in this fixed case; inspect how much release skew survives chunk
parallelism before changing expert ownership. No extra GPU submission for that idea.

Refined attribution from exact pair markers: tile-SF late pairs are metadataCTAs0..7,
not a global all-expert barrier. Their entireCTA waits for genphase return before
GEMM/cluster initialization. Sample1 CTA0 IndexReady16.416→dispatchEnd16.448
→metadataAcquire17.216→gather18.528→MMA19.392us; CTA18 MMA14.784us. Their scheduled
FC1/FC2 work is also heavier (132Ksteps) than tailCTAs132..149(100/112). Prepared
CPU7238856 remaps same17dispatchworkers to tail pairs, no extraCTAs/SMs/senders.
GPU decision pending parallel7238788. Summary now exposes these late-pair markers.

7238788 parallelSF result confirms bottleneck: ordinary42.848 vs serial44.6784,
per-pair compute~23.2us matches history; first-to-lastMMAspread~4.5us and lastfinish
always metadataCTAs0..7. User requests actual dependency removal, not placement.
Hold tail placement GPU. New concurrent_dispatch runs genphase on first epilogueWG
while gather/TMA/MMA run; metadata counts use32-arrival release/acquire mbarrier
instead of wholeCTA join. Initial cluster joins retained; disjoint scratch in unused
compact dispatch SMEM anddepth1 sender preserve current one-token-per-warp case.
CPU7238978 template type error corrected;7239077 validation in progress.
Memory-ordering reference: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html
(mbarrier release arrivals/acquire waits order each participating lane's prior writes).

Concurrent normal+IKET device/register gates pass after explicituint32_t fix;
CPU7239077 selectors finishing, GPU7239114/analysis7239115 afterok-gated. Candidate
runs first,240s per capture subprocess bounds new synchronization risk. Existing
first-MMA + late-pair markers will show whether MMA starts before worker retirement.

CPU7239077 COMPLETE2m06s all gates PASS; concurrent GPU7239114 ordinary exact/replay
PASS, kernel41.392us so far; controls/decomposition pending. Count-barrier source
ordering audited against generated SASS: STS5200→ARRIVE5250→TRYWAIT52b0→LDS53e0.
Add compiler memory clobbers around wrapper before promotion, current sources frozen.
Possible next improvement (not submitted): split second cluster arrive/wait around
startup WG, so dispatch can overlap other warps' TMEM/barrier initialization too.

CPU7239386 prepares explicit compiler ordering plus a separate split cluster
arrive/startup/wait variant (main_ordered/main_init_overlap); no followupGPU yet.
7239258 TU-path typo FAILED43s,7239310 cancelled28s; both fixed in combinedplan.
PTX cluster.arrive is nonblocking, aligned requires warp-uniform participation;
worker scratch does not require GEMM initialization, and worker later acquires wait.
Reference: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
Summary now reports actualMMA-minus-localGenphaseEnd per startup pair. Current
7239114 prefetch mean45.2832us contains40.928–49.344us range; don't overclaim a gain.

7239114/7239115 COMPLETE,96exact+96replay PASS. Concurrent ordinary41.392 vs
parallel42.7712; compute24.864 median matches history25.216. Four metadata pairs
issueMMA.768–1.600us before own worker retirement in all samples; release spread
.6–1us vs4.2–4.6us. However earliestMMA15.936 vs13.600parallel/9.760history, so
startup still fails acceptance. Some sender retirement now~2us late despite modest
bulk-drain shift. Next7239646/7239647 compares ordered concurrent/init overlap/best
prefetch/history and adds TailPublishStart after fence/join, before tail stores.
Replaces redundant SenderDone to retain30types, GenphaseEnd retains retirement.
CPU7239590 gates instrumented variants+bothselectors. No sender fence removed.

7239646 live ordinary init40.880us vs ordered42.0512 (medians40.880/41.568;
ordered range40.736–44.928, preserve outlier), both exact/replay pass. Await traces.
Sender interval summary now pairs exact CTA+warp: issue→drain→tail-publication-start
→GenphaseEnd; validates completeness/order and retains raw counts, avoiding global
extrema subtraction. New marker absent on older captures is explicitly no measurement.

Static open idea, no implementation/submission: SF bytes already exist after gate
quantization before the selector. In gate_topk_select_cta_kernel only warp0 does
selection/remote route-slot atomics;7warps mostly wait for publication. Could push
SF into existing plain-SF inbox with these otherwise idle warps while warp0 handles
route atomics, covering SF with existing route_ready release. Then local SF packing
need not wait for activation row-tail; gather warps would acquire A tails separately.
This retains all-push data transfer but changes router work/PDL timing, so compare
common metadata/selector origin, not just MegaMoE kernel time. Needs explicit peer
mask handoff, unique per-destination SF ownership, existing epoch mapping, and no
extra timed allocation. Inspect new tail-publication data before pursuing. No new
assumption that an SF byte pattern can safely serve as a Lamport sentinel.

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

Receiver-validation research constraint: deployed CUDA13.0/PTX9.0 calls bulk TMA
copies weak operations; do not infer generic polling atomicity from alignment or
copy size. Strong system-scope generic vector stores/loads are modelled per scalar
element, providing a clearer per-u32 sentinel path while preserving compute TMA.
This would change sender transport, not restore the failed per-K generic gather.
Need release/acquire plus generic→async visibility proof before implementation;
no receiver-validation prototype or GPU run yet. Primary reference:
https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html
(sections8.2.4,8.7,8.10.3 and cp.async.bulk).

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


Prepared follow-up (not submitted): receiver_validate/wide_metadata doubles
metadata participants128→256 using the second epilogue warpgroup on physical
CTAs0..7 only. Sender/helper participation stays128; senderCTAs stay8; all150
computeCTAs retain existing gather/TMA/MMA roles. Prefix scan has8warp sums,
metadata bar3 participant count256, each16-row tile assigns2rows/warp instead4.
Source/materialized storage, two-phase counter and instruction-level batching
unchanged. Disjoint scratch uses160bytes for16expert histogram+offsets+8sums,
within the existing dispatch arena. Final256-thread join completes before either
warpgroup returns to its normal epilogue/bar3-or4 lifecycle. No extra barrier,
flag or allocation. Both epilogue groups are delayed until metadata completes;
trace must show that faster readiness outweighs lost early epilogue availability.
Local diff reviewed; sources and bounded include overlay synced to HSG. No CPU
or GPU job yet; batched7242329 evidence will decide whether to proceed.

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


Deferred bounded-window idea (unimplemented, no job): current validation still
requires full4KB rows before first128B K-stage MMA. Preserve TMA compute but split
receiver materialization into first512B (fourK-stages) then remaining3584B. Existing
counter phases could be SF/index=valid_m, firstwindow=2*valid_m, fullA=3*valid_m.
Gather accepts>=2*valid_m forfirstfourstages andacquires>=3*valid_m once before
stage4 with proxyfence; avoid per-K generic gathers/joins (old109us failure).
All consumed words still validated/copied, no weakTMA inbox reread. Firstwindow
can stripe256threads over16rows×16lanes/row,2uint4 loads/thread, so allfirstwindow
rows issue in onebatch; remainder can retain proven full-row batched loading.
Tradeoff: extra localcounter/proxy boundary and possible stage4 stall; reject if
compute~25us lost. Keep reset outside timing. No additional allocations required.

Complementary sender idea only if that window is tested: precompute peer bases
once, hold distinct source vectors, then chunk-major SYSstores issue first512B to
allpeers beforelaterchunks. Earlier unbatched chunk-major sender repeated address
setup/reusedregisters and was slow; do not repeat it. Precomputed8peer basepairs
must fit startup128registerbudget and SASS must prove independent issue. Compare
firstMMA/fullkernel/broadstarts, not source opcode count. Preserve whole-row path.
Instrumentation could replace old last IndexReady marker (End already covers
last retirement) with firsttile FullAReady, alongside existing first gather-ready
markers, to test whether MMA actually precedes fullrow completion; needs explicit
analyzer contract update, not overloaded marker meaning or added injection.

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
