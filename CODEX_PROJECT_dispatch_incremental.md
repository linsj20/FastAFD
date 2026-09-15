## Handoff prepared — work paused at latest user request

Read [final MegaMoE handoff](CODEX_PROJECT_final_megamoe.md) for current source,
results, risks and exact remaining commands. Explicit per-buffer combine CPU7153123
PASS native+host in3:28. No running task jobs/collectors; new exact264 and five-config
Q3/Q35 unprofiled comparison plans are prepared but UNSUBMITTED. No index/commit
changes. Resume remaining validation/selection only when instructed; earlier ACTIVE
entries below are historical checkpoints. Do not claim final version or best-config
proof complete. Current state: pull_metadata_restore_20260914/selected_explicit/STATE.json.

# Latest selection follow-up

User now requires proof from tests that early metadata push + payload pull + push
combine is best among retained configs; if another path wins STOP/report. Initial
cleanup CPU/exact264 passed; both timing jobs7152479/7152485 complete/audited.
Q35 candidate_repeat broadly slower forrows22/28, pooled+5.127/+8.147percent;
do not dismiss as outliers or claim performance equivalence. Follow-up explicit
per-buffer combine/AFD=True code, CPU7153123; exact plan then five-config
unprofiled max-rank comparison. Root CODEX_PROJECT.md newest entry and
scratch/megamoe/pull_metadata_restore_20260914/selected_explicit/STATE.json are
current authority. Older STOP applies only to prior optimization search.

# Incremental dispatch optimization — September14

**Latest user request supersedes STOP for a bounded restoration/cleanup/test:**
retain best dense metadata + TMA payload pull; remove rejected active optimization
paths; run fresh decomposition. No optimization search or old sweep resumption.
Seven active files cleaned; CPU7152139 all5steps PASS2:56; native7152230 PASS1:42.
Exact264 GPU7152282 PASS/5records8:13. Fresh selected-vs-clean ABBA Q3EP8
7152479 and Q35EP16 7152485 submitted22:25UTC; collectors running. Seed3689699192022641276 Q3EP8 rows28/21/4, Q35EP16 rows38/28/22
unchanged after scope expanded to cleanup. Evidence pull_metadata_restore_20260914.
Production backup before_cleanup, selected byte identity verified before edits;
remote ffn_pull_metadata_clean_20260914 contains cleaned seven-file overlay.
Retain accepted combine, legacy routing correctness, and existing buffer offsets.
Detailed status now tracked in that artifact's restoration.json and root memory.

**STOPPED PER USER after completing and auditing ongoing experiments.**
7150076Q3/7150075Q35 COMPLETED0:0 in24:22/25:38; collectors90292/55335 EXIT0.
All48captures240phase1440ordinary retained/audited. Earlystore exactPASS352;
timing0/6dispatch targets(history−11.76..−3.85%), REJECTED_AS_STANDALONE.
Fullpath gainsvsdense Q3−0.62..+0.09%, Q35−0.60..−0.13%; versuscompact
Q3−2.59..−0.86%,Q35+4.62..+7.89%. Parentvschild effect only−0.33..+0.71%.
No qualifying replacement or promotion; optimization goal NOT achieved.

Currentcompact compute span24.70–25.65usQ3/37.18–37.60usQ35 dominates measured
kernel envelope(~69–76%); exposeddispatch5.47–5.92us. Q35compact saves1.15–1.33us
exposeddispatch vsdense, whileproducer finishes0.51–2.68us later andcompute grows
1.57–3.30us. These are overlapping spans, not additive latency accounting or a
hardware-stall attribution. Rank0NSys/IKET/SASS only; noPIC-C/NCUGPUprofile.
CPUinventory7150136/CLI7150281 completeandcollected; NCUavailable, noGPUclaim.

Final evidence: C/packed_source_early_store/e2e_rejection.json and
[FINAL_STATUS.md](scratch/megamoe/dispatch_incremental_20260914/FINAL_STATUS.md).
STATE.json STOPPED_PER_USER_AFTER_COMPLETED_EXPERIMENTS, no jobs/collectors or
nextactions. No further experiment, profiling, validation, alignment or optimization.
All older ACTIVE/go/next-action text below is historical and superseded by this
stop instruction. Resume only on a new user request.

User follow-up: have we tried same dense layout with push? Verified YES:
direct_push vector and direct_bulk_push TMA both send insideMegaMoE to identical
final contiguous rows, native extraordinalstore only. Exact264/352; fullpath
12.15–19.08%/10.25–17.82% slowerdense. Direction reversal also changes remote
address lookup and completion/publication ownership, so not an otherwise identical
instruction swap. Detailed7133350 trace localizes delay afteraddressready through
payloadissue/publication/tileavailability; expert0 no-prefix source still delayed.
Do not claim prefix lookup or pure atomic latency dominates. Full clarification
in FINAL_STATUS.md; no new work launched, stop remains in effect.

Further user proposal: stable source-owned receive slots, payload+expert tags,
destination CTAs wait for required payloads and compute. Clarified feasible;
sender final-row resolution only necessary for direct writes to dynamic packed
expert pool. Stable inbox removes prefix lookup, retains receiver expert grouping
and ordered generation/readiness. Compact source slots+gather partially match,
but tested rank-readiness/dispatch-tile-arrival protocol is not exact per-payload
destination-driven proposal. Options gather/local packing/expert-tile placement
have distinct costs; no impossibility claim or new run. Details FINAL_STATUS.md.

Correction after user's negative-zero reminder: older Lamport dispatch DID test
stable source-indexed inbox and per-word in-band readiness; prior answer missed
this history and overstated need for separate ready flag. Current retained optional
DG_MEGAMOE_LAMPORT_DISPATCH has kUnarrivedWord0x80000000, volatile uint4 all-word
checks, reservedzero canonicalization, separate SF/weightpolls, dispatchshared→
contiguousexpertTMAcopy, localtilearrival, readerjoin/clear/two-generationreuse.
One changed word is not whole-token completion; all needed words checked, or
one explicit marker must follow ordered fullpayload completion. Matched historical
7107319Q3EP16/7107761Q35EP32 exactPASS; pull/inbox kernels29.7259/30.0416us and
35.7568/38.2037us, samecombine. Primary36.2656/37.3304us and46.3845/48.4568us.
Sentinel mechanism tested; compute-direct inbox consumption without receiver
packing not established by those tests. Memory sources qwen35_push_comm Protocol
and early_inbox_generations_20260912 results. No runs or optimization resumed.

User asks direct contiguous writes plus signal stored elsewhere: YES, existing
direct_push/direct_bulk_push already use final expert activation/SF/weight pools
and separate get_l1_arrival_count_ptr tile counters. Bulk waits remote completion,
proxy fence/warp join, system atomic increment; consumer acquires expectedtilecount
then contiguousTMA. No inboxrepack. Distinguish from Lamportinbox packing. A
single-writer per-row flag array differs from tested shared tilecounter; do not
attribute counter cost or claim all separate-signal implementations tested.
Signal bitpattern/location does not remove payload-before-signal ordering or
finalrow placement; fixedexpertbase removesprefix but retains slotassignment.

Further clarification: fixed per-source (optionally expert×source) regions allow
sender-local row placement, eliminating remote prefix/slot discovery. Expert×source
base+localordinal permits contiguous segments and source-owned readiness flags;
consumer counts remain necessary for scheduling/partial tiles, not sender addresses.
Choices: source-segment tiles may be underfilled; aggregate segments into one shared
MMA tile via multiple loads; reserve sufficient fixed capacity. Costs need measurement,
not assumed failure. Historical rank_segments_20260911 used dynamic rank-prefix
placement and packed_layout still published destination-computed bases, so it is
not an exact test of static expert×source partitions. No implementation/run authorized.

User asks metadata-push/payload-pull vs all-pull: verified compact descriptor
campaign dispatch_dense_descriptors_20260913. Matched sixcase dispatch+20.99–30.65%;
full464 historical meanpercase dispatch+14.108%Q3/+23.108%Q35, rank0fullchain only
+0.478%/+0.583% with31/120 regressions, MegaMoEstart+1.735/+1.710us. Prior baseline
already pushed route/count metadata; this is dense descriptor construction vs old
count aggregation/inverse lookup with payloadpull unchanged, not puremetadata
direction A/B. No substantial fullpath/model gain or current strictgoalpass claim.

**User correction16:23UTC: the final goal is faster end-to-end kernel latency.
Reject candidates that improve dispatch by sacrificing the full path.** Prior
phase-only acceptance and promotion are superseded. Require faster matched ordinary
producer-start→MegaMoE-end on every selected case, including repeated controls;
retain >5% dispatch and correctness requirements. Only a qualifying E2E source
proceeds to fresh model alignment and full decomposition. Frozen history stays
frozen; establish E2E history independently from dispatch-phase minima. All changes
must be general; direct push remains a priority, without an impossibility claim.
Goal ACTIVE, NOT achieved.

**Latest user objective18:17UTC: preserve the current compact push's dispatch efficiency while removing its two overheads (producer preparation/pushing before MegaMoE, scattered subsequent-tile activation gathers during compute).** This is the current focus. Direct contiguous bulk push already running is an architectural diagnostic; do not expand its EP2/timing work unless it can preserve current push gains. No substitute goal of merely returning to ordinary pull. Continue consumer study from async timing and producer changes that reduce actual work. Prior early-PDL-only attempts already shifted waiting into dispatch without E2E gain (7135122,7136661); do not retry launch relocation alone. Real Q35 two-stream serving also deliberately defers PDL for shared-expert priority, so early-launch-only improvements would not apply there.

**Latest clarification18:20UTC:** payload transfer method need not stay. Treat fast push as evidence of an advantage to isolate and preserve, not a fixed implementation. Alternatives should retain fast transfer while removing preparation and scattered-consumer costs. Do not assume which factor is causal from the aggregate timing alone.

Evidence root: `scratch/megamoe/dispatch_incremental_20260914` (C). Remote R:
`/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel/FastAFD`.
Sources R/source/ffn_dispatch_dense_VARIANT_20260914; artifacts R/artifacts/
dispatch_incremental_20260914/VARIANT. Baseline.json and history/qwen3,qwen35 freeze
minima across464cases/threefullgenerations+denseABBA;384older rejected trial medians
checked,none better. Source overlays isolated; main/staged history preserved.

User reaffirmed: every new six-case performance experiment must draw fresh random3Q3EP8+3Q35EP16 from full eligible manifests, without timing filters or result-based rerolls. Keep its draw fixed across controls/repeats for paired comparisons. Current stride/shared-route draws are independent seeds2858878927204158955/7530739210575995699; all12arm plans match their recorded selections. Future new experiments must use a fresh draw, not reuse the reported two winning Q3 examples.

Max2outstanding GPU jobs. CPU jobs separate; all host/native preparation offGPU.
Plain ssh oci-hsg; remote rg unavailable.300s monitoring,2–3min nearcompletion;
no resubmission after observation timeout. No subagents, /tmp, py_check/hash
checks, silentfallback, staging/commits. Separate unstaged GDN work untouched.
Missing AGENTS.md preference was asked earlier; supplied globalinstructions apply.


## User scope correction — September14,18:10UTC

User explicitly requires dispatch/metadata optimization without degrading other parts: remove added pre-MegaMoE producer preparation/push and avoid compact activation gathers reducing subsequent-tile compute efficiency. Fullpath improvement alone does not excuse those regressions. Prefer transfer changes inside MegaMoE with original contiguous expert-row consumption; current direct_bulk_push directly tests that architecture. Existing compact async timing remains diagnostic, not promotion of its already-regressing producer.

`audit_fullpath` now additionally guards nonincreasing pooled ordinary producer completion/MegaMoE start and phase compute/combine medians versus matched controls. All records retained; component worst-repeat deltas reported separately. Marker spans include overlap/stalls, not isolated efficiency measurements. Critical in-memory validation also proves a faster fullpath cannot bypass component regressions and rejects nonfinite phases (component_guard_validation.json). Retained shared-route Q35 check rejects all3: producer+0.555..1.187us, compute+2.608..4.256us, combine+1.920..2.976us. Evidence component_guard_check.json. Future alignment workflow recomputes these gates. Collectors started before this edit imported the old function; recompute their fullpath_gate/completion after collection before any selection.

Authoritative continuation check: GPU7147418/7147425 both RUNNING, CPU7147393 COMPLETED0:0/4:14. Old collector/session handles missing and process inspection confirmed no collectors; restarted collection only (no GPU resubmission) as sessions51217/64746, observing same live jobs at11:06/11:07 elapsed. Direct-bulk CPU all8steps PASS; three ordinary geometries explicit registers fit48/40 budgets (dispatch maxR42/R42/R45). Direct-send loop disassembly has actual UBLKCP.G.S, full completion before original system publication, no local stores in changed loop. Assertion/error paths retain local stores/indirect calls; no whole-kernel spill-free claim. CPU completion/register audit saved; exact job name corrected, GPU plan still unsubmitted pending slot. Prior turn classified progress; this turn adds verified CPU evidence and enforceable component guards.


Direct bulk EP2 supplement prepared (validation_ep2), control/candidate Q3rows4/96,30 unique exact comparisons and actual M16/M32 requirements. Still unsubmitted. Native diff versus integrated dense is one local uint32 ordinal store per route; no activation preparation/push in producer. Must measure that store's cost, not claim unchanged producer.

18:15UTC: Q3 async7147418 COMPLETED0:0/16:53; Slurm COMPLETING release observed before next submit. Q35 async7147425 stillRUNNING17:36. Direct-bulk exact7147839 submitted18:14:57, eight records/352comparisons expected; EP2supplement still unsubmitted. No timing/selection for directbulk before exact passes.

18:22UTC async-gather fullyaudited/rejected:36captures180phase1080ordinary, all6dispatchgatesPASS(history9.04..19.68%) but fullpath59.30..91.39%slower thancompact,65.46..98.92%slower thandense. Producer within−.031..+.053us vscompact, compute+34.35..56.48us. TMA→lane cp.async replacement unsuitable; per-stage waiting/prefetch loss is hypothesis, not isolated attribution. BothcollectorEXIT1 onlyatfinalphasecomparison because suppliedbaseline.json insteadhistory/MODEL; allcaptures already fetched/audited. Localanalysisrecovery completed withcurrentcomponentguards, originalfailurelogs retained. NoGPUreruns.

Direct-bulk timing plans now include compact/compact_repeat reference arms in addition to dense/vector/bulk, eight symmetrical arms/model, same original fresh seed14608292657801473/cases. Added before any timing submission; source caches preserve actualcompact parent. Explicitly compares retained currentpush advantage, not merely beating slower vectorpush. Stillunsubmitted pendingexact.

18:28UTC direct-bulk exact7147839 COMPLETED0:0/11:10, strict8record352comparison PASS (all6actualM branches via large record). Collector4786 EXIT0. Immediate one-shot terminal collection avoided holding a ready next run for monitoring cadence; background collector subsequently finished same audit. No source changes after exact. EP2supplement remains pre-promotion requirement, unsubmitted to avoid expansion before targettiming evidence.

Eight-arm fresh original draw launched Q37148045 at18:27:54 andQ357148053 at18:28:11: densepull/vectorpush/currentcompact/bulkpush and reverse repeats. Q3rows6/10/28,Q35rows51/22/12, seed14608292657801473 unchanged from original unrun preparation. This explicitly tests preserving compact transfer gains while removing producer/consumer costs. Correctphasehistory paths are history/qwen3,qwen35. Collectors12039Q3 and38792 read new component guard code; monitor300s. At most2pending/runningGPUjobs; bothinitialPENDING/Priority. Finalalignment, actualserving/MAXrank throughput, and anypromotedsource integration remain unproven. No commits/staging/mainkernel changes.

## Producer handoff candidate — September14,18:34UTC

Previous turn PROGRESS: async consumer replacement conclusively rejected after full audit; directbulk exactPASS352 and fresh eight-arm timing submitted. This turn revalidated both live timingjobs7148045/7148053 RUNNING00:21 and bothcollectorhandles. No restarts.

New isolated **fixed_pool_shared_targets** on fixed_pool_preload: native CTA selected-route lanes retain remote ordinal long enough to compute activation/SF destination pointers and hand those through existing CTAshared barrier. Removes CTA globalordinalstore and copywarps' ID/ordinalglobalreloads and repeated address computation. Staticshared16*topk bytes; no newbarrier/globalallocation/transfer pass/consumer change. Genericrouter retains originalordinalpath. SFissue remains afterbulkcompletion forcausal isolation. Existing fixedpool memorycost and atomicallocation dependency remain; added pointerwork might negate savings.

Remote fullsourcecopy finished before applying singlefileoverlay; CPU7148132 submitted18:33:25, actualnativebuild+compatibleunchangedhostseed, noGPU. Six-record180comparison exactplan preparedUNSUBMITTED: Q3rows4/48,Q35rows4/160 pluswideH32768rows1/6, actualM16/M32 configsrequired. No redundantallM consumer testing because consumerunchanged. GPU awaits CPU success and timing-slot release; fresh performancedraw unprepared.

Report to user: strongest enabler is decoupling payload movement from metadata-dependent finalexpertrow placement. Source-indexed addresses allow bulkpayload andmetadataoverlap and avoid receiverglobalcompaction. Shorterdispatchwindow partly movesworkbeforeMegaMoE; not proof networktransfer intrinsicallyfaster. Consumer then paysgathercost. Preserve overlap and efficientbulktransfer inalternatives, but causal contribution ofeach is not isolated by aggregate phasegain. Fanoutdeduplication cannot explain balanced EP8top8/EP16top10 cases, where routes already targetdistinctranks.

## Continuation checkpoint — September14,18:45UTC

CPU sharedtargets7148132 COMPLETED0:0/5:39, actualnativebuild+compatiblehostseed2stepsPASS, logs/seed/completioncollected. No deviceheader or receiver changes; no new GPU correctness claim. CPUmonitor59965 observes recordedterminaljob; GPUcollectors12039/38792 remainlive.

Fresh sharedtargets eight-arm timing prepared with reusable preparer: seed1402188729198213786, Q3EP8rows3/4/8,Q35EP16rows13/5/30. Symmetric densepull/currentcompact/fixedbulk/sharedtargets/repeats; actualvalidated controlcaches preserved. Frozenfullpaththresholds saved beforetiming. Six-record180comparison exact awaits availableGPUslot; bothdirectbulk timingjobs stillRUNNING10:25 onlatestobservation. All newsharedtargets GPUplans UNSUBMITTED. No reroll or timingfilter.

New design-only tiled_expert_push_design.md records an alternative: dynamic contiguousexpert tiles allow a single tile-directory lookup and original TMA loads, with payloadsend insideMegaMoE and no finalexpert-prefixdependency. Explicitly unresolved: extraallocation andpublicationcost, jointmetadata/payloadreadiness, generationreuse/padding/capacity. Any sender must initialize a tile; ordinal-zero-owner scheme can deadlock on queuedowners. No implementation/build/GPUjob. Awaitdirectbulk evidence before choosing this larger change.

## Compact-transfer attribution correction and continuation — September14,19:04UTC

User correction: metadata/payload disaggregation was already present before the
last compact-push change. Earlier key-enabler report over-attributed the latest
gain to that architecture. Preserve the actual compact-frame benefit: activation,
source SF and exact weights sent together, auxiliary reads become local, one frame
per destination. Same-pitch/storage Q3 comparison7138284 isolates frame auxiliary
push: vectorframe dispatch11.707/15.330/16.497% faster than activation-only storage
control. Bulkframe did not consistently beat vectorframe. Cooperative single-CTA
staging later removed duplicated source preparation; its comparison also changed
overlap, so separate contributions remain unresolved. Compact pool removes memory
expansion; capacity reduction itself has not been isolated as latency cause.
Balanced EP8top8/EP16top10 does not benefit from destination deduplication.

Previous report-only turn was informational/no new experiment evidence; this
continuation revalidated live processes and completed concrete work. Directbulk
7148045/7148053 both COMPLETED0:0/23:00,25:02; collectors12039/38792 EXIT0. Full
48captures240phase1440ordinary audited. Bulk vs vector fullpath gains−0.083..2.338%;
vs currentcompact6.776..18.349% slower and vsdense10.252..17.824% slower. Allsix
dispatch gates fail. Saved direct_bulk_push/e2e_rejection.json, selection REJECTED;
EP2supplement not worth expanding. No promotion/source integration.

Sharedtargets exact7148396 submitted18:54:23, RUNNING6:52 latest; monitor74987/180s.
CPU7148132 alreadyPASS. Fresh eight-arm timing plans remain unsubmitted until exact.

New fixed_pool_packed_frame implemented independently on fixed_pool_preload:
activation+sourceSF share final expert row; removes scalar SF sends after bulk
activation wait, while original contiguous TMA descriptor uses physical frame
rowpitch. No receiver activation recopy, no extra kernel. Native/layout/device/host
four-file overlay; actual host rebuild required (no donor). SF source pointer uses
same generation/LCM row as activation. Wide rows may add a tail chunk; remaining
producer/ordinal costs and fixedpool footprint are explicit limitations. CPU staging
partition checkPASS15chunks/11hidden sizes; target Q3/Q35capacities768/3840 add0bytes
versus fixedpool_preload, but not versus dense. Remote basecopy completed before
overlay sync. CPU7148579 submitted18:58:40, RUNNING;5steps EP2/EP8/native/actualhost/
SASS, monitor28799/180s. Eightrecord352 exact inclwide/all6M prepared+synced,
UNSUBMITTED untilCPU. Generic/legacy/EP2 stillrequired beforepromotion.
Fresh timing seed1406568242933500168: Q3rows9/6/12,Q35rows22/2/6; all8arms with
dense/compact/fixedpool controls, frozenfullpaththresholds. No timing submitted.

## Packed source pull implementation — September14,19:13UTC

User withdrew FFN-side quant relocation; no quantization math/location/AFDtransport
change was made. This turn revalidated terminalsharedtargets7148396 and CPU7148579:
exactPASS180/6records/8:37; packedframe CPU actualnative+host5stepsPASS3:52. Submitted
packedframe exact7148680 and sharedtargets Q3timing7148682 at19:05:38/40, bothRUNNING;
monitors88331/41795. SharedQ35 timing awaits freeslot; no unvalidated timing.

New independent packed_source_pull five-file implementation onoriginaldense:
source activation+SF views interleave per row, existing fused quant producers
honor byte strides and write directly to those destinations (samequant/math/work).
Dispatch uses one remote bulk frame load into shared, then local SF transpose and
activation store into original contiguous final pool. Metadata/weightremote
transfer/consumer/combine unchanged. No receiver global inbox/recopy, producer
packing kernel, or producer payload push. Preserves compacttransfer principle
without sourceindexed consumergathers. SF-only packing deliberately isolates
remoteSF removal; weight still remote so not allframe advantages represented.
Shared dispatch staging grows by SF bytes/1KiBrounding; hostheuristics adjusted,
stage-count/producer/performance effects must be measured. Source allocation
unchanged for legalDataalignment shapes; CPU boundarycheck24geometriesPASS.
Nativegate+MiniMax fusedquant outputs both honor strides; nonquantized paths and
Lamportsource layout unchanged. Initial localedit nonuniqueanchor failed before
writing; corrected kernel-specific anchor. Sourcebasecopy completed beforeoverlay
sync. CPU7148793 submitted19:12:48, EP2/EP8/nativegate+MiniMax/actualhost/SASS5steps.
Exact/performance plans pending; noGPUclaim. Previous acknowledgment-onlyturn
was no newexperimentprogress; current turn adds implementation/build/validation.

19:17UTC packedframe7148680 COMPLETED0:0/9:39, strict8record352comparison PASS
includingall6M. Saved exact_audit.json. Freedslot immediatelyusedsharedtargets Q35
7148922, completing originaldraw withQ37148682 stillrunning. Packedframe fresh
timingplans remainUNsubmitted; no performanceclaim.
Packedsource CPU7148793 FAILED0:50 atfirstEP2compile: source_frame_layout const
was notconstexpr for sharedbuffer bytecount. Fixedconstexpr; failures/logsretained,
noGPUaffected. RecoveryCPU7148900 submitted19:15:17,monitor94739/180s, same5steps
actualnativegate+MiniMax/host. Twelve-record496comparison exact prepared+synced:
Q3/Q35/wide/all6M plusgeneric1024top2 andlegacy unprepared, sources originaldense
vs packedsource. Freshseed4482461993793792015 Q3rows18/9/10,Q35rows1/13/34.
Sixarm compact/dense/packed/reverse; redundantdense reference pair excluded before
anysubmission, no reroll. Frozenfullpaththresholds saved. Collector must use
--dispatch-control control --fullpath-control control --fullpath-control compact
(no pull arm in actualorder). Standalone MiniMax output stride runtime supplement
remains required beforepromotion; current exact usesQwenroute producer.

19:21UTC packedsource recovery7148900 passed both EP2/EP8 cubins and nativegate
compilation, then FAILED2:30 at MiniMax nonquantized launch missing newstrideargs.
Added zero unusedstrides there; candidatequantized mathunchanged. Logs/failure
retained. CompletionCPU7148965 submitted19:20:23, RUNNING37s, monitor34441; reuse
unchanged cubins and only finish native/actualhost/SASS (3steps). NoGPUexecution.
TargetH4096 shared+1024B fits existingstagecounts acrossall6M; exact expectations
now enforce actual num_stages andsmem_size, not just M. WideH32768 shared+4096B
can reduce stages; wide compute efficiency remains a limitation, not proven
unchanged. Avoid broad consumer-efficiency claims from just targetwidth checks.
SharedQ35collector76209 observes7148922RUNNING; Q3collector41795stillactive.

19:24UTC packedsource completionCPU7148965 COMPLETED0:0/3:01,3stepsPASS:
nativegate+MiniMax, actualhostextension, SASS. ReusesunchangedEP2/EP8cubins
from7148900 passedsteps0/1. ActualQ3EP8 explicitregisters dispatchR41<48 and
consumerR23<40; unresolvedassertioncalls retained, no wholekernelproof. Logs,
completion/register_audit.json collected. Exact496 nowCPUready butUNsubmitted;
GPUslots occupiedsharedtargets7148682/7148922, collectors41795/76209.
NextfreeGPU: packedsource exact496; nextotherfree slot packedexpertframe Q3timing
thenQ35 on frozenoriginaldraw. Currentgoalnotachieved; no alignment/promotion.

## Shared controls for packed alternatives — September14,19:28UTC

Previous turn PROGRESS: packedsource implementation/actualbuild and expertframe
exactPASS352. Revalidated sharedtiming7148682RUNNING22:12/7148922RUNNING10:45;
collectors41795/76209 live. Packedsource exact496 stillUNsubmitted untilfreeslot.
Combined both UNRUN packedtransfer timings to save4arms/model (24captures total)
and directly compare methods. Preserve originalpackedsource seed4482461993793792015
and sixcases without reroll/filter. Tenarms: compact/fixed/packed/control/candidate
and reverse. fixed=fixed_pool_preload, packed=fixed_pool_packed_frame,
control=originaldense, candidate=packed_source_pull. Actualsource/host/native/JIT
caches preserved from each validatedparent; cases/policies identical. Bothjobs
time45min. Standalone fixed_pool_packed_frame/selection.json SUPERSEDED_UNRUN,
originaldraw retained but MUSTNOTsubmit its standalone plans. Bothcombined timings
awaitpackedsource exactPASS. Collector --dispatch-control control, fullpathcontrols
control/compact/fixed/packed; separately evaluate packed reference as candidate
from samecaptured data, no rerun/refetch. No phase-only or methodpromotion yet.

19:30UTC live scheduler7148682 COMPLETED0:0/22:32,7148922 RUNNING12:58.
Collector41795 stilllive (nextscheduled observation); no restart or refetch.
Immediate freedGPUslot used packedsource exact7149145, 12records496comparisons
plus targetpipeline config checks. Combined10arm timing unsubmitted.

19:32UTC sharedtargets Q37148682 collector41795 EXIT0 COMPLETE_AUDITED:24captures
120phase720ordinary allretained. Rows3/4/8 dispatchhistory+15.33/18.69/19.48%,
all3dispatchgatesPASS. Versusfixedpool control producercompletion−0.916/−0.879/
−0.897us, MegaMoEstart−1.052/−0.991/−1.042us, fullpath+1.722/+1.729/+1.712%.
Versusdense producer+1.358/+1.378/+1.455us despitefullpath+1.261/+1.142/+2.142%;
thereforecomponentgoalFAIL. Versuscompact fullpath−0.940/−0.922/+0.103%, compute
span improves0.56..0.85us butproducerstill+1.21..1.22us. This establishes a useful
partialproducer saving, not qualifying replacement. Q35stillrunning; do notclaim
allsix outcome. Keep sharedtarget idea forcomposition onlyif independentpacked
expertframe test removes complementary remainingcost; sourcepull avoidsproducer
payloadwork entirely. Evidence sharedtargets/qwen3/decision_summary.json.
Packedsource exact7149145 initialPENDING/Priority verified; monitor7412/180s.

19:34UTC directlivequeue packedsource7149145 RUNNING3:02 nvl72094-T04,
sharedQ357148922 RUNNING16:54 nvl72094-T12..15. No observationtimeout/restart.
Currentturn PROGRESS: Q3completeaudit and quantitativehandoffbenefit, combined
comparison saves24captures withoriginaldraw, packedsource exactsubmitted/running.
No qualifying sixcase/fullpath/alignment result yet; goalactive.

## Multi-candidate audit support — September14,19:38UTC

Previous turn PROGRESS: Q3handoff quantitativeaudit and packedsourceexact launch.
Revalidated7149145RUNNING3:34 and7148922RUNNING17:26 atturnstart, no restarts.
Packedsource firstsmall prepared record passed includingnormal/paddedroutes;
remaining records pending, no wholejobPASSclaim.
Reusable collect_megamoe_sweep_arms.py now accepts --evaluate-candidate PREFIX,
fetches/auditsallarms once, and writes eachsecondary dispatch/fullpath report
withsuffix. Alluse samecase/job/source/repeat/history/component guards, exclude
self from suppliedfullpathcontrols, rejectselfdispatch or emptycontrols. Primary
completion boolean retainsoldmeaning; candidate_evaluations map reports each
method separately. Offline regression oncapturedQ3 sharedtargets reproduces
primaryreports exactly, separatelyrejectsfixedpoolcontrol, rejects selfcomparison.
Evidence multi_candidate_collector_validation.json. NoGPUrerun/refetch.
Packed combined collectors must use --candidate candidate --evaluate-candidate
packed --dispatch-control control --fullpath-control control --fullpath-control
compact --fullpath-control fixed --fullpath-control packed. Secondarypacked
comparesagainstcontrol/compact/fixed; primarysource comparesagainstallfour.
Newcollector synced into packedsource workflow; alreadyrunningQ35collector76209
retainsloadedoldcode and doesnotneedrestart.

19:40UTC exact7149145 live9:02, Q35timing7148922 live22:54. Exact snapshot
now proves wideH32768 actualnum_stages4/smem214532, versusoriginal5; previously
projected wide-depth reduction is confirmed, no wide-efficiency claim. Target
H4096 stagedepth/smemchecks await finalstrictaudit. No source changed inrunningjob.

19:45UTC sharedtargets Q357148922 collector76209 EXIT0 COMPLETE_AUDITED; both
models48captures240phase1440ordinary retained. All6dispatchgatesPASS(history
13.59..19.48%). Q35producer vsfixedpool−1.103/−0.780/−1.246us(rows13/30/5),
fullpath+2.783/+1.034/+1.824%. But vsdense producer+1.955/+2.950/+1.702us and
fullpath+0.473/−3.538/−0.125%; producergoal stillfails. Versuscompact compute
span−2.208..−3.584us demonstrates contiguousrow benefit, but producer+1.531..
1.691us remains. Selection REJECTED_AS_STANDALONE, e2e_rejection.json recordsall
six. Preserve partialhandoff benefit for a supportedcombination; no promotion.
Packedsource7149145 live14:05, lastrecordlargeallM done at11:57snapshot; legacy
records pendingfinalaudit. BothtimingGPUjobs terminal, so launchbothcombined
models immediatelyonce exact496/configaudit passes.

## Packed alternatives timing launched — September14,19:49UTC

Packedsource exact7149145 COMPLETED0:0/14:32, strict496comparison/12orderedrecord
PASS: all requested case/rank/source matches, legacy and all6M, targetH4096
actual stages/shared sizes. WideH32768 confirmed stages5→4; no general compute
nonregression claim. User withdrew FP8 relocation; math/location stay unchanged.
Existing producer output strides implement source layout only.

Combined timing7149421Q3 and7149422Q35 submitted19:48:56/57UTC, initialRUNNING
24/25s on nvl72172-T13,15 and nvl72150-T03,04,09,11. Both45min jobs;
collectors35996/92478 monitor300s and evaluate primarycandidate plus packed arm.
Original seed4482461993793792015, Q3rows18/9/10,Q35rows1/13/34 unchanged.
Ten symmetric arms: compact/fixed/packed/control/candidate then reverse repeats;
60captures300phase1800ordinary expected across bothmodels. No new draw/refetch,
source changes, extraGPUjobs, or performanceclaim. Standalone packedexpertframe
plans remain SUPERSEDED_UNRUN; do not submit. Currentturn progress is verified
exact/configuration evidence and timing launch, goal ACTIVE. Next audit BOTH
candidate_evaluations, not primary completion boolean alone. Any qualifying
candidate still needs EP2/runtime stride coverage and fresh serving alignment.

Continuation: both live jobs7149421/7149422 verified RUNNING7:02. Each finished
compact and fixedpool reference steps (about3min/arm); packedexpertframe is now
running. Empty top-level monitor tails did not indicate a hang: direct bounded
stdout/accounting confirmed actual completed captures and progressing job steps.
No restart/resubmission. Collectors35996/92478 remain live; no finaltiming data yet.

Static review of saved packedsource EP8 SASS confirms framepull16a0, independent
remoteweightload16c0 before framewait1740, localsharedSFload17d0, activationstore
1a40, fullstorecompletion1a90, arrivalpublication1b30. dispatch_order_review.json
records exact instructions/scope. Thus remaining weightread was not inadvertently
serialized behind framewait. No latency-causality claim. Existing contiguous TMA
consumer retained. Prior packedinput prototype used producer recopy; its rejection
does not measure the new direct-strided producer, but argues against assuming
packing alone guarantees benefit. Early local activation store overlapping SF
transpose remains a conditional idea, unimplemented and unproven.

Reviewed pre-promotion runtime gap: MiniMax quant4/quant16/quant16_sigmoid plus
nonquantized launch need stride/guard/reuse coverage; EP2 M16/M32 supplement and
fresh serving alignment remain. Defer extraGPU validation until targetperformance
qualifies. Previous turn PROGRESS (exact evidence and authoritative state update);
this turn static-order evidence plus verified live timing wait. Goal ACTIVE.

Supplement prepared while7149421/7149422 remain live: reusable
codex_scripts/validate_fused_quant_outputs.py, synced to packedsource workflow.
Checks independent Torch FP8/SF bytes, contiguous/frame uint8/frame int32/separate
padded views, untouched guard bytes, two graph replays with changed input values,
Qwen producer and MiniMax quant4/quant16/sigmoid16 plus nonquantized4/8/16.
Default H1024/4096, rows17, experts128 gives70checks. UNRUN: no correctness claim
or extraGPUsubmission. The generic MiniMax8 launch specifically covers the prior
missing-argument correction. Use existing run_cuda_check.sbatch and prebuilt native
cache; can run within a later EP2 allocation before distributed exact checks to
avoid another allocation. Only expand validation after targetperformance qualifies.
Previous turn PROGRESS/static-order review plusverifiedwait. Currentturn creates
necessary reusable runtime coverage; goalactive and currentGPUjobs unchanged.

First-pass diagnostic (not acceptance):7149421/7149422 live13:10, both completed
compact/fixed/packed/dense arms and started sourcepacked. Existing timeline helper
read24completecaptures/720ordinary records without refetching traces; snapshots
packed_source_pull/first_pass_reference_summary.json and first_pass_packed_analysis.json.
Expertpacked vsfixed producer−0.321..−1.492us, but still+dense1.287..3.636us.
Q3 fullpathvsdense+1.163..1.576%; Q35rows1/13/34 +3.833/−1.340/−4.324%.
Q35rows34 dispatchvsfixed−24.155% inthisfirstpass. No repeat/frozenhistory gate
claim; all5phase samples preserved (including long initial phase spans). This
argues against promoting expertpacked alone; awaits sourcepacked/repeats before
next implementation. Later collectors confirmedbothRUNNING15:26/27, no failures
or restarts. Currentturn PROGRESS: quantified intended producer-saving mechanism
and residualcost, plusverifiedwait. No extraGPUjob or timedsource modification.

## Early-store overlap candidate — September14,20:14UTC

Forward sourcepacked passes complete bothmodels atlive16:58; bothrepeat jobs
continue. first_pass_candidate_summary.json/first_pass_candidate_analysis.json
retain all6cases180ordinary and30phase samples. Versusdense sourcepacked producer
−0.121..+0.114us (roughlyunchanged), dispatch−3.288..−9.701%, fullpath−0.989..
+0.097%. Versuscompact Q3fullpath−1.400..−3.176%, Q35+3.743..+4.816%, but
all6dispatch−42.079..−78.673%. No acceptance conclusion before final repeats;
sourcepacking alone has not recovered the transfer advantage.

Prepared independent packed_source_early_store, one-header delta on sourcepacked:
local activation TMA store/commit immediately after remote frame wait, before SF
transpose/weight stores. Persistent elected leader owns issue/commit/wait; original
all-lane SF join, full store wait and release arrival retained. Lamport keeps its
original store/commit position. No extra global copy or new layout/producer/
consumer changes. Source/SASS establishes serialization opportunity; not a claim
it is large enough to pass >5%/E2E gates. Inherited wideH32768 stages5→4 unresolved.

Full remote parentcopy completed before overlay. CPU7149735 submitted20:10:35,
initialRUNNING1:00cpu-0052, monitor66267/180s. Five steps EP2/EP8/nativegate+MiniMax/
actualhost/SASS; original running timing sources unchanged. Exact plan352/8records
Q3/Q35/wide/all6M prepared+synced, UNSUBMITTED pendingCPU andGPUslot. EP2 and
standalone stride runtime coverage remain pre-promotion work; standalonevalidator
prepared earlier is available in workflow but UNRUN.

Fresh seed6974469513186867194 from reusablepreparer, no reroll/filter:
Q3rows9/28/10,Q35rows51/48/20. Eight symmetric arms compact/source/control/candidate
and reverse; source=packed_source_pull, control=originaldense. Preserved each
source's actual cache lineage; frozenfullpaththresholds beforetiming. Both35min
plans synced butUNSUBMITTED. Futurecollector --dispatch-control control with
--fullpath-control control --fullpath-control compact --fullpath-control source.
Do not evaluate source as a new candidate; it is the causal scheduling control.
Currentturn PROGRESS: first-pass causal diagnosis and new scheduling implementation,
CPUbuild plus exact/freshplans. At mosttwoGPUjobs7149421/7149422 remain running;
CPUjobseparate. GoalACTIVE; no promotion, finalalignment, or main/index changes.

20:17UTC earlystoreCPU7149735 COMPLETED0:0/3:46,5stepsPASS, monitor66267 EXIT0.
Actualnativegate+MiniMax/host rebuilt, EP2/EP8cubins. Logs/SASS collected and
explicitregisterreviewPASS R41dispatch<48/R23consumer<40; assertioncallsunresolved.
Compiledorder confirms localstore1810/commit1820 precedeSFload1890/store1940;
fullwait1a90 precedesreleasearrival1b30. PersistentP6 store/waitowner preserved.
completion.json/register_audit.json/dispatch_order_review.json saved. Exact352 is
ready atfirstfreeGPUslot; bothprior timingjobs7149421/7149422 stilllive25:28/29.
No CPU-only work will hold a GPU allocation. No newGPUsubmission yet.

20:20UTC Q37149421 COMPLETED0:0/30:02, collector35996 activelyauditing10arms.
Q357149422 stillRUNNING30:49. Immediate freedslot used earlystore exact7149885,
submitted20:20:10, initialRUNNING20s nvl72086-T06. Collector21453 uses maintained
collect_megamoe_abba.py every180s, then strict audit_megamoe_exact_outputs.py352.
At most2GPUjobs remain (7149422 and7149885); CPUpreparationalreadydone. Neither
firstpass diagnostic nor compiledstoreorder constitutes performanceacceptance.

Q3 collector35996 EXIT0 COMPLETE_AUDITED30captures150phase900ordinary for7149421.
Both candidate_evaluations false, allrecords/repeats retained. Sourcepacked dispatch
history−6.05/−9.31/−7.21% (rows18/9/10), all3FAIL; fullpathvsdense−0.365..−0.765%,
vscompact−1.600..−2.683%. Producerroughlyunchanged (−.081..+.008us vsdense) does
notrescue transferregression. Expertpacked all3dispatchPASS(history15.12..20.23%),
fullpathvsdense+1.196..1.695%, butproducer+1.375..1.458us; vscompact−.685/+.401/
+.436% (rows18/9/10), cannotpromote. Q3decision_summary.json records pairedresults.
Q357149422 alsoCOMPLETED0:0/31:52; collector92478awaitscheduledfinalcollection.
Earlystore7149885 remains soleGPUjob RUNNING (lastdirect1:25); collector21453live.
Q3 failures alreadyruleoutstandalonepromotion; fullQ35evidence stillrequired for
completeexperimentreport. No resubmission or extraGPUjob.

## Both packed alternatives rejected after full audit — September14,20:25UTC

Q35collector92478 EXIT0; bothmodels60captures300phase1800ordinary fullyaudited,
allrecords/repeats retained. Eachmethod independently evaluated via the new
candidate_evaluations map. combined_rejection.json plus eachvariant's
 e2e_rejection.json recordsall6cases; no reruns/refetches or partial-pass promotion.
Sourcepacked0/6dispatchPASS, fullpath0.216..1.256%slower vsdense, producer−.081..
+.105us roughlyunchanged. Q3 vscompact1.600..2.683%slower; Q35vscompact3.525..
3.751%faster, butstillfailsoriginaldense/history. Exact496 remains correctnessonly.
Expertpacked5/6dispatchPASS(history15.12..27.81% onpassingcases); Q35rows34history
−5.856%. Producerstill+dense1.336..3.506us despite−fixed0.419..1.505us. Q3fullpath
+dense1.196..1.695%; Q35rows1/13/34+dense2.351/−1.282/−4.427%. Both standalone
variantsREJECTED. Expertpackedoldseed/plans remainSUPERSEDED_UNRUN; measuredsixcases
come fromcombinedseed4482461993793792015, do not misattributeoldunrunselection.

Earlystore7149885 soleGPUjob RUNNING, collector21453 last3:20/2records (Q3 pair).
CPU/pass/storeorder alreadyvalidated; exact352 incomplete and fresh eight-arm
performance unsubmitted. Goalactive; no servingalignment/integration/promotion.
Newnative stride validator remainsUNRUN until a targetperformancequalifier.
Currentturn PROGRESS: fulltwo-methodaudit/rejection and nextschedulingcandidate
implementation/CPUproof/exactlaunch. No blockers, waiting onrecordedlivejob.

## Early-store timing launch — September14,20:35UTC

Previous turn PROGRESS: fulltwo-methodrejection and newearlystoreCPU/exactlaunch.
Revalidated7149885live8:35, thenCOMPLETED0:0/11:11. Immediate terminalcollection
and strict8record352auditPASS (all6M, wide, actualtargetstage/smem). Collector21453
subsequently EXIT0 withsameaudit, noGPUrerun. Fullperformance remainsunproven.
Submitted fresh eightarm timing7150076Q3/7150075Q35 at20:32:00, initialPENDING;
collectors90292/55335 monitor300s. Originalseed6974469513186867194 unchanged,
Q3rows9/28/10,Q35rows51/48/20. Compact/sourcepacked/dense/earlystore and reverse,
48captures240phase1440ordinary expected. --dispatch-control control;
fullpathcontrolscontrol/compact/source. No furtherGPUjobs.

Applied PIC-C skill to prepare causalprofiling if timing stillfails. Existing
same-rank smoke already brackets warmed measuredgraphreplays with CUDAprofilerAPI.
OlderOCI memory recordsPIC-C/NCU absent (6544129/6544501/6544786); currentloginPATH
also hasnone, not proof aboutcurrentimage. Bounded CPU-only inventory7150136
submitted20:35:28 (1CPU/5min) inside exactcurrentbenchmarkimage, knownbinarypaths
andonelevel/opt/nvidia only, noGPU/metric/version/hash checks. Artifacts under
picc_profiles/tool_inventory. No profilerGPUsubmission or fallback behavior;
collect capabilityevidence before planning expensive hardware profiling.

## Active experiments — September14,17:57UTC

- **compact_warp_async_gather**: consumer-only cp.async.cg16B gathers from rank-major pushed frames directly into final swizzled MMA shared tiles. No global recopy/allocation; producer, metadata, SF/B/combine and publication unchanged. Each lane waits/fences/syncwarps before existing cluster arrival; completed A writes contribute0 TMA transaction bytes. Per-stage wait may reduce prefetch overlap, so performance unproven. CPU7146967 PASS3:09/10steps: native+compatiblehostseed,four ordinary builds (Q3M16,Q35M16/M32/M64). Changed copy-role R25/18/23/18<40, no local memory or unresolved branches; two indirect calls remain in unchanged dispatch, no whole-kernel CFG proof. CPU address/swizzle/shuffle4224vectors covers all supported M16/32/64/96/128/192.
- Warp-async exact7147145 **PASS352 comparisons/8orderedrecords/10:03**. Q3/Q35/wide each60 and general E8 record172; explicit configs prove all six host M branches actually executed. Exactcollector39427 EXIT0; audit synced. Original fresh seed4959874324339898400 Q3EP8rows16/6/15,Q35EP16rows3/18/19 fixed across six arms/model, historical thresholds frozen. Timing7147418Q3 submitted17:55:39,7147425Q35 submitted17:56:22; both verifiedPENDING/Priority on initial observations. Collectors16230Q3/43766Q35 watch300s, no submissions/restarts. Both pending jobs count toward max2outstanding GPU jobs.
- Independent **direct_bulk_push** implemented on general_storage: replace distributed remote activation vector stores with local staging+one elected TMA bulk store into identical contiguous expert row. Reuse existing per-dispatch-warp pull_buffer, no new allocation/global scratch/kernel. SF/weight/return metadata overlap bulk transfer; full remote completion precedes unchanged proxy fence/warp join/SYSacq_rel arrival. Prefix, ordinals, native routing, contiguous consumer/combine unchanged. Different causal test from compact consumer gather and old Lamport kMegaPushDispatch branch; no assumed performance. Full remote base copy completed before three-file overlay sync.
- Direct-bulk CPU7147393 submitted17:54:36, verifiedRUNNING/00:30; monitor45275/180s. Eight steps: actualnative+compatiblehostseed,Q3/Q35/EP2 builds/disassembly. Existing pipelined host donor found; three host allocation/descriptor files byte-identical to general_storage base (host_donor_compatibility.json). Eight-record352-check EP4 exact plan preparedUNSUBMITTED with general_storage vector control; fresh timingseed14608292657801473 Q3rows6/10/28,Q35rows51/22/12, six arms/model and historicalthresholdsfrozen. All plans synced; GPU awaits CPU/register PASS and slot. Changed-path EP2 runtime verification remains a pre-promotion requirement, not claimed by CPU compile or EP4 exact.
- **compact_shared_routes REJECTED_E2E_REGRESSION_DO_NOT_RESUME**. CPU7146328 PASS2:10; exact7146777 PASS180/7:08. Timing7146910Q3 COMPLETED16:50,7146911Q35 COMPLETED18:06; collectors26164/93777 EXIT0 COMPLETE_AUDITED,36captures180phase1080ordinary allretained. Dispatch history+15.58–23.45% all6, but fullpath all6 slower thanmatchedlinear: Q3−0.646..−0.851%; Q35rows15−0.624%,rows21−1.278%,rows7−6.291%. Producer launch delayed0.341–0.685us inall6; Q35vsdense−4.636..−8.898%. Rows7slowrepeatretained, notsole rejectionbasis. Evidence compact_shared_routes/e2e_rejection.json. No alignment/sweep/stacking into another candidate.
- **compact_preload_token_stride REJECTED_E2E_REGRESSION_DO_NOT_RESUME**. Exact7146196 PASS180; timing7146329Q3 COMPLETE16:48,7146330Q35 COMPLETE18:59;36captures180phase1080ordinary allretained. Dispatch history+12.07–22.81% all6, Q3 fullpath+1.54–1.77% vsdense but Q35−3.92..−6.72%. Versuslinear Q3−0.38..−0.65%,Q35−0.42/+0.27/−0.16%. Q35 compute-span+3.70–4.19us vsdense persists after stride change; locates regression, not hardware attribution. Evidence compact_preload_token_stride/e2e_rejection.json. No alignment/full sweep.
- Reusable collection fix: fetch had received .sbatch instead of sibling megamoe_input_sweep.py. Failed Q3 collection log preserved; old15215 EXIT1,old87478 deliberately stopped143; corrected46534/80175 EXIT0 using saved captures, no GPU reruns. Primary dispatch gate now includes worst candidate repeat versus fastest dense repeat>5% plus history. Exact auditor optionally requires actual runtime tuples from MegaMoEConfig lines; positive check on saved metadata exact180/Q3M16/M32 and negative nonexistent rows4/M192 PASS. Warp-async expected six configs explicitly enforced; no missing-config success.

## Rejection checkpoint (16:25UTC)

- **compact_frame_linear_index REJECTED_E2E_REGRESSION_DO_NOT_RESUME**. Authoritative evidence linear/e2e_rejection.json. Latest matched six cases: Q3rows1−0.3558%,rows10+1.6611%,rows14+0.9328%;Q35rows4−2.5524%,rows5−2.1955%,rows51−3.5927% fullpath gain vsdense. Four fail, even though allsixdispatchgatesPASS. This disqualifies the implementation, not push as a method.
- Local validation/sweep/report runners stopped deliberately: sessions42842/34979 EXIT143 and all3locks released. GPU7145382Q35EP16 CANCELLED16:48 elapsed and7145806Q3EP8 CANCELLED1:51. No active campaignGPUjobs; no automatic submits/reports. Do NOT resume rejected plans. run_megamoe_sweeps.validate now rejects manifest statuses starting REJECTED; selection, launch order, both manifests and report plan marked rejected.
- Four completed groups retained: Q3EP2 7143497/46:13 (86cases),Q35EP4 7143496/1:28:27 (139),Q35EP8 7144300/54:59 (83),Q3EP4 7145082/28:34 (42). Total350cases1750phase10500ordinary. Partial captures from cancelled jobs retained; do not complete the remaining rejected-source matrix merely to finish the old plan.
- Group dispatch mean gains vsfrozenhistory: Q3EP2 +11.313622%(85/86>5%,allimprove);Q35EP4+15.524362%(128/139>5%,5regress/1tie);Q35EP8+15.938877%(all83>5%);Q3EP4+13.644933%(39/42>5%,rows4/7regress,rows34 numericaltie reported as−1.3e−14%). No numerical-classification code change made yet; raw samples retained.
- Group fullpath mean per-case gains vsunpairedoriginaldense: Q3EP2−1.949544%(82regress),Q35EP4−7.075292%(all139regress),Q35EP8−4.777033%(82regress),Q3EP4−1.692692%(41regress). Group reports sweeps/MODEL/epN/{frozen_history_comparison,fullchain_comparison}.{json,csv}; all ordinary records retained. Unpaired means separate historical runs, not same-session interleaved arms; onlyrank0, not distributed completion orrequestlatency.
- Both new selected-source normal-routing model alignments remain PASS: Q3 7142922 all816top1/top10;Q35 7142931 all2856top10/2853top1, knownrank2 firsttokens119/123/143. They prove alignment, not faster E2E. Actualall8workerlogs/model confirm EP8/twolanes/requestedrows518/1557; selected frame/L1 memory projections remain useful.
- NEXT: rebuild acceptance around matched ordinary E2E latency; audit historical fullpath baselines separately from phase minima; distinguish rank0 timeline fromslowest-rank completion; isolate producer added launch/work andconsumer kernel regression. No newGPUjob until a concrete general candidate/causal comparison is prepared. Prior elected/shuffle/fixedbulk alternatives also fail common E2E promotion; do not rebrand their phase gains as successes.
- Preserve main/index and all completed/partial measurements. No subagents, fallback, /tmp, py_check/hash/version checks orcommits. The deferred stride/SF/metadata ideas below may now be reconsidered for E2E; the previous prohibition on branches until completing the rejected sweep is superseded.


## Full-path acceptance enforcement — September14,16:33UTC

`run_megamoe_validation.audit_fullpath` now recomputes ordinary producer-start→MegaMoE-end from all30 raw records/arm/case and both repeats, requiring identical case coverage, expected source in both candidate/control repeats, same recorded job, finite positive timings and kernel/interval closure. Future selections must explicitly name `fullpath_control_arm` and `fullpath_control_source`; missing fields fail closed. Every case must improve pooled latency and the slowest candidate repeat must beat the fastest control repeat before validation can proceed. This is a conservative rank0 E2E prerequisite; it does not establish historical-best E2E or slowest-rank completion. Those remain required investigation work.

Critical check on existing six matched cases reproduces all saved pooled/worst-repeat gains, rejects both model groups, rejects mismatched job identity, and verifies rejected sweep manifests cannot resume. Evidence: compact_frame_linear_index/fullpath_gate_audit.json. Stale runner status files now explicitly STOPPED with prior snapshots retained. No new GPU jobs or kernel changes.

Causal split vs dense: Q35rows4/5 add1.3269/1.0469us inside MegaMoE and0.1733/0.2496us before its start; row51 adds2.6528us before start despite saving0.3067us inside MegaMoE. Investigate consumer SF/gather work separately from larger-row producer/metadata costs; this split is evidence of where costs accrue, not proof of their hardware cause.


## Current candidate preparation — September14,16:41UTC

Previous goal turn was progress: raw-record E2E gate implemented and verified against retained matched regressions. New isolated `compact_preload_token_stride` changes both source-row helper overloads on compact linear to token-major; no other kernel changes. Bijection/bank/address agreement checked across EP1..64 and capacities1/3/16/768/1920/3840 (1,663,192 points, under1second). General locality hypothesis only; old-stride Q3 result on old producer was near parity, Q35 unrun. Allocation, stride within each row, metadata and publication unchanged.

Fresh seed2858878927204158955: Q3EP8rows3/2/28;Q35EP16rows14/21/27. Prepared six-arm pull/control/candidate/candidate_repeat/control_repeat/pull_repeat with unchanged30ordinary+5phase policy. Updated reusable preparer accepts named reference arms from existing multi-arm templates. Timing jobs UNSUBMITTED, gated on CPU+exact. CPU7146137 submitted16:38:42, immediate scheduler PENDING; builds actual router/cache and ordinary consumer specializations without GPU. Paired EP4 exact plan has6records (Q3/Q35 M16/M32 plus H32768 chunks), wide rows1/6 avoids diagnosed unrelated fixture.

Source snapshot sync initially raced full copy and overlay (copy reported an existing-directory error); recovered by completing full base rsync, then applying overlay sequentially before CPU submission. No GPU ran on incomplete source. Main/index preserved.

Historical E2E reconstruction reads actual old SQLite captures using current timeline helper. Initial Q3 extracted164cases/492samples. Initial Q35 first remote CSV was stale299cases; synced authoritative local300-case CSV before re-extraction. Old dense ABBA remote aggregate CSVs absent; synced only those collected CSVs before extracting all8arms. No samples discarded, no GPU job for extraction. Three full generations plus original dense ABBA will form an explicit frozen E2E reference independent of phase minima, retaining differing sample counts.


## E2E baseline and new exact job — September14,16:48UTC

CPU7146137 COMPLETED0:0/2:15: actual router build, compatible host seed, both ordinary consumer builds/disassembly all6steps PASS. Dispatch highest registers Q3R41/Q35R31<48; TMA regions stay<40 (Q3max26,Q35max22). Paired exact7146196 submitted16:44:53; immediate scheduler PENDING/Priority, one outstanding GPU job. Timing plans remain unsubmitted.

Historical E2E freeze complete: `e2e_history/{plan,baseline}.json`, reusable `freeze_megamoe_fullpath_history.py`. Three prior full generations plus all8original dense ABBA arms, exactly the explicitly listed pre-campaign roots. Q3:164cases/504capturemeans/10,692ordinary records; Q35:300cases/911capturemeans/19,257ordinary records. Earliest Q35 sweep actually has299cases (previous note incorrectly expected300); newer generations supply union300. Earliest captures have3samples vs30later; preserve this distinction. Validates original audits, case/job identity, input contracts and raw interval closure, selects minimum FULL-PATH capture mean independently of dispatch minima. Does not claim an exhaustive search of every old rejected branch or statistical equivalence across sessions.

Validation now also requires explicit `fullpath_history` and both candidate repeats faster than each historical case mean, with matching input contracts. Six retained linear cases recomputed against this history still reject Q3rows1 and all3Q35; Q3rows10/14 pass. Evidence e2e_history/rejected_linear_gate_check.json. Six future stride thresholds frozen before timing in compact_preload_token_stride/frozen_fullpath_thresholds.json.

Local rsync does not trim `/./` in relative paths as expected: an earlier -aR copy created nested scratch paths remotely instead of authoritative targets. Corrected by setting subprocess cwd and passing plain relative paths; actual remote selection/runner status now verified REJECTED/STOPPED. Used same correction for historical aggregate CSVs; all8ABBA timelines now extracted. No active rejected-source workflow.

Ordinary baseline.json `milliseconds` is not usable as isolated kernel latency: retained Q35rows51 reports~33ms under Nsight versus~65us producer→MegaMoE intervals. Existing unprofiled `_time_graph` does CUDA events around repeated graphs then MAX across ranks, and normal-routing campaigns use graph_batch20. A qualifying source needs that unprofiled slowest-rank check; do not substitute profiled event wall time for kernel E2E.

Further metadata/producer idea, unimplemented: warp0 already knows route IDs, but payload helper's7copy warps each reload global IDs, recompute destination division and match_any/ballot. Share a per-CTA destination list and unique mask produced by the routing warp through the existing barrier. Could reduce producer work without changing remote ordinal/descriptor protocol or publication; needs actual SASS/latency evidence and general invalid-route/topk handling. Keep separate from current stride causal test.

## Prior dispatch-only selection evidence (historical; rejected for E2E)

- Latest fresh seed1951216027023626904 ten-arm timing: **Q37142487 COMPLETED0:0/29:47**, **Q357142492 COMPLETED0:0/31:55**. Each30captures150phase900ordinary+900timelines fullyaudited; ALLrepeats/outliersretained. Arms originaldense/fixedvector/compactlinear/fixedbulk/shufflegather thenreversed. Q3rows10/1/14;Q35rows4/51/5(M32at51). Reports pool_followup/MODEL/{linear,candidate,shuffle}_dispatch_comparison.json and *_user_gate.json, fullchain_comparison.json,completion.json.
- **Linear ALL6 USERdispatchgatesPASS**, including worst repeats versus frozenhistory+matchedoriginaldense. Q3history16.512/21.739/18.692%, dense15.925/18.873/19.258%(rows10/1/14). Q35history20.047/15.837/12.564%, dense14.824/18.421/15.802%(rows4/51/5). Causal5%vsincrementalcontrols mayFAIL; retained separately, baseline nevermoved.
- Latest linear fullchain vsdense: Q3rows1/10/14 **−.3558/+1.6611/+.9328%**; Q35rows4/5/51 **−2.5524/−2.1955/−3.5927%**. Balancedrank0producer→MegaMoE, notrequestlatency/throughput; noindependenceclaim. Prior phase-only selection was insufficient; now rejected because fullpath regresses.
- **Shuffled gather rejected for promotion**: all6fullchainregress vslinear includingworstrepeats. Q3rows1/10/14 −2.2765/−.8047/−.9618%; Q35rows4/5/51 −.4937/−.8793/−.5787%. Staticreductionofelection/descriptorissueoverhead didnotproduceimprovement. Itstillpassesall6phasegates; phasealonewouldhideconsumerregression.
- **Fixed bulk preload retained as alternative, not selected**: native-onlybulkproducer vsoldfixedvector improveschain5/6; Q3+4.616/+2.870/+3.053%, Q35+2.857/+2.732/−.364%(rows4/5/51). Versuscompactlinear losesallQ3(.86–1.81%)andQ35row51(1.187%), winsQ35rows4/5 by.944/.675%; largerpoolcostnotjustifiedforcommonpath. FullchainstillregressesvsdenseALL6. Bothnewvariants420exactPASS; exact gatepreserved below. No push-impossibility claim.
- Consolidated new exact420PASS/12records: main7142265 first6Qwen records PASS240 butwholejobFAILED inunchangedwidecontrolunprepared24-routefixture; wide recovery7142411 COMPLETED4:31/PASS120, EP2supp7142412 COMPLETED4:35/PASS60. ActualEP4Q3/Q35M16/M32,wideH32768two/threechunks,EP2M16/M32audited. compact_frame_shuffled_gather/combined_exact_audit.json. Supersededstandalonefixedvalidation MUSTNOTsubmit. Generic/legacy pathsunchanged,parentcoverage, no freshcoverageclaim.
- Earlier linear/elected exact7141337+7141684 PASS620aggregatecomparisons. Earlier8armlinearALL6gatesPASS; electedfullchain4.38–4.92%slowlinearQ3,2.01–3.89%slowlinearQ35; electedrejected. LinearCPUnative/cache7141058PASS4:52; addressidentity4704boundarychecksPASS, sourceformula unchanged since. No newselectedsource edits aftertiming.
- NEW selected-source serving memory evidence: all8worker logs/model from actual7142922/7142931 independently confirm max_rows518/1557, EP8,2lanes,normalrouting. Selected host/layout formula: compacttwo-bankframes51/127.5MiBperlane versusoriginalL1activationregions204/648MiB, addedL1activation0bytesacross2lanesbothmodels. Otherbufferregions/weights/KV/allocator reservations excluded. Evidence linear/serving_selected_worker_capacity_evidence.json andserving_compact_pool_projection.json; priorTorchcounterevidence remains scopedtooldpreload.
- Serving capacity confirmed fromall8expertworker logs/model: Q3requested518→aligned768,Q35requested1557→aligned1920, EP8and2lanesboth. Buffersreusedacrosslayers. Fixedpoolprojectedaddedactivation+rowmajorSFtwo-lane1.1484375/14.203125GiBperGPU vsoriginalactivationpool, othermetadataexcluded. Torchpeakreserved36.217/40.123GiB doesnotmeasuretotalfree/externalallocations. Evidence preload/serving_worker_capacity_evidence.json,serving_fixed_pool_projection.json. CompacttargetQ3EP8/Q35EP16allocationunchangedatcaps1..3840; Q3EP64adds22.5/118.5MiBperbufferat768/3840.
- PriorPRELOAD whole-model alignments7140414/7140417PASS(816/816top1Q3;2856top10/2853top1Q35,3knownrank2firsttokens), ordinarysyntheticMAXrank1584exactPASSbutnearparityvsdense. DoNOTapplytheseasnewlinearalignment. Extreme-skewperformance/requestthroughputunmeasured; Q35token-majorstrideunrun, noimpossibilityclaim.
- PriorPRELOAD partialsweep retained, paused: Q3EP2 7140728 86cases(history85>5%,mean12.566%;chain83regress,mean−2.326%unpaired);Q35EP4 7140727 139cases(history127>5%,mean15.491%,4regress;chainALL139regressmean−7.492%). Previousrunner6387STOPPED143, noGPUcancel. Missing239preloadcases nofullclosure; currentselectionrequiresnewall464linear.
- Fullreport plan prepared at linear/sweeps/report_plan.json: frozen CSV minima (includingABBA), threefullhistory comparisons, unpairedordinaryfullchain reference, bothsame-inputHTML templates andPNG/SVG/PDFplot. BothHTML --plan-only membership/alias checksPASS againstnewmanifests. No newtimingsrenderedbeforefullcoverage. Visualize skill read forreport renderer; reloadfullskillbeforecreating/updatingvisualaftercompaction. Renderer pathrecordedinplan.
- compare_megamoe_fullchain.py now validatesauditedcandidate/referencecontracts, actualtimelinecase/jobidentity, ordinarysamplecounts, finitetimestamps/ordering andkernelend-startclosure, thenreportsallmeans/regressionsunpaired. All86savedpreloadQ3EP2 cases reproduced including83regressions,mean−2.325744%. report_plan.json nowhasremoteCPUtimelinecollection+localfullchaincommands; noGPUtest.
- Final reporting: compare_megamoe_frozen_history.py ROOT --history-csv C/history/MODEL/decomposition.csv --output PREFIX (optional --ep) reuses full/EP actualcaptureaudit loader and retains frozen minima/provenance, CSV+JSON andallregressions. ExistingpreloadQ3EP2 all86savedcasecomparisons reproduced exactly; noGPUrun. This frozen report includespre-campaignABBA minima that threefullhistoryroots alone canmiss. Ordinary/fullchainreportsremainseparate.
- CPU-prewarm workflow gap reviewed: existingDeepGemm Compiler::build asksdevice_runtime forCUDAproperties andreturns loadedKernelRuntime; no explicitofflinearch/cache-build mode exposed. Native/host CPUgatesandpreparedcaches alreadyused; no manualcacheidentitysubstitution orselectedsourcechanges attempted. FutureofflineJITworkflow couldreducefirst-specializationcompilecost, butnotintroducedduringfrozen-sourcevalidation.
- Analysis helpers: compare_megamoe_sweep_arms.py now supports --control-arm(defaultcontrol) so the namedcontrolarmcanbe evaluated as candidate without aliases; defaultsemanticsunchanged. Actualall10-armreportsexecutedPASS. compare_megamoe_router_timelines.py recomputesallpairedmeans/worstrepeats from30finite actualrecords percase/arm, all900/modelretained. SweepfetchauditsactualCSVinstancecounts, notassumednsysJSONinstancesfield. Currentrelease_push/workflowtimelinehelper supports --sweep; oldpreloadcopystale, newlinearworkflowupdated. Exactcollector isnotforphase sweeps.

## Current compact-pool direction (implemented, exact testing)

compact_pool_push sends once per unique destination into two source-indexed final
L1 banks; consumer gather4.2CTA reads local rows directly into MMAshared, no global
receivercopy. Matched compact_pool_pull keeps contiguous pull. Capacity max(original
compactL1,2*EP*capacity), no addedQwenactivationmemory at12testedcapacitygeometries;
unusedlocalordinalstorage/stores removed, so Qwenallocation matchesoriginaldense.
CompactSF/L2/arrival stay unchanged; denseTMAboxheight1,legacyLOAD_BLOCK_M; invalid
rows OOB-fill. Five-file overlays, actualhostrebuild. Gather4 ISAprobe7136335 PASS1:03,
SASS UTMALDG.2D.GATHER4.2CTA emitted; register gates PASS; functional exact528 PASS.
Freshsixarmseed2796530709587003306 Q3rows15/3/10,Q35rows51/4/3. Originaltimingplansheldforframevariants; pairedexact7137493 PASS13:16/528.

## Earlier fixed-pool architecture (causal baseline; not servingready)

**fixed_pool_pull / router_pool_push:** two physical final L1 activation banks,
row=(generation*local_experts+expert)*align(EP*capacity,384)+ordinal. Compact SF,
weights,returnmetadata,arrival,L2 and scheduler unchanged. Host allocation/TMA
extent and receiver L1 indexing match.64bit math; explicit signedrow bounds.
Both genericwarp and CTA selectors push activation after dense ordinal assignment
before existing last-CTA/GPUticket/system-rank publication; receiver skips only
activation TMA, retains SF/weight/returnmetadata/localarrival. Matched fixed-pool
pullcontrol uses same physical layout/fence but receiver pulls payload.

Generation proof: n+1 producer writes opposite bank while remote n maycompute.
Before any source reaches n+2, its n+1 MegaMoE acquired every rank publication
n+1, implying all n kernelsfinished; bankn safelyreused. Singlebank earlypush
would be unsafe. Costs atcapacity3840: +2.801GiB/rank Q3EP8,+12.633GiB Q35EP16.
126 bank/expert/finalrounded-block geometry checks PASS. Detailed proof, budget,
and deferred chunk-directory idea: C/router_pool_design/DESIGN.md.

CPU control7134242 PASS3:55,producer7134243 PASS4:05: EP2/EP8receiver+native+
realhostbuild. SASScontrolR43/producerR39<48,5local each. Paired exact7134470
**PASS12:45,9records/528saved comparisons** (264eachvariant, includinggeneric/
CTA and legacygraphs/routing). Own caches prepared. Source5file overlays:
layout/mega_moe.cuh,impls/sm100...cuh,gate_topk_fused.cu,apis/mega.hpp,
host jit_kernels/impls/sm100...hpp. Production code remainsunchanged.

Router timing seed2990911191664381143: Q3rows16/12/15,Q35rows19/4/11. Sixarms
densepull/fixedpoolpull/routerpush/routerpush/fixedpoolpull/densepull.
Initial router pushes BEFORE PDL launch, so it can delay MegaMoE startup: report
full producer→MegaMoE chain, not merely phase shift.

**router_pool_overlap:** native-only controlled variant signals PDL afterordinal
allocation BEFORE activationcopy, exactlyonce/CTA; postcopyCTAbarrier, GPUticket
and finalSYSrankpublication unchanged. Both selectors, deferflag respected.
Receiver/layout/host identical to validatedrouter. NativeCPU7134734 PASS2:01;
exact7134958 PASS264 and Q37135122 COMPLETED16:40. Earlier launch shifts~3.4us
into waiting; fullchain6.12–6.48%slowervsdense, no materialimprovementvsvector. Sixarmseed
3340288658780047407: Q3rows15/6/10,Q35rows16/3/8. Controlrouter_pool_push; extra
densepull. Full timeline/comparison artifacts saved; no qualifyinggain.

## Completed causal evidence

**push_diagnosis7133350 PASS9:12:** marker-only paired densepull/direct sources,
fixedQ3EP8rows12,4arms/20phase/120ordinary/1920pairedmessages, allretained/audited
including50uscontroloutlier and duplicate publication markers. Final pooled
earliest medians (pull/push,us): transport2.368/2.416,address3.376/3.280,
payloadissue3.472/5.248,aux4.048/5.632,publication5.904/8.896,tileusable7.040/
12.560,compute7.632/13.120. Ordinarykernel38.313/46.328us.
CTA0warp0 has no preceding-expert prefix loads: address3.696/3.792,payloadissued
3.824/7.232,published6.944/11.040. Its addresslookup is not thelarge delay.
Per-messageaux→publication1.064/4.936us includes remote-write drain, NOTpure
atomicexecution. Issue≠completion,rank0cannotidentifyalllastremotecontributors.
See C/push_diagnosis/FINDINGS.md,summary.json,perarmdispatch_detail.json.

| Variant | Correctness/build | Performance/conclusion |
|---|---|---|
| direct_push | CPU7132009; exact7132066 PASS264 | Q37132218/Q357132220 PASSscheduler;24captures/120phase/720ordinary audited. All6fail:83.33–94.13% slower matched,75.92–92.33% slowerhistory;fullchain12.15–19.08%slower. Reject implementation, notpush. |
| general_storage | CPU7133636 PASS54s; EP2prepared90saved comparisons in7133665 | Appended max_tokens*topk uint32 ordinals removesEP>=4descriptor-slice restriction. Same-storagecontrol forpublication/width/SF. Original128b/acq_rel. |
| release_push | CPU7133003 PASS4:17; EP2prepared180comparisons(original+release); legacyEP4 7134031 PASS84 | Q37134131 COMPLETED18:04;18/90/540audited. Gains0.265/1.304/1.169% vsoriginal;72.87–82.49%slowervspull,76.98–79.05%slowerhistory. ALL3fail. Acquire removal notmaincause. |
| wide_push | CPU7133768 PASS55s; exact7133821 PASS6:58/264 | Q37134174 COMPLETED17:19;18/90/540audited.256b4pairs vs128b8pairs improves3.924/9.424/7.554% vsoriginal;67.15–82.01%slowervspull. Kernel0.40–1.28%worse;fullchain0.34–1.00%worsevsoriginal,14.18–17.16%vs pull. ALL3fail. |
| sf_blocked | CPU7132795 PASS3:07+host; exact7132948 PASS6:06/264 | Q37133218 COMPLETED18:07;18/90/540audited. Sameallocation/tilecontiguousSF pitch512B,still32sectors/warp. -1.02/-1.71/+1.14% vsdirect;ALL3fail. |
| sf_row4 | 56geometries; CPU7133930 PASS2:42+host; exact7134040 PASS6:56/264 | General4rowSF interleave,16Bcolumnstride+matching3DTMA.32lanes touch16sectorsvs32. Samealloc/otheraddresses/publication/activation. Q37134541running;firstpass~11.8–12.3us vsoriginal12.0–12.4, notlargegain. |
| sf_stride | CPU7132668 PASS4:16+host; exact7132851 PASS6:11/264 | +128SFrows/512Bcolumnpadding,also shiftslaterbufferbases. Sixarmplansready,unrun. |
| activation_push | CPU7132564; exact7132666 PASS6:07/264 | Q37132863 COMPLETED12:15,12/60/360audited;97.67–137.56%slowermatched,ALL3fail. Halfwarpsactivation/halfSFpull,two16bitarrivalcounts;notSF-only causaltest. |
| pipelined_push | CPU7134114 PASS55s; actual4independent128bloads before4stores,R31<48/5local | Originalwidth/SF/acq_rel fixed, loopretained. Host/native seeded,EP4exact264ready;noGPU/perf yet. |
| packed_weight | CPU7132219; exact7132610 PASS7:10/264 | uint64descriptor+FP32weight, payloadpullfixed. Timingreadyunsubmitted;oldEP>=4slice storageconflictswithdirectordinals,needs generalstorage redesign beforepromotion. |

All variants' exactrandom seeds/cases in their selection.json; never reroll/filter.
Q35 unrun for rejected Q3causalvariants is deliberately held, NOTclaimed6-case
completion. Important selectedcases: release seed3534985140580525039,Q3rows7/8/10,
Q35rows18/34/28; wide4502695627838642418,Q3rows28/5/12,Q35rows19/3/32;
sf_row4 852420333524602538,Q3rows7/6/20,Q35rows10/39/2.

## Reusable workflow / pitfalls

- prepare_megamoe_sweep_abba.py creates fresh random4armplans; --control-arm picks
  templatecontrol. Sixarmplans add named pull pair and set maincausalcontrol.
  Compare with compare_megamoe_sweep_arms.py --history C/history/MODEL
  --additional-control pull; all selectedcases+bothrepeats+historygates required.
- submit_megamoe_plan.py syncs plan, recordsjobonce. Source/workflow/cache sync
  MUST precede. No third GPU job; job.json existing means never resubmit.
- collect_megamoe_abba.py is for source/exact plans ONLY, expects MEGAMOE_ABBA_ORDER.
  Exact audit savedcomparison counts by CASE= +MEGAMOE_RESULT_COMPARE_OK; require
  schedulerCOMPLETED0:0 andexpectedrecords.
- Sweep fetch: megamoe_input_sweep.py fetch ARM --remote-root REMOTE_ARM
  --remote-script REMOTE/workflow/megamoe_input_sweep.py. Fetch includesmanifest
  and normal/profiled/source/contracts/actualCSVinstancecount audits.3cases/arm=
  15phase/90ordinary. Fullmanifest intentionallyincomplete for randomsamples.
- Reusable compare_megamoe_router_timelines.py ROOT --arm PREFIX (repeatable) validates matching nonempty cases/30 finite records per arm, recomputes pooled and worst-repeat chain/launch/kernel comparisons from actual records. Existing electedQ3 report reproduced across720records (case-keyed; initial positional audit failed lexical versus numeric ordering only); use for fresh ten-arm comparisons after timeline collection. No GPU rerun.
- Fullchain: current summarize_megamoe_router_timeline.py --sweep readsremote
  SQLite and decomposition.csv. Sync localcollected decomposition.csv backfirst;
  olderworkflowcopy lacked --sweep. Currenthelper syncedrelease/workflow, usable
  foranyvariant. Save30samples/case/arm JSON/CSV;never discardoutliers.
- seed_deepgemm_extension.py donor --cache-root EXACTPATH/deepgemm --copy requires
  emptycache andprovenhostABI/behavior match. Must seed AFTER sourcesync.
  Changedallocation/TMAdescriptor requires realhostrebuild. Cacheparentwrongonce
  causedhostcompileonGPU; corrected. Nativecache path explicitlyset inenv/wrapper.
- review_megamoe_dispatch_sass.py handles two4registergroups in SM100256b op.
  Linearrole span only,notwhole-controlflow proof. Diagnosticmarkersuse48dispatch/
  40TMAeventregisterbounds. summarize_megamoe_dispatch_detail.py preservesduplicate
  markerexecutions usinglastbranchprogressperroute;notremotecompletion.
- EP2legacy numericfixture creates4experts butrouterminimum8, so release7133665
  failedUNCHANGEDbaseline onlyafter9preparedrecords/180comparisons passed. Tested
  remaininglegacyatEP4 without rerunningprepared. Earlier360/444count predictions
  were wrong because EP2 halves per-rank comparisoncount.
- sf_row4 firstsbatch failedmissingworkflow beforejobcreation; syncedthenretry.
  DiagnosticCPU initialinclude-pathtypofailed7133251; fixed7133313 PASS55s.

Root header overrides olderhistoricalclosure claims. Full incrementalchronology
archived C/campaign_notes_0605.md; further designideas in C/router_pool_design.

06:08 initial routertrialcontrol: densepull rows16/12/15 dispatch6.880/6.880/
7.648us, fixedpoolpull6.912/6.976/6.944us; candidatecaptures notcompleteyet.
No large fixedstride-only penalty in firstpass; repeatsrequired.

sf_row4 Q3 six-arm7134541 COMPLETED17:42, allarms fetching for finalaudit.
One GPU slotfree; next routerQ35 if initialroutercandidate supportsbenefit,
otherwise inspectbeforeallocatingfurther. No blocked-resource claim.

06:13: first routerpush Q3candidate rows16/12 dispatch6.560/6.656us, ordinary
kernel38.443/37.753us; nownear pull (~6.88us), farbetter thanlatepush~12us, but
notyet5%overhistory. Repeatsrunning. Q35heldpendingcausalresults. Next GPU slot
used for PDL-overlap exact7134958 submitted06:13:13, standard264checks.

SFrow4 final7134541:18/90/540audited, gains2.368/1.589/0.908% vsoriginal
(rows7/6/20),82.76/78.61/85.44%slowervspull,all3historyfail.

General metadata followup drafted: router_weight_storage/router_packed_weight
5file overlays on validatedrouter_pool_push. Dedicated8B-aligned two-generation
uint64 descriptor+FP32weight table afterlocalordinals, independentofrank-slice
capacity (supportsEP1/2 withoutEP>=4restriction). Matchingcontrol allocatessame
tablebutkeepsold32bdescriptor+remoteweightread; candidatepublishesexactFP32bits
withdescriptor in bothselectors, receiverloads64b locally andeliminatesremote
scalarweightrequest. SF andactivation unchanged. get_numbytes matchesaligned
ordinal-end plus2*localexperts*maxrecv*8, final128alignment. Subsequentbufferbases
shiftidenticallyinbothsources. Fullnative/device/host rebuild needed; source
syncing, CPU plansreadyunsubmitted. NoGPU/perf.

06:22: routerQ3 7134733 COMPLETED18:10, sixarms fetching. Overlapexact7134958
COMPLETED6:23,5records/264comparisons PASS; overlapQ3timing submittednext.
Packedmetadata CPU jobs7135032(controlstorage),7135033(packedcandidate) submitted
06:19:50/51, receiversEP2/EP8 compiled; native/hostpending. Pairedexact9records/
528 ready/workflow synced, unsubmitted untilCPU PASS. No3rdGPU.

RouterQ3 final7134733:18/90/540audited. Dispatch6.608/6.528/6.400us forrows
16/12/15; gainsvsfixedpull4.176/5.991/6.977%,vsdensepull3.505/3.546/6.103%,
versushistory6.561/2.392/6.977%. All-casegatefails; materiallybetterthanlatepush
butnotthegoal. Wholekernel~0.81–1.29%betterthanfixedpull; fullchainextractnext.
PDL-overlap Q37135122 submitted06:23:22, running.

Followup router_aux_storage/router_packed_aux drafted on packedweight. Allocate
two row-major SF receivegenerations afterregisteredinputs,beforefinalL1. Size
aligned128B,derivedhidden/routes; noBLOCKM argument. Matchedcontrolsameallocation
keepsremoteSFread; candidatepushesexactSFwords inrouterbefore rankpublication,
receiverreadslocalSFthentransposestocompactL1SF. Activationstilldirectfinalpool;
SF has an intermediate table (DO NOT call SF zero-copy). Samebankproof, extra
0.469GiB/rank atH4096,E512,cap3840. FullCPU/native/hostbuilds being submitted;
noGPU/perf. This answers residualremoteSF dependencyafteractivation/weight
changes; no claimSFstagehelpsuntilmeasured.

Packedweight CPU7135033 PASS6:13; same-storagecontrol7135032 all5buildstepsPASS
at7:42 butschedulerstillRUNNING, finalstatuspending. Aux CPUjobs7135230(control),
7135231(candidate) submitted06:27:09/10. Pairedauxexact9records/528comparisons
readyunsubmitted. PDLoverlapQ37135122 running; onefreeGPUslot reservedonlyby
plan(notallocated), willusepackedweight exactonceCPUcontrolgateconfirmed.

Important fullchainresult(routerbeforePDL): producer→MegaMoE end is50.740/
51.517/51.668us(rows12/15/16), ~6.07–6.23%slowervsdensepull. MegaMoE start
shifted+~3.5us (12.89–13.21us vs9.37–9.69). So near-pull dispatch does NOT
establish lower total latency. PDL-overlap beingmeasured; no end-to-endwinclaim.
Packedweight CPUcontrol7135032 PASS7:51,candidate7135033 PASS6:13; R39/R37<48,
5local each. Paired exactsubmittednext. Aux CPU7135230/7135231 running.
New open direction: fixed-address finalpool enables source-shared→remote-global
bulk TMA (no receiver inbox/pull). Generalperwarp4KBchunk staging <=32KB/CTA,
worksforarbitraryalignedhidden viachunks, notmodel-specific. Must use full
cp.async.bulk.wait_group0 before publication (not .read-only), validateproxy/
visibility semantics andexact; notimplementedyet. SourceSMEM is intermediate,
so distinguish directnetwork path from literallynointermediateanywhere.

06:40 bulk native variants implemented: router_pool_bulk and router_pool_bulk_overlap
on validated router_pool_push / PDL-overlap respectively. Per-warp 4KB source shared
stage (<=32KB/CTA), loops arbitrary aligned hidden; lane0 issues bulk shared->remote
final L1 for assigned experts, full cp.async.bulk.wait_group 0 before reuse/publication.
No receiver inbox/copy. Both selector launch shared sizes updated; same host/receiver
and generation layout, compatible host donor seeded after sync. CPU-only native
build jobs7135683/7135684 submitted06:39:46/47; exact/performance not yet run.
Aux CPU7135230 PASS4:10 /7135231 PASS3:41, all5steps (EP2/EP8 receiver,native,
host,SASS) pass; both dispatchR37<48,5local. Paired exact ready, GPUslots currently
PDL-overlapQ37135122 and packedweight exact7135269. Packedweight fresh sweeps
converted to6arms (densepull,storagecontrol,packedcandidate,repeats), fixedseed
2011414784822190889 Q3rows14/32/12,Q35rows9/28/48; no timing filtering.

User steering06:43: 5% is minimum acceptance, NOT stopping target; pursue larger
clearly supported gains and full producer-to-consumer latency. Bulk CPU initial
compile caught host-launch kWarps out of scope; corrected to kTPB/kWarpSize in
both native-only overlays. New CPU retry dirs preserve failed logs/jobs.

PDL-overlap Q37135122 COMPLETED16:40;18captures/90phase/540ordinary and540
router-timeline samples audited, allretained. Rows15/6/10 dispatch10.240/9.792/
10.128us:54.77–58.02%slower than router-before-PDL,46.41–50.23%slower dense,
all3historyfail. Fullchain rows6/10/15 candidate50.499/50.579/51.419us vsvector
50.550/50.689/51.132, vsdense47.553/47.501/48.454. EarlyPDL shiftsMoe start
~3.4us earlierbutadds~3.4uswaiting; fullchain-0.56..+0.22%vsvector,6.12–6.48%
slowerdense. Strong causal evidence launch scheduling alone doesn't remove cost;
no push-impossibility claim. Bulk path and metadata dependencies remainopen.
Packedweight exact7135269 PASS11:58,9records528comparisons, audit saved;
freshQ3sixarm7135821 submitted06:43:15. Auxexact7135813 submitted06:42:39.
BothpendingPriority at06:44; do not submit3rdGPU. Bulk retryCPU7135819 PASS1:54,
7135840 stillrunning. User additionallyreiterated optimize metadata; weightpacking
andlocalSFtargetremote dependencies, routeallocation/publication remain further
optimization candidates, notjustpayload.

Serving applicability steering: optimizations must fit actual Qwen3/Qwen3.5 decode,
normal/skewed routing, empty/padded sources, repeated graph generations and memory
budget; random balanced decomposition alone is not serving evidence. Keep original
six-case/finalalignment requirements, add memory/throughput tradeoff assessment.
Fixedpool capacity3840 overhead+2.8/+12.6GiB remains prototype limitation; don't
promote as serving-ready. Memory-efficient chunkdirectory discussed, notimplemented.
General metadata-only dense_weight_storage/dense_packed_weight drafted from original
dense-pull snapshots/old packed-weight experiment, replaces EP>=4 rank-slice table
with dedicated aligned two-gen uint64 table (same as validated routerpackedweight),
removes old rank restriction. Original compact L1 + activation pull unchanged, no
fixed pool. Matching storage control shifts identical subsequent addresses. Extra
packed table atcap3840 =7.5MiB Q3/30MiB Q35 plus small localordinal storage. CPU
plans full EP2/EP8/native/actualhost; snapshots being prepared, unsubmitted.
Bulk initial CPU7135683/7135684 failed only launch-scope kWarps typo; retry7135840
PASS1:56,7135819 PASS1:54. Paired bulk exact ready9records528. Fresh bulk8arm plan
seed3646407820730108324 Q3rows9/14/7,Q35rows12/38/22: densepull/vector/bulk/
bulkoverlap/reverse repeats (shared controls). Comparator supports --candidate-arm
and --output; default results identical on completed overlapQ3 data. NoGPUyet.

06:55 metadata-only builds dense_weight_storage7136049/dense_packed_weight7136050
running, fullreceiverEP2/8/native/host. Pairedexact ready9records528, workflowsync.
Additional general metadata_credit experiment drafted against original densepull:
no new storage or payload change. EveryselectorCTA releases directly to eachrank;
CTA0 adds capacity-gridDim+1, others add1. Positivecredits eachsource sumcapacity,
so receiver acquires EP*capacity onlyafterallsourceCTAs release; supportsunequal
sourcegrids within existingcapacity, padded/invalidrows stillpublish. Removesdense
local lastCTAacqrelticket and finalfanoutdependency, legacybranchunchanged. Same
rank-counterbanks/reset. More remote atomics O(EP*CTAs) mayloseatlargerbatches;
no assumedwin. CPU EP2/8/native/SASS plan submittednext, compatible originaldense
hostdonor seededafterheadersync. Exact5records264 ready, noGPUyet.

Metadata_credit CPU7136155 submitted06:55:16, EP2/EP8/native/SASS (host unchanged,
originaldense donor seedconfirmed). CPUnoGPU. Pairedcompactmetadata CPU
7136050 PASS3:38; control7136049 pendinghost at4:21. GPUauxexact7135813 running12:01,
weightQ37135821 running. Main/index implementations untouched; allnewnative/layout/
receiver code in isolatedscratch overlays + remotefrozen snapshots.

06:57 auxexact7135813 PASS12:41,9records528comparisons audited. Bulk pairedexact
7136219 submitted06:56:48 afterauxslotreleased; weightQ37135821 remainsotherGPU.
AuxQ3freshseed2567057506297742172 rows32/16/1, Q35rows8/18/9 sixarmsready; next
freeGPUtiming afterweightQ3terminal. Bulk source helper chunkloop H>4096 remains
untested by standard1024/4096 exact: add general larger-hidden coverage before
promotion (not required to claim current Qwen shape benchmark correctness).
Metadatacredit freshseed953298678368895035 Q3rows7/2/21 Q35rows57/9/20 fourarms;
compactmetadata freshseed2838981218980795331 Q3rows14/8/24 Q35rows27/3/18 sixarms.
Neither submittedGPU. Both preserve original compact payload pull. Snapshotnames
for compact variants intentionallyffn_dispatch_dense_dense_weight_storage and
ffn_dispatch_dense_dense_packed_weight (double dense prefix), plansmatch.

07:02 weightQ37135821 COMPLETED18:20; sixarmsfetching. ItsGPUslotusedforauxQ3
next, bulkexact7136219 remainsotheractiveGPU. CompactstorageCPU7136049 PASS7:43,
packed7136050 PASS3:38; all5stepsandR43<48/5local. MetadatacreditCPU7136155 PASS2:11,
all4steps/R43<48/5local. Both exactplansready awaitingGPUslots. CPU-only gather4
ISAprobe submitting(group1/group2 inonecubin), nofunctional/perfclaim.

WeightQ37135821 final18/90/540audited plus540timelinesamples. Rows14/32/12
packeddispatch6.336/6.480/6.160us; gainsvsstoragepush0/0.246/4.467%,vsdense
7.042/6.682/8.551%,vshistory7.477/2.644/7.895%. All-casegatefails(row32history;
metadataincrementalso<5). Fullchain rows12/14/32 packed50.866/50.770/53.021us,
control50.857/51.097/52.947,dense47.777/47.953/49.876. ~5.88–6.47%slowerdense;
weightreadremovalnotmainfullchainbottleneck. Q35heldforbettercandidate.
AuxQ37136351 submitted07:02:45; bulkexact7136219 otherGPU. Monitor8151aux,
1418bulk (300s). Gather4CPU7136335 PASS1:03, bothCTAgroup1/2 assembleSM100f;
CPU/SASS artifactfetch84386pendingpoll. Nofunctionaltest/gainclaimed.
Serving-orientedgather direction refined: allocate max(originalcompactL1capacity,
2*EP*capacity) rows, use2source-indexedbanks ONLYdensepath, originalcompactrow
mappingforlegacy. With Qwentopk8/10 existingcompactL1alreadylargeenough: expected
noextragrossactivationallocation, unlikefixedperexpertpool. Denseconsumergather4
references source descriptors; nativefanoutonceperuniquedestination, notexpert.
KeepcompactSF/L2/arrival. HostL1TMAboxheight1fordensegather,originalLOAD_BLOCK_M
forlegacy; sameFP8/BF16 activationtypes. 2SMbarriercompletion/bytecount and per4row
OOBpadding critical. Implementnext; no sourcecodeyet. Underlyingconstructor
num_maxpooltokens mustbeverified to substantiate no-growth claim pergeometry.

07:14 compact_pool_pull/compact_pool_push five-file overlays implemented and synced,
fullCPU buildsubmitting. Candidate source-indexed2banks share original L1allocation
max(compactPool,2*EP*capacity); 12Qwengeometries capacity1..3840 shownoaddedL1bytes.
Removed now-unusedlocalordinaltable/stores fromBOTHcompactvariants: allocation
matches originaldense forQwen shapes, no150KBextra. Nativevectorfanoutdeduplicates
bydestination, helpergetsroute_rank; no perexpertactivationduplicate. Consumer
loads4sourceindices/lane beforeKloop, issuesgather4.2CTA intoshared atlane*4*BK,
barrierpointer maskedtoleader likeexisting2SMhelper; invalidrows useTMAOOB. SF/L2
remaincompact. DensehostTMAboxheight1; legacyLOAD_BLOCK_M. Matchedpullcontrolsame
layout, payloadhelperdisabled, normalcontiguousTMA. NeedTMAA Rmax<40 audit beforeGPU,
notjustdispatch48; CPUplansreceiverEP2/8/native/actualhost/SASS. NoGPUyet.
Mode-transitionproof: unpreparedlegacy cannotrankReady (staticchecks); its global
post-MMA cleanupbarrier ordersoldcompactconsumer before nextdensepush. Dense->legacy
fullstartbarrier orderspriorgenerationbeforecompactreuse. Lamportenvchanges onan
existingbuffer werealreadyunsupported byenvironment-dependentconstructorlayout;
no new mixed-envsupportclaim. Two-bankproof holdsforrepeateddensegraphserving.
Bulkexact7136219 PASS12:35,9records528. BulkQ3eightarm7136661 submitted07:12:43;
auxQ37136351 remainsotherGPU. Bothoutstanding; don'tsubmit3rd. Candidatecompact
firstdraftscript stoppedhostreplacementassert becauseL1/L2bothmatch; completedby
changingL1only, L2descriptorunchanged. Main/indexcode untouched.

Compact CPU7136704 failed46s: dead kDirectPoolPush=false branch stillnamed removed
get_dense_send_ordinal_ptr member. Removed obsolete late-sender branch inbothcompact
overlays (notanactualpath); also consumerdescriptorbankEP1 usesoriginalslice0rule.
CPU retrydirsready, candidateheadersresyncing. ControlCPU7136702 stillrunning2:17
witholdheaders; don'tmutate its snapshotuntilterminal. Newpairedcompactexact9records/
528ready. Workflowsyncing. CPU retryfull actualhost remainsneeded.
NextfreeGPU shouldgive metadata_credit exact (readyCPU PASS) so newpayloadtrials
cannotstarve metadata investigation; compact/dense-packed exact alsoawaitslots.

Compact candidateCPUretry7136815 submitted07:18:11; control7136702 FAILED3:54 same
removed-memberparseerror, snapshotupdatedafterterminal thencontrolretry submitting.
SASSreviewhelper now exposes inspect_register_regions and inspect_reachable_registers:
followdirect/annotatedindirectbranches untilEXIT/registerchange, reportunresolvedcalls.
Originalinspect outputsunchangedondensepacked/creditSASS. BaselineTMAroles R23<40,
0unresolved; dispatchR43<48 but4unresolvedexternal transfers requiremanualreview.
Do notcalllinearspananalysis CFGproof. MustauditnewgatherTMArolesbeforeGPUexact.

Compact controlCPUretry7136888 submitted07:20:28. Compactfreshsixarmplan seed
2796530709587003306 Q3rows15/3/10,Q35rows51/4/3 fixedbeforetiming; noGPUyet.
SASSreachablehelper fixed@!PTpredicateparsing, baselineTMA R23<40/0unresolved;
dispatchonly2CALL.ABS.NOINC unresolved (existingassert/library paths), R43<48.
Bothlinear/reachablehelpers reusable; unchangedoriginaldispatchinspectionoutputs
verifiedonactualdensepackedSASS. No fabricatedregister-safetyclaimfornewgatheryet.

07:22 auxQ37136351 COMPLETED18:08, sixarmsfetching. Freedslot usedmetadata_credit
exact next (5records264), notthirdGPU; bulk8arm7136661 remainsrunning. Compact
candidateCPUretry7136815 running3:46/controlretry7136888 running1:29.

AuxQ3 fullchain60samples/case/group(allretained): rows1/16/32 candidate38.351/
50.182/51.756us, control37.717/53.959/53.162,dense34.931/48.270/50.065.
Gainsvsstoragecontrol-1.681/+7.000/+2.644%, vsdense-9.791/-3.961/-3.379%.
DispatchgatesPASS doesnotestablishservinglatencygain. Q35stillnext tocheckrequired
modelcoverage; fixedpool result is diagnostic/candidatebaseline, notpromotion.
Sourcefanoutdedup + localaux incompactsourcepool is promisingcompositionafter
compactexact: source-indexedframe[activation,SF,topkweights], pitchaligned16,
TMAgatherdescriptorusesframepitch and 2*EP*capacity rows. ReceiverSF/weight reads
localframe viaexistingdescriptor rank/token/topk; no dedicatedpacked-weighttable
needed andonlyoneframe peruniquedestination. FitswithinoriginalQwencompactL1bytes
(expectedcapacitymustbecalculated), legacy usesoriginalcontiguouspool/descriptor.
Caninitiallyuseactivationvectorcopy+smallcoalescedSF/weighttailstores, sameSYSrank
publication. This composes measuredSFbenefit with noextragrossQwenmemory andlower
fanout; notimplemented yet, do notconflatewith currentcompact_pool_push activationonly.

Compact candidateCPUretry7136815 PASS5:47. EP2/8,native,actualhost,SASS all5steps.
ReachableTMA/MMAregions R23<40,0unresolved,0local; gather4present4instructions.
DispatchR41<48,5local,2existingexternalCALLs unresolved (sameasbaseline). Thusno
registerbudgetincrease needed. Control7136888 stillrunning7:14 atlastcheck.
Metadataexact7136914 running5:13; nextGPUauxQ35 onceitfinishes/audited.

07:30 metadataexact7136914 COMPLETED7:11; collector/auditnext. SlotusedauxQ35
(requiredQ35EP16 coverageafterall3Q3phasegatesPASS); bulkQ37136661 stillotherGPU.
CompactmatchedcontrolCPUretry7136888 PASS7:17 all5steps. ItsdispatchR43<48/5local,
TMA/MMAreachableR23<40,0unresolved/0local; candidateR41dispatch/R23TMA. Bothcompact
fullhost/native cachesprepared, pairedexact528 nowready. Correctnessharness includes
normalrouting (3phases,zero/emptydest), emptyranks/prefix andnoncontiguouspadding,
alternateinputs andgraphrepeats evenwhen timedcaseperfect. Fullmodelservingalignment
andnormal-routing performance stillpending; do notclaimtiminggeneralization.

07:36 auxQ35submission transientSSH/sync delay resolvedwithoutresubmission;
job7137352 submitted07:36:26 (not07:30). Metadataexact7136914 collectorcompleted,
5records264comparisons PASS. GPUbulkQ37136661 (22minatlastquery) andauxQ35 only.
Compactframe_storage/compact_frame_push5file overlaysdrafted: framepitchalign128
ofH+H/32+topk*4; bothgather4 sourceframes samepitch; candidatepushesSF/exactweights
intoownframetail, receiverreadsSF/weightlocally; controlkeepsremoteauxreads.
HostdenseTMAactualsourceRowsextent+framepitch; legacycontiguousview unchanged.
Allocate max(originalcompactL1,ceil(2EPcap*framepitch/H)); 12Qwengeometriesprove
noaddedL1/workspacebytes, inclpadding/aux. Same2bankproof. Removedextraordinalalready.
Sourceframe initial16pitch changedbeforeexecution to128forconsistentTMA128alignment;
no strideperformanceclaim. CPUplansfullreceiver/native/host/SASS prepared, snapshots
copying, notsubmitted. Compactgatherbase exactstillawaitGPUslot, nofunctionalclaim.

07:40 bulkQ37136661 COMPLETED24:49 eightarmsfetching; includesphaseoutlier106us
inlastpullrow7, retainall. Its freedGPUslot usedcompact_pool_push pairedexactnext,
auxQ357137352 remainsotherGPU. FrameCPUstorage/candidate submittingfull5steps,
workflow syncing. Don'tprematurelyclaimbulkadvantage fromoutlier; comparephase
medians andwholechainmean+median/distribution whenavailable, nofiltering.

BulkQ3 final7136661 24captures120phase720ordinary audited. Rows9/14/7 bulk-beforePDL
6.464/6.464/5.952us, gainsvsvector-1/+7.763/+4.370%, vshistory+0.980/+5.607/
+11.429%; all-casegatefails. Bulk-overlap9.200/9.232/8.960us,31.74–43.96%slower
vectorphase; requiresfullchaincomparison toassesscopycost, collecting8timelines.
Candidatebulk noall3phasegate, noQ35forstandaloneyet. AuxQ35running.
Compactexact7137493 submitted07:39:55; frameCPU7137494(storage)07:39:57 and
7137506(candidate)07:40:17 fullbuilds underway. Main/index staysuntouched.

07:51 bulkQ3 wholechain all60samples/case/group: rows7/9/14 dense47.678/47.510/
47.781us, vector50.278/50.394/50.834, bulk49.720/49.594/49.941, bulk-overlap50.252/
49.960/50.447. Bulk saves0.558/0.800/0.893usvsvector; launch0.756–0.878usearlier,
but4.28–4.52%slowdense. EarlybulkPDL adds0.366–0.532uschainvsbulkbeforePDL. Medians
agree; alloutliersretained. pipeline_distribution_comparison.json +8timelines saved.
Frame fullCPU/host/SASS gates PASS7137494/7137506; candidateTMAhighestR25<40,
controlR23<40, bothdispatchR41<48. Freshsixarmplanspreparedseed4077014237227445728.

Serving integration audit: MegaMoEAfdAdapter.forward uses the candidate's exact existing interface: quant_out=(buffer.x,buffer.x_sf), route_prepare_args(rank_ready=True), rank-ready/route-ready MegaMoE flags, per-lane persistent buffers, and num_token_non_padded maps. No additional host launch required by compact/frame paths. Same-bank graph lifetime depends on serialized reuse per lane; no claim of cross-stream concurrent reuse of one buffer. Correctness harness runs normal/zero routing, empty destinations, prefix-empty source ranks and noncontiguous padding plus graph replays; timing samples still balanced. Normal-route timing and full-model alignment remain required before serving promotion. Zero physical rows select unprepared path in adapter (existing behavior); current empty-rank coverage uses padded nonzero graph buckets, not heterogeneous physical zero/nonzero source shapes.

07:54 compact_pool pairedexact7137493 COMPLETED13:16,9records528comparisons allPASS (60each6prepared arms,84each2legacy variants). Gather4 functional path and graph/padding/normal-route fixtures passed. NextGPU compact_frame pairedexact, replacing separate compact-base Q3 timing priority: the frame's same-stride storage control measures activation-only gather plus unchanged remoteaux, so six-arm frame timing isolates aux benefit while comparing direct path to originaldense. Basecompact freshplans preserved/unrun, no reroll. Metadata_credit Q3 stillnext afterauxQ35.

07:58 router_packed_aux Q357137352 COMPLETED19:15,18captures90phase540ordinary audited. Rows8/18/9 dispatch5.408/5.616/5.344us, gains vshistory19.139/22.687/20.476%, same-SFstoragecontrol11.979/11.139/14.578%, originaldense17.359/20.408/18.137%; all3gates andbothrepeatsPASS. TogetherQ3 this isfirst qualifying6-case phase result. No servingpromotion: fixedpoolmemory cost andQ3fullchainregression; Q35fullchaincollecting. Pursuecompactcomposition before costlyfullsweeps/alignment.
Compactframepairedexact7137902 submitted07:54:31; metadatacreditQ37137990 submitted07:57:17. Thesearethe2GPUjobs. Newcompact_frame_bulk native-onlycompositionCPU7137992 submitted07:57:36; source/receiver/hostlayout identicalframepush, ownnativecache. Bulk chunk6016B=(48KiB-1KiBstaticreserve)/8warps; vectoractivation+coalescedaux tosourceSMEM, bulkdirectunique-destinationfinalframe, fullwritecompletion beforepublication; no receiverrecopy. Exact7records324prepared, includingnewH8192genericchunkloopreference; awaitsframeexact andCPU. Unexecuted/perfunproven.

08:00 auxQ35 fullchain all60samples/case/group: rows8/9/18 dense59.415/59.684/
60.216us, storagecontrol68.115/62.956/63.996, packedaux61.785/62.207/63.451.
Candidate3.989/4.227/5.372%slowdense; mediansagree. Controlrow8outlierretained
(mean68.115 median64.432). SixphasegatesPASS doesnotjustifyservinglatencyclaim.
Bulkframefreshsixarmseed3406116041880725457 Q3rows4/21/3,Q35rows16/2/19, plansready;
matchedvectorframecontrol +originaldense. Actualhostidentical, donorcopyseedingafter
sourcesync; firsthelperpath wasabsent inarchivesnapshot, syncingreusablehelper to
artifactworkflow. CPU7137992 running. No additionalGPUsubmission.

08:04 generalityreview caught compact_frame_bulk initialsharedbudget assumed8warps, but generic kTokensPerBlock=16 (CTA256threads=8warps). NoGPUexecuted. Corrected chunk=min(framebytes,align_down128((48KiB-1KiB)/actualblockwarps)), hostdynamicallocation usesidenticalformula. ActualQwen4352Bframe allocates34816B/CTA; generic16warpscap2944B/warp (47104Btotal). 256CPU geometrychecksPASS H1024/4096/8192/16384,topk2/8/10/32,warps1..16; shared+4Bstatic<=48KiB, allchunks128aligned. SourceupdatedafterCPU7137992terminal, nativeCPUretry submitting. Exact H8192generic rows3/24 replaces3/6 toexercise16warps andmultichunks. Thisalsoexposesolderrouter_pool_bulk's fixed4096*16 potentialgenericlarge-rowlimit; its validatedsmallfixture3/6 didnotcoverthat, don'tpromoteoldbulk asgeneral.

08:07 bulkframeCPUretry7138057 PASS1:54; correctedgenericsharedbudgetnativecacheprepared. Efficiencydecisionbeforeanyframe timing: combine vector+bulkframe inone8armtrial (densepull / frame_storage activation-only / vectorframe / bulkframe / reverse), retainoriginalframefreshseed4077014237227445728 andallcases. Newbulkstandalonefreshplans preservedunrun. Saves4armspermodel versus2sixarmtrials and isolatesauxpluscopymechanismonidenticalcases. Requiresbulkexact324first. ThusnextfreeGPUafterframeexact ->bulkexact; aftercreditQ3 ->dense-packedmetadataexact ifready, thencombinedframe8armQ3. No reroll, no candidate timing happenedbeforeselection/orderchange.

08:08 compact_frame pairedexact7137902 COMPLETED12:28,9records528saved comparisons PASS. Bothgatherframe-stride storagecontrol andlocalSF/FP32weightcandidate exact acrossgeneric/CTA Qwenfixtures, normal/zero/paddedranks andlegacygraph paths. NoextraQwenallocation provenbeforetests; nowfunctional proof too. FreedGPUslot usedcorrectedcompact_frame_bulk exact7records324 (H8192rows3/24), creditQ3stillotherGPU. Combined8armframeQ3awaitsbulkexact.

08:11 metadata_credit Q37137990 COMPLETED12:07,12captures60phase360ordinary audited. Seed953298678368895035 rows7/2/21 dispatch6.576/6.448/6.704us; historygains2.143/4.502/0.711%, matched1.202/-5.497/3.678%; all3gatesFAIL. Wholekernelchanges-2.565/+0.103/-0.377%; fullchaincollecting. DirectperCTAcredits do notprovidequalifyingmetadatawin, don'tpromoteorcompose. Q35held. Freedslotuseddense_packed_weight pairedexact528, CPU/host/SASS PASSalready; bulkframeexactstillotherGPU.

Metadata_credit wholechain all60samples/case/group: ep8_rows2: control=47.114 candidate=47.082us gain=0.067%; ep8_rows7: control=47.564 candidate=48.519us gain=-2.008%; ep8_rows21: control=49.633 candidate=49.778us gain=-0.292%. Allrecordsretained, timelineanddistributionJSONsaved.

Normal-routing serving follow-up workflow identified: existing megamoe_source_abba.sbatch + MEGAMOE_SMOKE_FIXED_PERFECT_ONLY=0 / PERFECT_ROUTING=0 times ordinary fused random-input routing through _exact_qwen_case, with graphreuse andsource-reference checks. This is syntheticnormalrouting, not realmodelrouting; fullmodelalignment remainsseparate. Prefer this existingworkflow once winningcompactcandidate isselected; no newstandaloneharnessneeded. Currentnormal-route exactchecks remaincorrectness-only, nottimingproof.

08:15 bulkframeexact7138156 5/7primaryrecordspassed: candidate_small/Q3/Q35 pluscontrol_large/candidate_large H8192rows3/24. Multichunk+16warpgeneric functional testPASS, legacy2recordsremain. Packedmetadataexact7138178 3/9recordssofar. No GPU timing ofcompactframe yet; combined8armQ3stillawaitfinalbulkexact.

08:18 bulkframeexact7138156 COMPLETED9:45,7records324comparisons PASS, inclH8192rows3/24 andlegacy. Combined8armcompactframeQ3 now submitting onoriginalseed4077014237227445728 rows21/20/6; fullCPUs/host/exactgatesallPASS. Packedmetadataexact7138178 remainsotherGPU. No newsourcechangesinrunningvariants.

08:21 concretegatherworkreduction identified: compactconsumer alwaysissues48gather4 requests/Kblock (96rows/CTA), but scheduler.get_valid_m<true>() aligns actualM onlyto16 andUMMA instrdesc uses thatsmallerextent. Newcompact_frame_gather receiver-only overlayonbulkframe limitsperCTAissuedrows to alignedM/2 and leaderexpectedAbytes toalignedM*BK*sizeof(A). AllrowsinsideUMMAextentstillgather/OOBzero-filled; onlyunconsumedrows skipped. SF/L2/source/publication/hostTMA unchanged. <=16validrows ->4requests vs48;192valid-MCPUgeometry checksPASS. CPUplanEP2/8/native/SASS (samehostdonor aftersync) submitting, exact7records324 inclH8192ready; no GPUtestorperfclaimyet. Currentframe8armtiming remainsfrozen, notmodified.

Gather CPU7138359 submitted08:21:24; ownnativecache builds andhostdonorcopied aftersourcesync (receiver-onlychange, hostallocation/TMAdescriptor/layout/nativeABIidenticalbulk). Freshsixarmseed2176874827476714375 Q3rows20/8/6,Q35rows33/8/13; densepull +bulkframecontrol +boundedgather. NoGPUyet. GatherM==schedulerUMMAalignedM, so skippedrows areunconsumed, notjustinvalidinputrows; zero-padding insideactualMMAshape ispreserved. Sourcecopies andmemorycapacity unchanged.

08:24 densepackedmetadata7138178 COMPLETED12:30,9records528allPASS. BoundedgatherCPU7138359 COMPLETED2:26 EP2/EP8/native/SASS all4steps; dispatchR41<48, TMAloadR26<40 (otherTMA/MMA R25),0unresolved TMA transfers/0local; same2externaldispatchcallsbaseline. Hostdonoralreadyseeded. NextfreedGPUslot usedgatherexact324; frameQ38arm7138284 stillrunning. DensepackedQ3timing ready, afterthispair freeslot so metadatafollowup remainsmeasured.

Servingfanout audit: sweepwrapper sets MEGAMOE_SMOKE_BALANCE_RANKS=1; fixed_ids/expert_map remap(id%EP)*localExperts+id/EP. ThusbalancedtimedQ3top8/EP8 sends8uniquedestinations/token, Q35top10/EP16 sends10—not grouped1–2destinations. Exhaustiverouting-cycleCPUcheckPASS. Compactdedupdoesnotartificiallyreducefanoutinthese timingcases; realroutes mayshare destinationsandbenefit, butskew/loadbalance stillneedsnormal-route andmodelmeasurements. Savedcompact_frame_push/balanced_fanout_audit.json.

08:35 compact_frame_preload native-onlyfollowup drafted onboundedgather: CTAwarps1..7prestageactivation/SF/paddingwhilewarp0selects+publishesroutedescriptors andwritesexactFP32weightstoSMEMtail. ExistingCTAbarrierafterproxysharedfence makesoneframeready; warpsbulkpushuniquedestinations fromsharedframe. QwenCTAshared4352Bvs34816B, noextrabarrierinonechunkcase; lastchunkcompletion joinsatexistingcallerCTAbarrier beforePDL/publication. Genericselectorunchanged. Forframes>48KiB-1KiB, subsequentchunkscooperativelystaged, withfullwritecompletion+CTAbarrierbeforereuse. No newpublicationprotocol orcounterchanges. CPUplanready, snapshotcopying; basegatherexactstillrunning, noGPUpreloadyet. NeedCTAwideframe exacttocovercooperativechunkloopbeforepromotion.

BoundedgatherQ37138899 submitted08:36:23; preloadnativeCPU7138900 same08:36:23. Preloadexact9records384ready: standardcandidate180+genericH8192extra60+CTA H65536/intermediate128extra60+legacy84. Widecontrolrecordsnewreference, rows1/3, reducedwarmup1/iters2/graph2; validatescooperativechunkloop beyond48KiBsourceSMEMbudget (frame67712B, chunks48128+19584). Standardtargetgraphs20retained. H65536testisfunctionalonly, notperformancecaseormodelspecialization. HostsupportsH/I divisible128. NoGPUpreloadyet; boundedgatherandoriginalframeQ3 are2activeGPUjobs.

08:40 preloadCPU7138900 PASS2:02/nativebuild marker; ownhostdonorseededaftersync. Freshsixarmseed2450173393863508490 Q3rows6/24/4,Q35rows10/2/11 ready, matchedboundedgathercontrol +originaldense. Exact9records384stillunsubmitted (standard+genericlarge+cooperativeCTAwide+legacy). Originalframe8armQ37138284 ~20minrunning, boundedgatherQ37138899 ~3minrunning. NextfreeGPUshouldrunreadydensepackedmetadataQ3 so additionalpayloadworkdoesnotstarvemetadata.

08:45 compactframeQ37138284 COMPLETED24:30,24captures120phase720ordinary audited. Seed4077014237227445728 rows21/20/6: vectorframe dispatch5.792/5.744/5.264us, history+14.218/+14.115/+24.194%, same-stridestoragecontrol+11.707/+15.330/+16.497%, dense+14.014/+15.728/+17.750%; all3gatesandbothrepeatsPASS. Bulkframe5.840/5.728/5.600us, history+13.507/+14.354/+19.355%, storage+10.976/+15.566/+11.168%, dense+13.302/+15.962/+12.500%; all3gatesandbothrepeatsPASStoo. TheseachieveQ3phasegateswithnoaddedQwenallocation. Fullchaincollecting; ordinarykernelrows21/20stilllongerdense, boundedgather targetsconsumerloadwork. FreedslotdensepackedQ37139030 submitted08:44:37; boundedgatherQ37138899 otherGPU. Q35originalframeheldpendingbettergatherresult ratherthanpay8armsforpotentiallyinferiorconsumer.

CompactframeQ3 wholechain all60samples/case/group (720totaltimelines): rows6/20/21 dense47.235/49.536/49.733us, activation-onlyframe50.573/55.934/56.091, vectorframe49.987/55.493/55.462, bulkframe50.722/55.871/55.948. Vector5.827/12.025/11.521%slowdense; bulk7.382/12.790/12.497%slowdense. Mediansagree. Bulklaunch13.622/14.347/14.241us vsvector12.974/13.760/13.605 anddense8.865/9.754/9.658. Unlikeearlieractivation-onlyfixedpoolbulk, packingthisfullframewithperwarpstageadds0.6uslaunchdelay. Both3phasegatesPASS but notservinglatencywin. Boundedgatheraddressesconsumerwork; sourcepreload addressesstage/serialization.
08:51 finalunsubmittedpreload revisedafterCPU7138900terminal; initialserial-metadataprototype archivedpre_overlap_gate_topk_fused.cu (neverGPUtested). Warp0 nowpublishesdensemetadata AFTERfirstCTAbarrier, concurrentwith7copywarps' bulkpayload; source-indexedaddressdoesnotdependonordinalallocation. ExistinglastCTAbarrier joinsmetadata+completedpayload beforeunchangedPDL/ticket/SYSpublication. Cooperativechunkreuse uses barrier.cta.sync 1,224 WITHOUT.aligned (conditionalcopygroup), warp0 usesCTAbarrier0. PTX9.4primarydocs verifiedcountmultiplewarp, conditional.alignedrestriction, and memoryvisibility amongparticipants: https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar. Fullbulkdestinationwait +proxyfences retained. SourceonlychangedbeforeanyGPU/perf; freshseed unchanged. NativeCPUretry submitting; owncompatiblehostcache remainsvalid. Exact384inclH65536 nowtestsmetadata/payloadoverlap+copygroupbarrier. No newGPUjob.

09:00 boundedgatherQ37138899 COMPLETED18:39, sixarmscollecting. Finaloverlap-preloadCPUretry7139255 COMPLETED3:42, nativebuildPASS; initialprototypeCPU7138900superseded. FreedGPUslotusedpreloadexact9records384 next, existinghostdonor/receivercachelayoutcompatible; metadataQ37139030 ~16minrunningotherGPU. PreloadfinalsourcefrozenbeforeGPU, archivedpre-overlapprototype remainsunexecuted.

BoundedgatherQ3 final7138899:18captures90phase540ordinary+540timelines audited. Rows20/8/6 dispatch5.792/5.616/5.520us, history+13.397/+17.606/+20.507%, matchedoriginaldense+14.824/+19.863/+10.156%, bothrepeatsPASS userhistory+dense5%target. Causalbulkframe dispatchdelta+1.093/-0.286/-0.583%, so NOTanother5%dispatchstep; originaldispatch_comparison retainsFAILcausalgates, target_comparison.json explicitlyseparatesuserbaselinegates. Fullchainrows6/8/20 candidate50.534/50.906/52.726us, bulkcontrol50.797/54.171/55.941,dense47.057/48.991/49.639: +0.518/+6.027/+5.747%vsbulk, -7.390/-3.907/-6.218%vsdense. Bulkrow8outlierretained (median52.864 vsmean54.171); candidatemedians50.448/50.752/52.704. Consumerworkreductionhelpslargerrowkernel, source delaystilllimitsserving. Finalpreloadmetadata/payloadoverlapexact7139364 submitted09:01:33; CPUretry7139255PASS3:42.

09:06 densepackedmetadataQ37139030 COMPLETED18:19,18/90/540audited. Seed2838981218980795331 rows14/8/24 dispatch6.672/6.896/6.816us, history+2.570/-1.174/+13.061%, same-allocationcontrol+5.442/+0.462/-1.914%, dense+3.695/-2.864/-2.651%; all3gatesFAIL. Densecontrols themselvesbeatoldhistoryrow24, so don'tclaimits13%aspackinggain. Q35held; fullchaincollecting. NextfreeGPUboundedgatherQ35 submitting(originalfreshselection33/8/13), tocheckEP16/modelcoveragewhilepreloadexactruns. No newpublication/weightpackingpromotion.

09:08 boundedgatherQ357139418 submitted09:06:58 (fresh33/8/13), alongsidepreloadexact7139364. Densepackedmetadatafullchainall60/case/group: rows8/14/24 candidate47.479/47.852/49.619us, dense47.887/47.998/49.626, storage48.034/48.100/49.615; gainsvsdense0.851/0.306/0.014%, storage1.155/0.517/-0.008%. Smallmeasuredwholechainimprovementdoesnotmeetdispatchgate; no promotion/Q35. All540timeline records retained.

09:18UTC preload wide failure isolated to preexisting weight SF transform: contiguous kernel requests block_mn48*sf_k2048*4=393216B in smxx_layout.hpp; handle.hpp178 cuFuncSetAttribute rejects it. No candidate dispatch failure. H32768 needs196608B; recovery uses this valid conversion extent and general16KiB cooperative tile (33920B frame→3chunks) to test subgroup barrier reuse. All Qwen source frames remain4352B/onechunk; generic chunkbudget unchanged. Partial exact audit records240 saved comparisons. No silent runtime fallback added.

09:20UTC preload16KiB CPU7139551 PASS2:06; recovery submitted after DENSE_ROUTER_BUILD_OK. Token-major stride isolated on boundedgather: sole get_source_l1_row mapping (bank*capacity+token)*EP+rank, unchangedallocation/pitch/format; allnative/receiver/gatheraccessorscentralized. 60bankgeometriesbijective/bounded. CPU7139571running, hostdonorseedpending, exact5records264 andfreshsixarmsplansprepared. Source copy completed and actualcentralhelperread confirmedtoken-majorbeforeacceptingCPU; initialworkflowrsyncmissingparentrepaired. Notiming/stridecausalityclaim.

Serving measurement note: rank0 Nsight router_timeline measures producerstart→MegaMoEend and is NOT distributedrequestlatency. Existing smoke _time_graph(run_stage) uses CUDAevents aroundcaptured producer+MegaMoE and reducesMAX acrossEP ranks; use graph_batch20 onnormalrouting forservingstress, label syntheticrandom separatelyfromactualmodelalignment. Exacttest can time normalrouting with FIXED_PERFECT_ONLY=0, PERFECT_ROUTING=0 (no kernel/harness change). Existingzero-hidden/padding correctnesscovershotrankbehavior butnotitsperformance. Full-modelalignment stillrequiredafterselection.

09:24UTC recovery7139575 FAILED1:40 beforefirstrecord: control bufferlayout Data(intermediate/32) requires16B alignment, I128gave4B. CorrectedfixtureI512 (existingAPIactuallyrequiresI%512 despitelater%128assert). H32768scaleconvert196608B, sourceframe33920B=16384+16384+1152, H/I SFrows1024/16B aligned. validation_recovery_aligned four-recordplanprepared; no candidate kernel failure and no runtimefallback.

Ordinary-routing serving-chain plans prepared at compact_frame_preload/normal_qwen3 andnormal_qwen35 (NOTsubmitted): originaldense ABBA, freshphaseselection3rows, perfectroutingoff, 20calls/graph100timedcalls, MAX-rank CUDAevents, exactcompare3armsagainstfirstcontrol. Candidatechoicepending; syntheticrandomnotactualmodelrouting, nofullmodelthroughputclaim.

09:30UTC boundedgatherQ35 7139418 COMPLETED19:00, all6arms3cases/15phase/90ordinaryaudited =18captures90phase540ordinary plus540rank0timelines. Rows33/8/13 dispatch5.728/5.840/6.128us, history20.444/12.679/10.930%, dense16.159/14.720/14.509%; all3usergatesinclrepeatsPASS, compact_frame_gather/six_case_gate.json savesALL6targetPASS. Causalbulkphase+0.279/-4.885/-5.220%, originalcausalFAILretained. Fullchain rows8/13/33 candidate65.713/66.621/68.210, dense59.127/59.510/62.661, bulk65.558/66.699/72.911us: candidate11.139/11.949/8.855%slowdense; bulk-relative-.237/+.117/+6.448%. No serving-latencywin. Tokenstrideexact7139805 submitted09:27:52 inQ35freedslot; preloadalignedrecovery7139659 remainsrunningwithfirst3recordsPASS.

09:31UTC preloadalignedrecovery7139659 COMPLETED5:36 PASS4records144savedcomparisons; wideH32768/I5123chunk andlegacy. Combined384PASS (original240standard+genericlarge remainvalidbecauseQwenonechunk4352Bandgenericpathunchangedby16KiBcooperativelimit; changedwidepath+legacytestedfinalcode). combined_exact_audit.json recordsbothfixturefailuresandactualsourcechange. Q3freshsixarmtimingsubmittedaftergate.

09:34UTC serving-priority scheduling decision: after tokenstrideexact frees a slot, run preload/normal_qwen35 ABBA (384exactPASS permitsstress; balancedQ3perf7139860stillrunning). OrdinaryrandomroutingtestsfanoutdedupandMAX-rankgraphlatency onEP16, independentofbalancedphaseoutcome; strideperformanceremainsreadybutheld. This yieldsrealisticrouting evidencebeforepayingfullsweeps/alignment.

Fullmodelalignment plans prepared (NOTsubmitted) for preload/qwen3_alignment andqwen35_alignment fromvalidatedSep13templates:48prompts816tokensQ3,168prompts2856tokensQ35; actual128Knormalrouting,multiMB2graphsoverlap, pinnedrefs. HostdonorupdatedtoactualcompactpreloadABI/alloccache, sourceupdated, taskpathsnew, guardregexallfastafdjobs. BothalignmenttopologiesFFNEP8 (Q316GPUs,Q3532GPUs); phaseQ35EP16 coverage separate. Mustdryrunlauncherandensurefinalselectionbeforelaunch.

Offline actual ordinary-Q35 inputrouting audit uses48small result snapshots acrossEP16, noGPU work. Newgeneral summarize_megamoe_routing.py validatedactualtensors: rows2/10/11 fanout7.906/7.694/7.676destinations/token (vsbalanced10), logicaldedup20.9375/23.0625/23.2386%; max/meanrankload1.30/1.11/1.091. Counts matchEP*rows*top10andrankloadsum; do NOTequatetheselogicalcountswithmeasuredNVLinkbytes. FullABBAwaitingfourthrecordbeforeperformanceconclusion.

09:42UTC preload ordinary-Q35 7139920 COMPLETED5:50, fourrecords3cases,1056savedexactcomparisonsPASS; everyrecordscope router_and_megamoe,20calls/graph100timedcalls, ranksMAXevents. Rows2/10/11 candidate45.3496/61.8667/62.3054us vsdense45.6618/61.7419/61.3128: +0.684/-0.202/-1.619%gain. Tinyrowcontrol drift2.166% exceedsapparentgain; row11candidate61.723/62.888spread retained. No reliablewholechainwin; nearparityisqualifiedonlyfor thiscandidate/routing/measurement, notcausalcomparisonagainstearlierbalancedrank0timings. normalQ3submittedinfreedslot tocoverbothmodels; strideperformancestillheld.

OrdinaryQ3actualrouting24snapshotsEP8 audit: rows4/6/24 fanout5.25/5.3125/5.36979 ranks/token (balanced8), logicaldedup34.375/33.59375/32.8776%, max/meanrankload1.156/1.146/1.130. CountsvalidatedagainstEP*rows*top8. Usesinitialnormal-inputresult.0 savedtensor, matching restoredtimedinput; noGPUrerun.

09:50UTC ordinaryQ3 7139982 COMPLETED5:34,528savedcomparisonsPASS, fourrecords3cases20calls/graph100timedcalls. Rows4/6/24 candidate48.13296/48.21632/51.76528us vsdense48.14352/48.39488/51.76592: +.02193/+.36896/+.00124%gain, controlsdrift1.65/2.94/.73%; effectivelyparity, no reliablelatencygain. TogethernormalQ35supportsnearparityfortestedsyntheticrouting, notwholemodelwin. BalancedpreloadQ3 7139860 COMPLETED17:52, collectingsixarms. TokenstrideQ3submittedafterordinaryQ3free; otherGPUslotavailableforpreloadQ35ifbalancedQ3qualifies.

09:52UTC preloadQ3 7139860 COMPLETED17:52,18captures90phase540ordinary+540timelinesaudited. Rows6/24/4 dispatch5.472/5.680/5.552us; history21.198/27.551/15.777%, dense17.191/15.072/17.577%; ALL3usergatesinclrepeatsPASS. Causalboundedgatherphase0/+.838/-2.967% (retainedcausalFAIL, noanother5%claim). Fullchainrows4/6/24 candidate46.7531/46.8960/49.2032us, dense46.9701/47.5573/49.9045, control51.1227/51.3925/53.0464: +.462/+1.391/+1.405%vsdense and+8.547/+8.749/+7.245%vsmatchedpreviouspush. Candidate launch9.5728/9.9765/10.0576 vscontrol13.7739/14.1339/14.2171 removes4.201/4.157/4.159us; originaldense9.3792/9.7019/9.7237. This isstrongcausalevidenceforthecombinedcooperativepreload+metadata/payloadoverlapchange (notisolatingeachsubcomponent). Normalroutingmax-rankresultsstillnearparitywithdrift, donotextrapolate1.4%rank0balancedtofullservingthroughput. PreloadQ35 **7140110** submitted09:51:39 afterALL3phasegate; strideQ3 **7140098** submitted09:50:31. BothGPUslotsoccupied.

09:55UTC both preparedpreloadfullmodelalignment dryruns PASS usingFASTAFD_DRY_RUN=1 (noGPUsubmissions): paths/profilehelper/pinnedprompt/topology/normalrouting/forceMultiMBGraphOverlap allresolved. Sourceverificationdisabledforarchives asbefore; hostdonoractualcompactABI. Candidatechoice stillpendingQ35phaseandstrideprobe. Logs qwen3_alignment/dry_run.log,qwen35_alignment/dry_run.log.

09:58UTC downstreamfullsweep plans preparedunderpreload/sweeps,11groups464cases withsamefrozenbaselinecoverage/policies (Q3164,Q35300). Existing run_megamoe_sweeps.py --validate-only PASS; NOTsubmitted. Freshmanifests/source/cache/taskpaths, workflowwithnativecachepropagation; startonlyafterselection+sixfreshphasegates+bothNEWalignments. Neverreuseoldclosureevidence. Plansrequireworkflow/manifestsremote syncbeforeactualrunner.

PreloadQ3 fullchainrepeat audit: slowercandidate30-recordmean versusfasterdensecontrolmean still+.261/+1.227/+1.017% (rows4/6/24), versusfasterpreviouspushcontrol+8.025/+8.631/+6.994%. Bothrepeatsretained; no statisticalindependenceclaim. Supportsbalancedrank0improvement whileordinaryroutingMAXrankremainsnearparity. fullchain_repeat_comparison.json.

General memory qualification: previous12geometryzero-extra result covers TARGET Q3EP8/Q35EP16 (caps1..3840), notallEPs. full_ep_memory_audit.json evaluatesEP2..64 atcap768/3840: Q3EP64(localexperts2) framepaddinggrowsactivationpoolby23592960B/124256256B perbuffer (22.5/118.5MiB); otherfullsweepmodel/EP geometrieszeroextra. Two-lane servingmultipliestheperbuffercost. No claimofuniversalzeroaddedallocation. Generalmax(original,sourceframes) isintentionalcapacityformula, notfallback.

10:12UTC tokenstrideQ3 7140098 COMPLETED17:42,18/90/540+540timelinesaudited. Rows7/4/8 phase5.648/5.440/5.712us; allhistory+dense5%targetsPASS butcausalstrideincrement0/+.585/-.563%, no meaningfulbenefit. Fullchainrows4/7/8 candidate50.5984/50.8037/50.8597 vsrankmajor50.2469/50.4635/50.9381us, gains-.699/-.674/+.154%; remains7.026–7.325%slowdense. DoNOTpromotestridebasedonbaselinegatesalone. Q35strideunrun; notproofallstridesirrelevant. PreloadQ35 7140110 COMPLETED18:49, collecting6arms; zeroGPUjobsuntilnewselectionstage.

10:16UTC preloadQ35 7140110 COMPLETED18:49,18/90/540+540timelinesauditPASS. Rows10/2/11 phase5.344/5.280/5.440us, history20.853/13.158/21.659%, dense18.137/19.118/20%; all3inclrepeatsPASS => preloadALL6gatesPASS. Causalphase+.299/+1.198/-2.410% retainedFAIL. Fullchainrows2/10/11 candidate49.8315/62.9909/61.6848, dense47.9557/60.0293/59.9525, previouspush54.1504/66.5115/66.4832us: candidate3.911/4.934/2.889%slowdense, but7.976/5.293/7.217%fasterpreviouspush. Launchcandidate10.4651/10.9941/10.9691 vscontrol14.6320/15.4981/15.4768, so4.17–4.51usupstreamgapremoved; remainingbalancedQ35gaplargelyconsumer. Selection: preloadforfullvalidation (strongesttestedgeneralpushsource organization, sampledphaseobjectiveachieved), no claimgeneralservinglatencywin or defaultreplacementofpull. Q3stridenogain; Q35strideunrun/open, notblockingoriginalphase→alignment→fullsweepsequence. BothalignmentplansmarkedREADYaftergate+dryrun; launchingwithmax2. Main/indexuntouched.

10:17UTC fullmodelalignment submitted **Q37140414**, **Q357140417**; Q3running,Q35prioritypending, bothcounttowardmax2. DryrunsalreadyPASS. Newalignment_audit_plan.json usesactualjobrunpaths andpinned priorrefs (Q3megamoe_final_e2e_20260909/alignment_score_retry, Q35qwen35_full_attention_20260910/qwen35_score). MustrequireCOMPLETED0:0,48/168orderedprompts,17tokens each,finitechosenandcandidatescores,fullprefixcoverage100%top10; changedprefixrequiresfreshscorer, no inheritedscores. Fullsweepplans11/464validated, sourcefrozenpreload, runonlyafterbothnewalignmentsPASS.

10:20UTC referencechecking consolidatedintomaintained compare_cached_alignment.py: validateschosenrank/logprob/top1, completeposition/token coverage,candidateuniqueness/finitevalues andsamplemodel. ActualpinnedQ348/816,Q35168/2856 PASS; injectedNaNchosen,Inf candidate,andwrongposition rejected foreachreference. CPU-onlyvalidation; notfreshmodelalignment. reference_validation.json collected.

Reviewable selectedartifact: preload/final_overlay contains5files (framepushhostAPI/hostJIT/layout +boundedgatherreceiver +preloadnative), candidate.patch593lines againstunchangedmain, candidate_files.json recordslayers/status. No implementation/indexedits. Fullsweepplans/workflow/manifests now syncedremote; cannotstartuntilbothNEWalignmentPASS.

10:37UTC Q3 fullmodel7140414 COMPLETED0:0/17:25. Freshgeneration audit PASS48orderedprompts×17tokens; pinned-prefix score audit816/816covered/top1/top10, finitechosen/candidatescoresvalidated. Newalignment_summary.json collected. Initialrsynccollectionusedwrongparent(results/) andfailed23; correctedtoexplicitoutputpathfromauditplan, bothJSONscollected; noGPUrerun. Q35stillrunning; fullsweepsnotstarted.

10:42UTC Q35fullmodel7140417 COMPLETED0:0/20:51; ordered168prompts×17tokens PASS, all2856covered/top10,2853top1. Existingrank2firsttokensprompt119/123/143 preservepinnedprefixes. BothNEWalignmentsPASS, nofreshscoringneeded; summariescollected/synced. Fullsweepmanifest/launchorderstatusREADY; zerooutstandingGPUjobsbeforelaunch. Sourcefrozen, max2/poll300.

Offlinepooledphaseaudit(bothrepeats,10samples/case/arm): Q35compute-spanincreases2.752–3.104usvsdense, dispatchdecreases1.184–1.360us, exposedcombinetaildecreases.064–.128us. Q3compute-span+.160–.400us, dispatch−1.008–1.184us. phase_delta_diagnosis.json retainsallmetrics; overlappingphasemediansnotadditive, notproofpureTMAoperationcost. Supportsconsumerinputgatherinvestigation, notpushimpossibility. ReviewableREADME.mdnowrecordsservingapplicabilityandlimits.

10:49UTC fullsweepfirstjobs7140727(Q35EP4/139cases),7140728(Q3EP2/86cases), runnerSESSION6387, bothprogressing atfirst300spoll. Do notstartanother runner or additionalGPUjobs.

IndependentCPU-onlyfollowup compact_frame_linear_index: actualbounded-gatherQ3 SASS3fa0–40b0/41e0–4230 retainsrankdivision/remainder+token/topk+rankrecombination forfourTMAgatherrows. Identity row=bank*EP*capacity+descriptor/topk eliminatesredundantwork forrank-majorlayout. New2fileoverlay addslinear-source-tokenget_source_l1_row overload andusesitonlyinconsumer gather; sender/publication/allocationunchanged. 4704boundarychecksPASS (EP1..64,cap1..3840,topk1..16,bothbanks). Sourceclonedfromfrozenpreloadtoindependentlinear_index; CPUplancompilesmatchedold/newordinaryQ3EP8/Q35EP16+disassembles. CPUonly, noGPUcorrectness/timingyet, NOTselected, currentpreloadsweepunchanged. Initialoverlaypreparationassertambiguousget_num_bytesanchorfailedbeforewriting; correcteduniquesource-rowanchor.

10:52UTC linear_index CPU7140824 COMPLETED0:0/1:06; all8steps(matchedQ3EP8/Q35EP16 old/newcompile+SASS)PASS. TMA-AlinearregioninstructionsQ3 501→479, Q35 535→504; Q3maxR26→25,Q35R24unchanged,noTMA local-memoryinstructions. Allreachable-roleexplicitregistersbelowexistingbudgets; existingassertCALLsremainreviewedsamecategory. Countsstatic, NOTexecution/latencygain. Twofilecandidate.patchvsfrozenpreload saved. StatusCPU_BUILD_AND_ADDRESS_IDENTITY_PASS_GPU_UNRUN; no native/hostallocation/publication changes, no GPUplan submitted, NOTselected. Currentfullsweepcandidate remainspreload.

10:56UTC firstpartialfullsweepfetch: Q3EP2 22completed/110phasesamples (of164modelcases), ALL22>5%vsfrozenhistory, mean13.366%,min7.254%,max22.120%. Q35EP4 18completed/90samples (of300),14improved/4regressed,7>5%,mean4.834%,min−10.056%,max18.947%. partial_frozen_history_comparison.json explicitlyPARTIAL_NOT_FULL_EP_OR_SWEEP; thisprovesno universalEPgain andmustretainregressions. Bothjobscontinue, runner6387max2/poll300; no changedsource/defaultpolicy. Collectedartifactcontracts40PASS. Frozenfullsweep goalstillactive.

10:58UTC firstpartialtimelineaudit: Q322cases660ordinaryrecords,Q3518cases540records. Versusoriginaldensefullsweep (unpairedrank0means), Q3meanpercasechain−1.115%,19/22regress,launch−.082us,kernel+1.451us; Q35−6.555%,all18regress,launch+.316us,kernel+7.265us. partial_fullchain_comparison.json retainedallrecords/scope; no servinglatencywinclaim. Initialtimelineanalysiscommandusedstalehelperfromcopiedworkflow lacking--sweep (bothCLIexit2beforeanalysis); correctedtoexistingrelease_push/workflowmaintainedhelper, bothPASS. sweeps/analysis_plan.json correctedforthispath; GPUjobs/sourceunchanged.

11:04UTC independentCPU-only **compact_frame_leader_gather 7140952** submitted. Builds4matchedordinarygeometrypairs: Q3EP8 M16,Q35EP16 M16/M32/M64 (lattertwocompilergeometryprobes). Base=linear_index; forcompiletimeBLOCK_M<=32, explicitlane0loadsall4-rowgatherindices(8/16registervalues)andissuesrequestswithouttheper-laneelectionloop; largerMkeepdistributedindices. Addresses, OOBpadding, totalrequests/bytes, pipelineexpect_txandpublicationunchanged. MotivationactualSASS: ELECT/PLOP/R2UR/BRA.U.ANY loopat4630–46c0 repeatedforeachlane/gather/stage; staticinstructioncountalonenotlatency. Mustinspectregisterbudget40/spillsandactualissueloweringbeforeanyGPUtest. Source2fileoverlayisolated, frozenpreloadsweepsunchanged. CPUmonitor41725, sweep runner6387(max2/poll300). No newGPUplanssubmitted.

11:08UTC leader_gather CPU7140952 COMPLETED0:0/1:15, all16stepsPASS. TMAlinearregions (base=linear_index): Q3M16 479→449instructions,R25→19; Q35M16 504→460,R24→19; Q35M32 486→727,R27→23 (staticunrollgrowth, no latencyclaim); M64 505→505,R24unchanged. AllreachableexplicitregisterbudgetsPASS,noTMAspills. CPUcompilationpromisingbutGPUunrun. Decision: pauseautomaticfurtherpreloadsweeps toinsertGPUexact+freshmatchedconsumercomparisonwhenfirstcurrentjobfinishes; initialQ35EP4 7140727/Q3EP2 7140728 continue. Runner6387terminated143 viaexact-commandpkill; localpgrepfailedsysmon in sandbox andattemptpkill -f -INT hadBSDoptionerror2; correcteddefaultTERMmatchedonlythisrunner, noGPUjobcancel. runner_status/launch_orderexplicitlypaused. Full464requirementremains; resumeorprepareselected-sourcefullsweepsafterevidence, notdeclarepartialcompletion.

11:12–11:14UTC consumerfollowupconcreteplansprepared: leader_gather/validation11records, freshpreloadcontroloutputs versuslinear_indexand leader_gather forsmall/Q3/Q35; Q3rows4/96/256,Q35rows4/256/768atEP4 targetM16/32/64 (mustverifyactualGPUconfigs), legacycontrol/candidate. NativecacheCPU **7141058** preparesbothnewsourcecachesandABI-compatibleunchangedhostdonors(emptytargets). GPUvalidationNOTsubmitteduntilCPUcachePASSandfreeslot. Eightarmfreshphaseplansseed3514216109826126197: Q3EP8rows28/21/8,Q35EP16rows7/18/6, orderdensepull/preload/linear/leader/leader/linear/preload/densepull,30ordinary5phaseeach. WaitexactPASSbeforeruns; noGPUtemplatesstarted. Source/workflowisolationpreservescurrentpreloadsweepjobs. leader_gather/candidate.patch containsbothconsumerchangesvsalignedpreload.

11:17UTC followupvalidation/qwen3/qwen35 directoriesandworkflows syncedremote; localbash-ncandidate.envPASS. NativecacheCPU7141058stillrunningat11:16(3:45); noGPUfollowupsubmissionuntilitsPASSandoneoftwosweepjobsfinishes.

11:19UTC cacheCPU7141058 COMPLETED0:0/4:52; collectedoutandbothseedJSONs, exactly4CPU_BUILD_STEP_OK and2DENSE_ROUTER_BUILD_OK, eachhostdonorcopied1emptytarget. cache_build/completion.jsonPASS, validationplanREADY_WAITING_FOR_GPU_SLOT. No newGPUjobyet; currenttwofullsweepjobscontinue.

11:21UTC verifiedwaitafterCPUcachegate: GPUmonitor89966 lastscheduled11:19showsQ3EP2currentcaseep2_rows90(index69/86),Q35EP4ep4_rows76(index58/139),bothRUNNING36:10. Exactconsumerplanandcachesready; awaitfirstjobterminalsuccess, collectcompletedEPgroup, submitexactinfreedslot. NoadditionalCPUvariantsorGPUjobsstarted.

11:25UTC correctedCPUleaderinterpretation: directSASSinspection shows explicitlane0 leader STILLcontains ELECT/PLOP/BRA.U.ANY loop aroundeachGATHER4, plusper-stageR2UR. Staticcount/registerdecrease doesNOTproveissue-loopelimination. ExistingGPUplansremainvalidcomparisons, but donotclaimremovedloop. BoundedCPU-only compact_frame_elected_gather overlayprobe changesbothloaderandissuer tocute::elect_one_sync (fullwarp/fullmask). NVIDIA PTX9.4spec confirms sameleaderdeterministicallyforsamemembermask; no assumptionleaderislane0. Localcutehelperverifiedfullmask0xffffffff. AllTMAwarpthreadsreachbothcallsinuniformL1phase; largerMpathunchanged. CPU4geometriesoverlayincludesoverexistingleadersource; noruntimesource/caches/GPUplanschanged. Evidence primary https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync .

11:28–11:33UTC electedCPU7141236 COMPLETED0:0/1:00,4ordinarygeometryprobesPASS. Explicit-lane0→elect: Q3M16linear443vs449instructions,R22vs19; Q35M16 465vs460,R22vs19; M32 681vs727,R23both; M64 505/R24unchanged. Allreachableexplicitregistersbelowbudgets,noTMAspills. DirectSASSbranch-targetreview confirms smalltileper-gatherELECT/PLOP/BRA.U.ANYloopsremoved; interveningBRAisforwardtoscopeexit. Statementnotbasedonlyonstaticcounts. Elected selectedforGPUfollowup; explicit-laneplansSUPERSEDED_BEFORE_GPU_SUBMISSION, unrunseed3514216109826126197retained. Runtime electedsourcecopiedfromleaderthen2fileoverlaysynced. NativecacheCPU **7141272** submitted11:30:34,2steps(actualelectednative+emptyhostseed), RUNNINGlast11:32(1:49). Linearcache7141058alreadyPASS,doNOTreseed.

Q3fullsweepEP2 **7140728 COMPLETED0:0/45:46**, all86cases/430phase/2580ordinaryaudited; completion.jsonwritten. Frozenhistory86/86improve,85>5%,mean12.5658%,min2.8302%,max22.1198%. Fullmodelstill78casesmissing; NOTfullsweepclosure. Q35EP4 **7140727** stillrunning. OneGPUslotfree; submit electedexactoncecache7141272PASS. Fresh electedeightarmseed6421203082086643303: Q3EP8rows9/10/8,Q35EP16rows7/20/8. Same8armcausalorderdense/preload/linear/elected/elected/linear/preload/dense,30ordinary5phase. Allsource/cachescontrolsthosevalidated; noGPUfollowupjobyet. Newvalidation/phaseworkflowssharedsame11recordcoverage withcandidate=elected; syncedruntime/workflow, modelrsyncsessions54011/8462stillactiveatlastobservation, validationrsync10460completed. Completependingtransfersbeforetimingjobsubmissions; exactplanfullysynced.

11:34–11:36UTC electedcache7141272 COMPLETED0:0/2:05; actualnativebuild+oneemptyhostseedPASS,completion.jsoncollected. Allnewmodel/validationrsynctransferscompleted. SubmittedGPUexact **7141337**11:34:41, RUNNINGnvl72078-T12, monitor1678. Q35preloadEP4stillrunning;max2. Q3EP2fulltimeline86cases2580recordscollected: meanpercasechain−2.3257%,83/86regress,min−10.3959%,max+1.6592%,launch+.272us,kernel+2.600us versusoriginaldensehistoricalfullsweep(unpaired). NewconsumerGPUsourcefullyisolatedandpendingcorrectness/timing; electedcandidate.patchsaved.

11:43–11:45UTC criticalactual-configaudit: main7141337 first6recordsPASSsofar (smallbothnew60comparisonseach;Q3bothnew88each). ActualQ3configsrows4→M16,96→M64,256→M96, so priorplannedM32coverageWRONG. Hostheuristic get_block_config_for_mega_moe uses expectedtokens/expert<=8.5→M16,<=16.5→M32,<=32.5→M64,<=64.5→M96. No change to runningjobinputs. Preparedminimal4-record validation_m32: controlpreload/elected Q3EP4rows48(expected12/expert),Q35EP4rows160(expected12.5) selectM32; verifyactualconfigs. BothinitialexactandsupplementmustPASSbeforephase. Existinglargecasesstillprovidevaluablelarger-pathcoverage; notdiscarded. No newGPUjob yet.

11:56–11:57UTC main elected7141337 COMPLETED0:0/13:33, all11records/556exactcomparisons PASS; collected completion.json. Submitted required M32 supplement **7141684**, RUNNING, monitor15184/300s. Q35EP4 sweep7140727 stillrunning. Timing waits supplemental exact plus actualM32 audit. User real-serving applicability remains essential: paired fullchain and MAXrank ordinary routing are separate from dispatch, alignment proves correctness not throughput. No source/index changes.

Serving memory check: existing preload full-model afd.log files only expose attention KV allocations (~150.5GiB), no expert-worker free/peak-memory records. Saved alignment_memory_log_evidence.json; this cannot support a fixed-expert-pool fits/does-not-fit conclusion. Open evidence-driven followup if elected gather retains consumer regression: earlier router_packed_aux fixed-pool path used per-warp vector activation sends and did not receive cooperative source preload. Single cooperative stage plus bulk copies could preserve contiguous consumer access, but ordinal-dependent destinations prevent the same metadata/copy overlap as source-indexed frames. No implementation or new job for this hypothesis; memory and fullchain require measurement.

12:02–12:03UTC supplemental7141684 COMPLETED0:0/5:03, all4recordsok/64exact comparisons; actualQ3rows48 andQ35rows160 bothM32 forcontrolandcandidate. Main+supplement620PASS, elected/linear correctness gateclosed. Submitted fresh8arm Q3EP8 timing **7141706**12:03:08; qwen35phaseplanreadywaitslot. Q35EP4preloadfullsweep7140727 continues; automaticremainingfullgroupspausedpendingconsumerdecision. No implementation/indexedits; active serving note elected/serving_applicability.md recordsgeneralgeometryrules, fullchain/normalrouting/memorylimits andnewalignmentrequirement. Goal ACTIVE.

Timing7141706 confirmedRUNNINGnvl72150-T[11-12], monitor **23923**/300s. Currentturn classifiedprogress+verifiedwait: correctnessgapclosed, freshtimingsubmitted; ongoingGPUjobs7141706and7140727, no thirdjob.

12:08UTC independent CPU candidate **fixed_pool_preload** implemented/synced fromrouter_packed_aux; one nativegatefile changed, bounded16KiBcooperativeactivationstage overlapsselector, warp-leaderbulkstores intoexistingcontiguousexpertrows, fullcompletionwaitbeforepublication. SF/metadata/receiver/allocation/genericwarp/legacyunchanged. Sourceandworkflowisolated; actualnativeCPUbuild+unchangedhostdonoremptycopy submitted (jobrecordfixed_pool_preload/cpu_build/job.json). No GPUplan submitted/selected. Currentelectedtimingcontinuesunchanged. This tests a remaining supported transport idea while preserving its explicit larger-memorytradeoff, not claiming a win.

12:10UTC fixed_pool_preload CPU7141857 FAILED1:0/43s beforecontainerentry: workdirretainedterminal/cache_build whileonly/cpu_build existed. No compiler/cachestep ran. Correctedworkdir/output/commandpaths innewcpu_build_retry; CPU **7141963** submitted12:10:16, RUNNING,monitor65664/180s. Sourceunchanged; hosttargetstillempty. Six-recordfixed_pool_preload/validation preparedUNSUBMITTED: freshsame-layoutfixedcontrol/candidate Q3rows4/48,Q35rows4/160,H32768/I512top8wideCTArows3/6(twoactivationchunks). Existinggeneric/legacycodeunchanged,parentvalidated; thisplanmakesno newcoverageclaimforthosepaths.

12:11–12:12UTC preloadQ35EP4 **7140727 COMPLETED0:0/1:27:12**; full139caseartifactfetch/audit ongoing(session60041), notyetclaimedcompleteaudit. FreedGPUslot usedforfresh electedQ35EP16 eight-arm **7142095**, submitted12:11:45. BothCPU/exactgatesalreadyPASS620; Q35consumerpenaltylarger thanQ3, so independentmatchedcomparisonwarrantedbeforeseeingQ3finalgain. CurrenttwoGPUtimings7141706/7142095; nofixedpoolGPUjob.

12:14–12:15UTC Q35EP4 fullgroup139/695phase/4170ordinary COMPLETE_AUDITED. Frozenhistory135improve/4regress,127>5%,mean15.4909%,min−10.0559%,max29.0735%. Timeline139cases4170records audited: versusoriginaldensehistoricalunpaired fullchainALL139regress,mean−7.49225%,range−10.90539..−3.38727%,launch+2.07556us,kernel+10.52641us. Fullmodelstill161casesmissing, overall464sweep239casesmissing. Strong support for investigating contiguousconsumer alternative; no generalpush-impossibilityclaim. InitialCPUanalysis assumednsys_kernel_us.instances fieldandfailedbeforewriting; correctedtoalready-auditedbaselineCSVinstances/manifest30, noGPUrerun.

CPU **7141963 COMPLETED0:0/2:05**, actualfixed_pool_preloadnativebuild+oneemptycompatiblehostseedPASS,2CPUsteps; completion.jsoncollected. Six-recordexactplansyncedREADYbutUNSUBMITTED(max2currenttimings). No resubmissionofCPU7141857or7141963, no reseedingcache. Potentialmetadatafollowupiffixedpoolpromising: CTAsharedordinals insteadlocalglobalstore/reload forcopywarps; notimplemented, currentcandidatepreservesmetadataforcausalcomparison.

12:19–12:24UTC electedQ3partial first5arms15captures75phase450ordinary+450timelines audited, controlsrepeatsPENDING. Bothcandidatepasses havecompute~26.7–27.1us vsfirstlinear24.7–24.9us; fullchaincandidate48.6–49.0us vslinear46.35–46.94us, provisional−4.40..−5.17%. DoNOTfinalrejectuntilreversecontrols. StaticuniqueGATHER-bearingTMAspan confirmsQ3plainLDG.E sites4(linear)→8(elected),notlatencyproof. IndependentCPUprobe **compact_frame_shuffled_gather 7142199** submitted12:23:17: retain4distributedindexloads, fullwarpSHFLtoexistingelectedissuer forM<=32; M>32unchanged. Source/layout/addresses/requests/padding/publicationunchanged. CPU4ordinarygeometrybuild/registergatepending, no runtime/cache/GPUplan yet. Monitor77023/180s. Analysisinitialrawindex/first40span selectedotherrole(zeros),correctedbyuniqueGATHERregion; savedplanprovenance.

12:25–12:28UTC shuffleCPU **7142199 COMPLETED0:0/58s**, all8steps/fourordinarygeometriesPASS. Elected→shuffled TMAregions: Q3M16 443/R22/LDG8→386/R25/LDG4;Q35M16 465/R22/8→404/R22/4;M32 681/R23/16→449/R25/4;M64 505/R24/4unchanged. AllreachableexplicitregisterbudgetsPASS,noTMAspills; smalltileissuecontextsretain no per-gather election backedges. StaticproofNOTlatency. Actualruntimesourcecopiedfromelected+2fileoverlay, workflowssynced; nativecache+emptycompatiblehostdonorCPU **7142250** submitted12:27:19.

Combined **compact_frame_shuffled_gather/validation**9recordplanprepared/synced, UNSUBMITTED untilcachePASS+oneGPUslotfree: controlalignedpreload, referencefixed_pool_preload, candidateshuffle; Q3EP4rows4/48,Q35EP4rows4/160(H4096),wideH32768/I512/top8rows3/6. CoversM16/M32,widechunks,normal/alternate/padding/graph. Generic/legacy/largerMunchangedparentcoverageexplicit, no newclaim. Fixed_pool_preload standalone6recordvalidation **SUPERSEDED_UNRUN**; doNOTsubmit it. ExistingelectedQ3/Q35timingsmustfinishnormally, onefreeGPUslot sufficient(no requirementwaitboth). No finalconsumerselectionyet.

12:41–12:46UTC electedQ35 **7142095 COMPLETED0:0/25:14**,24captures120phase720ordinary+720timelinesAUDITED. Bothlinearelectedall3USERphasegatesPASS => bothALL6overall. LinearQ35history17.703–20.574%,dense15.271–19.266%, fullchainstill2.190–4.876%slowdense, vspreload+.527/+.080/−.305%(rows7/8/20). Electedfullchain2.011–3.892%slowlinear,6.167–6.985%slowdense. Rejectelectedissuer onbothmodels; simplerlinearkeptascontrol, no universalservingwin.

Combinedexact7142265 FAILED1:0/9:43 onlyafter6QwenrecordsPASS240. ActualM16/M32allsourcevariantsconfirmed. control_wide minrows3*top8=24 triggeredtests/_exact_qwen_case appendedunprepared balanced24/one_rank32 fixture; in-bandcombineassertfiredinUNCHANGEDpreloadcontrol, afternormalwidechecks butbeforewholecontrolrecord. Sourceunchanged. New validation_wide_recovery only3remainingwidecontrol/referencefixed/candidateshuffle rows1/6H32768I512; minrows1 usesparentvalidatedfixtureavoidsunrelated24routebranch; exact graph/alternate/paddingretained, shorttiming1/2/2. Mainfirst6preserved, neverclaimedwholejobPASS.

PreparedEP2supplement3Q3recordsrows4/96(M16/M32), parentrealE128 avoidsoldinvalid4expertfixture. FirstsubmissionrejectedBEFOREjobcreation becauseinheritedntasks-per-node4 conflictsntasks2; explicit--ntasks-per-node2matchespriorvalidEP2plan. No allocation/jobIDfromrejection. Correctedsubmissionandwiderecoverysubmitted; currentjobIDsintheirjob.json. Bothrequiredbeforepool_followup timing.

12:49UTC both recoveryGPUjobs confirmedRUNNING: wide7142411 nvl72098-T08,EP27142412 nvl72098-T09, sharedmonitor **85447**/180s. Nextideas/STATE refreshed to distinguishhistoricalstalejobsfromcurrentstate. Remainingstrideidea explicitlyQ35/ordinaryroutingunmeasured; no newvariantuntilcurrentfixedbulk/shuffled measurements.
