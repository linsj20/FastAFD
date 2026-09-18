**MegaMoE latency comparison**

Times are µs. Each capture uses its own producer metadata-transfer start as zero. Ranges are the first–last observed markers in that capture; they are not transfer durations. Missing markers are shown as —. All captures and outliers are retained.

**Ordinary kernel timing**

| Arm | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| helper2 | 37.277 | 37.280 | 36.864 | 37.888 |
| best1536 | 37.331 | 37.312 | 36.736 | 37.664 |
| best1536_repeat | 37.334 | 37.344 | 36.992 | 37.664 |
| helper2_repeat | 37.206 | 37.216 | 36.960 | 37.568 |
| parallel1024 | 37.517 | 37.520 | 37.280 | 37.824 |
| history32 | 37.894 | 37.952 | 37.312 | 38.528 |

Ordinary passes appear in the input's execution order; each row's statistics belong to that pass. Medians are not pooled across repeats.

The following tables are instrumented. Do not add their overlapping phases or subtract them from the ordinary kernel timings above.

**Capture 0, rank 0**

| Milestone / duration | helper2 | best1536 | parallel1024 | history32 |
|---|---:|---:|---:|---:|
| Metadata transfer starts | 0.000 | 0.000 | 0.000 | 0.000 |
| Kernel entry | 3.552 | 3.552 | 3.616 | 3.296 |
| Producer metadata stores issued | 2.016–2.464 | 2.048–2.432 | 2.080–2.496 | 2.048–2.400 |
| Producer SF transfer starts | -0.352–0.672 | -0.352–0.512 | -0.352–0.576 | — |
| Producer SF stores issued | 0.192–0.768 | 0.160–0.608 | 0.192–0.672 | — |
| Producer metadata-ready publication marker | 5.088 | 5.024 | 5.056 | 4.608 |
| Startup workers enter | 4.480–4.960 | 4.384–4.928 | 4.576–4.960 | — |
| Receiver route metadata acquired | 5.376–5.664 | 5.440–5.664 | 5.248–5.472 | — |
| Sizes-ready publication issue | 5.888 | 5.952 | 5.792 | — |
| SF/index phase starts (see ownership notes) | 5.920–6.080 | 6.048–6.272 | 5.824–5.984 | — |
| SF worker stores issued (variant warp scope) | 6.720–8.000 | 6.848–8.160 | 6.720–7.648 | — |
| Local SF readiness publication issue | 7.232–8.352 | 7.392–8.384 | 7.136–7.872 | — |
| Payload source-load phase starts (variant scope) | 5.216–5.408 | 5.248–5.312 | 5.280–5.536 | 15.808–17.280 |
| Payload send phase starts | 5.920–6.144 | 5.984–6.048 | 5.984–6.272 | — |
| Payload stores issued | 8.000–8.736 | 8.288–8.608 | 8.320–8.736 | — |
| Receiver activation validation starts | 6.208–6.464 | 6.368–6.496 | 6.176–6.272 | — |
| First-window A readiness publication issue | 7.456–8.416 | 7.744–8.704 | 7.328–8.128 | — |
| Receiver remainder validation starts | 6.240–6.592 | 7.744–8.704 | 6.112–6.304 | — |
| Remaining A readiness publication issue (see scope) | 9.088–10.400 | 9.600–10.688 | 9.568–10.848 | — |
| Startup workers retire | 5.408–12.160 | 5.344–12.608 | 5.568–12.128 | — |
| Compute CTAs: metadata acquired | 5.248–5.696 | 5.344–5.696 | 5.184–5.856 | 15.040–15.232 |
| Pull issue markers | — | — | — | 15.936–17.440 |
| Pull completion observations | — | — | — | 16.416–19.904 |
| First MMA pair: index-path marker (variant scope) | 6.112–6.144 | 6.016 | 5.920–5.952 | — |
| First MMA pair: activation readiness observed | 8.320–8.352 | 8.672 | 8.384 | — |
| First MMA pair: SF readiness observed | 8.288–8.384 | 8.384–8.448 | 8.384–8.416 | — |
| First MMA pair: activation load issued | 8.448–8.480 | 8.832 | 8.512 | — |
| First MMA / operand-ready boundary | 8.864 (actual MMA) | 9.088 (actual MMA) | 8.864 (actual MMA) | 20.160 (actual MMA) |
| Pair-start spread: first / median / last | 8.864 / 9.312 / 9.824 | 9.088 / 9.728 / 10.080 | 8.864 / 9.184 / 9.600 | 20.160 / 21.056 / 21.280 |
| All pairs: median latest A minus latest SF observation | 0.000 | 0.352 | 0.128 | — |
| All pairs: median last activation-load marker to MMA | 0.320 | 0.288 | 0.320 | — |
| Remainder issuer wait, median duration | 0.480 | 0.224 | 1.312 | — |
| Final accumulator-ready observation | 34.144 | 34.208 | 34.944 | 44.768 |
| Compute span, duration | 25.280 | 25.120 | 26.080 | 24.608 |
| Kernel end | 40.320 | 40.064 | 40.416 | 48.864 |
| First MMA from selector entry, duration | 11.296 | 11.456 | 11.200 | 22.400 |

**Capture 1, rank 0**

| Milestone / duration | helper2 | best1536 | parallel1024 | history32 |
|---|---:|---:|---:|---:|
| Metadata transfer starts | 0.000 | 0.000 | 0.000 | 0.000 |
| Kernel entry | 3.424 | 3.424 | 3.520 | 3.104 |
| Producer metadata stores issued | 1.888–2.240 | 1.920–2.304 | 1.952–2.368 | 1.792–2.208 |
| Producer SF transfer starts | -0.352–0.544 | -0.352–0.576 | -0.352–0.608 | — |
| Producer SF stores issued | 0.192–0.640 | 0.192–0.672 | 0.192–0.704 | — |
| Producer metadata-ready publication marker | 4.832 | 4.896 | 4.960 | 4.416 |
| Startup workers enter | 4.384–4.800 | 4.384–4.768 | 4.448–4.864 | — |
| Receiver route metadata acquired | 5.216–5.472 | 5.280–5.536 | 5.376–5.600 | — |
| Sizes-ready publication issue | 5.504 | 5.536 | 5.632 | — |
| SF/index phase starts (see ownership notes) | 5.568–6.048 | 5.728–6.176 | 5.760–6.176 | — |
| SF worker stores issued (variant warp scope) | 6.592–7.840 | 6.720–8.480 | 6.624–8.096 | — |
| Local SF readiness publication issue | 7.040–8.576 | 7.008–8.608 | 7.136–8.320 | — |
| Payload source-load phase starts (variant scope) | 5.088–5.344 | 5.152–5.216 | 5.152–5.376 | 5.408–7.296 |
| Payload send phase starts | 5.792–6.112 | 5.856–5.952 | 5.856–6.144 | — |
| Payload stores issued | 7.712–8.448 | 7.904–8.288 | 7.840–8.640 | — |
| Receiver activation validation starts | 5.952–6.528 | 6.016–6.592 | 6.048–6.560 | — |
| First-window A readiness publication issue | 7.136–8.448 | 7.456–8.928 | 7.232–8.544 | — |
| Receiver remainder validation starts | 5.952–6.528 | 7.456–8.960 | 6.048–6.592 | — |
| Remaining A readiness publication issue (see scope) | 9.024–10.624 | 9.376–11.072 | 9.728–11.104 | — |
| Startup workers retire | 5.376–12.320 | 5.248–12.064 | 5.408–12.000 | — |
| Compute CTAs: metadata acquired | 5.152–5.824 | 5.184–5.888 | 5.280–5.984 | 4.704–5.152 |
| Pull issue markers | — | — | — | 5.536–7.424 |
| Pull completion observations | — | — | — | 5.920–9.856 |
| First MMA pair: index-path marker (variant scope) | 6.208 | 5.664 | 5.760 | — |
| First MMA pair: activation readiness observed | 8.608–8.704 | 8.864–8.896 | 8.640 | — |
| First MMA pair: SF readiness observed | 8.416–8.448 | 8.480–8.832 | 8.512–8.544 | — |
| First MMA pair: activation load issued | 8.768–8.864 | 9.024 | 8.768 | — |
| First MMA / operand-ready boundary | 9.088 (actual MMA) | 9.280 (actual MMA) | 9.024 (actual MMA) | 10.112 (actual MMA) |
| Pair-start spread: first / median / last | 9.088 / 9.376 / 10.048 | 9.280 / 9.440 / 10.144 | 9.024 / 9.312 / 9.728 | 10.112 / 10.496 / 11.232 |
| All pairs: median latest A minus latest SF observation | 0.192 | 0.320 | 0.000 | — |
| All pairs: median last activation-load marker to MMA | 0.320 | 0.288 | 0.320 | — |
| Remainder issuer wait, median duration | 0.768 | 0.352 | 1.472 | — |
| Final accumulator-ready observation | 34.368 | 34.464 | 35.456 | 35.136 |
| Compute span, duration | 25.280 | 25.184 | 26.432 | 25.024 |
| Kernel end | 39.520 | 39.776 | 39.904 | 39.936 |
| First MMA from selector entry, duration | 11.648 | 11.616 | 11.360 | 12.416 |

**Capture 2, rank 0**

| Milestone / duration | helper2 | best1536 | parallel1024 | history32 |
|---|---:|---:|---:|---:|
| Metadata transfer starts | 0.000 | 0.000 | 0.000 | 0.000 |
| Kernel entry | 3.552 | 3.840 | 3.552 | 3.328 |
| Producer metadata stores issued | 2.176–2.464 | 2.112–2.496 | 2.080–2.432 | 2.016–2.368 |
| Producer SF transfer starts | -0.352–0.608 | -0.352–0.576 | -0.448–0.480 | — |
| Producer SF stores issued | 0.192–0.704 | 0.192–0.672 | 0.192–0.576 | — |
| Producer metadata-ready publication marker | 5.088 | 5.056 | 4.992 | 4.672 |
| Startup workers enter | 4.480–4.960 | 4.832–5.248 | 4.416–4.928 | — |
| Receiver route metadata acquired | 5.504–5.728 | 5.344–5.888 | 5.408–5.632 | — |
| Sizes-ready publication issue | 6.016 | 5.920 | 5.920 | — |
| SF/index phase starts (see ownership notes) | 6.080–6.208 | 6.048–6.176 | 5.984–6.112 | — |
| SF worker stores issued (variant warp scope) | 6.880–7.936 | 6.880–7.840 | 6.816–7.968 | — |
| Local SF readiness publication issue | 7.424–8.640 | 7.264–8.000 | 7.392–8.160 | — |
| Payload source-load phase starts (variant scope) | 5.216–5.440 | 5.504–5.728 | 5.216–5.408 | 5.632–7.424 |
| Payload send phase starts | 5.920–6.176 | 6.208–6.464 | 5.920–6.144 | — |
| Payload stores issued | 8.128–8.576 | 8.512–8.896 | 8.000–8.896 | — |
| Receiver activation validation starts | 6.368–6.592 | 6.432–6.752 | 6.272–6.432 | — |
| First-window A readiness publication issue | 7.488–8.480 | 7.808–8.640 | 7.456–8.384 | — |
| Receiver remainder validation starts | 6.368–6.656 | 7.808–8.640 | 6.304–6.464 | — |
| Remaining A readiness publication issue (see scope) | 9.280–10.656 | 9.504–10.528 | 9.728–11.104 | — |
| Startup workers retire | 5.472–12.224 | 5.760–12.768 | 5.408–12.128 | — |
| Compute CTAs: metadata acquired | 5.408–5.856 | 5.344–5.824 | 5.312–5.824 | 4.832–5.312 |
| Pull issue markers | — | — | — | 5.792–7.552 |
| Pull completion observations | — | — | — | 6.272–10.016 |
| First MMA pair: index-path marker (variant scope) | 6.112 | 6.048–6.080 | 6.048–6.112 | — |
| First MMA pair: activation readiness observed | 8.480 | 8.544 | 8.416–8.512 | — |
| First MMA pair: SF readiness observed | 8.416 | 8.352 | 8.448–8.480 | — |
| First MMA pair: activation load issued | 8.608 | 8.704 | 8.544–8.640 | — |
| First MMA / operand-ready boundary | 8.928 (actual MMA) | 8.992 (actual MMA) | 8.928 (actual MMA) | 10.080 (actual MMA) |
| Pair-start spread: first / median / last | 8.928 / 9.472 / 10.016 | 8.992 / 9.504 / 9.984 | 8.928 / 9.280 / 9.856 | 10.080 / 10.688 / 11.072 |
| All pairs: median latest A minus latest SF observation | 0.032 | 0.288 | 0.032 | — |
| All pairs: median last activation-load marker to MMA | 0.320 | 0.288 | 0.384 | — |
| Remainder issuer wait, median duration | 0.480 | 0.320 | 1.488 | — |
| Final accumulator-ready observation | 34.752 | 34.144 | 34.816 | 35.840 |
| Compute span, duration | 25.824 | 25.152 | 25.888 | 25.760 |
| Kernel end | 39.936 | 40.576 | 39.936 | 39.616 |
| First MMA from selector entry, duration | 11.232 | 11.328 | 11.360 | 12.384 |

**Marker scope and sources**

Producer events describe the profiled rank's outgoing work; they do not time incoming network latency. Publication issue is not completion. Readiness markers are consumer observations; differences between A and SF observations are not isolated operand stalls. Remainder wait measures the activation issuer, not an exact MMA stall. Compute span runs from the indicated first-MMA boundary to the final accumulator-ready observation. Consult each capture's variant notes for ownership and transport details.

- helper2: [helper2_decomposition_7257013.json](</Users/shengjiel/Workspace/github/FastAFD/scratch/megamoe/lamport_payload/concurrent_dispatch/receiver_validate/remainder_helper/helper2_decomposition_7257013.json>)

  genphase_index_scatter: IndexScatter is SF-owner phase entry. Four SF work warps issue stores and publish SF1 independently. Receiver route/score materialization and its marker are absent; epilogue reads original producer metadata.

  sf_ready_contract: SF1 published after unchanged SF128-thread barrier4. PrefixA2 after A256-thread barrier3. RemainingA4 after independent remainder128-thread barrier5; bit4 alone does NOT certify prefix. Compute acquires bit2 beforeK0 and bit4 beforeK8, both proxyordered. Original metadata directprefetch unchanged. DispatchDone includes allworkers, then existing retirement/cleanup. FullPayloadIssue legacy marker names remainder publication, now SF CTA workerwarp4, not A CTAwarp0.

  index_path: MegaIndicesReady marks one activation issuer perCTA before preparation of contiguous materialized row coordinate; no indexarray or gather4. Source materializedbase+poolblock*32+(leader?0:alignedM/2). SF coordinator remains independent.

  activation_load: Regular2D TMA from dispatch warp0 only, twoCTAgroup. Native compact-A descriptor boxM8 instead ofgather1. One eight-row box/CTA foralignedM16, two foralignedM32; smemoffset row*BLOCK_K. Exact parent activation bytevolume: fixedcasecluster2048B plusSF1024B; alignedM*BLOCK_K expectedbytes. No duplicated peer/padding reads beyond parent MMAalignedextent. Separate whole-tile first1024B/remainder readiness, independent SFcoordinator retained. MegaGatherIssued marks regularTMAissue.

  remainder_wait: First FC1 activation warp0 perCTA: WaitStart before full-A bit4 acquire at Kstage8 (1024B); Ready after acquire/syncwarp/proxyfence before empty-stage wait.150 locations. Issuer wait is not exact MMA stall. Direct GenphaseRemainderValidateStart precedes first remainder row descriptor/load; GenphaseFullPayloadIssue follows stores, proxyfence, its producer-group join and release issue. They share the producer CTA/warp and are paired separately from consumer wait.
- best1536: [best1536_decomposition_7257013.json](</Users/shengjiel/Workspace/github/FastAFD/scratch/megamoe/lamport_payload/concurrent_dispatch/receiver_validate/remainder_helper/best1536_decomposition_7257013.json>)

  genphase_index_scatter: IndexScatter is SF-owner phase entry. Four SF work warps issue stores and publish SF1 independently. Receiver route/score materialization and its marker are absent; epilogue reads original producer metadata.

  sf_ready_contract: SF1/firstA2/fullA4 retain original L1 readiness. No metadata8 signal/acquire. Original route-ready SYS acquire and shared-count relay order producer metadata; epilogue prefetches scores/return descriptors before accumulator wait. Same SF128-thread join, A validation and retirement. ComputeEnd remains an epilogue observation.

  index_path: MegaIndicesReady marks one activation issuer perCTA before preparation of contiguous materialized row coordinate; no indexarray or gather4. Source materializedbase+poolblock*32+(leader?0:alignedM/2). SF coordinator remains independent.

  activation_load: Regular2D TMA from dispatch warp0 only, twoCTAgroup. Native compact-A descriptor boxM8 instead ofgather1. One eight-row box/CTA foralignedM16, two foralignedM32; smemoffset row*BLOCK_K. Exact parent activation bytevolume: fixedcasecluster2048B plusSF1024B; alignedM*BLOCK_K expectedbytes. No duplicated peer/padding reads beyond parent MMAalignedextent. Original whole-tile first1536B/full readiness, independent SFcoordinator retained. MegaGatherIssued marks regularTMAissue.

  remainder_wait: FirstFC1tile activation warp0 perCTA: WaitStart before full-A bit4 acquire atKstage12; Ready after acquire/syncwarp/proxyfence before empty-stage wait.150locations. Issuerwait is not exactMMAstall; parent has300locations across two gatherwarps/CTA. Direct GenphaseRemainderValidateStart precedes first remainder row descriptor/load; GenphaseFullPayloadIssue follows stores, proxyfence, its producer-group join and release issue. They share the producer CTA/warp and are paired separately from consumer wait.
- parallel1024: [parallel1024_decomposition_7257013.json](</Users/shengjiel/Workspace/github/FastAFD/scratch/megamoe/lamport_payload/concurrent_dispatch/receiver_validate/remainder_helper/parallel1024_decomposition_7257013.json>)

  genphase_index_scatter: IndexScatter is SF-owner phase entry. Four SF work warps issue stores and publish SF1 independently. Receiver route/score materialization and its marker are absent; epilogue reads original producer metadata.

  sf_ready_contract: SF1 published after unchanged SF128-thread barrier4. PrefixA2 after A256-thread barrier3. RemainingA4 after independent remainder128-thread barrier5; bit4 alone does NOT certify prefix. Compute acquires bit2 beforeK0 and bit4 beforeK8, both proxyordered. Original metadata directprefetch unchanged. DispatchDone includes allworkers, then existing retirement/cleanup. FullPayloadIssue legacy marker names remainder publication, now SF CTA workerwarp4, not A CTAwarp0.

  index_path: MegaIndicesReady marks one activation issuer perCTA before preparation of contiguous materialized row coordinate; no indexarray or gather4. Source materializedbase+poolblock*32+(leader?0:alignedM/2). SF coordinator remains independent.

  activation_load: Regular2D TMA from dispatch warp0 only, twoCTAgroup. Native compact-A descriptor boxM8 instead ofgather1. One eight-row box/CTA foralignedM16, two foralignedM32; smemoffset row*BLOCK_K. Exact parent activation bytevolume: fixedcasecluster2048B plusSF1024B; alignedM*BLOCK_K expectedbytes. No duplicated peer/padding reads beyond parent MMAalignedextent. Separate whole-tile first1024B/remainder readiness, independent SFcoordinator retained. MegaGatherIssued marks regularTMAissue.

  remainder_wait: First FC1 activation warp0 perCTA: WaitStart before full-A bit4 acquire at Kstage8 (1024B); Ready after acquire/syncwarp/proxyfence before empty-stage wait.150 locations. Issuer wait is not exact MMA stall. Direct GenphaseRemainderValidateStart precedes first remainder row descriptor/load; GenphaseFullPayloadIssue follows stores, proxyfence, its producer-group join and release issue. They share the producer CTA/warp and are paired separately from consumer wait.
- history32: [history32_decomposition_7257013.json](</Users/shengjiel/Workspace/github/FastAFD/scratch/megamoe/lamport_payload/concurrent_dispatch/receiver_validate/remainder_helper/history32_decomposition_7257013.json>)

  genphase_index_scatter: IndexScatter to IndexReady covers index/score scatter and publication only. SF is gathered within FC1 stages, and payload readiness is acquired at FC1 consumption. Earlier variants include SF scatter in dispatch; compare whole-kernel time and compute span as well as exposed dispatch.

  sf_ready_contract: MegaGatherSfReady follows first-stage shared SF writes and proxy fence, sampled in warp0 or split-SF warp2; it does not mean all producers completed.

  index_path: MegaIndicesReady is a variant-specific index-path boundary. In receiver-materialization parents it follows the SF/index acquire and precedes HBM index loads, not their completion. Consult source and the run contract.

  activation_load: MegaGatherIssued marks activation-load issue; the source and run contract distinguish indexed gather from regular TMA and specify active issuer warps and completion bytes.

  remainder_wait: Absent unless explicitly enabled by the captured source.
