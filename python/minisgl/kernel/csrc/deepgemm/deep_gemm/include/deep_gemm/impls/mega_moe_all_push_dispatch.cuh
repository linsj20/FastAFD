#pragma once

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

#ifdef IKET_BUILD_ENABLED
#define IKET_DEFAULT_INSTR_METHOD_WITHOUT_PAYLOAD_PMTRIG5
#include <iket/iket_device_apis.cuh>
CREATE_IKET_MARKER(GenphaseStart);
CREATE_IKET_MARKER(GenphaseInputsReady);
CREATE_IKET_MARKER(GenphaseMetadataFromProducer);
CREATE_IKET_MARKER(GenphaseTopkReceived);
CREATE_IKET_MARKER(GenphaseSizesReady);
CREATE_IKET_MARKER(GenphaseMiddlePayloadIssue);
CREATE_IKET_MARKER(GenphaseWindowPayloadIssue);
CREATE_IKET_MARKER(GenphaseSfReadyIssue);
CREATE_IKET_MARKER(GenphaseWindowValidateStart);
CREATE_IKET_MARKER(GenphaseRemainderValidateStart);
CREATE_IKET_MARKER(GenphaseIndexScatter);
CREATE_IKET_MARKER(GenphaseEnd);
CREATE_IKET_MARKER(MegaPayloadLoadStart);
CREATE_IKET_MARKER(GenphasePayloadStoresIssued);
#define GENPHASE_MARK(name) do { if (warp == 0) IKET_MARK(name); } while (0)
#define GENPHASE_MARK_WARP(name, owner) do { if (warp == (owner)) IKET_MARK(name); } while (0)
#define GENPHASE_MARK_ALL(name) IKET_MARK(name)
#else
#define GENPHASE_MARK(name) ((void)0)
#define GENPHASE_MARK_WARP(name, owner) ((void)0)
#define GENPHASE_MARK_ALL(name) ((void)0)
#endif

namespace deep_gemm {

inline constexpr uint32_t kFirstPayloadWindowBytes = 1024;
// Each disjoint window has exactly one publisher per tile. Prefix/middle/tail
// certify only their own materialized bytes. Consumers acquire each certificate
// before issuing a TMA that reads that window; SF remains independent.
inline constexpr uint32_t kDispatchSfReady = 1u;
inline constexpr uint32_t kDispatchFirstAReady = 2u;
inline constexpr uint32_t kDispatchMiddleAReady = 4u;
inline constexpr uint32_t kDispatchTailAReady = 8u;

template <uint32_t kPartitions>
__host__ __device__ constexpr uint32_t dispatch_ready_mask(const uint32_t bit) {
    static_assert(kPartitions >= 1 && kPartitions <= 8);
    return bit * static_cast<uint32_t>(((1ull << (4 * kPartitions)) - 1) / 15);
}

// Keep the polling implementation outside the main kernel's optimizer region.
// No timing markers live here: all phase boundaries stay in the caller.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kTopk,
          uint32_t kBegin, uint32_t kEnd, uint32_t kOwners,
          bool kLinearSource = false, bool kHalfWarp = false,
          bool kStickyValidation = false>
__device__ __noinline__ void validate_window_rows(
    const uint32_t* descriptors, const uint8_t* input_base, uint8_t* output_base,
    uint32_t ready, uint32_t owner, uint32_t lane) {
    // A full-warp vote covers both rows. Half-warp ownership halves the
    // row-address state while doubling vectors per lane; both modes retain
    // the exact final observations for receiver-owned materialization.
    DG_STATIC_ASSERT(kBegin < kEnd && kEnd <= kHidden &&
                     kBegin % 512 == 0 && kEnd % 512 == 0, "Complete disjoint row windows required");
    DG_STATIC_ASSERT(kOwners >= 1 && kOwners <= 8,
                     "Window owner count must match its barrier");
    constexpr uint32_t kRows = kHalfWarp ? 1 : 2;
    constexpr uint32_t kLanes = kHalfWarp ? 16 : 32;
    constexpr uint32_t kVectors = (kEnd - kBegin) / (16 * kLanes);
    DG_STATIC_ASSERT(!kStickyValidation || kRows * kVectors <= 32, "Validation mask overflow");
    // The loop condition is warp-uniform, including a single remaining row.
    for (uint32_t first = owner * (kHalfWarp ? 2 : 1); first < ready; first += kOwners * 2) {
        const uint4* input[kRows];
        uint4* output[kRows];
        uint4 values[kRows][kVectors];
        #pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            const uint32_t i = first + kOwners * r + lane / kLanes;
            if (i < ready) {
                const uint32_t descriptor = descriptors[i];
                uint64_t source;
                if constexpr (kLinearSource) {
                    // descriptor = (rank * kTokens + token) * kTopk + slot.
                    source = descriptor / kTopk;
                } else {
                    const uint32_t rank = descriptor / (kTokens * kTopk);
                    const uint32_t token = (descriptor % (kTokens * kTopk)) / kTopk;
                    source = static_cast<uint64_t>(rank) * kTokens + token;
                }
                input[r] = reinterpret_cast<const uint4*>(input_base + source * kHidden);
                output[r] = reinterpret_cast<uint4*>(output_base + static_cast<uint64_t>(i) * kHidden);
            }
        }
        // Payload is immutable until all dispatch participants retire. Keep each
        // fully validated vector while waiting for the remaining transport words.
        uint32_t complete = 0;
        bool missing;
        do {
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                if (first + kOwners * r + lane / kLanes < ready) {
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v)
                        if (!kStickyValidation || !(complete & (1u << (r * kVectors + v)))) {
                            asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(values[r][v].x), "=r"(values[r][v].y),
                                   "=r"(values[r][v].z), "=r"(values[r][v].w)
                                 : "l"(input[r] + lane % kLanes + kBegin / 16 + v * kLanes) : "memory");
                        }
                }
            }
            missing = false;
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                if (first + kOwners * r + lane / kLanes < ready) {
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v) {
                        bool pending = values[r][v].x == 0x80000000u || values[r][v].y == 0x80000000u ||
                                       values[r][v].z == 0x80000000u || values[r][v].w == 0x80000000u;
                        missing |= pending;
                        if constexpr (kStickyValidation)
                            if (!pending) complete |= 1u << (r * kVectors + v);
                    }
                }
            }
        } while (__any_sync(0xffffffff, missing));
        #pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            if (first + kOwners * r + lane / kLanes < ready) {
                #pragma unroll
                for (uint32_t v = 0; v < kVectors; ++v) {
                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                             "l"(output[r] + lane % kLanes + kBegin / 16 + v * kLanes),
                             "r"(values[r][v].x), "r"(values[r][v].y),
                             "r"(values[r][v].z), "r"(values[r][v].w) : "memory");
                }
            }
        }
    }
}

// All-push dispatch: producer-owned routes/SF, deduplicated activation sends,
// and receiver-owned materialization with independent K-window readiness.
// FC1 consumes materialized rows through regular eight-row TMA.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kExperts, uint32_t kTopk,
          uint32_t kRanks, uint32_t BLOCK_M, uint32_t kSFPoolTokens, uint32_t kPushers,
          uint32_t kMetadataExpertCTAs, bool kRecycle = false,
          bool kPrioritizeExperts = false, bool kDedicatedDispatch = false,
          bool kLinearSource = false, bool kCacheDestinations = false, uint32_t kNarrowSf = 0,
          uint32_t kCoalescedSf = 0, bool kPrefetchSend = false,
          uint32_t kMiddlePayloadWindowEnd = 2048, bool kHalfWarp = false,
          uint32_t kFirstReadinessWindowBytes = 1024, bool kStickyValidation = false,
          bool kPrefixSweep = false, bool kProducerPrefix = false, uint32_t kSenderLdg = 0,
          bool kPrefixWide = false, uint32_t kCachePriority = 0, uint32_t kGroupPriority = 0,
          uint32_t kRowPartitions = 1, bool kSendMiddleFirst = false>
__device__ __forceinline__ void mega_moe_dispatch(
    uint32_t num_tokens, const layout::SymBuffer<kRanks>& peers, uint8_t* storage,
    uint32_t tid, const layout::Buffer& combine_token_buffer) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    using W = layout::Workspace;
    DG_STATIC_ASSERT(kFirstReadinessWindowBytes == 512 || kFirstReadinessWindowBytes == 1024,
                     "Invalid first readiness window");
    DG_STATIC_ASSERT(kHidden > kFirstReadinessWindowBytes && kHidden % 512 == 0 &&
                     kFirstReadinessWindowBytes % 512 == 0, "Readiness windows require complete 512B chunks");
    DG_STATIC_ASSERT(kMiddlePayloadWindowEnd > kFirstReadinessWindowBytes &&
                     kMiddlePayloadWindowEnd % 512 == 0 &&
                     (kMiddlePayloadWindowEnd == 2048 || kMiddlePayloadWindowEnd <= kHidden),
                     "Invalid middle-window endpoint");
    constexpr uint32_t kThreads = 128, kWarps = 4, kDepth = 1;
    DG_STATIC_ASSERT(!kPrefixSweep || (!kPrefetchSend && kDepth == 1),
                     "Prefix sweep requires one ordinary sender slot");
    DG_STATIC_ASSERT(!kPrefixWide || (kHidden == 4096 && kTokens % 8 == 0 &&
                     kFirstReadinessWindowBytes == 1024),
                     "Wide prefix validation requires aligned rows and a 1KB first window");
    DG_STATIC_ASSERT(!kSendMiddleFirst || kSenderLdg != 2 || kMiddlePayloadWindowEnd % 1024 == 0,
                     "Middle-first wide stores require a whole 1KB window boundary");
    constexpr uint32_t kLocalExperts = kExperts / kRanks;
    DG_STATIC_ASSERT(!kPrioritizeExperts ||
        layout::get_num_max_pool_tokens(kRanks, kTokens, kTopk, kLocalExperts) >= 2 * kTokens,
        "Banked sender queues exceed gather-index capacity");
    DG_STATIC_ASSERT(kCachePriority <= 4 && (!kCachePriority || kPrioritizeExperts) &&
        ((kCachePriority != 1 && kCachePriority != 3) ||
         layout::get_num_max_pool_tokens(kRanks, kTokens, kTopk, kLocalExperts) >= 4ull * kTokens),
        "Invalid priority cache mode or global key capacity");
    DG_STATIC_ASSERT(kCachePriority != 4 || (kTopk % 4 == 0 && kHidden % 1024 == 0), "Priority vector alignment");
    constexpr uint32_t kHistogramStride = math::constexpr_align<uint32_t>(kLocalExperts + 1, 32u);
    constexpr uint32_t kHistograms = kGroupPriority == 2 ? kWarps : 1;
    constexpr uint32_t kKeyOffset = kHistogramStride * kHistograms;
    DG_STATIC_ASSERT((kCachePriority < 2 && kGroupPriority != 2) ||
                     kKeyOffset < 32 + kHidden + kHidden / 32,
                     "Priority histogram exceeds dispatch scratch");
    DG_STATIC_ASSERT(kGroupPriority <= 2 && (!kGroupPriority || kPrioritizeExperts), "Priority histograms require prioritized queues");
    DG_STATIC_ASSERT(kRowPartitions >= 1 && kRowPartitions <= 8 &&
                     kMetadataExpertCTAs % kRowPartitions == 0, "Invalid row partitions");
    DG_STATIC_ASSERT(kRowPartitions == 1 || (!kRecycle &&
                     !kCoalescedSf), "Unsupported partitioned materializer");
    constexpr uint32_t kTileWorkers = kMetadataExpertCTAs / kRowPartitions;
    DG_STATIC_ASSERT(kTileWorkers > 0 &&
                     (kTileWorkers <= kLocalExperts || kTileWorkers % kLocalExperts == 0),
                     "Extra expert workers require complete disjoint tile partitions");
    constexpr uint32_t kMetadataCTAs = 2 * kMetadataExpertCTAs;
    constexpr uint32_t kSourceRows = 2 * kRanks * kTokens;
    const uint32_t lane = tid % 32, warp = tid / 32;
    const uint32_t cta = blockIdx.x;
    GENPHASE_MARK(GenphaseStart);
    const W ws(peers.get_base_ptr(), kRanks, kExperts, kTokens, kTopk);
    // Input-row releases cover routes/SF. Eight auxiliary selector CTAs add
    // destination-ready releases when recycling; sender acquisition joins both.
    // Metadata workers still acquire all-rank route publication independently.
    // Dedicated mode uses the freed dispatch warpgroup, allowing epilogue
    // work to overlap. Its named barriers must not alias the epilogue's 2/3/4.
    // Metadata consumers acquire the producer's SYS publications below. Payload
    // senders acquire its local input counter, independently of the network tail.
    constexpr uint32_t kMetadataThreads = kDedicatedDispatch ? 128 : 256;
    constexpr uint32_t kMetadataWarps = kMetadataThreads / 32;
    constexpr uint32_t kJoinBarrier = kDedicatedDispatch ? 5 : 3;
    constexpr uint32_t kSfWarps = kNarrowSf ? kNarrowSf : 4;
    constexpr uint32_t kSfThreads = kSfWarps * 32;
    constexpr uint32_t kMiddleFirstWarp = kDedicatedDispatch && !kNarrowSf ? 0 : kSfWarps;
    constexpr uint32_t kMiddleWarps = kMetadataWarps - kMiddleFirstWarp;
    constexpr uint32_t kSFBarrier = kDedicatedDispatch ? 6 : 4;
    constexpr uint32_t kMiddleBarrier = kDedicatedDispatch ? 7 : 5;
    const auto sync = [&]() {
        if (cta < kMetadataCTAs)
            asm volatile("bar.sync %0, %1;" :: "n"(kJoinBarrier), "n"(kMetadataThreads) : "memory");
        else
            asm volatile("bar.sync %0, 128;" :: "n"(kJoinBarrier) : "memory");
    };

    const auto* x = static_cast<const uint8_t*>(ws.get_end_ptr());
    const auto* sf = x + static_cast<uint64_t>(kTokens) * kHidden;
    const auto* topk = reinterpret_cast<const int64_t*>(sf + static_cast<uint64_t>(kTokens) * kHidden / 32);
    auto* data = static_cast<uint8_t*>(ws.get_compact_frame_base_ptr(kHidden, kTopk));
    auto* plain_sf = data + static_cast<uint64_t>(kSourceRows) * kHidden;
    auto* canonical_sf = reinterpret_cast<uint32_t*>(
        data + ws.get_compact_l1_num_tokens(kHidden) * kHidden);

    if (cta < kMetadataCTAs) {
        // The first expert-CTA group owns SF/index; the second validates A.
        // Each tile keeps independent SF/prefix/middle/tail publication.
        const bool owns_sf = cta < kMetadataExpertCTAs;
        const uint32_t physical_worker = owns_sf ? cta : cta - kMetadataExpertCTAs;
        const uint32_t worker = physical_worker / kRowPartitions;
        const uint32_t row_partition = physical_worker % kRowPartitions;
        const uint32_t first_expert = kTileWorkers > kLocalExperts ? worker % kLocalExperts : worker;
        const uint32_t first_tile = kTileWorkers > kLocalExperts ? worker / kLocalExperts : 0;
        constexpr uint32_t kExpertStride = kTileWorkers < kLocalExperts ? kTileWorkers : kLocalExperts;
        constexpr uint32_t kTileStride = kTileWorkers > kLocalExperts ? kTileWorkers / kLocalExperts : 1;
        GENPHASE_MARK(GenphaseMetadataFromProducer);
        // The router writes expert-local source descriptors and scores before
        // PDL, then publishes the same all-rank readiness used by pull.
        if (tid == 0)
            while (ptx::ld_acq_sys(ws.get_route_ready_count_ptr()) != kRanks);
        sync();
        GENPHASE_MARK(GenphaseTopkReceived);
        auto* histogram = reinterpret_cast<uint32_t*>(storage);
        auto* offsets = histogram + kLocalExperts;
        auto* total_padded = offsets + kLocalExperts;
        // One warp already owns all 16 counts in the EP8 case. Let it finish
        // the prefix before handing offsets to the eight SF/A worker warps.
        // Larger expert sets use the same 32-lane scan with a running carry.
        if (warp == 0) {
            for (uint32_t e = lane; e < kLocalExperts; e += 32) {
                const uint32_t count = static_cast<uint32_t>(*ws.get_expert_recv_count_sum_ptr(e));
                histogram[e] = math::align(count, BLOCK_M);
            }
            __syncwarp();
            if (cta == 0) {
                if (lane == 0)
                    asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;"
                                 :: "l"(ws.get_genphase_flag_ptr(W::SizesReady)) : "memory");
                GENPHASE_MARK(GenphaseSizesReady);
            }
            uint32_t carry = 0;
            for (uint32_t start = 0; start < kLocalExperts; start += 32) {
                const uint32_t e = start + lane;
                const uint32_t padded = e < kLocalExperts ? histogram[e] : 0;
                uint32_t sum = padded;
                #pragma unroll
                for (uint32_t step = 1; step < 32; step *= 2) {
                    const uint32_t prev = __shfl_up_sync(0xffffffff, sum, step);
                    if (lane >= step) sum += prev;
                }
                if (e < kLocalExperts) offsets[e] = carry + sum - padded;
                carry += __shfl_sync(0xffffffff, sum, 31);
            }
            if (lane == 0) *total_padded = carry;
        }
        sync();
        const uint32_t carry = *total_padded;
        // The SF role materializes scales once per tile; route readiness covers
        // its source bytes. The independent A role validates transport words.
        // Materialized tiles use this rank's unused source-row capacity.
        // Active source rows [0,num_tokens) remain dedicated to wire payloads.
        // Reject insufficient capacity instead of aliasing live receive rows.
        if (num_tokens > kTokens || carry > kTokens - num_tokens)
            asm volatile("trap;");
        const uint64_t materialized_base = ws.get_source_l1_row(peers.rank_idx, num_tokens);
        constexpr uint32_t kSFBlock = math::constexpr_align(BLOCK_M, 128u);
        bool first_validation = true;
        bool first_remainder = true;
        bool first_sf = true;
        bool first_tail = true;
        if (owns_sf) GENPHASE_MARK(GenphaseIndexScatter);
        // Tile workers stride over experts/tiles. Row partitions own disjoint
        // four-bit lanes; even an empty partition must publish its certificates.
        for (uint32_t e = first_expert; e < kLocalExperts; e += kExpertStride) {
            const uint32_t count = static_cast<uint32_t>(*ws.get_expert_recv_count_sum_ptr(e));
            for (uint32_t base = first_tile * BLOCK_M; base < count; base += kTileStride * BLOCK_M) {
                const uint32_t ready = min(count - base, BLOCK_M);
                const uint32_t partition_width = math::align(
                    (ready + kRowPartitions - 1) / kRowPartitions, kMetadataThreads / 16);
                const uint32_t row_begin = kRowPartitions == 1 ? 0 : min(row_partition * partition_width, ready);
                const uint32_t partition_rows = kRowPartitions == 1 ? ready : min(partition_width, ready - row_begin);
                const uint32_t block = (offsets[e] + base) / BLOCK_M;
                if (owns_sf) {
                    // Scores remain in the producer bank. Optional return-address
                    // caching shares this tile's SF-ready publication.
                    // SF moves H/32 bytes per row; the middle window moves1024.
                    // Optionally give SF one to three warps and middle the rest.
                    // Default dedicated mode uses all four warps sequentially.
                    if (warp < kSfWarps) {
                        constexpr uint32_t kChunksPerRow = kHidden / 512;
                        constexpr uint32_t kChunksPerTask = kCoalescedSf == 2 ? kChunksPerRow : 1;
                        const uint32_t work = (kCoalescedSf ? kSFBlock : partition_rows) *
                                              (kChunksPerRow / kChunksPerTask);
                        auto* destination = canonical_sf + static_cast<uint64_t>(block) * kSFBlock * (kHidden / 128);
                        // Uniform loop bound preserves a full-warp readiness vote even
                        // for partial rows/chunks; inactive lanes never access payload.
                        for (uint32_t batch = 0; batch < work; batch += kSfThreads) {
                            const uint32_t task = batch + tid;
                            uint32_t i = row_begin + task / kChunksPerRow;
                            uint32_t atom = (task % kChunksPerRow) * 4;
                            if constexpr (kCoalescedSf) {
                                // Visit contiguous physical SF rows per store. Invert
                                // the 128-row swizzle; padded rows never read or write.
                                const uint32_t physical = task % kSFBlock;
                                i = (physical & ~127u) + ((physical & 3u) << 5) +
                                    ((physical & 127u) >> 2);
                                atom = kCoalescedSf == 2 ? 0 : (task / kSFBlock) * 4;
                            }
                            const bool active = task < work && (!kCoalescedSf || i < ready);
                            uint32_t source = 0;
                            if (active) {
                                const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                                    e, kRanks == 1 ? 0 : ws.metadata_slot, base + i);
                                if constexpr (kCacheDestinations) {
                                    if (atom == 0) {
                                        const uint32_t rank = descriptor / (kTokens * kTopk);
                                        const uint32_t route = descriptor % (kTokens * kTopk);
                                        const auto token = combine_token_buffer
                                            .get_rank_buffer(ws.metadata_slot * kTopk + route % kTopk)
                                            .get_data_buffer(route / kTopk);
                                        ws.get_cached_destinations_ptr()[block * BLOCK_M + i] =
                                            reinterpret_cast<uint64_t>(peers.map(token.get_base_ptr(), rank));
                                    }
                                }
                                if constexpr (kLinearSource) {
                                    source = ws.get_source_l1_row(descriptor / kTopk);
                                } else {
                                    const uint32_t rank = descriptor / (kTokens * kTopk);
                                    const uint32_t route = descriptor % (kTokens * kTopk);
                                    source = ws.get_source_l1_row(rank, route / kTopk);
                                }
                            }
                            // The acquired producer route-ready release covers SF.
                            // A readiness is published separately by its validator CTA.
                            if (active) {
                                const uint32_t physical = (i & ~127u) + (i & 31u) * 4 + ((i >> 5) & 3u);
                                const auto* source_sf = reinterpret_cast<const uint4*>(
                                    plain_sf + static_cast<uint64_t>(source) * (kHidden / 32));
                                // Row tasks reuse one descriptor/source address for
                                // every chunk while retaining coalesced SF stores.
                                #pragma unroll
                                for (uint32_t chunk = 0; chunk < kChunksPerTask; ++chunk) {
                                    const uint4 values = source_sf[atom / 4 + chunk];
                                    destination[(atom + chunk * 4 + 0) * kSFBlock + physical] = values.x;
                                    destination[(atom + chunk * 4 + 1) * kSFBlock + physical] = values.y;
                                    destination[(atom + chunk * 4 + 2) * kSFBlock + physical] = values.z;
                                    destination[(atom + chunk * 4 + 3) * kSFBlock + physical] = values.w;
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("bar.sync %0, %1;" :: "n"(kSFBarrier), "n"(kSfThreads) : "memory");
                        if (tid == 0)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchSfReady << (4 * row_partition)) : "memory");
                        if (first_sf) GENPHASE_MARK(GenphaseSfReadyIssue);
                    }
                    if (warp >= kMiddleFirstWarp) {
                        // Bit4 certifies only [1024, min(kMiddlePayloadWindowEnd,H));
                        // prefix and tail have separate owners and certificates.
                        if (first_remainder) GENPHASE_MARK_WARP(GenphaseRemainderValidateStart, kMiddleFirstWarp);
                        validate_window_rows<kTokens, kHidden, kTopk, kFirstReadinessWindowBytes,
                                               (kHidden < kMiddlePayloadWindowEnd ? kHidden : kMiddlePayloadWindowEnd),
                                               kMiddleWarps, kLinearSource, kHalfWarp, kStickyValidation>(
                            ws.get_src_token_topk_idx_ptr(e, kRanks == 1 ? 0 : ws.metadata_slot, base + row_begin),
                            data + ws.get_source_l1_row(0, 0) * kHidden,
                            data + (materialized_base + offsets[e] + base + row_begin) * kHidden,
                            partition_rows, warp - kMiddleFirstWarp, lane);
                        // TMA will read receiver-owned materialized bytes, never reread
                        // the concurrently written source inbox after validation.
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("bar.sync %0, %1;" :: "n"(kMiddleBarrier), "n"(kMiddleWarps * 32) : "memory");
                        if (tid == kMiddleFirstWarp * 32)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchMiddleAReady << (4 * row_partition)) : "memory");
                        if (first_remainder) GENPHASE_MARK_WARP(GenphaseMiddlePayloadIssue, kMiddleFirstWarp);
                        first_remainder = false;
                    }
                    first_sf = false;
                } else {
                    // No SF or remainder dependency: validate/materialize the prefix
                    // as soon as its transport words arrive, then publish bit2.
                    if (first_validation) GENPHASE_MARK(GenphaseWindowValidateStart);
                    // Validate the first window across all rows before later K bytes.
                    // Sixteen lanes own each row; issue independent vectors together.
                    // A full-warp vote includes both rows, including partial tails.
                    for (uint32_t batch = 0; batch < partition_rows * 16; batch += kMetadataThreads) {
                        const uint32_t task = batch + tid;
                        const bool active = task < partition_rows * 16;
                        const uint32_t i = row_begin + task / 16, chunk = task % 16;
                        const uint4* input = nullptr;
                        uint4* output = nullptr;
                        if (active) {
                            const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                                e, kRanks == 1 ? 0 : ws.metadata_slot, base + i);
                            if constexpr (kLinearSource) {
                                input = reinterpret_cast<const uint4*>(data +
                                    ws.get_source_l1_row(descriptor / kTopk) * kHidden);
                            } else {
                                const uint32_t rank = descriptor / (kTokens * kTopk);
                                const uint32_t token = (descriptor % (kTokens * kTopk)) / kTopk;
                                input = reinterpret_cast<const uint4*>(data +
                                    ws.get_source_l1_row(rank, token) * kHidden);
                            }
                            output = reinterpret_cast<uint4*>(data +
                                (materialized_base + offsets[e] + base + i) * kHidden);
                        }
                        constexpr uint32_t kFirstVectors = kFirstReadinessWindowBytes / (16 * 16);
                        // Wide pairs cover [lane*32 + pair*512, +32). Retain a
                        // pair only when all eight words pass; no vector atomicity assumed.
                        constexpr uint32_t kVectorStep = kPrefixWide ? 2 : 1;
                        uint4 values[kFirstVectors];
                        uint32_t complete = 0;
                        bool pending;
                        do {
                            if (active) {
                                #pragma unroll
                                for (uint32_t v = 0; v < kFirstVectors; v += kVectorStep)
                                    if (!kStickyValidation || !(complete & (1u << v))) {
                                        if constexpr (kPrefixWide)
                                            asm volatile("ld.relaxed.sys.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                 : "=r"(values[v].x), "=r"(values[v].y), "=r"(values[v].z), "=r"(values[v].w),
                                                   "=r"(values[v + 1].x), "=r"(values[v + 1].y),
                                                   "=r"(values[v + 1].z), "=r"(values[v + 1].w)
                                                 : "l"(input + chunk * 2 + v * 16) : "memory");
                                        else
                                            asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                                 : "=r"(values[v].x), "=r"(values[v].y),
                                                   "=r"(values[v].z), "=r"(values[v].w)
                                                 : "l"(input + chunk + v * 16) : "memory");
                                    }
                            }
                            pending = false;
                            if (active) {
                                #pragma unroll
                                for (uint32_t v = 0; v < kFirstVectors; v += kVectorStep) {
                                    bool missing = values[v].x == 0x80000000u || values[v].y == 0x80000000u ||
                                                   values[v].z == 0x80000000u || values[v].w == 0x80000000u;
                                    if constexpr (kPrefixWide)
                                        missing |= values[v + 1].x == 0x80000000u || values[v + 1].y == 0x80000000u ||
                                                   values[v + 1].z == 0x80000000u || values[v + 1].w == 0x80000000u;
                                    pending |= missing;
                                    if constexpr (kStickyValidation)
                                        if (!missing) complete |= 1u << v;
                                }
                            }
                        } while (__any_sync(0xffffffff, pending));
                        if (active) {
                            #pragma unroll
                            for (uint32_t v = 0; v < kFirstVectors; v += kVectorStep) {
                                if constexpr (kPrefixWide)
                                    asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::
                                                 "l"(output + chunk * 2 + v * 16),
                                                 "r"(values[v].x), "r"(values[v].y), "r"(values[v].z), "r"(values[v].w),
                                                 "r"(values[v + 1].x), "r"(values[v + 1].y),
                                                 "r"(values[v + 1].z), "r"(values[v + 1].w) : "memory");
                                else
                                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                             "l"(output + chunk + v * 16), "r"(values[v].x), "r"(values[v].y),
                                             "r"(values[v].z), "r"(values[v].w) : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    sync();
                    if (tid == 0)
                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                     :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchFirstAReady << (4 * row_partition)) : "memory");
                    if (first_validation) GENPHASE_MARK(GenphaseWindowPayloadIssue);
                    first_validation = false;
                    if constexpr (kHidden > kMiddlePayloadWindowEnd) {
                        // Prefix has already been certified. Its eight warps now
                        // validate the disjoint tail while consumers use earlier K.
                        if (first_tail) GENPHASE_MARK(GenphaseWindowValidateStart);
                        validate_window_rows<kTokens, kHidden, kTopk,
                                             kMiddlePayloadWindowEnd, kHidden, kMetadataWarps, kLinearSource, kHalfWarp, kStickyValidation>(
                            ws.get_src_token_topk_idx_ptr(e, kRanks == 1 ? 0 : ws.metadata_slot, base + row_begin),
                            data + ws.get_source_l1_row(0, 0) * kHidden,
                            data + (materialized_base + offsets[e] + base + row_begin) * kHidden,
                            partition_rows, warp, lane);
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        sync();
                        if (tid == 0)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchTailAReady << (4 * row_partition)) : "memory");
                        if (first_tail) GENPHASE_MARK(GenphaseWindowPayloadIssue);
                        first_tail = false;
                    }
                }
            }
        }
        sync();
        if (tid == 0)
            asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;"
                         :: "l"(ws.get_genphase_flag_ptr(W::DispatchDone)) : "memory");
        GENPHASE_MARK(GenphaseEnd);
        return;
    }

    // Recycled input readiness joins physical rows and eight auxiliary CTAs.
    // This acquire replaces a wait for the entire selector's network tail.
    if (lane == 0)
        while (ptx::ld_acq(ws.get_genphase_flag_ptr(W::CountsReady)) !=
               num_tokens + (kRecycle ? 8u : 0u));
    __syncwarp();
    // Compose the acquired generic writes with each issuer's async TMA reads.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    GENPHASE_MARK(GenphaseInputsReady);

    if constexpr (kPrioritizeExperts) {
        if (cta == kMetadataCTAs) {
            // Reuse the idle helper and unused gather-index arena. Queue banks
            // follow the route generation's existing retirement contract.
            // Wire addresses and receiver route order do not change.
            auto* offsets = reinterpret_cast<uint32_t*>(storage);
            auto* warp_offsets = offsets + (kGroupPriority == 2 ? warp * kHistogramStride : 0);
            // Each thread reloads its own keys. Mode1 uses retired global banks;
            // Modes2/3/4 use private CTA scratch, disjoint from offsets and GEMM.
            // Mode3 reads selector-prepared keys after the CountsReady acquire.
            // The host bounds actual rows against scratch capacity for modes2/3.
            auto* keys = kCachePriority >= 2 ? offsets + kKeyOffset :
                ws.get_gather_index_ptr((2 + ws.metadata_slot) * kTokens);
            const auto priority = [&](uint32_t token) {
                if constexpr (kCachePriority == 3)
                    return ws.get_gather_index_ptr((2 + ws.metadata_slot) * kTokens)[token];
                uint32_t key = kLocalExperts;
                if constexpr (kCachePriority == 4) {
                    // Read four aligned IDs while preserving full-width validity.
                    constexpr uint32_t kIdsPerLoad = 4;
                    #pragma unroll
                    for (uint32_t k = 0; k < kTopk; k += kIdsPerLoad) {
                        uint32_t words[2 * kIdsPerLoad];
                        asm volatile("ld.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]),
                              "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7])
                            : "l"(topk + token * kTopk + k) : "memory");
                        #pragma unroll
                        for (uint32_t i = 0; i < kIdsPerLoad; ++i)
                            if (words[2 * i + 1] == 0 && words[2 * i] < kExperts)
                                key = min(key, words[2 * i] % kLocalExperts);
                    }
                    return key;
                }
                #pragma unroll
                for (uint32_t k = 0; k < kTopk; ++k) {
                    const int64_t expert = topk[token * kTopk + k];
                    if (expert >= 0 && expert < kExperts)
                        key = min(key, static_cast<uint32_t>(expert) % kLocalExperts);
                }
                return key;
            };
            const auto claim_priority = [&](uint32_t key, uint32_t active) {
                const uint32_t group = __match_any_sync(active, key);
                const uint32_t leader = __ffs(group) - 1;
                uint32_t base = 0;
                if (lane == leader) base = atomicAdd(offsets + key, __popc(group));
                return __shfl_sync(active, base, leader) + __popc(group & ((1u << lane) - 1));
            };
            constexpr uint32_t kHistogramWords = kGroupPriority == 2 ? kKeyOffset : kLocalExperts + 1;
            for (uint32_t e = tid; e < kHistogramWords; e += kThreads) offsets[e] = 0;
            sync();
            if constexpr (kGroupPriority == 1) {
                // Warp-uniform rounds keep partial tails in the ballot. Only
                // valid lanes enter match/shuffle with that exact active mask.
                for (uint32_t first = warp * 32; first < num_tokens; first += kThreads) {
                    const uint32_t token = first + lane;
                    const uint32_t active = __ballot_sync(0xffffffffu, token < num_tokens);
                    if (token < num_tokens) {
                        const uint32_t key = priority(token);
                        if constexpr (kCachePriority) keys[token] = key;
                        claim_priority(key, active);
                    }
                }
            } else {
                for (uint32_t token = tid; token < num_tokens; token += kThreads) {
                    const uint32_t key = priority(token);
                    if constexpr (kCachePriority) keys[token] = key;
                    atomicAdd(warp_offsets + key, 1u);
                }
            }
            sync();
            if (tid == 0) {
                uint32_t total = 0;
                for (uint32_t e = 0; e <= kLocalExperts; ++e) {
                    // Expert-major, then warp-major spans preserve priorities
                    // while keeping each warp's ordinary atomic claims private.
                    #pragma unroll
                    for (uint32_t w = 0; w < kHistograms; ++w) {
                        auto& offset = offsets[w * kHistogramStride + e];
                        const uint32_t count = offset;
                        offset = total;
                        total += count;
                    }
                }
            }
            sync();
            if constexpr (kGroupPriority == 1) {
                for (uint32_t first = warp * 32; first < num_tokens; first += kThreads) {
                    const uint32_t token = first + lane;
                    const uint32_t active = __ballot_sync(0xffffffffu, token < num_tokens);
                    if (token < num_tokens)
                        ws.get_gather_index_ptr(ws.metadata_slot * kTokens)[
                            claim_priority(kCachePriority ? keys[token] : priority(token), active)] = token;
                }
            } else {
                for (uint32_t token = tid; token < num_tokens; token += kThreads)
                    ws.get_gather_index_ptr(ws.metadata_slot * kTokens)[
                        atomicAdd(warp_offsets + (kCachePriority ? keys[token] : priority(token)), 1u)] = token;
            }
            __threadfence();
            sync();
            if (tid == 0)
                asm volatile("st.release.gpu.global.u32 [%0], 1;"
                             :: "l"(ws.get_genphase_flag_ptr(W::IndicesReady)) : "memory");
        } else {
            if (lane == 0)
                while (ptx::ld_acq(ws.get_genphase_flag_ptr(W::IndicesReady)) != 1u);
            __syncwarp();
        }
    }

    if (cta > kMetadataCTAs) {
        // One slot per warp fits the disjoint dispatch region. The fixed EP8
        // case sends one token per warp; no lookahead slot is needed. Reuse on
        // later iterations follows this warp's completed local shared reads.
        constexpr uint32_t kRowBytes = kHidden;
        constexpr uint32_t kTokenStride = kPushers * kWarps;
        auto* barriers = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(storage);
        auto* stage = storage + 128;
        if (!kSenderLdg && tid < kWarps * kDepth)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(cute::cast_smem_ptr_to_uint(barriers + tid)) : "memory");
        if constexpr (!kSenderLdg)
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        sync();
        // Classify once while the row TMA is in flight. Each mask word is
        // warp-uniform, so fanout needs no per-slot shuffle or vote.
        constexpr uint32_t kMaskWords = (kRanks + 31) / 32;
        struct Route { uint32_t peers[kMaskWords]; };
        const auto source_token = [&](uint32_t position) {
            if constexpr (kPrioritizeExperts)
                return ws.get_gather_index_ptr(ws.metadata_slot * kTokens)[position];
            else return position;
        };
        const auto prefetch = [&](uint32_t position, uint32_t slot,
                                  uint32_t begin, uint32_t bytes) {
            const uint32_t token = source_token(position);
            // Every invocation names a valid local input row. Issue its load
            // before route lookup/dedup so that work can overlap TMA progress.
            // Empty routes still wait for this load but issue no payload stores.
            if (!kSenderLdg && lane == 0) {
                auto* tile = stage + slot * kRowBytes + begin;
                ptx::mbarrier_arrive_and_set_tx(barriers + slot, bytes);
                ptx::tma_load_1d(tile, x + static_cast<uint64_t>(token) * kHidden + begin, barriers + slot, bytes);
            }
            const int64_t expert = lane < kTopk ? topk[token * kTopk + lane] : -1;
            const int destination = expert >= 0 && expert < kExperts ? expert / kLocalExperts : -1;
            Route route{};
            #pragma unroll
            for (uint32_t word = 0; word < kMaskWords; ++word) {
                const uint32_t bit = destination >= 0 &&
                    uint32_t(destination) / 32 == word
                    ? (1u << (uint32_t(destination) % 32)) : 0u;
                route.peers[word] = __reduce_or_sync(0xffffffff, bit);
            }
            return route;
        };
        const uint32_t first_base = (cta - kMetadataCTAs - 1) * kWarps;
        const uint32_t first_token = first_base + warp;
        // Each warp sweeps prefixes before tails; no inter-warp wait is added.
        // Count only this warp's real loads when carrying its barrier phase.
        const uint32_t warp_rows = num_tokens > first_token
            ? (num_tokens - first_token - 1) / kTokenStride + 1 : 0;
        #pragma unroll
        for (uint32_t pass = 0; pass < (kPrefixSweep ? 2u : 1u); ++pass) {
            // The selector owns the prefix when enabled; native writers own
            // only the tail. Metadata retirement acquires selector completion.
            const uint32_t begin = kProducerPrefix || (kPrefixSweep && pass == 1) ? kFirstPayloadWindowBytes : 0;
            const uint32_t end = kPrefixSweep && pass == 0 ? kFirstPayloadWindowBytes : kHidden;
            if (first_base < num_tokens) {
                if (pass == 0) GENPHASE_MARK(MegaPayloadLoadStart);
                Route current = first_token < num_tokens ?
                    prefetch(first_token, warp * kDepth, begin, end - begin) : Route{};
                Route next{};
                for (uint32_t position = first_token, step = 0; first_base + step * kTokenStride < num_tokens; position += kTokenStride, ++step) {
                    const uint32_t slot = warp * kDepth + step % kDepth;
                    auto* tile = stage + slot * kRowBytes;
                    uint32_t phase = (step / kDepth + (kPrefixSweep ? pass * warp_rows : 0)) & 1;
                    if (!kSenderLdg && position < num_tokens)
                        ptx::mbarrier_wait_and_flip_phase(barriers + slot, phase);
                    uint32_t any_peer = 0;
                    #pragma unroll
                    for (uint32_t word = 0; word < kMaskWords; ++word)
                        any_peer |= current.peers[word];
                    if (any_peer) {
                        const uint32_t token = source_token(position);
                        const uint64_t row = ws.get_source_l1_row(peers.rank_idx, token);
                        constexpr uint32_t kVectors = kHidden / 512;
                        constexpr uint32_t kVectorStep = kSenderLdg == 2 ? 2 : 1;
                        uint4 values[kVectors];
                        #pragma unroll
                        for (uint32_t v = begin / 512; v < end / 512; v += kVectorStep) {
                            if constexpr (kSenderLdg == 2) {
                                // Pair adjacent 16B vectors per lane; one warp
                                // covers each 1KB window with aligned 32B accesses.
                                const auto* source = x + static_cast<uint64_t>(token) *
                                    kHidden + lane * 32 + v * 512;
                                asm volatile("ld.global.cg.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                             : "=r"(values[v].x), "=r"(values[v].y),
                                               "=r"(values[v].z), "=r"(values[v].w),
                                               "=r"(values[v + 1].x), "=r"(values[v + 1].y),
                                               "=r"(values[v + 1].z), "=r"(values[v + 1].w)
                                             : "l"(source) : "memory");
                            } else if constexpr (kSenderLdg) {
                                // CountsReady acquired producer writes before
                                // these L2 reads. No async/shared staging here.
                                const auto* source = x + static_cast<uint64_t>(token) *
                                    kHidden + lane * 16 + v * 512;
                                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                                             : "=r"(values[v].x), "=r"(values[v].y),
                                               "=r"(values[v].z), "=r"(values[v].w)
                                             : "l"(source) : "memory");
                            } else {
                                const uint32_t source = cute::cast_smem_ptr_to_uint(tile + lane * 16);
                                asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                                             : "=r"(values[v].x), "=r"(values[v].y),
                                               "=r"(values[v].z), "=r"(values[v].w)
                                             : "r"(source + v * 512) : "memory");
                            }
                        }
                        if constexpr (kPrefetchSend) {
                            // All lanes have consumed this warp's private shared
                            // slot. Refill it while current payload stores use only
                            // registers; the next iteration waits its normal phase.
                            __syncwarp();
                            next = position + kTokenStride < num_tokens ?
                                prefetch(position + kTokenStride, warp * kDepth, begin, end - begin) : Route{};
                        }
                        #pragma unroll
                        for (uint32_t v = begin / 512; v < end / 512; ++v) {
                            values[v].x = values[v].x == 0x80000000u ? 0 : values[v].x;
                            values[v].y = values[v].y == 0x80000000u ? 0 : values[v].y;
                            values[v].z = values[v].z == 0x80000000u ? 0 : values[v].z;
                            values[v].w = values[v].w == 0x80000000u ? 0 : values[v].w;
                        }
                        // Release the first consumable K window toward every peer
                        // before issuing unrelated remainder bytes. Keep the same
                        // deduplicated peers, exact encoded values and store owners.
                        // No completion wait or new synchronization between windows.
                        #pragma unroll
                        for (uint32_t window = 0; window < (kSendMiddleFirst ? 3u : 2u); ++window) {
                            // Optionally issue middle stores to every peer before
                            // tail stores; retain one writer per peer/word.
                            const uint32_t window_begin = window == 0 ? 0u :
                                window == 1 ? kFirstPayloadWindowBytes / 512 : kMiddlePayloadWindowEnd / 512;
                            const uint32_t window_end = window == 0 ? kFirstPayloadWindowBytes / 512 :
                                kSendMiddleFirst && window == 1 ? kMiddlePayloadWindowEnd / 512 : kVectors;
                            const uint32_t first = max(begin / 512, window_begin);
                            const uint32_t last = min(end / 512, window_end);
                            if (first >= last) continue;
                            #pragma unroll
                            for (uint32_t word = 0; word < kMaskWords; ++word) {
                                uint32_t pending_peers = current.peers[word];
                                while (pending_peers) {
                                    const uint32_t peer = word * 32 + __ffs(pending_peers) - 1;
                                    pending_peers &= pending_peers - 1;
                                    auto* output = peers.map(data + row * kHidden + lane * 16 * kVectorStep, peer);
                                    #pragma unroll
                                    for (uint32_t v = first; v < last; v += kVectorStep) {
                                        if constexpr (kSenderLdg == 2) {
                                            asm volatile("st.relaxed.sys.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::
                                                "l"(output + v * 512), "r"(values[v].x), "r"(values[v].y),
                                                "r"(values[v].z), "r"(values[v].w),
                                                "r"(values[v + 1].x), "r"(values[v + 1].y),
                                                "r"(values[v + 1].z), "r"(values[v + 1].w) : "memory");
                                        } else {
                                            asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                                "l"(output + v * 512), "r"(values[v].x), "r"(values[v].y),
                                                "r"(values[v].z), "r"(values[v].w) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (step == 0 && pass == 0) GENPHASE_MARK(GenphasePayloadStoresIssued);
                    // Stores source registers, not the shared slot. Once all shared
                    // reads are issued this warp may refill its private load slot.
                    __syncwarp();
                    // A partial final iteration must not reuse a prior route in
                    // inactive warps: its output row aliases materialized storage.
                    if (!kPrefetchSend || !any_peer)
                        next = position + kTokenStride < num_tokens ?
                            prefetch(position + kTokenStride, warp * kDepth, begin, end - begin) : Route{};
                    current = next;
                }
            }
        }
        // Retirement alone waits for remote store completion. Receiver
        // validation/local tile release independently admits gather/MMA.
        sync();
        if (tid == 0) __threadfence_system();
        sync();
        // Completion only; readiness travels in each payload word.
        if (!kSenderLdg && tid < kWarps * kDepth)
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                         :: "r"(cute::cast_smem_ptr_to_uint(barriers + tid)) : "memory");
        sync();
    }

    // External buffer preparation orders all previous invocations before reuse.
    // Every participant retires, including the helper and empty pushers.
    // The compute dispatch warps acquire all retirements before arena cleanup.
    if (tid == 0)
        asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;"
                     :: "l"(ws.get_genphase_flag_ptr(W::DispatchDone)) : "memory");
    GENPHASE_MARK(GenphaseEnd);
#endif
}

} // namespace deep_gemm

#undef GENPHASE_MARK_ALL
#undef GENPHASE_MARK_WARP
#undef GENPHASE_MARK
