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
inline constexpr uint32_t kMiddlePayloadWindowEnd = 2048;
// Each disjoint window has exactly one publisher per tile. Prefix/middle/tail
// certify only their own materialized bytes. Consumers acquire each certificate
// before issuing a TMA that reads that window; SF remains independent.
inline constexpr uint32_t kDispatchSfReady = 1u;
inline constexpr uint32_t kDispatchFirstAReady = 2u;
inline constexpr uint32_t kDispatchMiddleAReady = 4u;
inline constexpr uint32_t kDispatchTailAReady = 8u;

// Keep the polling implementation outside the main kernel's optimizer region.
// No timing markers live here: all phase boundaries stay in the caller.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kTopk,
          uint32_t kBegin, uint32_t kEnd, uint32_t kOwners>
__device__ __noinline__ void validate_window_rows(
    const uint32_t* descriptors, const uint8_t* input_base, uint8_t* output_base,
    uint32_t ready, uint32_t owner, uint32_t lane) {
    // Keep both row load batches in flight, with one uniform
    // polling loop. The tile certificate requires both rows;
    // store the exact final observations after the joint vote.
    DG_STATIC_ASSERT(kBegin < kEnd && kEnd <= kHidden &&
                     kBegin % 512 == 0 && kEnd % 512 == 0, "Complete disjoint row windows required");
    DG_STATIC_ASSERT(kOwners == 4 || kOwners == 8, "Window owner count must match its barrier");
    constexpr uint32_t kRows = 2;
    constexpr uint32_t kVectors = (kEnd - kBegin) / 512;
    for (uint32_t first = owner; first < ready; first += kOwners * kRows) {
        const uint4* input[kRows];
        uint4* output[kRows];
        uint4 values[kRows][kVectors];
        #pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            const uint32_t i = first + kOwners * r;
            if (i < ready) {
                const uint32_t descriptor = descriptors[i];
                const uint32_t rank = descriptor / (kTokens * kTopk);
                const uint32_t token = (descriptor % (kTokens * kTopk)) / kTopk;
                const uint64_t source = static_cast<uint64_t>(rank) * kTokens + token;
                input[r] = reinterpret_cast<const uint4*>(input_base + source * kHidden);
                output[r] = reinterpret_cast<uint4*>(output_base + static_cast<uint64_t>(i) * kHidden);
            }
        }
        bool missing;
        do {
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                if (first + kOwners * r < ready) {
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v)
                        asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(values[r][v].x), "=r"(values[r][v].y),
                                       "=r"(values[r][v].z), "=r"(values[r][v].w)
                                     : "l"(input[r] + lane + kBegin / 16 + v * 32) : "memory");
                }
            }
            missing = false;
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                if (first + kOwners * r < ready) {
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v)
                        missing |= values[r][v].x == 0x80000000u || values[r][v].y == 0x80000000u ||
                                   values[r][v].z == 0x80000000u || values[r][v].w == 0x80000000u;
                }
            }
        } while (__any_sync(0xffffffff, missing));
        #pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            if (first + kOwners * r < ready) {
                #pragma unroll
                for (uint32_t v = 0; v < kVectors; ++v)
                    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                 "l"(output[r] + lane + kBegin / 16 + v * 32),
                                 "r"(values[r][v].x), "r"(values[r][v].y),
                                 "r"(values[r][v].z), "r"(values[r][v].w) : "memory");
            }
        }
    }
}

// All-push dispatch: producer-owned routes/SF, deduplicated activation sends,
// and receiver-owned materialization with independent K-window readiness.
// FC1 consumes materialized rows through regular eight-row TMA.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kExperts, uint32_t kTopk,
          uint32_t kRanks, uint32_t BLOCK_M, uint32_t kSFPoolTokens, uint32_t kPushers, bool kRecycle = false>
__device__ __forceinline__ void mega_moe_dispatch(
    uint32_t num_tokens, const layout::SymBuffer<kRanks>& peers, uint8_t* storage,
    uint32_t tid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    using W = layout::Workspace;
    DG_STATIC_ASSERT(kHidden > kFirstPayloadWindowBytes && kHidden % 512 == 0 &&
                     kFirstPayloadWindowBytes % 512 == 0, "Readiness windows require complete 512B chunks");
    constexpr uint32_t kThreads = 128, kWarps = 4, kDepth = 1;
    constexpr uint32_t kLocalExperts = kExperts / kRanks;
    constexpr uint32_t kMetadataExpertCTAs = kLocalExperts < 2 * kRanks ? kLocalExperts : 2 * kRanks;
    constexpr uint32_t kMetadataCTAs = 2 * kMetadataExpertCTAs;
    constexpr uint32_t kSourceRows = 2 * kRanks * kTokens;
    const uint32_t lane = tid % 32, warp = tid / 32;
    const uint32_t cta = blockIdx.x;
    GENPHASE_MARK(GenphaseStart);
    const W ws(peers.get_base_ptr(), kRanks, kExperts, kTokens, kTopk);
    // Input-row releases cover routes/SF. Eight auxiliary selector CTAs add
    // destination-ready releases when recycling; sender acquisition joins both.
    // Metadata workers still acquire all-rank route publication independently.
    // Separate SF/index and A-validator CTAs each use both epilogue warpgroups.
    // Senders and the input helper use the first only.
    // Metadata consumers acquire the producer's SYS publications below. Payload
    // senders acquire its local input counter, independently of the network tail.
    const auto sync = [&]() {
        if (cta < kMetadataCTAs)
            asm volatile("bar.sync 3, 256;" ::: "memory");
        else
            asm volatile("bar.sync 3, 128;" ::: "memory");
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
        // Keep all eight validator warps and independent publication domains.
        // All 256 participants use barrier3 until dispatch returns; epilogue
        // later reuses its normal per-warpgroup barriers after this last join.
        constexpr uint32_t kMetadataThreads = 256, kMetadataWarps = 8;
        const bool owns_sf = cta < kMetadataExpertCTAs;
        const uint32_t first_expert = owns_sf ? cta : cta - kMetadataExpertCTAs;
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
        for (uint32_t e = first_expert; e < kLocalExperts; e += kMetadataExpertCTAs) {
            const uint32_t count = static_cast<uint32_t>(*ws.get_expert_recv_count_sum_ptr(e));
            for (uint32_t base = 0; base < count; base += BLOCK_M) {
                const uint32_t ready = min(count - base, BLOCK_M);
                const uint32_t block = (offsets[e] + base) / BLOCK_M;
                if (owns_sf) {
                    // The epilogue reads the original producer route/score banks;
                    // no local metadata copy or second ready publication is needed.
                    DG_STATIC_ASSERT(BLOCK_M <= 32, "SF role handles at most32 rows");
                    // Four SF warps retain the same fixed-case task mapping.
                    // For larger valid tiles they process successive 128-chunk
                    // batches. Barrier4 belongs only to these 128 participants;
                    // barrier3 remains the full worker initialization/retirement join.
                    if (warp < 4) {
                        constexpr uint32_t kChunksPerRow = kHidden / 512;
                        const uint32_t work = ready * kChunksPerRow;
                        auto* destination = canonical_sf + static_cast<uint64_t>(block) * kSFBlock * (kHidden / 128);
                        // Uniform loop bound preserves a full-warp readiness vote even
                        // for partial rows/chunks; inactive lanes never access payload.
                        for (uint32_t batch = 0; batch < work; batch += 128) {
                            const uint32_t task = batch + tid;
                            const bool active = task < work;
                            const uint32_t i = task / kChunksPerRow;
                            const uint32_t atom = (task % kChunksPerRow) * 4;
                            uint32_t source = 0;
                            if (active) {
                                const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                                    e, kRanks == 1 ? 0 : ws.metadata_slot, base + i);
                                const uint32_t rank = descriptor / (kTokens * kTopk);
                                const uint32_t route = descriptor % (kTokens * kTopk);
                                source = ws.get_source_l1_row(rank, route / kTopk);
                            }
                            // The acquired producer route-ready release covers SF.
                            // A readiness is published separately by its validator CTA.
                            if (active) {
                                const uint32_t physical = (i & ~127u) + (i & 31u) * 4 + ((i >> 5) & 3u);
                                const auto* source_sf = reinterpret_cast<const uint4*>(
                                    plain_sf + static_cast<uint64_t>(source) * (kHidden / 32));
                                const uint4 values = source_sf[atom / 4];
                                destination[(atom + 0) * kSFBlock + physical] = values.x;
                                destination[(atom + 1) * kSFBlock + physical] = values.y;
                                destination[(atom + 2) * kSFBlock + physical] = values.z;
                                destination[(atom + 3) * kSFBlock + physical] = values.w;
                            }
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("bar.sync 4, 128;" ::: "memory");
                        if (tid == 0)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchSfReady) : "memory");
                        if (first_sf) GENPHASE_MARK(GenphaseSfReadyIssue);
                    } else {
                        // Independent middle window. Bit4 certifies only [1024, min(2048,H));
                        // prefix and tail have separate owners and certificates.
                        if (first_remainder) GENPHASE_MARK_WARP(GenphaseRemainderValidateStart, 4);
                        validate_window_rows<kTokens, kHidden, kTopk, kFirstPayloadWindowBytes,
                                               (kHidden < kMiddlePayloadWindowEnd ? kHidden : kMiddlePayloadWindowEnd), 4>(
                            ws.get_src_token_topk_idx_ptr(e, kRanks == 1 ? 0 : ws.metadata_slot, base),
                            data + ws.get_source_l1_row(0, 0) * kHidden,
                            data + (materialized_base + offsets[e] + base) * kHidden,
                            ready, warp - 4, lane);
                        // TMA will read receiver-owned materialized bytes, never reread
                        // the concurrently written source inbox after validation.
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("bar.sync 5, 128;" ::: "memory");
                        if (tid == 128)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchMiddleAReady) : "memory");
                        if (first_remainder) GENPHASE_MARK_WARP(GenphaseMiddlePayloadIssue, 4);
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
                    for (uint32_t batch = 0; batch < ready * 16; batch += kMetadataThreads) {
                        const uint32_t task = batch + tid;
                        const bool active = task < ready * 16;
                        const uint32_t i = task / 16, chunk = task % 16;
                        const uint4* input = nullptr;
                        uint4* output = nullptr;
                        if (active) {
                            const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                                e, kRanks == 1 ? 0 : ws.metadata_slot, base + i);
                            const uint32_t rank = descriptor / (kTokens * kTopk);
                            const uint32_t token = (descriptor % (kTokens * kTopk)) / kTopk;
                            input = reinterpret_cast<const uint4*>(data +
                                ws.get_source_l1_row(rank, token) * kHidden);
                            output = reinterpret_cast<uint4*>(data +
                                (materialized_base + offsets[e] + base + i) * kHidden);
                        }
                        constexpr uint32_t kFirstVectors = kFirstPayloadWindowBytes / (16 * 16);
                        uint4 values[kFirstVectors];
                        bool pending;
                        do {
                            if (active) {
                                #pragma unroll
                                for (uint32_t v = 0; v < kFirstVectors; ++v)
                                    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                                 : "=r"(values[v].x), "=r"(values[v].y),
                                                   "=r"(values[v].z), "=r"(values[v].w)
                                                 : "l"(input + chunk + v * 16) : "memory");
                            }
                            pending = false;
                            if (active) {
                                #pragma unroll
                                for (uint32_t v = 0; v < kFirstVectors; ++v)
                                    pending |= values[v].x == 0x80000000u || values[v].y == 0x80000000u ||
                                               values[v].z == 0x80000000u || values[v].w == 0x80000000u;
                            }
                        } while (__any_sync(0xffffffff, pending));
                        if (active) {
                            #pragma unroll
                            for (uint32_t v = 0; v < kFirstVectors; ++v)
                                asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                             "l"(output + chunk + v * 16), "r"(values[v].x), "r"(values[v].y),
                                             "r"(values[v].z), "r"(values[v].w) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    sync();
                    if (tid == 0)
                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                     :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchFirstAReady) : "memory");
                    if (first_validation) GENPHASE_MARK(GenphaseWindowPayloadIssue);
                    first_validation = false;
                    if constexpr (kHidden > kMiddlePayloadWindowEnd) {
                        // Prefix has already been certified. Its eight warps now
                        // validate the disjoint tail while consumers use earlier K.
                        if (first_tail) GENPHASE_MARK(GenphaseWindowValidateStart);
                        validate_window_rows<kTokens, kHidden, kTopk,
                                             kMiddlePayloadWindowEnd, kHidden, kMetadataWarps>(
                            ws.get_src_token_topk_idx_ptr(e, kRanks == 1 ? 0 : ws.metadata_slot, base),
                            data + ws.get_source_l1_row(0, 0) * kHidden,
                            data + (materialized_base + offsets[e] + base) * kHidden,
                            ready, warp, lane);
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        sync();
                        if (tid == 0)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchTailAReady) : "memory");
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

    if (cta > kMetadataCTAs) {
        // One slot per warp fits the disjoint dispatch region. The fixed EP8
        // case sends one token per warp; no lookahead slot is needed. Reuse on
        // later iterations follows this warp's completed local shared reads.
        constexpr uint32_t kRowBytes = kHidden;
        constexpr uint32_t kTokenStride = kPushers * kWarps;
        auto* barriers = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(storage);
        auto* stage = storage + 128;
        if (tid < kWarps * kDepth)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(cute::cast_smem_ptr_to_uint(barriers + tid)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        sync();
        // Classify once while the row TMA is in flight. Each mask word is
        // warp-uniform, so fanout needs no per-slot shuffle or vote.
        constexpr uint32_t kMaskWords = (kRanks + 31) / 32;
        struct Route { uint32_t peers[kMaskWords]; };
        const auto prefetch = [&](uint32_t token, uint32_t slot) {
            // Every invocation names a valid local input row. Issue its load
            // before route lookup/dedup so that work can overlap TMA progress.
            // Empty routes still wait for this load but issue no payload stores.
            if (lane == 0) {
                auto* tile = stage + slot * kRowBytes;
                ptx::mbarrier_arrive_and_set_tx(barriers + slot, kRowBytes);
                ptx::tma_load_1d(tile, x + static_cast<uint64_t>(token) * kHidden, barriers + slot, kHidden);
            }
            const int64_t expert = lane < kTopk ? topk[token * kTopk + lane] : -1;
            const int destination = expert >= 0 && expert < kExperts ? expert / kLocalExperts : -1;
            Route route{};
            #pragma unroll
            for (uint32_t word = 0; word < kMaskWords; ++word) {
                const uint32_t bit = destination >= 0 && uint32_t(destination) / 32 == word
                    ? (1u << (uint32_t(destination) % 32)) : 0u;
                route.peers[word] = __reduce_or_sync(0xffffffff, bit);
            }
            return route;
        };
        const uint32_t first_base = (cta - kMetadataCTAs - 1) * kWarps;
        const uint32_t first_token = first_base + warp;
        if (first_base < num_tokens) {
            GENPHASE_MARK(MegaPayloadLoadStart);
            Route current = first_token < num_tokens ?
                prefetch(first_token, warp * kDepth) : Route{};
            Route next{};
            for (uint32_t token = first_token, step = 0; first_base + step * kTokenStride < num_tokens; token += kTokenStride, ++step) {
                const uint32_t slot = warp * kDepth + step % kDepth;
                auto* tile = stage + slot * kRowBytes;
                uint32_t phase = step / kDepth & 1;
                if (token < num_tokens)
                    ptx::mbarrier_wait_and_flip_phase(barriers + slot, phase);
                uint32_t any_peer = 0;
                #pragma unroll
                for (uint32_t word = 0; word < kMaskWords; ++word)
                    any_peer |= current.peers[word];
                if (any_peer) {
                    const uint64_t row = ws.get_source_l1_row(peers.rank_idx, token);
                    constexpr uint32_t kVectors = kHidden / 512;
                    uint4 values[kVectors];
                    const uint32_t source = cute::cast_smem_ptr_to_uint(tile + lane * 16);
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v)
                        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                                     : "=r"(values[v].x), "=r"(values[v].y),
                                       "=r"(values[v].z), "=r"(values[v].w)
                                     : "r"(source + v * 512) : "memory");
                    #pragma unroll
                    for (uint32_t v = 0; v < kVectors; ++v) {
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
                    for (uint32_t window = 0; window < 2; ++window) {
                        const uint32_t first = window == 0 ? 0 : kFirstPayloadWindowBytes / 512;
                        const uint32_t last = window == 0 ? kFirstPayloadWindowBytes / 512 : kVectors;
                        #pragma unroll
                        for (uint32_t word = 0; word < kMaskWords; ++word) {
                            uint32_t pending_peers = current.peers[word];
                            while (pending_peers) {
                                const uint32_t peer = word * 32 + __ffs(pending_peers) - 1;
                                pending_peers &= pending_peers - 1;
                                auto* output = peers.map(data + row * kHidden + lane * 16, peer);
                                #pragma unroll
                                for (uint32_t v = first; v < last; ++v)
                                    asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                        "l"(output + v * 512), "r"(values[v].x), "r"(values[v].y),
                                        "r"(values[v].z), "r"(values[v].w) : "memory");
                            }
                        }
                    }
                }
                if (step == 0) GENPHASE_MARK(GenphasePayloadStoresIssued);
                // Stores source registers, not the shared slot. Once all shared
                // reads are issued this warp may refill its private load slot.
                __syncwarp();
                // A partial final iteration must not reuse a prior route in
                // inactive warps: its output row aliases materialized storage.
                next = token + kTokenStride < num_tokens ?
                    prefetch(token + kTokenStride, warp * kDepth) : Route{};
                current = next;
            }
        }
        // Retirement alone waits for remote store completion. Receiver
        // validation/local tile release independently admits gather/MMA.
        sync();
        if (tid == 0) __threadfence_system();
        sync();
        // Completion only; readiness travels in each payload word.
        if (tid < kWarps * kDepth)
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
