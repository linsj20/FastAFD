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
CREATE_IKET_MARKER(GenphaseFullPayloadIssue);
CREATE_IKET_MARKER(GenphaseFirstPayloadIssue);
CREATE_IKET_MARKER(GenphaseSfStoresIssued);
CREATE_IKET_MARKER(GenphaseSfReadyIssue);
CREATE_IKET_MARKER(GenphasePayloadValidateStart);
CREATE_IKET_MARKER(GenphaseRemainderValidateStart);
CREATE_IKET_MARKER(GenphaseIndexScatter);
CREATE_IKET_MARKER(GenphaseEnd);
CREATE_IKET_MARKER(MegaPayloadLoadStart);
CREATE_IKET_MARKER(GenphasePayloadSendStart);
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

inline constexpr uint32_t kFirstPayloadWindowBytes = 1536;
// One publisher adds each unique bit once per tile. Disjoint bit additions
// commute, so SF and activation may finish independently. Epilogue metadata
// stays in the producer-owned banks covered by the original route-ready release.
inline constexpr uint32_t kDispatchSfReady = 1u;
inline constexpr uint32_t kDispatchFirstAReady = 2u;
inline constexpr uint32_t kDispatchFullAReady = 4u;

// Pull-style producer expert lists with independent deduplicated payload push.
// Local index scatter consumes only destination routes; FC1 keeps TMA gather
// with row-local in-band readiness and the existing push combine.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kExperts, uint32_t kTopk,
          uint32_t kRanks, uint32_t BLOCK_M, uint32_t kSFPoolTokens, uint32_t kPushers>
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
        bool first_sf = true;
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
                        if (first_sf) GENPHASE_MARK_ALL(GenphaseSfStoresIssued);
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("bar.sync 4, 128;" ::: "memory");
                        if (tid == 0)
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                         :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchSfReady) : "memory");
                        if (first_sf) GENPHASE_MARK(GenphaseSfReadyIssue);
                    }
                    first_sf = false;
                } else {
                    // No SF dependency: validate and materialize A as soon as routes
                    // and its transport words are ready. Bits2/4 publish A windows.
                    if (first_validation) GENPHASE_MARK(GenphasePayloadValidateStart);
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
                    if (first_validation) GENPHASE_MARK(GenphaseFirstPayloadIssue);
                    if (first_validation) GENPHASE_MARK(GenphaseRemainderValidateStart);
                    // Bit2 releases the first window. Validate/copy the remaining
                    // activation vectors while gather and MMA can progress.
                    for (uint32_t i = warp; i < ready; i += kMetadataWarps) {
                        const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                            e, kRanks == 1 ? 0 : ws.metadata_slot, base + i);
                        const uint32_t rank = descriptor / (kTokens * kTopk);
                        const uint32_t token = (descriptor % (kTokens * kTopk)) / kTopk;
                        const uint64_t source = ws.get_source_l1_row(rank, token);
                        const auto* input = reinterpret_cast<const uint4*>(data + source * kHidden);
                        auto* output = reinterpret_cast<uint4*>(data +
                            (materialized_base + offsets[e] + base + i) * kHidden);
                        constexpr uint32_t kVectors = (kHidden - kFirstPayloadWindowBytes) / 512;
                        uint4 values[kVectors];
                        bool pending;
                        do {
                            // Issue the full independent load batch before testing
                            // any result; keep separate destination registers live.
                            #pragma unroll
                            for (uint32_t v = 0; v < kVectors; ++v)
                                asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                             : "=r"(values[v].x), "=r"(values[v].y),
                                               "=r"(values[v].z), "=r"(values[v].w)
                                             : "l"(input + lane + kFirstPayloadWindowBytes / 16 + v * 32) : "memory");
                            pending = false;
                            #pragma unroll
                            for (uint32_t v = 0; v < kVectors; ++v)
                                pending |= values[v].x == 0x80000000u || values[v].y == 0x80000000u ||
                                           values[v].z == 0x80000000u || values[v].w == 0x80000000u;
                        } while (__any_sync(0xffffffff, pending));
                        #pragma unroll
                        for (uint32_t v = 0; v < kVectors; ++v)
                            asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                         "l"(output + lane + kFirstPayloadWindowBytes / 16 + v * 32), "r"(values[v].x), "r"(values[v].y),
                                         "r"(values[v].z), "r"(values[v].w) : "memory");
                    }
                    // TMA will read receiver-owned materialized bytes, never reread
                    // the concurrently written source inbox after validation.
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    sync();
                    if (tid == 0)
                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                     :: "l"(ws.get_l1_arrival_count_ptr(block)), "r"(kDispatchFullAReady) : "memory");
                    if (first_validation) GENPHASE_MARK(GenphaseFullPayloadIssue);
                    first_validation = false;
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

    // CountsReady counts physical input rows published by the selector CTAs.
    // This acquire replaces a wait for the entire selector's network tail.
    if (lane == 0)
        while (ptx::ld_acq(ws.get_genphase_flag_ptr(W::CountsReady)) != num_tokens);
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
        struct Route { int peer; uint32_t mask; };
        const auto prefetch = [&](uint32_t token, uint32_t slot) {
            const int64_t expert = lane < kTopk ? topk[token * kTopk + lane] : -1;
            const int destination = expert >= 0 && expert < kExperts ? expert / kLocalExperts : -1;
            const uint32_t equal = __match_any_sync(0xffffffff, destination);
            const uint32_t mask = __ballot_sync(0xffffffff, destination >= 0 && lane == __ffs(equal) - 1);
            if (lane == 0) {
                auto* tile = stage + slot * kRowBytes;
                // Empty routes still advance the slot phase, without reading data.
                ptx::mbarrier_arrive_and_set_tx(barriers + slot, mask ? kRowBytes : 0);
                if (mask) {
                    ptx::tma_load_1d(tile, x + static_cast<uint64_t>(token) * kHidden, barriers + slot, kHidden);
                }
            }
            return Route{destination, mask};
        };
        const uint32_t first_base = (cta - kMetadataCTAs - 1) * kWarps;
        const uint32_t first_token = first_base + warp;
        if (first_base < num_tokens) {
            GENPHASE_MARK(MegaPayloadLoadStart);
            Route current = first_token < num_tokens ?
                prefetch(first_token, warp * kDepth) : Route{-1, 0};
            Route next = {-1, 0};
            for (uint32_t token = first_token, step = 0; first_base + step * kTokenStride < num_tokens; token += kTokenStride, ++step) {
                const uint32_t slot = warp * kDepth + step % kDepth;
                auto* tile = stage + slot * kRowBytes;
                uint32_t phase = step / kDepth & 1;
                if (token < num_tokens)
                    ptx::mbarrier_wait_and_flip_phase(barriers + slot, phase);
                if (current.mask) {
                    if (step == 0) GENPHASE_MARK(GenphasePayloadSendStart);
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
                    // One warp-uniform peer decision/address for the complete
                    // row; eight stores use distinct data registers and offsets.
                    #pragma unroll
                    for (uint32_t k = 0; k < kTopk; ++k) {
                        const int peer = __shfl_sync(0xffffffff, current.peer, k);
                        if (__any_sync(0xffffffff, (current.mask & (1u << k)) != 0)) {
                            auto* output = peers.map(data + row * kHidden + lane * 16, peer);
                            #pragma unroll
                            for (uint32_t v = 0; v < kVectors; ++v)
                                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                                    "l"(output + v * 512), "r"(values[v].x), "r"(values[v].y),
                                    "r"(values[v].z), "r"(values[v].w) : "memory");
                        }
                    }
                }
                if (step == 0) GENPHASE_MARK(GenphasePayloadStoresIssued);
                // Stores source registers, not the shared slot. Once all shared
                // reads are issued this warp may refill its private load slot.
                __syncwarp();
                if (token + kTokenStride < num_tokens)
                    next = prefetch(token + kTokenStride, warp * kDepth);
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
