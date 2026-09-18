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
CREATE_IKET_MARKER(GenphaseIndexReady);
CREATE_IKET_MARKER(GenphasePayloadSenderDone);
CREATE_IKET_MARKER(GenphaseIndexScatter);
CREATE_IKET_MARKER(GenphaseEnd);
CREATE_IKET_MARKER(MegaPayloadLoadStart);
CREATE_IKET_MARKER(GenphasePayloadSendStart);
CREATE_IKET_MARKER(GenphasePayloadDrained);
#define GENPHASE_MARK(name) do { if (warp == 0) IKET_MARK(name); } while (0)
#else
#define GENPHASE_MARK(name) ((void)0)
#endif

namespace deep_gemm {

// Pull-style producer expert lists with independent deduplicated payload push.
// Local index scatter consumes only destination routes; FC1 keeps TMA gather
// with row-local in-band readiness and the existing push combine.
template <uint32_t kTokens, uint32_t kHidden, uint32_t kExperts, uint32_t kTopk,
          uint32_t kRanks, uint32_t BLOCK_M, uint32_t kSFPoolTokens, uint32_t kPushers>
__device__ __forceinline__ void mega_moe_dispatch(
    uint32_t num_tokens, const layout::SymBuffer<kRanks>& peers, uint8_t* storage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    using W = layout::Workspace;
    constexpr uint32_t kThreads = 128, kWarps = 4, kDepth = 2;
    constexpr uint32_t kLocalExperts = kExperts / kRanks;
    constexpr uint32_t kSourceRows = 2 * kRanks * kTokens;
    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const uint32_t cta = blockIdx.x;
    GENPHASE_MARK(GenphaseStart);
    const W ws(peers.get_base_ptr(), kRanks, kExperts, kTokens, kTopk);
    // First dispatch warpgroup, before this CTA initializes GEMM shared memory.
    // Metadata consumers acquire the producer's SYS publications below. Payload
    // senders acquire its local input counter, independently of the network tail.
    const auto sync = [&]() { asm volatile("bar.sync 3, 128;" ::: "memory"); };

    const auto* x = static_cast<const uint8_t*>(ws.get_end_ptr());
    const auto* sf = x + static_cast<uint64_t>(kTokens) * kHidden;
    const auto* topk = reinterpret_cast<const int64_t*>(sf + static_cast<uint64_t>(kTokens) * kHidden / 32);
    auto* data = static_cast<uint8_t*>(ws.get_compact_frame_base_ptr(kHidden, kTopk));
    auto* plain_sf = data + static_cast<uint64_t>(kSourceRows) * kHidden;
    auto* canonical_sf = reinterpret_cast<uint32_t*>(
        data + ws.get_compact_l1_num_tokens(kHidden) * kHidden);
    auto* routed_scores = reinterpret_cast<float*>(canonical_sf +
        static_cast<uint64_t>(kSFPoolTokens) * (kHidden / 128));

    if (cta < kRanks) {
        GENPHASE_MARK(GenphaseMetadataFromProducer);
        // The router writes expert-local source descriptors and scores before
        // PDL, then publishes the same all-rank readiness used by pull.
        if (tid == 0)
            while (ptx::ld_acq_sys(ws.get_route_ready_count_ptr()) != kRanks);
        sync();
        GENPHASE_MARK(GenphaseTopkReceived);
        auto* histogram = reinterpret_cast<uint32_t*>(storage);
        auto* offsets = histogram + kLocalExperts;
        auto* warp_sums = offsets + kLocalExperts;
        for (uint32_t e = tid; e < kLocalExperts; e += kThreads) {
            const uint32_t count = static_cast<uint32_t>(*ws.get_expert_recv_count_sum_ptr(e));
            histogram[e] = math::align(count, BLOCK_M);
            offsets[e] = 0;
        }
        sync();
        if (cta == 0) {
            if (tid == 0)
                asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;"
                             :: "l"(ws.get_genphase_flag_ptr(W::SizesReady)) : "memory");
            GENPHASE_MARK(GenphaseSizesReady);
        }
        // Exclusive padded-row prefix in natural expert order.
        uint32_t carry = 0;
        for (uint32_t start = 0; start < kLocalExperts; start += kThreads) {
            const uint32_t e = start + tid;
            const uint32_t padded = e < kLocalExperts ? histogram[e] : 0;
            uint32_t sum = padded;
            #pragma unroll
            for (uint32_t step = 1; step < 32; step *= 2) {
                const uint32_t prev = __shfl_up_sync(0xffffffff, sum, step);
                if (lane >= step) sum += prev;
            }
            if (lane == 31) warp_sums[warp] = sum;
            sync();
            uint32_t base = carry + sum - padded;
            for (uint32_t w = 0; w < warp; ++w) base += warp_sums[w];
            if (e < kLocalExperts) offsets[e] += base;
            for (uint32_t w = 0; w < kWarps; ++w) carry += warp_sums[w];
            sync();
        }
        // Indices do not depend on payload arrival. Publish them while the
        // payload is in flight; FC1 acquires payload at its first consumption.
        GENPHASE_MARK(GenphaseIndexScatter);
        for (uint32_t e = cta; e < kLocalExperts; e += kRanks) {
            const uint32_t count = static_cast<uint32_t>(*ws.get_expert_recv_count_sum_ptr(e));
            for (uint32_t i = tid; i < count; i += kThreads) {
                const uint32_t descriptor = *ws.get_src_token_topk_idx_ptr(
                    e, kRanks == 1 ? 0 : ws.metadata_slot, i);
                const uint32_t rank = descriptor / (kTokens * kTopk);
                const uint32_t route = descriptor % (kTokens * kTopk);
                const uint32_t row = offsets[e] + i;
                *ws.get_gather_index_ptr(row) = ws.get_source_l1_row(rank, route / kTopk);
                *ws.get_token_src_metadata_ptr(row) = {rank, route / kTopk, route % kTopk};
                routed_scores[row] = ws.get_score_inbox_ptr(rank)[route];
            }
            sync();
            if (tid == 0) {
                const uint32_t end = offsets[e] + count;
                for (uint32_t row = offsets[e]; row < end;) {
                    const uint32_t ready = min(end - row, BLOCK_M - row % BLOCK_M);
                    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                                 :: "l"(ws.get_l1_arrival_count_ptr(row / BLOCK_M)), "r"(ready) : "memory");
                    row += ready;
                }
            }
        }
        sync();
        if (tid == 0)
            asm volatile("red.async.release.gpu.global.add.u32 [%0], 1;"
                         :: "l"(ws.get_genphase_flag_ptr(W::DispatchDone)) : "memory");
        GENPHASE_MARK(GenphaseIndexReady);
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

    if (cta > kRanks) {
        // Two complete token slots per warp: stage FP8 and SF concurrently,
        // then overlap the next token's load with the current token's pushes.
        constexpr uint32_t kSFBytes = kHidden / 32, kRowBytes = kHidden + kSFBytes;
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
                    ptx::tma_load_1d(tile + kHidden, sf + static_cast<uint64_t>(token) * kSFBytes, barriers + slot, kSFBytes);
                }
            }
            return Route{destination, mask};
        };
        const uint32_t first_base = (cta - kRanks - 1) * kWarps;
        const uint32_t first_token = first_base + warp;
        if (first_base < num_tokens) {
            GENPHASE_MARK(MegaPayloadLoadStart);
            Route current = first_token < num_tokens ?
                prefetch(first_token, warp * kDepth) : Route{-1, 0};
            Route next = {-1, 0};
            if (first_token + kTokenStride < num_tokens)
                next = prefetch(first_token + kTokenStride, warp * kDepth + 1);
            for (uint32_t token = first_token, step = 0; first_base + step * kTokenStride < num_tokens; token += kTokenStride, ++step) {
                const uint32_t slot = warp * kDepth + step % kDepth;
                auto* tile = stage + slot * kRowBytes;
                uint32_t phase = step / kDepth & 1;
                if (token < num_tokens)
                    ptx::mbarrier_wait_and_flip_phase(barriers + slot, phase);
                uint32_t ready_tail = 0;
                if (current.mask) {
                    // The final FP8 word is data and this row's completion
                    // signal. Keep it at -0 during bulk transfer, then publish
                    // its real bits once every activation/SF byte has arrived.
                    auto* tail = reinterpret_cast<uint32_t*>(tile + kHidden - 4);
                    const uint32_t tail_bits = *tail;
                    ready_tail = tail_bits == 0x80000000u ? 0 : tail_bits;
                    __syncwarp();
                    if (lane == 0) *tail = 0x80000000u;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (step == 0) GENPHASE_MARK(GenphasePayloadSendStart);
                    const uint64_t row = ws.get_source_l1_row(peers.rank_idx, token);
                    #pragma unroll
                    for (uint32_t k = 0; k < kTopk; ++k) {
                        const int peer = __shfl_sync(0xffffffff, current.peer, k);
                        if (lane == 0 && (current.mask & (1u << k))) {
                            ptx::tma_store_1d(peers.map(plain_sf + row * kSFBytes, peer), tile + kHidden, kSFBytes);
                            ptx::tma_store_1d(peers.map(data + row * kHidden, peer), tile, kHidden);
                        }
                    }
                    if (lane == 0)
                        asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0; fence.proxy.async.global;" ::: "memory");
                }
                if (step == 0) GENPHASE_MARK(GenphasePayloadDrained);
                // All four warps contribute completed rows. One CTA-wide
                // release fence orders them, avoiding four concurrent SYS
                // fences on this SM. Inactive warps still join this iteration.
                sync();
                if (tid == 0) __threadfence_system();
                sync();
                if (current.mask & (1u << lane)) {
                    const uint64_t row = ws.get_source_l1_row(peers.rank_idx, token);
                    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::
                        "l"(peers.map(data + row * kHidden + kHidden - 4, current.peer)),
                        "r"(ready_tail) : "memory");
                }
                // The second token is preloaded. Thereafter refill the previous
                // slot while this token's outgoing reads may remain in flight.
                if (step && token + kTokenStride < num_tokens) {
                    if (lane == 0) {
                        if (current.mask)
                            asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                        else
                            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
                    }
                    __syncwarp();
                    next = prefetch(token + kTokenStride, warp * kDepth + (step + 1) % kDepth);
                }
                current = next;
            }
        }
        // Each row's bulk transfer has already completed before its tail store.
        // Tail stores use registers, so only the shared-stage users must join
        // before barrier invalidation/reuse. No global publication remains that
        // would require another system fence here.
        sync();
        // Completion only; readiness travels in each payload word.
        GENPHASE_MARK(GenphasePayloadSenderDone);
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

#undef GENPHASE_MARK
