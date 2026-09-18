#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t metadata_slot = 0;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank), num_topk(num_topk) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
#if defined(__CUDA_ARCH__)
        metadata_slot = *reinterpret_cast<volatile uint32_t*>(get_nvl_barrier_counter_ptr()) & 1u;
#endif
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_compact_l1_num_tokens(uint32_t hidden) const {
        const uint64_t bytes = 2ull * num_ranks * num_max_tokens_per_rank * (hidden + hidden / 32);
        const uint64_t source_rows = math::ceil_div<uint64_t>(bytes, hidden);
        return source_rows > num_max_pool_tokens ? source_rows : num_max_pool_tokens;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_source_l1_row(uint32_t source_rank, uint32_t source_token) const {
        return (static_cast<uint64_t>(metadata_slot) * num_ranks + source_rank) *
               num_max_tokens_per_rank + source_token;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_source_l1_row(uint32_t linear_source_token) const {
        return static_cast<uint64_t>(metadata_slot) * num_ranks *
               num_max_tokens_per_rank + linear_source_token;
    }

    CUTLASS_HOST_DEVICE
    void* get_compact_frame_base_ptr(uint32_t hidden, uint32_t topk) const {
        return math::advance_ptr(get_end_ptr(), static_cast<uint64_t>(num_max_tokens_per_rank) *
            (hidden + hidden / 32 + topk * (sizeof(int64_t) + sizeof(float))));
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Rank-level route publication completion
        num_bytes += sizeof(uint64_t);

        // Per-expert-rank combine readiness epoch
        num_bytes += num_ranks * sizeof(uint64_t);

        // Reserved padding preserves the validated metadata and payload offsets.
        num_bytes += num_ranks * 2 * sizeof(uint64_t);

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        return math::align<uint64_t>(num_bytes + get_metadata_bank_bytes(), 128) + get_genphase_bytes();
    }

    // Double-buffer only cross-rank inboxes/counters. Local indices are consumed
    // before the next local invocation, while a peer can already send its next plane.
    CUTLASS_HOST_DEVICE
    uint64_t get_genphase_bank_bytes() const {
        return math::align<uint64_t>((num_ranks + static_cast<uint64_t>(NumGenphaseFlags)) * sizeof(uint32_t), 128) +
            2ull * num_ranks * num_max_tokens_per_rank * num_topk * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_genphase_bytes() const {
        return 2 * get_genphase_bank_bytes() +
            math::align<uint64_t>(num_max_pool_tokens * sizeof(uint32_t), 128);
    }

    enum GenphaseFlag { CountsReady, SizesReady, IndicesReady, PayloadReady, DispatchDone, NumGenphaseFlags };

    CUTLASS_DEVICE
    uint32_t* get_genphase_flag_ptr(uint32_t index) const {
        return math::advance_ptr<uint32_t>(base,
            get_num_bytes() - get_genphase_bytes() + metadata_slot * get_genphase_bank_bytes()) + index;
    }

    CUTLASS_DEVICE
    uint32_t* get_genphase_flag_ptr(GenphaseFlag flag) const {
        return get_genphase_flag_ptr(num_ranks + static_cast<uint32_t>(flag));
    }

    CUTLASS_DEVICE
    int32_t* get_topk_inbox_ptr(uint32_t rank = 0) const {
        return math::advance_ptr<int32_t>(get_genphase_flag_ptr(0u),
            math::align<uint64_t>((num_ranks + static_cast<uint64_t>(NumGenphaseFlags)) * sizeof(uint32_t), 128)) +
            static_cast<uint64_t>(rank) * num_max_tokens_per_rank * num_topk;
    }

    CUTLASS_DEVICE
    float* get_score_inbox_ptr(uint32_t rank = 0) const {
        return reinterpret_cast<float*>(get_topk_inbox_ptr(num_ranks)) +
            static_cast<uint64_t>(rank) * num_max_tokens_per_rank * num_topk;
    }

    CUTLASS_DEVICE
    uint32_t* get_gather_index_ptr(uint32_t row = 0) const {
        return math::advance_ptr<uint32_t>(base,
            get_num_bytes() - get_genphase_bytes() + 2 * get_genphase_bank_bytes()) + row;
    }

    CUTLASS_HOST_DEVICE
    static constexpr uint32_t genphase_pusher_ctas(uint32_t tokens) {
        return math::constexpr_min(math::constexpr_ceil_div(tokens, 128u), 8u) + (tokens == 0);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_metadata_bank_bytes() const {
        return (1ull + 3ull * num_ranks + 2ull * num_experts + num_experts_per_rank) * sizeof(uint64_t);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint32_t* get_route_prepare_counter_ptr() const {
        // The final word in the fixed 32-byte barrier region is reserved for
        // the fused router's last-block publication ticket.
        return static_cast<uint32_t*>(base) + 7;
    }

    CUTLASS_DEVICE
    uint64_t* get_route_ready_count_ptr() const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes +
            metadata_slot * get_metadata_bank_bytes());
    }

    CUTLASS_DEVICE
    uint64_t* get_combine_ready_epoch_ptr(const uint32_t& rank_idx = 0) const {
        return get_route_ready_count_ptr() + 1 + rank_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return get_combine_ready_epoch_ptr(num_ranks) + num_ranks * 2 + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        auto* counts = math::advance_ptr<uint32_t>(base,
            kNumBarrierSignalBytes + 2 * get_metadata_bank_bytes());
        return counts + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        // Pad L1 entry count to even so that the `l2_arrival_mask` is 8-byte aligned
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& max_num_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

} // namespace deep_gemm::layout
