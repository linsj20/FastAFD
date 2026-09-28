#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    int num_dispatch_pull_stages;
    int num_dispatch_push_ctas;
    bool all_push_full_tma;
    int num_dispatch_expert_ctas;
    bool all_push_prioritize_experts;
    int compute_cta_offset;
    int compute_warmup_rounds;
    int compute_busy_sender_ctas;
    bool all_push_unified_tma;
    bool all_push_dedicated_dispatch;
    int num_epilogue_stages;
    bool all_push_linear_source;
    bool all_push_cache_destinations;
    bool all_push_interleave_m;
    int all_push_narrow_sf;
    int all_push_coalesced_sf;
    bool all_push_prefetch_send;
    int all_push_middle_window_end;
    bool all_push_half_warp;
    int all_push_first_window_end;
    bool all_push_sticky_validation;
    int all_push_full_row_combine;
    bool all_push_weight_cache;
    uint32_t all_push_warp_combine;
    bool all_push_prefix_sweep;
    bool all_push_producer_prefix;
    int all_push_sender_ldg;
    bool all_push_prefix_wide;
    int all_push_cache_priority;
    int all_push_group_priority;
    int all_push_return_wide;
    bool all_push_return_wg_sync;
    bool all_push_balance_epilogue;
    int all_push_extent_tma;
    bool all_push_late_count_lane;
    bool all_push_fc2_stage_ready;
    uint32_t all_push_activation_issuers;
    int all_push_metadata_row_partitions;
    int all_push_weight_stages;
    int all_push_activation_credit_limit;
    bool all_push_fc2_credit_only;
    int all_push_relaxed_poll;
    int all_push_send_middle_first;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_dispatch_pull_stages=" << config.num_dispatch_pull_stages
           << ", num_dispatch_push_ctas=" << config.num_dispatch_push_ctas
           << ", all_push_full_tma=" << config.all_push_full_tma
           << ", num_dispatch_expert_ctas=" << config.num_dispatch_expert_ctas
           << ", all_push_prioritize_experts=" << config.all_push_prioritize_experts
           << ", compute_cta_offset=" << config.compute_cta_offset
           << ", compute_warmup_rounds=" << config.compute_warmup_rounds
           << ", compute_busy_sender_ctas=" << config.compute_busy_sender_ctas
           << ", all_push_unified_tma=" << config.all_push_unified_tma
           << ", all_push_dedicated_dispatch=" << config.all_push_dedicated_dispatch
           << ", num_epilogue_stages=" << config.num_epilogue_stages
           << ", all_push_linear_source=" << config.all_push_linear_source
           << ", all_push_cache_destinations=" << config.all_push_cache_destinations
           << ", all_push_interleave_m=" << config.all_push_interleave_m
           << ", all_push_narrow_sf=" << config.all_push_narrow_sf
           << ", all_push_coalesced_sf=" << config.all_push_coalesced_sf
           << ", all_push_prefetch_send=" << config.all_push_prefetch_send
           << ", all_push_middle_window_end=" << config.all_push_middle_window_end
           << ", all_push_half_warp=" << config.all_push_half_warp
           << ", all_push_first_window_end=" << config.all_push_first_window_end
           << ", all_push_sticky_validation=" << config.all_push_sticky_validation
           << ", all_push_full_row_combine=" << config.all_push_full_row_combine
           << ", all_push_weight_cache=" << config.all_push_weight_cache
           << ", all_push_warp_combine=" << config.all_push_warp_combine
           << ", all_push_prefix_sweep=" << config.all_push_prefix_sweep
           << ", all_push_producer_prefix=" << config.all_push_producer_prefix
           << ", all_push_sender_ldg=" << config.all_push_sender_ldg
           << ", all_push_prefix_wide=" << config.all_push_prefix_wide
           << ", all_push_cache_priority=" << config.all_push_cache_priority
           << ", all_push_group_priority=" << config.all_push_group_priority
           << ", all_push_return_wide=" << config.all_push_return_wide
           << ", all_push_return_wg_sync=" << config.all_push_return_wg_sync
           << ", all_push_balance_epilogue=" << config.all_push_balance_epilogue
           << ", all_push_extent_tma=" << config.all_push_extent_tma
           << ", all_push_late_count_lane=" << config.all_push_late_count_lane
           << ", all_push_fc2_stage_ready=" << config.all_push_fc2_stage_ready
           << ", all_push_activation_issuers=" << config.all_push_activation_issuers
           << ", all_push_metadata_row_partitions=" << config.all_push_metadata_row_partitions
           << ", all_push_weight_stages=" << config.all_push_weight_stages
           << ", all_push_activation_credit_limit=" << config.all_push_activation_credit_limit
           << ", all_push_fc2_credit_only=" << config.all_push_fc2_credit_only
           << ", all_push_relaxed_poll=" << config.all_push_relaxed_poll
           << ", all_push_send_middle_first=" << config.all_push_send_middle_first << ")";
        return os;
    }
};

static std::tuple<int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens, const bool all_push = false) {
    // Rejected experiments live in frozen sources. Refuse stale launch settings.
    if (all_push) {
        for (const auto* removed : {
                 "DG_MEGAMOE_ALL_PUSH_READY_SCHEDULE",
                 "DG_MEGAMOE_ALL_PUSH_ROTATE_PEERS",
                 "DG_MEGAMOE_ALL_PUSH_SFA_MULTICAST",
                 "DG_MEGAMOE_ALL_PUSH_METADATA_REGISTER_SCAN",
                 "DG_MEGAMOE_ALL_PUSH_METADATA_SINGLE_JOIN",
                 "DG_MEGAMOE_ALL_PUSH_SKIP_IDLE_DISPATCH",
                 "DG_MEGAMOE_ALL_PUSH_METADATA_WARP_GROUP_LIMIT",
                 "DG_MEGAMOE_ALL_PUSH_SKIP_EMPTY_PARTITIONS",
                 "DG_MEGAMOE_ALL_PUSH_SF_EARLY_RETURN",
                 "DG_MEGAMOE_ALL_PUSH_SINGLE_GEMM_EPILOGUE",
                 "DG_MEGAMOE_ALL_PUSH_FC1_WG_READY",
                 "DG_MEGAMOE_ALL_PUSH_FC1_ASYNC_READY",
                 "DG_MEGAMOE_ALL_PUSH_METADATA_COOPERATIVE_GROUPS",
                 "DG_MEGAMOE_ALL_PUSH_METADATA_AFTER_REGISTERS",
                 "DG_MEGAMOE_ALL_PUSH_EARLY_SF",
                 "DG_MEGAMOE_ALL_PUSH_DIRECT_SHARED",
                 "DG_MEGAMOE_ALL_PUSH_WINDOW_WIDE",
                 "DG_MEGAMOE_ALL_PUSH_LOCAL_SOURCE",
                 "DG_MEGAMOE_ALL_PUSH_ROW_RELEASE"})
            DG_HOST_ASSERT(get_env<int>(removed, 0) == 0);
    }
    const int compact_tiles = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_COMPACT_TILES") : 0;
    DG_HOST_ASSERT(compact_tiles == 0 || compact_tiles == 1);
    const auto& [cluster_size, block_m, store_block_m, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        const int requested_tile = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_BLOCK_M") : 0;
        const bool use_m64 = all_push && requested_tile == 64;
        if (num_expected_tokens_per_expert <= 8.5) {
            // Really small token-per-expert (e.g. RL long-tail rollout), use the smallest block_m
            return {2, 16, 8, 2};
        } else if (num_expected_tokens_per_expert <= 16.5 or
                   (all_push and ((num_expected_tokens_per_expert <= 32.5 &&
                                  !use_m64) or
                                 not get_env<int>("DG_MEGAMOE_ALL_PUSH_WIDE")))) {
            // Preserve small all-push tiles and two epilogue warpgroups.
            // Wider tiles are opt-in until matched throughput validation.
            // Small batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 128
            return {2, 32, 16, 2};
        } else if (use_m64) {
            // Preserve two epilogue warpgroups for all-push dispatch and combine.
            return {2, 64, 16, 2};
        } else if (num_expected_tokens_per_expert <= 32.5) {
            // Medium batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 256
            return {2, 64, 32, 1};
        } else if (all_push && get_env<int>("DG_MEGAMOE_ALL_PUSH_BLOCK_M") == 80) {
            // Opt-in dense 80-row tile; preserve the small and WIDE=0 branches.
            return {2, 80, 8, 2};
        } else if (num_expected_tokens_per_expert <= 64.5 ||
                   (all_push && get_env<int>("DG_MEGAMOE_ALL_PUSH_BLOCK_M") == 96)) {
            // Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512
            // The earlier branches preserve small tiles and WIDE=0.
            return {2, 96, 16, 2};
        } else if (num_expected_tokens_per_expert <= 96.5) {
            // Medium batch size, Medium EP, decoding, e.g. 6/384 experts, EP16, bsz 256, or EP32, bsz128
            return {2, 128, 32, 2};
        } else {
            // Minimize padded rows, retaining the larger tile on ties.
            int tile = 192;
            if (compact_tiles) {
                const int rows = static_cast<int>(std::ceil(num_expected_tokens_per_expert));
                for (const int candidate : {160, 128})
                    if (ceil_div(rows, candidate) * candidate < ceil_div(rows, tile) * tile)
                        tile = candidate;
            }
            const int explicit_tile = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_BLOCK_M") : 0;
            DG_HOST_ASSERT(explicit_tile == 0 || explicit_tile == 128 || explicit_tile == 160 ||
                           explicit_tile == 192 || explicit_tile == 256);
            if (explicit_tile)
                tile = explicit_tile;
            return {2, tile, tile == 160 ? 16 : 32, 2};
        }
    }();

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT((all_push && (block_m == 80 || block_m == 256)) || std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs
    // Smaller output staging leaves room for another operand stage at M128.
    // Keep the established small tiles and other geometries unchanged.
    const int output_rows = all_push && block_m == 128 &&
        get_env<int>("DG_MEGAMOE_ALL_PUSH_COMPACT_STORE") ? 16 : store_block_m;
    return {cluster_size, block_m, output_rows, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms,
    const int& blocks_per_sm = 2) {

    float expected_tokens_per_expert = static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1) {
        // Most experts don't have tokens, calculate all experts at once
        return num_experts_per_rank;
    }

    // Count L1 blocks per expert assuming tokens are evenly spread across experts
    const int num_m_blocks = ceil_div(static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
    const int num_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

    // Cover the requested grid count; the default allows for routing imbalance.
    int num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(blocks_per_sm * num_sms, num_l1_blocks_per_expert) : 1;
    num_experts_per_wave = std::min(num_experts_per_wave, num_experts_per_rank);

    // Round up to the nearest divisor of num_experts_per_rank so every wave processes the same count
    while (num_experts_per_wave < num_experts_per_rank and num_experts_per_rank % num_experts_per_wave != 0)
        ++ num_experts_per_wave;

    return num_experts_per_wave;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden, const int& num_topk,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const int& num_dispatch_pull_stages = 1, const int& num_epilogue_stages = 2) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumTMAStoreStages = 2;
    DG_HOST_ASSERT(num_dispatch_pull_stages >= 1);

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, num_dispatch_pull_stages).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Match the device CTA-combine barrier count, preserving other shapes.
    const int num_combine_barriers = std::max(
        num_epilogue_warps * 2,
        (hidden == 4096 and (num_topk == 8 or num_topk == 10)) ? num_topk : 0);
    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine
    const int smem_barriers = (
        num_dispatch_warps * num_dispatch_pull_stages +
        num_epilogue_stages * 2 + num_combine_barriers) * 8;

    // Amax reduction
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = sf_block_m * 4;
    const int smem_sfb_per_stage = sf_block_n * 4;

    // Each stage owns full/empty barriers shared by operands and scales.
    const int smem_per_stage = load_block_m * block_k + block_n * block_k + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // Select maximum num_stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const int& num_dispatch_pull_stages = 1,
    const bool all_push = false, const int explicit_num_sms = 0) {
    // Block config
    const auto [cluster_size, block_m, store_block_m, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, all_push);
    const int block_n = 128;
    const int block_k = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    if (block_m > layout::kMaxCandidateBlockM) {
        // Keep the workspace ABI and small kernels unchanged. The larger tile
        // is legal only when existing capacity covers worst-case route padding.
        const int64_t max_rows = int64_t(num_ranks) * num_tokens *
            std::min(num_topk, num_experts_per_rank);
        DG_HOST_ASSERT(max_rows + int64_t(num_experts_per_rank) * (block_m - 1) <= num_max_pool_tokens);
    }
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // Waves
    const int num_sms = explicit_num_sms > 0 ? explicit_num_sms : device_runtime->get_num_sms();
    // Mode 2 also tests adaptive waves on small tiles; mode 1 preserves their schedule.
    const int wave_mode = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_WAVES") : 0;
    DG_HOST_ASSERT(wave_mode >= 0 && wave_mode <= 2);
    const bool single_wave = all_push && (wave_mode == 0 || (block_m <= 32 && wave_mode == 1));
    const int wave_blocks_per_sm = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_WAVE_BLOCKS_PER_SM", 2) : 2;
    DG_HOST_ASSERT(wave_blocks_per_sm >= 1 && wave_blocks_per_sm <= 8);
    DG_HOST_ASSERT(!single_wave || wave_blocks_per_sm == 2);
    const int num_experts_per_wave = single_wave ? num_experts_per_rank : get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms, wave_blocks_per_sm);

    // Mode 2 also tests full activation boxes on small tiles; mode 1 is wide-only.
    const int full_tma_mode = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_FULL_TMA") : 0;
    DG_HOST_ASSERT(full_tma_mode >= 0 && full_tma_mode <= 2);
    const bool full_activation_tma = all_push && full_tma_mode != 0 && (block_m > 32 || full_tma_mode == 2);
    const int extent_tma = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_EXTENT_TMA") : 0;
    DG_HOST_ASSERT(extent_tma >= 0 && extent_tma <= 2);
    const int late_count_lane = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_LATE_COUNT_LANE") : 0;
    DG_HOST_ASSERT(late_count_lane == 0 || late_count_lane == 1);

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;
    // Keep the small-tile sender schedule; tune large tiles independently.
    const int num_dispatch_push_ctas = !all_push ? 0 : block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_CTAS", 8)
        : layout::Workspace::genphase_pusher_ctas(num_max_tokens_per_rank);
    DG_HOST_ASSERT(!all_push || (num_dispatch_push_ctas > 0 && num_dispatch_push_ctas <= num_sms));
    const int default_expert_ctas = std::min(num_experts_per_rank, 2 * num_ranks);
    const int requested_expert_ctas = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_EXPERT_CTAS", default_expert_ctas) : default_expert_ctas;
    DG_HOST_ASSERT(requested_expert_ctas > 0);
    const int num_dispatch_expert_ctas = all_push && block_m > 32
        ? requested_expert_ctas : default_expert_ctas;
    // Retired expert grouping is unsupported; reject stale experiment settings.
    DG_HOST_ASSERT(!all_push || get_env<int>("DG_MEGAMOE_ALL_PUSH_METADATA_EXPERT_GROUPS", 1) == 1);
    DG_HOST_ASSERT(!all_push || (num_dispatch_expert_ctas > 0 &&
        2 * num_dispatch_expert_ctas + 1 + num_dispatch_push_ctas <= num_sms &&
        (num_dispatch_expert_ctas <= num_experts_per_rank || num_dispatch_expert_ctas % num_experts_per_rank == 0)));
    const int compute_warmup_all_tiles = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_COMPUTE_WARMUP_ALL_TILES", 0) : 0;
    DG_HOST_ASSERT(compute_warmup_all_tiles == 0 || compute_warmup_all_tiles == 1);
    const int compute_warmup_rounds = all_push && (block_m > 32 || compute_warmup_all_tiles)
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_COMPUTE_WARMUP_ROUNDS") : 0;
    DG_HOST_ASSERT(compute_warmup_rounds >= 0 && compute_warmup_rounds <= 8);
    DG_HOST_ASSERT(!compute_warmup_all_tiles || compute_warmup_rounds > 0);
    DG_HOST_ASSERT(!compute_warmup_rounds || (num_dispatch_expert_ctas % 2 == 0 &&
        !get_env<int>("DG_MEGAMOE_ALL_PUSH_ROTATE_COMPUTE")));
    const int compute_busy_sender_ctas = all_push && block_m > 32 &&
        get_env<int>("DG_MEGAMOE_ALL_PUSH_COMPUTE_BUSY_SENDERS")
        ? math::align(num_dispatch_push_ctas + 1, 2) : 0;
    DG_HOST_ASSERT(!compute_busy_sender_ctas || (compute_warmup_rounds &&
        num_sms > 2 * num_dispatch_expert_ctas + compute_busy_sender_ctas));
    const bool all_push_unified_tma = all_push && block_m > 32 &&
        get_env<int>("DG_MEGAMOE_ALL_PUSH_UNIFIED_TMA");
    DG_HOST_ASSERT(!all_push_unified_tma || get_env<int>("DG_MEGAMOE_ALL_PUSH_FULL_TMA"));

    const bool all_push_dedicated_dispatch = all_push && block_m > 32 &&
        get_env<int>("DG_MEGAMOE_ALL_PUSH_DEDICATED_DISPATCH");
    DG_HOST_ASSERT(!all_push_dedicated_dispatch || (all_push_unified_tma &&
        !compute_warmup_rounds && !compute_busy_sender_ctas &&
        !get_env<int>("DG_MEGAMOE_ALL_PUSH_ROTATE_COMPUTE")));

    const int all_push_narrow_sf = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_NARROW_SF") : 0;
    DG_HOST_ASSERT(all_push_narrow_sf >= 0 && all_push_narrow_sf <= 3);
    const int all_push_coalesced_sf = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_COALESCED_SF") : 0;
    DG_HOST_ASSERT(all_push_coalesced_sf >= 0 && all_push_coalesced_sf <= 2);
    const int all_push_prefetch_send = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_PREFETCH_SEND") : 0;
    DG_HOST_ASSERT(all_push_prefetch_send == 0 || all_push_prefetch_send == 1);

    const int all_push_middle_window_end = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_MIDDLE_WINDOW_END", 2048) : 2048;
    DG_HOST_ASSERT(all_push_middle_window_end == 1536 || all_push_middle_window_end == 2048 ||
                   all_push_middle_window_end == 2560 || all_push_middle_window_end == 3072);
    DG_HOST_ASSERT(all_push_middle_window_end == 2048 || all_push_middle_window_end <= hidden);

    const int all_push_half_warp = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_HALF_WARP") : 0;
    DG_HOST_ASSERT(all_push_half_warp == 0 || all_push_half_warp == 1);
    const int all_push_first_window_end = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_FIRST_WINDOW_END", 1024) : 1024;
    DG_HOST_ASSERT(all_push_first_window_end == 512 || all_push_first_window_end == 1024);
    const int all_push_sticky_validation = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_STICKY_VALIDATION", 0) : 0;
    DG_HOST_ASSERT(all_push_sticky_validation == 0 || all_push_sticky_validation == 1);
    const int all_push_full_row_combine = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_FULL_ROW_COMBINE", 0) : 0;
    DG_HOST_ASSERT(all_push_full_row_combine >= 0 && all_push_full_row_combine <= 4);
    const int all_push_weight_cache = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_WEIGHT_CACHE", 0) : 0;
    DG_HOST_ASSERT(all_push_weight_cache == 0 || all_push_weight_cache == 1);
    const int all_push_warp_combine = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_WARP_COMBINE", 0) : 0;
    DG_HOST_ASSERT(all_push_warp_combine >= 0 && all_push_warp_combine <= 4);
    DG_HOST_ASSERT(all_push_warp_combine < 2 || all_push_full_row_combine != 0);
    const int all_push_prefix_sweep = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_PREFIX_SWEEP", 0) : 0;
    DG_HOST_ASSERT(all_push_prefix_sweep == 0 || all_push_prefix_sweep == 1);
    DG_HOST_ASSERT(!all_push_prefix_sweep || !all_push_prefetch_send);
    const int all_push_producer_prefix = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_PRODUCER_PREFIX", 0) : 0;
    DG_HOST_ASSERT(all_push_producer_prefix == 0 || all_push_producer_prefix == 1);
    DG_HOST_ASSERT(!all_push_producer_prefix || (hidden == 4096 && all_push_first_window_end == 1024 &&
                   !all_push_prefix_sweep && !all_push_prefetch_send));
    const int all_push_sender_ldg = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_SENDER_LDG", 0) : 0;
    DG_HOST_ASSERT(all_push_sender_ldg >= 0 && all_push_sender_ldg <= 2);
    DG_HOST_ASSERT(!all_push_sender_ldg || (hidden == 4096 && !all_push_producer_prefix &&
                   !all_push_prefix_sweep && !all_push_prefetch_send));
    DG_HOST_ASSERT(all_push_sender_ldg != 2 ||
                   (num_max_tokens_per_rank % 8 == 0 && all_push_first_window_end == 1024));
    const int all_push_prefix_wide = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_PREFIX_WIDE", 0) : 0;
    DG_HOST_ASSERT(all_push_prefix_wide == 0 || all_push_prefix_wide == 1);
    DG_HOST_ASSERT(!all_push_prefix_wide || (hidden == 4096 && num_max_tokens_per_rank % 8 == 0 &&
                   all_push_first_window_end == 1024));
    const int all_push_group_priority = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_GROUP_PRIORITY", 0) : 0;
    DG_HOST_ASSERT(all_push_group_priority >= 0 && all_push_group_priority <= 2);
    DG_HOST_ASSERT(!all_push_group_priority || get_env<int>("DG_MEGAMOE_ALL_PUSH_PRIORITIZE_EXPERTS"));
    const int all_push_cache_priority = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_CACHE_PRIORITY", 0) : 0;
    DG_HOST_ASSERT(all_push_cache_priority >= 0 && all_push_cache_priority <= 4);
    DG_HOST_ASSERT(all_push_cache_priority != 4 || (num_topk % 4 == 0 && hidden % 1024 == 0));
    const int all_push_return_wide = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_RETURN_WIDE", 0) : 0;
    DG_HOST_ASSERT(all_push_return_wide >= 0 && all_push_return_wide <= 1);
    const int all_push_return_wg_sync = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_RETURN_WG_SYNC", 0) : 0;
    DG_HOST_ASSERT(all_push_return_wg_sync == 0 || all_push_return_wg_sync == 1);
    const int all_push_balance_epilogue = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_BALANCE_EPILOGUE", 0) : 0;
    DG_HOST_ASSERT(all_push_balance_epilogue == 0 || all_push_balance_epilogue == 1);
    const int all_push_fc2_stage_ready = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_FC2_STAGE_READY", 0) : 0;
    DG_HOST_ASSERT(all_push_fc2_stage_ready == 0 || all_push_fc2_stage_ready == 1);
    const int all_push_activation_issuers = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_ACTIVATION_ISSUERS", 1) : 1;
    DG_HOST_ASSERT(all_push_activation_issuers == 1 || all_push_activation_issuers == 2 ||
                   all_push_activation_issuers == 4);
    DG_HOST_ASSERT(all_push_activation_issuers == 1 || (full_activation_tma && !all_push_unified_tma));
    // Partition the full-activation materializer's rows, independently of
    // worker-count selection. Small tiles keep their original dispatch layout.
    const int metadata_row_partitions = full_activation_tma
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_METADATA_ROW_PARTITIONS", 1) : 1;
    DG_HOST_ASSERT(metadata_row_partitions >= 1 && metadata_row_partitions <= 8);
    DG_HOST_ASSERT(num_dispatch_expert_ctas % metadata_row_partitions == 0);
    const int tile_workers = num_dispatch_expert_ctas / metadata_row_partitions;
    DG_HOST_ASSERT(tile_workers <= num_experts_per_rank || tile_workers % num_experts_per_rank == 0);
    DG_HOST_ASSERT(metadata_row_partitions == 1 || (full_activation_tma &&
                   !all_push_coalesced_sf && !compute_warmup_rounds));
    DG_HOST_ASSERT(all_push_return_wide != 1 || (store_block_m % 16 == 0 &&
                   hidden % 1024 == 0 && intermediate_hidden % 128 == 0 &&
                   num_padded_sf_pool_tokens % 32 == 0 && num_max_tokens_per_rank % 8 == 0));
    DG_HOST_ASSERT(!all_push_cache_priority || get_env<int>("DG_MEGAMOE_ALL_PUSH_PRIORITIZE_EXPERTS"));
    DG_HOST_ASSERT(all_push_cache_priority != 3 || !all_push_producer_prefix);
    DG_HOST_ASSERT((all_push_cache_priority != 1 && all_push_cache_priority != 3) ||
                   int64_t(num_max_pool_tokens) >= 4ll * num_max_tokens_per_rank);
    // Modes2/3/4 reuse private dispatch scratch after the helper's histogram.
    // Bound actual rows, not the larger workspace input capacity.
    DG_HOST_ASSERT((all_push_cache_priority < 2 && all_push_group_priority != 2) ||
                   int64_t(ceil_div(num_experts_per_rank + 1, 32)) * 32 *
                       (all_push_group_priority == 2 ? 4 : 1) +
                       (all_push_cache_priority >= 2 ? num_tokens : 0) <=
                       32ll + hidden + hidden / 32);

    // Accumulator depth is independent of the operand and output-store pipelines.
    const int num_epilogue_stages = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_ACCUM_STAGES", 2) : 2;
    DG_HOST_ASSERT(num_epilogue_stages >= 1 && num_epilogue_stages <= 3 &&
        block_m * num_epilogue_stages + (sf_block_m + sf_block_n) / 32 <= 512);

    // Pipeline
    auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden, num_topk,
        block_m, block_n, block_k, store_block_m,
        sf_block_m, sf_block_n,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        num_dispatch_pull_stages, num_epilogue_stages);

    // Spend remaining SMEM on weight lookahead without coupling token credits.
    // Small tiles retain their existing shared pipeline.
    const int weight_stage_extra = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_WEIGHT_STAGE_EXTRA", 0) : 0;
    const int activation_stage_limit = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_ACTIVATION_STAGE_LIMIT", 0) : 0;
    const int keep_smem = all_push ? get_env<int>("DG_MEGAMOE_ALL_PUSH_KEEP_SMEM", 0) : 0;
    DG_HOST_ASSERT(weight_stage_extra >= 0 && weight_stage_extra <= 4);
    DG_HOST_ASSERT(activation_stage_limit == 0 || (activation_stage_limit >= 2 && activation_stage_limit <= 32));
    DG_HOST_ASSERT(keep_smem == 0 || (keep_smem == 1 && (weight_stage_extra || activation_stage_limit)));
    int weight_stages = 0;
    if ((weight_stage_extra || activation_stage_limit) && block_m > 32) {
        DG_HOST_ASSERT(full_activation_tma);
        const int original_smem_size = smem_size;
        const int activation_bytes = load_block_m * block_k + sf_block_m * 4 + 16;
        const int weight_bytes = block_n * block_k + sf_block_n * 4 + 16;
        const int fixed_bytes = smem_size - num_stages * (activation_bytes + weight_bytes - 16);
        weight_stages = num_stages + weight_stage_extra;
        num_stages = std::min(num_stages,
            (SM100ArchSpec::smem_capacity - 1024 - fixed_bytes - weight_stages * weight_bytes) / activation_bytes);
        if (activation_stage_limit)
            num_stages = std::min(num_stages, activation_stage_limit);
        DG_HOST_ASSERT(num_stages >= 2 && weight_stages <= 32);
        smem_size = fixed_bytes + num_stages * activation_bytes + weight_stages * weight_bytes;
        if (keep_smem) {
            // Isolate allocation size from ring depth without changing device code.
            DG_HOST_ASSERT(smem_size <= original_smem_size);
            smem_size = original_smem_size;
        }
    }

    const int activation_credit_limit = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_ACTIVATION_CREDIT_LIMIT", 0) : 0;
    const int fc2_credit_only = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_FC2_CREDIT_ONLY", 0) : 0;
    DG_HOST_ASSERT(activation_credit_limit == 0 ||
                   (activation_credit_limit >= 2 && activation_credit_limit <= 32));
    DG_HOST_ASSERT(fc2_credit_only == 0 || (fc2_credit_only == 1 && activation_credit_limit));
    DG_HOST_ASSERT(block_m <= 32 || activation_credit_limit <= num_stages);
    const int relaxed_poll = all_push
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_RELAXED_POLL", 0) : 0;
    DG_HOST_ASSERT(relaxed_poll >= 0 && relaxed_poll <= 3);

    const int send_middle_first = all_push && block_m > 32
        ? get_env<int>("DG_MEGAMOE_ALL_PUSH_SEND_MIDDLE_FIRST", 0) : 0;
    DG_HOST_ASSERT(send_middle_first == 0 || send_middle_first == 1);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_dispatch_pull_stages, num_dispatch_push_ctas,
        full_activation_tma,
        num_dispatch_expert_ctas,
        all_push && block_m > 32 && get_env<int>("DG_MEGAMOE_ALL_PUSH_PRIORITIZE_EXPERTS"),
        all_push && block_m > 32 && get_env<int>("DG_MEGAMOE_ALL_PUSH_ROTATE_COMPUTE")
            ? num_sms - 2 * num_dispatch_expert_ctas : 0,
        compute_warmup_rounds,
        compute_busy_sender_ctas, all_push_unified_tma, all_push_dedicated_dispatch, num_epilogue_stages,
        all_push && block_m > 32 && get_env<int>("DG_MEGAMOE_ALL_PUSH_LINEAR_SOURCE"),
        all_push && block_m > 32 && get_env<int>("DG_MEGAMOE_ALL_PUSH_CACHE_DESTINATIONS"),
        all_push && block_m > 32 && get_env<int>("DG_MEGAMOE_ALL_PUSH_INTERLEAVE_M"),
        all_push_narrow_sf,
        all_push_coalesced_sf,
        all_push_prefetch_send != 0,
        all_push_middle_window_end, all_push_half_warp != 0, all_push_first_window_end,
        all_push_sticky_validation != 0, all_push_full_row_combine, all_push_weight_cache != 0,
        static_cast<uint32_t>(all_push_warp_combine), all_push_prefix_sweep != 0,
        all_push_producer_prefix != 0, all_push_sender_ldg, all_push_prefix_wide != 0,
        all_push_cache_priority, all_push_group_priority, all_push_return_wide,
        all_push_return_wg_sync != 0, all_push_balance_epilogue != 0,
        full_activation_tma ? extent_tma : 0, late_count_lane != 0,
        all_push_fc2_stage_ready != 0, static_cast<uint32_t>(all_push_activation_issuers),
        metadata_row_partitions, weight_stages,
        block_m > 32 ? activation_credit_limit : 0, block_m > 32 && fc2_credit_only != 0,
        block_m > 32 ? relaxed_poll : 0, send_middle_first
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
