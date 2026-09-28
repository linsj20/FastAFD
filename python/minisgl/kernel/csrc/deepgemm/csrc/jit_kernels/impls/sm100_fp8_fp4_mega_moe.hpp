#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool use_fp8_weights;
        bool trace_recycling;
        bool routes_prepared;
        bool rank_gated_combine;
        bool route_ready_dispatch;
        bool rank_ready_route_publish;
        bool inband_combine;
        bool all_push;
        int reset_rows;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        unsigned long long* debug_timings;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        layout::TmaDescriptors<CUtensorMap, 16> tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.config.all_push_metadata_row_partitions == 1 || args.reset_rows == 0);
        return fmt::format(R"(
#include <deep_gemm/impls/{}>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}{}
    >);
}};
)", args.all_push ? "sm100_fp8_fp4_mega_moe_all_push.cuh" : "sm100_fp8_fp4_mega_moe.cuh",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.use_fp8_weights ? "true" : "false",
    args.trace_recycling ? "true" : "false",
    args.routes_prepared ? "true" : "false",
    args.rank_gated_combine ? "true" : "false",
    args.route_ready_dispatch ? "true" : "false",
    args.rank_ready_route_publish ? "true" : "false",
    args.inband_combine ? "true" : "false",
    args.all_push ? "true" : "false",
    args.all_push ? fmt::format(", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}", args.reset_rows, args.config.num_dispatch_push_ctas,
                               args.config.all_push_full_tma ? "true" : "false",
                               args.config.num_dispatch_expert_ctas,
                               args.config.all_push_prioritize_experts ? "true" : "false",
                               args.config.compute_cta_offset, args.config.compute_warmup_rounds,
                               args.config.compute_busy_sender_ctas,
                               args.config.all_push_unified_tma ? 1 : 0,
                               args.config.all_push_dedicated_dispatch ? 1 : 0,
                               args.config.num_epilogue_stages,
                               args.config.all_push_linear_source ? 1 : 0,
                               args.config.all_push_cache_destinations ? 1 : 0,
                               args.config.all_push_interleave_m ? 1 : 0,
                               args.config.all_push_narrow_sf,
                               args.config.all_push_coalesced_sf,
                               args.config.all_push_prefetch_send ? 1 : 0,
                               args.config.all_push_middle_window_end,
                               args.config.all_push_half_warp ? 1 : 0,
                               args.config.all_push_first_window_end,
                               args.config.all_push_sticky_validation ? 1 : 0,
                               args.config.all_push_full_row_combine,
                               args.config.all_push_weight_cache ? 1 : 0,
                               args.config.all_push_warp_combine,
                               args.config.all_push_prefix_sweep ? 1 : 0,
                               args.config.all_push_producer_prefix ? 1 : 0,
                               args.config.all_push_sender_ldg,
                               args.config.all_push_prefix_wide ? 1 : 0,
                               args.config.all_push_cache_priority,
                               args.config.all_push_group_priority,
                               args.config.all_push_return_wide,
                               args.config.all_push_return_wg_sync ? 1 : 0,
                               args.config.all_push_balance_epilogue ? 1 : 0,
                               args.config.all_push_extent_tma,
                               args.config.all_push_late_count_lane ? 1 : 0,
                               args.config.all_push_fc2_stage_ready ? 1 : 0,
                               args.config.all_push_activation_issuers,
                               args.config.all_push_metadata_row_partitions,
                               args.config.all_push_weight_stages,
                               args.config.all_push_activation_credit_limit,
                               args.config.all_push_fc2_credit_only ? 1 : 0,
                               args.config.all_push_relaxed_poll,
                               args.config.all_push_send_middle_first) : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.debug_timings,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

static void sm100_fp8_mega_moe_impl(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> debug_timings,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& use_fp8_weights,
    const bool& routes_prepared,
    const bool& rank_gated_combine,
    const bool& route_ready_dispatch,
    const bool& rank_ready_route_publish,
    const bool& inband_combine, const bool& all_push, const int& reset_rows = 0
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        /* num_dispatch_pull_stages */ 1, all_push);
    DG_HOST_ASSERT(config.all_push_full_row_combine != 4 ||
                   reinterpret_cast<uintptr_t>(y.data_ptr()) % 32 == 0);
    if (config.all_push_return_wide) {
        DG_HOST_ASSERT(inband_combine);
        for (const auto ptr : sym_buffer_ptrs)
            DG_HOST_ASSERT(static_cast<uint64_t>(ptr) % 32 == 0);
    }

    // Make tensormap
    constexpr int kGranK = 32;
    layout::TmaDescriptors<CUtensorMap, 16> tensor_map_l1_acts{};
    const int num_activation_maps = config.all_push_extent_tma ? config.block_m / 16 : 1;
    DG_HOST_ASSERT(num_activation_maps > 0 && num_activation_maps <= 16);
    for (int i = 0; i < num_activation_maps; ++i) {
        const int rows = config.all_push_extent_tma ? 8 * (i + 1)
            : all_push && !config.all_push_full_tma ? 8 : config.load_block_m;
        tensor_map_l1_acts.maps[i] = make_tma_2d_desc(l1_acts,
            hidden, all_push ? 2 * num_ranks * num_max_tokens_per_rank : static_cast<int>(l1_acts.size(0)),
            config.block_k, rows, static_cast<int>(l1_acts.stride(-2)), config.swizzle_acts_mode);
    }
    // Genphase stores all SF atoms of a pool block together, so refinement can
    // bulk-store contiguous atom runs instead of striding across the full pool.
    const auto tensor_map_l1_acts_sf = all_push ? make_tma_2d_desc(l1_acts_sf,
                                                        config.sf_block_m,
                                                        config.num_padded_sf_pool_tokens / config.sf_block_m * (hidden / 128),
                                                        config.sf_block_m, 1, config.sf_block_m, 0) :
                                                    make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    unsigned long long* debug_timings_ptr = nullptr;
    if (debug_timings.has_value())
        debug_timings_ptr = reinterpret_cast<unsigned long long*>(
            debug_timings->data_ptr<int64_t>());

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .use_fp8_weights = use_fp8_weights,
        .trace_recycling = debug_timings_ptr != nullptr,
        .routes_prepared = routes_prepared,
        .rank_gated_combine = rank_gated_combine,
        .route_ready_dispatch = route_ready_dispatch,
        .rank_ready_route_publish = rank_ready_route_publish,
        .inband_combine = inband_combine,
        .all_push = all_push,
        .reset_rows = reset_rows,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .debug_timings = debug_timings_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    if (all_push) {
        DG_HOST_ASSERT(num_tokens <= 8192);
        DG_HOST_ASSERT(num_topk > 0 && num_topk <= 32);
        DG_HOST_ASSERT(hidden > 1024 && hidden % 512 == 0);
        DG_HOST_ASSERT(config.block_m <= 256 && config.num_epilogue_threads >= 256);
        DG_HOST_ASSERT(config.num_experts_per_wave > 0 && num_experts_per_rank % config.num_experts_per_wave == 0);
        const int metadata_ctas = 2 * config.num_dispatch_expert_ctas;
        const int pushers = config.num_dispatch_push_ctas;
        DG_HOST_ASSERT(num_sms >= metadata_ctas + 1 + pushers);
        // Device static assertions cover disjoint scratch and TMA transaction sizes.
        // Actual padded materialization capacity is checked by the receiver.
    }
    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build(
        use_fp8_weights ? "sm100_fp8_fp8_mega_moe" : "sm100_fp8_fp4_mega_moe", code);
    // Only the route-ready producer has an independent release/acquire epoch
    // that lets MegaMoE consume data without waiting for the whole selector
    // grid. Other callers retain ordinary stream ordering.
    // Router auxiliary CTAs prepare the next activation bank. Norm follows
    // this producer directly and joins its output through hardware PDL.
    SM100FP8FP4MegaMoERuntime::launch(
        runtime, args, /* force_pdl */ route_ready_dispatch);
}

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> debug_timings,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& routes_prepared,
    const bool& rank_gated_combine,
    const bool& route_ready_dispatch,
    const bool& rank_ready_route_publish,
    const bool& inband_combine, const bool& all_push, const int& reset_rows = 0
) {
    sm100_fp8_mega_moe_impl(
        y, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
        l1_weights, l2_weights, l1_weights_sf, l2_weights_sf,
        cumulative_local_expert_recv_stats, debug_timings, sym_buffer_ptrs,
        rank_idx, num_max_tokens_per_rank, num_experts_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden,
        activation_clamp, fast_math, /* use_fp8_weights */ false,
        routes_prepared, rank_gated_combine, route_ready_dispatch, rank_ready_route_publish, inband_combine, all_push, reset_rows);
}

static void sm100_fp8_fp8_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> debug_timings,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& routes_prepared,
    const bool& rank_gated_combine,
    const bool& route_ready_dispatch,
    const bool& rank_ready_route_publish,
    const bool& inband_combine, const bool& all_push, const int& reset_rows = 0
) {
    sm100_fp8_mega_moe_impl(
        y, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf,
        l1_weights, l2_weights, l1_weights_sf, l2_weights_sf,
        cumulative_local_expert_recv_stats, debug_timings, sym_buffer_ptrs,
        rank_idx, num_max_tokens_per_rank, num_experts_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden,
        activation_clamp, fast_math, /* use_fp8_weights */ true,
        routes_prepared, rank_gated_combine, route_ready_dispatch, rank_ready_route_publish, inband_combine, all_push, reset_rows);
}

} // namespace deep_gemm
