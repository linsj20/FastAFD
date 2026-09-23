#pragma once

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"

namespace deep_gemm::runtime {

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    // Generate the serving specialization without tensors, TMA descriptors,
    // CUDA context creation, or a device-runtime query. CPU sweep preparation
    // uses the same heuristic and source generator as the actual launch.
    m.def("generate_megamoe_kernel", [](int ranks, int experts, int bucket,
            int rows, int topk, int hidden, int intermediate, int sms,
            bool all_push, bool inband_combine, bool rank_gated_combine) {
        DG_HOST_ASSERT(ranks >= 2 && ranks <= 64 && experts > 0 && experts % ranks == 0);
        DG_HOST_ASSERT(bucket > 0 && bucket % layout::kLCMCandidateBlockM == 0);
        DG_HOST_ASSERT(rows > 0 && rows <= bucket && sms > 0 && sms <= 160);
        DG_HOST_ASSERT(topk > 0 && topk <= 32 && hidden > 1024 && hidden % 512 == 0);
        DG_HOST_ASSERT(intermediate > 0 && intermediate % 128 == 0);
        DG_HOST_ASSERT(!inband_combine || rank_gated_combine);
        const int local_experts = experts / ranks;
        const int pool = layout::get_num_max_pool_tokens(ranks, bucket, topk, local_experts);
        int sf_pool = 0;
        for (int block_m: layout::kCandidateBlockM)
            sf_pool = std::max(sf_pool, layout::get_num_padded_sf_pool_tokens(pool, block_m));
        const auto config = get_mega_moe_config(ranks, experts, local_experts,
            bucket, rows, topk, hidden, intermediate, sf_pool, 1, all_push, sms);
        if (all_push) {
            DG_HOST_ASSERT(rows <= 8192 && config.block_m <= 32 && config.num_epilogue_threads >= 256);
            DG_HOST_ASSERT(config.num_experts_per_wave == local_experts);
            DG_HOST_ASSERT(sms >= 2 * std::min(local_experts, 2 * ranks) + 1 +
                layout::Workspace::genphase_pusher_ctas(bucket));
        }
        SM100FP8FP4MegaMoERuntime::Args args{
            .launch_args = LaunchArgs(sms,
                config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                config.smem_size, 2)
        };
        args.num_max_tokens_per_rank = bucket;
        args.hidden = hidden;
        args.intermediate_hidden = intermediate;
        args.num_experts = experts;
        args.num_topk = topk;
        args.num_ranks = ranks;
        args.activation_clamp = std::numeric_limits<float>::infinity();
        args.fast_math = true;
        args.routes_prepared = args.route_ready_dispatch = args.rank_ready_route_publish = true;
        args.rank_gated_combine = rank_gated_combine;
        args.inband_combine = inband_combine;
        args.all_push = all_push;
        args.config = config;
        std::ostringstream description;
        description << config;
        return pybind11::make_tuple(SM100FP8FP4MegaMoERuntime::generate_impl(args),
            description.str(), config.block_m, config.num_experts_per_wave);
    });
    m.def("precompile_kernel", [](const std::string& name, const std::string& code,
                                  const std::string& arch) {
        DG_HOST_ASSERT(std::regex_match(arch, std::regex(R"([0-9]+[af])")));
        const auto generated = fmt::format("// Includes' hash value: {}\n{}",
            include_parser->get_hash_value(code), code);
        return NVCCCompiler(arch).precompile(name, generated).string();
    });
#endif
    m.def("set_num_sms", [&](const int& new_num_sms) {
        device_runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return device_runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        device_runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return device_runtime->get_tc_util();
    });
    m.def("set_pdl", [&](const bool& new_enable_pdl) {
        device_runtime->set_pdl(new_enable_pdl);
    });
    m.def("get_pdl", [&]() {
        return device_runtime->get_pdl();
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
    m.def("init", [&](const std::string& library_root_path, const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
        IncludeParser::prepare_init(library_root_path);
#endif
    });
}

} // namespace deep_gemm::runtime
