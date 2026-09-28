#pragma once

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/impls/mega_moe_all_push_dispatch.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

#ifdef IKET_BUILD_ENABLED
#define IKET_DEFAULT_INSTR_METHOD_WITHOUT_PAYLOAD_PMTRIG5
#include <iket/iket_device_apis.cuh>
CREATE_IKET_MARKER(MegaDispatchStart);
CREATE_IKET_MARKER(MegaDispatchEnd);
CREATE_IKET_MARKER(MegaMetadataStart);
CREATE_IKET_MARKER(MegaComputeEnd);
CREATE_IKET_MARKER(MegaCombineStart);
CREATE_IKET_MARKER(MegaCombineEnd);
CREATE_IKET_MARKER(MegaLocalReduceStart);
CREATE_IKET_MARKER(MegaLocalReduceEnd);
CREATE_IKET_MARKER(MegaMetadataAcquired);
CREATE_IKET_MARKER(MegaIndicesReady);
CREATE_IKET_MARKER(MegaPayloadReady);
CREATE_IKET_MARKER(MegaGatherSfReady);
CREATE_IKET_MARKER(MegaGatherIssued);
CREATE_IKET_MARKER(MegaWindowWaitStart);
CREATE_IKET_MARKER(MegaWindowReady);
CREATE_IKET_MARKER(MegaMmaIssue);
#define MEGAMOE_IKET_MARK(name) IKET_MARK(name)
#define MEGAMOE_IKET_DETAIL_MARK(name) IKET_MARK(name)
#define MEGAMOE_IKET_FIRST(name) bool name = true
#define MEGAMOE_IKET_ONCE(name, first) do { if (first) IKET_MARK(name); } while (0)
#define MEGAMOE_IKET_DONE(first) first = false
#else
#define MEGAMOE_IKET_MARK(name) ((void)0)
#define MEGAMOE_IKET_DETAIL_MARK(name) ((void)0)
#define MEGAMOE_IKET_FIRST(name) ((void)0)
#define MEGAMOE_IKET_ONCE(name, first) ((void)0)
#define MEGAMOE_IKET_DONE(first) ((void)0)
#endif

namespace deep_gemm {

// Keep the ready fast path acquiring. Only unsuccessful spins use relaxed
// loads; an acquire must observe readiness before any consumer may proceed.
template <typename T, bool kExact = false>
CUTLASS_DEVICE T wait_ready_mask_relaxed(const T* ptr, const T mask) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    const auto ready = [=](T value) { return kExact ? value == mask : (value & mask) == mask; };
    T value;
    while (true) {
        if constexpr (sizeof(T) == 4)
            value = ptx::ld_acq(ptr);
        else
            value = ptx::ld_acq_gpu(ptr);
        if (ready(value)) return value;
        do {
            if constexpr (sizeof(T) == 4)
                asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
            else
                asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
        } while (!ready(value));
    }
}

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kUseFP8Weights = false,
    bool kTraceRecycling = false,
    bool kRoutesPrepared = false,
    bool kRankGatedCombine = false,
    bool kRouteReadyDispatch = false,
    bool kRankReadyRoutePublish = false,
    bool kInbandCombine = false,
    bool kAllPush = false,
    uint32_t kResetRows = 0,
    uint32_t kNumDispatchPushCTAs = layout::Workspace::genphase_pusher_ctas(kNumMaxTokensPerRank),
    bool kFullActivationTMA = false,
    uint32_t kNumDispatchExpertCTAs = (kNumExperts / kNumRanks < 2 * kNumRanks ? kNumExperts / kNumRanks : 2 * kNumRanks),
    bool kPrioritizeExperts = false,
    uint32_t kComputeCTAOffset = 0,
    uint32_t kComputeWarmupRounds = 0,
    uint32_t kComputeBusySenderCTAs = 0,
    uint32_t kUnifiedTMA = 0,
    uint32_t kDedicatedDispatch = 0,
    uint32_t kNumEpilogueStages = 2,
    uint32_t kLinearSource = 0,
    uint32_t kCacheDestinations = 0,
    uint32_t kInterleaveM = 0,
    uint32_t kNarrowSf = 0,
    uint32_t kCoalescedSf = 0,
    uint32_t kPrefetchSend = 0,
    uint32_t kMiddlePayloadWindowEnd = 2048,
    uint32_t kHalfWarp = 0,
    uint32_t kFirstReadinessWindowBytes = 1024,
    uint32_t kStickyValidation = 0,
    uint32_t kFullRowCombine = 0,
    uint32_t kWeightCache = 0,
    uint32_t kWarpCombine = 0,
    uint32_t kPrefixSweep = 0,
    uint32_t kProducerPrefix = 0,
    uint32_t kSenderLdg = 0,
    uint32_t kPrefixWide = 0,
    uint32_t kCachePriority = 0,
    uint32_t kGroupPriority = 0,
    uint32_t kReturnWide = 0,
    uint32_t kReturnWarpgroupSync = 0,
    uint32_t kBalanceEpilogue = 0,
    uint32_t kExtentActivationTMA = 0,
    uint32_t kLateCountLane = 0,
    uint32_t kFC2StageReady = 0,
    uint32_t kActivationIssuers = 1,
    uint32_t kMetadataRowPartitions = 1,
    uint32_t kWeightStages = 0,
    uint32_t kActivationCreditLimit = 0,
    uint32_t kFC2CreditOnly = 0,
    uint32_t kRelaxedPoll = 0,
    uint32_t kSendMiddleFirst = 0,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,
                            int* cumulative_local_expert_recv_stats,
                            unsigned long long* debug_timings,
                            const uint32_t num_tokens,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ std::conditional_t<(kExtentActivationTMA != 0),
                                layout::TmaDescriptors<cute::TmaDescriptor, BLOCK_M / 16>,
                                cute::TmaDescriptor> tensor_map_l1_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kAllPush, "This implementation requires all-push storage");
    DG_STATIC_ASSERT(kLinearSource <= 1, "Invalid source decoding choice");
    DG_STATIC_ASSERT(kInterleaveM <= 1, "Invalid tile order choice");
    DG_STATIC_ASSERT(kNarrowSf <= 3, "Invalid SF worker split");
    DG_STATIC_ASSERT(kCoalescedSf <= 2, "Invalid SF packing order");
    DG_STATIC_ASSERT(kPrefetchSend <= 1, "Invalid sender prefetch setting");
    DG_STATIC_ASSERT(kHalfWarp <= 1, "Invalid validator row ownership");
    DG_STATIC_ASSERT(kStickyValidation <= 1, "Invalid sticky validation option");
    DG_STATIC_ASSERT(kFullRowCombine <= 4, "Invalid combine row option");
    DG_STATIC_ASSERT(kWeightCache <= 1, "Invalid weight cache option");
    DG_STATIC_ASSERT(kReturnWarpgroupSync <= 1, "Invalid return synchronization option");
    DG_STATIC_ASSERT(kBalanceEpilogue <= 1, "Invalid epilogue balance option");
    DG_STATIC_ASSERT(kLateCountLane <= 1, "Invalid count lane option");
    DG_STATIC_ASSERT(kFC2StageReady <= 1, "Invalid FC2 readiness option");
    DG_STATIC_ASSERT((kActivationIssuers == 1 || kActivationIssuers == 2 || kActivationIssuers == 4) &&
                     kActivationIssuers <= kNumDispatchWarps && kActivationIssuers <= kNumStages &&
                     (kActivationIssuers == 1 || (kFullActivationTMA && !kUnifiedTMA)),
                     "Activation issuers require existing dispatch warps and separate full-box loads");
#ifdef IKET_BUILD_ENABLED
    DG_STATIC_ASSERT(kActivationIssuers == 1, "Multiple activation issuers need a separate profile marker contract");
#endif
    DG_STATIC_ASSERT(kExtentActivationTMA <= 2 && (!kExtentActivationTMA ||
                     (kFullActivationTMA && BLOCK_M % 16 == 0 && BLOCK_K == 128)),
                     "Extent TMA requires full boxes and one FP8 inner atom");
    const auto l1_activation_map = [&](const uint32_t aligned_rows) -> const cute::TmaDescriptor* {
        if constexpr (kExtentActivationTMA)
            return &tensor_map_l1_acts.maps[aligned_rows / 16 - 1];
        else
            return &tensor_map_l1_acts;
    };
    DG_STATIC_ASSERT(kPrefixSweep <= 1 && (!kPrefixSweep || !kPrefetchSend),
                     "Prefix sweep requires ordinary sender retirement");
    DG_STATIC_ASSERT(kProducerPrefix <= 1 && (!kProducerPrefix ||
                     (kHidden == 4096 && kFirstReadinessWindowBytes == 1024 &&
                      !kPrefixSweep && !kPrefetchSend)),
                     "Invalid producer prefix option");
    DG_STATIC_ASSERT(kSenderLdg <= 2 && (!kSenderLdg ||
                     (kHidden == 4096 && !kProducerPrefix && !kPrefixSweep && !kPrefetchSend)),
                     "Direct sender loads require ordinary full-row retirement");
    DG_STATIC_ASSERT(kSenderLdg != 2 || (kNumMaxTokensPerRank % 8 == 0 && kFirstReadinessWindowBytes == 1024),
                     "Wide sender vectors require aligned rows and a 1KB first window");
    DG_STATIC_ASSERT(kPrefixWide <= 1 && (!kPrefixWide || (kHidden == 4096 &&
                     kNumMaxTokensPerRank % 8 == 0 && kFirstReadinessWindowBytes == 1024)),
                     "Wide prefix validation requires aligned rows and a 1KB first window");
    DG_STATIC_ASSERT(kCachePriority <= 4 && (!kCachePriority || kPrioritizeExperts) &&
                     ((kCachePriority != 1 && kCachePriority != 3) ||
                      kNumMaxPoolTokens >= 4ull * kNumMaxTokensPerRank),
                     "Invalid priority cache mode or global key capacity");
    DG_STATIC_ASSERT(kCachePriority != 4 || (kNumTopk % 4 == 0 && kHidden % 1024 == 0), "Priority vector alignment");
    DG_STATIC_ASSERT(kGroupPriority <= 2 && (!kGroupPriority || kPrioritizeExperts),
                     "Grouped priority atomics require prioritized queues");
    DG_STATIC_ASSERT(kWarpCombine <= 4 && (!kWarpCombine || kInbandCombine) &&
                     (kWarpCombine < 2 || kFullRowCombine != 0),
                     "Warp-local combine requires in-band payloads");
    DG_STATIC_ASSERT((kFirstReadinessWindowBytes == 512 || kFirstReadinessWindowBytes == 1024) &&
                     kFirstReadinessWindowBytes % BLOCK_K == 0, "Invalid first readiness window");
    DG_STATIC_ASSERT(kCacheDestinations <= 1 && (!kCacheDestinations || kInbandCombine),
                     "Destination cache requires banked in-band combine");
    DG_STATIC_ASSERT(kReturnWide <= 1 && (!kReturnWide || kInbandCombine) && (kReturnWide != 1 || (
                     STORE_BLOCK_M % 16 == 0 && kHidden % 1024 == 0 &&
                     kIntermediateHidden % 128 == 0 && kNumPaddedSFPoolTokens % 32 == 0 && kNumMaxTokensPerRank % 8 == 0)),
                     "Wide return requires aligned in-band rows and sixteen-row staging");
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    constexpr bool kSplitPipeline = kWeightStages != 0;
    constexpr uint32_t kNumWeightStages = kSplitPipeline ? kWeightStages : kNumStages;
    DG_STATIC_ASSERT(!kSplitPipeline || (kFullActivationTMA &&
                     kNumStages >= 2 && kNumWeightStages >= 2 && kNumWeightStages <= 32),
                     "Separate weight stages require complete activation boxes and operand barriers");

    DG_STATIC_ASSERT(kRoutesPrepared && kRouteReadyDispatch && kRankReadyRoutePublish,
                     "All-push requires prepared rank-ready routes");
    constexpr bool kLastCTACleanup = kRankGatedCombine and kInbandCombine;
    DG_STATIC_ASSERT(
        static_cast<uint64_t>(kNumRanks) * kNumMaxTokensPerRank * kNumTopk <= 0xffffffffu,
        "Dense source descriptor exceeds uint32");

    constexpr uint32_t kMetadataExpertCTAs = kNumDispatchExpertCTAs;
    constexpr uint32_t kMetadataCTAs = 2 * kMetadataExpertCTAs;
    DG_STATIC_ASSERT(kMetadataRowPartitions == 1 || (kFullActivationTMA &&
                     !kResetRows && !kCoalescedSf && !kComputeWarmupRounds),
                     "Row partitions require ordinary full-activation materialization");
    constexpr uint32_t kSfReadyMask = dispatch_ready_mask<kMetadataRowPartitions>(kDispatchSfReady);
    constexpr uint32_t kFirstAReadyMask = dispatch_ready_mask<kMetadataRowPartitions>(kDispatchFirstAReady);
    constexpr uint32_t kMiddleAReadyMask = dispatch_ready_mask<kMetadataRowPartitions>(kDispatchMiddleAReady);
    constexpr uint32_t kTailAReadyMask = dispatch_ready_mask<kMetadataRowPartitions>(kDispatchTailAReady);
    constexpr uint64_t kSourceL1Tokens = 2ull * kNumRanks * kNumMaxTokensPerRank;
    constexpr uint64_t kSourceStorageRows = (kSourceL1Tokens * (kHidden + kHidden / 32) + kHidden - 1) / kHidden;
    constexpr uint64_t kFinalL1Tokens = kSourceStorageRows > kNumMaxPoolTokens ?
        kSourceStorageRows : kNumMaxPoolTokens;
    DG_STATIC_ASSERT(kFinalL1Tokens <= 0x7fffffffull, "L1 tensor exceeds signed row indexing");

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(not kInbandCombine or kRankGatedCombine,
                     "In-band combine requires rank-ready routes and ordered arena invocations");
    DG_STATIC_ASSERT(!kDedicatedDispatch || (kUnifiedTMA && kFullActivationTMA &&
                     !kComputeWarmupRounds && !kComputeBusySenderCTAs && !kComputeCTAOffset),
                     "Dedicated dispatch requires unified TMA and uniform compute scheduling");
    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    DG_STATIC_ASSERT(kComputeCTAOffset < kNumSMs && kComputeCTAOffset % 2 == 0,
                     "Compute rotation must preserve adjacent CTA pairs");
    const uint32_t compute_sm_idx = kComputeCTAOffset == 0 ? sm_idx : (sm_idx + kComputeCTAOffset) % kNumSMs;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    if (warp_idx < kNumDispatchWarps)
        MEGAMOE_IKET_MARK(MegaDispatchStart);

    // Let a compute warp prefetch while dispatch starts its routing exchange.
    if (warp_idx == kNumDispatchWarps) {
        if constexpr (kExtentActivationTMA == 1) {
            #pragma unroll
            for (uint32_t rows = 16; rows <= BLOCK_M; rows += 16)
                cute::prefetch_tma_descriptor(l1_activation_map(rows));
        } else if constexpr (kExtentActivationTMA == 0) {
            cute::prefetch_tma_descriptor(l1_activation_map(BLOCK_M));
        }
        // Mode 2 leaves extent descriptors to demand fetch, avoiding startup
        // prefetches for row extents that this CTA may never use.

        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // Workspaces
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    const uint32_t reset_epoch = kResetRows ?
        ptx::ld_acq(workspace.get_nvl_barrier_counter_ptr()) : 0;
    DG_STATIC_ASSERT(kResetRows == 0 || kLastCTACleanup,
                     "Tail reset requires all-push in-band combine");
    DG_STATIC_ASSERT(kResetRows <= kNumMaxTokensPerRank, "Reset exceeds capacity");
    DG_STATIC_ASSERT(kResetRows == 0 || kNumRanks >= 2, "Recycling signals require two ranks");

    // Token and buffer layouts
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Registered inputs
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, static_cast<uint32_t>(kFinalL1Tokens),
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk * (kInbandCombine ? 2 : 1), kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );

    // Match the split AG/EG MegaMoE compute path when kUseFP8Weights is set.
    // Packed FP4 and FP8 weights both occupy one byte per shared-memory value;
    // only the global-memory transaction size and MMA operand type differ.
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = std::conditional_t<kUseFP8Weights,
                                         cutlass::float_e4m3_t,
                                         cutlass::detail::float_e2m1_unpacksmem_t>;

    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    DG_STATIC_ASSERT(kNumEpilogueStages >= 1 && kNumEpilogueStages <= 3, "Invalid accumulator depth");
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * SMEM_A_SIZE_PER_STAGE + kNumWeightStages * SMEM_B_SIZE_PER_STAGE;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    // NOTES: GEMM shared memory starts after the dispatch region, aligned to 1024 bytes
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE
    );

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SF shared memory: SFA and SFB per pipeline stage
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + kNumWeightStages * SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Epilogue amax reduction shared memory
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumWeightStages]);

    // CTA combine uses one barrier per selected expert. Some Qwen3.5
    // geometries have four epilogue warps but ten selected experts.
    constexpr uint32_t kNumCombineBarriers =
        ((kNumTopk == 8 or kNumTopk == 10) and kHidden == 4096 and
         kNumTopk > kNumEpilogueWarps * 2) ? kNumTopk : kNumEpilogueWarps * 2;

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i); });
    auto combine_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + i); });
    auto weight_barrier_start = barrier_start_ptr +
        (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumCombineBarriers);
    auto weight_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return kSplitPipeline ? weight_barrier_start + i : full_barriers[i];
    });
    auto weight_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return kSplitPipeline ? weight_barrier_start + kNumWeightStages + i : empty_barriers[i];
    });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(weight_barrier_start +
        (kSplitPipeline ? 2 * kNumWeightStages : 0));

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(i == 0 ? 32 : 1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at all CTAs
                // A multiwave activation issuer must participate even in FC2;
                // observing empty phases alone allows the other warps to lap it.
                full_barriers[i]->init((kNumExpertsPerWave == kNumExpertsPerRank || kUnifiedTMA ? 4 : 6) -
                    (kSplitPipeline ? 2 : 0));
                empty_barriers[i]->init(1);
            }
            if constexpr (kSplitPipeline) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumWeightStages; ++i) {
                    weight_full_barriers[i]->init(2);  // One weight producer in each CTA.
                    weight_empty_barriers[i]->init(1);
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                tmem_full_barriers[i]->init(1);
                // Arrive only at the leader CTA
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumCombineBarriers; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    // Adjust registers
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = kDedicatedDispatch ? 168 : 208;
    constexpr uint32_t kNumDispatchRegisters = kDedicatedDispatch ? 128 : 48;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Set the dispatch register budget before transport so epilogue allocation can overlap it.
    if constexpr (kDedicatedDispatch) {
        if (warp_idx < kNumDispatchWarps)
            cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    // Startup uses disjoint scratch and can overlap other warps' initialization.
    cute::cluster_arrive_relaxed();

    {
        constexpr uint32_t kRefiners = kNumDispatchPushCTAs;
        constexpr uint32_t kWorkerThread = kDedicatedDispatch ? 0 : kNumDispatchThreads + kNumNonEpilogueThreads;
        constexpr uint32_t kScratchOffset = math::constexpr_align<uint32_t>(kNumExpertsPerRank * sizeof(uint32_t), 128u);
        constexpr uint32_t kScratchBytes = 128 + 4 * (kHidden + kHidden / 32);
        DG_STATIC_ASSERT(kNumDispatchThreads == 128 && kNumEpilogueThreads >= 256,
                         "Wide metadata requires two complete epilogue warpgroups");
        DG_STATIC_ASSERT(kRefiners > 0 && kNumSMs >= kMetadataCTAs + 1 + kRefiners,
                         "Invalid dispatch participant count");
        DG_STATIC_ASSERT(kScratchOffset + kScratchBytes <= SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE,
                         "Concurrent dispatch scratch overlaps GEMM storage");
        // Unified TMA frees dispatch warps for transport; the original path
        // borrows epilogue warps until startup completes. Scratch stays disjoint.
        const uint32_t worker_threads = !kDedicatedDispatch && sm_idx < kMetadataCTAs ? 256 : 128;
        if (sm_idx < kMetadataCTAs + 1 + kRefiners &&
            thread_idx >= kWorkerThread && thread_idx < kWorkerThread + worker_threads) {
            mega_moe_dispatch<kNumMaxTokensPerRank, kHidden, kNumExperts,
                kNumTopk, kNumRanks, BLOCK_M, kNumPaddedSFPoolTokens, kRefiners, kMetadataExpertCTAs,
                (kResetRows != 0), kPrioritizeExperts, (kDedicatedDispatch != 0), (kLinearSource != 0),
                (kCacheDestinations != 0), kNarrowSf, kCoalescedSf,
                (kPrefetchSend != 0), kMiddlePayloadWindowEnd, (kHalfWarp != 0), kFirstReadinessWindowBytes, (kStickyValidation != 0), (kPrefixSweep != 0), (kProducerPrefix != 0), kSenderLdg, (kPrefixWide != 0), kCachePriority, kGroupPriority, kMetadataRowPartitions, (kSendMiddleFirst != 0)>(
                    num_tokens, sym_buffer, smem_buffer + kScratchOffset, thread_idx - kWorkerThread,
                    combine_token_buffer);
            if (thread_idx / 32 == kWorkerThread / 32)
                MEGAMOE_IKET_MARK(MegaDispatchEnd);
        }
    }

    // Acquires all barrier/TMEM initialization before any role consumes it.
    // Worker warps have only used their separate dispatch scratch so far.
    cute::cluster_wait();

    // Rank-ready publication replaces the per-expert completion atomics.
    // Its release/acquire epoch is the route-data dependency, so a PDL launch
    // can overlap all descriptor and CTA-local setup above without adding a
    // redundant full-selector-grid wait here.
    // Aggregate each local expert once per CTA, then share exact totals with
    // every scheduler role through the existing expert-count shared memory.
    {
        // An idle compute warp publishes counts while dispatch performs transport.
        constexpr uint32_t kCountWarp = kDedicatedDispatch ? kNumDispatchWarps + 3 : 0;
        if (warp_idx == kCountWarp) {
            MEGAMOE_IKET_DETAIL_MARK(MegaMetadataStart);
            // The producer's all-rank SYS release already finalizes dense counts.
            // Acquire it directly, as pull does; no local SizesReady relay is
            // needed before this warp reads those same immutable counters.
            while (ptx::ld_acq_sys(workspace.get_route_ready_count_ptr()) != kNumRanks);
            MEGAMOE_IKET_MARK(MegaMetadataAcquired);
            uint32_t count_lane = lane_idx;
            // Producer/transport changes can extend this address across register
            // changes. Recompute it at its actual use.
            if constexpr (kLateCountLane)
                asm volatile("mov.u32 %0, %%laneid;" : "=r"(count_lane));
            for (uint32_t local_expert_idx = count_lane;
                 local_expert_idx < kNumExpertsPerRank;
                 local_expert_idx += 32) {
                // The existing all-rank acquire orders descriptor allocation and stores.
                smem_expert_count[local_expert_idx] = static_cast<uint32_t>(
                    ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(local_expert_idx)));
            }
        }
        // One release arrival from each metadata-writing lane. Consumers
        // acquire the count stores without waiting for startup workers.
        // CUTLASS's inline PTX wrappers omit compiler memory clobbers.
        // Keep the generic shared stores/loads on the ordered side too.
        asm volatile("" ::: "memory");
        if (warp_idx == kCountWarp)
            dispatch_barriers[0]->arrive();
        dispatch_barriers[0]->wait(0);
        asm volatile("" ::: "memory");
    }

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks,
        /* kRecvCountHighTarget = */ kNumSMs * kNumRanks,
        /* kAcquireRecvCounts = */ kRouteReadyDispatch,
        /* kRankReadyRecvCounts = */ kRankReadyRoutePublish,
        layout::Workspace, (kComputeWarmupRounds != 0), (kInterleaveM != 0),
        (kLateCountLane != 0)>(workspace, smem_expert_count);
    // Put the shorter tail assignments on receivers without moving dispatch,
    // combine or cleanup ownership. Every role in a CTA uses the same rotation.
    scheduler.block_idx = compute_sm_idx;
    if constexpr (kComputeWarmupRounds != 0) {
        DG_STATIC_ASSERT(kComputeCTAOffset == 0 && kMetadataExpertCTAs % 2 == 0,
                         "Startup assignment requires even receiver groups and no separate rotation");
        // Estimate busy receiver pairs; a routing imbalance changes the load
        // balance, never tile ownership. Optionally include the helper/sender
        // pairs, whose epilogue warps remain occupied by dispatch.
        DG_STATIC_ASSERT(kComputeBusySenderCTAs == 0 ||
            (kComputeBusySenderCTAs == math::constexpr_align(kNumDispatchPushCTAs + 1, 2u) &&
             kNumSMs > 2 * kMetadataExpertCTAs + kComputeBusySenderCTAs),
            "Busy senders require complete CTA pairs and remaining fast CTAs");
        const uint32_t active_workers = cute::min(kMetadataExpertCTAs, math::align<uint32_t>(
            kNumExpertsPerRank * math::ceil_div(num_tokens * kNumRanks * kNumTopk, kNumExperts * BLOCK_M), 2u));
        const uint32_t fast_ctas = kNumSMs - 2 * active_workers - kComputeBusySenderCTAs;
        const uint32_t logical_idx = sm_idx < active_workers ? fast_ctas + sm_idx
            : sm_idx < kMetadataExpertCTAs ? sm_idx - active_workers
            : sm_idx < kMetadataExpertCTAs + active_workers
                ? fast_ctas + active_workers + sm_idx - kMetadataExpertCTAs
            : sm_idx < 2 * kMetadataExpertCTAs ? sm_idx - 2 * active_workers
            : sm_idx < 2 * kMetadataExpertCTAs + kComputeBusySenderCTAs
                ? fast_ctas + 2 * active_workers + sm_idx - 2 * kMetadataExpertCTAs
                : sm_idx - 2 * active_workers - kComputeBusySenderCTAs;
        const bool busy = logical_idx >= fast_ctas;
        scheduler.block_idx = logical_idx + (busy ? kComputeWarmupRounds * fast_ctas : 0);
        scheduler.initial_stride = fast_ctas;
        scheduler.initial_blocks = busy ? 0 : kComputeWarmupRounds;
    }

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    uint32_t weight_stage_idx = 0, weight_phase = 0;
    DG_STATIC_ASSERT(kActivationCreditLimit == 0 ||
                     (kActivationCreditLimit >= 2 && kActivationCreditLimit <= kNumStages),
                     "Activation credit limit must fit the physical ring");
    DG_STATIC_ASSERT(kFC2CreditOnly <= 1 && (!kFC2CreditOnly || kActivationCreditLimit),
                     "FC2 credit scope requires an activation credit limit");
    auto limit_activation_lookahead = [&](const sched::BlockPhase& block_phase) {
        if constexpr (kActivationCreditLimit != 0 && kActivationCreditLimit < kNumStages) {
            if (!kFC2CreditOnly || block_phase == sched::BlockPhase::Linear2) {
                // Wait for stage t-limit, retaining the original physical-slot wait.
                const bool previous_cycle = stage_idx < kActivationCreditLimit;
                const uint32_t credit_idx = previous_cycle
                    ? stage_idx + kNumStages - kActivationCreditLimit
                    : stage_idx - kActivationCreditLimit;
                empty_barriers[credit_idx]->wait(phase ^ static_cast<uint32_t>(previous_cycle));
            }
        }
    };
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
        if constexpr (kSplitPipeline) {
            weight_stage_idx = weight_stage_idx == kNumWeightStages - 1 ? 0 : weight_stage_idx + 1;
            weight_phase ^= weight_stage_idx == 0;
        } else {
            weight_stage_idx = stage_idx;
            weight_phase = phase;
        }
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags

    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;
    constexpr uint32_t kCleanupPublishCounterIndex = 2;

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        if constexpr (!kDedicatedDispatch)
            cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Only the legacy NVLink barriers share a signal phase.
        if constexpr (not kRankGatedCombine)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        uint32_t expert_pool_block_offset = 0;
        // SF is packed once by the tile's worker. Keep regular activation
        // TMA independent; the coordinator continues to issue SF TMA.
        DG_STATIC_ASSERT(LOAD_BLOCK_M * 2 == BLOCK_M && LOAD_BLOCK_M % 8 == 0, "Eight-row TMA boxes must fit the owned tile");
        MEGAMOE_IKET_FIRST(first_gather);
        MEGAMOE_IKET_FIRST(first_remainder);
        MEGAMOE_IKET_FIRST(first_tail);
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t&) {
            const uint32_t pool_block = scheduler.get_current_pool_block_offset() + m_block_idx;
            const bool issues_activation = warp_idx < kActivationIssuers && !kUnifiedTMA;
            const uint32_t rows = scheduler.template get_valid_m<true>() / 2;
            uint32_t materialized_row = 0;
            uint32_t acquired_windows = 0;
            if (block_phase == sched::BlockPhase::Linear1 && issues_activation) {
                // Both box shapes cover the aligned MMA extent.
                // Acquire the first window; later windows are acquired at their K boundaries.
                MEGAMOE_IKET_ONCE(MegaIndicesReady, first_gather);
                const uint32_t peer_offset = is_leader_cta ? 0 : scheduler.template get_valid_m<true>() / 2;
                materialized_row = static_cast<uint32_t>(workspace.get_source_l1_row(sym_buffer.rank_idx, num_tokens) +
                                                         pool_block * BLOCK_M + peer_offset);
                if constexpr (kRelaxedPoll & 1)
                    wait_ready_mask_relaxed<uint32_t>(workspace.get_l1_arrival_count_ptr(pool_block), kFirstAReadyMask);
                else
                    while ((ptx::ld_acq(workspace.get_l1_arrival_count_ptr(pool_block)) & kFirstAReadyMask) != kFirstAReadyMask);
                __syncwarp();
                asm volatile("fence.proxy.async.global;" ::: "memory");
                MEGAMOE_IKET_ONCE(MegaPayloadReady, first_gather);
            }
            for (uint32_t k = 0; k < num_k_blocks; advance_pipeline(k)) {
                if (!issues_activation) continue;
                // A physical slot keeps its owner through every phase and
                // FC1/FC2 wave. Its producer cannot be lapped on slot reuse.
                if constexpr (kActivationIssuers > 1)
                    if (stage_idx % kActivationIssuers != warp_idx) continue;
                if (block_phase != sched::BlockPhase::Linear1) {
                    // Coordinator owns FC2 loads. Arrival credit prevents
                    // phase reuse before this issuer advances to the next wave.
                    if constexpr (kNumExpertsPerWave != kNumExpertsPerRank) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        limit_activation_lookahead(block_phase);
                        if (cute::elect_one_sync()) full_barriers[stage_idx]->arrive(0u);
                        __syncwarp();
                    }
                    continue;
                }
                DG_STATIC_ASSERT(BLOCK_K <= kFirstReadinessWindowBytes && kFirstReadinessWindowBytes % BLOCK_K == 0,
                                 "Readiness window must contain complete TMA stages");
                if (kActivationIssuers == 1 ? k == kFirstReadinessWindowBytes / BLOCK_K :
                    k >= kFirstReadinessWindowBytes / BLOCK_K && !(acquired_windows & kMiddleAReadyMask)) {
                    MEGAMOE_IKET_ONCE(MegaWindowWaitStart, first_remainder);
                    if constexpr (kRelaxedPoll & 1)
                        wait_ready_mask_relaxed<uint32_t>(workspace.get_l1_arrival_count_ptr(pool_block), kMiddleAReadyMask);
                    else
                        while ((ptx::ld_acq(workspace.get_l1_arrival_count_ptr(pool_block)) & kMiddleAReadyMask) != kMiddleAReadyMask);
                    __syncwarp();
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    MEGAMOE_IKET_ONCE(MegaWindowReady, first_remainder);
                    MEGAMOE_IKET_DONE(first_remainder);
                    if constexpr (kActivationIssuers > 1) acquired_windows |= kMiddleAReadyMask;
                }
                if constexpr (kHidden > kMiddlePayloadWindowEnd) {
                    DG_STATIC_ASSERT(kMiddlePayloadWindowEnd % BLOCK_K == 0,
                                     "Tail readiness must align with complete TMA stages");
                    if (kActivationIssuers == 1 ? k == kMiddlePayloadWindowEnd / BLOCK_K :
                        k >= kMiddlePayloadWindowEnd / BLOCK_K && !(acquired_windows & kTailAReadyMask)) {
                        MEGAMOE_IKET_ONCE(MegaWindowWaitStart, first_tail);
                        if constexpr (kRelaxedPoll & 1)
                            wait_ready_mask_relaxed<uint32_t>(workspace.get_l1_arrival_count_ptr(pool_block), kTailAReadyMask);
                        else
                            while ((ptx::ld_acq(workspace.get_l1_arrival_count_ptr(pool_block)) & kTailAReadyMask) != kTailAReadyMask);
                        __syncwarp();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        MEGAMOE_IKET_ONCE(MegaWindowReady, first_tail);
                        MEGAMOE_IKET_DONE(first_tail);
                        if constexpr (kActivationIssuers > 1) acquired_windows |= kTailAReadyMask;
                    }
                }
                empty_barriers[stage_idx]->wait(phase ^ 1);
                limit_activation_lookahead(block_phase);
                if (cute::elect_one_sync()) {
                    if constexpr (kFullActivationTMA) {
                        // One issue; the extent descriptor omits rows beyond
                        // the aligned MMA extent while retaining physical staging.
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            l1_activation_map(rows * 2), full_barriers[stage_idx], smem_a[stage_idx],
                            k * BLOCK_K, materialized_row, 2);
                    } else {
                        #pragma unroll
                        for (uint32_t row = 0; row < LOAD_BLOCK_M; row += 8) {
                            if (row < rows)
                                tma::copy<BLOCK_K, 8, kSwizzleAMode, a_dtype_t>(
                                    l1_activation_map(rows * 2), full_barriers[stage_idx], smem_a[stage_idx] + row * BLOCK_K,
                                    k * BLOCK_K, materialized_row + row, 2);
                        }
                    }
                    if constexpr (kNumExpertsPerWave != kNumExpertsPerRank)
                        full_barriers[stage_idx]->arrive(0u);
                }
                __syncwarp();
                MEGAMOE_IKET_ONCE(MegaGatherIssued, first_gather);
                MEGAMOE_IKET_DONE(first_gather);
            }
        });

        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        // Keep following cleanup accesses behind the retirement handoff in
        // the compiler as well as in the named barrier's memory ordering.
        asm volatile("" ::: "memory");

        auto* cleanup_counter = workspace.template get_grid_sync_count_ptr<kCleanupPublishCounterIndex>();
        if constexpr (kLastCTACleanup) {
            // Every local epilogue has stopped reading routing data before its
            // ticket. Only the last CTA clears it; combine need not join a grid.
            // Dispatch's completed mbarrier is outside the combine scratch region.
            auto* is_last = reinterpret_cast<uint32_t*>(dispatch_barriers[0]);
            if (thread_idx == 0) {
                asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                    :: "r"(cute::cast_smem_ptr_to_uint(is_last)) : "memory");
                *is_last = ptx::atomic_add_acqrel_gpu(cleanup_counter, 1) == kNumSMs - 1;
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (not *is_last)
                return;
        }

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (kLastCTACleanup or sm_idx != 0) {
            // The last CTA assigns experts to warps; legacy cleanup uses CTAs.
            for (uint32_t i = kLastCTACleanup ? warp_idx : sm_idx - 1;
                 i < kNumExpertsPerRank; i += kLastCTACleanup ? kNumDispatchWarps : kNumSMs - 1) {
                const uint32_t expert = i;
                // Read expert token count before clearing
                uint32_t num_recv_tokens;
                num_recv_tokens = scheduler.get_num_tokens(i);
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                if constexpr (not kLastCTACleanup)
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (kLastCTACleanup or warp_idx == 0) {
                    if (cute::elect_one_sync())
                        *workspace.get_expert_recv_count_sum_ptr(expert) = 0;
                }
                if (kLastCTACleanup or warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + expert, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                __syncwarp();

                // Clean L1 and L2 arrival stuffs
                for (uint32_t j = kLastCTACleanup ? lane_idx : thread_idx;
                     j < num_recv_m_blocks; j += kLastCTACleanup ? 32 : kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }

        // Only blocks that can own cleanup work join the local publication
        // ticket. Block 0 waits for that exact set before retaining the same
        // cross-rank cleanup barrier.
        constexpr uint32_t kNumCleanupBlocks =
            kNumExpertsPerRank + 1 < kNumSMs ? kNumExpertsPerRank + 1 : kNumSMs;
        if (not kLastCTACleanup and sm_idx < kNumCleanupBlocks) {
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            if (thread_idx == 0)
                ptx::atomic_add_acqrel_gpu(cleanup_counter, 1);
        }
        if (kLastCTACleanup or sm_idx == 0) {
            constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;
            const auto start_clock = clock64();
            if (not kLastCTACleanup and thread_idx < kNumRanks) {
                while (ptx::ld_acq(cleanup_counter) != kNumCleanupBlocks) {
                    if (thread_idx == 0 and
                        clock64() - start_clock >= kNumTimeoutCycles) {
                        printf("MegaMoE cleanup ticket timeout (300s): rank=%d ready=%u target=%u\n",
                               sym_buffer.rank_idx, ptx::ld_acq(cleanup_counter),
                               kNumCleanupBlocks);
                        DG_DEVICE_ASSERT(false and
                            "Cleanup ticket timeout");
                    }
                }
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            {
                for (uint32_t i = thread_idx; i < kNumRanks + layout::Workspace::NumGenphaseFlags; i += kNumDispatchThreads)
                    *workspace.get_genphase_flag_ptr(i) = 0;
            }
            {
                if (thread_idx == 0)
                    *workspace.get_route_ready_count_ptr() = 0;
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            }
            if constexpr (kRankGatedCombine) {
                // All-rank routes in g+1 imply every rank finished g. Alternating
                // metadata therefore permits g+2 publication without this join.
                if (thread_idx == 0) {
                    const uint32_t status = *workspace.get_nvl_barrier_counter_ptr() & 3u;
                    // Retire the unused signal phase for later non-pipelined invocations.
                    *workspace.get_nvl_barrier_signal_ptr(status & 1u) = (status >> 1) ? 0 : kNumRanks;
                    if constexpr (kResetRows != 0) {
                        // Publish all metadata retirement before output release
                        // can allow a later invocation to prepare this buffer.
                        *cleanup_counter = 0;
                        asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
                            "l"(workspace.get_nvl_barrier_counter_ptr()),
                            "r"(reset_epoch + 1) : "memory");
                    } else {
                        ptx::red_add(workspace.get_nvl_barrier_counter_ptr(), 1);
                    }
                }
            } else {
                comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                     kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                    workspace, sym_buffer, sm_idx, thread_idx,
                    [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                    /* Local cleanup ticket replaces the grid sync */ false,
                    /* At the end of kernel does not need to sync */ false
                );
            }
            if (thread_idx == 0 && kResetRows == 0)
                *cleanup_counter = 0;
        }

    } else if (warp_idx == kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();
        DG_STATIC_ASSERT(kUnifiedTMA <= 1 && (!kUnifiedTMA || kFullActivationTMA),
                         "Unified TMA requires full activation boxes");

        // GEMM TMA load warp for tokens with SFA
        MEGAMOE_IKET_FIRST(first_tile);
        MEGAMOE_IKET_FIRST(first_gather);
        MEGAMOE_IKET_FIRST(first_remainder);
        MEGAMOE_IKET_FIRST(first_tail);
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : l1_activation_map(scheduler.template get_valid_m<true>());
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;

            // Compute pool block offset for this expert
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait the entire token arrival for linear 1
            if (block_phase == sched::BlockPhase::Linear1) {
                if constexpr (kUnifiedTMA)
                    MEGAMOE_IKET_ONCE(MegaIndicesReady, first_tile);
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                {
                    // SF/index has its own bit; A publication is independent.
                    constexpr uint32_t ready = kSfReadyMask | (kUnifiedTMA ? kFirstAReadyMask : 0u);
                    if constexpr (kRelaxedPoll & 1)
                        wait_ready_mask_relaxed<uint32_t>(ptr, ready);
                    else
                        while ((ptx::ld_acq(ptr) & ready) != ready);
                }
                {
                    __syncwarp();
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    if constexpr (kUnifiedTMA)
                        MEGAMOE_IKET_ONCE(MegaPayloadReady, first_tile);
                    // Packed-SF-ready observation, not a per-stage SMEM store.
                    MEGAMOE_IKET_ONCE(MegaGatherSfReady, first_tile);
                }
                MEGAMOE_IKET_DONE(first_tile);

            } else if constexpr (!kFC2StageReady) {
                // The L1 output's block N is halved into `BLOCK_K / 2`, so we have to wait 2x L1 blocks' arrival
                // NOTES: Originally we wait blocks on-demand to overlap L1 calculation
                // with L2, but this optimization is negative when `num_experts_per_wave`
                // guarantees L1's completion when L2 starts. So we remove it.
                // In the future, if `num_experts_per_wave` is not large enough
                // due to small `num_experts_per_rank`, we may need to add it back or add a switch
                DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                // NOTES: Equivalent to `(1ull << (2 * num_k_blocks)) - 1`, but split into two shifts
                // to avoid undefined behavior when `num_k_blocks == 32`
                const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;
                if constexpr (kRelaxedPoll & 2)
                    wait_ready_mask_relaxed<uint64_t, true>(ptr, expected);
                else
                    while (ptx::ld_acq_gpu(ptr) != expected);

            }

            // Each released bit covers one FC1 output block and its scales.
            // Cache acquired arrivals: bits only accumulate until every CTA retires.
            uint64_t l2_arrivals = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (kFC2StageReady) {
                    if (block_phase == sched::BlockPhase::Linear2) {
                        DG_STATIC_ASSERT(BLOCK_K == BLOCK_N && L2_SHAPE_K / BLOCK_K <= 32,
                                         "Each FC2 slice needs two FC1 arrival bits");
                        const uint64_t needed = 3ull << (2 * k_block_idx);
                        if constexpr (kRelaxedPoll & 2) {
                            if ((l2_arrivals & needed) != needed)
                                l2_arrivals = wait_ready_mask_relaxed<uint64_t>(
                                    workspace.get_l2_arrival_mask_ptr(pool_block_idx), needed);
                        } else {
                        while ((l2_arrivals & needed) != needed)
                            l2_arrivals = ptx::ld_acq_gpu(workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                        }
                    }
                }
                if constexpr (kUnifiedTMA) {
                    if (block_phase == sched::BlockPhase::Linear1) {
                        // The real producer owns every stage in both FC1 and FC2;
                        // no separate activation observer can be lapped on wrap.
                        const uint32_t bit = k_block_idx == kFirstReadinessWindowBytes / BLOCK_K
                            ? kMiddleAReadyMask
                            : k_block_idx == kMiddlePayloadWindowEnd / BLOCK_K && kHidden > kMiddlePayloadWindowEnd
                                ? kTailAReadyMask : 0u;
                        if (bit) {
                            MEGAMOE_IKET_ONCE(MegaWindowWaitStart,
                                bit == kMiddleAReadyMask ? first_remainder : first_tail);
                            if constexpr (kRelaxedPoll & 1)
                                wait_ready_mask_relaxed<uint32_t>(workspace.get_l1_arrival_count_ptr(pool_block_idx), bit);
                            else
                                while ((ptx::ld_acq(workspace.get_l1_arrival_count_ptr(pool_block_idx)) & bit) != bit);
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            MEGAMOE_IKET_ONCE(MegaWindowReady,
                                bit == kMiddleAReadyMask ? first_remainder : first_tail);
                            if (bit == kMiddleAReadyMask) {
                                MEGAMOE_IKET_DONE(first_remainder);
                            } else {
                                MEGAMOE_IKET_DONE(first_tail);
                            }
                        }
                    }
                }
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);
                limit_activation_lookahead(block_phase);

                // Compute token offset from pool block index
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx;
                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;
                if constexpr (kUnifiedTMA) {
                    if (block_phase == sched::BlockPhase::Linear1)
                        m_idx += static_cast<uint32_t>(workspace.get_source_l1_row(sym_buffer.rank_idx, num_tokens));
                }

                {
                    if (block_phase == sched::BlockPhase::Linear1) {
                        sfa_m_idx = 0;
                        sfa_k_idx += pool_block_idx * (kHidden / 128);
                    }
                }
                if (cute::elect_one_sync()) {
                    if (block_phase != sched::BlockPhase::Linear1 || kUnifiedTMA)
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx],
                        smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        // Both CTA boxes and the expected bytes use the same extent.
                        const uint32_t activation_bytes = block_phase == sched::BlockPhase::Linear1 && (!kFullActivationTMA || kExtentActivationTMA) ?
                            scheduler.template get_valid_m<true>() * BLOCK_K * sizeof(a_dtype_t) : SMEM_A_SIZE_PER_STAGE * 2;
                        full_barriers[stage_idx]->arrive_and_expect_tx(activation_bytes + SF_BLOCK_M * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
                if constexpr (kUnifiedTMA) {
                    if (block_phase == sched::BlockPhase::Linear1) {
                        MEGAMOE_IKET_ONCE(MegaGatherIssued, first_gather);
                        MEGAMOE_IKET_DONE(first_gather);
                    }
                }
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        constexpr auto kWeightCacheHint = static_cast<uint64_t>(kWeightCache
            ? cute::TMA::CacheHintSm100::EVICT_LAST : cute::TMA::CacheHintSm100::EVICT_NORMAL);
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                weight_empty_barriers[weight_stage_idx]->wait(weight_phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, false, kWeightCacheHint>(
                        tensor_map_b_ptr, weight_full_barriers[weight_stage_idx], smem_b[weight_stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0, uint32_t, false, kWeightCacheHint>(
                        tensor_map_sfb_ptr, weight_full_barriers[weight_stage_idx],
                        smem_sfb[weight_stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        constexpr uint32_t kBArrivalBytes =
                            kUseFP8Weights ? SMEM_B_SIZE_PER_STAGE * 2 : SMEM_B_SIZE_PER_STAGE;
                        weight_full_barriers[weight_stage_idx]->arrive_and_expect_tx(
                            kBArrivalBytes + BLOCK_N * sizeof(uint32_t) * 2);
                    } else {
                        weight_full_barriers[weight_stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        const auto release_dependents = [&]() {
            if constexpr (kResetRows != 0) {
                if (lane_idx == 0) {
                    asm volatile("fence.release.gpu;" ::: "memory");
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
            }
        };
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);
            // Form SF addresses in descriptor units. Only immutable addresses
            // are prepared here; SF loads remain after operand completion.
            const uint32_t sfa_base_units = cute::cast_smem_ptr_to_uint(smem_sfa[0]) >> 4;
            const uint32_t sfb_base_units = cute::cast_smem_ptr_to_uint(smem_sfb[0]) >> 4;
            DG_STATIC_ASSERT(SMEM_SFA_SIZE_PER_STAGE % 16 == 0 &&
                             SMEM_SFB_SIZE_PER_STAGE % 16 == 0 &&
                             (kNumUTCCPAlignedElems * sizeof(uint32_t)) % 16 == 0,
                             "SF descriptor offsets require whole 16-byte units");
            DG_STATIC_ASSERT(static_cast<uint64_t>(SMEM_BEFORE_BARRIER_SIZE) +
                             static_cast<uint64_t>(kNumStages) * SMEM_SFA_SIZE_PER_STAGE +
                             static_cast<uint64_t>(kNumWeightStages) * SMEM_SFB_SIZE_PER_STAGE < (1ull << 32),
                             "SF address arithmetic must not wrap before scaling");

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            // Keep immutable stage-zero descriptor words: the working descriptors
            // below advance within each K block as MMAs are issued.
            const uint32_t a_desc_stage0_lo = a_desc.lo;
            const uint32_t b_desc_stage0_lo = b_desc.lo;
            DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 16 == 0 && SMEM_B_SIZE_PER_STAGE % 16 == 0,
                             "Descriptor stage strides must be in whole 16-byte units");
            DG_STATIC_ASSERT(static_cast<uint64_t>(kNumStages) * SMEM_A_SIZE_PER_STAGE <= 0xffffffffull &&
                             static_cast<uint64_t>(kNumWeightStages) * SMEM_B_SIZE_PER_STAGE <= 0xffffffffull,
                             "Byte-stride multiplication must not overflow before scaling");

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);

                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if constexpr (kSplitPipeline)
                        umma_arrive(reinterpret_cast<uint64_t*>(weight_empty_barriers[weight_stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Descriptor words are immutable addresses; select the current
                    // stage while operand completion can still be in flight.
                    const uint32_t a_desc_base_lo = a_desc_stage0_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                    const uint32_t b_desc_base_lo = b_desc_stage0_lo + weight_stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                    // Operands and scales share the stage completion barrier.
                    full_barriers[stage_idx]->wait(phase);
                    if constexpr (kSplitPipeline)
                        weight_full_barriers[weight_stage_idx]->wait(weight_phase);

                    ptx::tcgen05_after_thread_sync();

                    if (cute::elect_one_sync()) {
                        // UTCCP copy SFA and SFB to TMEM
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            sf_desc.start_address_ = static_cast<uint16_t>(
                                sfa_base_units + stage_idx * (SMEM_SFA_SIZE_PER_STAGE / 16) +
                                i * (kNumUTCCPAlignedElems * sizeof(uint32_t) / 16));
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            sf_desc.start_address_ = static_cast<uint16_t>(
                                sfb_base_units + weight_stage_idx * (SMEM_SFB_SIZE_PER_STAGE / 16) +
                                i * (kNumUTCCPAlignedElems * sizeof(uint32_t) / 16));
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                        // Issue UMMA
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                            MEGAMOE_IKET_ONCE(MegaMmaIssue, current_iter_idx == 1 && k_block_idx == 0 && k == 0);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // All MMAs have been issued; output readiness still waits for the
            // complete producer, including remote writeback and metadata cleanup.
            release_dependents();
            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        } else {
            // Follower CTAs do not issue MMA but must participate in PDL.
            release_dependents();
        }

    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        // Ordered invocations and all-rank route publication make slot(g-2)
        // fully consumed/cleared before any rank writes it again in g.
        // Capture parity before cleanup can advance the counter.
        const uint32_t combine_slot = kInbandCombine
            ? *reinterpret_cast<volatile uint32_t*>(workspace.get_nvl_barrier_counter_ptr()) & 1u
            : 0;

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // NOTES:
        //  - each warpgroup owns BM / 2 rows, contiguous or store-block interleaved
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        constexpr uint32_t kReturnVectors = kReturnWide == 1 ? 2 : 1;
        constexpr uint32_t kReturnLanesPerRow = 16 / kReturnVectors;
        constexpr uint32_t kReturnRowsPerWarp = 2 * kReturnVectors;
        constexpr uint32_t kReturnRowsPerRound = 8 * kReturnVectors;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Round-robin store blocks balance a valid row prefix without changing
        // the MMA tile or the physical warpgroup's private shared staging.
        const auto store_row = [&](const uint32_t s) {
            if constexpr (kBalanceEpilogue)
                return (s * kNumEpilogueWarpgroups + epilogue_wg_idx) * STORE_BLOCK_M;
            else
                return epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M;
        };

        // Match the dispatch-side ordering of legacy NVLink barriers.
        if constexpr (not kRankGatedCombine)
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks

        uint32_t current_iter_idx = 0;
        bool previous_return = false;
#if defined(IKET_BUILD_ENABLED) && defined(MEGAMOE_IKET_CRITICAL)
        // Round-robin scheduling carries its index across experts and both GEMMs.
        // Count this CTA's work once; mark only its final accumulator readiness.
        uint32_t iket_num_blocks = 0;
        bool iket_combine_started = false;
        if (epilogue_warp_idx == 0) {
            for (uint32_t e = 0; e < kNumExpertsPerRank; ++e)
                iket_num_blocks += math::ceil_div(smem_expert_count[e], BLOCK_M) *
                    ((L1_SHAPE_N + L2_SHAPE_N) / BLOCK_N);
            iket_num_blocks = scheduler.get_num_blocks(iket_num_blocks);
        }
#endif
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // Producer metadata is already ordered by the route-ready acquire
            // and shared-count handoff. Prefetch it while MMA is independent.
            if constexpr (kReturnWarpgroupSync) {
                // FC2 regions are disjoint by warpgroup. FC1 reuses their storage
                // with a different layout, so join both groups at that transition.
                if (previous_return && block_phase == sched::BlockPhase::Linear1)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                previous_return = block_phase == sched::BlockPhase::Linear2;
            }
            // Compute offsets
            // NOTES: use shuffle here to let NVCC know warp divergence won't happen
            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            float producer_weights[math::constexpr_ceil_div(WG_BLOCK_M, 32u)] = {};
            float4* producer_destinations[WG_BLOCK_M / STORE_BLOCK_M][STORE_BLOCK_M / kReturnRowsPerRound];
            {
                // Bank g stays live until every rank completes g; starting g+2
                // requires all-rank route-ready g+1 and therefore that retirement.
                if (block_phase == sched::BlockPhase::Linear1) {
                    #pragma unroll
                    for (uint32_t chunk = 0; chunk < math::constexpr_ceil_div(WG_BLOCK_M, 32u); ++chunk) {
                        const uint32_t row_in_wg = chunk * 32 + lane_idx;
                        const uint32_t row = store_row(row_in_wg / STORE_BLOCK_M) + row_in_wg % STORE_BLOCK_M;
                        if (row_in_wg < WG_BLOCK_M && row < valid_m) {
                            const auto* source = workspace.get_src_token_topk_idx_ptr(
                                local_expert_idx, kNumRanks == 1 ? 0 : workspace.metadata_slot,
                                m_block_idx * BLOCK_M + row);
                            uint32_t descriptor;
                            asm volatile("ld.global.u32 %0, [%1];"
                                         : "=r"(descriptor) : "l"(source) : "memory");
                            const uint32_t rank = descriptor / (kNumMaxTokensPerRank * kNumTopk);
                            const uint32_t route = descriptor % (kNumMaxTokensPerRank * kNumTopk);
                            asm volatile("ld.global.f32 %0, [%1];"
                                         : "=f"(producer_weights[chunk])
                                         : "l"(workspace.get_score_inbox_ptr(rank) + route) : "memory");
                        }
                    }
                } else {
                    if constexpr (kCacheDestinations) {
                        // SF workers write the address cache before their tile release.
                        // Producer route readiness alone does not cover those writes.
                        if constexpr (kRelaxedPoll & 1)
                            wait_ready_mask_relaxed<uint32_t>(workspace.get_l1_arrival_count_ptr(pool_block_idx), kSfReadyMask);
                        else
                            while ((ptx::ld_acq(workspace.get_l1_arrival_count_ptr(pool_block_idx)) &
                                    kSfReadyMask) != kSfReadyMask);
                        __syncwarp();
                    }
                    #pragma unroll
                    for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++s) {
                        #pragma unroll
                        for (uint32_t j = 0; j < STORE_BLOCK_M / kReturnRowsPerRound; ++j) {
                            const uint32_t row = store_row(s) +
                                j * kReturnRowsPerRound + warp_idx_in_wg * kReturnRowsPerWarp + lane_idx / kReturnLanesPerRow;
                            if (row < valid_m) {
                                if constexpr (kCacheDestinations) {
                                    const uint64_t destination = workspace.get_cached_destinations_ptr()[m_idx + row];
                                    producer_destinations[s][j] = math::advance_ptr<float4>(
                                        reinterpret_cast<void*>(destination),
                                        n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                        (lane_idx % kReturnLanesPerRow) * static_cast<uint32_t>(sizeof(float4) * kReturnVectors));
                                } else {
                                    const auto* source = workspace.get_src_token_topk_idx_ptr(
                                        local_expert_idx, kNumRanks == 1 ? 0 : workspace.metadata_slot,
                                        m_block_idx * BLOCK_M + row);
                                    uint32_t descriptor;
                                    asm volatile("ld.global.u32 %0, [%1];"
                                                 : "=r"(descriptor) : "l"(source) : "memory");
                                    // The route generation is already acquired. Resolve
                                    // the peer base and output address while MMA runs.
                                    const uint32_t rank = descriptor / (kNumMaxTokensPerRank * kNumTopk);
                                    const uint32_t route = descriptor % (kNumMaxTokensPerRank * kNumTopk);
                                    const auto token = combine_token_buffer
                                        .get_rank_buffer(combine_slot * kNumTopk + route % kNumTopk)
                                        .get_data_buffer(route / kNumTopk);
                                    producer_destinations[s][j] = sym_buffer.map(math::advance_ptr<float4>(
                                        token.get_base_ptr(), n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                                        (lane_idx % kReturnLanesPerRow) * static_cast<uint32_t>(sizeof(float4) * kReturnVectors)), rank);
                                }
                            }
                        }
                    }
                }
            }

            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);

            // The accumulator is complete here, not when the MMA issue warp
            // exits. Keep every L2 tile; the analyzer takes the final event.
            if (block_phase == sched::BlockPhase::Linear2) {
#if defined(IKET_BUILD_ENABLED) && defined(MEGAMOE_IKET_CRITICAL)
                if (epilogue_warp_idx == 0 and current_iter_idx == iket_num_blocks)
#endif
                    MEGAMOE_IKET_MARK(MegaComputeEnd);
            }

            ptx::tcgen05_after_thread_sync();

            if (block_phase == sched::BlockPhase::Linear1) {
                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                //   (values[0], values[2]), (values[1], values[3]),
                //   (values[4], values[6]), (values[5], values[7])

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (store_row(s) >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    float2 swiglu_values[kNumAtomsPerStore * 2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // Load weights from global into register cache per 32 tokens
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");

                        // Load weights from register cache
                        const float2 weights = {
                            ptx::exchange(producer_weights[(j * ATOM_M) / 32], (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(producer_weights[(j * ATOM_M) / 32], (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + store_row(s) + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Apply SwiGLU: silu(gate) * up
                        // Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // Clamp
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SwiGLU
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        // Amax reduction
                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());
                        if (lane_idx < 4)
                            smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `smem_amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax
                        const float2 wp_amax =
                            smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        // Cast
                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM
                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                     + i * ATOM_M * L1_OUT_BLOCK_N
                                                                     + row * L1_OUT_BLOCK_N
                                                                     + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        // Store SF to `l2_sf_buffer` as UE8M0 (MN-major layout)
                        // Only one warp per pair writes (both hold the same SF after cross-warp reduce)
                        // Each lane < 4 holds SF for 2 rows (sf.x and sf.y)
                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // Each atom base is 8-row aligned and remains inside
                            // this tile under either warpgroup mapping. The lane's
                            // low three row bits transform independently of it.
                            const uint32_t token_base_idx = store_row(s) + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Issue TMA store after all atoms in this store block
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                            out_n_idx,
                            m_idx + store_row(s));
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // Notify L2
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
            } else {
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / kReturnRowsPerRound;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                // This phase includes BF16 packing, remote writes, readiness,
                // and source-side reduction; it overlaps later compute tiles.
#if defined(IKET_BUILD_ENABLED) && defined(MEGAMOE_IKET_CRITICAL)
                if (epilogue_warp_idx == 0 and not iket_combine_started) {
                    MEGAMOE_IKET_MARK(MegaCombineStart);
                    iket_combine_started = true;
                }
#else
                MEGAMOE_IKET_MARK(MegaCombineStart);
#endif
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (store_row(s) >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    // Prefetch remote destinations before BF16 shared staging.
                    float4* destinations[kNumRowsPerWarp];
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++j) {
                        const uint32_t row = store_row(s) +
                            j * kReturnRowsPerRound + warp_idx_in_wg * kReturnRowsPerWarp + lane_idx / kReturnLanesPerRow;
                        if (row < valid_m) {
                            destinations[j] = producer_destinations[s][j];
                        }
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + store_row(s) + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Store into shared memory
                        // NOTES: only use first 16 lanes for address
                        // NOTES: 2 warps share a BF16 swizzle atom
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Write into remote buffers
                    // Read the same swizzled staging with 16 or 8 lanes per row.
                    const uint32_t row_in_atom = (warp_idx_in_wg * kReturnRowsPerWarp + lane_idx / kReturnLanesPerRow) % ATOM_M;
                    const uint32_t bank_group_idx = (lane_idx % (8 / kReturnVectors)) * kReturnVectors;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * kReturnRowsPerRound + warp_idx_in_wg * kReturnRowsPerWarp + lane_idx / kReturnLanesPerRow;
                        const uint32_t m_idx_in_block = store_row(s) + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        // Read from shared memory
                        const auto smem_base = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % kReturnLanesPerRow / (8 / kReturnVectors)) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode;
                        const auto smem_ptr = smem_base + (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        if constexpr (kReturnWide == 1) {
                            float4 packed[2];
                            #pragma unroll
                            for (uint32_t v = 0; v < 2; ++v) {
                                packed[v] = ptx::ld_shared(reinterpret_cast<float4*>(
                                    smem_base + ((bank_group_idx + v) ^ row_in_atom) * kNumBankGroupBytes));
                                auto* words = reinterpret_cast<uint32_t*>(&packed[v]);
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i)
                                    if (words[i] == 0x80000000u) words[i] = 0;
                            }
                            const auto* words = reinterpret_cast<const uint32_t*>(packed);
                            asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::
                                "l"(destinations[j]), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]),
                                "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]) : "memory");
                        } else {
                            auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));
                            if constexpr (kInbandCombine) {
                                auto* words = reinterpret_cast<uint32_t*>(&packed);
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i)
                                    if (words[i] == 0x80000000u) words[i] = 0;
                            }

                            *destinations[j] = packed;
                        }
                    }
                }

                // Protect each FC2 staging region before its next tile reuses it.
                if constexpr (kReturnWarpgroupSync)
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                else
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

            }
        });

        // Deallocate tensor memory
        // NOTES: must be called by the same logical warp ID on both CTAs
        if constexpr (kReturnWarpgroupSync)
            if (previous_return)
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // The local join protects cleanup metadata. Epoch/TMA consumers also
        // need producer publication; in-band consumers poll the payload itself.
        uint64_t combine_ready_epoch = 0;
        if constexpr (kRankGatedCombine) {
            if constexpr (not kLastCTACleanup)
                comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                    workspace, sm_idx, epilogue_thread_idx,
                    [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
                );
            if constexpr (not kInbandCombine) {
                combine_ready_epoch =
                    static_cast<uint64_t>(*workspace.get_nvl_barrier_counter_ptr()) + 1;
                if (sm_idx == 0 and epilogue_thread_idx < kNumRanks)
                    ptx::st_release_sys(
                        sym_buffer.map(
                            workspace.get_combine_ready_epoch_ptr(sym_buffer.rank_idx),
                            epilogue_thread_idx),
                        combine_ready_epoch);
            }
        } else {
            // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
            );
        }

        // Barrier with dispatch warps, so that they can do clean workspace
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results and write back
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);

        // A warp cannot represent EP>32 with one rank bit per lane. In those
        // worlds let each valid top-k lane acquire its own producer instead.
        // Duplicate producer acquires are harmless; every contributing slot
        // is ordered before the warp/CTA proceeds to bulk loads.
        const auto wait_combine_producer = [&](const int expert_rank_idx,
                                               const uint32_t token_idx) {
            if (expert_rank_idx >= 0) {
                constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;
                const auto ready_epoch_ptr =
                    workspace.get_combine_ready_epoch_ptr(expert_rank_idx);
                const auto start_clock = clock64();
                while (ptx::ld_acq_sys(ready_epoch_ptr) != combine_ready_epoch) {
                    if (clock64() - start_clock >= kNumTimeoutCycles) {
                        printf("MegaMoE producer combine timeout (300s): rank=%d token=%d expert_rank=%d ready=%llu target=%llu\n",
                               sym_buffer.rank_idx, token_idx, expert_rank_idx,
                               static_cast<unsigned long long>(ptx::ld_acq_sys(ready_epoch_ptr)),
                               static_cast<unsigned long long>(combine_ready_epoch));
                        DG_DEVICE_ASSERT(false and "Producer combine timeout");
                    }
                }
            }
        };

        if constexpr (kInbandCombine or ((kNumTopk == 8 or kNumTopk == 10) and kHidden == 4096)) {
            // Chunked combine uses one vector per thread. Full-row work reuses
            // those registers for successive vectors and amortizes mask/barrier work.
            // Each inbox/output vector still has one owner and the same sum order.
            DG_STATIC_ASSERT(kNumTopk <= 32, "Combine top-k must fit in a warp");
            constexpr uint32_t kNumUint4 = kNumHiddenBytes / sizeof(uint4);
            constexpr uint32_t kCombineVectorsPerThread = kFullRowCombine >= 2 ? 2 : 1;
            // Modes 2/3/4 assign each row to one/two/four independent warps.
            constexpr uint32_t kCombineThreads = kWarpCombine >= 2 ? 32u << (kWarpCombine - 2) : kNumEpilogueThreads;
            constexpr uint32_t kCombineWorkersPerCTA = kNumEpilogueThreads / kCombineThreads;
            constexpr uint32_t kCombineVectorStride = kFullRowCombine >= 3 ? 1 : kCombineThreads;
            DG_STATIC_ASSERT(kNumEpilogueThreads % kCombineThreads == 0,
                             "Combine row groups must partition epilogue warps");
            DG_STATIC_ASSERT(kFullRowCombine < 2 ||
                             (kInbandCombine && kNumUint4 == 2 * kNumEpilogueThreads && kNumTopk <= 16),
                             "Paired combine requires two complete in-band vectors and a 32-bit pending mask");
            DG_STATIC_ASSERT(
                kInbandCombine or kNumTopk <= kNumCombineBarriers,
                "CTA combine requires one transaction barrier per top-k slot");
            DG_STATIC_ASSERT(
                kInbandCombine or kNumTopk * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE,
                "CTA combine top-k staging exceeds shared memory");
            // Without the grid wait, an idle scheduler warp may still read counts.
            auto* combine_masks = reinterpret_cast<uint32_t*>(
                smem_buffer + (kLastCTACleanup ? SMEM_EXPERT_COUNT_SIZE : 0));
            auto* combine_tma_buffer = reinterpret_cast<uint4*>(smem_buffer);
            constexpr uint32_t kCombineChunkVectors =
                kInbandCombine && !kFullRowCombine ? kNumEpilogueThreads : kNumUint4;
            constexpr uint32_t kCombineChunksPerToken =
                (kNumUint4 + kCombineChunkVectors - 1) / kCombineChunkVectors;
            uint32_t combine_phase = 0;
            // Reverse the CTA-to-work permutation so the final CTAs can begin
            // row-chunk combine after their own compute work retires.
            const uint32_t first_combine_work = kNumSMs * kCombineWorkersPerCTA - 1 -
                (sm_idx * kCombineWorkersPerCTA +
                 (kWarpCombine >= 2 ? epilogue_warp_idx / (kCombineThreads / 32) : 0));
            for (uint64_t combine_work = first_combine_work;
                 combine_work < static_cast<uint64_t>(num_tokens) * kCombineChunksPerToken;
                 combine_work += kNumSMs * kCombineWorkersPerCTA) {
                const uint32_t token_idx = combine_work / kCombineChunksPerToken;
                const uint32_t vector_begin =
                    (combine_work % kCombineChunksPerToken) * kCombineChunkVectors;
                const uint32_t vector_end =
                    min(kNumUint4, vector_begin + kCombineChunkVectors);
                uint32_t warp_mask = 0;
                if (kWarpCombine || epilogue_warp_idx == 0) {
                    const int stored_expert_idx = lane_idx < kNumTopk
                        ? static_cast<int>(__ldg(
                              input_topk_idx_buffer.get_base_ptr<int64_t>() +
                              token_idx * kNumTopk + lane_idx))
                        : -1;
                    const uint32_t total_mask = __ballot_sync(
                        0xffffffff, stored_expert_idx >= 0);
                    if constexpr (kWarpCombine) {
                        // Each warp owns disjoint inbox/output vectors. Its
                        // read-only route mask needs no shared-memory handoff.
                        warp_mask = total_mask;
                    } else if (lane_idx == 0) {
                        combine_masks[0] = total_mask;
                    }

                    if constexpr (kRankGatedCombine and not kInbandCombine) {
                        const int expert_rank_idx = stored_expert_idx >= 0
                            ? stored_expert_idx /
                                  static_cast<int>(kNumExpertsPerRank)
                            : -1;
                        if constexpr (kNumRanks > 32) {
                            wait_combine_producer(expert_rank_idx, token_idx);
                        } else {
                            uint32_t required_rank_mask = 0;
                            #pragma unroll
                            for (uint32_t rank_idx = 0;
                                 rank_idx < kNumRanks;
                                 ++rank_idx) {
                                if (__any_sync(
                                        0xffffffff,
                                        expert_rank_idx ==
                                            static_cast<int>(rank_idx)))
                                    required_rank_mask |= 1u << rank_idx;
                            }
                            if (lane_idx == 0)
                                combine_masks[1] = required_rank_mask;
                        }
                    }
                }
                if constexpr (!kWarpCombine)
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                const uint32_t total_mask = kWarpCombine ? warp_mask : combine_masks[0];

                if constexpr (kRankGatedCombine and not kInbandCombine and kNumRanks <= 32) {
                    const uint32_t required_rank_mask = combine_masks[1];
                    if (epilogue_warp_idx == 0 and lane_idx < kNumRanks and
                        (required_rank_mask & (1u << lane_idx)) != 0) {
                        constexpr int64_t kNumTimeoutCycles =
                            300ll * 2000000000ll;
                        const auto ready_epoch_ptr =
                            workspace.get_combine_ready_epoch_ptr(lane_idx);
                        const auto start_clock = clock64();
                        while (ptx::ld_acq_sys(ready_epoch_ptr) !=
                               combine_ready_epoch) {
                            if (clock64() - start_clock >= kNumTimeoutCycles) {
                                printf("MegaMoE CTA combine timeout (300s): rank=%d token=%d expert_rank=%d ready=%llu target=%llu\n",
                                       sym_buffer.rank_idx, token_idx,
                                       lane_idx,
                                       static_cast<unsigned long long>(
                                           ptx::ld_acq_sys(ready_epoch_ptr)),
                                       static_cast<unsigned long long>(
                                           combine_ready_epoch));
                                DG_DEVICE_ASSERT(false and
                                    "CTA combine timeout");
                            }
                        }
                    }
                    // The producer epochs order every remote combine slot.
                    // Make the warp-0 acquires visible to all CTA threads
                    // before the elected lane issues bulk TMA loads.
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

                if constexpr (not kInbandCombine) {
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t topk_idx = 0;
                             topk_idx < kNumTopk;
                             ++topk_idx) {
                            if ((total_mask & (1u << topk_idx)) != 0) {
                                // Keep large combine offsets in 64-bit integer
                                // arithmetic across the final address addition.
                                const auto sym_base = reinterpret_cast<uint64_t>(
                                    sym_buffer.get_base_ptr());
                                const uint64_t combine_offset =
                                    reinterpret_cast<uint64_t>(
                                        combine_token_buffer.get_base_ptr()) - sym_base;
                                const uint64_t combine_end_offset = combine_offset +
                                    static_cast<uint64_t>(kNumTopk) *
                                        kNumMaxTokensPerRank * kNumHiddenBytes;
                                const void* src_ptr;
                                if (combine_end_offset > 0x7fffffffULL) {
                                    const uint64_t source_offset = combine_offset +
                                        (static_cast<uint64_t>(topk_idx) *
                                             kNumMaxTokensPerRank + token_idx) *
                                            kNumHiddenBytes;
                                    uint64_t source_address;
                                    asm volatile("add.u64 %0, %1, %2;"
                                        : "=l"(source_address)
                                        : "l"(sym_base), "l"(source_offset));
                                    src_ptr = reinterpret_cast<const void*>(source_address);
                                } else {
                                    // Preserve the established small-buffer path.
                                    src_ptr = combine_token_buffer
                                        .get_rank_buffer(topk_idx)
                                        .get_data_buffer(token_idx)
                                        .get_base_ptr();
                                }
                                auto* dst_ptr = combine_tma_buffer +
                                    topk_idx * kNumUint4;
                                ptx::tma_load_1d(
                                    dst_ptr, src_ptr,
                                    combine_barriers[topk_idx], kNumHiddenBytes);
                                ptx::mbarrier_arrive_and_set_tx(
                                    combine_barriers[topk_idx], kNumHiddenBytes);
                            }
                        }
                    }
                    #pragma unroll
                    for (uint32_t topk_idx = 0;
                         topk_idx < kNumTopk;
                         ++topk_idx) {
                        if ((total_mask & (1u << topk_idx)) != 0)
                            combine_barriers[topk_idx]->wait(combine_phase);
                    }
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }

                if (combine_work == first_combine_work)
                    MEGAMOE_IKET_MARK(MegaLocalReduceStart);
                for (uint32_t vector_idx = vector_begin + (kWarpCombine >= 2
                         ? lane_idx + (epilogue_warp_idx % (kCombineThreads / 32)) * 32 : epilogue_thread_idx) *
                         (kFullRowCombine >= 3 ? 2 : 1);
                     vector_idx < vector_end;
                     vector_idx += kCombineThreads * kCombineVectorsPerThread) {
                    uint4 values[kCombineVectorsPerThread][kNumTopk] = {};
                    if constexpr (kInbandCombine) {
                        const auto slot_ptr = [&](uint32_t vector, uint32_t slot) {
                            const uint64_t base =
                                reinterpret_cast<uint64_t>(combine_token_buffer.get_base_ptr());
                            const uint64_t offset =
                                ((static_cast<uint64_t>(combine_slot * kNumTopk + slot) *
                                  kNumMaxTokensPerRank + token_idx) * kNumUint4 + vector_idx +
                                  vector * kCombineVectorStride) * sizeof(uint4);
                            uint64_t address;
                            asm volatile("add.u64 %0, %1, %2;" : "=l"(address) : "l"(base), "l"(offset));
                            return reinterpret_cast<uint4*>(address);
                        };
                        uint32_t pending = 0;
                        if constexpr (kFullRowCombine >= 3) {
                            pending = total_mask;
                        } else {
                            #pragma unroll
                            for (uint32_t vector = 0; vector < kCombineVectorsPerThread; ++vector)
                                pending |= total_mask << (vector * kNumTopk);
                        }
                        const auto start = clock64();
                        while (pending) {
                            if constexpr (kFullRowCombine >= 3) {
                                // Adjacent vectors share a 32-byte transaction. Every
                                // word must arrive before retiring the pair; no atomicity assumed.
                                #pragma unroll
                                for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
                                    if (pending & (1u << slot)) {
                                        auto& a = values[0][slot];
                                        auto& b = values[1][slot];
                                        asm volatile("ld.volatile.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w),
                                              "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                                            : "l"(slot_ptr(0, slot)) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
                                    const auto& a = values[0][slot];
                                    const auto& b = values[1][slot];
                                    if ((pending & (1u << slot)) &&
                                        a.x != 0x80000000u && a.y != 0x80000000u &&
                                        a.z != 0x80000000u && a.w != 0x80000000u &&
                                        b.x != 0x80000000u && b.y != 0x80000000u &&
                                        b.z != 0x80000000u && b.w != 0x80000000u) {
                                        asm volatile("st.global.v8.u32 [%0], {%1, %1, %1, %1, %1, %1, %1, %1};" ::
                                            "l"(slot_ptr(0, slot)), "r"(0x80000000u) : "memory");
                                        pending &= ~(1u << slot);
                                    }
                                }
                            } else {
                                // Issue independent reads first; retain and clear each
                                // completed vector once, polling only missing slots.
                                #pragma unroll
                                for (uint32_t vector = 0; vector < kCombineVectorsPerThread; ++vector) {
                                    #pragma unroll
                                    for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
                                        if (pending & (1u << (vector * kNumTopk + slot))) {
                                            auto& v = values[vector][slot];
                                            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                                : "l"(slot_ptr(vector, slot)) : "memory");
                                        }
                                    }
                                }
                                #pragma unroll
                                for (uint32_t vector = 0; vector < kCombineVectorsPerThread; ++vector) {
                                    #pragma unroll
                                    for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
                                        const auto& v = values[vector][slot];
                                        if ((pending & (1u << (vector * kNumTopk + slot))) and v.x != 0x80000000u and
                                            v.y != 0x80000000u and v.z != 0x80000000u and v.w != 0x80000000u) {
                                            *slot_ptr(vector, slot) = make_uint4(
                                                0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
                                            pending &= ~(1u << (vector * kNumTopk + slot));
                                        }
                                    }
                                }
                            }
                            if (pending and clock64() - start > 10000000000ull)
                                DG_DEVICE_ASSERT(false and "In-band combine arrival timeout");
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t topk_idx = 0;
                             topk_idx < kNumTopk;
                             ++topk_idx) {
                            if ((total_mask & (1u << topk_idx)) != 0) {
                                values[0][topk_idx] = combine_tma_buffer[
                                    topk_idx * kNumUint4 + vector_idx];
                            }
                        }
                    }
                    uint4 paired_output[2];
                    #pragma unroll
                    for (uint32_t vector = 0; vector < kCombineVectorsPerThread; ++vector) {
                        const auto& contributions = values[vector];
                        uint4 casted;
                        auto* casted_bf16 =
                            reinterpret_cast<nv_bfloat162*>(&casted);
                        constexpr uint32_t kNumBF16x2 =
                            sizeof(uint4) / sizeof(nv_bfloat162);
                        #pragma unroll
                        for (uint32_t element = 0;
                             element < kNumBF16x2;
                             ++element) {
                            if constexpr (kNumTopk != 10 or kHidden != 4096) {
                                // Match the original warp combine exactly: start at
                                // +0 and accumulate valid slots in ascending order.
                                float2 reduced = {};
                                #pragma unroll
                                for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
                                    if ((total_mask & (1u << slot)) != 0) {
                                        ptx::accumulate(
                                            reduced,
                                            reinterpret_cast<const nv_bfloat162*>(
                                                &contributions[slot])[element]);
                                    }
                                }
                                casted_bf16[element] =
                                    __float22bfloat162_rn(reduced);
                            } else {
                                float2 pair_sums[5];
                                #pragma unroll
                                for (uint32_t pair = 0; pair < 5; ++pair) {
                                    const auto lhs = __bfloat1622float2(
                                        reinterpret_cast<const nv_bfloat162*>(
                                            &contributions[pair * 2])[element]);
                                    const auto rhs = __bfloat1622float2(
                                        reinterpret_cast<const nv_bfloat162*>(
                                            &contributions[pair * 2 + 1])[element]);
                                    pair_sums[pair] =
                                        make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
                                }
                                pair_sums[0].x += pair_sums[1].x;
                                pair_sums[0].y += pair_sums[1].y;
                                pair_sums[2].x += pair_sums[3].x;
                                pair_sums[2].y += pair_sums[3].y;
                                pair_sums[0].x += pair_sums[2].x;
                                pair_sums[0].y += pair_sums[2].y;
                                pair_sums[0].x += pair_sums[4].x;
                                pair_sums[0].y += pair_sums[4].y;
                                casted_bf16[element] =
                                    __float22bfloat162_rn(pair_sums[0]);
                            }
                        }
                        if constexpr (kFullRowCombine == 4) {
                            paired_output[vector] = casted;
                        } else {
                            reinterpret_cast<uint4*>(y)[
                                static_cast<uint64_t>(token_idx) * kNumUint4 +
                                vector_idx + vector * kCombineVectorStride] = casted;
                        }
                    }
                    if constexpr (kFullRowCombine == 4) {
                        // Adjacent output vectors share one aligned 32-byte store.
                        const auto dst = reinterpret_cast<uint4*>(y) +
                            static_cast<uint64_t>(token_idx) * kNumUint4 + vector_idx;
                        const auto& a = paired_output[0];
                        const auto& b = paired_output[1];
                        asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::
                            "l"(dst), "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
                            "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w) : "memory");
                    }
                }
                if (combine_work == first_combine_work)
                    MEGAMOE_IKET_MARK(MegaLocalReduceEnd);
                // The warp-local path reuses no shared combine state between
                // rows. Inbox retirement and reduction stay per-vector.
                if constexpr (!kWarpCombine)
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                combine_phase ^= 1;
            }
            if constexpr (kTraceRecycling && kResetRows != 0) {
                if (epilogue_thread_idx == 0 &&
                    first_combine_work < num_tokens * kCombineChunksPerToken) {
                    uint64_t stamp;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
                    debug_timings[24 + 3 * num_tokens + sm_idx] = stamp;
                }
            }
        } else {
        // Retain the established pipelined TMA path for every other model
        // shape until an identical-input source comparison proves otherwise.
        // NOTES: reuse shared memory from start up to the barriers
        // 1 token, 1 topk latency: ~3 us
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        // 3 slots of chunk is needed: 2 load stages and 1 store
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;

        // NOTES: either 1 or 2 chunks for simplicity
        // NOTES: Restrict on both smem and register
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        // Per-warp buffer: 2 stage load buffers + 1 store buffer
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // Per-warp barriers
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        // Iterate over all tokens
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // Read top-k slot indices: each lane reads one slot, then broadcast via exchange
            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            const int stored_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_expert_idx >= 0);

            if constexpr (kRankGatedCombine) {
                const int expert_rank_idx = stored_expert_idx >= 0
                    ? stored_expert_idx / static_cast<int>(kNumExpertsPerRank)
                    : -1;
                if constexpr (kNumRanks > 32) {
                    wait_combine_producer(expert_rank_idx, token_idx);
                } else {
                    uint32_t required_rank_mask = 0;
                    #pragma unroll
                    for (uint32_t rank_idx = 0; rank_idx < kNumRanks; ++ rank_idx) {
                        if (__any_sync(0xffffffff,
                                       expert_rank_idx == static_cast<int>(rank_idx)))
                            required_rank_mask |= 1u << rank_idx;
                    }
                    if (lane_idx < kNumRanks and
                        (required_rank_mask & (1u << lane_idx)) != 0) {
                        constexpr int64_t kNumTimeoutCycles =
                            300ll * 2000000000ll;
                        const auto ready_epoch_ptr =
                            workspace.get_combine_ready_epoch_ptr(lane_idx);
                        const auto start_clock = clock64();
                        while (ptx::ld_acq_sys(ready_epoch_ptr) !=
                               combine_ready_epoch) {
                            if (clock64() - start_clock >= kNumTimeoutCycles) {
                                printf("MegaMoE rank-gated combine timeout (300s): rank=%d token=%d expert_rank=%d ready=%llu target=%llu\n",
                                       sym_buffer.rank_idx, token_idx, lane_idx,
                                       static_cast<unsigned long long>(
                                           ptx::ld_acq_sys(ready_epoch_ptr)),
                                       static_cast<unsigned long long>(
                                           combine_ready_epoch));
                                DG_DEVICE_ASSERT(false and
                                    "Rank-gated combine timeout");
                            }
                        }
                    }
                }
                // All producer acquires must finish before combine loads.
                __syncwarp();
            }

            // Iterate all chunks
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // Move mask and load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        // Move
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;

                        // Load
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // Load the first selection
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // Accumulate all top-k contributions for this chunk in float registers
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // Prefetch next top-k into the buffer while current is being accumulated
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // Accumulate
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                // Cast
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // Wait share memory release and write
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA store the token chunk
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        }
        MEGAMOE_IKET_MARK(MegaCombineEnd);

    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm

#undef MEGAMOE_IKET_MARK
#undef MEGAMOE_IKET_DETAIL_MARK
#undef MEGAMOE_IKET_FIRST
#undef MEGAMOE_IKET_ONCE
#undef MEGAMOE_IKET_DONE
