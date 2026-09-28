// Two-stage fused router shared by the routed-expert FFN paths:
// 1. Tensor-core gate GEMM and optional activation quantization, tiled over
//    tokens and experts.
// 2. Top-k selection publishes destination expert indices/counts before PDL.
//    All-push sends scores and SF; MegaMoE independently pushes activation rows.
// The supported renormalized 128/8 and 512/10 shapes share one CTA per token;
// the generic selector retains one warp per token for other configurations.
// PDL overlaps the producer tail with dependent work without removing its
// data-readiness dependencies.

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/valid_tokens.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/impls/mega_moe_recycle.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <type_traits>

#ifdef IKET_BUILD_ENABLED
#define IKET_DEFAULT_INSTR_METHOD_WITHOUT_PAYLOAD_PMTRIG5
#include <iket/iket_device_apis.cuh>
CREATE_IKET_MARKER(RouteMetadataTransferStart);
CREATE_IKET_MARKER(RouteMetadataIssued);
CREATE_IKET_MARKER(RouteSfTransferStart);
CREATE_IKET_MARKER(RouteSfTransferIssued);
CREATE_IKET_MARKER(RouteCountsTransferStart);
CREATE_IKET_MARKER(RouteMetadataPublished);
#define ROUTE_IKET_MARK(name) IKET_MARK(name)
#else
#define ROUTE_IKET_MARK(name) ((void)0)
#endif

#include "gate_topk_fused_common.cuh"

namespace {

// Second kernel: top-k over the assembled logits (one block per token tile,
// one warp per token). PDL orders the completed local inputs before dispatch.
template <typename IndexT, typename MapT, bool kPrepareMegaMoERoutes,
          bool kRankReadyRoutePublish, bool kPerfectBalance,
          bool kRenormalize, bool kAllPush>
__launch_bounds__(kTokensPerBlock * kWarpSize) __global__ void gate_topk_select_kernel(
    const float* __restrict__ logits_gmem,  // [mtiles, 16, kMaxExperts]
    float* __restrict__ topk_weights,
    IndexT* __restrict__ topk_ids,
    const MapT* __restrict__ expert_map,    // optional global -> group-local map
    const int64_t* __restrict__ valid_token_count,
    int valid_count_size,
    float* __restrict__ w_stage,            // optional symm-buffer staging copy
    void* __restrict__ route_workspace,
    const int64_t* __restrict__ route_buffer_ptrs,
    const int route_rank, const int route_num_ranks,
    const int route_bucket, const int route_num_sms,
    const int num_tokens, const int num_experts, const int topk,
    const bool defer_route_dependent_launch) {
  const int mtile = blockIdx.x;
  const int m0 = mtile * kTokensPerBlock;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  device::PDL::wait<true>();

  const float* block_logits =
      logits_gmem + static_cast<size_t>(mtile) * kTokensPerBlock * kMaxExperts;

  constexpr int kMaxExpertsPerLane = kMaxExperts / kWarpSize;  // 16
  const int per_lane = (num_experts + kWarpSize - 1) / kWarpSize;
  {
    const int t = warp;
    const int token = m0 + t;
    const auto validity =
        minisgl::token_validity(token, num_tokens, valid_token_count, valid_count_size);
    if (token >= num_tokens) {
      if constexpr (!kPrepareMegaMoERoutes)
        return;
    } else if (!validity.valid) {
      if (lane < topk) {
        topk_ids[static_cast<size_t>(token) * topk + lane] =
            static_cast<IndexT>(-1);
        topk_weights[static_cast<size_t>(token) * topk + lane] = 0.0f;
        if (w_stage != nullptr)
          w_stage[static_cast<size_t>(token) * topk + lane] = 0.0f;
      }
      if constexpr (!kPrepareMegaMoERoutes)
        return;
    } else {

    float logit[kMaxExpertsPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) {
      const int e = i * kWarpSize + lane;
      logit[i] = (i < per_lane && e < num_experts)
                     ? block_logits[t * kMaxExperts + e]
                     : -FLT_MAX;
    }

    // Qwen3.5 renormalizes only the selected experts.  Specializing this
    // policy at launch time avoids an otherwise unused 512-way softmax while
    // keeping the non-renormalized model path bit-for-bit unchanged.
    float m_all = -FLT_MAX;
    float denom_all = 0.f;
    if constexpr (!kRenormalize) {
      #pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i)
        m_all = fmaxf(m_all, logit[i]);
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m_all = fmaxf(m_all, __shfl_xor_sync(0xffffffffu, m_all, off));
      #pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i)
        denom_all += logit[i] > -FLT_MAX ? __expf(logit[i] - m_all) : 0.f;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        denom_all += __shfl_xor_sync(0xffffffffu, denom_all, off);
    }

    uint32_t used = 0;
    float my_logit = -FLT_MAX;
    int my_expert = -1;
    float top0 = -FLT_MAX;
    for (int j = 0; j < topk; ++j) {
      float best = -FLT_MAX;
      int best_i = -1;
      #pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i) {
        if (!((used >> i) & 1u) && logit[i] > best) {
          best = logit[i];
          best_i = i;
        }
      }
      const uint32_t value_key = ordered_float_key(best);
      const uint32_t max_value_key = warp_redux_max_u32(value_key);
      const uint32_t lane_key = value_key == max_value_key
                                    ? 0xffffffffu - lane
                                    : 0;
      const uint32_t src = 0xffffffffu - warp_redux_max_u32(lane_key);
      const float v = __shfl_sync(0xffffffffu, best, src);
      const int wbi = __shfl_sync(0xffffffffu, best_i, src);
      if (lane == static_cast<int>(src)) used |= 1u << wbi;
      if (j == 0) top0 = v;
      if (lane == j) {
        my_logit = v;
        if constexpr (kPerfectBalance) {
          // Replace the original expert-id finalization in place. Perfect mode
          // admits only power-of-two expert counts, so the balanced sequence
          // uses 32-bit multiply/add/mask arithmetic rather than a division.
          // The normal GEMM, softmax, and top-k work above remains unchanged,
          // with no extra kernel, graph node, synchronization, copy, or
          // device-side runtime branch.
          my_expert =
              ((route_rank * validity.total + validity.compact_index) * topk + lane) &
              (num_experts - 1);
        } else {
          my_expert = wbi * kWarpSize + static_cast<int>(src);
        }
      }
    }

    const float normalization_max = kRenormalize ? top0 : m_all;
    float w = lane < topk ? __expf(my_logit - normalization_max) : 0.f;
    float w_sum = w;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
    if (lane < topk) {
      const auto mapped_expert = expert_map == nullptr
                                     ? static_cast<MapT>(my_expert)
                                     : __ldg(expert_map + my_expert);
      const auto token_topk_idx = static_cast<size_t>(token) * topk + lane;
      topk_ids[token_topk_idx] = static_cast<IndexT>(mapped_expert);
      const float w_out = kRenormalize ? w / w_sum : w / denom_all;
      topk_weights[token_topk_idx] = w_out;
      if (w_stage != nullptr)
        w_stage[token_topk_idx] = w_out;

      if constexpr (kPrepareMegaMoERoutes) {
        if (mapped_expert >= 0 && mapped_expert < num_experts) {
          const auto workspace = deep_gemm::layout::Workspace(
              route_workspace, route_num_ranks, num_experts,
              route_bucket, topk);
          if constexpr (kRankReadyRoutePublish) {
            // Allocate compact expert rows before the existing all-rank release.
            const uint32_t expert = static_cast<uint32_t>(mapped_expert);
            const uint32_t dst_rank = expert / workspace.num_experts_per_rank;
            const uint32_t local_expert = expert % workspace.num_experts_per_rank;
            const auto sum_offset = reinterpret_cast<uintptr_t>(
                workspace.get_expert_recv_count_sum_ptr(local_expert)) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_sum = reinterpret_cast<uint64_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + sum_offset);
            ROUTE_IKET_MARK(RouteMetadataTransferStart);
            const uint64_t slot = deep_gemm::ptx::atomic_add_sys(remote_sum, 1);
            // Source-indexed scores do not depend on the returned expert slot.
            // Issue them while its remote atomic response is outstanding.
            if constexpr (kAllPush) {
              const auto score_offset = reinterpret_cast<uintptr_t>(
                  workspace.get_score_inbox_ptr(route_rank) + token_topk_idx) -
                  reinterpret_cast<uintptr_t>(route_workspace);
              *reinterpret_cast<float*>(static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                                        score_offset) = w_out;
            }
            DG_DEVICE_ASSERT(slot < workspace.num_max_recv_tokens_per_expert);
            const auto offset = reinterpret_cast<uintptr_t>(
                workspace.get_src_token_topk_idx_ptr(local_expert,
                  workspace.num_ranks == 1 ? 0 : workspace.metadata_slot, static_cast<uint32_t>(slot))) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_dst = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + offset);
            *remote_dst = static_cast<uint32_t>(route_rank) * route_bucket * topk +
                static_cast<uint32_t>(token_topk_idx);
            ROUTE_IKET_MARK(RouteMetadataIssued);
          } else {
            auto* send_count = reinterpret_cast<unsigned int*>(
                workspace.get_expert_send_count_ptr(
                    static_cast<uint32_t>(mapped_expert)));
            const auto dst_slot = atomicAdd(send_count, 1u);
            const auto dst_rank = static_cast<uint32_t>(mapped_expert) /
                                  workspace.num_experts_per_rank;
            const auto dst_local_expert =
                static_cast<uint32_t>(mapped_expert) %
                workspace.num_experts_per_rank;
            auto* local_dst = workspace.get_src_token_topk_idx_ptr(
                dst_local_expert, route_rank, dst_slot);
            const auto byte_offset =
                reinterpret_cast<uintptr_t>(local_dst) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_dst = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                byte_offset);
            *remote_dst = static_cast<uint32_t>(token_topk_idx);
          }
        }
      }
    }
    }
  }


  if constexpr (kPrepareMegaMoERoutes) {
    // Every route is now grouped and its source index is already in the
    // destination's symmetric workspace.  A release/acquire last-block ticket
    // lets one block finalize the per-source and aggregate expert counts
    // without another kernel or a cooperative grid launch.
    __syncthreads();
    if constexpr (kAllPush) {
      const auto input_workspace = deep_gemm::layout::Workspace(
          route_workspace, route_num_ranks, num_experts, route_bucket, topk);
      if (threadIdx.x == 0)
        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                     :: "l"(input_workspace.get_genphase_flag_ptr(deep_gemm::layout::Workspace::CountsReady)),
                        "r"(static_cast<uint32_t>(min(kTokensPerBlock, num_tokens - m0))) : "memory");
    }
    // DeepGEMM launches MegaMoE as a programmatic dependent. All selector
    // blocks have published their route indices/count increments at this
    // point, so MegaMoE may overlap descriptor/barrier initialization with
    // the last block's remote count and readiness publication. MegaMoE polls
    // route_ready before consuming any of those values.
    if (!defer_route_dependent_launch)
      device::PDL::launch<true>();
    const auto workspace = deep_gemm::layout::Workspace(
        route_workspace, route_num_ranks, num_experts, route_bucket, topk);
    __shared__ int is_last_block;
    if (threadIdx.x == 0) {
      is_last_block = atom_add_acqrel_gpu(
          reinterpret_cast<int*>(workspace.get_route_prepare_counter_ptr()), 1) ==
          gridDim.x - 1;
    }
    __syncthreads();
    if (is_last_block) {
      if constexpr (!kRankReadyRoutePublish) {
        for (int expert = static_cast<int>(threadIdx.x);
             expert < num_experts;
             expert += static_cast<int>(blockDim.x)) {
          const auto count = static_cast<uint32_t>(
              *workspace.get_expert_send_count_ptr(expert));
          const auto dst_rank = expert / workspace.num_experts_per_rank;
          const auto dst_local_expert = expert % workspace.num_experts_per_rank;
          const auto recv_count_ptr = workspace.get_expert_recv_count_ptr(
              route_rank, dst_local_expert);
          const auto recv_count_offset =
              reinterpret_cast<uintptr_t>(recv_count_ptr) -
              reinterpret_cast<uintptr_t>(route_workspace);
          auto* remote_recv_count = reinterpret_cast<uint64_t*>(
              static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
              recv_count_offset);
          *remote_recv_count = count;

          if constexpr (not kRankReadyRoutePublish) {
            const auto recv_sum_ptr =
                workspace.get_expert_recv_count_sum_ptr(dst_local_expert);
            const auto recv_sum_offset =
                reinterpret_cast<uintptr_t>(recv_sum_ptr) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_recv_sum = reinterpret_cast<uint64_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                recv_sum_offset);
            const uint64_t completed_count =
                (static_cast<uint64_t>(route_num_sms) << 32) | count;
            deep_gemm::ptx::atomic_add_rel_sys(remote_recv_sum, completed_count);
          }
        }
      }
      __syncthreads();
      if constexpr (kRankReadyRoutePublish) {
        for (int dst = static_cast<int>(threadIdx.x);
             dst < route_num_ranks; dst += static_cast<int>(blockDim.x)) {
          const auto route_ready_offset =
              reinterpret_cast<uintptr_t>(
                  workspace.get_route_ready_count_ptr()) -
              reinterpret_cast<uintptr_t>(route_workspace);
          auto* remote_route_ready = reinterpret_cast<uint64_t*>(
              static_cast<uintptr_t>(route_buffer_ptrs[dst]) +
              route_ready_offset);
          // One readiness release orders the published source descriptors and
          // whichever count representation this specialization produces.

          deep_gemm::ptx::atomic_add_rel_sys(remote_route_ready, 1);
        }
        __syncthreads();
        if (threadIdx.x < kWarpSize) ROUTE_IKET_MARK(RouteMetadataPublished);
      }
      if (threadIdx.x == 0)
        *workspace.get_route_prepare_counter_ptr() = 0;
    }
  }

}

// Share one CTA per token across the supported renormalized MoE shapes.
// Warp 0 selects in registers with the original strict tie order; the CTA
// cooperates in route publication.
// Shape-derived constants are fixed across graph replays; the bank epoch is not.
struct RouteLayout {
    uint32_t experts_per_rank;
    uint32_t max_recv_tokens_per_expert;
    uint32_t max_pool_tokens;
    uint32_t max_pool_blocks;
};

__device__ __forceinline__ deep_gemm::layout::Workspace make_route_workspace(
    const RouteLayout& layout, void* base, uint32_t ranks, uint32_t experts,
    uint32_t tokens, uint32_t topk) {
    // Preserve the constructor's live device epoch read. Its pure shape math
    // becomes dead after these fields are replaced with the host-computed values.
    auto workspace = deep_gemm::layout::Workspace(base, ranks, experts, tokens, topk);
    workspace.num_experts_per_rank = layout.experts_per_rank;
    workspace.num_max_recv_tokens_per_expert = layout.max_recv_tokens_per_expert;
    workspace.num_max_pool_tokens = layout.max_pool_tokens;
    workspace.num_max_pool_blocks = layout.max_pool_blocks;
    return workspace;
}

template <int kExperts, int kTopK, typename IndexT, typename MapT, bool kPrepareMegaMoERoutes,
          bool kRankReadyRoutePublish, bool kPerfectBalance, bool kAllPush, bool kProducerPrefix,
          bool kPreparePriority, bool kSfVector64>
__launch_bounds__(kTPB) __global__ void gate_topk_select_cta_kernel(
    const float* __restrict__ logits_gmem,
    float* __restrict__ topk_weights,
    IndexT* __restrict__ topk_ids,
    const MapT* __restrict__ expert_map,
    const int64_t* __restrict__ valid_token_count,
    int valid_count_size,
    float* __restrict__ w_stage,
    void* __restrict__ route_workspace,
    const RouteLayout route_layout,
    const int64_t* __restrict__ route_buffer_ptrs,
    const int route_rank, const int route_num_ranks,
    const int route_bucket, const int route_num_sms,
    const int num_tokens, const bool defer_route_dependent_launch,
    const uint8_t* __restrict__ source_sf, const int hidden_size,
    const int reset_rows, uint64_t* reset_timings,
    void* ahead_workspace, const int64_t* ahead_peers,
    const uint8_t* __restrict__ source_xq) {
  static_assert(!kProducerPrefix || (kAllPush && kRankReadyRoutePublish));
  static_assert(!kPreparePriority || (kAllPush && kRankReadyRoutePublish && !kProducerPrefix));
  static_assert(!kSfVector64 || (kAllPush && kRankReadyRoutePublish));
  static_assert(kExperts <= kMaxExperts && kExperts % kWarpSize == 0);
  static_assert(kTopK > 0 && kTopK <= kWarpSize);
  const int token = static_cast<int>(blockIdx.x);
  if constexpr (kAllPush) {
    if (token >= num_tokens) {
      // Auxiliary CTAs touch only the alternate activation bank. The prior
      // producer retired before the ordinary gate GEMM launch; these workers
      // need no current logits/SF and may overlap that GEMM's tail.
      const auto workspace = make_route_workspace(route_layout,
          route_workspace, route_num_ranks, kExperts, route_bucket, kTopK);
#ifdef MEGAMOE_RECYCLE_AHEAD
      const auto ahead = make_route_workspace(route_layout,
          ahead_workspace, route_num_ranks, kExperts, route_bucket, kTopK);
      if (reset_timings)
        deep_gemm::mega_moe_recycle_ahead_worker<true>(workspace, route_buffer_ptrs,
            ahead, ahead_peers, hidden_size, reset_rows, token - num_tokens,
            reset_timings, num_tokens, route_num_sms);
      else
        deep_gemm::mega_moe_recycle_ahead_worker<false>(workspace, route_buffer_ptrs,
            ahead, ahead_peers, hidden_size, reset_rows, token - num_tokens,
            nullptr, num_tokens, route_num_sms);
#else
      if (reset_timings)
        deep_gemm::mega_moe_recycle_worker<true>(workspace, route_buffer_ptrs, hidden_size, reset_rows,
            token - num_tokens, reset_timings, num_tokens, route_num_sms);
      else
        deep_gemm::mega_moe_recycle_worker<false>(workspace, route_buffer_ptrs, hidden_size, reset_rows,
            token - num_tokens, nullptr, num_tokens, route_num_sms);
#endif
      return;
    }
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  device::PDL::wait<true>();

  const auto validity =
      minisgl::token_validity(token, num_tokens, valid_token_count, valid_count_size);
  const bool valid = validity.valid;

  float selected_logit = -FLT_MAX, selected_max = -FLT_MAX;
  if (valid && warp == 0) {
    constexpr int kPerLane = kExperts / kWarpSize;
    float logits[kPerLane];
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i)
      logits[i] = logits_gmem[static_cast<size_t>(token) * kMaxExperts + i * kWarpSize + lane];
    uint32_t used = 0;
    float my_logit = -FLT_MAX, top0 = -FLT_MAX;
    int my_expert = -1;
    #pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      float best = -FLT_MAX;
      int best_i = -1;
      #pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        if (!(used & (1u << i)) && logits[i] > best) {
          best = logits[i];
          best_i = i;
        }
      }
      const uint32_t key = ordered_float_key(best);
      const uint32_t max_key = warp_redux_max_u32(key);
      const uint32_t src = 0xffffffffu - warp_redux_max_u32(
          key == max_key ? 0xffffffffu - lane : 0);
      const float value = __shfl_sync(0xffffffffu, best, src);
      const int winner_i = __shfl_sync(0xffffffffu, best_i, src);
      if (lane == static_cast<int>(src)) used |= 1u << winner_i;
      if (j == 0) top0 = value;
      if (lane == j) {
        my_logit = value;
        my_expert = winner_i * kWarpSize + static_cast<int>(src);
      }
    }
    {
      float w = 0.f, w_sum = 0.f;
      if constexpr (kAllPush) {
        selected_logit = my_logit;
        selected_max = top0;
      } else {
        w = tid < kTopK ? __expf(my_logit - top0) : 0.f;
        w_sum = w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
      }
      if (tid < kTopK) {
        int expert = my_expert;
        if constexpr (kPerfectBalance) {
          expert =
              ((route_rank * validity.total + validity.compact_index) * kTopK + tid) &
              (kExperts - 1);
        }
        const auto mapped_expert = expert_map == nullptr
                                       ? static_cast<MapT>(expert)
                                       : __ldg(expert_map + expert);
        const auto token_topk_idx =
            static_cast<size_t>(token) * kTopK + tid;
        topk_ids[token_topk_idx] = static_cast<IndexT>(mapped_expert);
        if constexpr (!kAllPush) {
          const float w_out = w / w_sum;
          topk_weights[token_topk_idx] = w_out;
          if (w_stage != nullptr)
            w_stage[token_topk_idx] = w_out;
        }
      }
    }
  } else if (!valid && tid < kTopK) {
    const auto token_topk_idx = static_cast<size_t>(token) * kTopK + tid;
    topk_ids[token_topk_idx] = static_cast<IndexT>(-1);
    topk_weights[token_topk_idx] = 0.0f;
    if (w_stage != nullptr)
      w_stage[token_topk_idx] = 0.0f;
  }


  if constexpr (kPrepareMegaMoERoutes) {
    // This invocation owns the bank until all token CTAs publish their routes.
    // Keep one live device epoch across route, SF and final publication work.
    const auto workspace = make_route_workspace(route_layout,
        route_workspace, route_num_ranks, kExperts, route_bucket, kTopK);
    // Hand off selected IDs once; independent SF warps overlap warp0's
    // remote slot allocation. Each destination has exactly one SF writer.
    __syncthreads();
    if constexpr (kAllPush) {
      static_assert(kRankReadyRoutePublish && kPrepareMegaMoERoutes);
      if (valid && warp == 0) {
        const auto token_topk_idx = static_cast<size_t>(token) * kTopK + lane;
        const auto mapped_expert = lane < kTopK ? topk_ids[token_topk_idx] : IndexT(-1);
        const bool routed = lane < kTopK && mapped_expert >= 0 && mapped_expert < kExperts;
        uint32_t dst_rank = 0, local_expert = 0;
        // Existing dense-descriptor host bounds cap ALL claims at UINT32_MAX.
        // Full-word reset precedes this epoch; consumers read after rank-ready.
        uint32_t slot = 0;
        uintptr_t remote_base = 0, remote_sum = 0;
        if (routed) {
          const uint32_t expert = static_cast<uint32_t>(mapped_expert);
          dst_rank = expert * static_cast<uint32_t>(route_num_ranks) / kExperts;
          local_expert = expert - dst_rank * workspace.num_experts_per_rank;
          remote_base = static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]);
          const auto sum_offset = reinterpret_cast<uintptr_t>(
              workspace.get_expert_recv_count_sum_ptr(local_expert)) -
              reinterpret_cast<uintptr_t>(route_workspace);
          remote_sum = remote_base + sum_offset;
          ROUTE_IKET_MARK(RouteMetadataTransferStart);
        }
        // Predicate only the memory operation, after address-path reconvergence.
        // Initialize inactive-lane outputs before issue, avoiding a later write
        // to the still-pending slot register in the inactive branch.
        asm volatile("{ .reg .pred active; setp.ne.u32 active, %3, 0; "
                     "@active atom.sys.global.add.u32 %0, [%1], %2; }"
                     : "+r"(slot) : "l"(remote_sum), "r"(uint32_t(1)),
                       "r"(static_cast<uint32_t>(routed)) : "memory");
        // Every lane participates in the unchanged normalization tree. SF warps
        // already have IDs, and no returned slot is needed by this arithmetic.
        float w = lane < kTopK ? __expf(selected_logit - selected_max) : 0.f;
        float w_sum = w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
        if (lane < kTopK) {
          const float w_out = w / w_sum;
          topk_weights[token_topk_idx] = w_out;
          if (w_stage != nullptr)
            w_stage[token_topk_idx] = w_out;
          if (routed) {
            const auto score_offset = reinterpret_cast<uintptr_t>(
                workspace.get_score_inbox_ptr(route_rank) + token_topk_idx) -
                reinterpret_cast<uintptr_t>(route_workspace);
            *reinterpret_cast<float*>(remote_base + score_offset) = w_out;
            DG_DEVICE_ASSERT(slot < workspace.num_max_recv_tokens_per_expert);
            const auto offset = reinterpret_cast<uintptr_t>(
                workspace.get_src_token_topk_idx_ptr(local_expert,
                workspace.num_ranks == 1 ? 0 : workspace.metadata_slot,
                static_cast<uint32_t>(slot))) - reinterpret_cast<uintptr_t>(route_workspace);
            *reinterpret_cast<uint32_t*>(remote_base + offset) =
                static_cast<uint32_t>(route_rank) * route_bucket * kTopK +
                static_cast<uint32_t>(token_topk_idx);
            ROUTE_IKET_MARK(RouteMetadataIssued);
          }
        }
      }
    } else {
      if (valid && warp == 0 && lane < kTopK) {
        const auto token_topk_idx = static_cast<size_t>(token) * kTopK + lane;
        const auto mapped_expert = topk_ids[token_topk_idx];
        if (mapped_expert >= 0 && mapped_expert < kExperts) {
          if constexpr (kRankReadyRoutePublish) {
            // Allocate compact expert rows before the existing all-rank release.
            const uint32_t expert = static_cast<uint32_t>(mapped_expert);
            // Host validation guarantees kExperts is divisible by route_num_ranks.
            // Use the compile-time expert divisor on the slot-claim address path.
            const uint32_t dst_rank = expert / workspace.num_experts_per_rank;
            const uint32_t local_expert = expert % workspace.num_experts_per_rank;
            const auto sum_offset = reinterpret_cast<uintptr_t>(
                workspace.get_expert_recv_count_sum_ptr(local_expert)) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_sum = reinterpret_cast<uint64_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + sum_offset);
            ROUTE_IKET_MARK(RouteMetadataTransferStart);
            const uint64_t slot = deep_gemm::ptx::atomic_add_sys(remote_sum, 1);
            DG_DEVICE_ASSERT(slot < workspace.num_max_recv_tokens_per_expert);
            const auto offset = reinterpret_cast<uintptr_t>(
                workspace.get_src_token_topk_idx_ptr(local_expert,
                workspace.num_ranks == 1 ? 0 : workspace.metadata_slot, static_cast<uint32_t>(slot))) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_dst = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + offset);
            *remote_dst = static_cast<uint32_t>(route_rank) * route_bucket * kTopK +
                static_cast<uint32_t>(token_topk_idx);
            ROUTE_IKET_MARK(RouteMetadataIssued);
          } else {
            auto* send_count = reinterpret_cast<unsigned int*>(
                workspace.get_expert_send_count_ptr(
                    static_cast<uint32_t>(mapped_expert)));
            const auto dst_slot = atomicAdd(send_count, 1u);
            const auto dst_rank = static_cast<uint32_t>(mapped_expert) /
                                  workspace.num_experts_per_rank;
            const auto dst_local_expert =
                static_cast<uint32_t>(mapped_expert) %
                workspace.num_experts_per_rank;
            auto* local_dst = workspace.get_src_token_topk_idx_ptr(
                dst_local_expert, route_rank, dst_slot);
            const auto byte_offset =
                reinterpret_cast<uintptr_t>(local_dst) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_dst = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                byte_offset);
            *remote_dst = static_cast<uint32_t>(token_topk_idx);
          }
        }
      }
    }
    if constexpr (kAllPush) {
      if (valid && warp != 0) {
        const int expert = lane < kTopK ?
            static_cast<int>(topk_ids[static_cast<size_t>(token) * kTopK + lane]) : -1;
        const int peer = expert >= 0 && expert < kExperts ?
            static_cast<uint32_t>(expert) * static_cast<uint32_t>(route_num_ranks) / kExperts : -1;
        if constexpr (kPreparePriority) {
          // SF warp1 already has mapped IDs and peers; keep key preparation
          // independent of the routing warp's slot claims and normalization.
          if (warp == 1) {
            const uint32_t key = __reduce_min_sync(0xffffffffu, peer >= 0 ?
                static_cast<uint32_t>(expert) - peer * workspace.num_experts_per_rank :
                workspace.num_experts_per_rank);
            if (lane == 0)
              workspace.get_gather_index_ptr((2 + workspace.metadata_slot) * route_bucket)[token] = key;
          }
        }
        const uint64_t source_rows = 2ull * route_num_ranks * route_bucket;
        const uint64_t row = workspace.get_source_l1_row(route_rank, token);
        const auto sf_offset = reinterpret_cast<uintptr_t>(
            workspace.get_compact_frame_base_ptr(hidden_size, kTopK)) -
            reinterpret_cast<uintptr_t>(route_workspace) +
            source_rows * hidden_size + row * (hidden_size / 32);
        uint4 prefix[2];
        uint32_t epoch = 0;
        if constexpr (kProducerPrefix) {
          if (reset_rows)
            epoch = deep_gemm::ptx::ld_acq(workspace.get_nvl_barrier_counter_ptr());
          const auto* src = reinterpret_cast<const uint4*>(
              source_xq + static_cast<size_t>(token) * hidden_size);
          #pragma unroll
          for (int v = 0; v < 2; ++v) {
            auto value = src[lane + v * kWarpSize];
            value.x = value.x == 0x80000000u ? 0 : value.x;
            value.y = value.y == 0x80000000u ? 0 : value.y;
            value.z = value.z == 0x80000000u ? 0 : value.z;
            value.w = value.w == 0x80000000u ? 0 : value.w;
            prefix[v] = value;
          }
        }
        for (int dst = warp - 1; dst < route_num_ranks; dst += kTPB / kWarpSize - 1) {
          if (__any_sync(0xffffffffu, peer == dst)) {
            if constexpr (kProducerPrefix) {
              // Recycled banks require a destination acquire before any write.
              // Never wait CountsReady here: this CTA contributes to that count.
              // Nonrecycled calls retain their external preparation contract.
              if (reset_rows) {
                const auto ready_offset = reinterpret_cast<uintptr_t>(workspace.get_payload_ready_ptr()) -
                                          reinterpret_cast<uintptr_t>(route_workspace);
                const auto* ready = reinterpret_cast<const uint32_t*>(
                    static_cast<uintptr_t>(route_buffer_ptrs[dst]) + ready_offset);
                if (lane == 0)
                  while ((deep_gemm::ptx::ld_acq_sys(ready) - epoch) & 0x80000000u);
                __syncwarp();
              }
              const auto data_offset = reinterpret_cast<uintptr_t>(
                  workspace.get_compact_frame_base_ptr(hidden_size, kTopK)) -
                  reinterpret_cast<uintptr_t>(route_workspace) + row * hidden_size;
              #pragma unroll
              for (int v = 0; v < 2; ++v) {
                const auto address = static_cast<uintptr_t>(route_buffer_ptrs[dst]) +
                    data_offset + (lane + v * kWarpSize) * sizeof(uint4);
                const auto value = prefix[v];
                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
                    "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
              }
            }
            // SF is never cleared. Auxiliary CTAs publish destination payload
            // readiness separately into the sender's input-ready counter.

            ROUTE_IKET_MARK(RouteSfTransferStart);
            const auto* src = reinterpret_cast<const uint32_t*>(
                source_sf + static_cast<size_t>(token) * (hidden_size / 32));
            auto* out = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst]) + sf_offset);
            if constexpr (kSfVector64) {
              const auto* src64 = reinterpret_cast<const uint64_t*>(src);
              auto* out64 = reinterpret_cast<uint64_t*>(out);
              #pragma unroll 1
              for (int pair = lane; pair < hidden_size / 256; pair += kWarpSize)
                out64[pair] = src64[pair];
            } else {
              for (int word = lane; word < hidden_size / 128; word += kWarpSize)
                out[word] = src[word];
            }
            ROUTE_IKET_MARK(RouteSfTransferIssued);
          }
        }
      } else if constexpr (kPreparePriority) {
        // Invalid physical rows also overwrite their bank entry before CountsReady.
        if (tid == kWarpSize)
          workspace.get_gather_index_ptr((2 + workspace.metadata_slot) * route_bucket)[token] =
              workspace.num_experts_per_rank;
      }
    }
    // Existing last-CTA acqrel ticket and system route-ready publication
    // transitively cover route descriptors, SF and optional prefix stores.
    __syncthreads();
    if constexpr (kAllPush) {
      if (threadIdx.x == 0)
        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                     :: "l"(workspace.get_genphase_flag_ptr(deep_gemm::layout::Workspace::CountsReady)),
                        "r"(static_cast<uint32_t>(1)) : "memory");
    }
    if (!defer_route_dependent_launch)
      device::PDL::launch<true>();
    __shared__ int is_last_block;
    if (tid == 0) {
      is_last_block = atom_add_acqrel_gpu(
          reinterpret_cast<int*>(workspace.get_route_prepare_counter_ptr()), 1) ==
          num_tokens - 1;
    }
    __syncthreads();
    if (is_last_block) {
      if constexpr (!kRankReadyRoutePublish) {
        for (int expert = tid; expert < kExperts; expert += kTPB) {
          const auto count = static_cast<uint32_t>(
              *workspace.get_expert_send_count_ptr(expert));
          const auto dst_rank = expert / workspace.num_experts_per_rank;
          const auto dst_local_expert =
              expert % workspace.num_experts_per_rank;
          const auto recv_count_ptr = workspace.get_expert_recv_count_ptr(
              route_rank, dst_local_expert);
          const auto recv_count_offset =
              reinterpret_cast<uintptr_t>(recv_count_ptr) -
              reinterpret_cast<uintptr_t>(route_workspace);
          auto* remote_recv_count = reinterpret_cast<uint64_t*>(
              static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
              recv_count_offset);
          *remote_recv_count = count;

          if constexpr (not kRankReadyRoutePublish) {
            const auto recv_sum_ptr =
                workspace.get_expert_recv_count_sum_ptr(dst_local_expert);
            const auto recv_sum_offset =
                reinterpret_cast<uintptr_t>(recv_sum_ptr) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_recv_sum = reinterpret_cast<uint64_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                recv_sum_offset);
            const uint64_t completed_count =
                (static_cast<uint64_t>(route_num_sms) << 32) | count;
            deep_gemm::ptx::atomic_add_rel_sys(remote_recv_sum, completed_count);
          }
        }
      }
      __syncthreads();
      if constexpr (kRankReadyRoutePublish) {
        for (int dst = tid; dst < route_num_ranks; dst += kTPB) {
          const auto route_ready_offset =
              reinterpret_cast<uintptr_t>(
                  workspace.get_route_ready_count_ptr()) -
              reinterpret_cast<uintptr_t>(route_workspace);
          auto* remote_route_ready = reinterpret_cast<uint64_t*>(
              static_cast<uintptr_t>(route_buffer_ptrs[dst]) +
              route_ready_offset);

          deep_gemm::ptx::atomic_add_rel_sys(remote_route_ready, 1);
        }
        __syncthreads();
        if (tid < kWarpSize) ROUTE_IKET_MARK(RouteMetadataPublished);
      }
      if (tid == 0)
        *workspace.get_route_prepare_counter_ptr() = 0;
    }
  }

}

}  // namespace

struct GateTopkAllPushKernel {
  static void run(
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView counters,
      const tvm::ffi::TensorView expert_map,
      const tvm::ffi::TensorView valid_token_count,
      bool renormalize,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_sf,
      tvm::ffi::Optional<tvm::ffi::TensorView> w_stage,
      tvm::ffi::Optional<tvm::ffi::TensorView> route_workspace,
      tvm::ffi::Optional<tvm::ffi::TensorView> route_buffer_ptrs,
      int route_rank,
      int route_num_ranks,
      int route_bucket,
      int route_num_sms,
      bool route_rank_ready,
      bool perfect_balance,
      bool defer_route_dependent_launch,
      bool all_push, int reset_rows,
      tvm::ffi::Optional<tvm::ffi::TensorView> reset_timings,
      tvm::ffi::Optional<tvm::ffi::TensorView> ahead_workspace,
      tvm::ffi::Optional<tvm::ffi::TensorView> ahead_peers) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto num_experts = SymbolicSize{"E"};
    auto hidden_size = SymbolicSize{"H"};
    auto device_ref = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto index_dtype_ = SymbolicDType{};
    auto map_dtype_ = SymbolicDType{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(topk_weights);
    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t, int64_t>(index_dtype_)
        .verify(topk_indices);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(data_dtype_)
        .verify(hidden);
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(data_dtype_)
        .verify(gate_weight);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t, int64_t>(map_dtype_)
        .verify(expert_map);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int64_t>()
        .verify(valid_token_count);
    {
      const auto dt = data_dtype_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "hidden/gate_weight must be bf16");
    }

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    const auto topk_value = static_cast<int>(topk.unwrap());
    const char* prefix_env = std::getenv("DG_MEGAMOE_ALL_PUSH_PRODUCER_PREFIX");
    RuntimeCheck(!prefix_env || std::string(prefix_env) == "0" || std::string(prefix_env) == "1",
                 "DG_MEGAMOE_ALL_PUSH_PRODUCER_PREFIX must be 0 or 1");
    const char* wide_env = std::getenv("DG_MEGAMOE_ALL_PUSH_WIDE");
    const char* sf_vector_env = std::getenv("DG_MEGAMOE_ALL_PUSH_SF_VECTOR");
    RuntimeCheck(!sf_vector_env || std::string(sf_vector_env) == "0" ||
                 std::string(sf_vector_env) == "1",
                 "DG_MEGAMOE_ALL_PUSH_SF_VECTOR must be 0 or 1");
    const bool sf_vector = all_push && sf_vector_env && std::string(sf_vector_env) == "1" &&
        wide_env && std::atoi(wide_env) != 0 &&
        static_cast<float>(rows) * route_num_ranks * topk_value / experts > 32.5f;
    const char* priority_env = std::getenv("DG_MEGAMOE_ALL_PUSH_CACHE_PRIORITY");
    RuntimeCheck(!priority_env || std::string(priority_env) == "0" ||
                 std::string(priority_env) == "1" || std::string(priority_env) == "2" ||
                 std::string(priority_env) == "3" || std::string(priority_env) == "4",
                 "DG_MEGAMOE_ALL_PUSH_CACHE_PRIORITY must be 0, 1, 2, 3 or 4");
    const bool prepare_priority = all_push && priority_env && std::string(priority_env) == "3" &&
        wide_env && std::atoi(wide_env) != 0 &&
        static_cast<float>(rows) * route_num_ranks * topk_value / experts > 32.5f;
    const bool producer_prefix = all_push && prefix_env && std::string(prefix_env) == "1" &&
        wide_env && std::atoi(wide_env) != 0 &&
        static_cast<float>(rows) * route_num_ranks * topk_value / experts > 32.5f;
    RuntimeCheck(!producer_prefix || hsize == 4096, "Producer prefix requires hidden_size=4096");
    RuntimeCheck(!prepare_priority || !producer_prefix,
                 "Prepared priority keys require producer prefix disabled");
    RuntimeCheck(!sf_vector || (!prepare_priority && !producer_prefix && renormalize &&
                 route_rank_ready && ((experts == 128 && topk_value == 8) ||
                                      (experts == 512 && topk_value == 10))),
                 "Vector SF requires supported rank-ready routing without prefix or prepared keys");
    if (rows == 0) return;
    RuntimeCheck(hsize % (kKChunk * (kTPB / kWarpSize)) == 0, "hidden_size must be divisible by 512");
    RuntimeCheck(experts % 8 == 0 && experts <= kMaxExperts,
                 "num_experts must be a multiple of 8 and <= 512");
    RuntimeCheck(topk_value <= 32, "topk must be <= 32");
    RuntimeCheck(expert_map.size(0) == 0 || expert_map.size(0) == experts,
                 "expert_map must be empty or have num_experts entries, got ",
                 expert_map.size(0), " vs ", experts);
    RuntimeCheck(valid_token_count.size(0) == 0 ||
                     valid_token_count.size(0) == 1 ||
                     valid_token_count.size(0) == rows + 1,
                 "valid-token metadata must be empty, scalar, or [total, row map], got ",
                 valid_token_count.size(0));

    const bool prepare_routes = route_workspace.has_value();
    RouteLayout route_layout{}; // unused when route preparation is disabled
#ifdef MEGAMOE_RECYCLE_AHEAD
    RuntimeCheck(ahead_workspace.has_value() && ahead_peers.has_value() && route_workspace.has_value(),
                 "ahead recycling requires a distinct equal-layout arena");
    TensorMatcher({route_workspace.value().size(0)}).with_device<kDLCUDA>(device_ref)
        .with_dtype<int8_t>().verify(ahead_workspace.value());
    TensorMatcher({route_num_ranks}).with_device<kDLCUDA>(device_ref)
        .with_dtype<int64_t>().verify(ahead_peers.value());
    RuntimeCheck(ahead_workspace.value().data_ptr() != route_workspace.value().data_ptr(),
                 "ahead recycling must use a different arena");
#else
    RuntimeCheck(!ahead_workspace.has_value() && !ahead_peers.has_value(),
                 "ahead arena requires ahead recycling module");
#endif
#ifdef MEGAMOE_RECYCLE
    RuntimeCheck(all_push && reset_rows >= rows && reset_rows <= route_bucket,
                 "recycling requires an all-push reset prefix covering every row");
#else
    RuntimeCheck(reset_rows == 0, "reset prefix requires recycling module");
#endif
    uint64_t* reset_timings_ptr = nullptr;
    if (reset_timings.has_value()) {
      RuntimeCheck(reset_rows > 0, "reset trace requires recycling");
      TensorMatcher({-1}).with_device<kDLCUDA>(device_ref).with_dtype<int64_t>()
          .verify(reset_timings.value());
      RuntimeCheck(reset_timings.value().size(0) >= 24 + 3 * rows + route_num_sms + 16,
                   "reset trace buffer is too small");
      reset_timings_ptr = static_cast<uint64_t*>(reset_timings.value().data_ptr());
    }
    RuntimeCheck(!all_push || (renormalize && ((experts == 128 && topk_value == 8) || (experts == 512 && topk_value == 10))),
                 "SF-producer all-push requires the CTA selector");
    RuntimeCheck(!all_push || (prepare_routes && route_rank_ready && x_q.has_value() && x_sf.has_value()),
                 "all-push requires rank-ready metadata and fused quantization");
    RuntimeCheck(
        prepare_routes == route_buffer_ptrs.has_value(),
        "route_workspace and route_buffer_ptrs must be provided together");
    RuntimeCheck(prepare_routes || !route_rank_ready,
                 "rank-ready publication requires route preparation");
    RuntimeCheck(prepare_routes || !perfect_balance,
                 "perfect balance requires route preparation");
    RuntimeCheck(prepare_routes || !defer_route_dependent_launch,
                 "deferred route dependent launch requires route preparation");
    RuntimeCheck(!perfect_balance ||
                     (experts > 0 && (experts & (experts - 1)) == 0),
                 "perfect balance requires a power-of-two expert count");
    RuntimeCheck(!perfect_balance ||
                     static_cast<int64_t>(route_num_ranks) * rows * topk_value <=
                         std::numeric_limits<int>::max(),
                 "perfect balance route sequence exceeds int32 range");
    if (prepare_routes) {
      TensorMatcher({-1})
          .with_device<kDLCUDA>(device_ref)
          .with_dtype<int8_t>()
          .verify(route_workspace.value());
      TensorMatcher({route_num_ranks})
          .with_device<kDLCUDA>(device_ref)
          .with_dtype<int64_t>()
          .verify(route_buffer_ptrs.value());
      RuntimeCheck(route_num_ranks > 0 && experts % route_num_ranks == 0,
                   "route num_ranks must positively divide num_experts");
      RuntimeCheck(route_rank >= 0 && route_rank < route_num_ranks,
                   "route rank is outside the symmetric rank group");
      RuntimeCheck(route_bucket >= rows,
                   "route bucket is smaller than the routed token count");
      RuntimeCheck(route_num_sms > 0,
                   "route num_sms must be positive");
      const auto workspace = deep_gemm::layout::Workspace(
          nullptr, route_num_ranks, experts, route_bucket, topk_value);
      RuntimeCheck(!prepare_priority ||
                   uint64_t(workspace.num_max_pool_tokens) >= 4ull * route_bucket,
                   "Prepared priority key banks exceed gather-index capacity");
      route_layout = {workspace.num_experts_per_rank,
                      workspace.num_max_recv_tokens_per_expert,
                      workspace.num_max_pool_tokens, workspace.num_max_pool_blocks};
      RuntimeCheck(
          route_workspace.value().size(0) >=
              static_cast<int64_t>(workspace.get_num_bytes()),
          "route workspace is smaller than the MegaMoE metadata region");
    }

    RuntimeCheck(!prepare_routes || !route_rank_ready ||
        static_cast<uint64_t>(route_num_ranks) * route_bucket * topk_value <= 0xffffffffu,
        "Dense source descriptor exceeds uint32");
    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    const int num_mtiles = (rows + kTokensPerBlock - 1) / kTokensPerBlock;
    const int num_expert_splits = experts / 8;
    constexpr int kNumWarps = kTPB / kWarpSize;
    constexpr int kSmemBytes =
        kNumWarps * kStages * (kTokensPerBlock + 8) * kRowPad * 2 +
        kNumWarps * kTokensPerBlock * 8 * static_cast<int>(sizeof(float));
    static bool attr_set = false;
    if (!attr_set) {
      const auto status = cudaFuncSetAttribute(gate_topk_fused_tc_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      RuntimeCheck(status == cudaSuccess, "all-push router shared-memory setup failed: ",
                   cudaGetErrorString(status));
      attr_set = true;
    }
    uint8_t* x_q_ptr = nullptr;
    uint8_t* x_sf_ptr = nullptr;
    float* w_stage_ptr = nullptr;
    if (x_q.has_value()) {
      RuntimeCheck(x_sf.has_value(), "x_sf required with x_q");
      RuntimeCheck(hsize % (kQuantSplits * kKChunk) == 0,
                   "fused quant needs hidden_size % 1024 == 0");
      x_q_ptr = static_cast<uint8_t*>(x_q.value().data_ptr());
      x_sf_ptr = static_cast<uint8_t*>(x_sf.value().data_ptr());
    }
    RuntimeCheck(!sf_vector || (prepare_routes && x_sf_ptr && hsize % 256 == 0 &&
                 reinterpret_cast<uintptr_t>(x_sf_ptr) % 8 == 0 &&
                 reinterpret_cast<uintptr_t>(route_workspace.value().data_ptr()) % 8 == 0),
                 "Vector SF requires aligned source and symmetric workspace");
    if (w_stage.has_value())
      w_stage_ptr = static_cast<float*>(w_stage.value().data_ptr());
    gate_topk_fused_tc_kernel<<<num_mtiles * num_expert_splits, kTPB, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const __nv_bfloat16*>(gate_w_ptr(gate_weight)),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(topk_indices.data_ptr()),
        static_cast<float*>(partials.data_ptr()),
        static_cast<int*>(counters.data_ptr()),
        x_q_ptr, x_sf_ptr,
        rows, experts, hsize, num_expert_splits, topk_value, renormalize);
    const auto* count_ptr = valid_token_count.size(0) == 0
                                ? nullptr
                                : static_cast<const int64_t*>(
                                      valid_token_count.data_ptr());
    const int count_size = static_cast<int>(valid_token_count.size(0));
    const bool use_int32_indices = index_dtype_.unwrap().bits == 32;
    const bool use_int32_map = map_dtype_.unwrap().bits == 32;
    auto launch_select = [&](auto index_tag, auto map_tag) {
      using IndexT = typename decltype(index_tag)::type;
      using MapT = typename decltype(map_tag)::type;
      const auto* map_ptr = expert_map.size(0) == 0
                                ? nullptr
                                : static_cast<const MapT*>(expert_map.data_ptr());
      auto* route_workspace_ptr = prepare_routes
          ? route_workspace.value().data_ptr()
          : nullptr;
      const auto* route_buffer_ptrs_ptr = prepare_routes
          ? static_cast<const int64_t*>(route_buffer_ptrs.value().data_ptr())
          : nullptr;
      const auto launch = [&](auto prepare_tag, auto rank_ready_tag,
                              auto perfect_tag, auto renormalize_tag, auto compact_tag) {
        constexpr bool kAllPush = decltype(compact_tag)::value;
        constexpr bool kPrepareRoutes = decltype(prepare_tag)::value;
        constexpr bool kRankReady = decltype(rank_ready_tag)::value;
        constexpr bool kPerfect = decltype(perfect_tag)::value;
        constexpr bool kRenormalize = decltype(renormalize_tag)::value;
        if constexpr (kRenormalize) {
          const auto launch_cta = [&](auto experts_tag, auto topk_tag) {
            constexpr int kExperts = decltype(experts_tag)::value;
            constexpr int kTopK = decltype(topk_tag)::value;
            const auto launch_prefix = [&](auto prefix_tag, auto priority_tag, auto sf_vector_tag) {
              const auto kernel = gate_topk_select_cta_kernel<
                  kExperts, kTopK, IndexT, MapT, kPrepareRoutes, kRankReady, kPerfect,
                  kAllPush, decltype(prefix_tag)::value, decltype(priority_tag)::value,
                  decltype(sf_vector_tag)::value>;
              LaunchKernel(rows + (kAllPush && reset_rows ? deep_gemm::kMegaMoERecycleCTAs : 0),
                         kTPB, stream).with_attr(true)(kernel,
                static_cast<const float*>(partials.data_ptr()),
                static_cast<float*>(topk_weights.data_ptr()),
                static_cast<IndexT*>(topk_indices.data_ptr()),
                map_ptr, count_ptr, count_size, w_stage_ptr,
                route_workspace_ptr, route_layout, route_buffer_ptrs_ptr,
                route_rank, route_num_ranks, route_bucket, route_num_sms,
                rows, defer_route_dependent_launch, x_sf_ptr, hsize, reset_rows, reset_timings_ptr,
                ahead_workspace.has_value() ? ahead_workspace.value().data_ptr() : nullptr,
                ahead_peers.has_value() ? static_cast<const int64_t*>(ahead_peers.value().data_ptr()) : nullptr,
                x_q_ptr);
            };
            if constexpr (kAllPush) {
              if (producer_prefix) {
                launch_prefix(std::true_type{}, std::false_type{}, std::false_type{});
                return;
              }
              if (prepare_priority) {
                launch_prefix(std::false_type{}, std::true_type{}, std::false_type{});
                return;
              }
              if constexpr (kRankReady) {
                if (sf_vector) {
                  launch_prefix(std::false_type{}, std::false_type{}, std::true_type{});
                  return;
                }
              }
            }
            launch_prefix(std::false_type{}, std::false_type{}, std::false_type{});
          };
          if (experts == 128 && topk_value == 8) {
            launch_cta(std::integral_constant<int, 128>{}, std::integral_constant<int, 8>{});
            return;
          }
          if (experts == 512 && topk_value == 10) {
            launch_cta(std::integral_constant<int, 512>{}, std::integral_constant<int, 10>{});
            return;
          }
        }
        const auto kernel =
            gate_topk_select_kernel<IndexT, MapT, kPrepareRoutes, kRankReady,
                                    kPerfect, kRenormalize, kAllPush>;
        LaunchKernel(num_mtiles,
                     std::min(kTokensPerBlock, rows) * kWarpSize,
                     stream).with_attr(true)(kernel,
            static_cast<const float*>(partials.data_ptr()),
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<IndexT*>(topk_indices.data_ptr()),
            map_ptr, count_ptr, count_size, w_stage_ptr,
            route_workspace_ptr, route_buffer_ptrs_ptr,
            route_rank, route_num_ranks, route_bucket, route_num_sms,
            rows, experts, topk_value, defer_route_dependent_launch);
      };
      const auto launch_policy = [&](auto prepare_tag, auto rank_ready_tag,
                                     auto perfect_tag) {
        const auto launch_renormalize = [&](auto compact_tag) {
          if (renormalize)
            launch(prepare_tag, rank_ready_tag, perfect_tag, std::true_type{}, compact_tag);
          else
            launch(prepare_tag, rank_ready_tag, perfect_tag, std::false_type{}, compact_tag);
        };
        if constexpr (decltype(prepare_tag)::value && decltype(rank_ready_tag)::value) {
          if (all_push) {
            launch_renormalize(std::true_type{});
            return;
          }
        }
        launch_renormalize(std::false_type{});
      };
      if (prepare_routes) {
        if (route_rank_ready) {
          if (perfect_balance)
            launch_policy(std::true_type{}, std::true_type{}, std::true_type{});
          else
            launch_policy(std::true_type{}, std::true_type{}, std::false_type{});
        } else {
          if (perfect_balance)
            launch_policy(std::true_type{}, std::false_type{}, std::true_type{});
          else
            launch_policy(std::true_type{}, std::false_type{}, std::false_type{});
        }
      } else {
        launch_policy(std::false_type{}, std::false_type{}, std::false_type{});
      }
    };
    auto launch_index_types = [&]() {
      if (use_int32_indices && use_int32_map) {
        launch_select(TypeTag<int32_t>{}, TypeTag<int32_t>{});
      } else if (use_int32_indices && !use_int32_map) {
        launch_select(TypeTag<int32_t>{}, TypeTag<int64_t>{});
      } else if (!use_int32_indices && use_int32_map) {
        launch_select(TypeTag<int64_t>{}, TypeTag<int32_t>{});
      } else {
        launch_select(TypeTag<int64_t>{}, TypeTag<int64_t>{});
      }
    };
    launch_index_types();
  }

 private:
  static const void* gate_w_ptr(const tvm::ffi::TensorView& t) {
    return t.data_ptr();
  }
};
