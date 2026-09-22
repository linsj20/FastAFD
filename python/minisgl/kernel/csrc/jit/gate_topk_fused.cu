// Two-stage fused router shared by the routed-expert FFN paths:
// 1. Tensor-core gate GEMM and optional activation quantization, tiled over
//    tokens and experts.
// 2. Top-k selection publishes destination expert indices/counts before PDL.
//    Compact mode pushes scores too; MegaMoE independently pushes payload rows.
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
            DG_DEVICE_ASSERT(slot < workspace.num_max_recv_tokens_per_expert);
            const auto offset = reinterpret_cast<uintptr_t>(
                workspace.get_src_token_topk_idx_ptr(local_expert,
                  workspace.num_ranks == 1 ? 0 : workspace.metadata_slot, static_cast<uint32_t>(slot))) -
                reinterpret_cast<uintptr_t>(route_workspace);
            auto* remote_dst = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + offset);
            *remote_dst = static_cast<uint32_t>(route_rank) * route_bucket * topk +
                static_cast<uint32_t>(token_topk_idx);
            if constexpr (kAllPush) {
              const auto score_offset = reinterpret_cast<uintptr_t>(
                  workspace.get_score_inbox_ptr(route_rank) + token_topk_idx) -
                  reinterpret_cast<uintptr_t>(route_workspace);
              *reinterpret_cast<float*>(static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                                        score_offset) = w_out;
            }
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
template <int kExperts, int kTopK, typename IndexT, typename MapT, bool kPrepareMegaMoERoutes,
          bool kRankReadyRoutePublish, bool kPerfectBalance, bool kAllPush>
__launch_bounds__(kTPB) __global__ void gate_topk_select_cta_kernel(
    const float* __restrict__ logits_gmem,
    float* __restrict__ topk_weights,
    IndexT* __restrict__ topk_ids,
    const MapT* __restrict__ expert_map,
    const int64_t* __restrict__ valid_token_count,
    int valid_count_size,
    float* __restrict__ w_stage,
    void* __restrict__ route_workspace,
    const int64_t* __restrict__ route_buffer_ptrs,
    const int route_rank, const int route_num_ranks,
    const int route_bucket, const int route_num_sms,
    const int num_tokens, const bool defer_route_dependent_launch) {
  static_assert(kExperts <= kMaxExperts && kExperts % kWarpSize == 0);
  static_assert(kTopK > 0 && kTopK <= kWarpSize);
  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  device::PDL::wait<true>();

  const auto validity =
      minisgl::token_validity(token, num_tokens, valid_token_count, valid_count_size);
  const bool valid = validity.valid;

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
      float w = tid < kTopK ? __expf(my_logit - top0) : 0.f;
      float w_sum = w;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
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
        const float w_out = w / w_sum;
        topk_weights[token_topk_idx] = w_out;

        if (w_stage != nullptr)
          w_stage[token_topk_idx] = w_out;

        if constexpr (kPrepareMegaMoERoutes) {
          if (mapped_expert >= 0 && mapped_expert < kExperts) {
            const auto workspace = deep_gemm::layout::Workspace(
                route_workspace, route_num_ranks, kExperts,
                route_bucket, kTopK);
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
              DG_DEVICE_ASSERT(slot < workspace.num_max_recv_tokens_per_expert);
              const auto offset = reinterpret_cast<uintptr_t>(
                  workspace.get_src_token_topk_idx_ptr(local_expert,
                  workspace.num_ranks == 1 ? 0 : workspace.metadata_slot, static_cast<uint32_t>(slot))) -
                  reinterpret_cast<uintptr_t>(route_workspace);
              auto* remote_dst = reinterpret_cast<uint32_t*>(
                  static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) + offset);
              *remote_dst = static_cast<uint32_t>(route_rank) * route_bucket * kTopK +
                  static_cast<uint32_t>(token_topk_idx);
              if constexpr (kAllPush) {
                const auto score_offset = reinterpret_cast<uintptr_t>(
                    workspace.get_score_inbox_ptr(route_rank) + token_topk_idx) -
                    reinterpret_cast<uintptr_t>(route_workspace);
                *reinterpret_cast<float*>(static_cast<uintptr_t>(route_buffer_ptrs[dst_rank]) +
                                          score_offset) = w_out;
              }
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
  } else if (!valid && tid < kTopK) {
    const auto token_topk_idx = static_cast<size_t>(token) * kTopK + tid;
    topk_ids[token_topk_idx] = static_cast<IndexT>(-1);
    topk_weights[token_topk_idx] = 0.0f;
    if (w_stage != nullptr)
      w_stage[token_topk_idx] = 0.0f;
  }


  if constexpr (kPrepareMegaMoERoutes) {
    __syncthreads();
    if constexpr (kAllPush) {
      const auto input_workspace = deep_gemm::layout::Workspace(
          route_workspace, route_num_ranks, kExperts, route_bucket, kTopK);
      if (threadIdx.x == 0)
        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;"
                     :: "l"(input_workspace.get_genphase_flag_ptr(deep_gemm::layout::Workspace::CountsReady)),
                        "r"(static_cast<uint32_t>(1)) : "memory");
    }
    if (!defer_route_dependent_launch)
      device::PDL::launch<true>();
    const auto workspace = deep_gemm::layout::Workspace(
        route_workspace, route_num_ranks, kExperts, route_bucket, kTopK);
    __shared__ int is_last_block;
    if (tid == 0) {
      is_last_block = atom_add_acqrel_gpu(
          reinterpret_cast<int*>(workspace.get_route_prepare_counter_ptr()), 1) ==
          gridDim.x - 1;
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

struct GateTopkFusedKernel {
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
      bool all_push) {
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
      cudaFuncSetAttribute(gate_topk_fused_tc_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
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
            const auto kernel = gate_topk_select_cta_kernel<
                kExperts, kTopK, IndexT, MapT, kPrepareRoutes, kRankReady, kPerfect, kAllPush>;
            LaunchKernel(rows, kTPB, stream).with_attr(true)(kernel,
                static_cast<const float*>(partials.data_ptr()),
                static_cast<float*>(topk_weights.data_ptr()),
                static_cast<IndexT*>(topk_indices.data_ptr()),
                map_ptr, count_ptr, count_size, w_stage_ptr,
                route_workspace_ptr, route_buffer_ptrs_ptr,
                route_rank, route_num_ranks, route_bucket, route_num_sms,
                rows, defer_route_dependent_launch);
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
