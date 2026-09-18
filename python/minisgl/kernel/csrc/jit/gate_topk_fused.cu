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

namespace {

inline constexpr int kWarpSize = 32;
inline constexpr int kTokensPerBlock = 16;  // MMA M
inline constexpr int kKChunk = 64;          // K tile per stage
inline constexpr int kRowPad = 72;         // chunk + 8 pad
inline constexpr int kMaxExperts = 512;     // one token-tile grid owns all experts
inline constexpr int kTPB = 256;            // 8 warps

template <typename T>
struct TypeTag {
  using type = T;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)),
               "l"(src));
}

__device__ __forceinline__ void cp_async_16_zfill(void* dst, const void* src) {
  // src-size 0 => zero-fill the 16 bytes
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(smem_u32(dst)),
               "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Release-acquire ticket replaces two full __threadfence()s (expensive on
// GB200): syncthreads collects the block's writes, the acq_rel increment
// publishes them and acquires the other blocks' writes for the reader.
__device__ __forceinline__ int atom_add_acqrel_gpu(int* p, int v) {
  int old;
  asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

// Blackwell warp reduction used by TRT-LLM's MoE top-k. Encode the float as
// an unsigned monotonic key, then resolve equal values by the same lower-lane
// rule as the original shuffle tree. A pair of REDUX instructions replaces
// five value shuffles, five lane shuffles, and their comparisons.
__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  uint32_t bits = __float_as_uint(value);
  // CUDA float comparisons consider -0 and +0 equal. Canonicalize them so the
  // packed reduction preserves the selector's existing tie behavior.
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  const uint32_t mask =
      static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u;
  return bits ^ mask;
}

__device__ __forceinline__ uint32_t warp_redux_max_u32(uint32_t value) {
  uint32_t result;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
               : "=r"(result) : "r"(value));
  return result;
}

inline constexpr int kQuantSplits = 16;
inline constexpr int kStages = 4;   // cp.async pipeline depth (latency-bound kernel)

__launch_bounds__(kTPB) __global__ void gate_topk_fused_tc_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate_w,  // [num_experts, hidden_size]
    float* __restrict__ topk_weights,          // [num_tokens, topk]
    int32_t* __restrict__ topk_ids,            // [num_tokens, topk]
    float* __restrict__ logits_gmem,           // [mtiles, 16, kMaxExperts]
    int* __restrict__ counters,                // [mtiles], zero-initialized
    uint8_t* __restrict__ x_q,                 // optional [num_tokens, hidden] e4m3
    uint8_t* __restrict__ x_sf,                // optional [num_tokens, hidden/32] ue8m0
    const int num_tokens, const int num_experts, const int hidden_size,
    const int num_expert_splits,
    const int topk, const bool renormalize) {
  // v3 layout: grid = mtiles x num_expert_splits blocks. A block owns 8 experts with
  // the FULL K (no cross-block reduction); its 8 warps split K and run
  // fully independent per-warp cp.async pipelines (cp.async groups are
  // per-thread state, so the main loop needs no __syncthreads at all),
  // then reduce in shared memory.  The last block of each token tile runs
  // the top-k over the assembled [16, num_experts] logits.
  constexpr int kNumWarps = kTPB / kWarpSize;          // 8
  constexpr int kWarpABytes = kStages * kTokensPerBlock * kRowPad * 2;
  constexpr int kWarpBBytes = kStages * 8 * kRowPad * 2;
  extern __shared__ char smem_raw[];
  // per-warp staging: [warp][A stages][16][kRowPad] + [warp][B stages][8][kRowPad]
  auto a_smem = [&](int w, int stage, int row, int col) -> __nv_bfloat16& {
    return reinterpret_cast<__nv_bfloat16*>(smem_raw + w * (kWarpABytes + kWarpBBytes))
        [(stage * kTokensPerBlock + row) * kRowPad + col];
  };
  auto b_smem = [&](int w, int stage, int row, int col) -> __nv_bfloat16& {
    return reinterpret_cast<__nv_bfloat16*>(smem_raw + w * (kWarpABytes + kWarpBBytes) + kWarpABytes)
        [(stage * 8 + row) * kRowPad + col];
  };
  // reduce area after the staging region
  float* red_smem = reinterpret_cast<float*>(
      smem_raw + kNumWarps * (kWarpABytes + kWarpBBytes));  // [kNumWarps][16][8]

  const int mtile = blockIdx.x / num_expert_splits;
  const int nsplit = blockIdx.x % num_expert_splits;
  const int m0 = mtile * kTokensPerBlock;
  const int n_base = nsplit * 8;  // this block's 8 experts
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  // Per-warp K slice
  const int k_slice = hidden_size / kNumWarps;
  const int k_begin = warp * k_slice;

  // Per-warp async loader: A 16x64 (8 chunks of 16B per lane covers 128
  // 16B-chunks with 32 lanes x4) + B 8x64 (64 chunks, 32 lanes x2)
  constexpr int kChunks16B = kKChunk / 8;  // 16B chunks per row per stage
  const auto load_stage = [&](int stage, int kc) {
    #pragma unroll
    for (int i = 0; i < kTokensPerBlock * kChunks16B / kWarpSize; ++i) {
      const int idx = lane + i * kWarpSize;
      const int row = idx / kChunks16B, chunk = idx % kChunks16B;
      void* dst = &a_smem(warp, stage, row, chunk * 8);
      if (m0 + row < num_tokens)
        cp_async_16(dst, hidden + static_cast<size_t>(m0 + row) * hidden_size + kc + chunk * 8);
      else
        cp_async_16_zfill(dst, hidden);
    }
    #pragma unroll
    for (int i = 0; i < 8 * kChunks16B / kWarpSize; ++i) {
      const int idx = lane + i * kWarpSize;
      const int row = idx / kChunks16B, chunk = idx % kChunks16B;
      cp_async_16(&b_smem(warp, stage, row, chunk * 8),
                  gate_w + static_cast<size_t>(n_base + row) * hidden_size + kc + chunk * 8);
    }
    cp_async_commit();
  };

  float acc[4] = {};
  const int num_chunks = k_slice / kKChunk;

  // Each warp's two K halves have distinct logical owners. Fold those owners
  // onto available expert blocks for small grids; retain the original owners
  // for 128+ experts. Every hidden value still has exactly one quantizer.
  const bool full_quant_grid = num_expert_splits >= kQuantSplits;
  const int first_owner = full_quant_grid ? warp * 2 : (warp * 2) % num_expert_splits;
  const int second_owner = full_quant_grid ? warp * 2 + 1 : (warp * 2 + 1) % num_expert_splits;
  const bool quant_first = x_q != nullptr && nsplit == first_owner;
  const bool quant_second = x_q != nullptr && nsplit == second_owner;
  const auto quant_chunk = [&](int stage, int c) {
    const int row = lane / 2, grp = lane % 2;
    if (m0 + row >= num_tokens) return;
    const int k0 = k_begin + c * kKChunk + grp * 32;
    float vals[32];
    #pragma unroll
    for (int j = 0; j < 32; ++j)
      vals[j] = __bfloat162float(a_smem(warp, stage, row, grp * 32 + j));
    float amax = 1e-4f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(vals[j]));
    const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
    const uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
    x_sf[static_cast<size_t>(m0 + row) * (hidden_size / 32) + k0 / 32] =
        static_cast<uint8_t>(e);
    const float inv_scale = __uint_as_float((254u - e) << 23);
    uint32_t packed[8];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[j] = __nv_cvt_float2_to_fp8x2(
          make_float2(vals[2 * j] * inv_scale, vals[2 * j + 1] * inv_scale),
          __NV_SATFINITE, __NV_E4M3);
    }
    auto dst = reinterpret_cast<uint4*>(x_q + static_cast<size_t>(m0 + row) * hidden_size + k0);
    dst[0] = *reinterpret_cast<uint4*>(packed);
    dst[1] = *reinterpret_cast<uint4*>(packed + 4);
  };
  // Preserve the wait-group distance even when K cannot fill the pipeline.
  // Empty groups keep the first consumed copy older than kStages - 1 groups.
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < num_chunks)
      load_stage(st, k_begin + st * kKChunk);
    else
      cp_async_commit();
  }
  for (int c = 0; c < num_chunks; ++c) {
    const int cur = c % kStages;
    if (c + kStages - 1 < num_chunks)
      load_stage((c + kStages - 1) % kStages, k_begin + (c + kStages - 1) * kKChunk);
    else
      cp_async_commit();
    cp_async_wait<kStages - 1>();
    __syncwarp();

    #pragma unroll
    for (int ks = 0; ks < kKChunk / 16; ++ks) {
      uint32_t a0, a1, a2, a3;
      {
        const int grp = lane / 8;
        const int row = (grp % 2) * 8 + (lane % 8);
        const int col = ks * 16 + (grp / 2) * 8;
        const uint32_t addr = smem_u32(&a_smem(warp, cur, row, col));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "r"(addr));
      }
      uint32_t b0, b1;
      {
        const int r = lane % 8;
        const int half = (lane / 8) % 2;
        const uint32_t addr = smem_u32(&b_smem(warp, cur, r, ks * 16 + half * 8));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
      }
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (c < num_chunks / 2 ? quant_first : quant_second)
      quant_chunk(cur, c);
    __syncwarp();
  }

  // ---- intra-block K reduction in shared memory --------------------------
  {
    const int row = lane / 4;
    const int col = (lane % 4) * 2;
    float* mine = red_smem + warp * kTokensPerBlock * 8;
    mine[row * 8 + col] = acc[0];
    mine[row * 8 + col + 1] = acc[1];
    mine[(row + 8) * 8 + col] = acc[2];
    mine[(row + 8) * 8 + col + 1] = acc[3];
  }
  __syncthreads();
  // 16x8 = 128 outputs; thread t sums over the 8 warps
  float* block_logits = logits_gmem +
      (static_cast<size_t>(mtile) * kTokensPerBlock) * kMaxExperts;
  if (tid < kTokensPerBlock * 8) {
    float v = 0.f;
    #pragma unroll
    for (int w = 0; w < kNumWarps; ++w)
      v += red_smem[w * kTokensPerBlock * 8 + tid];
    block_logits[(tid / 8) * kMaxExperts + n_base + (tid % 8)] = v;
  }

  // Let the dependent selector become resident while the tail of the gate
  // grid drains. Its grid-dependency wait below preserves the full producer
  // barrier before any logit is consumed.
  device::PDL::launch<true>();
}

// Second kernel: top-k over the assembled logits (one block per token tile,
// one warp per token). PDL orders the completed local inputs before dispatch.
template <typename IndexT, typename MapT, bool kPrepareMegaMoERoutes,
          bool kRankReadyRoutePublish, bool kPerfectBalance,
          bool kRenormalize, bool kCompactPush>
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
            if constexpr (kCompactPush) {
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
    if constexpr (kCompactPush) {
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
          bool kRankReadyRoutePublish, bool kPerfectBalance, bool kCompactPush>
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
              if constexpr (kCompactPush) {
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
    if constexpr (kCompactPush) {
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
      bool compact_push) {
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
    RuntimeCheck(!compact_push || (prepare_routes && route_rank_ready && x_q.has_value() && x_sf.has_value()),
                 "compact payload push requires rank-ready metadata and fused quantization");
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
        constexpr bool kCompactPush = decltype(compact_tag)::value;
        constexpr bool kPrepareRoutes = decltype(prepare_tag)::value;
        constexpr bool kRankReady = decltype(rank_ready_tag)::value;
        constexpr bool kPerfect = decltype(perfect_tag)::value;
        constexpr bool kRenormalize = decltype(renormalize_tag)::value;
        if constexpr (kRenormalize) {
          const auto launch_cta = [&](auto experts_tag, auto topk_tag) {
            constexpr int kExperts = decltype(experts_tag)::value;
            constexpr int kTopK = decltype(topk_tag)::value;
            const auto kernel = gate_topk_select_cta_kernel<
                kExperts, kTopK, IndexT, MapT, kPrepareRoutes, kRankReady, kPerfect, kCompactPush>;
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
                                    kPerfect, kRenormalize, kCompactPush>;
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
          if (compact_push) {
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
