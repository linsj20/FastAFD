#pragma once

// Shared gate GEMM, quantization and selector primitives. Included by both
// routing implementations after their CUDA and MiniSGL dependencies.
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
#ifdef MEGAMOE_RECYCLE_EARLY_PDL
  // Reset CTAs need only the previous arena generation. Token selectors still
  // acquire this producer with PDL::wait before reading logits or quantization.
  device::PDL::launch<true>();
#endif
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
#ifndef MEGAMOE_RECYCLE_EARLY_PDL
  device::PDL::launch<true>();
#endif
}

}  // namespace
