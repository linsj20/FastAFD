// Per-128-group UE8M0 FP8 quantization + Qwen3.5 shared-expert scalar gate.
//
// The post-attention add/RMSNorm deliberately remains in FlashInfer. Its
// H=4096 specialization uses a 512-thread/16-warp reduction, and changing that
// reduction order changes decode numerics enough to break model alignment.
// This kernel fuses only consumers of the already-rounded BF16 norm output:
// the shared-expert FP8 input, scalar gate sigmoid, and peer-lane release.

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarpSize * kWarps;
constexpr int kGroup = 128;

// One CTA per token. Each warp owns kGroupsPerWarp 128-value groups and each
// lane owns four contiguous BF16 values in every group.
template <int kGroupsPerWarp>
__launch_bounds__(kThreads) __global__ void fp8_quant_shared_gate_kernel(
    const __nv_bfloat16* __restrict__ input,              // [m, H]
    const __nv_bfloat16* __restrict__ shared_gate_weight, // [H]
    uint8_t* __restrict__ out_q,                          // [m, H] e4m3 bytes
    uint32_t* __restrict__ out_s,                         // packed UE8M0
    __nv_bfloat16* __restrict__ shared_gate_scale,        // [m, 1]
    uint32_t* __restrict__ completion_counter,
    int64_t* __restrict__ release_turn,
    const int64_t next_turn,
    const int tma_aligned_mn) {
  constexpr int H = kGroup * kWarps * kGroupsPerWarp;
  const int row = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const __nv_bfloat16* x_row = input + static_cast<int64_t>(row) * H;
  uint8_t* q_row = out_q + static_cast<int64_t>(row) * H;

  float shared_gate_sum = 0.f;
  #pragma unroll
  for (int j = 0; j < kGroupsPerWarp; ++j) {
    const int g_k = warp + j * kWarps;
    const int base = g_k * kGroup + lane * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x_row + base);
    const uint2 gwv = *reinterpret_cast<const uint2*>(
        shared_gate_weight + base);
    const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* gwh = reinterpret_cast<const __nv_bfloat16*>(&gwv);
    float values[4];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float value = __bfloat162float(xh[i]);
      values[i] = value;
      amax = fmaxf(amax, fabsf(value));
      shared_gate_sum += value * __bfloat162float(gwh[i]);
    }

    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    float y_s = fmaxf(amax / 448.0f, 1e-10f);
    const uint32_t bits = __float_as_uint(y_s);
    const uint8_t exp_byte = static_cast<uint8_t>(
        ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));
    const float inv_y =
        1.0f / __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);

    __nv_fp8x2_storage_t out2[2];
    #pragma unroll
    for (int i = 0; i < 4; i += 2)
      out2[i / 2] = __nv_cvt_float2_to_fp8x2(
          make_float2(values[i] * inv_y, values[i + 1] * inv_y),
          __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>(q_row + base) =
        *reinterpret_cast<const uint32_t*>(out2);

    if (lane == 0) {
      const int pack_idx = g_k / 4;
      const int pos = g_k % 4;
      const int64_t out_idx =
          static_cast<int64_t>(pack_idx) * tma_aligned_mn + row;
      reinterpret_cast<uint8_t*>(out_s)[out_idx * 4 + pos] = exp_byte;
    }
  }

  #pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1)
    shared_gate_sum += __shfl_xor_sync(
        0xffffffffu, shared_gate_sum, off);
  __shared__ float warp_gate[kWarps];
  if (lane == 0) warp_gate[warp] = shared_gate_sum;
  __syncthreads();
  float gate_value = 0.f;
  if (threadIdx.x == 0) {
    float total_gate = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w)
      total_gate += warp_gate[w];
    // Preserve the two exposed BF16 boundaries in the established path:
    // the gate GEMM first returns BF16 logits, then sigmoid returns BF16.
    const __nv_bfloat16 gate_logit = __float2bfloat16_rn(total_gate);
    gate_value = __bfloat162float(gate_logit);
    shared_gate_scale[row] = gate_logit;
  }

  // The peer shared expert needs only FP8 and packed scales. Release it as soon
  // as those inputs and the intermediate BF16 logit are globally visible;
  // finish the consumer-only sigmoid while the peer starts gate/up.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t ticket = atomicAdd(completion_counter, 1u);
    if (ticket + 1u == static_cast<uint32_t>(gridDim.x)) {
      atomicExch(completion_counter, 0u);
      asm volatile(
          "st.release.gpu.global.u64 [%0], %1;"
          :
          : "l"(release_turn), "l"(next_turn));
    }
    shared_gate_scale[row] = __float2bfloat16_rn(
        1.0f / (1.0f + expf(-gate_value)));
  }
}

}  // namespace

struct Fp8QuantSharedGateKernel {
  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView shared_gate_weight,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView shared_gate_scale,
      const tvm::ffi::TensorView completion_counter,
      const tvm::ffi::TensorView release_turn,
      int64_t next_turn) {
    using namespace host;
    auto m = SymbolicSize{"M"};
    auto Hd = SymbolicSize{"H"};
    auto device_ref = SymbolicDevice{};
    auto bf16 = SymbolicDType{};

    TensorMatcher({m, Hd})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(bf16)
        .verify(input);
    TensorMatcher({1, Hd})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(bf16)
        .verify(shared_gate_weight);
    TensorMatcher({m, 1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(bf16)
        .verify(shared_gate_scale);
    TensorMatcher({1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(completion_counter);
    TensorMatcher({1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int64_t>()
        .verify(release_turn);
    const auto dt = bf16.unwrap();
    RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                 "input and shared gate weight must be bf16");

    const int rows = static_cast<int>(m.unwrap());
    const int H = static_cast<int>(Hd.unwrap());
    if (rows == 0) return;
    RuntimeCheck(H % (kGroup * kWarps) == 0,
                 "fp8_quant_shared_gate needs H % 1024 == 0");
    RuntimeCheck(H <= kGroup * kWarps * 4,
                 "fp8_quant_shared_gate needs H <= 4096");
    RuntimeCheck(next_turn >= 0,
                 "fp8_quant_shared_gate needs a nonnegative turn");
    const int tma_aligned_mn = (rows + 3) / 4 * 4;
    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());

#define LAUNCH_FP8_QUANT_SHARED_GATE(groups_per_warp)                         \
    fp8_quant_shared_gate_kernel<groups_per_warp>                             \
        <<<rows, kThreads, 0, stream>>>(                                      \
            static_cast<const __nv_bfloat16*>(input.data_ptr()),             \
            static_cast<const __nv_bfloat16*>(                               \
                shared_gate_weight.data_ptr()),                              \
            static_cast<uint8_t*>(out_q.data_ptr()),                         \
            static_cast<uint32_t*>(out_s.data_ptr()),                        \
            static_cast<__nv_bfloat16*>(shared_gate_scale.data_ptr()),       \
            static_cast<uint32_t*>(completion_counter.data_ptr()),           \
            static_cast<int64_t*>(release_turn.data_ptr()),                   \
            next_turn, tma_aligned_mn)
    switch (H / (kGroup * kWarps)) {
      case 1: LAUNCH_FP8_QUANT_SHARED_GATE(1); break;
      case 2: LAUNCH_FP8_QUANT_SHARED_GATE(2); break;
      case 3: LAUNCH_FP8_QUANT_SHARED_GATE(3); break;
      case 4: LAUNCH_FP8_QUANT_SHARED_GATE(4); break;
      default:
        RuntimeCheck(false, "unsupported fp8_quant_shared_gate hidden size: ", H);
    }
#undef LAUNCH_FP8_QUANT_SHARED_GATE
  }
};
