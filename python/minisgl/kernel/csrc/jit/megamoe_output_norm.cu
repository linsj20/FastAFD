/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from FlashInfer FusedAddRMSNormKernel. Arithmetic, vector width,
// reduction order and launch geometry are retained; only the incoming data
// dependency is replaced by MegaMoE's explicit output publication.
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>

namespace {
using namespace flashinfer;
template <uint32_t VEC_SIZE, typename T, bool kTrace>
__global__ void megamoe_output_add_rmsnorm(T* __restrict__ input, T* __restrict__ residual,
                                      T* __restrict__ weight, const uint32_t d,
                                      const uint32_t stride_input, const uint32_t stride_residual,
                                      float weight_bias, float eps, const uint32_t* epoch_ptr,
                                      const uint32_t* ready_ptr, uint64_t* timings) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;
#ifdef MEGAMOE_NORM_PRELOAD_WEIGHT
  // Weights are independent of the producer. Keep the first vector in
  // registers across the dependency wait and the unchanged reduction.
  vec_t<T, VEC_SIZE> first_weight;
  first_weight.fill(0.f);
  if (thread_id * VEC_SIZE < d)
    first_weight.load(weight + thread_id * VEC_SIZE);
#endif

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
    if (i == 0) {
      // Preload the residual while combine is still producing input. The
      // diagnostic stamp depends on a loaded value, not just a memory issue.
      if constexpr (kTrace) {
        const float loaded = float(residual_vec[0]);
        asm volatile("" :: "f"(loaded) : "memory");
        __syncthreads();
        if (thread_id == 0) {
          uint64_t stamp;
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
          timings[24 + 2 * gridDim.x + bx] = stamp;
        }
      }
      // The producer is now the immediate predecessor. Every thread acquires
      // its completed stores after independent residual preload.
      asm volatile("griddepcontrol.wait;" ::: "memory");
      if constexpr (kTrace) {
        if (thread_id == 0) {
          uint64_t stamp;
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
          timings[24 + 2 * bx] = stamp;
        }
      }
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d)
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
#ifdef MEGAMOE_NORM_PRELOAD_WEIGHT
      if (i == 0) weight_vec = first_weight;
      else
#endif
        weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
  if constexpr (kTrace) {
    __syncthreads();
    if (thread_id == 0) {
      uint64_t stamp;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
      timings[25 + 2 * bx] = stamp;
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}


}  // namespace

struct MegaMoEOutputNorm {
  template <bool kTrace>
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
                  tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace,
                  int ranks, double eps, tvm::ffi::TensorView timings) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto dtype = SymbolicDType{};
    dtype.set_value(DLDataType{kDLBfloat, 16, 1});
    auto rows = SymbolicSize{"rows"}, hidden = SymbolicSize{"hidden"};
    TensorMatcher({rows, hidden}).with_strides({-1, 1})
        .with_dtype(dtype).with_device<kDLCUDA>(device).verify(input);
    TensorMatcher({rows, hidden}).with_strides({-1, 1})
        .with_dtype(dtype).with_device<kDLCUDA>(device).verify(residual);
    TensorMatcher({hidden}).with_dtype(dtype)
        .with_device<kDLCUDA>(device).verify(weight);
    RuntimeCheck(ranks >= 2 && input.size(1) % 512 == 0, "Invalid MegaMoE norm shape");
    const uint32_t d = input.size(1);
    constexpr uint32_t vec = 8;
    const uint32_t warps = std::min(1024u, d / vec) / 32;
    const auto shared = ((warps + 3) / 4 * 4 + d) * sizeof(float);
    const auto* base = static_cast<const uint32_t*>(workspace.data_ptr());
    LaunchKernel(dim3(input.size(0)), dim3(32, warps), input.device(), shared).with_attr(true)(
        megamoe_output_add_rmsnorm<vec, __nv_bfloat16, kTrace>,
        static_cast<__nv_bfloat16*>(input.data_ptr()),
        static_cast<__nv_bfloat16*>(residual.data_ptr()),
        static_cast<__nv_bfloat16*>(weight.data_ptr()), d,
        static_cast<uint32_t>(input.stride(0)), static_cast<uint32_t>(residual.stride(0)),
        0.f, static_cast<float>(eps), base + 8 + (1 + ranks) * 2 + 4,
        base + 8 + (1 + ranks) * 2 + 1,
        static_cast<uint64_t*>(timings.data_ptr()));
  }
};
