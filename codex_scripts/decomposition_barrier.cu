// Standalone decomposition preparation only; never called by serving kernels.
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ unsigned long long timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void decomposition_collective_barrier(
    unsigned long long* local, const int64_t* peers, int rank, int ranks) {
  __shared__ unsigned long long epoch;
  if (threadIdx.x == 0) epoch = local[ranks] + 1;
  __syncthreads();
  if (threadIdx.x < ranks) {
    const int peer = threadIdx.x;
    auto* remote = reinterpret_cast<unsigned long long*>(peers[peer]) + rank;
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :: "l"(remote), "l"(epoch) : "memory");
    const auto begin = timer_ns();
    unsigned long long arrived;
    do {
      asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                   : "=l"(arrived) : "l"(local + peer) : "memory");
      if (timer_ns() - begin > 60000000000ull) asm volatile("trap;");
    } while (arrived < epoch);
  }
  __syncthreads();
  if (threadIdx.x == 0) local[ranks] = epoch;
}
}  // namespace

struct DecompositionBarrier {
  static void run(tvm::ffi::TensorView local, tvm::ffi::TensorView peers,
                  int64_t rank, int64_t ranks) {
    using namespace host;
    RuntimeCheck(ranks >= 1 && ranks <= 64 && rank >= 0 && rank < ranks,
                 "decomposition barrier requires 1..64 ranks");
    auto device = SymbolicDevice{};
    TensorMatcher({ranks + 1}).with_device<kDLCUDA>(device)
        .with_dtype<int64_t>().verify(local);
    TensorMatcher({ranks}).with_device<kDLCUDA>(device)
        .with_dtype<int64_t>().verify(peers);
    auto stream = LaunchKernel::resolve_device(device.unwrap());
    decomposition_collective_barrier<<<1, 64, 0, stream>>>(
        static_cast<unsigned long long*>(local.data_ptr()),
        static_cast<const int64_t*>(peers.data_ptr()), rank, ranks);
    RuntimeCheck(cudaGetLastError() == cudaSuccess, "decomposition barrier launch failed");
  }
};
