#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
 auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
  3840,
  4096,
  1536,
  128,
  8,
  8,
  32,
  128,
  128,
  16,
  128,
  128,
  493056,
  7888896,
  10,
  128,
  128,
  256,
  150,
  16,
  cute::numeric_limits<float>::infinity(),
  true,
  false,
  false,
  true,
  true,
  true,
  true,
  true,
  true
 >);
}
