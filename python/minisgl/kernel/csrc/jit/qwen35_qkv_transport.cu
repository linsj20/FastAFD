#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;
constexpr int kRotaryHalf = kRotaryDim / 2;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kValuesPerLane = kHeadDim / kWarpSize;
constexpr int kRotaryPartnerLane = kRotaryHalf / kValuesPerLane;
constexpr int kSignalThreads = 32;
constexpr int kTransportSlots = 2;
constexpr int64_t kMaxPublicationBlocks = 1024;

struct alignas(16) Bf16Pack8 {
  __nv_bfloat16 values[kValuesPerLane];
};

struct alignas(8) Bf16Pack4 {
  __nv_bfloat16 values[4];
};

struct Qwen35QKVParams {
  const __nv_bfloat16* q_gate;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const void* positions;
  const __nv_bfloat16* q_weight;
  const __nv_bfloat16* k_weight;
  const float* cos_sin_cache;
  const void* out_loc;
  const int64_t* source_offsets;
  const int64_t* q_desc;
  const int64_t* kv_desc;
  const int64_t* ready_desc;
  int* completion_counter;
  int64_t* release_turn;
  int64_t next_turn;
  int64_t ready_value;
  int64_t q_gate_stride;
  int64_t k_stride;
  int64_t v_stride;
  int64_t cache_stride;
  int q_heads;
  int q_edges;
  int kv_edges;
  int ready_edges;
  int layer;
  int slot;
  int rows;
  bool positions_i64;
  bool loc_i64;
  float eps;
};

struct Qwen35SharedRMSNormParams {
  __nv_bfloat16* hidden;
  const __nv_bfloat16* shared;
  const __nv_bfloat16* shared_gate;
  __nv_bfloat16* residual;
  const __nv_bfloat16* weight;
  int64_t hidden_stride;
  int64_t shared_stride;
  int64_t shared_gate_stride;
  int64_t residual_stride;
  int rows;
  int width;
  float eps;
};

__device__ __forceinline__ float load_bf16_shared_scale(
    const __nv_bfloat16* gate_scale) {
  return __bfloat162float(*gate_scale);
}

inline unsigned publication_blocks(int64_t warps) {
  return static_cast<unsigned>(std::min<int64_t>(
      (warps + kWarpsPerBlock - 1) / kWarpsPerBlock,
      kMaxPublicationBlocks));
}

__device__ __forceinline__ void store_release_system(
    int64_t* ptr, int64_t value) {
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(ptr), "l"(value));
}

__device__ __forceinline__ void store_release_device(
    int64_t* ptr, int64_t value) {
  asm volatile(
      "st.release.gpu.global.u64 [%0], %1;"
      :
      : "l"(ptr), "l"(value));
}

template <typename Params>
__device__ __forceinline__ void release_peer_lane(const Params& params) {
  if (params.release_turn != nullptr && blockIdx.x == 0 && threadIdx.x == 0)
    store_release_device(params.release_turn, params.next_turn);
}

__device__ __forceinline__ int64_t* ready_pointer(
    const int64_t* descriptor, int slot) {
  auto* base = reinterpret_cast<int64_t*>(descriptor[0]);
  return base + static_cast<int64_t>(slot) * descriptor[2] + descriptor[1];
}

__device__ __forceinline__ int64_t first_flattened_task(
    int64_t warp, int64_t stride, int64_t edge_base) {
  if (warp >= edge_base)
    return warp;
  return warp + (edge_base - warp + stride - 1) / stride * stride;
}

__device__ __forceinline__ int64_t load_index(
    const void* pointer, int row, bool i64) {
  return i64 ? static_cast<const int64_t*>(pointer)[row]
             : static_cast<const int32_t*>(pointer)[row];
}

__device__ __forceinline__ Bf16Pack8 normalize_and_rope(
    const __nv_bfloat16* source,
    const __nv_bfloat16* weight,
    const float* cos_sin_cache,
    int64_t cache_stride,
    int64_t position,
    float eps,
    int lane) {
  const Bf16Pack8 input = reinterpret_cast<const Bf16Pack8*>(source)[lane];
  const Bf16Pack8 scale = reinterpret_cast<const Bf16Pack8*>(weight)[lane];
  float values[kValuesPerLane];
  float square_sum = 0.0f;
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    values[index] = __bfloat162float(input.values[index]);
    square_sum += values[index] * values[index];
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    square_sum += __shfl_xor_sync(0xffffffffu, square_sum, offset);
  const float inverse_rms = rsqrtf(square_sum / kHeadDim + eps);

  // The established path materializes BF16 RMSNorm output before RoPE. Keep
  // that rounding boundary so fusion changes traffic and launches, not math.
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const __nv_bfloat16 normalized = __float2bfloat16_rn(
        values[index] * inverse_rms * __bfloat162float(scale.values[index]));
    values[index] = __bfloat162float(normalized);
  }

  if (lane < kRotaryDim / kValuesPerLane) {
    const float* cos = cos_sin_cache + position * cache_stride;
    const float* sin = cos + kRotaryHalf;
#pragma unroll
    for (int index = 0; index < kValuesPerLane; ++index) {
      float partner = __shfl_xor_sync(
          0x000000ffu, values[index], kRotaryPartnerLane);
      if (lane < kRotaryPartnerLane)
        partner = -partner;
      const int dimension = lane * kValuesPerLane + index;
      const int frequency = dimension % kRotaryHalf;
      values[index] = values[index] * cos[frequency] + partner * sin[frequency];
    }
  }

  Bf16Pack8 output;
#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index)
    output.values[index] = __float2bfloat16_rn(values[index]);
  return output;
}

__device__ __forceinline__ void finish_publication(
    const Qwen35QKVParams& params) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x >= kSignalThreads)
    return;
  int arrived = 0;
  if (threadIdx.x == 0)
    arrived = atomicAdd(params.completion_counter, 1);
  arrived = __shfl_sync(0xffffffffu, arrived, 0);
  if (arrived != static_cast<int>(gridDim.x) - 1)
    return;
  if (threadIdx.x == 0)
    atomicExch(params.completion_counter, 0);
  for (int edge = threadIdx.x; edge < params.ready_edges; edge += kSignalThreads)
    store_release_system(
        ready_pointer(params.ready_desc + static_cast<int64_t>(edge) * 3,
                      params.slot),
        params.ready_value);
}

__global__ void qwen35_prepare_publish_qkv_kernel(
    const Qwen35QKVParams params) {
  // The producer GEMM is stream-ordered before this kernel. Hand model
  // compute ownership to the peer lane at kernel entry so the entire
  // RMSNorm/RoPE/fabric-publication tail remains overlapped.
  release_peer_lane(params);
  const int lane = threadIdx.x % kWarpSize;
  const int64_t warp =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpSize;
  const int64_t warp_stride =
      static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  int64_t edge_base = 0;
  for (int edge = 0; edge < params.kv_edges; ++edge) {
    const int64_t* desc = params.kv_desc + static_cast<int64_t>(edge) * 8;
    auto* cache = reinterpret_cast<__nv_bfloat16*>(desc[0]);
    const int src_head = static_cast<int>(desc[1]);
    const int dst_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[3]);
    const int dst_local_heads = static_cast<int>(desc[4]);
    const int64_t cache_rows = desc[5];
    const int num_layers = static_cast<int>(desc[6]);
    const int source = static_cast<int>(desc[7]);
    const int64_t source_start = params.source_offsets[source];
    const int64_t source_rows =
        params.source_offsets[source + 1] - source_start;
    const int64_t edge_end = edge_base + source_rows * head_count;
    const int64_t layer_stride = cache_rows * dst_local_heads * kHeadDim;
    auto* k_cache = cache + static_cast<int64_t>(params.layer) * layer_stride;
    auto* v_cache = cache +
        (static_cast<int64_t>(num_layers) + params.layer) * layer_stride;
    for (int64_t global_task =
             first_flattened_task(warp, warp_stride, edge_base);
         global_task < edge_end;
         global_task += warp_stride) {
      const int64_t task = global_task - edge_base;
      const int head = static_cast<int>(task % head_count);
      const int source_row = static_cast<int>(task / head_count);
      const int64_t row = source_start + source_row;
      const int64_t cache_row = load_index(params.out_loc, row, params.loc_i64);
      const __nv_bfloat16* k_source = params.k + row * params.k_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim;
      const __nv_bfloat16* v_source = params.v + row * params.v_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim;
      const int64_t destination_offset =
          cache_row * dst_local_heads * kHeadDim +
          static_cast<int64_t>(dst_head + head) * kHeadDim;
      const int64_t position =
          load_index(params.positions, static_cast<int>(row), params.positions_i64);
      const Bf16Pack8 prepared_k = normalize_and_rope(
          k_source,
          params.k_weight,
          params.cos_sin_cache,
          params.cache_stride,
          position,
          params.eps,
          lane);
      reinterpret_cast<Bf16Pack8*>(k_cache + destination_offset)[lane] =
          prepared_k;
      reinterpret_cast<Bf16Pack8*>(v_cache + destination_offset)[lane] =
          reinterpret_cast<const Bf16Pack8*>(v_source)[lane];
    }
    edge_base = edge_end;
  }

  edge_base = 0;
  for (int edge = 0; edge < params.q_edges; ++edge) {
    const int64_t* desc = params.q_desc + static_cast<int64_t>(edge) * 7;
    auto* destination = reinterpret_cast<__nv_bfloat16*>(desc[0]);
    const int src_head = static_cast<int>(desc[1]);
    const int dst_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[3]);
    const int dst_local_heads = static_cast<int>(desc[4]);
    const int64_t slot_rows = desc[5];
    const int source = static_cast<int>(desc[6]);
    const int64_t source_start = params.source_offsets[source];
    const int64_t source_rows =
        params.source_offsets[source + 1] - source_start;
    const int64_t edge_end = edge_base + source_rows * head_count;
    destination += static_cast<int64_t>(params.slot) * slot_rows *
        dst_local_heads * kHeadDim;
    const bool is_gate = src_head >= params.q_heads;
    const int raw_head_start = is_gate ? src_head - params.q_heads : src_head;
    for (int64_t global_task =
             first_flattened_task(warp, warp_stride, edge_base);
         global_task < edge_end;
         global_task += warp_stride) {
      const int64_t task = global_task - edge_base;
      const int head = static_cast<int>(task % head_count);
      const int source_row = static_cast<int>(task / head_count);
      const int64_t row = source_start + source_row;
      const __nv_bfloat16* source_ptr = params.q_gate +
          row * params.q_gate_stride +
          static_cast<int64_t>(raw_head_start + head) * 2 * kHeadDim +
          (is_gate ? kHeadDim : 0);
      const int64_t destination_offset =
          (static_cast<int64_t>(source_row) * dst_local_heads + dst_head + head) *
          kHeadDim;
      Bf16Pack8 output;
      if (is_gate) {
        output = reinterpret_cast<const Bf16Pack8*>(source_ptr)[lane];
      } else {
        const int64_t position = load_index(
            params.positions, static_cast<int>(row), params.positions_i64);
        output = normalize_and_rope(
            source_ptr,
            params.q_weight,
            params.cos_sin_cache,
            params.cache_stride,
            position,
            params.eps,
            lane);
      }
      reinterpret_cast<Bf16Pack8*>(destination + destination_offset)[lane] =
          output;
    }
    edge_base = edge_end;
  }

  finish_publication(params);
}

__launch_bounds__(kThreads) __global__ void qwen35_shared_add_rmsnorm_kernel(
    const Qwen35SharedRMSNormParams params) {
  constexpr int kGroup = 128;
  constexpr int kValuesPerGroupLane = kGroup / kWarpSize;
  constexpr int kMaxGroupsPerWarp = 8;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= params.rows)
    return;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int groups = params.width / kGroup;
  const int groups_per_warp = groups / kWarpsPerBlock;
  __shared__ float warp_square_sum[kWarpsPerBlock];
  auto* hidden_row = params.hidden +
      static_cast<int64_t>(row) * params.hidden_stride;
  const auto* shared_row = params.shared +
      static_cast<int64_t>(row) * params.shared_stride;
  float shared_scale = 1.0f;
  if (params.shared_gate != nullptr) {
    if (lane == 0)
      shared_scale = load_bf16_shared_scale(
          params.shared_gate +
          static_cast<int64_t>(row) * params.shared_gate_stride);
    shared_scale = __shfl_sync(0xffffffffu, shared_scale, 0);
  }
  auto* residual_row = params.residual +
      static_cast<int64_t>(row) * params.residual_stride;
  float values[kMaxGroupsPerWarp][kValuesPerGroupLane];
  float square_sum = 0.0f;
#pragma unroll 1
  for (int index = 0; index < groups_per_warp; ++index) {
    const int group = warp + index * kWarpsPerBlock;
    const int element = group * kGroup + lane * kValuesPerGroupLane;
    const Bf16Pack4 hidden_pack =
        reinterpret_cast<const Bf16Pack4*>(hidden_row + element)[0];
    const Bf16Pack4 shared_pack =
        reinterpret_cast<const Bf16Pack4*>(shared_row + element)[0];
    const Bf16Pack4 residual_pack =
        reinterpret_cast<const Bf16Pack4*>(residual_row + element)[0];
#pragma unroll
    for (int item = 0; item < kValuesPerGroupLane; ++item) {
      // Match `(routed_bf16 + shared_bf16) -> bf16` before the existing
      // residual-add/RMSNorm arithmetic.
      const __nv_bfloat16 scaled_shared = __float2bfloat16_rn(
          __bfloat162float(shared_pack.values[item]) * shared_scale);
      const __nv_bfloat16 combined = __float2bfloat16_rn(
          __bfloat162float(hidden_pack.values[item]) +
          __bfloat162float(scaled_shared));
      const float value = __bfloat162float(combined) +
          __bfloat162float(residual_pack.values[item]);
      values[index][item] = value;
      square_sum += value * value;
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    square_sum += __shfl_xor_sync(0xffffffffu, square_sum, offset);
  if (lane == 0)
    warp_square_sum[warp] = square_sum;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int index = 0; index < kWarpsPerBlock; ++index)
    total += warp_square_sum[index];
  const float inverse_rms = rsqrtf(total / params.width + params.eps);

#pragma unroll 1
  for (int index = 0; index < groups_per_warp; ++index) {
    const int group = warp + index * kWarpsPerBlock;
    const int element = group * kGroup + lane * kValuesPerGroupLane;
    const Bf16Pack4 weight_pack =
        reinterpret_cast<const Bf16Pack4*>(params.weight + element)[0];
    Bf16Pack4 residual_output;
    Bf16Pack4 normalized_output;
#pragma unroll
    for (int item = 0; item < kValuesPerGroupLane; ++item) {
      residual_output.values[item] =
          __float2bfloat16_rn(values[index][item]);
      normalized_output.values[item] = __float2bfloat16_rn(
          values[index][item] * inverse_rms *
          __bfloat162float(weight_pack.values[item]));
    }
    reinterpret_cast<Bf16Pack4*>(residual_row + element)[0] = residual_output;
    reinterpret_cast<Bf16Pack4*>(hidden_row + element)[0] = normalized_output;
  }
}

}  // namespace

struct Qwen35QKVTransportKernel {
  static void publish(
      const tvm::ffi::TensorView q_gate,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView source_offsets,
      const tvm::ffi::TensorView q_desc,
      const tvm::ffi::TensorView kv_desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      tvm::ffi::Optional<tvm::ffi::TensorView> release_turn,
      int64_t next_turn,
      int64_t layer,
      int64_t slot,
      int64_t ready_value,
      double eps) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto q_gate_width = SymbolicSize{"q_gate_width"};
    auto kv_width = SymbolicSize{"kv_width"};
    auto data_dtype = SymbolicDType{};
    auto position_dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    auto cache_rows = SymbolicSize{"cache_rows"};
    auto rotary_dim = SymbolicSize{"rotary_dim"};
    auto source_offset_count = SymbolicSize{"source_offset_count"};
    auto q_edges = SymbolicSize{"q_edges"};
    auto kv_edges = SymbolicSize{"kv_edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    TensorMatcher({rows, q_gate_width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(q_gate);
    TensorMatcher({rows, kv_width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(k)
        .verify(v);
    TensorMatcher({rows})
        .with_dtype<int32_t, int64_t>(position_dtype)
        .with_device<kDLCUDA>(device)
        .verify(positions);
    TensorMatcher({kHeadDim})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(q_weight)
        .verify(k_weight);
    TensorMatcher({cache_rows, rotary_dim})
        .with_strides({-1, 1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(cos_sin_cache);
    TensorMatcher({rows})
        .with_dtype<int32_t, int64_t>(loc_dtype)
        .with_device<kDLCUDA>(device)
        .verify(out_loc);
    TensorMatcher({source_offset_count})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(source_offsets);
    TensorMatcher({q_edges, 7})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(q_desc);
    TensorMatcher({kv_edges, 8})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(kv_desc);
    TensorMatcher({ready_edges, 3})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ready_desc);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(completion_counter);
    int64_t* release_turn_ptr = nullptr;
    if (release_turn.has_value()) {
      TensorMatcher({1})
          .with_dtype<int64_t>()
          .with_device<kDLCUDA>(device)
          .verify(release_turn.value());
      release_turn_ptr = static_cast<int64_t*>(release_turn.value().data_ptr());
    }
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
        dtype.lanes == 1,
        "Qwen3.5 QKV transport requires BF16 inputs and norm weights");
    RuntimeCheck(
        q_gate_width.unwrap() > 0 and
        q_gate_width.unwrap() % (2 * kHeadDim) == 0 and
        kv_width.unwrap() > 0 and kv_width.unwrap() % kHeadDim == 0);
    RuntimeCheck(rotary_dim.unwrap() == kRotaryDim);
    RuntimeCheck(
        source_offset_count.unwrap() > 1 and q_edges.unwrap() > 0 and
        kv_edges.unwrap() > 0 and ready_edges.unwrap() > 0);
    RuntimeCheck(
        layer >= 0 and slot >= 0 and slot < kTransportSlots and
        ready_value > 0 and eps > 0.0 and
        (release_turn_ptr == nullptr || next_turn >= 0));

    const int q_heads = static_cast<int>(
        q_gate_width.unwrap() / (2 * kHeadDim));
    const Qwen35QKVParams params{
        .q_gate = static_cast<const __nv_bfloat16*>(q_gate.data_ptr()),
        .k = static_cast<const __nv_bfloat16*>(k.data_ptr()),
        .v = static_cast<const __nv_bfloat16*>(v.data_ptr()),
        .positions = positions.data_ptr(),
        .q_weight = static_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
        .k_weight = static_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .out_loc = out_loc.data_ptr(),
        .source_offsets = static_cast<const int64_t*>(source_offsets.data_ptr()),
        .q_desc = static_cast<const int64_t*>(q_desc.data_ptr()),
        .kv_desc = static_cast<const int64_t*>(kv_desc.data_ptr()),
        .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
        .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
        .release_turn = release_turn_ptr,
        .next_turn = next_turn,
        .ready_value = ready_value,
        .q_gate_stride = q_gate.stride(0),
        .k_stride = k.stride(0),
        .v_stride = v.stride(0),
        .cache_stride = cos_sin_cache.stride(0),
        .q_heads = q_heads,
        .q_edges = static_cast<int>(q_edges.unwrap()),
        .kv_edges = static_cast<int>(kv_edges.unwrap()),
        .ready_edges = static_cast<int>(ready_edges.unwrap()),
        .layer = static_cast<int>(layer),
        .slot = static_cast<int>(slot),
        .rows = static_cast<int>(rows.unwrap()),
        .positions_i64 = position_dtype.unwrap().bits == 64,
        .loc_i64 = loc_dtype.unwrap().bits == 64,
        .eps = static_cast<float>(eps),
    };
    const int64_t logical_warps = std::max<int64_t>(
        1,
        rows.unwrap() *
            (q_gate_width.unwrap() / kHeadDim + kv_width.unwrap() / kHeadDim));
    const LaunchKernel launch(
        publication_blocks(logical_warps), kThreads, device.unwrap());
    launch(qwen35_prepare_publish_qkv_kernel, params);
  }

  static void shared_add_rmsnorm(
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView shared,
      tvm::ffi::Optional<tvm::ffi::TensorView> shared_gate,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      double eps) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(hidden)
        .verify(shared)
        .verify(residual);
    const __nv_bfloat16* shared_gate_ptr = nullptr;
    int64_t shared_gate_stride = 0;
    if (shared_gate.has_value()) {
      TensorMatcher({rows, 1})
          .with_strides({-1, 1})
          .with_dtype(data_dtype)
          .with_device<kDLCUDA>(device)
          .verify(shared_gate.value());
      shared_gate_ptr = static_cast<const __nv_bfloat16*>(
          shared_gate.value().data_ptr());
      shared_gate_stride = shared_gate.value().stride(0);
    }
    TensorMatcher({width})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(weight);
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
        dtype.lanes == 1,
        "Qwen3.5 shared-add/RMSNorm requires BF16 tensors");
    RuntimeCheck(
        rows.unwrap() > 0 and width.unwrap() % 1024 == 0 and
        width.unwrap() <= 8192 and eps > 0.0);
    const Qwen35SharedRMSNormParams params{
        .hidden = static_cast<__nv_bfloat16*>(hidden.data_ptr()),
        .shared = static_cast<const __nv_bfloat16*>(shared.data_ptr()),
        .shared_gate = shared_gate_ptr,
        .residual = static_cast<__nv_bfloat16*>(residual.data_ptr()),
        .weight = static_cast<const __nv_bfloat16*>(weight.data_ptr()),
        .hidden_stride = hidden.stride(0),
        .shared_stride = shared.stride(0),
        .shared_gate_stride = shared_gate_stride,
        .residual_stride = residual.stride(0),
        .rows = static_cast<int>(rows.unwrap()),
        .width = static_cast<int>(width.unwrap()),
        .eps = static_cast<float>(eps),
    };
    const LaunchKernel launch(
        static_cast<unsigned>(rows.unwrap()), kThreads, device.unwrap());
    launch(qwen35_shared_add_rmsnorm_kernel, params);
  }
};
