#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kSignalThreads = 32;
constexpr int kTransportSlots = 2;
constexpr int64_t kMaxPublicationBlocks = 1024;

inline unsigned publication_blocks(int64_t tasks) {
  // Every publication CTA executes a system-scoped release fence before the
  // final CTA publishes readiness.  Keep enough CTAs to saturate the GPU, but
  // let their grid-stride loops absorb large grouped-prefill payloads instead
  // of issuing hundreds of thousands of fenced CTAs.
  return static_cast<unsigned>(std::min<int64_t>(
      (tasks + static_cast<int64_t>(kThreads) - 1) /
          static_cast<int64_t>(kThreads),
      kMaxPublicationBlocks));
}

struct PublishQKVParams {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const void* out_loc;
  const int64_t* source_offsets;
  const int64_t* q_desc;
  const int64_t* kv_desc;
  const int64_t* ready_desc;
  int* completion_counter;
  int64_t ready_value;
  int64_t q_stride;
  int64_t k_stride;
  int64_t v_stride;
  int q_edges;
  int kv_edges;
  int ready_edges;
  int layer;
  int slot;
  int rows;
  bool loc_i64;
};

struct PublishOParams {
  const __nv_bfloat16* o;
  const int64_t* desc;
  const int64_t* ready_desc;
  int* completion_counter;
  int64_t ready_value;
  int64_t o_stride;
  int edges;
  int ready_edges;
  int slot;
  int rows;
  int destination_source_stride;
};

struct PublishOReleaseTurnParams {
  PublishOParams publish;
  int64_t* turn;
  int64_t next_turn;
};

struct PublishOFp8Params {
  const uint8_t* o;
  const int32_t* scales;
  const int64_t* desc;
  const int64_t* ready_desc;
  int* completion_counter;
  int64_t ready_value;
  int64_t o_stride;
  int64_t scale_stride_row;
  int64_t scale_stride_group;
  int edges;
  int ready_edges;
  int slot;
  int rows;
  int destination_source_stride;
};

struct PublishOFp8ReleaseTurnParams {
  PublishOFp8Params publish;
  int64_t* turn;
  int64_t next_turn;
};

struct QuantizePublishOFp8Params {
  const __nv_bfloat16* o;
  uint8_t* staged_o;
  uint8_t* staged_scales;
  const int64_t* desc;
  const int64_t* ready_desc;
  int* completion_counter;
  int* quantization_counter;
  int64_t* turn;
  int64_t ready_value;
  int64_t next_turn;
  int64_t o_stride;
  int64_t staged_o_stride;
  int64_t staged_scale_stride;
  int edges;
  int ready_edges;
  int slot;
  int rows;
  int destination_source_stride;
};

__device__ __forceinline__ uint64_t load_acquire_system(const int64_t* ptr) {
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.L1::no_allocate.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint64_t load_acquire_device(const int64_t* ptr) {
  uint64_t value;
  asm volatile(
      "ld.acquire.gpu.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(ptr));
  return value;
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

__device__ __forceinline__ int64_t* ready_pointer(
    const int64_t* descriptor, int slot) {
  auto* base = reinterpret_cast<int64_t*>(descriptor[0]);
  const int64_t writer_index = descriptor[1];
  const int64_t writers_per_slot = descriptor[2];
  return base + static_cast<int64_t>(slot) * writers_per_slot + writer_index;
}

__device__ __forceinline__ int64_t first_flattened_task(
    int64_t thread,
    int64_t stride,
    int64_t edge_base) {
  if (thread >= edge_base)
    return thread;
  return thread +
      (edge_base - thread + stride - 1) / stride * stride;
}

template <bool kSingleReady>
__device__ __forceinline__ void finish_payload_publication(
    int* completion_counter,
    const int64_t* ready_desc,
    int ready_edges,
    int slot,
    int64_t ready_value) {
  // Every payload writer makes its own remote stores system-visible before
  // its CTA arrives. The last CTA resets the reusable local counter and sends
  // exactly one release signal to each destination. Receivers poll only these
  // words, so publication needs no cooperative grid barrier or payload scan.
  __threadfence_system();
  __syncthreads();
  if constexpr (kSingleReady) {
    if (threadIdx.x != 0)
      return;
    const int arrived = atomicAdd(completion_counter, 1);
    if (arrived != static_cast<int>(gridDim.x) - 1)
      return;
    atomicExch(completion_counter, 0);
    store_release_system(ready_pointer(ready_desc, slot), ready_value);
    return;
  }
  if (threadIdx.x >= kSignalThreads)
    return;
  int arrived = 0;
  if (threadIdx.x == 0)
    arrived = atomicAdd(completion_counter, 1);
  arrived = __shfl_sync(0xffffffffu, arrived, 0);
  if (arrived != static_cast<int>(gridDim.x) - 1)
    return;
  if (threadIdx.x == 0)
    atomicExch(completion_counter, 0);
  // All payload fences completed before the existing CTA rendezvous. Spread
  // the already-required destination releases across the last warp instead
  // of serializing them on lane zero. The shuffle is only a register exchange;
  // it adds no memory, CTA, grid, or cross-GPU barrier.
  for (int edge = threadIdx.x; edge < ready_edges; edge += kSignalThreads)
    store_release_system(
        ready_pointer(ready_desc + edge * 3, slot), ready_value);
}

__global__ void wait_turn_kernel(
    int64_t* turn,
    uint64_t timeout_cycles,
    int64_t* timeout_record,
    int64_t expected_turn,
    int64_t next_turn) {
  if (threadIdx.x != 0)
    return;
  const uint64_t start = clock64();
  while (load_acquire_device(turn) != static_cast<uint64_t>(expected_turn)) {
    if (clock64() - start > timeout_cycles) {
      timeout_record[0] = 1;
      timeout_record[1] = expected_turn;
      __threadfence_system();
      asm volatile("trap;");
      return;
    }
    __nanosleep(64);
  }
  store_release_device(turn, next_turn);
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady>
__global__ void publish_qkv_kernel(const PublishQKVParams params) {
  static_assert(kHeadDim % 4 == 0);
  constexpr int kPacksPerHead = kHeadDim / 4;
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // Each descriptor names one designated model-TP writer for an AG KV head
  // slice. Q, K, and V are one payload push; the final ready release orders all
  // of them, so there is no intra-kernel K/V-before-Q grid synchronization.
  // Map each phase's source edges into one logical task space. Reusing the
  // same per-edge task index would make only the first few CTAs copy every
  // source at high fan-in while the rest of the publication grid sat idle.
  int64_t edge_base = 0;
  const int kv_edges = kSingleEdge ? 1 : params.kv_edges;
  for (int edge = 0; edge < kv_edges; ++edge) {
    const int64_t* desc = params.kv_desc + static_cast<int64_t>(edge) * 8;
    auto* cache = reinterpret_cast<__nv_bfloat16*>(desc[0]);
    const int src_head = static_cast<int>(desc[1]);
    const int dst_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[3]);
    const int dst_local_heads = static_cast<int>(desc[4]);
    const int64_t cache_rows = desc[5];
    const int num_layers = static_cast<int>(desc[6]);
    int64_t source_start = 0;
    int64_t source_rows = params.rows;
    if constexpr (!kSingleEdge) {
      const int source = static_cast<int>(desc[7]);
      source_start = params.source_offsets[source];
      source_rows = params.source_offsets[source + 1] - source_start;
    }
    const int64_t layer_stride = cache_rows * dst_local_heads * kHeadDim;
    auto* k_cache = cache + static_cast<int64_t>(params.layer) * layer_stride;
    auto* v_cache = cache +
                    (static_cast<int64_t>(num_layers) + params.layer) * layer_stride;
    const int64_t tasks =
        source_rows * head_count * kPacksPerHead;
    const int64_t edge_end = edge_base + tasks;
    for (int64_t global_task =
             first_flattened_task(thread, stride, edge_base);
         global_task < edge_end;
         global_task += stride) {
      const int64_t task = global_task - edge_base;
      const int pack = static_cast<int>(task % kPacksPerHead);
      const int64_t row_head = task / kPacksPerHead;
      const int head = static_cast<int>(row_head % head_count);
      const int source_row = static_cast<int>(row_head / head_count);
      const int64_t row =
          kSingleEdge ? source_row : source_start + source_row;
      const int64_t cache_row = params.loc_i64
          ? static_cast<const int64_t*>(params.out_loc)[row]
          : static_cast<const int32_t*>(params.out_loc)[row];
      const int64_t src_offset =
          static_cast<int64_t>(row) * params.k_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim + pack * 4;
      const int64_t v_src_offset =
          static_cast<int64_t>(row) * params.v_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim + pack * 4;
      const int64_t dst_offset =
          cache_row * dst_local_heads * kHeadDim +
          static_cast<int64_t>(dst_head + head) * kHeadDim + pack * 4;
      reinterpret_cast<uint64_t*>(k_cache + dst_offset)[0] =
          reinterpret_cast<const uint64_t*>(params.k + src_offset)[0];
      reinterpret_cast<uint64_t*>(v_cache + dst_offset)[0] =
          reinterpret_cast<const uint64_t*>(params.v + v_src_offset)[0];
    }
    if constexpr (!kSingleEdge)
      edge_base = edge_end;
  }
  edge_base = 0;
  const int q_edges = kSingleEdge ? 1 : params.q_edges;
  for (int edge = 0; edge < q_edges; ++edge) {
    const int64_t* desc = params.q_desc + static_cast<int64_t>(edge) * 7;
    auto* destination = reinterpret_cast<__nv_bfloat16*>(desc[0]);
    const int src_head = static_cast<int>(desc[1]);
    const int dst_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[3]);
    const int dst_local_heads = static_cast<int>(desc[4]);
    const int64_t slot_rows = desc[5];
    int64_t source_start = 0;
    int64_t source_rows = params.rows;
    if constexpr (!kSingleEdge) {
      const int source = static_cast<int>(desc[6]);
      source_start = params.source_offsets[source];
      source_rows = params.source_offsets[source + 1] - source_start;
    }
    destination +=
        static_cast<int64_t>(params.slot) * slot_rows * dst_local_heads * kHeadDim;
    const int64_t tasks =
        source_rows * head_count * kPacksPerHead;
    const int64_t edge_end = edge_base + tasks;
    for (int64_t global_task =
             first_flattened_task(thread, stride, edge_base);
         global_task < edge_end;
         global_task += stride) {
      const int64_t task = global_task - edge_base;
      const int pack = static_cast<int>(task % kPacksPerHead);
      const int64_t row_head = task / kPacksPerHead;
      const int head = static_cast<int>(row_head % head_count);
      const int source_row = static_cast<int>(row_head / head_count);
      const int64_t row =
          kSingleEdge ? source_row : source_start + source_row;
      const int64_t src_offset =
          static_cast<int64_t>(row) * params.q_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim + pack * 4;
      const int64_t dst_offset =
          (static_cast<int64_t>(source_row) * dst_local_heads + dst_head + head) *
              kHeadDim +
          pack * 4;
      reinterpret_cast<uint64_t*>(destination + dst_offset)[0] =
          reinterpret_cast<const uint64_t*>(params.q + src_offset)[0];
    }
    if constexpr (!kSingleEdge)
      edge_base = edge_end;
  }
  finish_payload_publication<kSingleReady>(
      params.completion_counter,
      params.ready_desc,
      params.ready_edges,
      params.slot,
      params.ready_value);
}

template <int kHeadDim, bool kSingleEdge>
__device__ __forceinline__ void publish_o_payload(const PublishOParams params) {
  static_assert(kHeadDim % 4 == 0);
  constexpr int kPacksPerHead = kHeadDim / 4;
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int edges = kSingleEdge ? 1 : params.edges;
  for (int edge = 0; edge < edges; ++edge) {
    const int64_t* desc = params.desc + static_cast<int64_t>(edge) * 7;
    auto* destination = reinterpret_cast<__nv_bfloat16*>(desc[0]);
    const int src_head = static_cast<int>(desc[1]);
    const int dst_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[3]);
    const int dst_local_heads = static_cast<int>(desc[4]);
    const int64_t slot_rows = desc[5];
    const int source = static_cast<int>(desc[6]);
    destination +=
        static_cast<int64_t>(params.slot) * slot_rows * dst_local_heads * kHeadDim;
    // A single outgoing edge does not imply source zero. In grouped fan-in,
    // every attention rank has one edge to its model rank, but that edge can
    // own any source partition in the destination arena.
    destination += static_cast<int64_t>(source) *
        params.destination_source_stride * dst_local_heads * kHeadDim;
    const int64_t tasks =
        static_cast<int64_t>(params.rows) * head_count * kPacksPerHead;
    for (int64_t task = thread; task < tasks; task += stride) {
      const int pack = static_cast<int>(task % kPacksPerHead);
      const int64_t row_head = task / kPacksPerHead;
      const int head = static_cast<int>(row_head % head_count);
      const int row = static_cast<int>(row_head / head_count);
      const int64_t src_offset =
          static_cast<int64_t>(row) * params.o_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim + pack * 4;
      const int64_t dst_offset =
          (static_cast<int64_t>(row) * dst_local_heads + dst_head + head) *
              kHeadDim +
          pack * 4;
      reinterpret_cast<uint64_t*>(destination + dst_offset)[0] =
          reinterpret_cast<const uint64_t*>(params.o + src_offset)[0];
    }
  }
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady>
__global__ void publish_o_kernel(const PublishOParams params) {
  publish_o_payload<kHeadDim, kSingleEdge>(params);
  finish_payload_publication<kSingleReady>(
      params.completion_counter,
      params.ready_desc,
      params.ready_edges,
      params.slot,
      params.ready_value);
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady>
__global__ void publish_o_kernel_release_turn(
    const PublishOReleaseTurnParams params) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // The turn ticket is local to this GPU. Remote O visibility is
    // independently ordered below.
    store_release_device(params.turn, params.next_turn);
  }
  // Release the opposite attention lane at the start of the ordinary O push,
  // while the completion counter publishes O readiness only after every block
  // has made its payload stores visible.
  publish_o_payload<kHeadDim, kSingleEdge>(params.publish);
  finish_payload_publication<kSingleReady>(
      params.publish.completion_counter,
      params.publish.ready_desc,
      params.publish.ready_edges,
      params.publish.slot,
      params.publish.ready_value);
}

template <int kHeadDim, bool kSingleEdge>
__device__ __forceinline__ void publish_o_fp8_payload(
    const PublishOFp8Params params) {
  static_assert(kHeadDim % 16 == 0);
  constexpr int kPacksPerHead = kHeadDim / 16;
  const int64_t thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int edges = kSingleEdge ? 1 : params.edges;
  for (int edge = 0; edge < edges; ++edge) {
    const int64_t* desc = params.desc + static_cast<int64_t>(edge) * 10;
    auto* destination = reinterpret_cast<uint8_t*>(desc[0]);
    auto* destination_scales = reinterpret_cast<int32_t*>(desc[1]);
    const int src_head = static_cast<int>(desc[2]);
    const int dst_head = static_cast<int>(desc[3]);
    const int head_count = static_cast<int>(desc[4]);
    const int dst_local_heads = static_cast<int>(desc[5]);
    const int64_t slot_rows = desc[6];
    const int64_t scale_slot_elements = desc[7];
    const int source = static_cast<int>(desc[8]);
    const int source_count = static_cast<int>(desc[9]);
    destination +=
        static_cast<int64_t>(params.slot) * slot_rows * dst_local_heads * kHeadDim;
    destination += static_cast<int64_t>(source) *
        params.destination_source_stride * dst_local_heads * kHeadDim;
    destination_scales +=
        static_cast<int64_t>(params.slot) * scale_slot_elements;

    const int64_t payload_tasks =
        static_cast<int64_t>(params.rows) * head_count * kPacksPerHead;
    for (int64_t task = thread; task < payload_tasks; task += stride) {
      const int pack = static_cast<int>(task % kPacksPerHead);
      const int64_t row_head = task / kPacksPerHead;
      const int head = static_cast<int>(row_head % head_count);
      const int row = static_cast<int>(row_head / head_count);
      const int64_t src_offset =
          static_cast<int64_t>(row) * params.o_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim + pack * 16;
      const int64_t dst_offset =
          (static_cast<int64_t>(row) * dst_local_heads + dst_head + head) *
              kHeadDim +
          pack * 16;
      reinterpret_cast<uint4*>(destination + dst_offset)[0] =
          reinterpret_cast<const uint4*>(params.o + src_offset)[0];
    }

    const int packed_head_count = head_count / 4;
    const int64_t scale_tasks =
        static_cast<int64_t>(params.rows) * packed_head_count;
    const int64_t destination_rows =
        static_cast<int64_t>(params.destination_source_stride) * source_count;
    const int64_t destination_scale_stride =
        (destination_rows + 3) / 4 * 4;
    for (int64_t task = thread; task < scale_tasks; task += stride) {
      const int packed_head = static_cast<int>(task % packed_head_count);
      const int row = static_cast<int>(task / packed_head_count);
      const int64_t src_offset =
          static_cast<int64_t>(row) * params.scale_stride_row +
          static_cast<int64_t>(src_head / 4 + packed_head) *
              params.scale_stride_group;
      const int64_t destination_row =
          static_cast<int64_t>(source) * params.destination_source_stride + row;
      const int64_t dst_offset =
          destination_row + static_cast<int64_t>(dst_head / 4 + packed_head) *
              destination_scale_stride;
      destination_scales[dst_offset] = params.scales[src_offset];
    }
  }
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady>
__global__ void publish_o_fp8_kernel(const PublishOFp8Params params) {
  publish_o_fp8_payload<kHeadDim, kSingleEdge>(params);
  finish_payload_publication<kSingleReady>(
      params.completion_counter,
      params.ready_desc,
      params.ready_edges,
      params.slot,
      params.ready_value);
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady>
__global__ void publish_o_fp8_kernel_release_turn(
    const PublishOFp8ReleaseTurnParams params) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    store_release_device(params.turn, params.next_turn);
  publish_o_fp8_payload<kHeadDim, kSingleEdge>(params.publish);
  finish_payload_publication<kSingleReady>(
      params.publish.completion_counter,
      params.publish.ready_desc,
      params.publish.ready_edges,
      params.publish.slot,
      params.publish.ready_value);
}

template <int kHeadDim, bool kSingleEdge>
__device__ __forceinline__ void quantize_o_fp8_to_stage(
    const QuantizePublishOFp8Params params) {
  // Keep quantization as the attention compute tail, but materialize its small
  // FP8 result locally so remote publication can remain a communication tail.
  static_assert(kHeadDim == 128);
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kLanesPerHead = 8;
  constexpr int kHeadsPerWarp = 32 / kLanesPerHead;
  constexpr int kValuesPerLane = kHeadDim / kLanesPerHead;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int head_in_warp = lane / kLanesPerHead;
  const int lane_in_head = lane % kLanesPerHead;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t warp_stride =
      static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int edges = kSingleEdge ? 1 : params.edges;

  for (int edge = 0; edge < edges; ++edge) {
    const int64_t* desc = params.desc + static_cast<int64_t>(edge) * 10;
    const int src_head = static_cast<int>(desc[2]);
    const int head_count = static_cast<int>(desc[4]);

    const int64_t tasks = static_cast<int64_t>(params.rows) * head_count;
    const int64_t task_group_stride = warp_stride * kHeadsPerWarp;
    for (int64_t first_task = global_warp;
         first_task < tasks;
         first_task += task_group_stride) {
      // Each 8-lane subgroup quantizes one head while the other subgroups
      // process three additional tasks owned by this warp.  This preserves
      // the publication mapping below and shortens the pre-handoff phase
      // without changing the grid, payload layout, or scale definition.
      const int64_t task = first_task + head_in_warp * warp_stride;
      if (task >= tasks)
        continue;
      const int head = static_cast<int>(task % head_count);
      const int row = static_cast<int>(task / head_count);
      const auto* source_values = params.o +
          static_cast<int64_t>(row) * params.o_stride +
          static_cast<int64_t>(src_head + head) * kHeadDim +
          lane_in_head * kValuesPerLane;
      const uint4 packed_bf16[2] = {
          *reinterpret_cast<const uint4*>(source_values),
          *reinterpret_cast<const uint4*>(source_values + 8),
      };
      const auto* values =
          reinterpret_cast<const __nv_bfloat16*>(packed_bf16);
      float converted[kValuesPerLane];
      float amax = 1.0e-4f;
      #pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        converted[i] = __bfloat162float(values[i]);
        amax = fmaxf(amax, fabsf(converted[i]));
      }
      #pragma unroll
      for (int offset = kLanesPerHead / 2; offset > 0; offset >>= 1)
        amax = fmaxf(
            amax,
            __shfl_xor_sync(
                0xffu << (head_in_warp * kLanesPerHead),
                amax,
                offset,
                kLanesPerHead));

      const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
      const uint8_t exponent = static_cast<uint8_t>(
          ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));
      const float inverse_scale = __uint_as_float(
          static_cast<uint32_t>(254u - exponent) << 23);
      alignas(16) __nv_fp8x2_storage_t output[kValuesPerLane / 2];
      #pragma unroll
      for (int i = 0; i < kValuesPerLane; i += 2)
        output[i / 2] = __nv_cvt_float2_to_fp8x2(
            make_float2(
                converted[i] * inverse_scale,
                converted[i + 1] * inverse_scale),
            __NV_SATFINITE,
            __NV_E4M3);

      const int source_head = src_head + head;
      const int64_t staged_offset =
          static_cast<int64_t>(row) * params.staged_o_stride +
          static_cast<int64_t>(source_head) * kHeadDim +
          lane_in_head * kValuesPerLane;
      *reinterpret_cast<uint4*>(params.staged_o + staged_offset) =
          *reinterpret_cast<const uint4*>(output);
      if (lane_in_head == 0)
        params.staged_scales[
            static_cast<int64_t>(row) * params.staged_scale_stride +
            source_head] = exponent;
    }
  }
}

template <int kHeadDim, bool kSingleEdge>
__device__ __forceinline__ void publish_staged_o_fp8_payload(
    const QuantizePublishOFp8Params params) {
  static_assert(kHeadDim == 128);
  constexpr int kWarpsPerBlock = kThreads / 32;
  constexpr int kPacksPerHead = kHeadDim / 16;
  constexpr int kHeadsPerWarp = 32 / kPacksPerHead;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int head_in_warp = lane / kPacksPerHead;
  const int pack = lane % kPacksPerHead;
  const int64_t global_warp =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t warp_stride =
      static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  const int edges = kSingleEdge ? 1 : params.edges;

  for (int edge = 0; edge < edges; ++edge) {
    const int64_t* desc = params.desc + static_cast<int64_t>(edge) * 10;
    auto* destination = reinterpret_cast<uint8_t*>(desc[0]);
    auto* destination_scales = reinterpret_cast<uint8_t*>(desc[1]);
    const int src_head = static_cast<int>(desc[2]);
    const int dst_head = static_cast<int>(desc[3]);
    const int head_count = static_cast<int>(desc[4]);
    const int dst_local_heads = static_cast<int>(desc[5]);
    const int64_t slot_rows = desc[6];
    const int64_t scale_slot_elements = desc[7];
    const int source = static_cast<int>(desc[8]);
    const int source_count = static_cast<int>(desc[9]);
    destination +=
        static_cast<int64_t>(params.slot) * slot_rows * dst_local_heads * kHeadDim;
    destination += static_cast<int64_t>(source) *
        params.destination_source_stride * dst_local_heads * kHeadDim;
    destination_scales +=
        static_cast<int64_t>(params.slot) * scale_slot_elements * sizeof(int32_t);

    const int64_t tasks = static_cast<int64_t>(params.rows) * head_count;
    const int64_t task_group_stride = warp_stride * kHeadsPerWarp;
    for (int64_t first_task = global_warp;
         first_task < tasks;
         first_task += task_group_stride) {
      // Quantization assigned every warp tasks separated by warp_stride.  Fold
      // four of those owned heads into one full-warp publication instruction
      // group, matching the accepted publisher's lane utilization without
      // reading staging produced by another CTA.
      const int64_t task = first_task + head_in_warp * warp_stride;
      if (task >= tasks)
        continue;
      const int head = static_cast<int>(task % head_count);
      const int row = static_cast<int>(task / head_count);
      const int source_head = src_head + head;
      const int64_t staged_offset =
          static_cast<int64_t>(row) * params.staged_o_stride +
          static_cast<int64_t>(source_head) * kHeadDim +
          pack * 16;
      const int64_t dst_offset =
          (static_cast<int64_t>(row) * dst_local_heads + dst_head + head) *
              kHeadDim +
          pack * 16;
      *reinterpret_cast<uint4*>(destination + dst_offset) =
          *reinterpret_cast<const uint4*>(params.staged_o + staged_offset);
      if (pack == 0) {
        const int64_t destination_rows =
            static_cast<int64_t>(params.destination_source_stride) * source_count;
        const int64_t destination_scale_stride =
            (destination_rows + 3) / 4 * 4;
        const int destination_head = dst_head + head;
        const int64_t destination_row =
            static_cast<int64_t>(source) * params.destination_source_stride + row;
        const int64_t destination_word =
            destination_row + static_cast<int64_t>(destination_head / 4) *
                destination_scale_stride;
        destination_scales[destination_word * sizeof(int32_t) +
                           destination_head % 4] =
            params.staged_scales[
                static_cast<int64_t>(row) * params.staged_scale_stride +
                source_head];
      }
    }
  }
}

template <int kHeadDim, bool kSingleEdge, bool kSingleReady, bool kReleaseTurn>
__global__ void quantize_publish_o_fp8_kernel(
    const QuantizePublishOFp8Params params) {
  quantize_o_fp8_to_stage<kHeadDim, kSingleEdge>(params);
  __syncthreads();
  if constexpr (kReleaseTurn) {
    // Release attention ownership as soon as every CTA has stopped reading BF16
    // O. Each CTA then publishes only its own staged tasks, so publication can
    // overlap the peer lane without a cooperative grid barrier.
    if (threadIdx.x == 0) {
      const int arrived = atomicAdd(params.quantization_counter, 1);
      if (arrived == static_cast<int>(gridDim.x) - 1) {
        atomicExch(params.quantization_counter, 0);
        store_release_device(params.turn, params.next_turn);
      }
    }
  }
  publish_staged_o_fp8_payload<kHeadDim, kSingleEdge>(params);
  finish_payload_publication<kSingleReady>(
      params.completion_counter,
      params.ready_desc,
      params.ready_edges,
      params.slot,
      params.ready_value);
}

__global__ void wait_ready_kernel(
    const int64_t* ready,
    int ready_count,
    uint64_t timeout_cycles,
    int64_t* timeout_record,
    int64_t expected_ready,
    const int64_t* turn,
    int64_t expected_turn) {
  if (threadIdx.x != 0)
    return;
  const uint64_t start = clock64();
  while (true) {
    bool complete = turn == nullptr ||
        load_acquire_device(turn) == static_cast<uint64_t>(expected_turn);
    for (int index = 0; index < ready_count && complete; ++index)
      complete = load_acquire_system(ready + index) ==
          static_cast<uint64_t>(expected_ready);
    if (complete)
      return;
    if (clock64() - start > timeout_cycles) {
      timeout_record[0] = 1;
      timeout_record[1] = ready_count;
      __threadfence_system();
      asm volatile("trap;");
      return;
    }
    __nanosleep(64);
  }
}

template <int kHeadDim>
void launch_quantize_publish_o_fp8(
    const tvm::ffi::TensorView o,
    const tvm::ffi::TensorView staged_o,
    const tvm::ffi::TensorView staged_scales,
    const tvm::ffi::TensorView desc,
    const tvm::ffi::TensorView ready_desc,
    const tvm::ffi::TensorView completion_counter,
    const tvm::ffi::TensorView* quantization_counter,
    const tvm::ffi::TensorView* turn,
    int64_t slot,
    int64_t destination_source_stride,
    int64_t next_turn,
    int64_t ready_value) {
  using namespace host;
  static_assert(kHeadDim == 128);
  auto device = SymbolicDevice{};
  auto rows = SymbolicSize{"rows"};
  auto width = SymbolicSize{"width"};
  auto heads = SymbolicSize{"heads"};
  auto edges = SymbolicSize{"edges"};
  auto ready_edges = SymbolicSize{"ready_edges"};
  auto data_dtype = SymbolicDType{};
  auto staged_dtype = SymbolicDType{};
  TensorMatcher({rows, width})
      .with_strides({-1, 1})
      .with_dtype(data_dtype)
      .with_device<kDLCUDA>(device)
      .verify(o);
  TensorMatcher({rows, width})
      .with_strides({-1, 1})
      .with_dtype(staged_dtype)
      .with_device<kDLCUDA>(device)
      .verify(staged_o);
  TensorMatcher({rows, heads})
      .with_strides({-1, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(staged_scales);
  TensorMatcher({edges, 10})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(desc);
  TensorMatcher({ready_edges, 3})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(ready_desc);
  TensorMatcher({1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(completion_counter);
  const bool release_turn = turn != nullptr;
  RuntimeCheck(
      release_turn == (quantization_counter != nullptr),
      "fused O quantization turn release requires its completion counter");
  if (release_turn) {
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(*quantization_counter);
    TensorMatcher({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(*turn);
  }
  const auto dtype = data_dtype.unwrap();
  const auto output_dtype = staged_dtype.unwrap();
  RuntimeCheck(
      dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
      dtype.lanes == 1,
      "fused AFD O quantization requires BF16 input");
  RuntimeCheck(
      output_dtype.bits == 8 and output_dtype.lanes == 1,
      "fused AFD O staging requires an 8-bit payload");
  RuntimeCheck(heads.unwrap() * kHeadDim == width.unwrap());
  RuntimeCheck(width.unwrap() % (4 * kHeadDim) == 0);
  RuntimeCheck(
      rows.unwrap() > 0 and edges.unwrap() > 0 and ready_edges.unwrap() > 0);
  RuntimeCheck(
      slot >= 0 and slot < kTransportSlots and
      destination_source_stride > 0 and ready_value > 0 and
      (!release_turn or next_turn >= 0));
  const QuantizePublishOFp8Params params{
      .o = static_cast<const __nv_bfloat16*>(o.data_ptr()),
      .staged_o = static_cast<uint8_t*>(staged_o.data_ptr()),
      .staged_scales = static_cast<uint8_t*>(staged_scales.data_ptr()),
      .desc = static_cast<const int64_t*>(desc.data_ptr()),
      .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
      .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
      .quantization_counter = release_turn
          ? static_cast<int*>(quantization_counter->data_ptr())
          : nullptr,
      .turn = release_turn ? static_cast<int64_t*>(turn->data_ptr()) : nullptr,
      .ready_value = ready_value,
      .next_turn = next_turn,
      .o_stride = o.stride(0),
      .staged_o_stride = staged_o.stride(0),
      .staged_scale_stride = staged_scales.stride(0),
      .edges = static_cast<int>(edges.unwrap()),
      .ready_edges = static_cast<int>(ready_edges.unwrap()),
      .slot = static_cast<int>(slot),
      .rows = static_cast<int>(rows.unwrap()),
      .destination_source_stride = static_cast<int>(destination_source_stride),
  };
  const int64_t tasks =
      std::max<int64_t>(1, rows.unwrap() * width.unwrap() / 16);
  const LaunchKernel launch(
      publication_blocks(tasks), kThreads, device.unwrap());
  const bool single_ready = ready_edges.unwrap() == 1;
  const bool single_edge = single_ready and edges.unwrap() == 1;
  if (release_turn) {
    if (single_edge)
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, true, true, true>, params);
    else if (single_ready)
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, false, true, true>, params);
    else
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, false, false, true>, params);
  } else {
    if (single_edge)
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, true, true, false>, params);
    else if (single_ready)
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, false, true, false>, params);
    else
      launch(
          quantize_publish_o_fp8_kernel<kHeadDim, false, false, false>, params);
  }
}

template <int kHeadDim>
struct AfdFmhaTransportKernel {
  static void publish_qkv(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView source_offsets,
      const tvm::ffi::TensorView q_desc,
      const tvm::ffi::TensorView kv_desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      int64_t layer,
      int64_t slot,
      int64_t ready_value) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto q_width = SymbolicSize{"q_width"};
    auto kv_width = SymbolicSize{"kv_width"};
    auto q_edges = SymbolicSize{"q_edges"};
    auto kv_edges = SymbolicSize{"kv_edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    auto source_offset_count = SymbolicSize{"source_offset_count"};
    auto loc_dtype = SymbolicDType{};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, q_width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(q);
    TensorMatcher({rows, kv_width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(k)
        .verify(v);
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
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
        dtype.lanes == 1,
        "AFD FMHA transport requires BF16 payloads");
    RuntimeCheck(q_edges.unwrap() > 0 and source_offset_count.unwrap() > 1);
    RuntimeCheck(ready_edges.unwrap() > 0);
    RuntimeCheck(
        layer >= 0 and slot >= 0 and slot < kTransportSlots and ready_value > 0);
    const PublishQKVParams params{
        .q = static_cast<const __nv_bfloat16*>(q.data_ptr()),
        .k = static_cast<const __nv_bfloat16*>(k.data_ptr()),
        .v = static_cast<const __nv_bfloat16*>(v.data_ptr()),
        .out_loc = out_loc.data_ptr(),
        .source_offsets = static_cast<const int64_t*>(source_offsets.data_ptr()),
        .q_desc = static_cast<const int64_t*>(q_desc.data_ptr()),
        .kv_desc = static_cast<const int64_t*>(kv_desc.data_ptr()),
        .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
        .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
        .ready_value = ready_value,
        .q_stride = q.stride(0),
        .k_stride = k.stride(0),
        .v_stride = v.stride(0),
        .q_edges = static_cast<int>(q_edges.unwrap()),
        .kv_edges = static_cast<int>(kv_edges.unwrap()),
        .ready_edges = static_cast<int>(ready_edges.unwrap()),
        .layer = static_cast<int>(layer),
        .slot = static_cast<int>(slot),
        .rows = static_cast<int>(rows.unwrap()),
        .loc_i64 = loc_dtype.unwrap().bits == 64,
    };
    const int64_t tasks = std::max<int64_t>(
        1,
        rows.unwrap() * std::max(q_width.unwrap(), 2 * kv_width.unwrap()) / 4);
    const LaunchKernel launch(
        publication_blocks(tasks), kThreads, device.unwrap());
    const bool single_ready = ready_edges.unwrap() == 1;
    const bool single_edge =
        single_ready and source_offset_count.unwrap() == 2 and
        q_edges.unwrap() == 1 and kv_edges.unwrap() == 1;
    if (single_edge)
      launch(publish_qkv_kernel<kHeadDim, true, true>, params);
    else if (single_ready)
      launch(publish_qkv_kernel<kHeadDim, false, true>, params);
    else
      launch(publish_qkv_kernel<kHeadDim, false, false>, params);
  }

  static void publish_o(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t ready_value) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto edges = SymbolicSize{"edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(o);
    TensorMatcher({edges, 7})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(desc);
    TensorMatcher({ready_edges, 3})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ready_desc);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(completion_counter);
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
        dtype.lanes == 1,
        "AFD FMHA transport requires BF16 payloads");
    RuntimeCheck(edges.unwrap() > 0 and ready_edges.unwrap() > 0);
    RuntimeCheck(
        slot >= 0 and slot < kTransportSlots and
        destination_source_stride > 0 and ready_value > 0);
    const PublishOParams params{
        .o = static_cast<const __nv_bfloat16*>(o.data_ptr()),
        .desc = static_cast<const int64_t*>(desc.data_ptr()),
        .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
        .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
        .ready_value = ready_value,
        .o_stride = o.stride(0),
        .edges = static_cast<int>(edges.unwrap()),
        .ready_edges = static_cast<int>(ready_edges.unwrap()),
        .slot = static_cast<int>(slot),
        .rows = static_cast<int>(rows.unwrap()),
        .destination_source_stride = static_cast<int>(destination_source_stride),
    };
    const int64_t tasks = std::max<int64_t>(1, rows.unwrap() * width.unwrap() / 4);
    const LaunchKernel launch(
        publication_blocks(tasks), kThreads, device.unwrap());
    const bool single_ready = ready_edges.unwrap() == 1;
    const bool single_edge = single_ready and edges.unwrap() == 1;
    if (single_edge)
      launch(publish_o_kernel<kHeadDim, true, true>, params);
    else if (single_ready)
      launch(publish_o_kernel<kHeadDim, false, true>, params);
    else
      launch(publish_o_kernel<kHeadDim, false, false>, params);
  }

  static void publish_o_release_turn(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      const tvm::ffi::TensorView turn,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t next_turn,
      int64_t ready_value) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto edges = SymbolicSize{"edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(o);
    TensorMatcher({edges, 7})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(desc);
    TensorMatcher({ready_edges, 3})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ready_desc);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(completion_counter);
    TensorMatcher({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(turn);
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.code == DLDataTypeCode::kDLBfloat and dtype.bits == 16 and
        dtype.lanes == 1,
        "AFD FMHA transport requires BF16 payloads");
    RuntimeCheck(edges.unwrap() > 0 and ready_edges.unwrap() > 0);
    RuntimeCheck(
        slot >= 0 and slot < kTransportSlots and
        destination_source_stride > 0 and next_turn >= 0 and ready_value > 0);
    const PublishOReleaseTurnParams params{
        .publish = PublishOParams{
            .o = static_cast<const __nv_bfloat16*>(o.data_ptr()),
            .desc = static_cast<const int64_t*>(desc.data_ptr()),
            .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
            .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
            .ready_value = ready_value,
            .o_stride = o.stride(0),
            .edges = static_cast<int>(edges.unwrap()),
            .ready_edges = static_cast<int>(ready_edges.unwrap()),
            .slot = static_cast<int>(slot),
            .rows = static_cast<int>(rows.unwrap()),
            .destination_source_stride = static_cast<int>(destination_source_stride),
        },
        .turn = static_cast<int64_t*>(turn.data_ptr()),
        .next_turn = next_turn,
    };
    const int64_t tasks =
        std::max<int64_t>(1, rows.unwrap() * width.unwrap() / 4);
    const LaunchKernel launch(
        publication_blocks(tasks), kThreads, device.unwrap());
    const bool single_ready = ready_edges.unwrap() == 1;
    const bool single_edge = single_ready and edges.unwrap() == 1;
    if (single_edge)
      launch(publish_o_kernel_release_turn<kHeadDim, true, true>, params);
    else if (single_ready)
      launch(publish_o_kernel_release_turn<kHeadDim, false, true>, params);
    else
      launch(publish_o_kernel_release_turn<kHeadDim, false, false>, params);
  }

  static void publish_o_fp8(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView scales,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t ready_value) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto packed_groups = SymbolicSize{"packed_groups"};
    auto edges = SymbolicSize{"edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(o);
    TensorMatcher({rows, packed_groups})
        .with_strides({1, -1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(scales);
    TensorMatcher({edges, 10})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(desc);
    TensorMatcher({ready_edges, 3})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ready_desc);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(completion_counter);
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.bits == 8 and dtype.lanes == 1,
        "AFD FMHA FP8 O transport requires an 8-bit payload");
    RuntimeCheck(width.unwrap() % kHeadDim == 0);
    RuntimeCheck(
        packed_groups.unwrap() ==
        (width.unwrap() / kHeadDim + 3) / 4);
    RuntimeCheck(scales.stride(0) == 1);
    RuntimeCheck(scales.stride(1) == (rows.unwrap() + 3) / 4 * 4);
    RuntimeCheck(edges.unwrap() > 0 and ready_edges.unwrap() > 0);
    RuntimeCheck(
        slot >= 0 and slot < kTransportSlots and
        destination_source_stride > 0 and ready_value > 0);
    const PublishOFp8Params params{
        .o = static_cast<const uint8_t*>(o.data_ptr()),
        .scales = static_cast<const int32_t*>(scales.data_ptr()),
        .desc = static_cast<const int64_t*>(desc.data_ptr()),
        .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
        .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
        .ready_value = ready_value,
        .o_stride = o.stride(0),
        .scale_stride_row = scales.stride(0),
        .scale_stride_group = scales.stride(1),
        .edges = static_cast<int>(edges.unwrap()),
        .ready_edges = static_cast<int>(ready_edges.unwrap()),
        .slot = static_cast<int>(slot),
        .rows = static_cast<int>(rows.unwrap()),
        .destination_source_stride = static_cast<int>(destination_source_stride),
    };
    const int64_t tasks = std::max<int64_t>(
        1,
        std::max<int64_t>(
            rows.unwrap() * width.unwrap() / 16,
            rows.unwrap() * packed_groups.unwrap()));
    const LaunchKernel launch(
        publication_blocks(tasks), kThreads, device.unwrap());
    const bool single_ready = ready_edges.unwrap() == 1;
    const bool single_edge = single_ready and edges.unwrap() == 1;
    if (single_edge)
      launch(publish_o_fp8_kernel<kHeadDim, true, true>, params);
    else if (single_ready)
      launch(publish_o_fp8_kernel<kHeadDim, false, true>, params);
    else
      launch(publish_o_fp8_kernel<kHeadDim, false, false>, params);
  }

  static void publish_o_fp8_release_turn(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView scales,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      const tvm::ffi::TensorView turn,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t next_turn,
      int64_t ready_value) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto packed_groups = SymbolicSize{"packed_groups"};
    auto edges = SymbolicSize{"edges"};
    auto ready_edges = SymbolicSize{"ready_edges"};
    auto data_dtype = SymbolicDType{};
    TensorMatcher({rows, width})
        .with_strides({-1, 1})
        .with_dtype(data_dtype)
        .with_device<kDLCUDA>(device)
        .verify(o);
    TensorMatcher({rows, packed_groups})
        .with_strides({1, -1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(scales);
    TensorMatcher({edges, 10})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(desc);
    TensorMatcher({ready_edges, 3})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ready_desc);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(completion_counter);
    TensorMatcher({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(turn);
    const auto dtype = data_dtype.unwrap();
    RuntimeCheck(
        dtype.bits == 8 and dtype.lanes == 1,
        "AFD FMHA FP8 O transport requires an 8-bit payload");
    RuntimeCheck(width.unwrap() % kHeadDim == 0);
    RuntimeCheck(
        packed_groups.unwrap() ==
        (width.unwrap() / kHeadDim + 3) / 4);
    RuntimeCheck(scales.stride(0) == 1);
    RuntimeCheck(scales.stride(1) == (rows.unwrap() + 3) / 4 * 4);
    RuntimeCheck(edges.unwrap() > 0 and ready_edges.unwrap() > 0);
    RuntimeCheck(
        slot >= 0 and slot < kTransportSlots and
        destination_source_stride > 0 and next_turn >= 0 and ready_value > 0);
    const PublishOFp8ReleaseTurnParams params{
        .publish = PublishOFp8Params{
            .o = static_cast<const uint8_t*>(o.data_ptr()),
            .scales = static_cast<const int32_t*>(scales.data_ptr()),
            .desc = static_cast<const int64_t*>(desc.data_ptr()),
            .ready_desc = static_cast<const int64_t*>(ready_desc.data_ptr()),
            .completion_counter = static_cast<int*>(completion_counter.data_ptr()),
            .ready_value = ready_value,
            .o_stride = o.stride(0),
            .scale_stride_row = scales.stride(0),
            .scale_stride_group = scales.stride(1),
            .edges = static_cast<int>(edges.unwrap()),
            .ready_edges = static_cast<int>(ready_edges.unwrap()),
            .slot = static_cast<int>(slot),
            .rows = static_cast<int>(rows.unwrap()),
            .destination_source_stride =
                static_cast<int>(destination_source_stride),
        },
        .turn = static_cast<int64_t*>(turn.data_ptr()),
        .next_turn = next_turn,
    };
    const int64_t tasks = std::max<int64_t>(
        1,
        std::max<int64_t>(
            rows.unwrap() * width.unwrap() / 16,
            rows.unwrap() * packed_groups.unwrap()));
    const LaunchKernel launch(
        publication_blocks(tasks), kThreads, device.unwrap());
    const bool single_ready = ready_edges.unwrap() == 1;
    const bool single_edge = single_ready and edges.unwrap() == 1;
    if (single_edge)
      launch(publish_o_fp8_kernel_release_turn<kHeadDim, true, true>, params);
    else if (single_ready)
      launch(publish_o_fp8_kernel_release_turn<kHeadDim, false, true>, params);
    else
      launch(publish_o_fp8_kernel_release_turn<kHeadDim, false, false>, params);
  }

  static void quantize_publish_o_fp8(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView staged_o,
      const tvm::ffi::TensorView staged_scales,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t ready_value) {
    launch_quantize_publish_o_fp8<kHeadDim>(
        o,
        staged_o,
        staged_scales,
        desc,
        ready_desc,
        completion_counter,
        nullptr,
        nullptr,
        slot,
        destination_source_stride,
        0,
        ready_value);
  }

  static void quantize_publish_o_fp8_release_turn(
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView staged_o,
      const tvm::ffi::TensorView staged_scales,
      const tvm::ffi::TensorView desc,
      const tvm::ffi::TensorView ready_desc,
      const tvm::ffi::TensorView completion_counter,
      const tvm::ffi::TensorView quantization_counter,
      const tvm::ffi::TensorView turn,
      int64_t slot,
      int64_t destination_source_stride,
      int64_t next_turn,
      int64_t ready_value) {
    launch_quantize_publish_o_fp8<kHeadDim>(
        o,
        staged_o,
        staged_scales,
        desc,
        ready_desc,
        completion_counter,
        &quantization_counter,
        &turn,
        slot,
        destination_source_stride,
        next_turn,
        ready_value);
  }

  static void wait_turn(
      const tvm::ffi::TensorView turn,
      const tvm::ffi::TensorView timeout_record,
      int64_t timeout_ms,
      int64_t expected_turn,
      int64_t next_turn) {
    using namespace host;
    auto device = SymbolicDevice{};
    TensorMatcher({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(turn);
    TensorMatcher({2})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(timeout_record);
    RuntimeCheck(timeout_ms > 0 and expected_turn >= 0 and next_turn >= 0);
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clock_khz, cudaDevAttrClockRate, device.unwrap().device_id));
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(clock_khz) * static_cast<uint64_t>(timeout_ms);
    LaunchKernel(1, kThreads, device.unwrap())(
        wait_turn_kernel,
        static_cast<int64_t*>(turn.data_ptr()),
        timeout_cycles,
        static_cast<int64_t*>(timeout_record.data_ptr()),
        expected_turn,
        next_turn);
  }

  static void wait_ready(
      int64_t ready_ptr,
      int64_t ready_count,
      const tvm::ffi::TensorView timeout_record,
      int64_t timeout_ms,
      int64_t expected_ready) {
    using namespace host;
    auto device = SymbolicDevice{};
    TensorMatcher({2})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(timeout_record);
    RuntimeCheck(
        ready_ptr > 0 and ready_count > 0 and
        timeout_ms > 0 and expected_ready > 0);
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clock_khz, cudaDevAttrClockRate, device.unwrap().device_id));
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(clock_khz) * static_cast<uint64_t>(timeout_ms);
    LaunchKernel(1, kSignalThreads, device.unwrap())(
        wait_ready_kernel,
        reinterpret_cast<const int64_t*>(ready_ptr),
        static_cast<int>(ready_count),
        timeout_cycles,
        static_cast<int64_t*>(timeout_record.data_ptr()),
        expected_ready,
        nullptr,
        0);
  }

  static void wait_ready_turn(
      int64_t ready_ptr,
      int64_t ready_count,
      const tvm::ffi::TensorView timeout_record,
      const tvm::ffi::TensorView turn,
      int64_t timeout_ms,
      int64_t expected_turn,
      int64_t expected_ready) {
    using namespace host;
    auto device = SymbolicDevice{};
    TensorMatcher({2})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(timeout_record);
    TensorMatcher({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(turn);
    RuntimeCheck(
        ready_ptr > 0 and ready_count > 0 and timeout_ms > 0 and
        expected_turn >= 0 and expected_ready > 0);
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clock_khz, cudaDevAttrClockRate, device.unwrap().device_id));
    const uint64_t timeout_cycles =
        static_cast<uint64_t>(clock_khz) * static_cast<uint64_t>(timeout_ms);
    LaunchKernel(1, kSignalThreads, device.unwrap())(
        wait_ready_kernel,
        reinterpret_cast<const int64_t*>(ready_ptr),
        static_cast<int>(ready_count),
        timeout_cycles,
        static_cast<int64_t*>(timeout_record.data_ptr()),
        expected_ready,
        static_cast<const int64_t*>(turn.data_ptr()),
        expected_turn);
  }
};

}  // namespace
