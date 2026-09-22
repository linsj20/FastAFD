#pragma once

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

// Prepare the next activation bank while the following producer computes.
// Its output consumer depends directly on the producer, not this maintenance grid.
template <bool kTrace>
__device__ __forceinline__ void mega_moe_recycle_worker(
    const layout::Workspace& workspace, const int64_t* peer_bases,
    uint32_t hidden, uint32_t reset_rows,
    uint32_t worker, uint64_t* timings, uint32_t num_tokens, uint32_t num_sms) {
    constexpr uint32_t kResetCTAs = 8;
    const uint32_t epoch = ptx::ld_acq(workspace.get_nvl_barrier_counter_ptr());
    const uint32_t target = epoch + 1;
    // Serialize reuse of the join counter with the previous maintenance grid.
    // Every thread latches epoch before any CTA can release the following main;
    // the main cannot advance metadata until all these CTAs have participated.
    if (threadIdx.x == 0) {
        while (ptx::ld_acq_sys(workspace.get_payload_ready_ptr()) != epoch);
        const auto offset = reinterpret_cast<uintptr_t>(workspace.get_payload_ready_ptr()) -
                            reinterpret_cast<uintptr_t>(workspace.base);
        for (uint32_t peer = worker; peer < workspace.num_ranks; peer += kResetCTAs) {
            const auto* ready = reinterpret_cast<const uint32_t*>(
                static_cast<uintptr_t>(peer_bases[peer]) + offset);
            while ((ptx::ld_acq_sys(ready) - epoch) & 0x80000000u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Join destination readiness with the selector's input-row releases.
        // Sender acquire of rows+8 transitively orders every destination clear.
        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" ::
            "l"(workspace.get_genphase_flag_ptr(layout::Workspace::CountsReady)),
            "r"(1u) : "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const uint32_t trace_slot = 24 + 3 * num_tokens + num_sms + 2 * worker;
    if constexpr (kTrace) {
        if (threadIdx.x == 0) {
            uint64_t stamp;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
            timings[trace_slot] = stamp;
            if (worker == 0) timings[22] = stamp;
        }
        __syncthreads();
    }
    auto* data = static_cast<uint4*>(workspace.get_compact_frame_base_ptr(hidden, workspace.num_topk));
    const uint32_t kVectorsPerRow = hidden / sizeof(uint4);
    // Ordered per-arena calls have retired the prior local producer using this
    // bank. Current A/SF live in the opposite bank; SF is never cleared here.
    for (uint32_t i = worker * blockDim.x + threadIdx.x;
         i < workspace.num_ranks * reset_rows * kVectorsPerRow; i += kResetCTAs * blockDim.x) {
        const uint32_t row = i / kVectorsPerRow;
        const uint64_t source_row =
            (static_cast<uint64_t>(target & 1u) * workspace.num_ranks + row / reset_rows) *
            workspace.num_max_tokens_per_rank + row % reset_rows;
        data[source_row * kVectorsPerRow + i % kVectorsPerRow] =
            make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
    }
    __syncthreads();
    if constexpr (kTrace) {
        if (threadIdx.x == 0)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[trace_slot + 1]) :: "memory");
    }
    if (threadIdx.x == 0 && ptx::atomic_add_acqrel_gpu(
            workspace.get_reset_retired_ptr(), 1) == kResetCTAs - 1) {
        *workspace.get_reset_retired_ptr() = 0;
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::
            "l"(workspace.get_payload_ready_ptr()), "r"(target) : "memory");
        if constexpr (kTrace)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[23]) :: "memory");
    }
}

} // namespace deep_gemm
