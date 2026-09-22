#pragma once

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

#ifndef MEGAMOE_RECYCLE_CTAS
#define MEGAMOE_RECYCLE_CTAS 8
#endif
inline constexpr uint32_t kMegaMoERecycleCTAs = MEGAMOE_RECYCLE_CTAS;
static_assert(kMegaMoERecycleCTAs == 1 || kMegaMoERecycleCTAs == 4 ||
              kMegaMoERecycleCTAs == 8, "Reset requires one, four or eight CTAs");

// Prepare the next activation bank while the following producer computes.
// Its output consumer depends directly on the producer, not this maintenance grid.
template <bool kTrace>
__device__ __forceinline__ void mega_moe_recycle_worker(
    const layout::Workspace& workspace, const int64_t* peer_bases,
    uint32_t hidden, uint32_t reset_rows,
    uint32_t worker, uint64_t* timings, uint32_t num_tokens, uint32_t num_sms) {
    constexpr uint32_t kResetCTAs = kMegaMoERecycleCTAs;
    const uint32_t epoch = ptx::ld_acq(workspace.get_nvl_barrier_counter_ptr());
    const uint32_t target = epoch + 1;
    // Serialize reuse of the join counter with the previous maintenance grid.
    // Every thread latches epoch before any CTA can release the following main;
    // the main cannot advance metadata until all these CTAs have participated.
    if (threadIdx.x == 0)
        while (ptx::ld_acq_sys(workspace.get_payload_ready_ptr()) != epoch);
    if (threadIdx.x < 32) {
        const auto offset = reinterpret_cast<uintptr_t>(workspace.get_payload_ready_ptr()) -
                            reinterpret_cast<uintptr_t>(workspace.base);
        for (uint32_t peer = worker + threadIdx.x * kResetCTAs;
             peer < workspace.num_ranks; peer += 32 * kResetCTAs) {
            const auto* ready = reinterpret_cast<const uint32_t*>(
                static_cast<uintptr_t>(peer_bases[peer]) + offset);
            while ((ptx::ld_acq_sys(ready) - epoch) & 0x80000000u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Join destination readiness with the selector's input-row releases.
        // Keep the main's rows+8 signal independent of reset-worker count.
        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" ::
            "l"(workspace.get_genphase_flag_ptr(layout::Workspace::CountsReady)),
            "r"(8u / kResetCTAs) : "memory");
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
    if (threadIdx.x == 0 && (kResetCTAs == 1 || ptx::atomic_add_acqrel_gpu(
            workspace.get_reset_retired_ptr(), 1) == kResetCTAs - 1)) {
        if constexpr (kResetCTAs != 1)
            *workspace.get_reset_retired_ptr() = 0;
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::
            "l"(workspace.get_payload_ready_ptr()), "r"(target) : "memory");
        if constexpr (kTrace)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[23]) :: "memory");
    }
}

// Prepare the other arena one layer earlier, retaining exact per-generation
// upkeep for repeated-layer calls. The ordinary gate retires all preceding
// maintenance and arena uses before this selector starts.
template <bool kTrace>
__device__ __forceinline__ void mega_moe_recycle_ahead_worker(
    const layout::Workspace& current, const int64_t* current_peers,
    const layout::Workspace& ahead, const int64_t* ahead_peers,
    uint32_t hidden, uint32_t reset_rows, uint32_t worker,
    uint64_t* timings, uint32_t num_tokens, uint32_t num_sms) {
    constexpr uint32_t kWorkers = kMegaMoERecycleCTAs;
#ifdef MEGAMOE_RECYCLE_TMA
    __shared__ __align__(16) uint4 reset_tile[256];
#endif
    const auto maintain = [&](const layout::Workspace& ws, const int64_t* peers,
                              bool admit_current) {
        // Current main can advance only after every worker has latched this
        // generation. The other arena cannot advance until the next gate.
        const uint32_t epoch = ptx::ld_acq(ws.get_nvl_barrier_counter_ptr());
        const uint32_t target = epoch + 1;
        uint32_t ready;
        do { ready = ptx::ld_acq_sys(ws.get_payload_ready_ptr()); }
        while ((ready - epoch) & 0x80000000u);
        const bool clear = (ready - target) & 0x80000000u;
        if (threadIdx.x < 32 && (admit_current || clear)) {
            const auto offset = reinterpret_cast<uintptr_t>(ws.get_payload_ready_ptr()) -
                                reinterpret_cast<uintptr_t>(ws.base);
            for (uint32_t peer = worker + threadIdx.x * kWorkers;
                 peer < ws.num_ranks; peer += 32 * kWorkers) {
                const auto* flag = reinterpret_cast<const uint32_t*>(
                    static_cast<uintptr_t>(peers[peer]) + offset);
                while ((ptx::ld_acq_sys(flag) - epoch) & 0x80000000u);
            }
        }
        __syncthreads();
        if (admit_current && threadIdx.x == 0) {
            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" ::
                "l"(ws.get_genphase_flag_ptr(layout::Workspace::CountsReady)),
                "r"(8u / kWorkers) : "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        const uint32_t slot = admit_current ? 24 + 3 * num_tokens + num_sms : 2;
        const uint32_t issuer = admit_current ? 22 : 0;
        if constexpr (kTrace) {
            if (threadIdx.x == 0) {
                uint64_t stamp;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
                timings[slot + 2 * worker] = stamp;
                if (worker == 0) {
                    timings[issuer] = stamp;
                    timings[admit_current ? 18 : 19] = target;
                    if (admit_current) timings[20] = clear ? 1 : 0;
                    else timings[20] |= clear ? 2 : 0;
                }
            }
            __syncthreads();
        }
#ifdef MEGAMOE_RECYCLE_TMA
        if (clear) {
            for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x)
                reset_tile[i] = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
            // Order every thread's generic shared stores before thread0's TMA.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (threadIdx.x == 0) {
                auto* data = static_cast<uint8_t*>(ws.get_compact_frame_base_ptr(hidden, ws.num_topk));
                const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(reset_tile));
                const uint32_t prefix_bytes = reset_rows * hidden;
                for (uint32_t peer = worker; peer < ws.num_ranks; peer += kWorkers) {
                    const uint64_t base =
                        (static_cast<uint64_t>(target & 1u) * ws.num_ranks + peer) *
                        ws.num_max_tokens_per_rank * hidden;
                    for (uint32_t offset = 0; offset < prefix_bytes; offset += sizeof(reset_tile)) {
                        const uint32_t bytes = min(static_cast<uint32_t>(sizeof(reset_tile)), prefix_bytes - offset);
                        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                            "l"(data + base + offset), "r"(shared), "r"(bytes) : "memory");
                    }
                }
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                // Full completion, including destination writes; never .read.
                // Completion supplies the implicit async-to-generic proxy fence.
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
        }
#else
        if (clear) {
            auto* data = static_cast<uint4*>(ws.get_compact_frame_base_ptr(hidden, ws.num_topk));
            const uint32_t vectors = hidden / sizeof(uint4);
            // Each peer prefix is contiguous; no division/modulo per store.
            for (uint32_t peer = worker; peer < ws.num_ranks; peer += kWorkers) {
                const uint64_t base =
                    (static_cast<uint64_t>(target & 1u) * ws.num_ranks + peer) *
                    ws.num_max_tokens_per_rank * vectors;
                for (uint32_t i = threadIdx.x; i < reset_rows * vectors; i += blockDim.x)
                    data[base + i] = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
            }
        }
#endif
        __syncthreads();
        if constexpr (kTrace) {
            if (threadIdx.x == 0)
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[slot + 2 * worker + 1]) :: "memory");
        }
        if (clear && threadIdx.x == 0 && (kWorkers == 1 ||
                ptx::atomic_add_acqrel_gpu(ws.get_reset_retired_ptr(), 1) == kWorkers - 1)) {
            if constexpr (kWorkers != 1) *ws.get_reset_retired_ptr() = 0;
            asm volatile("st.release.sys.global.u32 [%0], %1;" ::
                "l"(ws.get_payload_ready_ptr()), "r"(target) : "memory");
            if constexpr (kTrace)
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[issuer + 1]) :: "memory");
        } else if constexpr (kTrace) {
            // A skipped clear observes an earlier publication. Distinguish
            // that observation explicitly with the action bits in slot20.
            if (!clear && worker == 0 && threadIdx.x == 0)
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timings[issuer + 1]) :: "memory");
        }
    };
    maintain(current, current_peers, true);
    maintain(ahead, ahead_peers, false);
}

} // namespace deep_gemm
