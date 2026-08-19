from __future__ import annotations

from typing import Any

import ray

from .afd_protocol import AfdProfilerCmd, AfdProfilerReply
from .afd_support import log_line


def start_nsys_runtime_capture(
    coord: Any,
    *,
    sync: bool,
    reason: str,
    via_worker_queue: bool = False,
) -> None:
    if not coord.server_args.ray_nsys or coord._nsys_runtime_started:
        return
    if via_worker_queue:
        coord._broadcast_cmd_to_workers(
            AfdProfilerCmd(action="start", sync=bool(sync), ack=True)
        )
        expected = len(coord._cmd_to_workers)
        acknowledged: set[int] = set()
        while len(acknowledged) < expected:
            for reply in coord._recv_worker_inbox():
                if isinstance(reply, AfdProfilerReply):
                    if reply.action != "start":
                        raise RuntimeError(
                            "Unexpected profiler acknowledgment while starting "
                            f"Nsight: {reply.action!r}"
                        )
                    worker_rank = int(reply.worker_rank)
                    if worker_rank in acknowledged:
                        raise RuntimeError(
                            f"Duplicate profiler-start acknowledgment from {worker_rank}"
                        )
                    acknowledged.add(worker_rank)
                    continue
                coord._afd_store_step_reply(
                    reply,
                    attn_tp=int(coord.attn_tp_size),
                )
        if len(acknowledged) != expected:
            raise RuntimeError(
                f"Profiler-start barrier mismatch: {len(acknowledged)} != {expected}"
            )
    else:
        ray.get(
            [
                worker.start_cuda_profiler.remote(sync=bool(sync))
                for worker in coord._all_workers()
            ]
        )
    coord._nsys_runtime_started = True
    log_line(
        coord.log_path,
        f"[afd-coordinator] nsys profiler:start reason={reason} "
        f"sync={int(bool(sync))} via_worker_queue={int(bool(via_worker_queue))}",
        flush=True,
    )


def stop_nsys_runtime_capture(
    coord: Any,
    *,
    sync: bool,
    reason: str,
    via_worker_queue: bool = False,
) -> None:
    if not coord.server_args.ray_nsys or not coord._nsys_runtime_started:
        return
    if via_worker_queue:
        coord._broadcast_cmd_to_workers(
            AfdProfilerCmd(action="stop", sync=bool(sync))
        )
    else:
        ray.get(
            [
                worker.stop_cuda_profiler.remote(sync=bool(sync))
                for worker in coord._all_workers()
            ]
        )
    coord._nsys_runtime_started = False
    log_line(
        coord.log_path,
        f"[afd-coordinator] nsys profiler:stop reason={reason} "
        f"sync={int(bool(sync))} via_worker_queue={int(bool(via_worker_queue))}",
        flush=True,
    )


def prepare_nsys_runtime_capture_for_decode_step(
    coord: Any,
    step_id: int,
    *,
    phase: str,
    real_batch_sizes: tuple[int, ...],
) -> bool:
    """Trace the first target decode, then select the following 15 launches.

    This is deliberately state-driven.  It never predicts a global scheduler
    step from prompt length or prefill chunking.  The first full-resident decode
    is retained as the cold-transition warmup diagnostic.  The next 15 exact
    target-batch decode launches are the stability-guarded measurement window.
    """
    if not coord.server_args.ray_nsys:
        return False
    capture_steps = int(coord._nsys_capture_decode_steps)
    captured = coord._nsys_capture_step_ids
    traced = coord._nsys_trace_step_ids
    trace_steps = capture_steps + 1
    if len(traced) >= trace_steps:
        return False

    target = int(coord._nsys_target_batch_per_dp)
    is_target_decode = (
        str(phase) == "decode"
        and bool(real_batch_sizes)
        and all(int(size) == target for size in real_batch_sizes)
    )
    if not coord._nsys_runtime_started:
        if not is_target_decode:
            return False
        start_nsys_runtime_capture(
            coord,
            sync=True,
            reason=(
                f"first_target_decode step={int(step_id)} "
                f"target_batch_per_dp={target} capture_steps={capture_steps}"
            ),
            via_worker_queue=True,
        )
    elif not is_target_decode:
        raise RuntimeError(
            "Nsight target-decode capture encountered a non-target step: "
            f"step={int(step_id)} phase={phase} real_batch_sizes={real_batch_sizes} "
            f"target_batch_per_dp={target}"
        )

    traced.append(int(step_id))
    if len(traced) > 1:
        captured.append(int(step_id))
    return True


def finish_nsys_runtime_capture_step_launch(
    coord: Any,
    step_id: int,
    *,
    included: bool,
) -> None:
    """Stop after one traced warmup plus 15 measured target-step commands.

    Worker commands are FIFO, so placing stop after step 15 and before any
    step-16 command captures exactly the intended window even when the
    coordinator keeps multiple inference steps in flight.
    """
    if not included:
        return
    captured = coord._nsys_capture_step_ids
    traced = coord._nsys_trace_step_ids
    if not traced or int(traced[-1]) != int(step_id):
        raise RuntimeError(
            f"Nsight capture launch accounting mismatch at step {int(step_id)}"
        )
    capture_steps = int(coord._nsys_capture_decode_steps)
    trace_steps = capture_steps + 1
    if len(traced) < trace_steps:
        return
    if len(traced) != trace_steps or len(captured) != capture_steps:
        raise RuntimeError(
            "Nsight capture accounting mismatch: "
            f"traced={len(traced)}/{trace_steps} "
            f"measured={len(captured)}/{capture_steps}"
        )
    target = int(coord._nsys_target_batch_per_dp)
    step_ids = ",".join(str(value) for value in captured)
    log_line(
        coord.log_path,
        f"[afd-coordinator] nsys profiler:target_decode_window "
        f"target_batch_per_dp={target} warmup_step_id={traced[0]} "
        f"step_ids={step_ids} count={len(captured)} trace_count={len(traced)}",
        flush=True,
    )
    stop_nsys_runtime_capture(
        coord,
        sync=True,
        reason=(
            f"target_decode_window_queued warmup_step={traced[0]} "
            f"first_step={captured[0]} "
            f"last_step={captured[-1]} count={len(captured)}"
        ),
        via_worker_queue=True,
    )


__all__ = [
    "finish_nsys_runtime_capture_step_launch",
    "prepare_nsys_runtime_capture_for_decode_step",
    "start_nsys_runtime_capture",
    "stop_nsys_runtime_capture",
]
