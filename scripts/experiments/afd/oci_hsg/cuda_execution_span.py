#!/usr/bin/env python3
"""Reusable Nsight helpers for per-step AFD CUDA execution spans."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import subprocess


CUDA_SPAN_API_VERSION = "20260817-attention-cuda-execution-span-v1"
AFD_SPAN_BASIS = "first_attention_cuda_start_to_last_attention_cuda_end"
AFD_TABLES = (
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "NVTX_EVENTS",
    "StringIds",
)
AFD_PIPELINE_LEAD_STEPS = 2


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def nvtx_ranges(
    connection: sqlite3.Connection,
) -> list[tuple[int, int, str]]:
    if not table_exists(connection, "NVTX_EVENTS"):
        raise RuntimeError("NVTX_EVENTS table is absent")
    return [
        (int(start), int(end), str(name))
        for start, end, name in connection.execute(
            """
            SELECT n.start,n.end,coalesce(n.text,s.value)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id=n.textId
            WHERE n.end IS NOT NULL
              AND coalesce(n.text,s.value) IS NOT NULL
            ORDER BY n.start
            """
        )
    ]


def afd_attention_cuda_spans(
    sqlite_path: Path,
) -> tuple[dict[int, tuple[int, int]], dict[str, int]]:
    connection = sqlite3.connect(sqlite_path)
    try:
        required = (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "NVTX_EVENTS",
            "StringIds",
        )
        missing = [table for table in required if not table_exists(connection, table)]
        if missing:
            raise RuntimeError(f"trace is missing required tables: {missing}")

        ranges = nvtx_ranges(connection)
        replies = [
            (start, end, int(match.group(1)))
            for start, end, name in ranges
            if name.startswith("AFD_Worker_SendReply_Put[")
            and (match := re.search(r"\[step=(\d+)\]", name)) is not None
        ]
        retire_starts = [
            start
            for start, _, name in ranges
            if name.startswith("AFD_AG_ProcessLast[")
        ]
        hot_loops: list[tuple[int, int, int, int]] = []
        for loop_start, loop_end, loop_name in ranges:
            if loop_name != "AFD_AttnCentralizedHotLoop_Iter":
                continue
            contained_replies = [
                step
                for reply_start, reply_end, step in replies
                if reply_start >= loop_start and reply_end <= loop_end
            ]
            if len(contained_replies) != 1:
                continue
            contained_retires = [
                start
                for start in retire_starts
                if loop_start <= start <= loop_end
            ]
            enqueue_end = min(contained_retires, default=loop_end)
            hot_loops.append(
                (loop_start, loop_end, enqueue_end, contained_replies[0])
            )

        launches = [
            (int(runtime_start), int(runtime_end), int(correlation_id))
            for runtime_start, runtime_end, correlation_id in connection.execute(
                """
                SELECT r.start,r.end,r.correlationId
                FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
                JOIN StringIds AS runtime_name ON runtime_name.id=r.nameId
                WHERE runtime_name.value LIKE 'cudaGraphLaunch%'
                ORDER BY r.start
                """
            )
        ]
        if not launches:
            raise RuntimeError("attention trace has no captured cudaGraphLaunch call")
        mapped: dict[int, tuple[int, int]] = {}
        graph_trace_counts: list[int] = []
        graph_kernel_counts: list[int] = []
        full_kernel_counts: list[int] = []
        for launch_start, _, correlation_id in launches:
            containing_loops = [
                (enqueue_end, reply_step)
                for loop_start, loop_end, enqueue_end, reply_step in hot_loops
                if loop_start <= launch_start <= loop_end
            ]
            if len(containing_loops) != 1:
                continue
            enqueue_end, reply_step = containing_loops[0]
            if table_exists(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
                trace_start, trace_end, graph_trace_count = connection.execute(
                    """
                    SELECT min(start),max(end),count(*)
                    FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                    WHERE correlationId=?
                    """,
                    (correlation_id,),
                ).fetchone()
            else:
                trace_start, trace_end, graph_trace_count = None, None, 0
            node_start, node_end, graph_kernel_count = connection.execute(
                """
                SELECT min(start),max(end),count(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE correlationId=? AND graphId IS NOT NULL
                """,
                (correlation_id,),
            ).fetchone()
            if graph_trace_count:
                graph_start, graph_end = trace_start, trace_end
            elif graph_kernel_count:
                graph_start, graph_end = node_start, node_end
            else:
                continue

            # Include attention-side sampling and writeback kernels launched
            # after the graph and before the same loop retires.
            extra_start, extra_end, extra_count = connection.execute(
                """
                SELECT min(k.start),max(k.end),count(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                  ON r.correlationId=k.correlationId
                WHERE r.start>=? AND r.start<?
                """,
                (launch_start, enqueue_end),
            ).fetchone()
            kernel_start = min(
                int(value)
                for value in (graph_start, extra_start)
                if value is not None
            )
            kernel_end = max(
                int(value)
                for value in (graph_end, extra_end)
                if value is not None
            )
            if kernel_end <= kernel_start:
                raise RuntimeError("attention CUDA span has non-positive duration")
            decode_step = reply_step + AFD_PIPELINE_LEAD_STEPS
            if decode_step in mapped:
                raise RuntimeError(
                    f"duplicate attention CUDA span for decode step {decode_step}"
                )
            mapped[decode_step] = (kernel_start, kernel_end)
            graph_trace_counts.append(int(graph_trace_count))
            graph_kernel_counts.append(int(graph_kernel_count))
            full_kernel_counts.append(int(extra_count))
        if not mapped:
            raise RuntimeError(
                "no attention CUDA execution could be mapped to a decode step"
            )
        return mapped, {
            "cuda_graph_launch_count": len(launches),
            "mapped_step_count": len(mapped),
            "hot_loop_anchor_count": len(hot_loops),
            "graph_trace_record_count_min": min(graph_trace_counts, default=0),
            "graph_trace_record_count_max": max(graph_trace_counts, default=0),
            "graph_kernel_count_min": min(graph_kernel_counts, default=0),
            "graph_kernel_count_max": max(graph_kernel_counts, default=0),
            "attention_kernel_count_min": min(full_kernel_counts),
            "attention_kernel_count_max": max(full_kernel_counts),
        }
    finally:
        connection.close()


def export_nsys(
    nsys: Path,
    report: Path,
    sqlite_path: Path,
    tables: tuple[str, ...],
    time_range: tuple[int, int] | None = None,
) -> None:
    command = [
        str(nsys),
        "export",
        "--type",
        "sqlite",
        "--tables",
        ",".join(tables),
        "--force-overwrite=true",
        "--quiet=true",
    ]
    if time_range is not None:
        start, end = time_range
        if end <= start:
            raise RuntimeError(f"invalid Nsight export range {start}/{end}")
        command.extend(["--times", f"{start}/{end}"])
    command.extend(["--output", str(sqlite_path), str(report)])
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not sqlite_path.is_file():
        detail = completed.stderr.strip()
        raise RuntimeError(
            f"Nsight export failed for {report} "
            f"(exit {completed.returncode}): {detail}"
        )


def gpu_worker_reports(ray_logs: Path, gpu_workers: int) -> list[Path]:
    indexed: dict[int, Path] = {}
    for report in ray_logs.glob("minisgl_rank*_*.nsys-rep"):
        match = re.fullmatch(r"minisgl_rank(\d+)_\d+\.nsys-rep", report.name)
        if not match:
            continue
        rank = int(match.group(1))
        if 1 <= rank <= gpu_workers:
            if rank in indexed:
                raise RuntimeError(f"duplicate attention-rank report {rank}")
            indexed[rank] = report
    expected = set(range(1, gpu_workers + 1))
    if set(indexed) != expected:
        raise RuntimeError(
            "rank-level Nsight coverage mismatch: "
            f"missing={sorted(expected - set(indexed))} "
            f"extra={sorted(set(indexed) - expected)}"
        )
    return [indexed[rank] for rank in sorted(indexed)]


def choose_afd_steps(
    complete_windows: list[list[int]],
    selected_window: list[int],
    common_steps: set[int],
) -> tuple[list[int], str]:
    def captured_subset(window: list[int]) -> list[int]:
        return [step for step in window if step in common_steps]

    selected_steps = captured_subset(selected_window)
    if selected_steps:
        measured_steps = selected_steps
        selection_policy = "captured subset of run-selected full-bucket wave"
    else:
        captured_windows = [captured_subset(window) for window in complete_windows]
        measured_steps = max(captured_windows, key=len)
        selection_policy = "largest captured subset of a complete full-bucket wave"
    if not measured_steps:
        raise RuntimeError(
            "no full-bucket decode step has a mapped attention CUDA span; "
            f"selected={selected_steps}, common={sorted(common_steps)}"
        )
    return measured_steps, selection_policy
