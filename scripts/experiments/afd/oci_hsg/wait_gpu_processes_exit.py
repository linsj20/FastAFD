#!/usr/bin/env python3
"""Wait for local CUDA compute processes to exit using Linux pidfd events."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def query_gpu_pids(nvidia_smi: str) -> set[int]:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit() or int(value) <= 0:
            raise RuntimeError(f"invalid nvidia-smi compute PID: {value!r}")
        pids.add(int(value))
    return pids


def wait_for_pid_events(pids: set[int], timeout_seconds: float) -> set[int]:
    poller = select.poll()
    pid_by_fd: dict[int, int] = {}
    try:
        for pid in sorted(pids):
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            pid_by_fd[fd] = pid
            poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        if not pid_by_fd:
            return set()
        timeout_ms = max(1, math.ceil(timeout_seconds * 1000.0))
        events = poller.poll(timeout_ms)
        if not events:
            raise TimeoutError(f"GPU processes did not exit: {sorted(pid_by_fd.values())}")
        return {pid_by_fd[fd] for fd, _ in events}
    finally:
        for fd in pid_by_fd:
            os.close(fd)


def wait_for_gpu_processes_exit(
    nvidia_smi: str, timeout_seconds: float
) -> dict[str, object]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    started = time.monotonic()
    deadline = started + timeout_seconds
    observed: set[int] = set()
    while True:
        pids = query_gpu_pids(nvidia_smi)
        observed.update(pids)
        if not pids:
            return {
                "status": "clear",
                "observed_pids": sorted(observed),
                "elapsed_seconds": time.monotonic() - started,
                "wait_mechanism": "linux_pidfd_poll",
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"GPU processes remain after timeout: {sorted(pids)}")
        exited = wait_for_pid_events(pids, remaining)
        if not exited:
            repeated = query_gpu_pids(nvidia_smi)
            if repeated == pids:
                raise RuntimeError(
                    "nvidia-smi PIDs are not visible to pidfd: "
                    f"{sorted(pids)}"
                )


def main() -> None:
    args = parse_args()
    report = wait_for_gpu_processes_exit(args.nvidia_smi, args.timeout_seconds)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
