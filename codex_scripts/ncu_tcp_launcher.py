#!/usr/bin/env python3
"""Start Slurm NCU peers with one server, reserving its port on client nodes."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def wait_for(test, description, timeout=30):
    deadline = time.monotonic() + timeout
    while not test():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.05)


def owns_listener(pid, port):
    """Observe the real server without opening a connection to its protocol."""
    inodes = set()

    def visit(current):
        proc = Path(f"/proc/{current}")
        if not proc.exists():
            return
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, PermissionError):
                    # A child can exit or change dumpability while NCU starts.
                    # Keep inspecting accessible descriptors; readiness still
                    # requires a listener owned by this process tree.
                    continue
                if target.startswith("socket:["):
                    inodes.add(target[8:-1])
            children = (proc / f"task/{current}/children").read_text().split()
        except (FileNotFoundError, PermissionError):
            return
        for child in children:
            visit(int(child))

    visit(pid)
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if (fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port
                and fields[9] in inodes):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1024 <= args.port <= 65535:
        parser.error("An NCU command and unprivileged TCP port are required")
    rank = int(os.environ["SLURM_PROCID"])
    node = int(os.environ["SLURM_NODEID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    args.root.mkdir(parents=True, exist_ok=True)
    # Each capture must have a fresh coordination directory.
    with (args.root / f"rank-{rank}.json").open("x") as out:
        json.dump(dict(rank=rank, node=node, host=socket.gethostname(),
                       port=args.port, command=command), out)
    reservation = None
    process = None
    try:
        if node != 0:
            ready = args.root / f"reserved-node-{node}"
            if local_rank == 0:
                reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                reservation.bind(("0.0.0.0", args.port))
                ready.write_text("reserved\n")
            else:
                wait_for(ready.exists, "Client-node port reservation")
        server_ready = args.root / "server-listening"
        if rank != 0:
            wait_for(server_ready.exists, "Head NCU listener")
        process = subprocess.Popen(command, start_new_session=True)

        def terminate(signum, _frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        if rank == 0:
            def listening():
                if process.poll() is not None:
                    raise RuntimeError("Head NCU exited before opening its listener")
                return owns_listener(process.pid, args.port)

            wait_for(listening, "Head NCU listener")
            server_ready.write_text("listening\n")
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if reservation is not None:
            reservation.close()


if __name__ == "__main__":
    raise SystemExit(main())
