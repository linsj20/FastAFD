#!/usr/bin/env python3
"""Collect MegaMoE ABBA runs described by local env.json and job.json files."""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import time
from pathlib import Path
from statistics import mean

from summarize_megamoe_abba import parse_records, summarize_profiles


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--host", default="oci-hsg")
    parser.add_argument("--captures", action="store_true",
                        help="Collect every rank-0 triple after successful completion")
    parser.add_argument("--watch-seconds", type=float, default=0,
                        help="Poll at this interval until all jobs terminate; zero collects once")
    args = parser.parse_args()
    if args.watch_seconds < 0:
        parser.error("watch interval must be nonnegative")
    runs = []
    for root in args.roots:
        env = json.loads((root / "env.json").read_text())
        job = str(json.loads((root / "job.json").read_text())["job"])
        if not job.isdigit():
            raise ValueError(f"{root}: requires a numeric job")
        runs.append((root, env, job))
    while runs:
        started = time.monotonic()
        finished = collect(runs, args.host, args.captures)
        if not args.watch_seconds:
            break
        runs = [run for run in runs if run[2] not in finished]
        if runs:
            time.sleep(max(0, args.watch_seconds - (time.monotonic() - started)))


def collect(runs: list, host: str, captures: bool) -> set[str]:
    command = ["sacct", "-n", "-X", "-j", ",".join(job for _, _, job in runs),
               "-o", "JobIDRaw,State,ExitCode,Elapsed,NodeList", "-P"]
    raw = subprocess.check_output(["ssh", host, shlex.join(command)], text=True)
    states = {}
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 5:
            states[fields[0]] = dict(zip(
                ("state", "exit_code", "elapsed", "nodes"), fields[1:5]))
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    finished = set()
    for root, env, job in runs:
        state = states[job]
        if state["state"].split()[0] in {
                "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
            finished.add(job)
        save(root / "scheduler.json", {"job": job, "observed_utc": observed, **state})
        if state["state"] == "PENDING":
            print(json.dumps({"root": str(root), "job": job, **state}), flush=True)
            continue
        remote = env["OUTPUT_ROOT"].rstrip("/")
        copied = subprocess.run(["rsync", "-a", *(
            f"{host}:{remote}/compare-{job}.{ext}" for ext in ("out", "err")
        ), str(root) + "/"], capture_output=True, text=True)
        if copied.returncode:
            elapsed = state["elapsed"].split(":")
            starting = (state["state"] in {"RUNNING", "CONFIGURING"}
                        and len(elapsed) == 3 and elapsed[0] == "00"
                        and int(elapsed[1]) < 3)
            if (starting and copied.returncode == 23
                    and "link_stat" in copied.stderr
                    and "No such file or directory" in copied.stderr):
                # Allocation state precedes batch startup and log creation.
                save(root / "comparison.json", {
                    "status": "AWAITING_BATCH_LOGS", "scheduler": state,
                })
                print(json.dumps({"root": str(root), "job": job, **state,
                                  "logs": "not created during batch startup"}), flush=True)
                continue
            raise RuntimeError(f"{root}: log collection failed: {copied.stderr}")
        records = parse_records(root / f"compare-{job}.out")
        expected = env["MEGAMOE_ABBA_ORDER"].split()
        if [label for label, _ in records] != expected[:len(records)]:
            raise ValueError(f"{root}: unexpected completed trial order")
        values: dict[str, list[float]] = {}
        timing_inputs = set()
        has_general_records = False
        for label, record in records:
            # Per-arm env files can override the job's fixed/general setting.
            # Classify the actual record rather than the submission default.
            metrics = record["exact_qwen"]
            fixed = [metric for metric in metrics.values()
                     if metric.get("timed_scope") == "route_producer_and_megamoe"]
            if not fixed:
                has_general_records = True
                continue
            if len(fixed) != len(metrics):
                raise ValueError(f"{root}: mixed timing scopes in one fixed-input trial")
            for metric in fixed:
                timing_inputs.add(json.dumps({
                    key: metric[key] for key in (
                        "ep_size", "global_tokens", "model_dimensions", "routing",
                        "timed_scope", "timing_iterations", "timing_warmups")
                }, sort_keys=True))
                values.setdefault(label.split("_", 1)[0], []).append(
                    1000 * float(metric["milliseconds"]))
        if len(timing_inputs) > 1 or has_general_records:
            # A joint correctness job can use fixed routing on different models.
            # Keep each record, but do not average unlike timing inputs.
            values = {}
        primary = {
            "primary_records": dict(records), "primary_us": values,
            "primary_mean_us": {arm: mean(samples) for arm, samples in values.items()},
            "primary_count": len(records), "expected_primary_count": len(expected),
        }
        save(root / "primary.json", primary)
        complete = state["state"] == "COMPLETED" and state["exit_code"] == "0:0"
        if complete and len(records) != len(expected):
            raise ValueError(f"{root}: successful job lacks expected primary records")
        profile_arms = env.get("MEGAMOE_ABBA_PROFILE_VARIANTS", "").split()
        if captures and complete:
            for arm in profile_arms:
                local = root / arm
                local.mkdir(exist_ok=True)
                base = f"{host}:{remote}/decomposition/{arm}/{job}/"
                subprocess.run(["rsync", "-a", base + "decomposition.json",
                                base + "ranks/rank-0/nsys/megamoe-rank0.cuda_gpu_kern_sum.csv",
                                base + "ranks/rank-0/nsys/megamoe-rank0.nsys-rep",
                                str(local) + "/"], check=True)
        profiles = summarize_profiles(root) if profile_arms else {}
        captures_complete = all(p.get("status") != "missing_capture"
                                for p in profiles.values())
        save(root / "comparison.json", {
            "status": "COMPLETED" if complete and captures_complete
            else "INCOMPLETE_EVIDENCE",
            "scheduler": state, "primary": primary, "profiles": profiles,
        })
        print(json.dumps({"root": str(root), "job": job, **state,
                          "primary_count": len(records),
                          "primary_us": values, "captures_complete": captures_complete}), flush=True)

    return finished


if __name__ == "__main__":
    main()
