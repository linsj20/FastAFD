#!/usr/bin/env python3
"""Run an authorized sweep launch order with bounded concurrency and audited gates.

Uses existing submission, monitoring and collection helpers. Stops on any failed
job or incomplete terminal group; never retries or modifies an experiment plan.
"""

import argparse
import csv
import datetime
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def helper(name, *args):
    subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], check=True)


def validate(groups):
    seen = set()
    for group in groups:
        root = Path(group["plan"])
        env = read(root / "env.json")
        manifest = read(root.parent / "manifest.json")
        if manifest["final_validation"].get("status", "").startswith("REJECTED"):
            raise ValueError(f"Sweep candidate is rejected; do not submit: {root}")
        expected = {c["case_id"] for c in manifest["cases"] if c["ep"] == group["ep"]}
        actual = env["MEGAMOE_SWEEP_CASES"].split(",")
        if set(actual) != expected or len(actual) != len(expected) or len(actual) != group["cases"]:
            raise ValueError(f"Case coverage mismatch: {root}")
        key = (group["model"], group["ep"])
        if key in seen:
            raise ValueError(f"Duplicate group: {key}")
        seen.add(key)
        if Path(env["FASTAFD_SOURCE_REPO"]).name != manifest["final_validation"]["source"]:
            raise ValueError(f"Source mismatch: {root}")
        if int(env["MEGAMOE_BASELINE_ITERATIONS"]) != manifest["final_validation"]["baseline_iterations"]:
            raise ValueError(f"Ordinary sample policy mismatch: {root}")
        if int(env["MEGAMOE_IKET_ITERATIONS"]) != manifest["final_validation"]["phase_samples"]:
            raise ValueError(f"Phase sample policy mismatch: {root}")
        if group["ep"] == 64 and env.get("MEGAMOE_SMOKE_COMPARE_COMBINE") != "1":
            raise ValueError(f"Missing EP64 exact combine gate: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_order", type=Path)
    parser.add_argument("--remote-script", required=True)
    parser.add_argument("--host", default="oci-hsg")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--watch-seconds", type=int, default=180)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 2 or args.watch_seconds < 180:
        raise ValueError("Require 1–2 jobs and monitoring interval >=180 seconds")
    groups = read(args.launch_order)["groups"]
    validate(groups)
    if args.validate_only:
        print(json.dumps({"validated_groups": len(groups), "cases": sum(g["cases"] for g in groups)}))
        return
    with args.launch_order.with_suffix(".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            pending, active = [], []
            for group in groups:
                root = Path(group["plan"])
                if (root / "completion.json").exists():
                    completion = read(root / "completion.json")
                    if completion["status"] != "COMPLETE_AUDITED" or completion["cases"] != group["cases"]:
                        raise ValueError(f"Invalid completion record: {root}")
                elif (root / "job.json").exists():
                    active.append(group)
                else:
                    pending.append(group)
            if len(active) > args.max_jobs:
                raise ValueError("Outstanding job limit exceeded")
            if active:
                helper("monitor_afd_jobs.py", *(g["plan"] for g in active),
                       "--host", args.host, "--watch-seconds", 0)
            completed = []
            for group in active:
                root = Path(group["plan"])
                state = read(root / "monitor_snapshot.json")
                if state["state"].split()[0] not in TERMINAL:
                    continue
                if state["state"] != "COMPLETED" or state["exit_code"] != "0:0":
                    raise RuntimeError(f"Job requires diagnosis: {state['job']} {state['state']} {state['exit_code']}")
                env = read(root / "env.json")
                helper("megamoe_input_sweep.py", "fetch", root.parent,
                       "--remote-root", env["MEGAMOE_SWEEP_ROOT"],
                       "--remote-script", args.remote_script, "--host", args.host)
                audit = read(root.parent / "audit_summary.json")
                rows = list(csv.DictReader((root.parent / "decomposition.csv").open()))
                found = {r["case_id"] for r in rows}
                expected = set(env["MEGAMOE_SWEEP_CASES"].split(","))
                if not expected <= found or expected.intersection(audit["missing"]):
                    raise RuntimeError(f"Completed job has missing cases: {root}")
                save(root / "completion.json", {
                    "job": state["job"], "status": "COMPLETE_AUDITED", "cases": group["cases"],
                    "phase_samples": group["cases"] * int(env["MEGAMOE_IKET_ITERATIONS"]),
                    "ordinary_kernel_instances": group["cases"] * int(env["MEGAMOE_BASELINE_ITERATIONS"]),
                    "elapsed": state["elapsed"], "source": Path(env["FASTAFD_SOURCE_REPO"]).name,
                    "remaining_model_cases": len(audit["missing"]),
                })
                completed.append(group)
            active = [g for g in active if g not in completed]
            for group in pending[:args.max_jobs - len(active)]:
                helper("submit_plan.py", "sbatch", group["plan"], "--host", args.host)
                active.append(group)
            status = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "active": [read(Path(g["plan"]) / "job.json") for g in active],
                      "completed_groups": sum((Path(g["plan"]) / "completion.json").exists() for g in groups)}
            save(args.launch_order.with_name("runner_status.json"), status)
            print(json.dumps(status), flush=True)
            if not active:
                for model_root in {Path(g["plan"]).parent for g in groups}:
                    if not read(model_root / "audit_summary.json")["complete"]:
                        raise RuntimeError(f"Final model audit incomplete: {model_root}")
                return
            time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
