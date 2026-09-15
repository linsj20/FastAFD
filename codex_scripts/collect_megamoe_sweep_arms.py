#!/usr/bin/env python3
"""Collect one recorded multi-arm timing job and evaluate dispatch plus E2E gates.

Submits no jobs. Waits only for the recorded job, retains all captures and repeats,
and fails on scheduler, source, coverage or collection errors. A failed performance
gate is a measured result, not a reason to retry the experiment.
"""

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

from compare_megamoe_fullchain import audit_fullpath


SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def evaluate_candidate(root, candidate, dispatch_control, fullpath_controls,
                       phase_history, history, envs, cases, job, run,
                       *, suffix=""):
    controls = [control for control in fullpath_controls if control != candidate]
    if candidate == dispatch_control or not controls:
        raise ValueError("Each candidate needs a distinct dispatch and full-path control")
    phase_path = root / f"dispatch_comparison{suffix}.json"
    run([sys.executable, str(SCRIPTS / "compare_megamoe_sweep_arms.py"),
         str(root), "--history", str(phase_history), "--candidate-arm", candidate,
         "--control-arm", dispatch_control, "--output", str(phase_path)])
    candidate_env = envs[candidate]
    fullpath = {}
    for control in controls:
        fullpath[control] = audit_fullpath(
            root, candidate, control,
            Path(candidate_env["FASTAFD_SOURCE_REPO"]).name,
            Path(envs[control]["FASTAFD_SOURCE_REPO"]).name,
            set(cases), job, samples=int(candidate_env["MEGAMOE_BASELINE_ITERATIONS"]),
            history=history)
    fullpath_path = root / f"fullpath_gate{suffix}.json"
    save(fullpath_path, fullpath)
    return {
        "dispatch_report": phase_path.name,
        "fullpath_report": fullpath_path.name,
        "fullpath_controls": controls,
        "passed": read(phase_path)["all_dispatch_gates_pass"] and
                  all(report["all_pass"] for report in fullpath.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase-history", type=Path, required=True)
    parser.add_argument("--fullpath-history", type=Path, required=True)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--evaluate-candidate", action="append", default=[],
                        help="Also evaluate this arm using the same captures and controls; "
                             "its own arm is excluded from full-path controls")
    parser.add_argument("--dispatch-control", default="pull")
    parser.add_argument("--fullpath-control", action="append", required=True)
    parser.add_argument("--host", default="oci-hsg")
    parser.add_argument("--watch-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.watch_seconds < 180:
        parser.error("Monitoring interval must be at least 180 seconds")
    if not (args.phase_history / "decomposition.csv").is_file():
        parser.error("--phase-history must name a model history directory containing decomposition.csv")
    root = args.root
    arms = read(root / "env.json")["MEGAMOE_SWEEP_ABBA_ORDER"].split()
    prefixes = [a for a in arms if not a.endswith("_repeat")]
    candidates = [args.candidate, *args.evaluate_candidate]
    required = {*candidates, args.dispatch_control, *args.fullpath_control}
    if (len(set(arms)) != len(arms) or not required <= set(prefixes)
            or len(set(candidates)) != len(candidates)
            or args.dispatch_control in candidates
            or any(not set(args.fullpath_control) - {candidate} for candidate in candidates)
            or args.candidate in args.fullpath_control
            or args.candidate == args.dispatch_control
            or len(set(args.fullpath_control)) != len(args.fullpath_control)
            or any(Path(a).name != a or a in {".", ".."} for a in arms)
            or arms != prefixes + [a + "_repeat" for a in reversed(prefixes)]):
        raise ValueError("Require distinct controls and symmetric named-arm repeats")
    envs = {a: read(root / a / "env.json") for a in arms}
    candidate_env = envs[args.candidate]
    cases = candidate_env["MEGAMOE_SWEEP_CASES"].split(",")
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Require unique nonempty sampled cases")
    policy = ("MEGAMOE_SWEEP_CASES", "MEGAMOE_BASELINE_ITERATIONS",
              "MEGAMOE_BASELINE_WARMUPS", "MEGAMOE_IKET_ITERATIONS",
              "MEGAMOE_IKET_WARMUPS", "MEGAMOE_IKET_EP")
    if any(e[k] != candidate_env[k] for e in envs.values() for k in policy):
        raise ValueError("Arm case or timing policy mismatch")
    for prefix in prefixes:
        if envs[prefix]["FASTAFD_SOURCE_REPO"] != envs[prefix + "_repeat"]["FASTAFD_SOURCE_REPO"]:
            raise ValueError("Repeated arm source mismatch")
    history = read(args.fullpath_history)["models"][args.model]
    if not set(cases) <= set(history):
        raise ValueError("Missing historical full-path cases")
    job = read(root / "job.json")["job"]
    with (root / "collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "completion.json").exists():
            raise ValueError("Collection already completed; inspect existing evidence")
        print(json.dumps({"status": "WAITING_FOR_RECORDED_JOB", "job": job,
                          "root": str(root)}), flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / "monitor_afd_jobs.py"),
                        str(root), "--host", args.host, "--watch-seconds",
                        str(args.watch_seconds)], check=True)
        state = read(root / "monitor_snapshot.json")
        if state["job"] != job or state["state"] != "COMPLETED" or state["exit_code"] != "0:0":
            raise RuntimeError(f"Timing job did not complete successfully: {state}")
        with (root / "collection.log").open("w") as log:
            def run(command):
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

            for arm, env in envs.items():
                local, remote = root / arm, env["MEGAMOE_SWEEP_ROOT"]
                run([sys.executable, str(SCRIPTS / "megamoe_input_sweep.py"), "fetch",
                     str(local), "--remote-root", remote, "--remote-script",
                     str(Path(env["MEGAMOE_SWEEP_SCRIPT"]).with_name("megamoe_input_sweep.py")),
                     "--host", args.host])
                audit = read(local / "audit_summary.json")
                if (audit["artifact_contracts_checked"] != len(cases)
                        or set(cases).intersection(audit["missing"])
                        or audit["samples_checked"] != len(cases) * int(env["MEGAMOE_IKET_ITERATIONS"])):
                    raise ValueError(f"Incomplete arm audit: {arm}")
                run(["rsync", "-a", str(local / "decomposition.csv"),
                     args.host + ":" + remote + "/decomposition.csv"])
                helper = str(Path(env["MEGAMOE_SWEEP_SCRIPT"]).with_name(
                    "summarize_megamoe_router_timeline.py"))
                run(["ssh", args.host, shlex.join(["python3", helper, remote, "--sweep",
                     "--output", remote + "/router_timeline.json",
                     "--csv", remote + "/router_timeline.csv"])])
                run(["rsync", "-a", args.host + ":" + remote + "/router_timeline.json",
                     args.host + ":" + remote + "/router_timeline.csv", str(local) + "/"])
            run([sys.executable, str(SCRIPTS / "compare_megamoe_router_timelines.py"),
                 str(root), "--samples", candidate_env["MEGAMOE_BASELINE_ITERATIONS"],
                 *(v for a in prefixes for v in ("--arm", a))])
            evaluations = {
                candidate: evaluate_candidate(
                    root, candidate, args.dispatch_control, args.fullpath_control,
                    args.phase_history, history, envs, cases, job, run,
                    suffix="" if candidate == args.candidate else "_" + candidate)
                for candidate in candidates
            }
        passed = evaluations[args.candidate]["passed"]
        result = {"status": "COMPLETE_AUDITED", "job": job, "elapsed": state["elapsed"],
                  "cases": len(cases), "arms": len(arms), "captures": len(cases) * len(arms),
                  "phase_samples": len(cases) * len(arms) * int(candidate_env["MEGAMOE_IKET_ITERATIONS"]),
                  "ordinary_kernel_instances": len(cases) * len(arms) * int(candidate_env["MEGAMOE_BASELINE_ITERATIONS"]),
                  "sampled_dispatch_and_fullpath_gates_pass": passed,
                  "candidate_evaluations": evaluations,
                  "all_records_retained": True,
                  "scope": "Sampled rank0 gates only; unprofiled slowest-rank timing, "
                           "serving alignment and full sweeps remain separate requirements."}
        save(root / "completion.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
