#!/usr/bin/env python3
"""Prepare fresh random same-allocation sweep plans from a validated template.

Writes plans only. The caller must validate candidate correctness, prepare its
host extension, sync its source/workflows and observe the GPU-job limit before
using submit_plan.py sbatch. Existing control source/cache paths are preserved.
"""

import argparse
import copy
import json
import random
import secrets
import shutil
from pathlib import Path, PurePosixPath


ARMS = ("control", "candidate", "candidate_repeat", "control_repeat")
POLICY = {
    "MEGAMOE_BASELINE_WARMUPS": "30",
    "MEGAMOE_BASELINE_ITERATIONS": "30",
    "MEGAMOE_IKET_WARMUPS": "3",
    "MEGAMOE_IKET_ITERATIONS": "5",
    "DG_MEGAMOE_LAMPORT_DISPATCH": "0",
    "DG_MEGAMOE_LAMPORT_COMBINE": "1",
    "DG_MEGAMOE_TMA_PUSH": "0",
    "DG_MEGAMOE_PUSH_IN_MEGAMOE": "0",
}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--control-arm", choices=ARMS,
                        help="Use this template arm's validated source/cache for both controls")
    parser.add_argument("--reference-arm", action="append", default=[],
                        help="Retain a named template reference and its repeat around the ABBA arms")
    parser.add_argument("--model", action="append", required=True, metavar="NAME:EP")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, help="Omit for a fresh random seed")
    parser.add_argument("--time", default="00:20:00")
    args = parser.parse_args()
    references = args.reference_arm
    arms = (*references, *ARMS, *(r + "_repeat" for r in reversed(references)))
    if (len(set(arms)) != len(arms)
            or any(Path(a).name != a or a in {".", ".."} for a in arms)):
        parser.error("Reference arms must be distinct directory names without arm/repeat overlap")
    remote = PurePosixPath(args.remote_root)
    source = PurePosixPath(args.candidate_source)
    if not remote.is_absolute() or not source.is_absolute() or args.count < 1:
        parser.error("remote paths must be absolute and count must be positive")
    seed = secrets.randbits(63) if args.seed is None else args.seed
    rng = random.Random(seed)
    selection = {"seed": seed, "cases": {}, "scope":
                 "Fresh random cases from each full manifest; no timing filtering. "
                 "Identical cases and sample policies across all arms.", "arms": arms}
    files = {}
    for model_spec in args.model:
        model, ep_text = model_spec.rsplit(":", 1)
        ep = int(ep_text)
        if (not model or model in {".", ".."} or Path(model).name != model
                or model in selection["cases"]):
            raise ValueError(f"Invalid or duplicate model: {model}")
        model_template = args.template / model
        manifest = read(model_template / "control/manifest.json")
        eligible = [case for case in manifest["cases"] if case["ep"] == ep]
        cases = rng.sample(eligible, args.count)
        selection["cases"][model] = cases
        for arm in arms:
            template_arm = args.control_arm if arm in ("control", "control_repeat") and args.control_arm else arm
            env = read(model_template / template_arm / "env.json")
            if int(env["MEGAMOE_IKET_EP"]) != ep:
                raise ValueError(f"{model}/{arm}: EP must match the allocation template")
            if any(env.get(key) != value for key, value in POLICY.items()):
                raise ValueError(f"{model}/{arm}: template changes the fixed trial policy")
            env["OUTPUT_ROOT"] = env["MEGAMOE_SWEEP_ROOT"] = str(remote / model / arm)
            env["MEGAMOE_SWEEP_CASES"] = ",".join(case["case_id"] for case in cases)
            env["MEGAMOE_IKET_EP"] = str(ep)
            for key in ("MEGAMOE_SWEEP_SCRIPT", "MEGAMOE_SWEEP_PROFILE_SCRIPT"):
                env[key] = str(remote / "workflow" / PurePosixPath(env[key]).name)
            if arm in ("candidate", "candidate_repeat"):
                env["FASTAFD_SOURCE_REPO"] = str(source)
                env["FASTAFD_IKET_DEEPGEMM_CACHE_ROOT"] = str(
                    remote / "cache/candidate/deepgemm")
                env["FASTAFD_IKET_TVM_CACHE_ROOT"] = str(
                    remote / "cache/candidate/tvm_ffi")
                env["DG_JIT_CACHE_DIR"] = str(remote / "cache" / model / "candidate/jit")
            arm_manifest = copy.deepcopy(manifest)
            arm_manifest["final_validation"] = {
                "status": "PLANNED_RANDOM_SAMPLE",
                "source": PurePosixPath(env["FASTAFD_SOURCE_REPO"]).name,
                "baseline_iterations": int(env["MEGAMOE_BASELINE_ITERATIONS"]),
                "phase_samples": int(env["MEGAMOE_IKET_ITERATIONS"]),
            }
            files[Path(model) / arm / "env.json"] = env
            files[Path(model) / arm / "manifest.json"] = arm_manifest
        model_env = read(model_template / "env.json")
        template_order = model_env["MEGAMOE_SWEEP_ABBA_ORDER"].split()
        if (len(set(template_order)) != len(template_order)
                or not set(arms) <= set(template_order)):
            raise ValueError(f"{model}: template lacks unique required arms")
        model_env["MEGAMOE_SWEEP_ABBA_ORDER"] = " ".join(arms)
        model_env["OUTPUT_ROOT"] = model_env["MEGAMOE_SWEEP_ABBA_ROOT"] = str(remote / model)
        files[Path(model) / "env.json"] = model_env
        command = read(model_template / "submit.json")
        if command[:2] != ["sbatch", "--parsable"]:
            raise ValueError(f"{model}: expected structured sbatch command")
        updates = {
            "--time=": args.time,
            "--job-name=": f"fastafd:{args.output.name}-{model}",
            "--output=": str(remote / model / "compare-%j.out"),
            "--error=": str(remote / model / "compare-%j.err"),
        }
        for prefix, value in updates.items():
            matches = [i for i, item in enumerate(command) if item.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"{model}: expected one {prefix} argument")
            command[matches[0]] = prefix + value
        command[-1] = str(remote / "workflow" / PurePosixPath(command[-1]).name)
        files[Path(model) / "submit.json"] = command
    files[Path("selection.json")] = selection
    # Validate every destination before writing any plan; never replace a trial.
    if any((args.output / model).exists() for model in selection["cases"]):
        raise ValueError("Output already contains a model plan; use a fresh trial")
    if any((args.output / path).exists() for path in files):
        raise ValueError("Output already contains a selection or plan")
    workflows = list((args.template / "workflow").iterdir())
    if not workflows or any(not path.is_file() for path in workflows):
        raise ValueError("Expected a directory of workflow files")
    for path, data in files.items():
        destination = args.output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "workflow").mkdir(exist_ok=True)
    for path in workflows:
        destination = args.output / "workflow" / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    print(json.dumps({"seed": seed, "cases": {
        model: [case["case_id"] for case in cases]
        for model, cases in selection["cases"].items()}}))


if __name__ == "__main__":
    main()
