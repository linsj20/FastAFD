#!/usr/bin/env python3
"""Capture or CPU-analyze explicit same-EP model groups without submitting jobs."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def load_groups(plan):
    """Read every input before capture, so a late missing manifest cannot waste GPUs."""
    if not plan["groups"] or plan["ep"] < 1:
        raise ValueError("Expected nonempty positive-EP sweep")
    prepared = []
    outputs = set()
    for group in plan["groups"]:
        env = dict(os.environ, **group["env"])
        root = Path(env["MEGAMOE_SWEEP_ROOT"])
        manifest = json.loads((root / "manifest.json").read_text())
        cases = {c["case_id"]: c for c in manifest["cases"]}
        selected = env["MEGAMOE_SWEEP_CASES"].split(",")
        if len(cases) != len(manifest["cases"]) or len(selected) != len(set(selected)):
            raise ValueError(f"Duplicate cases: {root}")
        if int(env["MEGAMOE_IKET_EP"]) != plan["ep"]:
            raise ValueError(f"Group EP mismatch: {root}")
        for name in selected:
            phase = env.get("FASTAFD_IKET_CAPTURE_PHASE", "both")
            if phase not in {"normal", "profile", "both"}:
                raise ValueError(f"Unknown capture phase: {phase}")
            phases = ("normal", "profile") if phase == "both" else (phase,)
            for capture_phase in phases:
                output = (root.resolve(), name, capture_phase)
                if output in outputs:
                    raise ValueError(f"Duplicate capture output: {root}/{name} ({capture_phase})")
                outputs.add(output)
            case = cases[name]
            contract = dict(manifest["contract"], **case.get("capture_contract", {}))
            if case["ep"] != plan["ep"] or case["global_tokens"] < 1 or case["rows_per_rank"] != (
                    case["global_tokens"] + plan["ep"] - 1) // plan["ep"]:
                raise ValueError(f"Case topology mismatch: {root}/{name}")
            if any(int(contract[k]) < 1 for k in ("hidden", "intermediate", "experts", "top_k", "num_sms", "bucket")):
                raise ValueError(f"Invalid shape: {root}/{name}")
            if contract["experts"] % plan["ep"] or contract["top_k"] > contract["experts"] or contract["bucket"] < case["rows_per_rank"]:
                raise ValueError(f"Invalid expert/bucket bounds: {root}/{name}")
            if not contract["model"]:
                raise ValueError(f"Missing model: {root}/{name}")
        prepared.append((env, root, manifest, cases, selected))
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("validate", "capture", "capture-rank", "analyze"))
    p.add_argument("plan", type=Path)
    p.add_argument("--shared-container", action="store_true",
                   help="Keep one container step across capture groups, preserving plan order")
    p.add_argument("--all-ranks", action="store_true",
                   help="Export ordinary timelines and decode phases for every captured rank")
    p.add_argument("--ordinary-only", action="store_true",
                   help="Export saved ordinary traces without requiring instrumented captures")
    p.add_argument("--profiles-only", action="store_true",
                   help="Decode profiles after ordinary traces have already been exported")
    args = p.parse_args()
    if args.shared_container and args.phase != "capture":
        p.error("--shared-container applies only to capture")
    if args.all_ranks and args.phase != "analyze":
        p.error("--all-ranks applies only to saved-capture analysis")
    if args.ordinary_only and args.phase != "analyze":
        p.error("--ordinary-only applies only to saved-capture analysis")
    if args.profiles_only and (args.phase != "analyze" or args.ordinary_only):
        p.error("--profiles-only requires analyze without --ordinary-only")
    plan = json.loads(args.plan.read_text())
    prepared = load_groups(plan)
    if args.shared_container:
        images = {env["FASTAFD_IMAGE"] for env, *_ in prepared}
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()
        per_node, remainder = divmod(plan["ep"], len(nodes))
        if (len(images) != 1 or remainder or not 1 <= per_node <= 4
                or len(nodes) != int(os.environ["SLURM_NNODES"])
                or int(os.environ["SLURM_NTASKS"]) != plan["ep"]
                or any(env.get("MEGAMOE_IKET_PROFILE_RANK", "0") != "0" for env, *_ in prepared)
                or len({re.sub(r"-T\d+$", "", node) for node in nodes}) != 1):
            raise ValueError("Shared capture requires one image and one matching NVL72 allocation")
        env = dict(os.environ, MASTER_ADDR=nodes[0], FASTAFD_IKET_SINGLE_FABRIC="1",
                   FASTAFD_SWEEP_RANK_WORKER="1", FASTAFD_IKET_IN_CONTAINER="1")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run(["srun", f"--nodes={len(nodes)}", f"--ntasks={plan['ep']}",
            f"--ntasks-per-node={per_node}", "--kill-on-bad-exit=1",
            f"--container-image={images.pop()}", "--container-mount-home",
            "--container-mounts=/lustre:/lustre", "--no-container-remap-root",
            "env", "-u", "CUDA_VISIBLE_DEVICES",
            "python3", "-B", str(Path(__file__).resolve()), "capture-rank",
            str(args.plan.absolute())], env=env, check=True)
        return
    if args.phase == "validate":
        for env, root, manifest, cases, selected in prepared:
            for key in ("MEGAMOE_SWEEP_SCRIPT", "MEGAMOE_SWEEP_PROFILE_SCRIPT", "MEGAMOE_SMOKE_TEST_SCRIPT", "FASTAFD_IMAGE"):
                if not Path(env[key]).is_file():
                    raise FileNotFoundError(f"{key}: {env[key]}")
            for key in ("MEGAMOE_ROUTER_MODULE", "MEGAMOE_ROUTER_MODULES",
                        "MEGAMOE_ROUTER_SOURCE", "MEGAMOE_TEST_ENTRY"):
                if env.get(key) and not Path(env[key]).is_file():
                    raise FileNotFoundError(f"{key}: {env[key]}")
            for key in ("FASTAFD_SOURCE_REPO", "FASTAFD_VENV", "FASTAFD_IKET_INCLUDE", "FASTAFD_IKET_PYTHON_DEPS", "FASTAFD_IKET_DEEPGEMM_CACHE_ROOT", "FASTAFD_IKET_TVM_CACHE_ROOT", "DG_JIT_CACHE_DIR"):
                if not Path(env[key]).is_dir():
                    raise FileNotFoundError(f"{key}: {env[key]}")
            if not (Path(env["FASTAFD_VENV"]) / "bin/python").is_file():
                raise FileNotFoundError(f"Missing venv interpreter: {env['FASTAFD_VENV']}")
        print(json.dumps(dict(status="PASS_PLAN_INPUTS", groups=len(prepared),
                              roots=len({str(row[1]) for row in prepared}), ep=plan["ep"])))
        return
    for env, root, manifest, cases, selected in prepared:
        if args.phase in ("capture", "capture-rank"):
            if int(os.environ["SLURM_NTASKS"]) != plan["ep"]:
                raise ValueError("Allocation task count differs from planned EP")
            if args.phase == "capture-rank":
                if (env.get("FASTAFD_SWEEP_RANK_WORKER") != "1"
                        or env.get("FASTAFD_IKET_IN_CONTAINER") != "1"
                        or "SLURM_PROCID" not in env or "MASTER_ADDR" not in env):
                    raise ValueError("capture-rank requires a shared container rank worker")
                specs = []
                for name in selected:
                    if list((root / name).glob("*/decomposition.json")):
                        raise ValueError(f"Case already complete: {root}/{name}")
                    case = cases[name]
                    c = dict(manifest["contract"], **case.get("capture_contract", {}))
                    specs.append("\t".join(map(str, [name, case["rows_per_rank"], case["global_tokens"],
                        *(c[k] for k in ("model", "hidden", "intermediate", "experts", "top_k", "num_sms", "bucket"))])))
                env["FASTAFD_SWEEP_CASE_SPECS"] = "\n".join(specs)
            subprocess.run(["bash", env["MEGAMOE_SWEEP_SCRIPT"]], env=env, check=True)
            continue
        for name in selected:
            candidates = [p.parent for p in (root / name).glob("*/baseline.json")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one ordinary capture: {root / name}")
            run = candidates[0]
            if not args.ordinary_only and not (run / "profiled.json").is_file():
                raise ValueError(f"Missing instrumented capture: {run}")
            case_env = dict(env, MEGAMOE_IKET_GLOBAL_TOKENS=str(cases[name]["global_tokens"]))
            contract = dict(manifest["contract"], **cases[name].get("capture_contract", {}))
            for key in ("model", "hidden", "intermediate", "experts", "top_k", "num_sms", "bucket"):
                variable = "SMS" if key == "num_sms" else key.upper()
                case_env["MEGAMOE_IKET_" + variable] = str(contract[key])
            for rank in range(plan["ep"] if args.all_ranks else 1):
                rank_env = dict(case_env, MEGAMOE_IKET_PROFILE_RANK=str(rank))
                prefix = run / f"ranks/rank-{rank}/nsys/megamoe-rank{rank}"
                if not args.profiles_only:
                    with prefix.with_suffix(".cuda_gpu_kern_sum.csv").open("w") as out:
                        subprocess.run(["/usr/local/cuda/bin/nsys", "stats", "--force-export=true",
                            "--report", "cuda_gpu_kern_sum", "--format", "csv",
                            str(prefix.with_suffix(".nsys-rep"))], env=rank_env, stdout=out, check=True)
                if not args.ordinary_only:
                    subprocess.run(["bash", env["MEGAMOE_SWEEP_PROFILE_SCRIPT"], "analyze", str(run)],
                                   env=rank_env, check=True)
            # Preserve the standard sweep schema alongside rank-specific output.
            for filename in (() if args.ordinary_only else ("decomposition.json", "decomposition.csv")):
                shutil.copy2(run / "rank_decomposition/rank-0" / filename, run / filename)
        suffix = "ordinary_analysis" if args.ordinary_only else "analysis"
        (args.plan.parent / f"{root.name}_{suffix}_complete.json").write_text(
            json.dumps(dict(ep=plan["ep"], cases=len(selected)), indent=2) + "\n")


if __name__ == "__main__":
    main()
