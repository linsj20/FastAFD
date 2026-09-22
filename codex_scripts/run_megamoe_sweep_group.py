#!/usr/bin/env python3
"""Capture or CPU-analyze explicit same-EP model groups without submitting jobs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def load_groups(plan):
    """Read every input before capture, so a late missing manifest cannot waste GPUs."""
    if not plan["groups"] or plan["ep"] < 1:
        raise ValueError("Expected nonempty positive-EP sweep")
    prepared = []
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
    p.add_argument("phase", choices=("validate", "capture", "analyze"))
    p.add_argument("plan", type=Path)
    p.add_argument("--all-ranks", action="store_true",
                   help="Export ordinary timelines and decode phases for every captured rank")
    p.add_argument("--ordinary-only", action="store_true",
                   help="Export saved ordinary traces without requiring instrumented captures")
    args = p.parse_args()
    if args.all_ranks and args.phase != "analyze":
        p.error("--all-ranks applies only to saved-capture analysis")
    if args.ordinary_only and args.phase != "analyze":
        p.error("--ordinary-only applies only to saved-capture analysis")
    plan = json.loads(args.plan.read_text())
    prepared = load_groups(plan)
    if args.phase == "validate":
        for env, root, manifest, cases, selected in prepared:
            for key in ("MEGAMOE_SWEEP_SCRIPT", "MEGAMOE_SWEEP_PROFILE_SCRIPT", "MEGAMOE_SMOKE_TEST_SCRIPT", "FASTAFD_IMAGE"):
                if not Path(env[key]).is_file():
                    raise FileNotFoundError(f"{key}: {env[key]}")
            for key in ("MEGAMOE_ROUTER_MODULE", "MEGAMOE_ROUTER_SOURCE", "MEGAMOE_TEST_ENTRY"):
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
        if args.phase == "capture":
            if int(os.environ["SLURM_NTASKS"]) != plan["ep"]:
                raise ValueError("Allocation task count differs from planned EP")
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
