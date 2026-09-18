from pathlib import Path
import argparse, json, os, subprocess

p = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("analyze_job", nargs="?")
parser.add_argument("--plan", type=Path, default=p / "capture_plan.json")
args = parser.parse_args()
base = json.loads((p / "env_iket.json").read_text())
cases = {c["label"]: c for c in json.loads((p / "capture_cases.json").read_text())}
steps = ([{"label": c["label"], "phase": "analyze"} for c in cases.values()
          if "profile" in c["phases"]] if args.analyze_job else
         json.loads(args.plan.read_text()))
for step in steps:
    case = cases[step["label"]]
    phase = step["phase"]
    assert phase == "analyze" or phase in case["phases"], step
    cmd = ["bash", str(p / "profile_megamoe_iket.sbatch")]
    env = {**os.environ, **base, **case["env"], **step.get("env", {})}
    print("CAPTURE_STEP=" + json.dumps(step), flush=True)
    if phase == "analyze":
        cmd += ["analyze", str(p / "iket" / case["label"] / args.analyze_job)]
    else:
        env["FASTAFD_IKET_CAPTURE_PHASE"] = phase
    subprocess.run(cmd, env=env, check=True, timeout=240)
