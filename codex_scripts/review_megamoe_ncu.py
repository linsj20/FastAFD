#!/usr/bin/env python3
"""Audit planned all-rank NCU captures and export diagnostic hardware counters."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import subprocess

from analyze_megamoe_throughput import audit_correctness


def parse_raw_csv(data):
    """NCU raw CSV has metric columns, then a units row and workload rows."""
    lines = data.splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith('"ID",') and '"Kernel Name"' in line)
    rows = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(rows) != 3 or len(set(rows[0])) != len(rows[0]) or any(
            len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("Expected raw metric headers, units, and exactly one workload")
    units, values = (dict(zip(rows[0], row)) for row in rows[1:])
    if units["ID"] or units["Kernel Name"] or not values["ID"]:
        raise ValueError("Invalid NCU raw units/workload rows")
    return units, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--completed-prefix", action="store_true",
                        help="Audit a failed job's fully completed leading captures; label results partial")
    args = parser.parse_args()
    root = args.root
    job = json.loads((root / "gpu/job.json").read_text())["job"]
    plan = json.loads((root / "plan.json").read_text())
    log = (root / f"capture-{job}.out").read_text()
    executions = [(group, case) for group in plan["groups"]
                  for case in group["env"]["MEGAMOE_SWEEP_CASES"].split(",")]
    planned_captures = len(executions)
    if args.completed_prefix:
        completed = len(re.findall(r"^SWEEP_CASE_COMPLETE ", log, re.M))
        starts = list(re.finditer(r"^SWEEP_CASE_START .*?$", log, re.M))
        if not 0 < completed <= planned_captures or len(starts) not in (completed, completed + 1):
            raise ValueError("Expected one nonempty completed prefix and at most one failed capture")
        if len(starts) > completed:
            log = log[:starts[completed].start()]
        executions = executions[:completed]
    expected = []
    for group, case in executions:
        env = group["env"]
        manifest = json.loads((Path(env["MEGAMOE_SWEEP_ROOT"]) / "manifest.json").read_text())
        spec = next(c for c in manifest["cases"] if c["case_id"] == case)
        contract = dict(manifest["contract"], **spec.get("capture_contract", {}))
        label = f"fixed_bucket{contract['bucket']}_rows{spec['rows_per_rank']}_sms{contract['num_sms']}"
        if env.get("MEGAMOE_RESULT_MODE"):
            expected.append((env["MEGAMOE_RESULT_MODE"], label))
        expected.append(("replay", f"{spec['rows_per_rank']}/{env.get('MEGAMOE_SMOKE_TIMING_SCOPE', 'pipeline')}"))
    for event in ("START", "COMPLETE"):
        if len(re.findall(rf"^SWEEP_CASE_{event} ", log, re.M)) != len(executions):
            raise ValueError(f"Incomplete NCU capture coverage: {event}")
    starts = list(re.finditer(r"^SWEEP_CASE_START .*?$", log, re.M))
    for index, (group, case) in enumerate(executions):
        section = log[starts[index].start():starts[index + 1].start()
                      if index + 1 < len(starts) else len(log)]
        marker = f"case={case} ep={plan['ep']} "
        if marker not in starts[index].group() or len(re.findall(
                r"^SWEEP_CASE_COMPLETE " + re.escape(marker), section, re.M)) != 1:
            raise ValueError(f"Out-of-order or incomplete capture: {group['model']}/{case}")
    correctness = audit_correctness(log, expected, plan["ep"])
    records = []
    for group, case in executions:
        env = group["env"]
        mode = env.get("MEGAMOE_SMOKE_NCU_GRAPH_PROFILING", "node")
        if mode not in ("node", "graph"):
            raise ValueError(f"Unsupported profiling scope: {mode}")
        run = Path(env["MEGAMOE_SWEEP_ROOT"]) / case / job
        for rank in range(plan["ep"]):
            report = run / f"ranks/rank-{rank}/ncu/megamoe-rank{rank}.ncu-rep"
            if not report.is_file() or not report.stat().st_size:
                raise ValueError(f"Missing NCU report: {report}")
            output = root / "analysis" / group["model"] / group["arm"] / case / f"rank-{rank}.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            exported = subprocess.run([env["MEGAMOE_SMOKE_NCU"], "--import", str(report),
                "--csv", "--page", "raw", "--print-units", "base"],
                text=True, capture_output=True)
            output.write_text(exported.stdout)
            if exported.returncode:
                output.with_suffix(".err").write_text(exported.stderr)
                raise RuntimeError(f"NCU export failed ({exported.returncode}); see {output}: "
                                   f"{exported.stdout[-1000:]} {exported.stderr[-1000:]}")
            data = exported.stdout
            metric_units, values = parse_raw_csv(data)
            name = values["Kernel Name"]
            if ((mode == "node" and "sm100_fp8_fp4_mega_moe_impl" not in name)
                    or (mode == "graph" and "graph" not in name.lower())):
                raise ValueError(f"Unexpected {mode} workload {name}: {report}")
            metrics = {}
            for name, units in [("gpu__time_duration.sum", ("ns", "nsecond")),
                                ("dram__bytes_read.sum", ("byte",)),
                                ("dram__bytes_write.sum", ("byte",))]:
                if metric_units.get(name) not in units:
                    raise ValueError(f"Invalid {name} unit in {report}: {metric_units.get(name)}")
                value = float(values[name].replace(",", ""))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {name} value: {value}")
                metrics[name] = value
            if metrics["gpu__time_duration.sum"] <= 0:
                raise ValueError(f"Nonpositive duration: {report}")
            profiler = {k: v for k, v in values.items() if k.startswith("profiler__")}
            records.append(dict(model=group["model"], case=case, arm=group["arm"], rank=rank,
                                report=str(report), workload=values["Kernel Name"], graph_profiling=mode,
                                metrics=metrics, profiler=profiler))
    result = dict(status=("PARTIAL_CORRECTNESS_REPORT_COVERAGE_REVIEW_REQUIRED" if args.completed_prefix
                          else "PASS_CORRECTNESS_REPORT_COVERAGE_REVIEW_REQUIRED"), correctness=correctness,
        planned_captures=planned_captures,
        captures=len(executions), reports=len(records), records=records,
        scope="Diagnostic counters only. Review replay passes/cache effects before attribution. "
              "Durations are not ordinary latency. Graph mode includes preparation/reset, routing and "
              "MegaMoE. DRAM bytes include all profiled traffic, not only weights.")
    filename = "summary_partial.json" if args.completed_prefix else "summary.json"
    (root / "analysis" / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
