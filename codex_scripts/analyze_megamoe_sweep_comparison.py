#!/usr/bin/env python3
"""CPU-analyze mirrored sweep arms, retaining every rank and measured call."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
import sys


def comparison_arms(plan):
    """The first prefix is the control; every other prefix has a paired repeat."""
    prefixes = plan.get("arm_prefixes", ["control", "candidate"])
    if (not isinstance(prefixes, list) or len(prefixes) < 2
            or any(not isinstance(p, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", p)
                   for p in prefixes)):
        raise ValueError("Require at least two distinct directory-safe arm prefixes")
    arms = prefixes + [p + "_repeat" for p in reversed(prefixes)]
    if len(set(arms)) != len(arms):
        raise ValueError("Arm prefixes and repeat names overlap")
    return prefixes, arms


def compact_summary(root):
    """Reduce audited, retained rank-local records; never align rank clocks."""
    audit = json.loads((root / "analysis/audit.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("Compact summary requires completed comparison audit")
    job = json.loads((root / "job.json").read_text())["job"]
    result = dict(status="AUDITED_COMPARISON", gpu_job=job, audit=audit, cases={},
                  scope="Ordinary full-chain timing is authoritative; instrumented phases and CUDA-event rankmax are separate scopes. Every sample retained; nullable metrics remain null if any required sample is missing.")
    plan = json.loads((root / "plan.json").read_text())
    prefixes, arms = comparison_arms(plan)
    selections = {(Path(g["env"]["MEGAMOE_SWEEP_ROOT"]).parent.name, case)
                  for g in plan["groups"] for case in g["env"]["MEGAMOE_SWEEP_CASES"].split(",")}
    for model, case in sorted(selections):
        output = root / "analysis" / model / case
        startup = json.loads((output / "startup.json").read_text())["cases"]
        rows, phases, events = {}, {}, {}
        for arm in arms:
            data = startup[f"{arm}/{case}/{job}"]
            if len(data["ranks"]) != audit["ep"] or any(len(samples) != data["samples"] for samples in data["ranks"].values()):
                raise ValueError(f"Incomplete startup coverage: {model}/{case}/{arm}")
            rows[arm] = [dict(row) for samples in data["ranks"].values() for row in samples]
            for row in rows[arm]:
                row["first_mma_from_selector_start_us"] = row["producer_to_main_us"] + row["first_mma_issue_us"]
            phase = json.loads((output / f"{arm}_phase_means.json").read_text())
            phases[arm] = {key: phase[key] for key in ("means", "phase_duration_means")}
            baseline = json.loads((root / model / arm / case / job / "baseline.json").read_text())
            if baseline["status"] != "ok" or len(baseline["exact_qwen"]) != 1:
                raise ValueError(f"Unexpected ordinary result: {model}/{case}/{arm}")
            events[arm] = next(iter(baseline["exact_qwen"].values()))["milliseconds"] * 1000

        def reduce_records(records):
            keys = {key for row in records for key in row if key.endswith("_us")}
            return {key: mean(row[key] for row in records) if all(row.get(key) is not None for row in records) else None
                    for key in sorted(keys)}

        ordinary = json.loads((output / "comparison.json").read_text())
        if ordinary["status"] != "PASS_TIMESTAMP_AND_AGGREGATION_AUDIT":
            raise ValueError(f"Missing ordinary audit: {model}/{case}")
        result["cases"][f"{model}/{case}"] = dict(
            ordinary=ordinary, phases=phases, startup={arm: reduce_records(rows[arm]) for arm in arms},
            pooled_startup={kind: reduce_records(rows[kind] + rows[kind + "_repeat"])
                            for kind in prefixes}, cuda_event_rankmax_us=events)
    (root / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    deltas = {key: {p: value["ordinary"]["comparisons"][p]["delta_us"]["megamoe_end_us"]
                    for p in prefixes[1:]} for key, value in result["cases"].items()}
    print(json.dumps({key: next(iter(values.values())) if len(values) == 1 else values
                      for key, values in deltas.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--summary-only", action="store_true", help="Reduce existing audited outputs without trace exports")
    parser.add_argument("--ordinary-only", action="store_true",
                        help="Audit completed ordinary captures separately when profiles are incomplete")
    args = parser.parse_args()
    root = args.root
    if args.summary_only and args.ordinary_only:
        parser.error("--summary-only requires the full profile and ordinary audit")
    if args.summary_only:
        compact_summary(root)
        return
    workflow = Path(__file__).resolve().parent
    plan = json.loads((root / "plan.json").read_text())
    prefixes, arms = comparison_arms(plan)
    control, candidates = prefixes[0], prefixes[1:]
    job = json.loads((root / "job.json").read_text())["job"]
    ep = plan["ep"]
    models = {}
    executions = records = ordinary_calls = profile_samples = 0
    for group in plan["groups"]:
        env = group["env"]
        base = Path(env["MEGAMOE_SWEEP_ROOT"])
        arm, model = base.name, base.parent.name
        phase = env["FASTAFD_IKET_CAPTURE_PHASE"]
        cases = env["MEGAMOE_SWEEP_CASES"].split(",")
        if arm not in arms or phase not in ("normal", "profile") or not cases:
            raise ValueError(f"Expected explicit normal/profile paired group: {group['model']}")
        entry = models.setdefault(model, {})
        key = (arm, phase)
        if key in entry or base.parent != root / model:
            raise ValueError(f"Duplicate group or unexpected output root: {base}")
        entry[key] = (cases, env)
        # run-iket executes the correctness harness in its dry run and capture.
        repeats = 1 if phase == "normal" else 2
        executions += len(cases) * ep * repeats
        if env["MEGAMOE_RESULT_MODE"] == "record":
            records += len(cases) * ep * repeats
        if phase == "normal":
            ordinary_calls += len(cases) * ep * int(env["MEGAMOE_BASELINE_ITERATIONS"])
        else:
            profile_samples += len(cases) * ep * int(env["MEGAMOE_IKET_ITERATIONS"])
    if not models or ep < 1:
        raise ValueError("Expected nonempty positive-EP comparison")
    expected_groups = {(arm, phase) for arm in arms for phase in ("normal", "profile")}
    for model, groups in models.items():
        if set(groups) != expected_groups:
            raise ValueError(f"Incomplete paired-arm coverage: {model}")
        cases, env = groups[(control, "normal")]
        if any(c != cases for c, _ in groups.values()):
            raise ValueError(f"Cases differ across arms/phases: {model}")
        if any(int(e["MEGAMOE_IKET_EP"]) != ep or
               e["MEGAMOE_BASELINE_ITERATIONS"] != env["MEGAMOE_BASELINE_ITERATIONS"] or
               e["MEGAMOE_IKET_ITERATIONS"] != env["MEGAMOE_IKET_ITERATIONS"] or
               e.get("MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT", "separate") !=
               env.get("MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT", "separate") or
               e.get("MEGAMOE_SMOKE_PREPARE_BARRIER", "symmetric") !=
               env.get("MEGAMOE_SMOKE_PREPARE_BARRIER", "symmetric")
               for _, e in groups.values()):
            raise ValueError(f"EP/sample/preparation policy mismatch: {model}")

    def run(name, *arguments):
        subprocess.run([sys.executable, "-B", str(workflow / name),
                        *map(str, arguments)], check=True)

    output_root = root / ("ordinary_analysis" if args.ordinary_only else "analysis")
    output_root.mkdir(exist_ok=True)
    if args.ordinary_only:
        # The explicit mode requires a completed ordinary prefix, never selects
        # successful arms out of a partially completed ordinary comparison.
        normal = [g for g in plan["groups"] if g["env"]["FASTAFD_IKET_CAPTURE_PHASE"] == "normal"]
        if plan["groups"][:len(normal)] != normal:
            raise ValueError("Ordinary-only audit requires ordinary groups before profiles")
        log = (root / f"capture-{job}.out").read_text()
        starts = list(re.finditer(r"^SWEEP_CASE_START .*?$", log, re.M))
        expected_cases = [case for g in normal for case in g["env"]["MEGAMOE_SWEEP_CASES"].split(",")]
        if len(starts) < len(expected_cases):
            raise ValueError("Incomplete ordinary capture log")
        counts = Counter()
        for index, (group, case) in enumerate((g, c) for g in normal for c in g["env"]["MEGAMOE_SWEEP_CASES"].split(",")):
            env = group["env"]
            text = log[starts[index].start():starts[index+1].start() if index+1 < len(starts) else len(log)]
            marker = f"case={case} ep={ep} "
            if marker not in starts[index].group() or len(re.findall(r"^SWEEP_CASE_COMPLETE " + re.escape(marker), text, re.M)) != 1:
                raise ValueError(f"Incomplete ordinary case: {group['model']}/{case}")
            mode = env["MEGAMOE_RESULT_MODE"]
            for kind, pattern in (("record", r"MEGAMOE_RESULT_RECORD_OK case=\S+ rank=(\d+)"),
                                  ("compare", r"MEGAMOE_RESULT_COMPARE_OK case=\S+ rank=(\d+)"),
                                  ("replay", r"MEGAMOE_REPLAY_EXACT_OK rank=(\d+) rows=\d+ scope=\w+")):
                ranks = sorted(map(int, re.findall(pattern, text)))
                expected = list(range(ep)) if kind in (mode, "replay") else []
                if ranks != expected:
                    raise ValueError(f"Ordinary {kind} coverage: {group['model']}: {ranks}")
                counts[kind] += len(ranks)
            module = env.get("MEGAMOE_ROUTER_MODULE")
            modules = re.findall(r"MEGAMOE_ROUTER_MODULE_READY path=(\S+)", text)
            if Counter(modules) != (Counter({str(Path(module).resolve(strict=True)): ep}) if module else Counter()):
                raise ValueError(f"Ordinary module coverage: {group['model']}")
        ordinary_plan = dict(plan, groups=normal)
        (output_root / "plan.json").write_text(json.dumps(ordinary_plan, indent=2) + "\n")
        run("run_megamoe_sweep_group.py", "analyze", output_root / "plan.json", "--all-ranks", "--ordinary-only")
    else:
        run("run_megamoe_sweep_group.py", "analyze", root / "analysis/plan.json", "--all-ranks")
    for model, groups in models.items():
        cases, env = groups[(control, "normal")]
        for case in cases:
            output = output_root / model / case
            output.mkdir(parents=True, exist_ok=True)
            base = root / model
            runs = {arm: base / arm / case / job for arm in arms}
            samples = env["MEGAMOE_BASELINE_ITERATIONS"]
            layout = env.get("MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT", "separate")
            if layout not in ("separate", "joined") or (layout == "joined" and
                    env.get("MEGAMOE_SMOKE_PREPARE_BARRIER") != "collective"):
                raise ValueError("Graph identity audit requires separate or joined collective preparation")
            run("summarize_megamoe_router_timeline.py", base, f"{case}/{job}",
                "--arms", *arms, "--samples", samples, "--rank-mean", ep,
                "--require-graph-identity", "--require-preinitialized-events",
                *(["--joined-collective-preparation"] if layout == "joined" else []),
                "--retain-timestamps", "--output", output / "ordinary.json")
            run("compare_megamoe_rank_timelines.py", output / "ordinary.json",
                "--control", control,
                *[arg for candidate in candidates for arg in ("--candidate", candidate)], "--ranks", ep,
                "--samples", samples, "--output", output / "comparison.json")
            if args.ordinary_only:
                continue
            for arm, path in runs.items():
                run("summarize_megamoe_latency.py",
                    *[f"rank{r}={path}/rank_decomposition/rank-{r}/decomposition.json"
                      for r in range(ep)], "--output", output / f"{arm}_phases.json",
                    "--rank-mean-output", output / f"{arm}_phase_means.json")
            run("summarize_megamoe_startup.py", *runs.values(),
                "--samples", env["MEGAMOE_IKET_ITERATIONS"],
                "--output", output / "startup.json")
    if args.ordinary_only:
        audit = dict(status="PASS_ORDINARY_ONLY", counts=dict(counts), ordinary_calls=ordinary_calls,
                     measured_profile_samples=0, models=list(models), ep=ep,
                     scope="Complete ordinary prefix: exactness, replay, module selection and all rank/call timelines. Instrumented decomposition remains incomplete.")
        (output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit), flush=True)
        return
    log = (root / f"capture-{job}.out").read_text()
    patterns = dict(record=r"MEGAMOE_RESULT_RECORD_OK case=\S+ rank=\d+",
                    compare=r"MEGAMOE_RESULT_COMPARE_OK case=\S+ rank=\d+",
                    replay=r"MEGAMOE_REPLAY_EXACT_OK rank=\d+ rows=\d+ scope=\w+")
    counts = {key: len(re.findall(pattern, log)) for key, pattern in patterns.items()}
    expected = dict(record=records, compare=executions-records, replay=executions)
    if counts != expected:
        raise ValueError(f"Correctness log coverage: {counts}, expected {expected}")
    (root / "analysis/audit.json").write_text(json.dumps(dict(
        status="PASS", counts=counts, ordinary_calls=ordinary_calls,
        measured_profile_samples=profile_samples, models=list(models), ep=ep), indent=2) + "\n")
    print("SWEEP_COMPARISON_AUDIT_PASS", flush=True)
    compact_summary(root)


if __name__ == "__main__":
    main()
