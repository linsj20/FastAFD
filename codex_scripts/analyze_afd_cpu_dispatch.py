#!/usr/bin/env python3
"""Correlate coordinator CPU ranges and one worker's Nsight graph launches.

The two CPU traces must come from the same host (shared monotonic clock).
Match worker CPU/NVTX ranges to align Nsight without wall-clock assumptions.
"""
import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path

from analyze_afd_attention_gap import graphs, span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinator", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coordinator = json.loads(args.coordinator.read_text())
    worker = json.loads(args.worker.read_text())
    if coordinator["metadata"]["hostname"] != worker["metadata"]["hostname"]:
        raise ValueError("CPU traces require a shared host monotonic clock")
    ce, we = coordinator["events"], worker["events"]
    with sqlite3.connect(args.sqlite) as db:
        db.row_factory = sqlite3.Row
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        ranges = [dict(r) for r in db.execute("SELECT * FROM NVTX_EVENTS")]
        for r in ranges:
            r["label"] = r.get("text") or names.get(r.get("textId"), "")
        runtime = {r["correlationId"]: dict(r) for r in db.execute(
            "SELECT start,end,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        main_tids = {r["globalTid"] for r in ranges
                     if r["label"].startswith("AFD_AG_MaterializeState[step=")}
        if len(main_tids) != 1:
            raise ValueError(f"Expected one materialization thread, got {main_tids}")
        main_tid = main_tids.pop()
        scheduler = [dict(r) for r in db.execute(
            "SELECT * FROM SCHED_EVENTS WHERE globalTid=? ORDER BY start", (main_tid,))]
    gpu = graphs(args.sqlite)
    offsets = []
    for step in range(1, len(gpu) + 1):
        label = f"AFD_AG_MaterializeState[step={step}]"
        cpu = sorted((e for e in we if e["name"] == label), key=lambda e: e["start_perf_ns"])
        nvtx = sorted((r for r in ranges if r["label"] == label), key=lambda r: r["start"])
        if len(cpu) != len(nvtx) or not cpu:
            raise ValueError(f"Unmatched CPU/NVTX step {step}: {len(cpu)} vs {len(nvtx)}")
        offsets.extend(e["start_perf_ns"] - r["start"] for e, r in zip(cpu, nvtx))
    offset = round(statistics.median(offsets))
    running = []
    for left, right in zip(scheduler, scheduler[1:]):
        if left["isSchedIn"] and not right["isSchedIn"]:
            running.append((left["start"]+offset, right["start"]+offset))
    def on_cpu_ms(lo, hi):
        if not scheduler or lo < scheduler[0]["start"]+offset or hi > scheduler[-1]["start"]+offset:
            return None
        return sum(max(0, min(hi, end)-max(lo, start)) for start, end in running)/1e6
    rows = []
    for step, graph in enumerate(gpu, 1):
        events = [e for e in ce if e["step_id"] == step]
        stages = collections.defaultdict(list)
        for e in events:
            stages[e["name"].split("[")[0].removeprefix("AFD_Coordinator_")].append(e)
        launch = stages["LaunchAG"][0]
        mat = [e for e in we if e["name"] == f"AFD_AG_MaterializeState[step={step}]"]
        api = runtime[graph[0]["correlationId"]]
        assert "cudaGraphLaunch" in names[api["nameId"]]
        gs, ge = span(graph)
        api_start = api["start"] + offset
        first_mat = min(e["start_perf_ns"] for e in mat)
        last_mat = max(e["end_perf_ns"] for e in mat)
        results = stages["CompleteResults"]
        row = dict(step=step,
            schedule_ms=sum(e["duration_ns"] for e in stages["ScheduleDps"])/1e6,
            build_ms=sum(e["duration_ns"] for e in stages["BuildDpLaunches"])/1e6,
            launch_attention_ms=launch["duration_ns"]/1e6,
            launch_ffn_group_envelope_ms=(stages["LaunchEG"][0]["start_perf_ns"]-launch["end_perf_ns"])/1e6,
            collect_ms=sum(e["duration_ns"] for e in stages["CompleteCollect"])/1e6,
            complete_results_scopes_ms=sum(e["duration_ns"] for e in results)/1e6,
            complete_results_envelope_ms=(max(e["end_perf_ns"] for e in results)-min(e["start_perf_ns"] for e in results))/1e6,
            worker_materialize_ms=(last_mat-first_mat)/1e6,
            worker_first_materialize_to_launch_ms=(api_start-first_mat)/1e6,
            worker_preparation_on_cpu_ms=on_cpu_ms(first_mat, api_start),
            attention_dispatch_start_to_ffn_api_ms=(api_start-launch["start_perf_ns"])/1e6,
            attention_dispatch_end_to_ffn_materialize_ms=(first_mat-launch["end_perf_ns"])/1e6,
            graph_ms=(ge-gs)/1e6,
            cpu_launch_start_ns=api_start,
            coordinator_launch_start_ns=launch["start_perf_ns"])
        if step > 1:
            prev_end = span(gpu[step-2])[1] + offset
            row.update(previous_gpu_end_to_materialize_ms=(first_mat-prev_end)/1e6,
                       previous_gpu_end_to_launch_api_ms=(api_start-prev_end)/1e6,
                       launch_cadence_ms=(launch["start_perf_ns"]-rows[-1]["coordinator_launch_start_ns"])/1e6)
        rows.append(row)
    cycles = []
    starts = [next(e for e in ce if e["name"].startswith(f"AFD_Coordinator_ScheduleDps[step={s}]"))
              for s in range(1, len(gpu)+1)]
    for step, (left, right) in enumerate(zip(starts, starts[1:]), 1):
        lo, hi = left["start_perf_ns"], right["start_perf_ns"]
        times = collections.defaultdict(float)
        for e in ce:
            if e["depth"] != 0:
                continue
            overlap = max(0, min(hi, e["end_perf_ns"])-max(lo, e["start_perf_ns"]))
            times[e["name"].split("[")[0].removeprefix("AFD_Coordinator_")] += overlap/1e6
        cycle_ms = (hi-lo)/1e6
        cycles.append(dict(step=step, cycle_ms=cycle_ms,
                           stages_ms={k:v for k,v in times.items() if v},
                           outside_ranges_ms=cycle_ms-sum(times.values())))
    output = dict(coordinator=str(args.coordinator), worker=str(args.worker), sqlite=str(args.sqlite),
                  alignment_offset_ns=offset, alignment_spread_us=(max(offsets)-min(offsets))/1e3,
                  alignment_samples=len(offsets), sqlite_tables=tables, steps=rows, cycles=cycles,
                  scheduler=dict(main_global_tid=main_tid, events=scheduler))
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
