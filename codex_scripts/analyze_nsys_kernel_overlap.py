#!/usr/bin/env python3
"""Summarize a kernel's launch resources and concurrent work in a Nsight export."""
import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path


def summary(rows):
    values = sorted(r["duration_us"] for r in rows)
    if not values:
        return {"count": 0}
    return dict(count=len(values), min_us=values[0], median_us=statistics.median(values),
                mean_us=statistics.mean(values), p95_us=values[int(.95*(len(values)-1))],
                max_us=values[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite", type=Path)
    p.add_argument("--kernel", required=True)
    p.add_argument("--skip-graphs", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with sqlite3.connect(args.sqlite) as db:
        db.row_factory = sqlite3.Row
        kernels = [dict(r) for r in db.execute(
            "SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start")]
        gpu = [dict(r) for r in db.execute("SELECT * FROM TARGET_INFO_GPU")]
    grouped = collections.defaultdict(list)
    for k in kernels:
        if k.get("graphId") is not None:
            grouped[k["correlationId"]].append(k)
    groups = sorted(grouped.values(), key=lambda g: g[0]["start"])
    rows = []
    for step, graph in enumerate(groups, 1):
        if step <= args.skip_graphs:
            continue
        selected = [k for k in graph if args.kernel in k["name"]]
        for ordinal, k in enumerate(selected):
            others = []
            for o in graph:
                if o is k or o["end"] <= k["start"] or o["start"] >= k["end"]:
                    continue
                others.append(dict(name=o["name"], stream=o["streamId"],
                    start_us=(o["start"]-k["start"])/1e3,
                    end_us=(o["end"]-k["start"])/1e3,
                    overlap_us=(min(o["end"],k["end"])-max(o["start"],k["start"]))/1e3,
                    grid=[o.get("grid"+d) for d in "XYZ"],
                    shared=o.get("staticSharedMemory",0)+o.get("dynamicSharedMemory",0)))
            rows.append(dict(step=step, ordinal=ordinal, duration_us=(k["end"]-k["start"])/1e3,
                start_in_graph_us=(k["start"]-graph[0]["start"])/1e3,
                kernel={key:value for key,value in k.items() if key not in ("start","end")},
                overlaps=others))
    categories = {}
    for name, predicate in (
        ("bootstrap", lambda r:r["ordinal"]<2),
        ("later", lambda r:r["ordinal"]>=2),
        ("with_megamoe",lambda r:any("mega_moe_impl" in o["name"] for o in r["overlaps"])),
        ("without_megamoe",lambda r:not any("mega_moe_impl" in o["name"] for o in r["overlaps"])),
        ("without_other_compute",lambda r:not any("wait_" not in o["name"] for o in r["overlaps"])),
    ):
        categories[name] = summary([r for r in rows if predicate(r)])
    result = dict(source=str(args.sqlite),gpu=gpu,summary=summary(rows),categories=categories,
                  steps={str(s):summary([r for r in rows if r["step"]==s]) for s in range(args.skip_graphs+1,len(groups)+1)},
                  rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(args.output, json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
