#!/usr/bin/env python3
"""Summarize device metrics inside kernels from the same Nsight SQLite report."""

import argparse
from bisect import bisect_left, bisect_right
import json
from pathlib import Path
import sqlite3
from statistics import mean, median


def percentage(name):
    return name.endswith(('[Throughput %]', '[Ratio %]'))


def summarize(path, kernel, samples, margin):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        kernels = list(db.execute(
            "SELECT k.start,k.end,k.deviceId,k.globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,?)>0 ORDER BY k.start",
            (kernel,)))
        if len(kernels) != samples or len({(k['deviceId'], k['globalPid']) for k in kernels}) != 1:
            raise ValueError(f"Require {samples} kernels on one process/device: {path}")
        if any(a['end'] > b['start'] for a, b in zip(kernels, kernels[1:])):
            raise ValueError(f"Overlapping selected kernels: {path}")
        cuda = list(db.execute("SELECT * FROM TARGET_INFO_CUDA_DEVICE WHERE cudaId=?",
                               (kernels[0]['deviceId'],)))
        if len(cuda) != 1:
            raise ValueError(f"Ambiguous CUDA device mapping: {path}")
        gpu = dict(db.execute("SELECT * FROM TARGET_INFO_GPU WHERE id=? AND uuid=?",
                              (cuda[0]['gpuId'], cuda[0]['uuid'])).fetchone())
        types = list(db.execute("SELECT DISTINCT typeId FROM TARGET_INFO_GPU_METRICS "
                                "WHERE (typeId & 255)=?", (gpu['id'],)))
        if len(types) != 1:
            raise ValueError(f"Ambiguous hardware metric device mapping: {path}")
        type_id = types[0][0]
        names = dict(db.execute("SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS "
                                "WHERE typeId=?", (type_id,)))
        values = {i: [] for i, n in names.items()
                  if percentage(n) or 'Clock Frequency' in n}
        for row in db.execute("SELECT timestamp,metricId,value FROM GPU_METRICS "
                              "WHERE typeId=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
                              (type_id, kernels[0]['start'], kernels[-1]['end'])):
            if row['metricId'] in values:
                values[row['metricId']].append((row['timestamp'], row['value']))
        diagnostics = [dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT WHERE severity>=2')]
        if any('overflow' in r['text'].lower() or 'gpu metric' in r['text'].lower()
               for r in diagnostics):
            raise ValueError(f"Hardware metric diagnostic requires review: {diagnostics}")
        result = {}
        for metric_id, rows in values.items():
            name = names[metric_id]
            if not rows:
                raise ValueError(f"Missing metric {name}: {path}")
            timestamps = [t for t, _ in rows]
            gaps = [b-a for a, b in zip(timestamps, timestamps[1:])]
            if not gaps or max(gaps) > margin or min(gaps) <= 0:
                raise ValueError(f"Margin must cover every sampling interval: {name}/{path}")
            metric = {'median_interval_ns': median(gaps), 'max_interval_ns': max(gaps)}
            for label, edge in [('whole_kernel', 0), ('interior', margin)]:
                per_call, all_values = [], []
                for k in kernels:
                    lo = bisect_left(timestamps, k['start'] + edge)
                    hi = bisect_right(timestamps, k['end'] - edge)
                    selected = [v for _, v in rows[lo:hi]]
                    if not selected:
                        raise ValueError(f"No {label} samples for a kernel: {name}/{path}")
                    per_call.append(mean(selected))
                    all_values.extend(selected)
                if percentage(name) and not all(0 <= v <= 100 for v in all_values):
                    raise ValueError(f"Invalid percentage samples: {name}/{path}")
                ordered = sorted(all_values)
                metric[label] = dict(kernel_mean=mean(per_call), sample_mean=mean(all_values),
                                     sample_count=len(all_values), sample_p90=ordered[int(.9*(len(ordered)-1))],
                                     sample_max=ordered[-1])
            # Clock exports retain raw integer units; do not infer scaling from display labels.
            metric['units'] = 'percent' if percentage(name) else 'raw_export_integer'
            result[name] = metric
        return dict(database=str(path), gpu={k: gpu[k] for k in ('id', 'uuid', 'busLocation', 'name')},
                    type_id=type_id, kernels=len(kernels), edge_margin_ns=margin,
                    kernel_mean_us=mean(k['end']-k['start'] for k in kernels)/1000,
                    diagnostics=diagnostics, metrics=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('databases', nargs='+', type=Path)
    parser.add_argument('--kernel', required=True, help='Substring of demangled kernel name')
    parser.add_argument('--samples', required=True, type=int)
    parser.add_argument('--edge-margin-ns', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.edge_margin_ns <= 0:
        parser.error('Require at least two kernels and a positive edge margin')
    records = [summarize(p, args.kernel, args.samples, args.edge_margin_ns) for p in args.databases]
    result = dict(scope='Device-wide samples aligned only to kernels in the same report and GPU. '
                  'Interior excludes the requested margin at both edges. Not weight-specific bytes '
                  'or per-instruction stall attribution; other ranks are not time-aligned.',
                  records=records)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS',reports=len(records),output=str(args.output))))


if __name__ == '__main__':
    main()
