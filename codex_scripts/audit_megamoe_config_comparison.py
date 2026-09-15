#!/usr/bin/env python3
"""Audit source comparisons with one or several fixed shapes per process."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from summarize_megamoe_abba import parse_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def label_cases(label, spec):
    if 'groups' in spec:
        match = re.fullmatch(r'(.+)_g(\d+)_([ab])', label)
        require(match is not None, f'unrecognized grouped label {label}')
        config, group, repeat = match.groups()
        rows = spec['groups'][int(group)]
        return config, rows if repeat == 'a' else list(reversed(rows)), repeat
    match = re.fullmatch(r'(.+)_rows(\d+)_([ab])', label)
    require(match is not None, f'unrecognized label {label}')
    config, rows, repeat = match.groups()
    return config, [int(rows)], repeat


def audit_run(root, plan, spec, allow_stopped):
    env, job, scheduler = read(root / 'env.json'), read(root / 'job.json')['job'], read(root / 'scheduler.json')
    require(str(scheduler['job']) == str(job), f'{root}: scheduler job mismatch')
    complete = scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0'
    stopped = scheduler['state'].split()[0] == 'CANCELLED'
    require(complete or (allow_stopped and stopped), f'{root}: job not successful')
    require(env['MEGAMOE_ABBA_NSYS'] == '0', f'{root}: profiling enabled')
    log = root / f'compare-{job}.out'
    records, planned = parse_records(log), env['MEGAMOE_ABBA_ORDER'].split()
    require(planned == [label for label in spec['order'] if label in planned], f'{root}: changed trial order')
    require([label for label, _ in records] == planned[:len(records)], f'{root}: unexpected records')
    require(not complete or len(records) == len(planned), f'{root}: missing trials')
    sections = re.split(r'(?m)^CASE=(\S+) SOURCE=(\S+)\s*$', log.read_text())
    require((len(sections) - 1) // 3 in ({len(records)} if complete else {len(records), len(records) + 1}),
            f'{root}: missing source markers')
    require(sections[1::3] == planned[:len(sections[1::3])], f'{root}: source order mismatch')
    cases = {sections[i]: sections[i + 1:i + 3] for i in range(1, len(sections), 3)}
    # A following CASE proves the preceding distributed srun exited zero.
    successful_trials = {label for label, _ in records} if complete else set(sections[1::3][:-1])
    grouped, measurements, successful_measurements = defaultdict(dict), {}, []
    for label, record in records:
        config, rows_list, repeat = label_cases(label, spec)
        variant = config.split('_', 1)[0]
        source, section = cases[label]
        require(source == plan['configs'][variant]['source'] == env[f'FASTAFD_{variant.upper()}_SOURCE'],
                f'{label}: source mismatch')
        require(record['status'] == 'ok' and record['expert_weight_dtype'] == env['MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE'],
                f'{label}: failed/wrong dtype')
        expected_keys = {f"bucket{env['MEGAMOE_SMOKE_BUCKET']}_rows{rows}_sms{env['MEGAMOE_SMOKE_SMS']}"
                         for rows in rows_list}
        require(set(record['exact_qwen']) == expected_keys, f'{label}: missing/extra shapes')
        if 'groups' in spec:
            require([int(x) for x in re.findall(r'MEGAMOE_FIXED_CASE_OK rows=(\d+)', section)] == rows_list,
                    f'{label}: shape execution order mismatch')
        checks = sorted((int(rank), int(tokens)) for rank, tokens in re.findall(
            r'MEGAMOE_COMBINE_EXACT_OK rank=(\d+) global_tokens=(\d+)', section))
        require(checks == sorted((rank, rows * spec['ep']) for rows in rows_list for rank in range(spec['ep'])),
                f'{label}: incomplete exact checks')
        for rows in rows_list:
            require(rows in spec['rows'], f'{label}: unexpected rows')
            metric = record['exact_qwen'][f"bucket{env['MEGAMOE_SMOKE_BUCKET']}_rows{rows}_sms{env['MEGAMOE_SMOKE_SMS']}"]
            expected = {
                'ep_size': spec['ep'], 'rows_per_rank': rows, 'global_tokens': rows * spec['ep'],
                'valid_rows_per_rank': [rows] * spec['ep'], 'timing_warmups': 200,
                'timing_iterations': 2000, 'timing_graph_batch': 100,
                'timing_rank_reduction': 'max', 'timing_cuda_profiler_range': 0,
                'combine_exact_checked': True, 'routes_prepared': True,
                'timed_scope': 'route_producer_and_megamoe',
                'routing': 'fixed_global_sequence_rank_interleaved_balance',
                'model_dimensions': {key: int(env['MEGAMOE_SMOKE_' + var]) for key, var in
                                     [('hidden', 'HIDDEN'), ('intermediate', 'INTERMEDIATE'),
                                      ('experts', 'NUM_EXPERTS'), ('top_k', 'TOP_K')]},
            }
            for key, value in expected.items():
                require(metric.get(key) == value, f'{label}/rows{rows}: {key}: {metric.get(key)!r} != {value!r}')
            if config in plan.get('payload_modes', {}):
                require(metric.get('payload_mode') == plan['payload_modes'][config],
                        f'{label}/rows{rows}: actual payload mode differs from plan')
            for setting, key in [('WARMUPS', 'timing_warmups'), ('ITERATIONS', 'timing_iterations'),
                                 ('GRAPH_BATCH', 'timing_graph_batch')]:
                require(int(env['MEGAMOE_SMOKE_' + setting]) == metric[key], f'{label}: env mismatch')
            value = float(metric['milliseconds']) * 1000
            require(math.isfinite(value) and value > 0, f'{label}: invalid latency')
            key = f'{config}_rows{rows}_{repeat}'
            require(key not in measurements, f'{label}: duplicate measurement')
            measurements[key] = {'trial': label, 'metric': metric}
            if label in successful_trials:
                successful_measurements.append(key)
            grouped[rows].setdefault(config, {})[repeat] = value
    return {'root': str(root), 'job': job, 'scheduler': scheduler, 'records': len(records),
            'case_measurements': len(measurements), 'exact_rank_checks': len(measurements) * spec['ep'],
            'stopped': stopped, 'successful_trial_labels': sorted(successful_trials),
            'successful_measurements': successful_measurements, 'grouped': dict(grouped),
            'measurements': measurements, 'all_records': dict(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--stopped', action='store_true', help='Audit an explicitly stopped comparison and preserve partial samples')
    args = parser.parse_args()
    plan = read(args.root / 'plan.json')
    results, audit, partial, missing, summaries = [], {}, [], [], {}
    for model, spec in plan['models'].items():
        require(len(spec['order']) == spec['expected_records'], f'{model}: invalid plan count')
        expected_labels = defaultdict(set)
        for label in spec['order']:
            config, rows_list, repeat = label_cases(label, spec)
            for rows in rows_list:
                expected_labels[rows].add(f'{config}_rows{rows}_{repeat}')
        require(set(expected_labels) == set(spec['rows']), f'{model}: inconsistent plan rows')
        roots = [args.root / model] + [args.root / path for path in spec.get('additional_runs', [])]
        runs = [audit_run(root, plan, spec, args.stopped) for root in roots]
        audit[model] = [{k: v for k, v in run.items() if k != 'grouped'} for run in runs]
        for rows in spec['rows']:
            complete_groups = []
            for run in runs:
                labels = {label for label in run['measurements'] if f'_rows{rows}_' in label}
                if labels == expected_labels[rows] and labels <= set(run['successful_measurements']):
                    complete_groups.append(run)
                else:
                    for label in sorted(labels):
                        partial.append({'model': model, 'job': run['job'], 'label': label,
                                        'us': run['measurements'][label]['metric']['milliseconds'] * 1000})
            if not complete_groups and args.stopped:
                missing.append({'model': model, 'rows': rows})
                continue
            require(len(complete_groups) == 1, f'{model}/{rows}: need exactly one complete within-allocation comparison')
            run, expected_count = complete_groups[0], len(expected_labels[rows]) // 2
            configs = run['grouped'][rows]
            output_maxima = {run['measurements'][label]['metric']['output_max']
                             for label in expected_labels[rows]}
            require(len(output_maxima) == 1 and all(math.isfinite(v) for v in output_maxima),
                    f'{model}/{rows}: output maxima differ across configurations/repeats')
            require(len(configs) == expected_count and all(set(v) == {'a', 'b'} for v in configs.values()),
                    f'{model}/{rows}: missing configs/repeats')
            means = {key: mean(values.values()) for key, values in configs.items()}
            selected = means['candidate_selected']
            result = {'model': model, 'rows': rows, 'ep': spec['ep'], 'job': run['job'],
                      'nodes': run['scheduler']['nodes'], 'repeats_us': configs, 'means_us': means,
                      'selected_latency_delta_percent': {key: (selected / value - 1) * 100 for key, value in means.items()},
                      'lowest_mean_config': min(means, key=means.get)}
            historical = plan.get('historical_configs', [])
            if historical:
                require(set(historical) <= set(means), f'{model}/{rows}: missing historical configuration')
                best = min(historical, key=means.get)
                result.update(
                    fastest_historical_config=best,
                    historical_mean_us=means[best],
                    latency_delta_vs_history_percent={key: 100 * (value / means[best] - 1)
                                                      for key, value in means.items()},
                    lower_than_history_both_repeats={
                        key: all(values[r] < configs[best][r] for r in ('a', 'b'))
                        for key, values in configs.items()},
                )
            if 'alternate_compact' in means:
                pull, push = configs['candidate_selected'], configs['alternate_compact']
                result.update(compact_latency_reduction_percent=100 * (1 - means['alternate_compact'] / selected),
                              compact_lower_both_repeats=all(push[r] < pull[r] for r in ('a', 'b')),
                              selected_lower_both_repeats=all(pull[r] < push[r] for r in ('a', 'b')),
                              repeat_ranges_overlap=not (max(push.values()) < min(pull.values()) or
                                                         max(pull.values()) < min(push.values())))
            results.append(result)
        model_cases = [c for c in results if c['model'] == model]
        if model_cases:
            summaries[model] = {'cases': len(model_cases)}
            if all('compact_latency_reduction_percent' in c for c in model_cases):
                changes = [c['compact_latency_reduction_percent'] for c in model_cases]
                summaries[model].update({
                    'compact_lower_mean': sum(v > 0 for v in changes),
                    'selected_lower_mean': sum(v < 0 for v in changes),
                    'compact_lower_both_repeats': sum(c['compact_lower_both_repeats'] for c in model_cases),
                    'selected_lower_both_repeats': sum(c['selected_lower_both_repeats'] for c in model_cases),
                    'overlapping_repeat_ranges': sum(c['repeat_ranges_overlap'] for c in model_cases),
                    'mean_case_compact_latency_reduction_percent': mean(changes),
                    'median_case_compact_latency_reduction_percent': median(changes),
                    'min_max_case_compact_latency_reduction_percent': [min(changes), max(changes)],
                })
            if plan.get('historical_configs'):
                summaries[model]['historical_comparison'] = {
                    config: {
                        'lower_mean_cases': sum(c['means_us'][config] < c['historical_mean_us'] for c in model_cases),
                        'lower_both_repeats_cases': sum(c['lower_than_history_both_repeats'][config] for c in model_cases),
                        'mean_case_latency_delta_percent': mean(c['latency_delta_vs_history_percent'][config] for c in model_cases),
                    } for config in model_cases[0]['means_us']
                }
    status = 'PARTIAL_STOPPED' if missing or any(run['stopped'] for runs in audit.values() for run in runs) else 'PASS'
    output = {'status': status, 'scope': plan['scope'], 'audit': audit, 'cases': results, 'summary': summaries,
              'partial_case_samples': partial, 'incomplete_cases': missing,
              'interpretation': 'Lowest sampled mean is descriptive. Repeat ranges are not confidence intervals; no universal optimality claim.'}
    (args.root / 'audit.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Unprofiled configuration comparison', '', plan['scope'], '', 'Audit status: ' + status, '',
             'Microseconds per invocation, reduced to the slowest rank. Each complete case uses one allocation.', '']
    if plan.get('historical_configs'):
        lines += ['## Against historical pull-dispatch / push-combine versions', '',
                  'For each case, the historical reference is the lower two-repeat mean of: ' +
                  ', '.join(plan['historical_configs']) + '. These are matched reruns of frozen sources.',
                  'Negative delta means lower latency. Means below average per-case percentage changes. '
                  'Old NSys minima are separate provenance, not directly compared with these CUDA-event timings.', '',
                  '| Model | Configuration | Lower mean cases | Lower in both repeats | Mean latency delta |',
                  '|---|---|---:|---:|---:|']
        for model, summary in summaries.items():
            for config, values in summary.get('historical_comparison', {}).items():
                lines.append(f"| {model} | {config} | {values['lower_mean_cases']}/{summary['cases']} | "
                             f"{values['lower_both_repeats_cases']}/{summary['cases']} | "
                             f"{values['mean_case_latency_delta_percent']:+.3f}% |")
        lines += ['', 'Choosing the best observed historical mean is descriptive; these two repeats do not establish statistical optimality.', '']
    for model, runs in audit.items():
        for run in runs:
            lines.append(f"- {model} job {run['job']}: {run['scheduler']['state']}, exit {run['scheduler']['exit_code']}, "
                         f"{run['records']} process records / {run['case_measurements']} case measurements.")
    lines.append('')
    csv_rows = []
    for case in results:
        lines += [f"## {case['model']} EP{case['ep']}, rows/rank {case['rows']}", '',
                  f"Job {case['job']}; nodes {case['nodes']}.", '',
                  '| Configuration | Repeat A | Repeat B | Mean | Selected latency vs config |',
                  '|---|---:|---:|---:|---:|']
        for config, values in case['repeats_us'].items():
            delta = case['selected_latency_delta_percent'][config]
            lines.append(f"| {config} | {values['a']:.3f} | {values['b']:.3f} | {case['means_us'][config]:.3f} | {delta:+.3f}% |")
            csv_rows.append({'model': case['model'], 'ep': case['ep'], 'rows': case['rows'], 'job': case['job'],
                             'config': config, 'repeat_a_us': values['a'], 'repeat_b_us': values['b'],
                             'mean_us': case['means_us'][config], 'selected_latency_delta_percent': delta})
        lines.append('')
    if missing:
        lines += ['## Incomplete cases', '', ', '.join(f"{c['model']} rows{c['rows']}" for c in missing), '']
    if partial:
        lines += ['## Interrupted case samples — retained separately', '',
                  'These incomplete cases are not used for a winner claim.', '',
                  '| Model | Job | Trial | Microseconds |', '|---|---|---|---:|']
        for sample in partial:
            lines.append(f"| {sample['model']} | {sample['job']} | {sample['label']} | {sample['us']:.3f} |")
    (args.root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if csv_rows:
        with (args.root / 'timings.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(json.dumps({'status': status, 'cases': len(results), 'partial_records': len(partial), 'summary': summaries}))


if __name__ == '__main__':
    main()
