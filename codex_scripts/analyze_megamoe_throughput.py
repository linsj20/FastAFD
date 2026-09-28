#!/usr/bin/env python3
"""Audit ordinary mirrored throughput runs, including small-first group ordering."""

import argparse
import json
import math
from statistics import mean, median
from pathlib import Path
import re
import subprocess
import sys

from compare_megamoe_rank_timelines import METRICS, compare


def audit_correctness(log, expected, ep):
    # Slurm merges independent stdout streams; rank5's completed case can arrive
    # after rank0 starts the next one. Preserve each rank's order, not that merge.
    actual = {rank: [] for rank in range(ep)}
    pattern = (r"MEGAMOE_RESULT_(RECORD|COMPARE)_OK case=(\S+) rank=(\d+)"
               r"|MEGAMOE_REPLAY_EXACT_OK rank=(\d+) rows=(\d+) scope=(\w+)")
    counts = dict(record=0, compare=0, replay=0)
    for match in re.finditer(pattern, log):
        kind, case, rank, replay_rank, rows, scope = match.groups()
        rank = int(rank if kind else replay_rank)
        event = (kind.lower(), case) if kind else ("replay", f"{rows}/{scope}")
        if rank not in actual:
            raise ValueError(f"Unexpected correctness rank: {rank}")
        actual[rank].append(event)
        counts[event[0]] += 1
    for rank, events in actual.items():
        if events != expected:
            raise ValueError(f"Correctness sequence differs for rank {rank}: {events}")
    return counts


def geometric_speedups(cases, candidate, control):
    """Equal case weights; each latency includes both orders and all ranks."""
    if not cases:
        raise ValueError("Geometric mean requires a nonempty comparison set")
    result = {}
    for metric in ("kernel_us", "megamoe_end_us"):
        ratios = {}
        for case, comparison in cases.items():
            paired = comparison["paired_means"]
            baseline, trial = paired[control][metric], paired[candidate][metric]
            if any(not math.isfinite(value) or value <= 0 for value in (baseline, trial)):
                raise ValueError(f"Require positive finite latency: {case}/{metric}")
            ratios[case] = baseline / trial
        result[metric] = dict(case_speedups=ratios, case_count=len(ratios),
                             geometric_mean_speedup=math.exp(mean(map(math.log, ratios.values()))))
    return result


def review_normal(argv):
    from collect_cuda_rank_timings import aggregate_rank_timings

    parser = argparse.ArgumentParser(description='Audit mirrored normal-routing graph comparisons')
    parser.add_argument('root', type=Path)
    args = parser.parse_args(argv)
    root = args.root
    plan = json.loads((root / 'scope.json').read_text())
    ep, arms, candidate = plan['ep'], plan['arm_prefixes'], plan['candidate']
    expected_keys = {(m, t, a + suffix) for m, totals in plan['specs'].items()
                     for t in totals for a in arms for suffix in ('', '_repeat')}
    keys = [(x['model'], x['total'], x['arm']) for x in plan['runs']]
    if (plan['routing'] != 'normal' or len(set(keys)) != len(keys) or set(keys) != expected_keys
            or candidate not in arms or not keys):
        raise ValueError('Normal-route pair/order coverage differs from the plan')
    expected, records = [], {}
    for run in plan['runs']:
        path, env = Path(run['path']), run['env']
        result = json.loads((path / 'result.json').read_text())
        if (result['status'] != 'ok' or result['expert_weight_dtype'] != 'fp4'
                or result['numeric'] is not None or len(result['exact_qwen']) != 1):
            raise ValueError('Unexpected normal-route smoke result')
        info = next(iter(result['exact_qwen'].values()))
        required = dict(MEGAMOE_SMOKE_FIXED_PERFECT_ONLY='0', MEGAMOE_SMOKE_PERFECT_ROUTING='0',
                        MEGAMOE_SMOKE_SPECIALIZATION_MODES='1:1:1:1',
                        MEGAMOE_SMOKE_PREPARED_GRAPH_LAYOUT='joined', MEGAMOE_SMOKE_GRAPH_BATCH='1',
                        MEGAMOE_SMOKE_PREPARE_BARRIER='collective', DG_JIT_REQUIRE_CACHE='1')
        if (any(env.get(k) != v for k, v in required.items())
                or info['timed_scope'] != 'router_and_megamoe' or info['calls_per_graph'] != 1
                or info['prepared_graph_layout'] != 'joined' or info['dispatch_reset_outside_timing']
                or env.get('MEGAMOE_SMOKE_WEIGHT_CACHE_INPUT') or env.get('MEGAMOE_SMOKE_WEIGHT_CACHE_OUTPUT')):
            raise ValueError('Normal-route graph contract differs')
        saved = json.loads((path / 'rank_timings.json').read_text())
        audit = aggregate_rank_timings(saved['ranks'])
        if (audit['world_size'] != ep or audit['iterations'] != plan['calls_per_arm']
                or 'preparation barrier/reset' not in audit['scope']
                or abs(info['milliseconds'] - audit['maximum_rank_mean_ms']) >= 1e-9
                or any(abs(saved[k] - audit[k]) >= 1e-12
                       for k in ('mean_rank_mean_ms', 'maximum_rank_mean_ms'))):
            raise ValueError('Normal-route timing audit failed')
        ranks = []
        for rank in sorted(saved['ranks'], key=lambda x: x['rank']):
            vals = sorted(v * 1000 for v in rank['durations_ms'])
            ranks.append(dict(rank=rank['rank'], mean_us=mean(vals), median_us=median(vals),
                              p95_us=vals[math.ceil(.95 * len(vals)) - 1], max_us=vals[-1]))
        key = run['model'], run['total'], run['arm']
        records[key] = dict(model=run['model'], total=run['total'], arm=run['arm'], ranks=ranks,
                            mean_rank_mean_us=audit['mean_rank_mean_ms'] * 1000,
                            maximum_rank_mean_us=audit['maximum_rank_mean_ms'] * 1000,
                            max_active_experts=info['max_active_experts'])
        mode = env.get('MEGAMOE_RESULT_MODE')
        rows = int(env['MEGAMOE_SMOKE_ROWS'])
        if mode:
            if mode not in ('record', 'compare'):
                raise ValueError('Unexpected exactness mode')
            label = f"exact_bucket{env['MEGAMOE_SMOKE_BUCKET']}_rows{rows}_sms{env['MEGAMOE_SMOKE_SMS']}_trial0"
            expected += [(mode, x) for x in ['exact_inputs', label, label + '_alternate',
                         *[label + f'_normal{i}' for i in range(3)],
                         *[label + f'_padding{i}' for i in range(3)]]]
        expected.append(('replay', f'{rows}/pipeline'))
    logs = sorted((root / 'logs').glob('rank-*.log'))
    if {p.name for p in logs} != {f'rank-{rank}.log' for rank in range(ep)}:
        raise ValueError('Missing normal-route rank log')
    counts = audit_correctness('\n'.join(p.read_text() for p in logs), expected, ep)
    comparisons, geomeans = [], {}
    for control in (a for a in arms if a != candidate):
        ratios = {}
        for model, totals in plan['specs'].items():
            for total in totals:
                orders = []
                for suffix in ('', '_repeat'):
                    a, b = records[model, total, candidate + suffix], records[model, total, control + suffix]
                    delta = [dict(rank=ca['rank'], **{k: ca[k] - co[k] for k in ('mean_us', 'median_us', 'p95_us')})
                             for ca, co in zip(a['ranks'], b['ranks'])]
                    orders.append(dict(order='reverse' if suffix else 'forward',
                        candidate_mean_us=a['mean_rank_mean_us'], control_mean_us=b['mean_rank_mean_us'],
                        mean_delta_us=a['mean_rank_mean_us'] - b['mean_rank_mean_us'], rank_deltas=delta))
                comparisons.append(dict(model=model, total=total, candidate=candidate, control=control, orders=orders))
                ratios[f'{model}/ep{ep}_tokens{total}'] = (mean(x['control_mean_us'] for x in orders)
                                                         / mean(x['candidate_mean_us'] for x in orders))
        geomeans[control] = dict(case_speedups=ratios, geometric_mean_speedup=math.exp(mean(map(math.log, ratios.values()))))
    report = dict(status='PASS_NORMAL_ROUTE_COMPLETENESS', runs=len(records), correctness=counts,
                  rank_samples=len(records) * ep * plan['calls_per_arm'], records=list(records.values()),
                  comparisons=comparisons, geometric_speedups=geomeans,
                  scope='All-sample normal-routing joined-graph times including preparation/reset; separate from ordinary kernel/span scores. Rank-local clocks, no noise correction.')
    (root / 'analysis/review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'comparisons')}), flush=True)


def timeline_windows(trace, ranks, samples, window_size):
    """Summarize contiguous ordinal windows without trimming slow samples."""
    if (min(ranks, samples, window_size) <= 0 or trace["rank_count"] != ranks
            or trace["samples_per_rank"] != samples
            or set(trace["per_rank"]) != {str(rank) for rank in range(ranks)}):
        raise ValueError("Unexpected rank/sample coverage or window size")
    series = {}
    for rank, data in trace["per_rank"].items():
        rows = data["records"]
        gaps = data["inter_call_gaps_us"]
        if len(rows) != samples or len(gaps) != samples - 1:
            raise ValueError(f"Incomplete trace: rank {rank}")
        metrics = {key: [row[key] for row in rows] for key in METRICS}
        metrics["selector_duration_us"] = [row["selector_end_us"] - row["selector_start_us"]
                                           for row in rows]
        metrics["following_gap_us"] = gaps
        if any(not math.isfinite(value) or value < 0
               for values in metrics.values() for value in values):
            raise ValueError(f"Invalid duration: rank {rank}")
        series[rank] = metrics

    def summarize(start, stop):
        metrics = {}
        for key in next(iter(series.values())):
            values_by_rank = {rank: data[key][start:stop] for rank, data in series.items()}
            values = sorted(value for vals in values_by_rank.values() for value in vals)
            if not values:  # A final one-call window has no following gap.
                metrics[key] = dict(samples=0)
                continue
            rank_means = {rank: mean(vals) for rank, vals in values_by_rank.items()}
            metrics[key] = dict(samples=len(values), mean=mean(values), median=median(values),
                                p95=values[math.ceil(.95 * len(values)) - 1],
                                rank_mean_range=[min(rank_means.values()), max(rank_means.values())])
        return metrics

    overall = summarize(0, samples)
    for key in METRICS:
        if not math.isclose(overall[key]["mean"], trace["means"][key], rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Saved trace mean differs: {key}")
    return dict(rank_count=ranks, samples_per_rank=samples, overall=overall,
                windows=[dict(first_call=start, calls_per_rank=min(window_size, samples - start),
                              metrics=summarize(start, start + window_size))
                         for start in range(0, samples, window_size)])


def review_windows(argv):
    parser = argparse.ArgumentParser(description="Inspect saved all-rank timing windows; no new GPU run")
    parser.add_argument("root", type=Path)
    parser.add_argument("--cases", required=True, help="Comma-separated model/epN_tokensM keys")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads((args.root / "summary.json").read_text())
    cases = args.cases.split(",")
    if (summary["status"] != "PASS" or not cases or len(set(cases)) != len(cases)
            or set(cases) - set(summary["cases"]) or args.window_size <= 0):
        raise ValueError("Require passing summary, distinct recorded cases and positive window size")
    if args.output.exists():
        raise ValueError("Window review output already exists")
    report = dict(scope="Diagnostic subset, all recorded arms and samples retained. Contiguous "
                       "rank-local call ordinals, not aligned cross-rank clocks. Following gaps "
                       "belong to their preceding call; the final call has none. Stage co-variation "
                       "does not prove a cause or change performance acceptance.",
                  source=str(args.root), window_size=args.window_size, cases={})
    for case in cases:
        comparison = summary["cases"][case]
        arms = report["cases"][case] = {}
        for prefix in comparison["paired_means"]:
            for suffix in ("", "_repeat"):
                label = prefix + suffix
                path = args.root / "analysis" / case / (label + ".json")
                trace = json.loads(path.read_text())[label]
                arms[label] = timeline_windows(trace, comparison["rank_count"],
                                               comparison["samples_per_rank_pass"], args.window_size)
                arms[label]["source"] = str(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reviewed timing windows for {len(cases)} cases, all arms, orders and ranks", flush=True)


def review_interleaved(argv):
    """Audit same-process graph comparisons and separate AA/BB calibration."""
    import statistics
    from collect_cuda_rank_timings import aggregate_rank_timings

    parser = argparse.ArgumentParser(description=review_interleaved.__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="Separate diagnostic report; default analysis/review.json")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise ValueError("Interleaved review output already exists")
    r = args.root
    plan = json.loads((r / "scope.json").read_text())
    ep = plan['ep']
    iterations = plan['calls_per_arm']
    warmups = plan['warmups_per_arm']
    control = plan.get('control', 'control')
    candidate = plan.get('candidate', 'rows')
    assert control != candidate
    entry = plan.get('entry_protocol', 'queued')
    assert entry in ('queued', 'synchronized')
    pairs = plan.get('pairs', ['ab'])
    assert pairs in (['ab'], ['aa', 'ab', 'bb'])
    assert plan['routing'] in ('balanced', 'normal')
    assert ep > 0 and iterations > 0 and (iterations % 100 == 0)
    batch_field = 'timing_graph_batch' if plan['routing'] == 'balanced' else 'calls_per_graph'
    expected_keys = {(x.get('pair', 'ab'), x['model'], x['total'], x['order']) for x in plan['runs']}
    assert len(expected_keys) == len(plan['runs']) and plan['runs']
    for (model, total) in {(m, t) for (_, m, t, _) in expected_keys}:
        assert {(p, o) for (p, m, t, o) in expected_keys if (m, t) == (model, total)} == {(p, o) for p in pairs for o in ['abba', 'baab']}
    reviews = []
    for run in plan['runs']:
        p = Path(run['path'])
        v = json.loads((p / 'interleaved.json').read_text())
        result = json.loads((p / 'result.json').read_text())
        assert v['status'] == 'PASS_INTERLEAVED_EXACT' and result['status'] == 'ok' and (result['expert_weight_dtype'] == 'fp4') and (result['numeric'] is None)
        assert v['entry_protocol'] == entry and v['warmups_per_arm'] == warmups
        expected = json.loads(run['env']['MEGAMOE_INTERLEAVED_VARIANTS'])
        assert v['variants'] == expected
        labels = list(expected)
        assert labels == ([control, candidate] if run['order'] == 'abba' else [candidate, control])
        pair = run.get('pair', 'ab')
        if pair in ('aa', 'bb'):
            assert expected[control] == expected[candidate]
        (a, b) = labels
        assert v['schedule'] == [a, b, b, a] * (iterations // 2) and v['calls_per_arm'] == iterations
        assert set(v['arms']) == {control, candidate} and v['scalar_result_arm'] == a
        assert len(result['exact_qwen']) == 1
        info = next(iter(result['exact_qwen'].values()))
        assert info['prepared_graph_layout'] == 'joined' and info[batch_field] == 1 and (not info['dispatch_reset_outside_timing'])
        for (label, arm) in v['arms'].items():
            audit = aggregate_rank_timings(arm['ranks'])
            assert audit['world_size'] == ep and audit['iterations'] == iterations and ('preparation' in audit['scope'])
            for key in ['mean_rank_mean_ms', 'maximum_rank_mean_ms']:
                assert abs(audit[key] - arm[key]) < 1e-12
            assert all((x['metadata']['variant'] == expected[label] for x in arm['ranks']))
        assert abs(info['milliseconds'] - v['arms'][a]['maximum_rank_mean_ms']) < 1e-09
        values = {label: {x['rank']: [t * 1000 for t in x['durations_ms']] for x in arm['ranks']} for (label, arm) in v['arms'].items()}
        ranks = []
        for rank in range(ep):
            (c, d) = (values[control][rank], values[candidate][rank])
            ranks.append(dict(rank=rank, mean_delta_us=statistics.mean(d) - statistics.mean(c), median_delta_us=statistics.median(d) - statistics.median(c), p95_delta_us=sorted(d)[math.ceil(0.95 * iterations) - 1] - sorted(c)[math.ceil(0.95 * iterations) - 1]))
        windows = []
        for start in range(0, iterations, 100):
            means = {label: statistics.mean((x for rank in vals.values() for x in rank[start:start + 100])) for (label, vals) in values.items()}
            windows.append(dict(first_call_per_arm=start, calls_per_arm=100, means_us=means, delta_us=means[candidate] - means[control]))
        cycles = [statistics.mean(((values[candidate][rank][i] + values[candidate][rank][i + 1] - values[control][rank][i] - values[control][rank][i + 1]) / 2 for rank in range(ep))) for i in range(0, iterations, 2)]
        means = {label: arm['mean_rank_mean_ms'] * 1000 for (label, arm) in v['arms'].items()}
        reviews.append(dict(pair=pair, entry_protocol=v['entry_protocol'], model=run['model'], total=run['total'], order=run['order'], means_us=means, mean_delta_us=means[candidate] - means[control], slowest_rank_mean_delta_us=(v['arms'][candidate]['maximum_rank_mean_ms'] - v['arms'][control]['maximum_rank_mean_ms']) * 1000, rank_deltas=ranks, windows=windows, cycles_improved=sum((x < 0 for x in cycles)), cycles=len(cycles), cycle_delta_range_us=[min(cycles), max(cycles)]))
    logs = sorted((r / 'logs').glob('rank-*.log'))
    assert {p.name for p in logs} == {f'rank-{rank}.log' for rank in range(ep)}
    log = '\n'.join((p.read_text() for p in logs))
    counts = {key: len(re.findall('MEGAMOE_' + key + '_OK', log)) for key in ['RESULT_COMPARE', 'REPLAY_EXACT', 'INTERLEAVED_EXACT']}
    expected = plan['expected_correctness']
    assert counts == expected, counts
    for path in logs:
        per_rank = {key: len(re.findall('MEGAMOE_' + key + '_OK', path.read_text())) for key in expected}
        assert per_rank == {k: v // ep for (k, v) in expected.items()}, (path, per_rank)
    out = dict(status='PASS_INTERLEAVED_COVERAGE_AND_EXACT_OUTPUTS', processes=len(reviews), rank_samples=len(reviews) * ep * iterations * 2, correctness=counts, control=control, candidate=candidate, scope='Unfiltered rank-local joined graph times. Control and candidate share inputs, buffers, process and allocation; ABBA and BAAB capture/schedule orders. No pull comparison and no cross-rank clock alignment. Cycle observations are not assumed independent.', runs=reviews)
    out['geometric_speedups_by_pair'] = {}
    for pair in pairs:
        ratios = {}
        for (model, total) in sorted({(x['model'], x['total']) for x in reviews}):
            selected = [x for x in reviews if (x['model'], x['total'], x['pair']) == (model, total, pair)]
            ratios[f'{model}/ep{ep}_tokens{total}'] = statistics.mean((x['means_us'][control] for x in selected)) / statistics.mean((x['means_us'][candidate] for x in selected))
        out['geometric_speedups_by_pair'][pair] = dict(case_count=len(ratios), case_speedups=ratios, geometric_mean_speedup=math.exp(statistics.mean(map(math.log, ratios.values()))), scope='Equal case weights, both-order mean joined-graph ratios. AA/BB quantify identical-policy variation separately; AB is the policy comparison. No pull comparison or noise subtraction.')
    out['geometric_speedup'] = out['geometric_speedups_by_pair']['ab']
    (args.output if args.output else r / 'analysis/review.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for (k, v) in out.items() if k != 'runs'}))
    for x in reviews:
        print(json.dumps({k: v for (k, v) in x.items() if k not in ['rank_deltas', 'windows']}))


def review_ranks(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads((args.root / "summary.json").read_text())
    if summary["status"] != "PASS":
        raise ValueError("Requires a completed, passing campaign audit")
    report = dict(scope="All samples retained. Rank-local times; no cross-rank clock "
                       "alignment. p95 uses the nearest-rank definition. "
                       "Timing variation on shared code is retained, not corrected.", cases={})
    for case, comparison in summary["cases"].items():
        case_report = report["cases"][case] = {}
        for suffix in ("", "_repeat"):
            arms = {}
            for prefix in (args.control, args.candidate):
                label = prefix + suffix
                trace = json.loads((args.root / "analysis" / case / (label + ".json")).read_text())[label]
                if trace["rank_count"] != comparison["rank_count"]:
                    raise ValueError(f"Rank coverage differs: {case}/{label}")
                arms[prefix] = {}
                for metric in ("kernel_us", "megamoe_end_us"):
                    rank_means, rank_stats, values = {}, {}, []
                    for rank, data in trace["per_rank"].items():
                        samples = sorted(row[metric] for row in data["records"])
                        if len(samples) != comparison["samples_per_rank_pass"]:
                            raise ValueError(f"Sample coverage differs: {case}/{label}/{rank}")
                        rank_means[rank] = mean(samples)
                        rank_stats[rank] = dict(mean=rank_means[rank], median=median(samples),
                            p95=samples[(95 * len(samples) + 99) // 100 - 1])
                        values.extend(samples)
                    if len(rank_stats) != comparison["rank_count"]:
                        raise ValueError(f"Rank records differ: {case}/{label}")
                    values.sort()
                    arms[prefix][metric] = dict(mean=mean(values), median=median(values),
                        p95=values[(95 * len(values) + 99) // 100 - 1],
                        slowest_rank_mean=max(rank_means.values()), rank_means=rank_means,
                        rank_stats=rank_stats)
            deltas = {}
            for metric in ("kernel_us", "megamoe_end_us"):
                control, candidate = (arms[p][metric] for p in (args.control, args.candidate))
                if control["rank_stats"].keys() != candidate["rank_stats"].keys():
                    raise ValueError(f"Rank identities differ: {case}/{metric}/{suffix}")
                deltas[metric] = {k: candidate[k] - control[k]
                                 for k in ("mean", "median", "p95", "slowest_rank_mean")}
                deltas[metric]["rank_mean_deltas"] = {
                    r: candidate["rank_means"][r] - value
                    for r, value in control["rank_means"].items()}
                for stat in ("median", "p95"):
                    deltas[metric][f"rank_{stat}_deltas"] = {
                        r: candidate["rank_stats"][r][stat] - values[stat]
                        for r, values in control["rank_stats"].items()}
            case_report["reverse" if suffix else "forward"] = dict(arms=arms, deltas=deltas)
    report["aggregate_scope"] = ("Equal weight per listed case; ratio of baseline to candidate "
                                 "paired mean latency. Greater than one is faster. This covers "
                                 "only this campaign; small-batch preservation is a separate gate.")
    report["geometric_speedups"] = geometric_speedups(summary["cases"], args.candidate, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reviewed {len(report['cases'])} cases, both orders, all samples")


def aggregate_summaries(summaries, scope, candidates, baselines):
    """Score the declared matrix without dropping cases or mixing extra controls."""
    for names in (candidates, baselines):
        if not names or not all(names) or len(names) != len(set(names)):
            raise ValueError("Require distinct nonempty arm names")
    groups = {name: scope[key] for name, key in (
        ("comparison", "cases"), ("context", "context_cases"),
        ("small_controls", "retained_focus_controls"))}
    specs = {}
    for name, rows in groups.items():
        keys = [f"{row['model']}/{row['case_id']}" for row in rows]
        if not rows or len(keys) != len(set(keys)):
            raise ValueError(f"Require a nonempty, distinct scope: {name}")
        for key, row in zip(keys, rows):
            if key in specs and specs[key] != row:
                raise ValueError(f"Conflicting case contracts: {key}")
            specs[key] = row
    if len(groups["comparison"]) != scope["case_count"]:
        raise ValueError("Declared comparison count differs from scope")
    expected_eps = {row["ep"] for row in specs.values()}
    if set(summaries) != expected_eps:
        raise ValueError("EP coverage differs from scope")
    cases, sample_counts = {}, set()
    for ep, summary in summaries.items():
        if (summary["status"] != "PASS" or summary["ep"] != ep
                or not summary["matched_collective_preparation"]):
            raise ValueError(f"Requires a passing matched capture audit: EP{ep}")
        for key, case in summary["cases"].items():
            if key not in specs or key in cases or specs[key]["ep"] != ep:
                raise ValueError(f"Unexpected or duplicate case: {key}")
            if case["rank_count"] != ep:
                raise ValueError(f"Rank coverage differs: {key}")
            cases[key] = case
            sample_counts.add(case["samples_per_rank_pass"])
    if set(cases) != set(specs):
        raise ValueError(f"Missing scope cases: {sorted(set(specs) - set(cases))}")
    if len(sample_counts) != 1 or next(iter(sample_counts)) <= 0:
        raise ValueError("Require equal positive sample counts")

    def score(rows):
        selected = {f"{r['model']}/{r['case_id']}": cases[f"{r['model']}/{r['case_id']}"] for r in rows}
        comparisons = {}
        for candidate in candidates:
            for baseline in baselines:
                result = geometric_speedups(selected, candidate, baseline)
                for metric in result.values():
                    ratios = metric["case_speedups"]
                    metric["regressions"] = {key: value for key, value in ratios.items() if value < 1}
                comparisons[f"{candidate}_vs_{baseline}"] = result
        return dict(case_count=len(selected), comparisons=comparisons)

    fixed = groups["comparison"]
    return dict(status="PASS_COVERAGE_PERFORMANCE_REVIEW_REQUIRED",
        scope="Equal case weights, all ranks and both orders. Extra controls do not change the fixed comparison score. Small preservation and historical comparisons remain separate requirements; no per-case policy selection.",
        unique_cases=len(cases), samples_per_rank_pass=sample_counts.pop(),
        scores={name: score(rows) for name, rows in groups.items()},
        by_model={model: score([r for r in fixed if r["model"] == model])
                  for model in sorted({r["model"] for r in fixed})},
        by_ep={str(ep): score([r for r in fixed if r["ep"] == ep])
               for ep in sorted({r["ep"] for r in fixed})})


def load_campaign_summaries(root, scope):
    eps = {row["ep"] for key in ("cases", "context_cases", "retained_focus_controls") for row in scope[key]}
    summaries = {ep: json.loads((root / f"ep{ep}/summary.json").read_text()) for ep in eps}
    for ep, summary in summaries.items():
        job = json.loads((root / f"ep{ep}/gpu/job.json").read_text())["job"]
        if str(summary["job"]) != str(job):
            raise ValueError(f"Summary belongs to a different GPU job: EP{ep}")
    return summaries


def aggregate(argv):
    parser = argparse.ArgumentParser(description="Aggregate audited EP sweeps over a fixed comparison scope")
    parser.add_argument("root", type=Path)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--candidates", default="control,workers,rows")
    parser.add_argument("--baselines", default="pull,reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="INCOMPLETE")) + "\n")
    scope = json.loads(args.scope.read_text())
    summaries = load_campaign_summaries(args.root, scope)
    result = aggregate_summaries(summaries, scope, args.candidates.split(","), args.baselines.split(","))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {result['unique_cases']} cases; performance review still required")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--job", help="Defaults to ROOT/gpu/job.json")
    p.add_argument("--correctness-only", action="store_true",
                   help="Audit complete capture and exact-result coverage without extracting timelines")
    args = p.parse_args()
    root = args.root
    correctness_output = root / "capture_correctness_review.json"
    correctness_output.write_text(json.dumps(dict(status="INCOMPLETE")) + "\n")
    if args.job is None:
        args.job = json.loads((root / "gpu/job.json").read_text())["job"]
    plan = json.loads((root / "plan.json").read_text())
    workflow = Path(__file__).parent
    log = (root / f"capture-{args.job}.out").read_text()
    starts = list(re.finditer(r"^SWEEP_CASE_START .*?$", log, re.M))
    executions = [(g, c) for g in plan["groups"]
                  for c in g["env"]["MEGAMOE_SWEEP_CASES"].split(",")]
    if len(starts) != len(executions):
        raise ValueError("Capture coverage differs from plan")
    stages = [g["stage"] for g in plan["groups"]]
    if stages != sorted(stages, key=lambda s: {"small_control": 0, "large_target": 1, "profile": 2}[s]):
        raise ValueError("Small controls must precede large targets")
    expected_checks = []
    cases = {}
    for i, (group, case) in enumerate(executions):
        env = group["env"]
        phase = env.get("FASTAFD_IKET_CAPTURE_PHASE", "normal")
        if phase not in ("normal", "profile") or (phase == "profile") != (group["stage"] == "profile"):
            raise ValueError("Require ordinary captures before separate profile groups")
        text = log[starts[i].start():starts[i + 1].start() if i + 1 < len(starts) else len(log)]
        marker = f"case={case} ep={plan['ep']} "
        if marker not in starts[i].group() or len(re.findall(r"^SWEEP_CASE_COMPLETE " + re.escape(marker), text, re.M)) != 1:
            raise ValueError(f"Incomplete capture: {group['model']}/{case}")
        manifest = json.loads((Path(env["MEGAMOE_SWEEP_ROOT"]) / "manifest.json").read_text())
        spec = next(c for c in manifest["cases"] if c["case_id"] == case)
        contract = dict(manifest["contract"], **spec.get("capture_contract", {}))
        label = f"fixed_bucket{contract['bucket']}_rows{spec['rows_per_rank']}_sms{contract['num_sms']}"
        for _ in range(2 if phase == "profile" else 1):
            if env.get("MEGAMOE_RESULT_MODE"):
                expected_checks.append((env["MEGAMOE_RESULT_MODE"], label))
            expected_checks.append(("replay", f"{spec['rows_per_rank']}/{env.get('MEGAMOE_SMOKE_TIMING_SCOPE', 'pipeline')}"))
        if phase == "profile":
            continue
        entry = cases.setdefault((group["model"], case), {})
        if group["arm"] in entry:
            raise ValueError("Duplicate arm/case")
        entry[group["arm"]] = env
    counts = audit_correctness(log, expected_checks, plan["ep"])
    correctness = dict(status="PASS_FULL_CAPTURE_EXACT_OUTPUTS", gpu_job=args.job,
        ep=plan["ep"], captures=len(executions), cases=len(cases), correctness=counts,
        scope="All planned captures and ranks; configured source comparisons and exact replay. "
              "Timing/graph audits and performance acceptance remain separate requirements.")
    correctness_output.write_text(json.dumps(correctness, indent=2) + "\n")
    if args.correctness_only:
        print(json.dumps(correctness))
        return

    def run(script, *args):
        subprocess.run([sys.executable, "-B", str(workflow / script), *map(str, args)], check=True)

    normal_groups = [g for g in plan["groups"] if g["stage"] != "profile"]
    profile_groups = [g for g in plan["groups"] if g["stage"] == "profile"]
    normal_plan = root / "analysis/ordinary_plan.json"
    normal_plan.write_text(json.dumps(dict(plan, groups=normal_groups), indent=2) + "\n")
    run("run_megamoe_sweep_group.py", "analyze", normal_plan, "--ordinary-only", "--all-ranks")
    matched_preparation = all(
        g["env"].get("MEGAMOE_SMOKE_RESET_DISPATCH_PAYLOAD") == "1"
        or g["env"].get("MEGAMOE_SMOKE_PREPARE_BARRIER_ONLY") == "1"
        for g in plan["groups"])
    summary = dict(job=args.job, ep=plan["ep"],
                   status="PASS" if matched_preparation else "OBSERVATION_ONLY_MIXED_PREPARATION",
                   matched_collective_preparation=matched_preparation, correctness=counts,
                   scope="Unfiltered mean rank-local producer-start to main-end and main-kernel duration; all ranks and both orders. Preparation excluded. Pull has exact replay but no cross-source byte comparison. Mixed preparation is not a final pull speedup gate.", cases={})
    prefixes = plan["arm_prefixes"]
    for (model, case), arms in cases.items():
        case_prefixes = [p for p in prefixes if p in arms or p + "_repeat" in arms]
        expected_arms = {a for p in case_prefixes for a in (p, p + "_repeat")}
        if prefixes[0] not in case_prefixes or len(case_prefixes) < 2 or set(arms) != expected_arms:
            raise ValueError(f"Case requires a control and complete mirrored arms: {model}/{case}")
        output = root / "analysis" / model / case
        output.mkdir(parents=True, exist_ok=True)
        timelines = {}
        samples = {env["MEGAMOE_BASELINE_ITERATIONS"] for env in arms.values()}
        if len(samples) != 1:
            raise ValueError("Mismatched sample counts")
        samples = int(samples.pop())
        for arm, env in arms.items():
            path = output / f"{arm}.json"
            reset = env.get("MEGAMOE_SMOKE_RESET_DISPATCH_PAYLOAD") == "1"
            barrier_only = env.get("MEGAMOE_SMOKE_PREPARE_BARRIER_ONLY") == "1"
            run("summarize_megamoe_router_timeline.py", root / model, f"{case}/{args.job}",
                "--arms", arm, "--samples", samples, "--rank-mean", plan["ep"],
                "--require-graph-identity", "--retain-timestamps",
                *(["--require-preinitialized-events"] if reset or barrier_only else []),
                *(["--joined-collective-preparation"] if reset else []),
                *(["--joined-collective-barrier-only"] if barrier_only else []),
                "--output", path)
            timelines.update(json.loads(path.read_text()))
        result = compare(timelines, case_prefixes[0], case_prefixes[1:], [], plan["ep"], samples)
        (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["cases"][f"{model}/{case}"] = result
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({case: result["paired_means"] for case, result in summary["cases"].items()}), flush=True)
    if profile_groups:
        profile_plan = root / "analysis/profile_plan.json"
        profile_plan.write_text(json.dumps(dict(plan, groups=profile_groups), indent=2) + "\n")
        run("run_megamoe_sweep_group.py", "analyze", profile_plan, "--profiles-only", "--all-ranks")
        for group in profile_groups:
            for case in group["env"]["MEGAMOE_SWEEP_CASES"].split(","):
                output = root / "analysis" / group["model"] / case
                capture = Path(group["env"]["MEGAMOE_SWEEP_ROOT"]) / case / args.job
                run("summarize_megamoe_latency.py",
                    *[f"rank{rank}={capture}/rank_decomposition/rank-{rank}/decomposition.json"
                      for rank in range(plan["ep"])],
                    "--kernel-origin", "--output", output / f"{group['arm']}_phases.json",
                    "--rank-mean-output", output / f"{group['arm']}_phase_means.json")


if __name__ == "__main__":
    if sys.argv[1:2] == ["review-normal"]:
        review_normal(sys.argv[2:])
    elif sys.argv[1:2] == ["review-interleaved"]:
        review_interleaved(sys.argv[2:])
    elif sys.argv[1:2] == ["review-windows"]:
        review_windows(sys.argv[2:])
    elif sys.argv[1:2] == ["review-ranks"]:
        review_ranks(sys.argv[2:])
    elif sys.argv[1:2] == ["aggregate"]:
        aggregate(sys.argv[2:])
    else:
        main()
