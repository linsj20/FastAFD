#!/usr/bin/env python3
"""Plot audited per-case latency changes and paired-repeat ranges."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Comparison directory containing audit.json')
    parser.add_argument('--all-configs', action='store_true', help='Plot absolute latency for every audited configuration')
    args = parser.parse_args()
    audit = json.loads((args.root / 'audit.json').read_text())
    if audit['status'] != 'PASS':
        raise ValueError('Plot requires complete successful comparison coverage')
    models = list(audit['summary'])
    if args.all_configs:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 7.5), squeeze=False)
        names = {'control_dense': 'Historical dense pull',
                 'reference_previous_metadata': 'Historical previous-metadata pull',
                 'candidate_selected': 'Selected pull',
                 'candidate_compact': 'Cleaned compact push',
                 'alternate_precleanup': 'Pull before cleanup',
                 'alternate_compact': 'Frozen compact push'}
        colors = ['#6b7280', '#8b5eaa', '#246fb4', '#13866a', '#cd7133']
        for ax, model in zip(axes[0], models):
            cases = sorted((c for c in audit['cases'] if c['model'] == model), key=lambda c: c['rows'])
            configs = list(cases[0]['means_us'])
            for i, config in enumerate(configs):
                offset = (i - (len(configs) - 1) / 2) * .16
                values = [c['means_us'][config] for c in cases]
                low = [v - min(c['repeats_us'][config].values()) for c, v in zip(cases, values)]
                high = [max(c['repeats_us'][config].values()) - v for c, v in zip(cases, values)]
                ax.errorbar(values, [y + offset for y in range(len(cases))], xerr=[low, high],
                            fmt='o', markersize=4, capsize=2, color=colors[i % len(colors)],
                            label=names.get(config, config))
            ax.set_yticks(range(len(cases)), [str(c['rows']) for c in cases])
            ax.invert_yaxis()
            ax.set_title(f"{'Qwen3.5' if model == 'qwen35' else model.replace('qwen', 'Qwen')} · EP{cases[0]['ep']}")
            ax.set_xlabel('Producer + MegaMoE latency (µs)')
            ax.set_ylabel('Rows per rank')
            ax.grid(axis='x', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .06))
        fig.suptitle('Historical pull / push-combine baselines and current variants', fontsize=15)
        fig.text(.5, .015, 'Points: two-repeat means. Whiskers: repeat ranges, not confidence intervals.\n'
                 'Slowest-rank CUDA events; 2,000 calls, 200 warmups, graph batch 100; fixed perfect routing; profiling off.',
                 ha='center', fontsize=8.5)
        fig.tight_layout(rect=(0, .15, 1, .95))
        for suffix in ('png', 'svg'):
            fig.savefig(args.root / ('all_config_latencies.' + suffix), dpi=180)
        print(args.root / 'all_config_latencies.png')
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6.5), squeeze=False, sharex=True)
    for ax, model in zip(axes[0], models):
        cases = sorted((c for c in audit['cases'] if c['model'] == model), key=lambda c: c['rows'])
        for y, case in enumerate(cases):
            pull = case['repeats_us']['candidate_selected']
            push = case['repeats_us']['alternate_compact']
            paired = [100 * (1 - push[r] / pull[r]) for r in ('a', 'b')]
            center = case['compact_latency_reduction_percent']
            color = '#13866a' if center >= 0 else '#bc692e'
            ax.barh(y, center, height=.58, color=color, alpha=.85)
            ax.errorbar(center, y, xerr=[[max(0, center - min(paired))], [max(0, max(paired) - center)]],
                        fmt='o', color='#202631', capsize=3, markersize=3, linewidth=1.2)
        ax.set_yticks(range(len(cases)), [str(c['rows']) for c in cases])
        ax.invert_yaxis()
        ax.axvline(0, color='#444444', linewidth=1)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylabel('Rows per rank')
        title = {'qwen3': 'Qwen3', 'qwen35': 'Qwen3.5'}.get(model, model)
        ax.set_title(f"{title} · EP{cases[0]['ep']}")
        ax.set_xlabel('Compact latency reduction (%)\n← selected pull faster      compact push faster →')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Selected pull versus compact payload push', fontsize=16, y=.99)
    fig.text(.5, .92, 'Producer + MegaMoE latency · slowest-rank CUDA events · profiling off', ha='center', fontsize=10)
    fig.text(.5, .025, 'Bars: reduction from the two-repeat means. Whiskers: range of the two paired-repeat reductions, not confidence intervals.\n'
             'Each repeat: 2,000 calls, 200 warmups, graph batch 100. Fixed perfect routing; sampled shapes shown by rows per rank.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .08, 1, .9))
    for suffix in ('png', 'svg'):
        fig.savefig(args.root / ('latency_comparison.' + suffix), dpi=180)
    print(args.root / 'latency_comparison.png')


if __name__ == '__main__':
    main()
