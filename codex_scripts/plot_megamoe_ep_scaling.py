#!/usr/bin/env python3
"""Plot fixed-total-token MegaMoE time versus EP with anchored ideal scaling.

Default mode uses audited ordinary Nsight kernel medians. The throughput
subcommand reports paired all-rank means with exact declared-scope coverage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def load_ordinary_sweep(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "ordinary_audit.json").read_text())
    assert audit["complete"] and not audit["missing"] and not audit["instrumented"]
    assert audit["rank"] == 0 and audit["metric"] == "nsys_kernel_median_us"
    rows = list(csv.DictReader((root / "ordinary.csv").open()))
    cases = {c["case_id"]: c for c in manifest["cases"]}
    assert len(rows) == len(cases) == audit["artifact_contracts_checked"]
    assert {r["case_id"] for r in rows} == set(cases)
    groups = defaultdict(list)
    for row in rows:
        case = cases[row["case_id"]]
        ep, tokens, time_us = int(row["ep"]), int(row["global_tokens"]), float(row["nsys_kernel_median_us"])
        assert (ep, tokens) == (case["ep"], case["global_tokens"])
        assert int(row["ordinary_samples"]) == audit["ordinary_samples"]
        assert math.isfinite(time_us) and time_us > 0
        groups[tokens].append(dict(ep=ep, tokens=tokens, measured_us=time_us,
                                  case_id=row["case_id"], artifact=row["artifact"]))
    for points in groups.values():
        points.sort(key=lambda p: p["ep"])
        assert len({p["ep"] for p in points}) == len(points)
    return dict(model=manifest["contract"]["model"], contract=manifest["contract"],
                source=str(root.resolve()), source_report=str((root / "ordinary.csv").resolve()),
                audited_cases=len(rows), checked_report_rows=len(rows),
                ordinary_samples=audit["ordinary_samples"], groups=groups)


def load_sweep(root: Path, report_subdir: str = "."):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit_summary.json").read_text())
    assert audit["complete"] and not audit["missing"], f"Incomplete sweep: {root}"
    rows = list(csv.DictReader((root / "decomposition.csv").open()))
    cases = {c["case_id"]: c for c in manifest["cases"]}
    by_case = {r["case_id"]: r for r in rows}
    assert len(rows) == len(by_case) == audit["artifact_contracts_checked"]
    assert set(by_case) == set(cases)
    report_root = root / report_subdir
    reports = list(report_root.glob("*-megamoe-comparisons.html"))
    assert len(reports) == 1, f"Expected one source report: {root}"
    match = re.search(r'<script[^>]*id="megamoe-data"[^>]*>(.*?)</script>', reports[0].read_text(), re.S)
    assert match, f"Missing report dataset: {reports[0]}"
    displayed = 0
    exclusions_path = report_root / "outlier_audit.json"
    exclusions = json.loads(exclusions_path.read_text())["cases"] if exclusions_path.exists() else {}
    for section in json.loads(match[1]):
        for row in section["rows"]:
            source = by_case[row["case_id"]]
            expected = float(source["nsys_kernel_median_us"])
            if row["case_id"] in exclusions:
                ordinary = exclusions[row["case_id"]]["ordinary"]
                assert math.isclose(expected, ordinary["raw_median_us"], abs_tol=1e-9)
                expected = ordinary["filtered_median_us"]
            assert (int(row["ep"]), int(row["global_tokens"]), float(row["nsys_kernel_median_us"])) == (
                int(source["ep"]), int(source["global_tokens"]), expected)
            displayed += 1
    groups = defaultdict(list)
    for row in rows:
        case = cases[row["case_id"]]
        ep, tokens, time_us = int(row["ep"]), int(row["global_tokens"]), float(row["nsys_kernel_median_us"])
        assert (ep, tokens) == (case["ep"], case["global_tokens"])
        assert ep > 0 and tokens > 0 and math.isfinite(time_us) and time_us > 0
        groups[tokens].append(dict(ep=ep, tokens=tokens, measured_us=time_us,
                                  case_id=row["case_id"], artifact=row["artifact"]))
    for points in groups.values():
        points.sort(key=lambda p: p["ep"])
        assert len({p["ep"] for p in points}) == len(points)
    return dict(model=manifest["contract"]["model"], contract=manifest["contract"],
                source=str(root.resolve()), source_report=str(reports[0].resolve()),
                audited_cases=len(rows), checked_report_rows=displayed,
                ordinary_samples=manifest["final_validation"]["baseline_iterations"], groups=groups)


def compare_sweeps(candidate, baseline):
    assert candidate["contract"] == baseline["contract"], "Different input contracts"
    def points(model):
        return {p["case_id"]: p for group in model["groups"].values() for p in group}
    current, previous = points(candidate), points(baseline)
    assert current.keys() == previous.keys(), "Different case coverage"
    rows = []
    for case_id, new in current.items():
        old = previous[case_id]
        assert (new["ep"], new["tokens"]) == (old["ep"], old["tokens"])
        rows.append(dict(case_id=case_id, ep=new["ep"], total_tokens=new["tokens"],
                         baseline_us=old["measured_us"], candidate_us=new["measured_us"],
                         saving_us=old["measured_us"] - new["measured_us"],
                         reduction_pct=100 * (1 - new["measured_us"] / old["measured_us"])))
    def summarize(group):
        old = sum(r["baseline_us"] for r in group) / len(group)
        new = sum(r["candidate_us"] for r in group) / len(group)
        return dict(cases=len(group), mean_case_reduction_pct=sum(r["reduction_pct"] for r in group) / len(group),
                    baseline_mean_us=old, candidate_mean_us=new, mean_saving_us=old-new,
                    reduction_of_mean_pct=100*(1-new/old), faster_cases=sum(r["saving_us"] > 0 for r in group))
    return dict(model=candidate["model"], baseline=baseline["source"], candidate=candidate["source"],
                metric="Unfiltered rank-0 ordinary kernel median per case",
                baseline_samples=baseline["ordinary_samples"], candidate_samples=candidate["ordinary_samples"],
                interpretation="Historical cross-allocation comparison; preparation differs; not a causal paired A/B",
                weighting="Each unique case has equal weight; aliases are not counted again",
                summary=summarize(rows), by_ep={str(ep): summarize([r for r in rows if r["ep"] == ep])
                                               for ep in sorted({r["ep"] for r in rows})}, cases=rows)


def throughput_history(root, scope, candidates, history_roots):
    """Compare rank-0 medians across allocations, separately from paired means."""
    specs = {f"{r['model']}/{r['case_id']}": r
             for group in ("cases", "context_cases") for r in scope[group]}
    expected_models = {r["model"] for r in specs.values()}
    if set(history_roots) != expected_models:
        raise ValueError("Require one historical root for every scoped model")
    previous, sources = {}, {}
    for model, path in history_roots.items():
        sweep = load_ordinary_sweep(path)
        sources[model] = {k: sweep[k] for k in ("source", "source_report", "contract", "ordinary_samples")}
        previous.update({f"{model}/{p['case_id']}": p for points in sweep["groups"].values() for p in points})
    if not set(specs) <= set(previous):
        raise ValueError("Historical sweep is missing declared comparison/context cases")
    reviews = {(ep, arm): json.loads((root / f"ep{ep}/analysis/{arm}_vs_reference.json").read_text())
               for ep in {r["ep"] for r in specs.values()} for arm in candidates}
    records = []
    for key, spec in specs.items():
        old = previous[key]
        if (old["ep"], old["tokens"]) != (spec["ep"], spec["total"]):
            raise ValueError(f"Historical case contract differs: {key}")
        for arm in candidates:
            review = reviews[spec["ep"], arm]["cases"][key]
            medians = {order: review[order]["arms"][arm]["kernel_us"]["rank_stats"]["0"]["median"]
                       for order in ("forward", "reverse")}
            if any(not math.isfinite(v) or v <= 0 for v in medians.values()):
                raise ValueError(f"Invalid rank-0 median: {key}/{arm}")
            current = sum(medians.values()) / len(medians)
            records.append(dict(case=key, model=spec["model"], ep=spec["ep"], total_tokens=spec["total"],
                arm=arm, historical_kernel_median_us=old["measured_us"],
                historical_artifact=old["artifact"],
                current_forward_median_us=medians["forward"], current_reverse_median_us=medians["reverse"],
                current_mean_of_medians_us=current, speedup=old["measured_us"] / current))
    scores = {}
    for group, field in (("comparison", "cases"), ("context", "context_cases")):
        keys = {f"{r['model']}/{r['case_id']}" for r in scope[field]}
        scores[group] = {}
        for arm in candidates:
            ratios = {r["case"]: r["speedup"] for r in records if r["arm"] == arm and r["case"] in keys}
            scores[group][arm] = dict(case_count=len(ratios),
                geometric_mean_speedup=math.exp(sum(map(math.log, ratios.values())) / len(ratios)),
                regressions={k: v for k, v in ratios.items() if v < 1})
    return dict(scope="Historical single rank-0 kernel median versus average of current forward/reverse "
        "rank-0 medians. Separate allocations and potentially different tuning; descriptive only, "
        "not paired cleanup attribution. No historical pipeline comparison. All declared comparison "
        "and context cases retained; extra small controls excluded.", sources=sources, scores=scores, records=records)


def throughput_report(argv):
    """Plot both paired metrics without treating them as historical medians."""
    from analyze_megamoe_throughput import aggregate_summaries, load_campaign_summaries
    parser = argparse.ArgumentParser(description=throughput_report.__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-script", type=Path, required=True)
    parser.add_argument("--candidates", default="control,workers,rows")
    parser.add_argument("--baseline", default="pull")
    parser.add_argument("--history", action="append", default=[], metavar="MODEL=ROOT",
                        help="Audited historical ordinary sweep; repeat for each scoped model")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_path = args.output.with_suffix(".comparison.json")
    result_path.write_text(json.dumps(dict(status="INCOMPLETE")) + "\n")
    # A failed refresh must not leave a stale passing report at the requested path.
    args.output.with_suffix(".html").write_text("<p>Incomplete sweep; report unavailable.</p>\n")
    scope = json.loads(args.scope.read_text())
    summaries = load_campaign_summaries(args.root, scope)
    candidates = args.candidates.split(",")
    baselines = list(dict.fromkeys([args.baseline, "reference"]))
    result = aggregate_summaries(summaries, scope, candidates, baselines)
    specs = {f"{r['model']}/{r['case_id']}": r for group in
             ("cases", "context_cases", "retained_focus_controls") for r in scope[group]}
    cases = {key: case for summary in summaries.values() for key, case in summary["cases"].items()}
    history_roots = {}
    for value in args.history:
        model, separator, path = value.partition("=")
        if not separator or not path or model in history_roots:
            raise ValueError("Require distinct MODEL=ROOT historical sweeps")
        history_roots[model] = Path(path)
    history = throughput_history(args.root, scope, candidates, history_roots) if history_roots else None
    history_ratios = {(r["case"], r["arm"]): r["speedup"] for r in history["records"]} if history else {}
    metrics = {"kernel_us": "Kernel", "megamoe_end_us": "Producer start to kernel end"}
    exports = [dict(model=r["model"], ep=r["ep"], total_tokens=r["total"], case=key,
                    arm=arm, metric=metric, mean_us=case["paired_means"][arm][metric])
               for key, case in cases.items() for r in [specs[key]]
               for arm in dict.fromkeys([*candidates, *baselines]) for metric in metrics]
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    figures = []
    for model in sorted({r["model"] for r in specs.values()}):
        eps = sorted({r["ep"] for r in specs.values() if r["model"] == model})
        plotted_metrics = dict(metrics)
        if history:
            plotted_metrics["historical_rank0"] = "Historical rank-0 kernel medians"
        fig, axes = plt.subplots(len(plotted_metrics), len(eps), squeeze=False, sharey=True,
                                 figsize=(3.1 * len(eps), 3.1 * len(plotted_metrics)), layout="constrained")
        for row, (metric, label) in enumerate(plotted_metrics.items()):
            for col, ep in enumerate(eps):
                ax = axes[row][col]
                keys = sorted((k for k, r in specs.items() if r["model"] == model and r["ep"] == ep),
                              key=lambda k: specs[k]["total"])
                if metric == "historical_rank0":
                    keys = [k for k in keys if (k, candidates[0]) in history_ratios]
                for i, candidate in enumerate(candidates):
                    ratios = ([history_ratios[k, candidate] for k in keys] if metric == "historical_rank0" else
                              [cases[k]["paired_means"][args.baseline][metric] /
                               cases[k]["paired_means"][candidate][metric] for k in keys])
                    ax.plot([specs[k]["total"] for k in keys], ratios, marker=("o", "s", "^")[i % 3],
                            markersize=3, label=candidate, linewidth=1.2)
                ax.axhline(1, color="0.4", linewidth=.8)
                ax.set_xscale("log", base=2)
                ax.set_xticks([32, 256, 2048, 8192])
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.set_xlim(min(specs[k]["total"] for k in keys) / 1.2, 10000)
                ax.set_title(f"{label} · EP{ep}", fontsize=10)
                ax.set_xlabel("Total tokens")
                ax.grid(alpha=.2)
                if col == 0:
                    baseline_label = "history" if metric == "historical_rank0" else args.baseline
                    ax.set_ylabel(f"Speedup over {baseline_label} (×)")
                if row == col == 0:
                    ax.legend(fontsize=9)
        title = f"{model} · paired rows: all ranks, both orders, unfiltered means"
        if history:
            title += "\nHistory: separate allocations/tuning · rank-0 median / current two-order median average"
        fig.suptitle(title)
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg")
        if args.preview:
            fig.savefig(args.output.with_suffix(f".{model}.preview.png"), dpi=120)
        plt.close(fig)
        svg = buffer.getvalue()
        svg = svg[svg.index("<svg "):].replace("<svg ",
            f'<svg role="img" aria-label="{html.escape(model)} kernel and producer-to-kernel speedups" ', 1)
        figures.append(svg)
    table = ["<table><caption>Equal-case geometric speedups; greater than one is faster. "
             "Performance and small-case preservation review remain required.</caption>"
             "<thead><tr><th>Scope</th><th>Cases</th><th>Comparison</th><th>Kernel</th>"
             "<th>Producer to kernel end</th><th>Slower cases: kernel / span</th></tr></thead><tbody>"]
    for name, group in result["scores"].items():
        for pair, scores in group["comparisons"].items():
            values = "".join(f"<td>{scores[m]['geometric_mean_speedup']:.6f}×</td>" for m in metrics)
            losses = " / ".join(str(len(scores[m]["regressions"])) for m in metrics)
            table.append(f"<tr><th>{html.escape(name.replace('_', ' '))}</th><td>{group['case_count']}</td>"
                         f"<td>{html.escape(pair.replace('_', ' '))}</td>{values}<td>{losses}</td></tr>")
    table.append("</tbody></table>")
    if history:
        result["historical_rank0_kernel_comparison"] = history
        table.append("<table><caption>Historical rank-0 kernel medians; separate allocations and tuning. "
                     "Descriptive comparison, not paired cleanup attribution.</caption><thead><tr>"
                     "<th>Scope</th><th>Cases</th><th>Arm</th><th>Geometric speedup</th>"
                     "<th>Slower cases</th></tr></thead><tbody>")
        for group, arms in history["scores"].items():
            for arm, score in arms.items():
                table.append(f"<tr><th>{html.escape(group)}</th><td>{score['case_count']}</td>"
                             f"<td>{html.escape(arm)}</td><td>{score['geometric_mean_speedup']:.6f}×</td>"
                             f"<td>{len(score['regressions'])}</td></tr>")
        table.append("</tbody></table>")
    result["source_summaries"] = {str(ep): str((args.root / f"ep{ep}/summary.json").resolve()) for ep in summaries}
    result["scope_source"] = str(args.scope.resolve())
    result["declared_scope_status"] = scope.get("status")
    template = Path(__file__).with_name("megamoe_ep_scaling.html").read_text()
    fragment = template.replace("<!--SCALING_FIGURE-->", "\n".join([*figures, *table]))
    fragment = fragment.replace("<!--SCALING_POINTS-->", json.dumps(exports).replace("<", "\\u003c"))
    fragment = fragment.replace("<!--SCALING_PROVENANCE-->", json.dumps(result).replace("<", "\\u003c"))
    source, destination = args.output.with_suffix(".source.html"), args.output.with_suffix(".html")
    source.write_text(fragment)
    subprocess.run([sys.executable, str(args.render_script), str(source.resolve()), str(destination.resolve()),
                    "--title", "MegaMoE paired throughput sweep"], check=True)
    from build_megamoe_comparison import StandaloneFrame
    rendered = StandaloneFrame()
    rendered.feed(destination.read_text())
    assert len(rendered.documents) == 1 and fragment in rendered.documents[0]
    destination.write_text(rendered.documents[0])
    with args.output.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    if history:
        with args.output.with_suffix(".history.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(history["records"][0]))
            writer.writeheader()
            writer.writerows(history["records"])
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Reported {len(cases)} cases; performance and small-case preservation review still required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path, help="Standalone HTML filename or prefix")
    parser.add_argument("--tokens", default="powers-of-two", help="powers-of-two, all, or comma-separated totals")
    parser.add_argument("--render-script", required=True, type=Path, help="Visualize skill scripts/render.py")
    parser.add_argument("--report-subdir", default=".", help="Source report directory relative to each sweep root")
    parser.add_argument("--ordinary-only", action="store_true",
                        help="Read audited ordinary.csv captures without decomposition profiles")
    parser.add_argument("--label", default="September 12 communication-final sweep")
    parser.add_argument("--baseline-roots", type=Path, nargs="+", help="Matching historical sweeps, in model order")
    parser.add_argument("--preview", action="store_true", help="Save a PNG of the embedded chart for visual review")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, ScalarFormatter
    import numpy as np

    models = [load_ordinary_sweep(root) if args.ordinary_only else load_sweep(root, args.report_subdir)
              for root in args.roots]
    comparisons = []
    if args.baseline_roots:
        assert len(args.baseline_roots) == len(models), "One baseline root is required per model"
        comparisons = [compare_sweeps(model, load_sweep(root))
                       for model, root in zip(models, args.baseline_roots)]
    explicit = None if args.tokens in ("all", "powers-of-two") else {int(x) for x in args.tokens.split(",")}
    curves, exports = [], []
    for model in models:
        chosen = {tokens: points for tokens, points in sorted(model["groups"].items())
                  if len(points) >= 2 and (args.tokens == "all" or
                  (args.tokens == "powers-of-two" and tokens & (tokens-1) == 0) or
                  (explicit is not None and tokens in explicit))}
        assert chosen, f"No matching multi-EP inputs: {model['model']}"
        curves.append(chosen)
        for tokens, points in chosen.items():
            first = points[0]
            for point in points:
                ideal = first["measured_us"] * first["ep"] / point["ep"]
                assert ideal > 0 and math.isfinite(ideal)
                exports.append(dict(model=model["model"], total_tokens=tokens,
                    ep=point["ep"], measured_us=point["measured_us"], ideal_us=ideal,
                    anchor_ep=first["ep"], anchor_us=first["measured_us"],
                    case_id=point["case_id"], artifact=point["artifact"]))
            assert exports[-len(points)]["ideal_us"] == first["measured_us"]
    token_values = sorted({t for group in curves for t in group})
    categorical = len(token_values) <= 10
    color_map = plt.get_cmap("tab10" if categorical else "viridis")
    norm = LogNorm(vmin=min(token_values), vmax=max(token_values))
    colors = {t: color_map(i if categorical else norm(t)) for i, t in enumerate(token_values)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(models), squeeze=False, sharey=True,
                             figsize=(7.1 * len(models), 6.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.245, top=.82, wspace=.20)
    fig.suptitle("MegaMoE time vs. expert parallelism — fixed total tokens", y=.985, fontsize=16)
    style_legend = [Line2D([], [], color="0.2", marker="o", lw=1.8, markersize=4, label="Measured: Nsight kernel median"),
                    Line2D([], [], color="0.2", ls="--", lw=1.5, label="Ideal: anchor time × anchor EP / EP")]
    fig.legend(handles=style_legend, loc="upper center", bbox_to_anchor=(.5, .936),
               ncol=2, frameon=False, columnspacing=2.8)
    ymin = min(min(row["measured_us"], row["ideal_us"]) for row in exports)
    ymax = max(max(row["measured_us"], row["ideal_us"]) for row in exports)
    for ax, model, chosen in zip(axes[0], models, curves):
        for tokens, points in chosen.items():
            first, last = points[0], points[-1]
            xs = [p["ep"] for p in points]
            ys = [p["measured_us"] for p in points]
            # Dense ideal samples make the 1/EP curve smooth on the log2 x axis.
            ideal_ep = np.geomspace(first["ep"], last["ep"], 120)
            ideal_us = first["measured_us"] * first["ep"] / ideal_ep
            ax.plot(ideal_ep, ideal_us, color=colors[tokens], ls=(0,(4,3)), lw=1.2,
                    alpha=.72, zorder=1)
            ax.plot(xs, ys, color=colors[tokens], marker="o", markersize=4 if categorical else 2.5,
                    lw=1.8 if categorical else 1.0, alpha=1 if categorical else .72,
                    label=f"{tokens:,}", zorder=2)
        eps = sorted({p["ep"] for points in chosen.values() for p in points})
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps, labels=[str(ep) for ep in eps])
        ax.set_xlim(min(eps)/1.22, max(eps)*1.13)
        ax.set_yscale("log", base=2)
        ax.set_ylim(ymin/1.15, ymax*1.15)
        ax.yaxis.set_major_locator(LogLocator(base=2, numticks=12))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(model["model"], loc="left", pad=12)
        ax.set_xlabel("Expert parallelism (EP)", labelpad=9)
        ax.grid(axis="both", color="0.88", lw=.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("0.55")
        ax.tick_params(axis="both", length=3, color="0.55")
        ax.tick_params(axis="y", labelleft=True)
    axes[0][0].set_ylabel("MegaMoE time (µs, log scale)", labelpad=8)
    if categorical:
        handles = [Line2D([], [], color=colors[t], lw=2, label=f"{t:,}") for t in token_values]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52,.145),
                   ncol=min(9,len(handles)), title="Total input tokens per invocation", frameon=False,
                   handlelength=2, columnspacing=1.6)
    else:
        from matplotlib.cm import ScalarMappable
        fig.subplots_adjust(bottom=.29)
        cax = fig.add_axes([.20,.135,.62,.023])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=cax, orientation="horizontal")
        ticks = [t for t in token_values if t & (t-1) == 0]
        colorbar.set_ticks(ticks, labels=[f"{t:,}" for t in ticks])
        colorbar.set_label("Total input tokens per invocation", labelpad=5)
        colorbar.ax.minorticks_off()
    # Label every curve at its leftmost measured point. Display-space spacing
    # separates close curves while short leaders preserve the exact anchor.
    fig.canvas.draw()
    for panel, (ax, chosen) in enumerate(zip(axes[0], curves)):
        anchors = []
        for tokens, points in chosen.items():
            first = points[0]
            x, y = first["ep"], first["measured_us"]
            pixel_y = ax.transData.transform((x, y))[1]
            anchors.append((pixel_y, tokens, x, y))
        anchors.sort()
        ys = np.array([row[0] for row in anchors])
        spacing = 12 * fig.dpi / 72
        steps = np.arange(len(anchors)) * spacing
        placed = np.maximum.accumulate(ys - steps) + steps
        placed -= np.mean(placed - ys)
        placed += max(0, ax.bbox.ymin + spacing / 2 - placed[0])
        placed -= max(0, placed[-1] - ax.bbox.ymax + spacing / 2)
        for label_y, (pixel_y, tokens, x, y) in zip(placed, anchors):
            caption = f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)
            label = ax.annotate(caption, xy=(x, y),
                                xytext=(-4, (label_y - pixel_y) * 72 / fig.dpi),
                                textcoords="offset points", ha="right", va="center",
                                fontsize=9, fontweight="bold", color="0.42", zorder=5,
                                bbox=dict(facecolor="white", edgecolor="none", pad=.3),
                                arrowprops=dict(arrowstyle="-", color="0.55",
                                                linewidth=.6, shrinkA=2, shrinkB=0))
            label.set_gid(f"curve-label-{panel}-{tokens}")
    samples = sorted({m["ordinary_samples"] for m in models})
    assert len(samples) == 1, "Different measurement sample counts"
    fig.text(.065,.021, f"{args.label} · rank 0 · {samples[0]} ordinary calls per point · unfiltered medians",
             fontsize=9, color="0.35")
    # Keep the vector figure and exact data inside HTML; no separate format exports.
    figure = io.StringIO()
    fig.savefig(figure, format="svg", facecolor="white")
    if args.preview:
        fig.savefig(args.output.with_suffix(".preview.png"), dpi=140, facecolor="white")
    plt.close(fig)
    summary = dict(metric="nsys_kernel_median_us", selection=args.tokens,
                   label=args.label, sample_policy="Unfiltered ordinary medians from " + (
                       "ordinary.csv" if args.ordinary_only else "decomposition.csv"),
                   baseline_comparisons=comparisons,
                   x_scale="log2", y_scale="log2", y_limits=[ymin/1.15, ymax*1.15],
                   ideal_formula="T_ideal(EP) = T_measured(EP_min) * EP_min / EP",
                   ideal_extent="smallest through largest measured EP for that token total",
                   missing_ep_policy="Only markers are observations; lines connect available measurements",
                   curve_labels="All total tokens left of the leftmost measured points, small grey bold labels with leaders",
                   excluded_single_ep_policy="No scaling curve for a total measured at only one EP",
                   models=[dict(model=m["model"], contract=m["contract"],
                                source=m["source"], source_report=m["source_report"],
                                audited_cases=m["audited_cases"], checked_report_rows=m["checked_report_rows"],
                                plotted_curves=len(g), plotted_points=sum(map(len,g.values())),
                                token_totals=list(g), ordinary_samples=m["ordinary_samples"])
                           for m,g in zip(models,curves)])
    svg = figure.getvalue()
    svg = svg[svg.index("<svg "):].strip()
    svg = svg.replace("<svg ", '<svg role="img" aria-label="MegaMoE time versus EP, measured and ideal fixed-token scaling" ', 1)
    template = Path(__file__).with_name("megamoe_ep_scaling.html").read_text()
    fragment = template.replace("<!--SCALING_FIGURE-->", svg)
    fragment = fragment.replace("<!--SCALING_POINTS-->", json.dumps(exports).replace("<", "\\u003c"))
    fragment = fragment.replace("<!--SCALING_PROVENANCE-->", json.dumps(summary).replace("<", "\\u003c"))
    source = args.output.with_suffix(".source.html")
    destination = args.output.with_suffix(".html")
    source.write_text(fragment)
    subprocess.run([sys.executable, str(args.render_script), str(source.resolve()),
                    str(destination.resolve()), "--title", "Fixed-token MegaMoE EP scaling"], check=True)
    # Unwrap the renderer's iframe to use the full browser width.
    from build_megamoe_comparison import StandaloneFrame
    rendered = StandaloneFrame()
    rendered.feed(destination.read_text())
    assert len(rendered.documents) == 1 and fragment in rendered.documents[0]
    destination.write_text(rendered.documents[0])
    if comparisons:
        comparison_path = args.output.with_suffix(".comparison.json")
        comparison_path.write_text(json.dumps(comparisons, indent=2) + "\n")
        print(json.dumps([dict(model=c["model"], **c["summary"]) for c in comparisons]))
    print(json.dumps(dict(html=str(destination), curves=sum(len(g) for g in curves), points=len(exports))))


if __name__ == "__main__":
    if sys.argv[1:2] == ["throughput"]:
        throughput_report(sys.argv[2:])
    else:
        main()
