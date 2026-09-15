#!/usr/bin/env python3
"""Plot audited MegaMoE input sweeps without mixing ordinary and IKET timing."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    args = parser.parse_args()
    metrics = (
        ("nsys_kernel_median_us", "Ordinary MegaMoE kernel", "Nsight median"),
        ("pre_compute_dispatch_exposure_us_median", "Exposed dispatch", "IKET median"),
        ("compute_span_us_median", "Compute span", "IKET median"),
        ("post_compute_combine_exposure_us_median", "Exposed combine", "IKET median"),
    )
    colors = dict(zip((2, 4, 8, 16, 32, 64),
                      ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#555555")))
    fig, axes = plt.subplots(len(metrics), len(args.roots), squeeze=False,
                             figsize=(6.4 * len(args.roots), 12), layout="constrained")
    coverage, buckets = [], set()
    for column, root in enumerate(args.roots):
        manifest = json.loads((root / "manifest.json").read_text())
        audit = json.loads((root / "audit_summary.json").read_text())
        with (root / "decomposition.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len(rows) != audit["artifact_contracts_checked"]:
            raise ValueError(f"Collected rows and audit coverage differ: {root}")
        expected = {case["case_id"] for case in manifest["cases"]} - set(audit["missing"])
        if len({row["case_id"] for row in rows}) != len(rows) or {row["case_id"] for row in rows} != expected:
            raise ValueError(f"Collected inputs and audited inputs differ: {root}")
        model = manifest["contract"]["model"]
        buckets.add(manifest["contract"]["bucket"])
        status = "complete" if audit["complete"] else "partial"
        coverage.append(f"{model}: {status} {len(rows)}/{audit['expected_cases']}")
        for index, (metric, title, scope) in enumerate(metrics):
            ax = axes[index, column]
            for ep in sorted({int(row["ep"]) for row in rows}):
                group = sorted((row for row in rows if int(row["ep"]) == ep),
                               key=lambda row: int(row["global_tokens"]))
                ax.plot([int(row["global_tokens"]) for row in group],
                        [float(row[metric]) for row in group],
                        color=colors[ep], marker=".", markersize=3.5,
                        linewidth=1.2, label=f"EP{ep}")
            ax.set_xscale("log", base=2)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{model}\n{title}", fontsize=11, loc="left")
            ax.set_ylabel(f"{scope} (µs)")
            ax.set_xlabel("Global input tokens per invocation")
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if index == 0:
                ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.suptitle("MegaMoE final decomposition\n" + " · ".join(coverage), fontsize=13)
    fig.supxlabel("Fixed balanced routes · bucket " + ", ".join(map(str, sorted(buckets))) + " · rank 0\n"
                  "Phase medians are instrumented; do not add them to explain ordinary kernel or full-model latency.",
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
