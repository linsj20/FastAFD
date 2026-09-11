#!/usr/bin/env python3
"""Plot previous, matched-current, and added AFD Pareto points from sweep.csv."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frontier(points: list[dict]) -> list[dict]:
    return sorted(
        [p for p in points if not any(
            q["x"] >= p["x"] and q["y"] >= p["y"]
            and (q["x"] > p["x"] or q["y"] > p["y"])
            for q in points
        )], key=lambda p: p["x"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--baseline-label", default="Previous EP4/8")
    parser.add_argument("--candidate-label", default="Optimized EP4/8")
    parser.add_argument("--added-label", default="Optimized EP16")
    args = parser.parse_args()
    with args.sweep.open() as stream:
        rows = list(csv.DictReader(stream))
    previous = []
    for path in sorted(args.baseline.glob("*.json")):
        metric = json.loads(path.read_text())
        previous.append({
            "id": metric["case_id"],
            "x": 1000 / metric["cuda_wall_time_mean_ms"],
            "y": metric["tps_per_active_gpu"],
        })
    current, added = [], []
    for row in rows:
        point = {
            "id": row["case_id"], "x": float(row["tps_per_user"]),
            "y": float(row["tps_per_gpu"]),
            "detail": f"A{row['attention_gpus']}:F{row['ffn_gpus']}, EP{row['ep']}, batch {row['batch']}",
        }
        (current if row["old_latency_ms"] else added).append(point)
    fig, ax = plt.subplots(figsize=(10.4, 6.7), layout="constrained")
    tooltips, fronts = {}, {}
    series = [
        (args.baseline_label, previous, "#707780", "o"),
        (args.candidate_label, current, "#0072b2", "^"),
        (args.added_label, added, "#d55e00", "s"),
    ]
    for index, (label, points, color, marker) in enumerate(series):
        if not points:
            continue
        front = frontier(points)
        fronts[label] = [p["id"] for p in front]
        front_ids = set(fronts[label])
        for number, point in enumerate(points):
            on_front = point["id"] in front_ids
            gid = f"series-{index}-point-{number}"
            artist, = ax.plot(
                point["x"], point["y"], marker=marker, linestyle="none",
                markersize=6 if on_front else 4.5, color=color,
                markerfacecolor=color if on_front else "none",
                alpha=1 if on_front else 0.35, zorder=3 if on_front else 2,
            )
            artist.set_gid(gid)
            detail = point.get("detail", point["id"])
            tooltips[gid] = f"{label}: {detail}; {point['x']:.3f} TPS/user; {point['y']:.3f} TPS/GPU"
        ax.plot([p["x"] for p in front], [p["y"] for p in front],
                color=color, linewidth=1.8, label=f"{label} ({len(points)} points)")
    ax.set_xlabel("Throughput per user (tokens/s)", fontsize=12)
    ax.set_ylabel("Throughput per active GPU (tokens/s)", fontsize=12)
    summary = json.loads(args.sweep.with_name("sweep_summary.json").read_text())
    suffix = "" if summary["status"] == "complete" else f" — partial {len(rows)}/{summary['planned']}"
    ax.set_title("FastAFD · 128K Pareto comparison" + suffix, fontsize=15, loc="left", pad=18)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.margins(x=0.05, y=0.07)
    fig.text(0.5, -0.025,
             "Filled markers and lines: each series’ Pareto frontier.  Current runs: no prefill, perfect balance, MB2 overlap.",
             ha="center", fontsize=9, color="#555555")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output_prefix.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    svg = args.output_prefix.with_suffix(".svg")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg)
    for element in tree.iter():
        if element.get("id") in tooltips:
            title = ET.Element("{http://www.w3.org/2000/svg}title")
            title.text = tooltips[element.get("id")]
            element.insert(0, title)
    tree.write(svg, encoding="unicode", xml_declaration=True)
    args.output_prefix.with_suffix(".frontiers.json").write_text(json.dumps(fronts, indent=2) + "\n")


if __name__ == "__main__":
    main()
