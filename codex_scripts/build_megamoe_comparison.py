#!/usr/bin/env python3
"""Refresh a previous MegaMoE comparison fragment from a complete audited sweep.

Preserves its layout, three input selections and representative AFD aliases.
Every timing and timeline comes from the new sweep; no historical timing is reused.
"""

import argparse
import csv
import json
import math
import re
import sqlite3
import subprocess
import sys
from html.parser import HTMLParser
from html import escape
from pathlib import Path
from statistics import median


DATA = re.compile(r'(<script[^>]*id="megamoe-data"[^>]*>)(.*?)(</script>)', re.S)
BOUNDARIES = ("measured_ordinal", "dispatch_start_us", "dispatch_end_us",
              "compute_start_us", "compute_end_us", "combine_start_us",
              "combine_end_us", "kernel_envelope_us")


def timing_inliers(values):
    """One symmetric, per-case mask; never compare ordinary and IKET values."""
    if len(values) < 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Outlier filtering requires at least three finite positive timings")
    center = median(values)
    mad = median(abs(v - center) for v in values)
    radius = max(3 * 1.4826 * mad, 0.05 * center)
    kept = [i for i, value in enumerate(values) if abs(value - center) <= radius]
    if len(kept) < 2:
        raise ValueError("Outlier filtering must retain at least two samples")
    return kept, dict(raw_count=len(values), retained_count=len(kept),
                     raw_median_us=center, filtered_median_us=median(values[i] for i in kept),
                     mad_us=mad, rejection_radius_us=radius,
                     excluded=[dict(ordinal=i, duration_us=v) for i, v in enumerate(values)
                               if i not in kept])


def filter_case(record, run, ordinary_count):
    database = run / "ranks/rank-0/nsys/megamoe-rank0.sqlite"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        ordinary = [r[0] for r in conn.execute(
            "SELECT (k.end-k.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.demangledName "
            "WHERE s.value LIKE '%sm100_fp8_fp4_mega_moe_impl%' ORDER BY k.start")]
    if len(ordinary) != ordinary_count or not math.isclose(
            median(ordinary), record["nsys_kernel_us"]["median"], abs_tol=1e-9):
        raise ValueError(f"Ordinary timings disagree with the audited capture: {run}")
    _, normal_audit = timing_inliers(ordinary)
    samples = record["samples"]
    indices, profile_audit = timing_inliers([s["kernel_envelope_us"] for s in samples])
    for excluded in profile_audit["excluded"]:
        excluded["ordinal"] = samples[excluded["ordinal"]]["measured_ordinal"]
    return [samples[i] for i in indices], dict(ordinary=normal_audit, iket=profile_audit)


class StandaloneFrame(HTMLParser):
    """Extract the rendered document so desktop reports can use the page width."""

    def __init__(self):
        super().__init__()
        self.documents = []

    def handle_starttag(self, tag, attrs):
        if tag == "iframe" and "srcdoc" in dict(attrs):
            self.documents.append(dict(attrs)["srcdoc"])


def overlap(sample, phase):
    return max(0, min(sample[phase + "_end_us"], sample["compute_end_us"])
               - max(sample[phase + "_start_us"], sample["compute_start_us"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--template", type=Path, required=True, help="Previous editable fragment")
    parser.add_argument("--output", type=Path, required=True, help="New editable fragment")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--standalone", type=Path)
    parser.add_argument("--render-script", type=Path)
    parser.add_argument("--title", default="MegaMoE decomposition")
    parser.add_argument("--subtitle", help="Visible standalone report heading and subtitle")
    parser.add_argument("--source-format", choices=("afd", "trtllm-original"), default="afd")
    parser.add_argument("--phase-samples", type=int, default=3,
                        help="Expected odd number of captured IKET samples per case")
    parser.add_argument("--exclude-outliers", action="store_true",
                        help="Filter each timing series at max(3 scaled MAD, 5%% of median)")
    args = parser.parse_args()
    if args.phase_samples < 1 or args.phase_samples % 2 == 0:
        parser.error("--phase-samples must be positive and odd")
    if bool(args.standalone) != bool(args.render_script):
        parser.error("--standalone and --render-script must be supplied together")
    if args.exclude_outliers and (args.source_format != "afd" or args.phase_samples < 3):
        parser.error("Outlier filtering requires AFD captures with at least three phase samples")
    template = args.template.read_text()
    # Derive lengths from the axis mapping: independently scaled widths can
    # reverse the apparent phase ordering when a report expands its time axis.
    template = re.sub(r"width:\(end-start\)/\d+\*plot",
                      "width:x(end)-x(start)", template)
    template = re.sub(r"width:\(overlapEnd-overlapStart\)/\d+\*plot",
                      "width:x(overlapEnd)-x(overlapStart)", template)
    if ("width:x(end)-x(start)" not in template or
            "width:x(overlapEnd)-x(overlapStart)" not in template):
        raise ValueError("Phase bars and overlap bars must use the shared axis mapping")
    match = DATA.search(template)
    if not match or "<!doctype" in template.lower():
        raise ValueError("Expected an editable fragment with exactly one megamoe-data script")
    if len(DATA.findall(template)) != 1:
        raise ValueError("Duplicate data blocks")
    sections = json.loads(match[2])
    manifest = json.loads((args.root / "manifest.json").read_text())
    validation = (manifest["final_validation"] if args.source_format == "afd" else
                  json.loads((args.root / "audit_summary.json").read_text()))
    if args.source_format == "trtllm-original" and validation["source_format"] != args.source_format:
        raise ValueError("Expected an audited original TRT-LLM sweep")
    if "phase_samples" in validation and validation["phase_samples"] != args.phase_samples:
        raise ValueError("Requested phase sample count differs from the sweep manifest")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    for section in sections:
        for row in section["rows"]:
            case = cases[row["case_id"]]
            if (int(row["ep"]), int(row["global_tokens"])) != (case["ep"], case["global_tokens"]):
                raise ValueError(f"Previous input definition differs: {case['case_id']}")
            aliases = {(x["attention_to_ffn_ratio"], x["batch_per_attention_gpu"])
                       for x in case["equivalent_inputs"]}
            for example in row["typical_cases"]:
                m = re.fullmatch(r"A(\d+):F(\d+) · B(\d+)", example["configuration"])
                if not m or (f"{m[1]}:{m[2]}", int(m[3])) not in aliases:
                    raise ValueError(f"Uncovered AFD example: {example}")
    plan = [{"title": s["title"], "cases": [r["case_id"] for r in s["rows"]]} for s in sections]
    if args.plan_only:
        print(json.dumps(plan))
        return
    audit = json.loads((args.root / "audit_summary.json").read_text())
    if not audit["complete"] or audit["missing"] or audit["artifact_contracts_checked"] != len(cases):
        raise ValueError("Final HTML requires a complete audited matrix")
    with (args.root / "decomposition.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    by_case = {r["case_id"]: r for r in rows}
    if len(rows) != len(by_case) or set(by_case) != set(cases):
        raise ValueError("Collected rows do not match the full manifest")
    filtered_cases = {}
    for section in sections:
        updated = []
        for previous in section["rows"]:
            row = dict(by_case[previous["case_id"]])
            case = cases[row["case_id"]]
            record = json.loads((args.root / row["artifact"]).read_text())
            samples = record["samples"]
            if args.source_format == "trtllm-original":
                samples = [dict(s, measured_ordinal=s["sample"]) for s in samples]
            if len(samples) != args.phase_samples:
                raise ValueError(f"Expected {args.phase_samples} retained IKET samples")
            if args.exclude_outliers:
                if row["case_id"] not in filtered_cases:
                    filtered_cases[row["case_id"]] = filter_case(
                        record, (args.root / row["artifact"]).parent,
                        validation["baseline_iterations"])
                samples, exclusion = filtered_cases[row["case_id"]]
                row["nsys_kernel_median_us"] = exclusion["ordinary"]["filtered_median_us"]
                for kind, prefix in (("ordinary", "nsys"), ("iket", "iket")):
                    row[prefix + "_samples_kept"] = exclusion[kind]["retained_count"]
                    row[prefix + "_samples_total"] = exclusion[kind]["raw_count"]
                for key in row:
                    if key.endswith(("_min", "_median", "_max")):
                        metric, stat = key.rsplit("_", 1)
                        if metric in samples[0]:
                            row[key] = {"min": min, "median": median, "max": max}[stat](
                                s[metric] for s in samples)
            center = median(s["kernel_envelope_us"] for s in samples)
            selected = min(samples, key=lambda s: (abs(s["kernel_envelope_us"] - center),
                                                   s["measured_ordinal"]))
            row["timeline"] = {key: selected[key] for key in BOUNDARIES}
            row["tokens_per_ffn_gpu"] = case["global_tokens"] / case["ep"]
            row["typical_cases"] = previous["typical_cases"]
            if "attention_to_ffn_ratio" in previous:
                row["attention_to_ffn_ratio"] = previous["attention_to_ffn_ratio"]
            for phase in ("dispatch", "combine"):
                key = phase + "_compute_overlap_us"
                row["timeline"][key] = overlap(selected, phase)
                row[key + "_median"] = median(overlap(sample, phase) for sample in samples)
            updated.append(row)
        section["rows"] = updated
    axis = re.search(r"t/(\d+)\*plot", template)
    if not axis:
        raise ValueError("Template shared axis was not found")
    largest = max(max(float(r["nsys_kernel_median_us"]), r["timeline"]["kernel_envelope_us"])
                  for section in sections for r in section["rows"])
    if largest > int(axis[1]):
        raise ValueError(f"Shared axis {axis[1]} would clip {largest}; explicitly expand the template axis")
    encoded = json.dumps(sections, ensure_ascii=False).replace("<", "\\u003c")
    fragment = template[:match.start(2)] + encoded + template[match.end(2):]
    fragment = re.sub(r"'text-anchor':t===\d+\?'end':'middle'",
                      f"'text-anchor':t==={axis[1]}?'end':'middle'", fragment)
    count = validation["baseline_iterations"]
    fragment = re.sub(r"Tables: (?:three|\d+)-sample medians(?: · Nsight: \d+ ordinary samples)?",
                      f"Tables: {args.phase_samples}-sample medians · Nsight: {count} ordinary samples",
                      fragment)
    fragment = re.sub(r"\b(?:three|\d+)-sample medians",
                      f"{args.phase_samples}-sample medians", fragment)
    if args.exclude_outliers:
        fragment = fragment.replace(f"Tables: {args.phase_samples}-sample medians · Nsight: {count} ordinary samples",
                                    f"Tables: retained-sample medians · Captured: {args.phase_samples} IKET / {count} ordinary samples")
        fragment = fragment.replace(f"{args.phase_samples}-sample medians", "retained-sample medians")
        fragment = fragment.replace("median-duration IKET launch", "IKET launch nearest retained median")
        fragment = fragment.replace("selected by median IKET kernel duration",
                                    "selected nearest the retained median IKET kernel duration")
        fragment = fragment.replace("computed per launch before taking medians.",
            "computed per retained launch before taking medians. Samples exceeding "
            "max(3 × 1.4826 × MAD, 5% of the case median) in either direction are excluded "
            "by kernel duration, separately for Nsight and IKET. All phases use the same "
            "retained IKET launches; all cases and raw measurements are preserved.")
        fragment = fragment.replace("; Nsight median ${f(r.nsys_kernel_median_us)} microseconds.",
            "; Nsight median ${f(r.nsys_kernel_median_us)} microseconds. Retained: "
            "${r.iket_samples_kept}/${r.iket_samples_total} IKET, "
            "${r.nsys_samples_kept}/${r.nsys_samples_total} ordinary.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    (args.output.parent / "comparison_data.json").write_text(json.dumps(sections, indent=2) + "\n")
    if args.exclude_outliers:
        (args.output.parent / "outlier_audit.json").write_text(json.dumps(dict(
            rule="abs(duration - median) > max(3 * 1.4826 * MAD, 0.05 * median)",
            scope="Displayed cases only; separate masks for ordinary and IKET kernel durations. "
                  "Every phase uses the retained IKET launches. Raw sweep audits remain unfiltered.",
            cases={key: value[1] for key, value in filtered_cases.items()}), indent=2) + "\n")
    for name, section in zip(("max_inputs", "fixed_global_tokens", "fixed_tokens_per_ffn_gpu"), sections):
        table = [{k: v for k, v in row.items() if k not in {"timeline", "typical_cases"}}
                 for row in section["rows"]]
        with (args.output.parent / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    if args.standalone:
        subprocess.run([sys.executable, str(args.render_script), str(args.output.resolve()),
                        str(args.standalone.resolve()), "--title", args.title], check=True)
        rendered = StandaloneFrame()
        rendered.feed(args.standalone.read_text())
        if len(rendered.documents) != 1 or fragment not in rendered.documents[0]:
            raise ValueError("Renderer must retain the complete fragment in exactly one frame")
        document = rendered.documents[0]
        if args.subtitle:
            style = ('<style>html>body{padding:2rem;max-width:1920px;margin:0 auto}'
                     'header{margin-bottom:1.8rem}header h1{margin:0 0 .35rem}'
                     'header .text-small{opacity:.8}'
                     '@media(max-width:600px){html>body{padding:1rem}}</style>')
            heading = (f'<header><h1>{escape(args.title)}</h1>'
                       f'<div class="text-small">{escape(args.subtitle)}</div></header>')
            document = document.replace("</head>", style + "\n</head>", 1)
            document = document.replace("<body>", "<body>\n" + heading, 1)
        args.standalone.write_text(document)
    print(json.dumps({"fragment": str(args.output), "sections": len(sections),
                      "displayed_rows": sum(len(s["rows"]) for s in sections),
                      "audited_cases": len(cases), "shared_axis_us": int(axis[1])}))


if __name__ == "__main__":
    main()
