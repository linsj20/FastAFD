#!/usr/bin/env python3
"""Refresh a previous MegaMoE comparison fragment from a complete audited sweep.

Preserves its layout, three input selections and representative AFD aliases.
Every timing and timeline comes from the new sweep; no historical timing is reused.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from statistics import median


DATA = re.compile(r'(<script[^>]*id="megamoe-data"[^>]*>)(.*?)(</script>)', re.S)
BOUNDARIES = ("measured_ordinal", "dispatch_start_us", "dispatch_end_us",
              "compute_start_us", "compute_end_us", "combine_start_us",
              "combine_end_us", "kernel_envelope_us")


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
    parser.add_argument("--phase-samples", type=int, default=3,
                        help="Expected odd number of retained IKET samples per case")
    args = parser.parse_args()
    if args.phase_samples < 1 or args.phase_samples % 2 == 0:
        parser.error("--phase-samples must be positive and odd")
    if bool(args.standalone) != bool(args.render_script):
        parser.error("--standalone and --render-script must be supplied together")
    template = args.template.read_text()
    match = DATA.search(template)
    if not match or "<!doctype" in template.lower():
        raise ValueError("Expected an editable fragment with exactly one megamoe-data script")
    if len(DATA.findall(template)) != 1:
        raise ValueError("Duplicate data blocks")
    sections = json.loads(match[2])
    manifest = json.loads((args.root / "manifest.json").read_text())
    validation = manifest["final_validation"]
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
    for section in sections:
        updated = []
        for previous in section["rows"]:
            row = dict(by_case[previous["case_id"]])
            case = cases[row["case_id"]]
            record = json.loads((args.root / row["artifact"]).read_text())
            samples = record["samples"]
            if len(samples) != args.phase_samples:
                raise ValueError(f"Expected {args.phase_samples} retained IKET samples")
            selected = sorted(samples, key=lambda x: x["kernel_envelope_us"])[len(samples) // 2]
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
    count = manifest["final_validation"]["baseline_iterations"]
    fragment = re.sub(r"Tables: (?:three|\d+)-sample medians(?: · Nsight: \d+ ordinary samples)?",
                      f"Tables: {args.phase_samples}-sample medians · Nsight: {count} ordinary samples",
                      fragment)
    fragment = re.sub(r"\b(?:three|\d+)-sample medians",
                      f"{args.phase_samples}-sample medians", fragment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    (args.output.parent / "comparison_data.json").write_text(json.dumps(sections, indent=2) + "\n")
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
        args.standalone.write_text(rendered.documents[0])
    print(json.dumps({"fragment": str(args.output), "sections": len(sections),
                      "displayed_rows": sum(len(s["rows"]) for s in sections),
                      "audited_cases": len(cases), "shared_axis_us": int(axis[1])}))


if __name__ == "__main__":
    main()
