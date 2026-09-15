#!/usr/bin/env python3
"""Inspect ordinary MegaMoE dispatch register regions in a source-scoped JIT cache.

This checks the linear region between dispatch register deallocation and the next
role's register change. It does not prove control flow, synchronization correctness,
or the budgets of native senders that execute before dispatch deallocation.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def inspect(sass):
    changes = list(re.finditer(r"USETMAXREG\.[^;]+;", sass))
    starts = [i for i, match in enumerate(changes)
              if re.search(r"DEALLOC\.CTAPOOL 0x(?:30|60)\s*;", match[0])]
    if len(starts) != 1 or starts[0] + 1 >= len(changes):
        raise ValueError("Expected one dispatch deallocation followed by another role")
    i = starts[0]
    budget = int(re.search(r"0x([0-9a-f]+)", changes[i][0])[1], 16)
    region = sass[changes[i].end():changes[i + 1].start()]
    return {
        "scope": "linear dispatch region; excludes any pre-deallocation sender",
        "start": changes[i][0], "end": changes[i + 1][0], "budget": budget,
        **inspect_region(region),
    }


def inspect_register_regions(sass):
    """Inspect each deallocation span, including TMA/MMA roles; not a CFG proof."""
    changes = list(re.finditer(r"USETMAXREG\.[^;]+;", sass))
    regions = []
    for current, following in zip(changes, changes[1:]):
        match = re.search(r"DEALLOC\.CTAPOOL 0x([0-9a-f]+)\s*;", current[0])
        if match:
            regions.append({
                "scope": "linear register region; control-flow paths require separate review",
                "start": current[0], "end": following[0],
                "budget": int(match[1], 16),
                **inspect_region(sass[current.end():following.start()]),
            })
    return regions


def inspect_reachable_registers(sass):
    """Follow nvdisasm branch labels from each deallocation until exit/reallocation.

    Indirect branches require nvdisasm BRANCH_TARGETS annotations. Unresolved
    transfers are reported explicitly; this does not prove runtime predicates.
    """
    instructions, labels = [], {}
    for line in sass.splitlines():
        label = re.match(r"\s*(\.L[\w.]+):", line)
        if label:
            labels[label[1]] = len(instructions)
        if re.search(r"/\*[0-9a-f]+\*/", line):
            instructions.append(line)
    reports = []
    for entry, line in enumerate(instructions):
        allocation = re.search(r"USETMAXREG\.DEALLOC\.CTAPOOL 0x([0-9a-f]+)", line)
        if not allocation:
            continue
        pending, visited, unresolved = [entry + 1], set(), set()
        while pending:
            index = pending.pop()
            if index in visited or index >= len(instructions):
                continue
            current = instructions[index]
            if "USETMAXREG." in current:
                continue
            visited.add(index)
            match = re.search(r"\*/\s+(@!?[A-Z]+\d*\s+)?([A-Z][\w.]*)\s*(.*)", current)
            if not match:
                unresolved.add(current.strip())
                continue
            predicate, operation, operands = match.groups()
            transfer = operation.startswith(("BRA", "BRX", "JMP", "JMX", "CALL", "BSSY"))
            terminal = operation.startswith(("EXIT", "RET"))
            if transfer:
                targets = re.findall(r"\.L[\w.]+", operands)
                if not targets:
                    unresolved.add(current.strip())
                for target in targets:
                    if target in labels:
                        pending.append(labels[target])
                    else:
                        unresolved.add(target)
            conditional = bool(predicate or re.match(r"!?UP(?:\d+|T)\s*,", operands))
            if (not transfer and not terminal) or conditional or operation.startswith(("BSSY", "CALL")):
                pending.append(index + 1)
        reports.append({
            "scope": "Reachable explicit instructions between register changes; unresolved transfers require review",
            "entry": line.strip(), "budget": int(allocation[1], 16),
            "instructions_checked": len(visited), "unresolved_transfers": sorted(unresolved),
            **inspect_region("\n".join(instructions[i] for i in sorted(visited))),
        })
    return reports


def inspect_region(region):
    registers = [int(n) for n in re.findall(r"\bR(\d+)\b", region)]
    wide = [int(n) + 1 for n in re.findall(r"\bR(\d+)\.64\b", region)]
    for line in region.splitlines():
        instruction = re.search(r"\*/\s+(?:@!?[A-Z]+\d*\s+)?([A-Z][\w.]+)\s+(.*);", line)
        if not instruction:
            continue
        op, operands = instruction.groups()
        width = 8 if ".256" in op else 4 if ".128" in op else 2 if ".64" in op else 1
        data = re.findall(r"\bR(\d+)\b", operands)
        if ".256" in op and op.startswith("LDG"):
            # SM100 disassembly names two four-register groups for LDG.256.
            wide.extend(int(n) + 3 for n in data[:2])
        elif ".256" in op and op.startswith("STG"):
            wide.extend(int(n) + 3 for n in data[-2:])
        elif data and op.startswith(("LDG", "LDS", "LDC")):
            wide.append(int(data[0]) + width - 1)
        elif data and op.startswith(("STG", "STS")):
            wide.append(int(data[-1]) + width - 1)
    return {
        "highest_explicit_register": max(registers, default=-1),
        "highest_register_including_wide_memory_operands": max(registers + wide, default=-1),
        "local_memory_instructions": re.findall(r"\b(?:LDL|STL)[^;]*;", region),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Ordinary source-scoped jit/cache directory")
    parser.add_argument("--nvdisasm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--include-instrumented", action="store_true",
                        help="Also inspect IKET builds, labeled separately in the results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sorted(args.cache.glob("kernel.sm100_fp8_fp4_mega_moe.*/kernel.cu")):
        text = source.read_text()
        arguments = text.split("sm100_fp8_fp4_mega_moe_impl<", 1)[1].split(">);", 1)[0].strip()
        values = [v.strip() for v in arguments.split(",")]
        if args.bucket is not None and int(values[0]) != args.bucket:
            continue
        if args.ranks is not None and int(values[19]) != args.ranks:
            continue
        cubin = source.with_suffix(".cubin")
        sass = subprocess.check_output([args.nvdisasm, str(cubin)], text=True)
        instrumented = "__iket_meta_info" in sass
        if instrumented and not args.include_instrumented:
            continue
        destination = args.output / f"kernel_{len(results)}_sass.txt"
        destination.write_text(sass)
        results.append({"template_arguments": arguments, "cubin": str(cubin),
                        "sass": str(destination), "iket_build": instrumented,
                        **inspect(sass)})
    if not results:
        raise ValueError("No ordinary kernels matched the requested geometry")
    (args.output / "compiled_review.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(json.dumps({**{key: value for key, value in result.items()
                             if key not in {"template_arguments", "local_memory_instructions"}},
                          "local_memory_instruction_count": len(result["local_memory_instructions"])}))
    for result in results:
        if result["highest_register_including_wide_memory_operands"] >= result["budget"]:
            print("MANUAL_CONTROL_FLOW_REVIEW: linear span includes registers above the "
                  "dispatch budget; other warp roles may legitimately use them.")


if __name__ == "__main__":
    main()
