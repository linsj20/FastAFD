#!/usr/bin/env python3
"""Extract selected cuobjdump functions without copying whole module dumps.

Run on a CPU allocation for large dumps. Instruction counts and explicit register
names are static evidence, not dynamic costs or a resource/ordering proof.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def extract(path, contains):
    functions = []
    current = None
    with path.open() as stream:
        for line in stream:
            match = re.search(r"Function\s*:\s*(\S+)", line)
            if match:
                name = match[1]
                current = None
                if all(fragment in name for fragment in contains):
                    current = dict(name=name, instructions=[])
                    functions.append(current)
                continue
            instruction = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.+?);", line)
            if current is not None and instruction:
                current["instructions"].append(dict(
                    address=int(instruction[1], 16), text=instruction[2].strip()))
    if not functions or any(not f["instructions"] for f in functions):
        raise ValueError(f"Missing or empty selected functions: {path}")
    if len({f['name'] for f in functions}) != len(functions):
        raise ValueError(f"Repeated function names; select a single architecture: {path}")
    for function in functions:
        operations = Counter()
        local = []
        for instruction in function["instructions"]:
            text = instruction["text"]
            opcode = re.sub(r"^@\S+\s+", "", text).split()[0]
            operations[opcode] += 1
            if opcode.startswith(("LDL", "STL")):
                local.append(instruction)
        function.update(operation_counts=dict(sorted(operations.items())),
                        local_memory=local)
    return dict(source=str(path), functions=functions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--contains", action="append", required=True,
                        help="Require every given substring in the mangled function name")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [extract(path, args.contains) for path in args.dumps]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        scope="Static cuobjdump instructions; no dynamic timing or resource proof",
        records=records), separators=(",", ":")) + "\n")
    print(json.dumps({record["source"]: len(record["functions"])
                      for record in records}), flush=True)


if __name__ == "__main__":
    main()
