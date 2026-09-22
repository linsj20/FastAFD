#!/usr/bin/env python3
"""Compile saved DeepGEMM specializations on CPUs for an explicit CUDA target.

Inputs are kernel.cu files or cache directories containing kernel.*/kernel.cu.
The serving source supplies headers and the native compiler; only the saved
template arguments are reused. Run with the serving Python environment/image.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--header-root", type=Path,
                        help="Explicit device library overlay; native source stays --source")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {}
    for path in args.inputs:
        files = sorted(path.glob("kernel.*/kernel.cu")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"No saved kernel specializations: {path}")
        for file in files:
            parts = file.parent.name.split(".", 2)
            if file.name != "kernel.cu" or len(parts) != 3 or parts[0] != "kernel":
                raise ValueError(f"Expected a generated DeepGEMM kernel path: {file}")
            first, separator, code = file.read_text().partition("\n")
            if not separator or not first.startswith("// Includes' hash value: "):
                raise ValueError(f"Missing generated-source preamble: {file}")
            sources.setdefault((parts[1], code), []).append(str(file))
    os.environ["DG_JIT_CACHE_DIR"] = str(args.cache_root.resolve())
    sys.path.insert(0, str(args.source.resolve() / "python"))
    import torch
    from minisgl.kernel import deepgemm

    if args.header_root is not None:
        header_root = args.header_root.resolve(strict=True)
        if not (header_root / "include/deep_gemm").is_dir():
            raise ValueError("Header root must contain include/deep_gemm")
        deepgemm._library_root = lambda: header_root

    records = []
    for (name, code), inputs in sources.items():
        target = deepgemm.precompile_kernel(name, code, args.arch)
        records.append({"name": name, "inputs": inputs, "target": target})
        print(f"PRECOMPILED {name}: {target}", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU cache preparation unexpectedly initialized CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.source), "arch": args.arch,
        "header_root": str(args.header_root) if args.header_root else None,
        "cache_root": str(args.cache_root), "cuda_initialized": False,
        "kernels": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
