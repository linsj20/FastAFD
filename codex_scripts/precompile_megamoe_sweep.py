#!/usr/bin/env python3
"""CPU-prepare ordinary/instrumented MegaMoE specializations from sweep manifests."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests", nargs="+", type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--metadata-output", type=Path,
                   help="Write preparation records separately when reusing an existing cache")
    p.add_argument("--header-root", type=Path, help="Explicit device header overlay")
    p.add_argument("--router-source", type=Path, help="Explicit router source for the overlay")
    p.add_argument("--router-wrapper", default="GateTopkAllPushKernel")
    p.add_argument("--iket-include", required=True)
    p.add_argument("--ordinary-only", action="store_true",
                   help="Prepare only ordinary kernels, without IKET variants")
    p.add_argument("--payload-mode", choices=("all_push", "pull"), default="all_push")
    p.add_argument("--preparation-inputs", nargs="*", type=Path, default=[],
                   help="Saved kernel.cu specializations used by untimed weight/input preparation")
    p.add_argument("--arch", default="100f")
    args = p.parse_args()
    if args.router_source and not args.header_root:
        p.error('--router-source requires --header-root')
    root = args.cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = (args.metadata_output or root).resolve()
    metadata.mkdir(parents=True, exist_ok=True)
    os.environ.update(MINISGL_DEEPGEMM_BUILD_DIR=str(root / "native"),
                      DG_JIT_CACHE_DIR=str(root / "jit"),
                      TVM_FFI_CACHE_DIR=str(root / "tvm"),
                      TVM_FFI_CUDA_ARCH_LIST="10.0a", MAX_JOBS="4")
    sys.path.insert(0, str(args.source.resolve() / "python"))
    import torch
    from minisgl.kernel import deepgemm
    if args.header_root:
        os.environ['MEGAMOE_HEADER_ROOT'] = str(args.header_root.resolve(strict=True))
        if args.router_source:
            os.environ['MEGAMOE_ROUTER_SOURCE'] = str(args.router_source.resolve(strict=True))
            os.environ['MEGAMOE_ROUTER_WRAPPER'] = args.router_wrapper
        from megamoe_header_smoke import configure
        configure()
    from minisgl.kernel.moe_topk import _jit_gate_topk_fused_module

    specializations, cases = {}, []
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        c = manifest["contract"]
        for case in manifest["cases"]:
            c = dict(manifest["contract"], **case.get("capture_contract", {}))
            inputs = (case["ep"], c["experts"], c["bucket"], case["rows_per_rank"],
                      c["top_k"], c["hidden"], c["intermediate"], c["num_sms"])
            # The EP64 ordinary gate compares selected versus conventional combine.
            for inband, rank_gated in ([(True, True), (False, False)]
                                       if case["ep"] == 64 else [(True, True)]):
                code, config, block_m, wave = deepgemm.generate_megamoe_kernel(
                    *inputs, args.payload_mode == "all_push", inband, rank_gated)
                if args.payload_mode == "all_push":
                    assert block_m <= 32 and wave == c["experts"] // case["ep"]
                q, r = divmod(case["global_tokens"] * c["top_k"], c["experts"])
                padded = sum(((q + int(e * case["ep"] < r) + block_m - 1) // block_m) * block_m
                             for e in range(c["experts"] // case["ep"]))
                if args.payload_mode == "all_push" and case["rows_per_rank"] + padded > c["bucket"]:
                    raise ValueError(f"Insufficient materialization capacity: {case['case_id']}")
                key = f"{path.parent.name}/{case['case_id']}/{'selected' if inband else 'combine_reference'}"
                specializations.setdefault(code, {"config": config, "cases": []})["cases"].append(key)
                cases.append(dict(case=key, block_m=block_m, experts_per_wave=wave))
    (metadata / "configurations.json").write_text(json.dumps(cases, indent=2) + "\n")
    records = []
    modes = [("ordinary", "")]
    if not args.ordinary_only:
        modes.append(("iket", args.iket_include))
    for mode, include in modes:
        os.environ["DG_JIT_IKET_INCLUDE"] = include
        os.environ["DG_JIT_IKET_CRITICAL"] = "1"
        _jit_gate_topk_fused_module.cache_clear()
        _jit_gate_topk_fused_module(all_push=args.payload_mode == "all_push")
        for code, info in specializations.items():
            target = deepgemm.precompile_kernel("sm100_fp8_fp4_mega_moe", code, args.arch)
            records.append(dict(mode=mode, target=target, **info))
            print(f"PRECOMPILED {mode} {info['cases'][0]} -> {target}", flush=True)
    # Cache-only execution also covers untimed input/weight preparation. Reuse
    # its exact saved template code, with the selected source's current headers.
    os.environ["DG_JIT_IKET_INCLUDE"] = ""
    preparation = []
    seen = set()
    for path in args.preparation_inputs:
        parts = path.parent.name.split(".", 2)
        first, separator, code = path.read_text().partition("\n")
        if (path.name != "kernel.cu" or len(parts) != 3 or parts[0] != "kernel"
                or not separator or not first.startswith("// Includes' hash value: ")):
            raise ValueError(f"Expected a saved DeepGEMM kernel.cu: {path}")
        key = parts[1], code
        if key in seen:
            continue
        seen.add(key)
        target = deepgemm.precompile_kernel(parts[1], code, args.arch)
        preparation.append(dict(name=parts[1], input=str(path), target=target))
        print(f"PREPARATION_READY {parts[1]} -> {target}", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation initialized CUDA")
    (metadata / "manifest.json").write_text(json.dumps(dict(source=str(args.source),
        cuda_initialized=False, configurations=cases, specializations=records,
        preparation_kernels=preparation), indent=2) + "\n")
    print(f"SWEEP_CPU_READY cases={len(cases)} builds={len(records)}", flush=True)


if __name__ == "__main__":
    main()
