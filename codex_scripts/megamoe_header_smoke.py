#!/usr/bin/env python3
"""Run the standard smoke test with explicit headers or a prepared router."""

import functools
import json
import os
import runpy
import sys
from pathlib import Path


def loaded_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and fields[5].endswith(".so"):
            paths.add(Path(fields[5]).resolve(strict=True))
    return paths


def prepare_routers(argv):
    import argparse

    parser = argparse.ArgumentParser(description="CPU-build ordinary routers and record their loaded libraries")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    cache = args.cache_root.absolute()
    cache.mkdir(parents=True, exist_ok=True)
    cache = cache.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(TVM_FFI_CACHE_DIR=str(cache), TVM_FFI_CUDA_ARCH_LIST="10.0a",
                      DG_JIT_IKET_INCLUDE="", MAX_JOBS="4")
    sys.path.insert(0, str(source / "python"))
    import torch
    from minisgl.kernel import moe_topk

    modules = {}
    for all_push in (False, True):
        name = "gate_topk_fused_all_push" if all_push else "gate_topk_fused"
        before = loaded_libraries()
        module = moe_topk._jit_gate_topk_fused_module(all_push=all_push)
        matches = [p for p in loaded_libraries() - before
                   if p.is_relative_to(cache) and p.parent.name.startswith(f"minisgl__{name}_")]
        if len(matches) != 1 or not callable(module.launch):
            raise RuntimeError(f"Expected exactly one loaded {name} library: {matches}")
        path = matches[0]
        expected = source / "python/minisgl/kernel/csrc/jit" / f"{name}.cu"
        if f'#include "{expected}"' not in (path.parent / "cuda.cu").read_text():
            raise RuntimeError(f"Prepared router uses a different source: {path}")
        modules["all_push" if all_push else "pull"] = str(path)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU router preparation initialized CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(dict(source=str(source), modules=modules, cuda_initialized=False), output, indent=2)
        output.write("\n")
    print("PREPARED_ORDINARY_ROUTER_MODULES", json.dumps(modules), flush=True)


def configure():
    from minisgl.kernel import deepgemm

    header_root = os.environ.get("MEGAMOE_HEADER_ROOT")
    router = os.environ.get("MEGAMOE_ROUTER_SOURCE")
    router_module = os.environ.get("MEGAMOE_ROUTER_MODULE")
    router_modules = os.environ.get("MEGAMOE_ROUTER_MODULES")
    if sum(bool(value) for value in (router, router_module, router_modules)) > 1:
        raise ValueError("Choose one router source, prepared module, or module map")
    if not header_root and not router_module and not router_modules:
        raise ValueError("Require headers or a prepared router module/map")
    root = Path(header_root).resolve(strict=True) if header_root else None
    if root is not None:
        if not (root / "include/deep_gemm").is_dir():
            raise ValueError("MEGAMOE_HEADER_ROOT must contain include/deep_gemm")
        deepgemm._library_root = lambda: root
        print(f"MEGAMOE_HEADER_ROOT_READY path={root}", flush=True)
    native_source = os.environ.get("MEGAMOE_NATIVE_SOURCE_ROOT")
    if native_source:
        native_root = Path(native_source).resolve(strict=True)
        if not (native_root / "csrc" / "python_api.cpp").is_file():
            raise ValueError("MEGAMOE_NATIVE_SOURCE_ROOT must contain csrc/python_api.cpp")
        # Select the same host descriptor source for both cache checks and rebuilds.
        # A device-header override alone does not change the native extension source.
        deepgemm._source_root = lambda: native_root
    if router_modules:
        from minisgl.kernel import moe_topk
        from tvm_ffi import load_module

        spec = json.loads(Path(router_modules).read_text())
        source = Path(spec["source"]).resolve(strict=True)
        if not Path(moe_topk.__file__).resolve().is_relative_to(source):
            raise ValueError("Prepared routers belong to a different Python source")
        if set(spec["modules"]) != {"pull", "all_push"}:
            raise ValueError("Prepared router map must contain pull and all_push")
        modules = {}
        for mode, value in spec["modules"].items():
            path = Path(value).resolve(strict=True)
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Empty prepared router: {path}")
            modules[mode] = load_module(str(path))
            if not callable(modules[mode].launch):
                raise ValueError(f"Prepared router must export launch: {path}")

        @functools.cache
        def prepared_routers(all_push=False, recycle=False, ahead=False):
            if recycle or ahead:
                raise ValueError("Prepared router map supports ordinary, non-recycling routes only")
            return modules["all_push" if all_push else "pull"]

        moe_topk._jit_gate_topk_fused_module = prepared_routers
        print(f"MEGAMOE_ROUTER_MODULES_READY path={router_modules}", flush=True)
    elif router_module:
        from minisgl.kernel import moe_topk
        from tvm_ffi import load_module

        path = Path(router_module).resolve(strict=True)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("MEGAMOE_ROUTER_MODULE must name a prepared library")
        module = load_module(str(path))
        if not callable(module.launch):
            raise ValueError("Prepared router must export launch")
        original_router = moe_topk._jit_gate_topk_fused_module
        all_push_only = os.environ.get("MEGAMOE_ROUTER_WRAPPER") == "GateTopkAllPushKernel"

        @functools.cache
        def prepared_router(all_push=False, recycle=False, ahead=False):
            if all_push_only and not all_push:
                return original_router(all_push=all_push, recycle=recycle, ahead=ahead)
            return module

        moe_topk._jit_gate_topk_fused_module = prepared_router
        print(f"MEGAMOE_ROUTER_MODULE_READY path={path}", flush=True)
    elif router:
        from minisgl.kernel import moe_topk
        from minisgl.kernel.utils import load_jit

        @functools.cache
        def build_router(*_args, **_kwargs):
            iket_include = os.environ.get("DG_JIT_IKET_INCLUDE")
            return load_jit(
                "gate_topk_fused",
                cuda_files=[str(Path(router).resolve(strict=True))],
                cuda_wrappers=[("launch", os.environ.get(
                    "MEGAMOE_ROUTER_WRAPPER", "GateTopkFusedKernel") + "::run")],
                extra_include_paths=[str(root / "include")] + ([iket_include] if iket_include else []),
                extra_cuda_cflags=["-DIKET_BUILD_ENABLED=1"] if iket_include else [],
            )

        moe_topk._jit_gate_topk_fused_module = build_router


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare-routers"]:
        prepare_routers(sys.argv[2:])
        raise SystemExit(0)
    stack_timeout = float(os.environ.get("MEGAMOE_SMOKE_STACK_TIMEOUT", "0"))
    if stack_timeout < 0:
        raise ValueError("MEGAMOE_SMOKE_STACK_TIMEOUT must be nonnegative")
    if stack_timeout:
        import faulthandler

        stack_path = (Path(os.environ["FASTAFD_IKET_RUN_ROOT"]) / "ranks" /
                      f"rank-{os.environ['SLURM_PROCID']}" / "python_stack.txt")
        stack_path.parent.mkdir(parents=True, exist_ok=True)
        stack_file = stack_path.open("w")
        faulthandler.dump_traceback_later(stack_timeout, file=stack_file)
    if any(os.environ.get(key) for key in (
        "MEGAMOE_ROUTER_SOURCE", "MEGAMOE_ROUTER_MODULE", "MEGAMOE_ROUTER_MODULES"
    )):
        if os.environ.get("MEGAMOE_SMOKE_TIMING_SCOPE", "pipeline") != "pipeline":
            raise RuntimeError("Producer dispatch requires producer replay in every timed invocation")
        if os.environ.get("MEGAMOE_SMOKE_PRODUCTION_ROUTES") != "1":
            raise RuntimeError("Producer dispatch requires the production route producer")
    configure()
    entry = Path(os.environ["MEGAMOE_TEST_ENTRY"]).resolve(strict=True)
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
