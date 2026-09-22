#!/usr/bin/env python3
"""Run the standard smoke test with explicit headers or a prepared router."""

import functools
import os
from pathlib import Path
import runpy


def configure():
    from minisgl.kernel import deepgemm

    header_root = os.environ.get("MEGAMOE_HEADER_ROOT")
    router = os.environ.get("MEGAMOE_ROUTER_SOURCE")
    router_module = os.environ.get("MEGAMOE_ROUTER_MODULE")
    if router and router_module:
        raise ValueError("Choose a router source or a prepared module, not both")
    if not header_root and not router_module:
        raise ValueError("Require MEGAMOE_HEADER_ROOT or MEGAMOE_ROUTER_MODULE")
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
    if router_module:
        from minisgl.kernel import moe_topk
        from tvm_ffi import load_module

        path = Path(router_module).resolve(strict=True)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("MEGAMOE_ROUTER_MODULE must name a prepared library")
        module = load_module(str(path))
        if not callable(module.launch):
            raise ValueError("Prepared router must export launch")

        @functools.cache
        def prepared_router(*_args, **_kwargs):
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
    if os.environ.get("MEGAMOE_ROUTER_SOURCE") or os.environ.get("MEGAMOE_ROUTER_MODULE"):
        if os.environ.get("MEGAMOE_SMOKE_TIMING_SCOPE", "pipeline") != "pipeline":
            raise RuntimeError("Producer dispatch requires producer replay in every timed invocation")
        if os.environ.get("MEGAMOE_SMOKE_PRODUCTION_ROUTES") != "1":
            raise RuntimeError("Producer dispatch requires the production route producer")
    configure()
    runpy.run_path(os.environ["MEGAMOE_TEST_ENTRY"], run_name="__main__")
