#!/usr/bin/env python3
"""Run the standard MegaMoE smoke test with a bounded header/router overlay."""

import functools
import os
from pathlib import Path
import runpy


def configure():
    from minisgl.kernel import deepgemm

    root = Path(os.environ["MEGAMOE_HEADER_ROOT"])
    deepgemm._library_root = lambda: root
    router = os.environ.get("MEGAMOE_ROUTER_SOURCE")
    if router:
        from minisgl.kernel import moe_topk
        from minisgl.kernel.utils import load_jit

        @functools.cache
        def build_router():
            iket_include = os.environ.get("DG_JIT_IKET_INCLUDE")
            return load_jit(
                "gate_topk_fused",
                cuda_files=[str(Path(router).resolve(strict=True))],
                cuda_wrappers=[("launch", "GateTopkFusedKernel::run")],
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
    if os.environ.get("MEGAMOE_ROUTER_SOURCE"):
        if os.environ.get("MEGAMOE_SMOKE_TIMING_SCOPE", "pipeline") != "pipeline":
            raise RuntimeError("Producer dispatch requires producer replay in every timed invocation")
        if os.environ.get("MEGAMOE_SMOKE_PRODUCTION_ROUTES") != "1":
            raise RuntimeError("Producer dispatch requires the production route producer")
    configure()
    runpy.run_path(os.environ["MEGAMOE_TEST_ENTRY"], run_name="__main__")
