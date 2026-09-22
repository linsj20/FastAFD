"""Output-scoped PDL consumer for the model all-push recycling kernel."""

import functools
from pathlib import Path

from .utils import load_jit


@functools.cache
def _jit_module(trace=False):
    import flashinfer

    return load_jit(
        "megamoe_output_norm" + ("_trace" if trace else ""),
        cuda_files=["megamoe_output_norm.cu"],
        cuda_wrappers=[("launch", f"MegaMoEOutputNorm::run<{str(trace).lower()}>")],
        extra_include_paths=[str(Path(flashinfer.__file__).parent / "data" / "include")],
    )


def megamoe_output_add_rmsnorm(hidden, residual, weight, eps, buffer, *, debug_timings=None):
    if residual is None:
        raise ValueError("MegaMoE output normalization requires its model residual")
    _jit_module(debug_timings is not None).launch(
        hidden, residual, weight, buffer.buffer, buffer.num_ranks, float(eps),
        buffer.buffer if debug_timings is None else debug_timings,
    )
