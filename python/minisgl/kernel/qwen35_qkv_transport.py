from __future__ import annotations

import functools

import torch

from .utils import load_jit


@functools.cache
def _jit_module():
    return load_jit(
        "qwen35_qkv_transport",
        cuda_files=["qwen35_qkv_transport.cu"],
        cuda_wrappers=[
            ("publish", "Qwen35QKVTransportKernel::publish"),
            ("publish_linear", "Qwen35QKVTransportKernel::publish_linear"),
            ("shared_add_rmsnorm", "Qwen35QKVTransportKernel::shared_add_rmsnorm"),
        ],
    )


def ensure_qwen35_qkv_transport_built() -> None:
    _jit_module()


def publish_qwen35_qkv(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_loc: torch.Tensor,
    source_offsets: torch.Tensor,
    q_descriptors: torch.Tensor,
    kv_descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    release_turn: torch.Tensor | None = None,
    next_turn: int | None = None,
    layer: int,
    slot: int,
    ready_value: int,
    eps: float,
) -> None:
    """Prepare and remotely publish one Qwen3.5 full-attention QKV payload."""

    rows = int(q_gate.shape[0]) if q_gate.ndim else -1
    for name, tensor in (("q_gate", q_gate), ("k", k), ("v", v)):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or tensor.ndim != 2
            or int(tensor.shape[0]) != rows
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                f"Qwen3.5 {name} must be a rank-2 strided CUDA BF16 matrix "
                f"with {rows} rows, got shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype} device={tensor.device} "
                f"stride={tensor.stride()}"
            )
    head_dim = 256
    if int(q_gate.shape[1]) % (2 * head_dim):
        raise ValueError("Qwen3.5 q_gate width must contain interleaved Q/gate heads")
    if int(k.shape[1]) == 0 or int(k.shape[1]) % head_dim or k.shape != v.shape:
        raise ValueError(
            "Qwen3.5 K/V must have equal, nonempty widths divisible by 256"
        )
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_cuda
        or not positions.is_contiguous()
        or tuple(positions.shape) != (rows,)
    ):
        raise ValueError("Qwen3.5 positions must be one contiguous CUDA int vector per row")
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or not weight.is_contiguous()
            or tuple(weight.shape) != (head_dim,)
        ):
            raise ValueError(f"Qwen3.5 {name} must be contiguous CUDA BF16[256]")
    if (
        cos_sin_cache.dtype != torch.float32
        or not cos_sin_cache.is_cuda
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.stride(1) != 1
        or int(cos_sin_cache.shape[1]) != 64
    ):
        raise ValueError(
            "Qwen3.5 full-attention RoPE cache must be a CUDA FP32 matrix "
            "with rotary_dim=64"
        )
    if (
        out_loc.dtype not in (torch.int32, torch.int64)
        or not out_loc.is_cuda
        or not out_loc.is_contiguous()
        or tuple(out_loc.shape) != (rows,)
    ):
        raise ValueError("Qwen3.5 out_loc must be one contiguous CUDA int vector per row")
    if (
        source_offsets.dtype != torch.int64
        or not source_offsets.is_cuda
        or not source_offsets.is_contiguous()
        or source_offsets.ndim != 1
        or source_offsets.numel() < 2
    ):
        raise ValueError("Qwen3.5 source offsets must be a contiguous CUDA int64 vector")
    descriptor_specs = (
        ("q", q_descriptors, 7),
        ("kv", kv_descriptors, 8),
        ("ready", ready_descriptors, 3),
    )
    for name, descriptor, width in descriptor_specs:
        if (
            descriptor.dtype != torch.int64
            or not descriptor.is_cuda
            or not descriptor.is_contiguous()
            or descriptor.ndim != 2
            or descriptor.shape[0] < 1
            or int(descriptor.shape[1]) != width
        ):
            raise ValueError(
                f"Qwen3.5 {name} descriptors must be nonempty contiguous "
                f"CUDA int64[:, {width}]"
            )
    if (
        completion_counter.dtype != torch.int32
        or not completion_counter.is_cuda
        or not completion_counter.is_contiguous()
        or completion_counter.numel() != 1
    ):
        raise ValueError("Qwen3.5 completion counter must be one CUDA int32")
    _validate_optional_turn(release_turn, next_turn)
    if layer < 0 or slot not in (0, 1) or ready_value <= 0 or eps <= 0:
        raise ValueError(
            "Qwen3.5 publication requires a nonnegative layer, slot 0/1, "
            "positive readiness ticket, and positive epsilon"
        )

    _jit_module().publish(
        q_gate,
        k,
        v,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        out_loc,
        source_offsets,
        q_descriptors,
        kv_descriptors,
        ready_descriptors,
        completion_counter,
        release_turn,
        -1 if next_turn is None else int(next_turn),
        int(layer),
        int(slot),
        int(ready_value),
        float(eps),
    )


def publish_qwen35_linear_input(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    source_offsets: torch.Tensor,
    q_descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    shared: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
    release_turn: torch.Tensor | None = None,
    next_turn: int | None = None,
    slot: int,
    ready_value: int,
    eps: float,
) -> None:
    """Fuse residual-add/RMSNorm with Qwen3.5 linear-input publication."""

    if (
        hidden.dtype != torch.bfloat16
        or not hidden.is_cuda
        or hidden.ndim != 2
        or hidden.stride(1) != 1
        or int(hidden.shape[0]) < 1
        or int(hidden.shape[1]) % 1024
        or int(hidden.shape[1]) > 8192
    ):
        raise ValueError(
            "Qwen3.5 linear input must be a nonempty strided CUDA BF16 matrix "
            "with hidden size divisible by 1024 and at most 8192"
        )
    if shared is not None and residual is None:
        raise ValueError(
            "Qwen3.5 shared linear input requires an existing residual"
        )
    shared_input = hidden if shared is None else shared
    if shared_gate is not None and shared is None:
        raise ValueError("Qwen3.5 shared gate requires a shared output")
    if (
        shared_input.dtype != hidden.dtype
        or not shared_input.is_cuda
        or shared_input.shape != hidden.shape
        or shared_input.stride(1) != 1
    ):
        raise ValueError("Qwen3.5 shared output must match the linear hidden input")
    if shared_gate is not None and (
        shared_gate.dtype != hidden.dtype
        or not shared_gate.is_cuda
        or tuple(shared_gate.shape) != (int(hidden.shape[0]), 1)
        or not shared_gate.is_contiguous()
    ):
        raise ValueError("Qwen3.5 shared gate must be contiguous CUDA BF16 [rows, 1]")
    residual_input = hidden if residual is None else residual
    if (
        residual_input.dtype != hidden.dtype
        or not residual_input.is_cuda
        or residual_input.shape != hidden.shape
        or residual_input.stride(1) != 1
    ):
        raise ValueError("Qwen3.5 residual must match the linear hidden input")
    if (
        weight.dtype != hidden.dtype
        or not weight.is_cuda
        or not weight.is_contiguous()
        or tuple(weight.shape) != (int(hidden.shape[1]),)
    ):
        raise ValueError("Qwen3.5 linear RMSNorm weight must match hidden width")
    if (
        source_offsets.dtype != torch.int64
        or not source_offsets.is_cuda
        or not source_offsets.is_contiguous()
        or source_offsets.ndim != 1
        or source_offsets.numel() < 2
    ):
        raise ValueError("Qwen3.5 source offsets must be a contiguous CUDA int64 vector")
    for name, descriptor, width in (
        ("q", q_descriptors, 7),
        ("ready", ready_descriptors, 3),
    ):
        if (
            descriptor.dtype != torch.int64
            or not descriptor.is_cuda
            or not descriptor.is_contiguous()
            or descriptor.ndim != 2
            or descriptor.shape[0] < 1
            or int(descriptor.shape[1]) != width
        ):
            raise ValueError(
                f"Qwen3.5 {name} descriptors must be nonempty contiguous "
                f"CUDA int64[:, {width}]"
            )
    if (
        completion_counter.dtype != torch.int32
        or not completion_counter.is_cuda
        or not completion_counter.is_contiguous()
        or completion_counter.numel() != 1
    ):
        raise ValueError("Qwen3.5 completion counter must be one CUDA int32")
    _validate_optional_turn(release_turn, next_turn)
    if slot not in (0, 1) or ready_value <= 0 or eps <= 0:
        raise ValueError(
            "Qwen3.5 publication requires slot 0/1, a positive readiness "
            "ticket, and positive epsilon"
        )
    _jit_module().publish_linear(
        hidden,
        shared_input,
        shared_gate,
        residual_input,
        weight,
        source_offsets,
        q_descriptors,
        ready_descriptors,
        completion_counter,
        release_turn,
        -1 if next_turn is None else int(next_turn),
        int(slot),
        int(ready_value),
        float(eps),
        int(shared is not None),
        int(residual is not None),
    )


def _validate_optional_turn(
    turn: torch.Tensor | None, next_turn: int | None
) -> None:
    if (turn is None) != (next_turn is None):
        raise ValueError("Qwen3.5 publication turn and next value must be paired")
    if turn is None:
        return
    if (
        turn.dtype != torch.int64
        or not turn.is_cuda
        or not turn.is_contiguous()
        or turn.numel() != 1
        or next_turn is None
        or next_turn < 0
    ):
        raise ValueError(
            "Qwen3.5 publication turn must be one contiguous CUDA int64 "
            "with a nonnegative next value"
        )


def qwen35_shared_add_rmsnorm(
    hidden: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    shared_gate: torch.Tensor | None = None,
) -> None:
    """Fold a deferred shared-expert output into the next RMSNorm in place."""

    if (
        hidden.dtype != torch.bfloat16
        or not hidden.is_cuda
        or hidden.ndim != 2
        or hidden.stride(1) != 1
        or int(hidden.shape[0]) < 1
        or int(hidden.shape[1]) % 1024
        or int(hidden.shape[1]) > 8192
    ):
        raise ValueError(
            "Qwen3.5 shared-add/RMSNorm hidden input must be a nonempty "
            "strided CUDA BF16 matrix with hidden size divisible by 1024 "
            "and at most 8192"
        )
    for name, tensor in (("shared", shared), ("residual", residual)):
        if (
            tensor.dtype != hidden.dtype
            or not tensor.is_cuda
            or tensor.shape != hidden.shape
            or tensor.stride(1) != 1
        ):
            raise ValueError(f"Qwen3.5 {name} must match the hidden input")
    if shared_gate is not None and (
        shared_gate.dtype != hidden.dtype
        or not shared_gate.is_cuda
        or tuple(shared_gate.shape) != (int(hidden.shape[0]), 1)
        or not shared_gate.is_contiguous()
    ):
        raise ValueError("Qwen3.5 shared gate must be contiguous CUDA BF16 [rows, 1]")
    if (
        weight.dtype != hidden.dtype
        or not weight.is_cuda
        or not weight.is_contiguous()
        or tuple(weight.shape) != (int(hidden.shape[1]),)
    ):
        raise ValueError("Qwen3.5 shared-add/RMSNorm weight must match hidden width")
    if eps <= 0:
        raise ValueError("Qwen3.5 shared-add/RMSNorm epsilon must be positive")
    _jit_module().shared_add_rmsnorm(
        hidden, shared, shared_gate, residual, weight, float(eps)
    )


__all__ = [
    "ensure_qwen35_qkv_transport_built",
    "publish_qwen35_linear_input",
    "publish_qwen35_qkv",
    "qwen35_shared_add_rmsnorm",
]
