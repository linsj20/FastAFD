from __future__ import annotations

import functools

import torch

from .utils import load_jit, make_cpp_args


@functools.cache
def _jit_module(head_dim: int):
    args = make_cpp_args(head_dim)
    return load_jit(
        "afd_fmha_transport",
        *args,
        cuda_files=["afd_fmha_transport.cu"],
        cuda_wrappers=[
            (name, f"AfdFmhaTransportKernel<{args}>::{name}")
            for name in (
                "publish_qkv", "publish_o", "quantize_publish_o_fp8",
                "wait_turn", "release_turn", "wait_ready", "wait_ready_turn",
            )
        ],
    )


def ensure_afd_fmha_transport_built(head_dim: int) -> None:
    _jit_module(int(head_dim))


def publish_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    source_offsets: torch.Tensor,
    q_descriptors: torch.Tensor,
    kv_descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    layer: int,
    slot: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_bf16_rows("q", q, head_dim)
    _validate_bf16_rows("k", k, head_dim)
    _validate_bf16_rows("v", v, head_dim)
    _validate_source_offsets(source_offsets)
    _validate_completion_counter(completion_counter)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_qkv(
        q,
        k,
        v,
        out_loc,
        source_offsets,
        q_descriptors,
        kv_descriptors,
        ready_descriptors,
        completion_counter,
        int(layer),
        int(slot),
        int(ready_value),
    )


def publish_o(
    o: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    slot: int,
    destination_source_stride: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_bf16_rows("o", o, head_dim)
    gate_input = o if gate is None else gate
    _validate_bf16_rows("gate", gate_input, head_dim)
    if gate_input.shape != o.shape:
        raise ValueError(
            f"AFD O gate shape {tuple(gate_input.shape)} does not match "
            f"payload shape {tuple(o.shape)}"
        )
    _validate_completion_counter(completion_counter)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_o(
        o,
        gate_input,
        descriptors,
        ready_descriptors,
        completion_counter,
        int(slot),
        int(destination_source_stride),
        int(ready_value),
        int(gate is not None),
    )


def quantize_publish_o_fp8(
    o: torch.Tensor,
    staged_o: torch.Tensor,
    staged_scales: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    slot: int,
    destination_source_stride: int,
    head_dim: int,
    ready_value: int,
) -> None:
    """Quantize O locally and publish it remotely in one kernel launch."""
    _validate_bf16_rows("o", o, head_dim)
    _validate_fp8_o_quant_shape(o, head_dim)
    _validate_fp8_staging(o, staged_o, staged_scales, head_dim)
    _validate_completion_counter(completion_counter)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).quantize_publish_o_fp8(
        o,
        staged_o,
        staged_scales,
        descriptors,
        ready_descriptors,
        completion_counter,
        int(slot),
        int(destination_source_stride),
        int(ready_value),
    )


def wait_ready(
    ready: torch.Tensor,
    timeout_record: torch.Tensor,
    *,
    timeout_ms: int,
    head_dim: int,
    expected_ready: int,
    turn: torch.Tensor | None = None,
    expected_turn: int | None = None,
) -> None:
    _validate_ready("AFD readiness", ready)
    _validate_ready_value(expected_ready)
    module = _jit_module(int(head_dim))
    if turn is None and expected_turn is None:
        module.wait_ready(
            int(ready.data_ptr()),
            int(ready.numel()),
            timeout_record,
            int(timeout_ms),
            int(expected_ready),
        )
        return
    _validate_turn(turn, expected_turn)
    module.wait_ready_turn(
        int(ready.data_ptr()),
        int(ready.numel()),
        timeout_record,
        turn,
        int(timeout_ms),
        int(expected_turn),
        int(expected_ready),
    )


def wait_turn(
    turn: torch.Tensor,
    timeout_record: torch.Tensor,
    *,
    timeout_ms: int,
    expected_turn: int,
    next_turn: int,
    head_dim: int,
) -> None:
    _validate_turn(turn, expected_turn)
    _validate_turn(turn, next_turn)
    _jit_module(int(head_dim)).wait_turn(
        turn,
        timeout_record,
        int(timeout_ms),
        int(expected_turn),
        int(next_turn),
    )


def release_turn(
    turn: torch.Tensor,
    *,
    next_turn: int,
    head_dim: int,
) -> None:
    """Release the peer compute lane before enqueueing publication work."""
    _validate_turn(turn, next_turn)
    _jit_module(int(head_dim)).release_turn(turn, int(next_turn))


def _validate_turn(turn: torch.Tensor | None, value: int | None) -> None:
    if (
        turn is None
        or value is None
        or turn.dtype != torch.int64
        or not turn.is_cuda
        or not turn.is_contiguous()
        or turn.numel() != 1
        or int(value) < 0
    ):
        raise ValueError(
            "AFD attention turn requires one contiguous CUDA int64 and a "
            f"non-negative value, got turn={turn} value={value}"
        )


def _validate_ready(name: str, ready: torch.Tensor) -> None:
    if (
        ready.dtype != torch.int64
        or not ready.is_cuda
        or not ready.is_contiguous()
        or ready.ndim != 1
        or ready.numel() < 1
    ):
        raise ValueError(f"{name} must be a nonempty contiguous CUDA int64 vector")


def _validate_ready_value(value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"AFD readiness ticket must be positive, got {value}")


def _validate_completion_counter(counter: torch.Tensor) -> None:
    if (
        counter.dtype != torch.int32
        or not counter.is_cuda
        or not counter.is_contiguous()
        or counter.numel() != 1
    ):
        raise ValueError(
            "AFD publication completion counter must be one contiguous CUDA int32"
        )


def _validate_source_offsets(offsets: torch.Tensor) -> None:
    if (
        offsets.dtype != torch.int64
        or not offsets.is_cuda
        or not offsets.is_contiguous()
        or offsets.ndim != 1
        or offsets.numel() < 2
    ):
        raise ValueError(
            "AFD fan-in source offsets must be a contiguous CUDA int64 vector"
        )


def _validate_fp8_rows(name: str, tensor: torch.Tensor, head_dim: int) -> None:
    if (
        tensor.dtype != torch.float8_e4m3fn
        or not tensor.is_cuda
        or tensor.ndim != 2
        or tensor.shape[1] % int(head_dim)
        or tensor.stride(1) != 1
    ):
        raise ValueError(
            f"AFD {name} must be a rank-2 CUDA FP8 E4M3 head matrix with "
            f"head_dim={head_dim}, got shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} device={tensor.device} stride={tensor.stride()}"
        )


def _validate_fp8_o_quant_shape(o: torch.Tensor, head_dim: int) -> None:
    if int(head_dim) % 128 or o.shape[1] % 512:
        raise ValueError(
            "Fused AFD O quantization requires 128-element UE8M0 groups and "
            f"four-group alignment, got width={o.shape[1]} head_dim={head_dim}"
        )


def _validate_fp8_staging(
    o: torch.Tensor,
    staged_o: torch.Tensor,
    staged_scales: torch.Tensor,
    head_dim: int,
) -> None:
    _validate_fp8_rows("staged_o", staged_o, head_dim)
    scale_groups = int(o.shape[1]) // 128
    if tuple(staged_o.shape) != tuple(o.shape):
        raise ValueError(
            "AFD staged FP8 O must match the BF16 O shape: "
            f"o={tuple(o.shape)} staged_o={tuple(staged_o.shape)}"
        )
    if (
        staged_scales.dtype != torch.uint8
        or not staged_scales.is_cuda
        or not staged_scales.is_contiguous()
        or tuple(staged_scales.shape) != (int(o.shape[0]), scale_groups)
    ):
        raise ValueError(
            "AFD staged O scales must be contiguous CUDA uint8 with one UE8M0 "
            f"byte per row and 128 elements, got shape={tuple(staged_scales.shape)} "
            f"dtype={staged_scales.dtype} device={staged_scales.device} "
            f"stride={staged_scales.stride()}"
        )


def _validate_bf16_rows(name: str, tensor: torch.Tensor, head_dim: int) -> None:
    if (
        tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
        or tensor.ndim != 2
        or tensor.shape[1] % int(head_dim)
        or tensor.stride(1) != 1
    ):
        raise ValueError(
            f"AFD {name} must be a rank-2 CUDA BF16 head matrix with "
            f"head_dim={head_dim}, got shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} device={tensor.device} stride={tensor.stride()}"
        )


__all__ = [
    "ensure_afd_fmha_transport_built",
    "publish_o",
    "quantize_publish_o_fp8",
    "publish_qkv",
    "release_turn",
    "wait_ready",
    "wait_turn",
]
