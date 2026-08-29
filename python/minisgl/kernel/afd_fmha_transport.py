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
            ("publish_qkv", f"AfdFmhaTransportKernel<{args}>::publish_qkv"),
            ("publish_o", f"AfdFmhaTransportKernel<{args}>::publish_o"),
            (
                "publish_o_release_turn",
                f"AfdFmhaTransportKernel<{args}>::publish_o_release_turn",
            ),
            ("publish_o_fp8", f"AfdFmhaTransportKernel<{args}>::publish_o_fp8"),
            (
                "publish_o_fp8_release_turn",
                f"AfdFmhaTransportKernel<{args}>::publish_o_fp8_release_turn",
            ),
            (
                "quantize_publish_o_fp8",
                f"AfdFmhaTransportKernel<{args}>::quantize_publish_o_fp8",
            ),
            (
                "quantize_publish_o_fp8_release_turn",
                f"AfdFmhaTransportKernel<{args}>::quantize_publish_o_fp8_release_turn",
            ),
            ("wait_turn", f"AfdFmhaTransportKernel<{args}>::wait_turn"),
            ("wait_ready", f"AfdFmhaTransportKernel<{args}>::wait_ready"),
            ("wait_ready_turn", f"AfdFmhaTransportKernel<{args}>::wait_ready_turn"),
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
    slot: int,
    destination_source_stride: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_bf16_rows("o", o, head_dim)
    _validate_completion_counter(completion_counter)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_o(
        o,
        descriptors,
        ready_descriptors,
        completion_counter,
        int(slot),
        int(destination_source_stride),
        int(ready_value),
    )


def publish_o_release_turn(
    o: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    turn: torch.Tensor,
    *,
    slot: int,
    destination_source_stride: int,
    next_turn: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_bf16_rows("o", o, head_dim)
    _validate_completion_counter(completion_counter)
    _validate_turn(turn, next_turn)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_o_release_turn(
        o,
        descriptors,
        ready_descriptors,
        completion_counter,
        turn,
        int(slot),
        int(destination_source_stride),
        int(next_turn),
        int(ready_value),
    )


def publish_o_fp8(
    o: torch.Tensor,
    scales: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    *,
    slot: int,
    destination_source_stride: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_fp8_rows("o", o, head_dim)
    _validate_packed_scales(scales, o.shape[0], o.shape[1], head_dim)
    _validate_completion_counter(completion_counter)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_o_fp8(
        o,
        scales,
        descriptors,
        ready_descriptors,
        completion_counter,
        int(slot),
        int(destination_source_stride),
        int(ready_value),
    )


def publish_o_fp8_release_turn(
    o: torch.Tensor,
    scales: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    turn: torch.Tensor,
    *,
    slot: int,
    destination_source_stride: int,
    next_turn: int,
    head_dim: int,
    ready_value: int,
) -> None:
    _validate_fp8_rows("o", o, head_dim)
    _validate_packed_scales(scales, o.shape[0], o.shape[1], head_dim)
    _validate_completion_counter(completion_counter)
    _validate_turn(turn, next_turn)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).publish_o_fp8_release_turn(
        o,
        scales,
        descriptors,
        ready_descriptors,
        completion_counter,
        turn,
        int(slot),
        int(destination_source_stride),
        int(next_turn),
        int(ready_value),
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


def quantize_publish_o_fp8_release_turn(
    o: torch.Tensor,
    staged_o: torch.Tensor,
    staged_scales: torch.Tensor,
    descriptors: torch.Tensor,
    ready_descriptors: torch.Tensor,
    completion_counter: torch.Tensor,
    quantization_counter: torch.Tensor,
    turn: torch.Tensor,
    *,
    slot: int,
    destination_source_stride: int,
    next_turn: int,
    head_dim: int,
    ready_value: int,
) -> None:
    """Fuse quantize/publish while releasing ownership after quantization."""
    _validate_bf16_rows("o", o, head_dim)
    _validate_fp8_o_quant_shape(o, head_dim)
    _validate_fp8_staging(o, staged_o, staged_scales, head_dim)
    _validate_completion_counter(completion_counter)
    _validate_completion_counter(quantization_counter)
    _validate_turn(turn, next_turn)
    _validate_ready_value(ready_value)
    _jit_module(int(head_dim)).quantize_publish_o_fp8_release_turn(
        o,
        staged_o,
        staged_scales,
        descriptors,
        ready_descriptors,
        completion_counter,
        quantization_counter,
        turn,
        int(slot),
        int(destination_source_stride),
        int(next_turn),
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


def _validate_packed_scales(
    scales: torch.Tensor, rows: int, width: int, head_dim: int
) -> None:
    groups = (int(width) + int(head_dim) - 1) // int(head_dim)
    packed_groups = (groups + 3) // 4
    expected_stride = ((int(rows) + 3) // 4) * 4
    if (
        scales.dtype != torch.int32
        or not scales.is_cuda
        or scales.ndim != 2
        or tuple(scales.shape) != (int(rows), packed_groups)
        or scales.stride(0) != 1
        or scales.stride(1) != expected_stride
    ):
        raise ValueError(
            "AFD O scales must use DeepGEMM packed UE8M0 layout: "
            f"shape={tuple(scales.shape)} stride={scales.stride()} "
            f"expected_shape={(int(rows), packed_groups)} "
            f"expected_stride={(1, expected_stride)} dtype={scales.dtype} "
            f"device={scales.device}"
        )


def _validate_fp8_o_quant_shape(o: torch.Tensor, head_dim: int) -> None:
    if int(head_dim) != 128 or o.shape[1] % (4 * int(head_dim)):
        raise ValueError(
            "Fused AFD O quantization requires 128-wide heads and the existing "
            f"four-head alignment, got width={o.shape[1]} head_dim={head_dim}"
        )


def _validate_fp8_staging(
    o: torch.Tensor,
    staged_o: torch.Tensor,
    staged_scales: torch.Tensor,
    head_dim: int,
) -> None:
    _validate_fp8_rows("staged_o", staged_o, head_dim)
    heads = int(o.shape[1]) // int(head_dim)
    if tuple(staged_o.shape) != tuple(o.shape):
        raise ValueError(
            "AFD staged FP8 O must match the BF16 O shape: "
            f"o={tuple(o.shape)} staged_o={tuple(staged_o.shape)}"
        )
    if (
        staged_scales.dtype != torch.uint8
        or not staged_scales.is_cuda
        or not staged_scales.is_contiguous()
        or tuple(staged_scales.shape) != (int(o.shape[0]), heads)
    ):
        raise ValueError(
            "AFD staged O scales must be contiguous CUDA uint8 with one UE8M0 "
            f"byte per row and head, got shape={tuple(staged_scales.shape)} "
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
    "publish_o_release_turn",
    "publish_o_fp8",
    "publish_o_fp8_release_turn",
    "quantize_publish_o_fp8",
    "quantize_publish_o_fp8_release_turn",
    "publish_qkv",
    "wait_ready",
    "wait_turn",
]
