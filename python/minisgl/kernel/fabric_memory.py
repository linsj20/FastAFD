from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Sequence

import torch

from .deepep_moe import _load_extension


@dataclass(frozen=True)
class FabricTensor:
    """CUDA tensor plus the physical owner or imported fabric mapping."""

    tensor: torch.Tensor
    allocation: Any
    byte_offset: int = 0

    @property
    def handle(self) -> bytes:
        return bytes(self.allocation.export_handle())


def allocate_fabric_tensor(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
) -> FabricTensor:
    return allocate_fabric_tensors(((shape, dtype),))[0]


def allocate_fabric_tensors(
    specs: Sequence[tuple[Sequence[int], torch.dtype]],
) -> tuple[FabricTensor, ...]:
    """Co-allocate typed views so one exported handle covers one publication arena."""
    if not specs:
        raise ValueError("fabric allocation requires at least one tensor view")
    layouts: list[tuple[Sequence[int], torch.dtype, int, int]] = []
    next_offset = 0
    for shape, dtype in specs:
        num_bytes = _required_bytes(shape, dtype)
        alignment = max(16, torch.empty((), dtype=dtype).element_size())
        next_offset = _align_up(next_offset, alignment)
        layouts.append((shape, dtype, next_offset, num_bytes))
        next_offset += num_bytes
    allocation = _load_extension().FabricAllocation.allocate(next_offset)
    storage = allocation.tensor()
    return tuple(
        FabricTensor(
            tensor=_typed_view(
                storage,
                shape,
                dtype,
                num_bytes,
                byte_offset=byte_offset,
            ),
            allocation=allocation,
            byte_offset=byte_offset,
        )
        for shape, dtype, byte_offset, num_bytes in layouts
    )


def import_fabric_tensor(
    handle: bytes,
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    byte_offset: int = 0,
    allocation: Any | None = None,
) -> FabricTensor:
    num_bytes = _required_bytes(shape, dtype)
    byte_offset = int(byte_offset)
    element_size = torch.empty((), dtype=dtype).element_size()
    if byte_offset < 0 or byte_offset % element_size:
        raise ValueError(
            f"fabric view offset must be nonnegative and {element_size}-byte aligned: "
            f"offset={byte_offset}"
        )
    if allocation is None:
        allocation = _load_extension().FabricAllocation.import_handle(handle)
    if int(allocation.size) < byte_offset + num_bytes:
        raise ValueError(
            "fabric mapping is smaller than the requested tensor: "
            f"mapping_bytes={int(allocation.size)} offset={byte_offset} "
            f"required_bytes={num_bytes}"
        )
    return FabricTensor(
        tensor=_typed_view(
            allocation.tensor(),
            shape,
            dtype,
            num_bytes,
            byte_offset=byte_offset,
        ),
        allocation=allocation,
        byte_offset=byte_offset,
    )


def _required_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim < 1 for dim in normalized):
        raise ValueError(f"fabric tensor shape must contain positive dimensions: {normalized}")
    return prod(normalized) * torch.empty((), dtype=dtype).element_size()


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _typed_view(
    storage: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    num_bytes: int,
    *,
    byte_offset: int = 0,
) -> torch.Tensor:
    return (
        storage.narrow(0, int(byte_offset), num_bytes)
        .view(dtype)
        .view(tuple(int(dim) for dim in shape))
    )


__all__ = [
    "FabricTensor",
    "allocate_fabric_tensor",
    "allocate_fabric_tensors",
    "import_fabric_tensor",
]
