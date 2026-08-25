from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """Concrete paged MHA key-value cache.

    Stores keys and values in a single contiguous
    [2, num_layers, num_pages, page_size, local_kv_heads, head_dim] buffer.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        storage: torch.Tensor | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        storage_shape = (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
        if storage is None:
            storage = torch.empty(storage_shape, device=device, dtype=dtype)
        elif (
            tuple(storage.shape) != storage_shape
            or storage.device != device
            or storage.dtype != dtype
            or not storage.is_contiguous()
        ):
            raise ValueError(
                "MHA KV backing storage does not match the required layout: "
                f"required={storage_shape}/{dtype}/{device} "
                f"actual={tuple(storage.shape)}/{storage.dtype}/{storage.device} "
                f"contiguous={storage.is_contiguous()}"
            )
        self._kv_buffer = storage
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype
