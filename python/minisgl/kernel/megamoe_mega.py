"""Same-rank MegaMoE wrapper backed by symmetric-memory communication."""

from __future__ import annotations

import os
import types
from typing import Any

import torch
import torch.distributed as dist

from . import deepgemm as _dg
from .megamoe_m2n_mega import align, get_token_alignment


def _ext():
    return _dg._load_extension()


class MegaMoESymmBuffer:
    """Symmetric storage with communication protocols fixed at allocation.

    ``payload_mode`` is ``"pull"`` (default) or ``"compact_push"``. Both use
    pushed metadata with prepared rank-ready routes. The compact mode pushes
    one FP8/scale/weight frame per source token and destination rank, then
    gathers expert rows during compute. Pull materializes contiguous expert rows.
    An explicit mode overrides ``MINISGL_MEGAMOE_PAYLOAD_MODE``.
    """

    def __init__(
        self,
        group: Any,
        *,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        inband_combine: bool | None = None,
        payload_mode: str | None = None,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.rank = int(dist.get_rank(group=group))
        self.num_ranks = int(dist.get_world_size(group))
        self.num_experts = int(num_experts)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.num_max_tokens_per_rank = align(
            num_max_tokens_per_rank, get_token_alignment()
        )
        # Pin the protocol to this allocation; launch must use the same arena layout.
        # The environment override remains for standalone/reference callers.
        if inband_combine is None:
            setting = os.environ.get("DG_MEGAMOE_LAMPORT_COMBINE", "0")
            if setting not in ("0", "1"):
                raise ValueError("DG_MEGAMOE_LAMPORT_COMBINE must be 0 or 1")
            inband_combine = setting == "1"
        if not isinstance(inband_combine, bool):
            raise TypeError("inband_combine must be a bool")
        self.inband_combine = inband_combine
        if payload_mode is None:
            payload_mode = os.environ.get("MINISGL_MEGAMOE_PAYLOAD_MODE", "pull")
        if payload_mode not in ("pull", "compact_push"):
            raise ValueError("payload_mode must be 'pull' or 'compact_push'")
        self.payload_mode = payload_mode
        self.compact_push = payload_mode == "compact_push"
        if self.num_experts % self.num_ranks:
            raise RuntimeError(
                "MegaMoE experts must divide the symmetric rank group: "
                f"experts={self.num_experts} ranks={self.num_ranks}"
            )

        num_bytes, slice_input_buffers = _ext().get_symm_buffer_size_for_mega_moe(
            self.num_ranks,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            True,
            "swiglu",
            self.inband_combine,
            self.compact_push,
        )
        allocator = torch if self.num_ranks == 1 else symm_mem
        self.buffer = allocator.empty(int(num_bytes), dtype=torch.int8, device="cuda")
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if self.num_ranks == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer_ptrs_device = torch.tensor(
            self.handle.buffer_ptrs,
            dtype=torch.int64,
            device=self.buffer.device,
        )
        self.buffer.zero_()
        input_views = slice_input_buffers(self.buffer)
        if self.inband_combine:
            combine_bytes = 2 * self.num_topk * self.num_max_tokens_per_rank * self.hidden * 2
            self.buffer[-combine_bytes:].view(torch.int32).fill_(-(1 << 31))
        torch.cuda.synchronize()
        # No peer may publish until every destination's initialization finished.
        dist.barrier(group=group)

        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = input_views
        self.y = torch.empty(
            (self.num_max_tokens_per_rank, self.hidden),
            dtype=torch.bfloat16,
            device=self.buffer.device,
        )

    @property
    def buffer_ptrs(self) -> list[int]:
        return list(self.handle.buffer_ptrs)

    def route_prepare_args(
        self, *, rank_ready: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, bool, bool]:
        if self.compact_push and not rank_ready:
            raise ValueError("compact payload push requires rank-ready metadata")
        return (
            self.buffer,
            self.buffer_ptrs_device,
            self.rank,
            self.num_ranks,
            self.num_max_tokens_per_rank,
            int(_dg.get_num_sms()),
            bool(rank_ready),
            self.compact_push,
        )


def fp8_fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    *,
    debug_timings: torch.Tensor | None = None,
    routes_prepared: bool = False,
    rank_gated_combine: bool = False,
    route_ready_dispatch: bool = False,
    rank_ready_route_publish: bool = False,
) -> None:
    """Run one fused FP8-activation/FP8-weight MegaMoE kernel."""
    _ext().fp8_fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        None,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs,
        sym_buffer.rank,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        (1, 1, 32),
        "swiglu",
        None,
        True,
        debug_timings,
        bool(routes_prepared),
        bool(rank_gated_combine),
        bool(route_ready_dispatch),
        bool(rank_ready_route_publish),
        sym_buffer.inband_combine,
        sym_buffer.compact_push,
    )


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: MegaMoESymmBuffer,
    *,
    debug_timings: torch.Tensor | None = None,
    routes_prepared: bool = False,
    rank_gated_combine: bool = False,
    route_ready_dispatch: bool = False,
    rank_ready_route_publish: bool = False,
) -> None:
    """Run one fused FP8-activation/MXFP4-weight MegaMoE kernel."""
    _ext().fp8_fp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        None,
        sym_buffer.buffer,
        sym_buffer.buffer_ptrs,
        sym_buffer.rank,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        (1, 1, 32),
        "swiglu",
        None,
        True,
        debug_timings,
        bool(routes_prepared),
        bool(rank_gated_combine),
        bool(route_ready_dispatch),
        bool(rank_ready_route_publish),
        sym_buffer.inband_combine,
        sym_buffer.compact_push,
    )


__all__ = ["MegaMoESymmBuffer", "fp8_fp4_mega_moe", "fp8_fp8_mega_moe"]
