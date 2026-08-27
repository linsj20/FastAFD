"""Same-rank FP8-activation MegaMoE adapter for FMHA-only model workers."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from minisgl.kernel import megamoe_mega as _mega
from minisgl.kernel import megamoe_m2n_mega as _weights
from minisgl.kernel.moe_topk import gate_topk


class MegaMoEAfdAdapter:
    """Route and run one fused MegaMoE kernel on the FFN rank set."""

    backend = "megamoe"

    def __init__(
        self,
        *,
        group: Any,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        num_max_tokens_per_rank: int,
        num_lanes: int,
        tp_size: int,
        tp_rank: int,
        gate_renormalize: bool,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("MegaMoEAfdAdapter requires torch.distributed")
        if hidden_size % 512 or intermediate_size % 512:
            raise RuntimeError(
                "megamoe requires hidden and MoE intermediate dimensions divisible "
                f"by 512, got hidden={hidden_size} intermediate={intermediate_size}"
            )
        self.group = group
        self.rank = int(dist.get_rank(group=group))
        self.group_size = int(dist.get_world_size(group))
        self.real_num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.num_experts = self.group_size * self.num_local_experts
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.bucket = int(num_max_tokens_per_rank)
        self.num_lanes = max(1, int(num_lanes))
        self.gate_renormalize = bool(gate_renormalize)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.expert_weight_dtype = os.environ.get(
            "MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp8"
        ).strip().lower()
        if self.expert_weight_dtype not in ("fp8", "fp4"):
            raise RuntimeError(
                "MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE must be fp8 or fp4, "
                f"got {self.expert_weight_dtype!r}"
            )
        self.precision = f"fp8_{self.expert_weight_dtype}"
        self._run_mega_moe = (
            _mega.fp8_fp8_mega_moe
            if self.expert_weight_dtype == "fp8"
            else _mega.fp8_fp4_mega_moe
        )
        if self.group_size * self.tp_size * self.num_local_experts < self.real_num_experts:
            raise RuntimeError(
                "MegaMoE rank group does not cover the model experts: "
                f"rank_group={self.group_size} tp={self.tp_size} "
                f"local_experts={self.num_local_experts} "
                f"model_experts={self.real_num_experts}"
            )

        expert_map = torch.full(
            (self.real_num_experts,), -1, dtype=torch.int64, device="cuda"
        )
        for dp_rank in range(self.group_size):
            global_ep_rank = dp_rank * self.tp_size + self.tp_rank
            global_start = global_ep_rank * self.num_local_experts
            global_end = min(
                self.real_num_experts, global_start + self.num_local_experts
            )
            if global_start < global_end:
                group_start = dp_rank * self.num_local_experts
                expert_map[global_start:global_end] = torch.arange(
                    group_start,
                    group_start + global_end - global_start,
                    dtype=torch.int64,
                    device=expert_map.device,
                )
        self.expert_map = expert_map
        self.buffers: list[_mega.MegaMoESymmBuffer] = []
        self.weights: dict[
            int,
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}

    def register_layer_weights(self, layer_id: int, experts: Any) -> None:
        """Requantize Qwen FP8 block weights to the selected MegaMoE layout."""
        if getattr(experts, "_fp8_scale_format", None) != "block":
            raise RuntimeError("megamoe requires block-scaled FP8 expert weights")
        block_shape = tuple(
            getattr(getattr(experts, "quant", None), "weight_block_size", ())
        )
        if block_shape != (128, 128):
            raise RuntimeError(
                "megamoe requires Qwen 128x128 FP8 block scales, "
                f"got block_shape={block_shape}"
            )
        if self.expert_weight_dtype == "fp8":
            l1 = _weights.requant_qwen_fp8_weights_per32(
                experts.gate_up_proj, experts.gate_up_proj_scale, inplace=True
            )
            l2 = _weights.requant_qwen_fp8_weights_per32(
                experts.down_proj, experts.down_proj_scale, inplace=True
            )
        else:
            l1 = _weights.requant_qwen_fp8_weights_to_fp4(
                experts.gate_up_proj, experts.gate_up_proj_scale
            )
            l2 = _weights.requant_qwen_fp8_weights_to_fp4(
                experts.down_proj, experts.down_proj_scale
            )
        self.weights[int(layer_id)] = _weights.transform_weights_for_mega_moe(
            l1, l2
        )

    def allocate_buffers(self) -> None:
        if self.buffers:
            return
        self.buffers = [
            _mega.MegaMoESymmBuffer(
                self.group,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=self.bucket,
                num_topk=self.top_k,
                hidden=self.hidden_size,
                intermediate_hidden=self.intermediate_size,
            )
            for _ in range(self.num_lanes)
        ]

    def forward(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        *,
        lane: int,
        num_token_non_padded: int | torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = hidden_states.contiguous().view(-1, self.hidden_size)
        num_tokens = int(hidden.shape[0])
        if num_tokens > self.bucket:
            raise RuntimeError(
                f"MegaMoE tokens={num_tokens} exceeds dispatch bucket={self.bucket}"
            )
        try:
            l1_weights, l2_weights = self.weights[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"layer {int(layer_id)} has no registered MegaMoE weights"
            ) from exc
        buffer = self.buffers[int(lane) % self.num_lanes]
        ids = buffer.topk_idx[:num_tokens]
        weights = buffer.topk_weights[:num_tokens]
        gate_topk(
            hidden,
            gate_weight,
            self.top_k,
            renormalize=self.gate_renormalize,
            expert_map=self.expert_map,
            num_token_non_padded=num_token_non_padded,
            topk_idx_dtype=torch.int64,
            quant_out=(buffer.x[:num_tokens], buffer.x_sf[:num_tokens], None),
            out=(ids, weights),
        )
        y = buffer.y[:num_tokens]
        self._run_mega_moe(y, l1_weights, l2_weights, buffer)
        return y.view_as(hidden_states)


__all__ = ["MegaMoEAfdAdapter"]
