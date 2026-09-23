"""FP8-activation MegaMoE adapter for FFN ranks in the AFD runtime."""

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
    trace_root = None
    # Fixed FFN-rank source + padded materialization row budget for model decode.
    # Sources already aggregate this FFN rank's attention-rank fan-in.
    # Prompt length only sizes the separate pull-prefill allocation.
    all_push_capacity = 1024

    def __init__(
        self,
        *,
        group: Any,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        top_k: int,
        num_max_tokens_per_rank: int,
        num_lanes: int,
        tp_size: int,
        tp_rank: int,
        gate_renormalize: bool,
        defer_route_dependent_launch: bool = False,
        num_max_decode_tokens_per_rank: int = 1024,
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
        self.num_layers = int(num_layers)
        self.top_k = int(top_k)
        self.bucket = int(num_max_tokens_per_rank)
        # Reset every source row any captured decode shape can address. This
        # bound is fixed for the adapter; it never changes the allocation size.
        self.all_push_reset_rows = int(num_max_decode_tokens_per_rank)
        self.all_push_capacity = int(os.environ.get("MINISGL_MEGAMOE_ALL_PUSH_CAPACITY", "1024"))
        if self.all_push_capacity < 1024:
            raise ValueError("MegaMoE all-push capacity must be at least 1024 rows")
        self.num_lanes = max(1, int(num_lanes))
        self.gate_renormalize = bool(gate_renormalize)
        self.defer_route_dependent_launch = bool(defer_route_dependent_launch)
        perfect_routing = os.environ.get(
            "MINISGL_AFD_PERFECT_ROUTING", "0"
        ).strip()
        if perfect_routing not in ("0", "1"):
            raise RuntimeError(
                "MINISGL_AFD_PERFECT_ROUTING must be 0 or 1, "
                f"got {perfect_routing!r}"
            )
        self.perfect_routing = perfect_routing == "1"
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.expert_weight_dtype = os.environ.get(
            "MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE", "fp4"
        ).strip().lower()
        if self.expert_weight_dtype not in ("fp8", "fp4"):
            raise RuntimeError(
                "MegaMoE requires MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE "
                "to be fp8 or fp4, "
                f"got {self.expert_weight_dtype!r}"
            )
        if self.expert_weight_dtype == "fp8":
            self.precision = "fp8_fp8"
            self._run_mega_moe = _mega.fp8_fp8_mega_moe
        else:
            self.precision = "fp8_fp4"
            self._run_mega_moe = _mega.fp8_fp4_mega_moe
        weight_path = os.environ.get("MINISGL_MEGAMOE_EXPERT_WEIGHT_PATH", "").strip()
        if weight_path and self.expert_weight_dtype != "fp4":
            raise RuntimeError(
                "a preconverted MegaMoE expert checkpoint requires fp4 weights"
            )
        self._weight_loader = (
            _weights.Mxfp4ExpertWeightLoader(
                weight_path,
                torch.device("cuda", torch.cuda.current_device()),
                num_layers=self.num_layers,
                num_experts=self.real_num_experts,
                hidden=self.hidden_size,
                intermediate_hidden=self.intermediate_size,
            )
            if weight_path
            else None
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
        self.alternate_buffers: list[_mega.MegaMoESymmBuffer] = []
        self._output_buffers: dict[int, _mega.MegaMoESymmBuffer] = {}
        ahead = os.environ.get("MINISGL_MEGAMOE_RECYCLE_AHEAD", "0")
        if ahead not in ("0", "1"):
            raise ValueError("MINISGL_MEGAMOE_RECYCLE_AHEAD must be 0 or 1")
        self.recycle_ahead = ahead == "1"
        self.trace_root = os.environ.get("MINISGL_MEGAMOE_TRACE_ROOT")
        self._trace_timings = {}
        self._trace_output = {}
        self.prefill_buffers: list[_mega.MegaMoESymmBuffer] = []
        self.weights: dict[
            int,
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor],
            ],
        ] = {}

    def register_layer_weights(self, layer_id: int, experts: Any) -> None:
        """Requantize Qwen FP8 block weights to the selected MegaMoE layout."""
        if self._weight_loader is not None:
            self.weights[int(layer_id)] = self._weight_loader.load_layer(
                layer_id,
                expert_offset=int(experts.local_expert_offset),
                expert_count=self.num_local_experts,
            )
            return
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
        mode = os.environ.get("MINISGL_MEGAMOE_PAYLOAD_MODE", "pull")
        if mode == "all_push":
            if not 0 < self.all_push_reset_rows <= self.all_push_capacity:
                raise ValueError("all-push maximum decode rows exceed the configured capacity")
            if self.expert_weight_dtype != "fp4":
                raise ValueError("Model all-push recycling requires FP4 expert weights")
        def allocate(payload_mode=mode):
            return _mega.MegaMoESymmBuffer(
                self.group,
                num_experts=self.num_experts,
                num_max_tokens_per_rank=(self.all_push_capacity
                                         if payload_mode == "all_push" else self.bucket),
                num_topk=self.top_k,
                hidden=self.hidden_size,
                intermediate_hidden=self.intermediate_size,
                inband_combine=True,
                payload_mode=payload_mode,
            )
        self.buffers = [allocate() for _ in range(self.num_lanes)]
        if self.buffers[0].all_push:
            self.alternate_buffers = [allocate() for _ in range(self.num_lanes)]
            self.prefill_buffers = [allocate("pull") for _ in range(self.num_lanes)]
            # One-time setup only. Runtime reuse is a destination-local epoch
            # acquire, and retiring MegaMoE CTAs reset the next internal bank.
            for pair in zip(self.buffers, self.alternate_buffers):
                for buffer in pair:
                    buffer.reset_dispatch_payload(self.all_push_capacity)
                    if self.recycle_ahead:
                        # Both banks were cleared above. A monotonic ready word
                        # certifies generations 0 and 1 before any rank publishes.
                        # Offset matches Workspace::get_payload_ready_ptr().
                        offset = 32 + (1 + buffer.num_ranks) * 8
                        buffer.buffer[offset:offset + 4].view(torch.int32).fill_(1)
                    buffer.handle.barrier(channel=0)
                    self._output_buffers[buffer.y.data_ptr()] = buffer
            torch.cuda.synchronize()
        print(
            f"MEGAMOE_AFD_PROTOCOL rank={self.rank} ranks={self.group_size} "
            f"payload_mode={self.buffers[0].payload_mode} "
            f"lanes={self.num_lanes} buffers_per_lane={1 + bool(self.alternate_buffers)} "
            f"capacity={self.buffers[0].num_max_tokens_per_rank} "
            f"decode_reset_rows={self.all_push_reset_rows} "
            f"prefill_capacity={self.bucket} prefill_payload_mode=pull "
            f"precision={self.precision}",
            flush=True,
        )

    def forward(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        *,
        lane: int,
        num_token_non_padded: int | torch.Tensor | None,
        phase: str = "decode",
        debug_timings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden_states.contiguous().view(-1, self.hidden_size)
        num_tokens = int(hidden.shape[0])
        try:
            l1_weights, l2_weights = self.weights[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"layer {int(layer_id)} has no registered MegaMoE weights"
            ) from exc
        lane = int(lane) % self.num_lanes
        if phase not in ("decode", "prefill"):
            raise ValueError("MegaMoE phase must be decode or prefill")
        bank = int(layer_id) % 2
        buffer = (self.prefill_buffers[lane] if phase == "prefill" and self.prefill_buffers
                  else self.buffers[lane])
        capacity = self.all_push_reset_rows if buffer.all_push else self.bucket
        if num_tokens > capacity:
            raise RuntimeError(
                f"MegaMoE {phase} tokens={num_tokens} exceeds row capacity={capacity}"
            )
        if buffer.all_push:
            if num_tokens == 0:
                raise RuntimeError("all-push requires a physical row; pad empty ranks")
            if bank:
                buffer = self.alternate_buffers[lane]
        if self.trace_root and buffer.all_push:
            from minisgl.kernel import deepgemm

            key = (int(layer_id), lane)
            if key not in self._trace_timings:
                self._trace_timings[key] = torch.empty(
                    24 + 3 * self.all_push_reset_rows + deepgemm.get_num_sms() + 16,
                    dtype=torch.int64, device=hidden.device,
                )
            debug_timings = self._trace_timings[key]
            debug_timings.zero_()
            self._trace_output[buffer.y.data_ptr()] = debug_timings
        ids = buffer.topk_idx[:num_tokens]
        weights = buffer.topk_weights[:num_tokens]
        routes_prepared = num_tokens > 0
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
            route_prepare=(buffer.route_prepare_args(rank_ready=True) if routes_prepared else None),
            perfect_balance=self.perfect_routing,
            recycle_payload=buffer.all_push,
            recycle_rows=self.all_push_reset_rows if buffer.all_push else 0,
            recycle_timings=debug_timings if buffer.all_push else None,
            recycle_ahead=(
                (
                    (self.buffers[lane] if bank else self.alternate_buffers[lane]).buffer,
                    (self.buffers[lane] if bank else self.alternate_buffers[lane]).buffer_ptrs_device,
                )
                if buffer.all_push and self.recycle_ahead else None
            ),
            defer_route_dependent_launch=self.defer_route_dependent_launch,
        )
        y = buffer.y[:num_tokens]
        self._run_mega_moe(
            y,
            l1_weights,
            l2_weights,
            buffer,
            routes_prepared=routes_prepared,
            rank_gated_combine=True,
            route_ready_dispatch=routes_prepared,
            rank_ready_route_publish=routes_prepared,
            **({"reset_rows": self.all_push_reset_rows} if buffer.all_push else {}),
            **({"debug_timings": debug_timings} if debug_timings is not None else {}),
        )
        return y.view_as(hidden_states)

    def normalize_output(self, norm, hidden, residual, *, debug_timings=None) -> bool:
        """Acquire output while retiring producer CTAs recycle the next payload bank."""
        buffer = self._output_buffers.get(hidden.data_ptr())
        if buffer is None:
            return False
        from minisgl.kernel.megamoe_output_norm import megamoe_output_add_rmsnorm

        if self.trace_root:
            debug_timings = self._trace_output[hidden.data_ptr()]
        megamoe_output_add_rmsnorm(
            hidden, residual, norm.weight, norm.eps, buffer, debug_timings=debug_timings
        )
        return True

    def save_trace(self, step_id: int, rows_per_lane: list[int]) -> None:
        """Export a separately instrumented model replay after its stream finishes."""
        import json
        from pathlib import Path
        from minisgl.kernel import deepgemm

        if not self._trace_timings:
            return
        sms = deepgemm.get_num_sms()
        root = Path(self.trace_root) / f"step-{step_id}"
        root.mkdir(parents=True, exist_ok=True)
        calls = {}
        for (layer, lane), values in self._trace_timings.items():
            rows = rows_per_lane[lane]
            raw = values.tolist()
            reset_offset = 24 + 3 * rows + sms
            calls[f"{layer}:{lane}"] = {
                "norm_rows": rows, "hidden": self.hidden_size,
                "stamps": raw[21:reset_offset],
                "cycle_timings": raw[:21],
                "reset_worker_stamps": raw[reset_offset:reset_offset + 16],
                **({"reset_ahead": raw[:21]} if self.recycle_ahead else {}),
            }
        (root / f"rank-{self.rank}.json").write_text(
            json.dumps({"rank": self.rank, "calls": calls}) + "\n"
        )


__all__ = ["MegaMoEAfdAdapter"]
