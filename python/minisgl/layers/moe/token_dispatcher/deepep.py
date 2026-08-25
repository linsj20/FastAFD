from __future__ import annotations

from typing import Any, NamedTuple

import torch

from minisgl.core import get_global_ctx
from minisgl.kernel.deepep_moe import deepep_topk_idx_dtype

from .base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from ..moe_runner.base import MoeRunnerConfig
from ..permute_registry import register_post_permute, register_pre_permute


class DeepEPElasticDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    handle: Any
    topk_ids: torch.Tensor | None
    topk_weights: torch.Tensor | None
    recv_count: torch.Tensor | None = None
    hidden_states_scale: torch.Tensor | None = None
    expected_m_per_expert: int = 0

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_ELASTIC


assert isinstance(DeepEPElasticDispatchOutput, DispatchOutput)


class DeepEPElasticCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    handle: Any

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_ELASTIC


assert isinstance(DeepEPElasticCombineInput, CombineInput)


class DeepEPElasticPreparedDispatch(NamedTuple):
    dispatch_x: torch.Tensor
    dispatch_x_scale: torch.Tensor | None
    group_topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_alignment: int
    expected_m_per_expert: int


class DeepEPDispatcher(BaseDispatcher):
    """DeepEP V2 elastic dispatcher for one dense-TP column.

    The group is all MLP-DP ranks with the same dense TP rank.  Router top-k
    expert ids are global model expert ids, while DeepEP expects ids local to
    this TP-column group, so we map global ids into the group expert namespace.
    """

    def __init__(self, moe_runner_config: MoeRunnerConfig):
        self.config = moe_runner_config
        self.dp_size = int(moe_runner_config.dp_size)
        self.tp_size = int(moe_runner_config.tp_size)
        self.tp_rank = int(moe_runner_config.tp_rank)
        self.num_experts = int(moe_runner_config.num_experts or 0)
        self.num_local_experts = int(moe_runner_config.num_local_experts or 0)
        self.top_k = int(moe_runner_config.top_k or 0)
        self._group_expert_map: torch.Tensor | None = None

    def _expected_m_per_expert(self, max_tokens_per_rank: int) -> int:
        """Return a graph-static DeepGEMM hint from the real routing capacity."""
        if self.num_experts <= 0:
            raise RuntimeError("DeepEP routing-density hint requires expert counts")
        numerator = (
            4
            * max(1, int(max_tokens_per_rank))
            * max(1, self.dp_size)
            * max(1, self.top_k)
        )
        return max(1, (numerator + self.num_experts - 1) // self.num_experts)

    @staticmethod
    def _deepgemm_expert_alignment(expected_m_per_expert: int) -> int:
        from minisgl.kernel import deepgemm as deep_gemm

        alignment = int(
            deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
                int(expected_m_per_expert)
            )
        )
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        return alignment

    def _buffer(self):
        buffer = getattr(get_global_ctx(), "mlp_deepep_buffer", None)
        if buffer is None:
            raise RuntimeError(
                "DeepEPDispatcher needs ctx.mlp_deepep_buffer; "
                "AFD DeepEP MoE runtime was not initialized"
            )
        return buffer

    def _map_global_to_group_experts(self, device: torch.device) -> torch.Tensor:
        cached = self._group_expert_map
        if cached is not None and cached.device == device:
            return cached
        if self.num_experts <= 0 or self.num_local_experts <= 0:
            raise RuntimeError("DeepEPDispatcher requires MoE expert counts")
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int64)
        for dp_rank in range(self.dp_size):
            global_ep_rank = dp_rank * self.tp_size + self.tp_rank
            global_start = global_ep_rank * self.num_local_experts
            group_start = dp_rank * self.num_local_experts
            expert_map[
                global_start : global_start + self.num_local_experts
            ] = torch.arange(
                group_start,
                group_start + self.num_local_experts,
                dtype=torch.int64,
            )
        self._group_expert_map = expert_map.to(device=device)
        return self._group_expert_map

    def prepare_for_capture(self, device: torch.device) -> None:
        """Materialize dispatcher state that CUDA graph capture cannot allocate."""
        self._map_global_to_group_experts(device)

    def _should_use_fp8_dispatch(self) -> bool:
        return self.config.params_dtype == torch.float8_e4m3fn

    def _topk_idx_dtype(self, buffer: Any) -> torch.dtype:
        topk_idx_dtype = getattr(buffer, "topk_idx_dtype", None)
        return deepep_topk_idx_dtype() if topk_idx_dtype is None else topk_idx_dtype

    def _finish_prepared_dispatch(
        self,
        hidden_states: torch.Tensor,
        group_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expected_m_per_expert: int,
    ) -> DeepEPElasticPreparedDispatch:
        buffer = self._buffer()
        if not bool(getattr(buffer, "is_elastic", False)):
            raise RuntimeError(
                "DeepEPDispatcher requires DeepEP V2 ElasticBuffer; "
                f"got buffer={type(buffer).__name__}"
            )
        dispatch_x = hidden_states.contiguous()
        dispatch_x_scale = None
        if self._should_use_fp8_dispatch():
            from minisgl.kernel.fp8_quant import per_token_cast_to_fp8

            dispatch_x, dispatch_x_scale = per_token_cast_to_fp8(
                dispatch_x,
                use_ue8m0=True,
                gran_k=128,
                use_packed_ue8m0=True,
                backend="cuda",
            )
        return DeepEPElasticPreparedDispatch(
            dispatch_x=dispatch_x,
            dispatch_x_scale=dispatch_x_scale,
            group_topk_ids=group_topk_ids.contiguous(),
            topk_weights=topk_weights.contiguous(),
            expert_alignment=self._deepgemm_expert_alignment(
                expected_m_per_expert
            ),
            expected_m_per_expert=expected_m_per_expert,
        )

    def prepare_dispatch(
        self, hidden_states: torch.Tensor, topk_output: Any
    ) -> DeepEPElasticPreparedDispatch:
        """Run router top-k and FP8 quantization, stopping before 24-SM comm."""
        if not isinstance(topk_output, torch.Tensor):
            raise RuntimeError("DeepEPDispatcher expects raw router logits")
        from minisgl.kernel import topk_softmax_group_local

        ctx = get_global_ctx()
        buffer = self._buffer()
        max_tokens_per_rank = int(ctx.moe_deepep_dispatch_max_tokens_per_rank)
        expected_m_per_expert = self._expected_m_per_expert(max_tokens_per_rank)
        topk_idx_dtype = self._topk_idx_dtype(buffer)
        topk_weights = torch.empty(
            (hidden_states.shape[0], self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        group_topk_ids = torch.empty(
            (hidden_states.shape[0], self.top_k),
            dtype=topk_idx_dtype,
            device=hidden_states.device,
        )
        topk_softmax_group_local(
            topk_weights,
            group_topk_ids,
            topk_output,
            self._map_global_to_group_experts(topk_output.device),
            num_token_non_padded=ctx.moe_num_token_non_padded,
            renormalize=bool(self.config.renormalize),
        )
        return self._finish_prepared_dispatch(
            hidden_states,
            group_topk_ids,
            topk_weights,
            expected_m_per_expert,
        )

    def prepare_dispatch_from_gate(
        self, hidden_states: torch.Tensor, gate_weight: torch.Tensor
    ) -> DeepEPElasticPreparedDispatch:
        """Fuse the decode router GEMM and group-local top-k selection."""
        from minisgl.kernel.moe_topk import gate_topk

        if hidden_states.ndim != 2 or gate_weight.ndim != 2:
            raise RuntimeError(
                "fused DeepEP router expects hidden/gate matrices, got "
                f"hidden={tuple(hidden_states.shape)} gate={tuple(gate_weight.shape)}"
            )
        if gate_weight.shape != (self.num_experts, hidden_states.shape[1]):
            raise RuntimeError(
                "fused DeepEP router shape mismatch: "
                f"hidden={tuple(hidden_states.shape)} gate={tuple(gate_weight.shape)} "
                f"experts={self.num_experts}"
            )
        ctx = get_global_ctx()
        buffer = self._buffer()
        max_tokens_per_rank = int(ctx.moe_deepep_dispatch_max_tokens_per_rank)
        expected_m_per_expert = self._expected_m_per_expert(max_tokens_per_rank)
        group_topk_ids, topk_weights = gate_topk(
            hidden_states,
            gate_weight,
            self.top_k,
            renormalize=bool(self.config.renormalize),
            expert_map=self._map_global_to_group_experts(hidden_states.device),
            num_token_non_padded=ctx.moe_num_token_non_padded,
            topk_idx_dtype=self._topk_idx_dtype(buffer),
        )
        return self._finish_prepared_dispatch(
            hidden_states,
            group_topk_ids,
            topk_weights,
            expected_m_per_expert,
        )

    def dispatch_prepared(
        self, prepared: DeepEPElasticPreparedDispatch
    ) -> DeepEPElasticDispatchOutput:
        """Launch DeepEP dispatch from a pre-quantized routing packet."""
        ctx = get_global_ctx()
        buffer = self._buffer()
        if not bool(getattr(buffer, "is_elastic", False)):
            raise RuntimeError(
                "DeepEPDispatcher requires DeepEP V2 ElasticBuffer; "
                f"got buffer={type(buffer).__name__}"
            )
        (
            recv_x,
            recv_x_scale,
            recv_group_topk_ids,
            recv_topk_weights,
            handle,
        ) = buffer.dispatch(
            prepared.dispatch_x,
            prepared.group_topk_ids,
            prepared.topk_weights,
            hidden_states_scale=prepared.dispatch_x_scale,
            expert_alignment=prepared.expert_alignment,
            num_max_dispatch_tokens_per_rank=ctx.moe_deepep_dispatch_max_tokens_per_rank,
        )
        return DeepEPElasticDispatchOutput(
            hidden_states=recv_x,
            handle=handle,
            # ElasticBuffer returns top-k ids in the receiving rank's
            # local expert namespace.  The runner must consume them without
            # applying the global expert_map again.
            topk_ids=recv_group_topk_ids,
            topk_weights=recv_topk_weights[: recv_x.shape[0]],
            recv_count=handle.handle.psum_num_recv_tokens_per_expert,
            hidden_states_scale=recv_x_scale,
            expected_m_per_expert=prepared.expected_m_per_expert,
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: Any
    ) -> DeepEPElasticDispatchOutput:
        prepared = self.prepare_dispatch(hidden_states, topk_output)
        return self.dispatch_prepared(prepared)

    def combine(
        self,
        combine_input: CombineInput,
        *,
        release_turn: torch.Tensor | None = None,
        release_value: int = 0,
    ) -> torch.Tensor:
        if isinstance(combine_input, DeepEPElasticCombineInput):
            return self.combine_elastic(
                combine_input,
                release_turn=release_turn,
                release_value=release_value,
            )
        raise RuntimeError(
            f"DeepEPDispatcher expected DeepEPElasticCombineInput, "
            f"got {type(combine_input).__name__}"
        )

    def combine_elastic(
        self,
        combine_input: DeepEPElasticCombineInput,
        *,
        release_turn: torch.Tensor | None = None,
        release_value: int = 0,
    ) -> torch.Tensor:
        return self._buffer().combine(
            combine_input.hidden_states,
            None,
            combine_input.handle,
            release_turn=release_turn,
            release_value=release_value,
        )


@register_pre_permute("deepep_elastic", "deep_gemm")
def _pre_permute_deepep_elastic_to_deep_gemm(
    dispatch_output: DeepEPElasticDispatchOutput,
) -> DeepEPElasticDispatchOutput:
    if dispatch_output.recv_count is None:
        raise RuntimeError("DeepGEMM requires DeepEP V2 expanded dispatch output")
    return dispatch_output


@register_post_permute("deep_gemm", "deepep_elastic")
def _post_permute_runner_to_deepep_elastic(
    runner_output: torch.Tensor,
    dispatch_output: DeepEPElasticDispatchOutput,
) -> DeepEPElasticCombineInput:
    return DeepEPElasticCombineInput(
        hidden_states=runner_output,
        handle=dispatch_output.handle,
    )


__all__ = [
    "DeepEPDispatcher",
    "DeepEPElasticCombineInput",
    "DeepEPElasticDispatchOutput",
    "DeepEPElasticPreparedDispatch",
]
