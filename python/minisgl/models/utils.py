from __future__ import annotations

from typing import Any, NamedTuple

import torch

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            quant=config.quant,
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant=config.quant,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class _MoEMLPPrepared(NamedTuple):
    experts: Any
    num_tokens: int
    hidden_dim: int


class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant=config.quant,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states

    def prepare_deepep(self, hidden_states: torch.Tensor) -> _MoEMLPPrepared:
        """Fuse decode routing and prepare FP8 dispatch before communication."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        return _MoEMLPPrepared(
            experts=self.experts.prepare_deepep_from_gate(
                hidden_states, self.gate.weight
            ),
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
        )

    def finish_deepep(self, prepared: _MoEMLPPrepared) -> torch.Tensor:
        """Complete communication and expert execution for a prepared MoE input."""
        dispatched = self.dispatch_deepep(prepared)
        expert_output = self.run_deepep_experts(dispatched)
        return self.combine_deepep(expert_output)

    def dispatch_deepep(self, prepared: _MoEMLPPrepared) -> _MoEMLPPrepared:
        return _MoEMLPPrepared(
            experts=self.experts.dispatch_deepep(prepared.experts),
            num_tokens=prepared.num_tokens,
            hidden_dim=prepared.hidden_dim,
        )

    def run_deepep_experts(self, dispatched: _MoEMLPPrepared) -> _MoEMLPPrepared:
        return _MoEMLPPrepared(
            experts=self.experts.run_deepep_experts(dispatched.experts),
            num_tokens=dispatched.num_tokens,
            hidden_dim=dispatched.hidden_dim,
        )

    def combine_deepep(
        self,
        expert_output: _MoEMLPPrepared,
        *,
        release_turn: torch.Tensor | None = None,
        release_value: int = 0,
    ) -> torch.Tensor:
        final_hidden_states = self.experts.combine_deepep(
            expert_output.experts,
            release_turn=release_turn,
            release_value=release_value,
        )
        return final_hidden_states.view(
            expert_output.num_tokens, expert_output.hidden_dim
        )


class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
            quant=config.quant,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant=config.quant,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.prepare_qkv(x)
        del x
        o = self.attn.forward(qkv)
        return self.finish_attention(o)

    def prepare_qkv(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj.forward(x)

    def finish_attention(self, o: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(o)

    def finish_attention_fp8(
        self, o_fp8: torch.Tensor, o_scale: torch.Tensor
    ) -> torch.Tensor:
        return self.o_proj.forward_fp8_prequant(
            o_fp8,
            o_scale,
            output_shape_prefix=tuple(o_fp8.shape[:-1]),
            output_dtype=torch.bfloat16,
        )


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
