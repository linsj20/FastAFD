from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel.qwen35_gdn import (
    qwen35_causal_conv1d_decode_packed,
    qwen35_causal_conv1d_prefill,
    qwen35_gdn_decode,
    qwen35_gdn_gates,
    qwen35_gdn_prefill,
    qwen35_rmsnorm_gated,
    qwen35_rmsnorm_gated_fp8,
)
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVGatedMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormOffset,
    RMSNormOffsetFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


_GDN_PROJECTION_AUX_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


class GdnProjectionSync:
    """Reusable device handshakes for one serialized GDN projection lane.

    The owner publishes input readiness; the peer consumes and resets it, then
    publishes gate completion. The owner consumes/resets completion before any
    subsequent call can reuse the inputs or signals. Capture joins only at the
    enclosing graph boundary, preserving two execution lanes.
    """

    def __init__(self, device, timeout_record, timeout_ms: int, head_dim: int):
        self.ready = torch.zeros(1, dtype=torch.int64, device=device)
        self.done = torch.zeros_like(self.ready)
        self.timeout_record = timeout_record
        self.timeout_ms = timeout_ms
        self.head_dim = head_dim

    def release(self, turn: torch.Tensor) -> None:
        from minisgl.kernel.afd_fmha_transport import release_turn

        release_turn(turn, next_turn=1, head_dim=self.head_dim)

    def wait(self, turn: torch.Tensor) -> None:
        from minisgl.kernel.afd_fmha_transport import wait_turn

        wait_turn(
            turn, self.timeout_record, timeout_ms=self.timeout_ms,
            expected_turn=1, next_turn=0, head_dim=self.head_dim,
        )


def _qwen35_gdn_projection_aux_stream(
    main_stream: torch.cuda.Stream, device: torch.device
) -> torch.cuda.Stream:
    """Return one persistent projection side stream per decode lane."""

    device_index = (
        int(device.index) if device.index is not None else torch.cuda.current_device()
    )
    key = (device_index, int(main_stream.cuda_stream))
    stream = _GDN_PROJECTION_AUX_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _GDN_PROJECTION_AUX_STREAMS[key] = stream
    return stream


class _Qwen3_5DepthwiseConv1d(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Qwen3.5 depthwise convolution requires request state")


class Qwen3_5LinearAttention(BaseOP):
    """TP-sharded Qwen3.5 Gated DeltaNet with request-scoped recurrent state."""

    def __init__(self, config: ModelConfig, layer_id: int):
        tp_size = get_tp_info().size
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        local_key_heads = div_even(config.linear_num_key_heads, tp_size)
        local_value_heads = div_even(config.linear_num_value_heads, tp_size)
        local_conv_dim = (
            2 * local_key_heads * config.linear_key_head_dim
            + local_value_heads * config.linear_value_head_dim
        )
        module_prefix = f"model.layers.{layer_id}.linear_attn"

        self.in_proj_qkvz = LinearColParallelMerged(
            config.hidden_size,
            [key_dim, key_dim, value_dim, value_dim],
            has_bias=False,
            quant=config.quant_for_modules(
                f"{module_prefix}.in_proj_qkv",
                f"{module_prefix}.in_proj_z",
            ),
        )
        self.in_proj_ba = LinearColParallelMerged(
            config.hidden_size,
            [config.linear_num_value_heads, config.linear_num_value_heads],
            has_bias=False,
            # Qwen3.5 keeps these narrow projections in BF16. Their independent
            # checkpoint scales cannot be represented by one shared 128-row
            # FP8 block when b/a are packed together.
            quant=None,
        )
        self.conv1d = _Qwen3_5DepthwiseConv1d(
            local_conv_dim, config.linear_conv_kernel_dim
        )
        self.dt_bias = torch.empty(local_value_heads)
        self.A_log = torch.empty(local_value_heads, dtype=torch.float32)
        self.norm = RMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearOProj(
            value_dim,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.out_proj"),
        )
        self._num_key_heads = local_key_heads
        self._num_value_heads = local_value_heads
        self._key_head_dim = config.linear_key_head_dim
        self._value_head_dim = config.linear_value_head_dim
        self._local_key_dim = local_key_heads * self._key_head_dim
        self._local_value_dim = local_value_heads * self._value_head_dim
        self._norm_eps = config.rms_norm_eps
        self._layer_id = layer_id
        self._projection_events: dict[
            tuple[int, int], tuple[torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]
        ] = {}

    def _project_inputs(
        self, hidden_states: torch.Tensor, *, aux_stream: torch.cuda.Stream | None = None,
        projection_sync: GdnProjectionSync | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.cuda.Event | None,
    ]:
        """Fuse projections and overlap BA plus gate preparation with QKVZ."""

        if not hidden_states.is_cuda:
            projected_qkvz = self.in_proj_qkvz.forward(hidden_states)
            projected_ba = self.in_proj_ba.forward(hidden_states)
            b, a = projected_ba.view(-1, 2 * self._num_value_heads).split(
                self._num_value_heads, dim=-1
            )
            g, beta = qwen35_gdn_gates(a, b, self.A_log, self.dt_bias)
            return projected_qkvz, projected_ba, g, beta, None

        main_stream = torch.cuda.current_stream(hidden_states.device)
        start_event = done_event = None
        if projection_sync is not None:
            if aux_stream is None or aux_stream.cuda_stream == main_stream.cuda_stream:
                raise ValueError("GDN device dependencies require a distinct peer stream")
            projection_sync.release(projection_sync.ready)
        else:
            stream_key = (
                int(main_stream.cuda_stream),
                0 if aux_stream is None else int(aux_stream.cuda_stream),
            )
            sync = self._projection_events.get(stream_key)
            if sync is None:
                if aux_stream is None:
                    aux_stream = _qwen35_gdn_projection_aux_stream(main_stream, hidden_states.device)
                sync = (aux_stream, torch.cuda.Event(), torch.cuda.Event())
                self._projection_events[stream_key] = sync
            aux_stream, start_event, done_event = sync
            start_event.record(main_stream)
        projected_qkvz = self.in_proj_qkvz.forward(hidden_states)
        with torch.cuda.stream(aux_stream):
            if projection_sync is not None:
                projection_sync.wait(projection_sync.ready)
            else:
                aux_stream.wait_event(start_event)
            projected_ba = self.in_proj_ba.forward(hidden_states)
            b, a = projected_ba.view(-1, 2 * self._num_value_heads).split(
                self._num_value_heads, dim=-1
            )
            g, beta = qwen35_gdn_gates(a, b, self.A_log, self.dt_bias)
            if projection_sync is not None:
                projection_sync.release(projection_sync.done)
            else:
                done_event.record(aux_stream)
        hidden_states.record_stream(aux_stream)
        projected_ba.record_stream(main_stream)
        g.record_stream(main_stream)
        beta.record_stream(main_stream)
        return projected_qkvz, projected_ba, g, beta, done_event

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        projection_stream: torch.cuda.Stream | None = None,
        projection_sync: GdnProjectionSync | None = None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        state_pool = ctx.gdn_state_pool
        if state_pool is None:
            raise RuntimeError("Qwen3.5 GDN state pool is not initialized")
        layer_state = state_pool.layer_state(self._layer_id)

        projected_qkvz, projected_ba, g, beta, side_done_event = (
            self._project_inputs(
                hidden_states, aux_stream=projection_stream, projection_sync=projection_sync
            )
        )
        qkv_width = 2 * self._local_key_dim + self._local_value_dim
        mixed_qkv = projected_qkvz[:, :qkv_width]
        z = projected_qkvz[:, qkv_width:].view(
            -1, self._num_value_heads, self._value_head_dim
        )
        b, a = projected_ba.view(-1, 2 * self._num_value_heads).split(
            self._num_value_heads, dim=-1
        )

        if batch.is_decode:
            query, key, value, q_norm_scale, k_norm_scale = (
                qwen35_causal_conv1d_decode_packed(
                    mixed_qkv,
                    self.conv1d.weight,
                    layer_state.conv,
                    batch.state_indices,
                    batch.positions,
                    key_dim=self._local_key_dim,
                    value_dim=self._local_value_dim,
                    key_head_dim=self._key_head_dim,
                )
            )
        else:
            cu_seqlens, sequence_slots, first_positions = self._prefill_state_plan(batch)
            initial_conv = layer_state.conv.index_select(0, sequence_slots)
            mixed_qkv, final_conv = qwen35_causal_conv1d_prefill(
                mixed_qkv,
                self.conv1d.weight,
                initial_conv,
                cu_seqlens,
                first_positions,
            )
            layer_state.conv.index_copy_(0, sequence_slots, final_conv)
            query, key, value = mixed_qkv.split(
                [self._local_key_dim, self._local_key_dim, self._local_value_dim],
                dim=-1,
            )
        query = query.view(-1, self._num_key_heads, self._key_head_dim)
        key = key.view(-1, self._num_key_heads, self._key_head_dim)
        value = value.view(-1, self._num_value_heads, self._value_head_dim)

        # Keep the independent BA GEMM and fused gate preparation behind QKVZ
        # plus causal convolution; only recurrent-state update needs to join.
        if projection_sync is not None:
            projection_sync.wait(projection_sync.done)
        elif side_done_event is not None:
            torch.cuda.current_stream(hidden_states.device).wait_event(side_done_event)

        if batch.is_decode:
            core_output = qwen35_gdn_decode(
                query,
                key,
                value,
                a,
                b,
                self.A_log,
                self.dt_bias,
                layer_state.recurrent,
                batch.state_indices,
                batch.positions,
                q_norm_scale,
                k_norm_scale,
                g,
                beta,
            )
        else:
            initial_recurrent = layer_state.recurrent.index_select(0, sequence_slots)
            keep = (first_positions != 0).to(initial_recurrent.dtype)
            initial_recurrent = initial_recurrent * keep[:, None, None, None]
            core_output, final_recurrent = qwen35_gdn_prefill(
                query,
                key,
                value,
                g,
                beta,
                initial_recurrent,
                cu_seqlens,
            )
            layer_state.recurrent.index_copy_(0, sequence_slots, final_recurrent)

        if batch.is_decode and self.out_proj.weight.dtype == torch.float8_e4m3fn:
            gated_fp8, gated_scale = qwen35_rmsnorm_gated_fp8(
                core_output, z, self.norm.weight, self._norm_eps
            )
            return self.out_proj.forward_fp8_prequant(
                gated_fp8.flatten(1),
                gated_scale,
                output_shape_prefix=(core_output.shape[0],),
                output_dtype=core_output.dtype,
                output=output,
            )

        if output is not None:
            raise RuntimeError(
                "Qwen3.5 direct linear-attention output requires the decode "
                "FP8 out-projection path"
            )
        gated = qwen35_rmsnorm_gated(core_output, z, self.norm.weight, self._norm_eps)
        return self.out_proj.forward(gated.flatten(1))

    @staticmethod
    def _prefill_state_plan(batch):
        cached = getattr(batch, "_qwen35_gdn_prefill_plan", None)
        if cached is not None:
            return cached
        lengths = [int(req.extend_len) for req in batch.padded_reqs]
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu_seqlens = torch.tensor(
            boundaries, dtype=torch.int32, device=batch.positions.device
        )
        starts = cu_seqlens[:-1].to(torch.int64)
        sequence_slots = batch.state_indices.index_select(0, starts)
        first_positions = batch.positions.index_select(0, starts)
        plan = (cu_seqlens, sequence_slots, first_positions)
        batch._qwen35_gdn_prefill_plan = plan
        return plan


class Qwen3_5FullAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.q_size = self.num_qo_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        module_prefix = f"model.layers.{layer_id}.self_attn"
        self.qkv_proj = LinearQKVGatedMerged(
            config.hidden_size,
            config.head_dim,
            config.num_qo_heads,
            config.num_kv_heads,
            has_bias=False,
            quant=config.quant_for_modules(
                f"{module_prefix}.q_proj",
                f"{module_prefix}.k_proj",
                f"{module_prefix}.v_proj",
            ),
        )
        self.q_norm = RMSNormOffset(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormOffset(config.head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * config.head_dim,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.o_proj"),
        )

    def split_query_gate(self, q_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leading = q_gate.shape[:-1]
        per_head = q_gate.view(*leading, self.num_qo_heads, 2 * self.head_dim)
        query, gate = per_head.chunk(2, dim=-1)
        return query.reshape(*leading, self.q_size), gate.reshape(*leading, self.q_size)

    def prepare_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj.forward(hidden_states)

    def finish_attention_fp8(
        self, output: torch.Tensor, output_scale: torch.Tensor
    ) -> torch.Tensor:
        return self.o_proj.forward_fp8_prequant(
            output,
            output_scale,
            output_shape_prefix=tuple(output.shape[:-1]),
            output_dtype=torch.bfloat16,
        )

    def finish_attention(self, output: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(output)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(hidden_states)
        q_gate, key, value = qkv.split(
            [2 * self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query, gate = self.split_query_gate(q_gate)
        attn_output = self.attn.forward(torch.cat((query, key, value), dim=-1))
        return self.finish_attention(attn_output * torch.sigmoid(gate))


class Qwen3_5SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        intermediate_size = config.shared_expert_intermediate_size
        module_prefix = f"model.layers.{layer_id}.mlp.shared_expert"
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            has_bias=False,
            quant=config.quant_for_modules(
                f"{module_prefix}.gate_proj",
                f"{module_prefix}.up_proj",
            ),
        )
        self.down_proj = LinearRowParallel(
            intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant=config.quant_for_modules(f"{module_prefix}.down_proj"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden_states)))

    def forward_fp8_prequant(
        self,
        input_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        output_shape_prefix: tuple[int, ...],
    ) -> torch.Tensor:
        """Run gate/up from a caller-produced normalized FP8 activation."""
        gate_up = self.gate_up_proj.forward_fp8_prequant(
            input_fp8,
            input_scale,
            output_shape_prefix=output_shape_prefix,
            output_dtype=torch.bfloat16,
        )
        return self.down_proj.forward(silu_and_mul(gate_up))


def qwen35_shared_output(
    shared_hidden: torch.Tensor, gate_logits: torch.Tensor
) -> torch.Tensor:
    return shared_hidden * torch.sigmoid(gate_logits)


class Qwen3_5MoeMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant=config.quant,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = Qwen3_5SharedExpert(config, layer_id)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        routed = self.experts.forward(
            hidden_states=hidden_states,
            router_logits=self.gate.forward(hidden_states),
        )
        shared = qwen35_shared_output(
            self.shared_expert.forward(hidden_states),
            self.shared_expert_gate.forward(hidden_states),
        )
        return routed + shared


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5LinearAttention(config, layer_id)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(config, layer_id)
        else:
            raise ValueError(f"Invalid Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen3_5MoeMLP(config, layer_id)
        self.input_layernorm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormOffsetFused(
            config.hidden_size, config.rms_norm_eps
        )
        self._layer_id = layer_id

    def forward_token_mixer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn.forward(hidden_states)
        return self.self_attn.forward(hidden_states)

    def prepare_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_type != "full_attention":
            raise RuntimeError(
                f"Qwen3.5 layer {self._layer_id} is not a full-attention layer"
            )
        hidden_states, residual = self.input_layernorm.forward(
            hidden_states, residual
        )
        return self.self_attn.prepare_qkv(hidden_states), residual

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
        hidden_states = self.forward_token_mixer(hidden_states)
        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        return self.mlp.forward(hidden_states), residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormOffsetFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        residual = None
        for layer in self.layers.op_list:
            hidden_states, residual = layer.forward(hidden_states, residual)
        return self.norm.forward(hidden_states, residual)[0]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens.forward(input_ids)

    def finalize_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.norm.forward(hidden_states, residual)[0]


class Qwen3_5MoeForConditionalGeneration(BaseLLMModel):
    """Text-only Qwen3.5-MoE implementation for the multimodal checkpoint."""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def finalize_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.finalize_hidden(hidden_states, residual)

    def forward_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward(hidden_states)


__all__ = [
    "Qwen3_5FullAttention",
    "Qwen3_5LinearAttention",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5SharedExpert",
    "qwen35_shared_output",
]
