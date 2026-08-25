from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import minisgl.core as core

from minisgl.afd_attention_worker import AfdAttentionState
from minisgl.afd_attention_runtime import AfdAttentionRuntime
from minisgl.afd_expert_worker import AfdModelState
from minisgl.afd_protocol import AfdAGStepPlan, AfdSamplingPlan, AfdTokenBlock
from minisgl.afd_support import AfdRuntimeConfig
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.kernel.deepep_moe import DeepEPMoeElasticBuffer
from minisgl.kernel.afd_fmha_transport import (
    publish_o,
    publish_qkv,
    wait_ready,
)
from minisgl.kernel.fabric_memory import allocate_fabric_tensor, import_fabric_tensor
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.layers import set_rope_device
from minisgl.models import load_weight
from minisgl.models.qwen3_moe import Qwen3MoeForCausalLM
from minisgl.utils import div_ceil, torch_dtype


@dataclass(frozen=True)
class _TransportSignals:
    q_ready: torch.Tensor
    q_ready_descriptors: torch.Tensor
    q_publish_counters: torch.Tensor
    o_ready: torch.Tensor
    o_ready_descriptors: torch.Tensor
    o_publish_counters: torch.Tensor


def _transport_signals(
    *,
    q_ready: torch.Tensor,
    q_ready_destination: torch.Tensor,
    o_ready: torch.Tensor,
    o_ready_destination: torch.Tensor,
    device: torch.device,
) -> _TransportSignals:
    def descriptor(destination: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[destination.data_ptr(), 0, 1]], dtype=torch.int64, device=device
        )

    return _TransportSignals(
        q_ready=q_ready,
        q_ready_descriptors=descriptor(q_ready_destination),
        q_publish_counters=torch.zeros((2,), dtype=torch.int32, device=device),
        o_ready=o_ready,
        o_ready_descriptors=descriptor(o_ready_destination),
        o_publish_counters=torch.zeros((2,), dtype=torch.int32, device=device),
    )


def _load_case(
    alignment_path: str,
    legacy_sample_path: str,
    case_index: int,
) -> tuple[torch.Tensor, int]:
    with open(alignment_path, encoding="utf-8") as handle:
        prompt_ids = json.load(handle)["results"][case_index]["prompt_token_ids"]
    with open(legacy_sample_path, encoding="utf-8") as handle:
        expected = json.load(handle)["samples"][case_index]["generated_token_ids"][0]
    return torch.tensor(prompt_ids, dtype=torch.int32), int(expected)


def _forward_split_attention(
    model: Qwen3MoeForCausalLM,
    batch: Batch,
    *,
    q_slots: torch.Tensor,
    q_descriptors: torch.Tensor,
    kv_descriptors: torch.Tensor,
    signals: _TransportSignals,
    timeout_record: torch.Tensor,
    backend,
    o_slots: torch.Tensor | None = None,
    o_descriptors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mirror FMHA-only QKV publication while keeping producer/consumer local.

    This preserves the full checkpoint and EP4 model path while isolating the
    qk_norm_rope -> publication -> prepared-attention composition from fabric
    mapping and inter-node synchronization.
    """

    rows = int(batch.positions.numel())
    model_config = model.model.layers.op_list[0].self_attn.attn
    q_heads = int(model_config.num_qo_heads)
    kv_heads = int(model_config.num_kv_heads)
    head_dim = int(model_config.head_dim)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    hidden = model.embed_input_ids(batch.input_ids)
    residual = None
    for layer_id, layer in enumerate(model.model.layers.op_list):
        qkv, residual = layer.prepare_attention(hidden, residual)
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        q, k = layer.self_attn.attn.apply_qk_norm_rope(batch.positions, q, k)
        slot = layer_id & 1
        publish_qkv(
            q,
            k,
            v,
            batch.out_loc,
            q_descriptors,
            kv_descriptors,
            signals.q_ready_descriptors,
            signals.q_publish_counters[slot : slot + 1],
            layer=layer_id,
            slot=slot,
            head_dim=head_dim,
            consumed_o_ready=signals.o_ready[slot],
        )
        q_slot = q_slots[slot, :rows]
        wait_ready(
            signals.q_ready[slot],
            timeout_record,
            timeout_ms=30_000,
            head_dim=head_dim,
        )
        o = backend.forward_prepared(q_slot, layer_id, batch)
        if o_descriptors is not None:
            if o_slots is None:
                raise RuntimeError("peer split attention requires O receive slots")
            publish_o(
                o.view(rows, -1),
                o_descriptors,
                signals.o_ready_descriptors,
                signals.o_publish_counters[slot : slot + 1],
                signals.q_ready[slot],
                slot=slot,
                head_dim=head_dim,
            )
            o_slot = o_slots[slot, :rows]
            wait_ready(
                signals.o_ready[slot],
                timeout_record,
                timeout_ms=30_000,
                head_dim=head_dim,
            )
            hidden = layer.self_attn.finish_attention(o_slot.view(rows, -1))
        else:
            signals.q_ready[slot].zero_()
            hidden = layer.self_attn.finish_attention(o.view(rows, -1))
        hidden, residual = layer.post_attention_layernorm.forward(hidden, residual)
        hidden = layer.mlp.forward(hidden)
    return model.finalize_hidden(hidden, residual)


def _forward_runtime_split_local(
    model: Qwen3MoeForCausalLM,
    model_batch: Batch,
    attention_batch: Batch,
    *,
    q_slots: torch.Tensor,
    q_descriptors: torch.Tensor,
    kv_descriptors: torch.Tensor,
    signals: _TransportSignals,
    timeout_record: torch.Tensor,
    backend,
) -> torch.Tensor:
    """Run the split model with the two production-materialized batches."""

    rows = int(model_batch.positions.numel())
    attention = model.model.layers.op_list[0].self_attn.attn
    q_heads = int(attention.num_qo_heads)
    kv_heads = int(attention.num_kv_heads)
    head_dim = int(attention.head_dim)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    hidden = model.embed_input_ids(model_batch.input_ids)
    residual = None
    for layer_id, layer in enumerate(model.model.layers.op_list):
        qkv, residual = layer.prepare_attention(hidden, residual)
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        q, k = layer.self_attn.attn.apply_qk_norm_rope(
            model_batch.positions, q, k
        )
        slot = layer_id & 1
        publish_qkv(
            q,
            k,
            v,
            model_batch.out_loc,
            q_descriptors,
            kv_descriptors,
            signals.q_ready_descriptors,
            signals.q_publish_counters[slot : slot + 1],
            layer=layer_id,
            slot=slot,
            head_dim=head_dim,
            consumed_o_ready=signals.o_ready[slot],
        )
        q_slot = q_slots[slot, :rows]
        wait_ready(
            signals.q_ready[slot],
            timeout_record,
            timeout_ms=30_000,
            head_dim=head_dim,
        )
        o = backend.forward_prepared(q_slot, layer_id, attention_batch)
        signals.q_ready[slot].zero_()
        hidden = layer.self_attn.finish_attention(o.view(rows, -1))
        hidden, residual = layer.post_attention_layernorm.forward(hidden, residual)
        hidden = layer.mlp.forward(hidden)
    return model.finalize_hidden(hidden, residual)


def _runtime_prefill_plan(
    *,
    step_id: int,
    start: int,
    end: int,
    prompt_ids: torch.Tensor,
) -> AfdAGStepPlan:
    rows = int(end) - int(start)
    return AfdAGStepPlan(
        step_id=int(step_id),
        phase="prefill",
        real_size=1,
        table_indices=(0,),
        extend_lens=(rows,),
        microbatch_offsets=(0, 1),
        microbatch_token_offsets=(0, rows),
        microbatch_real_token_counts=(rows,),
        exec_table_indices=(0,),
        exec_start_positions=(int(start),),
        exec_extend_lens=(rows,),
        exec_sample_indices=(0,),
        exec_writebacks=(end == int(prompt_ids.numel()),),
        token_blocks=(
            AfdTokenBlock(
                table_idx=0,
                start_pos=int(start),
                tokens=prompt_ids[start:end].clone(),
            ),
        ),
        sampling=AfdSamplingPlan.from_sampling_params((SamplingParams(),)),
        dispatch_bucket=rows,
    )


def _main_runtime_split_local() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 4:
        raise RuntimeError(
            f"runtime split-local checkpoint requires four ranks, got {world_size}"
        )
    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ["MINISGL_DEVICE_INDEX"] = str(local_rank)

    model_path = os.environ["AFD_CHECKPOINT_MODEL_PATH"]
    alignment_path = os.environ["AFD_CHECKPOINT_ALIGNMENT_JSON"]
    legacy_sample_path = os.environ["AFD_CHECKPOINT_LEGACY_SAMPLE_JSON"]
    case_indices = tuple(
        int(value)
        for value in os.environ.get("AFD_CHECKPOINT_CASE_INDICES", "0,1,2,3").split(",")
    )
    if len(case_indices) != world_size:
        raise RuntimeError("runtime split-local checkpoint requires four cases")
    prompt_ids, expected_token = _load_case(
        alignment_path, legacy_sample_path, case_indices[rank]
    )
    sequence_length = int(prompt_ids.numel())
    page_size = 64
    chunk_rows = int(os.environ.get("AFD_CHECKPOINT_CHUNK_ROWS", "512"))
    num_pages = div_ceil(sequence_length + 1, page_size)
    tp_groups = tuple((worker_rank,) for worker_rank in range(world_size))
    config = AfdRuntimeConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="trtllm",
        moe_backend="fused",
        cuda_graph_max_bs=0,
        page_size=page_size,
        memory_ratio=0.9,
        max_extend_tokens=chunk_rows,
        max_seq_len_override=sequence_length + 1,
        num_page_override=num_pages,
        ep_size=world_size,
        dp_size=world_size,
        dp_rank=rank,
        cache_type="naive",
        distributed_host=os.environ["MASTER_ADDR"],
        distributed_port=int(os.environ["MASTER_PORT"]),
        distributed_rank=rank,
        distributed_world_size=world_size,
        distributed_tp_groups=tp_groups,
        afd_moe_a2a_backend="deepep",
        afd_moe_runner_backend="deep_gemm",
        afd_model_placement="fmha-only",
    )

    attention_state = AfdAttentionState(config)
    attention_runtime = AfdAttentionRuntime(
        config,
        attention_state,
        owns_model_state=False,
        prepare_attention_metadata=True,
    )
    # This diagnostic intentionally hosts both production roles in one process;
    # production itself has one Context per actor. Expose the task-local CUDA
    # ordinal and release the attention Context before constructing model state.
    os.environ["MINISGL_DEVICE_INDEX"] = str(attention_state.device.index or 0)
    core._GLOBAL_CTX = None
    model_state = AfdModelState(config)
    model_runtime = AfdAttentionRuntime(
        config,
        model_state,
        owns_model_state=True,
        prepare_attention_metadata=False,
    )
    attention_runtime.configure_num_pages(num_pages)
    model_runtime.configure_num_pages(num_pages)
    model_ctx = model_state.ctx
    model_ctx.server_args = config
    model_ctx.replica_id = rank
    core._GLOBAL_CTX = model_ctx

    moe_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    set_rope_device(model_state.device)
    with torch.device("meta"), torch_dtype(config.dtype):
        model = Qwen3MoeForCausalLM(config.model_config)
    weights = {
        key: (
            value
            if value.dtype == torch.float8_e4m3fn
            or (value.dtype == torch.float32 and key.endswith("_scale"))
            else value.to(config.dtype)
        )
        for key, value in load_weight(model_path, model_state.device)
    }
    model.load_state_dict(weights)
    del weights
    local_experts = div_ceil(int(config.model_config.num_experts), world_size)
    model_ctx.mlp_deepep_buffer = DeepEPMoeElasticBuffer(
        group=moe_group,
        num_max_dispatch_tokens_per_rank=chunk_rows,
        hidden_size=int(config.model_config.hidden_size),
        num_experts=local_experts * world_size,
        top_k=int(config.model_config.num_experts_per_tok),
    )

    model_config = config.model_config
    q_heads = int(model_config.num_qo_heads)
    kv_heads = int(model_config.num_kv_heads)
    head_dim = int(model_config.head_dim)
    q_slots = torch.empty(
        (2, chunk_rows, q_heads, head_dim),
        dtype=torch.bfloat16,
        device=model_state.device,
    )
    q_ready = torch.zeros((2, 1), dtype=torch.int64, device=model_state.device)
    o_ready = torch.zeros_like(q_ready)
    signals = _transport_signals(
        q_ready=q_ready,
        q_ready_destination=q_ready,
        o_ready=o_ready,
        o_ready_destination=o_ready,
        device=model_state.device,
    )
    kv_buffer = attention_state.ctx.kv_cache._kv_buffer
    cache_rows = int(kv_buffer.shape[2]) * int(kv_buffer.shape[3])
    q_descriptors = torch.tensor(
        [[q_slots.data_ptr(), 0, 0, q_heads, q_heads, chunk_rows]],
        dtype=torch.int64,
        device=model_state.device,
    )
    kv_descriptors = torch.tensor(
        [[
            kv_buffer.data_ptr(),
            0,
            0,
            kv_heads,
            kv_heads,
            cache_rows,
            int(model_config.num_layers),
        ]],
        dtype=torch.int64,
        device=model_state.device,
    )
    timeout_record = torch.zeros((2,), dtype=torch.int64, device=model_state.device)
    backend = attention_state.attn_backends[0]
    actual_token = -1

    with torch.inference_mode():
        for step_id, start in enumerate(range(0, sequence_length, chunk_rows), start=1):
            end = min(sequence_length, start + chunk_rows)
            plan = _runtime_prefill_plan(
                step_id=step_id,
                start=start,
                end=end,
                prompt_ids=prompt_ids,
            )
            core._GLOBAL_CTX = attention_state.ctx
            attention_batch = attention_runtime.materialize_ag_plan(plan)
            core._GLOBAL_CTX = model_ctx
            model_batch = model_runtime.materialize_ag_plan(plan)
            attention_state.stream.wait_stream(attention_runtime.stream)
            model_state.stream.wait_stream(model_runtime.stream)
            model_state.stream.wait_stream(attention_runtime.stream)
            rows = end - start
            model_ctx.moe_num_token_non_padded = rows
            model_ctx.moe_deepep_dispatch_max_tokens_per_rank = rows
            with torch.cuda.stream(model_state.stream), model_ctx.forward_batch(model_batch):
                final_hidden = _forward_runtime_split_local(
                    model,
                    model_batch,
                    attention_batch,
                    q_slots=q_slots,
                    q_descriptors=q_descriptors,
                    kv_descriptors=kv_descriptors,
                    signals=signals,
                    timeout_record=timeout_record,
                    backend=backend,
                )
                if end == sequence_length:
                    selected = final_hidden.index_select(0, model_batch.afd_last_indices)
            if end == sequence_length:
                sampling_batch = model_runtime.build_sampling_batch(1)
                with torch.cuda.stream(model_state.stream), model_ctx.forward_batch(
                    sampling_batch
                ):
                    logits = model.forward_lm_head(selected)
                    actual_token = int(torch.argmax(logits, dim=-1).item())

    torch.cuda.synchronize(model_state.device)
    result = {
        "rank": rank,
        "case_index": case_indices[rank],
        "sequence_length": sequence_length,
        "attention_mode": "runtime-split-local",
        "actual_token": actual_token,
        "legacy_token": expected_token,
        "matches_legacy": actual_token == expected_token,
    }
    results: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(results, result)
    if rank == 0:
        print("AFD_CHECKPOINT_RUNTIME_SPLIT_RESULTS " + json.dumps(results, sort_keys=True))
    model_ctx.mlp_deepep_buffer.destroy()
    dist.barrier()
    model_state.shutdown()


def _forward_role_split_model(
    model: Qwen3MoeForCausalLM,
    batch: Batch,
    *,
    q_descriptors: torch.Tensor,
    kv_descriptors: torch.Tensor,
    o_slots: torch.Tensor,
    q_ready_descriptors: torch.Tensor,
    q_publish_counters: torch.Tensor,
    o_ready: torch.Tensor,
    timeout_record: torch.Tensor,
) -> torch.Tensor:
    rows = int(batch.positions.numel())
    attention = model.model.layers.op_list[0].self_attn.attn
    q_heads = int(attention.num_qo_heads)
    kv_heads = int(attention.num_kv_heads)
    head_dim = int(attention.head_dim)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    hidden = model.embed_input_ids(batch.input_ids)
    residual = None
    for layer_id, layer in enumerate(model.model.layers.op_list):
        qkv, residual = layer.prepare_attention(hidden, residual)
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        q, k = layer.self_attn.attn.apply_qk_norm_rope(batch.positions, q, k)
        slot = layer_id & 1
        publish_qkv(
            q,
            k,
            v,
            batch.out_loc,
            q_descriptors,
            kv_descriptors,
            q_ready_descriptors,
            q_publish_counters[slot : slot + 1],
            layer=layer_id,
            slot=slot,
            head_dim=head_dim,
            consumed_o_ready=o_ready[slot],
        )
        o_slot = o_slots[slot, :rows]
        wait_ready(
            o_ready[slot],
            timeout_record,
            timeout_ms=30_000,
            head_dim=head_dim,
        )
        hidden = layer.self_attn.finish_attention(o_slot.view(rows, -1))
        hidden, residual = layer.post_attention_layernorm.forward(hidden, residual)
        hidden = layer.mlp.forward(hidden)
    return model.finalize_hidden(hidden, residual)


def _forward_role_split_attention(
    batch: Batch,
    *,
    num_layers: int,
    head_dim: int,
    q_slots: torch.Tensor,
    q_ready: torch.Tensor,
    o_descriptors: torch.Tensor,
    o_ready_descriptors: torch.Tensor,
    o_publish_counters: torch.Tensor,
    timeout_record: torch.Tensor,
    backend,
) -> None:
    rows = int(batch.positions.numel())
    for layer_id in range(num_layers):
        slot = layer_id & 1
        q_slot = q_slots[slot, :rows]
        wait_ready(
            q_ready[slot],
            timeout_record,
            timeout_ms=30_000,
            head_dim=head_dim,
        )
        o = backend.forward_prepared(q_slot, layer_id, batch)
        publish_o(
            o.view(rows, -1),
            o_descriptors,
            o_ready_descriptors,
            o_publish_counters[slot : slot + 1],
            q_ready[slot],
            slot=slot,
            head_dim=head_dim,
        )


def _main_role_split() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 8:
        raise RuntimeError(f"checkpoint role split requires eight ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ["MINISGL_DEVICE_INDEX"] = str(local_rank)
    is_attention = rank < 4
    lane = rank if is_attention else rank - 4

    model_path = os.environ["AFD_CHECKPOINT_MODEL_PATH"]
    alignment_path = os.environ["AFD_CHECKPOINT_ALIGNMENT_JSON"]
    legacy_sample_path = os.environ["AFD_CHECKPOINT_LEGACY_SAMPLE_JSON"]
    case_indices = tuple(
        int(value)
        for value in os.environ.get("AFD_CHECKPOINT_CASE_INDICES", "0,1,2,3").split(",")
    )
    if len(case_indices) != 4:
        raise RuntimeError("role-split checkpoint cases must contain four indices")
    prompt_ids, expected_token = _load_case(
        alignment_path, legacy_sample_path, case_indices[lane]
    )
    sequence_length = int(prompt_ids.numel())
    page_size = 64
    chunk_rows = int(os.environ.get("AFD_CHECKPOINT_CHUNK_ROWS", "512"))
    num_pages = div_ceil(sequence_length + 1, page_size)
    tp_groups = tuple((worker_rank,) for worker_rank in range(world_size))
    config = AfdRuntimeConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="trtllm",
        moe_backend="fused",
        cuda_graph_max_bs=0,
        page_size=page_size,
        memory_ratio=0.9,
        max_extend_tokens=chunk_rows,
        max_seq_len_override=sequence_length + 1,
        num_page_override=num_pages,
        ep_size=4,
        dp_size=4,
        dp_rank=lane,
        cache_type="naive",
        distributed_host=os.environ["MASTER_ADDR"],
        distributed_port=int(os.environ["MASTER_PORT"]),
        distributed_rank=rank,
        distributed_world_size=world_size,
        distributed_tp_groups=tp_groups,
        afd_moe_a2a_backend="deepep",
        afd_moe_runner_backend="deep_gemm",
        afd_model_placement="fmha-only" if is_attention else "legacy",
    )
    state = AfdAttentionState(config)
    ctx = state.ctx
    ctx.server_args = config
    ctx.replica_id = lane
    model_ranks = list(range(4, 8))
    model_group = dist.new_group(ranks=model_ranks, backend="gloo")

    model = None
    if not is_attention:
        set_rope_device(state.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            model = Qwen3MoeForCausalLM(config.model_config)
        weights = {
            key: (
                value
                if value.dtype == torch.float8_e4m3fn
                or (value.dtype == torch.float32 and key.endswith("_scale"))
                else value.to(config.dtype)
            )
            for key, value in load_weight(model_path, state.device)
        }
        model.load_state_dict(weights)
        del weights
        local_experts = div_ceil(int(config.model_config.num_experts), 4)
        ctx.mlp_deepep_buffer = DeepEPMoeElasticBuffer(
            group=model_group,
            num_max_dispatch_tokens_per_rank=chunk_rows,
            hidden_size=int(config.model_config.hidden_size),
            num_experts=local_experts * 4,
            top_k=int(config.model_config.num_experts_per_tok),
        )

    ctx.page_table[0, :sequence_length].copy_(
        torch.arange(sequence_length, dtype=torch.int32, device=state.device)
    )
    prompt_device = (
        prompt_ids.pin_memory().to(state.device, non_blocking=True)
        if not is_attention
        else None
    )
    model_config = config.model_config
    q_heads = int(model_config.num_qo_heads)
    kv_heads = int(model_config.num_kv_heads)
    head_dim = int(model_config.head_dim)
    record: dict[str, object] = {"rank": rank}
    if is_attention:
        q_owner = allocate_fabric_tensor(
            (2, chunk_rows, q_heads, head_dim), dtype=torch.bfloat16
        )
        q_slots = q_owner.tensor
        q_ready_owner = allocate_fabric_tensor((2, 1), dtype=torch.int64)
        q_ready = q_ready_owner.tensor
        q_ready.zero_()
        kv_owner = state.fabric_kv
        kv_buffer = ctx.kv_cache._kv_buffer
        record.update(
            q_handle=q_owner.handle,
            q_shape=tuple(q_slots.shape),
            q_ready_handle=q_ready_owner.handle,
            q_ready_shape=tuple(q_ready.shape),
            kv_handle=kv_owner.handle,
            kv_shape=tuple(kv_buffer.shape),
        )
    else:
        o_owner = allocate_fabric_tensor(
            (2, chunk_rows, q_heads, head_dim), dtype=torch.bfloat16
        )
        o_slots = o_owner.tensor
        o_ready_owner = allocate_fabric_tensor((2, 1), dtype=torch.int64)
        o_ready = o_ready_owner.tensor
        o_ready.zero_()
        record.update(
            o_handle=o_owner.handle,
            o_shape=tuple(o_slots.shape),
            o_ready_handle=o_ready_owner.handle,
            o_ready_shape=tuple(o_ready.shape),
        )

    torch.cuda.synchronize(state.device)
    records: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(records, record)
    if is_attention:
        model_record = records[4 + lane]
        assert model_record is not None
        o_mapping = import_fabric_tensor(
            model_record["o_handle"],  # type: ignore[arg-type]
            model_record["o_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )
        o_ready_mapping = import_fabric_tensor(
            model_record["o_ready_handle"],  # type: ignore[arg-type]
            model_record["o_ready_shape"],  # type: ignore[arg-type]
            dtype=torch.int64,
        )
        o_descriptors = torch.tensor(
            [[o_mapping.tensor.data_ptr(), 0, 0, q_heads, q_heads, chunk_rows]],
            dtype=torch.int64,
            device=state.device,
        )
        o_ready_descriptors = torch.tensor(
            [[o_ready_mapping.tensor.data_ptr(), 0, 1]],
            dtype=torch.int64,
            device=state.device,
        )
        o_publish_counters = torch.zeros(
            (2,), dtype=torch.int32, device=state.device
        )
        backend = state.attn_backends[0]
    else:
        attention_record = records[lane]
        assert attention_record is not None
        q_mapping = import_fabric_tensor(
            attention_record["q_handle"],  # type: ignore[arg-type]
            attention_record["q_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )
        kv_mapping = import_fabric_tensor(
            attention_record["kv_handle"],  # type: ignore[arg-type]
            attention_record["kv_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )
        q_ready_mapping = import_fabric_tensor(
            attention_record["q_ready_handle"],  # type: ignore[arg-type]
            attention_record["q_ready_shape"],  # type: ignore[arg-type]
            dtype=torch.int64,
        )
        cache_rows = int(kv_mapping.tensor.shape[2]) * int(kv_mapping.tensor.shape[3])
        q_descriptors = torch.tensor(
            [[q_mapping.tensor.data_ptr(), 0, 0, q_heads, q_heads, chunk_rows]],
            dtype=torch.int64,
            device=state.device,
        )
        kv_descriptors = torch.tensor(
            [[
                kv_mapping.tensor.data_ptr(),
                0,
                0,
                kv_heads,
                kv_heads,
                cache_rows,
                int(model_config.num_layers),
            ]],
            dtype=torch.int64,
            device=state.device,
        )
        q_ready_descriptors = torch.tensor(
            [[q_ready_mapping.tensor.data_ptr(), 0, 1]],
            dtype=torch.int64,
            device=state.device,
        )
        q_publish_counters = torch.zeros(
            (2,), dtype=torch.int32, device=state.device
        )
    timeout_record = torch.zeros((2,), dtype=torch.int64, device=state.device)
    actual_token = -1

    with torch.inference_mode(), torch.cuda.stream(state.stream):
        for start in range(0, sequence_length, chunk_rows):
            end = min(sequence_length, start + chunk_rows)
            rows = end - start
            req = Req(
                input_ids=prompt_ids[:end],
                table_idx=0,
                cached_len=start,
                output_len=1,
                uid=case_indices[lane],
                sampling_params=SamplingParams(),
                cache_handle=NaiveCacheHandle(),
            )
            batch = Batch(reqs=[req], phase="prefill")
            batch.padded_reqs = [req]
            batch.positions = torch.arange(
                start, end, dtype=torch.int32, device=state.device
            )
            batch.out_loc = torch.arange(
                start, end, dtype=torch.int32, device=state.device
            )
            if is_attention:
                batch.input_ids = torch.empty((0,), dtype=torch.int32, device=state.device)
                backend.prepare_metadata(batch)
                with ctx.forward_batch(batch):
                    _forward_role_split_attention(
                        batch,
                        num_layers=int(model_config.num_layers),
                        head_dim=head_dim,
                        q_slots=q_slots,
                        q_ready=q_ready,
                        o_descriptors=o_descriptors,
                        o_ready_descriptors=o_ready_descriptors,
                        o_publish_counters=o_publish_counters,
                        timeout_record=timeout_record,
                        backend=backend,
                    )
            else:
                assert model is not None and prompt_device is not None
                batch.input_ids = prompt_device[start:end]
                ctx.moe_num_token_non_padded = rows
                ctx.moe_deepep_dispatch_max_tokens_per_rank = rows
                with ctx.forward_batch(batch):
                    final_hidden = _forward_role_split_model(
                        model,
                        batch,
                        q_descriptors=q_descriptors,
                        kv_descriptors=kv_descriptors,
                        o_slots=o_slots,
                        q_ready_descriptors=q_ready_descriptors,
                        q_publish_counters=q_publish_counters,
                        o_ready=o_ready,
                        timeout_record=timeout_record,
                    )
                    selected = final_hidden[-1:] if end == sequence_length else None
                if selected is not None:
                    sampling_batch = Batch(reqs=[req], phase="decode")
                    sampling_batch.padded_reqs = [req]
                    with ctx.forward_batch(sampling_batch):
                        logits = model.forward_lm_head(selected)
                        actual_token = int(torch.argmax(logits, dim=-1).item())

    torch.cuda.synchronize(state.device)
    result = {
        "rank": rank,
        "role": "attention" if is_attention else "model",
        "case_index": case_indices[lane],
        "sequence_length": sequence_length,
        "actual_token": actual_token,
        "legacy_token": expected_token,
        "matches_legacy": actual_token == expected_token,
    }
    results: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(results, result)
    if rank == 0:
        model_results = [item for item in results if item and item["role"] == "model"]
        print("AFD_CHECKPOINT_ROLE_SPLIT_RESULTS " + json.dumps(model_results, sort_keys=True))
    if not is_attention:
        ctx.mlp_deepep_buffer.destroy()
    dist.barrier()
    state.shutdown()


def main() -> None:
    attention_mode = os.environ.get("AFD_CHECKPOINT_ATTENTION_MODE", "local")
    if attention_mode == "split-roles":
        _main_role_split()
        return
    if attention_mode == "runtime-split-local":
        _main_runtime_split_local()
        return
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 4:
        raise RuntimeError(f"checkpoint prefill smoke requires four ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ["MINISGL_DEVICE_INDEX"] = str(local_rank)

    model_path = os.environ["AFD_CHECKPOINT_MODEL_PATH"]
    alignment_path = os.environ["AFD_CHECKPOINT_ALIGNMENT_JSON"]
    legacy_sample_path = os.environ["AFD_CHECKPOINT_LEGACY_SAMPLE_JSON"]
    case_indices = tuple(
        int(value)
        for value in os.environ.get("AFD_CHECKPOINT_CASE_INDICES", "0,1,2,3").split(",")
    )
    if len(case_indices) != world_size:
        raise RuntimeError(
            "AFD_CHECKPOINT_CASE_INDICES must contain one index per rank, "
            f"got {case_indices}"
        )
    prompt_ids, expected_token = _load_case(
        alignment_path,
        legacy_sample_path,
        case_indices[rank],
    )
    sequence_length = int(prompt_ids.numel())
    attention_mode = os.environ.get("AFD_CHECKPOINT_ATTENTION_MODE", "local")
    if attention_mode not in ("local", "split-local", "split-peer"):
        raise RuntimeError(
            "AFD_CHECKPOINT_ATTENTION_MODE must be 'local', 'split-local', "
            "or 'split-peer', "
            f"got {attention_mode!r}"
        )
    page_size = 64
    chunk_rows = int(os.environ.get("AFD_CHECKPOINT_CHUNK_ROWS", "512"))
    num_pages = div_ceil(sequence_length + 1, page_size)
    tp_groups = tuple((worker_rank,) for worker_rank in range(world_size))
    config = AfdRuntimeConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="trtllm",
        moe_backend="fused",
        cuda_graph_max_bs=0,
        page_size=page_size,
        memory_ratio=0.9,
        max_extend_tokens=chunk_rows,
        max_seq_len_override=sequence_length + 1,
        num_page_override=num_pages,
        ep_size=world_size,
        dp_size=world_size,
        dp_rank=rank,
        cache_type="naive",
        distributed_host=os.environ["MASTER_ADDR"],
        distributed_port=int(os.environ["MASTER_PORT"]),
        distributed_rank=rank,
        distributed_world_size=world_size,
        distributed_tp_groups=tp_groups,
        afd_moe_a2a_backend="deepep",
        afd_moe_runner_backend="deep_gemm",
        afd_model_placement=(
            "fmha-only" if attention_mode == "split-peer" else "legacy"
        ),
    )

    state = AfdAttentionState(config)
    ctx = state.ctx
    ctx.server_args = config
    ctx.replica_id = rank
    moe_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    set_rope_device(state.device)
    with torch.device("meta"), torch_dtype(config.dtype):
        model = Qwen3MoeForCausalLM(config.model_config)
    weights = {
        key: (
            value
            if value.dtype == torch.float8_e4m3fn
            or (value.dtype == torch.float32 and key.endswith("_scale"))
            else value.to(config.dtype)
        )
        for key, value in load_weight(model_path, state.device)
    }
    model.load_state_dict(weights)
    del weights

    local_experts = div_ceil(int(config.model_config.num_experts), world_size)
    ctx.mlp_deepep_buffer = DeepEPMoeElasticBuffer(
        group=moe_group,
        num_max_dispatch_tokens_per_rank=chunk_rows,
        hidden_size=int(config.model_config.hidden_size),
        num_experts=local_experts * world_size,
        top_k=int(config.model_config.num_experts_per_tok),
    )
    ctx.page_table[0, :sequence_length].copy_(
        torch.arange(sequence_length, dtype=torch.int32, device=state.device)
    )
    prompt_device = prompt_ids.pin_memory().to(state.device, non_blocking=True)
    backend = state.attn_backends[0]
    actual_token = -1

    if attention_mode in ("split-local", "split-peer"):
        model_config = config.model_config
        q_heads = int(model_config.num_qo_heads)
        kv_heads = int(model_config.num_kv_heads)
        head_dim = int(model_config.head_dim)
        if attention_mode == "split-peer":
            q_owner = allocate_fabric_tensor(
                (2, chunk_rows, q_heads, head_dim), dtype=torch.bfloat16
            )
            o_owner = allocate_fabric_tensor(
                (2, chunk_rows, q_heads, head_dim), dtype=torch.bfloat16
            )
            q_ready_owner = allocate_fabric_tensor((2, 1), dtype=torch.int64)
            o_ready_owner = allocate_fabric_tensor((2, 1), dtype=torch.int64)
            q_slots = q_owner.tensor
            o_slots = o_owner.tensor
            q_ready = q_ready_owner.tensor
            o_ready = o_ready_owner.tensor
            q_ready.zero_()
            o_ready.zero_()
        else:
            q_slots = torch.empty(
                (2, chunk_rows, q_heads, head_dim),
                dtype=torch.bfloat16,
                device=state.device,
            )
            o_slots = None
            q_ready = torch.zeros((2, 1), dtype=torch.int64, device=state.device)
            o_ready = torch.zeros_like(q_ready)
        kv_buffer = ctx.kv_cache._kv_buffer
        cache_rows = int(kv_buffer.shape[2]) * int(kv_buffer.shape[3])
        if attention_mode == "split-peer":
            kv_owner = state.fabric_kv
            torch.cuda.synchronize(state.device)
            record = {
                "q_handle": q_owner.handle,
                "q_shape": tuple(q_slots.shape),
                "q_ready_handle": q_ready_owner.handle,
                "q_ready_shape": tuple(q_ready.shape),
                "kv_handle": kv_owner.handle,
                "kv_shape": tuple(kv_buffer.shape),
                "o_handle": o_owner.handle,
                "o_shape": tuple(o_slots.shape),
                "o_ready_handle": o_ready_owner.handle,
                "o_ready_shape": tuple(o_ready.shape),
            }
            records: list[dict[str, object] | None] = [None] * world_size
            dist.all_gather_object(records, record)
            peer_delta = int(os.environ.get("AFD_CHECKPOINT_PEER_DELTA", "1"))
            if peer_delta <= 0 or peer_delta >= world_size:
                raise RuntimeError(
                    f"AFD_CHECKPOINT_PEER_DELTA must be in [1, {world_size}), "
                    f"got {peer_delta}"
                )
            attention_rank = (rank + peer_delta) % world_size
            producer_rank = (rank - peer_delta) % world_size
            attention_record = records[attention_rank]
            producer_record = records[producer_rank]
            assert attention_record is not None and producer_record is not None
            q_mapping = import_fabric_tensor(
                attention_record["q_handle"],  # type: ignore[arg-type]
                attention_record["q_shape"],  # type: ignore[arg-type]
                dtype=torch.bfloat16,
            )
            kv_mapping = import_fabric_tensor(
                attention_record["kv_handle"],  # type: ignore[arg-type]
                attention_record["kv_shape"],  # type: ignore[arg-type]
                dtype=torch.bfloat16,
            )
            o_mapping = import_fabric_tensor(
                producer_record["o_handle"],  # type: ignore[arg-type]
                producer_record["o_shape"],  # type: ignore[arg-type]
                dtype=torch.bfloat16,
            )
            q_ready_mapping = import_fabric_tensor(
                attention_record["q_ready_handle"],  # type: ignore[arg-type]
                attention_record["q_ready_shape"],  # type: ignore[arg-type]
                dtype=torch.int64,
            )
            o_ready_mapping = import_fabric_tensor(
                producer_record["o_ready_handle"],  # type: ignore[arg-type]
                producer_record["o_ready_shape"],  # type: ignore[arg-type]
                dtype=torch.int64,
            )
            q_destination = q_mapping.tensor
            kv_destination = kv_mapping.tensor
            o_descriptors = torch.tensor(
                [[
                    o_mapping.tensor.data_ptr(),
                    0,
                    0,
                    q_heads,
                    q_heads,
                    chunk_rows,
                ]],
                dtype=torch.int64,
                device=state.device,
            )
            signals = _transport_signals(
                q_ready=q_ready,
                q_ready_destination=q_ready_mapping.tensor,
                o_ready=o_ready,
                o_ready_destination=o_ready_mapping.tensor,
                device=state.device,
            )
        else:
            q_destination = q_slots
            kv_destination = kv_buffer
            o_descriptors = None
            signals = _transport_signals(
                q_ready=q_ready,
                q_ready_destination=q_ready,
                o_ready=o_ready,
                o_ready_destination=o_ready,
                device=state.device,
            )
        q_descriptors = torch.tensor(
            [[q_destination.data_ptr(), 0, 0, q_heads, q_heads, chunk_rows]],
            dtype=torch.int64,
            device=state.device,
        )
        kv_descriptors = torch.tensor(
            [[
                kv_destination.data_ptr(),
                0,
                0,
                kv_heads,
                kv_heads,
                cache_rows,
                int(model_config.num_layers),
            ]],
            dtype=torch.int64,
            device=state.device,
        )
        timeout_record = torch.zeros((2,), dtype=torch.int64, device=state.device)

    with torch.inference_mode(), torch.cuda.stream(state.stream):
        for start in range(0, sequence_length, chunk_rows):
            end = min(sequence_length, start + chunk_rows)
            rows = end - start
            req = Req(
                input_ids=prompt_ids[:end],
                table_idx=0,
                cached_len=start,
                output_len=1,
                uid=case_indices[rank],
                sampling_params=SamplingParams(),
                cache_handle=NaiveCacheHandle(),
            )
            batch = Batch(reqs=[req], phase="prefill")
            batch.padded_reqs = [req]
            batch.input_ids = prompt_device[start:end]
            batch.positions = torch.arange(
                start, end, dtype=torch.int32, device=state.device
            )
            batch.out_loc = torch.arange(
                start, end, dtype=torch.int32, device=state.device
            )
            backend.prepare_metadata(batch)
            ctx.moe_num_token_non_padded = rows
            ctx.moe_deepep_dispatch_max_tokens_per_rank = rows
            with ctx.forward_batch(batch):
                if attention_mode in ("split-local", "split-peer"):
                    final_hidden = _forward_split_attention(
                        model,
                        batch,
                        q_slots=q_slots,
                        q_descriptors=q_descriptors,
                        kv_descriptors=kv_descriptors,
                        signals=signals,
                        timeout_record=timeout_record,
                        backend=backend,
                        o_slots=o_slots,
                        o_descriptors=o_descriptors,
                    )
                else:
                    final_hidden = model.model.forward(batch.input_ids)
                selected = final_hidden[-1:] if end == sequence_length else None
            if selected is not None:
                sampling_batch = Batch(reqs=[req], phase="decode")
                sampling_batch.padded_reqs = [req]
                with ctx.forward_batch(sampling_batch):
                    logits = model.forward_lm_head(selected)
                    actual_token = int(torch.argmax(logits, dim=-1).item())

    torch.cuda.synchronize(state.device)
    result = {
        "rank": rank,
        "case_index": case_indices[rank],
        "sequence_length": sequence_length,
        "attention_mode": attention_mode,
        "actual_token": actual_token,
        "legacy_token": expected_token,
        "matches_legacy": actual_token == expected_token,
    }
    results: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(results, result)
    if rank == 0:
        print("AFD_CHECKPOINT_LOCAL_PREFILL_RESULTS " + json.dumps(results, sort_keys=True))
    require_legacy = os.environ.get("AFD_CHECKPOINT_REQUIRE_LEGACY", "1") == "1"
    if require_legacy and actual_token != expected_token:
        raise AssertionError(
            "checkpoint local-prefill first token differs from the passing legacy run: "
            f"case={case_indices[rank]} actual={actual_token} legacy={expected_token}"
        )
    dist.barrier()
    ctx.mlp_deepep_buffer.destroy()
    state.shutdown()


if __name__ == "__main__":
    main()
