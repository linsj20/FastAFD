from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from minisgl.attention.trtllm import TensorRTLLMBackend
from minisgl.core import Batch, Context, Req, SamplingParams, set_global_ctx
from minisgl.distributed import set_tp_info
from minisgl.kernel.afd_fmha_transport import publish_qkv, wait_ready
from minisgl.kernel.fabric_memory import allocate_fabric_tensors, import_fabric_tensor
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.naive_cache import NaiveCacheHandle


def _prepared_qkv(
    start: int,
    rows: int,
    *,
    phase_offset: float,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(start, start + rows, device=device, dtype=torch.float32)
    dims = torch.arange(head_dim, device=device, dtype=torch.float32)
    q_head_ids = torch.arange(q_heads, device=device, dtype=torch.float32)
    kv_head_ids = torch.arange(kv_heads, device=device, dtype=torch.float32)
    q_phase = (
        positions[:, None, None] * 0.0007
        + q_head_ids[None, :, None] * 0.031
        + dims[None, None, :] * 0.0023
        + phase_offset
    )
    k_phase = (
        positions[:, None, None] * 0.0009
        + kv_head_ids[None, :, None] * 0.047
        + dims[None, None, :] * 0.0017
        + phase_offset
    )
    v_phase = (
        positions[:, None, None] * 0.0005
        + kv_head_ids[None, :, None] * 0.053
        + dims[None, None, :] * 0.0011
        + phase_offset
    )
    return (
        (torch.sin(q_phase) * 0.25).to(torch.bfloat16).view(rows, -1),
        (torch.cos(k_phase) * 0.25).to(torch.bfloat16).view(rows, -1),
        (torch.sin(v_phase) * 0.5).to(torch.bfloat16).view(rows, -1),
    )


def _last_token_reference(
    q_last: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    kv_heads = int(keys.shape[1])
    q_heads = int(q_last.shape[0])
    heads_per_kv = q_heads // kv_heads
    q_grouped = q_last.float().view(kv_heads, heads_per_kv, -1)
    scores = torch.einsum("ghd,sgd->ghs", q_grouped, keys.float())
    scores.mul_(q_last.shape[-1] ** -0.5)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("ghs,sgd->ghd", probabilities, values.float()).view(
        q_heads, -1
    )


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size not in (2, 4):
        raise RuntimeError(f"FMHA prefill smoke requires two or four ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )
    set_tp_info(rank=0, size=1)

    pair_count = world_size // 2
    is_attention = rank < pair_count
    pair = rank if is_attention else rank - pair_count

    q_heads = 64
    kv_heads = 4
    head_dim = 128
    page_size = 64
    chunk_rows = int(os.environ.get("AFD_FMHA_PREFILL_CHUNK_ROWS", "512"))
    sequence_length = int(
        os.environ.get("AFD_FMHA_PREFILL_SEQUENCE_LENGTH", "8192")
    )
    num_layers = int(os.environ.get("AFD_FMHA_PREFILL_NUM_LAYERS", "1"))
    if (
        chunk_rows < 1
        or sequence_length < chunk_rows
        or sequence_length % chunk_rows
        or num_layers < 1
    ):
        raise RuntimeError(
            "FMHA prefill smoke requires positive layers and a positive sequence "
            "length exactly divisible by chunk rows, got "
            f"layers={num_layers} sequence/chunk={sequence_length}/{chunk_rows}"
        )
    num_pages = sequence_length // page_size + 1
    record: dict[str, object] = {"rank": rank}

    if is_attention:
        ctx = Context(page_size=page_size)
        set_global_ctx(ctx)
        q_owner, kv_owner, q_ready_owner = allocate_fabric_tensors(
            (
                ((2, chunk_rows, q_heads, head_dim), torch.bfloat16),
                (
                    (2, num_layers, num_pages, page_size, kv_heads, head_dim),
                    torch.bfloat16,
                ),
                ((2, 1), torch.int64),
            )
        )
        q_ready_owner.tensor.zero_()
        ctx.page_table = torch.zeros(
            (2, sequence_length), dtype=torch.int32, device=device
        )
        ctx.page_table[0].copy_(
            torch.arange(sequence_length, dtype=torch.int32, device=device)
        )
        model_config = SimpleNamespace(
            num_qo_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
        )
        ctx.kv_cache = create_kvcache_pool(
            model_config=model_config,
            num_pages=num_pages,
            page_size=page_size,
            dtype=torch.bfloat16,
            device=device,
            storage=kv_owner.tensor,
        )
        backend = TensorRTLLMBackend(model_config)
        ctx.attn_backend = backend
        attention_handle = q_owner.handle
        record.update(
            q_handle=attention_handle,
            q_shape=tuple(q_owner.tensor.shape),
            q_offset=q_owner.byte_offset,
            q_ready_handle=attention_handle,
            q_ready_shape=tuple(q_ready_owner.tensor.shape),
            q_ready_offset=q_ready_owner.byte_offset,
            kv_handle=attention_handle,
            kv_shape=tuple(kv_owner.tensor.shape),
            kv_offset=kv_owner.byte_offset,
        )

    torch.cuda.synchronize(device)
    records: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(records, record)

    if not is_attention:
        attention_record = records[pair]
        assert attention_record is not None
        q_mapping = import_fabric_tensor(
            attention_record["q_handle"],  # type: ignore[arg-type]
            attention_record["q_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
            byte_offset=attention_record["q_offset"],  # type: ignore[arg-type]
        )
        kv_mapping = import_fabric_tensor(
            attention_record["kv_handle"],  # type: ignore[arg-type]
            attention_record["kv_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
            byte_offset=attention_record["kv_offset"],  # type: ignore[arg-type]
            allocation=q_mapping.allocation,
        )
        q_ready_mapping = import_fabric_tensor(
            attention_record["q_ready_handle"],  # type: ignore[arg-type]
            attention_record["q_ready_shape"],  # type: ignore[arg-type]
            dtype=torch.int64,
            byte_offset=attention_record["q_ready_offset"],  # type: ignore[arg-type]
            allocation=q_mapping.allocation,
        )
        q_descriptors = torch.tensor(
            [
                [
                    q_mapping.tensor.data_ptr(),
                    0,
                    0,
                    q_heads,
                    q_heads,
                    chunk_rows,
                    0,
                ]
            ],
            dtype=torch.int64,
            device=device,
        )
        kv_descriptors = torch.tensor(
            [
                [
                    kv_mapping.tensor.data_ptr(),
                    0,
                    0,
                    kv_heads,
                    kv_heads,
                    num_pages * page_size,
                    num_layers,
                    0,
                ]
            ],
            dtype=torch.int64,
            device=device,
        )
        q_ready_descriptors = torch.tensor(
            [[q_ready_mapping.tensor.data_ptr(), 0, 1]],
            dtype=torch.int64,
            device=device,
        )
        q_publish_counters = torch.zeros((2,), dtype=torch.int32, device=device)
    else:
        timeout = torch.zeros((2,), dtype=torch.int64, device=device)
        reference_keys: list[torch.Tensor] = []
        reference_values: list[torch.Tensor] = []
        max_abs = torch.zeros((), dtype=torch.float32, device=device)
        max_rel = torch.zeros((), dtype=torch.float32, device=device)

    for chunk_index, start in enumerate(range(0, sequence_length, chunk_rows)):
        slot = chunk_index & 1
        q, k, v = _prepared_qkv(
            start,
            chunk_rows,
            phase_offset=pair * 0.19,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            device=device,
        )
        out_loc = torch.arange(
            start,
            start + chunk_rows,
            dtype=torch.int32,
            device=device,
        )
        if is_attention:
            input_ids = torch.zeros(start + chunk_rows, dtype=torch.int32)
            req = Req(
                input_ids=input_ids,
                table_idx=0,
                cached_len=start,
                output_len=1,
                uid=0,
                sampling_params=SamplingParams(),
                cache_handle=NaiveCacheHandle(),
            )
            batch = Batch(reqs=[req], phase="prefill")
            batch.padded_reqs = [req]
            backend.prepare_metadata(batch)
            reference_keys.append(k.view(chunk_rows, kv_heads, head_dim))
            reference_values.append(v.view(chunk_rows, kv_heads, head_dim))
            expected = _last_token_reference(
                q.view(chunk_rows, q_heads, head_dim)[-1],
                torch.cat(reference_keys, dim=0),
                torch.cat(reference_values, dim=0),
            )

        for layer in range(num_layers):
            ready_value = chunk_index * num_layers + layer + 1
            if not is_attention:
                publish_qkv(
                    q,
                    k,
                    v,
                    out_loc,
                    torch.tensor((0, chunk_rows), dtype=torch.int64, device=device),
                    q_descriptors,
                    kv_descriptors,
                    q_ready_descriptors,
                    q_publish_counters[slot : slot + 1],
                    layer=layer,
                    slot=slot,
                    head_dim=head_dim,
                    ready_value=ready_value,
                )
                torch.cuda.synchronize(device)
            else:
                wait_ready(
                    q_ready_owner.tensor[slot],
                    timeout,
                    timeout_ms=30_000,
                    head_dim=head_dim,
                    expected_ready=ready_value,
                )
                actual = backend.forward_prepared(q_owner.tensor[slot], layer, batch)
                difference = (actual[-1].float() - expected).abs()
                max_abs = torch.maximum(max_abs, difference.max())
                max_rel = torch.maximum(
                    max_rel,
                    (difference / expected.abs().clamp_min(0.125)).max(),
                )
                torch.cuda.synchronize(device)
            dist.barrier()

    if is_attention:
        abs_value = float(max_abs)
        rel_value = float(max_rel)
        if abs_value > 0.02 or rel_value > 0.06:
            raise AssertionError(
                "FMHA remote prefill reference mismatch: "
                f"max_abs={abs_value} max_rel={rel_value}"
            )
        print(
            "AFD_FMHA_PREFILL_REFERENCE_SMOKE_OK "
            f"pair={pair} sequence_length={sequence_length} chunk_rows={chunk_rows} "
            f"num_layers={num_layers} "
            f"q_heads={q_heads} kv_heads={kv_heads} page_size={page_size} "
            f"max_abs={abs_value:.8f} max_rel={rel_value:.8f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
