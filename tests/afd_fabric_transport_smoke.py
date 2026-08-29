from __future__ import annotations

import os

import torch
import torch.distributed as dist

from minisgl.kernel.afd_fmha_transport import (
    ensure_afd_fmha_transport_built,
    publish_qkv,
    quantize_publish_o_fp8_release_turn,
    wait_ready,
)
from minisgl.kernel.fabric_memory import (
    allocate_fabric_tensors,
    import_fabric_tensor,
)


def _descriptor(rows: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64, device=device).view(-1, width)


def _prebuild() -> None:
    """Build both extensions once before the four ranks enter collectives."""
    torch.cuda.set_device(0)
    ensure_afd_fmha_transport_built(128)
    from minisgl.kernel import deepep_moe
    from minisgl.kernel import deepgemm as deep_gemm

    deepep_moe._load_extension()
    deep_gemm.per_token_cast_to_fp8(
        torch.zeros((4, 64 * 128), dtype=torch.bfloat16, device="cuda"),
        use_ue8m0=True,
        gran_k=128,
        use_packed_ue8m0=True,
    )
    print("AFD_FABRIC_TRANSPORT_PREBUILD_OK head_dim=128 fabric_extension=ready")


def main() -> None:
    if os.environ.get("AFD_FABRIC_SMOKE_PREBUILD") == "1":
        _prebuild()
        return

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size not in (2, 4):
        raise RuntimeError(f"fabric smoke requires two or four ranks, got {world_size}")
    local_rank = int(os.environ["SLURM_LOCALID"])
    visible_gpus = int(torch.cuda.device_count())
    device_index = 0 if visible_gpus == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )

    pair_count = world_size // 2
    is_attention = rank < pair_count
    pair = rank if is_attention else rank - pair_count
    rows = int(os.environ.get("AFD_FABRIC_SMOKE_ROWS", "4"))
    iterations = int(os.environ.get("AFD_FABRIC_SMOKE_ITERATIONS", "1"))
    if rows < 1 or iterations < 1:
        raise RuntimeError(
            f"fabric smoke rows and iterations must be positive, got {rows}/{iterations}"
        )
    q_heads, kv_heads, head_dim = 64, 8, 128
    o_scale_packed_groups = (q_heads + 3) // 4
    o_scale_row_stride = ((rows + 3) // 4) * 4
    o_scale_slot_elements = o_scale_row_stride * o_scale_packed_groups
    num_layers = 3
    cache_rows = max(16, rows * 3 + 17)
    record: dict[str, object] = {"rank": rank}
    if is_attention:
        q_owner, kv_owner, q_ready_owner = allocate_fabric_tensors(
            (
                ((2, rows, q_heads, head_dim), torch.bfloat16),
                (
                    (2, num_layers, cache_rows, 1, kv_heads, head_dim),
                    torch.bfloat16,
                ),
                ((2, 1), torch.int64),
            )
        )
        q_ready_owner.tensor.zero_()
        o_publish_counters = torch.zeros((2,), dtype=torch.int32, device=device)
        o_quantization_counters = torch.zeros((2,), dtype=torch.int32, device=device)
        o_fp8_staging = torch.empty(
            (2, rows, q_heads * head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        o_scale_staging = torch.empty(
            (2, rows, q_heads), dtype=torch.uint8, device=device
        )
        turn = torch.zeros((1,), dtype=torch.int64, device=device)
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
    else:
        o_owner, o_scale_owner, o_ready_owner = allocate_fabric_tensors(
            (
                ((2, rows, q_heads, head_dim), torch.float8_e4m3fn),
                ((2, o_scale_slot_elements), torch.int32),
                ((2, 1), torch.int64),
            )
        )
        o_ready_owner.tensor.zero_()
        q_publish_counters = torch.zeros((2,), dtype=torch.int32, device=device)
        model_handle = o_owner.handle
        record.update(
            o_handle=model_handle,
            o_shape=tuple(o_owner.tensor.shape),
            o_offset=o_owner.byte_offset,
            o_scale_handle=model_handle,
            o_scale_shape=tuple(o_scale_owner.tensor.shape),
            o_scale_offset=o_scale_owner.byte_offset,
            o_ready_handle=model_handle,
            o_ready_shape=tuple(o_ready_owner.tensor.shape),
            o_ready_offset=o_ready_owner.byte_offset,
        )

    # Finish local initialization before publishing handles to remote writers.
    torch.cuda.synchronize()
    records: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(records, record)
    timeout = torch.zeros((2,), dtype=torch.int64, device=device)
    q_map = q_ready_map = kv_map = o_map = o_scale_map = o_ready_map = None
    if not is_attention:
        ag = records[pair]
        assert ag is not None
        q_map = import_fabric_tensor(
            ag["q_handle"],  # type: ignore[arg-type]
            ag["q_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
            byte_offset=ag["q_offset"],  # type: ignore[arg-type]
        )
        kv_map = import_fabric_tensor(
            ag["kv_handle"],  # type: ignore[arg-type]
            ag["kv_shape"],  # type: ignore[arg-type]
            dtype=torch.bfloat16,
            byte_offset=ag["kv_offset"],  # type: ignore[arg-type]
            allocation=q_map.allocation,
        )
        q_ready_map = import_fabric_tensor(
            ag["q_ready_handle"],  # type: ignore[arg-type]
            ag["q_ready_shape"],  # type: ignore[arg-type]
            dtype=torch.int64,
            byte_offset=ag["q_ready_offset"],  # type: ignore[arg-type]
            allocation=q_map.allocation,
        )
    else:
        eg = records[pair + pair_count]
        assert eg is not None
        o_map = import_fabric_tensor(
            eg["o_handle"],  # type: ignore[arg-type]
            eg["o_shape"],  # type: ignore[arg-type]
            dtype=torch.float8_e4m3fn,
            byte_offset=eg["o_offset"],  # type: ignore[arg-type]
        )
        o_scale_map = import_fabric_tensor(
            eg["o_scale_handle"],  # type: ignore[arg-type]
            eg["o_scale_shape"],  # type: ignore[arg-type]
            dtype=torch.int32,
            byte_offset=eg["o_scale_offset"],  # type: ignore[arg-type]
            allocation=o_map.allocation,
        )
        o_ready_map = import_fabric_tensor(
            eg["o_ready_handle"],  # type: ignore[arg-type]
            eg["o_ready_shape"],  # type: ignore[arg-type]
            dtype=torch.int64,
            byte_offset=eg["o_ready_offset"],  # type: ignore[arg-type]
            allocation=o_map.allocation,
        )

    out_loc = (
        torch.arange(rows, dtype=torch.int32, device=device) * 3 + 5 + pair
    ) % cache_rows
    for epoch in range(iterations):
        slot = epoch & 1
        layer = epoch % num_layers
        q_value = float(1 + pair + epoch * 2)
        k_value = float(11 + pair + epoch * 2)
        v_value = float(21 + pair + epoch * 2)
        o_value = float(31 + pair + epoch * 2)

        if not is_attention:
            assert q_map is not None and q_ready_map is not None and kv_map is not None
            q = torch.full(
                (rows, q_heads * head_dim), q_value, device=device, dtype=torch.bfloat16
            )
            k = torch.full(
                (rows, kv_heads * head_dim), k_value, device=device, dtype=torch.bfloat16
            )
            v = torch.full(
                (rows, kv_heads * head_dim), v_value, device=device, dtype=torch.bfloat16
            )
            q.view(-1)[::257] = -0.0
            k.view(-1)[::131] = -0.0
            v.view(-1)[::127] = 0.0
            q_desc = _descriptor(
                [[q_map.tensor.data_ptr(), 0, 0, q_heads, q_heads, rows, 0]],
                7,
                device,
            )
            kv_desc = _descriptor(
                [
                    [
                        kv_map.tensor.data_ptr(),
                        0,
                        0,
                        kv_heads,
                        kv_heads,
                        cache_rows,
                        num_layers,
                        0,
                    ]
                ],
                8,
                device,
            )
            q_ready_desc = _descriptor(
                [[q_ready_map.tensor.data_ptr(), 0, 1]], 3, device
            )
            publish_qkv(
                q,
                k,
                v,
                out_loc,
                torch.tensor((0, rows), dtype=torch.int64, device=device),
                q_desc,
                kv_desc,
                q_ready_desc,
                q_publish_counters[slot : slot + 1],
                layer=layer,
                slot=slot,
                head_dim=head_dim,
                ready_value=epoch + 1,
            )
        else:
            wait_ready(
                q_ready_owner.tensor[slot],
                timeout,
                timeout_ms=30_000,
                head_dim=head_dim,
                expected_ready=epoch + 1,
            )
            torch.cuda.synchronize()
            expected_q = torch.full_like(q_owner.tensor[slot], q_value)
            expected_q.view(-1)[::257] = -0.0
            expected_k = torch.full(
                (rows, kv_heads, head_dim),
                k_value,
                device=device,
                dtype=torch.bfloat16,
            )
            expected_v = torch.full_like(expected_k, v_value)
            expected_k.view(-1)[::131] = -0.0
            expected_v.view(-1)[::127] = 0.0
            torch.testing.assert_close(
                q_owner.tensor[slot].view(torch.int16),
                expected_q.view(torch.int16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                kv_owner.tensor[0, layer]
                .view(cache_rows, kv_heads, head_dim)[out_loc]
                .view(torch.int16),
                expected_k.view(torch.int16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                kv_owner.tensor[1, layer]
                .view(cache_rows, kv_heads, head_dim)[out_loc]
                .view(torch.int16),
                expected_v.view(torch.int16),
                rtol=0,
                atol=0,
            )

        dist.barrier()
        if is_attention:
            assert (
                o_map is not None
                and o_scale_map is not None
                and o_ready_map is not None
            )
            o = torch.full(
                (rows, q_heads * head_dim), o_value, device=device, dtype=torch.bfloat16
            )
            o.view(-1)[::251] = -0.0
            o_desc = _descriptor(
                [
                    [
                        o_map.tensor.data_ptr(),
                        o_scale_map.tensor.data_ptr(),
                        0,
                        0,
                        q_heads,
                        q_heads,
                        rows,
                        o_scale_slot_elements,
                        0,
                        1,
                    ]
                ],
                10,
                device,
            )
            o_ready_desc = _descriptor(
                [[o_ready_map.tensor.data_ptr(), 0, 1]], 3, device
            )
            quantize_publish_o_fp8_release_turn(
                o,
                o_fp8_staging[slot],
                o_scale_staging[slot],
                o_desc,
                o_ready_desc,
                o_publish_counters[slot : slot + 1],
                o_quantization_counters[slot : slot + 1],
                turn,
                slot=slot,
                destination_source_stride=rows,
                next_turn=epoch + 1,
                head_dim=head_dim,
                ready_value=epoch + 1,
            )
            torch.cuda.synchronize()
            if int(turn.item()) != epoch + 1:
                raise RuntimeError(
                    f"fused O publication did not release turn {epoch + 1}"
                )
        else:
            wait_ready(
                o_ready_owner.tensor[slot],
                timeout,
                timeout_ms=30_000,
                head_dim=head_dim,
                expected_ready=epoch + 1,
            )
            torch.cuda.synchronize()
            from minisgl.kernel import deepgemm as deep_gemm

            expected_o_bf16 = torch.full(
                (rows, q_heads * head_dim),
                o_value,
                device=device,
                dtype=torch.bfloat16,
            )
            expected_o_bf16.view(-1)[::251] = -0.0
            expected_o, expected_scale = deep_gemm.per_token_cast_to_fp8(
                expected_o_bf16,
                use_ue8m0=True,
                gran_k=128,
                use_packed_ue8m0=True,
            )
            torch.testing.assert_close(
                o_owner.tensor[slot].view(torch.uint8),
                expected_o.view(rows, q_heads, head_dim).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            received_scale = torch.as_strided(
                o_scale_owner.tensor[slot],
                size=(rows, o_scale_packed_groups),
                stride=(1, o_scale_row_stride),
            )
            torch.testing.assert_close(
                received_scale, expected_scale, rtol=0, atol=0
            )
        dist.barrier()

    if rank == 0:
        print(
            "AFD_FABRIC_TRANSPORT_SMOKE_OK "
            f"pairs={pair_count} rows={rows} iterations={iterations} "
            f"q_heads={q_heads} kv_heads={kv_heads}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
