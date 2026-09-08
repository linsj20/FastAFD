from __future__ import annotations

import os

import torch
import torch.distributed as dist

from minisgl.kernel.afd_fmha_transport import (
    ensure_afd_fmha_transport_built,
    publish_o,
    publish_o_release_turn,
    publish_qkv,
    quantize_publish_o_fp8_release_turn,
    release_turn,
    signal_direct_o_release_turn,
    wait_ready,
)
from minisgl.kernel.fabric_memory import (
    allocate_fabric_tensors,
    import_fabric_tensor,
)
from minisgl.kernel.qwen35_qkv_transport import (
    ensure_qwen35_qkv_transport_built,
    publish_qwen35_linear_input,
    publish_qwen35_qkv,
    qwen35_shared_add_rmsnorm,
)


def _descriptor(rows: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64, device=device).view(-1, width)


def _direct_o_gemm_operands(
    *,
    rows: int,
    output_width: int,
    input_width: int,
    epoch: int,
    pair: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    from minisgl.kernel import deepgemm as deep_gemm

    input_values = torch.arange(
        rows * input_width, dtype=torch.float32, device=device
    ).view(rows, input_width)
    weight_values = torch.arange(
        output_width * input_width, dtype=torch.float32, device=device
    ).view(output_width, input_width)
    input_bf16 = (
        ((input_values + epoch * 7 + pair * 3).remainder_(31) - 15.0)
        .mul_(0.03125)
        .to(torch.bfloat16)
    )
    weight_bf16 = (
        ((weight_values + epoch * 11 + pair * 5).remainder_(29) - 14.0)
        .mul_(0.03125)
        .to(torch.bfloat16)
    )
    input_fp8, input_scale = deep_gemm.per_token_cast_to_fp8(
        input_bf16,
        use_ue8m0=True,
        gran_k=128,
        use_packed_ue8m0=True,
    )
    weight_fp8, weight_scale = deep_gemm.per_block_cast_to_fp8(
        weight_bf16,
        use_ue8m0=True,
        gran_k=128,
    )
    weight_scale = deep_gemm.transform_sf_into_required_layout(
        weight_scale.unsqueeze(0),
        output_width,
        input_width,
        (1, 128, 128),
        1,
        False,
        False,
    ).squeeze(0)
    return (input_fp8, input_scale), (weight_fp8, weight_scale)


def _launch_direct_o_gemm(
    operands: tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    output: torch.Tensor,
) -> None:
    from minisgl.kernel import deepgemm as deep_gemm

    deep_gemm.fp8_gemm_nt(
        operands[0],
        operands[1],
        output,
        disable_ue8m0_cast=False,
    )


def _run_direct_o_gemm(
    *,
    rows: int,
    output_width: int,
    input_width: int,
    epoch: int,
    pair: int,
    device: torch.device,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the Qwen3.5-style FP8 output GEMM into an optional fabric tensor."""
    operands = _direct_o_gemm_operands(
        rows=rows,
        output_width=output_width,
        input_width=input_width,
        epoch=epoch,
        pair=pair,
        device=device,
    )
    if output is None:
        output = torch.empty(
            (rows, output_width), dtype=torch.bfloat16, device=device
        )
    elif (
        tuple(output.shape) != (rows, output_width)
        or output.dtype != torch.bfloat16
        or not output.is_cuda
        or not output.is_contiguous()
    ):
        raise ValueError(
            "direct O GEMM output must be contiguous BF16 with shape "
            f"{(rows, output_width)} in CUDA memory, got shape={tuple(output.shape)} "
            f"dtype={output.dtype} device={output.device} "
            f"contiguous={output.is_contiguous()}"
        )
    _launch_direct_o_gemm(operands, output)
    return output


def _prebuild() -> None:
    """Build both extensions once before the four ranks enter collectives."""
    torch.cuda.set_device(0)
    head_dim = int(os.environ.get("AFD_FABRIC_SMOKE_HEAD_DIM", "128"))
    ensure_afd_fmha_transport_built(128)
    if head_dim != 128:
        ensure_afd_fmha_transport_built(head_dim)
    if os.environ.get("AFD_FABRIC_SMOKE_QWEN35_QKV") == "1":
        ensure_qwen35_qkv_transport_built()
    from minisgl.kernel import deepep_moe
    from minisgl.kernel import deepgemm as deep_gemm

    deepep_moe._load_extension()
    q_heads = int(os.environ.get("AFD_FABRIC_SMOKE_Q_HEADS", "64"))
    deep_gemm.per_token_cast_to_fp8(
        torch.zeros((4, q_heads * head_dim), dtype=torch.bfloat16, device="cuda"),
        use_ue8m0=True,
        gran_k=128,
        use_packed_ue8m0=True,
    )
    if os.environ.get("AFD_FABRIC_SMOKE_DIRECT_O") == "1":
        _run_direct_o_gemm(
            rows=int(os.environ.get("AFD_FABRIC_SMOKE_ROWS", "3")),
            output_width=q_heads * head_dim,
            input_width=int(
                os.environ.get("AFD_FABRIC_SMOKE_DIRECT_O_INPUT_WIDTH", "1024")
            ),
            epoch=0,
            pair=0,
            device=torch.device("cuda", 0),
        )
    print(
        "AFD_FABRIC_TRANSPORT_PREBUILD_OK "
        f"head_dims=128,{head_dim} fabric_extension=ready"
    )


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
    q_heads = int(os.environ.get("AFD_FABRIC_SMOKE_Q_HEADS", "64"))
    kv_heads = int(os.environ.get("AFD_FABRIC_SMOKE_KV_HEADS", "8"))
    head_dim = int(os.environ.get("AFD_FABRIC_SMOKE_HEAD_DIM", "128"))
    qwen35_qkv = os.environ.get("AFD_FABRIC_SMOKE_QWEN35_QKV") == "1"
    direct_o = os.environ.get("AFD_FABRIC_SMOKE_DIRECT_O") == "1"
    direct_o_timing = os.environ.get("AFD_FABRIC_SMOKE_DIRECT_O_TIMING") == "1"
    direct_o_input_width = int(
        os.environ.get("AFD_FABRIC_SMOKE_DIRECT_O_INPUT_WIDTH", "1024")
    )
    if qwen35_qkv and head_dim != 256:
        raise RuntimeError("Qwen3.5 QKV transport smoke requires head_dim=256")
    q_payload_heads = q_heads * (2 if qwen35_qkv else 1)
    o_transport_dtype = os.environ.get("AFD_FABRIC_SMOKE_O_DTYPE", "fp8")
    if o_transport_dtype not in ("bf16", "fp8"):
        raise RuntimeError(
            "fabric smoke O dtype must be bf16 or fp8, "
            f"got {o_transport_dtype!r}"
        )
    if direct_o and (not qwen35_qkv or o_transport_dtype != "bf16"):
        raise RuntimeError(
            "direct O smoke requires Qwen3.5 QKV and BF16 O transport"
        )
    if direct_o_input_width <= 0 or direct_o_input_width % 128:
        raise RuntimeError(
            "direct O smoke GEMM input width must be a positive multiple of 128"
        )
    scale_groups = q_heads * head_dim // 128
    o_scale_packed_groups = (scale_groups + 3) // 4
    o_scale_row_stride = ((rows + 3) // 4) * 4
    o_scale_slot_elements = o_scale_row_stride * o_scale_packed_groups
    num_layers = 3
    cache_rows = max(16, rows * 3 + 17)
    record: dict[str, object] = {"rank": rank}
    if is_attention:
        q_owner, kv_owner, q_ready_owner = allocate_fabric_tensors(
            (
                ((2, rows, q_payload_heads, head_dim), torch.bfloat16),
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
        o_fp8_staging = (
            torch.empty(
                (2, rows, q_heads * head_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            if o_transport_dtype == "fp8"
            else None
        )
        o_scale_staging = (
            torch.empty(
                (2, rows, scale_groups), dtype=torch.uint8, device=device
            )
            if o_transport_dtype == "fp8"
            else None
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
        if o_transport_dtype == "fp8":
            o_owner, o_scale_owner, o_ready_owner = allocate_fabric_tensors(
                (
                    ((2, rows, q_heads, head_dim), torch.float8_e4m3fn),
                    ((2, o_scale_slot_elements), torch.int32),
                    ((2, 1), torch.int64),
                )
            )
        else:
            o_owner, o_ready_owner = allocate_fabric_tensors(
                (
                    ((2, rows, q_heads, head_dim), torch.bfloat16),
                    ((2, 1), torch.int64),
                )
            )
            o_scale_owner = None
        o_ready_owner.tensor.zero_()
        q_publish_counters = torch.zeros((2,), dtype=torch.int32, device=device)
        model_handle = o_owner.handle
        record.update(
            o_handle=model_handle,
            o_shape=tuple(o_owner.tensor.shape),
            o_offset=o_owner.byte_offset,
            o_ready_handle=model_handle,
            o_ready_shape=tuple(o_ready_owner.tensor.shape),
            o_ready_offset=o_ready_owner.byte_offset,
        )
        if o_scale_owner is not None:
            record.update(
                o_scale_handle=model_handle,
                o_scale_shape=tuple(o_scale_owner.tensor.shape),
                o_scale_offset=o_scale_owner.byte_offset,
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
            dtype=(
                torch.float8_e4m3fn
                if o_transport_dtype == "fp8"
                else torch.bfloat16
            ),
            byte_offset=eg["o_offset"],  # type: ignore[arg-type]
        )
        if o_transport_dtype == "fp8":
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

    def qwen35_inputs(epoch: int):
        q_elements = rows * q_heads * head_dim
        kv_elements = rows * kv_heads * head_dim
        q_values = torch.arange(q_elements, dtype=torch.int64, device=device)
        kv_values = torch.arange(kv_elements, dtype=torch.int64, device=device)
        q = (((q_values + epoch * 7) % 97).float() - 48.0).view(
            rows, q_heads, head_dim
        ).div_(16.0).to(torch.bfloat16)
        gate = (((q_values + epoch * 11) % 89).float() - 44.0).view(
            rows, q_heads, head_dim
        ).div_(13.0).to(torch.bfloat16)
        k = (((kv_values + epoch * 13) % 83).float() - 41.0).view(
            rows, kv_heads, head_dim
        ).div_(15.0).to(torch.bfloat16)
        v = (((kv_values + epoch * 17) % 79).float() - 39.0).view(
            rows, kv_heads, head_dim
        ).div_(14.0).to(torch.bfloat16)
        q_weight = (
            0.75
            + (torch.arange(head_dim, device=device) % 19).float() / 64.0
        ).to(torch.bfloat16)
        k_weight = (
            0.625
            + (torch.arange(head_dim, device=device) % 23).float() / 64.0
        ).to(torch.bfloat16)
        positions = torch.arange(rows, dtype=torch.int64, device=device) * 5 + 3
        frequency = 1.0 / (
            10000000.0
            ** (torch.arange(0, 64, 2, device=device).float() / 64.0)
        )
        phases = torch.outer(torch.arange(128, device=device).float(), frequency)
        cos_sin_cache = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return q, gate, k, v, q_weight, k_weight, positions, cos_sin_cache

    def normalize_rope_reference(
        values: torch.Tensor,
        weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        values_f32 = values.float()
        normalized = (
            values_f32
            * torch.rsqrt(values_f32.square().mean(dim=-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(torch.bfloat16)
        output = normalized.clone()
        rotary = normalized[..., :64].float()
        low, high = rotary.chunk(2, dim=-1)
        cache = cos_sin_cache.index_select(0, positions)
        cos, sin = cache.chunk(2, dim=-1)
        output[..., :32] = (low * cos[:, None] - high * sin[:, None]).to(
            torch.bfloat16
        )
        output[..., 32:64] = (high * cos[:, None] + low * sin[:, None]).to(
            torch.bfloat16
        )
        return output

    def linear_inputs(epoch: int):
        width = 7168
        elements = rows * width
        values = torch.arange(elements, dtype=torch.int64, device=device)
        hidden = (((values + pair * 5 + epoch * 7) % 101).float() - 50.0).view(
            rows, width
        ).div_(17.0).to(torch.bfloat16)
        residual = (((values + pair * 11 + epoch * 13) % 103).float() - 51.0).view(
            rows, width
        ).div_(19.0).to(torch.bfloat16)
        shared = (((values + pair * 17 + epoch * 19) % 107).float() - 53.0).view(
            rows, width
        ).div_(23.0).to(torch.bfloat16)
        weight = (
            0.75 + (torch.arange(width, device=device) % 29).float() / 64.0
        ).to(torch.bfloat16)
        return hidden, shared, residual, weight

    def linear_norm_reference(
        hidden: torch.Tensor,
        shared: torch.Tensor | None,
        residual: torch.Tensor | None,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summed = hidden.float()
        if shared is not None:
            summed = (summed + shared.float()).to(torch.bfloat16).float()
        if residual is not None:
            summed = summed + residual.float()
        normalized = (
            summed
            * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(torch.bfloat16)
        return normalized, summed.to(torch.bfloat16)

    for epoch in range(iterations):
        slot = epoch & 1
        layer = epoch % num_layers
        q_value = float(1 + pair + epoch * 2)
        k_value = float(11 + pair + epoch * 2)
        v_value = float(21 + pair + epoch * 2)
        o_value = float(31 + pair + epoch * 2)

        if not is_attention:
            assert q_map is not None and q_ready_map is not None and kv_map is not None
            if qwen35_qkv:
                q_input, gate, k_input, v_input, qw, kw, positions, rope = (
                    qwen35_inputs(epoch)
                )
                q_gate = torch.stack((q_input, gate), dim=2).reshape(rows, -1)
                storage = torch.cat(
                    (q_gate, k_input.flatten(1), v_input.flatten(1)), dim=-1
                )
                q_gate, k, v = storage.split(
                    (q_gate.shape[1], k_input.numel() // rows, v_input.numel() // rows),
                    dim=-1,
                )
                q_desc = _descriptor(
                    [
                        [
                            q_map.tensor.data_ptr(),
                            0,
                            0,
                            q_heads,
                            q_payload_heads,
                            rows,
                            0,
                        ],
                        [
                            q_map.tensor.data_ptr(),
                            q_heads,
                            q_heads,
                            q_heads,
                            q_payload_heads,
                            rows,
                            0,
                        ],
                    ],
                    7,
                    device,
                )
            else:
                q = torch.full(
                    (rows, q_heads * head_dim),
                    q_value,
                    device=device,
                    dtype=torch.bfloat16,
                )
                k = torch.full(
                    (rows, kv_heads * head_dim),
                    k_value,
                    device=device,
                    dtype=torch.bfloat16,
                )
                v = torch.full(
                    (rows, kv_heads * head_dim),
                    v_value,
                    device=device,
                    dtype=torch.bfloat16,
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
            source_offsets = torch.tensor(
                (0, rows), dtype=torch.int64, device=device
            )
            if qwen35_qkv:
                publish_qwen35_qkv(
                    q_gate,
                    k,
                    v,
                    positions,
                    qw,
                    kw,
                    rope,
                    out_loc,
                    source_offsets,
                    q_desc,
                    kv_desc,
                    q_ready_desc,
                    q_publish_counters[slot : slot + 1],
                    layer=layer,
                    slot=slot,
                    ready_value=epoch + 1,
                    eps=1e-6,
                )
            else:
                publish_qkv(
                    q,
                    k,
                    v,
                    out_loc,
                    source_offsets,
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
            if qwen35_qkv:
                q_input, gate, k_input, expected_v, qw, kw, positions, rope = (
                    qwen35_inputs(epoch)
                )
                expected_q = torch.cat(
                    (
                        normalize_rope_reference(q_input, qw, positions, rope),
                        gate,
                    ),
                    dim=1,
                )
                expected_k = normalize_rope_reference(k_input, kw, positions, rope)
                torch.testing.assert_close(
                    q_owner.tensor[slot], expected_q, rtol=0.02, atol=0.02
                )
            else:
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
                .view(cache_rows, kv_heads, head_dim)[out_loc],
                expected_k,
                rtol=0.02 if qwen35_qkv else 0,
                atol=0.02 if qwen35_qkv else 0,
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
        if qwen35_qkv:
            linear_ticket = iterations + epoch + 1
            linear_heads = 7168 // head_dim
            if not is_attention:
                assert q_map is not None and q_ready_map is not None
                hidden, shared_input, residual_input, linear_weight = linear_inputs(
                    epoch
                )
                has_residual = bool(epoch & 1)
                shared = shared_input if has_residual else None
                residual = residual_input.clone() if has_residual else None
                linear_desc = _descriptor(
                    [
                        [
                            q_map.tensor.data_ptr(),
                            0,
                            0,
                            linear_heads,
                            q_payload_heads,
                            rows,
                            0,
                        ]
                    ],
                    7,
                    device,
                )
                publish_qwen35_linear_input(
                    hidden,
                    residual,
                    linear_weight,
                    torch.tensor((0, rows), dtype=torch.int64, device=device),
                    linear_desc,
                    q_ready_desc,
                    q_publish_counters[slot : slot + 1],
                    shared=shared,
                    slot=slot,
                    ready_value=linear_ticket,
                    eps=1e-6,
                )
                torch.cuda.synchronize()
                if residual is not None:
                    _, expected_residual = linear_norm_reference(
                        hidden, shared, residual_input, linear_weight
                    )
                    torch.testing.assert_close(
                        residual, expected_residual, rtol=0, atol=0
                    )
                    consumer_hidden = hidden.clone()
                    consumer_residual = residual_input.clone()
                    qwen35_shared_add_rmsnorm(
                        consumer_hidden,
                        shared_input,
                        consumer_residual,
                        linear_weight,
                        1e-6,
                    )
                    expected_hidden, expected_residual = linear_norm_reference(
                        hidden, shared_input, residual_input, linear_weight
                    )
                    torch.testing.assert_close(
                        consumer_hidden, expected_hidden, rtol=0.02, atol=0.02
                    )
                    torch.testing.assert_close(
                        consumer_residual, expected_residual, rtol=0, atol=0
                    )
            else:
                wait_ready(
                    q_ready_owner.tensor[slot],
                    timeout,
                    timeout_ms=30_000,
                    head_dim=head_dim,
                    expected_ready=linear_ticket,
                )
                torch.cuda.synchronize()
                hidden, shared_input, residual_input, linear_weight = linear_inputs(
                    epoch
                )
                shared = shared_input if epoch & 1 else None
                expected_linear, _ = linear_norm_reference(
                    hidden,
                    shared,
                    residual_input if epoch & 1 else None,
                    linear_weight,
                )
                torch.testing.assert_close(
                    q_owner.tensor[slot, :, :linear_heads].reshape(rows, -1),
                    expected_linear,
                    rtol=0.02,
                    atol=0.02,
                )
            dist.barrier()

        if is_attention:
            assert o_map is not None and o_ready_map is not None
            o_ready_desc = _descriptor(
                [[o_ready_map.tensor.data_ptr(), 0, 1]], 3, device
            )
            if direct_o:
                _run_direct_o_gemm(
                    rows=rows,
                    output_width=q_heads * head_dim,
                    input_width=direct_o_input_width,
                    epoch=epoch,
                    pair=pair,
                    device=device,
                    output=o_map.tensor[slot].view(rows, -1),
                )
                signal_direct_o_release_turn(
                    o_ready_desc,
                    turn,
                    slot=slot,
                    next_turn=epoch + 1,
                    head_dim=head_dim,
                    ready_value=epoch + 1,
                )
            else:
                o = torch.full(
                    (rows, q_heads * head_dim),
                    o_value,
                    device=device,
                    dtype=torch.bfloat16,
                )
                o.view(-1)[::251] = -0.0
            if o_transport_dtype == "fp8":
                assert o_scale_map is not None
                assert o_fp8_staging is not None and o_scale_staging is not None
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
            else:
                o_desc = _descriptor(
                    [[o_map.tensor.data_ptr(), 0, 0, q_heads, q_heads, rows, 0]],
                    7,
                    device,
                )
                if direct_o:
                    pass
                elif qwen35_qkv:
                    release_turn(
                        turn, next_turn=epoch + 1, head_dim=head_dim
                    )
                    publish_o(
                        o,
                        o_desc,
                        o_ready_desc,
                        o_publish_counters[slot : slot + 1],
                        slot=slot,
                        destination_source_stride=rows,
                        head_dim=head_dim,
                        ready_value=epoch + 1,
                    )
                else:
                    publish_o_release_turn(
                        o,
                        o_desc,
                        o_ready_desc,
                        o_publish_counters[slot : slot + 1],
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
            if direct_o:
                expected_o_bf16 = _run_direct_o_gemm(
                    rows=rows,
                    output_width=q_heads * head_dim,
                    input_width=direct_o_input_width,
                    epoch=epoch,
                    pair=pair,
                    device=device,
                )
            else:
                expected_o_bf16 = torch.full(
                    (rows, q_heads * head_dim),
                    o_value,
                    device=device,
                    dtype=torch.bfloat16,
                )
                expected_o_bf16.view(-1)[::251] = -0.0
            if o_transport_dtype == "fp8":
                assert o_scale_owner is not None
                from minisgl.kernel import deepgemm as deep_gemm

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
            else:
                torch.testing.assert_close(
                    o_owner.tensor[slot].view(torch.int16),
                    expected_o_bf16.view(rows, q_heads, head_dim).view(torch.int16),
                    rtol=0,
                    atol=0,
                )
        dist.barrier()

    timing_results: dict[str, float] = {}
    if direct_o and direct_o_timing:
        timing_warmups = int(
            os.environ.get("AFD_FABRIC_SMOKE_TIMING_WARMUPS", "30")
        )
        timing_iterations = int(
            os.environ.get("AFD_FABRIC_SMOKE_TIMING_ITERATIONS", "1000")
        )
        if timing_warmups < 0 or timing_iterations < 1:
            raise RuntimeError("direct O timing warmups/iterations are invalid")
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        if is_attention:
            assert o_map is not None and o_ready_map is not None
            output_width = q_heads * head_dim
            operands = _direct_o_gemm_operands(
                rows=rows,
                output_width=output_width,
                input_width=direct_o_input_width,
                epoch=iterations,
                pair=pair,
                device=device,
            )
            local_output = torch.empty(
                (rows, output_width), dtype=torch.bfloat16, device=device
            )
            remote_output = o_map.tensor[0].view(rows, output_width)
            o_ready_desc = _descriptor(
                [[o_ready_map.tensor.data_ptr(), 0, 1]], 3, device
            )
            o_desc = _descriptor(
                [[o_map.tensor.data_ptr(), 0, 0, q_heads, q_heads, rows, 0]],
                7,
                device,
            )

            def control_publish() -> None:
                _launch_direct_o_gemm(operands, local_output)
                release_turn(turn, next_turn=1, head_dim=head_dim)
                publish_o(
                    local_output,
                    o_desc,
                    o_ready_desc,
                    o_publish_counters[:1],
                    slot=0,
                    destination_source_stride=rows,
                    head_dim=head_dim,
                    ready_value=1,
                )

            def direct_publish() -> None:
                _launch_direct_o_gemm(operands, remote_output)
                signal_direct_o_release_turn(
                    o_ready_desc,
                    turn,
                    slot=0,
                    next_turn=1,
                    head_dim=head_dim,
                    ready_value=1,
                )

            for label, runner in (
                ("control", control_publish),
                ("candidate", direct_publish),
            ):
                runner()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    runner()
                graphs[label] = graph
        dist.barrier()

        for trial, label in enumerate(
            ("control", "candidate", "candidate", "control")
        ):
            dist.barrier()
            elapsed_ms = 0.0
            if is_attention:
                graph = graphs[label]
                for _ in range(timing_warmups):
                    graph.replay()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(timing_iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end) / timing_iterations
            elapsed = torch.tensor(
                (elapsed_ms,), dtype=torch.float64, device=device
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            timing_results[f"{label}_trial{trial}"] = float(elapsed.item())
            dist.barrier()

    if rank == 0:
        print(
            "AFD_FABRIC_TRANSPORT_SMOKE_OK "
            f"pairs={pair_count} rows={rows} iterations={iterations} "
            f"q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim} "
            f"o_dtype={o_transport_dtype} qwen35_qkv={int(qwen35_qkv)} "
            f"direct_o={int(direct_o)}"
        )
        if timing_results:
            print(
                "AFD_DIRECT_LINEAR_O_TIMING "
                + " ".join(
                    f"{key}={value:.9f}ms"
                    for key, value in timing_results.items()
                )
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
