from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def qwen35_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 Q/K L2 normalization, evaluated in FP32."""

    x_f32 = x.float()
    return (x_f32 * torch.rsqrt(x_f32.square().sum(dim=-1, keepdim=True) + eps)).to(
        x.dtype
    )


def qwen35_gdn_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return log forget gate ``g`` and delta update gate ``beta`` in FP32."""

    if not a.is_cuda:
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        beta = torch.sigmoid(b.float())
        return g, beta
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"Qwen3.5 GDN gate inputs must be matching 2D tensors, "
            f"got a={tuple(a.shape)} b={tuple(b.shape)}"
        )
    if A_log.numel() != a.shape[1] or dt_bias.numel() != a.shape[1]:
        raise ValueError(
            "Qwen3.5 GDN gate parameter shape mismatch: "
            f"heads={a.shape[1]} A_log={tuple(A_log.shape)} "
            f"dt_bias={tuple(dt_bias.shape)}"
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("Qwen3.5 GDN gate head dimensions must be contiguous")

    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    block_heads = triton.next_power_of_2(a.shape[1])
    _qwen35_gdn_gates_kernel[(a.shape[0],)](
        a,
        b,
        A_log,
        dt_bias,
        g,
        beta,
        a.stride(0),
        b.stride(0),
        a.shape[1],
        BLOCK_HEADS=block_heads,
        num_warps=2,
    )
    return g, beta


@triton.jit
def _qwen35_gdn_gates_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    a_row_stride,
    b_row_stride,
    num_heads,
    BLOCK_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.arange(0, BLOCK_HEADS)
    mask = head < num_heads
    a = tl.load(
        a_ptr + token * a_row_stride + head, mask=mask, other=0.0
    ).to(tl.float32)
    b = tl.load(
        b_ptr + token * b_row_stride + head, mask=mask, other=0.0
    ).to(tl.float32)
    A_log = tl.load(A_log_ptr + head, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head, mask=mask, other=0.0).to(tl.float32)
    softplus_arg = a + dt_bias
    softplus = tl.where(
        softplus_arg <= 20.0,
        tl.log(1.0 + tl.exp(softplus_arg)),
        softplus_arg,
    )
    output_offset = token * num_heads + head
    tl.store(g_ptr + output_offset, -tl.exp(A_log) * softplus, mask=mask)
    tl.store(beta_ptr + output_offset, tl.sigmoid(b), mask=mask)


def _expand_qk_heads(
    q: torch.Tensor, k: torch.Tensor, num_value_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_key_heads = q.shape[-2]
    if num_key_heads == num_value_heads:
        return q, k
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "GDN value heads must be divisible by key heads: "
            f"{num_value_heads} % {num_key_heads} != 0"
        )
    repeats = num_value_heads // num_key_heads
    return (
        q.repeat_interleave(repeats, dim=-2),
        k.repeat_interleave(repeats, dim=-2),
    )


def qwen35_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float | None = None,
    normalize_qk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Portable GDN oracle/fallback for one sequence.

    Inputs use ``[T, H, D]`` and state uses the FlashInfer-compatible
    k-last layout ``[H_v, D_v, D_k]``.
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if initial_state.shape != (
        v.shape[1],
        v.shape[2],
        k.shape[2],
    ):
        raise ValueError(
            "invalid recurrent state shape: "
            f"got {tuple(initial_state.shape)}, expected "
            f"{(v.shape[1], v.shape[2], k.shape[2])}"
        )

    if normalize_qk:
        q, k = qwen35_l2norm(q), qwen35_l2norm(k)
    q, k = _expand_qk_heads(q, k, v.shape[1])
    scale = float(scale if scale is not None else q.shape[-1] ** -0.5)
    state = initial_state.float().clone()
    outputs = []
    for token_idx in range(q.shape[0]):
        state.mul_(torch.exp(g[token_idx].float())[:, None, None])
        q_t = q[token_idx].float() * scale
        k_t = k[token_idx].float()
        v_t = v[token_idx].float()
        memory = (state * k_t[:, None, :]).sum(dim=-1)
        delta = (v_t - memory) * beta[token_idx].float()[:, None]
        state.add_(delta[:, :, None] * k_t[:, None, :])
        outputs.append((state * q_t[:, None, :]).sum(dim=-1))
    output = torch.stack(outputs, dim=0).to(q.dtype)
    return output, state


def qwen35_recurrent_gated_delta_rule_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Portable packed-sequence wrapper used by CPU tests and fallback paths."""

    boundaries = [int(x) for x in cu_seqlens.to("cpu").tolist()]
    if len(boundaries) != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens and initial_state batch size disagree")
    outputs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        output, final_state = qwen35_recurrent_gated_delta_rule(
            q[start:end],
            k[start:end],
            v[start:end],
            g[start:end],
            beta[start:end],
            initial_state[seq_idx],
        )
        outputs.append(output)
        final_states.append(final_state)
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


def qwen35_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GDN prefill using FlashInfer's public SM90 API when available."""

    if not q.is_cuda:
        return qwen35_recurrent_gated_delta_rule_packed(
            q, k, v, g, beta, initial_state, cu_seqlens
        )

    device_major = torch.cuda.get_device_capability(q.device)[0]
    if device_major == 9 and q.dtype in (torch.float16, torch.bfloat16):
        from flashinfer import chunk_gated_delta_rule

        q_norm = qwen35_l2norm(q).contiguous()
        k_norm = qwen35_l2norm(k).contiguous()
        return chunk_gated_delta_rule(
            q=q_norm,
            k=k_norm,
            v=v.contiguous(),
            g=torch.exp(g.float()).contiguous(),
            beta=beta.float().contiguous(),
            scale=q.shape[-1] ** -0.5,
            initial_state=initial_state.float().contiguous(),
            output_final_state=True,
            cu_seqlens=cu_seqlens.to(device=q.device, dtype=torch.int64),
            use_qk_l2norm_in_kernel=False,
        )
    return _qwen35_gdn_prefill_triton(q, k, v, g, beta, initial_state, cu_seqlens)


def qwen35_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    first_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Packed depthwise causal convolution and final short-conv state."""

    width = weight.shape[-1]
    if initial_state.shape[-1] != width - 1:
        raise ValueError("convolution state width must equal kernel width - 1")
    boundaries = [int(v) for v in cu_seqlens.to("cpu").tolist()]
    outputs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        reset = (first_positions[seq_idx] == 0).to(initial_state.dtype)
        prefix = initial_state[seq_idx] * (1 - reset)
        sequence = x[start:end].transpose(0, 1)
        window = torch.cat((prefix, sequence), dim=-1)
        output = F.conv1d(
            window.unsqueeze(0),
            weight,
            bias=None,
            groups=x.shape[-1],
        ).squeeze(0).transpose(0, 1)
        outputs.append(F.silu(output))
        final_states.append(window[:, -(width - 1) :])
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


@triton.jit
def _qwen35_causal_conv1d_decode_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    idx_ptr,
    pos_ptr,
    out_ptr,
    q_norm_scale_ptr,
    k_norm_scale_ptr,
    num_channels,
    x_row_stride,
    state_slot_stride,
    state_channel_stride,
    state_tap_stride,
    w_channel_stride,
    w_tap_stride,
    out_row_stride,
    num_requests,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    PACK_QKV: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    COMPUTE_QK_SCALES: tl.constexpr,
    KEY_HEAD_DIM: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
):
    """One fused decode step of the Qwen3.5 depthwise causal convolution.

    Replaces index_select + mask + cat + weighted-sum + index_copy_ + SiLU, which
    ran as roughly seven ATen launches per linear-attention layer.

    The retained taps are held in scalar registers instead of a [BLOCK_C, taps]
    tile, mirroring vLLM's `_causal_conv1d_update_kernel`. That removes a second
    read of the state (the shifted window is already in registers) and the
    power-of-two tap padding, which together cost ~1.6x the reference's runtime.
    """
    req = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < num_channels

    slot = tl.load(idx_ptr + req)
    # A request at position 0 starts a fresh sequence, so its retained state is
    # discarded rather than convolved.
    keep = tl.where(tl.load(pos_ptr + req) != 0, 1.0, 0.0)

    state_base = state_ptr + slot * state_slot_stride + offs_c * state_channel_stride
    w_base = w_ptr + offs_c * w_channel_stride
    xv = tl.load(x_ptr + req * x_row_stride + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    # window = [state[0..W-2], x]; accumulate against the matching weight taps.
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    if KERNEL_WIDTH >= 2:
        tap0 = tl.load(state_base, mask=mask_c, other=0.0).to(tl.float32) * keep
        acc += tap0 * tl.load(w_base, mask=mask_c, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        tap1 = tl.load(
            state_base + state_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32) * keep
        acc += tap1 * tl.load(
            w_base + w_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        tap2 = tl.load(
            state_base + 2 * state_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32) * keep
        acc += tap2 * tl.load(
            w_base + 2 * w_tap_stride, mask=mask_c, other=0.0
        ).to(tl.float32)
    acc += xv * tl.load(
        w_base + (KERNEL_WIDTH - 1) * w_tap_stride, mask=mask_c, other=0.0
    ).to(tl.float32)

    # Shift the retained window by one: new_state[j] = window[j+1]. Every source
    # tap is already in a register, so the state is never re-read.
    state_ty = state_ptr.dtype.element_ty
    if KERNEL_WIDTH >= 3:
        tl.store(state_base, tap1.to(state_ty), mask=mask_c)
    if KERNEL_WIDTH >= 4:
        tl.store(state_base + state_tap_stride, tap2.to(state_ty), mask=mask_c)
    tl.store(
        state_base + (KERNEL_WIDTH - 2) * state_tap_stride,
        xv.to(state_ty),
        mask=mask_c,
    )

    out = acc * tl.sigmoid(acc)
    out_rounded = out.to(out_ptr.dtype.element_ty)
    if COMPUTE_QK_SCALES:
        # Recurrent value tiles share each Q/K head. Compute its normalization
        # multiplier once while the just-produced BF16 head is still resident,
        # instead of repeating two reductions in every recurrent tile.
        channel_block = tl.program_id(1)
        if channel_block < 2 * NUM_KEY_HEADS:
            norm_input = out_rounded.to(tl.float32)
            inv_norm = tl.rsqrt(
                tl.sum(norm_input * norm_input, axis=0) + 1.0e-6
            )
            if channel_block < NUM_KEY_HEADS:
                tl.store(
                    q_norm_scale_ptr + req * NUM_KEY_HEADS + channel_block,
                    inv_norm * (KEY_HEAD_DIM**-0.5),
                )
            else:
                tl.store(
                    k_norm_scale_ptr
                    + req * NUM_KEY_HEADS
                    + channel_block
                    - NUM_KEY_HEADS,
                    inv_norm,
                )
    if PACK_QKV:
        # Store the conv result directly as three globally contiguous tensors.
        # The recurrent kernel walks token-major Q/K/V independently; packing
        # here avoids materializing a row-interleaved tensor and then copying
        # each slice solely to repair its token stride.
        q_end = KEY_DIM
        k_end = 2 * KEY_DIM
        packed_offset = tl.where(
            offs_c < q_end,
            req * KEY_DIM + offs_c,
            tl.where(
                offs_c < k_end,
                num_requests * KEY_DIM + req * KEY_DIM + offs_c - q_end,
                2 * num_requests * KEY_DIM
                + req * VALUE_DIM
                + offs_c
                - k_end,
            ),
        )
        tl.store(
            out_ptr + packed_offset,
            out_rounded,
            mask=mask_c,
        )
    else:
        tl.store(
            out_ptr + req * out_row_stride + offs_c,
            out_rounded,
            mask=mask_c,
        )


def qwen35_causal_conv1d_decode_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation retained for parity testing and CPU fallback."""

    state = state_pool.index_select(0, state_indices)
    keep = (positions != 0).to(state.dtype)[:, None, None]
    state = state * keep
    window = torch.cat((state, x.unsqueeze(-1)), dim=-1)
    output = (window * weight.squeeze(1).unsqueeze(0)).sum(dim=-1)
    state_pool.index_copy_(0, state_indices, window[:, :, 1:])
    return F.silu(output)


def qwen35_causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Graph-safe single-token causal convolution with in-place slot updates."""

    # MINISGL_QWEN35_CONV1D_EAGER=1 selects the pre-fusion path so an A/B can be
    # interleaved without editing or checking out source between runs.
    if not x.is_cuda or os.environ.get("MINISGL_QWEN35_CONV1D_EAGER") == "1":
        return qwen35_causal_conv1d_decode_eager(
            x, weight, state_pool, state_indices, positions
        )

    num_requests, num_channels = x.shape
    kernel_width = weight.shape[-1]
    if not 2 <= kernel_width <= 4:
        raise ValueError(
            f"qwen35_causal_conv1d_decode supports kernel_width 2..4, got "
            f"{kernel_width} from weight shape {tuple(weight.shape)}; the decode "
            f"kernel unrolls the retained taps into registers. Extend the "
            f"KERNEL_WIDTH cascade in _qwen35_causal_conv1d_decode_kernel to "
            f"support a wider convolution."
        )
    block_c = 128
    out = torch.empty_like(x)
    grid = (num_requests, triton.cdiv(num_channels, block_c))
    _qwen35_causal_conv1d_decode_kernel[grid](
        x,
        weight,
        state_pool,
        state_indices,
        positions,
        out,
        out,
        out,
        num_channels,
        x.stride(0),
        state_pool.stride(0),
        state_pool.stride(1),
        state_pool.stride(2),
        weight.stride(0),
        weight.stride(-1),
        out.stride(0),
        num_requests,
        KERNEL_WIDTH=kernel_width,
        BLOCK_C=block_c,
        PACK_QKV=False,
        KEY_DIM=0,
        VALUE_DIM=0,
        COMPUTE_QK_SCALES=False,
        KEY_HEAD_DIM=0,
        NUM_KEY_HEADS=0,
        num_warps=4,
    )
    return out


def qwen35_causal_conv1d_decode_packed(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    key_dim: int,
    value_dim: int,
    key_head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fuse decode convolution with token-contiguous Q/K/V materialization."""

    if 2 * key_dim + value_dim != x.shape[-1]:
        raise ValueError(
            "Qwen3.5 packed convolution dimensions do not cover the input: "
            f"key={key_dim} value={value_dim} input={x.shape[-1]}"
        )
    num_requests = x.shape[0]
    if not x.is_cuda or os.environ.get("MINISGL_QWEN35_CONV1D_EAGER") == "1":
        mixed = qwen35_causal_conv1d_decode_eager(
            x, weight, state_pool, state_indices, positions
        )
        q, k, v = mixed.split((key_dim, key_dim, value_dim), dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_f32 = q.view(num_requests, -1, key_head_dim).float()
        k_f32 = k.view(num_requests, -1, key_head_dim).float()
        q_scale = torch.rsqrt(q_f32.square().sum(dim=-1) + 1.0e-6)
        q_scale = q_scale * (key_head_dim**-0.5)
        k_scale = torch.rsqrt(k_f32.square().sum(dim=-1) + 1.0e-6)
        return q, k, v, q_scale, k_scale

    num_channels = x.shape[1]
    kernel_width = weight.shape[-1]
    if not 2 <= kernel_width <= 4:
        raise ValueError(
            f"qwen35_causal_conv1d_decode_packed supports kernel_width 2..4, got "
            f"{kernel_width} from weight shape {tuple(weight.shape)}"
        )
    block_c = 128
    if key_dim % block_c or value_dim % block_c:
        raise ValueError(
            "Qwen3.5 packed convolution requires 128-aligned Q/K/V widths, "
            f"got key={key_dim} value={value_dim}"
        )
    if key_head_dim != block_c or key_dim % key_head_dim:
        raise ValueError(
            "Qwen3.5 packed convolution Q/K scale fusion requires "
            f"128-wide key heads, got key_dim={key_dim} head_dim={key_head_dim}"
        )

    packed = torch.empty(
        num_requests * num_channels, dtype=x.dtype, device=x.device
    )
    num_key_heads = key_dim // key_head_dim
    q_norm_scale = torch.empty(
        (num_requests, num_key_heads), dtype=torch.float32, device=x.device
    )
    k_norm_scale = torch.empty_like(q_norm_scale)
    grid = (num_requests, triton.cdiv(num_channels, block_c))
    _qwen35_causal_conv1d_decode_kernel[grid](
        x,
        weight,
        state_pool,
        state_indices,
        positions,
        packed,
        q_norm_scale,
        k_norm_scale,
        num_channels,
        x.stride(0),
        state_pool.stride(0),
        state_pool.stride(1),
        state_pool.stride(2),
        weight.stride(0),
        weight.stride(-1),
        0,
        num_requests,
        KERNEL_WIDTH=kernel_width,
        BLOCK_C=block_c,
        PACK_QKV=True,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        COMPUTE_QK_SCALES=True,
        KEY_HEAD_DIM=key_head_dim,
        NUM_KEY_HEADS=num_key_heads,
        num_warps=4,
    )
    q_elements = num_requests * key_dim
    v_elements = num_requests * value_dim
    q = packed[:q_elements].view(num_requests, key_dim)
    k = packed[q_elements : 2 * q_elements].view(num_requests, key_dim)
    v = packed[2 * q_elements : 2 * q_elements + v_elements].view(
        num_requests, value_dim
    )
    return q, k, v, q_norm_scale, k_norm_scale


def _qwen35_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    original_dtype = x.dtype
    x_f32 = x.float()
    # z was upcast twice here, which cost an extra full-size copy per layer.
    z_f32 = z.float()
    normalized = x_f32 * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    gated = normalized * (z_f32 * torch.sigmoid(z_f32))
    return gated.to(original_dtype)


@triton.jit
def _qwen35_rmsnorm_gated_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_ptr,
    z_token_stride,
    eps: tl.constexpr,
    rows: tl.constexpr,
    width: tl.constexpr,
    heads: tl.constexpr,
    row_block: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
    offsets = tl.arange(0, block)[None, :]
    mask = (row < rows) & (offsets < width)
    x = tl.load(x_ptr + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
    token = row // heads
    head = row % heads
    z_offset = token * z_token_stride + head * width + offsets
    z = tl.load(z_ptr + z_offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=offsets < width, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / width + eps)
    normalized = x * inv_rms * weight
    gate = z * tl.sigmoid(z)
    out = normalized * gate
    tl.store(
        out_ptr + row * width + offsets,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _qwen35_rmsnorm_gated_fp8_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_ptr,
    scale_ptr,
    z_token_stride,
    eps: tl.constexpr,
    rows: tl.constexpr,
    width: tl.constexpr,
    heads: tl.constexpr,
    scale_token_stride: tl.constexpr,
    scale_group_stride: tl.constexpr,
    row_block: tl.constexpr,
    block: tl.constexpr,
):
    row_base = tl.program_id(0) * row_block
    local_row_id = tl.arange(0, row_block)
    local_row = local_row_id[:, None]
    row = row_base + local_row
    offsets = tl.arange(0, block)[None, :]
    mask = (row < rows) & (offsets < width)
    x = tl.load(x_ptr + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
    token = row // heads
    head = row % heads
    z_offset = token * z_token_stride + head * width + offsets
    z = tl.load(z_ptr + z_offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=offsets < width, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / width + eps)
    normalized = x * inv_rms * weight
    gate = z * tl.sigmoid(z)

    # Preserve the standalone path's BF16 boundary before computing dynamic
    # per-128 UE8M0 scales and FP8 values.
    rounded = (normalized * gate).to(x_ptr.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(rounded), axis=1), 1.0e-10)
    scale_raw = tl.maximum(amax * (1.0 / 448.0), 1.0e-10)
    scale = tl.exp2(tl.ceil(tl.log2(scale_raw)))
    quantized = tl.clamp(rounded / scale[:, None], -448.0, 448.0).to(
        out_ptr.dtype.element_ty
    )
    tl.store(out_ptr + row * width + offsets, quantized, mask=mask)

    scale_exp = scale.to(tl.int32, bitcast=True) >> 23
    packed = tl.sum(scale_exp << (local_row_id.to(tl.int32) * 8), axis=0)
    token = row_base // heads
    packed_group = (row_base % heads) // row_block
    tl.store(
        scale_ptr
        + token * scale_token_stride
        + packed_group * scale_group_stride,
        packed,
    )


def qwen35_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-value-head RMSNorm followed by the Qwen3.5 SiLU output gate."""

    # MINISGL_QWEN35_GATE_EAGER=1 selects the eager path so an A/B can be
    # interleaved without editing or checking out source between runs, matching
    # MINISGL_QWEN35_CONV1D_EAGER above. Needed because the Triton reduction is
    # not bitwise-identical, so any correctness difference has to be attributable.
    if not x.is_cuda or os.environ.get("MINISGL_QWEN35_GATE_EAGER") == "1":
        return _qwen35_rmsnorm_gated(x, z, weight, eps)
    if x.shape != z.shape or x.shape[-1] != weight.numel():
        raise ValueError(
            f"Qwen3.5 gated RMSNorm shape mismatch: "
            f"x={tuple(x.shape)} z={tuple(z.shape)} weight={tuple(weight.shape)}"
        )
    if not (x.is_contiguous() and weight.is_contiguous()):
        raise ValueError("Qwen3.5 gated RMSNorm requires contiguous x/weight")
    if z.stride(-1) != 1 or z.stride(-2) != z.shape[-1]:
        raise ValueError(
            "Qwen3.5 gated RMSNorm requires contiguous per-token z heads"
        )
    out = torch.empty_like(x)
    width = int(x.shape[-1])
    heads = int(x.shape[-2])
    rows = x.numel() // width
    row_block = 2
    _qwen35_rmsnorm_gated_kernel[(triton.cdiv(rows, row_block),)](
        x,
        z,
        weight,
        out,
        z.stride(0),
        eps=float(eps),
        rows=rows,
        width=width,
        heads=heads,
        row_block=row_block,
        block=triton.next_power_of_2(width),
        num_warps=2,
    )
    return out


def qwen35_rmsnorm_gated_fp8(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse GDN gated RMSNorm with dynamic FP8 input quantization."""

    if not x.is_cuda:
        raise RuntimeError("Qwen3.5 fused gated RMSNorm FP8 requires CUDA")
    if x.shape != z.shape or x.shape[-1] != weight.numel():
        raise ValueError(
            f"Qwen3.5 gated RMSNorm FP8 shape mismatch: "
            f"x={tuple(x.shape)} z={tuple(z.shape)} weight={tuple(weight.shape)}"
        )
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise ValueError("Qwen3.5 fused gated RMSNorm FP8 requires BF16 inputs")
    if not (x.is_contiguous() and weight.is_contiguous()):
        raise ValueError("Qwen3.5 fused gated RMSNorm FP8 requires contiguous x/weight")
    if z.stride(-1) != 1 or z.stride(-2) != z.shape[-1]:
        raise ValueError(
            "Qwen3.5 fused gated RMSNorm FP8 requires contiguous per-token heads"
        )

    tokens, heads, width = x.shape
    row_block = 4
    if width != 128 or heads % row_block:
        raise ValueError(
            "Qwen3.5 fused gated RMSNorm FP8 requires width=128 and heads%4=0, "
            f"got heads={heads} width={width}"
        )
    rows = tokens * heads
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    aligned_tokens = (tokens + 3) // 4 * 4
    packed_groups = heads // row_block
    scales = torch.empty_strided(
        (tokens, packed_groups),
        (1, aligned_tokens),
        dtype=torch.int32,
        device=x.device,
    )
    _qwen35_rmsnorm_gated_fp8_kernel[(triton.cdiv(rows, row_block),)](
        x,
        z,
        weight,
        out,
        scales,
        z.stride(0),
        eps=float(eps),
        rows=rows,
        width=width,
        heads=heads,
        scale_token_stride=scales.stride(0),
        scale_group_stride=scales.stride(1),
        row_block=row_block,
        block=128,
        num_warps=4,
    )
    return out, scales


@triton.jit
def _qwen35_gdn_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    initial_state_ptr,
    cu_seqlens_ptr,
    output_ptr,
    final_state_ptr,
    H_Q: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    value_head = tl.program_id(1)
    value_block = tl.program_id(2)
    key_head = value_head // (H_V // H_Q)
    start = tl.load(cu_seqlens_ptr + sequence_idx)
    end = tl.load(cu_seqlens_ptr + sequence_idx + 1)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V
    state_offsets = (
        ((sequence_idx * H_V + value_head) * V + value_offsets[:, None]) * K
        + key_offsets[None, :]
    )
    state_mask = value_mask[:, None] & key_mask[None, :]
    state = tl.load(
        initial_state_ptr + state_offsets, mask=state_mask, other=0.0
    ).to(tl.float32)

    token_idx = start
    while token_idx < end:
        q = tl.load(
            q_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (K**-0.5)
        k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        head_offset = token_idx * H_V + value_head
        state *= tl.exp(tl.load(g_ptr + head_offset).to(tl.float32))
        v = tl.load(
            v_ptr + head_offset * V + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        memory = tl.sum(state * k[None, :], axis=1)
        delta = (v - memory) * tl.load(beta_ptr + head_offset).to(tl.float32)
        state += delta[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], axis=1)
        tl.store(
            output_ptr + head_offset * V + value_offsets,
            output,
            mask=value_mask,
        )
        token_idx += 1

    tl.store(final_state_ptr + state_offsets, state, mask=state_mask)


def _qwen35_gdn_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.float().contiguous()
    beta = beta.float().contiguous()
    initial_state = initial_state.float().contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    grid = (
        initial_state.shape[0],
        v.shape[1],
        triton.cdiv(value_dim, block_value),
    )
    _qwen35_gdn_prefill_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        output,
        final_state,
        H_Q=q.shape[1],
        H_V=v.shape[1],
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        num_warps=4,
        num_stages=2,
    )
    return output, final_state


@triton.jit
def _qwen35_gdn_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    state_indices_ptr,
    positions_ptr,
    output_ptr,
    q_norm_scale_ptr,
    k_norm_scale_ptr,
    g_ptr,
    beta_ptr,
    a_row_stride,
    b_row_stride,
    HAS_QK_NORM_SCALES: tl.constexpr,
    HAS_PRECOMPUTED_GATES: tl.constexpr,
    H_Q: tl.constexpr,
    H_V: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # The value block is the fastest-varying grid axis so that consecutive CTAs
    # walk one request's recurrent state contiguously. With the token axis first,
    # a scheduled wave touched one 16 KiB slice from each of B slots that sit
    # 4 MiB apart, and this bandwidth-bound kernel lost ~13% to the scatter.
    value_block = tl.program_id(0)
    token_idx = tl.program_id(1)
    value_head = tl.program_id(2)
    key_head = value_head // (H_V // H_Q)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V

    q = tl.load(
        q_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + (token_idx * H_Q + key_head) * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_QK_NORM_SCALES:
        norm_offset = token_idx * H_Q + key_head
        q *= tl.load(q_norm_scale_ptr + norm_offset)
        k *= tl.load(k_norm_scale_ptr + norm_offset)
    else:
        q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (K**-0.5)
        k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)

    slot = tl.load(state_indices_ptr + token_idx).to(tl.int64)
    position = tl.load(positions_ptr + token_idx)
    state_offsets = (
        ((slot * H_V + value_head) * V + value_offsets[:, None]) * K
        + key_offsets[None, :]
    )
    state_mask = value_mask[:, None] & key_mask[None, :]
    state = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )
    state = tl.where(position == 0, 0.0, state)

    gate_offset = token_idx * H_V + value_head
    if HAS_PRECOMPUTED_GATES:
        g = tl.load(g_ptr + gate_offset).to(tl.float32)
        beta = tl.load(beta_ptr + gate_offset).to(tl.float32)
    else:
        a = tl.load(a_ptr + token_idx * a_row_stride + value_head).to(tl.float32)
        b = tl.load(b_ptr + token_idx * b_row_stride + value_head).to(tl.float32)
        A_log = tl.load(A_log_ptr + value_head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
        softplus_arg = a + dt_bias
        softplus = tl.where(
            softplus_arg <= 20.0,
            tl.log(1.0 + tl.exp(softplus_arg)),
            softplus_arg,
        )
        g = -tl.exp(A_log) * softplus
        beta = tl.sigmoid(b)
    state *= tl.exp(g)

    v = tl.load(
        v_ptr + (token_idx * H_V + value_head) * V + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    memory = tl.sum(state * k[None, :], axis=1)
    delta = (v - memory) * beta
    state += delta[:, None] * k[None, :]
    output = tl.sum(state * q[None, :], axis=1)

    tl.store(state_ptr + state_offsets, state, mask=state_mask)
    tl.store(
        output_ptr + (token_idx * H_V + value_head) * V + value_offsets,
        output,
        mask=value_mask,
    )


def qwen35_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    positions: torch.Tensor,
    q_norm_scale: torch.Tensor | None = None,
    k_norm_scale: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused single-token GDN update; mutates selected recurrent-state slots."""

    if (g is None) != (beta is None):
        raise ValueError("Qwen3.5 GDN requires both precomputed gates")

    if not q.is_cuda:
        outputs = []
        for token_idx in range(q.shape[0]):
            slot = int(state_indices[token_idx])
            initial = state_pool[slot]
            if int(positions[token_idx]) == 0:
                initial = torch.zeros_like(initial)
            if g is None or beta is None:
                token_g, token_beta = qwen35_gdn_gates(
                    a[token_idx], b[token_idx], A_log, dt_bias
                )
            else:
                token_g = g[token_idx]
                token_beta = beta[token_idx]
            output, final_state = qwen35_recurrent_gated_delta_rule(
                q[token_idx : token_idx + 1],
                k[token_idx : token_idx + 1],
                v[token_idx : token_idx + 1],
                token_g.unsqueeze(0),
                token_beta.unsqueeze(0),
                initial,
            )
            state_pool[slot].copy_(final_state)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    # The optimized decode convolution already writes token-contiguous Q/K/V,
    # making these calls no-ops. Keep them for other callers: reading the old
    # row-interleaved views in place was measured and REJECTED because the
    # bandwidth-bound recurrent kernel lost more than the copy kernels saved.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("Qwen3.5 GDN a/b head dimensions must be contiguous")
    output = torch.empty_like(v)
    has_qk_norm_scales = q_norm_scale is not None or k_norm_scale is not None
    if has_qk_norm_scales:
        if q_norm_scale is None or k_norm_scale is None:
            raise ValueError(
                "Qwen3.5 GDN requires both Q and K normalization scales"
            )
        expected_scale_shape = (q.shape[0], q.shape[1])
        if (
            q_norm_scale.shape != expected_scale_shape
            or k_norm_scale.shape != expected_scale_shape
        ):
            raise ValueError(
                "Qwen3.5 GDN normalization scale shape mismatch: "
                f"expected={expected_scale_shape} q={tuple(q_norm_scale.shape)} "
                f"k={tuple(k_norm_scale.shape)}"
            )
        q_norm_scale = q_norm_scale.contiguous()
        k_norm_scale = k_norm_scale.contiguous()
    else:
        # Triton still requires pointer arguments when the compile-time branch
        # removes the loads.
        q_norm_scale = q
        k_norm_scale = k
    has_precomputed_gates = g is not None or beta is not None
    if has_precomputed_gates:
        if g is None or beta is None:
            raise ValueError("Qwen3.5 GDN requires both precomputed gates")
        expected_gate_shape = (v.shape[0], v.shape[1])
        if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
            raise ValueError(
                "Qwen3.5 precomputed gate shape mismatch: "
                f"expected={expected_gate_shape} g={tuple(g.shape)} "
                f"beta={tuple(beta.shape)}"
            )
        g = g.contiguous()
        beta = beta.contiguous()
    else:
        g = a
        beta = b
    key_dim = q.shape[-1]
    value_dim = v.shape[-1]
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    grid = (triton.cdiv(value_dim, block_value), q.shape[0], v.shape[1])
    _qwen35_gdn_decode_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state_pool,
        state_indices,
        positions,
        output,
        q_norm_scale,
        k_norm_scale,
        g,
        beta,
        a.stride(0),
        b.stride(0),
        HAS_QK_NORM_SCALES=has_qk_norm_scales,
        HAS_PRECOMPUTED_GATES=has_precomputed_gates,
        H_Q=q.shape[1],
        H_V=v.shape[1],
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        # Measured across B=32..512: 8 warps beat 4 by 7-12%. This is a
        # read-modify-write of a 4 MiB/request state, so it wants issue width to
        # cover memory latency rather than a narrow tile.
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = [
    "qwen35_causal_conv1d_decode",
    "qwen35_causal_conv1d_decode_packed",
    "qwen35_causal_conv1d_prefill",
    "qwen35_gdn_decode",
    "qwen35_gdn_gates",
    "qwen35_gdn_prefill",
    "qwen35_l2norm",
    "qwen35_recurrent_gated_delta_rule",
    "qwen35_recurrent_gated_delta_rule_packed",
    "qwen35_rmsnorm_gated",
    "qwen35_rmsnorm_gated_fp8",
]
