"""Opt-in Qwen3 TP1 small-decode LL128 transport for Blackwell NVLink peers."""
import functools

import torch

from .utils import load_jit


@functools.cache
def module():
    return load_jit(
        "afd_qkv_lamport", cuda_files=["afd_qkv_lamport.cu"],
        cuda_wrappers=[(name, f"AfdQkvLamport::{name}") for name in ("publish", "receive")],
    )


def publish(q, k, v, host_boxes, *, slot, epoch):
    if any(x.dtype != torch.bfloat16 or x.ndim != 2 or x.stride(1) != 1 for x in (q, k, v)):
        raise ValueError("Lamport QKV requires BF16 matrices with contiguous rows")
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("Lamport QKV row counts differ")
    module().publish(q, k, v, host_boxes, slot, epoch)


def receive(inbox, q, kv, locations, *, layer, slot, epoch, timeout_ms):
    module().receive(inbox, q, kv, locations, layer, slot, epoch, timeout_ms)
