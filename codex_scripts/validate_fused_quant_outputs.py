#!/usr/bin/env python3
"""Check fused quant producer output strides, guard bytes, and graph reuse on CUDA.

Run against the intended source through PYTHONPATH and its prebuilt native cache.
This is a correctness check, not a performance benchmark or model alignment.
"""

import argparse
import json
from pathlib import Path

import torch

from minisgl.kernel.fp8_quant import per_token_cast_to_fp8_torch
from minisgl.kernel.moe_topk import _jit_minimax_route_fused_module, gate_topk


def outputs(rows, hidden, layout):
    guards = []

    def allocation(width):
        storage = torch.full((rows + 2, width), 0xA5, dtype=torch.uint8, device="cuda")
        untouched = torch.ones_like(storage, dtype=torch.bool)
        guards.append((storage, untouched))
        return storage, untouched

    sf_width = hidden // 32
    if layout.startswith("frame"):
        storage, untouched = allocation(hidden + sf_width + 16)
        x = storage[1:-1, :hidden]
        sf = storage[1:-1, hidden:hidden + sf_width]
        untouched[1:-1, :hidden + sf_width] = False
    else:
        padding = 16 if layout == "separate_padded" else 0
        storage, untouched = allocation(hidden + padding)
        x = storage[1:-1, :hidden]
        untouched[1:-1, :hidden] = False
        storage, untouched = allocation(sf_width + 2 * padding)
        sf = storage[1:-1, :sf_width]
        untouched[1:-1, :sf_width] = False
    return (x.view(torch.float8_e4m3fn),
            sf if layout == "frame_u8" else sf.view(torch.int32), guards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 3 or not 2 <= args.experts <= 256 or any(h <= 0 or h % 1024 for h in args.hidden):
        parser.error("Require rows >= 3, 2 <= experts <= 256 and hidden widths divisible by 1024")
    if args.output.exists():
        parser.error("Output already exists; preserve the previous evidence")
    torch.cuda.set_device(0)
    torch.manual_seed(1742)
    native = _jit_minimax_route_fused_module()
    records = []
    for hidden in args.hidden:
        x_in = torch.randn((args.rows, hidden), device="cuda", dtype=torch.bfloat16)
        x_in[0].zero_()
        x_in[1].fill_(-1)
        gate = torch.randn((args.experts, hidden), device="cuda", dtype=torch.bfloat16)
        gate_fp32 = gate.float()
        for mode in ("qwen", "minimax4", "minimax16", "minimax16_sigmoid"):
            def run(x, sf):
                if mode == "qwen":
                    return gate_topk(x_in, gate, 2, quant_out=(x, sf, None))
                logits = torch.empty((args.rows, args.experts), device="cuda")
                method = {
                    "minimax4": native.launch_logits4_quant,
                    "minimax16": native.launch_logits16_quant,
                    "minimax16_sigmoid": native.launch_logits16_quant_sigmoid,
                }[mode]
                method(logits, x_in, gate_fp32, x, sf)
                return (logits,)

            for layout in ("contiguous", "frame_i32", "frame_u8", "separate_padded"):
                x, sf, guards = outputs(args.rows, hidden, layout)
                reference_x, reference_sf, _ = outputs(args.rows, hidden, "contiguous")
                # Warm native kernels and persistent workspaces before capture.
                run(x, sf)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual_route = run(x, sf)
                for replay in range(2):
                    x_in.mul_(-2)
                    expected_x, expected_sf = per_token_cast_to_fp8_torch(
                        x_in, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
                    expected_route = run(reference_x, reference_sf)
                    for storage, _ in guards:
                        storage.fill_(0xA5)
                    graph.replay()
                    torch.cuda.synchronize()
                    label = f"{mode}/{layout}/H{hidden}/replay{replay}"
                    for name, actual, expected in (
                        ("activation", x.view(torch.uint8), expected_x.view(torch.uint8)),
                        ("scale", sf.view(torch.uint8), expected_sf.view(torch.uint8)),
                        *((f"route{i}", a, b) for i, (a, b) in
                          enumerate(zip(actual_route, expected_route))),
                    ):
                        if not torch.equal(actual, expected):
                            raise RuntimeError(f"{label}: {name} mismatch")
                    if any(not bool((storage[mask] == 0xA5).all()) for storage, mask in guards):
                        raise RuntimeError(f"{label}: wrote outside output views")
                    records.append({"mode": mode, "layout": layout, "hidden": hidden,
                                    "replay": replay, "status": "PASS"})
                # Keep successive cases in the same modest input range.
                x_in.mul_(0.25)
            if mode in ("minimax4", "minimax16"):
                plain = torch.empty((args.rows, args.experts), device="cuda")
                method = native.launch_logits4 if mode == "minimax4" else native.launch_logits16
                method(plain, x_in, gate_fp32)
                quantized = run(reference_x, reference_sf)[0]
                if not torch.equal(plain, quantized):
                    raise RuntimeError(f"{mode}/H{hidden}: nonquantized logits mismatch")
                records.append({"mode": mode, "hidden": hidden,
                                "nonquantized": True, "status": "PASS"})
                if mode == "minimax16":
                    native.launch_logits(plain, x_in, gate_fp32)
                    if not torch.equal(plain, quantized):
                        raise RuntimeError(f"minimax8/H{hidden}: nonquantized logits mismatch")
                    records.append({"mode": "minimax8", "hidden": hidden,
                                    "nonquantized": True, "status": "PASS"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "PASS", "rows": args.rows, "experts": args.experts,
              "checks": len(records), "records": records,
              "scope": "Local fused output strides and graph reuse; not distributed alignment"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
