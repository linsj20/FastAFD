"""CUDA regression and graph timing for routing with padding inside fan-in."""

import json
import statistics

import torch

from minisgl.kernel.moe_topk import gate_topk, topk_softmax_group_local


def graph_time(fn):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return statistics.median(samples)


def main():
    torch.manual_seed(20260909)
    cases = []
    for experts, topk in ((128, 8), (512, 10)):
        for counts, span in (((3, 3), 4), ((1, 3, 2), 3), ((0, 4), 4), ((3,), 4)):
            rows = len(counts) * span
            hidden = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda")
            gate = torch.randn(experts, 4096, dtype=torch.bfloat16, device="cuda")
            metadata = [sum(counts)]
            prefix = 0
            for count in counts:
                metadata.extend(range(prefix, prefix + count))
                metadata.extend([-1] * (span - count))
                prefix += count
            validity = torch.tensor(
                metadata if len(counts) > 1 else [counts[0]], device="cuda"
            )
            valid = torch.tensor(
                [local < count for count in counts for local in range(span)], device="cuda"
            )
            ids, weights = gate_topk(hidden, gate, topk, topk_idx_dtype=torch.int64)
            actual_ids, actual_weights = gate_topk(
                hidden, gate, topk, topk_idx_dtype=torch.int64,
                num_token_non_padded=validity,
            )
            torch.testing.assert_close(actual_ids[valid], ids[valid], rtol=0, atol=0)
            torch.testing.assert_close(actual_weights[valid], weights[valid], rtol=0, atol=0)
            assert (actual_ids[~valid] == -1).all() and (actual_weights[~valid] == 0).all()
            # The non-fused group-local selector shares the validity contract.
            logits = torch.randn(rows, experts, device="cuda")
            expert_map = torch.arange(experts, dtype=torch.int64, device="cuda")
            group_ids, group_weights = torch.empty_like(ids), torch.empty_like(weights)
            topk_softmax_group_local(
                group_weights, group_ids, logits, expert_map, renormalize=True
            )
            topk_softmax_group_local(
                actual_weights, actual_ids, logits, expert_map,
                num_token_non_padded=validity, renormalize=True,
            )
            torch.testing.assert_close(actual_ids[valid], group_ids[valid], rtol=0, atol=0)
            torch.testing.assert_close(actual_weights[valid], group_weights[valid], rtol=0, atol=0)
            assert (actual_ids[~valid] == -1).all() and (actual_weights[~valid] == 0).all()
            cases.append({"experts": experts, "counts": counts, "span": span, "exact": True})
    # Identical routing work isolates indexing cost from changing active rows.
    gate = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    times = {}
    for rows in (8, 32, 128):
        hidden = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda")
        scalar = torch.tensor([rows], dtype=torch.int64, device="cuda")
        row_map = torch.tensor([rows, *range(rows)], dtype=torch.int64, device="cuda")
        out = (
            torch.empty(rows, 8, dtype=torch.int64, device="cuda"),
            torch.empty(rows, 8, dtype=torch.float32, device="cuda"),
        )
        times[rows] = {}
        for name, counts in (
            ("scalar_a", scalar), ("row_map_a", row_map),
            ("row_map_b", row_map), ("scalar_b", scalar),
        ):
            times[rows][name] = graph_time(
                lambda: gate_topk(
                    hidden, gate, 8, topk_idx_dtype=torch.int64,
                    num_token_non_padded=counts, out=out,
                )
            )
    print(json.dumps({"cases": cases, "router_graph_us": times}, indent=2))


if __name__ == "__main__":
    main()
