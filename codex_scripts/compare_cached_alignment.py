"""Check AFD tokens against saved vLLM scores at identical prompt/prefixes.

A changed prefix never inherits scores from the old continuation. Missing
reference coverage fails explicitly and is distinct from a ranked mismatch.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    references = defaultdict(list)
    model = None
    for path in args.reference_json:
        data = json.loads(path.read_text())
        if model is not None and data["tokenizer_model"] != model:
            raise ValueError("cached references use different tokenizer/model profiles")
        model = data["tokenizer_model"]
        for result in data["results"]:
            references[result["prompt"]].append(result)
    sample_data = json.loads(args.sample_json.read_text())
    samples = sample_data["samples"]
    if not samples:
        raise ValueError("sample bundle is empty")
    results = []
    total = covered = top1 = top10 = 0
    for index, sample in enumerate(samples):
        tokens = sample["generated_token_ids"]
        if not tokens:
            raise ValueError(f"prompt {index} has no generated tokens")
        ranks = []
        for position, token in enumerate(tokens):
            observed = set()
            for reference in references[sample["prompt"]]:
                if position >= len(reference["steps"]):
                    continue
                if tokens[:position] != reference["generated_token_ids"][:position]:
                    continue
                step = reference["steps"][position]
                if token == step["token_id"]:
                    observed.add(int(step["rank_in_vllm"]))
                if token == step["vllm_top1_token_id"]:
                    observed.add(1)
                for candidate in step.get("vllm_candidates", []):
                    if token == candidate["token_id"]:
                        observed.add(int(candidate["rank"]))
            if len(observed) > 1:
                raise ValueError(f"inconsistent cached ranks at prompt={index} position={position}")
            rank = next(iter(observed)) if observed else None
            ranks.append(rank)
            total += 1
            covered += rank is not None
            top1 += rank == 1
            top10 += rank is not None and rank <= 10
        results.append({
            "prompt_index": index,
            "ranks": ranks,
            "unknown_positions": [i for i, rank in enumerate(ranks) if rank is None],
            "non_top1_positions": [
                i for i, rank in enumerate(ranks) if rank is not None and rank != 1
            ],
        })
    summary = {
        "num_prompts": len(samples),
        "total_generated_tokens": total,
        "covered_tokens": covered,
        "missing_reference_tokens": total - covered,
        "top1_tokens": top1,
        "top10_tokens": top10,
        "overall_top1_agreement_rate": top1 / total,
        "overall_top10_hit_rate": top10 / total,
        "passed": covered == total and top10 == total,
    }
    args.output_json.write_text(
        json.dumps({
            "summary": summary,
            "sample_json": str(args.sample_json),
            "reference_json": list(map(str, args.reference_json)),
            "results": results,
        }, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("cached alignment incomplete or below 100% top-10")


if __name__ == "__main__":
    main()
