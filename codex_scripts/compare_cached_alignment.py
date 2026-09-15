"""Check AFD tokens against saved vLLM scores at identical prompt/prefixes.

A changed prefix never inherits scores from the old continuation. Missing
reference coverage fails explicitly and is distinct from a ranked mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def validate_reference(result: dict) -> None:
    """Reject malformed positions or scores before using cached prefix ranks."""
    tokens = result["generated_token_ids"]
    steps = result["steps"]
    if result["num_generated_tokens"] != len(tokens) or len(steps) != len(tokens):
        raise ValueError("cached reference token/step coverage differs")
    for position, (token, step) in enumerate(zip(tokens, steps)):
        if step["position"] != position or step["token_id"] != token:
            raise ValueError("cached reference position/token differs")
        rank = step["rank_in_vllm"]
        if (type(rank) is not int or rank < 1
                or not math.isfinite(step["logprob_in_vllm"])
                or step["is_top1_in_vllm"] != (rank == 1)
                or (rank == 1 and step["vllm_top1_token_id"] != token)):
            raise ValueError("cached reference has an invalid chosen-token score")
        candidate_tokens, candidate_ranks = set(), set()
        for candidate in step.get("vllm_candidates", []):
            candidate_token, candidate_rank = candidate["token_id"], candidate["rank"]
            if (type(candidate_token) is not int or candidate_token < 0
                    or type(candidate_rank) is not int or candidate_rank < 1
                    or not math.isfinite(candidate["logprob"])
                    or candidate_token in candidate_tokens or candidate_rank in candidate_ranks
                    or (candidate_rank == 1 and candidate_token != step["vllm_top1_token_id"])):
                raise ValueError("cached reference has an invalid candidate score")
            candidate_tokens.add(candidate_token)
            candidate_ranks.add(candidate_rank)


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
            validate_reference(result)
            references[result["prompt"]].append(result)
    sample_data = json.loads(args.sample_json.read_text())
    if sample_data["tokenizer_model"] != model:
        raise ValueError("sample and cached references use different tokenizer/model profiles")
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
