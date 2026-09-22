#!/usr/bin/env python3
"""Check complete ordered generation coverage, optionally with fresh vLLM scores."""

import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=int, required=True)
    parser.add_argument("--tokens-per-prompt", type=int, required=True)
    parser.add_argument("--score-json", type=Path)
    parser.add_argument("--max-top10-misses", type=int, default=0,
                        help="Allowed total tokens outside top10; default remains strict")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_top10_misses < 0 or (args.max_top10_misses and not args.score_json):
        parser.error("A nonzero miss allowance requires scores and must be nonnegative")
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#")]
    sample = json.loads(args.sample_json.read_text())
    rows = sample["samples"]
    require(sample["tokenizer_model"] == args.model, "Model profile mismatch")
    require(len(prompts) == len(rows) == args.prompts, "Prompt count mismatch")
    for index, (prompt, row) in enumerate(zip(prompts, rows)):
        require(row["prompt"] == prompt, f"Prompt order/content mismatch at {index}")
        tokens = row["generated_token_ids"]
        require(len(tokens) == args.tokens_per_prompt, f"Token count mismatch at {index}")
        require(all(type(t) is int and t >= 0 for t in tokens), "Invalid token IDs")
    count = args.prompts * args.tokens_per_prompt
    require(sample["summary"]["num_prompts"] == args.prompts, "Invalid prompt summary")
    require(sample["summary"]["total_generated_tokens"] == count, "Invalid token summary")
    result = {"passed": True, "sample_json": str(args.sample_json),
              "prompt_file": str(args.prompt_file), "model": args.model,
              "prompts": args.prompts, "tokens": count, "scored": False}
    if args.score_json:
        score = json.loads(args.score_json.read_text())
        require(score["tokenizer_model"] == args.model, "Scorer model mismatch")
        require(len(score["results"]) == len(rows), "Scorer prompt coverage mismatch")
        top1 = top10 = 0
        outside_top10 = []
        for prompt_index, (sample_row, row) in enumerate(zip(rows, score["results"])):
            require(row["prompt"] == sample_row["prompt"], "Scorer prompt order mismatch")
            tokens = sample_row["generated_token_ids"]
            require(row["generated_token_ids"] == tokens, "Scorer continuation mismatch")
            require(row["num_generated_tokens"] == len(tokens) == len(row["steps"]),
                    "Scorer token coverage mismatch")
            for position, (token, step) in enumerate(zip(tokens, row["steps"])):
                require(step["position"] == position and step["token_id"] == token,
                        "Scorer position/token mismatch")
                rank = step["rank_in_vllm"]
                require(type(rank) is int and rank >= 1, "Invalid reference rank")
                top10 += rank <= 10
                if rank > 10:
                    outside_top10.append(dict(prompt_index=prompt_index, position=position,
                                              token_id=token, rank=rank))
                require(math.isfinite(step["logprob_in_vllm"]), "Non-finite logprob")
                candidates = step["vllm_candidates"]
                require(bool(candidates), "Missing reference candidates")
                require(all(math.isfinite(candidate["logprob"]) for candidate in candidates),
                        "Non-finite reference candidate logprob")
                chosen = [candidate for candidate in candidates
                          if candidate["token_id"] == token]
                require(len(chosen) == 1 and chosen[0]["rank"] == rank and
                        chosen[0]["logprob"] == step["logprob_in_vllm"],
                        "Chosen token/reference candidate mismatch")
                require(step["is_top1_in_vllm"] == (rank == 1), "Invalid top1 flag")
                top1 += rank == 1
        summary = score["summary"]
        require(summary["num_prompts"] == args.prompts and
                summary["total_generated_tokens"] == count, "Scorer summary coverage mismatch")
        require(math.isclose(summary["overall_top10_hit_rate"], top10 / count),
                "Invalid top10 summary")
        result.update(passed=count - top10 <= args.max_top10_misses,
                      max_top10_misses=args.max_top10_misses, top10_misses=count - top10,
                      scored=True, score_json=str(args.score_json),
                      top1_tokens=top1, top10_tokens=top10, finite_logprobs=True,
                      finite_candidate_logprobs=True, outside_top10=outside_top10)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("Alignment failed: top10 miss allowance exceeded; see audit output")


if __name__ == "__main__":
    main()
