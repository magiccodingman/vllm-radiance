#!/usr/bin/env python3
"""Characterize fixed-prompt greedy divergence without weakening strict gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def common_prefix(left: list[Any] | str, right: list[Any] | str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def samples_by_prompt(result: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    samples = result.get("samples")
    if isinstance(samples, list) and samples:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            grouped.setdefault(int(sample["prompt_index"]), []).append(sample)
        for prompt_samples in grouped.values():
            prompt_samples.sort(key=lambda sample: int(sample["repetition"]))
        return grouped

    texts = result.get("generated_texts", [])
    input_lens = result.get("input_lens", [])
    output_lens = result.get("output_lens", [])
    return {
        index: [
            {
                "prompt_index": index,
                "repetition": 0,
                "generated_text": text,
                "input_len": input_lens[index],
                "output_len": output_lens[index],
            }
        ]
        for index, text in enumerate(texts)
    }


def repeatability(grouped: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    compared = 0
    mismatches: list[dict[str, int]] = []
    for prompt_index, samples in sorted(grouped.items()):
        reference = samples[0]
        for sample in samples[1:]:
            compared += 1
            if (
                sample.get("generated_text") != reference.get("generated_text")
                or sample.get("output_len") != reference.get("output_len")
            ):
                mismatches.append(
                    {
                        "prompt_index": prompt_index,
                        "repetition": int(sample["repetition"]),
                    }
                )
    return {
        "status": "passed" if not mismatches else "failed",
        "compared_repetitions": compared,
        "mismatches": mismatches,
    }


def logprob_margin(sample: dict[str, Any], index: int | None) -> float | None:
    if index is None:
        return None
    top = sample.get("top_logprobs")
    if not isinstance(top, list) or index >= len(top) or not isinstance(top[index], dict):
        return None
    values = sorted((float(value) for value in top[index].values()), reverse=True)
    return values[0] - values[1] if len(values) >= 2 else None


def token_context(tokens: list[Any] | None, index: int | None) -> list[Any] | None:
    if tokens is None or index is None:
        return None
    return tokens[max(0, index - 3) : index + 4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    base_grouped = samples_by_prompt(baseline)
    cand_grouped = samples_by_prompt(candidate)
    prompt_indices = sorted(base_grouped.keys() & cand_grouped.keys())
    divergences: list[dict[str, Any]] = []
    exact_matches = 0
    for prompt_index in prompt_indices:
        base = base_grouped[prompt_index][0]
        cand = cand_grouped[prompt_index][0]
        base_text = str(base["generated_text"])
        cand_text = str(cand["generated_text"])
        base_tokens = base.get("tokens")
        cand_tokens = cand.get("tokens")
        token_prefix = None
        if isinstance(base_tokens, list) and isinstance(cand_tokens, list):
            token_prefix = common_prefix(base_tokens, cand_tokens)
        exact = (
            base_text == cand_text and base.get("output_len") == cand.get("output_len")
        )
        if exact:
            exact_matches += 1
            continue
        char_prefix = common_prefix(base_text, cand_text)
        divergences.append(
            {
                "prompt_index": prompt_index,
                "common_prefix_characters": char_prefix,
                "first_divergence_token": token_prefix,
                "baseline_logprob_margin": logprob_margin(base, token_prefix),
                "candidate_logprob_margin": logprob_margin(cand, token_prefix),
                "baseline_token_context": token_context(base_tokens, token_prefix),
                "candidate_token_context": token_context(cand_tokens, token_prefix),
                "baseline_text_context": base_text[max(0, char_prefix - 40) : char_prefix + 80],
                "candidate_text_context": cand_text[max(0, char_prefix - 40) : char_prefix + 80],
            }
        )

    fixture_match = baseline.get("prompts_sha256") == candidate.get("prompts_sha256")
    report = {
        "status": (
            "passed"
            if fixture_match and prompt_indices and exact_matches == len(prompt_indices)
            else "failed"
        ),
        "fixture_match": fixture_match,
        "common_prompts": len(prompt_indices),
        "exact_matches": exact_matches,
        "baseline_repeatability": repeatability(base_grouped),
        "candidate_repeatability": repeatability(cand_grouped),
        "divergences": divergences,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
