#!/usr/bin/env python3
"""Run fixed, meaningful greedy prompts for cross-configuration correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"completion request failed ({exc.code}): {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(prompt, str) for prompt in prompts
    ):
        raise SystemExit("prompt fixture must be a non-empty JSON string list")

    generated_texts: list[str] = []
    input_lens: list[int] = []
    output_lens: list[int] = []
    for index, prompt in enumerate(prompts):
        response = post_json(
            f"{args.base_url.rstrip('/')}/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "seed": 20260822 + index,
            },
            args.timeout,
        )
        choices = response.get("choices")
        usage = response.get("usage")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
            raise RuntimeError(f"unexpected completion response for prompt {index}: {response}")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("text"), str):
            raise RuntimeError(f"completion text missing for prompt {index}")
        generated_texts.append(choice["text"])
        input_lens.append(int(usage["prompt_tokens"]))
        output_lens.append(int(usage["completion_tokens"]))

    fixture_bytes = args.prompts.read_bytes()
    result = {
        "workload": "correctness_fixed",
        "temperature": "0",
        "seed": "per-prompt:20260822+index",
        "prompts": prompts,
        "prompts_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "input_lens": input_lens,
        "output_lens": output_lens,
        "generated_texts": generated_texts,
        "completed": len(prompts),
        "failed": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
