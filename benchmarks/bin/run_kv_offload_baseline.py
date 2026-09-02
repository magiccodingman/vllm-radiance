#!/usr/bin/env python3
"""Bounded native CPU-KV restore and shared-prefix convoy baseline.

Each concurrency case uses a disjoint prompt. It primes the CPU tier once,
resets only local GPU APC, then releases an identical-prefix request wave at a
barrier. Streaming captures TTFT while Prometheus counters prove whether CPU
loads occurred and how much data moved. Result directories are immutable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import statistics
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SAMPLE_RE = re.compile(
    r'^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$'
)
METRICS = {
    "load_bytes": ("vllm:kv_offload_load_bytes_total", None),
    "load_time": ("vllm:kv_offload_load_time_total", None),
    "store_bytes": ("vllm:kv_offload_store_bytes_total", None),
    "store_time": ("vllm:kv_offload_store_time_total", None),
    "load_ops": ("vllm:kv_offload_size_count", 'transfer_type="CPU_to_GPU"'),
    "store_ops": ("vllm:kv_offload_size_count", 'transfer_type="GPU_to_CPU"'),
    "external_hits": ("vllm:external_prefix_cache_hits_total", None),
    "external_queries": ("vllm:external_prefix_cache_queries_total", None),
    "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", None),
    "accepted_tokens": ("vllm:spec_decode_num_accepted_tokens_total", None),
}


def request_json(method: str, url: str, timeout: float = 120) -> dict[str, Any]:
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def reset_cache(base_url: str, *, external: bool) -> int:
    query = urllib.parse.urlencode({"reset_external": str(external).lower()})
    for attempt in range(1, 481):
        result = request_json(
            "POST", f"{base_url}/reset_prefix_cache?{query}", timeout=30
        )
        if result.get("success") is True:
            return attempt
        time.sleep(0.25)
    raise TimeoutError("prefix-cache reset did not drain within 120 seconds")


def scrape_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        body = response.read().decode()
    totals = dict.fromkeys(METRICS, 0.0)
    for line in body.splitlines():
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        for key, (name, required_label) in METRICS.items():
            if match.group("name") != name:
                continue
            labels = match.group("labels") or ""
            if required_label is None or required_label in labels:
                totals[key] += float(match.group("value"))
    return totals


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def wait_for_increase(
    base_url: str, baseline: dict[str, float], key: str, timeout: float = 120
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    latest = scrape_metrics(base_url)
    while latest[key] <= baseline[key] and time.monotonic() < deadline:
        time.sleep(0.25)
        latest = scrape_metrics(base_url)
    return latest


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    first_token: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if first_token is None:
                    first_token = time.perf_counter()
                text_parts.append(choices[0]["text"])
    finished = time.perf_counter()
    return {
        "ttft_seconds": None if first_token is None else first_token - started,
        "elapsed_seconds": finished - started,
        "text": "".join(text_parts),
        "usage": usage,
    }


def make_prompt(case: str, repetitions: int) -> str:
    unit = (
        f"Radiance native KV offload baseline {case}. Preserve this shared "
        "prefix while analyzing deterministic CPU-to-GPU restoration, hybrid "
        "recurrent state, and simultaneous request scheduling. "
    )
    return unit * repetitions + "\nReturn one concise conclusion."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prompt-repetitions", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable result: {args.output}")
    if any(value < 1 for value in args.concurrencies):
        raise SystemExit("concurrencies must be positive")

    base_url = args.base_url.rstrip("/")
    result: dict[str, Any] = {
        "schema": 1,
        "started_unix": time.time(),
        "base_url": base_url,
        "model": args.model,
        "temperature": 0,
        "seed": args.seed,
        "prompt_repetitions": args.prompt_repetitions,
        "max_tokens": args.max_tokens,
        "cases": [],
    }
    for concurrency in args.concurrencies:
        prompt = make_prompt(f"c{concurrency}", args.prompt_repetitions)
        initial_reset = reset_cache(base_url, external=True)
        before_prime = scrape_metrics(base_url)
        prime = stream_completion(
            base_url, args.model, prompt, args.max_tokens, args.seed
        )
        after_prime = wait_for_increase(base_url, before_prime, "store_bytes")
        local_reset = reset_cache(base_url, external=False)
        before_wave = scrape_metrics(base_url)

        barrier = threading.Barrier(concurrency + 1)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = [
                executor.submit(
                    stream_completion,
                    base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.seed,
                    barrier,
                )
                for _ in range(concurrency)
            ]
            wave_started = time.perf_counter()
            barrier.wait(timeout=30)
            responses = [future.result() for future in futures]
            wave_seconds = time.perf_counter() - wave_started
        time.sleep(0.5)
        after_wave = scrape_metrics(base_url)

        output_tokens = sum(
            int((response.get("usage") or {}).get("completion_tokens") or 0)
            for response in responses
        )
        ttfts = [
            float(response["ttft_seconds"])
            for response in responses
            if response["ttft_seconds"] is not None
        ]
        accepted = after_wave["accepted_tokens"] - before_wave["accepted_tokens"]
        drafted = after_wave["draft_tokens"] - before_wave["draft_tokens"]
        result["cases"].append(
            {
                "concurrency": concurrency,
                "prompt_characters": len(prompt),
                "prompt_tokens": int(
                    (prime.get("usage") or {}).get("prompt_tokens") or 0
                ),
                "initial_reset_attempts": initial_reset,
                "local_reset_attempts": local_reset,
                "prime": prime,
                "prime_metrics_delta": delta(before_prime, after_prime),
                "wave": {
                    "wall_seconds": wave_seconds,
                    "request_throughput": concurrency / wave_seconds,
                    "output_tps": output_tokens / wave_seconds,
                    "output_tokens": output_tokens,
                    "median_ttft_seconds": statistics.median(ttfts),
                    "max_ttft_seconds": max(ttfts),
                    "median_request_seconds": statistics.median(
                        response["elapsed_seconds"] for response in responses
                    ),
                    "max_request_seconds": max(
                        response["elapsed_seconds"] for response in responses
                    ),
                    "acceptance_rate": None if drafted == 0 else accepted / drafted,
                    "responses": responses,
                },
                "wave_metrics_delta": delta(before_wave, after_wave),
            }
        )

    result["finished_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            [
                {
                    "c": case["concurrency"],
                    "prompt_tokens": case["prompt_tokens"],
                    "ttft": case["wave"]["median_ttft_seconds"],
                    "output_tps": case["wave"]["output_tps"],
                    "load_bytes": case["wave_metrics_delta"]["load_bytes"],
                    "load_ops": case["wave_metrics_delta"]["load_ops"],
                    "external_hits": case["wave_metrics_delta"]["external_hits"],
                }
                for case in result["cases"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
