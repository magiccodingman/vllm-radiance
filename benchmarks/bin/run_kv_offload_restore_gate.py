#!/usr/bin/env python3
"""Deterministic cold -> GPU hit -> CPU restore qualification gate.

The gate deliberately resets only vLLM's local GPU prefix cache between the
second and third requests.  The native OffloadingConnector cache remains
resident, so the third request must produce CPU->GPU bytes and externally
cached prompt tokens.  Outputs must remain byte-identical at temperature 0.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


METRICS = {
    "load_bytes": "vllm:kv_offload_load_bytes_total",
    "load_time": "vllm:kv_offload_load_time_total",
    "store_bytes": "vllm:kv_offload_store_bytes_total",
    "store_time": "vllm:kv_offload_store_time_total",
    "external_hits": "vllm:external_prefix_cache_hits_total",
    "external_queries": "vllm:external_prefix_cache_queries_total",
}
SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def scrape_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        text = response.read().decode()
    totals = dict.fromkeys(METRICS, 0.0)
    inverse = {metric: key for key, metric in METRICS.items()}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match and match.group("name") in inverse:
            totals[inverse[match.group("name")]] += float(match.group("value"))
    return totals


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def wait_for_metric_increase(
    base_url: str,
    baseline: dict[str, float],
    key: str,
    timeout: float,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    latest = scrape_metrics(base_url)
    while latest[key] <= baseline[key] and time.monotonic() < deadline:
        time.sleep(0.25)
        latest = scrape_metrics(base_url)
    return latest


def reset_cache(base_url: str, *, external: bool, timeout: float) -> int:
    query = urllib.parse.urlencode({"reset_external": str(external).lower()})
    deadline = time.monotonic() + timeout
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        result = request_json(
            "POST", f"{base_url}/reset_prefix_cache?{query}", timeout=30
        )
        if result.get("success") is True:
            return attempts
        time.sleep(0.25)
    raise TimeoutError(f"prefix-cache reset did not drain in {timeout}s")


def cached_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    response = request_json(
        "POST", f"{base_url}/v1/completions", payload, timeout=1800
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    return {
        "elapsed_seconds": elapsed,
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "cached_tokens": cached_tokens(response),
        "id": response.get("id"),
    }


def make_prompt(repetitions: int) -> str:
    # Repetition is intentional: the same immutable prompt is reused in every
    # phase, while the fixed preamble keeps this fixture disjoint from normal
    # benchmark warmups and production traffic.
    unit = (
        "Radiance KV restore fixture 2026-09-01. Analyze the invariant that "
        "a deterministic cache restore must preserve every target token, "
        "hybrid recurrent state, and draft verification boundary. "
    )
    return unit * repetitions + "\nReturn exactly one concise validation sentence."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-repetitions", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reset-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable result: {args.output}")
    if args.prompt_repetitions < 1 or args.max_tokens < 1:
        raise SystemExit("prompt repetitions and max tokens must be positive")

    base_url = args.base_url.rstrip("/")
    prompt = make_prompt(args.prompt_repetitions)
    result: dict[str, Any] = {
        "schema": 1,
        "started_unix": time.time(),
        "base_url": base_url,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "prompt_repetitions": args.prompt_repetitions,
        "prompt_characters": len(prompt),
        "max_tokens": args.max_tokens,
    }

    result["initial_reset_attempts"] = reset_cache(
        base_url, external=True, timeout=args.reset_timeout
    )
    metrics_0 = scrape_metrics(base_url)
    cold = completion(base_url, args.model, prompt, args.max_tokens, args.seed)
    metrics_1 = wait_for_metric_increase(
        base_url, metrics_0, "store_bytes", args.reset_timeout
    )
    gpu_hit = completion(base_url, args.model, prompt, args.max_tokens, args.seed)
    metrics_2 = scrape_metrics(base_url)

    # Preserve the external CPU tier while forcing the next lookup off GPU.
    result["local_reset_attempts"] = reset_cache(
        base_url, external=False, timeout=args.reset_timeout
    )
    metrics_3 = scrape_metrics(base_url)
    cpu_restore = completion(base_url, args.model, prompt, args.max_tokens, args.seed)
    metrics_4 = wait_for_metric_increase(
        base_url, metrics_3, "load_bytes", args.reset_timeout
    )

    cold_delta = metric_delta(metrics_0, metrics_1)
    gpu_delta = metric_delta(metrics_1, metrics_2)
    restore_delta = metric_delta(metrics_3, metrics_4)
    prompt_tokens = int((cold.get("usage") or {}).get("prompt_tokens") or 0)
    # The OpenAI usage extension is not enabled in the production server.  The
    # connector's query counter is nevertheless exact here: on the immediate
    # repeat it only sees the suffix that the local GPU APC did not satisfy.
    # On the restore phase, external_hits is the exact number served by CPU.
    gpu_local_cached_tokens = max(
        0, prompt_tokens - int(gpu_delta["external_queries"])
    )
    cpu_restored_tokens = int(restore_delta["external_hits"])
    cpu_restore_equal = cold["text"] == cpu_restore["text"]
    gpu_hit_equal = cold["text"] == gpu_hit["text"]
    checks = {
        "meaningful_prompt": prompt_tokens >= 1024,
        "cold_stored_to_cpu": cold_delta["store_bytes"] > 0,
        "gpu_repeat_cached": gpu_local_cached_tokens > 0,
        "cpu_restore_loaded_bytes": restore_delta["load_bytes"] > 0,
        "cpu_restore_external_hit": restore_delta["external_hits"] > 0,
        "cpu_restore_cached_tokens": cpu_restored_tokens > 0,
        "cpu_restore_byte_identical": cpu_restore_equal,
        "gpu_hit_byte_identical": gpu_hit_equal,
    }
    result.update(
        {
            "finished_unix": time.time(),
            "phases": {
                "cold": cold,
                "gpu_hit": gpu_hit,
                "cpu_restore": cpu_restore,
            },
            "metrics": {
                "before": metrics_0,
                "after_cold": metrics_1,
                "after_gpu_hit": metrics_2,
                "before_cpu_restore": metrics_3,
                "after_cpu_restore": metrics_4,
                "cold_delta": cold_delta,
                "gpu_hit_delta": gpu_delta,
                "cpu_restore_delta": restore_delta,
            },
            "derived": {
                "gpu_local_cached_tokens": gpu_local_cached_tokens,
                "cpu_restored_tokens": cpu_restored_tokens,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
