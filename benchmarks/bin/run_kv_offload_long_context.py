#!/usr/bin/env python3
"""Matched long-context GPU-cache and native CPU-KV restore baseline.

For every ``context:concurrency`` case this harness creates disjoint,
deterministic prompts and runs these phases:

1. ``cold`` after resetting local and external caches;
2. ``gpu_hit`` as an immediate repeat of the same prompts;
3. optionally, ``cpu_restore`` after resetting only local GPU APC;
4. optionally, ``post_restore_gpu_hit`` as an immediate repeat.

The server's prompt-token source counters distinguish local computation, local
GPU cache hits, and external CPU-KV transfers.  Streaming responses provide
TTFT and decode timing, while a Prometheus sampler captures peak running,
waiting, GPU-KV, and CPU-KV occupancy.  Result directories are immutable and a
checkpoint is written after every phase so failed pressure cases stay visible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


@dataclass(frozen=True)
class MetricSpec:
    candidates: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


def spec(*candidates: tuple[str, dict[str, str]]) -> MetricSpec:
    return MetricSpec(
        tuple((name, tuple(sorted(labels.items()))) for name, labels in candidates)
    )


METRICS = {
    "load_bytes": spec(
        ("vllm:kv_offload_total_bytes_total", {"transfer_type": "CPU_to_GPU"}),
        ("vllm:kv_offload_load_bytes_total", {}),
    ),
    "load_time": spec(
        ("vllm:kv_offload_total_time_total", {"transfer_type": "CPU_to_GPU"}),
        ("vllm:kv_offload_load_time_total", {}),
    ),
    "store_bytes": spec(
        ("vllm:kv_offload_total_bytes_total", {"transfer_type": "GPU_to_CPU"}),
        ("vllm:kv_offload_store_bytes_total", {}),
    ),
    "store_time": spec(
        ("vllm:kv_offload_total_time_total", {"transfer_type": "GPU_to_CPU"}),
        ("vllm:kv_offload_store_time_total", {}),
    ),
    "load_ops": spec(
        ("vllm:kv_offload_size_count", {"transfer_type": "CPU_to_GPU"}),
    ),
    "store_ops": spec(
        ("vllm:kv_offload_size_count", {"transfer_type": "GPU_to_CPU"}),
    ),
    "external_hits": spec(("vllm:external_prefix_cache_hits_total", {})),
    "external_queries": spec(("vllm:external_prefix_cache_queries_total", {})),
    "prefix_hits": spec(("vllm:prefix_cache_hits_total", {})),
    "prefix_queries": spec(("vllm:prefix_cache_queries_total", {})),
    "prompt_local_compute": spec(
        ("vllm:prompt_tokens_by_source_total", {"source": "local_compute"}),
    ),
    "prompt_local_cache": spec(
        ("vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}),
    ),
    "prompt_external_transfer": spec(
        ("vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}),
    ),
    "prompt_tokens": spec(("vllm:prompt_tokens_total", {})),
    "generation_tokens": spec(("vllm:generation_tokens_total", {})),
    "draft_tokens": spec(("vllm:spec_decode_num_draft_tokens_total", {})),
    "accepted_tokens": spec(("vllm:spec_decode_num_accepted_tokens_total", {})),
    "preemptions": spec(("vllm:num_preemptions_total", {})),
    "running": spec(("vllm:num_requests_running", {})),
    "waiting": spec(("vllm:num_requests_waiting", {})),
    "waiting_capacity": spec(
        ("vllm:num_requests_waiting_by_reason", {"reason": "capacity"}),
    ),
    "gpu_cache_usage": spec(("vllm:kv_cache_usage_perc", {})),
    "cpu_cache_usage": spec(("vllm:kv_offload_cpu_cache_usage_perc", {})),
    "cpu_cache_write_usage": spec(
        ("vllm:kv_offload_cpu_cache_write_usage_perc", {}),
    ),
    "cpu_cache_read_usage": spec(
        ("vllm:kv_offload_cpu_cache_read_usage_perc", {}),
    ),
}

GAUGES = (
    "running",
    "waiting",
    "waiting_capacity",
    "gpu_cache_usage",
    "cpu_cache_usage",
    "cpu_cache_write_usage",
    "cpu_cache_read_usage",
)


def parse_labels(raw: str) -> dict[str, str]:
    return dict(LABEL_RE.findall(raw))


def scrape_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        body = response.read().decode()
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in body.splitlines():
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        samples.append(
            (
                match.group("name"),
                parse_labels(match.group("labels") or ""),
                float(match.group("value")),
            )
        )

    values: dict[str, float] = {}
    for key, metric_spec in METRICS.items():
        value = 0.0
        for name, required_pairs in metric_spec.candidates:
            required = dict(required_pairs)
            matching = [
                sample_value
                for sample_name, labels, sample_value in samples
                if sample_name == name
                and all(labels.get(label) == wanted for label, wanted in required.items())
            ]
            if matching:
                value = sum(matching)
                break
        values[key] = value
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after[key] - before[key]
        for key in before
        if key not in GAUGES
    }


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


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
    raise TimeoutError(f"cache reset did not drain in {timeout:.0f}s")


def settle_metrics(
    base_url: str, *, timeout: float = 120, stable_seconds: float = 2.0
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    previous = scrape_metrics(base_url)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.25)
        current = scrape_metrics(base_url)
        changed = any(
            current[key] != previous[key]
            for key in ("load_bytes", "store_bytes", "load_ops", "store_ops")
        )
        idle = current["running"] == 0 and current["waiting"] == 0
        if changed or not idle:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= stable_seconds:
            return current
        previous = current
    return previous


class MetricMonitor:
    def __init__(self, base_url: str, interval: float = 0.5):
        self.base_url = base_url
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.samples.append(
                    {"unix": time.time(), "metrics": scrape_metrics(self.base_url)}
                )
            except Exception as error:  # Preserve transient scrape failures.
                self.samples.append({"unix": time.time(), "error": repr(error)})
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def summary(self) -> dict[str, float]:
        valid = [sample["metrics"] for sample in self.samples if "metrics" in sample]
        if not valid:
            return {}
        return {f"max_{key}": max(sample[key] for sample in valid) for key in GAUGES}


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait(timeout=60)
    started = time.perf_counter()
    first_token: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    response_id: str | None = None
    finish_reason: str | None = None
    chunk_count = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            response_id = response_id or chunk.get("id")
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                text = choice.get("text") or ""
                if text:
                    now = time.perf_counter()
                    if first_token is None:
                        first_token = now
                    chunk_count += 1
                    text_parts.append(text)
    finished = time.perf_counter()
    output_tokens = int(usage.get("completion_tokens") or 0)
    decode_seconds = None if first_token is None else finished - first_token
    tpot_seconds = (
        None
        if decode_seconds is None or output_tokens <= 1
        else decode_seconds / (output_tokens - 1)
    )
    return {
        "id": response_id,
        "elapsed_seconds": finished - started,
        "ttft_seconds": None if first_token is None else first_token - started,
        "decode_seconds": decode_seconds,
        "tpot_seconds": tpot_seconds,
        "decode_tps": None if not tpot_seconds else 1.0 / tpot_seconds,
        "chunk_count": chunk_count,
        "finish_reason": finish_reason,
        "usage": usage,
        "text": "".join(text_parts),
    }


def run_wave(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(len(prompts) + 1)
    monitor = MetricMonitor(base_url)
    before = scrape_metrics(base_url)
    monitor.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(
                stream_completion,
                base_url,
                model,
                prompt,
                max_tokens,
                seed + index,
                timeout,
                barrier,
            )
            for index, prompt in enumerate(prompts)
        ]
        barrier.wait(timeout=60)
        wave_started = time.perf_counter()
        responses = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - wave_started
    after = settle_metrics(base_url)
    monitor.stop()

    output_tokens = sum(
        int((response.get("usage") or {}).get("completion_tokens") or 0)
        for response in responses
    )
    prompt_tokens = sum(
        int((response.get("usage") or {}).get("prompt_tokens") or 0)
        for response in responses
    )
    ttfts = [
        float(response["ttft_seconds"])
        for response in responses
        if response["ttft_seconds"] is not None
    ]
    tpots = [
        float(response["tpot_seconds"])
        for response in responses
        if response["tpot_seconds"] is not None
    ]
    deltas = metric_delta(before, after)
    drafted = deltas["draft_tokens"]
    accepted = deltas["accepted_tokens"]
    return {
        "wall_seconds": wall_seconds,
        "request_throughput": len(prompts) / wall_seconds,
        "output_tps_end_to_end": output_tokens / wall_seconds,
        "total_token_tps_end_to_end": (prompt_tokens + output_tokens) / wall_seconds,
        "prompt_tps_to_last_first_token": (
            None if not ttfts else prompt_tokens / max(ttfts)
        ),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "median_ttft_seconds": None if not ttfts else statistics.median(ttfts),
        "max_ttft_seconds": None if not ttfts else max(ttfts),
        "median_tpot_seconds": None if not tpots else statistics.median(tpots),
        "median_decode_tps": None if not tpots else 1.0 / statistics.median(tpots),
        "median_request_seconds": statistics.median(
            response["elapsed_seconds"] for response in responses
        ),
        "max_request_seconds": max(
            response["elapsed_seconds"] for response in responses
        ),
        "acceptance_rate": None if drafted == 0 else accepted / drafted,
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": deltas,
        "monitor_summary": monitor.summary(),
        "monitor_samples": monitor.samples,
        "responses": responses,
    }


def make_prompt(tokenizer: Any, target_tokens: int, marker: str) -> tuple[str, int]:
    header = (
        f"Radiance long-context baseline {marker}. This request is deliberately "
        "disjoint from every other fixture. "
    )
    unit = (
        f"Preserve deterministic context marker {marker} while measuring cold "
        "prefill, GPU cache reuse, native CPU KV restoration, hybrid recurrent "
        "state, scheduling, and sustained overflow rotation. "
    )
    footer = (
        "\nContinue with a deterministic technical analysis of cache placement and "
        "scheduler behavior."
    )
    # Encode first, then decode the exact prefix. Qwen's tokenizer round-trips
    # these ordinary text tokens exactly, avoiding guessed repetition counts.
    source = header + unit * max(2, target_tokens // 20 + 16) + footer
    token_ids = tokenizer.encode(source, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        raise RuntimeError(f"prompt generator produced only {len(token_ids)} tokens")
    text = tokenizer.decode(
        token_ids[:target_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = tokenizer.encode(text, add_special_tokens=False)
    if len(roundtrip) > target_tokens:
        while len(roundtrip) > target_tokens:
            token_ids = token_ids[: -(len(roundtrip) - target_tokens)]
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            roundtrip = tokenizer.encode(text, add_special_tokens=False)
    return text, len(roundtrip)


def parse_case(raw: str, max_tokens: int) -> tuple[int, int, int]:
    context_raw, concurrency_raw = raw.split(":", 1)
    total_context = int(context_raw)
    concurrency = int(concurrency_raw)
    input_tokens = total_context - max_tokens
    if input_tokens < 1024 or concurrency < 1:
        raise argparse.ArgumentTypeError(f"invalid case {raw!r}")
    return total_context, input_tokens, concurrency


def atomic_checkpoint(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def phase_equality(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cold = phases.get("cold", {}).get("responses", [])
    comparisons: dict[str, Any] = {}
    for name, phase in phases.items():
        if name == "cold":
            continue
        responses = phase.get("responses", [])
        comparisons[name] = {
            "count_matches": len(responses) == len(cold),
            "per_request_byte_equal": [
                left.get("text") == right.get("text")
                for left, right in zip(cold, responses)
            ],
        }
        comparisons[name]["all_byte_equal"] = (
            comparisons[name]["count_matches"]
            and all(comparisons[name]["per_request_byte_equal"])
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="total-context:concurrency, e.g. 131072:4 262144:2",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--reset-timeout", type=float, default=600)
    parser.add_argument("--cpu-restore", action="store_true")
    parser.add_argument("--post-restore-gpu-hit", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite immutable run: {args.output_dir}")
    if args.post_restore_gpu_hit and not args.cpu_restore:
        raise SystemExit("--post-restore-gpu-hit requires --cpu-restore")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "results.json"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cases = [parse_case(raw, args.max_tokens) for raw in args.cases]
    result: dict[str, Any] = {
        "schema": 1,
        "label": args.label,
        "started_unix": time.time(),
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "cpu_restore": args.cpu_restore,
        "post_restore_gpu_hit": args.post_restore_gpu_hit,
        "requested_cases": args.cases,
        "cases": [],
        "complete": False,
    }
    atomic_checkpoint(checkpoint, result)

    for total_context, input_tokens, concurrency in cases:
        case_name = f"ctx{total_context}-c{concurrency}"
        case: dict[str, Any] = {
            "name": case_name,
            "total_context": total_context,
            "target_input_tokens": input_tokens,
            "concurrency": concurrency,
            "phases": {},
            "status": "running",
        }
        result["cases"].append(case)
        try:
            prompts: list[str] = []
            prompt_records: list[dict[str, Any]] = []
            for index in range(concurrency):
                prompt, actual_tokens = make_prompt(
                    tokenizer, input_tokens, f"{args.label}-{case_name}-r{index}"
                )
                prompts.append(prompt)
                prompt_records.append(
                    {
                        "index": index,
                        "actual_input_tokens": actual_tokens,
                        "characters": len(prompt),
                        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                )
            case["prompts"] = prompt_records
            case["external_reset_attempts"] = reset_cache(
                result["base_url"], external=True, timeout=args.reset_timeout
            )
            case["phases"]["cold"] = run_wave(
                result["base_url"],
                args.model,
                prompts,
                args.max_tokens,
                args.seed,
                args.request_timeout,
            )
            atomic_checkpoint(checkpoint, result)

            case["phases"]["gpu_hit"] = run_wave(
                result["base_url"],
                args.model,
                prompts,
                args.max_tokens,
                args.seed,
                args.request_timeout,
            )
            atomic_checkpoint(checkpoint, result)

            if args.cpu_restore:
                case["local_reset_attempts"] = reset_cache(
                    result["base_url"], external=False, timeout=args.reset_timeout
                )
                case["phases"]["cpu_restore"] = run_wave(
                    result["base_url"],
                    args.model,
                    prompts,
                    args.max_tokens,
                    args.seed,
                    args.request_timeout,
                )
                atomic_checkpoint(checkpoint, result)
                if args.post_restore_gpu_hit:
                    case["phases"]["post_restore_gpu_hit"] = run_wave(
                        result["base_url"],
                        args.model,
                        prompts,
                        args.max_tokens,
                        args.seed,
                        args.request_timeout,
                    )
                    atomic_checkpoint(checkpoint, result)
            case["output_equivalence"] = phase_equality(case["phases"])
            case["status"] = "complete"
        except Exception as error:
            case["status"] = "failed"
            case["error"] = repr(error)
            atomic_checkpoint(checkpoint, result)
            if not args.continue_on_error:
                raise
        atomic_checkpoint(checkpoint, result)

    result["finished_unix"] = time.time()
    result["complete"] = all(case["status"] == "complete" for case in result["cases"])
    atomic_checkpoint(checkpoint, result)
    summary = []
    for case in result["cases"]:
        row: dict[str, Any] = {"case": case["name"], "status": case["status"]}
        for phase_name, phase in case.get("phases", {}).items():
            row[phase_name] = {
                "ttft": phase.get("median_ttft_seconds"),
                "output_tps": phase.get("output_tps_end_to_end"),
                "decode_tps": phase.get("median_decode_tps"),
                "local_compute": phase.get("metrics_delta", {}).get(
                    "prompt_local_compute"
                ),
                "local_cache": phase.get("metrics_delta", {}).get(
                    "prompt_local_cache"
                ),
                "external_transfer": phase.get("metrics_delta", {}).get(
                    "prompt_external_transfer"
                ),
                "load_bytes": phase.get("metrics_delta", {}).get("load_bytes"),
            }
        summary.append(row)
    print(json.dumps(summary, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
