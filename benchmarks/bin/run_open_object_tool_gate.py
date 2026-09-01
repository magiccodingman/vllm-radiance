#!/usr/bin/env python3
"""Live wire gate for generic deferred-tool wrappers with nested arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Expected:
    name: str
    arguments: dict[str, Any]


CASES: dict[str, tuple[Expected, ...]] = {
    "command": (
        Expected(
            "mcp__smacx__smac_command",
            {
                "command": "acknowledge_popup",
                "match_id": "match-provider-wire-probe",
                "session_id": "session-provider-wire-probe",
                "expected_revision": "16893526205145507771",
            },
        ),
    ),
    "lan": (Expected("mcp__smacx__smac_lan", {"action": "status"}),),
    "parallel": (
        Expected("mcp__smacx__smac_list", {"kind": "factions"}),
        Expected("mcp__smacx__smac_chat", {"action": "list"}),
    ),
    "memory": (
        Expected(
            "mcp__smacx__smac_memory",
            {
                "action": "search",
                "match_id": "match-provider-wire-probe",
                "query": "match overview",
            },
        ),
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search additional tools loaded on demand.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_call",
            "description": "Invoke a deferred tool by name with its arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    # Intentionally omit additionalProperties. JSON Schema's
                    # default is true; this exact shape exposed the corruption.
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
]


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def request_body(
    expected: tuple[Expected, ...],
    *,
    model: str,
    stream: bool,
    request_id: str,
    preserve_thinking: bool,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    rendered = [
        {"name": call.name, "arguments": call.arguments} for call in expected
    ]
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Emit the requested deferred tool calls exactly. Preserve every "
                    "nested argument and emit no prose."
                ),
            },
            {
                "role": "user",
                "content": "Call tool_call with these values exactly:\n"
                + json.dumps(rendered, ensure_ascii=False, indent=2),
            },
        ],
        "tools": TOOLS,
        # A named choice is exactly one call in vLLM's structural-tag policy.
        # Use required for the explicit two-call fixture so parallel semantics
        # are tested without conflating them with named-choice policy.
        "tool_choice": (
            "required"
            if len(expected) > 1
            else {"type": "function", "function": {"name": "tool_call"}}
        ),
        "parallel_tool_calls": True,
        "stream": stream,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "reasoning_effort": "low",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": preserve_thinking,
        },
        "max_completion_tokens": 1024,
        "request_id": request_id,
        "return_token_ids": True,
        "include_reasoning": True,
    }


def collect_stream(response: Any, capture: Any) -> tuple[list[dict[str, Any]], str | None]:
    slots: dict[int, dict[str, Any]] = {}
    finish_reason = None
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        capture.write(compact({"kind": "wire_line", "line": line}) + "\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        for choice in event.get("choices", []):
            finish_reason = choice.get("finish_reason") or finish_reason
            for fragment in (choice.get("delta") or {}).get("tool_calls") or []:
                index = int(fragment.get("index", 0))
                slot = slots.setdefault(
                    index, {"function": {"name": "", "arguments": ""}}
                )
                function = fragment.get("function") or {}
                slot["function"]["name"] += str(function.get("name") or "")
                slot["function"]["arguments"] += str(
                    function.get("arguments") or ""
                )
    return [slots[index] for index in sorted(slots)], finish_reason


def validate(
    expected: tuple[Expected, ...], calls: list[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    if len(calls) != len(expected):
        issues.append(f"expected {len(expected)} calls, received {len(calls)}")
    for index, wanted in enumerate(expected):
        if index >= len(calls):
            break
        function = calls[index].get("function") or {}
        if function.get("name") != "tool_call":
            issues.append(f"call {index}: outer name={function.get('name')!r}")
        try:
            outer = json.loads(function.get("arguments") or "")
        except (TypeError, json.JSONDecodeError) as exc:
            issues.append(f"call {index}: invalid outer arguments: {exc}")
            continue
        if outer.get("name") != wanted.name:
            issues.append(f"call {index}: nested name={outer.get('name')!r}")
        if outer.get("arguments") != wanted.arguments:
            issues.append(
                f"call {index}: nested arguments={outer.get('arguments')!r}; "
                f"expected={wanted.arguments!r}"
            )
        extras = set(outer) - {"name", "arguments"}
        if extras:
            issues.append(f"call {index}: flattened outer fields={sorted(extras)}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--capture-root", default="benchmarks/results")
    parser.add_argument(
        "--preserve-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.capture_root).resolve() / f"{stamp}_open-object-tool-gate"
    run_dir.mkdir(parents=True, exist_ok=False)
    capture_path = run_dir / "wire.jsonl"
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    endpoint += "/chat/completions"

    results: list[dict[str, Any]] = []
    with capture_path.open("x", encoding="utf-8") as capture:
        for run in range(1, args.runs + 1):
            for case_name, expected in CASES.items():
                for stream in (True, False):
                    request_id = f"open-object-{stamp}-{run}-{case_name}-{'s' if stream else 'n'}"
                    body = request_body(
                        expected,
                        model=args.model,
                        stream=stream,
                        request_id=request_id,
                        preserve_thinking=args.preserve_thinking,
                        seed=args.seed + run - 1,
                        temperature=args.temperature,
                    )
                    capture.write(compact({"kind": "request", "body": body}) + "\n")
                    started = time.monotonic()
                    request = Request(
                        endpoint,
                        data=compact(body).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    try:
                        with urlopen(request, timeout=180) as response:
                            headers = dict(response.headers.items())
                            capture.write(
                                compact({"kind": "response_headers", "headers": headers})
                                + "\n"
                            )
                            if stream:
                                calls, finish_reason = collect_stream(response, capture)
                            else:
                                raw = response.read().decode("utf-8", errors="replace")
                                capture.write(
                                    compact({"kind": "wire_body", "body": raw}) + "\n"
                                )
                                payload = json.loads(raw)
                                choice = payload["choices"][0]
                                calls = choice["message"].get("tool_calls") or []
                                finish_reason = choice.get("finish_reason")
                        issues = validate(expected, calls)
                    except Exception as exc:  # preserve the request-level failure
                        calls, finish_reason = [], None
                        issues = [f"request failure: {exc!r}"]
                    result = {
                        "run": run,
                        "case": case_name,
                        "stream": stream,
                        "request_id": request_id,
                        # vLLM intentionally uses stop for a named tool choice;
                        # argument integrity, not that policy, is this gate.
                        "finish_reason": finish_reason,
                        "passed": not issues,
                        "issues": issues,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                    results.append(result)
                    capture.write(compact({"kind": "result", "result": result}) + "\n")
                    capture.flush()
                    print(compact(result), flush=True)

    summary = {
        "requests": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
