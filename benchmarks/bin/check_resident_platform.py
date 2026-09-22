#!/usr/bin/env python3
"""Bounded resident qualification. Saves wire evidence before asserting.

No benchmark sweep, output retokenization, tolerance or forced-EOS bypass.
"""
import argparse
import base64
import hashlib
import io
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:11436")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vision", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    def call(name, endpoint, body):
        begin = time.monotonic()
        with urlopen(Request(a.url + endpoint, data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"}), timeout=120) as r:
            result = json.load(r)
        (a.out / f"{name}.json").write_text(json.dumps(
            {"request": body, "response": result, "elapsed_s": time.monotonic()-begin}, indent=2))
        return result
    call("owners", "/collective_rpc", {"method": "platform_snapshot"})
    common = {"model": a.model, "temperature": 0, "seed": 0,
              "return_token_ids": True, "max_tokens": 32}
    prompt = "The capital of France is"
    salt = uuid.uuid4().hex
    rows = []
    for label, cache in (("fresh", salt), ("prefix", salt), ("independent", uuid.uuid4().hex)):
        body = {**common, "prompt": prompt, "cache_salt": cache,
                "stream": True, "stream_options": {"include_usage": True}}
        begin = time.monotonic()
        events, ids, text, usage = [], [], "", None
        with urlopen(Request(a.url + "/v1/completions", data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"}), timeout=120) as r:
            done = False
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                events.append({"t_s": time.monotonic()-begin, "event": event})
                usage = event.get("usage") or usage
                for c in event["choices"]:
                    ids.extend(c.get("token_ids") or [])
                    text += c.get("text", "")
        elapsed = time.monotonic()-begin
        token_times = [e["t_s"] for e in events if any(c.get("token_ids") for c in e["event"]["choices"])]
        record = {"request": body, "events": events, "usage": usage,
                  "ids": ids, "text": text, "done": done, "elapsed_s": elapsed,
                  "ttft_s": token_times[0] if token_times else None,
                  "steady_tps": (len(ids)-1)/(token_times[-1]-token_times[0]) if len(token_times)>1 else None,
                  "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        (a.out / f"{label}.json").write_text(json.dumps(record, indent=2))
        assert done and usage and usage["completion_tokens"] == len(ids) and ids
        rows.append(record)
    assert rows[0]["ids"] == rows[1]["ids"] == rows[2]["ids"], "fresh/prefix trajectory mismatch"
    # The hybrid attention block is 1568 tokens on this lane; a five-token
    # repeated request cannot prove actual prefix reuse.
    prefix_ids = call("prefix-tokenize", "/tokenize", {"model": a.model,
        "prompt": "A red apple sits on a table. " * 210})["tokens"][:1600]
    assert len(prefix_ids) == 1600
    prefix_salt = uuid.uuid4().hex
    prefix_results = []
    for index in range(2):
        with urlopen(a.url + "/metrics", timeout=10) as r:
            (a.out / f"prefix-metrics-{index}.prom").write_bytes(r.read())
        prefix_results.append(call(f"real-prefix-{index}", "/v1/completions", {
            **common, "prompt": prefix_ids, "cache_salt": prefix_salt, "max_tokens": 4}))
    with urlopen(a.url + "/metrics", timeout=10) as r:
        (a.out / "prefix-metrics-after.prom").write_bytes(r.read())
    assert prefix_results[0]["choices"][0]["token_ids"] == prefix_results[1]["choices"][0]["token_ids"]
    # Tokenize only the ORIGINAL prompt. Generated IDs come directly from API.
    original = call("tokenize", "/tokenize", {"model": a.model, "prompt": prompt})["tokens"]
    for step in range(min(3, len(rows[0]["ids"]))):
        result = call(f"recompute-{step}", "/v1/completions", {
            **common, "prompt": original + rows[0]["ids"][:step], "max_tokens": 1,
            "cache_salt": uuid.uuid4().hex})
        assert result["choices"][0]["token_ids"] == [rows[0]["ids"][step]]
    chat = {"model": a.model, "temperature": 0, "seed": 0, "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False}}
    schema = {"type": "object", "properties": {"answer": {"type": "object",
              "properties": {"city": {"type": "string"}}, "required": ["city"],
              "additionalProperties": False}}, "required": ["answer"], "additionalProperties": False}
    result = call("structured", "/v1/chat/completions", {**chat,
        "messages": [{"role": "user", "content": "Return Paris as answer.city."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": schema}}})
    assert json.loads(result["choices"][0]["message"]["content"]) == {"answer": {"city": "Paris"}}
    result = call("tool", "/v1/chat/completions", {**chat,
        "messages": [{"role": "user", "content": "Use lookup for Paris. Do not answer in text."}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": schema}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}}})
    tool = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert tool["name"] == "lookup" and json.loads(tool["arguments"]) == {"answer": {"city": "Paris"}}
    call("bounded-prefill", "/v1/completions", {**common, "prompt": "A red apple sits on a table. " * 64,
                                               "max_tokens": 16, "cache_salt": uuid.uuid4().hex})
    if a.vision:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (128, 128), (255, 0, 0)).save(buf, format="PNG")
        content = [{"type": "text", "text": "Name the single dominant color. Answer with one word."},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}]
        outputs = []
        for i in range(2):
            result = call(f"vision-{i}", "/v1/chat/completions", {**chat, "messages": [{"role": "user", "content": content}]})
            outputs.append(result["choices"][0]["message"]["content"])
        assert outputs[0] == outputs[1] and "red" in outputs[0].lower()
    if a.profile:
        call("profile-start", "/collective_rpc", {"method": "platform_profile_start"})
        try:
            call("profiled-short", "/v1/completions", {**common, "prompt": prompt, "max_tokens": 8,
                                                       "cache_salt": uuid.uuid4().hex})
        finally:
            call("profile-stop", "/collective_rpc", {"method": "platform_profile_stop"})
    (a.out / "result.json").write_text(json.dumps({"status": "PASS", "short_requests": [
        {k: row[k] for k in ("ttft_s", "steady_tps", "elapsed_s", "token_sha256", "usage")} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
