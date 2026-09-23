"""Actual Qwen parser over synthetic token streams, including tool boundaries."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from qwen_r9700_lab.conformance_transport import Completion, ProtocolError
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, write_private


def qualify_parser(checkpoint, expected_sha256, output, *, parser_config=None):
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.qwen3 import Qwen3Parser

    source = Path(inspect.getfile(Qwen3Parser))
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise DiagnosticError("native parser binding changed")
    parser_class = Qwen3Parser
    if parser_config is not None:
        from vllm.parser.parser_manager import ParserManager

        parser_class = ParserManager.get_parser(**parser_config)
        if parser_class is None:
            raise DiagnosticError("configured serving parser is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tool = {
        "type": "function",
        "function": {
            "name": "record",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    request = ChatCompletionRequest(
        model="fixture",
        messages=[{"role": "user", "content": "Record."}],
        tools=[tool],
        tool_choice="auto",
        stream=True,
    )
    xml = (
        "<tool_call>\n<function=record>\n<parameter=text>café λ 🙂</parameter>\n"
        "</function>\n</tool_call>"
    )
    fixtures = {
        "valid_tool": ("Plan.\n</think>\n\n" + xml, True),
        "marker_in_thinking": ('The token "<tool_call>" is syntax.\n</think>\n\n' + xml, True),
        "answer_only": ("Plan.\n</think>\n\nThe value is 37.", False),
        "colon_only": ("Plan.\n</think>\n\nThe next step is:", False),
        "unfinished_name": ("Plan.\n</think>\n\n<tool_call>\n<function=", False),
    }
    rows = []
    for label, (text, expected_tool) in fixtures.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        for width in sorted({1, 2, 7, len(ids)}):
            parser = parser_class(tokenizer, tools=request.tools)
            merged, previous, deltas = Completion(), "", []
            for offset in range(0, len(ids), width):
                end = min(offset + width, len(ids))
                current = tokenizer.decode(ids[:end], skip_special_tokens=False).rstrip("\ufffd")
                if not current.startswith(previous):
                    raise DiagnosticError("synthetic tokenizer stream changed an emitted prefix")
                delta = parser.parse_delta(
                    current[len(previous) :], ids[offset:end], request, finished=end == len(ids)
                )
                if delta is not None:
                    value = delta.model_dump(exclude_none=True)
                    deltas.append(value)
                    merged.accept({"choices": [{"index": 0, "delta": value}]})
                previous = current
            complete = []
            try:
                merged.accept(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if merged.tools else "stop",
                            }
                        ]
                    }
                )
                complete = merged.result()["tools"]
            except ProtocolError:
                if expected_tool:
                    raise
            passed = (
                (
                    len(complete) == 1
                    and complete[0]["name"] == "record"
                    and complete[0]["parsed_arguments"] == {"text": "café λ 🙂"}
                )
                if expected_tool
                else not complete
            )
            rows.append(
                {"fixture": label, "chunk_tokens": width, "passed": passed, "deltas": deltas}
            )
    result = {
        "source_sha256": expected_sha256,
        "entrypoint": "registered_serving_parser" if parser_config else "direct_Qwen3Parser",
        "parser_config": parser_config,
        "rows": rows,
        "passed": bool(rows) and all(r["passed"] for r in rows),
        "gpu_executed": False,
        "tools_executed": False,
    }
    write_private(output, result)
    if not result["passed"]:
        raise DiagnosticError("native parser fixture failed; raw synthetic deltas retained")
    return result
