#!/usr/bin/env python3
"""GPU-free regression check for open nested Qwen tool arguments."""

from __future__ import annotations

import json
import xgrammar as xgr

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.structural_tag_registry import (
    _normalize_qwen_tool_schemas,
    get_model_structural_tag,
)


def bridge_tool(arguments_schema: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "tool_call",
            "description": "Invoke a deferred tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": arguments_schema,
                },
                "required": ["name", "arguments"],
            },
        },
    }


def main() -> None:
    generic = bridge_tool({"type": "object"})
    tools = [generic]
    _normalize_qwen_tool_schemas(tools)

    parameters = generic["function"]["parameters"]  # type: ignore[index]
    assert isinstance(parameters, dict)
    # The function's root object remains exactly as supplied.
    assert "additionalProperties" not in parameters
    nested = parameters["properties"]["arguments"]  # type: ignore[index]
    assert nested == {"type": "object", "additionalProperties": True}

    # Idempotence and explicit caller intent are both preserved.
    _normalize_qwen_tool_schemas(tools)
    assert nested == {"type": "object", "additionalProperties": True}
    closed = bridge_tool({"type": "object", "additionalProperties": False})
    declared = bridge_tool(
        {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        }
    )
    controls = [closed, declared]
    _normalize_qwen_tool_schemas(controls)
    assert closed["function"]["parameters"]["properties"]["arguments"] == {  # type: ignore[index]
        "type": "object",
        "additionalProperties": False,
    }
    assert "additionalProperties" not in (  # type: ignore[operator]
        declared["function"]["parameters"]["properties"]["arguments"]  # type: ignore[index]
    )

    # Exercise the public structural-tag builder as vLLM does for named calls.
    supplied = bridge_tool({"type": "object"})
    supplied["function"]["strict"] = True
    tool = ChatCompletionToolsParam.model_validate(supplied)
    choice = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "tool_call"}}
    )
    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[tool],
        tool_choice=choice,
        reasoning=True,
    )
    assert tag is not None
    serialized = tag.model_dump_json()
    assert '"additionalProperties":true' in serialized, serialized
    xgr.Grammar.from_structural_tag(serialized)

    # Use the supported compiler/matcher API rather than a removed testing-only
    # EBNF converter. Check the actual wire representation, independent of rule
    # names: nested objects stay JSON-braced; nested XML and wrong types reject.
    plain_tag = get_model_structural_tag("qwen_3_coder", [tool], choice, False)
    assert plain_tag is not None
    tokenizer = xgr.TokenizerInfo(
        [bytes([index]) for index in range(256)] + [b"<eos>"], stop_token_ids=[256]
    )
    compiled = xgr.GrammarCompiler(tokenizer, max_threads=2).compile_grammar(
        xgr.Grammar.from_structural_tag(plain_tag.model_dump_json())
    )
    for content, expected in (
        ("{}", True),
        (json.dumps({"ref": "e12", "nested": {"count": 1}, "items": [{"id": 3}]}), True),
        ("<parameter=ref>e12</parameter>", False),
        ("[]", False),
        ('{"ref":', False),
    ):
        wire = ("<tool_call>\n<function=tool_call>\n"
                "<parameter=name>package_probe</parameter>\n"
                f"<parameter=arguments>{content}</parameter>\n"
                "</function>\n</tool_call>")
        matcher = xgr.GrammarMatcher(compiled)
        complete = (matcher.accept_string(wire) and matcher.accept_token(256)
                    and matcher.is_terminated())
        assert complete is expected, (content, complete)

    print("qwen open nested-object structural-tag regression check: PASS")


if __name__ == "__main__":
    main()
