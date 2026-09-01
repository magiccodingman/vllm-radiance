#!/usr/bin/env python3
"""GPU-free regression check for open nested Qwen tool arguments."""

from __future__ import annotations

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.structural_tag_registry import (
    _normalize_qwen_tool_schemas,
    get_model_structural_tag,
)
from xgrammar.testing import _qwen_xml_tool_calling_to_ebnf


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

    # XGrammar 0.2.3 must now select a JSON-braced nested object rather than
    # recursively nested Qwen <parameter> tags that the flat parser corrupts.
    ebnf = _qwen_xml_tool_calling_to_ebnf(json.dumps(parameters), False)
    arguments_rule = next(
        line for line in ebnf.splitlines() if line.startswith("root_part_0 ::=")
    )
    assert "root_prop_" in arguments_rule, arguments_rule
    assert "xml_object" not in arguments_rule, arguments_rule

    # Exercise the public structural-tag builder as vLLM does for named calls.
    tool = ChatCompletionToolsParam.model_validate(bridge_tool({"type": "object"}))
    tag = get_model_structural_tag(
        model="qwen_3_coder",
        tools=[tool],
        tool_choice=ChatCompletionNamedToolChoiceParam.model_validate(
            {"type": "function", "function": {"name": "tool_call"}}
        ),
        reasoning=True,
    )
    assert tag is not None
    serialized = tag.model_dump_json()
    assert '"additionalProperties":true' in serialized, serialized

    print("qwen open nested-object structural-tag regression check: PASS")


if __name__ == "__main__":
    main()
