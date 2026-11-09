#!/usr/bin/env python3
"""Installed-image CPU contracts, not live GPU or performance qualification."""
import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import xgrammar as xgr
from openai.types.responses import FunctionTool
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionToolsParam,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


def main():
    for key in ("RADIANCE_QWEN_AUTO_TOOL_GRAMMAR", "RADIANCE_QWEN_INT64_BOUND_COMPAT"):
        os.environ.pop(key, None)
    schema = {"type": "object", "properties": {"request": {
        "type": "object", "properties": {
            "expectedVersion": {"type": ["integer", "null"], "minimum": 0,
                                "maximum": float(2**63)},
            "items": {"type": "array", "items": {"type": "object"}},
        }, "required": ["items"]}}, "required": ["request"]}
    original = copy.deepcopy(schema)

    def tool(strict=None, responses=False):
        if responses:
            return FunctionTool(type="function", name="package_probe", parameters=schema,
                                strict=strict)
        f = {"name": "package_probe", "parameters": schema}
        if strict is not None:
            f["strict"] = strict
        return ChatCompletionToolsParam.model_validate({"type": "function", "function": f})

    assert get_model_structural_tag("qwen_3_coder", [tool()], "auto", False) is None
    assert get_model_structural_tag("qwen_3_coder", [tool(False)], "auto", False) is None
    try:
        get_model_structural_tag("qwen_3_coder", [tool(True)], "auto", False)
        raise AssertionError("rounded maximum silently accepted by default")
    except ValueError as exc:
        assert "Int64 range" in str(exc)
    os.environ["RADIANCE_QWEN_AUTO_TOOL_GRAMMAR"] = "1"
    os.environ["RADIANCE_QWEN_INT64_BOUND_COMPAT"] = "1"
    forced = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "package_probe"}})
    for responses in (False, True):
        for choice in ("auto", "required", forced):
            for reasoning in (False, True):
                value = tool(responses=responses)
                before = value.model_dump()
                tag = get_model_structural_tag("qwen_3_coder", [value], choice, reasoning)
                assert tag is not None
                raw = tag.model_dump_json()
                assert '9223372036854775807' in raw
                xgr.Grammar.from_structural_tag(raw)
                assert value.model_dump() == before and schema == original
        # The compatibility opt-in restores the old omitted-flag default;
        # explicit non-strict tools must still leave argument schemas free.
        value = tool(False, responses=responses)
        before = value.model_dump()
        tag = get_model_structural_tag("qwen_3_coder", [value], "auto", False)
        assert tag is not None
        raw = tag.model_dump_json()
        assert '9223372036854775807' not in raw
        xgr.Grammar.from_structural_tag(raw)
        assert value.model_dump() == before and schema == original
    assert get_model_structural_tag("qwen_3_coder", [tool()], "none", False) is None
    assert get_model_structural_tag("llama", [tool()], "auto", False) is None
    assert get_model_structural_tag("qwen_3_coder", [], "auto", False) is None

    # Real validation selection must never replace a modern-tag compiler failure
    # with a Guidance legacy-schema KeyError. No tokenizer/model is loaded.
    params = SamplingParams(structured_outputs=StructuredOutputsParams(
        structural_tag=json.dumps({"type": "structural_tag", "format": {"type": "invalid"}})))
    try:
        with patch('vllm.v1.structured_output.backend_guidance.validate_guidance_grammar') as fallback:
            params._validate_structured_outputs(SimpleNamespace(is_diffusion=False),
                                                StructuredOutputsConfig(backend="auto"), SimpleNamespace())
        raise AssertionError("invalid modern structural tag accepted")
    except VLLMValidationError as exc:
        assert "Guidance fallback does not support its format" in str(exc), str(exc)
        fallback.assert_not_called()
    for raw in (
        tag.model_dump_json(),
        json.dumps({"structures": [{"begin": "<test>", "schema": {"type": "integer"},
                                    "end": "</test>"}], "triggers": ["<test>"]}),
    ):
        accepted = SamplingParams(structured_outputs=StructuredOutputsParams(structural_tag=raw))
        accepted._validate_structured_outputs(SimpleNamespace(is_diffusion=False),
                                              StructuredOutputsConfig(backend="auto"), SimpleNamespace())
        assert accepted.structured_outputs._backend == "xgrammar"
    print("Installed Qwen grammar compiler contracts: PASS (12 Chat/Responses combinations)")


if __name__ == "__main__":
    main()
