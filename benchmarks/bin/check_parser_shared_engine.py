#!/usr/bin/env python3
"""GPU-free regression check for vLLM #52830's shared-engine parser fix."""

import argparse

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import DelegatingParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"ref": {"type": "string"}},
                    "required": ["ref"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    parser_cls = ParserManager.get_parser(
        tool_parser_name="qwen3_coder",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    assert issubclass(parser_cls, DelegatingParser), parser_cls
    assert parser_cls.tool_parser_cls.__name__ == "Qwen3EngineToolParser"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Click e12."}],
        tools=tools,
        tool_choice="required",
    )
    adjusted = parser_cls(tokenizer, request.tools).adjust_request(request)
    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.structural_tag is not None
    print("shared parser-engine structural-tag regression check: PASS")


if __name__ == "__main__":
    main()
