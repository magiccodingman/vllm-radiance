"""Separate raw generation differences from output-parser presentation differences."""

from __future__ import annotations

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal, write_private


def _ids(value):
    return (
        bool(value)
        and isinstance(value, list)
        and all(type(token) is int and token >= 0 for token in value)
    )


def _first_difference(a, b):
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def compare_protocol_pair(streamed, nonstreamed, evidence, *, kind="stream_nonstream"):
    """Preserve a bounded diagnostic, then enforce exact observed equivalence.

    Whitespace normalization is diagnostic only. A matching display is not enough
    without complete raw IDs; a mismatch cannot be assigned to the parser unless
    the prompt and generated IDs match. Tool IDs are transport-assigned identities
    and may differ between independent requests; tool names/arguments may not.
    """
    prompt = [result.get("prompt_token_ids") for result in (streamed, nonstreamed)]
    output = [result.get("token_ids") for result in (streamed, nonstreamed)]
    prompt_observed = all(_ids(ids) for ids in prompt)
    output_observed = all(_ids(ids) for ids in output)
    prompt_declared = [
        (result.get("usage") or {}).get("prompt_tokens") for result in (streamed, nonstreamed)
    ]
    prompt_complete = prompt_observed and all(
        type(count) is int and count > 0 and len(ids) == count
        for ids, count in zip(prompt, prompt_declared, strict=True)
    )
    declared = [
        (result.get("usage") or {}).get("completion_tokens") for result in (streamed, nonstreamed)
    ]
    counts_observed = all(type(count) is int and count > 0 for count in declared)
    output_complete = (
        output_observed
        and counts_observed
        and all(len(ids) == count for ids, count in zip(output, declared, strict=True))
    )
    prompt_equal = prompt_complete and prompt[0] == prompt[1]
    output_equal = output_complete and output[0] == output[1]
    content_equal = streamed["content"] == nonstreamed["content"]
    reasoning_equal = streamed["reasoning"] == nonstreamed["reasoning"]
    finish_equal = streamed["finish_reason"] == nonstreamed["finish_reason"]

    def tool_values(result):
        return [
            {key: tool[key] for key in ("name", "arguments", "parsed_arguments")}
            for tool in result["tools"]
        ]

    tools_equal = tool_values(streamed) == tool_values(nonstreamed)
    if not prompt_complete or not output_complete:
        classification = "MISSING_RAW_TOKEN_OBSERVATION"
    elif not prompt_equal:
        classification = "PROMPT_TOKEN_DIFFERENCE"
    elif not output_equal:
        classification = "GENERATED_TOKEN_DIFFERENCE"
    elif not (content_equal and reasoning_equal and finish_equal and tools_equal):
        classification = "PARSED_OUTPUT_DIFFERENCE"
    else:
        classification = "EXACT_OBSERVED_MATCH"

    report = seal(
        {
            "schema": "urn:qwen:protocol-pair-comparison:v1",
            "comparison_kind": kind,
            "classification": classification,
            "proof": "UNPROVED",
            "scope": "This observed request pair; no universal parser or backend guarantee.",
            "result_sha256": [digest(result) for result in (streamed, nonstreamed)],
            "prompt_observed": prompt_observed,
            "prompt_complete": prompt_complete,
            "prompt_equal": prompt_equal,
            "prompt_counts": [len(ids) if isinstance(ids, list) else None for ids in prompt],
            "output_complete": output_complete,
            "output_equal": output_equal,
            "output_counts": [len(ids) if isinstance(ids, list) else None for ids in output],
            "declared_completion_counts": declared,
            "declared_prompt_counts": prompt_declared,
            "first_prompt_difference": _first_difference(*prompt) if prompt_observed else None,
            "first_output_difference": _first_difference(*output) if output_complete else None,
            "content_equal": content_equal,
            "reasoning_equal": reasoning_equal,
            "finish_equal": finish_equal,
            "tools_equal": tools_equal,
            "content_lengths": [len(result["content"]) for result in (streamed, nonstreamed)],
            "reasoning_lengths": [len(result["reasoning"]) for result in (streamed, nonstreamed)],
            "content_equal_after_strip": streamed["content"].strip()
            == nonstreamed["content"].strip(),
            "whitespace_accepted_as_equal": False,
        }
    )
    write_private(evidence, report)
    if classification != "EXACT_OBSERVED_MATCH":
        raise DiagnosticError(f"response comparison ({kind}): {classification}")
    return report
