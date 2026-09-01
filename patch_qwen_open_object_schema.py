#!/usr/bin/env python3
"""Preserve open nested JSON objects in Qwen XML structural tags.

JSON Schema treats an omitted ``additionalProperties`` keyword as ``true``.
XGrammar 0.2.3's Qwen-XML compiler does not preserve that equivalence for a
nested ``{"type": "object"}``: omission selects nested ``<parameter>`` tags,
while explicit ``true`` selects a JSON-braced object.  vLLM's flat Qwen
argument converter cannot reconstruct nested parameter tags and moves their
fields beside the parent while returning the parent as an empty string.

Normalize only semantically open *nested* objects with no declared properties
or composition.  The tool's root parameter object and all declared/closed
schemas are untouched.  This makes the generated representation round-trip
through the existing parser without changing the accepted JSON Schema.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/tool_parsers/structural_tag_registry.py"

HELPER_ANCHOR = '''_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


'''

HELPER_NEW = '''_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


_QWEN_XML_STRUCTURAL_TAG_MODELS = frozenset(
    {"qwen_3", "qwen_3_5", "qwen_3_coder"}
)


def _normalize_qwen_open_nested_objects(
    schema: object,
    *,
    is_root: bool,
) -> None:
    """Make open nested objects use XGrammar's JSON-braced representation.

    Omitted and explicit-true ``additionalProperties`` are equivalent under
    JSON Schema.  The explicit form prevents XGrammar 0.2.3's Qwen-XML path
    from rendering an open nested object as recursively nested parameter tags,
    which vLLM's flat Qwen argument converter cannot round-trip.
    """
    if isinstance(schema, list):
        for item in schema:
            _normalize_qwen_open_nested_objects(item, is_root=False)
        return
    if not isinstance(schema, dict):
        return

    is_plain_open_object = (
        not is_root
        and schema.get("type") == "object"
        and "additionalProperties" not in schema
        and not schema.get("properties")
        and not schema.get("patternProperties")
        and not any(key in schema for key in ("$ref", "allOf", "anyOf", "oneOf"))
    )
    if is_plain_open_object:
        # JSON Schema's default is true; spelling it out only selects the
        # representation that the Qwen parser can faithfully reconstruct.
        schema["additionalProperties"] = True

    for value in tuple(schema.values()):
        _normalize_qwen_open_nested_objects(value, is_root=False)


def _normalize_qwen_tool_schemas(tools: list[dict[str, object]]) -> None:
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            _normalize_qwen_open_nested_objects(parameters, is_root=True)


'''

CALL_ANCHOR = '''    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

'''

CALL_NEW = '''    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    if model in _QWEN_XML_STRUCTURAL_TAG_MODELS:
        _normalize_qwen_tool_schemas(dumped_tools)
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

'''

HELPER_SENTINEL = "def _normalize_qwen_open_nested_objects("
CALL_SENTINEL = "_normalize_qwen_tool_schemas(dumped_tools)"


def main() -> None:
    apply(
        F,
        HELPER_ANCHOR,
        HELPER_NEW,
        HELPER_SENTINEL,
        "qwen-open-object-schema-helper",
    )
    apply(
        F,
        CALL_ANCHOR,
        CALL_NEW,
        CALL_SENTINEL,
        "qwen-open-object-schema-normalization",
    )


if __name__ == "__main__":
    main()
