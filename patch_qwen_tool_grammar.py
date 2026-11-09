#!/usr/bin/env python3
"""Guarded Qwen grammar interoperability and diagnostic overlay.

Both compatibility switches default off. No caller schema is mutated and no
constraint is silently removed. See docs/QWEN_TOOL_GRAMMAR_COMPATIBILITY.md.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])
REGISTRY = LIB / "vllm/tool_parsers/structural_tag_registry.py"
SAMPLING = LIB / "vllm/sampling_params.py"

HELPERS = '''def _radiance_qwen_auto_grammar(model: str) -> bool:
    import os
    return (
        model in _QWEN_XML_STRUCTURAL_TAG_MODELS
        and os.environ.get("RADIANCE_QWEN_AUTO_TOOL_GRAMMAR", "0") == "1"
    )


def _radiance_qwen_integer_bounds(schema: object) -> None:
    """Normalize schema nodes, never examples/defaults or instance data.

    The opt-in Int64 compatibility mode narrows ONLY a float maximum equal to
    2**63 to INT64_MAX. This repairs rounded Int64 schema exports; it is not
    an exact transformation for a caller deliberately requesting 2**63.
    """
    import os
    if not isinstance(schema, dict):
        return
    types = schema.get("type")
    integer = types == "integer" or (
        isinstance(types, list) and "integer" in types and "number" not in types
    )
    if integer:
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            value = schema.get(key)
            if not isinstance(value, float) or not value.is_integer():
                continue
            exact = int(value)
            if -(2**63) <= exact < 2**63:
                schema[key] = exact
            elif (
                key == "maximum" and exact == 2**63
                and os.environ.get("RADIANCE_QWEN_INT64_BOUND_COMPAT", "0") == "1"
            ):
                schema[key] = 2**63 - 1
            else:
                raise ValueError(
                    "Qwen tool schema has an integer bound outside XGrammar's "
                    "Int64 range. Export exact integer bounds instead of rounded "
                    "floating-point limits. For rounded Int64 maximum exports "
                    "only, an operator may explicitly enable "
                    "RADIANCE_QWEN_INT64_BOUND_COMPAT=1 (narrows to INT64_MAX)."
                )
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        mapping = schema.get(key)
        if isinstance(mapping, dict):
            for child in mapping.values():
                _radiance_qwen_integer_bounds(child)
    for key in ("items", "additionalProperties", "unevaluatedProperties", "unevaluatedItems",
                "contains", "propertyNames", "not", "if", "then", "else"):
        child = schema.get(key)
        if isinstance(child, list):
            for item in child:
                _radiance_qwen_integer_bounds(item)
        else:
            _radiance_qwen_integer_bounds(child)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                _radiance_qwen_integer_bounds(child)


'''


def main() -> None:
    apply(REGISTRY, "def _any_tool_strict(\n", HELPERS + "def _any_tool_strict(\n",
          "def _radiance_qwen_auto_grammar(", "qwen-grammar-helpers")
    apply(REGISTRY,
          '    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]\n',
          '    # radiance: Responses tool parameters may otherwise alias caller schemas.\n'
          '    from copy import deepcopy\n'
          '    dumped_tools = [deepcopy(_dump_tool_for_xgrammar(tool)) for tool in tools]\n',
          '# radiance: Responses tool parameters may otherwise alias caller schemas.',
          "qwen-tool-schema-copy")
    resolve = '    tools = resolve_tool_strictness(tools, tool_choice, strict_level)\n'
    previous = (
        '    # Preserve the legacy opt-in using v0.33 native strictness semantics.\n'
        '    if _radiance_qwen_auto_grammar(model):\n'
        '        strict_level = max(strict_level, ToolStrictLevel.FUNCTION)\n'
        + resolve
    )
    # Exact upgrade of the initial v0.33 development overlay, or a clean wheel.
    anchor = previous if REGISTRY.is_file() and previous in REGISTRY.read_text() else resolve
    apply(REGISTRY, anchor,
          '    # Preserve the legacy opt-in using v0.33 native strictness semantics.\n'
          '    if _radiance_qwen_auto_grammar(model):\n'
          '        strict_level = max(strict_level, ToolStrictLevel.FUNCTION)\n'
          '        # Legacy XGrammar treated an omitted strict flag as schema-strict.\n'
          '        # Preserve that opt-in contract while respecting explicit false.\n'
          '        tools = [\n'
          '            _with_tool_strict(tool, True)\n'
          '            if ((isinstance(tool, FunctionTool) and tool.strict is None)\n'
          '                or (isinstance(tool, ChatCompletionToolsParam)\n'
          '                    and tool.function.strict is None))\n'
          '            else tool for tool in tools\n'
          '        ]\n'
          '    tools = resolve_tool_strictness(tools, tool_choice, strict_level)\n',
          '# Legacy XGrammar treated an omitted strict flag as schema-strict.',
          "qwen-auto-grammar-opt-in")
    apply(REGISTRY,
          '        _normalize_qwen_tool_schemas(dumped_tools)\n',
          '        _normalize_qwen_tool_schemas(dumped_tools)\n'
          '        for tool in dumped_tools:\n'
          '            _radiance_qwen_integer_bounds(tool.get("function", {}).get("parameters"))\n',
          '_radiance_qwen_integer_bounds(tool.get(', "qwen-integer-bounds")
    apply(SAMPLING,
          '            except VLLMValidationError:\n'
          '                # The request either failed validation\n',
          '            except VLLMValidationError as xgrammar_error:\n'
          '                # radiance: Guidance accepts legacy structural tags only.\n'
          '                if self.structured_outputs.structural_tag:\n'
          '                    tag = json_mod.loads(self.structured_outputs.structural_tag)\n'
          '                    if "structures" not in tag:\n'
          '                        raise VLLMValidationError(\n'
          '                            "XGrammar rejected this modern structural tag; "\n'
          '                            "Guidance fallback does not support its format. "\n'
          '                            "Check tool schema constraints and compiler compatibility."\n'
          '                        ) from xgrammar_error\n'
          '                # The request either failed validation\n',
          '# radiance: Guidance accepts legacy structural tags only.',
          "modern-structural-tag-fallback")


if __name__ == "__main__":
    main()
