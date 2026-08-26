#!/usr/bin/env python3
"""Backport vLLM #52830 for shared reasoning/tool parser engines.

vLLM v0.28.0 collapses a reasoning parser and tool parser backed by the same
``ParserEngine`` into the raw engine class.  For the qualified
``qwen3``/``qwen3_coder`` pair this discards the registered tool adapter that
carries ``structural_tag_model = "qwen_3_coder"``.  Consequently
``tool_choice="required"`` is unconstrained and the model can omit required
arguments even though strict tool calling is enabled.

Upstream merge 46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3 (PR #52830)
removed this shared-engine shortcut so the existing ``DelegatingParser`` keeps
both registered adapters and installs the XGrammar structural tag.  This is the
smallest source-equivalent behavioral backport for the pinned v0.28.0 wheel.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/parser/parser_manager.py"

ANCHOR = '''        reasoning_engine_cls = cls._get_parser_engine_cls(reasoning_parser_cls)
        tool_engine_cls = cls._get_parser_engine_cls(tool_parser_cls)
        if reasoning_engine_cls is not None and reasoning_engine_cls is tool_engine_cls:
            return reasoning_engine_cls

'''

NEW = '''        # vLLM #52830: preserve the registered reasoning and tool adapters even
        # when both use the same parser engine. The DelegatingParser path below
        # attaches the structural tag required for strict/required tool calls.

'''

SENTINEL = "vLLM #52830: preserve the registered reasoning and tool adapters"


def main() -> None:
    apply(
        F,
        ANCHOR,
        NEW,
        SENTINEL,
        "parser: preserve adapters for shared reasoning/tool engine",
    )


if __name__ == "__main__":
    main()
