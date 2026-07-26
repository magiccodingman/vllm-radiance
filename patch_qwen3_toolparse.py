#!/usr/bin/env python3
"""Bake: fix streaming vs non-streaming tool-parser divergence on a truncated
tool call (vLLM issue #47137), for the engine-based parsers (qwen3_coder /
qwen3_xml + reasoning=qwen3, gemma4, ...).

Absent both halves of the upstream fix (the content-leak half from PR #46875, and
the arguments half that was still un-merged upstream), two related bugs reproduce:

  input '<tool_call>\\n<function'                     (opener truncated by max_tokens/stop)
    non-stream content = '<tool_call>\\n<function'  |  streaming content = None      -> LEAK
  input '...<parameter=city>\\nSan Fr'                (truncated mid-parameter value)
    non-stream args    = '{}'                       |  streaming args = '{"city": "San Fr'  -> DIVERGE

Both are made to match STREAMING (the sane direction: never surface raw tool
markup as assistant content; return the same partial args a streaming client
already accumulated). Two surgical edits, engine-parsers only:

1. content leak, vllm/parser/abstract_parser.py, DelegatingParser._extract_tool_calls
   When no tool call is promoted, engine parsers' ExtractedToolCallInformation.content
   already has the incomplete markup stripped; return THAT instead of the raw input.
   (Legacy/non-engine parsers keep the raw `content` path unchanged: zero risk.)

2. args divergence, vllm/parser/engine/parser_engine.py, _build_extracted_result
   A call truncated mid-parameter drops the unterminated value under the
   partial=False arg converter ('{}'), but the streaming path already streamed the
   partial value into slot.streamed_json ('{"city": "San Fr'). When partial=True
   parses further than partial=False (i.e. the body is truncated), emit the exact
   streamed value so both paths agree byte-for-byte. _fix_arg_types no-ops on the
   (intentionally unbalanced) partial JSON, so no double-fix guard is needed.

Idempotent; anchor-count-guarded; ast.parse guard before writing. NOOP once applied."""
import ast
import sysconfig
from pathlib import Path
from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])

# ── 1. content-leak fix (abstract_parser.py) ─────────────────────────────────
F_ABS = LIB / "vllm/parser/abstract_parser.py"
# 0.26.0 expanded the `else` block: a required/named tool-choice empty-content case now sits between
# the `# No tool calls.` comment and the `return None, content`, so the old contiguous anchor no
# longer matches. The anchor targets the tail of that block instead. The leak path (returning raw
# `content`) is unchanged in 0.26.0, so this fix is still needed.
ABS_ANCHOR = (
    "                if (is_required_tool_choice or is_named_tool_choice) and (\n"
    "                    content is None\n"
    "                    or (isinstance(content, str) and not content.strip())\n"
    "                ):\n"
    "                    return [], None\n"
    "                return None, content\n"
)
ABS_NEW = (
    "                if (is_required_tool_choice or is_named_tool_choice) and (\n"
    "                    content is None\n"
    "                    or (isinstance(content, str) and not content.strip())\n"
    "                ):\n"
    "                    return [], None\n"
    "                # Engine-based parsers already strip incomplete / un-promoted\n"
    "                # tool-call markup from content, so return that (drops it)\n"
    "                # rather than the raw input. Keeps non-streaming in agreement\n"
    "                # with streaming on a truncated <tool_call> opener (vLLM #47137).\n"
    "                if self._engine_based and tool_call_info is not None:\n"
    "                    return None, tool_call_info.content\n"
    "                return None, content\n"
)
ABS_SENTINEL = "if self._engine_based and tool_call_info is not None:\n                    return None, tool_call_info.content"

# ── 2. args-divergence fix (parser_engine.py) ────────────────────────────────
F_ENG = LIB / "vllm/parser/engine/parser_engine.py"
ENG_ANCHOR = (
    "                    try:\n"
    "                        args_json = converter(raw_body, False)\n"
    "                    except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                        logger.debug(\n"
    '                            "arg converter failed (extract): %s", raw_body[:80]\n'
    "                        )\n"
    "                        args_json = self._extract_args_json(raw_body, name)\n"
)
ENG_NEW = (
    "                    try:\n"
    "                        args_json = converter(raw_body, False)\n"
    "                    except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                        logger.debug(\n"
    '                            "arg converter failed (extract): %s", raw_body[:80]\n'
    "                        )\n"
    "                        args_json = self._extract_args_json(raw_body, name)\n"
    "                    else:\n"
    "                        # vLLM #47137: a call truncated mid-parameter drops\n"
    "                        # the unterminated value under partial=False, but a\n"
    "                        # streaming client already received it. When\n"
    "                        # partial=True parses further, the body is truncated;\n"
    "                        # emit the exact streamed args so non-streaming agrees\n"
    "                        # with streaming.\n"
    "                        try:\n"
    "                            partial_json = converter(raw_body, True)\n"
    "                        except (json.JSONDecodeError, ValueError, TypeError):\n"
    "                            partial_json = args_json\n"
    "                        if partial_json != args_json and slot.streamed_json:\n"
    "                            args_json = slot.streamed_json\n"
)
ENG_SENTINEL = "if partial_json != args_json and slot.streamed_json:\n                            args_json = slot.streamed_json"


def main():
    apply(F_ABS, ABS_ANCHOR, ABS_NEW, ABS_SENTINEL, "tool-content-leak")
    apply(F_ENG, ENG_ANCHOR, ENG_NEW, ENG_SENTINEL, "tool-args-truncation")


if __name__ == "__main__":
    main()
