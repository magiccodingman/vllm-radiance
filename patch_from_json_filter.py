#!/usr/bin/env python3
"""Bake: register a `from_json` Jinja filter in transformers' chat-template
environment (transformers/utils/chat_template_utils.py, _cached_compile_jinja_template).

Why: the community "fixed" Qwen3.6 chat template (qwen3.6-enhanced.jinja, passed
via --chat-template) parses a tool_call.arguments JSON *string* back into a
mapping (`arguments | from_json`) so an assistant tool-call turn replayed in
multi-turn history is re-rendered as <parameter=..> blocks in the exact XML form
the model was trained on. That path is what fixes the agentic "|items" crash: the
STOCK Qwen3.6 template does `tool_call.arguments | items` directly, which raises
`TypeError: Can only get item pairs from a mapping` the moment vLLM replays a
tool call (arguments come back as a string over the OpenAI API), stalling any
multi-turn tool loop.

transformers' jinja env only registers `tojson`, `raise_exception`,
`strftime_now`; no `from_json`, and the ImmutableSandboxedEnvironment
forbids calling json.loads from the template. So the enhanced template raises
`No filter named 'from_json' found` in this build. This patch adds the one missing
filter (json.loads on a string, NOOP otherwise). It is a superset-safe addition:
no existing template references `from_json`, so nothing
else changes.

Injection point (transformers 5.12.x): right where the other filters/globals are
attached, just before `return jinja_env.from_string(chat_template)`.

Idempotent; anchor-count-guarded; ast.parse guard before writing. NOOP once applied."""
import ast
import sysconfig
from pathlib import Path
from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])

F = LIB / "transformers/utils/chat_template_utils.py"

ANCHOR = (
    '    jinja_env.filters["tojson"] = tojson\n'
    '    jinja_env.globals["raise_exception"] = raise_exception\n'
    '    jinja_env.globals["strftime_now"] = strftime_now\n'
    "    return jinja_env.from_string(chat_template)\n"
)
NEW = (
    "    def from_json(value):\n"
    "        # radiance: expose a JSON parser to chat templates. Community 'fixed'\n"
    "        # tool templates (e.g. Qwen3.6-enhanced) parse a tool_call.arguments\n"
    "        # JSON *string* back into a mapping to re-render <parameter=..> blocks\n"
    "        # in multi-turn history; the sandbox forbids json.loads directly. NOOP\n"
    "        # on already-parsed (non-string) values.\n"
    "        return json.loads(value) if isinstance(value, str) else value\n"
    "\n"
    '    jinja_env.filters["tojson"] = tojson\n'
    '    jinja_env.filters["from_json"] = from_json\n'
    '    jinja_env.globals["raise_exception"] = raise_exception\n'
    '    jinja_env.globals["strftime_now"] = strftime_now\n'
    "    return jinja_env.from_string(chat_template)\n"
)
SENTINEL = 'jinja_env.filters["from_json"] = from_json'


def main():
    apply(F, ANCHOR, NEW, SENTINEL, "jinja-from_json-filter")


if __name__ == "__main__":
    main()
