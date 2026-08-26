#!/usr/bin/env python3
"""Backport vLLM #53046: validate post-reasoning speculative drafts.

Draft tokens generated before the grammar bitmask activates are not guaranteed
to satisfy that grammar.  When a reasoning-end marker lands inside a
speculative window, v0.28.0 feeds the remaining, pre-bitmask drafts directly to
``accept_tokens``. XGrammar correctly rejects them, but the direct call emits
FSM errors and risks leaving the speculative/grammar state out of sync.

Upstream commit c6e19b3be24338759a443e03c8325d76da9ee202 first probes those
drafts with the non-mutating ``validate_tokens`` path and advances only valid
ones. This is its exact source-equivalent backport for the stable v0.28.0 pin;
the fix merged after the release tag.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/v1/structured_output/__init__.py"

ANCHOR = '''                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
'''

NEW = '''                    if advance_grammar and not grammar.is_terminated():
                        if post_reasoning_end_in_window:
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
'''

SENTINEL = "accepted = bool(grammar.validate_tokens([token]))"


def main():
    apply(
        F,
        ANCHOR,
        NEW,
        SENTINEL,
        "xgrammar-validate-post-reasoning-spec-drafts",
    )


if __name__ == "__main__":
    main()
