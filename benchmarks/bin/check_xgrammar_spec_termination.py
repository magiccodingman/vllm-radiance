#!/usr/bin/env python3
"""Focused, GPU-free regression check for the XGrammar spec backports."""

import inspect

from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class Matcher:
    """Small matcher double that makes token 99 terminate the grammar."""

    def __init__(self):
        self.tokens: list[int] = []
        self.terminated = False

    def accept_token(self, token: int) -> bool:
        if self.terminated:
            return False
        self.tokens.append(token)
        self.terminated = token == 99
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, count: int) -> None:
        del self.tokens[-count:]
        self.terminated = bool(self.tokens and self.tokens[-1] == 99)

    def reset(self) -> None:
        self.tokens.clear()
        self.terminated = False


def grammar() -> XgrammarGrammar:
    return XgrammarGrammar(
        vocab_size=128,
        matcher=Matcher(),  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
    )


def main() -> None:
    manager_source = inspect.getsource(StructuredOutputManager.grammar_bitmask)
    assert "accepted = bool(grammar.validate_tokens([token]))" in manager_source

    accepted = grammar()
    assert accepted.accept_tokens("test", [1, 99, 2])
    assert accepted.matcher.tokens == [1, 99]
    assert accepted.num_processed_tokens == 2
    assert accepted.is_terminated()
    assert accepted.accept_tokens("test", [2])
    assert accepted.matcher.tokens == [1, 99]

    accepted.reset()
    assert not accepted.is_terminated()
    assert accepted.num_processed_tokens == 0
    assert accepted.matcher.tokens == []

    validated = grammar()
    assert validated.validate_tokens([1, 99, 2]) == [1, 99]
    assert validated.matcher.tokens == []
    assert not validated.matcher.is_terminated()

    assert validated.accept_tokens("test", [99])
    assert validated.validate_tokens([2]) == []
    print("xgrammar speculative termination regression check: PASS")


if __name__ == "__main__":
    main()
