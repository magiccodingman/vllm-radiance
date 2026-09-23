#!/usr/bin/env python3
"""Focused, GPU-free regression check for the XGrammar spec backports."""

from types import SimpleNamespace

import torch

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
    # Exercise the installed manager, not the spelling of an obsolete backport.
    manager = object.__new__(StructuredOutputManager)
    manager.vllm_config = SimpleNamespace(
        num_speculative_tokens=4, model_config=SimpleNamespace(is_diffusion=False)
    )
    manager._grammar_bitmask = torch.zeros((8, 4), dtype=torch.int32)
    manager.fill_bitmask_parallel_threshold = 16
    manager._get_constraint_start = lambda request, tokens: 0
    checked = grammar()
    request = SimpleNamespace(
        use_structured_output=True,
        structured_output_request=SimpleNamespace(grammar=checked),
    )
    assert manager.validate_tokens(request, [1, 99, 2, -1]) == [1, 99]
    rows = []
    manager._fill_bitmasks = lambda batch: rows.extend(
        (index, apply, tuple(g.matcher.tokens)) for g, index, apply in batch
    )
    manager.grammar_bitmask({"test": request}, ["test"], {"test": [1, 99, -1, -1]})
    assert len(rows) == 5
    assert rows[:2] == [(0, True, ()), (1, True, (1,))]
    assert rows[-1][1] is False  # No bonus grammar constraint after padding.
    assert checked.matcher.tokens == []  # Speculative advancement rolled back.
    assert not checked.is_terminated()

    manager._get_constraint_start = lambda request, tokens: 1
    assert manager.validate_tokens(request, [7, 1, 99, 2]) == [7, 1, 99]

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
