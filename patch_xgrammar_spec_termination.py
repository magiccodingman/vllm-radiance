#!/usr/bin/env python3
"""Backport vLLM PR #52805: stop XGrammar batches at termination.

vLLM v0.27.1 advances structured-output grammars with a batch of accepted
speculative tokens.  When a terminating token occurs before the end of that
batch, the remaining tokens are incorrectly passed to an already-terminated
XGrammar matcher.  Besides the noisy "Failed to advance FSM" diagnostics, the
overshoot can desynchronise constrained decoding and truncate tool-call JSON.

This is the source-equivalent v0.27.1 backport of upstream merge
12f64b39d29282437e35be9aa5db432fb2a1a6e6
(https://github.com/vllm-project/vllm/pull/52805).  It is intentionally kept as
an overlay instead of moving the stable vLLM pin.

Idempotent; exact-anchor guarded; ast.parse checked before writing.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/v1/structured_output/backend_xgrammar.py"

ANCHOR = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

NEW = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

RESET_ANCHOR = '''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''

RESET_NEW = '''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False  # radiance: clear cached grammar termination
'''

SENTINEL = "Tokens after termination are ignored. Returns False if the FSM"
RESET_SENTINEL = "radiance: clear cached grammar termination"


def main():
    apply(F, ANCHOR, NEW, SENTINEL, "xgrammar-spec-termination-batches")
    apply(
        F,
        RESET_ANCHOR,
        RESET_NEW,
        RESET_SENTINEL,
        "xgrammar-reset-termination-state",
    )


if __name__ == "__main__":
    main()
