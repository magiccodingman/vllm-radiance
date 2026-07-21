#!/usr/bin/env python3
"""Bake: let the RADIANCE draft controller short-circuit the MTP draft loop.

vLLM's SpecDecodeBaseProposer.propose runs a fixed `for token_index in
range(num_speculative_tokens - 1)` loop -- one draft-model forward per iteration,
no early exit. The whole point of the confidence-gated controller is to STOP
running forwards once every request has decided to stop (verify) or take an
n-gram: fewer serial draft forwards on low-headroom content. The controller sets
`self._radiance_stop` from its per-slot A/B/C decision; this patch injects the one
line that honours it, so the remaining forwards are never launched (the partial
`draft_token_ids_list` is what gets stacked and returned).

Idempotent; anchor-count-guarded; ast.parse guard before writing. NOOP once applied."""
import ast
import sysconfig
from pathlib import Path
from _patchlib import apply

LIB = Path(sysconfig.get_paths()["purelib"])
F = LIB / "vllm/v1/spec_decode/llm_base_proposer.py"

ANCHOR = (
    "        for token_index in range(self.num_speculative_tokens - 1):\n"
    "            # Update the inputs.\n"
)
NEW = (
    "        for token_index in range(self.num_speculative_tokens - 1):\n"
    "            if getattr(self, \"_radiance_stop\", False):\n"
    "                break  # radiance: controller stopped; skip the remaining draft forwards\n"
    "            # Update the inputs.\n"
)
SENTINEL = "radiance: controller stopped; skip the remaining draft forwards"


def main():
    apply(F, ANCHOR, NEW, SENTINEL, "mtp-draft-loop-break")


if __name__ == "__main__":
    main()
