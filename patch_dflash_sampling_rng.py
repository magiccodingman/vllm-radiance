#!/usr/bin/env python3
"""Separate DFlash2 selector proposal noise from target replacement noise.

Adapt the independent draft Philox stream from vLLM PR #54282 (commit
fe755c88995ad468882517b6c4bdd60138d46a3a) to the selector walk's direct call to
gumbel_noised_argmax. Sharing (seed, position) conditions the replacement draw
on the rejected proposal and biases probabilistic rejection sampling.

The selector uses an unclamped sampling-position buffer. Offset only its local
RNG index: model positions, cache addressing, and greedy sampling are unchanged.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


def main():
    path = (Path(sysconfig.get_paths()["purelib"])
            / "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py")
    apply(
        path,
        "        position = tl.load(sample_pos_ptr + flat) - 1\n",
        "        # vLLM #54282: proposal and target replacement need independent noise.\n"
        "        # This local RNG index does not change model/cache positions.\n"
        "        position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)\n",
        "position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)",
        "dflash2: separate selector proposal RNG stream",
    )
    apply(
        path,
        "# Candidate ids key the noise, matching the target's own sampling.",
        "# Candidate ids key the proposal's independent sampling noise.",
        "# Candidate ids key the proposal's independent sampling noise.",
        "dflash2: describe independent proposal noise",
    )
    # The direct helper has no implicit drafting salt in the pinned vLLM. A
    # future upstream sampler API change requires reviewing this adaptation.


if __name__ == "__main__":
    main()
