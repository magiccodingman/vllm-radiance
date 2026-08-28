#!/usr/bin/env python3
"""Give radiance_verifyhead a place to set its per-step gate.

The int2 target head can only run when the batch's sampling parameters keep the support inside the
reranked set, and `LogitsProcessor._apply_head` sees neither the sampling params nor the batch. The
V2 runner's `sample()` sees both, one line above the call, so the gate is evaluated there and left
on the LogitsProcessor for _apply_head to read.

`sample()` runs in eager Python -- the model forward is already done and `logits_indices` has
selected the sampled rows -- so a Python branch here is not a torch.compile graph break and not
inside a captured graph. That is the whole reason this is the right anchor and
`compute_logits` itself is not.

Inert unless RADIANCE_VERIFY_HEAD=1; the module returns immediately. Idempotent.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = (Path(sysconfig.get_paths()["purelib"])
     / "vllm/v1/worker/gpu/model_runner.py")

OLD = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
"""

NEW = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        # --- RADIANCE int2 verify head (patch_verify_head.py) ---
        try:
            import radiance_verifyhead as _radiance_vh
            _radiance_vh.before_compute_logits(self, input_batch, grammar_output)
        except Exception:
            pass
        logits = self.model.compute_logits(sample_hidden_states)
"""


def main():
    apply(F, OLD, NEW, "RADIANCE int2 verify head",
          "verify head: evaluate the sampling-param gate before compute_logits")


if __name__ == "__main__":
    main()
