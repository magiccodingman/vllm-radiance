#!/usr/bin/env python3
"""Bake: propagate seq_lens_cpu_upper_bound through CommonAttentionMetadata.unpadded().

vLLM's unpadded() reconstructs the attn metadata but drops seq_lens_cpu_upper_bound
(every other per-request field is copied). Under disable_padded_drafter_batch, when the real
batch != padded cudagraph batch (chunked prefill / mixed), unpadded() runs and the drafter's
prepare_inputs then hits `assert seq_lens_cpu_upper_bound is not None` -> EngineDeadError.
This restores the (already-correct) optimistic bound, sliced with the maybe_slice_reqs lambda
already defined in unpadded(). Zero correctness risk. Idempotent. Enables the MTP
disable_padded_drafter_batch decode win."""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backend.py"
ANCHOR = "            is_prefilling=maybe_slice_reqs(self.is_prefilling),\n"
ADD = "            seq_lens_cpu_upper_bound=maybe_slice_reqs(self.seq_lens_cpu_upper_bound),\n"


def main():
    apply(F, ANCHOR, ANCHOR + ADD, "seq_lens_cpu_upper_bound=maybe_slice_reqs", "unpad fix")


if __name__ == "__main__":
    main()
