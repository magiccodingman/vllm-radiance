#!/usr/bin/env python3
"""Bake: propagate seq_lens_cpu_upper_bound through CommonAttentionMetadata.unpadded().

vLLM's unpadded() reconstructs the attn metadata but drops seq_lens_cpu_upper_bound
(every other per-request field is copied). Under disable_padded_drafter_batch, when the real
batch != padded cudagraph batch (chunked prefill / mixed), unpadded() runs and the drafter's
prepare_inputs then hits `assert seq_lens_cpu_upper_bound is not None` -> EngineDeadError.
This restores the (already-correct) optimistic bound, sliced with the maybe_slice_reqs lambda
already defined in unpadded(). Zero correctness risk. Idempotent. Enables the MTP
disable_padded_drafter_batch decode win."""
import ast
import sysconfig
from pathlib import Path

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backend.py"
ANCHOR = "            is_prefilling=maybe_slice_reqs(self.is_prefilling),\n"
ADD = "            seq_lens_cpu_upper_bound=maybe_slice_reqs(self.seq_lens_cpu_upper_bound),\n"


def apply(path, anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new` in `path`. Skips if
    `sentinel` is already present; a missing file or non-unique anchor is fatal."""
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({path})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    path.write_text(s)
    print(f"  OK    {label}")


def main():
    apply(F, ANCHOR, ANCHOR + ADD, "seq_lens_cpu_upper_bound=maybe_slice_reqs", "unpad fix")


if __name__ == "__main__":
    main()
