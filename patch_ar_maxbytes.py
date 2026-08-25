#!/usr/bin/env python3
"""Restore RADIANCE_AR_MAX_KB / RADIANCE_AR_QUANT_MIN_KB as environment knobs.

Upstream hardcodes the P2P all-reduce size gate at 48 MB, sized to hold one prefill chunk at the
shipped `--max-num-batched-tokens 4096` (4096 x 5120 x bf16 = 40 MiB). Anything larger falls back
to RCCL, silently.

That is exactly the trap this fork hit once before, from the other direction. The gate compares the
raw bf16 byte count, and a chunked-prefill all-reduce is `--max-num-batched-tokens x hidden x 2`.
At 8192 tokens and hidden 5120 that is 80 MiB -- above the cap -- so *every prefill reduction* goes
to RCCL while the P2P kernel only ever sees the small decode messages. Measured on 2x R9700 (TP2,
Qwen3.8-27B) with a torch profile of a fixed 40k prompt: all-reduce was 18.8% of prefill GPU time
on RCCL at 3.145 ms per call; sizing the cap to fit moved all 924 reductions onto the P2P kernel at
1.317 ms each (2.18x), worth +0.9-7.3% prefill on fp8 and +3.1-12.8% on MXFP4. KV cache size did
not change -- the extra 2 x max_bytes of IPC scratch comes out of non-KV budget, not headroom.

The default here is upstream's 49152, so an unconfigured serve behaves exactly as upstream does.
This only makes the number reachable, which matters because it has to track the chunk size:
whenever `--max-num-batched-tokens` changes, `tokens x hidden x 2` has to stay under the cap or the
fast all-reduce disappears without a word in the log. At chunk 16384 the message is 160 MiB and
needs >= 196608.

Verify with a torch profile (CLI-only in current vLLM:
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=...`): `ncclDevKernel`
should be absent, and the R4D all-reduce call count should match `vllm::all_reduce`.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
AR = SP / "radiance_allreduce.py"

ANCHOR = (
    "# Largest message the kernel takes; anything above it falls back to RCCL. Sized to hold one prefill\n"
    "# chunk's all-reduce (4096 tokens x 5120 channels x bf16 = 40 MiB), so a serve running the shipped\n"
    "# --max-num-batched-tokens keeps the kernel for prefill as well as decode. Costs 2x this in IPC\n"
    "# scratch per rank, which is trivial on 32 GB. The kernel is bit-identical to RCCL at every size.\n"
    "_MAX_BYTES = 49152 * 1024\n"
    "# Below this the exact bf16 kernel wins: compression only pays once the transfer is bandwidth-bound.\n"
    "_QUANT_MIN_BYTES = 128 * 1024\n"
)

NEW = (
    "# Largest message the kernel takes; anything above it falls back to RCCL. Sized to hold one prefill\n"
    "# chunk's all-reduce (4096 tokens x 5120 channels x bf16 = 40 MiB), so a serve running the shipped\n"
    "# --max-num-batched-tokens keeps the kernel for prefill as well as decode. Costs 2x this in IPC\n"
    "# scratch per rank, which is trivial on 32 GB. The kernel is bit-identical to RCCL at every size.\n"
    "#\n"
    "# --- radiance (patch_ar_maxbytes.py): this has to track --max-num-batched-tokens ---\n"
    "# The gate compares the raw bf16 byte count, so the message is tokens x hidden x 2: at chunk 8192\n"
    "# and hidden 5120 it is 80 MiB, over the default, and every prefill reduction silently falls back\n"
    "# to RCCL. Measured cost of that fallback here: all-reduce 18.8% of prefill GPU time at 3.145 ms\n"
    "# per call vs 1.317 ms on the kernel (2.18x), worth +3.1-12.8% prefill on the MXFP4 build. The\n"
    "# default below is upstream's, so an unconfigured serve is unchanged.\n"
    "_MAX_BYTES = int(os.environ.get(\"RADIANCE_AR_MAX_KB\") or 49152) * 1024\n"
    "# Below this the exact bf16 kernel wins: compression only pays once the transfer is bandwidth-bound.\n"
    "_QUANT_MIN_BYTES = int(os.environ.get(\"RADIANCE_AR_QUANT_MIN_KB\") or 128) * 1024\n"
)


def main():
    # `os` is already imported at the top of radiance_allreduce.py, above these constants.
    apply(AR, ANCHOR, NEW,
          "this has to track --max-num-batched-tokens",
          "all-reduce: RADIANCE_AR_MAX_KB / _QUANT_MIN_KB knobs")


if __name__ == "__main__":
    main()
