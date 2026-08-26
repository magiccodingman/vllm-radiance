#!/usr/bin/env python3
"""Backport vLLM #53017: use the real DFlash logits-cache column stride.

The v0.28.0 DFlash sampler assumes each cache column is exactly the sampled
vocabulary width. Draft checkpoints may reserve an input-only mask/noise row,
making the cache wider than the output head. Every column after the first then
lands at the wrong address. Upstream commit
``d4f4d3f40fc5350a71777fcb0e5eb8a57bda631f`` passes both tensor strides and
rejects a cache that is too narrow.

Idempotent; exact-anchor guarded; ``ast.parse`` checked before writing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply


LIB = Path(sysconfig.get_paths()["purelib"])
GUMBEL = LIB / "vllm/v1/worker/gpu/sample/gumbel.py"
REJECTION = LIB / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"


def main() -> None:
    apply(
        GUMBEL,
        """def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
""",
        """def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
""",
        "def gumbel_block_argmax(\n    logits,\n    block,\n    mask,\n    token_idx,\n    expanded_idx_mapping_ptr,\n    temp_ptr,\n    seeds_ptr,\n    pos_ptr,\n    # [max_num_reqs, num_cols, vocab_size]",
        "dflash: declare block-argmax cache strides",
    )
    apply(
        GUMBEL,
        """def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
""",
        """def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
""",
        "def _gumbel_sample_kernel(\n    local_argmax_ptr,\n    local_argmax_stride,\n    local_max_ptr,\n    local_max_stride,\n    # [max_num_reqs, num_cols, vocab_size]",
        "dflash: declare sampler-kernel cache strides",
    )
    apply(
        GUMBEL,
        """            + req_state_idx * logits_cache_stride
            + col * vocab_size
""",
        """            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
""",
        "+ col * logits_cache_stride_1",
        "dflash: address cache with its real column stride",
    )
    apply(
        GUMBEL,
        """        logits_cache_ptr,
        logits_cache_stride,
        logits_cache_col_ptr,
""",
        """        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
""",
        "        logits_cache_stride_1,\n        logits_cache_col_ptr,",
        "dflash: forward both cache strides",
    )
    apply(
        GUMBEL,
        """    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
""",
        """    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
    BLOCK_SIZE = 1024
""",
        "draft logits cache vocab dim",
        "dflash: reject a narrow logits cache",
    )
    apply(
        GUMBEL,
        """        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache_col,
""",
        """        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
""",
        "logits_cache.stride(1) if logits_cache is not None else 0",
        "dflash: launch sampler with both cache strides",
    )
    apply(
        REJECTION,
        """        None,  # logits_cache_ptr
        0,  # logits_cache_stride
        None,  # logits_cache_col_ptr
""",
        """        None,  # logits_cache_ptr
        0,  # logits_cache_stride_0
        0,  # logits_cache_stride_1
        None,  # logits_cache_col_ptr
""",
        "0,  # logits_cache_stride_1",
        "dflash: update rejection sampler cache-stride ABI",
    )


if __name__ == "__main__":
    main()
