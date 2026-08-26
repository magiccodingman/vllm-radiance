"""Correctness fixes to the DFlash speculator base class that DFlash2 depends on.

PR #52816 (DFlash2) is written against a `dflash/speculator.py` that is materially newer than the
one in 0.27.1, and it does not carry that file's changes -- they landed separately. Backporting
only #52816 gives a DFlash2 that starts, drafts, and reports a healthy-looking acceptance curve
while producing garbled text: locally plausible tokens in the wrong places, worse the longer the
generation runs. Verified on Qwen3.8-27B, where MTP under the same V2 runner is clean.

Three fixes, all from main, none of them optional:

1. `sample_idx_mapping` is initialised and reset to -1 rather than 0. -1 is the "inert row"
   sentinel the prepare kernel already writes for padding and the consumers already test for it.
   Zero makes every untouched graph-padding row scatter into request slot 0, which is exactly the
   slot a single-stream request occupies.

2. Context loads are masked with `is_valid_ctx` (the context minus the rejected suffix) rather
   than `is_ctx` (the full allocated span). Without that distinction, rejected tokens from the
   previous step are loaded as valid context and receive draft KV, so the target can read rejected
   proposals as accepted history. This is the one that degrades text.

3. Neither the context nor the query slot checks for the null block. Block 0 is the null block, and
   a sliding-window block table can legitimately contain it after eviction; writing draft KV there
   corrupts an unrelated request.

Context parallelism is deliberately NOT backported. main's version of this file threads cp_rank /
cp_size / cp_interleave through the kernel and calls cp_local_slot(), but 0.27.1's BlockTables has
no cp attributes at all, so that path cannot be expressed here. At cp_size 1 -- the only thing this
box runs -- cp_local_slot() reduces exactly to `block_id * block_size + pos % block_size`, which is
what the unpatched arithmetic already computes. Copying main's file wholesale would AttributeError.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
DF = SP / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

# --- 1. the inert-row sentinel -----------------------------------------------------------------
MAP_OLD = """        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )"""
MAP_NEW = """        # -1 marks an inert sampling row. CUDA graph capture can execute the full buffer
        # before a real batch has populated it, so zero would make every padding row
        # scatter into request slot 0.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )"""

CAP_OLD = "        self.sample_idx_mapping.zero_()"
CAP_NEW = "        self.sample_idx_mapping.fill_(-1)"

# --- 2 + 3. rejected suffix is not context, and the null block is not a slot ---------------------
CTX_OLD = """    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)"""

CTX_NEW = """    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    # The rejected suffix of the previous step sits inside [0, num_ctx) but is NOT context:
    # loading it makes the target read back its own rejected proposals as accepted history.
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_valid_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map to it after
    # eviction, and rejected suffix rows are not valid context either. Neither kind of row
    # may write draft KV into physical block 0.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    ctx_slot = tl.where(
        ctx_resident, ctx_block_id * block_size + (ctx_pos % block_size), PAD_SLOT_ID
    )
    # Stored over the full [0, num_ctx) span while the loads above are masked to
    # [0, num_valid_ctx): the rejected rows in between get position 0 and PAD_SLOT_ID. They
    # write no KV and their positions are never consumed, but the span must stay fully
    # initialised so a replayed graph cannot observe a stale value from an earlier batch.
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)"""

NVC_OLD = "    valid_ctx_end = ctx_end - num_rejected"
NVC_NEW = ("    valid_ctx_end = ctx_end - num_rejected\n"
           "    num_valid_ctx = valid_ctx_end - ctx_start")

QSLOT_OLD = "    q_slot = q_block_id * block_size + (query_pos % block_size)"
QSLOT_NEW = """    # A null block is never a writable cache slot; a sliding-window block table can carry
    # evicted or global padding entries.
    q_resident = is_query & (q_block_id != 0)
    q_slot = tl.where(
        q_resident, q_block_id * block_size + (query_pos % block_size), PAD_SLOT_ID
    )"""

SEQ_OLD = ("        tl.store(out_seq_lens_ptr + req_idx, "
           "last_valid_pos + 1 + num_query_per_req)")
SEQ_NEW = """        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )"""


def main() -> None:
    apply(DF, MAP_OLD, MAP_NEW, "sample_idx_mapping = torch.full", "dflash: inert-row sentinel")
    apply(DF, CAP_OLD, CAP_NEW, "sample_idx_mapping.fill_(-1)", "dflash: capture resets to -1")
    apply(DF, NVC_OLD, NVC_NEW, "num_valid_ctx = valid_ctx_end", "dflash: num_valid_ctx")
    apply(DF, CTX_OLD, CTX_NEW, "is_valid_ctx = j < num_valid_ctx", "dflash: rejected suffix")
    apply(DF, QSLOT_OLD, QSLOT_NEW, "q_resident = is_query", "dflash: null block query slot")
    apply(DF, SEQ_OLD, SEQ_NEW, "tl.minimum(last_valid_pos + 1", "dflash: clamp seq_lens")


if __name__ == "__main__":
    main()
