#!/usr/bin/env python3
"""Bake: align the multimodal placeholder mask with the drafter's compacted token layout.

The MTP drafter builds its first-pass inputs by embedding `input_ids[:num_tokens]` and
overwriting the image-placeholder positions with the vision embeddings via `is_mm_embed`.
The mask comes from `_gather_mm_embeddings`, which sizes it to `total_num_scheduled_tokens`
(the target model's scheduled count). Under `disable_padded_drafter_batch` the target tokens
fed to the drafter are gathered by `token_indices` (rejected tokens dropped), so the drafter
buffer is shorter than the mask. `_merge_multimodal_embeddings` then does
`inputs_embeds[is_multimodal]` with a mask longer than the tensor ->
`IndexError: shape of the mask [S] does not match the indexed tensor [S-r, H]` -> EngineDeadError.
This hits any multimodal request that reaches the drafter with a non-empty placeholder set
(chunked-prefill step containing image tokens while verifying the previous draft).

Fix: re-index the mask with the same `token_indices` used to compact the target tokens, so its
length and ordering match the draft inputs_embeds buffer. Image placeholders are prompt tokens
(always accepted, never drafted or rejected), so within the image region the gather is contiguous
and the number of set positions is preserved -> the vision embeddings still land 1:1. A no-op when
the mask already matches (no rejection, or the padded drafter path where token_indices is a range).
Idempotent. Enables multimodal serving with MTP speculative decoding + disable_padded_drafter_batch."""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/worker/gpu_model_runner.py"

# 1. Give token_indices a defined default across all three drafter-input branches
#    (only the disable_padded_drafter_batch branch binds it locally).
INIT_ANCHOR = "            num_rejected_tokens_gpu = None\n"
INIT_ADD = "            token_indices = None\n"

# 2. Re-index the placeholder mask onto the compacted token layout.
MASK_ANCHOR = (
    "                mm_embed_inputs = self._gather_mm_embeddings(\n"
    "                    scheduler_output,\n"
    "                    shift_computed_tokens=1,\n"
    "                )\n"
)
MASK_NEW = (
    "                mm_embeds, is_mm_embed = self._gather_mm_embeddings(\n"
    "                    scheduler_output,\n"
    "                    shift_computed_tokens=1,\n"
    "                )\n"
    "                # RADIANCE: align the placeholder mask with the drafter's\n"
    "                # compacted token layout (disable_padded_drafter_batch drops\n"
    "                # rejected tokens via token_indices; the mask is built at\n"
    "                # scheduled scale). Without this the mask outlives the draft\n"
    "                # inputs_embeds buffer -> IndexError. Placeholders are prompt\n"
    "                # tokens (never rejected), so the gather preserves them 1:1.\n"
    "                if (\n"
    "                    is_mm_embed is not None\n"
    "                    and token_indices is not None\n"
    "                    and is_mm_embed.shape[0] != target_token_ids.shape[0]\n"
    "                ):\n"
    "                    is_mm_embed = is_mm_embed[token_indices.to(is_mm_embed.device)]\n"
    "                mm_embed_inputs = (mm_embeds, is_mm_embed)\n"
)


def main():
    apply(F, INIT_ANCHOR, INIT_ANCHOR + INIT_ADD,
          "            token_indices = None\n", "mtp mm-mask: token_indices default")
    apply(F, MASK_ANCHOR, MASK_NEW,
          "RADIANCE: align the placeholder mask", "mtp mm-mask: re-index on compaction")


if __name__ == "__main__":
    main()
