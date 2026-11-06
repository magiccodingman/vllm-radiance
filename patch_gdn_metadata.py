#!/usr/bin/env python3
"""Bounded v0.33 GDN bookkeeping optimization, preserving upstream state selection.

In particular, zero-token drafts MUST retain the speculative state-offset path.
The old 0.30 sum-of-drafted-tokens shortcut must never be restored here.
Upstream now owns the padded-row block-table slice and cross-group metadata reuse.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backends/gdn_attn.py"
HELPERS = '''_RADIANCE_GDN_META = __import__("os").environ.get("RADIANCE_GDN_META", "1") == "1"


def _radiance_gdn_indices(builder, size, device):
    cached = getattr(builder, "_radiance_gdn_arange", None)
    if cached is None or cached.device != device or cached.numel() < size:
        cached = torch.arange(size, dtype=torch.int32, device=device)
        builder._radiance_gdn_arange = cached
    empty = getattr(builder, "_radiance_gdn_empty", None)
    if empty is None or empty.device != device:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        builder._radiance_gdn_empty = empty
    return cached[:size], empty


'''
MASK_OLD = '''            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
'''
MASK_NEW = '''            if _RADIANCE_GDN_META:
                mask = num_decode_draft_tokens_cpu.numpy() >= 0
                spec_sequence_masks_cpu = torch.from_numpy(mask)
                num_spec_decodes = int(mask.sum())
            else:
                spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
                num_spec_decodes = spec_sequence_masks_cpu.sum().item()
'''
INDEX_OLD = '''                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
'''
INDEX_NEW = '''                if _RADIANCE_GDN_META:
                    spec_token_indx, non_spec_token_indx = _radiance_gdn_indices(
                        self, spec_token_size, query_start_loc.device
                    )
                else:
                    spec_token_indx = torch.arange(
                        spec_token_size, dtype=torch.int32, device=query_start_loc.device
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
'''


def main():
    anchor = "class GDNAttentionBackend(AttentionBackend):"
    apply(F, anchor, HELPERS + anchor, "def _radiance_gdn_indices(", "GDN cached index helper")
    apply(F, MASK_OLD, MASK_NEW, "mask = num_decode_draft_tokens_cpu.numpy() >= 0", "GDN CPU mask")
    apply(F, INDEX_OLD, INDEX_NEW, "= _radiance_gdn_indices(", "GDN persistent indices")


if __name__ == "__main__":
    main()
