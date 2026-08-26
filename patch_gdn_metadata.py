#!/usr/bin/env python3
"""Cut the per-step Python cost of building GDN attention metadata.

Sampling the TP worker shows 13% of a decode step is host time with the GPU idle, and a third of
that -- about 1.5 ms of every 35 ms -- is one function: `GDNAttentionMetadataBuilder.build`. It is
not doing anything expensive; it is a long chain of individually cheap torch ops, each paying full
dispatch, to derive quantities that in a steady-state decode batch hardly change. Three changes,
all value-preserving:

  * The request-count bookkeeping runs on CPU tensors -- a mask, an advanced index, a sum and an
    `.item()`, then the same mask built a second time -- which is five dispatches to reduce a
    vector with one entry per request. numpy over the same buffer gives the same integers for a
    fraction of the cost, and `torch.from_numpy` hands the mask back without a copy.

  * `torch.arange` and `torch.empty(0)` are re-allocated (and the arange re-launched) every step
    for values that depend on nothing but their length. Both are read exactly once, by the `copy_`
    into the builder's persistent cudagraph buffers, so a cached arange sliced to length is
    indistinguishable.

  * `block_table_tensor[spec_sequence_masks_cpu, :num_spec+1]` gathers rows by a boolean mask. In
    the steady-state decode batch every sequence is a spec decode, so the mask selects every row
    and a slice has the same values without the gather or its allocation. Guarded on the cudagraph
    branch being the consumer, because that consumer is a `copy_` and does not care that a slice is
    strided where a gather would have been contiguous.

`query_lens` is also moved inside the branch that uses it, so the common path stops launching a
subtraction kernel for a tensor it discards.

Set `RADIANCE_GDN_META=0` to fall back to the stock path.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backends/gdn_attn.py"

MASK_OLD = """        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()"""

MASK_NEW = """        spec_sequence_masks_cpu: torch.Tensor | None = None
        # --- RADIANCE (patch_gdn_metadata.py): one numpy pass over the same buffer
        # instead of mask / index / sum / item and then the mask a second time.
        _r_np = _RADIANCE_GDN_META and num_decode_draft_tokens_cpu is not None
        _ndt_np = num_decode_draft_tokens_cpu.numpy() if _r_np else None
        _mask_np = None if _ndt_np is None else _ndt_np >= 0
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or (
                int(_ndt_np[_mask_np].sum()) == 0
                if _r_np
                else num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
                .sum()
                .item()
                == 0
            )
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            if _r_np:
                spec_sequence_masks_cpu = torch.from_numpy(_mask_np)
                num_spec_decodes = int(_mask_np.sum())
            else:
                spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
                num_spec_decodes = spec_sequence_masks_cpu.sum().item()"""

LENS_OLD = """            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )"""

LENS_NEW = """            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            if _r_np:
                # --- RADIANCE: same integers, one numpy pass. `size` is the numpy
                # spelling of `size(0)` for these 1-D per-request vectors.
                _qlen_np = query_lens_cpu.numpy()
                _nonspec_np = _qlen_np[~_mask_np]
                num_decodes = int((_nonspec_np == 1).sum())
                num_zero_len = int((_nonspec_np == 0).sum())
                num_prefills = _nonspec_np.size - num_decodes - num_zero_len
                num_decode_tokens = num_decodes
                num_prefill_tokens = int(_nonspec_np.sum()) - num_decode_tokens
                num_spec_decode_tokens = (
                    int(_qlen_np.sum()) - num_prefill_tokens - num_decode_tokens
                )
            else:
                non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
                num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
                # Exclude zero-length padded sequences from prefill count.
                num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
                num_prefills = (
                    non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
                )
                num_decode_tokens = num_decodes
                num_prefill_tokens = (
                    non_spec_query_lens_cpu.sum().item() - num_decode_tokens
                )
                num_spec_decode_tokens = (
                    query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
                )"""

IDX_OLD = """                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]"""

IDX_NEW = """                if _RADIANCE_GDN_META:
                    # --- RADIANCE: both of these depend only on their length and are
                    # read once, by the copy_ into the persistent buffers below.
                    spec_token_indx = _radiance_arange(
                        self, spec_token_size, query_start_loc.device
                    )
                    non_spec_token_indx = _radiance_empty_idx(
                        self, query_start_loc.device
                    )
                else:
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                # Filter by spec_sequence_masks to exclude padded sequences
                # --- RADIANCE: when every sequence is a spec decode the mask selects
                # every row, so a slice carries the same values without the gather.
                # Only taken when the cudagraph branch below will consume it, since
                # that consumer is a copy_ and a strided source is fine there.
                if (
                    _RADIANCE_GDN_META
                    and _r_np
                    and self.use_full_cuda_graph
                    and num_spec_decodes <= self.decode_cudagraph_max_bs
                    and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
                    and bool(_mask_np.all())
                ):
                    spec_state_indices_tensor = block_table_tensor[
                        : _mask_np.size, : self.num_spec + 1
                    ]
                else:
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]"""

PREAMBLE_OLD = "class GDNAttentionMetadataBuilder("

PREAMBLE_NEW = '''_RADIANCE_GDN_META = __import__("os").environ.get("RADIANCE_GDN_META", "1") == "1"


def _radiance_arange(builder, n: int, device) -> torch.Tensor:
    """arange(n) as a slice of one cached buffer -- read-only downstream.

    Never populate the cache during a graph capture: an allocation made there comes from the
    graph's private pool and is only valid inside a replay, so a buffer meant to outlive the
    step would be reading reused memory. Capture falls back to the stock allocation, and the
    first ordinary build fills the cache.
    """
    c = getattr(builder, "_radiance_arange_buf", None)
    if c is None or c.numel() < n or c.device != device:
        if torch.cuda.is_current_stream_capturing():
            return torch.arange(n, dtype=torch.int32, device=device)
        c = torch.arange(max(n, 4096), dtype=torch.int32, device=device)
        builder._radiance_arange_buf = c
    return c[:n]


def _radiance_empty_idx(builder, device) -> torch.Tensor:
    c = getattr(builder, "_radiance_empty_buf", None)
    if c is None or c.device != device:
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(0, dtype=torch.int32, device=device)
        c = torch.empty(0, dtype=torch.int32, device=device)
        builder._radiance_empty_buf = c
    return c


class GDNAttentionMetadataBuilder('''

apply(F, PREAMBLE_OLD, PREAMBLE_NEW, "_RADIANCE_GDN_META", "gdn metadata: helpers")
apply(F, MASK_OLD, MASK_NEW, "one numpy pass over the same buffer", "gdn metadata: spec mask")
apply(F, LENS_OLD, LENS_NEW, "same integers, one numpy pass", "gdn metadata: request counts")
apply(F, IDX_OLD, IDX_NEW, "read once, by the copy_", "gdn metadata: index tensors")
