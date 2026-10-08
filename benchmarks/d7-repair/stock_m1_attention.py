"""Experimental D7 attention using independent native M1 query workgroups.

The queries share the same immutable paged KV storage. Each receives its own
causal sequence length, M1 split law and wave-level softmax control flow.
Only the pinned, single-sequence decode shape is admitted; prefill is unchanged.
"""

import functools
import hashlib
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError

ATTENTION_SOURCE = "6d4b4571c938003d79e2edc322d1eadc674e30781597845a59b59b8065a6fd0a"


def m1_splits(context):
    """Pinned libr4d split law for one sequence with four KV heads."""
    if type(context) is not int or context < 1:
        raise DiagnosticError("M1 attention requires a positive context bound")
    tiles = (context + 15) // 16
    splits = 32  # floor-power-of-two(192 / (four KV heads * one sequence))
    while splits > 16 and splits > tiles // 2:
        splits //= 2
    return max(1, min(splits, tiles))


def split_groups(width, bound):
    if type(width) is not int or not 1 <= width <= 8 or bound < width:
        raise DiagnosticError("independent attention query range is outside the contract")
    groups = []
    for row in range(width):
        splits = m1_splits(bound - width + row + 1)
        if groups and groups[-1][2] == splits:
            groups[-1] = (groups[-1][0], row + 1, splits)
        else:
            groups.append((row, row + 1, splits))
    return groups


class StockM1Attention:
    def __init__(self, hooks):
        import torch  # isort: skip

        import radiance_r4d_attn as native

        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != ATTENTION_SOURCE:
            raise DiagnosticError("independent attention requires the pinned Radiance wrapper")
        self.calls = self.rows = 0
        original = native.R4DAttentionImpl.forward

        @functools.wraps(original)
        def forward(
            impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            md = attn_metadata
            plan = getattr(md, "r4d_plan", None)
            if not plan or all(row[2] == 1 or row[2] > 8 for row in plan):
                return original(
                    impl,
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    md,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if (
                len(plan) != 1
                or plan[0][:2] != (0, 1)
                or plan[0][3] != 0
                or md.causal is not True
                or output_scale is not None
                or output_block_scale is not None
            ):
                raise DiagnosticError("independent attention requires one unmixed decode sequence")
            width = plan[0][2]
            if (impl.num_heads, impl.num_kv_heads, impl.head_size) != (24, 4, 256):
                raise DiagnosticError(
                    "independent attention geometry differs from the pinned model"
                )
            if (
                query.shape != (width, 24, 256)
                or output.shape != query.shape
                or not query.is_contiguous()
                or not output.is_contiguous()
                or query.dtype != torch.bfloat16
                or output.dtype != query.dtype
            ):
                raise DiagnosticError("independent attention query/output layout changed")
            variant, block_stride, head_stride = impl._geometry(kv_cache, query, output)
            groups = split_groups(width, md.r4d_max_ctx)
            table = md.block_table[:1].expand(width, -1).contiguous()
            lengths = md.seq_lens[:1] - torch.arange(
                width - 1, -1, -1, device=query.device, dtype=torch.int32
            )
            max_blocks = table.shape[1]
            scales = []
            for name in ("_k_scale_float", "_v_scale_float"):
                scale = float(getattr(layer, name, 1.0))
                scales.append(
                    None
                    if scale == 1.0
                    else torch.full((width, 4), scale, device=query.device, dtype=torch.float32)
                )
            for start, stop, splits in groups:
                rows = stop - start
                needed = native.r4d.attn_decode_h256_gqa6_scratch_bytes(
                    rows, 1, 24, 4, 256, md.r4d_max_ctx, splits
                )
                if needed > md.r4d_scratch.numel() * md.r4d_scratch.element_size():
                    raise DiagnosticError("independent attention scratch capacity is insufficient")
                native._DECODE[variant](
                    query[start:stop].data_ptr(),
                    kv_cache.data_ptr(),
                    table[start:stop].data_ptr(),
                    lengths[start:stop].data_ptr(),
                    output[start:stop].data_ptr(),
                    *(0 if s is None else s[start:stop].data_ptr() for s in scales),
                    md.r4d_scratch.data_ptr(),
                    rows,
                    1,
                    24,
                    4,
                    256,
                    16,
                    max_blocks,
                    block_stride,
                    head_stride,
                    impl.scale,
                    splits,
                    md.r4d_max_ctx,
                    torch.cuda.current_stream().cuda_stream,
                )
            self.calls += 1
            self.rows += width
            return output

        hooks.replace(native.R4DAttentionImpl, "forward", forward)

    def receipt(self):
        return {
            "calls": self.calls,
            "rows": self.rows,
            "mode": "independent-m1-query-workgroups",
            "scope": "Single active sequence; D7 widths 2 through 8, paged FP8/BF16 KV.",
        }
