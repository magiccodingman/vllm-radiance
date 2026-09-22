"""Use the corrected stock convolution for speculative rows as well as M1.

The R4D conversion rounds BF16 midpoints away from zero; stock uses nearest
even. Sharing the stock operation also shares its SiLU and FP32 accumulation.
This adapter preserves Radiance's raw-input and speculative-history ABI.
"""

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class StockConvolutionAdapter:
    def __init__(self, torch, update):
        self.torch = torch
        self.update = update
        self.calls = 0

    def __call__(
        self,
        x,
        weight,
        bias,
        history,
        state_len,
        indices,
        accepted,
        cu,
        num_seqs,
        tokens,
        heads,
        query_heads,
        max_query_len,
    ):
        t = self.torch
        channels = (2 * query_heads + heads) * 128
        if (
            num_seqs != 1
            or heads != 48
            or query_heads != 16
            or not 1 <= max_query_len <= 8
            or not 1 <= tokens <= max_query_len
            or state_len != 3 + max_query_len - 1
            or tuple(x.shape) != (tokens, channels)
            or tuple(weight.shape) != (channels, 4)
            or history.ndim != 3
            or tuple(history.shape[1:]) != (channels, state_len)
            or x.dtype != t.bfloat16
            or weight.dtype != t.bfloat16
            or history.dtype != t.bfloat16
            or tuple(indices.shape) != (1,)
            or tuple(cu.shape) != (2,)
            or bias is not None
        ):
            raise DiagnosticError("unsupported stock convolution diagnostic invocation")
        output = t.empty_like(x)
        result = self.update(
            x,
            history,
            weight,
            None,
            "silu",
            conv_state_indices=indices,
            num_accepted_tokens=accepted,
            query_start_loc=cu,
            max_query_len=max_query_len,
            validate_data=False,
            out=output,
        )
        if result is not output:
            raise DiagnosticError("stock convolution did not retain the supplied output")
        q, k, v = result.split((query_heads * 128, query_heads * 128, heads * 128), dim=-1)
        self.calls += 1
        return (
            q.contiguous().view(tokens, query_heads, 128),
            k.contiguous().view(tokens, query_heads, 128),
            v.contiguous().view(tokens, heads, 128),
        )
