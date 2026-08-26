#!/usr/bin/env python3
"""Let the DFlash context-KV precompute work with a quantized drafter.

`precompute_and_store_context_kv` fuses every layer's K/V projection into one GEMM, and it builds
that fused weight by slicing the raw parameter: `a.qkv_proj.weight[a.q_size:]`. Every assumption
behind that slice fails once the drafter checkpoint is quantized:

  * the parameter is float8_e4m3fn, so the F.linear against bf16 activations raises outright;
  * the rows are not where the slice expects them -- our preshuffle load hook rewrites block-fp8
    weights to [N//16, K*16], so row `q_size` is no longer the first K row; and
  * the buffer is built at the end of load_weights, before the quant method has processed the
    weights at all, so nothing can be read out of them yet.

Rather than teach this path about every weight layout, ask the layer for its own effective weight:
running the identity through `qkv_proj` returns W^T whatever the quant method and layout underneath.
A one-hot row has amax 1, so its activation quantization is exact and the result is the dequantized
weight rounded to the compute dtype -- precisely what the fused GEMM would have multiplied by. The
build is deferred to first use, which is the profile run, so the weights are fully processed and the
allocation still happens long before any CUDA graph capture. Steady-state memory is unchanged: the
fused buffer is the same bf16 tensor the unquantized path builds.

An unquantized drafter keeps the original eager slice, so the bf16 configuration is untouched.

The test is on the dense dtypes rather than on a list of known quantized ones.
The identity path asks the layer for its effective weight and is therefore
scheme-independent.  Enumerating quantized dtypes would silently misroute a
new scheme; notably, OCP MXFP4 stores packed uint8 weights with shape [N, K/2].
Zero and one are exactly representable under MXFP4, so the identity materialization
remains exact for that layout too.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/qwen3_dflash.py"

HELPER_OLD = """@support_torch_compile
class DFlashQwen3Model(nn.Module):"""

HELPER_NEW = '''_DFLASH_DENSE = (torch.bfloat16, torch.float16, torch.float32)


def _dflash_kv_weight_rows(qkv_proj, q_size: int) -> torch.Tensor:
    """The K/V rows of a qkv projection as a dense compute-dtype matrix."""
    weight = qkv_proj.weight
    if weight.dtype in _DFLASH_DENSE:
        return weight[q_size:]
    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)
    eye = torch.eye(
        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device
    )
    out = qkv_proj(eye)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, q_size:].t().contiguous()


@support_torch_compile
class DFlashQwen3Model(nn.Module):'''

SLICE_OLD = """        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)"""
SLICE_NEW = """        self._kv_source_attn = layers_attn
        if layers_attn[0].qkv_proj.weight.dtype not in _DFLASH_DENSE:
            # Deferred: a quantized qkv_proj cannot be read until its quant method has processed
            # the weights, which happens after load_weights returns.
            self._fused_kv_weight = None
        else:
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)"""

# Migration anchor for this fork's previous FP8-only implementation.  Release
# overlays can be built on an already-patched 0.8 image, so fail-closed patching
# must recognize and replace that exact block rather than requiring users to
# rebuild the compiler stack from scratch.
LEGACY_FP8_SLICE_OLD = '''        # KV projection weights: [num_layers * 2 * kv_size, hidden_size].
        #
        # DFlash bypasses LinearBase here and calls F.linear on one fused buffer.
        # A serialized FP8 draft therefore needs its QKV weights materialized in
        # the activation dtype first; otherwise raw float8 parameters reach
        # F.linear and fail (or, with implicit casts, ignore their scales).  This
        # only expands the small K/V slices used by context precomputation.  The
        # ordinary draft forward path remains quantized and uses its configured
        # FP8 kernel.
        def dense_qkv_weight(attn: nn.Module) -> torch.Tensor:
            proj = attn.qkv_proj
            weight = proj.weight
            if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                return weight

            scale = getattr(proj, "weight_scale_inv", None)
            if scale is None:
                scale = getattr(proj, "weight_scale", None)
            if scale is None:
                raise RuntimeError(
                    "DFlash FP8 QKV projection is missing its weight scale"
                )

            dtype = self._hidden_norm_weight.dtype
            dense = weight.to(dtype)
            scale = scale.to(dtype)
            if scale.numel() == 1:
                return dense * scale
            if scale.ndim == 2:
                block_shape = getattr(proj, "weight_block_size", None)
                if not block_shape or len(block_shape) != 2:
                    raise RuntimeError(
                        "DFlash block-FP8 QKV projection is missing block shape"
                    )
                expanded = scale.repeat_interleave(block_shape[0], dim=0)
                expanded = expanded.repeat_interleave(block_shape[1], dim=1)
                return dense * expanded[: dense.shape[0], : dense.shape[1]]
            if scale.ndim == 1 and scale.shape[0] == dense.shape[0]:
                return dense * scale.unsqueeze(1)
            raise RuntimeError(
                f"Unsupported DFlash FP8 QKV scale shape {tuple(scale.shape)} "
                f"for weight {tuple(dense.shape)}"
            )

        kv_weights = [dense_qkv_weight(a)[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)'''

LAZY_OLD = """        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )"""
LAZY_NEW = """        if self._fused_kv_weight is None:
            self._fused_kv_weight = torch.cat(
                [
                    _dflash_kv_weight_rows(a.qkv_proj, a.q_size)
                    for a in self._kv_source_attn
                ],
                dim=0,
            )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )"""


def main() -> None:
    apply(F, HELPER_OLD, HELPER_NEW, "_dflash_kv_weight_rows", "dflash: quantized fused KV helper")
    if "def dense_qkv_weight(attn: nn.Module)" in F.read_text(encoding="utf-8"):
        apply(F, LEGACY_FP8_SLICE_OLD, SLICE_NEW, "self._kv_source_attn",
              "dflash: migrate FP8-only fused KV build")
    else:
        apply(F, SLICE_OLD, SLICE_NEW, "self._kv_source_attn", "dflash: defer fused KV build")
    apply(F, LAZY_OLD, LAZY_NEW, "if self._fused_kv_weight is None",
          "dflash: materialize fused KV on first use")


if __name__ == "__main__":
    main()
