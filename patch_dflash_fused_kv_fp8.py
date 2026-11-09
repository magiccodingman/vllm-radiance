#!/usr/bin/env python3
"""Keep the fused context-KV optimization for quantized DFlash projections.

v0.33 owns the safe per-layer quantized fallback. Retain Radiance's effective-
weight materialization, deferred until all loader hooks have processed weights,
so serialized FP8 and runtime W4 drafts can still use one fused context GEMM.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply
F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/qwen3_dflash.py"

HELPER_OLD = '@support_torch_compile\nclass DFlashQwen3Model(nn.Module):'
HELPER_NEW = '''_DFLASH_DENSE = (torch.bfloat16, torch.float16, torch.float32)


def _dflash_kv_weight_rows(qkv_proj, q_size: int) -> torch.Tensor:
    """The K/V rows of a qkv projection as a dense compute-dtype matrix."""
    weight = qkv_proj.weight
    if weight.dtype in _DFLASH_DENSE and weight.ndim == 2:
        return weight[q_size:]
    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)
    eye = torch.eye(
        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device
    )
    out = qkv_proj.quant_method.apply(qkv_proj, eye, bias=None)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, q_size:].t().contiguous()


@support_torch_compile
class DFlashQwen3Model(nn.Module):'''

SOURCE_OLD = '        self._context_q_sizes = [a.q_size for a in layers_attn]\n'
SOURCE_NEW = SOURCE_OLD + '        self._kv_source_attn = layers_attn\n'
DENSE_OLD = "            isinstance(proj.quant_method, UnquantizedLinearMethod)\n"
DENSE_NEW = DENSE_OLD.rstrip() + "\n            and proj.weight.ndim == 2\n"
LAZY_OLD = '        if self._fused_kv_weight is not None:\n'
LAZY_NEW = '''        if self._fused_kv_weight is None and hasattr(self, "_kv_source_attn"):
            self._fused_kv_weight = torch.cat(
                [_dflash_kv_weight_rows(a.qkv_proj, a.q_size) for a in self._kv_source_attn],
                dim=0,
            )
            biases = [a.qkv_proj.bias for a in self._kv_source_attn]
            if biases[0] is not None:
                self._fused_kv_bias = torch.cat(
                    [bias[a.q_size:] for bias, a in zip(biases, self._kv_source_attn)], dim=0
                )
        if self._fused_kv_weight is not None:
'''

def main():
    apply(F, HELPER_OLD, HELPER_NEW, "_dflash_kv_weight_rows", "dflash: effective quantized KV rows")
    apply(F, SOURCE_OLD, SOURCE_NEW, "self._kv_source_attn", "dflash: retain processed projection sources")
    apply(F, DENSE_OLD, DENSE_NEW, "and proj.weight.ndim == 2",
          "dflash: require dense matrix storage for eager KV")
    apply(F, "    if weight.dtype in _DFLASH_DENSE:\n",
          "    if weight.dtype in _DFLASH_DENSE and weight.ndim == 2:\n",
          "weight.dtype in _DFLASH_DENSE and weight.ndim == 2",
          "dflash: detect packed storage with dense dtype")
    apply(F, "    out = qkv_proj(eye)\n",
          "    out = qkv_proj.quant_method.apply(qkv_proj, eye, bias=None)\n",
          "qkv_proj.quant_method.apply(qkv_proj, eye, bias=None)",
          "dflash: materialize weights without projection bias")
    apply(F, LAZY_OLD, LAZY_NEW, 'hasattr(self, "_kv_source_attn")', "dflash: defer fused KV materialization")

if __name__ == "__main__":
    main()
