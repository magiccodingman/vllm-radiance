#!/usr/bin/env python3
"""The whole libr4d integration, in one patch, switchable at run time with RADIANCE_USE_R4D.

libr4d is a library of hand-written gfx1201 (RDNA4) kernels. vLLM reaches into it from several
places, but only three of them need a source edit, and they are here together because they are one
feature: a build either has the R4D library wired in or it does not. A single switch for it is only
meaningful if there is a single place that says what "it" is.

  1. THE ATTENTION BACKEND. The enum in vllm/v1/attention/backends/registry.py is what the CLI, the
     speculative-decoding config and the platform selector all resolve names against, and a Python
     Enum cannot be extended after it is created -- so the member is added to the source. R4D is
     deliberately NOT added to the platform's auto-selection list: it is an explicit opt-in through
     `--attention-backend R4D`, and everything else keeps its defaults.

  2. THE GDN LAYER. A prologue on _forward_core runs a whole gated-delta-net step in five
     hand-written kernels -- conv_prep, kkt_solve and chunk_scan on the prefill path, conv_update
     and recurrent_update on the decode path -- and returns False for any step shape it does not
     cover, leaving the original body to run untouched. The reason this is at the layer and not at
     the op seams is the convolution: fusing it with the split, the qk l2 norm, the gating and the
     gate cumsum keeps its output (34 MB per layer per rank at this model's chunk size) out of HBM
     entirely, and no op-level hook can see both ends of that.

  3. THE FLA CHUNK PATH. For any step the layer hook declines, this still routes the last three of
     FLA's five prefill kernels -- recompute_w_u_fwd, chunk_gated_delta_rule_fwd_h and chunk_fwd_o
     -- to the fused chunk scan, so the fallback path is not the unaccelerated one.

RADIANCE_USE_R4D (default 1) is read by the radiance_* modules the injected code calls into, not by
the injected code here: this patch is applied once at image build, and what a serve needs is to be
able to take the library out of the picture without a rebuild. Set it to 0 and every hook is still
installed and every one of them declines, so the fallbacks -- FLA/Triton for the gated delta net,
the Triton kernel for vision attention, RCCL for the all-reduce, rocBLAS for the router GEMM -- run
exactly as they would in an image built without the library. `--attention-backend R4D` then refuses
at startup, which is better than quietly serving something slower than what was asked for.

Every edit is idempotent and anchors on source that would have to change for the replacement to be
wrong, so drift shows up as a failed build rather than as a silent divergence.

NOT PATCHED: the gated RMS norm. `_output_projection` runs INSIDE the compiled region, so a python
call there is a Dynamo `Unsupported method call` and the model fails to compile -- and the same
region is where vLLM's fuse_norm_quant pass may already have claimed the norm. The kernel exists
(`gdn_gated_rmsnorm_h128_bf16`, and the decode kernel can fold the same arithmetic into its
epilogue), but wiring it needs a registered custom op and a trace that prices what the norm actually
costs today. Left out rather than shipped on a guess."""
import sysconfig
from pathlib import Path

from _patchlib import apply

PURELIB = Path(sysconfig.get_paths()["purelib"])
REG = PURELIB / "vllm/v1/attention/backends/registry.py"
F = PURELIB / "vllm/third_party/flash_linear_attention/ops/chunk.py"
L = PURELIB / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"


# ---- 1. the attention backend enum ---------------------------------------------------------
REG_OLD = '    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"\n'
REG_NEW = REG_OLD + '    R4D = "radiance_r4d_attn.R4DAttentionBackend"\n'


# ---- 2. the gated-delta-net layer ------------------------------------------------------------
LAYER_IMPORT_OLD = "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
LAYER_IMPORT_NEW = (
    LAYER_IMPORT_OLD
    + "\n"
    + "try:\n"
    + "    import radiance_gdn as _radiance_gdn\n"
    + "except Exception:\n"
    + "    _radiance_gdn = None\n"
)

# Anchor the full standard-path prologue. Pinned vLLM main now has a second
# fused-norm helper with the same warm-up body, so the old four-line anchor is
# no longer unique even though the intended insertion point is unchanged.
LAYER_OLD = (
    "    def _forward_core(\n"
    "        self,\n"
    "        mixed_qkv: torch.Tensor,\n"
    "        b: torch.Tensor,\n"
    "        a: torch.Tensor,\n"
    "        core_attn_out: torch.Tensor,\n"
    "    ):\n"
    "        \"\"\"Core conv1d + recurrent attention (standard path).\n"
    "\n"
    "        Args:\n"
    "            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)\n"
    "            b: beta gating vector                   (num_tokens, num_heads)\n"
    "            a: alpha gating vector                  (num_tokens, num_heads)\n"
    "            core_attn_out: Pre-allocated output buffer for attention results.\n"
    "        \"\"\"\n"
    "        forward_context = get_forward_context()\n"
    "        attn_metadata_raw = forward_context.attn_metadata\n"
    "\n"
    "        if attn_metadata_raw is None:\n"
    "            self._warmup_prefill_kernels(mixed_qkv, 0)\n"
    "            return\n"
)
LAYER_NEW = (
    LAYER_OLD
    + "\n"
    + "        # --- RADIANCE all-R4D gated delta net (patch_r4d.py) ---\n"
    + "        # conv_prep -> kkt_solve -> chunk_scan for a prefill step, conv_update ->\n"
    + "        # recurrent_update for a speculative decode step. Returns False for any step it\n"
    + "        # does not cover, which leaves the Triton body below exactly as it was.\n"
    + "        if _radiance_gdn is not None and _radiance_gdn.ALL:\n"
    + "            if _radiance_gdn.forward_core_fused(self, mixed_qkv, b, a, core_attn_out):\n"
    + "                return\n"
)


# ---- 3. the FLA chunk path -------------------------------------------------------------------
IMPORT_OLD = (
    "def chunk_gated_delta_rule_fwd(\n"
    "    q: torch.Tensor,\n"
)
IMPORT_NEW = (
    "try:\n"
    "    import radiance_gdn as _radiance_gdn\n"
    "except Exception:\n"
    "    _radiance_gdn = None\n"
    "\n"
    + IMPORT_OLD
)

# The anchor ends where the replaced work begins, so a drift in any of the three kernels this
# bypasses shows up as a failed patch rather than as a silent divergence.
BODY_OLD = (
    "    A = solve_tril(\n"
    "        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype\n"
    "    )\n"
    "    w, u = recompute_w_u_fwd(\n"
)
BODY_NEW = (
    "    A = solve_tril(\n"
    "        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype\n"
    "    )\n"
    "    # --- RADIANCE fused GDN prefill kernel (patch_r4d.py) ---\n"
    "    # One kernel for the WY representation, the state scan and the output. It returns None\n"
    "    # for any shape it was not compiled for, which leaves the Triton path below untouched.\n"
    "    if _radiance_gdn is not None and _radiance_gdn.ENABLED and SUPPRESS_LEVEL < 3:\n"
    "        _fused = _radiance_gdn.fused_prefill(\n"
    "            q, k, v, A, g, beta, scale, initial_state, output_final_state, cu_seqlens,\n"
    "            core_attn_out,\n"
    "        )\n"
    "        if _fused is not None:\n"
    "            return g, _fused[0], A, _fused[1], None, None, None\n"
    "    w, u = recompute_w_u_fwd(\n"
)


def main():
    apply(REG, REG_OLD, REG_NEW, "radiance_r4d_attn", "R4D attention backend enum")
    apply(L, LAYER_IMPORT_OLD, LAYER_IMPORT_NEW, "import radiance_gdn as _radiance_gdn",
          "RADIANCE all-R4D GDN import")
    apply(L, LAYER_OLD, LAYER_NEW, "RADIANCE all-R4D gated delta net",
          "route the whole GDN layer -> R4D kernels")
    apply(F, IMPORT_OLD, IMPORT_NEW, "import radiance_gdn as _radiance_gdn",
          "RADIANCE fused GDN import")
    apply(F, BODY_OLD, BODY_NEW, "RADIANCE fused GDN prefill kernel (patch_r4d.py)",
          "route GDN prefill wy+delta_h+chunk_o -> fused R4D kernel")


if __name__ == "__main__":
    main()
