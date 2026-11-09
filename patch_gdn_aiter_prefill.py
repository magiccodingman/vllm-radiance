#!/usr/bin/env python3
"""Retain the explicit RDNA4 HIP GDN backend using v0.33's upstream schedule plumbing.

The default remains upstream auto (Triton on gfx1201). The CDNA FlyDSL backend
is upstream-owned; it must not be enabled on RDNA4 by widening its architecture gate.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply
SP = Path(sysconfig.get_paths()["purelib"])
QWEN = SP / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
META = SP / "vllm/v1/attention/backends/gdn_attn.py"

METHOD = '''    def forward_aiter(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        aiter_prefill_metadata: object | None = None,
    ):
        from aiter.ops.triton.gated_delta_net import (
            chunk_gated_delta_rule_opt_vk,
        )

        # AITER's VK layout matches vLLM's persistent GDN state layout.  Its
        # output buffer contract also matches the FLA path used by this layer.
        return chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            o=core_attn_out,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_chunk_hip=True,
            state_dtype=initial_state.dtype,
            snapshot_dtype=k.dtype,
            use_exp2=True,
            prefill_metadata=aiter_prefill_metadata,
        )

'''

def main():
    apply(QWEN,
          '"triton", "flashinfer", "cutedsl", "aiter_flydsl", "sycl"',
          '"triton", "flashinfer", "cutedsl", "aiter_flydsl", "sycl", "aiter"',
          '"sycl", "aiter"', "GDN HIP result annotation")
    apply(QWEN, '        "aiter_flydsl": "AITER FlyDSL",\n',
          '        "aiter_flydsl": "AITER FlyDSL",\n        "aiter": "Radiance AITER HIP/WMMA",\n',
          '"aiter": "Radiance AITER HIP/WMMA"', "GDN HIP diagnostic")
    anchor = '    backend = str(backend_cfg).strip().lower()\n'
    new = anchor + '''    # radiance: explicit HIP/WMMA alternative; never select CDNA FlyDSL on RDNA4.
    if backend == "aiter":
        if not current_platform.is_rocm() or not rocm_aiter_ops.is_rdna_gdn_triton_kernels_available():
            raise RuntimeError("The explicit Radiance AITER HIP backend requires supported RDNA4 kernels")
        return backend, "aiter"
'''
    apply(QWEN, anchor, new, '# radiance: explicit HIP/WMMA alternative', 'GDN explicit RDNA4 HIP backend')
    apply(QWEN, '        elif active_backend == "aiter_flydsl":\n',
          '        elif active_backend == "aiter":\n            self._forward_method = self.forward_aiter\n        elif active_backend == "aiter_flydsl":\n',
          'self._forward_method = self.forward_aiter\n', 'GDN HIP dispatch')
    apply(QWEN, '    def forward_aiter_flydsl(\n', METHOD + '    def forward_aiter_flydsl(\n',
          '    def forward_aiter(\n', 'GDN HIP method')
    # Same metadata builder and chunk size for the HIP and FlyDSL VK pipelines.
    for path in (QWEN, META):
        source = path.read_text()
        anchor = 'self.gdn_prefill_backend == "aiter_flydsl"'
        replacement = 'self.gdn_prefill_backend in ("aiter_flydsl", "aiter")'
        if replacement in source:
            continue
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(f"FAIL GDN schedule: {path}: {count} matches, expected 1")
        source = source.replace(anchor, replacement)
        import ast
        ast.parse(source)
        path.write_text(source)
        print(f"OK GDN upstream reusable schedule: {path.name}")

if __name__ == "__main__":
    main()
