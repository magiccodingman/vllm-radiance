#!/usr/bin/env python3
"""Add an opt-in AITER 0.1.20 GDN prefill backend for RDNA4.

Pinned vLLM main can use AITER's RDNA4 Triton kernels for GDN decode, but its
prefill path still selects the vendored Triton/FLA pipeline.  AITER 0.1.20 has
an optimized full prefill pipeline whose K5 recurrence uses the gfx1201 HIP
WMMA kernel.  This patch exposes it as the explicit experimental setting
``gdn_prefill_backend=aiter``; ``auto`` retains vLLM's established fallback.

The vLLM metadata builder already has host-resident sequence lengths.  It uses
them here to build AITER's reusable schedule once per batch, avoiding the
otherwise repeated device-to-host chunk-offset read in every GDN layer.

The patch is deliberately source-drift guarded against the pinned vLLM tree.
"""

import ast
import sysconfig
from pathlib import Path


SP = Path(sysconfig.get_paths()["purelib"])
QWEN = SP / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
META = SP / "vllm/v1/attention/backends/gdn_attn.py"


def _replace(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise SystemExit(
            f"  FAIL  AITER GDN {label}: anchor matched {count}x, expected 1"
        )
    return source.replace(anchor, replacement, 1)


def transform_qwen(source: str) -> str:
    if "def forward_aiter(" in source:
        return source

    source = _replace(
        source,
        ') -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:',
        ') -> tuple[str, Literal["triton", "flashinfer", "cutedsl", "aiter"]]:',
        "backend result type",
    )

    anchor = '''    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
'''
    replacement = '''    backend = str(backend_cfg).strip().lower()

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    # AITER 0.1.20's optimized VK-layout prefill pipeline includes a gfx1201
    # HIP/WMMA K5 recurrence.  Keep it behind the existing backend selector;
    # importing the implementation also verifies that the required AITER API
    # is present rather than treating all AITER releases as equivalent.
    supports_aiter = False
    if current_platform.is_rocm() and head_k_dim == 128:
        try:
            from aiter.ops.triton.gated_delta_net import (  # noqa: F401
                chunk_gated_delta_rule_opt_vk,
            )

            supports_aiter = (
                rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
            )
        except (ImportError, AttributeError):
            supports_aiter = False

    if backend == "aiter" and supports_aiter:
        return backend, "aiter"
    if not current_platform.is_cuda():
        return backend, "triton"
'''
    source = _replace(source, anchor, replacement, "RDNA4 selection")

    source = _replace(
        source,
        '''    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
''',
        '''    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "aiter": "AITER HIP/Triton",
        "triton": "Triton/FLA",
    }[active_backend]
''',
        "backend log label",
    )

    source = _replace(
        source,
        '        if backend in ("flashinfer", "cutedsl") and active_backend != backend:',
        '        if backend in ("flashinfer", "cutedsl", "aiter") and active_backend != backend:',
        "fallback warning",
    )

    source = _replace(
        source,
        '''        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native
''',
        '''        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        elif active_backend == "aiter":
            self._forward_method = self.forward_aiter
        else:
            self._forward_method = self.forward_native
''',
        "forward dispatch",
    )

    # All implementations receive the optional schedule because dispatch is
    # selected per model instance.  Non-AITER implementations simply ignore it.
    signature_anchor = '''        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
'''
    signature_replacement = '''        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        aiter_prefill_metadata: object | None = None,
    ):
'''
    if source.count(signature_anchor) != 3:
        raise SystemExit(
            "  FAIL  AITER GDN forward signatures: expected three backend signatures"
        )
    source = source.replace(signature_anchor, signature_replacement)

    insert_anchor = '''    def forward_cutedsl(
'''
    forward_aiter = '''    def forward_aiter(
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
    source = _replace(source, insert_anchor, forward_aiter + insert_anchor, "AITER forward")

    source = _replace(
        source,
        '''                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
''',
        '''                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                aiter_prefill_metadata=attn_metadata.aiter_prefill_metadata,
            )
''',
        "runtime schedule handoff",
    )

    warmup_anchor = '''        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
'''
    warmup_replacement = '''        # Backend-specific kernels require reusable metadata.
        chunk_indices = None
        chunk_offsets = None
        aiter_prefill_metadata = None
        if self.gdn_prefill_backend == "aiter":
            from aiter.ops.triton.gated_delta_net import (
                build_gated_delta_rule_prefill_metadata,
            )

            aiter_prefill_metadata = build_gated_delta_rule_prefill_metadata(
                [T], cu_seqlens=cu_seqlens, chunk_size=FLA_CHUNK_SIZE
            )
        if self.gdn_prefill_backend == "cutedsl":
'''
    source = _replace(source, warmup_anchor, warmup_replacement, "warmup metadata")

    warmup_call = '''                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
'''
    warmup_call_new = '''                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                aiter_prefill_metadata=aiter_prefill_metadata,
            )
'''
    source = _replace(source, warmup_call, warmup_call_new, "warmup schedule handoff")
    return source


def transform_metadata(source: str) -> str:
    if "aiter_prefill_metadata: object | None" in source:
        return source

    source = _replace(
        source,
        '''    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
''',
        '''    prefill_has_initial_state: torch.Tensor | None = None
    # Reusable AITER host/device chunk schedule.  Built once per batch so every
    # GDN layer avoids an offsets device-to-host synchronization.
    aiter_prefill_metadata: object | None = None

    # The following attributes are for triton implementation of causal_conv1d
''',
        "metadata field",
    )
    source = _replace(
        source,
        '        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]',
        '        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl", "aiter"]',
        "builder backend type",
    )

    source = _replace(
        source,
        '''        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
''',
        '''        prefill_has_initial_state: torch.Tensor | None = None
        aiter_prefill_metadata: object | None = None
        if num_prefills > 0:
''',
        "builder schedule variable",
    )

    build_anchor = '''            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )
'''
    build_replacement = build_anchor + '''
            if self.gdn_prefill_backend == "aiter":
                from aiter.ops.triton.gated_delta_net import (
                    build_gated_delta_rule_prefill_metadata,
                )
                from vllm.third_party.flash_linear_attention.ops.utils import (
                    FLA_CHUNK_SIZE,
                )

                prefill_seq_lens_cpu = (
                    prefill_query_start_loc_cpu[1:]
                    - prefill_query_start_loc_cpu[:-1]
                ).tolist()
                aiter_prefill_metadata = (
                    build_gated_delta_rule_prefill_metadata(
                        prefill_seq_lens_cpu,
                        cu_seqlens=prefill_query_start_loc,
                        chunk_size=FLA_CHUNK_SIZE,
                    )
                )
'''
    source = _replace(source, build_anchor, build_replacement, "schedule construction")

    source = _replace(
        source,
        '''            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
''',
        '''            prefill_has_initial_state=prefill_has_initial_state,
            aiter_prefill_metadata=aiter_prefill_metadata,
            spec_query_start_loc=spec_query_start_loc,
''',
        "metadata construction",
    )
    return source


def _patch(path: Path, transform, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"  FAIL  AITER GDN {label}: {path} missing")
    source = path.read_text()
    transformed = transform(source)
    if transformed == source:
        print(f"  NOOP  AITER GDN {label} already applied")
        return
    ast.parse(transformed)
    path.write_text(transformed)
    print(f"  OK    AITER GDN {label}")


def main() -> None:
    _patch(QWEN, transform_qwen, "Qwen backend")
    _patch(META, transform_metadata, "metadata bridge")


if __name__ == "__main__":
    main()
