#!/usr/bin/env python3
"""gfx1201 (R9700 / RDNA4) platform discovery patch for pinned vLLM main.

Current vLLM has its own deliberately separate RDNA4 AITER route. Do not widen
the CDNA AITER gate: that would expose gfx1201 to CK/MFMA/ASM kernels without
RDNA device code. The old sampler and Triton-driver workarounds are also gone;
upstream now owns those decisions.

The amdsmi-enumeration failures (platform undetected, device_count==0, get_device_name IndexError,
gcn-arch query) are not patched here: one root cause (amdsmi locked out after HIP init), fixed at
interpreter startup by radiance_amdsmi (amdsmi_init before HIP)."""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])


def main():
    # A0. _get_gcn_arch: honor RADIANCE_GFX_ARCH env. amdsmi's asic_info "target_graphics_version"
    #     is empty for gfx1201 so the query raises, and the torch.cuda fallback then crashes at import.
    #     A deterministic env read avoids both. VLLM_ROCM_GCN_ARCH is the pre-0.5.1 name for the same
    #     knob, still accepted; it is no longer the one the image sets, because vLLM warns about
    #     VLLM_*-prefixed variables it does not itself define.
    apply(
        SP / "vllm/platforms/rocm.py",
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '    import os as _os\n'
        '    _env = _os.environ.get("RADIANCE_GFX_ARCH") or _os.environ.get("VLLM_ROCM_GCN_ARCH")\n'
        '    if _env:\n'
        '        return _env\n'
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '_env = _os.environ.get("RADIANCE_GFX_ARCH")',
        "honor RADIANCE_GFX_ARCH env",
    )
if __name__ == "__main__":
    main()
