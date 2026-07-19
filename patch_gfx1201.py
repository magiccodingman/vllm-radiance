#!/usr/bin/env python3
"""gfx1201 (R9700 / RDNA4) logic patches for vLLM on ROCm. Idempotent string replacement on the
installed site-packages copies; re-running is safe.

The amdsmi-enumeration failures (platform undetected, device_count==0, get_device_name IndexError,
gcn-arch query) are not patched here: one root cause (amdsmi locked out after HIP init), fixed at
interpreter startup by radiance_amdsmi (amdsmi_init before HIP)."""
import ast
import sysconfig
from pathlib import Path

SP = Path(sysconfig.get_paths()["purelib"])


def apply(path, anchor, new, sentinel, label):
    """Idempotent one-shot source patch: replace the unique `anchor` with `new` in `path`. Skips if
    `sentinel` is already present; a missing file or non-unique anchor is fatal."""
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    n = s.count(anchor)
    if n != 1:
        raise SystemExit(f"  FAIL  {label}: anchor matched {n}x, expected 1 ({path})")
    s = s.replace(anchor, new, 1)
    ast.parse(s)  # never write a file that would not parse
    path.write_text(s)
    print(f"  OK    {label}")


def main():
    # A0. _get_gcn_arch: honor VLLM_ROCM_GCN_ARCH env. amdsmi's asic_info "target_graphics_version"
    #     is empty for gfx1201 so the query raises, and the torch.cuda fallback then crashes at import.
    #     A deterministic env read avoids both.
    apply(
        SP / "vllm/platforms/rocm.py",
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '    import os as _os\n'
        '    _env = _os.environ.get("VLLM_ROCM_GCN_ARCH")\n'
        '    if _env:\n'
        '        return _env\n'
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '_env = _os.environ.get("VLLM_ROCM_GCN_ARCH")',
        "honor VLLM_ROCM_GCN_ARCH env",
    )
    # A. AITER enablement: vLLM gates AITER on MI3xx; treat gfx12x as capable too.
    apply(
        SP / "vllm/_aiter_ops.py",
        "        from vllm.platforms.rocm import on_mi3xx\n\n        return on_mi3xx()",
        "        from vllm.platforms.rocm import on_gfx12x, on_mi3xx\n\n"
        "        return on_mi3xx() or on_gfx12x()",
        "on_mi3xx() or on_gfx12x()",
        "is_aiter_found_and_supported: allow gfx12x",
    )
    # B. Triton HIPDriver.is_active(): stock gates on torch.cuda.is_available(), which is False in
    #    vLLM's GPU-less inspection subprocess where aiter touches the driver at import. A ROCm torch
    #    build always targets HIP, so gate on torch.version.hip.
    apply(
        SP / "triton/backends/amd/driver.py",
        "            return torch.cuda.is_available() and (torch.version.hip is not None)",
        "            return torch.version.hip is not None",
        "            return torch.version.hip is not None",
        "Triton HIPDriver.is_active: gate on torch.version.hip",
    )
    # C. AITER sampler gate: VLLM_ROCM_USE_AITER=1 also selects AITER's top-k/top-p sampler, whose
    #    C++/HIP kernel fails to build on RDNA4. Gate to MI3xx; gfx12x uses the native sampler.
    apply(
        SP / "vllm/v1/sample/ops/topk_topp_sampler.py",
        '            logprobs_mode not in ("processed_logits", "processed_logprobs")\n'
        "            and rocm_aiter_ops.is_enabled()\n"
        "        ):",
        '            logprobs_mode not in ("processed_logits", "processed_logprobs")\n'
        "            and rocm_aiter_ops.is_enabled()\n"
        "            # gfx1201: AITER's sampler C++/HIP kernel fails to build on RDNA4.\n"
        "            # Gate to MI3xx; gfx12x uses the native sampler.\n"
        '            and __import__("vllm.platforms.rocm", fromlist=["on_mi3xx"]).on_mi3xx()\n'
        "        ):",
        "AITER's sampler C++/HIP kernel fails to build on RDNA4",
        "topk_topp_sampler: gate AITER sampler to MI3xx",
    )


if __name__ == "__main__":
    main()
