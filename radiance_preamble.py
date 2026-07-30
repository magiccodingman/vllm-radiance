#!/usr/bin/env python3
"""RADIANCE startup preamble: print the banner and run environment prechecks before
vLLM boots. Launched once by radiance_entrypoint.sh in the container's main process (not per
TP worker), which then exec's `vllm serve`. Every check is best-effort: a failure prints a
warning, never blocks the serve. The GPU topology/bandwidth sweep (rocm-bandwidth-test, on by
default via RADIANCE_RUN_BWTEST=1 in the image) runs in the background from the entrypoint; its
report follows a few seconds later.

Env knobs: NO_COLOR / RADIANCE_BANNER_PLAIN=1 disable ANSI color."""
import glob
import os
import sys
import importlib.metadata as md

# ------------------------------------------------------------------ color
_PLAIN = bool(os.environ.get("NO_COLOR")) or os.environ.get("RADIANCE_BANNER_PLAIN") == "1"


def c(code, s):
    return s if _PLAIN else f"\033[{code}m{s}\033[0m"


ACCENT = "38;5;39"   # radiance cyan-blue
DIM = "2"


def hdr(title):
    line = "─" * 3
    print("\n" + c(ACCENT + ";1", f"{line}[ {title} ]{'─' * max(4, 58 - len(title))}"))


def ok(s):   return c("32;1", s)
def bad(s):  return c("31;1", s)
def warn(s): return c("33;1", s)
def dim(s):  return c(DIM, s)


BANNER = r"""
██████╗  █████╗ ██████╗ ██╗ █████╗ ███╗   ██╗ ██████╗███████╗
██╔══██╗██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║██╔════╝██╔════╝
██████╔╝███████║██║  ██║██║███████║██╔██╗ ██║██║     █████╗
██╔══██╗██╔══██║██║  ██║██║██╔══██║██║╚██╗██║██║     ██╔══╝
██║  ██║██║  ██║██████╔╝██║██║  ██║██║ ╚████║╚██████╗███████╗
╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝"""


def _v(dist):
    try:
        return md.version(dist)
    except Exception:
        return None


def print_banner():
    ver = os.environ.get("RADIANCE_VERSION", "?")
    for ln in BANNER.splitlines():
        print(c(ACCENT + ";1", ln))
    print(dim("  vLLM inference server · AMD Radeon AI PRO R9700 · gfx1201 / RDNA4"))
    print(dim(f"  radiance ") + c(ACCENT, f"v{ver}"))


# ------------------------------------------------------------------ GPUs
def section_gpus():
    hdr("GPUs")
    try:
        import torch
    except Exception as e:
        print("  " + bad(f"torch import failed: {e!r}; cannot enumerate GPUs"))
        return
    try:
        n = torch.cuda.device_count()
    except Exception as e:
        print("  " + bad(f"device enumeration failed: {e!r}"))
        return

    print(f"  detected : {ok(str(n))} GPU(s)" + dim("  (HIP_VISIBLE_DEVICES="
          + os.environ.get("HIP_VISIBLE_DEVICES", "unset") + ")"))
    if n == 0:
        print("  " + bad("no GPUs visible; vLLM will not start"))
        return

    want = (os.environ.get("RADIANCE_GFX_ARCH")
            or os.environ.get("VLLM_ROCM_GCN_ARCH")   # pre-0.5.1 name
            or "gfx1201")
    good = 0
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            arch = getattr(p, "gcnArchName", "?").split(":")[0]
            gib = getattr(p, "total_memory", 0) / (1024 ** 3)
            match = arch == want
            good += match
            tag = ok(f"✓ {want}") if match else bad(f"✗ {arch} (expected {want})")
            print(f"  [{i}] {p.name:<26} {arch:<9} {gib:5.1f} GiB   {tag}")
        except Exception as e:
            print(f"  [{i}] " + bad(f"props failed: {e!r}"))

    verdict = ok("PASS") if good == n else bad("FAIL")
    print(f"  arch check : {verdict} " + dim(f"({good}/{n} {want})"))

    # P2P: bidirectional peer access across every ordered pair
    if n >= 2:
        try:
            pairs, enabled = [], True
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    a = torch.cuda.can_device_access_peer(i, j)
                    enabled = enabled and a
                    if i < j:
                        b = torch.cuda.can_device_access_peer(j, i)
                        pairs.append(f"{i}↔{j} " + (ok("✓") if (a and b) else bad("✗")))
            state = ok("ENABLED") if enabled else warn("DISABLED (RCCL fallback)")
            print(f"  P2P access : {state}   " + "  ".join(pairs))
        except Exception as e:
            print("  P2P access : " + warn(f"probe failed: {e!r}"))


# ------------------------------------------------------------------ radiance optimizations
def _val(name, default):
    return os.environ.get(name, default)


def section_opts():
    hdr("RADIANCE optimizations")

    print("  " + dim("feature toggles (set to 0 to disable):"))
    for name, dflt, desc in [
        ("RADIANCE_PRESHUFFLE",     "1", "preshuffled AITER FP8 blockscale GEMM"),
        ("RADIANCE_ATTN_TUNE",      "1", "RDNA4 unified-attention tiling (fp8 + bf16/auto KV)"),
        ("RADIANCE_GDN_WMMA",       "1", "gated-delta-net KKt gram + triangular solve on the fp16 matrix cores (vs fp32 scalar path)"),
        ("RADIANCE_VIT_FLASH",      "1", "native head_dim-72 flash attention for the vision encoder (multimodal)"),
        ("RADIANCE_FUSE_RMS_QUANT", "1", "fold group-FP8 quant into the RMSNorm epilogue"),
        ("RADIANCE_FAST_REDUCE",    "1", "P2P one-shot all-reduce for TP=2, byte-identical to RCCL"),
        ("RADIANCE_AR_QUANT",       "1", "fp8 all-reduce payload for large messages (on; not RCCL-identical)"),
        ("RADIANCE_DYNAMIC_DRAFT",  "1", "per-request MTP draft-depth controller (needs speculative mtp)"),
        ("RADIANCE_MOE_ROUTER",     "1", "custom bf16 MoE-gate GEMM for the n in [6,16] band (fine-grained MoE, e.g. Qwen3.6-35B-A3B)"),
    ]:
        badge = ok("ON ") if _val(name, dflt) == "1" else warn("OFF")
        print(f"    {badge} {name:<24} " + dim(desc))

    print("\n  " + dim("dynamic drafting (RADIANCE_DYNAMIC_DRAFT):"))
    if _val("RADIANCE_DYNAMIC_DRAFT", "1") == "1":
        print("        " + dim("per-slot confidence gate: draft while cum. confidence >= TAU, take a free-win"))
        print("        " + dim("n-gram, else verify; batch schedule caps serial MTP forwards at concurrency."))
        for name, dflt, desc in [
            ("RADIANCE_DRAFT_SCHEDULE",   "1:8,2:7,4:6,8:5,16:4", "batch-size MTP-forward ceiling (bs:max_depth, carry-forward)"),
            ("RADIANCE_DRAFT_TAU",        "0.35", "confidence-product stop threshold"),
        ]:
            print(f"        {dim('·')} {name} = {c(ACCENT, _val(name, dflt))}  " + dim(desc))
    else:
        print("        " + dim("off (stock MTP)"))

    print("\n  " + dim("all-reduce size gates (when RADIANCE_FAST_REDUCE=1):"))
    ar = [("RADIANCE_AR_MAX_KB", "32768"), ("RADIANCE_AR_QUANT_MIN_KB", "128")]
    print("        " + "  ".join(f"{n}={c(ACCENT, _val(n, d))}" for n, d in ar))

    print("\n  " + dim("NUMA binding (RADIANCE_NUMA_BIND / --numa-bind):"))
    active = os.environ.get("RADIANCE_NUMA_ACTIVE", "")
    spec = os.environ.get("RADIANCE_NUMA_BIND", "")
    if active:
        print("        " + ok(active))
    elif spec:
        print("        " + warn(f"requested '{spec}' but inactive (unbound)"))
    else:
        print("        " + dim("off (set =auto or pass --numa-bind to pin to the GPU-local node)"))

    print("\n  " + dim("startup:"))
    for name, dflt, desc in [
        ("RADIANCE_RUN_BWTEST",     "1", "GPU topology + bandwidth sweep at startup (on; backgrounded, ~1 s; set 0 to skip)"),
        ("RADIANCE_BWTEST_TIMEOUT", "150", "bandwidth sweep timeout, seconds"),
        ("RADIANCE_BANNER_PLAIN",   "0", "disable ANSI color (also NO_COLOR)"),
    ]:
        print(f"        {dim('·')} {name} = {c(ACCENT, _val(name, dflt))}  " + dim(desc))

    print("\n  " + dim("baked-in (always on; correctness + GEMM path):"))
    baked = [
        "block-FP8 GEMM dispatcher; preshuffle / AITER split-K / generic + tuned fp8-configs",
        "tuned fused-MoE Triton configs for fine-grained MoE (Qwen3.6-35B-A3B E=256,N=256; ~-12% TTFT prefill)",
        "bf16 / auto KV cache attention fit + tune (fits the RDNA4 64 KiB LDS at head_size 256)",
        "amdsmi gfx1201 GPU enumeration fix (device_count / platform detect)",
        "AITER enablement for gfx12x (upstream gates it to MI3xx / CDNA)",
        "native sampler fallback (AITER top-k/top-p kernel absent on RDNA4)",
        "MTP drafter unpad fix (enables disable_padded_drafter_batch single-stream path)",
        "tool-parser streaming vs non-streaming consistency (vLLM #47137)",
        "from_json Jinja filter for tool-calling chat templates",
    ]
    for b in baked:
        print(f"    {ok('•')} " + b)

    # AITER routing env that shapes which kernels run
    print("\n  " + dim("AITER routing (VLLM_ROCM_USE_AITER*):"))
    aiter_env = ["VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION",
                 "VLLM_ROCM_USE_AITER_MHA", "VLLM_ROCM_USE_AITER_MLA",
                 "VLLM_ROCM_USE_AITER_MOE", "VLLM_ROCM_USE_AITER_LINEAR",
                 "VLLM_ROCM_USE_AITER_RMSNORM", "VLLM_ROCM_USE_AITER_FP8BMM"]
    cells = []
    for e in aiter_env:
        v = os.environ.get(e)
        short = e.replace("VLLM_ROCM_USE_AITER", "AITER").replace("_UNIFIED_ATTENTION", "_UATTN")
        if v is None:
            cells.append(dim(f"{short}=-"))
        else:
            cells.append((ok if v == "1" else dim)(f"{short}={v}"))
    print("    " + "  ".join(cells))


# ------------------------------------------------------------------ versions
def _rocm_version():
    """ROCm userspace version. The layout moved in 7.14: the version file now lives under the
    rocm-core component dir (/opt/rocm/core-<ver>/.info/version, reachable via the `core`
    alternatives symlink) instead of directly under ROCM_PATH."""
    root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    cands = [os.path.join(root, ".info", "version"),
             os.path.join(root, "core", ".info", "version"),
             os.path.join(root, ".info", "version-dev")]
    cands += sorted(glob.glob(os.path.join(root, "core-*", ".info", "version")), reverse=True)
    for cand in cands:
        try:
            with open(cand) as f:
                v = f.read().strip()
            if v:
                return v
        except Exception:
            pass
    return _v("rocm-sdk-core") or _v("rocm-sdk") or None


def section_versions():
    hdr("component versions")
    import platform
    rows = [("radiance", os.environ.get("RADIANCE_VERSION", "?")),
            ("vllm", _v("vllm"))]
    hip = None
    try:
        import torch
        rows.append(("torch", torch.__version__))
        hip = torch.version.hip
    except Exception:
        rows.append(("torch", None))
    rows.append(("torchvision", _v("torchvision")))
    try:
        import triton
        rows.append(("triton", triton.__version__))
    except Exception:
        rows.append(("triton", _v("triton")))
    rows += [("aiter", _v("amd-aiter")),
             ("transformers", _v("transformers")),
             ("amdsmi", _v("amdsmi")),
             ("hip", hip),
             ("rocm", _rocm_version()),
             ("python", platform.python_version())]
    for name, val in rows:
        cell = c(ACCENT, str(val)) if val not in (None, "") else dim("not detected")
        print(f"    {name:<13} {cell}")


def main():
    print_banner()
    for fn in (section_gpus, section_opts, section_versions):
        try:
            fn()
        except Exception as e:
            print("  " + bad(f"{fn.__name__} failed: {e!r}"))
    if os.environ.get("RADIANCE_RUN_BWTEST", "0") == "1":
        print("\n  " + dim("GPU topology + bandwidth sweep runs in the background; report follows below."))
    print(c(ACCENT + ";1", "─" * 62) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
