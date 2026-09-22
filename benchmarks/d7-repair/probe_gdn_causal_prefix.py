"""Check native GDN causality under future-gate changes and partial chunks.

Synthetic inputs only. Requires an explicit GPU flag and a production-profile.json
containing the expected r4d.so hash. Exact prefix equality is checked separately
from diagnostic FP64 error. A discrepancy exits 2 after preserving its evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import private_json, seal, write_private

CASES = ("full", "repeat", "prefix32", "prefix48", "future_gates", "future_values")
COMPARISONS = CASES[1:]


def outcome(cases, comparisons):
    """Reject incomplete or invalid controls rather than treating them as passes."""
    if (
        [row.get("name") for row in cases] != list(CASES)
        or [row.get("other") for row in comparisons] != list(COMPARISONS)
        or not all(
            all(
                row.get(key) is True
                for key in ("finite", "guards_intact", "initial_state_unchanged")
            )
            for row in cases
        )
        or not all(
            all(
                row.get(key) is True
                for key in (
                    "identical_input_prefix",
                    "identical_initial_state",
                    "oracle_prefix_exact",
                )
            )
            for row in comparisons
        )
    ):
        return "INVALID_CONTROL"
    # Repeatability and unchanged-future-values are independent controls. A
    # failure still matters, but cannot isolate gate/partition arithmetic.
    controls = {row["other"]: row for row in comparisons}
    if not all(
        controls[name].get("output_prefix_exact_bytes") is True
        for name in ("repeat", "future_values")
    ):
        return "CONTROL_DISCREPANCY"
    if any(row.get("output_prefix_exact_bytes") is not True for row in comparisons):
        return "CAUSAL_PREFIX_DISCREPANCY"
    return "TESTED"


def probe(root: Path):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    # Torch loads this image's bundled ROCm dependencies. Loading r4d first
    # fails on libamdhip64.so.7 before the probe can execute any kernel.
    native = importlib.import_module("radiance_gdn")
    r4d = importlib.import_module("r4d")
    expected = private_json(root / "production-profile.json")["kernel_hashes"]["r4d.so"]
    observed = hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest()
    if observed != expected or not native.ENABLED:
        raise RuntimeError("native GDN binary does not match the declared profile")
    helper = Path(__file__).with_name("probe_gdn_numerics.py")
    spec = importlib.util.spec_from_file_location("independent_gdn_reference", helper)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(1536)
    heads, query_heads, width, prefix = 48, 16, 128, 32
    scale = width**-0.5
    q = torch.nn.functional.normalize(torch.randn(64, query_heads, width), dim=-1).to(
        torch.bfloat16
    )
    k = torch.nn.functional.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(
        torch.bfloat16
    )
    v = torch.randn(64, heads, width).to(torch.bfloat16)
    beta = torch.full((64, heads), 0.5, dtype=torch.float32)
    initial = torch.randn(1, heads, width, width, dtype=torch.float32) * 0.01
    steps = torch.full((64, heads), -0.02, dtype=torch.float32)
    binding = seal(
        {
            "r4d_so": observed,
            "radiance_gdn": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "helper": hashlib.sha256(helper.read_bytes()).hexdigest(),
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "installed_source_modified": False,
            "maximum_decay_span_below_repair_threshold": True,
        }
    )
    write_private(root / "binding.json", binding)
    observations, summaries = {}, []
    for name in CASES:
        n = 32 if name == "prefix32" else 48 if name == "prefix48" else 64
        ss, vv = steps[:n].clone(), v[:n].clone()
        if name == "future_gates":
            ss[prefix:] = -0.07
        if name == "future_values":
            vv[prefix:] *= 3
        cumulative = torch.from_numpy(np.cumsum(ss.double().numpy(), axis=0).astype(np.float32))
        qq, kk, vg, bg, g = [
            t.contiguous().cuda() for t in (q[:n], k[:n], vv, beta[:n], cumulative)
        ]
        h0 = initial.clone().cuda()
        cu = torch.tensor([0, n], device="cuda", dtype=torch.int32)
        matrix = native.kkt_solve(kk, bg, g, cu, 1, n, heads, query_heads)
        count = vg.numel()
        slab = torch.full((count + 512,), 37.0, device="cuda", dtype=torch.bfloat16)
        output = slab[256 : 256 + count].view(1, n, heads, width)
        y, final = native.fused_prefill(
            qq[None],
            kk[None],
            vg[None],
            matrix[None],
            g[None],
            bg[None],
            scale,
            h0,
            True,
            cu,
            None,
            out=output,
        )
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all() and torch.isfinite(final).all())
        guards = bool((slab[:256] == 37).all() and (slab[-256:] == 37).all())
        initial_unchanged = torch.equal(h0.cpu(), initial)
        mask = torch.tril(torch.ones(prefix, prefix, dtype=torch.bool, device="cuda"))
        lower = matrix[:prefix, :, :prefix].permute(1, 0, 2)[:, mask].cpu().contiguous()
        yy, ff = y[0].cpu().clone(), final.cpu().clone()
        oracle, oracle_final = reference.sequential_reference(
            q[:n], k[:n], vv, ss, beta[:n], initial, [0, n], scale
        )
        data = {
            "q": q[:n].contiguous(),
            "k": k[:n].contiguous(),
            "v": vv,
            "steps": ss,
            "cumulative": cumulative,
            "beta": beta[:n].contiguous(),
            "initial": initial,
            "output": yy,
            "final": ff,
            "matrix_prefix_lower": lower,
            "oracle": oracle,
            "oracle_final": oracle_final,
        }
        path = root / (name + ".safetensors")
        if path.exists():
            raise FileExistsError("refusing to overwrite prior probe evidence")
        save_file(data, path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        observations[name] = data
        row = {
            "name": name,
            "tokens": n,
            "capsule_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_relative_l2_vs_fp64": float((yy.double() - oracle).norm() / oracle.norm())
            if finite
            else None,
            "state_relative_l2_vs_fp64": float(
                (ff.double() - oracle_final).norm() / oracle_final.norm()
            )
            if finite
            else None,
            "guards_intact": guards,
            "initial_state_unchanged": initial_unchanged,
            "finite": finite,
        }
        summaries.append(row)
        write_private(root / (name + ".json"), seal(row))
    base, comparisons = observations["full"], []
    for name in COMPARISONS:
        data = observations[name]
        a, b = base["output"][:prefix].contiguous(), data["output"][:prefix].contiguous()
        different = a != b
        positions = torch.nonzero(different.any(-1).any(-1)).flatten().tolist()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        comparisons.append(
            {
                "other": name,
                "identical_input_prefix": all(
                    torch.equal(base[key][:prefix], data[key][:prefix])
                    for key in ("q", "k", "v", "beta", "steps", "cumulative")
                ),
                "identical_initial_state": torch.equal(base["initial"], data["initial"]),
                "oracle_prefix_exact": torch.equal(
                    base["oracle"][:prefix], data["oracle"][:prefix]
                ),
                "kkt_prefix_lower_exact": torch.equal(
                    base["matrix_prefix_lower"], data["matrix_prefix_lower"]
                ),
                "output_prefix_exact_bytes": torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                "different_values": int(different.sum()),
                "first_different_position": positions[0] if positions else None,
                "max_abs": float((a.float() - b.float()).abs().max()) if finite else None,
                "relative_l2": float((a.double() - b.double()).norm() / a.double().norm())
                if finite
                else None,
            }
        )
    result = seal(
        {
            "status": outcome(summaries, comparisons),
            "binding": binding["sha256"],
            "comparisons": comparisons,
            "cases": summaries,
            "scope": (
                "Six synthetic GDN calls; exact prefix invariance is distinct from FP64 error. "
                "No model-level loop rate or arbitrary-input theorem."
            ),
            "gpu_backend_unchanged": True,
        }
    )
    write_private(root / "probe-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("explicit --allow-gpu is required")
    os.umask(0o077)
    result = probe(args.root)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "TESTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
