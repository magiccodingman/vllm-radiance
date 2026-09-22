"""Replay pinned native RoPE and one product-rounding intervention on common inputs."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from compare_execution_modes_d7 import load as load_run
from compare_mode_boundaries_d7 import load

from qwen_r9700_lab.conformance_attention_cut import (
    attention_cut,
    rotary_formula,
    selected_rotary_coefficients,
)
from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, compare_arrays
from qwen_r9700_lab.conformance_precision_intervention import admit_precision_intervention
from qwen_r9700_lab.conformance_rotary_repair import SOURCE_SHA256, patch_source
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def run(args):
    import torch
    import triton
    import vllm.model_executor.layers.rotary_embedding.mrope as native

    torch.set_num_threads(1)
    sides = [load_run(r) for r in (args.eager_run, args.compiled_run)]
    admission = admit_precision_intervention(*sides)
    require(admission["modes"] == ["eager", "compiled-no-graphs"], "wrong replay modes")
    require(admission["captures"] == [True, True], "need observed capture runs")
    require(
        admission["declared_change"]["emulate_precision_casts"] == [False, True],
        "compiled output must use the declared precision-cast intervention",
    )
    roots = [args.eager_capture, args.compiled_capture]
    bridges = [private_json(p) for p in (args.eager_bridge, args.compiled_bridge)]
    source = Path(native.__file__).read_text()
    patched_path = args.output / "mrope_rne.py"
    patched_path.write_text(patch_source(source))
    name = "vllm.model_executor.layers.rotary_embedding._diagnostic_rne_mrope"
    spec = importlib.util.spec_from_file_location(name, patched_path)
    patched = importlib.util.module_from_spec(spec)
    sys.modules[name] = patched
    spec.loader.exec_module(patched)
    artifacts, previous = {}, []

    # Preserve the actual compiled program, not just the Python kernel source.
    def observe(label, module):
        jit = module._triton_mrope_forward
        original = jit.run

        def traced(*positional, **keywords):
            kernel = original(*positional, **keywords)
            if label not in artifacts:
                artifacts[label] = {}
                for stage, value in kernel.asm.items():
                    raw = value if isinstance(value, bytes) else str(value).encode()
                    path = args.output / f"{label}.{stage}"
                    path.write_bytes(raw)
                    artifacts[label][stage] = hashlib.sha256(raw).hexdigest()
            return kernel

        jit.run = traced
        previous.append((jit, original))

    observe("native", native)
    observe("rne_products", patched)
    results, capture_sources = {}, {}

    def tensor(value):
        return torch.from_numpy(value.copy()).view(torch.bfloat16).cuda()

    def array(value):
        return value.detach().cpu().view(torch.int16).numpy()

    try:
        for phase in ("prefill", "decode"):
            manifests = [
                private_json(
                    r / ("prefill-manifest.json" if phase == "prefill" else "manifest.json")
                )
                for r in roots
            ]
            for root, manifest, side, bridge in zip(roots, manifests, sides, bridges, strict=True):
                authenticate(manifest)
                admit_bridge(
                    bridge,
                    side["pass"],
                    private_json(root / "manifest.json"),
                    manifest if phase == "prefill" else None,
                )
            observed, totals = [], {}
            for a, b in zip(manifests[0]["batches"], manifests[1]["batches"], strict=True):
                ma, ta = load(roots[0], a)
                mb, tb = load(roots[1], b)
                positions = ma["positions"]
                require(positions == mb["positions"], "rotary capture positions differ")
                observed += positions
                left, right = attention_cut(ma, ta, 3), attention_cut(mb, tb, 3)
                for kind in ("query", "key"):
                    require(
                        np.array_equal(
                            left[kind + "_after_normalization"],
                            right[kind + "_after_normalization"],
                        ),
                        "rotary isolation requires identical normalized inputs",
                    )
                cosine, sine = selected_rotary_coefficients(mb, tb, 3)
                c = tensor(np.stack([cosine] * 3))
                s = tensor(np.stack([sine] * 3))
                for label, module, rounding, expected_cut in (
                    ("native", native, "rtz", left),
                    ("rne_products", patched, "rne", right),
                ):
                    q = tensor(right["query_after_normalization"]).reshape(len(positions), -1)
                    k = tensor(right["key_after_normalization"]).reshape(len(positions), -1)
                    q, k = module.triton_mrope(q, k, c, s, [11, 11, 10], 256, 64, True, True)
                    for kind, value, heads in (("query", q, 24), ("key", k, 4)):
                        actual = array(value).reshape(len(positions), heads, 256)
                        inputs = right[kind + "_after_normalization"]
                        oracle = rotary_formula(inputs, cosine, sine, rounding)
                        require(
                            np.array_equal(actual, oracle),
                            "native output does not follow declared rounding formula",
                        )
                        require(
                            np.array_equal(actual, expected_cut[kind + "_after_rotation"]),
                            "native replay does not reproduce the corresponding capture",
                        )
                        for other_label, other in (("eager", left), ("compiled_casts", right)):
                            name = f"{label}/{kind}/{other_label}"
                            counts = compare_arrays(
                                [actual], [other[kind + "_after_rotation"]], positions
                            )
                            require(counts is not None, "incomplete native rotary comparison")
                            total = totals.setdefault(name, {})
                            for field, count in counts.items():
                                if field != "different_positions":
                                    total[field] = total.get(field, 0) + count
                        fault = actual.copy()
                        fault.flat[0] ^= 1
                        require(
                            compare_arrays([actual], [fault], positions)["different_elements"] == 1,
                            "one-bit negative control missed",
                        )
                require(
                    np.array_equal(array(c), np.stack([cosine] * 3))
                    and np.array_equal(array(s), np.stack([sine] * 3)),
                    "rotary coefficients were mutated",
                )
            expected = [p for batch in manifests[0]["batches"] for p in batch["positions"]]
            require(
                observed == expected and len(set(observed)) == len(observed), "coverage changed"
            )
            require(len(observed) == (320 if phase == "decode" else 9), "incomplete native domain")
            results[phase], capture_sources[phase] = totals, [m["sha256"] for m in manifests]
    finally:
        for jit, original in previous:
            jit.run = original
        sys.modules.pop(name, None)
    return seal(
        {
            "schema": "qwen.rotary-rounding-isolation.v1",
            "status": "SAMPLE_CHECKED",
            "layer": 3,
            "admission": admission,
            "captures": capture_sources,
            "bridges": [b["sha256"] for b in bridges],
            "results": results,
            "sources": {
                "native_mrope": SOURCE_SHA256,
                "patched_mrope": hashlib.sha256(patched_path.read_bytes()).hexdigest(),
                "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "patcher": hashlib.sha256(
                    Path(sys.modules[patch_source.__module__].__file__).read_bytes()
                ).hexdigest(),
            },
            "compiled_artifacts": artifacts,
            "versions": {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "gpu": torch.cuda.get_device_name(),
            },
            "negative_controls_detected": True,
            "coefficients_unchanged": True,
            "scope": (
                "Layer 3 NeoX RoPE, common normalized Q/K and compiled selected BF16 "
                "coefficients; 320 decode and 9 sampled prefill positions. "
                "No coefficient-selection, multimodal, other-layout, full-model, "
                "performance or universal equivalence claim."
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "compiled-run",
        "eager-run",
        "compiled-capture",
        "eager-capture",
        "compiled-bridge",
        "eager-bridge",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        report = run(args)
    write_private(args.output / "result.json", report)
    print(json.dumps({"sha256": report["sha256"], "status": report["status"]}))


if __name__ == "__main__":
    main()
