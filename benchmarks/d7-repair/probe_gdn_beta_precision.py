"""Isolate beta rounding in one preserved GDN decode transition.

This changes a diagnostic copy of the stock kernel, not the reference contract
or installed backend. It cannot establish which precision a model ought to use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from analyze_gdn_decode_transition import analyze, array, call, stats

from qwen_r9700_lab.conformance_state import read_frame
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

MODULE = "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
SOURCE_SHA256 = "00a3b971b0dbb6ed26e246970a0e1a21a9a174030974685bdd3f0c8ab5fb4bfe"
MODEL_REPORT = "fb28dce4a21a28345ac624de569d55346dfb3254defe5d8c43a5e5412187be47"
KERNEL = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
WRAPPER = "fused_recurrent_gated_delta_rule_packed_decode"
OLD = "beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
NEW = "beta_val = tl.sigmoid(b_val).to(tl.float32)"


def ablation_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("stock GDN source differs from the reviewed preimage")
    text = source.decode()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == KERNEL]
    if len(nodes) != 1:
        raise ValueError("stock packed-decode kernel is ambiguous")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    if body.count(OLD) != 1 or NEW in body:
        raise ValueError("stock beta conversion is ambiguous")
    result = (
        "".join(lines[: node.lineno - 1])
        + body.replace(OLD, NEW)
        + "".join(lines[node.end_lineno :])
    )
    compile(result, "gdn_beta_fp32_diagnostic.py", "exec")
    return result.encode()


def exact_comparison(actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("exact replay requires matching tensor representations")
    return {**stats(actual, expected), "bit_equal": actual.tobytes() == expected.tobytes()}


def classify(rows):
    if set(rows) != {"stock", "stock_repeat", "beta_fp32_only"}:
        return "INCOMPLETE"
    if not all(r["guards_unchanged"] and r["inputs_unchanged"] for r in rows.values()):
        return "INVALID_CONTROL"
    if not all(
        rows[mode][field]["bit_equal"]
        for mode in ("stock", "stock_repeat")
        for field in ("state_to_captured_stock", "output_to_captured_stock")
    ):
        return "INVALID_CONTROL"
    return "DIAGNOSTIC_MEASURED"


def validate_capture(root):
    report = private_json(root / "probe-result.json")
    authenticate(report)
    if report["sha256"] != MODEL_REPORT:
        raise ValueError("GDN ablation requires the reviewed first-transition capture")
    diagnostic = analyze(root)
    if not all(row["equal"] for row in diagnostic["inputs"].values()):
        raise ValueError("cannot isolate beta precision with different transition inputs")
    stock = call(root, "serial", "stock.gdn.packed_decode")
    native = call(root, "d7", "radiance_gdn.recurrent_update")
    before = read_frame(stock / "before")
    manifest = private_json(stock.parent / "calls.json")
    authenticate(manifest)
    recorded = next(
        r for r in manifest["calls"] if r["index"] == int(stock.name.removeprefix("call-"))
    )
    if recorded["before"]["frame"] != before["sha256"]:
        raise ValueError("GDN call frame no longer matches its manifest")
    descriptors = recorded["before"]["descriptors"]
    if (
        before["consumed"] != 2050
        or descriptors["kwargs.scale"]["scalar"] != 128**-0.5
        or descriptors["kwargs.use_qk_l2norm_in_kernel"]["scalar"] is not True
    ):
        raise ValueError("unreviewed packed-decode scalar parameters")
    expected = {
        "mixed_qkv": ([1, 10240], "bf16"),
        "a": ([1, 48], "bf16"),
        "b": ([1, 48], "bf16"),
        "A_log": ([48], "<f4"),
        "dt_bias": ([48], "bf16"),
        "initial_state.selected_values": ([1, 48, 128, 128], "<f4"),
    }
    for name, (shape, dtype) in expected.items():
        descriptor = before["components"]["kwargs." + name]
        if descriptor["shape"] != shape or descriptor["dtype"] != dtype:
            raise ValueError("unreviewed packed-decode tensor layout")
    return stock, native, diagnostic


def replay(root, candidate_path):
    # Importing native libraries and allocating GPU tensors happens only after
    # acquiring the campaign lease in main().
    import torch

    stock_call, native_call, diagnostic = validate_capture(root)
    stock = importlib.import_module(MODULE)
    if hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("loaded stock kernel differs from the captured source")
    spec = importlib.util.spec_from_file_location(MODULE + "_qwen_beta_diagnostic", candidate_path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    inputs = {}
    for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias"):
        value = array(stock_call / "before", "kwargs." + name)
        dtype = torch.float32 if name == "A_log" else torch.bfloat16
        inputs[name] = torch.from_numpy(value.copy()).to(device="cuda", dtype=dtype)
    initial = array(stock_call / "before", "kwargs.initial_state.selected_values", 0).astype(
        np.float32
    )
    captured_stock_state = array(
        stock_call / "after", "kwargs.initial_state.selected_values", 0
    ).astype(np.float32)
    captured_stock_output = array(stock_call / "after", "kwargs.out")[0, 0].astype(np.float32)
    mapping = diagnostic["selected_storage_metadata"]
    output_slot = mapping["d7_indices"].index(mapping["d7_mapping"][0][0])
    captured_native_state = array(
        native_call / "after", "args.7.selected_values", output_slot
    ).astype(np.float32)
    captured_native_output = array(native_call / "after", "args.8")[0].astype(np.float32)
    rows = {}
    for name, module in (("stock", stock), ("stock_repeat", stock), ("beta_fp32_only", candidate)):
        args = {k: v.clone() for k, v in inputs.items()}
        pool = torch.full((3, 48, 128, 128), 17.0, dtype=torch.float32, device="cuda")
        pool[1].copy_(torch.from_numpy(initial).to("cuda"))
        out = torch.full((1, 1, 48, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
        indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        result = getattr(module, WRAPPER)(
            **args,
            scale=128**-0.5,
            initial_state=pool,
            out=out,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        if result[0] is not out or result[1] is not pool:
            raise ValueError("packed decode changed its in-place return convention")
        actual_state = pool[1].cpu().numpy()
        actual_output = out[0, 0].float().cpu().numpy()
        rows[name] = {
            "guards_unchanged": bool((pool[[0, 2]] == 17).all()),
            "inputs_unchanged": all(torch.equal(args[k], v) for k, v in inputs.items())
            and indices.tolist() == [1],
            "state_to_captured_stock": exact_comparison(actual_state, captured_stock_state),
            "output_to_captured_stock": exact_comparison(actual_output, captured_stock_output),
            "state_to_captured_r4d": exact_comparison(actual_state, captured_native_state),
            "output_to_captured_r4d": exact_comparison(actual_output, captured_native_output),
        }
    return seal(
        {
            "status": classify(rows),
            "scope": (
                "One preserved layer-zero decode transition; stock beta precision ablation only."
            ),
            "capture_report": MODEL_REPORT,
            "source": SOURCE_SHA256,
            "candidate": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "rows": rows,
            "installed_sources_modified": False,
            "reference_contract_changed": False,
            "repair_qualified": False,
        }
    )


def main(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    if not args.allow_gpu:
        raise ValueError("native diagnostic requires explicit --allow-gpu")
    os.umask(0o077)
    candidate = ablation_source(args.source.read_bytes())
    validate_capture(args.capture)
    args.output.mkdir(mode=0o700)
    path = args.output / "gdn_beta_fp32_diagnostic.py"
    path.write_bytes(candidate)
    with gpu_lease(args.output / "gpu-lease"):
        result = replay(args.capture, path)
    write_private(args.output / "probe-result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "DIAGNOSTIC_MEASURED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true", required=True)
    raise SystemExit(main(parser.parse_args()))
