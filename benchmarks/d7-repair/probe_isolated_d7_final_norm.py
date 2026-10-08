"""Replay the original compiled final norm on correct Pi inputs, then its head.

The original compiler fused a retained residual addition into the final norm.
Its cut therefore includes that addition's two original BF16 inputs, not only
their already-rounded sum. Uses the preserved generated kernel, not a newly
written approximation of it. All downstream heads use serial reference M1.
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

from probe_isolated_d7_head import file_digest, partial_progress, tensor_digest

from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.conformance_topk import compare_rows, require, summarize_logits
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private

OLD_SOURCE_SHA = "643e0d0306c2a7e710454ebd87ef03073060b797bc0629ae8888457152c7e933"
KERNEL = "triton_red_fused__to_copy_add_fused_add_rms_norm_3"


def cuts(metadata):
    authenticate(metadata)
    name = "qwen_d7_qualified.gemma_residual.default"
    final = [
        e
        for e in metadata["events"]
        if e["operation"] == name
        and any(
            re.fullmatch(r"(?:language_model\.)?model\.norm", x)
            for x in e.get("logical_identities", ())
        )
    ]
    carry = [
        e
        for e in metadata["events"]
        if e["operation"] == name
        and any(
            re.fullmatch(r"(?:language_model\.)?model\.layers\.63\.post_attention_layernorm", x)
            for x in e.get("logical_identities", ())
        )
    ]
    require(len(final) == len(carry) == 1, "final fused residual/norm boundary is ambiguous")
    require(carry[0]["index"] < final[0]["index"], "residual producer must precede final norm")
    return final[0], carry[0]


def run(args):
    spec = private_json(args.spec)
    os.environ.update(spec["environment"])
    import torch
    from safetensors import safe_open
    from stock_m1_norm import StockM1Norm
    from vllm.ir.ops import fused_add_rms_norm
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm

    from qwen_r9700_lab.conformance_radiance import verify_sources

    verify_sources(Path(importlib.util.find_spec("vllm").origin).parent.parent, spec["binding"])
    require(
        file_digest(args.old_source) == OLD_SOURCE_SHA, "original compiled kernel source changed"
    )
    module_spec = importlib.util.spec_from_file_location("d7_original_final_norm", args.old_source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    old = getattr(module, KERNEL)
    fixed = StockM1Norm(args.norm_build)
    native = fused_add_rms_norm.impls["native"].impl_fn
    torch.set_num_threads(2)
    manifest, reference = [
        private_json(p) for p in (args.captures / "manifest.json", args.reference)
    ]
    for value in (manifest, reference):
        authenticate(value)
    require(manifest["positions"] == len(reference["rows"]) == 320, "320 positions required")
    require(len(manifest["batches"]) == 40, "40 complete groups required")
    model = Path(spec["native_config"]["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(str(model / index[name]), framework="pt") as f:
            return f.get_tensor(name).to("cuda")

    head_weight, norm_weight = weight("lm_head.weight"), weight("model.language_model.norm.weight")
    weights_before = [tensor_digest(t) for t in (head_weight, norm_weight)]
    mlp, attention, previous = [
        torch.zeros((8, 5120), device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    old_hidden = torch.empty_like(mlp)

    def head(hidden):
        return torch.cat(
            [rocm_unquantized_gemm(None, row[None], head_weight, None) for row in hidden]
        )

    def serial():
        carry = (attention.float() + previous.float()).bfloat16()
        hidden = torch.cat(
            [
                native(mlp[i : i + 1], carry[i : i + 1], norm_weight.float() + 1, 1e-6)[0]
                for i in range(8)
            ]
        )
        return head(hidden), hidden, carry

    def original():
        old_hidden.copy_(mlp)
        old.run(
            old_hidden,
            attention,
            previous,
            norm_weight,
            8,
            5120,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        return head(old_hidden), old_hidden

    def repaired():
        carry = (attention.float() + previous.float()).bfloat16()
        hidden, _ = fixed(mlp, carry, norm_weight, 1e-6)
        return head(hidden), hidden

    graphs = {}
    for name, fn in (("reference", serial), ("old", original), ("fixed", repaired)):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        graphs[name] = graph, result
    sources = {
        "capture": manifest["sha256"],
        "reference": reference["sha256"],
        "norm_build": fixed.manifest["sha256"],
        "old_source": OLD_SOURCE_SHA,
        "probe": file_digest(Path(__file__)),
        "binding": spec["binding"]["sha256"],
    }
    records = {"old": [], "fixed": []}
    hidden_exact = {"old": 0, "fixed": 0}
    for n, batch in enumerate(manifest["batches"]):
        path = args.captures / batch["file"]
        metadata = private_json(path.with_suffix(".json"))
        authenticate(metadata)
        require(
            metadata["sha256"] == batch["sha256"]
            and file_digest(path) == metadata["tensor_sha256"],
            "capture changed",
        )
        final, post = cuts(metadata)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

        def array(event, suffix, values=saved):
            return values[str(event["index"]) + suffix].clone()

        arrays = [
            array(final, ".before.args.0"),
            array(post, ".before.args.0"),
            array(post, ".before.args.1"),
        ]
        expected_hidden = array(final, ".after.result.0")
        expected_carry = array(final, ".before.args.1")
        del saved, array
        require(
            all(t.shape == (8, 5120) and t.dtype == torch.bfloat16 for t in arrays),
            "fused final norm requires the exact captured M8 BF16 inputs",
        )
        for target, value in zip((mlp, attention, previous), arrays, strict=True):
            target.copy_(value)
        for graph, _ in graphs.values():
            graph.replay()
        require(
            all(
                torch.equal(t.cpu().view(torch.uint8), v.view(torch.uint8))
                for t, v in zip((mlp, attention, previous), arrays, strict=True)
            ),
            "isolated input changed",
        )
        require(
            torch.equal(
                graphs["reference"][1][1].cpu().view(torch.uint8),
                expected_hidden.view(torch.uint8),
            ),
            "serial final norm does not reproduce reference hidden rows",
        )
        require(
            torch.equal(
                graphs["reference"][1][2].cpu().view(torch.uint8),
                expected_carry.view(torch.uint8),
            ),
            "expanded residual cut does not reproduce its reference carry",
        )
        expected = graphs["reference"][1][0].float().cpu().numpy()
        summaries = [summarize_logits(row) for row in expected]
        rows = reference["rows"][n * 8 : n * 8 + 8]
        require(batch["positions"] == [r["absolute_position"] for r in rows], "position mismatch")
        require(
            summaries == [r["logits"] for r in rows],
            "reference remainder does not reproduce in-model logits",
        )
        for arm in records:
            logits, hidden = graphs[arm][1]
            records[arm] += [
                compare_rows(s, summarize_logits(r))
                for s, r in zip(summaries, logits.float().cpu().numpy(), strict=True)
            ]
            hidden_exact[arm] += int(
                (hidden.cpu().view(torch.uint8) == expected_hidden.view(torch.uint8))
                .all(dim=1)
                .sum()
            )
        replace_private(args.output, "progress.json", partial_progress(records, sources))
    require(
        [tensor_digest(t) for t in (head_weight, norm_weight)] == weights_before,
        "checkpoint weight modified",
    )
    import numpy as np

    fault = expected[0].copy()
    fault.view(np.uint32)[0] ^= 1
    require(
        not compare_rows(summaries[0], summarize_logits(fault))["full_logits_exact"],
        "injected fault missed",
    )
    return seal(
        {
            "schema": "qwen.isolated-d7-final-norm.v1",
            "status": "SAMPLE_CHECKED",
            "sources": sources,
            "results": partial_progress(records, sources)["results"],
            "hidden_exact_positions": hidden_exact,
            "reference_remainder_checked": 320,
            "weights_unchanged": True,
            "negative_control_detected": True,
            "scope": (
                "Original compiled final normalization including retained residual sum; "
                "fixed compiled norm; 320 correct Pi positions, common serial reference head. "
                "Not other layers or universal equivalence."
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "captures", "reference", "norm-build", "old-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = run(args)
    write_private(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "results": result["results"]}))


if __name__ == "__main__":
    main()
