"""Compare actual old/fixed M8 heads on 320 correct compiled Pi hidden rows.

All three implementations execute captured GPU graphs. The serial M1 result
must reproduce the saved in-model logits before a sample can be counted.
This is a head-only correctness comparison, not a throughput measurement.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.conformance_topk import compare_rows, require, summarize_logits
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def tensor_digest(value, chunk_bytes=1024 * 1024):
    """Hash contiguous tensor storage with bounded host copies, including BF16."""
    import torch

    require(value.is_contiguous(), "weight digest requires contiguous storage")
    require(chunk_bytes > 0, "digest chunk must be positive")
    raw = value.detach().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), chunk_bytes):
        chunk = raw[start : start + chunk_bytes].to(device="cpu", copy=True)
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def partial_progress(records, sources):
    """Persist useful counts without certifying pending weight/fault checks."""
    require(set(records) == {"old", "fixed"}, "both head arms required")
    positions = len(records["old"])
    require(positions == len(records["fixed"]), "head arms have different coverage")
    require(0 <= positions <= 320 and positions % 8 == 0, "incomplete M8 group")
    return seal(
        {
            "schema": "qwen.isolated-d7-head-progress.v1",
            "status": "INCOMPLETE",
            "positions": positions,
            "pending_checks": ["checkpoint weights unchanged", "injected output fault detected"],
            "sources": sources,
            "results": {
                arm: {
                    "positions": len(rows),
                    "full_logits_exact": sum(r["full_logits_exact"] for r in rows),
                    "top20_set_exact": sum(r["20"]["set_exact"] for r in rows),
                    "top20_order_exact": sum(r["20"]["ranked_exact"] for r in rows),
                    "top1_exact": sum(r["1"]["set_exact"] for r in rows),
                }
                for arm, rows in records.items()
            },
        }
    )


def run(args):
    spec = private_json(args.spec)
    os.environ.update(spec["environment"])
    import torch
    from safetensors import safe_open
    from stock_m1_head_pair import StockM1HeadPair
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm

    from qwen_r9700_lab.conformance_radiance import verify_sources

    package = Path(importlib.util.find_spec("vllm").origin).parent.parent
    verify_sources(package, spec["binding"])
    torch.set_num_threads(2)
    manifest = private_json(args.captures / "manifest.json")
    reference = private_json(args.reference)
    authenticate(manifest)
    authenticate(reference)
    require(manifest["positions"] == len(reference["rows"]) == 320, "320 positions required")
    require(len(manifest["batches"]) == 40, "40 complete M8 groups required")
    model = Path(spec["native_config"]["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    with safe_open(str(model / index["weight_map"]["lm_head.weight"]), framework="pt") as source:
        weight = source.get_tensor("lm_head.weight").to("cuda")
    weight_digest = tensor_digest(weight)
    props = torch.cuda.get_device_properties(0)
    require("gfx1201" in props.gcnArchName, "pinned R9700 required")
    fixed = StockM1HeadPair(args.build)
    hidden = torch.zeros((8, 5120), device="cuda", dtype=torch.bfloat16)
    functions = {
        "reference": lambda: torch.cat(
            [rocm_unquantized_gemm(None, row[None], weight, None) for row in hidden]
        ),
        "old": lambda: rocm_unquantized_gemm(None, hidden, weight, None),
        "fixed": lambda: fixed(weight, hidden),
    }
    graphs = {}
    for name, function in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            value = function()
        graphs[name] = (graph, value)
    records = {arm: [] for arm in ("old", "fixed")}
    sources = {
        "capture_manifest": manifest["sha256"],
        "reference_rows": reference["sha256"],
        "build": fixed.manifest["sha256"],
        "binding": spec["binding"]["sha256"],
        "source_sha256": file_digest(Path(__file__)),
    }
    replace_private(args.output, "progress.json", partial_progress(records, sources))
    references_checked = 0
    for number, batch in enumerate(manifest["batches"]):
        path = args.captures / batch["file"]
        metadata = private_json(path.with_suffix(".json"))
        authenticate(metadata)
        require(metadata["sha256"] == batch["sha256"], "capture metadata changed")
        require(
            file_digest(path) == metadata["tensor_sha256"],
            "captured tensors changed",
        )
        last = metadata["events"][-1]
        require(
            last["operation"] == "qwen_d7_qualified.gemma_residual.default", "final norm missing"
        )
        # Only touch the small final hidden tensor, not every captured boundary.
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        selected = saved[last["after"][0]["key"]].clone()
        require(selected.shape == (8, 5120) and selected.dtype == hidden.dtype, "head cut changed")
        hidden.copy_(selected)
        del saved
        for graph, _ in graphs.values():
            graph.replay()
        require(torch.equal(hidden.cpu(), selected), "a head modified its captured input")
        expected = graphs["reference"][1].float().cpu().numpy()
        summaries = [summarize_logits(row) for row in expected]
        offset = number * 8
        observed_rows = reference["rows"][offset : offset + 8]
        require(
            batch["positions"] == [r["absolute_position"] for r in observed_rows],
            "position mismatch",
        )
        require(
            all(s == row["logits"] for s, row in zip(summaries, observed_rows, strict=True)),
            "serial head does not reproduce the in-model reference",
        )
        references_checked += 8
        for arm in records:
            values = graphs[arm][1].float().cpu().numpy()
            for summary, row in zip(summaries, values, strict=True):
                records[arm].append(compare_rows(summary, summarize_logits(row)))
        replace_private(args.output, "progress.json", partial_progress(records, sources))
    require(
        tensor_digest(weight) == weight_digest,
        "a head modified the checkpoint weights",
    )
    corrupted = expected[0].copy()
    import numpy as np

    corrupted.view(np.uint32)[0] ^= 1
    require(
        not compare_rows(summaries[0], summarize_logits(corrupted))["full_logits_exact"],
        "the injected one-bit output fault was not detected",
    )
    result = {}
    for arm, rows in records.items():
        result[arm] = {
            "positions": len(rows),
            "layer_instances": [None],
            "evaluations": len(rows),
            "isolated_inputs_verified": True,
            "reference_remainder_verified": True,
            "reference_remainder": "identity: vocabulary head is the final numerical stage",
            "stage_output_exact": sum(r["full_logits_exact"] for r in rows),
            "stage_state_exact": 320,
            "state_scope": "head is stateless; captured input and checkpoint weights are read-only",
            "top20_set_exact": sum(r["20"]["set_exact"] for r in rows),
            "top20_order_exact": sum(r["20"]["ranked_exact"] for r in rows),
            "top1_exact": sum(r["1"]["set_exact"] for r in rows),
        }
    return seal(
        {
            "schema": "qwen.isolated-d7-head.v1",
            "status": "SAMPLE_CHECKED",
            "stage": "Full BF16 target head",
            "results": result,
            "reference_positions_checked": references_checked,
            "negative_control_detected": True,
            "head_weights_sha256": weight_digest,
            "graph_replay_all_arms": True,
            **sources,
            "scope": "320 actual Pi reference hidden rows; isolated head, not other decoder stages",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "captures", "reference", "build", "output"):
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
