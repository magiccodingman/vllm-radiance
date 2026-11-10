#!/usr/bin/env python3
"""Real Triton confidence contracts: CPU interpreter or explicitly selected GPU.

NVIDIA execution is portable-kernel evidence, not ROCm/R9700 qualification.
"""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu (Triton interpreter), cuda:0, cuda:1, ...")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["TRITON_INTERPRET"] = "1"
    elif os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Unset TRITON_INTERPRET for an actual GPU run")
    import torch
    import triton
    source = Path(os.environ.get("RADIANCE_DRAFT_GPU_SOURCE", ROOT / "radiance_draft_gpu.py"))
    spec = importlib.util.spec_from_file_location("tested_draft_gpu", source)
    gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu)
    device = torch.device(args.device)
    if device.type == "cuda":
        # CUDA/HIP graph contexts capture the current device's stream.
        torch.cuda.set_device(device)

    class ConfidenceTests(unittest.TestCase):
        def check(self, logits):
            logits = logits.contiguous()
            expected = logits.float().softmax(-1).amax(-1)
            output = torch.empty(logits.shape[0], device=device)
            maximum, total = torch.empty_like(output), torch.empty_like(output)
            scratch = gpu.make_scratch(logits.shape[0], device)
            gpu.capture_gpu(logits, output, scratch)
            gpu.capture_local(logits, maximum, total, scratch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            torch.testing.assert_close(output, expected, rtol=5e-5, atol=1e-7, equal_nan=True)
            expected_max = logits.float().amax(-1)
            # Empty shards contribute no exponential mass to another TP rank.
            safe_max = torch.where(torch.isneginf(expected_max), 0.0, expected_max)
            expected_sum = (logits.float() - safe_max[:, None]).exp().sum(-1)
            torch.testing.assert_close(maximum, expected_max, equal_nan=True)
            torch.testing.assert_close(total, expected_sum, rtol=5e-5, atol=1e-7)
            return maximum, total

        def random(self, width, dtype=torch.float32):
            return torch.randn(2, width, device=device, dtype=dtype)

        def test_finite_rows_and_partial_tiles(self):
            dtypes = (torch.float32,) if device.type == "cpu" else (torch.float32, torch.float16, torch.bfloat16)
            for width in (1, 511, 4097, 124160, 248320):
                for dtype in dtypes:
                    with self.subTest(width=width, dtype=dtype):
                        self.check(self.random(width, dtype))

        def test_dominant_token_beyond_old_vocabulary_limit(self):
            for width in (262145, 270000, 600000):
                with self.subTest(width=width):
                    logits = self.random(width)
                    logits[:, -1] = 30
                    self.check(logits)

        def test_entire_masked_block_is_neutral(self):
            logits = self.random(8192)
            logits[:, :4096] = -float("inf")
            self.check(logits)

        def test_finite_values_after_masked_iteration(self):
            logits = torch.full((2, 8192), -float("inf"), device=device)
            logits[:, 512:1024] = self.random(512)
            self.check(logits)

        def test_fully_masked_row_preserves_undefined_softmax(self):
            self.check(torch.full((2, 8192), -float("inf"), device=device))

        def test_empty_rank_does_not_poison_cross_rank_confidence(self):
            logits = self.random(16384)
            logits[:, :4096] = -float("inf")
            stats = [self.check(part.contiguous()) for part in logits.chunk(4, dim=-1)]
            maxima = torch.stack([stat[0] for stat in stats], dim=-1)
            sums = torch.stack([stat[1] for stat in stats], dim=-1)
            maximum = maxima.amax(-1)
            combined = (sums * (maxima - maximum[:, None]).exp()).sum(-1).reciprocal()
            torch.testing.assert_close(combined, logits.softmax(-1).amax(-1), rtol=5e-5, atol=1e-7)

        def test_tp1_caller_handles_fully_masked_logits(self):
            # Execute the actual caller without importing/installing vLLM or hooks.
            tree = ast.parse((ROOT / "radiance_draft.py").read_text(encoding="utf-8"))
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_local_draft"]
            scope = {"_GPU": [gpu]}
            exec(compile(tree, str(ROOT / "radiance_draft.py"), "exec"), scope)
            logits = torch.full((2, 8192), -float("inf"), device=device)
            proposer = SimpleNamespace(model=SimpleNamespace(
                logits_processor=SimpleNamespace(_apply_head=lambda *args: logits), lm_head=SimpleNamespace()))
            module = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1,
                                     tensor_model_parallel_all_gather=lambda *args, **kwargs: self.fail("TP1 gathered"))
            with patch.dict(sys.modules, {"vllm.distributed": module}):
                _, confidence = scope["_local_draft"](proposer, None)
            self.assertTrue(torch.isnan(confidence).all().item())

        @unittest.skipIf(device.type == "cpu", "CUDA/HIP graph execution requires a GPU")
        def test_graph_replay_reuses_scratch(self):
            logits = self.random(270000)
            logits[:, :4096] = -float("inf")
            output = torch.empty(2, device=device)
            scratch = gpu.make_scratch(2, device)
            for _ in range(3):
                gpu.capture_gpu(logits, output, scratch)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gpu.capture_gpu(logits, output, scratch)
            for dominant in (False, True):
                if dominant:
                    logits[:, -1] = 30
                graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(output, logits.softmax(-1).amax(-1), rtol=5e-5, atol=1e-7)

    torch.manual_seed(123)
    print(f"Confidence contracts: device={device}, torch={torch.__version__}, triton={triton.__version__}, source={source}", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConfidenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
