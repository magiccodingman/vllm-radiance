"""Protect immutable offline tuning outputs without loading Torch or a GPU."""

import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import radiance_tunableop as tuning


class TunableOpImmutabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="radiance-tunableop-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "untuned0.csv").write_text("collected signatures\n", encoding="utf-8")
        self.args = argparse.Namespace(
            root=str(self.root), input_pattern="untuned?.csv", num_gpus=1,
            numerical_check="1e-3_1e-3", max_duration_ms=30, max_iterations=100,
            rotating_buffer_mib=-1,
        )
        torch = types.ModuleType("torch")
        cuda = types.ModuleType("torch.cuda")
        tunable = types.ModuleType("torch.cuda.tunable")
        torch.version = types.SimpleNamespace(hip="fixture")
        torch.cuda = cuda
        cuda.device_count = lambda: 2
        cuda.get_device_name = lambda index: "fixture GPU"
        cuda.tunable = tunable
        self.tune_call = Mock()
        tunable.mgpu_tune_gemm_in_file = self.tune_call
        modules = patch.dict(sys.modules, {
            "torch": torch, "torch.cuda": cuda, "torch.cuda.tunable": tunable,
        })
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)

    def result(self, ordinal=0, content="tuned signatures\n"):
        path = self.root / f"results{ordinal}.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_second_manifest_write_cannot_bless_changed_results(self):
        self.result()
        manifest = tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=1)
        original = manifest.read_bytes()
        self.result(content="different results\n")
        with self.assertRaises(FileExistsError):
            tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=1)
        self.assertEqual(manifest.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            tuning.verify(self.root)

    def test_tune_rejects_existing_manifest_before_tuning_or_environment_changes(self):
        self.result()
        manifest = self.root / "manifest.json"
        manifest.write_text("original manifest", encoding="utf-8")
        before = dict(os.environ)
        with self.assertRaises(FileExistsError):
            tuning.tune(self.args)
        self.tune_call.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(manifest.read_text(encoding="utf-8"), "original manifest")

    def test_tune_preserves_partial_or_empty_results(self):
        for content in ("partial results\n", ""):
            with self.subTest(content=content):
                result = self.result(content=content)
                with self.assertRaises(FileExistsError):
                    tuning.tune(self.args)
                self.tune_call.assert_not_called()
                self.assertEqual(result.read_text(encoding="utf-8"), content)

    def test_manifest_requires_positive_integer_gpu_count(self):
        self.result()
        for count in (0, -1, True):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=count)
                self.assertFalse((self.root / "manifest.json").exists())

    def test_tune_rejects_invalid_gpu_count_before_invocation(self):
        self.args.num_gpus = 0
        with self.assertRaises(ValueError):
            tuning.tune(self.args)
        self.tune_call.assert_not_called()

    def test_manifest_rejects_missing_device_output(self):
        self.result()
        with self.assertRaisesRegex(ValueError, "results1.csv"):
            tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=2)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_complete_fresh_manifest_verifies(self):
        self.result()
        self.result(1)
        tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=2)
        self.assertEqual(tuning.verify(self.root)["num_gpus"], 2)

    def test_verify_rejects_invalid_gpu_count(self):
        self.result()
        path = tuning.write_manifest(self.root, input_pattern="untuned?.csv", num_gpus=1)
        data = json.loads(path.read_text(encoding="utf-8"))
        for count in (0, -1, True):
            with self.subTest(count=count):
                data["num_gpus"] = count
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    tuning.verify(self.root)

    def test_fresh_tune_produces_valid_manifest(self):
        self.tune_call.side_effect = lambda *args: self.result()
        tuning.tune(self.args)
        self.tune_call.assert_called_once()
        self.assertEqual(tuning.verify(self.root)["num_gpus"], 1)


if __name__ == "__main__":
    unittest.main()
