"""Reject invalid calibration inputs before allocating a model or writing scales."""

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import radiance_kv_calibration as calibration


class ReachedModelConstruction(RuntimeError):
    pass


class CalibrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.observations = {"layer.attn": {"q": 2.0, "k": 4.0, "v": 8.0, "calls": 1}}
        temp = tempfile.TemporaryDirectory(prefix="radiance-calibration-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        # Identity preflight hashes bytes; no actual checkpoint is loaded.
        (model / "model.safetensors").write_bytes(b"identity-only fixture")
        corpus = self.root / "corpus.jsonl"
        corpus.write_text('{"text":"hello"}\n', encoding="utf-8")
        self.args = calibration._parser().parse_args([
            "calibrate", "--model", str(model), "--corpus", str(corpus),
            "--output", str(self.root / "scales.safetensors"),
        ])

    def reports(self):
        return [{"rank": rank, "layers": copy.deepcopy(self.observations)} for rank in (0, 1)]

    def test_nonfinite_parameters_are_rejected(self):
        for field in ("fp8_max", "margin"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        calibration.make_scale_values(self.observations, **{field: value})

    def test_float32_unrepresentable_scales_are_rejected(self):
        for fp8, margin in ((1.0e308, 1.05), (1.0e-308, 1.05), (448.0, 1.0e308)):
            with self.subTest(fp8_max=fp8, margin=margin):
                with self.assertRaises(ValueError):
                    calibration.make_scale_values(self.observations, fp8_max=fp8, margin=margin)

    def test_invalid_observation_cannot_be_masked_by_another_rank(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                reports = self.reports()
                reports[0]["layers"]["layer.attn"]["q"] = value
                with self.assertRaises(ValueError):
                    calibration.aggregate_worker_stats(reports)

    def test_unobserved_rank_cannot_be_masked_by_another_rank(self):
        reports = self.reports()
        reports[0]["layers"]["layer.attn"]["calls"] = 0
        with self.assertRaises(ValueError):
            calibration.aggregate_worker_stats(reports)

    def test_zero_local_maximum_is_valid_with_positive_global_maximum(self):
        reports = self.reports()
        reports[0]["layers"]["layer.attn"]["q"] = 0.0
        self.assertEqual(calibration.aggregate_worker_stats(reports)["layer.attn"]["q"], 2.0)

    def test_valid_scales_preserve_existing_arithmetic(self):
        result = calibration.make_scale_values(self.observations)
        for kind in ("q", "k", "v"):
            self.assertEqual(result[f"layer.attn.{kind}_scale"], self.observations["layer.attn"][kind] * 1.05 / 448.0)
        self.assertEqual(result["layer.attn.prob_scale"], 1.05 / 448.0)

    def test_direct_scale_conversion_rejects_invalid_observations(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                observations = copy.deepcopy(self.observations)
                observations["layer.attn"]["q"] = value
                with self.assertRaises(ValueError):
                    calibration.make_scale_values(observations)

    def test_invalid_parameters_fail_before_importing_vllm(self):
        for field, value in (("fp8_max", float("inf")), ("margin", float("nan"))):
            with self.subTest(field=field), patch.dict(sys.modules, {"vllm": None}):
                args = copy.copy(self.args)
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    calibration.calibrate(args)

    def test_existing_output_is_rejected_before_importing_vllm(self):
        Path(self.args.output).write_bytes(b"preserve")
        with patch.dict(sys.modules, {"vllm": None}), self.assertRaises(FileExistsError):
            calibration.calibrate(self.args)
        self.assertEqual(Path(self.args.output).read_bytes(), b"preserve")

    def test_valid_preflight_reaches_model_factory(self):
        factory = Mock(side_effect=ReachedModelConstruction)
        module = types.ModuleType("vllm")
        module.LLM = factory
        module.SamplingParams = Mock()
        with patch.dict(sys.modules, {"vllm": module}), self.assertRaises(ReachedModelConstruction):
            calibration.calibrate(self.args)
        factory.assert_called_once()
        self.assertFalse(Path(self.args.output).exists())


if __name__ == "__main__":
    unittest.main()
