"""Exercise the actual performance CLI with measurable and unusable controls."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/bin/compare.py"


class PerformanceEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="radiance-performance-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.row = {
            "config": "fixture", "workload": "decode", "input_tokens": 128,
            "output_tokens": 64, "concurrency": 1, "tp": 2, "spec": "off",
            "cpu_offload_gb": 0, "temperature": 0,
            "median_output_throughput": 100.0,
        }

    def run_gate(self, baseline, candidate, threshold="0"):
        for name, rows in (("baseline", baseline), ("candidate", candidate)):
            (self.root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
        args = [sys.executable, str(SCRIPT), str(self.root / "baseline.json"),
                str(self.root / "candidate.json"), "--output", str(self.root / "comparison.md")]
        if threshold is not None:
            args += ["--fail-below", threshold]
        return subprocess.run(args, capture_output=True, text=True)

    def test_matching_finite_decode_passes(self):
        result = self.run_gate([self.row], [self.row])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_regression_fails_and_preserves_report(self):
        candidate = dict(self.row, median_output_throughput=90.0)
        result = self.run_gate([self.row], [candidate])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-10.00%", (self.root / "comparison.md").read_text(encoding="utf-8"))

    def test_empty_comparison_cannot_pass_gate(self):
        self.assertNotEqual(self.run_gate([], []).returncode, 0)

    def test_unmatched_decode_cannot_pass_gate(self):
        candidate = dict(self.row, concurrency=2)
        self.assertNotEqual(self.run_gate([self.row], [candidate]).returncode, 0)

    def test_prefill_only_cannot_pass_decode_gate(self):
        row = dict(self.row, workload="prefill")
        self.assertNotEqual(self.run_gate([row], [row]).returncode, 0)

    def test_unusable_decode_measurements_cannot_pass_gate(self):
        for side in ("baseline", "candidate"):
            for value in (None, True, 0, -1, float("nan"), float("inf"), float("-inf")):
                with self.subTest(side=side, value=value):
                    baseline, candidate = copy.deepcopy(self.row), copy.deepcopy(self.row)
                    (baseline if side == "baseline" else candidate)["median_output_throughput"] = value
                    self.assertNotEqual(self.run_gate([baseline], [candidate]).returncode, 0)

    def test_one_valid_case_cannot_hide_an_invalid_common_case(self):
        bad = dict(self.row, concurrency=2, median_output_throughput=None)
        self.assertNotEqual(self.run_gate([self.row, bad], [self.row, bad]).returncode, 0)

    def test_nonfinite_threshold_is_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(threshold=value):
                self.assertNotEqual(self.run_gate([self.row], [self.row], threshold=value).returncode, 0)

    def test_negative_finite_threshold_still_works(self):
        candidate = dict(self.row, median_output_throughput=99.0)
        self.assertEqual(self.run_gate([self.row], [candidate], threshold="-2").returncode, 0)

    def test_overflowing_relative_delta_cannot_pass_gate(self):
        baseline = dict(self.row, median_output_throughput=1.0e-308)
        candidate = dict(self.row, median_output_throughput=1.0e308)
        self.assertNotEqual(self.run_gate([baseline], [candidate]).returncode, 0)

    def test_report_only_mode_can_describe_empty_comparison(self):
        self.assertEqual(self.run_gate([], [], threshold=None).returncode, 0)


if __name__ == "__main__":
    unittest.main()
