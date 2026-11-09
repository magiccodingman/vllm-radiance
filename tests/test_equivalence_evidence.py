"""Exercise the real qualification command on valid and incomplete raw runs."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/bin/verify_outputs.py"


class EquivalenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="radiance-equivalence-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.result = {
            "temperature": "0", "seed": 17, "prompts": ["Hello"],
            "input_lens": [1], "generated_texts": ["world"],
            "output_lens": [1], "completed": 1, "failed": 0,
        }

    def compare(self, baseline, candidate=None):
        for name, value in (("baseline", baseline), ("candidate", candidate or baseline)):
            raw = self.root / name / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "case.json").write_text(json.dumps(value), encoding="utf-8")
        report_path = self.root / "report.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root / "baseline"),
             str(self.root / "candidate"), "--output", str(report_path)],
            capture_output=True, text=True,
        )
        return result, json.loads(report_path.read_text(encoding="utf-8"))

    def assert_rejected(self, value):
        result, report = self.compare(value)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["mismatches"])

    def test_valid_matching_run_passes(self):
        result, report = self.compare(self.result)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["compared_requests"], 1)
        self.assertEqual(report["status"], "passed")

    def test_missing_lengths_on_both_sides_fail(self):
        del self.result["output_lens"]
        self.assert_rejected(self.result)

    def test_empty_runs_fail(self):
        self.result.update(generated_texts=[], output_lens=[], completed=0, prompts=[], input_lens=[])
        self.assert_rejected(self.result)

    def test_non_string_texts_fail(self):
        self.result["generated_texts"] = [None]
        self.assert_rejected(self.result)

    def test_lengths_must_cover_every_text(self):
        self.result["output_lens"] = []
        self.assert_rejected(self.result)

    def test_invalid_token_lengths_fail(self):
        for value in (-1, True, 1.0, "1", None):
            with self.subTest(length=value):
                self.result["output_lens"] = [value]
                self.assert_rejected(self.result)

    def test_failed_requests_fail(self):
        self.result["failed"] = 1
        self.assert_rejected(self.result)

    def test_incomplete_request_count_fails(self):
        self.result["completed"] = 2
        self.assert_rejected(self.result)

    def test_empty_generated_string_with_zero_tokens_is_valid(self):
        self.result.update(generated_texts=[""], output_lens=[0])
        result, report = self.compare(self.result)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["compared_requests"], 1)

    def test_existing_text_divergence_still_fails(self):
        candidate = copy.deepcopy(self.result)
        candidate["generated_texts"] = ["different"]
        result, report = self.compare(self.result, candidate)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any("generated text differs" in item for item in report["mismatches"]))


if __name__ == "__main__":
    unittest.main()
