"""CPU-only CLI contracts with synthetic server responses and counters."""

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(os.environ.get(
    "RADIANCE_KV_GATE_TEST_SOURCE",
    str(Path(__file__).resolve().parents[1] / "benchmarks/bin/run_kv_offload_restore_gate.py"),
))
spec = importlib.util.spec_from_file_location("tested_kv_gate", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class KVGateOutputTests(unittest.TestCase):
    def run_gate(self, output, *, restored_text="same", race=False):
        before = dict.fromkeys(gate.METRICS, 0.0)
        stored = dict(before, store_bytes=4096.0)
        restored = dict(stored, load_bytes=4096.0, external_hits=1024.0)
        replies = iter(("same", "same", restored_text))

        def request(method, url, payload=None, timeout=None):
            if "/reset_prefix_cache" in url:
                if race and "reset_external=false" in url:
                    output.write_text("concurrent immutable result", encoding="utf-8")
                return {"success": True}
            self.assertEqual(method, "POST")
            self.assertTrue(url.endswith("/v1/completions"))
            return {"choices": [{"text": next(replies), "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2048}}

        argv = ["gate", "--model", "synthetic", "--output", str(output),
                "--prompt-repetitions", "1", "--max-tokens", "1"]
        with patch.object(sys, "argv", argv), patch.object(gate, "request_json", side_effect=request) as network, \
                patch.object(gate, "scrape_metrics", side_effect=[before, stored, stored]), \
                patch.object(gate, "wait_for_metric_increase", side_effect=[stored, restored]), \
                patch.object(sys, "stdout", io.StringIO()):
            gate.main()
        self.assertEqual(network.call_count, 5)
        return json.loads(output.read_text())

    def test_existing_parent_accepts_new_result(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_gate(Path(folder) / "result.json")
            self.assertTrue(result["passed"])
            self.assertTrue(all(result["checks"].values()))

    def test_missing_parent_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new" / "nested" / "result.json"
            self.assertTrue(self.run_gate(output)["passed"])

    def test_current_directory_output(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path.cwd()
            try:
                os.chdir(folder)
                self.assertTrue(self.run_gate(Path("result.json"))["passed"])
            finally:
                os.chdir(previous)

    def test_failed_checks_are_saved_before_failure_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failed.json"
            with self.assertRaises(SystemExit) as raised:
                self.run_gate(output, restored_text="different")
            self.assertEqual(raised.exception.code, 1)
            result = json.loads(output.read_text())
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["cpu_restore_byte_identical"])
            self.assertEqual(result["phases"]["cpu_restore"]["text"], "different")

    def test_existing_result_is_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing.json"
            output.write_text("immutable", encoding="utf-8")
            argv = ["gate", "--model", "synthetic", "--output", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.object(gate, "request_json", side_effect=AssertionError("unexpected request")) as network:
                with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                    gate.main()
            network.assert_not_called()
            self.assertEqual(output.read_text(), "immutable")

    def test_concurrent_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            with self.assertRaises(FileExistsError) as raised:
                self.run_gate(output, race=True)
            self.assertEqual(Path(raised.exception.filename), output)
            self.assertEqual(output.read_text(), "concurrent immutable result")


if __name__ == "__main__":
    unittest.main()
