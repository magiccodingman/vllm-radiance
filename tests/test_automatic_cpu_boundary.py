"""Automatic checks must never execute hardware/manual entries through truthiness."""

import contextlib
import io
import subprocess
import unittest
from unittest.mock import patch

from tools import radiance, validate_control_plane


def entry(**overrides):
    return {
        "id": "fixture", "class": "contract", "hardware": "cpu", "areas": ["fixture"],
        "command": "must not run a GPU probe", "automatic": True, "ci": False,
        **overrides,
    }


class AutomaticCPUBoundaryTests(unittest.TestCase):
    def run_plan(self, entries, dry_run=False):
        with patch.object(radiance.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            with contextlib.redirect_stdout(io.StringIO()):
                status = radiance.run_auto(entries, dry_run)
            return status, run.call_count

    def test_cpu_automatic_check_executes(self):
        self.assertEqual(self.run_plan([entry()]), (0, 1))

    def test_manual_hardware_entries_are_skipped(self):
        self.assertEqual(self.run_plan([entry(hardware="r9700-tp2", automatic=False)]), (0, 0))

    def test_non_boolean_flags_fail_without_execution(self):
        for flag in ("automatic", "ci"):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(flag=flag, value=value):
                    status, calls = self.run_plan([entry(**{flag: value})])
                    self.assertNotEqual(status, 0)
                    self.assertEqual(calls, 0)

    def test_non_cpu_automatic_entry_fails_without_execution(self):
        for hardware in ("candidate-image", "generic-rocm", "r9700", "r9700-tp2",
                         "maintenance-window-r9700-tp2"):
            with self.subTest(hardware=hardware):
                status, calls = self.run_plan([entry(hardware=hardware)])
                self.assertNotEqual(status, 0)
                self.assertEqual(calls, 0)

    def test_complete_plan_is_validated_before_any_command(self):
        status, calls = self.run_plan([entry(), entry(id="bad", hardware="r9700")])
        self.assertNotEqual(status, 0)
        self.assertEqual(calls, 0)

    def test_valid_cpu_dry_run_does_not_execute(self):
        self.assertEqual(self.run_plan([entry()], dry_run=True), (0, 0))

    def test_registry_validator_rejects_non_boolean_flags(self):
        for flag in ("automatic", "ci"):
            for value in ("false", "true", 0, 1, None):
                with self.subTest(flag=flag, value=value):
                    errors = []
                    doc = {"hardware_levels": ["cpu"], "tests": [entry(**{flag: value})]}
                    validate_control_plane.check_tests(radiance.REPO, doc, errors)
                    self.assertTrue(any(flag in error and "boolean" in error for error in errors), errors)

    def test_registry_validator_rejects_automatic_hardware(self):
        errors = []
        doc = {"hardware_levels": ["cpu", "r9700"], "tests": [entry(hardware="r9700")]}
        validate_control_plane.check_tests(radiance.REPO, doc, errors)
        self.assertTrue(any("automatic" in error and "cpu" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
