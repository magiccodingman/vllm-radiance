"""Reject typed package and numeric failures that Python equality can conceal."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_QWEN_GATE_SOURCE", ROOT / "benchmarks/bin/run_client_tool_gate.py"))
spec = importlib.util.spec_from_file_location("client_gate_under_test", SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def call(value, name="package_probe"):
    return {"function": {"name": name, "arguments": json.dumps(value)}}


class ClientToolValueTests(unittest.TestCase):
    def test_original_package_and_key_order_are_valid(self):
        value = json.loads(json.dumps(gate.PACKAGE, sort_keys=True))
        self.assertEqual(gate.validate("package_auto", [call(value)], "tool_calls"), [])

    def test_boolean_cannot_replace_zero_version(self):
        value = copy.deepcopy(gate.PACKAGE)
        value["request"]["expectedVersion"] = False
        self.assertTrue(gate.validate("package_auto", [call(value)], "tool_calls"))

    def test_nested_boolean_cannot_replace_integer_coordinates(self):
        for index, boolean in ((0, False), (1, True)):
            with self.subTest(index=index):
                value = copy.deepcopy(gate.PACKAGE)
                value["request"]["items"][1]["definition"]["blocks"][index]["row"] = boolean
                self.assertTrue(gate.validate("package_strict", [call(value)], "tool_calls"))

    def test_equal_json_numbers_remain_valid(self):
        # JSON Schema integer accepts numbers with a zero fractional part.
        value = copy.deepcopy(gate.PACKAGE)
        value["request"]["expectedVersion"] = 0.0
        self.assertEqual(gate.validate("package_auto", [call(value)], "tool_calls"), [])

    def test_changed_values_and_container_shapes_fail(self):
        for replacement in ("0", None, 0.5, [], {}):
            with self.subTest(replacement=replacement):
                value = copy.deepcopy(gate.PACKAGE)
                value["request"]["expectedVersion"] = replacement
                self.assertTrue(gate.validate("package_auto", [call(value)], "tool_calls"))

    def test_numeric_overflow_is_not_accepted_as_confidence(self):
        for exponent in ("1e400", "-1e400"):
            with self.subTest(exponent=exponent):
                value = {"function": {"name": "keypress", "arguments":
                         '{"reasoning":"Synthetic","confidence":' + exponent +
                         ',"stepComplete":true,"keys":["TAB"]}'}}
                self.assertTrue(gate.validate("keypress_auto", [value], "tool_calls"))

    def test_finite_numbers_and_large_integers_remain_valid(self):
        for confidence in (0, -2, 0.25, 1e300, 10**400):
            with self.subTest(kind=type(confidence).__name__):
                value = {"reasoning": "Synthetic", "confidence": confidence,
                         "stepComplete": True, "keys": ["TAB"]}
                self.assertEqual(gate.validate("keypress_auto", [call(value, "keypress")], "tool_calls"), [])

    def test_validation_does_not_modify_package_fixture(self):
        before = copy.deepcopy(gate.PACKAGE)
        gate.validate("package_auto", [call(gate.PACKAGE)], "tool_calls")
        self.assertEqual(gate.PACKAGE, before)


if __name__ == "__main__":
    unittest.main()
