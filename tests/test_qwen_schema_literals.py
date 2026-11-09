"""Protect Qwen grammar schema semantics without loading vLLM or a model."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_QWEN_SCHEMA_SOURCE", ROOT / "patch_qwen_open_object_schema.py"))
spec = importlib.util.spec_from_file_location("qwen_schema_under_test", SOURCE)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


def helpers(text=None):
    namespace = {"StructuralTagBuilder": object}
    exec(overlay.HELPER_NEW if text is None else text, namespace)
    return namespace["_normalize_qwen_open_nested_objects"]


class QwenSchemaLiteralTests(unittest.TestCase):
    def test_const_and_enum_keep_their_accepted_instances(self):
        for keyword, value in (("const", {"type": "object"}), ("enum", [{"type": "object"}])):
            with self.subTest(keyword=keyword):
                schema = {"type": "object", "properties": {"literal": {keyword: copy.deepcopy(value)}},
                          "required": ["literal"]}
                instance = {"literal": {"type": "object"}}
                self.assertTrue(Draft202012Validator(schema).is_valid(instance))
                helpers()(schema, is_root=True)
                self.assertTrue(Draft202012Validator(schema).is_valid(instance))
                self.assertFalse(Draft202012Validator(schema).is_valid(
                    {"literal": {"type": "object", "additionalProperties": True}}))

    def test_annotations_and_extension_payloads_are_unchanged(self):
        payload = {"type": "object", "properties": {"nested": {"type": "object"}}}
        schema = {"type": "object", **{key: copy.deepcopy(payload) for key in
                  ("default", "example", "x-client-data")}, "examples": [copy.deepcopy(payload)]}
        before = copy.deepcopy(schema)
        helpers()(schema, is_root=True)
        self.assertEqual(schema, before)

    def test_schema_named_properties_are_still_normalized(self):
        schema = {"type": "object", "properties": {name: {"type": "object"}
                  for name in ("default", "examples", "enum", "const", "type")}}
        helpers()(schema, is_root=True)
        for child in schema["properties"].values():
            self.assertEqual(child, {"type": "object", "additionalProperties": True})
        self.assertNotIn("additionalProperties", schema)

    def test_schema_maps_and_applicators_are_visited(self):
        for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"):
            with self.subTest(keyword=keyword):
                schema = {keyword: {"payload": {"type": "object"}}}
                helpers()(schema, is_root=True)
                self.assertIs(schema[keyword]["payload"].get("additionalProperties"), True)
        for keyword in ("items", "additionalItems", "additionalProperties", "unevaluatedProperties",
                        "unevaluatedItems", "contains", "propertyNames", "not", "if", "then", "else"):
            with self.subTest(keyword=keyword):
                schema = {keyword: {"type": "object"}}
                helpers()(schema, is_root=True)
                self.assertIs(schema[keyword].get("additionalProperties"), True)

    def test_schema_arrays_and_legacy_tuple_items_are_visited(self):
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems", "items"):
            with self.subTest(keyword=keyword):
                schema = {keyword: [{"type": "object"}, True, False]}
                helpers()(schema, is_root=True)
                self.assertIs(schema[keyword][0].get("additionalProperties"), True)
                self.assertEqual(schema[keyword][1:], [True, False])

    def test_root_declared_closed_and_composed_controls(self):
        for schema in ({"type": "object"}, {"type": "object", "additionalProperties": False},
                       {"type": "object", "properties": {"name": {"type": "string"}}},
                       {"type": "object", "$ref": "#/$defs/item"}):
            before = copy.deepcopy(schema)
            helpers()(schema, is_root=True)
            self.assertEqual(schema, before)
        schema = {"type": "object", "allOf": [{"required": ["name"]}]}
        before = copy.deepcopy(schema)
        helpers()(schema, is_root=False)
        self.assertEqual(schema, before)

    def test_booleans_and_dependency_name_arrays_are_unchanged(self):
        schema = {"dependencies": {"name": ["other", "type"]}, "additionalProperties": False}
        before = copy.deepcopy(schema)
        helpers()(schema, is_root=True)
        self.assertEqual(schema, before)

    def test_unevaluated_property_constraints_are_not_opened(self):
        for constraint in (False, {"type": "integer"}):
            with self.subTest(constraint=constraint):
                schema = {"properties": {"payload": {"type": "object", "unevaluatedProperties": constraint}}}
                instances = [{"payload": {}}, {"payload": {"extra": 1}}, {"payload": {"extra": "text"}}]
                before = [Draft202012Validator(schema).is_valid(value) for value in instances]
                helpers()(schema, is_root=True)
                self.assertEqual([Draft202012Validator(schema).is_valid(value) for value in instances], before)
                self.assertNotIn("additionalProperties", schema["properties"]["payload"])

    def test_same_instance_applicators_keep_unevaluated_closure(self):
        for keyword in ("allOf", "anyOf", "oneOf"):
            with self.subTest(keyword=keyword):
                schema = {"properties": {"payload": {
                    keyword: [{"type": "object"}], "unevaluatedProperties": False}}}
                instance = {"payload": {"extra": 1}}
                self.assertFalse(Draft202012Validator(schema).is_valid(instance))
                helpers()(schema, is_root=True)
                self.assertFalse(Draft202012Validator(schema).is_valid(instance))

    def test_parent_closure_does_not_disable_nested_open_object_fix(self):
        schema = {"type": "object", "unevaluatedProperties": False,
                  "properties": {"payload": {"type": "object"}}}
        instance = {"payload": {"extra": 1}}
        self.assertTrue(Draft202012Validator(schema).is_valid(instance))
        helpers()(schema, is_root=True)
        self.assertTrue(Draft202012Validator(schema).is_valid(instance))
        self.assertIs(schema["properties"]["payload"].get("additionalProperties"), True)
        # Explicit true already evaluates unknown properties, so spelling out
        # additionalProperties remains safe and must keep the original fix.
        schema = {"properties": {"payload": {"allOf": [
            {"type": "object", "unevaluatedProperties": True}], "unevaluatedProperties": False}}}
        self.assertTrue(Draft202012Validator(schema).is_valid(instance))
        helpers()(schema, is_root=True)
        self.assertIs(schema["properties"]["payload"]["allOf"][0].get("additionalProperties"), True)
        self.assertTrue(Draft202012Validator(schema).is_valid(instance))

    def test_fresh_and_already_baked_patch_are_idempotent(self):
        # This fixture retains the original installed helper to exercise updating
        # an image where the old broad helper sentinel is already present.
        old_helper = overlay.HELPER_NEW
        if hasattr(overlay, "NEW_NORMALIZER"):
            old_helper = old_helper.replace(overlay.NEW_NORMALIZER, overlay.LEGACY_NORMALIZER)
        elif hasattr(overlay, "SCHEMA_WALK"):
            old_helper = old_helper.replace(overlay.SCHEMA_WALK,
                "    for value in tuple(schema.values()):\n"
                "        _normalize_qwen_open_nested_objects(value, is_root=False)\n")
        fresh = (overlay.HELPER_ANCHOR + "def build(tools, tool_choice):\n" + overlay.CALL_ANCHOR +
                 "    return dumped_tools\n")
        baked = old_helper + "def build(tools, tool_choice):\n" + overlay.CALL_NEW + "    return dumped_tools\n"
        for label, source in (("fresh", fresh), ("baked", baked)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "registry.py"
                target.write_text(source, encoding="utf-8")
                with patch.object(overlay, "F", target):
                    overlay.main()
                    once = target.read_text(encoding="utf-8")
                    overlay.main()
                self.assertEqual(target.read_text(encoding="utf-8"), once)
                schema = {"properties": {"literal": {"const": {"type": "object"}}}}
                helpers(once)(schema, is_root=True)
                self.assertEqual(schema["properties"]["literal"]["const"], {"type": "object"})


if __name__ == "__main__":
    unittest.main()
