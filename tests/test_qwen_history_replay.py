"""Render real Qwen templates with Jinja and optional pinned Transformers integration."""
import json
import os
from pathlib import Path
import unittest

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


ROOT = Path(os.environ.get("RADIANCE_QWEN_TEMPLATE_ROOT", Path(__file__).resolve().parents[1]))


def compile_template(source):
    if os.environ.get("RADIANCE_QWEN_TEMPLATE_COMPILER") == "transformers":
        from transformers.utils.chat_template_utils import _compile_jinja_template
        compiled = _compile_jinja_template(source)
        # The image's guarded from_json overlay supplies this compatibility filter.
        compiled.environment.filters["from_json"] = json.loads
        return compiled
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters.update(from_json=json.loads, tojson=json.dumps)

    def raise_exception(message):
        raise TemplateError(message)

    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(source)


class QwenHistoryReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = {name: compile_template((ROOT / name).read_text(encoding="utf-8"))
                         for name in ("qwen3.8-enhanced.jinja", "qwen3.6-enhanced.jinja")}

    def render(self, name, messages, **kwargs):
        return self.templates[name].render(messages=messages, tools=None, add_generation_prompt=False, **kwargs)

    def test_literal_close_marker_in_answer_is_preserved(self):
        answer = "Use the literal </think> delimiter in the example. Keep this answer."
        messages = [{"role": "user", "content": "Explain the delimiter."},
                    {"role": "assistant", "content": "<think>Plan</think>\n" + answer}]
        for name in self.templates:
            with self.subTest(template=name):
                rendered = self.render(name, messages)
                self.assertIn("<think>\nPlan\n</think>", rendered)
                self.assertIn(answer, rendered)

    def test_inline_tool_payload_with_close_marker_is_preserved(self):
        body = '<tool_call>\n<function=echo>\n<parameter=text>\n</think>\n</parameter>\n</function>\n</tool_call>'
        messages = [{"role": "user", "content": "Echo a delimiter."},
                    {"role": "assistant", "content": "<think>Plan</think>\n" + body}]
        for name in self.templates:
            with self.subTest(template=name):
                self.assertIn(body, self.render(name, messages))

    def test_all_answer_markers_are_preserved(self):
        answer = "First </think>, second </think>, then the complete final text."
        messages = [{"role": "assistant", "content": "<think>Plan</think>" + answer}]
        for name in self.templates:
            with self.subTest(template=name):
                self.assertIn(answer, self.render(name, messages))

    def test_preserve_thinking_false_keeps_historical_answer(self):
        answer = "Historical answer with </think> as data."
        messages = [{"role": "user", "content": "First request"},
                    {"role": "assistant", "content": "<think>Old plan</think>" + answer},
                    {"role": "user", "content": "Next request"}]
        for name in self.templates:
            with self.subTest(template=name):
                rendered = self.render(name, messages, preserve_thinking=False)
                self.assertIn(answer, rendered)
                self.assertNotIn("Old plan", rendered)

    def test_ordinary_inline_reasoning_is_unchanged(self):
        messages = [{"role": "assistant", "content": "<think>Plan</think>\nAnswer"}]
        for name in self.templates:
            with self.subTest(template=name):
                self.assertIn("<|im_start|>assistant\n<think>\nPlan\n</think>\n\nAnswer<|im_end|>",
                              self.render(name, messages))

    def test_explicit_reasoning_content_is_unchanged(self):
        answer = "An explicit answer containing </think> as data."
        messages = [{"role": "assistant", "reasoning_content": "Explicit plan", "content": answer}]
        for name in self.templates:
            with self.subTest(template=name):
                rendered = self.render(name, messages)
                self.assertIn("<think>\nExplicit plan\n</think>", rendered)
                self.assertIn(answer, rendered)

    def test_mapping_and_json_tool_arguments_match(self):
        values = {"text": "</think>", "enabled": True, "payload": {"n": 1}}
        for name in self.templates:
            results = []
            for arguments in (values, json.dumps(values)):
                messages = [{"role": "assistant", "content": None, "tool_calls": [
                    {"function": {"name": "echo", "arguments": arguments}}]}]
                results.append(self.render(name, messages))
            with self.subTest(template=name):
                self.assertEqual(*results)
                self.assertIn("<parameter=enabled>\ntrue\n</parameter>", results[0])

    def test_boolean_and_null_parameters_use_json_literals(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"function": {
            "name": "echo", "arguments": {"yes": True, "no": False, "missing": None}}}]}]
        for name in self.templates:
            with self.subTest(template=name):
                rendered = self.render(name, messages)
                for parameter, value in (("yes", "true"), ("no", "false"), ("missing", "null")):
                    self.assertIn(f"<parameter={parameter}>\n{value}\n</parameter>", rendered)

    def test_vision_placeholders_and_qwen38_reasoning_defaults_remain(self):
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "video_url", "video_url": {"url": "unused"}}, {"type": "text", "text": "Describe"}]}]
        rendered = self.render("qwen3.8-enhanced.jinja", messages)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", rendered)
        self.assertIn("<|vision_start|><|video_pad|><|vision_end|>", rendered)
        self.assertIn("Reasoning effort is set to xhigh.", rendered)
        self.assertNotIn("Reasoning effort is set", self.render("qwen3.8-enhanced.jinja", messages, enable_thinking=False))


if __name__ == "__main__":
    unittest.main()
