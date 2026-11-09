"""Exercise actual response assembly without loading conformance GPU/process code."""
import ast
import json
import math
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('RADIANCE_TOOL_ASSEMBLY_SOURCE',
             ROOT / 'benchmarks/conformance/src/qwen_r9700_lab/conformance_transport.py'))


class ToolAssemblyTests(unittest.TestCase):
    def setUp(self):
        names = {'ProtocolError', 'BackendResponseError', 'check_response_event',
                 '_finite_json_number', 'Completion'}
        nodes = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        self.scope = dict(json=json, math=math, DiagnosticError=RuntimeError)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.scope)
        self.completion = self.scope['Completion']()

    def delta(self, index, name='', arguments='', identity=None):
        call = dict(index=index, function=dict(name=name, arguments=arguments))
        if identity:
            call['id'] = identity
        self.completion.accept({'choices': [{'index': 0, 'delta': {'tool_calls': [call]}}]})

    def finish(self):
        self.completion.accept({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})
        return self.completion.result()['tools']

    def test_tool_order_follows_index_not_first_arrival(self):
        self.delta(1, 'second', '{}', 'id-second')
        self.delta(0, 'first', '{}', 'id-first')
        self.assertEqual([tool['name'] for tool in self.finish()], ['first', 'second'])

    def test_interleaved_fragments_keep_index_order_and_payload(self):
        self.delta(1, 'sec', '{"x":', 'id-second')
        self.delta(0, 'first', '{"y":2}', 'id-first')
        self.delta(1, 'ond', '1}')
        tools = self.finish()
        self.assertEqual([(tool['name'], tool['parsed_arguments']) for tool in tools],
                         [('first', {'y': 2}), ('second', {'x': 1})])

    def test_nonfinite_json_constants_are_rejected(self):
        for value in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(value=value):
                self.completion = self.scope['Completion']()
                self.delta(0, 'first', '{"nested":[' + value + ']}', 'id-first')
                with self.assertRaises(self.scope['ProtocolError']):
                    self.finish()

    def test_overflowing_json_exponents_are_rejected(self):
        for value in ('1e999', '-1e999'):
            with self.subTest(value=value):
                self.completion = self.scope['Completion']()
                self.delta(0, 'first', '{"value":' + value + '}', 'id-first')
                with self.assertRaises(self.scope['ProtocolError']):
                    self.finish()

    def test_finite_numbers_and_numeric_strings_are_preserved(self):
        self.delta(0, 'first', '{"a":1e308,"b":-2.5,"c":"NaN","d":12345678901234567890}', 'id-first')
        self.assertEqual(self.finish()[0]['parsed_arguments'],
                         dict(a=1e308, b=-2.5, c='NaN', d=12345678901234567890))

    def test_nested_boolean_null_and_unicode_are_preserved(self):
        self.delta(0, 'first', '{"x":[true,null,"λ"]}', 'id-first')
        self.assertEqual(self.finish()[0]['parsed_arguments'], {'x': [True, None, 'λ']})

    def test_nonstreaming_tools_keep_declared_order(self):
        self.completion.accept({'choices': [{'message': {'tool_calls': [
            {'id': 'a', 'function': {'name': 'first', 'arguments': '{}'}},
            {'id': 'b', 'function': {'name': 'second', 'arguments': '{}'}},
        ]}, 'finish_reason': 'tool_calls'}]})
        self.assertEqual([t['name'] for t in self.completion.result()['tools']], ['first', 'second'])

    def test_tool_index_gaps_remain_rejected(self):
        self.delta(1, 'second', '{}', 'id-second')
        with self.assertRaises(self.scope['ProtocolError']):
            self.finish()

    def test_wrong_finish_boundary_remains_rejected(self):
        self.delta(0, 'first', '{}', 'id-first')
        self.completion.accept({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
        with self.assertRaises(self.scope['ProtocolError']):
            self.completion.result()

    def test_result_reassembly_does_not_mutate_fragments(self):
        self.delta(0, 'first', '{"x":2}', 'id-first')
        first = self.finish()
        self.assertEqual(self.completion.result()['tools'], first)
        self.assertEqual(self.completion.tools[0]['arguments'], '{"x":2}')


if __name__ == '__main__':
    unittest.main()
