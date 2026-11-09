#!/usr/bin/env python3
"""CPU negative controls for the client-shaped wire gate itself."""
import io
import json
import unittest
from run_client_tool_gate import PACKAGE, collect_stream, fixtures, validate


def call(args=None, name='keypress'):
    if args is None:
        args = {'reasoning': 'Synthetic', 'confidence': 1, 'stepComplete': True, 'keys': ['TAB']}
    return {'function': {'name': name, 'arguments': json.dumps(args)}}


class GateContracts(unittest.TestCase):
    def test_valid_parallel(self):
        self.assertEqual(validate('keypress_parallel', [call(), call()], 'tool_calls'), [])

    def test_bad_first_and_second_calls(self):
        for index in (0, 1):
            calls = [call(), call()]
            calls[index]['function']['arguments'] = '{"reasoning": broken'
            self.assertTrue(validate('keypress_parallel', calls, 'tool_calls'))

    def test_types_unknown_tool_missing_fields(self):
        bad = [call({}, 'keypress'), call(name='unknown'), call({'keys': 'TAB'}),
               call({'reasoning': 'x', 'confidence': True, 'stepComplete': 'true', 'keys': ['TAB']})]
        for value in bad:
            self.assertTrue(validate('keypress_auto', [value], 'tool_calls'))

    def test_missing_and_truncated(self):
        self.assertTrue(validate('keypress_auto', [], 'stop'))
        self.assertTrue(validate('keypress_auto', [call()], 'length'))
        self.assertTrue(validate('keypress_parallel', [call()], 'tool_calls'))

    def test_package_native_and_stringified(self):
        self.assertEqual(validate('package_auto', [call(PACKAGE, 'package_probe')], 'tool_calls'), [])
        for bad in ({'request': json.dumps(PACKAGE['request'])}, {'request': {'items': []}}):
            self.assertTrue(validate('package_auto', [call(bad, 'package_probe')], 'tool_calls'))

    def test_actual_auto_and_int64_fixture(self):
        cases = dict(fixtures())
        auto = cases['package_auto']
        self.assertEqual(auto['tool_choice'], 'auto')
        self.assertNotIn('strict', auto['tools'][0]['function'])
        bound = auto['tools'][0]['function']['parameters']['properties']['request']['properties']['expectedVersion']
        self.assertIsInstance(bound['maximum'], float)
        self.assertEqual(bound['maximum'], float(2**63))
        self.assertTrue(cases['package_strict']['tools'][0]['function']['strict'])

    def test_stream_fragments_every_call(self):
        calls = [call(), call()]
        lines = []
        for index, c in enumerate(calls):
            arg = c['function']['arguments']
            for f in ({'name': 'keypress', 'arguments': arg[:10]}, {'arguments': arg[10:]}):
                event = {'choices': [{'delta': {'tool_calls': [{'index': index, 'function': f}]}}]}
                lines.append(('data: ' + json.dumps(event) + '\n').encode())
        lines.extend([b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n', b'data: [DONE]\n'])
        reconstructed, finish = collect_stream(lines, io.StringIO())
        self.assertEqual(reconstructed, calls)
        self.assertEqual(validate('keypress_parallel', reconstructed, finish), [])
        with self.assertRaises(ValueError):
            collect_stream(lines[:-1], io.StringIO())
        with self.assertRaises(ValueError):
            collect_stream([b'data: {"error":"synthetic"}\n'], io.StringIO())


if __name__ == '__main__':
    unittest.main()
