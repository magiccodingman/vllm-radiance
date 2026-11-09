#!/usr/bin/env python3
"""Synthetic client-shaped wire regressions; never execute generated tools."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


def reject_constant(value):
    raise ValueError(f'non-JSON numeric constant: {value}')


def schema_for(value):
    """Closed synthetic fixture schema, not a schema inferred from customer data."""
    if isinstance(value, dict):
        return {'type': 'object', 'properties': {k: schema_for(v) for k, v in value.items()},
                'required': list(value), 'additionalProperties': False}
    if isinstance(value, list):
        return {'type': 'array', 'items': {'anyOf': [schema_for(v) for v in value]},
                'minItems': len(value), 'maxItems': len(value)}
    if isinstance(value, bool):
        return {'type': 'boolean'}
    if isinstance(value, int):
        return {'type': 'integer'}
    return {'type': 'string'}


PACKAGE = {'request': {'title': 'Synthetic reporting package', 'expectedVersion': 0,
    'items': [
        {'key': 'calculation', 'operation': 'Function', 'definition': {
            'sql': "SELECT category, amount FROM (VALUES ('Example A', 12), ('Example B', 7)) AS example(category, amount)",
            'outputs': [{'name': 'category', 'type': 'Text'}, {'name': 'amount', 'type': 'Integer'}]}},
        {'key': 'report', 'operation': 'Report', 'definition': {'dependencies': ['calculation'],
            'blocks': [{'id': 'heading', 'kind': 'Heading', 'column': 0, 'row': 0},
                       {'id': 'chart', 'kind': 'Bar', 'column': 0, 'row': 1},
                       {'id': 'table', 'kind': 'NextPageTable', 'column': 0, 'row': 2}]}},
        {'key': 'folder', 'operation': 'Folder', 'definition': {'reports': ['report']}}]}}


def fixtures():
    keys = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/tool-schema-multitool.json').read_text())
    tools = keys['tools']  # Same synthetic keypress shape used by the QA harness.
    package_schema = schema_for(PACKAGE)
    # Reproduces the .NET Int64 exporter boundary, including nullable typing.
    package_schema['properties']['request']['properties']['expectedVersion'] = {
        'type': ['integer', 'null'], 'minimum': 0, 'maximum': float(2**63)}
    package_tool = {'type': 'function', 'function': {'name': 'package_probe',
                    'description': 'Capture a synthetic package. No tool will execute.',
                    'parameters': package_schema}}
    cases = []
    key_calls = [dict(name='keypress', arguments=dict(reasoning='Synthetic independent call',
                 confidence=1, stepComplete=True, keys=[key])) for key in ('TAB', 'ENTER')]
    for name, choice in [('keypress_auto', 'auto'), ('keypress_parallel', 'required')]:
        cases.append((name, {'tools': tools, 'tool_choice': choice,
            'parallel_tool_calls': True, 'reasoning_effort': 'low',
            'messages': [{'role': 'system', 'content': 'Synthetic tool-serialization capture only. Nothing executes. The two calls are independent fixtures, not a sequential browser plan. Emit the exact requested calls without prose.'},
                         {'role': 'user', 'content': 'Emit these two separate keypress calls exactly:\n' + json.dumps(key_calls)}]}))
    for strict in (False, True):
        tool = copy.deepcopy(package_tool)
        if strict:
            tool['function']['strict'] = True
        cases.append(('package_strict' if strict else 'package_auto', {
            'tools': [tool], 'tool_choice': 'auto', 'parallel_tool_calls': False,
            'reasoning_effort': 'medium', 'messages': [{'role': 'user', 'content':
                'Call package_probe exactly once with these exact native nested values. Do not encode request or its nested objects as JSON strings. No tool executes.\n' + json.dumps(PACKAGE)}]}))
    return cases


def same_json_value(actual, expected):
    """Compare fixture values using JSON types, where booleans are not numbers."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(same_json_value(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(same_json_value(a, e) for a, e in zip(actual, expected)))
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def validate(case, calls, finish):
    issues = []
    if finish not in ('stop', 'tool_calls'):
        issues.append(f'incomplete finish: {finish}')
    if not calls:
        issues.append('no tool calls')
    if case == 'keypress_parallel' and len(calls) != 2:
        issues.append(f'expected two required keypress calls, received {len(calls)}')
    if case.startswith('package') and len(calls) != 1:
        issues.append('expected one package call')
    for index, call in enumerate(calls):
        try:
            f = call['function']
            a = json.loads(f['arguments'], parse_constant=reject_constant)
            if not isinstance(a, dict):
                raise ValueError('arguments must be an object')
            if case.startswith('package'):
                if f['name'] != 'package_probe' or not same_json_value(a, PACKAGE):
                    raise ValueError('package values/typed nested objects changed')
            else:
                if f['name'] != 'keypress':
                    raise ValueError('unexpected tool name')
                if not isinstance(a.get('reasoning'), str) or type(a.get('confidence')) not in (int, float):
                    raise ValueError('invalid reasoning/confidence type')
                if type(a['confidence']) is float and not math.isfinite(a['confidence']):
                    raise ValueError('non-finite confidence')
                if type(a.get('stepComplete')) is not bool:
                    raise ValueError('invalid stepComplete type')
                if not isinstance(a.get('keys'), list) or not a['keys'] or not all(isinstance(k, str) for k in a['keys']):
                    raise ValueError('invalid keys')
                if set(a) - {'reasoning', 'confidence', 'stepComplete', 'stepFailed', 'keys'}:
                    raise ValueError('unexpected fields')
                if 'stepFailed' in a and type(a['stepFailed']) is not bool:
                    raise ValueError('invalid stepFailed type')
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'call {index}: {exc}')
    return issues


def collect_stream(response, capture):
    slots, finish, done = {}, None, False
    for raw in response:
        line = raw.decode('utf-8').strip()
        capture.write(json.dumps({'sse': line}) + '\n')
        if not line.startswith('data:'):
            continue
        if line[5:].strip() == '[DONE]':
            done = True
            break
        event = json.loads(line[5:])
        if 'error' in event:
            raise ValueError('stream returned an error')
        for choice in event.get('choices', []):
            finish = choice.get('finish_reason') or finish
            for fragment in (choice.get('delta') or {}).get('tool_calls') or []:
                slot = slots.setdefault(fragment['index'], {'function': {'name': '', 'arguments': ''}})
                for key in ('name', 'arguments'):
                    slot['function'][key] += fragment.get('function', {}).get(key) or ''
    if not done:
        raise ValueError('truncated SSE: missing DONE')
    return [slots[i] for i in sorted(slots)], finish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--capture-root', required=True)
    args = p.parse_args()
    if args.runs < 1:
        p.error('--runs must be positive')
    root = Path(args.capture_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ_client-tool-gate')
    root.mkdir(parents=True, exist_ok=False)
    results = []
    with (root / 'wire.jsonl').open('x') as capture:
        for run in range(args.runs):
            for case, fixture in fixtures():
                for stream in (False, True):
                    body = dict(fixture, model=args.model, stream=stream, temperature=0.0,
                                seed=17 + run, max_completion_tokens=6144)
                    capture.write(json.dumps({'run': run, 'case': case, 'request': body}) + '\n')
                    calls, finish = [], None
                    try:
                        req = Request(args.base_url.rstrip('/') + '/chat/completions',
                                      data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                        with urlopen(req, timeout=180) as response:
                            if stream:
                                calls, finish = collect_stream(response, capture)
                            else:
                                data = json.load(response)
                                capture.write(json.dumps({'response': data}) + '\n')
                                choice = data['choices'][0]
                                calls = choice['message'].get('tool_calls') or []
                                finish = choice.get('finish_reason')
                        issues = validate(case, calls, finish)
                    except Exception as exc:
                        issues = [f'request/stream failure: {exc}']
                    result = dict(run=run, case=case, stream=stream, calls=len(calls), issues=issues, passed=not issues)
                    results.append(result)
                    capture.write(json.dumps({'result': result}) + '\n')
                    capture.flush()
                    print(json.dumps(result), flush=True)
    summary = dict(requests=len(results), passed=sum(r['passed'] for r in results),
                   failed=sum(not r['passed'] for r in results), run_dir=str(root))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return int(summary['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
