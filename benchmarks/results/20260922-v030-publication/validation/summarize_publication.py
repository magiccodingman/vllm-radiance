"""Render existing BetterBench statistics without changing their definitions."""
import hashlib
import json
import re
import sys
from pathlib import Path
from betterbench.report import single_rows, combined_score, concurrency_rows, prefill_rows

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'final-runs'
summary = {'methodology': 'BetterBench v0.2.2 / v1 / standard.json',
           'runtime_source': '487a62e829f16d35d86a0e5b6790050ed62a4c24',
           'image': 'sha256:6e4f456ab8617ebad12de4da50d297b96828365cab02bfd38a85d5d6d4b536db',
           'version': '1.1.0-rc1.vllm0.30.0',
           'vllm_commit': 'ced6857afa0ea7b2e3f0846a62e1394e90f15607',
           'production': 'STOPPED; original unless-stopped policy retained', 'lanes': {}}
for lane in ('non-spec', 'mtp-k4', 'dflash-k5', 'dflash-k7'):
    directory = root / lane
    path = directory / 'betterbench/results.json'
    if not path.exists():
        summary['lanes'][lane] = {'status': 'NOT_COMPLETE'}
        continue
    results = json.loads(path.read_text())
    records = [r for rows in results['single_stream'].values() for r in rows]
    concurrency = concurrency_rows(results)
    prefill = prefill_rows(results)
    success = (len(records) == 80 and all(r.get('ok') for r in records)
               and len(concurrency) == 4 and all(c['ok'] == c['requests'] == 24 for c in concurrency)
               and len(prefill) == 3 and all(not p['skipped'] for p in prefill)
               and all(len(p.get('pp_tps', [])) == 4 for p in results['prefill']))
    row = {'status': 'PERFORMANCE_COMPLETE' if success else 'INCOMPLETE_OR_FAILED_REQUESTS',
           'single_requests': len(records), 'single_success': sum(bool(r.get('ok')) for r in records),
           'weighted': combined_score(results), 'categories': single_rows(results),
           'concurrency': concurrency, 'prefill': prefill,
           'results_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'manifest': json.loads((directory / 'manifest.json').read_text()),
           'runtime': json.loads((directory / 'runtime-owners.json').read_text())}
    tool = directory / 'tool-schema-gate/summary.json'
    row['tool_schema'] = json.loads(tool.read_text()) if tool.exists() else {'status': 'NOT_RUN'}
    baseline = json.loads((root / 'non-spec/raw/correctness_fixed.json').read_text())
    actual = json.loads((directory / 'raw/correctness_fixed.json').read_text())
    row['fixed_outputs'] = {'requests': len(actual['generated_texts']),
        'matches_non_spec': sum(a == b and n == m for a, b, n, m in zip(
            baseline['generated_texts'], actual['generated_texts'],
            baseline['output_lens'], actual['output_lens'], strict=True)),
        'comparison': 'Exact generated text and reported token count; no retokenization'}
    row['qualification'] = ('FAILED_TOOL_QUALIFICATION' if not row['tool_schema'].get('pass')
        else 'NON_SPEC_CONTROL' if lane == 'non-spec' else 'EXPERIMENTAL_EQUIVALENCE_FAIL')
    end = directory / 'metrics-benchmark-end.prom'
    if end.exists() and lane != 'non-spec':
        def counters(path):
            return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
                for line in path.read_text().splitlines()
                if line.startswith('vllm:spec_decode') and '_created' not in line}
        begin, finish = counters(directory / 'metrics-benchmark-start.prom'), counters(end)
        delta = {key: value - begin.get(key, 0) for key, value in finish.items()}
        def value(name):
            return sum(v for key, v in delta.items() if key.startswith('vllm:' + name + '_total{'))
        drafts = value('spec_decode_num_drafts')
        proposed = value('spec_decode_num_draft_tokens')
        accepted = value('spec_decode_num_accepted_tokens')
        row['acceptance'] = {'scope': 'BetterBench only, including standard warmups',
            'drafts': drafts, 'proposed': proposed, 'accepted': accepted,
            'fraction': accepted / proposed, 'proposals_per_draft': proposed / drafts,
            'accepted_plus_bonus_per_draft': 1 + accepted / drafts, 'counter_deltas': delta}
    elif lane == 'mtp-k4':
        intervals = re.findall(r'Accepted: (\d+) tokens, Drafted: (\d+) tokens',
                              (directory / 'logs/server.log').read_text())
        accepted, proposed = map(sum, zip(*((int(a), int(p)) for a, p in intervals)))
        row['acceptance'] = {'scope': 'Sum of logged server intervals; includes correctness/tools; not benchmark-isolated',
            'accepted': accepted, 'proposed': proposed, 'fraction': accepted / proposed,
            'intervals': len(intervals), 'limitation': 'Original wrapper exited on failed tool gate before final Prometheus export'}
    metrics = directory / 'metrics-final.prom'
    if metrics.exists():
        row['speculative_counters_whole_server_lifetime'] = [line for line in metrics.read_text().splitlines()
            if line.startswith('vllm:spec_decode') and not '_created' in line]
    summary['lanes'][lane] = row
output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('publication-summary.json')
output.write_text(json.dumps(summary, indent=2))
print(json.dumps({name: {k: v for k, v in row.items() if k in
    ('status', 'weighted', 'concurrency', 'prefill', 'tool_schema')}
    for name, row in summary['lanes'].items()}, indent=2))
