#!/usr/bin/env python3
"""Summarize matched v1-006 tasks, preserving the reused/new-task distinction."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'env'))
from get_score import is_success


def wilson(success, count):
    z = 1.959963984540054
    p = success / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [100 * (center - half), 100 * (center + half)]


def error_type(record):
    reason = record.get('failure_reason') or ''
    if not reason:
        return 'none'
    if 'maximum context length' in reason or 'exceeds the model' in reason:
        return 'context_limit'
    return reason.split(':')[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'outputs/eval/dev500-manifest.json').read_text())
    reused = set(manifest['reused_task_ids'])
    added = set(manifest['added_task_ids'])
    groups = {'all500': reused | added, 'added300': added, 'reused200': reused}
    report = {'experiment': 'v1-006', 'checked_at': datetime.now(timezone.utc).isoformat(),
              'manifest': manifest, 'models': {}, 'pairs': {}}
    outcomes = {}
    for lane in manifest['lanes']:
        directory = ROOT / 'outputs/eval' / lane['batch_name']
        records = []
        for path in directory.glob('*.json'):
            if not path.stem.isdigit():
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                if args.progress:
                    continue
                raise
            if record['status'] in ('finished', 'failed'):
                records.append(record)
        label = lane['label']
        if args.progress:
            state_path = ROOT / 'outputs/eval/v1-006' / f'{label}-state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            print(json.dumps({'label': label, 'complete': len(records), 'added_complete': sum(r['task_id'] in added for r in records),
                              'state': state.get('status'), 'errors': dict(Counter(error_type(r) for r in records if r['task_id'] in added))}), flush=True)
            continue
        assert len(records) == len({r['task_id'] for r in records}) == 500, label
        assert {r['task_id'] for r in records} == groups['all500'], label
        outcomes[label] = {r['task_id']: is_success(r) for r in records}
        results = {}
        for group, ids in groups.items():
            selected = [r for r in records if r['task_id'] in ids]
            success = sum(is_success(r) for r in selected)
            results[group] = {'count': len(selected), 'success': success, 'success_pct': 100 * success / len(selected),
                              'wilson_95_pct': wilson(success, len(selected)),
                              'statuses': dict(Counter(r['status'] for r in selected)),
                              'failure_reasons': dict(Counter((r.get('failure_reason') or 'none').split(':')[0] for r in selected)),
                              'error_types': dict(Counter(error_type(r) for r in selected)),
                              'prompt_tokens': sum(r.get('token_usage_summary', {}).get('prompt_tokens', 0) or 0 for r in selected),
                              'completion_tokens': sum(r.get('token_usage_summary', {}).get('completion_tokens', 0) or 0 for r in selected),
                              'unknown_usage_requests': sum(r.get('token_usage_summary', {}).get('requests_with_unknown_usage', 0) or 0 for r in selected)}
        report['models'][label] = results
    if args.progress:
        return
    for first, second in [('a-ep2', 'b-ep2'), ('base', 'a-ep2'), ('base', 'b-ep2')]:
        report['pairs'][f'{first}_vs_{second}'] = {}
        for group, ids in groups.items():
            first_only = sum(outcomes[first][i] and not outcomes[second][i] for i in ids)
            second_only = sum(outcomes[second][i] and not outcomes[first][i] for i in ids)
            discordant = first_only + second_only
            p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(first_only, second_only) + 1)) / 2 ** discordant) if discordant else 1.0
            report['pairs'][f'{first}_vs_{second}'][group] = {'first_only': first_only, 'second_only': second_only,
                                                           'mcnemar_exact_two_sided_p': p}
    path = ROOT / 'outputs/eval/dev500-results.json'
    report['task_outcomes'] = outcomes
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'models': report['models'], 'a_vs_b': report['pairs']['a-ep2_vs_b-ep2']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
