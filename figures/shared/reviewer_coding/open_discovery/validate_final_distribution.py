"""Independent recount of the published matrix and all group denominators."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name: str) -> list[dict]:
    with (HERE / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    matrix = rows('paper_theme_matrix.csv')
    stats = rows('theme_statistics.csv')
    grouped = rows('theme_by_group.csv')
    book = json.loads((HERE / 'complete_screening/theme_codebook.json').read_text())
    theme_ids = {r['theme_id'] for r in book}
    reports = {r['pilot_id']: r for r in map(json.loads, (HERE / 'raw_reports.jsonl').read_text().splitlines())}
    assert len(matrix) == len(reports) == 994
    assert {r['pilot_id'] for r in matrix} == set(reports)
    assert len(theme_ids) == 100
    assert all('cohort' not in r for r in matrix + stats + grouped)
    for r in matrix:
        assert all(r[t] in ('0', '1') for t in theme_ids)
    for s in stats:
        part = matrix
        count = sum(int(r[s['theme_id']]) for r in part)
        assert count == int(s['present_papers'])
        assert abs(count * 100 / len(part) - float(s['percentage'])) < 1e-9
    for g in grouped:
        part = [r for r in matrix if reports[r['pilot_id']][g['group_dimension']] == g['group_value']]
        assert len(part) == int(g['completed_papers'])
        count = sum(int(r[g['theme_id']]) for r in part)
        assert count == int(g['present_papers'])
        assert abs(count * 100 / len(part) - float(g['percentage'])) < 1e-9
    mapped = json.loads((HERE / 'theme_legacy_comparison.json').read_text())['themes']
    nav = json.loads((HERE / 'theme_navigation.json').read_text())['groups']
    assert {r['theme_id'] for r in mapped} == theme_ids
    assert Counter(t for r in nav for t in r['theme_ids']) == Counter({t: 1 for t in theme_ids})
    evidence = [json.loads(l) for l in (HERE / 'theme_evidence.jsonl').read_text().splitlines()]
    active = [r for r in evidence if r['included'] and r['kind'] == 'match']
    pairs = Counter((r['pilot_id'], r['theme_id']) for r in active)
    expected = {(r['pilot_id'], t) for r in matrix for t in theme_ids if r[t] == '1'}
    assert set(pairs) == expected and all(n == 1 for n in pairs.values())
    result = dict(status='passed', papers=994, themes=100, overall_rows=len(stats),
                  group_rows=len(grouped), active_source_anchored_pairs=len(active),
                  journal_denominators=dict(Counter(r['journal'] for r in reports.values())),
                  field_denominators=dict(Counter(r['field'] for r in reports.values())),
                  checks=['exact eligible roster', 'pooled population denominator', 'all theme sums and percentages',
                          'all journal and field denominators/sums/percentages', 'unique paper-theme evidence',
                          'complete posthoc mapping and navigation'],
                  files_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ['paper_theme_matrix.csv', 'theme_statistics.csv',
                                             'theme_by_group.csv', 'theme_evidence.jsonl']})
    (HERE / 'final_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'papers', 'themes', 'group_rows', 'active_source_anchored_pairs']}))


if __name__ == '__main__':
    main()
