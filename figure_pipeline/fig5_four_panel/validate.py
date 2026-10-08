from __future__ import annotations

from collections import Counter

import numpy as np

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_robustness.materials import original_blocks, source_catalog

from .prepare import OUT, SOURCE


def main() -> None:
    parts = read(OUT / 'answer_parts.json')
    protocol = read(OUT / 'protocol.json')
    summary = read(OUT / 'summary.json')
    assert summary['new_calls'] <= protocol['new_request_cap'] == 30
    keys = [(r['paper_id'], r['condition'], r['question_id'], r['part_id']) for r in parts]
    assert len(keys) == len(set(keys))
    metrics = read(OUT / 'paper_metrics.json')
    for r in metrics:
        if r['aspect'] != 'all':
            continue
        other = [p for p in metrics if p['paper_id'] == r['paper_id'] and p['condition'] == r['condition'] and p['aspect'] != 'all']
        assert sum(p['fixed_denominator'] for p in other) == r['fixed_denominator']
        assert sum(p['grounded_answer_count'] for p in other) == r['grounded_answer_count']
    c = read(OUT / 'panel_c_summary.json')
    for group in ('retained', 'lost'):
        full = [r for r in c if r['group'] == group and r['condition'] == 'F']
        reduced = [r for r in c if r['group'] == group and r['condition'] == 'E50']
        assert full[0]['N'] == reduced[0]['N']
        for rows in (full, reduced):
            assert sum(r['count'] for r in rows) == rows[0]['N']
            assert np.isclose(sum(r['percent'] or 0 for r in rows), 100 if rows[0]['N'] else 0)
    d = read(OUT / 'panel_d_summary.json')
    assert all(r['paper_ids'] == d[0]['paper_ids'] for r in d)
    for r in read(OUT / 'costs.json'):
        if r['complete_cost']:
            assert np.isclose((r['new_spending_call_seconds'] or 0) + (r['reused_call_seconds'] or 0), r['call_seconds_sum'])
    violations = []
    for paper in protocol['selected']:
        aliases = source_catalog(read(SOURCE / 'inputs/F' / f'{paper}.json'))
        blocks = original_blocks(read(SOURCE / 'inputs/E50' / f'{paper}.json'))
        visible = {p['source_id'] for b in blocks for p in b['provenance']}
        for r in parts:
            if r['paper_id'] == paper and r['condition'] == 'E50' and r['grounded_answer'] == 1:
                invalid = [s for s in r['evidence_source_ids'] if s in aliases and s not in visible]
                if invalid:
                    violations.append({'paper_id': paper, 'part_id': r['part_id'], 'source_ids': invalid})
    assert not violations, violations
    result = {'state': 'passed', 'new_calls': summary['new_calls'],
              'fixed_part_rows': len(parts), 'part_states': dict(Counter(r['behavior'] for r in parts)),
              'common_cost_cohort_n': d[0]['n'], 'E50_invisible_grounding_ids': violations,
              'checks': ['unique fixed identities', 'fixed denominators across aspects', 'matched F/E50 point sets',
                         'stacked counts and percentages', 'four-condition identical cost cohort',
                         'new plus reused cost reconciliation', 'known E50 grounding source visibility', '30-request hard cap'],
              'limit': 'Structural/data checks and exact source visibility; not independent validation of scientific truth'}
    write(OUT / 'validation.json', result)
    print(result)


if __name__ == '__main__':
    main()
