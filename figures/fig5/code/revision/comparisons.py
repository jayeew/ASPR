from __future__ import annotations

from collections import defaultdict
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read

from .aggregate import interval, scored
from .models import Config


def export_paired_metrics(config: Config, metrics: list[dict[str, Any]]) -> None:
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in metrics}
    rows = []
    names = ('q_fixed', 'answer_coverage', 'unsupported_burden', 'answered_risk',
             'reasonable_abstention_sensitivity', 'unnecessary_abstention_rate')
    for current in metrics:
        if current['condition'] in ('F', 'E'):
            continue
        baseline = index.get((current['paper_id'], 'F', current['aspect']))
        for name in names:
            a, b = baseline.get(name) if baseline else None, current[name]
            rows.append({k: current[k] for k in ('paper_id', 'split', 'condition', 'aspect')} |
                        {'metric': name, 'baseline': a, 'perturbed': b,
                         'paired_delta': b - a if a is not None and b is not None else None})
    write_csv(config.output / 'paired_metrics.csv', rows)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ('split', 'condition', 'aspect', 'metric'))].append(row)
    summaries = [dict(zip(('split', 'condition', 'aspect', 'metric'), key), planned_papers=len(values),
                      **interval([r['paired_delta'] for r in values if r['paired_delta'] is not None], config.seed))
                 for key, values in sorted(groups.items())]
    write_csv(config.output / 'paired_metrics_summary.csv', summaries)


def export_independent_review(config: Config, primary: list[dict[str, Any]]) -> None:
    index = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r for r in primary}
    rows = []
    fields = ('response_type', 'status', 'scope_correct', 'grounding', 'unsupported_definitive',
              'conflicting_material_error', 'report_quotes', 'abstention_quotes', 'unsupported_quotes',
              'evidence_source_ids', 'reason', 'support_reason')
    for path in sorted((config.output / 'independent_aligned').glob('*/*.json')):
        mapping = read(config.output / 'independent_mapping' / path.parent.name / path.name)
        for candidate in read(path)['candidates']:
            condition = mapping[candidate['report_id']]
            for question in candidate['questions']:
                for part in question['parts']:
                    key = (path.stem, condition, question['question_id'], part['part_id'])
                    first = index.get(key, {'technical_state': 'missing'})
                    second = dict(part)
                    expected = {'target_abstention': True, 'substantive_answer': False, 'omission': False, 'unresolved': None}
                    if second.get('explicitly_abstains') is not expected.get(second.get('response_type')):
                        second['technical_state'] = 'invalid_response_taxonomy'
                    ready = first['technical_state'] == second['technical_state'] == 'completed'
                    score = scored(second)
                    row = dict(zip(('paper_id', 'condition', 'question_id', 'part_id'), key))
                    row.update(primary_technical_state=first['technical_state'], independent_technical_state=second['technical_state'],
                               primary_correct=first.get('correct'), independent_correct=score['correct'],
                               correct_agreement=first.get('correct') == score['correct'] if ready else None,
                               response_agreement=first.get('response_type') == second.get('response_type') if ready else None,
                               independent_source=str(path))
                    for field in fields:
                        row['primary_' + field], row['independent_' + field] = first.get(field), second.get(field)
                    rows.append(row)
    write_csv(config.output / 'independent_review_comparison.csv', rows)
