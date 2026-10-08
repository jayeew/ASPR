from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import ASPECTS
from figure_pipeline.fig4_explanation_study.aggregate import GROUNDED
from figure_pipeline.fig5_robustness.aggregate import costs, usage_summary
from figure_pipeline.fig5_robustness.models import Config

from .prepare import CONDITIONS, OUT, SOURCE, canonical_sources


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save(name: str, rows: list[dict[str, Any]]) -> None:
    write(OUT / (name + '.json'), rows)
    write_csv(OUT / (name + '.csv'), rows)


def interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'mean': None, 'low': None, 'high': None}
    a = np.asarray(values)
    means = a[np.random.default_rng(20261003).integers(0, len(a), (10000, len(a)))].mean(axis=1)
    return {'n': len(a), 'mean': float(a.mean()), 'low': float(np.quantile(means, .025)),
            'high': float(np.quantile(means, .975))}


def panel_a() -> None:
    backgrounds = {r['paper_id']: r for r in read(OUT / 'paper_background.json')}
    rows = []
    for r in csv_rows(Config().source / 'paper_aspect_metrics.csv'):
        if r['condition'] == 'F' and r['paper_id'] in backgrounds:
            rows.append({**backgrounds[r['paper_id']], **r})
    assert len({(r['paper_id'], r['aspect']) for r in rows}) == len(rows) == 500
    result = []
    for dimension, group in [('group', v) for v in ('life', 'physical_engineering', 'medicine', 'earth_environment')] + [('history_group', v) for v in ('lower', 'higher')]:
        for aspect in ASPECTS:
            selected = [r for r in rows if r[dimension] == group and r['aspect'] == aspect]
            values = [float(r['grounded_coverage']) for r in selected if r['state'] == 'completed' and r['grounded_coverage']]
            result.append({'dimension': dimension, 'group': group, 'aspect': aspect,
                           'planned_n': len(selected), **interval(values)})
    save('panel_a_papers', rows)
    save('panel_a_summary', result)


def classify(part: dict[str, Any], answerability: str) -> dict[str, Any]:
    if part.get('technical_state') != 'completed':
        return {'behavior': 'technical_missing', 'grounded_answer': None, 'reasonable_abstention': None,
                'confirmed_unsupported': None, 'scientific_unresolved': None}
    response = part['response_type']
    unknown = response == 'unresolved' or part['status'] == 'unresolved'
    if response == 'substantive_answer':
        unknown |= (part['scope_correct'] is None or part['grounding'] == 'unclear'
                    or part['unsupported_definitive'] is None or part['conflicting_material_error'] is None)
    if response == 'target_abstention':
        unknown |= part['abstention_appropriateness'] == 'unresolved' or part['unsupported_definitive'] is None
    correct = (response == 'substantive_answer' and part['status'] == 'correct'
               and part['scope_correct'] is True and part['grounding'] in GROUNDED
               and part['unsupported_definitive'] is False and part['conflicting_material_error'] is False)
    reasonable = (response == 'target_abstention' and part['abstention_appropriateness'] == 'reasonable'
                  and part['unsupported_definitive'] is False and part['conflicting_material_error'] is False)
    # References remain fallible; retain candidate-specific evidence-supported appropriateness.
    behavior = ('unsupported' if part['unsupported_definitive'] is True else 'unresolved' if unknown else
                'grounded' if correct else 'abstention' if response == 'target_abstention' else 'incomplete')
    return {'behavior': behavior, 'grounded_answer': int(correct), 'reasonable_abstention': int(reasonable),
            'confirmed_unsupported': int(part['unsupported_definitive'] is True),
            'scientific_unresolved': int(unknown),
            'reference_abstention_disagreement': reasonable and answerability == 'answerable'}


def answer_parts() -> list[dict[str, Any]]:
    protocol = read(OUT / 'protocol.json')
    corrected = {(r['paper_id'], r['question_id'], r['part_id']): r['E50_answerability']
                 for r in read(OUT / 'frozen_history_groups.json')}
    result = []
    for paper in protocol['selected']:
        fixed = read(SOURCE / 'baseline/reference/papers' / f'{paper}.json')
        refs = {v['view_id']: {(p['question_id'], p['part_id']): p for p in v['parts']}
                for v in read(SOURCE / 'validated_reference' / f'{paper}.json')['views']}
        observed = {}
        path = OUT / 'aligned' / f'{paper}.json'
        mapping = read(OUT / 'mapping' / f'{paper}.json')
        if path.exists():
            for c in read(path)['candidates']:
                for q in c['questions']:
                    for p in q['parts']:
                        observed[(mapping[c['report_id']], q['question_id'], p['part_id'])] = p
        conditions = [*CONDITIONS] + (['F_REPEAT'] if paper in protocol['repeated'] else [])
        for condition in conditions:
            view = condition if condition in ('F', 'E50', 'K5') else 'F'
            for question in fixed['questions']:
                if not question['applicable']:
                    continue
                for p in question['answer_parts']:
                    qid, pid = question['question_id'], p['part_id']
                    ref = refs[view][(qid, pid)]
                    answerability = corrected.get((paper, qid, pid), ref['answerability']) if condition == 'E50' else ref['answerability']
                    part = observed.get((condition, qid, pid), {'technical_state': 'missing'})
                    result.append({**part, **classify(part, answerability), 'paper_id': paper, 'condition': condition,
                                   'aspect': question['aspect'], 'question_id': qid, 'part_id': pid,
                                   'answerability': answerability, 'reference_original_answerability': ref['answerability'],
                                   'reference_path': str(SOURCE / 'validated_reference' / f'{paper}.json'),
                                   'report_path': str(SOURCE / 'reports' / condition / f'{paper}.json')})
    assert len({(r['paper_id'], r['condition'], r['question_id'], r['part_id']) for r in result}) == len(result)
    save('answer_parts', result)
    return result


def metrics(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(r['paper_id'], r['condition']) for r in parts})
    for paper, condition in keys:
        for aspect in ['all', *ASPECTS]:
            selected = [r for r in parts if r['paper_id'] == paper and r['condition'] == condition
                        and (aspect == 'all' or r['aspect'] == aspect)]
            complete = bool(selected) and all(r['technical_state'] == 'completed' for r in selected)
            row = {'paper_id': paper, 'condition': condition, 'aspect': aspect, 'fixed_denominator': len(selected),
                   'complete': complete, 'technical_missing': sum(r['technical_state'] != 'completed' for r in selected)}
            for field in ('grounded_answer', 'reasonable_abstention', 'confirmed_unsupported', 'scientific_unresolved'):
                row[field + '_count'] = sum(r[field] == 1 for r in selected)
                row[field + '_rate'] = row[field + '_count'] / len(selected) if complete else None
            result.append(row)
    save('paper_metrics', result)
    return result


def panel_b(rows: list[dict[str, Any]]) -> None:
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in rows}
    points = []
    for r in rows:
        if r['condition'] not in ('K5', 'LUNA', 'F_REPEAT') or r['aspect'] == 'all':
            continue
        base = index[(r['paper_id'], 'F', r['aspect'])]
        complete = r['complete'] and base['complete']
        points.append({'paper_id': r['paper_id'], 'condition': r['condition'], 'aspect': r['aspect'],
                       'complete': complete, 'delta_pp': 100 * (r['grounded_answer_rate'] - base['grounded_answer_rate']) if complete else None})
    summaries = []
    for condition in ('K5', 'LUNA', 'F_REPEAT'):
        for aspect in ASPECTS:
            selected = [r for r in points if r['condition'] == condition and r['aspect'] == aspect]
            summaries.append({'condition': condition, 'aspect': aspect, 'planned_n': len(selected),
                              **interval([r['delta_pp'] for r in selected if r['complete']])})
    save('panel_b_paired', points)
    save('panel_b_summary', summaries)


def panel_c(parts: list[dict[str, Any]]) -> None:
    audit = {r['paper_id']: r for r in read(OUT / 'e50_input_audit.json')}
    index = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r for r in parts}
    rows, exclusions = [], []
    for g in read(OUT / 'frozen_history_groups.json'):
        reason = None
        pair = [index[(g['paper_id'], c, g['question_id'], g['part_id'])] for c in ('F', 'E50')]
        if not g['eligible']:
            reason = 'not_independent_history_dependent'
        elif not audit[g['paper_id']]['eligible']:
            reason = 'E50_report_or_intervention_unavailable'
        elif g['group'] == 'reference_unresolved':
            reason = 'conditional_reference_unresolved'
        elif any(r['technical_state'] != 'completed' for r in pair):
            reason = 'paired_evaluation_technical_missing'
        if reason:
            exclusions.append({**g, 'exclusion_reason': reason})
            continue
        rows.extend({**r, 'group': g['group']} for r in pair)
    summary = []
    for group in ('retained', 'lost'):
        for condition in ('F', 'E50'):
            selected = [r for r in rows if r['group'] == group and r['condition'] == condition]
            counts = Counter(r['behavior'] for r in selected)
            for behavior in ('grounded', 'abstention', 'unsupported', 'incomplete', 'unresolved'):
                summary.append({'group': group, 'condition': condition, 'behavior': behavior,
                                'n': len({r['paper_id'] for r in selected}), 'N': len(selected),
                                'count': counts[behavior], 'percent': 100 * counts[behavior] / len(selected) if selected else None,
                                'reasonable_abstention_count': sum(r['reasonable_abstention'] == 1 for r in selected)})
    save('panel_c_parts', rows)
    save('panel_c_summary', summary)
    save('panel_c_exclusions', exclusions)


def cost_table() -> list[dict[str, Any]]:
    records, _, allocated = costs(Config())
    baseline = [read(p) for p in (SOURCE / 'baseline/calls').glob('*/record.json')]
    for row in allocated:
        paper, condition = row['paper_id'], row['condition']
        new = [r for r in records if r['paper_id'] == paper and r['stage'] in ('analysis', 'reports')
               and r['method'] in (condition, condition + '_gear', condition + '_graph')]
        reused = [r for r in baseline if r['paper_id'] == paper and
                  (condition == 'F' and (r['stage'], r['method']) in (('analysis', 'gear'), ('analysis', 'graph_F'), ('reports', 'F'))
                   or condition == 'K5' and (r['stage'], r['method']) == ('analysis', 'gear'))]
        row.update(new_spending_call_seconds=usage_summary(new)['call_seconds_sum'],
                   reused_call_seconds=usage_summary(reused)['call_seconds_sum'],
                   monetary_cost_complete=False, monetary_cost=None)
    save('costs', allocated)
    return allocated


def panel_d(rows: list[dict[str, Any]], costs_rows: list[dict[str, Any]]) -> None:
    quality = {(r['paper_id'], r['condition']): r for r in rows if r['aspect'] == 'all'}
    costs_index = {(r['paper_id'], r['condition']): r for r in costs_rows}
    eligible = {r['paper_id'] for r in read(OUT / 'e50_input_audit.json') if r['eligible']}
    common = sorted(p for p in eligible if all(quality[(p, c)]['complete'] and costs_index[(p, c)]['complete_cost'] for c in CONDITIONS))
    papers, summary = [], []
    for c in CONDITIONS:
        for p in common:
            papers.append({**quality[(p, c)], 'call_seconds_ratio': costs_index[(p, c)]['call_seconds_sum'] / costs_index[(p, 'F')]['call_seconds_sum']})
        selected = [r for r in papers if r['condition'] == c]
        summary.append({'condition': c, 'n': len(common), 'paper_ids': common,
                        'time_ratio_median': float(np.median([r['call_seconds_ratio'] for r in selected])) if selected else None,
                        **{k: float(np.mean([r[k] for r in selected])) if selected else None for k in
                           ('grounded_answer_rate', 'reasonable_abstention_rate', 'confirmed_unsupported_rate', 'scientific_unresolved_rate')}})
    save('panel_d_papers', papers)
    save('panel_d_summary', summary)


def conditions() -> None:
    audit = {r['paper_id']: r for r in read(OUT / 'e50_input_audit.json')}
    result = []
    for paper in read(OUT / 'protocol.json')['selected']:
        aliases, _ = canonical_sources(read(SOURCE / 'inputs/F' / f'{paper}.json'))
        full_sources = sorted(set(aliases.values()))
        for condition in CONDITIONS:
            exists = (SOURCE / 'reports' / condition / f'{paper}.json').exists()
            result.append({'paper_id': paper, 'condition': condition, 'report_available': exists,
                           'model': 'gpt-5.6-luna' if condition == 'LUNA' else 'gpt-6.1-sol',
                           'neighbor_cap': 5 if condition == 'K5' else 10,
                           'intervention_eligible': audit[paper]['eligible'] if condition == 'E50' else True,
                           'retained_sources': audit[paper].get('retained_sources') if condition == 'E50' else full_sources,
                           'retained_fraction': audit[paper].get('retained_fraction') if condition == 'E50' else 1,
                           'missing_reason': '' if exists else 'Original generation missing; no replacement report'})
    save('conditions_materials', result)


def main() -> None:
    panel_a()
    parts = answer_parts()
    rows = metrics(parts)
    panel_b(rows)
    panel_c(parts)
    panel_d(rows, cost_table())
    conditions()
    ledger = OUT / 'call_ledger.jsonl'
    summary = {'new_calls': len(ledger.read_text().splitlines()) if ledger.exists() else 0,
               'part_states': dict(Counter(r['behavior'] for r in parts)),
               'aligned_papers': len(list((OUT / 'aligned').glob('*.json'))),
               'reference_abstention_disagreements': sum(r.get('reference_abstention_disagreement', False) for r in parts)}
    write(OUT / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
