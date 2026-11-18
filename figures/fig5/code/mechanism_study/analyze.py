from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_four_panel.aggregate import classify, interval

from . import costs, interventions, summarize, transfer
from .diagnostic import OUT as DIAGNOSTIC

OUT = DIAGNOSTIC.parent
DATA = OUT / 'figure_data'


def save(name: str, rows: list[dict[str, Any]]) -> None:
    write(DATA / f'{name}.json', rows)
    write_csv(DATA / f'{name}.csv', rows)


def index_metrics(folder: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(r['paper_id'], r['condition'], r['aspect']): r for r in read(folder / 'paper_metrics.json')}


def dose_response() -> None:
    index = index_metrics(interventions.OUT)
    papers = read(interventions.OUT / 'protocol.json')['gradient_papers']
    conditions = {'Historical sources': ['F', 'E75', 'E50', 'E25'],
                  'Historical parent papers in graph': ['F', 'G50', 'G25']}
    fractions = {'F': 1., 'E75': .75, 'E50': .5, 'E25': .25, 'G50': .5, 'G25': .25}
    rows, summary = [], []
    for aspect in sorted({key[2] for key in index}):
        for kind, names in conditions.items():
            common = [p for p in papers if all(index.get((p, c, aspect), {}).get('complete') for c in names)]
            for condition in names:
                for paper in common:
                    row, base = index[paper, condition, aspect], index[paper, 'F', aspect]
                    rows.append({'paper_id': paper, 'intervention': kind, 'condition': condition, 'aspect': aspect,
                                 'nominal_retained_fraction': fractions[condition],
                                 'grounded_correct_percent': 100 * row['grounded_answer'],
                                 'paired_change_percentage_points': 100 * (row['grounded_answer'] - base['grounded_answer'])})
                values = [r for r in rows if r['intervention'] == kind and r['condition'] == condition and r['aspect'] == aspect]
                for metric in ('grounded_correct_percent', 'paired_change_percentage_points'):
                    summary.append({'intervention': kind, 'condition': condition, 'aspect': aspect,
                                    'nominal_retained_fraction': fractions[condition], 'planned_n': len(papers),
                                    'metric': metric, **interval([r[metric] for r in values])})
    save('dose_papers', rows)
    save('dose_summary', summary)


def mask_repeatability() -> None:
    index = index_metrics(interventions.OUT)
    papers = read(interventions.OUT / 'protocol.json')['gradient_papers']
    rows = []
    for aspect in sorted({key[2] for key in index}):
        for paper in papers:
            names = ['F', 'E50', 'E50_SEED2', 'E50_SEED3']
            if not all(index.get((paper, c, aspect), {}).get('complete') for c in names):
                continue
            values = [100 * (index[paper, c, aspect]['grounded_answer'] - index[paper, 'F', aspect]['grounded_answer']) for c in names[1:]]
            rows.append({'paper_id': paper, 'aspect': aspect, 'masks': 3,
                         'within_paper_mean_change': float(np.mean(values)),
                         'within_paper_min_change': min(values), 'within_paper_max_change': max(values),
                         **{f'mask_{i + 1}_change': value for i, value in enumerate(values)}})
    save('repeated_mask_papers', rows)
    save('repeated_mask_summary', [{'aspect': a, 'planned_n': len(papers),
                                  **interval([r['within_paper_mean_change'] for r in rows if r['aspect'] == a])}
                                 for a in sorted({key[2] for key in index})])


def target_states() -> None:
    decisions = [(p.stem, read(p)) for p in (interventions.OUT / 'eligibility').glob('*.json')]
    index = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r
             for r in read(interventions.OUT / 'answer_parts.json')}
    names = ['F', 'NONCRITICAL', 'CRITICAL', 'RESTORE']
    rows, exclusions = [], []
    design = read(interventions.OUT / 'protocol.json')['critical_design_papers']
    for paper in design:
        decision = next((d for p, d in decisions if p == paper), {})
        if not decision.get('eligible'):
            exclusions.append({'paper_id': paper, 'reason': decision.get('reason', 'Eligibility not completed'),
                               'eligible': decision.get('eligible'), 'reference_states': decision.get('states')})
            continue
        target = decision['target']
        parts = [index.get((paper, c, target['question_id'], target['part_id'])) for c in names]
        complete = all(p and p['technical_state'] == 'completed' for p in parts)
        for condition, part in zip(names, parts, strict=True):
            if not part or part['technical_state'] != 'completed':
                state = 'Evaluation unavailable'
            elif part['confirmed_unsupported']:
                state = 'Unsupported assertion'
            elif part['grounded_answer']:
                state = 'Grounded correct answer'
            elif part['reasonable_abstention']:
                state = 'Justified withholding'
            elif part['scientific_unresolved']:
                state = 'Scientific judgment unresolved'
            else:
                state = 'Other incomplete or incorrect answer'
            rows.append({'paper_id': paper, 'condition': condition, 'state': state, 'paired_complete': complete,
                         'question_id': target['question_id'], 'part_id': target['part_id']})
    save('target_states', rows)
    save('target_exclusions', exclusions)
    paired = [r for r in rows if r['paired_complete']]
    save('target_state_counts', [{'condition': c, 'state': s, 'count': sum(r['condition'] == c and r['state'] == s for r in paired),
                                 'n': len({r['paper_id'] for r in paired})}
                                for c in names for s in sorted({r['state'] for r in paired})])


def resource_tradeoff() -> None:
    rows, summaries = [], []
    for label, folder in [('Model-stage diagnostic', DIAGNOSTIC), ('Unified main cohort', transfer.OUT)]:
        index = index_metrics(folder)
        costs_index = {(r['paper_id'], r['condition']): r for r in read(folder / 'generation_costs.json')}
        protocol = read(folder / 'protocol.json')
        conditions = list(protocol['conditions'])
        common = [p for p in protocol['papers'] if all(index.get((p, c, 'all'), {}).get('complete')
                  and costs_index.get((p, c), {}).get('complete_cost') for c in conditions)]
        baseline = 'original_analysis_original_writer'
        for condition in conditions:
            current = []
            for paper in common:
                cost, base = costs_index[paper, condition], costs_index[paper, baseline]
                quality, qbase = index[paper, condition, 'all'], index[paper, baseline, 'all']
                current.append({'cohort': label, 'paper_id': paper, 'condition': condition,
                                'processed_token_ratio': cost['total_input_and_output_tokens'] / base['total_input_and_output_tokens'],
                                'cumulative_call_time_ratio': cost['call_seconds_sum'] / base['call_seconds_sum'],
                                'coverage_change_percentage_points': 100 * (quality['grounded_answer'] - qbase['grounded_answer']),
                                'grounded_correct_percent': 100 * quality['grounded_answer']})
            rows.extend(current)
            for metric in ('processed_token_ratio', 'cumulative_call_time_ratio', 'coverage_change_percentage_points', 'grounded_correct_percent'):
                values = [r[metric] for r in current]
                summaries.append({'cohort': label, 'condition': condition, 'metric': metric,
                                  'planned_n': len(protocol['papers']), **interval(values),
                                  'median': float(np.median(values)) if values else None})
    save('resource_papers', rows)
    save('resource_summary', summaries)


def independent_review() -> None:
    primary = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r
               for r in read(DIAGNOSTIC / 'answer_parts.json')}
    parts = read(DIAGNOSTIC / 'independent_error_parts.json')
    rows = []
    for p in parts:
        key = p['paper_id'], p['condition'], p['question_id'], p['part_id']
        first, second = primary[key], classify(p, 'answerable')
        rows.append({'paper_id': p['paper_id'], 'condition': p['condition'], 'question_id': p['question_id'],
                     'part_id': p['part_id'], 'primary_grounded': first['grounded_answer'],
                     'independent_grounded': second['grounded_answer'],
                     'agreement': first['grounded_answer'] == second['grounded_answer'],
                     'answer_completeness': p['answer_completeness'], 'error_categories': p['error_categories']})
    save('independent_review_parts', rows)
    summary = []
    for condition in sorted({r['condition'] for r in rows}):
        selected = [r for r in rows if r['condition'] == condition]
        summary.append({'condition': condition, 'papers': len({r['paper_id'] for r in selected}),
                        'answer_parts': len(selected), 'agreement_fraction': sum(r['agreement'] for r in selected) / len(selected),
                        'completeness': dict(Counter(r['answer_completeness'] for r in selected)),
                        'errors': dict(Counter(e for r in selected for e in r['error_categories'])),
                        'human_review': False})
    save('independent_review_summary', summary)
    paired = []
    for paper in sorted({r['paper_id'] for r in rows}):
        for judge in ('primary_grounded', 'independent_grounded'):
            values = {}
            for condition in sorted({r['condition'] for r in rows}):
                current = [r[judge] for r in rows if r['paper_id'] == paper and r['condition'] == condition]
                values[condition] = float(np.mean(current))
            baseline = values['original_analysis_original_writer']
            paired.extend({'paper_id': paper, 'judge': judge, 'condition': c,
                           'delta_percentage_points': 100 * (value - baseline)}
                          for c, value in values.items() if c != 'original_analysis_original_writer')
    save('independent_review_paired', paired)
    save('independent_review_paired_summary', [{'judge': j, 'condition': c,
          **interval([r['delta_percentage_points'] for r in paired if r['judge'] == j and r['condition'] == c])}
         for j, c in sorted({(r['judge'], r['condition']) for r in paired})])


def scenarios() -> None:
    backgrounds = read(OUT.parent / 'fig5_four_panel/paper_background.json')
    groups = {'life': 'Life sciences', 'physical_engineering': 'Physical sciences and engineering',
              'medicine': 'Medicine', 'earth_environment': 'Earth and environmental sciences'}
    rows, summary = [], []
    for label, folder, condition in [('Analysis-model replacement', transfer.OUT, 'replacement_analysis_original_writer'),
                                     ('Half of historical sources retained', interventions.OUT, 'E50')]:
        changes = [r for r in read(folder / 'paired_changes.json')
                   if r['condition'] == condition and r['aspect'] == 'all' and r['metric'] == 'grounded_answer']
        for group, name in groups.items():
            papers = {r['paper_id'] for r in backgrounds if r['group'] == group}
            selected = [r for r in changes if r['paper_id'] in papers]
            rows.extend({**r, 'scenario': name, 'comparison': label} for r in selected)
            summary.append({'scenario': name, 'comparison': label, 'planned_n': len(papers),
                            **interval([r['delta_percentage_points'] for r in selected])})
    save('scenario_papers', rows)
    save('scenario_summary', summary)


def baseline_protocol_agreement() -> None:
    single = {(r['paper_id'], r['question_id'], r['part_id']): r for r in read(interventions.OUT / 'answer_parts.json')
              if r['condition'] == 'F' and r['technical_state'] == 'completed'}
    rows = []
    for part in read(transfer.OUT / 'answer_parts.json'):
        if part['condition'] != 'original_analysis_original_writer' or part['technical_state'] != 'completed':
            continue
        key = part['paper_id'], part['question_id'], part['part_id']
        if key in single:
            other = single[key]
            rows.append({'paper_id': key[0], 'question_id': key[1], 'part_id': key[2],
                         'joint_candidate_grounded': part['grounded_answer'],
                         'single_candidate_grounded': other['grounded_answer'],
                         'agreement': part['grounded_answer'] == other['grounded_answer']})
    save('baseline_protocol_parts', rows)
    joint_index, single_index = index_metrics(transfer.OUT), index_metrics(interventions.OUT)
    paired = []
    for paper in read(transfer.OUT / 'protocol.json')['papers']:
        joint = joint_index.get((paper, 'original_analysis_original_writer', 'all'))
        alone = single_index.get((paper, 'F', 'all'))
        if joint and alone and joint['complete'] and alone['complete']:
            current = [r for r in rows if r['paper_id'] == paper]
            paired.append({'paper_id': paper, 'coverage_difference_percentage_points': 100 * (joint['grounded_answer'] - alone['grounded_answer']),
                           'part_agreement_fraction': sum(r['agreement'] for r in current) / len(current)})
    save('baseline_protocol_papers', paired)
    save('baseline_protocol_summary', [{'metric': m, **interval([r[m] for r in paired])}
                                      for m in ('coverage_difference_percentage_points', 'part_agreement_fraction')])


def main() -> None:
    interventions.aggregate()
    transfer.aggregate()
    summarize.main()
    costs.diagnostic_costs()
    costs.transfer_costs()
    dose_response()
    mask_repeatability()
    target_states()
    resource_tradeoff()
    independent_review()
    scenarios()
    baseline_protocol_agreement()


if __name__ == '__main__':
    main()
