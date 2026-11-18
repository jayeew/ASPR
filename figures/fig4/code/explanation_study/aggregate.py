from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES

from .input_information import input_information
from .models import ASPECTS, Config

GROUNDED = {'independent_history', 'native_graph', 'derived_visible_graph'}
COMPARISONS = {
    '加入GEAR（完整系统−仅Graph）': ('G', None),
    '加入Graph（完整系统−仅GEAR）': ('E', None),
    '联合图（完整系统−移除联合图）': ('F_noJ', 'joint_contribution'),
    '结构数值（完整系统−屏蔽结构数值）': ('F_noM', 'structural_resolution'),
    '引用路径（完整系统−屏蔽引用路径）': ('F_noP', 'citation_contact'),
}


def part_rows(config: Config, report_set: str, paper: str, reference: dict[str, Any]) -> list[dict[str, Any]]:
    mapping_path = config.output / 'evaluation_mapping' / report_set / f'{paper}.json'
    mapping = read(mapping_path) if mapping_path.exists() else {}
    unresolved_path = config.output / 'unresolved_tasks.json'
    upstream = [r for r in read(unresolved_path) if r['paper_id'] == paper] if unresolved_path.exists() else []
    rows = []
    for aspect in ASPECTS:
        path = config.output / 'aligned_evaluation' / report_set / aspect / f'{paper}.json'
        observed = {}
        for candidate in read(path)['candidates'] if path.exists() else []:
            for q in candidate['questions']:
                for part in q['parts']:
                    observed[(mapping[candidate['report_id']], q['question_id'], part['part_id'])] = part
        for q in reference['questions']:
            if q['aspect'] != aspect or not q['applicable']:
                continue
            for expected in q['answer_parts']:
                for condition in CONDITIONS:
                    value = observed.get((condition, q['question_id'], expected['part_id']), {})
                    row = {'report_set': report_set, 'paper_id': paper, 'condition': condition,
                           'configuration': NAMES[condition], 'aspect': aspect, 'aspect_name': ASPECTS[aspect],
                           'question_id': q['question_id'], 'part_id': expected['part_id'],
                           'question': q['question'], 'expected_content': expected['expected_content'],
                           'status': value.get('status'), 'scope_correct': value.get('scope_correct'),
                           'grounding': value.get('grounding'),
                           'technical_state': value.get('technical_state', 'missing_judgment'),
                           'report_segment_ids': value.get('report_segment_ids', []),
                           'report_quotes': value.get('report_quotes', []),
                           'evidence_source_ids': value.get('evidence_source_ids', []),
                           'reason': value.get('reason', ''), 'reference_evidence': expected['evidence']}
                    row['correct'] = int(row['status'] == 'correct' and row['scope_correct'] is True)
                    row['grounded_correct'] = int(row['correct'] and row['grounding'] in GROUNDED)
                    if not value and report_set == 'new' and upstream:
                        row['technical_state'] = 'not_evaluated_incomplete_seven_conditions'
                        row['reason'] = 'Full Graph input interpretation was restricted by provider; incomplete seven-condition comparison. Existing partial reports are preserved.'
                    if row['technical_state'] != 'completed':
                        row['correct'] = row['grounded_correct'] = None
                    rows.append(row)
    return rows


def paper_rows(parts: list[dict[str, Any]], references: dict[str, Any]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in parts:
        groups[(row['report_set'], row['paper_id'], row['condition'], row['aspect'])].append(row)
    rows = []
    for report_set in ('existing', 'new'):
        for paper, reference in references.items():
            if not reference['graph_useful']:
                continue
            for condition in CONDITIONS:
                for aspect in ASPECTS:
                    values = groups[(report_set, paper, condition, aspect)]
                    complete = bool(values) and all(v['technical_state'] == 'completed' for v in values)
                    counts = Counter(v['status'] for v in values if v['technical_state'] == 'completed')
                    row = {'report_set': report_set, 'paper_id': paper, 'condition': condition,
                           'configuration': NAMES[condition], 'aspect': aspect, 'aspect_name': ASPECTS[aspect],
                           'state': 'completed' if complete else 'technical_incomplete' if values else 'not_applicable',
                           'fixed_denominator': len(values),
                           'evaluated_parts': sum(v['technical_state'] == 'completed' for v in values),
                           'correct_parts': sum(v['correct'] or 0 for v in values) if complete else None,
                           'grounded_correct_parts': sum(v['grounded_correct'] or 0 for v in values) if complete else None,
                           'incorrect_parts': counts['incorrect'] if complete else None,
                           'unwritten_parts': counts['not_written'] if complete else None,
                           'unresolved_parts': counts['unresolved'] if complete else None}
                    row['content_coverage'] = row['correct_parts'] / len(values) if complete else None
                    row['grounded_coverage'] = row['grounded_correct_parts'] / len(values) if complete else None
                    rows.append(row)
    return rows


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['report_set'], row['condition'], row['aspect'])].append(row)
    results = []
    for (report_set, condition, aspect), values in groups.items():
        complete = [v for v in values if v['state'] == 'completed']
        results.append({'report_set': report_set, 'condition': condition, 'configuration': NAMES[condition],
                        'aspect': aspect, 'aspect_name': ASPECTS[aspect], 'complete_papers': len(complete),
                        'not_applicable_papers': sum(v['state'] == 'not_applicable' for v in values),
                        'technical_incomplete_papers': sum(v['state'] == 'technical_incomplete' for v in values),
                        'content_coverage': mean(v['content_coverage'] for v in complete) if complete else None,
                        'grounded_coverage': mean(v['grounded_coverage'] for v in complete) if complete else None,
                        'correct_parts': sum(v['correct_parts'] for v in complete),
                        'grounded_correct_parts': sum(v['grounded_correct_parts'] for v in complete),
                        'fixed_denominator': sum(v['fixed_denominator'] for v in complete),
                        'incorrect_parts': sum(v['incorrect_parts'] for v in complete),
                        'unwritten_parts': sum(v['unwritten_parts'] for v in complete),
                        'unresolved_parts': sum(v['unresolved_parts'] for v in complete)})
    return results


def contrasts(rows: list[dict[str, Any]], parts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(r['report_set'], r['paper_id'], r['condition'], r['aspect']): r for r in rows}
    part_index = {(r['report_set'], r['paper_id'], r['condition'], r['part_id']): r for r in parts}
    effects, changes = [], []
    for full in [r for r in rows if r['condition'] == 'F']:
        for name, (other, focused_aspect) in COMPARISONS.items():
            if focused_aspect is not None and focused_aspect != full['aspect']:
                continue
            comparison = index[(full['report_set'], full['paper_id'], other, full['aspect'])]
            valid = full['state'] == comparison['state'] == 'completed'
            effects.append({k: full[k] for k in ('report_set', 'paper_id', 'aspect', 'aspect_name')} | {
                'contrast': name, 'comparison_configuration': NAMES[other], 'complete_pair': valid,
                'content_delta': full['content_coverage'] - comparison['content_coverage'] if valid else None,
                'grounded_delta': full['grounded_coverage'] - comparison['grounded_coverage'] if valid else None,
                'full_coverage': full['content_coverage'], 'comparison_coverage': comparison['content_coverage']})
            for part in [p for p in parts if (p['report_set'], p['paper_id'], p['condition'], p['aspect']) ==
                         (full['report_set'], full['paper_id'], 'F', full['aspect'])]:
                before = part_index[(part['report_set'], part['paper_id'], other, part['part_id'])]
                if part['correct'] is None or before['correct'] is None:
                    continue
                if (part['correct'], part['grounded_correct']) == (before['correct'], before['grounded_correct']):
                    continue
                change = ('gained' if part['correct'] else 'lost') if part['correct'] != before['correct'] else (
                    'grounding_gained' if part['grounded_correct'] else 'grounding_lost')
                changes.append({k: part[k] for k in ('report_set', 'paper_id', 'aspect', 'aspect_name',
                                                      'part_id', 'expected_content', 'reference_evidence')} | {
                    'contrast': name, 'change': change,
                    'full_status': part['status'], 'comparison_status': before['status'],
                    'full_grounding': part['grounding'], 'comparison_grounding': before['grounding'],
                    'full_quotes': part['report_quotes'], 'comparison_quotes': before['report_quotes'],
                    'full_reason': part['reason'], 'comparison_reason': before['reason']})
    return effects, changes


def effect_summary(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in effects:
        groups[(row['report_set'], row['contrast'], row['aspect'])].append(row)
    result = []
    for (report_set, contrast, aspect), values in groups.items():
        paired = [v for v in values if v['complete_pair']]
        result.append({'report_set': report_set, 'contrast': contrast, 'aspect': aspect,
                       'aspect_name': ASPECTS[aspect], 'paired_papers': len(paired),
                       'content_delta': mean(v['content_delta'] for v in paired) if paired else None,
                       'grounded_delta': mean(v['grounded_delta'] for v in paired) if paired else None,
                       'content_improved_papers': sum(v['content_delta'] > 0 for v in paired),
                       'content_tied_papers': sum(v['content_delta'] == 0 for v in paired),
                       'content_reduced_papers': sum(v['content_delta'] < 0 for v in paired)})
    return result


def aggregate(config: Config) -> None:
    input_information(config)
    references = {p.stem: read(p) for p in (config.output / 'reference/papers').glob('*.json')}
    parts = [row for group in ('existing', 'new') for p, r in references.items() if r['graph_useful']
             for row in part_rows(config, group, p, r)]
    rows = paper_rows(parts, references)
    effects, changes = contrasts(rows, parts)
    write_csv(config.output / 'answer_parts.csv', parts)
    write(config.output / 'answer_parts.json', parts)
    write_csv(config.output / 'paper_aspect_metrics.csv', rows)
    write_csv(config.output / 'configuration_summary.csv', summaries(rows))
    write_csv(config.output / 'paired_effects.csv', effects)
    write_csv(config.output / 'paired_effects_summary.csv', effect_summary(effects))
    write(config.output / 'paired_changes.json', changes)
    write_csv(config.output / 'graph_screening.csv', [
        {'paper_id': p, 'included': r['graph_useful'], 'reason': r['eligibility_reason'],
         'not_applicable_questions': [q['question_id'] for q in r['questions'] if not q['applicable']]}
        for p, r in references.items()])
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    usage = Counter()
    for record in records:
        usage.update({k: v for k, v in (record.get('usage') or {}).items() if isinstance(v, (int, float))})
    lengths = []
    for report_set, root in [('existing', config.source), ('new', config.output)]:
        for paper, reference in references.items():
            if not reference['graph_useful']:
                continue
            for condition in CONDITIONS:
                path = root / 'reports' / condition / f'{paper}.json'
                if path.exists():
                    lengths.append({'report_set': report_set, 'paper_id': paper, 'condition': condition,
                                    'configuration': NAMES[condition], 'characters': len(read(path)['body'])})
    write_csv(config.output / 'report_lengths.csv', lengths)
    write(config.output / 'run_summary.json', {
        'scope': 'original_five_only', 'screened_papers': len(references),
        'included_papers': [p for p, r in references.items() if r['graph_useful']],
        'excluded_papers': [p for p, r in references.items() if not r['graph_useful']],
        'calls': len(records), 'calls_by_stage_state': dict(Counter(r['stage'] + ':' + r['state'] for r in records)),
        'report_counts': dict(Counter(r['report_set'] for r in lengths)),
        'known_usage': dict(usage), 'calls_with_unknown_usage': sum(r.get('usage') is None for r in records),
        'answer_part_technical_states': dict(Counter(r['technical_state'] for r in parts)),
        'answer_part_scientific_states': dict(Counter(r['status'] for r in parts)),
        'regression_tests_run': False, 'remaining_95_started': False,
        'unresolved_tasks': read(config.output / 'unresolved_tasks.json') if (config.output / 'unresolved_tasks.json').exists() else [],
    })
    print(f'Aggregated {len(parts)} answer-part slots, {len(rows)} paper-aspect slots.', flush=True)
