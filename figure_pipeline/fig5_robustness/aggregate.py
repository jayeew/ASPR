from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import ASPECTS
from figure_pipeline.fig4_explanation_study.aggregate import GROUNDED

from .models import CONDITIONS, Config

METRICS = ('grounded_correct', 'explicit_abstention', 'unsupported_definitive')


def summarize(values: list[float], config: Config, repeats: bool = False) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'mean': None, 'low': None, 'high': None, 'minimum': None, 'maximum': None}
    array = np.array(values, dtype=float)
    result = {'n': len(values), 'mean': float(array.mean()), 'minimum': float(array.min()), 'maximum': float(array.max())}
    if repeats:
        return {**result, 'low': None, 'high': None}
    rng = np.random.default_rng(config.seed)
    boot = array[rng.integers(0, len(array), (config.bootstrap_repeats, len(array)))].mean(axis=1)
    return {**result, 'low': float(np.quantile(boot, .025)), 'high': float(np.quantile(boot, .975))}


def part_values(part: dict[str, Any]) -> dict[str, Any]:
    if part.get('technical_state') != 'completed':
        return {m: None for m in METRICS} | {'scientific_unresolved': None}
    correct = (part['status'] == 'correct' and part['scope_correct'] is True and
               part['grounding'] in GROUNDED and part.get('unsupported_definitive') is False)
    return {'grounded_correct': int(correct),
                'explicit_abstention': None if part['explicitly_abstains'] is None else int(part['explicitly_abstains']),
                'unsupported_definitive': None if part['unsupported_definitive'] is None else int(part['unsupported_definitive']),
                'scientific_unresolved': int(part['status'] == 'unresolved' or part['explicitly_abstains'] is None or
                                          part['unsupported_definitive'] is None or part['scope_correct'] is None)}


def paired_rows(config: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = read(config.output / 'cohort.json')
    rows, metrics = [], []
    for paper in cohort['selected']:
        path = config.output / 'baseline/reference/papers' / f'{paper}.json'
        if not path.exists():
            for c in CONDITIONS:
                if c != 'F_REPEAT' or paper in cohort['repeated']:
                    for aspect in ASPECTS:
                        metrics.append(dict(paper_id=paper, condition=c, aspect=aspect, state='upstream_unavailable',
                                            fixed_denominator=None, technical_coverage=None, **{m: None for m in METRICS}))
            continue
        ref = read(path)
        for aspect in ASPECTS:
            expected = [(q['question_id'], p['part_id']) for q in ref['questions'] if q['aspect'] == aspect and q['applicable'] for p in q['answer_parts']]
            evaluation = config.output / 'aligned_evaluation' / aspect / f'{paper}.json'
            observed = {}
            if evaluation.exists():
                mapping = read(config.output / 'evaluation_mapping' / aspect / f'{paper}.json')
                for candidate in read(evaluation)['candidates']:
                    for q in candidate['questions']:
                        for part in q['parts']:
                            observed[(mapping[candidate['report_id']], q['question_id'], part['part_id'])] = part
            for condition in CONDITIONS:
                if condition == 'F_REPEAT' and paper not in cohort['repeated']:
                    continue
                current = []
                for question, part in expected:
                    value = observed.get((condition, question, part), {'technical_state': 'missing'})
                    row = {**value, **part_values(value), 'paper_id': paper, 'condition': condition,
                           'aspect': aspect, 'question_id': question, 'part_id': part}
                    rows.append(row)
                    current.append(row)
                metrics.append(metric_row(paper, condition, aspect, current))
    return rows, metrics


def metric_row(paper: str, condition: str, aspect: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(r['technical_state'] == 'completed' for r in rows)
    complete = bool(total) and completed == total
    result = {'paper_id': paper, 'condition': condition, 'aspect': aspect, 'fixed_denominator': total,
                  'evaluated_parts': completed, 'technical_coverage': completed / total if total else None,
                  'state': 'completed' if complete else 'technical_missing' if total else 'not_applicable',
                  'scientific_unresolved': sum(r.get('scientific_unresolved') == 1 for r in rows)}
    for metric in METRICS:
        numerator = sum(r.get(metric) == 1 for r in rows)
        result.update({metric + '_numerator': numerator if complete else None,
                       metric: numerator / total if complete else None,
                       metric + '_unresolved': sum(r.get(metric) is None for r in rows if r['technical_state'] == 'completed')})
    for kind in ('reasonable', 'unnecessary', 'unresolved'):
        result['abstention_' + kind] = sum(r.get('explicitly_abstains') is True and r.get('abstention_appropriateness') == kind for r in rows) if complete else None
    result['not_written'] = sum(r.get('status') == 'not_written' and r.get('explicitly_abstains') is False for r in rows) if complete else None
    return result


def contrasts(config: Config, metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in metrics}
    differences = []
    for row in metrics:
        if row['condition'] == 'F':
            continue
        baseline = index[(row['paper_id'], 'F', row['aspect'])]
        for metric in METRICS:
            valid = row.get(metric) is not None and baseline.get(metric) is not None
            differences.append({'paper_id': row['paper_id'], 'condition': row['condition'], 'aspect': row['aspect'], 'metric': metric,
                                    'condition_value': row.get(metric), 'baseline_value': baseline.get(metric),
                                    'delta': row[metric] - baseline[metric] if valid else None,
                                    'missing_reason': None if valid else f"F:{baseline['state']}; condition:{row['state']}"})
    grouped = defaultdict(list)
    for r in differences:
        grouped[(r['condition'], r['aspect'], r['metric'])].append(r)
    summaries = []
    for (condition, aspect, metric), values in grouped.items():
        good = [r for r in values if r['delta'] is not None]
        summaries.append(dict(condition=condition, aspect=aspect, metric=metric, planned=len(values),
                              missing=len(values) - len(good), paper_ids=[r['paper_id'] for r in good],
                              **summarize([r['delta'] for r in good], config, condition == 'F_REPEAT')))
    return differences, summaries


def observational(config: Config) -> list[dict[str, Any]]:
    rows = []
    for paper in read(config.output / 'paper_coverage.json'):
        ident = paper['paper_id']
        path = config.output / 'baseline/reference/papers' / f'{ident}.json'
        ref = read(path) if path.exists() else None
        for aspect in ASPECTS:
            expected = [(q['question_id'], p['part_id']) for q in ref['questions'] if q['aspect'] == aspect and q['applicable'] for p in q['answer_parts']] if ref else []
            observed = {}
            evaluation = config.output / 'baseline/aligned_evaluation' / aspect / f'{ident}.json'
            mapping = config.output / 'baseline/evaluation_mapping' / f'{ident}.json'
            if evaluation.exists() and mapping.exists():
                full = {k for k, v in read(mapping).items() if v == 'F'}
                for c in read(evaluation)['candidates']:
                    if c['report_id'] in full:
                        observed.update({(q['question_id'], p['part_id']): p for q in c['questions'] for p in q['parts']})
            complete = bool(expected) and all(observed.get(p, {}).get('technical_state') == 'completed' for p in expected)
            correct = [observed[p] for p in expected if p in observed and observed[p]['status'] == 'correct' and observed[p]['scope_correct'] is True]
            state = ('upstream_unavailable' if ref is None else 'graph_not_applicable' if not ref['graph_useful'] else
                     'completed' if complete else 'technical_missing' if expected else 'not_applicable')
            rows.append({**paper, 'aspect': aspect, 'state': state, 'fixed_denominator': len(expected) if ref else None,
                         'correct_parts': len(correct) if complete else None,
                         'grounded_parts': sum(p['grounding'] in GROUNDED for p in correct) if complete else None,
                         'content_coverage': len(correct) / len(expected) if complete else None,
                         'grounded_coverage': sum(p['grounding'] in GROUNDED for p in correct) / len(expected) if complete else None})
    return rows


def observational_summary(config: Config, rows: list[dict[str, Any]], exclude_development: bool) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if exclude_development and row['development']:
            continue
        for grouping, value in [('field', row['group']), ('neighborhood', 'sparse' if row['sparse'] else 'saturated')]:
            groups[(grouping, value, row['aspect'])].append(row)
    results = []
    for (grouping, value, aspect), items in groups.items():
        for metric in ('content_coverage', 'grounded_coverage'):
            values = [r[metric] for r in items if r[metric] is not None]
            results.append(dict(grouping=grouping, group=value, aspect=aspect, metric=metric,
                                planned=len(items), missing=len(items)-len(values), states=dict(Counter(r['state'] for r in items)),
                                **summarize(values, config)))
    return results


def usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    usage = Counter()
    missing = 0
    for r in records:
        if r.get('usage') is None:
            missing += 1
        else:
            usage.update({k: v for k, v in r['usage'].items() if isinstance(v, (int, float))})
    return dict(calls=len(records), failures=sum(r.get('state') != 'completed' for r in records),
                call_seconds_sum=sum(r.get('seconds', 0) for r in records) if records else None, usage_missing=missing,
                usage_complete=missing == 0, **{k: usage.get(k) for k in
                ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_output_tokens')})


def costs(config: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = [dict(read(p), record_path=str(p)) for p in (config.output / 'logs/calls').glob('*/record.json')]
    baseline = [dict(read(p), record_path=str(p)) for p in (config.output / 'baseline/calls').glob('*/record.json')]
    groups = defaultdict(list)
    for r in records:
        groups[(r['stage'], r['method'])].append(r)
    spending = [dict(stage=s, task=c, cost_scope='new_spending', **usage_summary(rs)) for (s, c), rs in groups.items()]
    allocated = []
    cohort = read(config.output / 'cohort.json')
    for paper in cohort['selected']:
        for condition in CONDITIONS:
            if condition == 'F_REPEAT' and paper not in cohort['repeated']:
                continue
            items = []
            required = [('analysis', 'gear'), ('analysis', 'graph'), ('reports', '')]
            missing_tasks = []
            for stage, branch in required:
                reused = condition == 'F' or (condition == 'K5' and branch == 'gear')
                method = ('gear' if branch == 'gear' else 'graph_F' if branch else 'F') if reused else condition + ('_' + branch if branch else '')
                found = [r for r in (baseline if reused else records) if (r['paper_id'], r['stage'], r['method']) == (paper, stage, method)]
                if not found:
                    missing_tasks.append(f'{stage}/{method}')
                items.extend(found)
            # End-to-end wall time is available only when all stages are newly run in this process.
            starts = [r.get('first_event_at') for r in items if r.get('first_event_at')]
            ends = [r.get('last_event_at') for r in items if r.get('last_event_at')]
            wall = max(ends)-min(starts) if starts and ends and condition not in ('F', 'K5') and not missing_tasks else None
            allocated.append(dict(paper_id=paper, condition=condition, cost_scope='attributable_fixed_input_generation',
                                  missing_task_logs=missing_tasks, complete_cost=not missing_tasks and all(r.get('usage') is not None for r in items),
                                  observed_event_wall_seconds=wall, **usage_summary(items)))
    return records, spending, allocated


def status(config: Config) -> dict[str, Any]:
    cohort = read(config.output / 'cohort.json')
    ledger = config.output / 'call_ledger.jsonl'
    entries = [json.loads(s) for s in ledger.read_text().splitlines() if s] if ledger.exists() else []
    states = [read(p) for p in (config.output / 'task_status').glob('*/*/*.json')]
    reports = []
    for paper in cohort['selected']:
        for condition in CONDITIONS:
            if condition == 'F_REPEAT' and paper not in cohort['repeated']:
                continue
            path = config.output / 'reports' / condition / f'{paper}.json'
            matching = [r for r in states if r['paper_id'] == paper and r['stage'] == 'reports' and r['condition'] == condition]
            upstream = [r for r in states if r['paper_id'] == paper and r['state'] == 'upstream_unavailable']
            reference_failure = [r for r in states if r['paper_id'] == paper and r['stage'] == 'reference' and r['state'] == 'unresolved']
            reports.append({'paper_id': paper, 'condition': condition, 'state': 'available' if path.exists() else
                                matching[-1]['state'] if matching else 'upstream_unavailable' if upstream else
                                'reference_unavailable' if reference_failure else 'not_run'})
    return {'planned_reports': 85, 'available_reports': sum(r['state'] == 'available' for r in reports),
                'new_calls': len(entries), 'additional_calls': sum(r['additional'] for r in entries),
                'budget_limit': 325, 'reports': reports, 'task_states': dict(Counter(r['state'] for r in states)),
                'unresolved_tasks': [r for r in states if r['state'] in ('unresolved', 'upstream_unavailable')]}


def aggregate(config: Config) -> None:
    parts, metrics = paired_rows(config)
    differences, summaries = contrasts(config, metrics)
    observed = observational(config)
    records, spending, allocated = costs(config)
    tables = {'answer_parts': parts, 'paper_condition_metrics': metrics, 'paired_differences': differences,
                  'paired_summary': [r for r in summaries if r['condition'] != 'F_REPEAT'],
                  'repeat_variation': [r for r in summaries if r['condition'] == 'F_REPEAT'],
                  'observational_papers': observed,
                  'observational_summary': observational_summary(config, observed, False),
                  'observational_summary_without_development': observational_summary(config, observed, True),
                  'model_calls': records, 'new_spending': spending, 'attributable_costs': allocated}
    for name, rows in tables.items():
        write(config.output / f'{name}.json', rows)
        write_csv(config.output / f'{name}.csv', rows)
    write(config.output / 'run_summary.json', status(config))
