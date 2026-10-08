from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from figure_pipeline.fig3_revision.aggregate import bootstrap, write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_mechanisms.io import jsonl
from figure_pipeline.fig4_rerun.config import CONDITIONS, METRICS, NAMES
from figure_pipeline.fig4_rerun.engine import ledger

from .config import DIMENSIONS, Config

CONTRASTS = [
    ('加入 GEAR（完整系统－仅 Graph）', 'G', 'scientific_increment'),
    ('加入 GEAR（完整系统－仅 Graph）', 'G', 'knowledge_relations'),
    ('加入 Graph（完整系统－仅 GEAR）', 'E', 'scientific_increment'),
    ('加入 Graph（完整系统－仅 GEAR）', 'E', 'knowledge_relations'),
    ('联合图（完整系统－移除联合图）', 'F_noJ', 'joint_explanation'),
    ('结构数值（完整系统－屏蔽结构数值）', 'F_noM', 'structure_explanation'),
    ('引用路径（完整系统－屏蔽引用路径）', 'F_noP', 'citation_explanation'),
]


def status(config: Config) -> dict[str, Any]:
    entries = ledger(config.output)
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    evaluations = [read(p) for p in (config.output / 'evaluation').glob('*/*.json')]
    usage = [r.get('usage') for r in records if r.get('usage')]
    return {
        'calls_started': len(entries), 'call_limit': 70,
        'generation_started': sum(e['stage'] == 'generate' for e in entries),
        'evaluation_started': sum(e['stage'] == 'evaluate' for e in entries),
        'reports': len(list((config.output / 'reports').glob('*/*.json'))),
        'evaluations': len(evaluations),
        'evaluation_states': dict(Counter(e['evaluation_status'] for e in evaluations)),
        'call_states': dict(Counter(r.get('state', 'unknown') for r in records)),
        'usage_records': len(usage),
        'input_tokens': sum(u.get('input_tokens', 0) for u in usage),
        'output_tokens': sum(u.get('output_tokens', 0) for u in usage),
        'failed_calls': [{k: r.get(k) for k in ('paper_id', 'method', 'stage', 'state', 'error')}
                         for r in records if r.get('state') not in ('completed', 'running', 'started')],
    }


def core_metrics(evaluation: dict[str, Any] | None, material: dict[str, Any]) -> dict[str, Any]:
    references = {r['core_id']: r for r in material['reference']}
    valid = evaluation is not None and evaluation['core_status'] == 'completed'
    items = {i['core_id']: i for i in evaluation['items']} if valid else {}
    applicable = {'C': list(references),
                  'H': [key for key, r in references.items() if r['historical_comparison_applicable']],
                  'S': [key for key, r in references.items() if r['scope_applicable']]}
    result = {}
    for metric, keys in applicable.items():
        numerator = sum(items[key]['addressed'] and (
            metric == 'C' or (items[key]['scope_correct'] is True and (
                metric == 'S' or items[key]['historical_comparison_correct'] is True))) for key in keys) if valid else None
        result.update({metric: numerator / len(keys) if numerator is not None and keys else None,
                       metric + '_numerator': numerator, metric + '_denominator': len(keys)})
    return result


def aggregate(config: Config) -> dict[str, Any]:
    ids = read(config.output / 'pilot.json')['paper_ids']
    rows, judgments = [], []
    for paper in ids:
        material = read(config.output / 'inputs/evaluation' / f'{paper}.json')
        for condition in CONDITIONS:
            path = config.output / 'evaluation' / condition / f'{paper}.json'
            evaluation = read(path) if path.exists() else None
            report_path = config.output / 'reports' / condition / f'{paper}.json'
            row = {'paper_id': paper, 'condition': condition, 'configuration': NAMES[condition],
                   'report_chars': read(report_path)['body_chars'] if report_path.exists() else None,
                   'evaluation_state': evaluation['evaluation_status'] if evaluation else 'missing',
                   **core_metrics(evaluation, material)}
            dimensions = {d['dimension']: d for d in evaluation['dimensions']} if evaluation else {}
            for dimension in DIMENSIONS:
                item = dimensions.get(dimension)
                row[dimension] = item['score'] if item else None
                row[dimension + '_state'] = item['status'] if item else 'missing'
                if item:
                    judgments.append({'paper_id': paper, 'condition': condition,
                                      'configuration': NAMES[condition], **item})
            rows.append(row)
    write_csv(config.output / 'paper_metrics.csv', rows)
    jsonl(config.output / 'capability_evaluations.jsonl', judgments)
    lookup = {(r['paper_id'], r['condition']): r for r in rows}
    metrics = {**DIMENSIONS, **METRICS}
    summary = []
    for metric, title in metrics.items():
        common = [p for p in ids if all(lookup[p, c][metric] is not None for c in CONDITIONS)]
        for condition in CONDITIONS:
            available = [lookup[p, condition][metric] for p in ids if lookup[p, condition][metric] is not None]
            values = [lookup[p, condition][metric] for p in common]
            summary.append({'condition': condition, 'configuration': NAMES[condition], 'metric': metric,
                            'metric_name': title, **bootstrap(values, config), 'common_paper_ids': common,
                            'available_n': len(available), 'available_mean': mean(available) if available else None})
    write_csv(config.output / 'condition_summary.csv', summary)
    main_summary = []
    for metric in ('scientific_increment', 'knowledge_relations'):
        common = [p for p in ids if all(lookup[p, c][metric] is not None for c in ('T', 'E', 'G', 'F'))]
        for condition in ('T', 'E', 'G', 'F'):
            main_summary.append({'condition': condition, 'configuration': NAMES[condition],
                                 'metric': metric, 'metric_name': DIMENSIONS[metric],
                                 **bootstrap([lookup[p, condition][metric] for p in common], config),
                                 'paper_ids': common})
    write_csv(config.output / 'main_four_conditions.csv', main_summary)
    pairs, effects = [], []
    for title, comparator, dimension in CONTRASTS:
        current = []
        for paper in ids:
            full, control = lookup[paper, 'F'][dimension], lookup[paper, comparator][dimension]
            delta = full - control if full is not None and control is not None else None
            row = {'contrast': title, 'paper_id': paper, 'metric': dimension,
                   'metric_name': DIMENSIONS[dimension], 'full': full, 'comparator': control, 'delta': delta}
            pairs.append(row)
            if delta is not None:
                current.append(delta)
        effects.append({'contrast': title, 'metric': dimension, 'metric_name': DIMENSIONS[dimension],
                        **bootstrap(current, config), 'higher_papers': sum(d > 0 for d in current),
                        'equal_papers': sum(d == 0 for d in current), 'lower_papers': sum(d < 0 for d in current)})
    write_csv(config.output / 'paired_effects_paper.csv', pairs)
    write_csv(config.output / 'paired_effects_summary.csv', effects)
    missing = [j for j in judgments if j['status'] in ('unresolved', 'technical_missing')]
    write(config.output / 'unresolved_judgments.json', missing)
    result = {**status(config), 'paper_count': len(ids), 'unresolved_dimensions': len(missing),
              'paired_effects': effects, 'scope': 'five-paper exploratory report-input pilot'}
    write(config.output / 'summary.json', result)
    return result
