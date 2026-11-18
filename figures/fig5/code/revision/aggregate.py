from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.aggregate import GROUNDED

from .models import Config


def all_tasks(config: Config) -> list[dict[str, Any]]:
    tasks = {(t['paper_id'], t['condition']): t for t in read(config.output / 'tasks.json')}
    for path in (config.output / 'paper_tasks').glob('*.json'):
        for t in read(path):
            # Static protocol remains authoritative for corrected view associations.
            tasks.setdefault((t['paper_id'], t['condition']), t)
    return list(tasks.values())


def current_eligibilities(config: Config) -> list[dict[str, Any]]:
    result = []
    for path in (config.output / 'eligibility').glob('*.json'):
        source = config.output / 'support_results' / path.name
        if not source.exists():
            continue
        support, decision = read(source), read(path)
        if support['targets']:
            if decision.get('target') != support['targets'][0] or len(decision.get('states', [])) != 3:
                continue
        elif decision.get('reason') != support['ineligible_reason']:
            continue
        result.append({'paper_id': path.stem, **decision})
    return result


def status(config: Config) -> dict[str, Any]:
    tasks = all_tasks(config)
    states = [read(p) for p in (config.output / 'task_status').glob('*/*/*.json')]
    failed = [r for r in states if r['state'] in ('unresolved', 'upstream_unavailable')]
    ledger = config.output / 'call_ledger.jsonl'
    calls = [json.loads(s) for s in ledger.read_text().splitlines() if s] if ledger.exists() else []
    from figure_pipeline.fig4_explanation_100.models import ASPECTS
    evaluated = 0
    for task in tasks:
        paths = [config.output / 'aligned' / task['condition'] / aspect / f"{task['paper_id']}.json" for aspect in ASPECTS]
        if all(path.exists() for path in paths) and all(part['technical_state'] == 'completed' for path in paths for q in read(path)['questions'] for part in q['parts']):
            evaluated += 1
    return {'planned_reports': len(tasks), 'fully_evaluated_reports': evaluated,
            'evaluated_report_aspects': len(list((config.output / 'aligned').glob('*/*/*.json'))),
            'expected_static_reference_views': len({(t['paper_id'], t['view']) for t in read(config.output / 'tasks.json')}),
            'available_reports': sum((config.output / 'reports' / t['condition'] / f"{t['paper_id']}.json").exists() for t in tasks),
            'validated_references': len(list((config.output / 'validated_reference').glob('*/*.json'))),
            'eligible_critical_papers': sum(r['eligible'] for r in current_eligibilities(config)),
            'support_designs_checked': len(current_eligibilities(config)),
            'support_proposals_awaiting_or_completed_reference_check': sum(bool(read(p)['targets']) for p in (config.output / 'support_results').glob('*.json')),
            'new_calls': len(calls), 'task_states': dict(Counter(r['state'] for r in states)), 'unresolved_tasks': failed}


def scored(part: dict[str, Any]) -> dict[str, Any]:
    if part.get('technical_state') != 'completed':
        return {'correct': None, 'answered': None, 'risk': None, 'unsupported': None, 'abstains': None, 'omission': None, 'scientific_unresolved': None}
    answered = part['response_type'] == 'substantive_answer'
    correct = (answered and part['status'] == 'correct' and part['scope_correct'] is True and
               part['grounding'] in GROUNDED and part['unsupported_definitive'] is False and part['conflicting_material_error'] is False)
    known_error = part['status'] == 'incorrect' or (answered and part['scope_correct'] is False) or part['unsupported_definitive'] is True or part['conflicting_material_error'] is True
    risk = int(known_error) if known_error else None if answered and (part['status'] == 'unresolved' or part['unsupported_definitive'] is None or part['conflicting_material_error'] is None or part['scope_correct'] is None) else 0
    unresolved = part['response_type'] == 'unresolved' or part['status'] == 'unresolved' or part['unsupported_definitive'] is None or part['conflicting_material_error'] is None
    return {'correct': int(correct), 'answered': int(answered), 'risk': risk, 'unsupported': part['unsupported_definitive'],
            'abstains': int(part['response_type'] == 'target_abstention'), 'omission': int(part['response_type'] == 'omission'), 'scientific_unresolved': int(unresolved)}


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total, completed = len(rows), sum(r['technical_state'] == 'completed' for r in rows)
    ready = completed == total and total > 0
    answerable = [r for r in rows if r['answerability'] == 'answerable']
    unanswerable = [r for r in rows if r['answerability'] == 'unanswerable']
    answered = [r for r in rows if r['answered'] == 1]
    result = {'fixed_denominator': total, 'evaluated_parts': completed, 'state': 'completed' if ready else 'technical_missing' if total else 'not_applicable',
              'answerable_denominator': len(answerable), 'unanswerable_denominator': len(unanswerable),
              'reference_unresolved': sum(r['answerability'] == 'unresolved' for r in rows), 'reference_missing': sum(r['answerability'] == 'missing' for r in rows)}
    specifications = {'q_fixed': (sum(r['correct'] == 1 for r in rows), total),
                      'answer_coverage': (len(answered), total),
                      'unsupported_burden': (sum(r['unsupported'] is True for r in rows), total),
                      'answered_risk': (sum(r['risk'] == 1 for r in answered), len(answered)),
                      'reasonable_abstention_sensitivity': (sum(r['abstains'] == 1 for r in unanswerable), len(unanswerable)),
                      'unnecessary_abstention_rate': (sum(r['abstains'] == 1 for r in answerable), len(answerable))}
    for name, (num, den) in specifications.items():
        result.update({name + '_numerator': num if ready else None, name + '_denominator': den,
                       name: num / den if den and ready else None})
    result.update(unsupported_unresolved=sum(r['unsupported'] is None for r in rows if r['technical_state'] == 'completed'),
                  answered_risk_unresolved=sum(r['risk'] is None for r in answered),
                  scientific_unresolved=sum(r['scientific_unresolved'] == 1 for r in rows), omissions=sum(r['omission'] == 1 for r in rows))
    unknown = result['answered_risk_unresolved']
    result['answered_risk_lower_bound'] = result['answered_risk']
    result['answered_risk_upper_bound'] = (result['answered_risk_numerator'] + unknown) / len(answered) if ready and answered else None
    if unknown:
        # Unknown answered judgments must not silently become a zero-error point estimate.
        result['answered_risk'] = None
    return result


def parts_and_metrics(config: Config, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, metrics = [], []
    for t in tasks:
        paper, condition = t['paper_id'], t['condition']
        fixed = read(config.output / 'baseline/reference/papers' / f'{paper}.json')
        refpath = config.output / 'validated_reference' / t['view'] / f'{paper}.json'
        refs = {(p['question_id'], p['part_id']): p for p in read(refpath)['parts']} if refpath.exists() else {}
        by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for q in fixed['questions']:
            path = config.output / 'aligned' / condition / q['aspect'] / f'{paper}.json'
            parts = {(qq['question_id'], p['part_id']): p for qq in read(path)['questions'] for p in qq['parts']} if path.exists() else {}
            by_aspect[q['aspect']]
            for p in q['answer_parts'] if q['applicable'] else []:
                key = (q['question_id'], p['part_id'])
                part = parts.get(key, {'technical_state': 'missing'})
                row = {**t, 'aspect': q['aspect'], 'question_id': key[0], 'part_id': key[1], **part,
                       'answerability': refs.get(key, {}).get('answerability', 'missing'), **scored(part)}
                rows.append(row)
                by_aspect[q['aspect']].append(row)
        for aspect, items in by_aspect.items():
            metrics.append({**t, 'aspect': aspect, **metric(items)})
    return rows, metrics


def paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r for r in rows}
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any] | None, dict[str, Any]]]] = defaultdict(list)
    for r in rows:
        if r['condition'] not in ('F', 'E'):
            base = index.get((r['paper_id'], 'F', r['question_id'], r['part_id']))
            groups[(r['paper_id'], r['condition'], r['aspect'])].append((base, r))
    result = []
    for (paper, condition, aspect), all_values in groups.items():
        refs_complete = all(a and a['answerability'] != 'missing' and b['answerability'] != 'missing' for a, b in all_values)
        values = [(a, b) for a, b in all_values if a and a['answerability'] == b['answerability'] == 'answerable']
        ready = bool(values) and refs_complete and all(a['technical_state'] == b['technical_state'] == 'completed' for a, b in values)
        result.append({'paper_id': paper, 'condition': condition, 'aspect': aspect,
                       'split': all_values[0][1]['split'], 'common_answerable_denominator': len(values) if refs_complete else None,
                       'delta_q_answerable': sum(b['correct'] - a['correct'] for a, b in values) / len(values) if ready else None,
                       'state': 'reference_missing' if not refs_complete else 'no_common_answerable_parts' if not values else 'completed' if ready else 'technical_missing',
                       'technical_complete': ready})
    return result


def interval(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'mean': None, 'low': None, 'high': None}
    data = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), (10000, len(data)))].mean(axis=1)
    return {'n': len(values), 'mean': float(data.mean()), 'low': float(np.quantile(means, .025)), 'high': float(np.quantile(means, .975))}


def costs(config: Config, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path in (config.output / 'logs/calls').glob('*/record.json'):
        r = read(path)
        usage = r.get('usage') or {}
        result.append({'paper_id': r['paper_id'], 'stage': r['stage'], 'condition': r['method'].removesuffix('__capacity_retry'), 'raw_condition': r['method'], 'model': r['model'], 'effort': r['effort'],
                       'state': r['state'], 'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                       'cached_input_tokens': usage.get('cached_input_tokens'), 'reasoning_output_tokens': usage.get('reasoning_output_tokens'), 'seconds': r.get('seconds'), 'reused': False,
                       'new_spending': True, 'cost_category': 'inference' if r['stage'] in ('analysis', 'reports') else 'reference_or_evaluation',
                       'timing_source': r.get('timing_source', 'client_monotonic_elapsed'), 'record_path': str(path), 'attempt_id': r.get('attempt_id'), 'usage_missing': not bool(usage)})
        result[-1]['started_at'] = r['time']
    # Historical attribution remains distinct from actual new spending.
    reuse = read(config.output / 'reuse.json')
    roots = [config.source, config.old]
    records = [(p, read(p)) for root in roots for p in (root / 'logs/calls').glob('*/record.json')]
    for t in tasks:
        adoption = next((r for r in reuse if (r['paper_id'], r['condition']) == (t['paper_id'], t['condition'])), None)
        specs = []
        if adoption:
            root = config.source if t['condition'] in ('F', 'E') else config.old
            specs = [(root, 'reports', t['condition'])]
            if root == config.source:
                specs += [(root, 'analysis', 'gear')]
                if t['condition'] == 'F':
                    specs += [(root, 'analysis', 'graph_F')]
            else:
                specs += [(root, 'analysis', t['condition'] + '_graph')]
                specs += [(config.source, 'analysis', 'gear')] if t['condition'] == 'K5' else [(root, 'analysis', t['condition'] + '_gear')]
        elif t['condition'] in ('K3', 'K5', 'G50', 'G25'):
            specs = [(config.source, 'analysis', 'gear')]
        for root, stage, method in specs:
            matches = [(p, r) for p, r in records if p.is_relative_to(root) and (r.get('paper_id'), r.get('stage'), r.get('method'), r.get('state')) == (t['paper_id'], stage, method, 'completed')]
            if not matches:
                result.append({'paper_id': t['paper_id'], 'condition': t['condition'], 'stage': stage, 'reused': True, 'new_spending': False, 'usage_missing': True, 'state': 'attribution_missing', 'cost_category': 'inference'})
                continue
            path, r = max(matches, key=lambda x: x[1]['time'])
            usage = r.get('usage') or {}
            result.append({'paper_id': t['paper_id'], 'condition': t['condition'], 'stage': stage, 'model': r['model'], 'effort': r['effort'],
                           'state': 'reused', 'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                           'cached_input_tokens': usage.get('cached_input_tokens'), 'reasoning_output_tokens': usage.get('reasoning_output_tokens'), 'seconds': r.get('seconds'), 'reused': True,
                           'new_spending': False, 'cost_category': 'inference', 'record_path': str(path), 'usage_missing': not bool(usage)})
    return result


def aggregate(config: Config) -> None:
    tasks = all_tasks(config)
    rows, metrics = parts_and_metrics(config, tasks)
    write_csv(config.output / 'answer_parts.csv', rows)
    write_csv(config.output / 'paper_metrics.csv', metrics)
    paired_rows = paired(rows)
    write_csv(config.output / 'paired_answerable.csv', paired_rows)
    paired_summary = []
    for split, condition, aspect in sorted({(r['split'], r['condition'], r['aspect']) for r in paired_rows}):
        current = [r for r in paired_rows if (r['split'], r['condition'], r['aspect']) == (split, condition, aspect)]
        paired_summary.append({'split': split, 'condition': condition, 'aspect': aspect, 'planned_papers': len(current),
                               **interval([r['delta_q_answerable'] for r in current if r['delta_q_answerable'] is not None], config.seed)})
    write_csv(config.output / 'paired_answerable_summary.csv', paired_summary)
    cost_rows = costs(config, tasks)
    write_csv(config.output / 'stage_costs.csv', cost_rows)
    conditions = []
    for t in tasks:
        mask = config.output / 'masks' / t['view'] / f"{t['paper_id']}.json"
        data = read(config.output / 'inputs' / t['view'] / f"{t['paper_id']}.json")
        from figure_pipeline.fig5_robustness.materials import original_blocks
        blocks = original_blocks(data)
        aliases = read(config.output / 'identities' / f"{t['paper_id']}.json")
        graph = data['graph'] or {'cards': [], 'joint': {'historical_edges': []}}
        conditions.append({**t, **(read(mask) if mask.exists() else {}), 'condition': t['condition'],
                           'visible_independent_works': len({aliases[p['source_id']] for b in blocks for p in b['provenance']}),
                           'unique_original_blocks': len(blocks), 'unique_original_characters': sum(len(b['text']) for b in blocks),
                           'visible_historical_nodes': len({n['claim_id'] for c in graph['cards'] for n in c['neighbors']}),
                           'visible_historical_edges': len(graph['joint']['historical_edges']),
                           'report_available': (config.output / 'reports' / t['condition'] / f"{t['paper_id']}.json").exists(),
                           'model': 'gpt-5.6-luna' if t['condition'] in ('LUNA', 'DW_LUNA') else config.model,
                           'effort': {'LOW': 'low', 'HIGH': 'high'}.get(t['condition'], 'medium')})
    write_csv(config.output / 'conditions.csv', conditions)
    refs = [{**p, 'paper_id': path.stem, 'view': path.parent.name} for path in (config.output / 'validated_reference').glob('*/*.json') for p in read(path)['parts']]
    write_csv(config.output / 'conditional_reference.csv', refs)
    supports = []
    for path in (config.output / 'support_results').glob('*.json'):
        value = read(path)
        supports.extend([{'paper_id': path.stem, **t} for t in value['targets']] or [{'paper_id': path.stem, 'ineligible_reason': value['ineligible_reason']}])
    write_csv(config.output / 'support_groups.csv', supports)
    samples = {r['paper_id']: r for r in read(config.output / 'samples.json')}
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in metrics}
    domain = []
    for r in metrics:
        if r['condition'] != 'F':
            continue
        other = index.get((r['paper_id'], 'E', r['aspect']), {})
        a, b = r['q_fixed'], other.get('q_fixed')
        domain.append({'paper_id': r['paper_id'], 'group': samples[r['paper_id']]['group'], 'aspect': r['aspect'],
                       'F_q_fixed': a, 'E_q_fixed': b, 'paired_delta': a - b if a is not None and b is not None else None})
    write_csv(config.output / 'domain_paired.csv', domain)
    summary = []
    for group in sorted({r['group'] for r in domain}):
        for aspect in sorted({r['aspect'] for r in domain}):
            current = [r for r in domain if r['group'] == group and r['aspect'] == aspect]
            summary.append({'group': group, 'aspect': aspect, 'planned_papers': len(current), **interval([r['paired_delta'] for r in current if r['paired_delta'] is not None], config.seed)})
    write_csv(config.output / 'domain_summary.csv', summary)
    from .diagnostics import (
        cost_quality,
        export_diagnostics,
        export_latency,
        export_material_tables,
        material_checks,
    )
    export_diagnostics(config, rows, metrics)
    from .diagnostics import model_completion_audit
    material_checks(config)
    model_completion_audit(config)
    export_material_tables(config)
    cost_quality(config, tasks, metrics, cost_rows)
    export_latency(config, cost_rows)
    from .comparisons import export_independent_review, export_paired_metrics
    export_paired_metrics(config, metrics)
    export_independent_review(config, rows)
    write(config.output / 'run_summary.json', status(config))
    print(json.dumps({k: v for k, v in status(config).items() if k != 'unresolved_tasks'}, ensure_ascii=False), flush=True)
