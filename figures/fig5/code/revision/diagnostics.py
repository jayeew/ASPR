from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write

from .aggregate import interval
from .materials import original_blocks, packet
from .models import Config


def state(row: dict[str, Any]) -> str:
    if row['technical_state'] != 'completed':
        return 'technical_missing'
    if row['risk'] == 1 or row['unsupported'] is True:
        return 'incorrect_or_unsupported'
    if row['correct'] == 1:
        return 'grounded_correct'
    if row['abstains'] == 1:
        return 'target_abstention'
    if row['omission'] == 1:
        return 'omission'
    return 'evaluation_unresolved'


def export_diagnostics(config: Config, rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    indexed = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r for r in rows}
    traces, behavior = [], []
    for path in (config.output / 'eligibility').glob('*.json'):
        eligible = read(path)
        if not eligible['eligible']:
            continue
        target = eligible['target']
        ident = (target['question_id'], target['part_id'])
        trajectory = {}
        for condition in ('F', 'NONCRITICAL', 'CRITICAL', 'RESTORE', 'DW_CRITICAL'):
            row = indexed.get((path.stem, condition, *ident))
            if row:
                trajectory[condition] = row
                traces.append({'paper_id': path.stem, 'condition': condition, 'question_id': ident[0], 'part_id': ident[1],
                               'answerability': row['answerability'], 'state': state(row), 'split': row['split'],
                               'report_quotes': row.get('report_quotes'), 'abstention_quotes': row.get('abstention_quotes')})
        baseline, control, critical, restore = [trajectory.get(c) for c in ('F', 'NONCRITICAL', 'CRITICAL', 'RESTORE')]
        pair_ready = lambda a, b: bool(a and b and a['technical_state'] == b['technical_state'] == 'completed')
        behavior.append({'paper_id': path.stem, 'question_id': ident[0], 'part_id': ident[1],
                         'noncritical_harmful_flip': int(control['correct'] != 1) if pair_ready(baseline, control) and baseline['correct'] == 1 else None,
                         'critical_target_abstention': critical['abstains'] if critical and critical['technical_state'] == 'completed' and critical['answerability'] == 'unanswerable' else None,
                         'restore_correct': restore['correct'] if pair_ready(baseline, restore) and baseline['correct'] == 1 else None,
                         'restore_after_valid_shrink': restore['correct'] if pair_ready(critical, restore) and critical['abstains'] == 1 and baseline and baseline['correct'] == 1 else None})
    write_csv(config.output / 'critical_trajectories.csv', traces)
    write_csv(config.output / 'critical_behavior.csv', behavior)
    decisions: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for r in rows:
        decisions[(r['split'], r['condition'], r['answerability'], state(r))] += 1
    write_csv(config.output / 'decision_matrix.csv', [{'split': k[0], 'condition': k[1], 'answerability': k[2], 'response_state': k[3], 'parts': v} for k, v in decisions.items()])
    repeat_rows = []
    for r in rows:
        if r['condition'] not in ('ORDER', 'REPEAT2', 'REPEAT3'):
            continue
        base = indexed.get((r['paper_id'], 'F', r['question_id'], r['part_id']))
        ready = base and base['technical_state'] == r['technical_state'] == 'completed'
        repeat_rows.append({'paper_id': r['paper_id'], 'condition': r['condition'], 'question_id': r['question_id'], 'part_id': r['part_id'],
                            'baseline_state': state(base) if base else 'technical_missing', 'changed_state': state(r),
                            'correct_delta': r['correct'] - base['correct'] if ready else None,
                            'harmful_flip': int(r['correct'] != 1) if ready and base['correct'] == 1 else None})
    write_csv(config.output / 'repeat_and_order.csv', repeat_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in metrics:
        grouped[(r['split'], r['condition'], r['aspect'])].append(r)
    summaries = []
    for key, values in grouped.items():
        for name in ('q_fixed', 'answer_coverage', 'unsupported_burden', 'answered_risk', 'reasonable_abstention_sensitivity', 'unnecessary_abstention_rate'):
            summaries.append(dict(split=key[0], condition=key[1], aspect=key[2], metric=name, planned_papers=len(values),
                                  **interval([r[name] for r in values if r[name] is not None], config.seed)))
    write_csv(config.output / 'condition_summary.csv', summaries)


def material_checks(config: Config) -> None:
    cohort = read(config.output / 'cohort.json')
    audits = []
    for paper in cohort['exploration']:
        base = packet(config, paper, 'F')
        e = [set(read(config.output / 'masks' / c / f'{paper}.json')['retained']) for c in ('E25', 'E50', 'E75')]
        g = [set(read(config.output / 'masks' / c / f'{paper}.json')['retained']) for c in ('G25', 'G50')]
        if not (e[0] <= e[1] <= e[2] and g[0] <= g[1]):
            raise ValueError(f'Non-nested masks: {paper}')
        for condition in ('E25', 'E50', 'E75'):
            if packet(config, paper, condition)['graph'] != base['graph']:
                raise ValueError('Original deletion changed graph')
        for condition in ('G25', 'G50', 'K3', 'K5'):
            if original_blocks(packet(config, paper, condition)) != original_blocks(base):
                raise ValueError('Graph change altered original evidence')
        audits.append({'paper_id': paper, 'nested_original_masks': True, 'nested_parent_masks': True, 'independent_manipulations': True})
    write(config.output / 'material_validation.json', audits)
    matches = []
    for paper in cohort['small']:
        for direct, full, view in [('DW_F', 'F', 'F'), ('DW_E50', 'E50', 'E50'), ('DW_LUNA', 'LUNA', 'F'), ('DW_CRITICAL', 'CRITICAL', 'CRITICAL')]:
            direct_path = config.output / 'report_inputs' / direct / f'{paper}.json'
            if not direct_path.exists():
                continue
            actual = read(direct_path)
            expected = packet(config, paper, view)
            valid = actual['manuscript'] == expected['manuscript'] and actual['graph'] == expected['graph'] and actual['original_evidence'] == original_blocks(expected)
            if not valid:
                raise ValueError(f'DW material mismatch {paper}/{direct}')
            matches.append({'paper_id': paper, 'direct': direct, 'full': full, 'material_match': True, 'truncated': False})
    write_csv(config.output / 'direct_material_checks.csv', matches)


def cost_quality(config: Config, tasks: list[dict[str, Any]], metrics: list[dict[str, Any]], costs: list[dict[str, Any]]) -> None:
    points = []
    for task in tasks:
        if task['condition'] not in ('F', 'REPEAT2', 'LOW', 'HIGH', 'DW_F', 'LUNA', 'DW_LUNA') or task['paper_id'] not in read(config.output / 'cohort.json')['small']:
            continue
        paper, condition = task['paper_id'], task['condition']
        calls = [r for r in costs if r['paper_id'] == paper and r.get('cost_category') == 'inference' and
                 (r['condition'] == condition or r['condition'] in (condition + '_gear', condition + '_graph'))]
        complete_usage = bool(calls) and all(r.get('input_tokens') is not None and r.get('output_tokens') is not None for r in calls)
        timing = config.output / 'timing' / condition / f'{paper}.json'
        latency = latency_scope(read(timing), calls) if timing.exists() else {'wall_seconds': None, 'wall_time_scope': 'unavailable', 'final_invocation_wall_seconds': None}
        for family, aspects in [('H', ['historical_verification']), ('structure', ['joint_contribution', 'structural_resolution', 'citation_contact'])]:
            quality = [r for r in metrics if (r['paper_id'], r['condition']) == (paper, condition) and r['aspect'] in aspects and r['fixed_denominator']]
            complete = bool(quality) and all(r['state'] == 'completed' for r in quality)
            denominator = sum(r['fixed_denominator'] for r in quality)
            answered = sum(r['answered_risk_denominator'] for r in quality)
            points.append({'paper_id': paper, 'condition': condition, 'family': family,
                           'model': 'gpt-5.6-luna' if condition in ('LUNA', 'DW_LUNA') else config.model,
                           'q_fixed': sum(r['q_fixed_numerator'] for r in quality) / denominator if complete else None,
                           'answered_risk': sum(r['answered_risk_numerator'] for r in quality) / answered if complete and answered and not any(r['answered_risk_unresolved'] for r in quality) else None,
                           'risk_unresolved': sum(r['answered_risk_unresolved'] for r in quality),
                           'total_input_output_tokens': sum(r['input_tokens'] + r['output_tokens'] for r in calls) if complete_usage else None,
                           'call_seconds_sum': sum(r['seconds'] for r in calls) if calls and all(r.get('seconds') is not None for r in calls) else None,
                           **latency,
                           'call_attempts': len(calls), 'failed_attempts': sum(r['state'] not in ('completed', 'reused') for r in calls),
                           'cost_scope': 'fixed_material_inference', 'currency_cost': None,
                           'operating_role': 'historical_standard_background' if condition == 'F' else 'standard_budget' if condition == 'REPEAT2' else condition,
                           'frontier_eligible': condition != 'F'})
    write_csv(config.output / 'cost_quality_points.csv', points)
    # Actual discrete operating points only, kept within a model and complete paired cohort.
    settings = []
    for model in {r['model'] for r in points}:
        for family in ('H', 'structure'):
            current = [r for r in points if r['model'] == model and r['family'] == family and r['frontier_eligible']]
            methods = sorted({r['condition'] for r in current})
            valid_by_method = {m: {r['paper_id'] for r in current if r['condition'] == m and all(r[k] is not None for k in ('q_fixed', 'answered_risk', 'total_input_output_tokens')) and r['risk_unresolved'] == 0} for m in methods}
            common = set.intersection(*valid_by_method.values()) if valid_by_method else set()
            for method in methods:
                observed = [r for r in current if r['condition'] == method and r['paper_id'] in common]
                settings.append({'model': model, 'family': family, 'condition': method, 'paired_papers': len(observed),
                                 'quality': sum(r['q_fixed'] for r in observed) / len(observed) if observed else None,
                                 'risk': sum(r['answered_risk'] for r in observed) / len(observed) if observed else None,
                                 'tokens': sum(r['total_input_output_tokens'] for r in observed) / len(observed) if observed else None})
    for point in settings:
        if not point['paired_papers']:
            point['observed_nondominated'] = None
            continue
        competitors = [p for p in settings if p['model'] == point['model'] and p['family'] == point['family'] and p['paired_papers']]
        point['observed_nondominated'] = not any(p['quality'] >= point['quality'] and p['risk'] <= point['risk'] and p['tokens'] <= point['tokens'] and
                                                (p['quality'] > point['quality'] or p['risk'] < point['risk'] or p['tokens'] < point['tokens']) for p in competitors)
    write_csv(config.output / 'observed_cost_quality_settings.csv', settings)


def latency_scope(timing: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    measured = timing.get('wall_seconds')
    if timing.get('state') != 'completed' or measured is None:
        return {'wall_seconds': None, 'final_invocation_wall_seconds': None, 'wall_time_scope': 'incomplete'}
    started = datetime.fromisoformat(timing['ended_at']) - timedelta(seconds=measured)
    earlier = [r for r in calls if not r.get('reused') and r.get('started_at') and
               datetime.fromisoformat(r['started_at']) < started - timedelta(seconds=5)]
    return {'wall_seconds': None if earlier else measured, 'final_invocation_wall_seconds': measured,
            'wall_time_scope': 'resumed_partial_invocation' if earlier else 'complete_generation_invocation',
            'reused_stage_count': sum(bool(r.get('reused')) for r in calls),
            'includes_queue_wait': timing.get('includes_queue_wait')}


def export_latency(config: Config, costs: list[dict[str, Any]]) -> None:
    rows = []
    for path in sorted((config.output / 'timing').glob('*/*.json')):
        condition, paper = path.parent.name, path.stem
        calls = [r for r in costs if r['paper_id'] == paper and r.get('cost_category') == 'inference' and
                 r['condition'] in (condition, condition + '_gear', condition + '_graph')]
        rows.append({'paper_id': paper, 'condition': condition, **latency_scope(read(path), calls)})
    write_csv(config.output / 'latency_audit.csv', rows)


def export_material_tables(config: Config) -> None:
    originals, graph_rows = [], []
    for path in sorted((config.output / 'inputs').glob('*/*.json')):
        paper, view = path.stem, path.parent.name
        base, current = packet(config, paper, 'F'), read(path)
        aliases = read(config.output / 'identities' / path.name)
        retained = {(b['block_id'], p['source_id']) for b in original_blocks(current) for p in b['provenance']}
        mapping_path = config.output / 'order_mapping' / path.name
        mapping = read(mapping_path) if view == 'ORDER' and mapping_path.exists() else {}
        for block in original_blocks(base):
            block_id = mapping.get(block['block_id'], block['block_id'])
            for source in block['provenance']:
                originals.append({'paper_id': paper, 'view': view, 'block_id': block_id, 'baseline_block_id': block['block_id'],
                                  'source_id': source['source_id'], 'work_id': aliases[source['source_id']],
                                  'source_type': source['source_type'], 'characters': len(block['text']),
                                  'retained': (block_id, source['source_id']) in retained})
        actual_graph = current['graph'] or {'cards': [], 'joint': {'historical_edges': []}}
        actual_edges = {tuple(sorted(e)) for e in actual_graph['joint']['historical_edges']}
        actual_pairs = {(card['claim']['claim_id'], n['claim_id']) for card in actual_graph['cards'] for n in card['neighbors']}
        for card in base['graph']['cards']:
            for neighbor in card['neighbors']:
                key = (card['claim']['claim_id'], neighbor['claim_id'])
                graph_rows.append({'paper_id': paper, 'view': view, 'kind': 'target_semantic_neighbor', 'source_node': key[0], 'target_node': key[1],
                                   'parent_paper_id': neighbor['parent_paper_id'], 'retained': key in actual_pairs,
                                   'cosine_similarity': neighbor['cosine_similarity'], 'semantic_rank': neighbor['semantic_rank'],
                                   'direct_citation': neighbor['direct_citation'], 'two_hop_path_count': neighbor['two_hop_path_count']})
        for edge in base['graph']['joint']['historical_edges']:
            graph_rows.append({'paper_id': paper, 'view': view, 'kind': 'historical_semantic_edge', 'source_node': edge[0], 'target_node': edge[1],
                               'retained': tuple(sorted(edge)) in actual_edges})
    write_csv(config.output / 'visible_materials.csv', originals)
    write_csv(config.output / 'graph_materials.csv', graph_rows)


def model_completion_audit(config: Config) -> None:
    from figure_pipeline.fig4_explanation_100.models import Report

    from .pipeline import public
    records = [(p, read(p)) for p in (config.old / 'logs/calls').glob('*/record.json')]
    rows = []
    for paper in read(config.output / 'cohort.json')['exploration']:
        full = packet(config, paper, 'F')
        path = config.old / 'report_inputs/LUNA' / f'{paper}.json'
        report = config.output / 'reports/LUNA' / f'{paper}.json'
        actual = read(path) if path.exists() else {}
        material_match = (actual.get('manuscript') == full['manuscript'] and actual.get('graph') == full['graph'] and
                          actual.get('original_evidence') == original_blocks(full) and actual.get('public_tasks') == public(config, paper))
        candidates = [(p, r) for p, r in records if (r.get('paper_id'), r.get('stage'), r.get('method'), r.get('state')) == (paper, 'reports', 'LUNA', 'completed')]
        selected = max(candidates, key=lambda pair: pair[1]['time']) if candidates else None
        record = selected[1] if selected else {}
        usage = record.get('usage') or {}
        valid = False
        if report.exists():
            Report.model_validate(read(report))
            valid = True
        rows.append({'paper_id': paper, 'condition': 'LUNA', 'full_material_match': material_match, 'report_schema_complete': valid,
                     'body_characters': len(read(report)['body']) if valid else None, 'requested_model': record.get('model'),
                     'requested_effort': record.get('effort'), 'turn_completed_usage_present': bool(usage), 'cli_returncode': record.get('returncode'),
                     'output_tokens': usage.get('output_tokens'), 'input_tokens': usage.get('input_tokens'),
                     'client_input_truncation_detected': False if material_match else None, 'server_truncation_status': 'not_exposed',
                     'scientific_omissions': 'requires_corrected_part_evaluation', 'call_record': str(selected[0]) if selected else None})
    write_csv(config.output / 'model_completion_audit.csv', rows)
