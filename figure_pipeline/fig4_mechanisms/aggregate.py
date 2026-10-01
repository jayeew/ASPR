"""Paper-level endpoints, paired contrasts and source tables; no rendering."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import PaperBootstrap
from figure_pipeline.fig3_revision.storage import Store, roster

from .config import COMPONENTS, CONDITIONS, RISK_TYPES, Config
from .io import artifact, jsonl, read, write, write_csv


def optional(path: Path) -> dict[str, Any] | None:
    return read(path) if path.is_file() else None


def paper_metric(config: Config, paper: str, condition: str) -> tuple[dict, list[dict]]:
    store = Store(config)
    extract = optional(store.path('extract', paper, condition))
    support = optional(store.path('support', paper, condition))
    novelty = optional(store.path('novelty', paper, condition))
    reference = store.get('reference', paper)
    clusters = optional(artifact(config.output, 'information_clusters', paper))
    relations = optional(artifact(config.output, 'cross_relations', paper))
    result = {'paper_id': paper, 'condition': condition, 'H': None, 'V': None, 'R': None,
              'V_cross': None, 'H_numerator': None, 'H_denominator': 0, 'R_numerator': None,
              'R_denominator': 0, 'unknown_support_share': None, 'improper_cross': None, 'missing_reasons': []}
    rows = []
    refs = [r for r in reference['items'] if r['state'] != 'insufficient_material'
            and r['historical_comparison_applicable']]
    result['H_denominator'] = len(refs)
    if extract is not None and novelty is not None:
        predictions = {p['core_id']: p for p in extract['predictions']}
        judgments = {v['core_id']: v for v in novelty['items']}
        if all(r['core_id'] in predictions and r['core_id'] in judgments for r in refs):
            count = sum(predictions[r['core_id']]['state'] != 'not_addressed'
                        and judgments[r['core_id']]['historical_comparison_correct'] is True
                        and judgments[r['core_id']]['scope_correct'] is True for r in refs)
            result.update(H_numerator=count, H=100*count/len(refs) if refs else None)
        else:
            result['missing_reasons'].append('historical_evaluation_incomplete')
    else:
        result['missing_reasons'].append('historical_evaluation_missing')
    labels = {r['unit_id']: r for r in support['units']} if support else {}
    units = [u for u in extract['units'] if u['needs_verification'] or u['substantive']] if extract else []
    complete = extract is not None and support is not None and all(u['unit_id'] in labels for u in units)
    result['R_denominator'] = len(units)
    for unit in extract['units'] if extract else []:
        rows.append({'paper_id': paper, 'condition': condition, **unit, 'assessment': labels.get(unit['unit_id'])})
    if complete:
        count = sum(bool(labels[u['unit_id']]['errors']) or labels[u['unit_id']]['support'] == 'contradicted' for u in units)
        unknown = sum(labels[u['unit_id']]['support'] in {'not_verifiable', 'partly_supported'}
                      or labels[u['unit_id']]['scope_correct'] is None for u in units)
        result.update(R_numerator=count, R=100*count/len(units) if units else None,
                      unknown_support_share=unknown/len(units) if units else None)
        result['improper_cross'] = sum(u['insight_type'] == 'cross_contribution'
            and (bool(labels[u['unit_id']]['errors']) or labels[u['unit_id']]['support'] == 'contradicted') for u in units)
    else:
        result['missing_reasons'].append('assertion_evaluation_incomplete')
    if complete and clusters is not None and relations is not None:
        cross = {r['unit_key'] for r in relations['items'] if r['valid'] and r['condition'] == condition}
        valid, cross_valid = set(), set()
        for cluster in clusters['clusters']:
            members = [k for k in cluster['unit_keys'] if clusters['mapping'][k]['condition'] == condition]
            supported = [k for k in members if clusters['mapping'][k]['support'] is not None
                         and clusters['mapping'][k]['support']['support'] == 'supported'
                         and clusters['mapping'][k]['support']['scope_correct'] is True
                         and not clusters['mapping'][k]['support']['errors']
                         and clusters['mapping'][k]['support']['nonparaphrase_insight']]
            if cluster['kind'] == 'cross_contribution':
                supported = [k for k in supported if k in cross]
            if supported:
                valid.add(cluster['cluster_id'])
                if cluster['kind'] == 'cross_contribution':
                    cross_valid.add(cluster['cluster_id'])
        result.update(V=len(valid), V_cross=len(cross_valid))
    else:
        result['missing_reasons'].append('information_evaluation_incomplete')
    return result, rows


def estimate(values: dict[str, float], config: Config) -> dict[str, Any]:
    return PaperBootstrap(sorted(values), config).estimate(values) if values else {
        'n': 0, 'estimate': None, 'low': None, 'high': None}


def component_effects(metrics: dict[tuple[str, str], dict], papers: list[dict], config: Config) -> None:
    rows, summaries = [], []
    for component, comparator in COMPONENTS.items():
        for metric in ('H', 'V', 'R'):
            values = {}
            for paper in papers:
                ident = paper['paper_id']
                full, comparison = metrics[ident, 'F'], metrics[ident, comparator]
                a, b = full[metric], comparison[metric]
                eligible = a is not None and b is not None
                delta = a-b if eligible else None
                if eligible:
                    values[ident] = delta
                denominator = metric+'_denominator'
                rows.append({'paper_id': ident, 'component': component, 'metric': metric,
                    'full_condition': 'F', 'comparator_condition': comparator, 'full_value': a,
                    'comparator_value': b, 'delta': delta, 'eligible': eligible,
                    'missing_reason': '' if eligible else 'missing_or_inapplicable_endpoint',
                    'full_denominator': full.get(denominator), 'comparator_denominator': comparison.get(denominator)})
            stats = estimate(values, config)
            quantiles = np.quantile(list(values.values()), [.25, .5, .75]).tolist() if values else [None]*3
            summaries.append({'component': component, 'metric': metric, 'n_papers': stats['n'],
                'mean_delta': stats['estimate'], 'ci_low': stats['low'], 'ci_high': stats['high'],
                'q1': quantiles[0], 'median': quantiles[1], 'q3': quantiles[2]})
    write_csv(config.output/'component_effects_paper.csv', rows)
    write_csv(config.output/'component_effects_summary.csv', summaries)


def interaction(metrics: dict[tuple[str, str], dict], papers: list[dict], config: Config) -> None:
    rows, values = [], {c: {} for c in ('T', 'E', 'G', 'F', 'I')}
    for paper in papers:
        ident = paper['paper_id']
        v = {c: metrics[ident, c]['V'] for c in ('T', 'E', 'G', 'F')}
        complete = all(x is not None for x in v.values())
        delta = v['F']-v['E']-v['G']+v['T'] if complete else None
        rows.append({'paper_id': ident, **{'V_'+c: x for c, x in v.items()}, 'I': delta,
                     'complete': complete, 'missing_reason': '' if complete else 'four_condition_V_incomplete'})
        if complete:
            for c, x in {**v, 'I': delta}.items():
                values[c][ident] = x
    stats = {c: estimate(v, config) for c, v in values.items()}
    means = {c: s['estimate'] for c, s in stats.items()}
    coefficients = None if not values['I'] else {'intercept': means['T'], 'gear': means['E']-means['T'],
        'graph': means['G']-means['T'], 'interaction': means['I']}
    write_csv(config.output/'interaction_paper.csv', rows)
    write(config.output/'interaction_summary.json', {'n_papers': len(values['I']), 'statistics': stats,
        'bilinear_coefficients': coefficients, 'interpolation': 'visual_only_four_observed_conditions'})


def joint_effects(metrics: dict[tuple[str, str], dict], papers: list[dict], config: Config) -> None:
    rows = []
    for paper in papers:
        ident = paper['paper_id']
        graph = optional(config.output/'inputs/joint_structure'/f'{ident}.json') or {}
        base = {k: v for k, v in graph.items() if k not in {'exclusive_pair_ids', 'historical_nodes',
                     'historical_edges', 'insertion_edges', 'claim_neighbors'}}
        full, no_joint = metrics[ident, 'F']['V_cross'], metrics[ident, 'F_noJ']['V_cross']
        complete = graph.get('J_topo') is not None and full is not None and no_joint is not None
        rows.append({'paper_id': ident, **base, 'V_cross_F': full, 'V_cross_F_noJ': no_joint,
            'improper_cross_F': metrics[ident, 'F'].get('improper_cross'),
            'improper_cross_F_noJ': metrics[ident, 'F_noJ'].get('improper_cross'),
            'delta_cross': full-no_joint if full is not None and no_joint is not None else None,
            'complete': complete, 'missing_reason': '' if complete else 'joint_or_cross_evaluation_incomplete'})
    write_csv(config.output/'joint_effects_paper.csv', rows)


def risk_exports(papers: list[dict], config: Config) -> tuple[int, int]:
    inputs, transitions, events, matrix = [], [], [], []
    assessed = 0
    ordered = sorted(papers, key=lambda p: (p.get('field_name') or 'Unknown', p['paper_id']))
    write_csv(config.output/'paper_order.csv', [{'paper_id': p['paper_id'], 'domain': p.get('field_name'),
              'paper_order': i} for i, p in enumerate(ordered)])
    for order, paper in enumerate(ordered):
        ident = paper['paper_id']
        bundle = optional(artifact(config.output, 'condition_inputs', ident, 'F'))
        if bundle:
            inputs.extend({'paper_id': ident, 'condition': 'F', **f} for f in bundle['findings'])
        outcomes = optional(artifact(config.output, 'fusion_transitions', ident))
        if outcomes:
            assessed += 1
            transitions.extend(outcomes['items'])
        risks = optional(artifact(config.output, 'fusion_risks', ident)) or {'events': [], 'assessments': []}
        events.extend(risks['events'])
        assessments = {a['risk_type']: a for a in risks['assessments']}
        for kind in RISK_TYPES:
            assessment = assessments.get(kind, {})
            found = [e for e in risks['events'] if e['risk_type'] == kind and e['confirmed']]
            if found:
                state = 'present'
            elif not assessment:
                state = 'incomplete'
            elif not assessment['applicability']:
                state = 'not_applicable'
            elif assessment['assessment_complete']:
                state = 'absent'
            else:
                state = 'unresolved'
            matrix.append({'paper_id': ident, 'domain': paper.get('field_name'), 'paper_order': order,
                'risk_family': 'new_error' if kind in RISK_TYPES[:4] else 'supported_content_change',
                'risk_type': kind, 'event_count_confirmed': len(found),
                'eligible_unit_count': assessment.get('eligible_unit_count'),
                'assessment_complete': assessment.get('assessment_complete', False),
                'unresolved_unit_count': assessment.get('unresolved_unit_count'),
                'applicability': assessment.get('applicability'), 'cell_state': state,
                'event_ids': [e['event_id'] for e in found]})
    counts = Counter((r['source_group'], r['full_status']) for r in transitions)
    source_counts = Counter(r['source_group'] for r in transitions)
    write_csv(config.output/'error_transition_counts.csv', [{'source_group': source, 'full_status': status,
        'count': counts[source, status] if assessed else None, 'source_denominator': source_counts[source] if assessed else None,
        'branch_dependent_correction_count': sum(r['source_group'] == source and r['full_status'] == status
            and r['branch_dependent_correction'] for r in transitions) if assessed else None,
        'assessed_papers': assessed, 'total_papers': len(papers)}
        for source in ('GEAR-only', 'Graph-only', 'Shared')
        for status in ('corrected', 'not_propagated', 'propagated', 'unresolved')])
    summaries = [{'risk_type': kind, 'n': sum(r['cell_state'] == 'present' for r in matrix if r['risk_type'] == kind),
        'N': sum(r['cell_state'] in {'present', 'absent'} for r in matrix if r['risk_type'] == kind),
        'unknown': sum(r['cell_state'] in {'unresolved', 'incomplete'} for r in matrix if r['risk_type'] == kind),
        'not_applicable': sum(r['cell_state'] == 'not_applicable' for r in matrix if r['risk_type'] == kind)} for kind in RISK_TYPES]
    write_csv(config.output/'paper_risk_matrix.csv', matrix)
    write_csv(config.output/'risk_summary.csv', summaries)
    jsonl(config.output/'fusion_inputs.jsonl', inputs)
    jsonl(config.output/'fusion_transitions.jsonl', transitions)
    jsonl(config.output/'fusion_risk_events.jsonl', events)
    return assessed, len(events)


def aggregate(config: Config) -> dict[str, Any]:
    papers = roster(config)
    metrics, units, clusters_out, relations_out = {}, [], [], []
    for paper in papers:
        ident = paper['paper_id']
        if not Store(config).path('reference', ident).exists():
            for condition in CONDITIONS:
                metrics[ident, condition] = {'paper_id': ident, 'condition': condition,
                    'H': None, 'V': None, 'R': None, 'V_cross': None, 'missing_reasons': ['input_missing']}
            continue
        for condition in CONDITIONS:
            metrics[ident, condition], rows = paper_metric(config, ident, condition)
            units.extend(rows)
        clusters = optional(artifact(config.output, 'information_clusters', ident))
        if clusters:
            clusters_out.extend({'paper_id': ident, **c,
                'members': {k: clusters['mapping'][k] for k in c['unit_keys']}} for c in clusters['clusters'])
        relations = optional(artifact(config.output, 'cross_relations', ident))
        if relations:
            relations_out.extend({'paper_id': ident, **r} for r in relations['items'])
    costs = usage_rows(config)
    costs_by_key = {(r['paper_id'], r['condition']): r for r in costs}
    for key, metric in metrics.items():
        report = optional(config.output/'reports'/key[1]/'papers'/f'{key[0]}.json')
        metric['body_chars'] = report.get('body_chars') if report else None
        metric['condition_calls'] = costs_by_key.get(key, {}).get('calls', 0)
        metric['input_tokens'] = costs_by_key.get(key, {}).get('input_tokens')
        metric['output_tokens'] = costs_by_key.get(key, {}).get('output_tokens')
    write_csv(config.output/'costs.csv', costs)
    write_csv(config.output/'paper_metrics.csv', list(metrics.values()))
    jsonl(config.output/'report_units.jsonl', units)
    jsonl(config.output/'information_clusters.jsonl', clusters_out)
    jsonl(config.output/'joint_relations.jsonl', relations_out)
    component_effects(metrics, papers, config)
    interaction(metrics, papers, config)
    joint_effects(metrics, papers, config)
    assessed, event_count = risk_exports(papers, config)
    summary = status(config)
    summary.update(e1_assessed_papers=assessed, risk_events=event_count,
        evaluable_endpoints={m: sum(r[m] is not None for r in metrics.values()) for m in ('H', 'V', 'R')},
        interpretation='fixed_existing_analysis_information_effect', report_length_target=None,
        scientific_review='gpt-5.6-luna; not new human validation')
    write(config.output/'summary.json', summary)
    (config.output/'summary.md').write_text('# Fig4 数据准备状态\n\n'+
        '\n'.join(f'- {k}: {v}' for k, v in summary.items())+'\n\n空值表示未完成或不适用，不是实验零值。\n', encoding='utf-8')
    return summary


def usage_rows(config: Config) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (config.output/'logs/calls').glob('*/record.json'):
        call = read(path)
        key = (call['paper_id'], call.get('method') or 'shared_evaluation')
        row = groups.setdefault(key, {'paper_id': key[0], 'condition': key[1], 'calls': 0,
            'failures': 0, 'seconds': 0.0, 'input_tokens': 0, 'output_tokens': 0,
            'usage_missing_calls': 0, 'shared_task': key[1] not in CONDITIONS})
        row['calls'] += 1
        row['failures'] += call['state'] not in {'running', 'completed'}
        row['seconds'] += call.get('seconds', 0)
        if call.get('usage'):
            row['input_tokens'] += call['usage'].get('input_tokens', 0)
            row['output_tokens'] += call['usage'].get('output_tokens', 0)
        else:
            row['usage_missing_calls'] += 1
    return list(groups.values())


def status(config: Config) -> dict[str, Any]:
    papers = roster(config)
    counts = {'prepared_papers': 0, 'reports': 0, 'extract': 0, 'support': 0, 'novelty': 0,
              'information_papers': 0, 'cross_relation_papers': 0, 'fusion_risk_papers': 0}
    store = Store(config)
    for p in papers:
        ident = p['paper_id']
        counts['prepared_papers'] += (config.output/'inputs/common'/f'{ident}.json').is_file()
        for c in CONDITIONS:
            counts['reports'] += (config.output/'reports'/c/'papers'/f'{ident}.json').is_file()
            for stage in ('extract', 'support', 'novelty'):
                counts[stage] += store.path(stage, ident, c).is_file()
        for name, stage in [('information_papers', 'information_clusters'), ('cross_relation_papers', 'cross_relations'),
                            ('fusion_risk_papers', 'fusion_risks')]:
            counts[name] += artifact(config.output, stage, ident).is_file()
    calls = [read(p) for p in (config.output/'logs/calls').glob('*/record.json')]
    usage = Counter()
    for call in calls:
        if call.get('usage'):
            usage.update({k: v for k, v in call['usage'].items() if isinstance(v, (int, float))})
    latest = {}
    log_path = config.output/'run_log.jsonl'
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            latest[row['stage'], row['paper_id'], row['condition']] = row
    failures = [row for row in latest.values() if row['state'] == 'failed']
    return {'papers': len(papers), 'expected_reports': len(papers)*len(CONDITIONS), **counts,
            'pending_reports': len(papers)*len(CONDITIONS)-counts['reports'], 'failed_objects': failures,
            'actual_calls': len(calls), 'call_states': dict(Counter(c['state'] for c in calls)),
            'usage': dict(usage), 'usage_missing_calls': sum(not c.get('usage') for c in calls),
            'model': config.model}
