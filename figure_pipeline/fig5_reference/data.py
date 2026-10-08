from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fig5_robustness'
OUT = ROOT / 'outputs/fig5_reference'
ASPECTS = ['historical_verification', 'knowledge_position', 'joint_contribution',
           'structural_resolution', 'citation_contact']
SHORT = ['History', 'Neighborhood', 'Joint structure', 'Structural values', 'Citation contact']
CONDITIONS = ['F', 'E50', 'K5', 'LUNA']
FIELDS = ['life', 'physical_engineering', 'medicine', 'earth_environment']


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(name: str, rows: list[dict[str, Any]]) -> None:
    write(OUT / 'data' / f'{name}.json', rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / 'data' / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'mean': None, 'low': None, 'high': None}
    a = np.array(values)
    rng = np.random.default_rng(20261002)
    boot = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    return {'n': len(a), 'mean': float(a.mean()), 'low': float(np.quantile(boot, .025)),
                'high': float(np.quantile(boot, .975))}


def coverage_tables(observed: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> None:
    definitions = [('sparse_claim_fraction', 0.0), ('mean_nearest_similarity', None),
                   ('readable_sources', None), ('fulltext_fraction', 1.0)]
    assignments, summaries, rules = [], [], []
    for variable, threshold in definitions:
        threshold = float(np.median([r[variable] for r in coverage])) if threshold is None else threshold
        lookup = {r['paper_id']: int(r[variable] == 1) if variable == 'fulltext_fraction'
                  else int(r[variable] > threshold) for r in coverage if r[variable] is not None}
        rules.append({'variable': variable, 'threshold': threshold,
                          'rule': '0: <1; 1: =1' if variable == 'fulltext_fraction' else '0: <=threshold; 1: >threshold'})
        for row in coverage:
            assignments.append({'paper_id': row['paper_id'], 'variable': variable, 'value': row[variable],
                                    'stratum': lookup.get(row['paper_id'])})
        for stratum in (0, 1):
            ids = [k for k, v in lookup.items() if v == stratum]
            vals = [r[variable] for r in coverage if r['paper_id'] in ids]
            for aspect in ASPECTS:
                records = [r for r in observed if r['paper_id'] in ids and r['aspect'] == aspect]
                good = [r for r in records if r['grounded_coverage'] is not None]
                summaries.append(dict(variable=variable, stratum=stratum, aspect=aspect,
                                      planned=len(ids), missing=len(records)-len(good), threshold=threshold,
                                      minimum=min(vals), maximum=max(vals), paper_ids=[r['paper_id'] for r in good],
                                      **interval([r['grounded_coverage'] for r in good])))
    table('b_coverage_assignments', assignments)
    table('b_stratum_summary', summaries)
    table('b_stratum_rules', rules)


def judgment_tables(metrics: list[dict[str, Any]]) -> None:
    summaries = []
    for condition in CONDITIONS:
        for aspect in ASPECTS:
            rows = [r for r in metrics if r['condition'] == condition and r['aspect'] == aspect]
            good = [r for r in rows if r['state'] == 'completed']
            result = {'condition': condition, 'aspect': aspect, 'planned': len(rows), 'n': len(good),
                          'missing': sum(r['state'] == 'technical_missing' for r in rows),
                          'not_applicable': sum(r['state'] == 'not_applicable' for r in rows),
                          'paper_ids': [r['paper_id'] for r in good]}
            for key in ['explicit_abstention_numerator', 'abstention_reasonable', 'abstention_unnecessary',
                        'abstention_unresolved', 'unsupported_definitive_numerator',
                        'unsupported_definitive_unresolved', 'scientific_unresolved']:
                result[key] = float(np.mean([r[key] / r['fixed_denominator'] for r in good])) if good else None
            summaries.append(result)
    table('d_judgment_rates', summaries)


def cost_tables(costs: list[dict[str, Any]]) -> None:
    base = {r['paper_id']: r for r in costs if r['condition'] == 'F'}
    rows, summary = [], []
    for row in costs:
        if row['condition'] not in ('E50', 'K5', 'LUNA'):
            continue
        f = base[row['paper_id']]
        for key in ['input_tokens', 'output_tokens', 'call_seconds_sum']:
            complete = row['complete_cost'] and f['complete_cost'] and not row['failures'] and not f['failures']
            rows.append({'paper_id': row['paper_id'], 'condition': row['condition'], 'metric': key,
                             'value': row[key], 'baseline': f[key], 'complete': complete,
                             'ratio': row[key] / f[key] if complete else None,
                             'reason': None if complete else 'Incomplete generation or unknown usage'})
    for condition in ('E50', 'K5', 'LUNA'):
        for key in ['input_tokens', 'output_tokens', 'call_seconds_sum']:
            vals = [r['ratio'] for r in rows if r['condition'] == condition and r['metric'] == key and r['complete']]
            summary.append({'condition': condition, 'metric': key, 'planned': 20, 'n': len(vals),
                                'median': float(np.median(vals)), 'q1': float(np.quantile(vals, .25)),
                                'q3': float(np.quantile(vals, .75)), 'minimum': min(vals), 'maximum': max(vals)})
    table('f_cost_ratios', rows)
    table('f_cost_summary', summary)


def prepare() -> None:
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    sources = ['observational_papers', 'observational_summary', 'observational_summary_without_development',
               'paper_coverage', 'paper_condition_metrics', 'paired_differences', 'paired_summary',
               'repeat_variation', 'attributable_costs', 'new_spending', 'model_calls', 'answer_parts',
               'run_summary', 'cohort', 'protocol']
    data = {name: read(SOURCE / f'{name}.json') for name in sources}
    for name in sources:
        for ext in ('.json', '.csv'):
            path = SOURCE / (name + ext)
            if path.exists():
                shutil.copy2(path, OUT / 'data' / path.name)
    table('a_field_coverage', [r for r in data['observational_summary'] if r['grouping'] == 'field'])
    coverage_tables(data['observational_papers'], data['paper_coverage'])
    judgment_tables(data['paper_condition_metrics'])
    cost_tables(data['attributable_costs'])
    table('e_repeat_paper_differences', [r for r in data['paired_differences'] if r['condition'] == 'F_REPEAT'])
    write(OUT / 'data/source_files.json', {'source_root': str(SOURCE), 'sources': sources,
          'note': 'Scientific labels reused unchanged. Only descriptive plotting aggregates are calculated.',
          'bootstrap_draws': 10000, 'seed': 20261002, 'paper_weighting': 'equal'})
    print('Prepared Fig5 plotting tables', flush=True)
