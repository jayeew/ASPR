"""Row exports, explicit denominators, shared paper bootstrap and forty component datasets."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import COMPARATORS, METHODS, Config
from .progress import log
from .storage import MissingInput, Store, read, roster, write

DIMENSIONS = ('contribution_fidelity', 'historical_increment', 'knowledge_relations', 'scope_uncertainty', 'whole_paper_synthesis')
PREF_DIMS = ('overall', 'increment_clarity', 'evidence_traceability', 'knowledge_usefulness', 'appropriate_limitations')
ERRORS = ('unsupported_definitive', 'false_antecedence', 'false_firstness', 'semantic_causal', 'omitted_scope')
KINDS = ('historical_increment', 'cross_work_relation', 'scope_correction', 'cross_contribution')
R_STATES = ('supported_difference', 'substantially_covered', 'bounded_increment', 'insufficient_material')
P_STATES = ('positive_increment', 'substantially_known', 'limited_increment', 'explicit_abstention', 'not_addressed')
COMPONENTS = {
'a1': ('Sample', 'table'), 'a2': ('Method inputs', 'table'), 'a3': ('Evaluation channels', 'table'), 'a4': ('Generation and evaluation materials', 'table'),
'b1': ('Historical reference composition', 'category'), 'b2': ('Core increment precision / recall', 'scatter'), 'b3': ('Reference × report judgment', 'matrix'), 'b4': ('Paired Full − GEAR', 'interval'), 'b5': ('Antecedence and overclaim', 'interval'),
'c1': ('Five quality dimensions', 'interval'), 'c2': ('Quality score space', 'scatter'), 'c3': ('Quality differences', 'interval'), 'c4': ('Shared contribution coverage', 'interval'), 'c5': ('Assertion composition', 'category'), 'c6': ('Quoted report example', 'case'),
'd1': ('Support × traceability', 'scatter'), 'd2': ('Support states', 'category'), 'd3': ('Error prevalence', 'interval'), 'd4': ('S − T', 'interval'),
'e1': ('Human issues and reasons', 'interval'), 'e2': ('Scientific disagreement', 'category'), 'e3': ('Reviewer attention', 'category'), 'e4': ('Tone and scientific stance', 'matrix'), 'e5': ('Across-round changes', 'matrix'), 'e6': ('Documented issue follow-up outcomes', 'category'),
'f1': ('Valid information count space', 'scatter'), 'f2': ('Quality and reliability', 'scatter'), 'f3': ('Six-method cluster intersection', 'category'), 'f4': ('Four information types', 'interval'), 'f5': ('Evidence chain', 'case'), 'f6': ('Information density', 'interval'),
'g1': ('Retention flow', 'flow'), 'g2': ('Branch union and Full', 'interval'), 'g3': ('Errors corrected, omitted, propagated, added', 'category'), 'g4': ('Important information retained', 'interval'),
'h1': ('Double-order categories', 'category'), 'h2': ('All decidable pairs: ternary preference', 'category'), 'h3': ('Order swap matrix', 'matrix'), 'h4': ('Per-dimension preference', 'interval'), 'h5': ('Repeat evaluation and controls', 'category'),
}


def rate(values: list[bool | float | None]) -> float | None:
    known = [float(v) for v in values if v is not None]
    return mean(known) if known else None


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator/denominator if denominator else None


def decode(choice: str, mapping: dict[str, str]) -> str:
    return mapping.get(choice, choice)


def paired_preference(first: dict[str, Any], second: dict[str, Any], dimension: str) -> str:
    choices = [decode(r[dimension], r['mapping']) for r in (first, second)]
    if 'cannot_judge' in choices:
        return 'cannot_judge'
    return choices[0] if choices[0] == choices[1] else 'order_sensitive'


def bootstrap(values: list[float], config: Config) -> dict[str, Any]:
    if not values:
        return {'n': 0, 'estimate': None, 'low': None, 'high': None}
    rng = random.Random(config.seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(config.bootstrap_repeats))
    return {'n': len(values), 'estimate': mean(values), 'low': draws[int(.025*len(draws))],
            'high': draws[min(len(draws)-1, int(.975*len(draws)))]}


class PaperBootstrap:
    """Every metric/condition uses the same sampled paper identities."""
    def __init__(self, ids: list[str], config: Config) -> None:
        import numpy as np
        self.ids = ids
        self.draws = np.random.default_rng(config.seed).integers(0, len(ids), (config.bootstrap_repeats, len(ids)))

    def estimate(self, values: dict[str, float | None]) -> dict[str, Any]:
        import numpy as np
        array = np.array([values.get(i) if values.get(i) is not None else np.nan for i in self.ids], dtype=float)
        n = int(np.isfinite(array).sum())
        if not n:
            return {'n': 0, 'missing': len(self.ids), 'estimate': None, 'low': None, 'high': None}
        sampled = array[self.draws]
        counts = np.isfinite(sampled).sum(axis=1)
        sums = np.nansum(sampled, axis=1)
        estimates = sums[counts > 0]/counts[counts > 0]
        return {'n': n, 'missing': len(self.ids)-n, 'estimate': float(np.nanmean(array)),
                'low': float(np.quantile(estimates, .025)), 'high': float(np.quantile(estimates, .975))}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v.replace('\x00', '\ufffd') if isinstance(v, str) else v
                         for k, v in row.items()} for row in rows)


class Aggregator:
    def __init__(self, config: Config, paper_ids: set[str] | None = None) -> None:
        self.config, self.store = config, Store(config)
        self.roster = roster(config)
        if paper_ids is not None:
            self.roster = [row for row in self.roster if row['paper_id'] in paper_ids]
        self.ids = [p['paper_id'] for p in self.roster]
        self.bootstrap = PaperBootstrap(self.ids, config)
        self.tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.metrics: list[dict[str, Any]] = []
        self.cache: dict[tuple[str, str, str], Any] = {}

    def get(self, stage: str, ident: str, method: str = '') -> Any:
        key = (stage, ident, method)
        if key not in self.cache:
            try:
                self.cache[key] = self.store.get(*key)
            except MissingInput as exc:
                self.cache[key] = None
                self.tables['missing'].append({'paper_id': ident, 'stage': stage, 'method': method, 'error': str(exc)})
        return self.cache[key]

    def rows(self, table: str, ident: str, method: str, rows: list[dict[str, Any]]) -> None:
        self.tables[table].extend({'paper_id': ident, 'method': method, **r} for r in rows)

    def metric(self, ident: str, method: str, name: str, numerator: float | None,
               denominator: float | None = 1, expected: int | None = None) -> None:
        value = ratio(numerator, denominator) if numerator is not None and denominator is not None else None
        self.metrics.append({'paper_id': ident, 'method': method, 'metric': name, 'value': value,
                             'numerator': numerator, 'denominator': denominator, 'expected': expected})

    def collect(self) -> None:
        for index, ident in enumerate(self.ids, 1):
            log(self.config, '汇总进度', f'论文{index}/{len(self.ids)}，读取已有评价并计算论文指标。', stage='aggregate', paper_id=ident)
            for stage, field in [('core', 'items'), ('reference', 'items'), ('checklist', 'concerns'),
                                 ('tone', 'items'), ('review_dynamics', 'changes')]:
                value = self.get(stage, ident)
                if value is not None:
                    self.rows(stage, ident, '', value[field])
            for method in METHODS:
                self.collect_method(ident, method)
            self.collect_fusion(ident)
            self.collect_preferences(ident)
            # Rechecks are expected only for the selected twenty.
            from .stages import recheck_ids
            if ident in recheck_ids(self.config):
                self.collect_recheck(ident)

    def collect_recheck(self, ident: str) -> None:
        value = self.get('recheck', ident)
        if value is None:
            return
        self.rows('rechecks', ident, '', value['checks'])
        self.rows('controls', ident, '', value['controls'])
        manifest = self.config.output/'inputs/tasks/recheck/papers'/ident/'repeat_batches.json'
        expected: dict[tuple[str, str], set[str]] = defaultdict(set)
        if manifest.exists():
            keys = {'support': 'unit_id', 'novelty': 'core_id', 'concerns': 'concern_id'}
            for batch in read(manifest)['tasks']:
                for job in batch['jobs']:
                    expected[(job['method'], job['kind'])].add(job['object'][keys[job['kind']]])
        grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for row in value['checks']:
            grouped[(row['method'], row['task'])].append(row)
            if row.get('different') is None:
                self.tables['missing'].append({'paper_id': ident, 'stage': 'recheck',
                    'method': row['method'], 'object_id': row['object_id'], 'task': row['task'],
                    'error': 'First-round object missing; repeat not comparable'})
        for method, kind in sorted(expected.keys() | grouped.keys()):
            rows = grouped[(method, kind)]
            missing = expected[(method, kind)]-{r['object_id'] for r in rows}
            self.rows('recheck_coverage', ident, method, [{'task': kind,
                'expected': len(expected[(method, kind)]) if manifest.exists() else None,
                'observed': len(rows), 'comparable': sum(r.get('different') is not None for r in rows),
                'same': sum(r.get('different') is False for r in rows),
                'different': sum(r.get('different') is True for r in rows),
                'uncomparable': sum(r.get('different') is None for r in rows), 'missing_objects': sorted(missing)}])
            for obj in sorted(missing):
                self.tables['missing'].append({'paper_id': ident, 'stage': 'recheck', 'method': method,
                    'task': kind, 'object_id': obj, 'error': 'Repeated object missing'})
        controls = {r['kind']: r for r in value['controls']}
        for kind in ('source_mismatch', 'firstness_overreach', 'scope_deletion'):
            pair = controls.get(kind)
            observed = sum(side in pair.get('judgments', {}) for side in ('original', 'altered')) if pair else 0
            self.rows('control_coverage', ident, '', [{'kind': kind, 'constructed': pair is not None,
                'expected_judgments': 2 if pair else 0, 'observed_judgments': observed,
                'status': 'not_constructed' if pair is None else 'completed' if observed == 2 else 'missing_judgment'}])

    def collect_method(self, ident: str, method: str) -> None:
        extract = self.get('extract', ident, method)
        support = self.get('support', ident, method)
        novelty = self.get('novelty', ident, method)
        quality = self.get('quality', ident, method)
        concerns = self.get('concerns', ident, method)
        if extract is not None:
            self.rows('assertions', ident, method, extract['units'])
            self.rows('predictions', ident, method, extract['predictions'])
        if support is not None:
            self.rows('support', ident, method, support['units'])
        if novelty is not None:
            self.rows('verdicts', ident, method, novelty['items'])
        if extract is not None and novelty is not None:
            self.core_metrics(ident, method, extract, novelty)
        if extract is not None and support is not None:
            self.support_metrics(ident, method, extract, support)
        if quality is not None:
            self.rows('quality', ident, method, quality['dimensions'])
            for dim in quality['dimensions']:
                self.metric(ident, method, 'quality_'+dim['dimension'], dim['score'])
        if concerns is not None:
            self.rows('concern_matches', ident, method, concerns['matches'])
            self.concern_metrics(ident, method, concerns)
        self.information_metrics(ident, method)

    def core_metrics(self, ident: str, method: str, extract: dict[str, Any], novelty: dict[str, Any]) -> None:
        reference = self.get('reference', ident)
        if reference is None:
            return
        predictions = {r['core_id']: r for r in extract['predictions']}
        verdicts = {r['core_id']: r for r in novelty['items']}
        determinate = [r for r in reference['items'] if r['state'] != 'insufficient_material']
        positive = [r for r in determinate if r['state'] in {'supported_difference', 'bounded_increment'}]
        available = [r for r in determinate if r['core_id'] in predictions and r['core_id'] in verdicts]
        complete = len(available) == len(determinate)
        asserted = [r for r in available if predictions[r['core_id']]['asserts_increment']]
        effective = {r['core_id'] for r in available if r['state'] in {'supported_difference', 'bounded_increment'}
            and predictions[r['core_id']]['asserts_increment'] and verdicts[r['core_id']]['effective_increment'] is True
            and verdicts[r['core_id']]['difference_correct'] is True and verdicts[r['core_id']]['scope_correct'] is True
            and verdicts[r['core_id']]['material_overclaim'] is False}
        self.metric(ident, method, 'recall', len(effective) if complete else None, len(positive), len(positive))
        self.metric(ident, method, 'precision', len(effective) if complete else None, len(asserted), len(determinate))
        self.metric(ident, method, 'core_technical_missing', len(determinate)-len(available))
        for metric, applicability, key in [('historical_comparison', 'historical_comparison_applicable', 'historical_comparison_correct'),
                                            ('scope_coverage', 'scope_applicable', 'scope_correct')]:
            refs = [r for r in determinate if r[applicability]]
            done = all(r['core_id'] in verdicts and r['core_id'] in predictions for r in refs)
            num = sum(predictions[r['core_id']]['state'] != 'not_addressed' and verdicts[r['core_id']][key] is True
                      for r in refs if r['core_id'] in verdicts and r['core_id'] in predictions)
            self.metric(ident, method, metric, num if done else None, len(refs), len(refs))
        for error in ('false_firstness', 'false_antecedence', 'material_overclaim'):
            values = [verdicts[r['core_id']][error] for r in available]
            self.metric(ident, method, error, sum(v is True for v in values) if complete else None, len(available))
        insufficient = [r for r in reference['items'] if r['state'] == 'insufficient_material']
        self.metric(ident, method, 'insufficient_material_assertions',
                    sum(predictions[r['core_id']]['asserts_increment'] for r in insufficient if r['core_id'] in predictions),
                    sum(r['core_id'] in predictions for r in insufficient), len(insufficient))
        for ref in reference['items']:
            pred, verdict = predictions.get(ref['core_id']), verdicts.get(ref['core_id'])
            self.rows('core_comparison', ident, method, [{'core_id': ref['core_id'], 'R': ref['state'],
                'P': pred['state'] if pred else None,
                'effective': ref['core_id'] in effective if verdict and pred and ref['state'] != 'insufficient_material' else None,
                'quotes': pred['quotes'] if pred else [], 'conflicting_quotes': pred['conflicting_quotes'] if pred else []}])

    def support_metrics(self, ident: str, method: str, extract: dict[str, Any], support: dict[str, Any]) -> None:
        units = [u for u in extract['units'] if u['needs_verification'] or u['substantive']]
        index = {r['unit_id']: r for r in support['units']}
        done = [index[u['unit_id']] for u in units if u['unit_id'] in index]
        complete = len(done) == len(units)
        supported = [r for r in done if r['support'] == 'supported' and r['scope_correct'] is True]
        traceable = [r for r in supported if r['originally_substantiated']]
        for name, count in [('S', len(supported)), ('T', len(traceable)), ('S_minus_T', len(supported)-len(traceable))]:
            self.metric(ident, method, name, count if complete else None, len(units), len(units))
        self.metric(ident, method, 'support_technical_missing', len(units)-len(done))
        for state in ('supported', 'partly_supported', 'not_verifiable', 'contradicted'):
            self.metric(ident, method, 'support_'+state, sum(r['support'] == state for r in done) if complete else None, len(units))
        for error in ERRORS:
            count = sum(error in r['errors'] for r in done)
            self.metric(ident, method, 'error_prevalence_'+error, int(count > 0) if complete and units else None)
            self.metric(ident, method, 'errors_per100_'+error, 100*count if complete else None, len(units))
        prepared = self.get('prepare', ident)
        if prepared is None:
            return
        for name, predicate in [('Mentioned', lambda u: True), ('Substantive', lambda u: u['substantive']),
                ('Source_supported', lambda u: u['substantive'] and u['unit_id'] in {r['unit_id'] for r in supported})]:
            ids = {cid for u in extract['units'] if predicate(u) for cid in u['claim_ids']}
            actual = {c['claim_id'] for c in prepared['claims']}
            self.metric(ident, method, name, len(ids & actual) if name != 'Source_supported' or complete else None, len(actual))

    def concern_metrics(self, ident: str, method: str, value: dict[str, Any]) -> None:
        checklist = self.get('checklist', ident)
        if checklist is None:
            return
        expected = [c for c in checklist['concerns'] if c['applies_to_input'] == 'yes']
        index = {r['concern_id']: r for r in value['matches']}
        done = [index[c['concern_id']] for c in expected if c['concern_id'] in index]
        complete = len(done) == len(expected)
        self.metric(ident, method, 'issue_coverage', sum(r['scope'] == 'same' for r in done) if complete else None, len(expected))
        reason_ids = {c['concern_id'] for c in expected if c['reasons']}
        self.metric(ident, method, 'reason_coverage', sum(r['reason_coverage'] == 'complete' for r in done
                    if r['concern_id'] in reason_ids) if complete else None, len(reason_ids))
        self.metric(ident, method, 'concern_technical_missing', len(expected)-len(done))

    def information_metrics(self, ident: str, method: str) -> None:
        clusters, prepared = self.get('clusters', ident), self.get('prepare', ident)
        if clusters is None or prepared is None:
            return
        if method in clusters.get('missing_methods', []):
            return
        eligible = set(clusters.get('eligible_unit_keys', clusters['mapping']))
        selected = [c for c in clusters['clusters'] if any(k in eligible and clusters['mapping'].get(k, '').startswith(method+'/') for k in c['unit_keys'])]
        report = self.get('full' if method == 'fusion' else method, ident)
        self.metric(ident, method, 'information_count', len(selected))
        for kind in KINDS:
            self.metric(ident, method, 'information_'+kind, sum(c['kind'] == kind for c in selected))
        if report is not None:
            chars = sum(not c.isspace() for c in report['body'])
            self.metric(ident, method, 'information_density', 1000*len(selected), chars)
        if method == 'fusion' and not clusters.get('missing_methods'):
            for c in clusters['clusters']:
                members = sorted({clusters['mapping'].get(k, '').split('/')[0] for k in c['unit_keys'] if k in eligible})
                if members:
                    self.rows('clusters', ident, '', [{**c, 'methods': members, 'mapping': clusters['mapping']}])

    def collect_fusion(self, ident: str) -> None:
        value = self.get('fusion', ident)
        if value is None:
            return
        sets = value['sets']
        self.rows('fusion_sets', ident, 'fusion', [sets])
        self.rows('fusion_errors', ident, 'fusion', value['errors'])
        for name in ('union', 'full', 'retained', 'not_retained', 'added'):
            self.metric(ident, 'fusion', 'fusion_'+name, len(sets[name]))
        for name in ('newly_supported', 'mentioned_without_support'):
            self.metric(ident, 'fusion', 'fusion_'+name, len(sets.get(name, [])))
        self.metric(ident, 'fusion', 'important_retention', len(sets['important_retained']), len(sets['important_union']))
        for name in ('explicitly_corrected', 'not_propagated', 'propagated', 'unknown'):
            self.metric(ident, 'fusion', 'fusion_error_'+name, sum(r['status'] == name for r in value['errors']))
        self.metric(ident, 'fusion', 'fusion_new_errors', len(set(value['newly_introduced_error_keys'])))

    def collect_preferences(self, ident: str) -> None:
        for opponent in COMPARATORS:
            ab, ba = [self.get('preference', ident, opponent+'_'+order) for order in ('AB', 'BA')]
            if ab is None or ba is None:
                self.rows('preference_pairs', ident, opponent, [{'opponent': opponent, 'category': 'technical_failure'}])
                continue
            for dimension in PREF_DIMS:
                choices = [decode(r[dimension], r['mapping']) for r in (ab, ba)]
                values = [1.0 if v == 'fusion' else .5 if v == 'tie' else 0.0 for v in choices if v != 'cannot_judge']
                # h4 averages available judged orders within paper; no imputed ties.
                self.metric(ident, opponent, 'preference_'+dimension, sum(values), len(values))
                if dimension == 'overall':
                    category = paired_preference(ab, ba, dimension)
                    self.rows('preference_pairs', ident, opponent, [{'opponent': opponent, 'category': category,
                        'AB': choices[0], 'BA': choices[1], 'both_decidable': 'cannot_judge' not in choices}])
            self.rows('preference_orders', ident, opponent, [{**r, 'opponent': opponent} for r in (ab, ba)])

    def summaries(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.metrics:
            grouped[(row['metric'], row['method'])].append(row)
        summaries, paired = [], []
        for (metric, method), rows in grouped.items():
            valid = [r for r in rows if r['value'] is not None]
            summaries.append({'metric': metric, 'method': method,
                **self.bootstrap.estimate({r['paper_id']: r['value'] for r in rows}),
                'micro': ratio(sum(r['numerator'] for r in valid), sum(r['denominator'] for r in valid)),
                'numerator': sum(r['numerator'] for r in valid), 'denominator': sum(r['denominator'] for r in valid)})
        for metric in sorted({r['metric'] for r in self.metrics}):
            full = {r['paper_id']: r['value'] for r in grouped.get((metric, 'fusion'), []) if r['value'] is not None}
            for opponent in COMPARATORS:
                other = {r['paper_id']: r['value'] for r in grouped.get((metric, opponent), []) if r['value'] is not None}
                common = sorted(full.keys() & other.keys())
                paired.append({'metric': metric, 'method': opponent, 'primary': opponent == 'gear', 'paper_ids': common,
                               **self.bootstrap.estimate({p: full[p]-other[p] for p in common})})
        return summaries, paired

    def output(self) -> None:
        log(self.config, '统计开始', f'执行{self.config.bootstrap_repeats}次论文级bootstrap，保留实际分母与缺失。', stage='aggregate')
        summaries, paired = self.summaries()
        self.tables['paper_metrics'] = self.metrics
        self.tables['summary'] = summaries
        self.tables['paired_differences'] = paired
        for name in ('missing', 'rechecks', 'controls', 'recheck_coverage', 'control_coverage'):
            self.tables.setdefault(name, [])
        for name, rows in self.tables.items():
            write_csv(self.config.output/'derived'/f'{name}.csv', rows)
            path = self.config.output/'derived'/f'{name}.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        components = component_data(self, summaries, paired)
        for key, value in components.items():
            write(self.config.output/'derived/components'/f'{key}.json', value)
            write_csv(self.config.output/'derived/components'/f'{key}.csv', value['rows'])
        prepared = [v for (s, _, _), v in self.cache.items() if s == 'prepare' and v is not None]
        write(self.config.output/'derived/summary.json', {'papers': len(self.ids),
            'shared_claims': sum(len(v['claims']) for v in prepared), 'summaries': summaries,
            'paired_differences': paired, 'missing': self.tables['missing'], 'components': list(components),
            'recheck': {'observed': sum(r['observed'] for r in self.tables['recheck_coverage']),
                        'comparable': sum(r['comparable'] for r in self.tables['recheck_coverage']),
                        'uncomparable': sum(r['uncomparable'] for r in self.tables['recheck_coverage']),
                        'missing_objects': sum(len(r['missing_objects']) for r in self.tables['recheck_coverage'])},
            'bootstrap': {'unit': 'paper', 'repeats': self.config.bootstrap_repeats, 'seed': self.config.seed}})
        costs(self.config, set(self.ids))
        log(self.config, '阶段完成', f'统计表及40份组件数据已保存；缺项记录={len(self.tables["missing"])}。', stage='aggregate')


def costs(config: Config, paper_ids: set[str] | None = None) -> None:
    path = config.output/'logs/calls/calls.jsonl'
    rows, malformed = [], []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError('Call record is not an object')
                rows.append(value)
            except (ValueError, TypeError):
                malformed.append({'line': number, 'raw': line, 'usage': None})
    for record_path in sorted((config.output/'logs/calls').glob('*/record.json')):
        try:
            value = read(record_path)
            if not isinstance(value, dict):
                raise MissingInput('Call record is not an object')
            rows.append(value)
        except MissingInput as exc:
            malformed.append({'path': str(record_path), 'error': str(exc), 'usage': None})
    write(config.output/'derived/unreadable_calls.json', malformed)
    if paper_ids is not None:
        rows = [row for row in rows if row.get('paper_id') in paper_ids]
    papers = [p for p in roster(config) if paper_ids is None or p['paper_id'] in paper_ids]
    calls = [{k: v for k, v in r.items() if k != 'raw'} for r in rows]
    write_csv(config.output/'derived/model_calls.csv', calls)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get('stage', ''), row.get('method', ''))].append(row)
    summary = [{'stage': s, 'method': m, 'calls': len(group), 'failures': sum(r.get('state') != 'completed' for r in group),
                'seconds': sum(r.get('seconds') or 0 for r in group),
                'input_tokens': sum(r['usage']['input_tokens'] for r in group if r.get('usage')) if any(r.get('usage') for r in group) else None,
                'output_tokens': sum(r['usage']['output_tokens'] for r in group if r.get('usage')) if any(r.get('usage') for r in group) else None,
                'usage_complete': all(r.get('usage') is not None for r in group),
                'usage_missing': sum(r.get('usage') is None for r in group)} for (s, m), group in grouped.items()]
    write_csv(config.output/'derived/costs.csv', summary)
    network_counts: Counter[tuple[str, str]] = Counter()
    event_path = config.output/'logs/events.jsonl'
    if event_path.exists():
        with event_path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # An interrupted trailing log line is not a scientific result.
                if event.get('event') == '网络请求':
                    network_counts[(event.get('paper_id', ''), event.get('stage', ''))] += 1
    retrieval = []
    for paper in papers:
        for method in ('eacl', 'reviewgrounder'):
            path = config.output/'evidence/retrieval'/paper['paper_id']/method/'result.json'
            if path.exists():
                value = read(path)
                retrieval.append({'paper_id': paper['paper_id'], 'method': method,
                    'network_calls': network_counts.get((paper['paper_id'], method), value['network_calls']),
                    'network_count_scope': '累计实际请求，含失败/续跑/显式重跑；旧日志缺失时采用已有检索计数', 'fulltexts': value['fulltext_count'],
                    'failures': len(value['failures']), 'queries': len(value['queries']), 'candidates': value.get('candidate_count', len(value.get('candidates', [])))})
    write_csv(config.output/'derived/retrieval_costs.csv', retrieval)


def aggregate(config: Config, paper_ids: set[str] | None = None) -> None:
    aggregator = Aggregator(config, paper_ids)
    aggregator.collect()
    aggregator.output()


def component_data(a: Aggregator, summaries: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, Any]:
    result = {key: {'id': key, 'panel': key[0], 'title': title, 'kind': kind, 'rows': [],
                   'missing_tasks': len(a.tables['missing'])} for key, (title, kind) in COMPONENTS.items()}

    def put(key: str, rows: list[dict[str, Any]], **extra: Any) -> None:
        result[key].update(rows=rows, **extra)

    def metrics(key: str, names: list[str], differences: bool = False) -> None:
        source = paired if differences else summaries
        put(key, [r for r in source if r['metric'] in names], axis='Full minus comparator' if differences else 'macro mean')

    def scatter(key: str, x: str, y: str, source: list[dict[str, Any]] = summaries) -> None:
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in source:
            if row['metric'] in {x, y}:
                grouped[row['method']][row['metric']] = row
        put(key, [{'method': method, 'x': rows.get(x, {}).get('estimate'), 'y': rows.get(y, {}).get('estimate'),
                   'x_low': rows.get(x, {}).get('low'), 'x_high': rows.get(x, {}).get('high'),
                   'y_low': rows.get(y, {}).get('low'), 'y_high': rows.get(y, {}).get('high'),
                   'n_x': rows.get(x, {}).get('n', 0), 'n_y': rows.get(y, {}).get('n', 0)}
                  for method, rows in grouped.items()], x_label=x, y_label=y)

    def categories(key: str, rows: list[dict[str, Any]], field: str, matrix_y: str | None = None,
                   levels: list[str] | None = None) -> None:
        # Proportions within each paper then macro means. Row counts are also retained.
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        labels: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            method = row.get('method') or 'all'
            category = str(row.get(field, 'unknown'))
            if matrix_y:
                category += ' × '+str(row.get(matrix_y, 'unknown'))
            labels[method].add(category)
            grouped[(method, row['paper_id'])].append({**row, '_category': category})
        values = []
        all_categories = set(levels or []) | {category for names in labels.values() for category in names}
        for method in labels:
            papers = {p for m, p in grouped if m == method}
            for label in sorted(all_categories):
                nums = {p: sum(r['_category'] == label for r in grouped[(method, p)]) for p in papers}
                dens = {p: len(grouped[(method, p)]) for p in papers}
                item = {'method': method, 'category': label, 'count': sum(nums.values()), 'denominator': sum(dens.values()),
                        'micro': ratio(sum(nums.values()), sum(dens.values())),
                        **a.bootstrap.estimate({p: nums[p]/dens[p] for p in papers})}
                if matrix_y:
                    item['x'], item['y'] = label.split(' × ', 1)
                values.append(item)
        put(key, values, denominator='within-paper available objects; papers without eligible objects are NA')

    prepared = [v for (s, _, _), v in a.cache.items() if s == 'prepare' and v is not None]
    fields = Counter(p.get('field_name', 'unknown') for p in a.roster)
    put('a1', [{'label': 'Roster papers', 'value': len(a.ids)}, {'label': 'Prepared papers', 'value': len(prepared)},
               {'label': 'Shared claims', 'value': sum(len(p['claims']) for p in prepared)},
               {'label': 'Selected core contributions', 'value': len(a.tables['core'])}]+
              [{'label': field, 'value': count} for field, count in sorted(fields.items())])
    descriptions = {'direct_a': 'Fresh manuscript-only fragment reading and synthesis', 'gear': 'Fresh shared claims and historical comparisons',
        'graph': 'Fresh single-claim + union joint Graph', 'fusion': 'Fresh GEAR + Graph claim fusion and joint synthesis',
        'eacl': 'Independent extraction → dual retrieval/ranking → historical comparison → writer',
        'reviewgrounder': 'Draft → literature/results/insights → rubric synthesis → writer'}
    put('a2', [{'method': method, 'input': descriptions[method]} for method in METHODS])
    put('a3', [{'channel': name, 'definition': definition} for name, definition in [
        ('R/P/V', 'Separate historical reference, report prediction and evidence judgment'),
        ('Quality', 'Five independent 0–3 dimensions; no novelty total'), ('Human issues', 'Existing source-bound reviewer records'),
        ('Support', 'Atomic source support and original traceability'), ('Information', 'Within-paper semantic clusters'),
        ('Preference', 'Independent AB and BA; technical failures separate')]])
    put('a4', [{'material': 'Generation', 'definition': 'Own manuscript, references, independent retrieval; no evaluation cores or human judgments'},
               {'material': 'Historical R', 'definition': 'Manuscript + prior dated sources; no method judgments or reviewer attitude'},
               {'material': 'Evaluation', 'definition': 'Common material pool and original reports; limitations remain explicit'},
               {'material': 'h5', 'definition': 'Same-model repeat and matched controls; no automatic main-result edits'}])
    categories('b1', a.tables['reference'], 'state')
    scatter('b2', 'recall', 'precision')
    categories('b3', [r for r in a.tables['core_comparison'] if r['P'] is not None], 'R', 'P')
    metrics('b4', ['recall', 'precision', 'historical_comparison', 'scope_coverage'], True)
    metrics('b5', ['historical_comparison', 'false_antecedence', 'false_firstness', 'material_overclaim'])
    metrics('c1', ['quality_'+d for d in DIMENSIONS])
    scatter('c2', 'quality_contribution_fidelity', 'quality_historical_increment')
    metrics('c3', ['quality_'+d for d in DIMENSIONS], True)
    metrics('c4', ['Mentioned', 'Substantive', 'Source_supported'])
    categories('c5', a.tables['assertions'], 'kind')
    scatter('d1', 'S', 'T')
    categories('d2', a.tables['support'], 'support')
    metrics('d3', ['error_prevalence_'+e for e in ERRORS])
    metrics('d4', ['S_minus_T'])
    metrics('e1', ['issue_coverage', 'reason_coverage'])
    applicable = {(r['paper_id'], r['concern_id']) for r in a.tables['checklist'] if r['applies_to_input'] == 'yes'}
    categories('e2', [r for r in a.tables['concern_matches'] if (r['paper_id'], r['concern_id']) in applicable], 'disagreement')
    categories('e3', a.tables['checklist'], 'dimension')
    categories('e4', a.tables['tone'], 'tone', 'scientific_stance')
    changes = [r for r in a.tables['review_dynamics'] if r['later_round'] is not None
               and r['later_round'] > r['earlier_round'] > 0 and r['same_reviewer_explicit']]
    categories('e5', changes, 'earlier_stance', 'later_stance')
    categories('e6', a.tables['review_dynamics'], 'resolution')
    scatter('f1', 'information_historical_increment', 'information_cross_work_relation')
    scatter('f2', 'information_count', 'S')
    categories('f3', [{**r, 'intersection': '+'.join(r['methods'])} for r in a.tables['clusters']], 'intersection')
    metrics('f4', ['information_'+k for k in KINDS])
    metrics('f6', ['information_density'])
    metrics('g1', ['fusion_retained', 'fusion_not_retained', 'fusion_added', 'fusion_newly_supported', 'fusion_mentioned_without_support'])
    metrics('g2', ['fusion_union', 'fusion_full'])
    metrics('g3', ['fusion_error_'+s for s in ('explicitly_corrected', 'not_propagated', 'propagated', 'unknown')]+['fusion_new_errors'])
    metrics('g4', ['important_retention'])
    h1 = []
    for row in a.tables['preference_pairs']:
        if row['category'] == 'technical_failure':
            continue
        category = ('stable_full_win' if row['category'] == 'fusion' else 'stable_tie' if row['category'] == 'tie'
                    else 'stable_opponent_win' if row['category'] == row['opponent'] else row['category'])
        h1.append({**row, 'category': category})
    categories('h1', h1, 'category', levels=['stable_full_win', 'stable_tie', 'stable_opponent_win',
                                          'order_sensitive', 'cannot_judge'])
    result['h1']['technical_failures'] = [r for r in a.tables['preference_pairs'] if r['category'] == 'technical_failure']
    decidable = [r for r in a.tables['preference_pairs'] if r.get('both_decidable')]
    orders = [{**r, 'choice': 'Full' if r[order] == 'fusion' else 'Tie' if r[order] == 'tie' else 'Opponent'}
              for r in decidable for order in ('AB', 'BA')]
    categories('h2', orders, 'choice')
    swap = [{**r, **{order: 'Full' if r[order] == 'fusion' else 'Opponent' if r[order] == r['opponent']
                     else r[order] for order in ('AB', 'BA')}} for r in a.tables['preference_pairs'] if 'AB' in r]
    categories('h3', swap, 'AB', 'BA')
    metrics('h4', ['preference_'+d for d in PREF_DIMS])
    repeat = [{**r, 'method': r['task'], 'category': 'different' if r['different'] else 'same'} for r in a.tables['rechecks'] if r.get('different') is not None]
    for row in a.tables['controls']:
        for side, verdict in row['judgments'].items():
            repeat.append({'paper_id': row['paper_id'], 'method': row['kind']+'/'+side,
                           'category': verdict['support']})
    categories('h5', repeat, 'category')
    from .recheck import COMPARE
    result['h5']['field_agreement'] = [
        {'task': task, 'field': field, 'n': len(rows),
         'same': sum(r['first'][field] == r['second'][field] for r in rows)}
        for task, fields in COMPARE.items()
        for rows in [[r for r in a.tables['rechecks'] if r['task'] == task and r.get('different') is not None]]
        for field in fields]
    result['h5']['recheck_coverage'] = a.tables['recheck_coverage']
    result['h5']['control_coverage'] = a.tables['control_coverage']
    result['h5']['uncomparable_rechecks'] = [r for r in a.tables['rechecks'] if r.get('different') is None]
    # Case eligibility uses available quoted evidence, never gain or method ranking.
    quote_rows = sorted([r for r in a.tables['core_comparison'] if r['method'] == 'fusion' and r['quotes']],
                        key=lambda r: (r['paper_id'], r['core_id']))
    used = quote_rows[0]['paper_id'] if quote_rows else None
    if used:
        put('c6', [r for r in a.tables['core_comparison'] if r['paper_id'] == used and r['core_id'] == quote_rows[0]['core_id']],
            selection='first paper ID with a quoted Full core judgment; no Full-gain selection')
    chains = sorted([r for r in a.tables['support'] if r['paper_id'] != used and r['support'] == 'supported'
                     and r['scope_correct'] and r['evidence']], key=lambda r: (r['paper_id'], r['method'], r['unit_id']))
    if chains:
        first = chains[0]
        originals = [r for r in a.tables['assertions'] if (r['paper_id'], r['method'], r['unit_id']) ==
                     (first['paper_id'], first['method'], first['unit_id'])]
        put('f5', [{**first, 'report_quote': originals[0]['quote'] if originals else ''}],
            selection='first distinct paper ID with supported scope-correct assertion and quoted evidence')
    return result
