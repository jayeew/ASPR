"""Summarize source-bound supplementary judgments without replacing originals."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv

from .abcd_followup import DEST, PREP
from .cd_sources import OUTPUT, read, write_json, write_rows
from .run_cd_first import rows

RUN = DEST / 'evaluation'
METHODS = ('gear', 'graph', 'fusion')


def target_results() -> list[dict]:
    packets = {p['target_id']: p for p in rows(RUN / 'final_packets.jsonl')}
    output = []
    for target in rows(PREP / 'targets.jsonl'):
        packet = packets.get(target['id'])
        path = RUN / 'results' / (packet['id'] + '.json') if packet else None
        record = read(path) if path and path.exists() else {}
        accepted = record.get('status') == 'completed'
        answer = record.get('answer') if accepted else None
        result = {**target, 'evaluation_status': record.get('status', 'deferred'),
                  'source': 'approved_source_bound_supplementary_evaluation',
                  'answer': answer, 'reason_coverage': 'uncertain',
                  'relation_treatment': 'uncertain', 'report_quotes': []}
        if answer:
            result.update({k: answer[k] for k in ('object_scope', 'reason_coverage',
                           'relation_treatment', 'report_quotes', 'reason')})
            payload = packet['payload']
            if target['kind'] == 'review_reason' and result['reason_coverage'] == 'complete' and result['object_scope'] != 'same':
                result['reason_coverage'] = 'uncertain'
                result['supplementary_limitation'] = 'Complete reason coverage conflicts with incomplete object/scope alignment.'
            if target['kind'] == 'relation_trace' and result['relation_treatment'] in {'same_scope', 'merged_same_scope'} and result['object_scope'] != 'same':
                result['relation_treatment'] = 'uncertain'
                result['supplementary_limitation'] = 'Same-scope treatment conflicts with incomplete object/scope alignment.'
            if target['kind'] == 'relation_trace' and result['relation_treatment'] == 'not_applicable':
                result['relation_treatment'] = 'uncertain'
                result['supplementary_limitation'] = 'No applicable relation treatment was established.'
            unresolved = not payload['screening_complete'] or payload['uncertain_chunk_count'] > 0
            if unresolved:
                if result['reason_coverage'] == 'none':
                    result['reason_coverage'] = 'uncertain'
                    result['supplementary_limitation'] = 'Absence not established across uncertain report chunks.'
                if result['relation_treatment'] == 'not_located':
                    result['relation_treatment'] = 'uncertain'
                    result['supplementary_limitation'] = 'Absence not established across uncertain report chunks.'
        output.append(result)
    return output


def combined_coverage(values: list[str]) -> str:
    """Require all explicit matched reasons for complete contribution coverage."""
    if 'uncertain' in values:
        return 'uncertain'
    applicable = [v for v in values if v != 'not_applicable']
    if not applicable:
        return 'not_applicable'
    if all(v == 'complete' for v in applicable):
        return 'complete'
    if all(v == 'none' for v in applicable):
        return 'none'
    return 'partial'


def coverage_summary(records: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[r['paper_id'], r['core_id'], r['method']].append(r)
    contribution_rows = []
    for (paper, core, method), values in sorted(groups.items()):
        contribution_rows.append({'id': f'{paper}/{core}/{method}', 'paper_id': paper,
            'core_id': core, 'method': method, 'state': combined_coverage([v['reason_coverage'] for v in values]),
            'concern_ids': sorted({v['concern_id'] for v in values}), 'record_count': len(values)})
    populations = {m: {(r['paper_id'], r['core_id']) for r in contribution_rows if r['method'] == m} for m in METHODS}
    if not populations['gear'] == populations['graph'] == populations['fusion']:
        raise ValueError('Contribution coverage populations differ between methods')
    write_rows(RUN / 'contribution_reason_coverage.jsonl', contribution_rows)
    return {'contribution_states': {m: dict(Counter(r['state'] for r in contribution_rows if r['method'] == m)) for m in METHODS},
            'matched_contribution_denominator': len(populations['gear']),
            'paper_denominator': len({p for p, _ in populations['gear']}),
            'reason_record_states': {m: dict(Counter(r['reason_coverage'] for r in records if r['method'] == m)) for m in METHODS},
            'aggregation': 'Complete requires all applicable explicitly matched reviewer reasons; mixed coverage is partial; any unresolved assessment remains uncertain.',
            'limitations': ['Automatic matching and coverage judgments, not new human annotations.', 'Reviewer silence is not a negative label.']}


def trace_summary(records: list[dict]) -> dict:
    verdicts = {r['id']: r for r in rows(OUTPUT / 'd_verification/verification_results.jsonl')}
    supported = {k for k, r in verdicts.items() if r['verified_state'] in {'supported', 'supported_after_narrowing'}}
    primary = [r for r in records if r['relation_id'] in supported]
    for method in ('gear', 'fusion'):
        if len([r for r in primary if r['method'] == method]) != len(supported):
            raise ValueError('Relation trace populations differ')
    return {'supported_relation_treatment': {m: dict(Counter(r['relation_treatment'] for r in primary if r['method'] == m)) for m in ('gear', 'fusion')},
            'supported_relation_denominator': len(supported),
            'all_assessed_relation_denominator': len(verdicts),
            'insufficient_evidence_treatment': {m: dict(Counter(r['relation_treatment'] for r in records if r['method'] == m and r['relation_id'] not in supported)) for m in ('gear', 'fusion')},
            'case_treatments': {r['relation_id']+'/'+r['method']: r['relation_treatment'] for r in records},
            'limitations': ['Candidate relations are not necessarily novel scientific discoveries.', 'Treatment reasons are evaluator descriptions, not inferred system motives.', 'Not located requires complete report screening without uncertain chunks.']}


def main() -> None:
    if not (RUN / 'completion.json').exists():
        raise RuntimeError('Approved evaluation is still running')
    results = target_results()
    c = rows(PREP / 'c_reused_reason_coverage.jsonl') + [r for r in results if r['kind'] == 'review_reason']
    d = [r for r in results if r['kind'] == 'relation_trace']
    if len(c) != 162 or len(d) != 104:
        raise ValueError('Approved analysis populations differ')
    write_rows(RUN / 'reason_coverage_records.jsonl', c)
    write_rows(RUN / 'report_trace_records.jsonl', d)
    coverage = coverage_summary(c)
    trace = trace_summary(d)
    write_json(RUN / 'coverage_statistics.json', coverage)
    write_json(RUN / 'trace_statistics.json', trace)
    calls = [read(p) for p in (RUN / 'logs/calls').glob('*/record.json')]
    if len(calls) > 1082 or any(r['model'] != 'gpt-5.6-luna' or r['effort'] != 'xhigh' for r in calls):
        raise ValueError('Calls exceed approved model or budget')
    usage = {'actual_calls': len(calls), 'format_repair_calls': sum(r.get('repair', 0) > 0 for r in calls)}
    write_json(RUN / 'summary.json', {'execution': read(RUN / 'completion.json'),
        'C': {'existing_records_reused': 36, 'supplementary_targets': 126},
        'D': {'supplementary_targets': 104},
        'target_statuses': dict(Counter(r['evaluation_status'] for r in results)), 'calls': usage})
    for name, data, field in [('C_reason_coverage', coverage['contribution_states'], 'coverage'),
                              ('D_supported_relation_treatment', trace['supported_relation_treatment'], 'treatment')]:
        path = DEST / (name + '.csv')
        if not path.exists():
            with path.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['method', field, 'count', 'denominator'])
                writer.writeheader()
                for method, counts in data.items():
                    for state, count in counts.items():
                        writer.writerow({'method': method, field: state, 'count': count, 'denominator': sum(counts.values())})
    flow_path = DEST / 'C_flow_counts.csv'
    if not flow_path.exists():
        joint = rows(OUTPUT / 'first_stage/c_method_review_joint.jsonl')
        with flow_path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['method', 'transition', 'from_state', 'to_state', 'count', 'denominator'])
            writer.writeheader()
            for method in METHODS:
                values = [r for r in joint if r['method'] == method]
                for transition, pairs in [('history_to_system', [(r['reference_state'], r['prediction_state']) for r in values]),
                                           ('system_to_review', [(r['prediction_state'], r['review_summary']['state']) for r in values])]:
                    for (left, right), count in sorted(Counter(pairs).items()):
                        writer.writerow({'method': method, 'transition': transition, 'from_state': left,
                                         'to_state': right, 'count': count, 'denominator': len(values)})
    print('Prepared contribution coverage and relation trace statistics.')


if __name__ == '__main__':
    main()
