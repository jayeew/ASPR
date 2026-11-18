"""Summarize approved first-stage results and prepare unapproved follow-up data."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .cd_contracts import PROMPTS, SCHEMAS, reviewer_summary
from .cd_sources import OUTPUT, write_json, write_rows
from .run_cd_first import EXECUTION, result_path, selected_packets


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def core_reviews(records: dict, packets: list[dict]) -> tuple[list[dict], list[dict]]:
    reviews = {r['id']: r for r in rows(OUTPUT/'tables/c_review_records.jsonl')}
    links = []
    for packet in packets:
        if packet['task'] != 'c_match':
            continue
        record = records[packet['id']]
        if record['answer'] is None:
            continue
        issues = record.get('integrity_issues', [])
        if 'core_concern_identity_or_completeness' in issues:
            continue
        paper = packet['payload']['paper_id']
        for link in record['answer']['links']:
            local_issues = [i for i in issues if i.startswith(link['core_id']+'/')]
            if any('/reason_quotes/' not in i for i in local_issues):
                continue
            review = reviews[f'{paper}/{link["concern_id"]}']
            strict = (review['applicability_for_strict_analysis'] and review['source_role_ready'] and
                      all(r['identity_explicit'] for r in review['source_origins']))
            links.append({'id': f'{paper}/{link["core_id"]}/{link["concern_id"]}',
                          'paper_id': paper, **link, 'source_ready': strict,
                          'existing_stance': review['stance'], 'original_record_id': review['id'],
                          'evaluation_id': record['id'], 'status': 'completed',
                          'reason_quote_status': 'pending' if local_issues else 'located_or_empty',
                          'reason_quote_issues': local_issues})
    by_core: dict[str, list[dict]] = defaultdict(list)
    for link in links:
        by_core[f'{link["paper_id"]}/{link["core_id"]}'].append(link)
    original = rows(OUTPUT/'tables/c_review_links.jsonl')
    expected = Counter(f'{r["paper_id"]}/{r["core_id"]}' for r in original)
    gaps = {r['paper_id'] for r in rows(OUTPUT/'tables/source_gaps.jsonl')}
    paper_gaps = {r['paper_id'] for r in rows(OUTPUT/'tables/papers.jsonl') if r['unresolved_source_sections']}
    summaries = []
    for core in rows(OUTPUT/'tables/c_cores.jsonl'):
        group = by_core[core['id']]
        complete = len(group) == expected[core['id']]
        gap = core['paper_id'] in gaps | paper_gaps
        summary = reviewer_summary(group, complete, gap)
        observed = reviewer_summary(group, True, gap)
        summaries.append({'id': core['id'], 'paper_id': core['paper_id'], 'core_id': core['core_id'],
            'reference_state': core['reference']['state'], 'expected_pairs': expected[core['id']],
            'evaluated_pairs': len(group), 'complete': complete, 'summary': summary,
            'observed_summary': observed, 'has_source_gap': gap,
            'matched_concern_ids': [r['concern_id'] for r in group if r['match'] == 'same'],
            'reviewer_silence_is_not_negative': True})
    return links, summaries


def method_review_join(core_summaries: list[dict]) -> tuple[list[dict], dict]:
    summaries = {r['id']: r for r in core_summaries}
    evidence = {r['id']: r for r in rows(OUTPUT/'tables/evidence_index.jsonl')}
    joint = []
    for method in rows(OUTPUT/'tables/c_methods.jsonl'):
        core = summaries[f'{method["paper_id"]}/{method["core_id"]}']
        quotes_ready = all(evidence[e]['location']['state'] in {'exact','normalized_whitespace_unicode'}
                           for e in method['report_evidence_ids'])
        if not method['report_evidence_ids'] and method['prediction']['state'] != 'not_addressed':
            quotes_ready = False
        joint.append({'id': method['id'], 'paper_id': method['paper_id'], 'core_id': method['core_id'],
            'method': method['method'], 'reference_state': core['reference_state'],
            'prediction_state': method['prediction']['state'], 'effective_increment': method['existing_effective_increment'],
            'report_quote_ready': quotes_ready, 'review_summary': core['summary'],
            'review_alignment_complete': core['complete'], 'review_alignment_source': 'source_bound_luna_matching',
            'new_human_annotation': False})
    recognized = {r['id'] for r in core_summaries if r['complete'] and r['summary']['state']=='recognized'
                  and not r['summary']['has_unresolved'] and r['reference_state'] in {'supported_difference','bounded_increment'}}
    by_core: dict[str,list[dict]] = defaultdict(list)
    for row in joint:
        by_core[f'{row["paper_id"]}/{row["core_id"]}'].append(row)
    comparable = {cid for cid in recognized if len(by_core[cid])==3 and all(r['report_quote_ready'] for r in by_core[cid])}
    stats = {'recognized_reference_positive_cores': len(recognized),
             'common_quote_ready_cores': len(comparable), 'excluded_quote_gap_cores': len(recognized-comparable),
             'methods': {}}
    for method in ('gear','graph','fusion'):
        selected = [r for cid in comparable for r in by_core[cid] if r['method']==method]
        numerator = sum(r['prediction_state'] in {'positive_increment','limited_increment'} and
                        r['effective_increment'] is True for r in selected)
        stats['methods'][method] = {'numerator': numerator, 'denominator': len(selected),
                                   'fraction': numerator/len(selected) if selected else None}
    return joint, stats


def relation_records(records: dict, packets: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates = {r['id']: r for r in rows(OUTPUT/'tables/d_candidates.jsonl')}
    evidence = {r['id']: r for r in rows(OUTPUT/'tables/evidence_index.jsonl')}
    classifications, relations = [], []
    for packet in packets:
        if packet['task'] != 'd_triage':
            continue
        record = records[packet['id']]
        payload = packet['payload']
        cid = f'{payload["paper_id"]}/{payload["cluster_id"]}'
        parts = record['answer']['parts'] if record['answer'] is not None else []
        categories = sorted({p['category'] for p in parts})
        classifications.append({'id': cid, 'paper_id': payload['paper_id'],
            'cluster_id': payload['cluster_id'], 'evaluation_id': record['id'],
            'status': record['status'], 'categories': categories if record['status']=='completed' else [],
            'mixed': len(categories)>1 if record['status']=='completed' else None,
            'parts': parts, 'limitations': record.get('answer', {}).get('limitations', []) if record['answer'] else []})
        if record['status'] != 'completed':
            continue
        for i, part in enumerate(parts, 1):
            if part['category'] != 'scientific_relation_candidate':
                continue
            ident = f'{cid}/R{i:03d}'
            refs = {eid for unit in candidates[cid]['units'] if unit['method']=='graph'
                    and unit['unit']['unit_id'] in part['unit_ids'] for eid in unit['source_evidence_ids']}
            proof, excluded = [], []
            for eid in sorted(refs):
                e = evidence[eid]
                meta = e['metadata']
                types = {m.get('source_type') for m in meta.get('owners', [])} or {meta.get('source_type')}
                if e['namespace']=='manuscript':
                    types = {'manuscript'}
                located = e['location']['state'] in {'exact','normalized_whitespace_unicode'}
                native_types = types & {'manuscript', 'abstract', 'fulltext'}
                forbidden = not native_types
                if not located or forbidden:
                    excluded.append({'evidence_id': eid, 'reason': 'unlocated_or_not_independent_original_text',
                                     'source_types': sorted(str(t) for t in types)})
                    continue
                original_meta = {k:v for k,v in meta.items() if k != 'evaluated_source_type'}
                if 'owners' in original_meta:
                    original_meta['owners'] = [o for o in original_meta['owners'] if o.get('source_type') in native_types]
                proof.append({'evidence_id': eid, 'source_types': sorted(native_types),
                              'quote': e['location']['source_quote'],
                              'source_metadata': original_meta, 'source_file': e['source_file']})
            payload_verify = {'relation_id': ident, 'relation_statement': part['relation_statement'],
                'subject_description': part['subject_description'], 'object_description': part['object_description'],
                'direction': part['direction'], 'scope': part['scope'], 'original_evidence': proof,
                'evidence_limitations': ['Only existing source-bound excerpts; no additional retrieval.',
                                         'Excluded graph observations, reviewer judgments, metadata-only and unlocated quotes.']}
            serialized = json.dumps(payload_verify, ensure_ascii=False)
            relations.append({'id': ident, 'paper_id': payload['paper_id'], 'cluster_id': payload['cluster_id'],
                'triage_part': part, 'status': 'pending_original_evidence_verification',
                'original_evidence': proof, 'excluded_evidence': excluded,
                'verification_payload': payload_verify,
                'payload_within_material_limit': len(serialized)<=12000,
                'requires_offline_packing': len(serialized)>12000,
                'verification_authorized': False})
    return classifications, relations


def call_usage() -> dict:
    states, tokens, efforts = Counter(), Counter(), Counter()
    repair_calls = 0
    for path in (EXECUTION/'logs/calls').glob('*/record.json'):
        record = json.loads(path.read_text())
        states[record['state']]+=1
        efforts[record['effort']]+=1
        repair_calls+=int(record.get('repair',0)>0)
        for key, value in (record.get('usage') or {}).items():
            if isinstance(value, (float,int)):
                tokens[key]+=value
    return {'actual_cli_calls': sum(states.values()), 'format_repair_calls': repair_calls,
            'states': dict(states), 'efforts': dict(efforts), 'usage': dict(tokens)}


def main() -> None:
    packets = selected_packets()
    if any(not result_path(p).exists() for p in packets):
        raise ValueError('Wait for all approved tasks before publishing summary')
    records = {p['id']: json.loads(result_path(p).read_text()) for p in packets}
    links, core_summaries = core_reviews(records, packets)
    classes, relations = relation_records(records, packets)
    joint, recognized_stats = method_review_join(core_summaries)
    first = OUTPUT/'first_stage'
    write_rows(first/'c_review_links.jsonl', links)
    write_rows(first/'c_core_summary.jsonl', core_summaries)
    write_rows(first/'c_method_review_joint.jsonl', joint)
    write_rows(first/'d_classification.jsonl', classes)
    write_rows(first/'d_relation_candidates.jsonl', relations)
    counts = Counter(r['status'] for r in records.values())
    summary = {'approved_requests': 681, 'request_statuses': dict(counts),
        'C': {'valid_link_records': len(links), 'match_states': dict(Counter(r['match'] for r in links)),
              'core_summary_states': dict(Counter(r['summary']['state'] for r in core_summaries)),
              'observed_core_states': dict(Counter(r['observed_summary']['state'] for r in core_summaries)),
              'complete_cores': sum(r['complete'] for r in core_summaries),
              'supported_recognized_identification': recognized_stats},
        'D': {'completed_clusters': sum(r['status']=='completed' for r in classes),
              'clusters_by_category': dict(Counter(c for r in classes if r['status']=='completed' for c in r['categories'])),
              'mixed_clusters': sum(r['mixed'] is True for r in classes),
              'scientific_relation_candidates': len(relations),
              'candidate_papers': len({r['paper_id'] for r in relations}),
              'verification_pending': True, 'verified_relation_count': None,
              'relations_with_located_original_excerpt': sum(bool(r['original_evidence']) for r in relations),
              'relations_needing_offline_packing': sum(r['requires_offline_packing'] for r in relations)},
        'call_usage': call_usage(), 'limitations': ['Scientific relation candidates are not verified discoveries.',
            'Missing, failed or source-unresolved review matches do not imply reviewer rejection.',
            'Downstream verification and report trace not authorized by this first-stage approval.']}
    write_json(first/'summary.json', summary)
    next_work = {'authorized': False, 'relation_verification_objects': len(relations),
                 'not_ready_for_execution': True,
                 'reason': 'Confirm scoped original evidence, finish offline packing, then assess calls with the user.',
                 'review_reason_core_concern_pairs': sum(r['match']=='same' and r['source_ready'] and
                    r['applies_to_input']=='yes' for r in links),
                 'method_reason_object_multiplier': 3,
                 'failed_or_integrity_tasks': [r['id'] for r in records.values() if r['status']!='completed'],
                 'source_recovery_objects': 15}
    write_json(first/'next_stage_assessment.json', next_work)
    for task in ('d_verify','c_reason','d_trace'):
        write_json(first/'schemas'/f'{task}.json', SCHEMAS[task].model_json_schema())
    csv_path = first/'descriptive_statistics.csv'
    if not csv_path.exists():
        with csv_path.open('x') as stream:
            writer = csv.writer(stream)
            writer.writerow(['group','category','count','unit'])
            for label, values, unit in [('C_match',summary['C']['match_states'],'core_concern_pair'),
                                        ('C_core',summary['C']['core_summary_states'],'core'),
                                        ('D_category',summary['D']['clusters_by_category'],'cluster_nonexclusive')]:
                for key,value in values.items():writer.writerow([label,key,value,unit])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
