"""Retrieve current IDs separately from recovered snapshot labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .stage0 import DEFAULT_OUT, FIELDS, ROOT, Client, write_csv, write_json

SELECT = ('id,doi,publication_date,publication_year,language,type,is_retracted,'
          'is_paratext,primary_topic,primary_location,updated_date')


def project(raw: dict[str, Any]) -> dict[str, Any]:
    topic = raw.get('primary_topic') or {}
    source = (raw.get('primary_location') or {}).get('source') or {}
    return {'work_id': raw['id'].rsplit('/', 1)[-1], 'doi': raw.get('doi'),
            'publication_date': raw.get('publication_date'), 'publication_year': raw.get('publication_year'),
            'language': raw.get('language'), 'work_type': raw.get('type'),
            'is_retracted': raw.get('is_retracted'), 'is_paratext': raw.get('is_paratext'),
            'source_id': source.get('id'), 'source_name': source.get('display_name'),
            'source_type': source.get('type'), 'field_id': (topic.get('field') or {}).get('id'),
            'field_name': (topic.get('field') or {}).get('display_name'),
            'subfield_id': (topic.get('subfield') or {}).get('id'),
            'primary_topic_id': topic.get('id'), 'primary_topic_name': topic.get('display_name'),
            'updated_date': raw.get('updated_date'), 'metadata_version': 'current_api_20261021'}


def recover_by_doi(client: Client, old: list[dict[str, Any]], current: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    missing = [r for r in old if r['work_id'] not in current]
    if not missing:
        return []
    dois = '|'.join(r['doi'].removeprefix('https://doi.org/') for r in missing)
    response = client.get('works', {'filter': 'doi:' + dois, 'per_page': 100, 'select': SELECT})
    by_doi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response['results']:
        by_doi[str(row.get('doi') or '').removeprefix('https://doi.org/').lower()].append(row)
    resolutions = []
    for row in missing:
        matches = by_doi.get(row['doi'].removeprefix('https://doi.org/').lower(), [])
        if len(matches) == 1:
            mapped = project(matches[0])
            mapped['current_work_id'] = mapped['work_id']
            mapped['work_id'] = row['work_id']
            mapped['identity_resolution'] = 'unique_exact_doi'
            current[row['work_id']] = mapped
        resolutions.append({'original_work_id': row['work_id'], 'doi': row['doi'],
                            'current_matches': [r['id'] for r in matches], 'resolved': len(matches) == 1})
    return resolutions


def main() -> None:
    client = Client(DEFAULT_OUT)
    old = pq.read_table(DEFAULT_OUT / 'canonical_target_works_ids_restored.parquet').to_pylist()
    ids = sorted(row['work_id'] for row in old)
    raw_rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(client.get, 'works', {'filter': 'openalex:' + '|'.join(ids[start:start + 100]),
                   'per_page': 100, 'select': SELECT}) for start in range(0, len(ids), 100)]
        for i, future in enumerate(as_completed(futures), 1):
            raw_rows.extend(future.result()['results'])
            if i % 25 == 0 or i == len(futures):
                print(f'Current ID recovery {i}/{len(futures)}', flush=True)
    current = {row['id'].rsplit('/', 1)[-1]: project(row) for row in raw_rows}
    returned_work_ids = len(current)
    for row in current.values():
        row['current_work_id'] = row['work_id']
        row['identity_resolution'] = 'exact_work_id'
    doi_resolutions = recover_by_doi(client, old, current)
    claim_ids = set(pq.read_table(ROOT / 'data/claim_graph/claim_nodes.parquet',
                                 columns=['parent_paper_id'])['parent_paper_id'].to_pylist())
    differences, rows, missing = [], [], []
    for row in old:
        live = current.get(row['work_id'])
        if live is None:
            missing.append(row['work_id']); continue
        live['nature_article_id'] = row['nature_article_id']
        live['has_legacy_claim'] = row['nature_article_id'] in claim_ids
        live['current_b_candidate'] = (str(live['field_id']).rsplit('/', 1)[-1] in FIELDS.values()
            and live['source_type'] == 'journal' and live['work_type'] in ['article', 'letter']
            and live['publication_year'] in [2023, 2024, 2025] and live['language'] == 'en'
            and live['is_retracted'] is False and live['is_paratext'] is False)
        for key in ['source_id', 'field_id', 'primary_topic_id', 'publication_year', 'publication_date', 'work_type']:
            if row.get(key) != live.get(key):
                differences.append({'work_id': row['work_id'], 'field': key,
                                    'snapshot_value': row.get(key), 'current_value': live.get(key)})
        rows.append(live)
    pq.write_table(pa.Table.from_pylist(rows), DEFAULT_OUT / 'legacy_papers_current_metadata.parquet')
    write_csv(DEFAULT_OUT / 'metadata_version_differences.csv', differences)
    write_csv(DEFAULT_OUT / 'metadata_conflicts.csv', [{
        'work_id': row['original_work_id'], 'doi': row['doi'],
        'issue': 'current_doi_identity_not_unique', 'candidate_ids': row['current_matches'],
        'status': 'unresolved_not_merged'} for row in doi_resolutions if not row['resolved']])
    write_json(DEFAULT_OUT / 'current_id_recovery_audit.json', {
        'requested': len(ids), 'returned_by_work_id': returned_work_ids, 'matched_requested': len(rows),
        'doi_resolutions': doi_resolutions,
        'missing_requested_ids': missing, 'unrequested_returned_ids': sorted(set(current) - set(ids)),
        'difference_counts': dict(Counter(r['field'] for r in differences)),
        'papers_with_any_difference': len({r['work_id'] for r in differences}),
        'current_b_candidate_with_legacy_claim_n': sum(r['current_b_candidate'] and r['has_legacy_claim'] for r in rows),
        'missing_current_ids': {key: sum(not r[key] for r in rows) for key in ['source_id', 'field_id', 'primary_topic_id']},
        'limitation': 'Current metadata sidecar; historical assets and labels preserved.'})


if __name__ == '__main__':
    main()
