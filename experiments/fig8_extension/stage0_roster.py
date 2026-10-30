"""Materialize the bounded Nature candidate roster without model calls."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .stage0 import DEFAULT_OUT, FIELDS, Client, filters, write_csv, write_json

SELECT = ('id,doi,title,publication_date,publication_year,type,language,primary_topic,'
          'primary_location,abstract_inverted_index,authorships,updated_date,is_retracted,is_paratext')


def abstract_text(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positions = [(position, token) for token, values in index.items() for position in values]
    return ' '.join(token for _, token in sorted(positions))


def fetch_year(client: Client, year: int) -> list[dict[str, Any]]:
    cursor, rows, seen = '*', [], set()
    query = filters('nature').replace('publication_year:2023-2025', f'publication_year:{year}')
    while cursor:
        if cursor in seen:
            raise RuntimeError('Repeated Nature roster cursor')
        seen.add(cursor)
        payload = client.get('works', {'filter': query, 'cursor': cursor, 'per_page': 200, 'select': SELECT})
        rows.extend(payload['results'])
        cursor = payload['meta'].get('next_cursor')
        print(f'Nature {year}: {len(rows)}/{payload["meta"]["count"]}', flush=True)
    if len(rows) != payload['meta']['count'] or len(rows) != len({r['id'] for r in rows}):
        raise RuntimeError('Nature roster denominator or identity mismatch')
    return rows


def project(record: dict[str, Any]) -> dict[str, Any]:
    topic = record.get('primary_topic') or {}
    location = record.get('primary_location') or {}
    source = location.get('source') or {}
    authorships = record.get('authorships') or []
    field = topic.get('field') or {}
    return {'work_id': record['id'], 'doi': record.get('doi'), 'title': record.get('title'),
            'publication_date': record.get('publication_date'), 'year': record['publication_year'],
            'work_type': record.get('type'), 'language': record.get('language'),
            'source_id': source.get('id'), 'issn_l': source.get('issn_l'),
            'field_id': field.get('id'), 'field_name': field.get('display_name'),
            'subfield_id': (topic.get('subfield') or {}).get('id'), 'topic_id': topic.get('id'),
            'panel_b_overlap': str(field.get('id') or '').rsplit('/', 1)[-1] in FIELDS.values(),
            'abstract_text': abstract_text(record.get('abstract_inverted_index')),
            'has_institution': any(a.get('institutions') for a in authorships),
            'has_country': any(a.get('countries') or any(i.get('country_code') for i in a.get('institutions') or []) for a in authorships),
            'author_n': len(authorships),
            'authors_with_institution_n': sum(bool(a.get('institutions')) for a in authorships),
            'raw_type': location.get('raw_type'), 'version': location.get('version'),
            'is_published': location.get('is_published'),
            'authorships_json': json.dumps(authorships, ensure_ascii=False),
            'openalex_updated_date': record.get('updated_date'),
            'article_type_verified': False, 'date_source': 'OpenAlex; online_vs_issue_not_publisher_verified'}


def main() -> None:
    client = Client(DEFAULT_OUT)
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch_year, client, year) for year in [2023, 2024, 2025]]
        for future in as_completed(futures):
            rows.extend(project(row) for row in future.result())
    rows.sort(key=lambda row: row['work_id'])
    pq.write_table(pa.Table.from_pylist(rows), DEFAULT_OUT / 'nature_candidate_roster.parquet')
    write_csv(DEFAULT_OUT / 'nature_candidate_roster.csv', [{k: v for k, v in row.items()
               if k not in ['authorships_json', 'abstract_text']} for row in rows])
    summary = {'candidate_n': len(rows), 'abstract_n': sum(bool(row['abstract_text']) for row in rows),
               'b_overlap_n': sum(row['panel_b_overlap'] for row in rows),
               'b_overlap_abstract_n': sum(row['panel_b_overlap'] and bool(row['abstract_text']) for row in rows),
               'nature_only_n': sum(not row['panel_b_overlap'] for row in rows),
               'nature_only_abstract_n': sum(not row['panel_b_overlap'] and bool(row['abstract_text']) for row in rows),
               'duplicate_doi_excess': len(rows) - len({row['doi'] for row in rows}),
               'article_type_verification': 'pending for all candidates'}
    write_json(DEFAULT_OUT / 'nature_roster_audit.json', summary)


if __name__ == '__main__':
    main()
