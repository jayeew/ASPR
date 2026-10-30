"""Resolve candidate DOI aliases and publication-year errors with source evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
import requests

from .nature_boundary import page_evidence
from .nature_types import OUT, Publisher
from .stage0 import write_json


def datacite(doi: str) -> dict[str, Any]:
    url = 'https://api.datacite.org/dois/' + doi
    path = OUT / 'datacite_raw' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
    if not path.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        write_json(path, {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                         'response': response.json()})
    return json.loads(path.read_text())['response']['data']['attributes']


def review(row: dict[str, str], catalogue_dois: set[str], publisher: Publisher) -> dict[str, Any]:
    doi = row['doi']
    if doi.startswith('10.1038/'):
        evidence = page_evidence(publisher, {'article_url': 'https://www.nature.com/articles/' + doi.split('/')[1]})
        metadata = evidence['metadata']
        if metadata.get('citation_doi') != [doi]:
            raise ValueError('Publisher DOI mismatch')
        date = metadata.get('citation_online_date', [''])[0].replace('/', '-')
        if not date or date[:4] in ['2023', '2024', '2025']:
            raise ValueError('Unexpected date needs manual review')
        return {'doi': doi, 'decision': 'excluded_publisher_online_date_outside_window',
                'publisher_date': date, 'canonical_doi': doi, 'evidence_url': evidence['article_url'],
                'evidence': metadata}
    data = datacite(doi)
    variants = [r['relatedIdentifier'].lower() for r in data.get('relatedIdentifiers', [])
                if r['relationType'] == 'IsVariantFormOf' and r['relatedIdentifierType'] == 'DOI']
    supplements = [r['relatedIdentifier'].lower() for r in data.get('relatedIdentifiers', [])
                   if r['relationType'] == 'IsSupplementTo' and r['relatedIdentifierType'] == 'DOI']
    if len(variants) == 1 and variants[0] in catalogue_dois:
        decision, canonical = 'excluded_duplicate_repository_version', variants[0]
    elif len(supplements) == 1 and doi.startswith('10.5281/zenodo.'):
        decision, canonical = 'excluded_supplementary_data_record', supplements[0]
    else:
        raise ValueError(f'Unresolved DOI identity: {doi}')
    return {'doi': doi, 'decision': decision, 'canonical_doi': canonical,
            'evidence_url': 'https://api.datacite.org/dois/' + doi,
            'evidence': data.get('relatedIdentifiers')}


def main() -> None:
    with (OUT / 'prior_candidate_decisions.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['decision'].startswith(('not_in_', 'excluded_duplicate_',
                 'excluded_supplementary_', 'excluded_publisher_online_'))]
    catalogue = pq.read_table(OUT / 'publisher_catalogue.parquet', columns=['doi']).to_pylist()
    publisher = Publisher(OUT)
    resolutions = [review(row, {r['doi'] for r in catalogue}, publisher) for row in rows]
    write_json(OUT / 'prior_candidate_resolutions.json', resolutions)
    print(f'Resolved {len(resolutions)} candidate identity/date exceptions')


if __name__ == '__main__':
    main()
