"""Cache ROR source evidence for exact institution IDs and affiliation matches."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
import requests

from .nature_backfill_fetch import BASE, OUT
from .stage0 import Client, write_json


def normalized(value: str) -> str:
    value = unicodedata.normalize('NFKD', value).casefold()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in value if not unicodedata.combining(c)).split())


class Registry:
    def __init__(self) -> None:
        self.local = threading.local()

    def get(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        query = {'url': 'https://api.ror.org/v2/organizations' + endpoint, 'params': params or {}}
        token = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = OUT / 'ror_raw' / (token + '.json')
        if path.exists():
            return {**json.loads(path.read_text()), 'receipt': str(path)}
        if getattr(self.local, 'session', None) is None:
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        status: int | str = 'network'
        for attempt in range(3):
            try:
                response = self.local.session.get(query['url'], params=query['params'], timeout=(10, 35))
                status = response.status_code
                if response.ok:
                    result = {**query, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'response': response.json()}
                    write_json(path, result)
                    return {**result, 'receipt': str(path)}
                if status not in [429, 500, 502, 503, 504]:
                    break
            except (requests.RequestException, ValueError):
                status = 'network_or_json'
            time.sleep(2 ** attempt)
        raise RuntimeError(f'ROR request failed {status}; cache token {token}')


def collect_countries() -> None:
    rows = pq.read_table(BASE / 'research_roster_locked.parquet', columns=['authorships_json']).to_pylist()
    institutions = {i['id']: i for r in rows for a in json.loads(r['authorships_json'])
                    for i in a['institutions'] if not i.get('country_code')}
    client, output = Registry(), []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(client.get, '/' + i['ror'].rsplit('/', 1)[-1]): i
                   for i in institutions.values() if i.get('ror')}
        for index, future in enumerate(as_completed(futures), 1):
            inst = futures[future]
            try:
                raw = future.result()
                record = raw['response']
                if record.get('id') != inst['ror']:
                    raise ValueError('ROR identity mismatch')
                countries = sorted({x['geonames_details']['country_code'] for x in record.get('locations', [])
                                    if x.get('geonames_details', {}).get('country_code')})
                output.append({'openalex_id': inst['id'], 'ror': inst['ror'], 'name': inst['display_name'],
                    'country_codes': countries, 'country_code': countries[0] if len(countries) == 1 else None,
                    'status': 'exact_ror_single_country' if len(countries) == 1 else 'unresolved_multi_or_no_country',
                    'source_receipt': raw['receipt'], 'source_url': raw['url'], 'registry_status': record.get('status')})
            except (RuntimeError, ValueError) as error:
                output.append({'openalex_id': inst['id'], 'ror': inst['ror'], 'status': 'failed', 'error': str(error)})
            if index % 50 == 0 or index == len(futures):
                print(f'ROR countries {index}/{len(futures)}', flush=True)
    for inst in institutions.values():
        if not inst.get('ror'):
            output.append({'openalex_id': inst['id'], 'status': 'missing_ror_id'})
    write_json(OUT / 'institution_country_repairs.json', output)


def collect_affiliations() -> None:
    candidates = json.loads((OUT / 'affiliation_query_inputs.json').read_text())
    client, output = Registry(), []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(client.get, '', {'affiliation': address}): address for address in candidates}
        for index, future in enumerate(as_completed(futures), 1):
            address = futures[future]
            try:
                raw = future.result()
                chosen = [item for item in raw['response'].get('items', []) if item.get('chosen')]
                output.append({'address': address, 'chosen': chosen, 'source_receipt': raw['receipt'],
                               'source_url': raw['url'], 'status': 'queried'})
            except (RuntimeError, ValueError) as error:
                output.append({'address': address, 'status': 'failed', 'error': str(error)})
            if index % 50 == 0 or index == len(futures):
                print(f'ROR affiliations {index}/{len(futures)}', flush=True)
    write_json(OUT / 'affiliation_query_results.json', output)


def collect_crosswalk() -> None:
    results = json.loads((OUT / 'affiliation_query_results.json').read_text())
    rors = sorted({item['organization']['id'] for row in results for item in row.get('chosen', [])})
    client = Client(OUT / 'openalex')
    client.local.session = requests.Session()
    client.local.session.trust_env = False
    records = []
    for start in range(0, len(rors), 100):
        query = {'filter': 'ror:' + '|'.join(rors[start:start + 100]), 'per_page': 200}
        raw = client.get('institutions', query)
        if raw['meta']['count'] > len(raw['results']):
            raise ValueError('Institution crosswalk exceeds one page')
        token = hashlib.sha256(json.dumps({'endpoint': 'institutions', 'params': query}, sort_keys=True).encode()).hexdigest()
        receipt = str(OUT / 'openalex/api_raw' / (token + '.json'))
        records.extend({**r, 'openalex_identity_source': receipt} for r in raw['results'])
        print(f'ROR/OpenAlex crosswalk {min(start + 100, len(rors))}/{len(rors)}', flush=True)
    write_json(OUT / 'ror_openalex_crosswalk.json', records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['countries', 'affiliations', 'crosswalk'], required=True)
    args = parser.parse_args()
    {'countries': collect_countries, 'affiliations': collect_affiliations, 'crosswalk': collect_crosswalk}[args.phase]()


if __name__ == '__main__':
    main()
