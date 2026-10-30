"""Second-pass institution review with cached registry evidence and explicit unknowns."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .nature_backfill_fetch import OUT as PREV
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import address_country, clean_address, country_names
from .stage0 import write_csv, write_json

OUT = PREV.parent / 'nature_institution_review'


def units() -> list[dict[str, Any]]:
    groups: dict[str, Any] = {}
    records = json.loads((PREV / 'unresolved_author_institutions.json').read_text())
    for row in records:
        for aff in row['affiliation_evidence']:
            address = aff['address']
            unit = groups.setdefault(address, {'address': address, 'papers': set(), 'authors': set()})
            unit['papers'].add(row['doi'])
            unit['authors'].add((row['doi'], row['author_position']))
    return sorted([{'address': k, 'paper_n': len(v['papers']), 'author_n': len(v['authors']),
                    'dois': sorted(v['papers'])} for k, v in groups.items()],
                  key=lambda r: (-r['paper_n'], -r['author_n'], r['address']))


def name_queries(address: str) -> list[str]:
    parts = [s.strip() for s in clean_address(address).split(',')]
    org = parts[:-2] if len(parts) > 2 else parts[:1]
    entity = [s for s in org if re.search(r'\b(university|universit|hospital|museum|company|inc\.?|ltd\.?|gmbh|llc)\b', s, re.IGNORECASE)]
    chosen = [x for x in entity if len(normalized(x).split()) > 1][-1:] or org[:1]
    return [re.sub(r'\s*\([^)]*\)', '', x).strip() for x in chosen if len(x) > 3]


class ReviewRegistry:
    def __init__(self) -> None:
        self.local = threading.local()

    def query(self, term: str) -> dict[str, Any]:
        path = OUT / 'ror_queries' / (hashlib.sha256(term.encode()).hexdigest() + '.json')
        if path.exists():
            return {**json.loads(path.read_text()), 'receipt': str(path)}
        if getattr(self.local, 'session', None) is None:
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        url = 'https://api.ror.org/v2/organizations'
        for attempt in range(3):
            try:
                response = self.local.session.get(url, params={'query': term}, timeout=(10, 30))
                response.raise_for_status()
                result = {'query': term, 'url': response.url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                          'response': response.json()}
                write_json(path, result)
                return {**result, 'receipt': str(path)}
            except (requests.RequestException, ValueError) as error:
                if attempt == 2:
                    return {'query': term, 'error': str(error), 'response': {'items': []}}
                time.sleep(2 ** attempt)
        raise RuntimeError('Unreachable retry state')


def collect(quoted: bool = False) -> None:
    ranked = units()
    write_json(OUT / 'priority_units.json', ranked)
    terms = list(dict.fromkeys(q for row in ranked for q in name_queries(row['address'])))
    if quoted:
        extra = [p.strip() for row in ranked for p in row['address'].split(',')[:-2]
                 if len(p.strip()) > 5 and not re.match(r'^(department|division|faculty|school|unit)\b', p.strip(), re.IGNORECASE)]
        terms = list(dict.fromkeys('\"' + q.replace('\"', '') + '\"' for q in terms + extra))
    client, results = ReviewRegistry(), []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(client.query, q): q for q in terms}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append({'query': futures[future], 'receipt': result.get('receipt'), 'error': result.get('error')})
            if i % 50 == 0 or i == len(terms):
                print(f'Name queries {i}/{len(terms)}', flush=True)
    write_json(OUT / ('query_log_quoted.json' if quoted else 'query_log.json'), results)


def canonical_name(value: str) -> str:
    value = re.sub(r'\s*\([^)]*\)\s*$', '', value)
    key = normalized(value)
    return re.sub(r'^the ', '', key)


def exact_candidate(address: str, org: dict[str, Any]) -> tuple[bool, str]:
    parts = [p.strip() for p in clean_address(address).split(',')]
    country = address_country(address, country_names())
    tail_n = 3 if country in {'US', 'CA', 'AU'} and len(parts) >= 4 else 2
    org_parts = parts[:-tail_n] if len(parts) > tail_n else parts[:1]
    keys = {canonical_name(p) for p in org_parts}
    aliases = {canonical_name(n['value']) for n in org.get('names', []) if 'acronym' not in n['types']}
    matches = {a for a in aliases.intersection(keys) if len(a) >= 5}
    if not matches:
        return False, 'no_exact_organizational_component'
    countries = {l['geonames_details']['country_code'] for l in org.get('locations', [])}
    country = address_country(address, country_names())
    if not country or country not in countries:
        return False, 'country_conflict_or_missing'
    cities = {normalized(l['geonames_details'].get('name', '')) for l in org.get('locations', [])} - {''}
    if not cities.intersection({normalized(p) for p in parts[-4:]}):
        return False, 'city_requires_review'
    if org.get('status') != 'active':
        return False, 'inactive_registry_record'
    return True, 'exact_component_country_city'


def propose() -> None:
    old_queries = {r['address']: r for r in json.loads((PREV / 'affiliation_query_results.json').read_text())}
    output = []
    for unit in units():
        address, candidates, rejected = unit['address'], {}, {}
        receipts = []
        old = old_queries.get(address, {}).get('source_receipt')
        if old:
            receipts.append(old)
        terms = name_queries(address)
        terms += ['\"' + q.replace('\"', '') + '\"' for q in terms + [p.strip() for p in address.split(',')[:-2]]]
        for term in terms:
            path = OUT / 'ror_queries' / (hashlib.sha256(term.encode()).hexdigest() + '.json')
            if path.exists():
                receipts.append(str(path))
        for receipt in receipts:
            for item in json.loads(Path(receipt).read_text())['response'].get('items', []):
                org = item.get('organization', item)
                ok, why = exact_candidate(address, org)
                if why == 'city_requires_review':
                    rejected[org['id']] = {'ror': org['id'], 'reason': why, 'receipt': receipt}
                if ok:
                    candidates[org['id']] = {'organization': org, 'receipt': receipt, 'method': why}
        output.append({**unit, 'candidates': list(candidates.values()), 'query_receipts': list(dict.fromkeys(receipts)),
                       'plausible_rejected_candidates': list(rejected.values()),
                       'status': 'accepted_exact_registry' if len(candidates) == 1 else 'unresolved'})
    write_json(OUT / 'registry_proposals.json', output)
    write_csv(OUT / 'registry_proposals.csv', [{'address': r['address'], 'paper_n': r['paper_n'],
        'author_n': r['author_n'], 'status': r['status'], 'candidates': [{'id': x['organization']['id'],
        'names': x['organization']['names'], 'types': x['organization']['types']} for x in r['candidates']]} for r in output])
    print(f"Exact registry units: {sum(r['status'] == 'accepted_exact_registry' for r in output)}/{len(output)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['collect', 'quoted', 'propose'])
    args = parser.parse_args()
    {'collect': collect, 'quoted': lambda: collect(True), 'propose': propose}[args.phase]()


if __name__ == '__main__':
    main()
