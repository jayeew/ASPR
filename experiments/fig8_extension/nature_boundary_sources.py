"""Source collection for the university-boundary affiliation review; prior runs are immutable."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pyarrow.parquet as pq
import requests

from .nature_backfill_fetch import BASE, DirectPublisher, parse_page
from .stage0 import write_json

ROOT = BASE.parent
PREV = ROOT / 'nature_weighted_review'
OUT = ROOT / 'nature_university_boundary_review'
FIELD = 'authorships_boundary_review_json'
SLOTS = 'institution_boundary_review_entities'
OLD_FIELD = 'authorships_weighted_review_json'
OLD_SLOTS = 'institution_weighted_review_entities'
LOCK = threading.Lock()
NEXT_REQUEST = 0.0
LOCAL = threading.local()


def load_rows() -> list[dict[str, Any]]:
    return pq.read_table(PREV / 'research_roster_weighted_review.parquet').to_pylist()


def session() -> requests.Session:
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def fetch_ror(ror: str) -> dict[str, Any]:
    global NEXT_REQUEST
    path = OUT / 'ror_records' / (ror.rsplit('/', 1)[-1] + '.json')
    if path.exists():
        return {**json.loads(path.read_text()), 'receipt': str(path)}
    url = 'https://api.ror.org/v2/organizations/' + ror.rsplit('/', 1)[-1]
    result: dict[str, Any] = {'ror': ror, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    for attempt in range(3):
        with LOCK:
            delay = max(0.0, NEXT_REQUEST - time.monotonic())
            NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic()) + 0.18
        time.sleep(delay)
        try:
            response = session().get(url, timeout=(10, 25))
            result.update(status=response.status_code, attempts=attempt + 1)
            if response.ok:
                org = response.json()
                if org.get('id') != ror:
                    raise ValueError('Registry identity mismatch')
                result['organization'] = org
                break
            if response.status_code not in [429, 500, 502, 503, 504]:
                break
        except (requests.RequestException, ValueError) as error:
            result['error'] = str(error)
        time.sleep(min(3 * (attempt + 1), 9))
    write_json(path, result)
    return {**result, 'receipt': str(path)}


def root_university_name(name: str) -> bool:
    # Only a discovery filter: never itself proves an ownership relationship.
    return bool(re.search(r'university|universit|universidad|univerz|universyt|politecn|polytechni|institute of technology', name, re.IGNORECASE)) and not bool(
        re.search(r'hospital|medicine|medical cent|medical college|college of|school|laborator|department|cancer|clinic|health|observatory|institute for|center for|centre for', name, re.IGNORECASE))


def collect_registry() -> None:
    inventory = json.loads((OUT / 'entity_inventory.json').read_text())
    by_id = {x['entity']['review_entity_id']: x for x in inventory}
    education = {k for k, x in by_id.items() if x['entity']['review_entity_type'] in ['education_proxy', 'university']}
    selected = []
    for item in inventory:
        ent = item['entity']
        if not ent.get('ror'):
            continue
        reasons = []
        if any(p in education and p != ent['review_entity_id'] for p in ent.get('lineage', [])):
            reasons.append('openalex_lineage_candidate_not_evidence')
        if ent['review_entity_type'] == 'healthcare':
            reasons.append('all_healthcare_entities')
        if ent['review_entity_type'] in ['education_proxy', 'university'] and not root_university_name(ent['display_name']):
            reasons.append('education_subunit_or_independent_college')
        if re.search(r'cornell|peking|harvard|stanford|massachusetts institute of technology', ent['display_name'], re.IGNORECASE):
            reasons.append('top_and_boundary_named_units')
        if reasons:
            selected.append(dict(item, reasons=reasons))
    write_json(OUT / 'registry_discovery_roster.json', selected)
    cache = json.loads((OUT / 'ror_cache_index.json').read_text())
    needed = sorted({x['entity']['ror'] for x in selected} - cache.keys())
    print(f'{len(selected)} candidate entities; {len(needed)} uncached exact registry requests', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fetch_ror, ror): ror for ror in needed}
        for i, future in enumerate(as_completed(jobs), 1):
            result = future.result()
            results.append(result)
            if result.get('organization'):
                cache[result['ror']] = result
            if i % 100 == 0:
                print(f'{i}/{len(needed)}; successful {sum(bool(x.get("organization")) for x in results)}', flush=True)
    write_json(OUT / 'ror_registry_index.json', cache)
    write_json(OUT / 'registry_fetch_receipts.json', results)
    print('Registry collection complete', flush=True)


@lru_cache(maxsize=512)
def publisher(doi: str) -> dict[str, Any]:
    suffix = doi.split('/')[1] + '.json'
    for folder in [ROOT / 'nature_metadata_backfill', PREV, OUT]:
        path = folder / 'publisher_records' / suffix
        if path.exists():
            return json.loads(path.read_text())
    url = 'https://www.nature.com/articles/' + doi.split('/')[1]
    token = hashlib.sha256(url.encode()).hexdigest()
    folder = BASE if (BASE / 'publisher_raw' / (token + '.html.gz')).exists() else OUT
    result = parse_page(DirectPublisher(folder).get(url), doi)
    result.update(source_url=url, source_receipt=str(folder / 'publisher_raw' / (token + '.html.json')))
    write_json(OUT / 'publisher_records' / suffix, result)
    return result


def capture(url: str) -> dict[str, Any]:
    path = OUT / 'primary_sources' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
    if path.exists():
        return {**json.loads(path.read_text()), 'receipt': str(path)}
    result: dict[str, Any] = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    try:
        response = session().get(url, timeout=(10, 25))
        result.update(status=response.status_code, final_url=response.url)
        if response.ok:
            raw = path.with_suffix('.raw.gz')
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(gzip.compress(response.content))
            result.update(raw_path=str(raw), sha256=hashlib.sha256(response.content).hexdigest())
    except requests.RequestException as error:
        result['error'] = str(error)
    write_json(path, result)
    return {**result, 'receipt': str(path)}


if __name__ == '__main__':
    collect_registry()
