"""Fetch v8 only for unresolved version identities already present in official batches."""
from __future__ import annotations

import fcntl
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .elife_batch import FOLDER, URL
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

OUT = FOLDER / 'version_details'
ACCEPT = 'application/vnd.elife.article-vor+json;version=8'


def fetch(identifier: str) -> dict:
    path = OUT / (identifier + '.json')
    if path.exists():
        return json.loads(path.read_text())
    if identifier == '95648':
        result = json.loads((FOLDER / 'version_representation_probe.json').read_text())
        atomic_json(path, result)
        return result
    url = URL + '/' + identifier
    result = {'url': url, 'headers': {'Accept': ACCEPT}, 'attempts': []}
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(3):
            paced(url)
            try:
                response = session.get(url, headers={'Accept': ACCEPT}, timeout=(10, 35))
                result['attempts'].append({'http_status': response.status_code})
                result.update(http_status=response.status_code, content_type=response.headers.get('Content-Type'))
                raw = OUT / (identifier + '.json.gz')
                atomic_bytes(raw, gzip.compress(response.content))
                result['raw_receipt'] = str(raw.relative_to(ROOT))
                if response.status_code not in {500, 502, 503, 504}:
                    break
                if response.headers.get('Retry-After'):
                    result['retry_after'] = response.headers['Retry-After']
                    break
            except requests.RequestException as error:
                result['attempts'].append({'error': type(error).__name__})
            if attempt < 2:
                time.sleep(2 ** attempt)
    atomic_json(path, result)
    return result


def run() -> None:
    scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
    targets = {p['doi'].split('elife.')[1].split('.')[0] for p in scope
               if len(p['doi'].split('elife.')[1].split('.')) > 1}
    batch = json.loads((FOLDER / 'batch_progress.json').read_text())
    for records in batch['matched'].values():
        for record in records:
            if record['native'].get('elocationId', '').startswith('RP'):
                targets.add(record['native']['id'])
    available = set()
    for path in (FOLDER / 'raw').glob('*.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('url') != URL or receipt.get('http_status') != 200:
            continue
        document = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
        available.update(item['id'] for item in document.get('items', []))
    ids = sorted(targets & available)
    atomic_json(OUT / 'scope.json', {'ids': ids, 'absent_from_article_batch': sorted(targets - available),
                                   'reason': 'Version DOI or RP publication-path identity unresolved; v8 exposes doiVersion',
                                   'scope_not_expanded': True})
    state = {'status': 'running', 'planned': len(ids), 'processed': 0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(ids), 4):
            for result in pool.map(fetch, ids[start:start + 4]):
                state['processed'] += 1
                if result.get('http_status') in {401, 403, 429}:
                    state.update(status='source_restricted_unattempted_retained', failure=result)
                    atomic_json(OUT / 'status.json', state)
                    return
            atomic_json(OUT / 'status.json', state)
    state['status'] = 'fixed_version_identity_queue_fetched_review_required'
    atomic_json(OUT / 'status.json', state)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
