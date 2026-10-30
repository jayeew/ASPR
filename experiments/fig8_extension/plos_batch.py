"""Fetch only the fixed candidate DOI list from the official PLOS search API."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def parameters(dois: list[str]) -> dict:
    if not dois or len(dois) > 50 or any(not d.startswith('10.1371/') or '"' in d for d in dois):
        raise ValueError('Expected at most 50 fixed PLOS DOIs')
    return {'q': 'id:(' + ' OR '.join('"' + d + '"' for d in dois) + ')',
            'fq': 'doc_type:full', 'rows': 100, 'wt': 'json',
            'fl': 'id,title_display,journal,eissn,article_type,publication_date,abstract_primary_display,author_display,publisher'}


def collect() -> dict:
    folder = ROOT / 'supplement/plos_batch'
    scope = json.loads((folder / 'scope.json').read_text())
    dois = sorted({p['doi'] for p in scope['papers']})
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (fixed DOI metadata batches)'
    state = {'fixed_dois': len(dois), 'batches': (len(dois) + 49) // 50,
             'completed_batches': 0, 'new_requests': 0, 'returned_records': 0,
             'status': 'running', 'source_policy': 'https://api.plos.org/solr/faq',
             'interval_seconds': 12.1, 'metadata_not_yet_applied': True}
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_request = 0.0
        for offset in range(0, len(dois), 50):
            group = dois[offset:offset + 50]
            index = offset // 50
            receipt = folder / f'batch_{index:04d}.json'
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record['dois'] != group:
                    raise ValueError('Fixed batch identity changed')
            else:
                record = {'dois': group, 'params': parameters(group), 'attempts': [],
                          'retrieved_at': datetime.now(timezone.utc).isoformat()}
            while record.get('http_status') != 200 and len(record['attempts']) < 3:
                if record.get('http_status') in {401, 403, 429}:
                    break
                if record.get('http_status') and record['http_status'] not in {500, 502, 503, 504}:
                    break
                time.sleep(max(0, 12.1 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                state['new_requests'] += 1
                try:
                    response = session.get('https://api.plos.org/search', params=record['params'], timeout=(10, 45))
                    body = response.content
                    raw = folder / f'batch_{index:04d}_attempt_{len(record["attempts"]) + 1}.raw.gz'
                    atomic_bytes(raw, gzip.compress(body, mtime=0))
                    attempt = {'http_status': response.status_code, 'raw_receipt': str(raw.relative_to(ROOT)),
                               'sha256': hashlib.sha256(body).hexdigest()}
                    record['attempts'].append(attempt)
                    record.update(attempt, url=response.url, retry_after=response.headers.get('Retry-After', ''))
                except requests.RequestException as error:
                    record['attempts'].append({'error': type(error).__name__})
                atomic_json(receipt, record)
            if record.get('http_status') != 200:
                state.update(status='source_stopped_unattempted_batches_remain', stopped_receipt=str(receipt.relative_to(ROOT)))
                break
            try:
                payload = json.loads(gzip.decompress((ROOT / record['raw_receipt']).read_bytes()))
                docs = payload['response']['docs']
                if payload['response']['numFound'] != len(docs) or any(d.get('id') not in group for d in docs):
                    raise ValueError('Incomplete page or off-scope result')
            except (KeyError, ValueError, OSError) as error:
                state.update(status='response_requires_review', error=type(error).__name__, stopped_receipt=str(receipt.relative_to(ROOT)))
                break
            state['completed_batches'] += 1
            state['returned_records'] += len(docs)
            atomic_json(folder / 'status.json', state)
        else:
            state['status'] = 'fixed_batches_collected_identity_type_date_review_required'
        atomic_json(folder / 'status.json', state)
    print(json.dumps(state), flush=True)
    return state


if __name__ == '__main__':
    collect()
