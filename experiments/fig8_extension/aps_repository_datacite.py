"""Retrieve fixed repository DOI relationships through public DataCite batches."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced


def run() -> dict:
    folder = ROOT / 'supplement/aps_repository_identity'
    scope = json.loads((folder / 'scope.json').read_text())['papers']
    requested = sorted({p['doi'] for p in scope})
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0'
    receipts = []
    for start in range(0, len(requested), 50):
        batch = requested[start:start + 50]
        params = {'query': 'doi:(' + ' OR '.join('"' + d + '"' for d in batch) + ')',
                  'page[size]': 100, 'affiliation': 'true', 'publisher': 'true'}
        token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = folder / (token + '.json')
        if path.exists():
            receipt = json.loads(path.read_text())
        else:
            receipt = {'url': 'https://api.datacite.org/dois', 'params': params,
                       'requested_dois': batch, 'attempts': [],
                       'retrieved_at': datetime.now(timezone.utc).isoformat()}
            for attempt in range(3):
                paced(receipt['url'])
                try:
                    response = session.get(receipt['url'], params=params, timeout=(10, 45))
                    receipt['attempts'].append({'http_status': response.status_code})
                    receipt['http_status'] = response.status_code
                    raw = path.with_suffix('.body.gz')
                    atomic_bytes(raw, gzip.compress(response.content))
                    receipt['raw_receipt'] = str(raw.relative_to(ROOT))
                    receipt['retry_after'] = response.headers.get('Retry-After', '')
                    if response.status_code not in {500, 502, 503, 504} or receipt['retry_after']:
                        break
                except requests.RequestException as error:
                    receipt['attempts'].append({'error': type(error).__name__})
                if attempt < 2:
                    time.sleep(2 ** attempt)
            atomic_json(path, receipt)
        receipts.append(str(path.relative_to(ROOT)))
        if receipt.get('http_status') != 200:
            break
    session.close()
    result = {'fixed_dois': len(requested), 'batch_receipts': receipts,
              'status': 'batch_metadata_collected_identity_review_required',
              'qualification_changed': False}
    atomic_json(folder / 'acquisition.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
