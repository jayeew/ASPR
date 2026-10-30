"""Cache public SciPost pagination for the fixed existing DOI roster."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import time
from urllib.parse import urlsplit, urlunsplit

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

FOLDER = ROOT / 'supplement/scipost_bulk'


def fetch(session: requests.Session, url: str) -> dict:
    token = hashlib.sha256(url.encode()).hexdigest()
    receipt = FOLDER / 'pages' / (token + '.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    result = {'url': url, 'attempts': []}
    for attempt in range(3):
        paced(url)
        try:
            response = session.get(url, timeout=(10, 35))
            result['attempts'].append({'http_status': response.status_code})
            raw = receipt.with_suffix('.raw.gz')
            atomic_bytes(raw, gzip.compress(response.content))
            result.update(http_status=response.status_code, raw_receipt=str(raw.relative_to(ROOT)))
            if response.status_code not in {500, 502, 503, 504} or response.headers.get('Retry-After'):
                break
        except requests.RequestException as error:
            result['attempts'].append({'error': type(error).__name__})
        if attempt < 2:
            time.sleep(2 ** attempt)
    atomic_json(receipt, result)
    return result


def run() -> None:
    (FOLDER / 'pages').mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roster = json.loads((FOLDER / 'scope.json').read_text())['papers']
        targets = {p['doi'].lower() for p in roster if p['doi']}
        receipt = json.loads((FOLDER / 'publications_first_receipt.json').read_text())
        receipt['raw_receipt'] = str((FOLDER / 'publications_first.json.gz').relative_to(ROOT))
        session = requests.Session()
        session.trust_env = False
        session.headers.update({'Accept': 'application/json', 'User-Agent': 'Fig8MetadataResearch/1.0'})
        seen = set()
        matches = {}
        reason = 'public_listing_exhausted'
        pages = 0
        while True:
            if receipt.get('http_status') != 200:
                reason = 'source_failed_no_automatic_repeat'
                break
            data = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
            for item in data['results']:
                doi = item['doi'].lower()
                if doi in targets:
                    matches[doi] = {'record': item, 'receipt': receipt['raw_receipt'], 'identity_and_type_not_yet_verified': True}
            pages += 1
            atomic_json(FOLDER / 'matches.json', matches)
            atomic_json(FOLDER / 'status.json', {'status': 'running', 'pages': pages, 'matched_dois': len(matches), 'fixed_dois': len(targets)})
            link = data.get('next')
            if not link or targets <= matches.keys():
                break
            parts = urlsplit(link)
            if parts.hostname != 'scipost.org' or parts.path != '/api/publications/':
                reason = 'unexpected_pagination_target'
                break
            url = urlunsplit(('https', parts.netloc, parts.path, parts.query, ''))
            if url in seen:
                reason = 'repeated_pagination_link'
                break
            seen.add(url)
            receipt = fetch(session, url)
        session.close()
        atomic_json(FOLDER / 'status.json', {'status': 'acquisition_finished_review_required', 'reason': reason, 'pages': pages, 'matched_dois': len(matches), 'fixed_dois': len(targets), 'unmatched_dois': sorted(targets - matches.keys()), 'qualification_completion_not_implied': True})


if __name__ == '__main__':
    run()
