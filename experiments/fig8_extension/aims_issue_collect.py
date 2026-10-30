"""Fetch only the AIMS journal/issues linked by the fixed candidate PDF URLs."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter

import requests

from .aims_native_collect import FOLDER
from .supplement_sources import atomic_json, paced


def run() -> None:
    folder = FOLDER / 'issues'; folder.mkdir(exist_ok=True)
    rows = json.loads((FOLDER / 'scope.json').read_text())['records']
    groups: dict[tuple, set[str]] = {}
    for row in rows:
        match = re.search(r'/aimspress-data/([^/]+)/(\d{4})/([^/]+)/', row['pdf_url'])
        if match:
            groups.setdefault(match.groups(), set()).add(row['doi'])
    atomic_json(FOLDER / 'issue_scope.json', {'groups': [
        {'publisher_id': k[0], 'year': k[1], 'issue': k[2], 'dois': sorted(v)} for k, v in groups.items()],
        'unrouted': [r for r in rows if not re.search(r'/aimspress-data/([^/]+)/(\d{4})/([^/]+)/', r['pdf_url'])]})
    session = requests.Session(); session.trust_env = False
    url = 'https://www.aimspress.com/data/article/archive-article-data'
    counts = Counter(); matched = set(); requested = 0
    for group, dois in groups.items():
        params = dict(zip(['publisherId', 'year', 'issue'], group)); params['volume'] = ''
        name = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        receipt = folder / (name + '.json'); raw_path = folder / (name + '.json.gz')
        if receipt.exists():
            result = json.loads(receipt.read_text())
        elif group == ('mbe', '2023', '12'):
            raw_path.write_bytes((FOLDER / 'issue_probe.json.gz').read_bytes())
            result = {'http_status': 200, 'reused_probe': True, 'params': params,
                      'raw_receipt': str(raw_path), 'url': url, 'attempts': []}
            atomic_json(receipt, result)
        else:
            result = {'url': url, 'params': params, 'attempts': []}
            for number in range(1, 4):
                paced(url)
                try:
                    response = session.post(url, params=params, timeout=(10, 30)); requested += 1
                    result['attempts'].append({'number': number, 'status': response.status_code})
                    result.update(http_status=response.status_code, retry_after=response.headers.get('Retry-After', ''))
                    if response.status_code == 200:
                        raw_path.write_bytes(gzip.compress(response.content))
                        result.update(raw_receipt=str(raw_path), sha256=hashlib.sha256(response.content).hexdigest())
                    if response.status_code < 500:
                        break
                except requests.RequestException as exc:
                    result['attempts'].append({'number': number, 'error': type(exc).__name__})
            atomic_json(receipt, result)
        if result.get('http_status') in {401, 403, 429}:
            atomic_json(FOLDER / 'issue_status.json', {'status': 'source_hold', 'receipt': str(receipt),
                        'completed_groups': sum(counts.values()), 'fixed_groups': len(groups), 'outcomes': dict(counts)})
            break
        if result.get('http_status') == 200:
            try:
                data = json.loads(gzip.decompress(raw_path.read_bytes()))
            except (ValueError, OSError):
                counts['invalid_response'] += 1
            else:
                articles = data.get('data', {}).get('articles', []) if isinstance(data.get('data'), dict) else []
                matched.update(v.get('doi', '').casefold() for v in articles if v.get('doi', '').casefold() in dois)
                counts['success' if data.get('result') == 'success' else 'api_error'] += 1
        else:
            counts['request_unresolved'] += 1
        atomic_json(FOLDER / 'issue_status.json', {'status': 'complete' if sum(counts.values()) == len(groups) else 'running',
                    'fixed_groups': len(groups), 'completed_groups': sum(counts.values()), 'outcomes': dict(counts),
                    'matched_dois': len(matched), 'new_requests': requested, 'qualification_changes': 0})
        print(sum(counts.values()), '/', len(groups), 'matched', len(matched), flush=True)
    session.close()


if __name__ == '__main__':
    run()
