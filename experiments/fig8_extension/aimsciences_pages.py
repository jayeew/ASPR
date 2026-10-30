"""Collect the fixed, publisher-linked remaining AIMS Sciences page roster."""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml.etree import ParserError

from .supplement_queue import ROOT
from .supplement_sources import atomic_json, paced, parse_publisher

FOLDER = ROOT / 'supplement/aimsciences_native'
STOP = threading.Event()


def collect(row: dict) -> dict:
    doi = row['doi']
    path = FOLDER / 'pages' / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
    if path.exists():
        return json.loads(path.read_text())
    result = {'doi': doi, 'work_id': row['work_id'], 'attempts': [],
              'url': 'https://www.aimsciences.org/article/doi/' + doi,
              'network_route': 'direct_trust_env_false'}
    if STOP.is_set():
        return {**result, 'status': 'not_requested_source_hold'}
    raw_path = path.with_suffix('.html.gz')
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(1, 4):
            if STOP.is_set():
                result['status'] = 'not_requested_source_hold'
                break
            paced(result['url'])
            try:
                response = session.get(result['url'], timeout=(10, 30))
                result['attempts'].append({'number': attempt, 'status': response.status_code})
                result.update(http_status=response.status_code,
                              retry_after=response.headers.get('Retry-After', ''))
                if response.status_code in {401, 403, 429}:
                    STOP.set()
                if response.status_code == 200:
                    raw_path.write_bytes(gzip.compress(response.content))
                    result.update(status='cached_for_identity_and_type_review',
                                  raw_receipt=str(raw_path.relative_to(ROOT)),
                                  raw_sha256=hashlib.sha256(response.content).hexdigest())
                    try:
                        result['parsed'] = parse_publisher(response.content)
                    except (ParserError, ValueError) as exc:
                        result.update(status='cached_parse_unresolved', parse_error=type(exc).__name__)
                    break
                result['status'] = 'http_unresolved'
                if response.status_code < 500:
                    break
            except requests.RequestException as exc:
                result['attempts'].append({'number': attempt, 'error': type(exc).__name__})
                result['status'] = 'transport_unresolved'
    if result['status'] != 'not_requested_source_hold':
        atomic_json(path, result)
    return result


def run() -> None:
    (FOLDER / 'pages').mkdir(exist_ok=True)
    records = json.loads((FOLDER / 'page_scope.json').read_text())['records']
    probe = next(r for r in records if r['doi'] == '10.3934/dcdss.2023169')
    raw = gzip.decompress((FOLDER / 'capability.html.gz').read_bytes())
    path = FOLDER / 'pages' / (hashlib.sha256(probe['doi'].encode()).hexdigest() + '.json')
    if not path.exists():
        atomic_json(path, {**probe, 'url': 'https://www.aimsciences.org/article/doi/' + probe['doi'],
                    'status': 'cached_for_identity_and_type_review', 'attempts': [],
                    'raw_receipt': 'supplement/aimsciences_native/capability.html.gz',
                    'raw_sha256': hashlib.sha256(raw).hexdigest(), 'parsed': parse_publisher(raw),
                    'reused_probe': True})
    counts = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(collect, records):
            counts[result['status']] += 1
            done = sum(counts.values())
            atomic_json(FOLDER / 'page_status.json', {
                'status': 'source_hold' if STOP.is_set() else ('complete' if done == len(records) else 'running'),
                'fixed_dois': len(records), 'reviewed_jobs': done, 'outcomes': dict(counts),
                'qualification_changes': 0, 'final_task_closure': False})
            if done % 10 == 0 or done == len(records):
                print(done, '/', len(records), dict(counts), flush=True)


if __name__ == '__main__':
    run()
