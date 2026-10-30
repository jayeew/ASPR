"""Collect a fixed AIMS DOI queue using small public pages and retained receipts."""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from lxml import html

from .supplement_queue import ROOT
from .supplement_sources import atomic_json, paced, parse_publisher

FOLDER = ROOT / 'supplement/aims_native'
STOP = threading.Event()


def parse(raw: bytes, doi: str) -> dict:
    """Keep literal header types; untyped Published dates remain non-electronic."""
    base = parse_publisher(raw)
    tree = html.fromstring(raw)
    kinds = list(dict.fromkeys(' '.join(n.itertext()).strip() for n in
        tree.xpath('//span[contains(concat(" ",normalize-space(@class)," ")," columnli ")]')))
    abstracts = tree.xpath('//meta[translate(@name,"ABCDEFGHIJKLMNOPQRSTUVWXYZ",'
                          '"abcdefghijklmnopqrstuvwxyz")="dc.description"]/@content')
    abstract = ' '.join(html.fromstring(abstracts[0]).itertext()) if abstracts else ''
    dates = [' '.join(n.getparent().itertext()).strip() for n in
             tree.xpath('//b[normalize-space(text())="Published:"]')]
    return {**base, 'literal_types': [v for v in kinds if v and v != 'Special Issues'],
            'header_labels': kinds, 'abstract': abstract,
            'published_labels_not_online_evidence': dates,
            'doi_matched': doi.casefold() in {v.casefold() for v in base.get('doi', [])}}


def collect(row: dict) -> dict:
    doi = row['doi']; name = hashlib.sha256(doi.encode()).hexdigest()
    receipt = FOLDER / 'pages' / (name + '.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    if STOP.is_set():
        return {'status': 'not_requested_source_hold', 'doi': doi}
    url = 'https://www.aimspress.com/article/doi/' + doi
    raw_path = receipt.with_suffix('.html.gz')
    result = {'doi': doi, 'work_id': row['work_id'], 'url': url,
              'network_route': 'direct_trust_env_false', 'attempts': []}
    session = requests.Session(); session.trust_env = False
    for number in range(1, 4):
        if STOP.is_set():
            result['status'] = 'not_requested_source_hold'; break
        paced(url)
        try:
            response = session.get(url, timeout=(10, 30))
            result['attempts'].append({'number': number, 'http_status': response.status_code})
            if response.status_code in {401, 403, 429}:
                STOP.set()
            if response.status_code == 200:
                raw = response.content
                raw_path.write_bytes(gzip.compress(raw))
                result.update(status='cached_for_identity_review', raw_receipt=str(raw_path.relative_to(ROOT)),
                              raw_sha256=hashlib.sha256(raw).hexdigest(), parsed=parse(raw, doi))
                break
            result.update(status='http_unresolved', http_status=response.status_code,
                          retry_after=response.headers.get('Retry-After', ''))
            if response.status_code < 500:
                break
        except requests.RequestException as exc:
            result['attempts'].append({'number': number, 'error': type(exc).__name__})
            result['status'] = 'transport_unresolved'
    session.close()
    atomic_json(receipt, result)
    return result


def run() -> None:
    (FOLDER / 'pages').mkdir(exist_ok=True)
    records = json.loads((FOLDER / 'scope.json').read_text())['records']
    unique = {r['doi']: r for r in records}
    probe = json.loads((FOLDER / 'page_capability.json').read_text())
    doi = '10.3934/mbe.2023909'
    raw = gzip.decompress((ROOT / probe['raw_receipt']).read_bytes())
    atomic_json(FOLDER / 'pages' / (hashlib.sha256(doi.encode()).hexdigest() + '.json'),
                {'doi': doi, 'work_id': unique[doi]['work_id'], 'url': probe['url'],
                 'status': 'cached_for_identity_review', 'raw_receipt': probe['raw_receipt'],
                 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'parsed': parse(raw, doi),
                 'attempts': [], 'reused_probe': True})
    start = time.monotonic(); counts = Counter(); completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(collect, unique.values()):
            completed += 1; counts[result['status']] += 1
            if completed % 25 == 0 or completed == len(unique):
                atomic_json(FOLDER / 'collection_status.json', {
                    'status': 'source_hold' if STOP.is_set() else ('complete' if completed == len(unique) else 'running'),
                    'fixed_dois': len(unique), 'reviewed_jobs': completed, 'outcomes': dict(counts),
                    'seconds': time.monotonic() - start, 'qualification_changes': 0,
                    'final_task_closure': False})
                print(completed, dict(counts), flush=True)


if __name__ == '__main__':
    run()
