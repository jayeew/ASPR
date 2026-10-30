"""Resolve fixed HAL journal-name differences against its structured journal authority."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import time
from collections import defaultdict

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/hal_metadata/journal_identity'


def collect() -> bool:
    titles = json.loads((FOLDER / 'scope.json').read_text())['titles']
    session = requests.Session()
    session.trust_env = False
    for number, start in enumerate(range(0, len(titles), 25)):
        batch = titles[start:start + 25]
        path = FOLDER / f'titles_{number:03}.json'
        body = path.with_suffix('.body.gz')
        state = json.loads(path.read_text()) if path.exists() else {'titles': batch, 'attempts': []}
        if state['titles'] != batch:
            raise ValueError('Fixed journal identity batch changed')
        while state.get('http_status') != 200 and len(state['attempts']) < 3:
            if state.get('http_status') in {400, 401, 403, 404, 429}:
                break
            attempt: dict = {}
            state['attempts'].append(attempt)
            params = {'q': 'title_s:(' + ' OR '.join(json.dumps(t) for t in batch) + ')',
                      'wt': 'json', 'fl': '*', 'rows': 1000}
            try:
                response = session.get('https://api.hal.science/ref/journal/', params=params,
                                       timeout=(10, 35))
                raw = response.content
                body.write_bytes(gzip.compress(raw))
                state.update(url=response.url, http_status=response.status_code,
                             raw_receipt=str(body), sha256=hashlib.sha256(raw).hexdigest())
                attempt['http_status'] = response.status_code
                if response.status_code == 429:
                    attempt['retry_after'] = response.headers.get('Retry-After', '')
            except requests.RequestException as error:
                state.pop('http_status', None)
                attempt['error'] = str(error)
            atomic_json(path, state)
            time.sleep(1)
        if state.get('http_status') in {401, 403, 429}:
            atomic_json(FOLDER / 'status.json', {'status': 'source_stopped', 'receipt': str(path)})
            return False
        if state.get('http_status') == 200:
            result = json.loads(gzip.decompress(body.read_bytes()))['response']
            if result['numFound'] > len(result['docs']):
                atomic_json(FOLDER / 'status.json', {'status': 'truncated_batch_held', 'receipt': str(path)})
                return False
    return True


def build_rules() -> dict:
    with (ROOT / 'fields/journal_registry.csv').open() as stream:
        registry = list(csv.DictReader(stream))
    owners: dict = defaultdict(set)
    for row in registry:
        for issn in json.loads(row['issn'] or '[]'):
            owners[issn].add(row['source_id'])
    wanted = set(json.loads((FOLDER / 'scope.json').read_text())['titles'])
    proposals: dict = defaultdict(list)
    held = []
    for path in sorted(FOLDER.glob('titles_*.json')):
        state = json.loads(path.read_text())
        if state.get('http_status') != 200:
            held.append({'receipt': str(path), 'reason': 'transport_unresolved'})
            continue
        raw = gzip.decompress(path.with_suffix('.body.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest() != state['sha256']:
            raise ValueError('Journal authority receipt checksum differs')
        for pos, record in enumerate(json.loads(raw)['response']['docs']):
            title = record.get('title_s', '')
            if title not in wanted:
                continue
            issns = {record.get('issn_s'), record.get('eissn_s')} - {None, ''}
            matched = set().union(*(owners[i] for i in issns)) if issns else set()
            if record.get('valid_s') != 'VALID' or len(matched) != 1:
                held.append({'title': title, 'docid': record['docid'], 'reason': 'unvalidated_or_ambiguous_issn'})
                continue
            sid = next(iter(matched))
            proposals[title].append({'source_id': sid, 'issns': sorted(issns),
                'receipt': str(path.relative_to(ROOT)), 'raw_receipt': str(path.with_suffix('.body.gz').relative_to(ROOT)),
                'raw_sha256': state['sha256'], 'position': pos, 'docid': record['docid'],
                'url': state['url'], 'basis': 'Exact HAL journal title; VALID authority record; ISSNs uniquely map to current registry. Identity only; no type, date or publisher inference.'})
    rules = {title: values for title, values in proposals.items()
             if len({v['source_id'] for v in values}) == 1}
    result = {'rules': rules, 'held': held, 'scope_titles': len(wanted), 'qualified_papers_added': 0}
    atomic_json(FOLDER / 'rules.json', result)
    atomic_json(FOLDER / 'status.json', {'status': 'authority_review_complete',
        'matched_titles': len(rules), 'scope_titles': len(wanted), 'final_task_closure': False})
    return {'matched_titles': len(rules), 'scope_titles': len(wanted)}


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if collect():
            print(json.dumps(build_rules()), flush=True)


if __name__ == '__main__':
    run()
