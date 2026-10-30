"""Discover explicit indexed study records in a fixed public date window."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .fig8b_qualification_decide import NLM_EXPLICIT_STUDY_TYPES, key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json

F = ROOT / 'supplement/target_500k/nlm_public_type_search'
URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
STATES = {'pending_no_primary_type_date', 'pending_primary_type', 'pending_primary_online_date'}


def fetch(session: requests.Session, params: dict) -> tuple[dict, Path]:
    token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = F / 'raw' / (token + '.json.gz')
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes())), path
    record = {'url': URL, 'params': params, 'at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
    for attempt in range(3):
        try:
            response = session.get(URL, params=params, timeout=(10, 60))
            record['attempts'].append({'http_status': response.status_code})
            record['http_status'] = response.status_code
            if response.ok:
                record['response'] = response.json()
                break
            if response.status_code not in {500, 502, 503, 504}:
                break
        except (requests.RequestException, ValueError) as error:
            record['attempts'].append({'error': type(error).__name__})
        if attempt < 2:
            time.sleep(2 ** attempt)
    atomic_bytes(path, gzip.compress(json.dumps(record).encode(), mtime=0))
    time.sleep(0.5)
    return record, path


def run() -> None:
    (F / 'raw').mkdir(parents=True, exist_ok=True)
    db = connect()
    targets = {r[1]: dict(zip(('work_id', 'doi', 'journal', 'current_status'), r))
               for r in db.execute('select work_id,doi,journal,status from papers') if r[3] in STATES and r[1]}
    db.close()
    query = json.loads(gzip.decompress((F / 'first_page.json.gz').read_bytes()))['params']['query']
    session = requests.Session()
    session.trust_env = False
    cursor = '*'
    matches: set[str] = set()
    seen: set[str] = set()
    counts: set[int] = set()
    pages = 0
    while True:
        params = dict(query=query, format='json', resultType='lite', pageSize=1000, cursorMark=cursor)
        receipt, _ = fetch(session, params)
        body = receipt.get('response', {})
        if receipt.get('http_status') != 200 or 'resultList' not in body:
            atomic_json(F / 'status.json', {'status': 'discovery_source_stopped', 'pages': pages, 'receipt': str(_), 'qualification_changes': 0})
            return
        counts.add(body['hitCount'])
        rows = body['resultList']['result']
        for item in rows:
            seen.add(item.get('source', '') + ':' + item['id'])
            doi = (item.get('doi') or '').lower()
            if doi in targets:
                matches.add(doi)
        pages += 1
        state = {'status': 'discovering', 'pages': pages, 'seen_records': len(seen), 'hit_counts': sorted(counts), 'pending_target_matches': len(matches), 'qualification_changes': 0}
        atomic_json(F / 'status.json', state)
        if pages % 10 == 0:
            print(json.dumps(state), flush=True)
        following = body.get('nextCursorMark')
        if not rows or not following or following == cursor or len(seen) >= max(counts):
            break
        cursor = following
    ordered = sorted(matches)
    records = {}
    failures = []
    for start in range(0, len(ordered), 60):
        subset = ordered[start:start + 60]
        params = dict(query='SRC:MED AND (' + ' OR '.join(f'DOI:"{doi}"' for doi in subset) + ')', format='json', resultType='core', pageSize=1000)
        receipt, path = fetch(session, params)
        body = receipt.get('response', {})
        rows = body.get('resultList', {}).get('result', [])
        if receipt.get('http_status') != 200 or len(rows) != body.get('hitCount'):
            failures.append({'dois': subset, 'receipt': str(path), 'reason': 'failed_or_incomplete_response'})
            continue
        for item in rows:
            doi = (item.get('doi') or '').lower()
            types = item.get('pubTypeList', {}).get('pubType', [])
            specific = [t for t in types if key(t) in NLM_EXPLICIT_STUDY_TYPES]
            if doi not in targets or not specific or item.get('source') != 'MED':
                continue
            row = records.setdefault(doi, {**targets[doi], 'types': [], 'secondary_evidence': []})
            row['types'] = sorted(set(row['types']) | set(specific))
            row['secondary_evidence'].append({'types': types, 'receipt': '../../' + str(path.relative_to(ROOT)), 'title': item.get('title', '')})
    atomic_json(F / 'inventory.json', {'records': list(records.values()), 'papers': len(records), 'failed_core_batches': failures})
    state.update(status='discovery_complete_review_required', complete_pagination=len(counts) == 1 and len(seen) == next(iter(counts)), specific_study_candidates=len(records), core_failed_batches=len(failures))
    atomic_json(F / 'status.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    run()
