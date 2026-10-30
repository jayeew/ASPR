"""Fetch fixed publisher XML links already deposited in the SCOAP3 cache."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import sqlite3
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import httpx
from lxml import etree

from .native_xml_document import document
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

FOLDER = ROOT / 'supplement/scoap3_native'
HOST = 'https://scoap3-prod-backend.s3.cern.ch/'


def scope() -> list[dict]:
    path = FOLDER / 'scope.json'
    if path.exists():
        return json.loads(path.read_text())['jobs']
    db = sqlite3.connect(f'file:{ROOT}/supplement/scoap3/cache_index.sqlite?mode=ro', uri=True)
    db.execute(f"ATTACH DATABASE 'file:{ROOT}/supplement/tasks.sqlite?mode=ro' AS q")
    jobs = {}
    for payload, receipt, wid in db.execute("SELECT r.payload,r.receipt,p.work_id FROM records r JOIN q.papers p ON p.doi=r.doi WHERE p.status LIKE 'pending%'"):
        record = json.loads(payload)
        for file in record.get('related_files', []):
            url = file['file']
            if '.xml' not in url or not url.startswith(HOST):
                continue
            item = jobs.setdefault(url, {'url': url, 'doi': record['doi'].lower(),
                                        'discovery_receipt': receipt, 'work_ids': []})
            if wid not in item['work_ids']:
                item['work_ids'].append(wid)
    db.close()
    result = sorted(jobs.values(), key=lambda r: (r['doi'], r['url']))
    atomic_json(path, {'created_at': datetime.now(timezone.utc).isoformat(),
                      'scope': 'Existing pending candidates with cached publisher-deposited XML links; no new cohort',
                      'papers': len({wid for j in result for wid in j['work_ids']}),
                      'urls': len(result), 'jobs': result})
    return result


def fetch(job: dict) -> dict:
    token = hashlib.sha256(job['url'].encode()).hexdigest()
    path = FOLDER / 'receipts' / (token + '.json')
    if path.exists():
        return json.loads(path.read_text())
    result = {**job, 'attempts': [], 'network_route': 'direct_trust_env_false',
              'retrieved_at': datetime.now(timezone.utc).isoformat()}
    with httpx.Client(trust_env=False, timeout=35, follow_redirects=False) as client:
        for attempt in range(3):
            paced(job['url'])
            try:
                response = client.get(job['url'])
                result['http_status'] = response.status_code
                result['attempts'].append({'http_status': response.status_code})
                if response.status_code not in {500, 502, 503, 504}:
                    raw = response.content
                    raw_path = FOLDER / 'raw' / (token + '.xml.gz')
                    atomic_bytes(raw_path, gzip.compress(raw, mtime=0))
                    result.update(raw_receipt=str(raw_path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())
                    if response.status_code == 200:
                        try:
                            tree, normalization = document(raw)
                            result.update(status='xml_cached_identity_review_required', root_tag=tree.tag, xml_normalization=normalization)
                        except etree.XMLSyntaxError:
                            result['status'] = 'invalid_xml'
                    else:
                        result['status'] = 'http_unresolved'
                    break
            except httpx.HTTPError as exc:
                result['attempts'].append({'error': type(exc).__name__, 'detail': str(exc)})
            if attempt < 2:
                time.sleep(2 ** attempt)
    result.setdefault('status', 'transport_or_server_attempts_exhausted')
    atomic_json(path, result)
    return result


def run() -> None:
    for directory in [FOLDER, FOLDER / 'receipts', FOLDER / 'raw']:
        directory.mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = scope()
        existing = [json.loads(p.read_text()) for p in (FOLDER / 'receipts').glob('*.json')]
        denied_dois = {r['doi'] for r in existing if r.get('http_status') == 403}
        restricted_prefix = HOST + 'media/files/'
        if any(r.get('http_status') in {401, 429} or (r.get('http_status') == 403 and not r['url'].startswith(restricted_prefix)) for r in existing):
            atomic_json(FOLDER / 'status.json', {'status': 'source_restriction_retained', 'no_automatic_retry': True})
            return
        known = {r['url'] for r in existing}
        restricted = [j for j in jobs if j['url'] not in known and (j['url'].startswith(restricted_prefix) or j['doi'] in denied_dois)] if denied_dois else []
        restricted_urls = {j['url'] for j in restricted}
        atomic_json(FOLDER / 'restricted_unattempted.json', {'jobs': restricted, 'reason': 'Reviewed AccessDenied path held; no alternate request for denied DOI', 'final_task_closure': False})
        pending = iter(j for j in jobs if j['url'] not in known and j['url'] not in restricted_urls)
        counts = Counter(r['status'] for r in existing); stop = ''
        with ThreadPoolExecutor(max_workers=8) as pool:
            active = {}
            def submit() -> None:
                job = next(pending, None)
                if job:active[pool.submit(fetch, job)] = job
            for _ in range(8):submit()
            while active:
                ready, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    active.pop(future); result = future.result(); counts[result['status']] += 1
                    if result.get('http_status') in {401, 403, 429}:
                        stop = 'permission_or_rate_restriction_no_further_dispatch'
                    atomic_json(FOLDER / 'status.json', {'status': 'draining_inflight' if stop else 'running',
                        'stop_reason': stop, 'fixed_urls': len(jobs), 'completed': sum(counts.values()),
                        'outcomes': dict(counts), 'workers': 8, 'request_start_interval_seconds': 1.05,
                        'qualification_not_implied': True})
                    if not stop:submit()
        atomic_json(FOLDER / 'status.json', {'status': 'source_stopped_unattempted_retained' if stop else ('fixed_public_xml_queue_processed_restricted_path_retained' if restricted else 'fixed_xml_queue_acquired_review_required'),
                    'stop_reason': stop, 'fixed_urls': len(jobs), 'completed': sum(counts.values()), 'outcomes': dict(counts)})


if __name__ == '__main__':run()
