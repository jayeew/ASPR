"""Resolve stale evidence-ready flags using fixed cached deposit receipts."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import sqlite3
from collections import Counter

from .fig8b_qualification_build import match, plain
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/crossref_missing_date_review'


def prepare() -> list[dict]:
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / 'review.json'
    if path.exists():
        return json.loads(path.read_text())['records']
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    rows = db.execute("SELECT t.task_id,t.entity_key,t.receipt,p.work_id,p.title,p.journal,p.source_id "
        "FROM tasks t JOIN task_papers l USING(task_id) JOIN papers p ON p.work_id=l.work_id "
        "WHERE t.kind='qualification' AND t.active=1 AND t.status='evidence_ready'").fetchall()
    db.close()
    with (ROOT / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: r for r in csv.DictReader(stream)}
    cache, records = {}, []
    for tid, doi, receipt, wid, title, journal, sid in rows:
        if not receipt.startswith('fields/crossref_raw/'):
            continue
        if receipt not in cache:
            raw = gzip.decompress((ROOT / receipt).read_bytes())
            cache[receipt] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
        data, digest = cache[receipt]
        items = [(i, r) for i, r in enumerate(data['response']['message']['items'])
                 if r.get('DOI', '').lower() == doi.lower()]
        if len(items) != 1:
            continue
        position, item = items[0]
        if item.get('published-online'):
            continue
        evidence = {'family': 'crossref_publisher_deposit',
            'title': plain((item.get('title') or [''])[0]), 'kind': item.get('type', ''),
            'issns': item.get('ISSN', []), 'journal_titles': item.get('container-title', [])}
        checked = match({'work_id': wid, 'title': title, 'journal_name': journal, 'source_id': sid},
                        evidence, journals[sid])
        records.append({'task_id': tid, 'doi': doi, 'work_id': wid,
            'raw_receipt': receipt, 'raw_sha256': digest,
            'locator': f'response.message.items[{position}]', 'source_url': data['url'],
            'identity': checked['identity'], 'explicit_online_date': None,
            'before_status': 'evidence_ready', 'status': 'matched_deposit_no_online_date'
                if checked['identity'] == 'matched' else 'deposit_identity_conflict',
            'reason': 'Cached deposit contains no explicit online date; do not repeat same response. '
                'Generic article type is not original-research confirmation. Other source paths remain open.',
            'qualification_changed': False, 'final_task_closure': False})
    atomic_json(path, {'records': records, 'raw_responses_read': len(cache),
        'network_requests': 0, 'outcomes': dict(Counter(r['status'] for r in records))})
    return records


def run() -> dict:
    records = prepare()
    receipt = str((FOLDER / 'review.json').relative_to(ROOT))
    atomic_json(FOLDER / 'status.json', {'status': 'waiting_for_writer', 'records': len(records)})
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for raw_path, digest in {(r['raw_receipt'], r['raw_sha256']) for r in records}:
            if hashlib.sha256(gzip.decompress((ROOT / raw_path).read_bytes())).hexdigest() != digest:
                raise ValueError('Reviewed deposit cache changed')
        db = connect()
        count = 0
        for r in records:
            if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (r['task_id'], 'cached_deposit_missing_date_review', receipt)).fetchone():
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                    (r['task_id'], 'cached_deposit_missing_date_review', r['status'], receipt, json.dumps(r)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (r['task_id'],))
            count += db.execute('UPDATE tasks SET status=?,reason=?,receipt=? WHERE task_id=? '
                'AND status=? AND active=1', (r['status'], r['reason'], receipt, r['task_id'],
                r['before_status'])).rowcount
        db.commit()
        db.close()
    result = {'status': 'source_review_recorded_qualification_still_unresolved',
              'reviewed': len(records), 'task_statuses_updated': count,
              'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'status.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
