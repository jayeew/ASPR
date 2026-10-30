"""Reuse exact per-paper primary URL evidence for remaining alternate-link tasks."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def url_key(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ''))


def run() -> None:
    folder = ROOT / 'supplement/cached_secondary_publication_links'
    links = sqlite3.connect(folder / 'leads.sqlite')
    current = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    rows = links.execute("SELECT d.work_id,d.url,d.status,d.evidence,r.doi FROM dispatch d JOIN reviewed r USING(work_id) WHERE d.status='source_path_review_required'").fetchall()
    cache = {}; counts = Counter(); reviews = []
    for wid, url, old_status, old_evidence, doi in rows:
        if wid not in cache:
            cache[wid] = [json.loads(r[0]) for r in current.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,))]
        evidence = [e for e in cache[wid] if e.get('doi') == doi and e.get('url') and url_key(e['url']) == url_key(url)]
        usable = [e for e in evidence if e.get('identity') == 'matched' and e.get('receipt')
                  and (ROOT / 'fields/qualification' / e['receipt']).is_file()]
        record = {'work_id': wid, 'doi': doi, 'url': url, 'qualification_changed': False, 'final_task_closure': False,
                  'evidence': [{'id': e['id'], 'identity': e.get('identity'), 'receipt': e.get('receipt'), 'family': e.get('family')} for e in evidence]}
        status = 'existing_primary_evidence_path_reused' if usable else 'same_url_identity_or_receipt_unresolved' if evidence else 'no_exact_primary_url_evidence'
        record['status'] = status; reviews.append(record); counts[status] += 1
        if not usable: continue
        before = json.dumps({'status': old_status, 'evidence': json.loads(old_evidence)})
        after = json.dumps({'status': status, 'evidence': record})
        links.execute('INSERT INTO dispatch_changes VALUES(?,?,?,?,?,?)', (wid,url,before,after,
                      'Same Work, DOI and exact primary source URL; matched identity and existing raw receipt; reuse outcome without new request',datetime.now(timezone.utc).isoformat()))
        links.execute('UPDATE dispatch SET status=?,evidence=? WHERE work_id=? AND url=?', (status,json.dumps(record),wid,url))
    atomic_json(folder / 'primary_cache_review.json', {'records': reviews, 'outcomes': dict(counts), 'network_requests': 0, 'qualification_changes': 0})
    links.commit(); links.close(); current.close()
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
