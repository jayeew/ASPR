"""Attach existing URL-specific evidence to cached publication leads without requests."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cached_primary_url_leads'
    db = sqlite3.connect(folder / 'leads.sqlite')
    pending = {r[0] for r in db.execute("SELECT url FROM url_dispatch WHERE status='source_path_review_pending'")}
    db.execute('CREATE TABLE IF NOT EXISTS cached_url_evidence(url TEXT,receipt TEXT,basis TEXT,status TEXT,PRIMARY KEY(url,receipt,basis))')
    source = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    counts = Counter()
    for table, field in [('current_evidence_index', 'url'), ('source_metadata', 'source_url')]:
        cursor = source.execute(f"SELECT json_extract(payload,'$.{field}'),payload FROM {table}")
        for url, payload in cursor:
            if url not in pending:
                continue
            item = json.loads(payload)
            receipt = item.get('receipt') if table == 'current_evidence_index' else item.get('evidence_receipt')
            if not receipt:
                continue
            if table == 'current_evidence_index':
                receipt = 'fields/qualification/' + receipt
            state = item.get('identity', 'cached_content_identity_review_required')
            db.execute('INSERT OR IGNORE INTO cached_url_evidence VALUES(?,?,?,?)',(url,receipt,table,state))
            counts[table] += 1
    source.close();db.commit()
    parsed_folder = ROOT / 'fields/qualification/supplement_parsed'
    for url in sorted(pending):
        path = parsed_folder / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if not path.exists():
            continue
        item = json.loads(path.read_text())
        if item.get('url') != url:
            continue
        db.execute('INSERT OR IGNORE INTO cached_url_evidence VALUES(?,?,?,?)',
                   (url,str(path.relative_to(ROOT)),'exact_url_acquisition_receipt',item.get('status','missing_status')))
        counts['exact_acquisition_receipts'] += 1
    db.commit()
    db.execute("UPDATE url_dispatch SET status='existing_exact_url_evidence_review_required',reason='Exact URL has cached evidence or acquisition result; inspect retained status, do not repeat unchanged response',evidence='supplement/cached_primary_url_leads/leads.sqlite#cached_url_evidence' WHERE status='source_path_review_pending' AND EXISTS(SELECT 1 FROM cached_url_evidence e WHERE e.url=url_dispatch.url)")
    db.commit()
    result = {'status':'exact_url_cache_match_complete_not_task_closure','network_requests':0,
        'input_urls':len(pending),'matches':dict(counts),
        'url_status_counts':dict(db.execute('SELECT status,count(*) FROM url_dispatch GROUP BY status')),
        'remaining_hosts':list(db.execute("SELECT l.host,count(DISTINCT l.work_id) n FROM leads l JOIN url_dispatch d USING(url) WHERE d.status='source_path_review_pending' GROUP BY l.host ORDER BY n DESC LIMIT 40")),
        'limitations':['Exact URL cache match alone does not establish paper identity or usable missing fields.',
                      'A different URL on a previously restricted host is not permission to request it.',
                      'Unmatched URLs still need source applicability review before requests.']}
    db.close();atomic_json(folder / 'evidence_match_summary.json',result);print(json.dumps(result),flush=True)


if __name__ == '__main__':
    run()
