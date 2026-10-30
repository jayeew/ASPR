"""Carry a retained exact-host restriction into pending URL queues, not paper closure."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cached_primary_url_leads'
    receipt = 'supplement/elsevier_redirects/target_check.json'
    response = json.loads((ROOT / receipt).read_text())['response']
    assert response['http_status'] == 403
    host = urlsplit(response['url']).hostname
    assert host == 'www.sciencedirect.com'
    evidence = {'receipt': receipt, 'attempted_url': response['url'], 'http_status': 403,
                'basis': 'Retained direct exact-host restriction; do not repeat on another URL. '
                         'These pending URLs were not individually visited; alternative paths remain open.',
                'content_fetched': False, 'final_task_closure': False}
    now = datetime.now(timezone.utc).isoformat()
    counts = {}
    with (folder / 'source_hold_reconcile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry_path = folder / 'host_hold_application.json'
        registry = json.loads(registry_path.read_text())
        old = registry['hosts'].get(host)
        if old != evidence:
            with (folder / 'host_registry_changes.jsonl').open('a') as stream:
                stream.write(json.dumps({'host': host, 'before': old, 'after': evidence, 'at': now}) + '\n')
            registry['hosts'][host] = evidence
            atomic_json(registry_path, registry)
        db = sqlite3.connect(folder / 'leads.sqlite', timeout=60)
        rows = db.execute("SELECT DISTINCT d.url,d.status,d.reason,d.evidence FROM url_dispatch d "
                          "JOIN leads l USING(url) WHERE l.host=? AND d.status='source_path_review_pending'", (host,)).fetchall()
        for url, *before in rows:
            after = ['existing_host_hold_alternatives_pending', evidence['basis'], json.dumps(evidence)]
            db.execute('INSERT INTO url_dispatch_changes VALUES(?,?,?,?,?)',
                       (url, json.dumps(before), json.dumps(after), 'retained_exact_host_restriction', now))
            db.execute('UPDATE url_dispatch SET status=?,reason=?,evidence=? WHERE url=?', (*after, url))
        db.commit()
        summary_path = folder / 'dispatch_summary.json'
        summary = json.loads(summary_path.read_text())
        summary['url_status_counts'] = dict(db.execute('SELECT status,count(*) FROM url_dispatch GROUP BY status'))
        summary['updated_at'] = now
        atomic_json(summary_path, summary)
        db.close()
        counts['primary_urls'] = len(rows)
        for name, table, pending in [
                ('cached_secondary_publication_links/leads.sqlite', 'dispatch', 'source_path_review_required'),
                ('snapshot_location_recovery/locations.sqlite', 'links', 'new_location_source_review_required')]:
            db = sqlite3.connect(ROOT / 'supplement' / name, timeout=60)
            db.execute('CREATE TABLE IF NOT EXISTS retained_host_changes(work_id TEXT,url TEXT,before_json TEXT,after_json TEXT,at TEXT)')
            query = ("SELECT d.work_id,d.url,d.status,d.evidence FROM dispatch d JOIN leads l USING(work_id,url) "
                     'WHERE l.host=? AND d.status=?' if table == 'dispatch' else
                     'SELECT work_id,url,status,evidence FROM links WHERE host=? AND status=?')
            rows = db.execute(query, (host, pending)).fetchall()
            for wid, url, status, stored in rows:
                merged = {**json.loads(stored), 'retained_host_hold': evidence}
                after = ['existing_host_hold_alternatives_pending', json.dumps(merged)]
                db.execute('INSERT INTO retained_host_changes VALUES(?,?,?,?,?)',
                           (wid, url, json.dumps([status, stored]), json.dumps(after), now))
                db.execute(f'UPDATE {table} SET status=?,evidence=? WHERE work_id=? AND url=?', (*after, wid, url))
            db.commit()
            db.close()
            counts[name] = len(rows)
    atomic_json(folder / 'sciencedirect_hold_reconciliation.json', {'host': host, 'changed': counts,
        'evidence': evidence, 'network_requests': 0, 'qualification_changes': 0,
        'final_task_closure': False, 'at': now})
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    run()
