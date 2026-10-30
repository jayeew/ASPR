"""Reuse exact publisher URL/DOI outcomes; preserve different-format alternatives."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def canonical(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in {'fbclid', 'gclid'}]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip('/'), urlencode(query), ''))


def cached_page(current: sqlite3.Connection, wid: str, url: str) -> dict | None:
    paper = current.execute('SELECT doi FROM papers WHERE work_id=?', (wid,)).fetchone()
    if not paper:
        return None
    for payload, in current.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,)):
        evidence = json.loads(payload)
        if (evidence.get('identity') != 'matched' or evidence.get('doi') != paper[0]
                or evidence.get('family') not in {'publisher_springer', 'publisher_nature', 'publisher_frontiers'}
                or canonical(evidence.get('url', '')) != canonical(url)):
            continue
        receipt = evidence.get('receipt', '')
        if receipt.endswith('.html.gz') and (ROOT / 'fields/qualification' / receipt).is_file():
            return {'reused_primary_evidence_id': evidence['id'],
                    'reused_publisher_receipt': str((ROOT / 'fields/qualification' / receipt).relative_to(ROOT)),
                    'reused_publisher_status': 'matched_cached_publisher_page', 'existing_application': True}
    return None


def run() -> None:
    current = connect()
    jobs = {r[0]: r[1:] for r in current.execute("""SELECT p.work_id,p.doi,j.route,j.status,j.receipt,j.applied
        FROM publisher_remaining_jobs j JOIN papers p USING(doi)
        WHERE j.route IN ('springer','nature','frontiers') AND j.status<>'pending'""")}
    db = sqlite3.connect(FOLDER / 'locations.sqlite', timeout=120)
    rows = db.execute("""SELECT work_id,url,host,evidence FROM links
        WHERE status='new_location_source_review_required'
        AND host IN ('link.springer.com','www.nature.com','www.frontiersin.org')""").fetchall()
    counts = Counter()
    updates = []
    for wid, url, host, payload in rows:
        job = jobs.get(wid)
        if not job:
            reused = cached_page(current, wid, url)
            if reused:
                evidence = json.loads(payload)
                evidence.update(reused, newly_visited=False, qualification_confirmation_implied=False,
                                final_task_closure=False, reuse_basis='Matched current DOI and exact publisher HTML page URL in existing primary evidence index')
                updates.append((wid, url, json.dumps(evidence)))
                counts['primary_index_page_outcome_selected'] += 1
                continue
            counts['no_completed_matching_publisher_job'] += 1
            continue
        doi, route, outcome, receipt, applied = job
        path = ROOT / receipt
        if not receipt or not path.is_file():
            counts['receipt_unavailable'] += 1
            continue
        record = json.loads(path.read_text())
        if record.get('doi', '').lower() != doi or canonical(record.get('url', '')) != canonical(url):
            counts['different_url_or_identity_alternative_retained'] += 1
            continue
        evidence = json.loads(payload)
        evidence.update(reused_publisher_receipt=receipt, reused_publisher_status=outcome,
                        existing_application=bool(applied), newly_visited=False,
                        qualification_confirmation_implied=False, final_task_closure=False,
                        reuse_basis='Exact current DOI and same URL after fragment/trailing slash/tracking-only normalization; PDF and other content paths remain distinct')
        updates.append((wid, url, json.dumps(evidence)))
    current.close()
    now = datetime.now(timezone.utc).isoformat()
    db.execute('CREATE TABLE IF NOT EXISTS routing_changes(work_id TEXT,url TEXT,previous_status TEXT,next_status TEXT,reason TEXT,recorded_at TEXT,PRIMARY KEY(work_id,url,next_status))')
    for offset in range(0, len(updates), 250):
        db.execute('BEGIN IMMEDIATE')
        for wid, url, evidence in updates[offset:offset + 250]:
            changed = db.execute("UPDATE links SET status='existing_exact_publisher_outcome_reused',evidence=? WHERE work_id=? AND url=? AND status='new_location_source_review_required'", (evidence, wid, url)).rowcount
            if changed:
                db.execute('INSERT OR IGNORE INTO routing_changes VALUES(?,?,?,?,?,?)',
                    (wid, url, 'new_location_source_review_required', 'existing_exact_publisher_outcome_reused',
                     'Existing exact URL and current DOI acquisition receipt reused; unresolved metadata preserved', now))
                counts['exact_publisher_outcomes_reused'] += 1
        db.commit()
    db.close()
    atomic_json(FOLDER / 'publisher_reuse_status.json', {'status': 'existing_exact_publisher_routes_reused',
        'reviewed_links': len(rows), 'outcomes': dict(counts), 'network_requests': 0,
        'qualification_changes': 0, 'final_task_closure': False, 'checked_at': now})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
