"""Remove duplicate DOI-resolver work from recovered locations without requests."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit

from .snapshot_location_recovery import FOLDER
from .supplement_queue import connect
from .supplement_sources import atomic_json


def run() -> None:
    db = sqlite3.connect(FOLDER / 'locations.sqlite', timeout=120)
    current = connect()
    db.execute('CREATE TABLE IF NOT EXISTS routing_changes(work_id TEXT,url TEXT,previous_status TEXT,next_status TEXT,reason TEXT,recorded_at TEXT,PRIMARY KEY(work_id,url,next_status))')
    rows = db.execute("SELECT work_id,url,status FROM links WHERE status='new_location_source_review_required' AND host IN ('doi.org','dx.doi.org','www.doi.org')").fetchall()
    counts = Counter(); updated = datetime.now(timezone.utc).isoformat()
    for wid, url, status in rows:
        paper = current.execute('SELECT doi FROM papers WHERE work_id=?', (wid,)).fetchone()
        matches = bool(paper and unquote(urlsplit(url).path).lstrip('/').casefold() == paper[0])
        target = 'doi_resolver_existing_route' if matches else 'resolver_doi_identity_review_required'
        db.execute('INSERT OR IGNORE INTO routing_changes VALUES(?,?,?,?,?,?)',
                   (wid, url, status, target, 'Exact decoded DOI resolver path comparison with current work DOI; not a fetched publication source or task closure', updated))
        db.execute('UPDATE links SET status=? WHERE work_id=? AND url=? AND status=?', (target, wid, url, status))
        counts[target] += 1
    db.commit(); current.close(); db.close()
    atomic_json(FOLDER / 'resolver_route_status.json', {'status': 'resolver_routes_reused', 'outcomes': dict(counts),
        'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False, 'updated_at': updated})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
