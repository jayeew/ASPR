"""Reuse exact cached PMC work/DOI associations without repeating acquisition."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache

from .pmc_cloud_inventory import FOLDER as CLOUD
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


@lru_cache(maxsize=2048)
def archive_outcomes(pmcid: str) -> tuple[dict, dict]:
    version = CLOUD / 'version_results' / (pmcid + '.1.json')
    xml = CLOUD / 'xml_results' / (pmcid + '.1.json')
    return (json.loads(version.read_text()) if version.exists() else {},
            json.loads(xml.read_text()) if xml.exists() else {})


def run() -> None:
    db = sqlite3.connect(FOLDER / 'locations.sqlite', timeout=120)
    db.execute('CREATE TABLE IF NOT EXISTS routing_changes(work_id TEXT,url TEXT,previous_status TEXT,next_status TEXT,reason TEXT,recorded_at TEXT,PRIMARY KEY(work_id,url,next_status))')
    rows = db.execute("SELECT work_id,url,evidence FROM links WHERE status='existing_archive_association_review_required'").fetchall()
    current = connect(); counts = Counter(); now = datetime.now(timezone.utc).isoformat()
    for wid, url, payload in rows:
        evidence = json.loads(payload); pmcid = evidence['pmcid']
        version, xml = archive_outcomes(pmcid)
        paper = current.execute('SELECT doi FROM papers WHERE work_id=?', (wid,)).fetchone()
        associated = bool(paper and any(p.get('work_id') == wid and p.get('doi') == paper[0]
                                        for p in version.get('paper_links', [])))
        if not associated:
            counts['association_still_requires_review'] += 1
            continue
        evidence.update(reused_version_result=str((CLOUD / 'version_results' / (pmcid + '.1.json')).relative_to(ROOT)),
            reused_version_status=version.get('status'), association_basis='Exact cached input Work ID and current DOI pair',
            source_identity_confirmation_implied=False, final_task_closure=False)
        if xml:
            evidence.update(reused_xml_result=str((CLOUD / 'xml_results' / (pmcid + '.1.json')).relative_to(ROOT)),
                            reused_xml_status=xml.get('status'))
        target = 'existing_archive_outcome_reused'
        db.execute('INSERT OR IGNORE INTO routing_changes VALUES(?,?,?,?,?,?)',
                   (wid, url, 'existing_archive_association_review_required', target,
                    'Exact cached work/DOI association; retain original acquisition outcome, no new retrieval or qualification claim', now))
        db.execute("UPDATE links SET status=?,evidence=? WHERE work_id=? AND url=? AND status='existing_archive_association_review_required'",
                   (target, json.dumps(evidence), wid, url))
        counts[target] += 1
        if counts[target] % 250 == 0:
            db.commit()  # Let the live location dispatcher commit between batches.
    db.commit(); db.close(); current.close()
    atomic_json(FOLDER / 'archive_reuse_status.json', {'status': 'cached_archive_associations_reused',
        'reviewed_links': len(rows), 'outcomes': dict(counts), 'network_requests': 0,
        'qualification_changes': 0, 'final_task_closure': False, 'updated_at': now})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
