"""Reconcile only completed fixed-cohort PMC links, retaining dispatch history."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .cached_secondary_dispatch import archive_pmcid
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run(cohort: Path | None = None) -> dict:
    folder = ROOT / 'supplement/cached_secondary_publication_links'
    cohort = cohort or ROOT / 'supplement/secondary_link_pmc'
    terminal = json.loads((cohort / 'status.json').read_text())
    if terminal['status'] != 'fixed_metadata_and_available_xml_processed':
        raise RuntimeError('Fixed cohort is not terminal')
    scope = json.loads((cohort / 'scope.json').read_text())['jobs']
    db = sqlite3.connect(folder / 'leads.sqlite')
    current = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    db.execute('CREATE TABLE IF NOT EXISTS dispatch_changes(work_id TEXT,url TEXT,before_json TEXT,after_json TEXT,reason TEXT,recorded_at TEXT)')
    rows = db.execute("SELECT d.work_id,d.url,d.status,d.evidence,r.doi FROM dispatch d JOIN reviewed r USING(work_id) WHERE d.status='archive_metadata_not_yet_requested'").fetchall()
    cache = {}; counts = Counter()
    for wid, url, before_status, before_evidence, doi in rows:
        pmc = archive_pmcid(url, urlsplit(url).hostname or '')
        if not pmc or pmc not in scope:
            counts['outside_completed_cohort_retained'] += 1
            continue
        if pmc not in cache:
            base = ROOT / 'supplement/pmc_cloud'
            vp = base / 'version_results' / f'{pmc}.1.json'
            xp = base / 'xml_results' / f'{pmc}.1.json'
            cache[pmc] = (vp, json.loads(vp.read_text()) if vp.exists() else {},
                          xp, json.loads(xp.read_text()) if xp.exists() else {})
        vp, version, xp, xml = cache[pmc]
        if not any(p.get('work_id') == wid and p.get('doi') == doi for p in version.get('paper_links', [])):
            counts['association_not_confirmed_retained'] += 1
            continue
        evidence = {'version_result': str(vp.relative_to(ROOT)), 'version_status': version['status'],
                    'qualification_changed': False, 'final_task_closure': False}
        status = 'existing_archive_metadata_outcome_reused'
        if xml:
            status = 'existing_archive_xml_outcome_reused'
            evidence.update(xml_result=str(xp.relative_to(ROOT)), xml_status=xml['status'])
            parsed = xml.get('parsed_receipt')
            if parsed:
                applied = current.execute('SELECT 1 FROM supplied_publisher_processed WHERE doi=? AND receipt=?', (doi, parsed)).fetchone()
                evidence.update(parsed_receipt=parsed, already_applied=bool(applied))
                if not applied:
                    status = 'existing_parsed_archive_application_review'
        after = json.dumps({'status': status, 'evidence': evidence}, sort_keys=True)
        before = json.dumps({'status': before_status, 'evidence': json.loads(before_evidence)}, sort_keys=True)
        db.execute('INSERT INTO dispatch_changes VALUES(?,?,?,?,?,?)', (wid, url, before, after,
                   'Completed fixed PMC cohort receipt; exact Work and DOI association', datetime.now(timezone.utc).isoformat()))
        db.execute('UPDATE dispatch SET status=?,evidence=? WHERE work_id=? AND url=?', (status, json.dumps(evidence), wid, url))
        counts[status] += 1
    db.commit()
    outcomes = dict(db.execute('SELECT status,count(*) FROM dispatch GROUP BY status'))
    cumulative = db.execute('SELECT count(*) FROM dispatch_changes').fetchone()[0]
    db.close(); current.close()
    result = {'changes': dict(counts), 'cumulative_audited_link_changes': cumulative, 'current_dispatch_outcomes': outcomes,
              'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False}
    atomic_json(folder / 'fixed_cohort_reconciliation.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
