"""Apply SciPost native identity/date evidence without inventing article types."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import date

from .publisher_delta_apply import apply_available
from .scipost_bulk_collect import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def stage(item: dict, receipt: str) -> dict | None:
    record = item['record']
    label = record.get('in_volume') or record.get('in_issue') or ''
    journal = re.split(r' Vol\.| issue ', label, maxsplit=1)[0].strip()
    if not journal or not record.get('doi') or not record.get('title'):
        return None
    published = record.get('publication_date', '')
    try:
        date.fromisoformat(published)
    except (ValueError, TypeError):
        published = ''
    # Only this journal's online-release workflow has been checked against authoring policy.
    online = published if journal == 'SciPost Physics' else ''
    return {'source': 'publisher', 'route': 'scipost_api', 'doi': record['doi'].lower(),
            'url': 'https://scipost.org' + record['url'],
            'raw_receipt': os.path.relpath(ROOT / receipt, ROOT / 'fields/qualification'),
            'type_limitation': 'Public API has no explicit article type; do not infer from journal or abstract.',
            'online_date_policy': 'supplement/scipost_bulk/authoring_receipt.json' if online else '',
            'parsed': {'doi': [record['doi'].lower()], 'title': [record['title']],
                       'issns': [], 'journal_titles': [journal], 'article_types': [],
                       'online_date': online, 'date_precision': 'day' if online else '',
                       'abstract': record.get('abstract', ''),
                       'publication_dates': [published] if published else []}}


def record_attempts() -> None:
    """Record each scoped DOI's acquisition outcome without closing unresolved tasks."""
    matches = json.loads((FOLDER / 'matches.json').read_text())
    scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for paper in scope:
            item = matches.get(paper['doi'].lower())
            receipt = item['receipt'] if item else 'supplement/scipost_bulk/status.json'
            status = 'batch_doi_found_identity_and_type_review_required' if item else 'absent_from_public_api_listing'
            for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (paper['doi'],)).fetchall():
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (task_id, 'scipost_public_api', receipt)).fetchone():
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, 'scipost_public_api', status, receipt, json.dumps({'final_task_closure': False, 'no_explicit_article_type': True})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
        db.commit()
        db.close()


def run() -> None:
    status = json.loads((FOLDER / 'status.json').read_text())
    if status['status'] != 'acquisition_finished_review_required':
        raise RuntimeError('Bulk acquisition is still active; apply after its fixed pagination finishes')
    (FOLDER / 'staged').mkdir(exist_ok=True)
    matches = json.loads((FOLDER / 'matches.json').read_text())
    jobs = []
    held = []
    for doi, item in matches.items():
        record = stage(item, item['receipt'])
        if record is None:
            held.append({'doi': doi, 'reason': 'Native API lacks explicit journal identity or title', 'receipt': item['receipt']})
            continue
        path = FOLDER / 'staged' / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
        atomic_json(path, record)
        jobs.append((doi, str(path.relative_to(ROOT))))
    atomic_json(FOLDER / 'application_scope.json', {'jobs': jobs, 'held': held, 'no_inferred_research_type': True})
    result = apply_available(supplied_jobs=jobs)
    atomic_json(FOLDER / 'application_result.json', result)
    record_attempts()


if __name__ == '__main__':
    run()
