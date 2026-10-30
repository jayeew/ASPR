"""Link completed fixed OAI source attempts to each existing candidate task."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter
from pathlib import Path

from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json


def run(folder: Path) -> dict:
    terminal = json.loads((folder / 'oai_status.json').read_text())
    if terminal['status'] == 'running': raise ValueError('Discovery scope not finished')
    stats = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX); db = connect()
        for line in (folder / 'oai_results.jsonl').read_text().splitlines():
            source = json.loads(line)
            path = folder / 'harvested' / (hashlib.sha256(source['url'].encode()).hexdigest() + '.json')
            harvest = json.loads(path.read_text()) if path.exists() else None
            if source['status'] == 'public_formats_available' and harvest is None:
                stats['harvest_still_pending'] += len(source['candidates']); continue
            receipt = str(path.relative_to(ROOT)) if harvest else source.get('raw_receipt') or source.get('restriction_receipt') or str((folder / 'oai_results.jsonl').relative_to(ROOT))
            for candidate in source['candidates']:
                wid = candidate['work_id']
                task = db.execute("SELECT DISTINCT t.task_id FROM tasks t JOIN task_papers l USING(task_id) WHERE t.kind='qualification' AND l.work_id=?", (wid,)).fetchall()
                if len(task) != 1: stats['task_identity_unresolved'] += 1; continue
                tid = task[0][0]
                previous = db.execute("SELECT 1 FROM attempts WHERE task_id=? AND source='publisher_oai_fixed_scope' AND receipt=?", (tid, receipt)).fetchone()
                if previous: stats['already_linked'] += 1; continue
                record = harvest.get('matched', {}).get(wid) if harvest else None
                status = ('matched_metadata_requires_type_date_review' if record and record['title_identity'] == 'exact' else
                          'native_title_conflict' if record else
                          'fixed_endpoint_no_matching_record:' + harvest['status'] if harvest else source['status'])
                detail = {'work_id': wid, 'endpoint': source['url'], 'landing_page_url': candidate['landing_page_url'],
                          'discovery_status': source['status'], 'harvest_status': harvest['status'] if harvest else None,
                          'candidate_individually_requested': False,
                          'method': 'Candidate-bound source batch result; no claim of individual page retrieval',
                          'final_task_closure': False, 'qualification_unchanged': True,
                          'next': 'Use retained matching metadata if sufficient; unconfirmed type/date or inaccessible native page remains unresolved; do not repeat unchanged source request'}
                attempt(db, tid, 'publisher_oai_fixed_scope', status, receipt, detail); stats[status] += 1
            db.commit()
        db.close()
    result = {'status': 'source_attempts_linked_not_final_candidate_closure', **stats}
    atomic_json(folder / 'disposition_ledger_status.json', result)
    print(json.dumps(result), flush=True)
    return result
