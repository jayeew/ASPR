"""Register per-DOI applicability of retained batch evidence, without closing tasks."""
from __future__ import annotations

import json

from .pensoft_bulk import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    records = []
    for path in (FOLDER / 'matched').glob('*.json'):
        r = json.loads(path.read_text())
        records.append((r['doi'], 'official_pensoft_mods', 'metadata_located_identity_type_date_checks_separate', path))
    for path in (FOLDER / 'missing_identifiers').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('doi') and r.get('receipt'):
            records.append((r['doi'], 'official_pensoft_oai_exact_identifier', r['status'], path))
    for path in (FOLDER / 'transferred_journals/rrpharm_matches').glob('*.json'):
        r = json.loads(path.read_text())
        records.append((r['doi'], 'official_rrpharmacology_oai', 'metadata_located_after_platform_transfer', path))
    db = connect()
    db.execute('PRAGMA busy_timeout=900000')
    added = 0
    for doi, source, status, path in records:
        receipt = str(path.relative_to(ROOT))
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (doi,)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (tid, source, receipt)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (tid,source,status,receipt,json.dumps({'applicability':'Exact DOI in fixed candidate scope and retained response record',
                       'final_task_closure':False,'no_research_eligibility_inferred_from_retrieval':True})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,))
            added += 1
        db.commit()
    db.close()
    result = {'source_records':len(records),'new_task_attempts':added,'network_requests':0,'task_statuses_not_closed':True}
    atomic_json(FOLDER / 'source_ledger_application.json',result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()))
