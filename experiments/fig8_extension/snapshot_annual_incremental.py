"""Consume new identity-matched archive evidence for unknown annual publisher labels."""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
import sqlite3
from collections import Counter

from . import jats_annual_fixed_scope as annual
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> None:
    folder = FOLDER / 'annual_imprints'
    with (folder / 'incremental.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = sqlite3.connect(folder / 'incremental.sqlite')
        state.execute('CREATE TABLE IF NOT EXISTS reviewed(work_id TEXT,evidence_id TEXT,receipt TEXT,batch TEXT,PRIMARY KEY(work_id,evidence_id))')
        done = {(w, e) for w, e in state.execute('SELECT work_id,evidence_id FROM reviewed')}
        old = json.loads((folder / 'review.json').read_text())['records']
        original = {(r['work_id'], str((ROOT / r['receipt']).resolve())) for r in old}
        unknown = {(r['source_id'], int(r['year'])) for r in
                   read_csv(ROOT / 'fields/journal_publisher_by_year.csv')
                   if r['nature_portfolio_year'] not in {'True', 'False'}
                   and int(r['year']) in {2023, 2024, 2025}}
        archive = sqlite3.connect(f'file:{FOLDER}/archive_jobs.sqlite?mode=ro', uri=True)
        recovered_rows = archive.execute(
            "SELECT pmcid,receipt FROM jobs WHERE status='metadata_and_available_xml_path_processed'").fetchall()
        recovered = {r[0] for r in recovered_rows}
        archive.close()
        recovered_ids = set()
        for _, receipt in recovered_rows:
            source = json.loads((ROOT / receipt).read_text())
            recovered_ids.update(p['work_id'] for p in source.get('paper_links', []))
        db = connect()
        db.execute('CREATE TEMP TABLE imprint_recovered(work_id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO imprint_recovered VALUES(?)', [(w,) for w in recovered_ids])
        records, keys, counts = [], [], Counter()
        query = '''SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,e.evidence_key,e.payload
            FROM imprint_recovered u CROSS JOIN papers p ON p.work_id=u.work_id
            CROSS JOIN current_evidence_index e INDEXED BY current_evidence_work_kind
            ON e.work_id=p.work_id AND e.kind='primary' '''
        for wid, doi, title, journal, sid, evidence_id, payload in db.execute(query):
            if (wid, evidence_id) in done:
                continue
            e = json.loads(payload)
            identifiers = set(re.findall(r'PMC\d+', e.get('url') or ''))
            if not identifiers & recovered:
                continue
            date = e.get('online_date') or ''
            if (e.get('family') != 'archived_jats' or e.get('identity') != 'matched'
                    or len(date) < 4 or (sid, int(date[:4])) not in unknown):
                continue
            receipt = e.get('receipt')
            if not receipt:
                counts['missing_native_receipt_preserved'] += 1
                continue
            if (wid, str((ROOT / 'fields/qualification' / receipt).resolve())) in original:
                counts['original_review_reused'] += 1
                continue
            records.append({'work_id': wid, 'doi': doi, 'title': title, 'journal': journal,
                            'source_id': sid, 'evidence': e})
            keys.append((wid, evidence_id, receipt))
        db.close()
        if not records:
            atomic_json(folder / 'incremental_status.json', {'status': 'no_new_applicable_cached_evidence', **counts,
                'network_requests': 0, 'final_task_closure': False})
            state.close()
            return
        token = hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()[:20]
        batch = folder / 'batches' / token
        batch.mkdir(parents=True, exist_ok=True)
        if not (batch / 'scope.json').exists():
            atomic_json(batch / 'scope.json', {'records': records, 'network_requests': 0,
                'scope': 'Current unknown journal-years; new indexed identity-matched native evidence only; acquisition receipts reused.'})
        annual.FOLDER = batch
        applied = batch / 'application_status.json'
        if not applied.exists() or json.loads(applied.read_text()).get('status') != 'fixed_annual_cache_applied':
            annual.run()
        state.executemany('INSERT OR IGNORE INTO reviewed VALUES(?,?,?,?)',
                          [(w, e, r, str(batch.relative_to(ROOT))) for w, e, r in keys])
        state.commit(); state.close()
        atomic_json(folder / 'incremental_status.json', {'status': 'new_cached_annual_evidence_processed',
            'reviewed_evidence': len(keys), 'batch_receipt': str(batch.relative_to(ROOT)),
            'application': json.loads(applied.read_text()), **counts,
            'network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
