"""Reconcile fixed PDF-date outcomes without treating unread sources as missing fields."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from .scoap3_pdf_dates import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def classify(record: dict) -> str:
    if record['status'] == 'explicit_online_date_identified':
        return 'explicit_online_date_identified'
    if record['status'] == 'current_eligibility_or_known_date_skipped':
        return 'current_eligibility_or_known_date_skipped'
    states = {e.get('status', '') for e in record['evidence']}
    if states == {'reviewed_restricted_path_or_doi_not_requested'}:
        return 'not_requested_previously_restricted_path_or_doi'
    if 'first_page_standalone_doi_not_unique_or_mismatched' in states:
        return 'first_page_identity_unresolved'
    if 'explicit_online_date_absent_or_conflicting' in states:
        return 'read_first_page_date_absent_or_conflicting'
    return 'retrieval_or_parse_unresolved'


def run() -> dict:
    source = json.loads((FOLDER / 'status.json').read_text())
    consumer = json.loads((FOLDER / 'consumer_status.json').read_text())
    if source['status'] != 'fixed_pdf_date_queue_processed' or consumer.get('unconsumed_results') != 0:
        raise ValueError('Fixed source and consumer must finish before reconciliation')
    scope = {j['doi'] for j in json.loads((FOLDER / 'fixed_scope.json').read_text())['jobs']}
    rows = []
    for path in sorted((FOLDER / 'results').glob('*.json')):
        record = json.loads(path.read_text())
        receipts = []
        for evidence in record['evidence']:
            receipt = evidence.get('receipt')
            if receipt:
                result = json.loads((ROOT / receipt).read_text())
                raw = result.get('raw_receipt')
                if raw and not (ROOT / raw).exists():
                    raise ValueError('Missing retained source: ' + raw)
                receipts.append({'receipt': receipt, 'raw_receipt': raw, 'http_status': result.get('http_status'),
                                 'attempts': result.get('attempts', []), 'evidence_status': evidence.get('status')})
        rows.append({'doi': record['doi'], 'status': classify(record), 'acquisition_status': record['status'],
                     'result_receipt': str(path.relative_to(ROOT)), 'source_receipts': receipts,
                     'restriction_evidence': 'supplement/scoap3_native/resource_restriction_review.json',
                     'final_task_closure': False})
    assert len(rows) == len(scope) and {r['doi'] for r in rows} == scope
    output = str((FOLDER / 'final_path_outcomes.json').relative_to(ROOT))
    atomic_json(ROOT / output, rows)
    db = connect(); db.execute('PRAGMA busy_timeout=900000')
    added = 0
    for row in rows:
        task_id = hashlib.sha256(('qualification:' + row['doi']).encode()).hexdigest()[:32]
        if not db.execute('SELECT 1 FROM tasks WHERE task_id=?', (task_id,)).fetchone():
            raise ValueError('Missing qualification ledger identity: ' + row['doi'])
        if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                      (task_id, 'scoap3_pdf_date_path_reconciliation', output)).fetchone():
            continue
        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                   (task_id, 'scoap3_pdf_date_path_reconciliation', row['status'], output,
                    json.dumps({'doi': row['doi'], 'result_receipt': row['result_receipt'], 'final_task_closure': False})))
        db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
        added += 1
    db.commit(); db.close()
    summary = {'fixed_dois': len(scope), 'outcomes': dict(Counter(r['status'] for r in rows)),
               'all_fixed_dois_accounted_for': True, 'unconsumed_results': 0,
               'ledger_attempts_added': added, 'network_requests': 0,
               'limitation': 'First-page date source only. Restricted paths were not fetched. Research type and alternative evidence paths remain separate.'}
    atomic_json(FOLDER / 'final_path_summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
