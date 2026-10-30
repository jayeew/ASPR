"""Record per-DOI cached-deposit limits for the completed fixed online-date queue."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict

from .current_online_date_queue import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    db = connect()
    scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
    pending = {p['doi']: p for p in scope if db.execute(
        "SELECT 1 FROM papers WHERE doi=? AND status='pending_primary_online_date'", (p['doi'],)).fetchone()}
    receipts: dict[str, list[dict]] = defaultdict(list)
    for path in (ROOT / 'fields/crossref_raw').glob('*.gz'):
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        items = {i['DOI'].lower(): i for i in record.get('response', {}).get('message', {}).get('items', []) if i.get('DOI')}
        for doi in (set(record.get('dois', [])) | set(items)) & set(pending):
            item = items.get(doi)
            receipts[doi].append({'receipt': str(path.relative_to(ROOT)), 'retrieval_status': record.get('status'),
                                  'doi_returned': item is not None,
                                  'explicit_online_date_parts': (item or {}).get('published-online', {}).get('date-parts', []),
                                  'selected_fields': record.get('params', {}).get('select', '')})
    for doi in pending:
        if receipts[doi]:
            continue
        task_id = hashlib.sha256(('qualification:' + doi).encode()).hexdigest()[:32]
        for receipt_path, in db.execute(
                "SELECT receipt FROM attempts WHERE task_id=? AND source='crossref_public_deposit_bounded_three_journals' AND status='evidence_ready'", (task_id,)):
            request = json.loads((ROOT / receipt_path).read_text())
            if request.get('http_status') != 200 or not request.get('raw_receipt'):
                continue
            with gzip.open(ROOT / request['raw_receipt'], 'rt') as stream:
                response = json.load(stream)
            items = [i for i in response.get('message', {}).get('items', []) if i.get('DOI', '').lower() == doi]
            if len(items) == 1:
                receipts[doi].append({'receipt': receipt_path, 'raw_receipt': request['raw_receipt'],
                    'retrieval_status': 'success', 'doi_returned': True,
                    'explicit_online_date_parts': items[0].get('published-online', {}).get('date-parts', []),
                    'selected_fields': request.get('params', {}).get('select', '')})
    reviewed = []
    for doi, paper in pending.items():
        evidence = receipts[doi]
        if any(e['explicit_online_date_parts'] for e in evidence):
            reason = 'explicit_date_present_but_qualification_still_unresolved_review_identity_or_conflict'
        elif any(e['doi_returned'] for e in evidence):
            reason = 'cached_matching_doi_deposit_has_no_explicit_published_online_field'
        elif any(e['retrieval_status'] == 'success' for e in evidence):
            reason = 'successful_cached_query_returned_no_matching_doi'
        elif evidence:
            reason = 'cached_retrieval_failed_or_restricted'
        else:
            reason = 'no_local_crossref_batch_receipt_linked_other_ledger_paths_require_review'
        reviewed.append({**paper, 'reason': reason, 'receipts': evidence, 'final_task_closure': False})
    output = 'supplement/current_online_date_queue/remaining_date_review.json'
    atomic_json(ROOT / output, reviewed)
    db.execute('PRAGMA busy_timeout=900000')
    for row in reviewed:
        for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (row['doi'],)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND status=?',
                          (task_id, 'cached_online_date_field_review', output, row['reason'])).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (task_id, 'cached_online_date_field_review', row['reason'], output,
                        json.dumps({'doi': row['doi'], 'work_id': row['work_id'], 'final_task_closure': False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
    db.commit(); db.close()
    summary = {'remaining': len(reviewed), 'reasons': dict(Counter(r['reason'] for r in reviewed)),
               'routes': dict(Counter(r['route'] for r in reviewed)), 'network_requests': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'remaining_date_review_summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
