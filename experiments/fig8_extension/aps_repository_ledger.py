"""Record shared DataCite batch applicability and each repository identity disposition."""
from __future__ import annotations

import fcntl
import json
from collections import Counter

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/aps_repository_identity'
    reviewed = json.loads((folder / 'relationship_review.json').read_text())['records']
    remaining = {r['work_id']: r for r in json.loads((folder / 'remaining_queue.json').read_text())['papers']}
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        for record in reviewed:
            disposition = remaining.get(record['work_id'])
            status = disposition['reason'] if disposition else 'repository_variant_confirmed_and_linked'
            detail = {'work_id': record['work_id'], 'requested_doi': record['doi'],
                      'source_receipt': record['raw_receipt'], 'source_locator': record['locator'],
                      'identity_title_matched': record['title_matched'],
                      'shared_batch_evidence': True, 'individual_http_request': False,
                      'qualification_disposition': status, 'final_task_closure': disposition is None,
                      'review': 'supplement/aps_repository_identity/relationship_review.json',
                      'remaining_queue': 'supplement/aps_repository_identity/remaining_queue.json' if disposition else '',
                      'dedup_changes': 'supplement/aps_repository_identity/dedup_changes.jsonl' if not disposition else ''}
            tasks = db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (record['doi'],)).fetchall()
            if not tasks:
                raise ValueError('Qualification task missing for reviewed repository DOI')
            for task_id, in tasks:
                exists = db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND status=?',
                                    (task_id, 'datacite_repository_identity', record['raw_receipt'], status)).fetchone()
                if exists:
                    counts['unchanged_attempts_reused'] += 1
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, 'datacite_repository_identity', status, record['raw_receipt'], json.dumps(detail)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
                counts['source_reviews_recorded'] += 1
                counts['confirmed_duplicate' if disposition is None else 'unresolved_retained'] += 1
        db.commit()
        db.close()
    result = {**counts, 'reviewed_papers': len(reviewed), 'network_requests': 0}
    atomic_json(folder / 'ledger_status.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
