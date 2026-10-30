"""Attach each retained native-source outcome to its applicable tasks, idempotently."""
from __future__ import annotations

import json
from collections import Counter

from .pensoft_bulk import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    db = connect()
    db.execute('PRAGMA busy_timeout=900000')
    counts = Counter()
    for subfolder in ['native/results', 'alternate_hosts/results']:
        for path in (FOLDER/subfolder).glob('*.json'):
            record = json.loads(path.read_text())
            doi, status = record['doi'], record['status']
            receipt = str(path.relative_to(ROOT))
            source = 'pensoft_' + subfolder.split('/')[0] + '_xml'
            tasks = db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=? "
                               "UNION SELECT task_id FROM tasks WHERE kind='metadata' AND entity_key IN "
                               "(SELECT work_id FROM papers WHERE doi=?)", (doi, doi)).fetchall()
            for tid, in tasks:
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (tid, source, receipt)).fetchone():
                    counts['existing_attempts_skipped'] += 1
                    continue
                detail = {'doi': doi, 'source_outcome': status, 'final_task_closure': False,
                          'qualification_not_implied': True, 'metadata_completeness_not_implied': True,
                          'http_status': record.get('receipt', {}).get('http_status'),
                          'applicability': 'Exact DOI in the retained per-paper native source request/result',
                          'evidence_receipt': record.get('staged_receipt', ''),
                          'next_step': 'Use evidence application and remaining field queues; this acquisition outcome alone does not close tasks'}
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, source, status, receipt, json.dumps(detail)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
                counts['new_attempts'] += 1
            counts['source_results_examined'] += 1
        db.commit()
    db.close()
    output = {**counts, 'network_requests': 0, 'task_statuses_not_changed': True}
    atomic_json(FOLDER/'native_ledger_status.json', output)
    return output


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
