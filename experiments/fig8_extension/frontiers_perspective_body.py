"""Acquire missing Perspective bodies within the fixed, previously reviewed scope."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter

from .frontiers_content_cache import FOLDER, extract
from .springer_remaining_collect import run as collect
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'missing_body_publisher'
    folder.mkdir(exist_ok=True)
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'status.json').exists():
            return
        scope = json.loads((FOLDER / 'missing_body_path_inventory.json').read_text())
        rows = scope['records']
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = connect()
            for row in rows:
                assert row['doi'].startswith('10.3389/')
                assert db.execute('SELECT doi FROM papers WHERE work_id=?',
                                  (row['work_id'],)).fetchone() == (row['doi'],)
                db.execute("INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status) "
                           "VALUES(?,'frontiers','pending')", (row['doi'],))
            db.commit()
            db.close()
        collect('frontiers', workers=8, selected_dois={r['doi'] for r in rows})
        db = connect()
        results = []
        for row in rows:
            state, receipt = db.execute('SELECT status,receipt FROM publisher_remaining_jobs '
                                       'WHERE doi=?', (row['doi'],)).fetchone()
            result = {'work_id': row['work_id'], 'doi': row['doi'],
                      'source_status': state, 'source_receipt': receipt,
                      'status': 'body_unresolved', 'final_task_closure': False}
            if state == 'parsed':
                record = json.loads((ROOT / receipt).read_text())
                try:
                    raw = gzip.decompress((ROOT / 'fields/qualification' / record['raw_receipt']).read_bytes())
                    body = extract(raw, row['doi'])
                    path = folder / (hashlib.sha256(row['doi'].encode()).hexdigest() + '.json')
                    atomic_json(path, {**body, 'work_id': row['work_id'],
                        'source_receipt': receipt, 'raw_receipt': record['raw_receipt'],
                        'raw_sha256': hashlib.sha256(raw).hexdigest(),
                        'qualification_adjudicated': False})
                    result.update(status='matched_body_ready_for_content_review',
                                  body_receipt=str(path.relative_to(ROOT)),
                                  body_characters=len(body['body']))
                except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
                    result['reason'] = str(error)
            results.append(result)
        db.close()
        atomic_json(folder / 'status.json', {'records': results,
            'outcomes': dict(Counter(r['status'] for r in results)),
            'fixed_papers': len(rows), 'qualification_changes': 0,
            'final_task_closure': False,
            'limitation': 'Body availability is not evidence of original research; content adjudication remains required.'})
        print(json.dumps(dict(Counter(r['status'] for r in results))), flush=True)


if __name__ == '__main__':
    run()
