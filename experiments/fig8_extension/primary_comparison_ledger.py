"""Append reviewed cached-byline outcomes to existing metadata tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(folder: Path) -> dict:
    events = folder/'review_events.jsonl'
    db = connect()
    db.execute('PRAGMA busy_timeout=900000')
    added = 0
    pairs = 0
    for line in events.read_text().splitlines():
        event = json.loads(line)
        review = event.get('review', event)
        attempt = event.get('attempt', event.get('application', {}))
        pairs += 1
        for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='metadata' AND entity_key=?",
                                   (review['work_id'],)).fetchall():
            source = 'cached_primary_byline_comparison'
            receipt = review['incoming_receipt']
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                          (task_id, source, receipt)).fetchone():
                continue
            detail = {'comparison':review['status'], 'raw_sha256':review.get('incoming_raw_sha256'),
                      'review_events':str(events.relative_to(ROOT)), 'application':attempt,
                      'final_task_closure':False}
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (task_id, source, attempt.get('status', review['status']), receipt, json.dumps(detail)))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
            added += 1
            if added % 250 == 0:
                db.commit()
    db.commit()
    db.close()
    result = {'reviewed_pairs':pairs, 'metadata_attempts_added':added, 'network_requests':0,
              'unresolved_disagreements_retained':True}
    atomic_json(folder/'ledger_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    run(parser.parse_args().folder)
