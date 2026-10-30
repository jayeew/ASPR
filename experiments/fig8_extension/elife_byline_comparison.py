"""Compare retained primary signatures with cached eLife v8 signatures."""
from __future__ import annotations

import fcntl
import json
import sqlite3

from .derived_refresh import refresh
from .elife_version_details import OUT
from .primary_byline_comparison import apply_missing_links, run as review
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def record_reviewed_pairs() -> int:
    folder = OUT / 'primary_comparison'
    audit = json.loads((folder / 'audit.json').read_text())
    attempts = {r['work_id']: r for r in json.loads((folder / 'missing_link_attempts.json').read_text())}
    done = 0
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = sqlite3.connect(ROOT / 'supplement/tasks.sqlite', timeout=60)
        with (folder / 'review_events.jsonl').open('a') as stream:
            for item in audit:
                attempt = attempts.get(item['work_id'], {})
                if attempt.get('status') not in {'missing_links_filled', 'no_additional_unique_exact_links'}:
                    continue
                after = 'primary_comparison_reviewed:' + item['status']
                cursor = db.execute("UPDATE springer_author_processed SET status=? WHERE work_id=? AND receipt=? AND digest=? AND status='existing_primary_requires_source_comparison'",
                                    (after, item['work_id'], item['incoming_receipt'], item['incoming_processed_digest']))
                if cursor.rowcount:
                    stream.write(json.dumps({'before': 'existing_primary_requires_source_comparison',
                                             'after': after, 'review': item, 'attempt': attempt,
                                             'unresolved_entity_mapping_not_closed': True}) + '\n')
                    done += 1
        db.commit()
        db.close()
    return done


def run() -> dict:
    folder = OUT / 'primary_comparison'
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    jobs = list(db.execute("""SELECT s.work_id,p.shard,s.receipt,s.digest
        FROM springer_author_processed s JOIN papers p USING(work_id)
        WHERE s.status='existing_primary_requires_source_comparison'
        AND s.receipt LIKE 'supplement/elife_official/%'
        AND p.status='included_verified_research'"""))
    db.close()
    if not jobs:
        return {'new_pairs': 0}
    result = {'review': review(folder, jobs), 'application': apply_missing_links(folder)}
    result['source_pairs_reviewed'] = record_reviewed_pairs()
    atomic_json(folder / 'result.json', result)
    if result['application'].get('papers_changed'):
        refresh()
    return result


if __name__ == '__main__':
    run()
