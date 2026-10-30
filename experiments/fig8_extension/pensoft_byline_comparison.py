"""Compare the fixed acquired Pensoft JATS alternatives and fill only missing links."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .derived_refresh import refresh
from .pensoft_bulk import FOLDER
from .primary_byline_comparison import apply_missing_links, run as review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(source_folder: Path = FOLDER, receipt_prefix: str = "supplement/pensoft_bulk/%") -> dict:
    folder = source_folder/'primary_comparison'
    folder.mkdir(exist_ok=True)
    with (folder/'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect()
        jobs = list(db.execute("SELECT s.work_id,p.shard,s.receipt,s.digest FROM springer_author_processed s "
                               "JOIN papers p USING(work_id) WHERE s.status='existing_primary_requires_source_comparison' "
                               "AND s.receipt LIKE ? AND p.status='included_verified_research'", (receipt_prefix,)))
        db.close()
        if len({j[0] for j in jobs}) != len(jobs):
            raise ValueError('One incoming receipt per work required for comparison')
        if not jobs:
            return {'new_pairs': 0}
        result = {'review': review(folder, jobs), 'application': apply_missing_links(folder)}
        audit = json.loads((folder/'audit.json').read_text())
        attempts = {r['work_id']: r for r in json.loads((folder/'missing_link_attempts.json').read_text())}
        changed = 0
        with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            db = connect()
            with (folder/'review_events.jsonl').open('a') as stream:
                for item in audit:
                    attempt = attempts.get(item['work_id'], {})
                    if attempt.get('status') not in {'missing_links_filled','no_additional_unique_exact_links'}:
                        continue
                    after = 'primary_comparison_reviewed:' + item['status']
                    cursor = db.execute("UPDATE springer_author_processed SET status=? WHERE work_id=? AND receipt=? AND digest=? AND status='existing_primary_requires_source_comparison'",
                                        (after,item['work_id'],item['incoming_receipt'],item['incoming_processed_digest']))
                    if cursor.rowcount:
                        stream.write(json.dumps({'before':'existing_primary_requires_source_comparison','after':after,
                                                 'review':item,'attempt':attempt,'entity_and_country_tasks_not_closed':True})+'\n')
                        changed += 1
            db.commit()
            db.close()
        result['source_pairs_reviewed'] = changed
        atomic_json(folder/'result.json',result)
        if result['application'].get('papers_changed'):
            refresh()
        return result


if __name__ == '__main__':
    print(json.dumps(run()),flush=True)
