"""Review only new archive-cache primary-link holds from snapshot recovery."""
from __future__ import annotations

import fcntl
import json
import sqlite3

from .derived_refresh import refresh
from .pmc_cloud_inventory import FOLDER as CLOUD
from .primary_byline_comparison import apply_missing_links, run as review
from .primary_comparison_ledger import run as record_attempts
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'byline_comparison'
    folder.mkdir(exist_ok=True)
    with (folder / 'review.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = folder / 'review_events.jsonl'
        done = set()
        if events.exists():
            for line in events.read_text().splitlines():
                item = json.loads(line)
                if item['status'] == 'source_hash_missing' and item.get('comparison_adapter') != 'pmc_retained_hash_2':
                    continue
                done.add((item['work_id'], item['incoming_receipt'], item['incoming_processed_digest']))
        state = sqlite3.connect(f'file:{FOLDER}/archive_jobs.sqlite?mode=ro', uri=True)
        identifiers = [r[0] for r in state.execute("SELECT pmcid FROM jobs WHERE status='metadata_and_available_xml_path_processed'")]
        state.close()
        receipts = set()
        for pmcid in identifiers:
            path = CLOUD / 'xml_results' / (pmcid + '.1.json')
            outcome = json.loads(path.read_text())
            if outcome.get('parsed_receipt'):
                receipts.add(outcome['parsed_receipt'])
        db = connect()
        jobs = [row for row in db.execute("""SELECT s.work_id,p.shard,s.receipt,s.digest
            FROM springer_author_processed s JOIN papers p USING(work_id)
            WHERE s.status='existing_primary_requires_source_comparison'
            AND p.status='included_verified_research'""")
            if row[2] in receipts and (row[0], row[2], row[3]) not in done]
        db.close()
        if not jobs:
            print(json.dumps({'new_comparison_pairs': 0}), flush=True)
            return
        atomic_json(folder / 'selected_jobs.json', {'jobs': jobs, 'network_requests': 0})
        result = {'review': review(folder, jobs), 'application': apply_missing_links(folder)}
        attempts = {r['work_id']: r for r in json.loads((folder / 'missing_link_attempts.json').read_text())}
        with events.open('a') as stream:
            for item in json.loads((folder / 'audit.json').read_text()):
                attempt = attempts.get(item['work_id'], {})
                if attempt.get('status') == 'retained_state_changed_requires_recomparison':
                    continue
                stream.write(json.dumps({**item, 'application': attempt,
                    'comparison_adapter': 'pmc_retained_hash_2',
                    'final_task_closure': False, 'entity_mapping_not_asserted': True}) + '\n')
        result['ledger'] = record_attempts(folder)
        atomic_json(folder / 'result.json', result)
        if result['application'].get('papers_changed'):
            refresh()
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
