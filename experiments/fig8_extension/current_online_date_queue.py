"""Resolve the current fixed online-date queue using cached and unseen DOI deposits."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .cached_date_apply import run as apply_dates
from .derived_refresh import refresh
from .supplement_crossref import run as collect
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/current_online_date_queue'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = FOLDER / 'scope.json'
        if not path.exists():
            db = connect()
            papers = [{'doi': doi, 'work_id': wid, 'route': route} for doi, wid, route in db.execute(
                "SELECT doi,work_id,route FROM papers WHERE status='pending_primary_online_date' AND doi<>''")]
            db.close()
            atomic_json(path, {'created_at': datetime.now(timezone.utc).isoformat(), 'papers': papers,
                              'scope': 'Fixed current candidates pending explicit online date; no new years, fields, or journals'})
        dois = {r['doi'] for r in json.loads(path.read_text())['papers']}
        atomic_json(FOLDER / 'status.json', {'status': 'applying_existing_date_cache', 'fixed_dois': len(dois)})
        before = apply_dates(dois, FOLDER)
        atomic_json(FOLDER / 'cache_application.json', before)
        db = connect()
        pending = {doi for doi in dois if db.execute(
            "SELECT 1 FROM papers WHERE doi=? AND status='pending_primary_online_date'", (doi,)).fetchone()}
        db.close()
        atomic_json(FOLDER / 'status.json', {'status': 'collecting_unqueried_deposits', 'remaining_dois': len(pending)})
        acquisition = collect(apply_cache=False, requested_dois=pending)
        atomic_json(FOLDER / 'acquisition.json', acquisition)
        after = apply_dates(pending, FOLDER)
        db = connect()
        counts: dict[str, int] = {}
        for doi in dois:
            for status, in db.execute('SELECT status FROM papers WHERE doi=?', (doi,)):
                counts[status] = counts.get(status, 0) + 1
        db.close()
        atomic_json(FOLDER / 'status.json', {'status': 'fixed_date_path_processed_other_fields_and_paths_remain',
                    'fixed_dois': len(dois), 'cache_application': before, 'acquisition': acquisition,
                    'new_cache_application': after, 'current_statuses': counts,
                    'final_task_closure': False, 'generic_type_not_promoted': True})
        refresh()


if __name__ == '__main__':
    run()
