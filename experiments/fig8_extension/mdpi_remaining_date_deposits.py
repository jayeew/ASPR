"""Prefetch missing online dates for the existing MDPI native-type acquisition scope."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .cached_date_apply import run as apply_dates
from .current_online_date_queue import FOLDER as PRIOR
from .derived_refresh import refresh
from .supplement_crossref import run as collect
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/mdpi_oai/remaining_date_deposits'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope_path = FOLDER / 'scope.json'
        if not scope_path.exists():
            candidates = json.loads((ROOT / 'supplement/mdpi_oai/targeted/scope.json').read_text())['papers']
            db = connect()
            papers = [p for p in candidates if db.execute(
                "SELECT 1 FROM papers WHERE work_id=? AND status IN ('pending_no_primary_type_date','pending_primary_online_date')",
                (p['work_id'],)).fetchone()]
            db.close()
            atomic_json(scope_path, {'papers': papers, 'created_at': datetime.now(timezone.utc).isoformat(),
                                    'reason': 'Existing fixed MDPI candidates with unresolved dates; native type acquisition continues independently'})
        dois = {p['doi'] for p in json.loads(scope_path.read_text())['papers']}
        atomic_json(FOLDER / 'status.json', {'status': 'waiting_for_existing_crossref_queue', 'fixed_dois': len(dois)})
        with (PRIOR / 'runner.lock').open('a') as prior_guard:
            fcntl.flock(prior_guard, fcntl.LOCK_EX)
            prior = json.loads((PRIOR / 'status.json').read_text())
            acquisition = json.loads((PRIOR / 'acquisition.json').read_text()) if (PRIOR / 'acquisition.json').exists() else {}
            if prior['status'] != 'fixed_date_path_processed_other_fields_and_paths_remain' or acquisition.get('source_stopped_http_status'):
                atomic_json(FOLDER / 'status.json', {'status': 'retained_unattempted_prior_source_not_complete_or_restricted',
                            'prior_status': prior, 'prior_acquisition': acquisition, 'final_task_closure': False})
                return
        atomic_json(FOLDER / 'status.json', {'status': 'selecting_unqueried_deposits_without_checkpoint_lock', 'fixed_dois': len(dois)})
        before = {'cache_reuse': 'Collector skips previously queried DOI receipts before making requests',
                  'field_application': 'Existing and new date evidence applied together after acquisition',
                  'network_parallel_with_report_refresh': True}
        atomic_json(FOLDER / 'cache_application.json', before)
        db = connect()
        remaining = {doi for doi in dois if db.execute(
            "SELECT 1 FROM papers WHERE doi=? AND status IN ('pending_no_primary_type_date','pending_primary_online_date')", (doi,)).fetchone()}
        db.close()
        atomic_json(FOLDER / 'status.json', {'status': 'collecting_only_unqueried_dates', 'remaining_dois': len(remaining)})
        result = collect(apply_cache=False, requested_dois=remaining)
        atomic_json(FOLDER / 'acquisition.json', result)
        after = apply_dates(remaining, FOLDER)
        atomic_json(FOLDER / 'status.json', {'status': 'fixed_deposit_path_processed_native_type_and_metadata_queues_continue',
                    'fixed_dois': len(dois), 'acquisition': result, 'cache_application': before,
                    'new_cache_application': after, 'final_task_closure': False})
        refresh()


if __name__ == '__main__':
    run()
