"""Consume cached deposits as the fixed native-type queue qualifies additional papers."""
from __future__ import annotations

import fcntl
import json
import time

from .derived_refresh import refresh
from .mdpi_remaining_date_deposits import FOLDER
from .mdpi_targeted_collect import TARGET
from .official_rule_reuse import run as reuse_entities
from .supplement_apply import apply
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'new_qualified_metadata.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = FOLDER / 'new_qualified_metadata_status.json'
        atomic_json(marker, {'status': 'waiting_for_initial_metadata_consumer', 'network_requests': 0})
        with (FOLDER / 'metadata_consumer.lock').open('a') as initial:
            fcntl.flock(initial, fcntl.LOCK_EX)
            previous = json.loads((FOLDER / 'metadata_status.json').read_text())
            if previous['status'] != 'qualified_metadata_applied':
                atomic_json(marker, {'status': 'initial_metadata_incomplete_retained', 'previous': previous})
                return
        scope = {p['doi'] for p in json.loads((FOLDER / 'scope.json').read_text())['papers']}
        completed_path = FOLDER / 'new_qualified_metadata_done.json'
        # The earlier interim scope was recorded before application, then confirmed successful.
        done = set(json.loads(completed_path.read_text())) if completed_path.exists() else set()
        interim = FOLDER / 'interim_metadata.json'
        if interim.exists():
            done.update(json.loads((FOLDER / 'interim_metadata_scope.json').read_text())['dois'])
        last_flush = time.monotonic()
        while True:
            db = connect()
            qualified = {doi for doi, in db.execute(
                "SELECT doi FROM papers WHERE route='mdpi' AND status='included_verified_research'")}
            db.close()
            pending = (qualified & scope) - done
            source = json.loads((TARGET / 'status.json').read_text())
            consumer = json.loads((TARGET / 'consumer_status.json').read_text())
            terminal = source['status'] not in {'running', 'draining_inflight'} and consumer['status'] == 'terminal_cache_processed'
            if pending and (len(pending) >= 2000 or terminal or time.monotonic() - last_flush >= 600):
                atomic_json(marker, {'status': 'applying_newly_qualified_cache', 'pending': len(pending), 'network_requests': 0})
                with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = apply(scope_dois=pending, index_sources=False)
                done.update(pending)
                atomic_json(completed_path, sorted(done))
                with (FOLDER / 'new_qualified_metadata_events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'qualified_dois': sorted(pending), 'result': result, 'network_requests': 0}) + '\n')
                reuse_entities()
                refresh()
                last_flush = time.monotonic()
                pending = set()
            atomic_json(marker, {'status': 'fixed_type_phase_metadata_processed' if terminal else 'waiting_for_new_qualifications',
                        'fixed_scope_dois': len(scope), 'processed_qualified_dois': len(done), 'pending': len(pending),
                        'network_requests': 0, 'final_task_closure': False,
                        'limitation': 'Metadata applied does not imply complete author links or resolved institutions; future evidence paths remain independent.'})
            if terminal:
                return
            time.sleep(60)


if __name__ == '__main__':
    run()
