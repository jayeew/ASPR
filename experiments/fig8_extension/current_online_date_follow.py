"""Consume qualified deposited metadata once the fixed date queue has drained."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .current_online_date_queue import FOLDER
from .derived_refresh import refresh
from .supplement_apply import apply
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(folder: Path = FOLDER, expected_status: str = 'fixed_date_path_processed_other_fields_and_paths_remain') -> None:
    with (folder / 'metadata_consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = folder / 'metadata_status.json'
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'qualified_metadata_applied':
            return
        atomic_json(marker, {'status': 'waiting_for_fixed_date_queue', 'network_requests': 0})
        with (folder / 'runner.lock').open('a') as acquisition:
            fcntl.flock(acquisition, fcntl.LOCK_EX)
            state = json.loads((folder / 'status.json').read_text())
            if state['status'] != expected_status:
                atomic_json(marker, {'status': 'date_queue_not_complete_no_duplicate_acquisition', 'source_state': state})
                return
            with (ROOT / 'supplement/checkpoint.lock').open('a') as write_lock:
                fcntl.flock(write_lock, fcntl.LOCK_EX)
                db = connect()
                selected = {p['doi'] for p in json.loads((folder / 'scope.json').read_text())['papers']
                            if db.execute("SELECT 1 FROM papers WHERE doi=? AND status='included_verified_research'", (p['doi'],)).fetchone()}
                db.close()
                atomic_json(marker, {'status': 'applying_qualified_metadata', 'qualified_dois': len(selected), 'network_requests': 0})
                result = apply(scope_dois=selected, index_sources=False) if selected else {}
            atomic_json(marker, {'status': 'qualified_metadata_applied', 'qualified_dois': len(selected),
                                'result': result, 'network_requests': 0, 'final_task_closure': False})
            refresh()


if __name__ == '__main__':
    run()
