"""Refresh current reports once after the fixed date and metadata batches finish."""
from __future__ import annotations

import fcntl
import json

from .current_online_date_queue import FOLDER
from .derived_refresh import refresh
from .supplement_sources import atomic_json


def run() -> None:
    marker = FOLDER / 'refresh_status.json'
    with (FOLDER / 'refresh_dispatch.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'current_reports_refreshed':
            return
        atomic_json(marker, {'status': 'waiting_for_metadata_application', 'network_requests': 0})
        with (FOLDER / 'metadata_consumer.lock').open('a') as consumer:
            fcntl.flock(consumer, fcntl.LOCK_EX)
            metadata = json.loads((FOLDER / 'metadata_status.json').read_text())
            if metadata['status'] != 'qualified_metadata_applied':
                atomic_json(marker, {'status': 'metadata_not_complete_refresh_deferred', 'metadata_state': metadata})
                return
            atomic_json(marker, {'status': 'refreshing_current_reports', 'network_requests': 0})
            refresh(force=True)
            atomic_json(marker, {'status': 'current_reports_refreshed', 'network_requests': 0,
                                 'final_task_closure': False, 'other_acquisition_queues_still_require_completion': True})


if __name__ == '__main__':
    run()
