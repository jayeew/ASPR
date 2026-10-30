"""Dispatch the fixed distribution-platform extension after the live collector ends."""
from __future__ import annotations

import fcntl
import json
import time

from .springer_remaining_collect import run as collect
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pleiades_route'
SPRINGER = ROOT / 'supplement/springer_remaining'


def collector_live() -> bool:
    with (SPRINGER / 'collector.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def run() -> None:
    marker = FOLDER / 'dispatch_status.json'
    while collector_live():
        atomic_json(marker, {'status': 'waiting_for_live_existing_collector',
                             'network_requests': 0, 'same_host_parallel_collectors': False})
        time.sleep(30)
    state = json.loads((SPRINGER / 'status.json').read_text())
    if state['status'] != 'fixed_acquisition_queue_processed_qualification_apply_required':
        atomic_json(marker, {'status': 'retained_unattempted_source_not_successfully_finished',
                             'previous_source_status': state['status'], 'reason': state.get('stop_reason', '')})
        return
    atomic_json(marker, {'status': 'dispatching_fixed_queue_extension'})
    collect('springer')
    atomic_json(marker, {'status': 'extension_collector_returned',
                         'source_status': json.loads((SPRINGER / 'status.json').read_text())['status']})


if __name__ == '__main__':
    with (FOLDER / 'dispatcher.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
