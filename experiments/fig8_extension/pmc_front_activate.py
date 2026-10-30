"""Drain the existing XML writer before activating conditional front captures."""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from .pmc_cloud_inventory import FOLDER
from .pmc_cloud_xml import run
from .supplement_sources import atomic_json


def run_activation(pid: int) -> None:
    process = Path('/proc') / str(pid)
    marker = FOLDER / 'front_activation_status.json'
    latch = FOLDER / 'source_stop.json'
    request = {'operator_transition': str(uuid.uuid4()),
               'reason': 'Drain current batch before enabling validated front range acquisition',
               'source_failure': False, 'old_pid': pid}
    if not process.exists():
        raise RuntimeError('Expected writer no longer live; inspect ownership before activation')
    start = (process / 'stat').read_text().split()[21]
    request['old_process_start'] = start
    # Exclusive creation never replaces an actual source failure or rate limit.
    with latch.open('x') as output:
        json.dump(request, output)
        output.flush()
        os.fsync(output.fileno())
    atomic_json(marker, {**request, 'status': 'waiting_for_old_batch_to_drain'})
    while process.exists():
        try:
            if (process / 'stat').read_text().split()[21] != start:
                break
        except FileNotFoundError:
            break
        time.sleep(5)
    with (FOLDER / 'xml_collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(latch.read_text()) if latch.exists() else {}
        if current.get('operator_transition') != request['operator_transition']:
            atomic_json(marker, {**request, 'status': 'real_source_stop_preserved', 'source_stop': current})
            return
        old_status = json.loads((FOLDER / 'xml_status.json').read_text())
        atomic_json(FOLDER / 'front_activation_previous_status.json', old_status)
        latch.unlink()
        atomic_json(FOLDER / 'xml_status.json', {'status': 'operator_drain_complete_resuming',
                    'previous_status_receipt': 'front_activation_previous_status.json',
                    'source_stopped': False, 'final_task_closure': False})
    atomic_json(marker, {**request, 'status': 'front_consumer_starting', 'active_pid': os.getpid()})
    run()
    atomic_json(marker, {**request, 'status': 'front_consumer_returned',
                        'xml_status': json.loads((FOLDER / 'xml_status.json').read_text())})


if __name__ == '__main__':
    with (FOLDER / 'front_activation.lock').open('a') as activation_guard:
        fcntl.flock(activation_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_activation(1240398)
