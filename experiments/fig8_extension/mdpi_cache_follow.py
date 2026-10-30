"""Consume MDPI native type evidence in increments, with one serialized consumer."""
from __future__ import annotations

import fcntl
import json
import time

from .mdpi_cache_apply import run as apply_cache
from .mdpi_oai_harvest import FOLDER
from .supplement_sources import atomic_json


def collector_live() -> bool:
    with (FOLDER / 'collector.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def run() -> None:
    path = FOLDER / 'consumer_status.json'
    prior = json.loads(path.read_text()) if path.exists() else {}
    consumed = prior.get('consumed', -1)
    while True:
        state = json.loads((FOLDER / 'harvest_status.json').read_text())
        count = state['matched_records']
        terminal = state['status'] != 'running' or not collector_live()
        if consumed < 0 or count - consumed >= 500 or (terminal and count > consumed):
            result = apply_cache()
            from .mdpi_primary_followup import run as followup_dates
            followup_dates()
            with (FOLDER / 'consumer_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'available': count, 'result': result}) + '\n')
            consumed = count
        atomic_json(path, {'status': 'terminal_cache_processed' if terminal else 'waiting_for_increment',
                          'consumed': consumed, 'available': count, 'network_requests': 0,
                          'date_and_affiliation_gaps_not_closed': True})
        if terminal:
            return
        time.sleep(30)


if __name__ == '__main__':
    with (FOLDER / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
