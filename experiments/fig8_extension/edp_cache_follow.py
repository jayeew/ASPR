"""Consume EDP cache increments until the fixed collector finishes or stops."""
from __future__ import annotations

import fcntl
import json
import time

from .edp_errata_apply import run as apply_errata
from .edp_oai_harvest import FOLDER
from .supplement_sources import atomic_json


def collector_live() -> bool:
    with (FOLDER / 'collector.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def cache_count() -> int:
    return sum(len(json.loads(path.read_text()).get('matched', {}))
               for path in FOLDER.glob('*_progress.json'))


def ready(count: int, consumed: int, terminal: bool) -> bool:
    return count > consumed and (count - consumed >= 1000 or terminal or consumed == 0)


def run() -> None:
    marker = FOLDER / 'consumer_status.json'
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    consumed = previous.get('consumed_records', 0)
    while True:
        harvest = json.loads((FOLDER / 'harvest_status.json').read_text())
        live = collector_live()
        terminal = harvest['status'] != 'running' or not live
        count = cache_count()
        if ready(count, consumed, terminal):
            result = apply_errata()
            consumed = count
            with (FOLDER / 'consumer_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'consumed_records': consumed, 'result': result}) + '\n')
        status = ('collector_terminal_cache_consumed' if terminal else 'waiting_for_cache_increment')
        atomic_json(marker, {'status': status, 'consumed_records': consumed,
                             'available_records': count, 'collector_live': live,
                             'harvest_status': harvest['status'], 'network_requests': 0,
                             'unresolved_not_closed': True})
        if terminal:
            return
        time.sleep(30)


if __name__ == '__main__':
    with (FOLDER / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
