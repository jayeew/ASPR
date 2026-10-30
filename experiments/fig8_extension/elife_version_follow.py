"""Consume fixed v8 identity batches and stop after the collector terminates."""
from __future__ import annotations

import fcntl
import json
import time

from .elife_duplicate_apply import run as duplicates
from .elife_version_apply import run as qualify
from .elife_version_details import OUT
from .elife_version_metadata import run as metadata
from .supplement_sources import atomic_json


def live() -> bool:
    with (OUT / 'collector.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def run() -> None:
    marker = OUT / 'consumer_status.json'
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    consumed = previous.get('consumed', 0)
    while True:
        state = json.loads((OUT / 'status.json').read_text())
        terminal = state['status'] != 'running' or not live()
        count = state['processed']
        if count > consumed and (count - consumed >= 250 or terminal):
            results = {'qualification': qualify(), 'duplicates': duplicates(), 'metadata': metadata()}
            consumed = count
            with (OUT / 'consumer_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'consumed': consumed, 'results': results}) + '\n')
        atomic_json(marker, {'status': 'terminal_cache_processed' if terminal else 'waiting_for_increment',
                             'consumed': consumed, 'available': count, 'network_requests': 0,
                             'unresolved_not_closed': True})
        if terminal:
            return
        time.sleep(30)


if __name__ == '__main__':
    with (OUT / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
