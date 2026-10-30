"""Consume fixed-scope Pensoft pages as they arrive; serialize with other writers."""
from __future__ import annotations

import fcntl
import json
import time

from .pensoft_bulk import FOLDER
from .pensoft_bulk_apply import run as apply
from .batch_memory import release_batch_memory
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            db = connect()
            known = {row[0] for row in db.execute("SELECT receipt FROM supplied_publisher_processed WHERE receipt LIKE 'supplement/pensoft_bulk/staged/%'")}
            db.close()
            pending = sum(str((FOLDER / 'staged' / p.name).relative_to(ROOT)) not in known
                          for p in (FOLDER / 'matched').glob('*.json'))
            state = json.loads((FOLDER / 'status.json').read_text())['status']
            if pending >= 250 or (state != 'running' and pending):
                result = apply()
                atomic_json(FOLDER / 'consumer_status.json', {'status': 'batch_processed', 'application': result})
            if state != 'running':
                atomic_json(FOLDER / 'consumer_status.json', {'status': 'available_metadata_processed_unknown_dates_and_authors_retained', 'source_status': state})
                return
            release_batch_memory()
            time.sleep(30)


if __name__ == '__main__':
    run()
