"""Ensure the one delayed PMC resume has a cache consumer, without duplicate writers."""
from __future__ import annotations

import fcntl
import json
import time

from .pmc_journal_follow import FOLDER, run as consume
from .supplement_sources import atomic_json


def run() -> None:
    marker = FOLDER/'resume_consumer_status.json'
    with (FOLDER/'resume_consumer.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            recovery = json.loads((FOLDER/'rate_resume_status.json').read_text())['status']
            state = json.loads((FOLDER/'status.json').read_text())['status']
            if recovery == 'no_resume_state_changed':
                atomic_json(marker,{'status':'resume_cancelled_no_consumer_started'});return
            if recovery == 'waiting_for_rate_cooldown' or (recovery == 'resuming_existing_checkpoint' and state == 'source_stopped'):
                atomic_json(marker,{'status':'waiting_for_acquisition_resume','network_requests':0})
                time.sleep(30);continue
            try:
                atomic_json(marker, {'status':'starting_or_running_cache_consumer',
                                     'detail_status':'consumer_status.json', 'network_requests':0})
                consume()
            except BlockingIOError:
                atomic_json(marker,{'status':'existing_cache_consumer_retained','network_requests':0})
                time.sleep(30);continue
            atomic_json(marker,{'status':'resumed_cache_consumption_finished','network_requests':0})
            return


if __name__ == '__main__':
    run()
