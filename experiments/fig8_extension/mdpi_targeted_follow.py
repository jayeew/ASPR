"""Serialize targeted OAI consumption with the previous bulk consumer."""
from __future__ import annotations

import fcntl
import json
import time

from .mdpi_cache_apply import run as apply_cache
from .mdpi_primary_followup import run as follow_dates
from .mdpi_targeted_collect import FOLDER, TARGET
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER/'consumer.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        marker=TARGET/'consumer_status.json'
        consumed=json.loads(marker.read_text()).get('consumed',0) if marker.exists() else 0
        while True:
            state=json.loads((TARGET/'status.json').read_text())
            available=state['matched_records'];terminal=state['status'] not in {'running','draining_inflight'}
            if available-consumed>=500 or (terminal and available>consumed):
                result=apply_cache();follow_dates();consumed=available
                with (TARGET/'consumer_events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'consumed':consumed,'type_application':result})+'\n')
            atomic_json(marker,{'status':'terminal_cache_processed' if terminal else 'waiting_for_increment',
                                'consumed':consumed,'available':available})
            if terminal:return
            time.sleep(30)


if __name__=='__main__':
    run()
