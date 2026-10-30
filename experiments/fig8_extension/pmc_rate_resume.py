"""Perform one delayed, slower resume of a retained PMC 429 checkpoint."""
from __future__ import annotations

import fcntl
import json
import signal
import time
from datetime import datetime, timezone

from . import pmc_journal_harvest as harvest
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT/'supplement/pmc_journal_batches'
    policy = json.loads((folder/'deferred_rate_retry.json').read_text())
    marker = folder/'rate_resume_status.json'
    with (folder/'rate_resume.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = datetime.fromisoformat(policy['not_before_utc'])
        atomic_json(marker, {'status':'waiting_for_rate_cooldown','not_before_utc':policy['not_before_utc'],
                             'requests_sent':0,'retry_rounds_allowed':1})
        while (delay := (deadline-datetime.now(timezone.utc)).total_seconds()) > 0:
            time.sleep(min(delay,30))
        with (ROOT/'supplement/pmc_api.lock').open('a') as api:
            fcntl.flock(api,fcntl.LOCK_EX)
            state = json.loads((folder/'status.json').read_text())
            if state.get('status') != 'source_stopped' or state.get('receipt',{}).get('http_status') != 429:
                atomic_json(marker, {'status':'no_resume_state_changed'});return
            atomic_json(marker, {'status':'resuming_existing_checkpoint','request_interval_seconds':1.05})
            harvest.run()
            atomic_json(marker, {'status':'resume_returned_no_automatic_repeat',
                                 'collector_state':json.loads((folder/'status.json').read_text())['status']})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, harvest.request_checkpoint_stop)
    run()
