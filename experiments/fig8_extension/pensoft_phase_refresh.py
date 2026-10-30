"""Coalesce the completed Pensoft phase into one current-data recomputation."""
from __future__ import annotations

import fcntl
import json

from .derived_refresh import refresh
from .pensoft_bulk import FOLDER
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER/'taxonomy_rule'
    status_path = folder/'phase_refresh.json'
    with (folder/'refresh_dispatch.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
            return
        atomic_json(status_path, {'status': 'waiting_for_taxonomy_application', 'network_requests': 0})
        with (folder/'dispatcher.lock').open('a') as phase:
            fcntl.flock(phase, fcntl.LOCK_EX)
            state = json.loads((folder/'status.json').read_text())
            if state.get('status') != 'reviewed_taxonomy_rule_applied_other_types_retained':
                atomic_json(status_path, {'status': 'phase_not_complete_refresh_not_started', 'phase_status': state.get('status')})
                return
            atomic_json(status_path, {'status': 'refreshing', 'network_requests': 0})
            refresh(force=True)
        atomic_json(status_path, {'status': 'complete', 'network_requests': 0,
                                 'full_preparation_complete': False,
                                 'scope': 'One consistent checkpoint after Pensoft; other source queues continue'})


if __name__ == '__main__':
    run()
