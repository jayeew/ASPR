"""Drain snapshot-derived cache work and refresh once after the fixed collector ends."""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

TERMINAL = {'discovered_archive_leads_processed_other_paths_pending',
            'source_stopped_unattempted_preserved'}
STEPS = ['snapshot_byline_comparison', 'byline_complete_cache', 'snapshot_annual_incremental',
         'official_rule_reuse', 'country_incremental', 'frontiers_perspective_policy', 'remaining_qualification_plan',
         'derived_refresh']


def collector_state() -> str:
    status = json.loads((FOLDER / 'archive_status.json').read_text()).get('status')
    for marker in ['incremental_rule_activation.json', 'type_rule_activation.json']:
        path = FOLDER / marker
        if not path.exists():
            continue
        pid = json.loads(path.read_text()).get('new_pid')
        proc = Path('/proc') / str(pid)
        try:
            command = (proc / 'cmdline').read_bytes()
        except FileNotFoundError:
            continue
        if b'experiments.fig8_extension.snapshot_archive_consumer' in command:
            return 'live'
    if status in TERMINAL:
        return 'terminal'
    return 'collector_absent_without_terminal_receipt'


def run() -> None:
    folder = FOLDER / 'phase_finish'; folder.mkdir(exist_ok=True)
    marker = folder / 'status.json'
    with (folder / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(marker.read_text()) if marker.exists() else {'completed_modules': []}
        if state.get('status') == 'snapshot_phase_refreshed':
            return
        while True:
            condition = collector_state()
            if condition == 'terminal':
                break
            state.update(status='waiting_for_live_collector' if condition == 'live' else condition,
                         checked_at=datetime.now(timezone.utc).isoformat(), goal_complete=False)
            atomic_json(marker, state)
            if condition != 'live':
                return
            time.sleep(30)
        for module in STEPS:
            if module in state['completed_modules']:
                continue
            state.update(status='running_cached_finish_step', current_module=module)
            atomic_json(marker, state)
            with (folder / (module + '.log')).open('a') as output:
                args = [sys.executable, '-u', '-m', 'experiments.fig8_extension.' + module]
                if module == 'derived_refresh':
                    args.append('--force')
                result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                state.update(status='failed', returncode=result.returncode)
                atomic_json(marker, state)
                raise RuntimeError(f'Cached finish failed: {module}; retained log available')
            state['completed_modules'].append(module)
            atomic_json(marker, state)
        state.update(status='snapshot_phase_refreshed', finished_at=datetime.now(timezone.utc).isoformat(),
                     goal_complete=False, next_required='Review full-scope remaining paths and unresolved entities; acquisition terminal does not establish final closure or formal comparison readiness')
        atomic_json(marker, state)


if __name__ == '__main__':
    run()
