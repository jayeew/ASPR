"""Refresh once after the fixed RCSI and both cached roster applications."""
from __future__ import annotations

import fcntl
import json
from contextlib import ExitStack
from datetime import datetime, timezone

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    base = ROOT / 'supplement'
    folder = base / 'rcsi_source_review'
    dependencies = [
        ('rcsi_source_review', 'finisher.lock', 'application_status.json',
         'available_fixed_source_records_applied'),
        ('cached_native_roster_reconciliation', 'application.lock', 'repair_execution.json',
         'fixed_cache_rosters_applied_refresh_pending'),
        ('cached_remaining_roster_reconciliation', 'application.lock', 'repair_execution.json',
         'fixed_cache_rosters_applied_refresh_pending'),
        ('lsa_native_headings', 'application.lock', 'application_status.json',
         'fixed_native_headings_applied'),
        ('pbj_native_formats', 'application.lock', 'status.json',
         'fixed_named_format_scope_applied'),
        ('pbj_native_rosters', 'application.lock', 'roster_progress.json',
         'fixed_scope_applied'),
        ('jats_annual_nonresearch_cache', 'application.lock', 'application_status.json',
         'fixed_annual_cache_applied'),
    ]
    with ExitStack() as stack:
        guard = stack.enter_context((folder / 'phase_refresh.lock').open('a'))
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'waiting_for_fixed_applications',
                 'started_at': datetime.now(timezone.utc).isoformat(),
                 'dependencies': [d[0] for d in dependencies], 'final_task_closure': False}
        atomic_json(folder / 'phase_refresh.json', state)
        for name, lock, receipt, expected in dependencies:
            handle = stack.enter_context((base / name / lock).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX)
            actual = json.loads((base / name / receipt).read_text())['status']
            if actual != expected:
                state.update(status='application_incomplete', dependency=name, actual=actual)
                atomic_json(folder / 'phase_refresh.json', state)
                raise RuntimeError(f'{name} is incomplete: {actual}')
        from .derived_refresh import refresh
        from .official_rule_reuse import run as reuse
        state['status'] = 'reusing_shared_institution_rules'
        atomic_json(folder / 'phase_refresh.json', state)
        reuse()
        state['status'] = 'refreshing'
        atomic_json(folder / 'phase_refresh.json', state)
        try:
            refresh(force=True)
        except (RuntimeError, OSError, ValueError) as error:
            state.update(status='failed', error=str(error),
                         finished_at=datetime.now(timezone.utc).isoformat())
            atomic_json(folder / 'phase_refresh.json', state)
            raise
        state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(folder / 'phase_refresh.json', state)


if __name__ == '__main__':
    run()
