"""Consume new annual and entity evidence once the fixed MDPI date phase ends."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .mdpi_dated_annual import run as apply_annual
from .mdpi_remaining_date_deposits import FOLDER
from .official_rule_reuse import run as reuse_entities
from .supplement_sources import atomic_json


def run() -> None:
    marker = FOLDER / 'phase_finish_status.json'
    with (FOLDER / 'phase_finish.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'phase_reports_refreshed':
            return
        state = {'status': 'waiting_for_qualified_metadata', 'network_requests': 0,
                 'final_task_closure': False, 'started_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(marker, state)
        with (FOLDER / 'metadata_consumer.lock').open('a') as consumer:
            fcntl.flock(consumer, fcntl.LOCK_EX)
            source = json.loads((FOLDER / 'status.json').read_text())
            metadata = json.loads((FOLDER / 'metadata_status.json').read_text())
            if (source['status'] != 'fixed_deposit_path_processed_native_type_and_metadata_queues_continue'
                    or metadata['status'] != 'qualified_metadata_applied'):
                state.update(status='phase_incomplete_retained', source=source, metadata=metadata)
                atomic_json(marker, state)
                return
            state.update(status='applying_new_dated_publisher_evidence')
            atomic_json(marker, state)
            state['annual_application'] = apply_annual()
            state.update(status='reusing_reviewed_entities_on_changed_authors')
            atomic_json(marker, state)
            reuse_entities()
            state.update(status='refreshing_phase_reports')
            atomic_json(marker, state)
            refresh(force=True)
            state.update(status='phase_reports_refreshed', finished_at=datetime.now(timezone.utc).isoformat(),
                         limitation='Other fixed acquisition queues remain open; this phase is not full Fig8 completion')
            atomic_json(marker, state)


if __name__ == '__main__':
    run()
