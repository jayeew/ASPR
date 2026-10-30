"""Finish the current acquisition phase once, without declaring goal completion."""
from __future__ import annotations

import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

LOCKS = (
    'pmc_cloud/resume_dispatch.lock', 'pmc_cloud/inventory.lock',
    'pmc_cloud/version_dispatch.lock', 'pmc_cloud/version_collector.lock',
    'pmc_cloud/xml_collector.lock', 'springer_remaining/collector.lock',
    'pleiades_route/dispatcher.lock', 'mdpi_oai/targeted/collector.lock',
    'mdpi_oai/consumer.lock',
    'mdpi_oai/remaining_date_deposits/new_qualified_metadata.lock',
)
FOLDER = ROOT / 'supplement/current_phase_finish'


def busy_sources() -> list[str]:
    busy = []
    for relative in LOCKS:
        path = ROOT / 'supplement' / relative
        if not path.exists():
            continue
        with path.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy.append(relative)
    manifest = ROOT / 'supplement/publisher_consumer_resume.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        proc = Path('/proc') / str(record['new_pid'])
        if proc.exists():
            command = (proc / 'cmdline').read_bytes().decode().replace('\0', ' ')
            same_process = record.get('process_start') in (None, (proc / 'stat').read_text().split()[21])
            if 'python' in command and same_process:
                busy.append('publisher_consumer_pid:' + str(record['new_pid']))
    return busy


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = FOLDER / 'status.json'
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'current_phase_report_refreshed':
            return
        previous = None
        while True:
            busy = busy_sources()
            if busy != previous:
                atomic_json(marker, {'status':'waiting_for_current_phase_workers', 'busy':busy,
                            'checked_at':datetime.now(timezone.utc).isoformat(),
                            'network_requests':0, 'goal_complete':False})
                previous = busy
            if not busy:
                break
            time.sleep(30)
        from .publisher_delta_apply import apply_available, unprocessed_author_jobs
        from .springer_author_cache import run as apply_authors
        from .official_rule_reuse import run as reuse_entities
        from .jats_annual_copyright import run as annual
        from .derived_refresh import refresh
        atomic_json(marker, {'status':'draining_cached_publisher_results','goal_complete':False})
        qualification = apply_available()
        db = connect()
        jobs = unprocessed_author_jobs(db)
        db.close()
        authors = apply_authors(selected_jobs=jobs, workers=8, fill_abstract=True) if jobs else {}
        reuse_entities()
        atomic_json(marker, {'status':'applying_remaining_cached_annual_evidence',
                    'qualification':qualification,'authors':authors,'goal_complete':False})
        annual_result = annual()
        atomic_json(marker, {'status':'refreshing_current_phase_report', 'annual':annual_result,
                    'qualification':qualification,'authors':authors,'goal_complete':False})
        refresh(force=True)
        atomic_json(marker, {'status':'current_phase_report_refreshed',
                    'finished_at':datetime.now(timezone.utc).isoformat(),
                    'qualification':qualification,'authors':authors,'annual':annual_result,
                    'network_requests':0,'goal_complete':False,
                    'next_required':'Review remaining acquisition, entity, provenance and readiness gates in goal_completion_audit.json; source stops and unknowns are not automatically final.'})


if __name__ == '__main__':
    run()
