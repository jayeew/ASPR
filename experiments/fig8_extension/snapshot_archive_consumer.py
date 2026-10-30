"""Consume newly discovered PMC leads continuously using the existing cached adapter."""
from __future__ import annotations

import fcntl
import json
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .pmc_cloud_inventory import FOLDER as CLOUD
from .archive_mapping_route import assess as mapping_routes
from .pmc_cloud_versions import STOP, journal_priority
from .pmc_cloud_xml import apply_batch
from .priority_cloud_leads import acquire
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'archive_consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = sqlite3.connect(FOLDER / 'archive_jobs.sqlite')
        state.execute('CREATE TABLE IF NOT EXISTS jobs(pmcid TEXT PRIMARY KEY,status TEXT,receipt TEXT)')
        links = sqlite3.connect(f'file:{FOLDER}/locations.sqlite?mode=ro', uri=True)
        while True:
            current = connect(); groups: dict[str, dict] = defaultdict(dict)
            done = {r[0] for r in state.execute('SELECT pmcid FROM jobs')}
            for wid, evidence in links.execute("SELECT work_id,evidence FROM links WHERE status='new_archive_metadata_lead'"):
                item = json.loads(evidence); pmcid = item['pmcid']
                if pmcid in done:
                    continue
                row = current.execute('SELECT doi,title,journal,status,source_id,shard FROM papers WHERE work_id=?', (wid,)).fetchone()
                if not row:
                    continue
                if row[3] == 'included_verified_research':
                    gap = current.execute("SELECT 1 FROM tasks WHERE kind='metadata' AND entity_key=? AND active=1 AND status<>'resolved'", (wid,)).fetchone()
                    if not gap:
                        continue
                elif not row[3].startswith('pending'):
                    continue
                groups[pmcid][wid] = {'work_id': wid, 'doi': row[0], 'title': row[1],
                    'journal': row[2], 'source_id': row[4], 'pmcid': pmcid,
                    'eligibility_status': row[3],
                    'shard': row[5],
                    'discovery_receipt': 'supplement/snapshot_location_recovery/locations.sqlite'}
            current.close()
            jobs = [(pmcid, list(papers.values())) for pmcid, papers in groups.items()]
            jobs.sort(key=lambda item: min((not p['eligibility_status'].startswith('pending'),
                journal_priority(p['source_id'], p['journal']), p['shard']) for p in item[1]))
            if STOP.is_set() or (CLOUD / 'source_stop.json').exists():
                atomic_json(FOLDER / 'archive_status.json', {'status': 'source_stopped_unattempted_preserved', 'queued_identifiers': len(jobs), 'final_task_closure': False})
                break
            # Refresh discovery and priorities after each operational batch so newly
            # recovered qualification gaps do not wait behind all metadata-only jobs.
            for start in range(0, min(len(jobs), 2000), 2000):
                batch = jobs[start:start + 2000]
                mapping = mapping_routes(batch)
                if mapping:
                    route_folder = FOLDER / 'entity_review_routes'
                    route_folder.mkdir(exist_ok=True)
                    for pmcid, proofs in mapping.items():
                        receipt = route_folder / (pmcid + '.json')
                        atomic_json(receipt, {'pmcid': pmcid, 'papers': proofs,
                            'status': 'existing_native_fields_present_entity_review_required',
                            'network_requests': 0, 'final_task_closure': False,
                            'reopen_on_new_identity_or_signature_conflict': True})
                        state.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?)',
                            (pmcid, 'entity_review_required_existing_native_fields', str(receipt.relative_to(ROOT))))
                    state.commit()
                    batch = [(pmcid, papers) for pmcid, papers in batch if pmcid not in mapping]
                with (CLOUD / 'version_collector.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    with ThreadPoolExecutor(max_workers=32) as pool:
                        results = list(pool.map(acquire, batch))
                handled = []
                for (pmcid, papers), result in zip(batch, results):
                    path = CLOUD / 'version_results' / (pmcid + '.1.json')
                    if path.exists():
                        associated = {(p['work_id'], p['doi']) for p in result.get('paper_links', [])}
                        if all((p['work_id'], p['doi']) in associated for p in papers):
                            handled.append(path)
                        else:
                            state.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?)', (pmcid, 'cached_association_review_required', str(path.relative_to(ROOT))))
                with (CLOUD / 'xml_collector.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    consumed_path = CLOUD / 'xml_consumed.json'
                    consumed = set(json.loads(consumed_path.read_text())) if consumed_path.exists() else set()
                    pending = [path for path in handled if str(path.relative_to(ROOT)) not in consumed]
                    if pending:
                        consumed.update(apply_batch(pending))
                        atomic_json(consumed_path, sorted(consumed))
                    for path in handled:
                        rel = str(path.relative_to(ROOT))
                        if rel in consumed:
                            state.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?)', (path.name.split('.')[0], 'metadata_and_available_xml_path_processed', rel))
                state.commit()
                atomic_json(FOLDER / 'archive_status.json', {'status': 'processing_new_archive_leads',
                    'current_discovered_identifiers': len(jobs), 'outcomes': dict(state.execute('SELECT status,count(*) FROM jobs GROUP BY status')),
                    'metadata_workers': 32, 'final_task_closure': False, 'full_refresh': 'deferred_until_combined_phase_end'})
                if STOP.is_set():
                    break
            status_path = FOLDER / 'dispatch_status.json'
            if not jobs and status_path.exists() and json.loads(status_path.read_text())['status'] == 'fixed_locations_dispatched':
                atomic_json(FOLDER / 'archive_status.json', {'status': 'discovered_archive_leads_processed_other_paths_pending',
                    'outcomes': dict(state.execute('SELECT status,count(*) FROM jobs GROUP BY status')), 'final_task_closure': False})
                break
            time.sleep(20)
        state.close(); links.close()


if __name__ == '__main__':
    run()
