"""Dispatch recovered cached PMC leads without duplicating the existing collector."""
from __future__ import annotations

import fcntl
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .pmc_cloud_inventory import FOLDER as CLOUD
from .pmc_cloud_versions import STOP
from .priority_cloud_leads import acquire
from .secondary_markup_recovery import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'dispatch.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (FOLDER / 'discovery.lock').open('a') as discovery:
            fcntl.flock(discovery, fcntl.LOCK_EX)
            state = json.loads((FOLDER / 'status.json').read_text())
            if state['status'] != 'fixed_cache_discovery_processed':
                raise ValueError('Cache recovery did not finish')
        scope_path = FOLDER / 'dispatch_scope.json'
        if not scope_path.exists():
            existing = {pmc for paper in json.loads((CLOUD / 'scope.json').read_text())['papers']
                        for pmc in paper['pmcids']}
            groups = defaultdict(dict)
            for path in sorted((FOLDER / 'parts').glob('*.json')):
                for paper in json.loads(path.read_text())['leads']:
                    groups[paper['pmcid']][paper['work_id']] = paper
            jobs = {}; reused = {}; queued = []
            for pmcid, group in groups.items():
                target = CLOUD / 'version_results' / (pmcid + '.1.json')
                if target.exists():
                    result = json.loads(target.read_text())
                    linked = {p['work_id'] for p in result.get('paper_links', [])}
                    reused[pmcid] = {'receipt': str(target.relative_to(ROOT)),
                        'covered_work_ids': sorted(set(group) & linked),
                        'unresolved_additional_work_ids': sorted(set(group) - linked)}
                elif pmcid in existing:
                    queued.append(pmcid)
                else:
                    jobs[pmcid] = list(group.values())
            atomic_json(scope_path, {'jobs': jobs, 'existing_results': reused,
                'existing_collector_pmcids': queued, 'recovered_unique_pmcids': len(groups),
                'qualification_unchanged': True})
        scope = json.loads(scope_path.read_text())
        jobs = sorted(scope['jobs'].items(), key=lambda item: min(
            {'Science': 0, 'Cell': 1, 'Proceedings of the National Academy of Sciences': 2}.get(p['journal'], 3)
            for p in item[1]))
        marker = FOLDER / 'dispatch_status.json'
        atomic_json(marker, {'status': 'waiting_for_existing_metadata_collector',
            'fixed_new_pmcids': len(jobs), 'existing_results': len(scope['existing_results']),
            'already_queued': len(scope['existing_collector_pmcids']), 'final_task_closure': False})
        counts = Counter()
        with (CLOUD / 'version_collector.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pending = []
            for pmcid, papers in jobs:
                path = CLOUD / 'version_results' / (pmcid + '.1.json')
                if path.exists():
                    counts[json.loads(path.read_text())['status']] += 1
                else:
                    pending.append((pmcid, papers))
            reused = sum(counts.values())
            queue = iter(pending)
            last_status = 0.0
            with ThreadPoolExecutor(max_workers=64) as pool:
                active = set()
                def submit() -> None:
                    if STOP.is_set() or (CLOUD / 'source_stop.json').exists():
                        return
                    item = next(queue, None)
                    if item is not None:
                        active.add(pool.submit(acquire, item))
                for _ in range(64):
                    submit()
                while active:
                    ready, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in ready:
                        active.remove(future)
                        result = future.result()
                        if result['receipt'].get('status') != 'not_requested_after_source_stop':
                            counts[result['status']] += 1
                        submit()
                    now = time.monotonic()
                    if now - last_status >= 5.0 or not active:
                        atomic_json(marker, {'status': 'reading_new_metadata', 'fixed_new_pmcids': len(jobs),
                            'processed': sum(counts.values()), 'outcomes': dict(counts), 'workers': 64,
                            'cached_results_reused': reused, 'scheduling': 'rolling_bounded_futures',
                            'final_task_closure': False})
                        last_status = now
        atomic_json(marker, {'status': 'fixed_metadata_path_processed' if sum(counts.values()) == len(jobs)
            else 'source_stopped_unattempted_retained', 'fixed_new_pmcids': len(jobs),
            'processed': sum(counts.values()), 'outcomes': dict(counts),
            'xml_application': 'Shared XML consumer; metadata is not qualification evidence',
            'final_task_closure': False})


if __name__ == '__main__':
    run()
