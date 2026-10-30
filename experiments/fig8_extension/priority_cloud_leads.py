"""Check fixed new PMC leads through public version metadata, preserving version limits."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .pmc_cloud_inventory import FOLDER as CLOUD
from .pmc_cloud_metadata import assess_version
from .pmc_cloud_versions import STOP, fetch_metadata
from .pmc_cloud_xml import apply_batch
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/priority_locations'


def acquire(item: tuple[str, list[dict]]) -> dict:
    pmcid, papers = item
    target = CLOUD / 'version_results' / (pmcid + '.1.json')
    if target.exists():
        return json.loads(target.read_text())
    receipt = fetch_metadata(f'metadata/{pmcid}.1.json')
    result = {'inventory': {'pmcid': pmcid, 'version': 1, 'metadata_key': f'metadata/{pmcid}.1.json',
        'discovery_basis': 'Known public version-one object pattern; not an inventory match or latest-version claim'},
        'receipt': receipt, 'assessments': [], 'paper_links': [{'work_id': p['work_id'], 'doi': p['doi']} for p in papers],
        'status': 'version_metadata_unavailable', 'final_task_closure': False}
    if receipt.get('http_status') == 200:
        try:
            metadata = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
            if not isinstance(metadata, dict) or str(metadata.get('version')) != '1':
                raise ValueError('Requested object version does not match metadata version')
        except (ValueError, OSError) as error:
            result.update(status='metadata_version_or_format_conflict', error=str(error))
        else:
            result.update(status='version_metadata_reviewed', metadata=metadata,
                assessments=[{**assess_version(metadata, {**p, 'pmcid': pmcid}),
                              'work_id': p['work_id'], 'doi': p['doi']} for p in papers])
    if receipt.get('status') != 'not_requested_after_source_stop':
        atomic_json(target, result)
    return result


def run() -> None:
    with (FOLDER / 'cloud_leads.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = {r['work_id']: r for r in json.loads((FOLDER / 'outcomes.json').read_text())}
        pmcs = json.loads((FOLDER / 'pmc_cache_review.json').read_text())['pmcids']
        existing = {v for p in json.loads((CLOUD / 'scope.json').read_text())['papers'] for v in p['pmcids']}
        jobs = [(pmcid, [rows[w] for w in work_ids]) for pmcid, work_ids in pmcs.items() if pmcid not in existing]
        jobs.sort(key=lambda item: min({'Science': 0, 'Cell': 1}.get(p['journal'], 2) for p in item[1]))
        atomic_json(FOLDER / 'cloud_leads_scope.json', {'pmcids': [p for p, _ in jobs],
            'scope_papers': len({p['work_id'] for _, group in jobs for p in group}),
            'discovery_source': 'supplement/priority_locations/outcomes.json',
            'version_limit': 'Probe version 1 only; unavailable or manuscript does not mean no later published version exists.'})
        counts = Counter(); handled = []
        with (CLOUD / 'version_collector.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with ThreadPoolExecutor(max_workers=16) as pool:
                for start in range(0, len(jobs), 16):
                    if STOP.is_set() or (CLOUD / 'source_stop.json').exists():
                        break
                    batch = list(pool.map(acquire, jobs[start:start + 16]))
                    for (pmcid, _), result in zip(jobs[start:start + 16], batch):
                        counts[result['status']] += 1
                        path = CLOUD / 'version_results' / (pmcid + '.1.json')
                        if path.exists():
                            handled.append(path)
                    atomic_json(FOLDER / 'cloud_leads_status.json', {'status': 'reading_metadata',
                        'fixed_pmcids': len(jobs), 'processed': sum(counts.values()), 'outcomes': dict(counts),
                        'final_task_closure': False})
        with (CLOUD / 'xml_collector.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done_path = CLOUD / 'xml_consumed.json'
            done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
            pending = [p for p in handled if str(p.relative_to(ROOT)) not in done]
            if pending:
                done.update(apply_batch(pending))
                atomic_json(done_path, sorted(done))
        atomic_json(FOLDER / 'cloud_leads_status.json', {'status': 'fixed_version_one_path_reviewed' if len(handled) == len(jobs) else 'source_stopped_unattempted_retained',
            'fixed_pmcids': len(jobs), 'processed': len(handled), 'outcomes': dict(counts),
            'final_task_closure': False, 'version_limit': 'Later versions and nonpublic manuscripts remain independent unresolved paths.'})


if __name__ == '__main__':
    run()
