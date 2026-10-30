"""Discover public repository leads for the fixed priority scope, without eligibility changes."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

from .fig8b_qualification_decide import key
from .ojs_metadata_harvest import fetch
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/priority_locations'


def matched(paper: dict, work: dict) -> bool:
    return (work.get('id') == paper['work_id']
            and (work.get('doi') or '').removeprefix('https://doi.org/').lower() == paper['doi'].lower()
            and key(work.get('title')) == key(paper['title']))


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
        records = json.loads((FOLDER / 'cached_records.json').read_text())
        pending = [p for p in scope if p['work_id'] not in records]
        batches = []; stop = ''
        for start in range(0, len(pending), 100):
            group = pending[start:start + 100]
            params = {'filter': 'ids.openalex:' + '|'.join(p['work_id'].rsplit('/', 1)[-1] for p in group),
                      'per_page': 100, 'select': 'id,doi,title,locations,open_access,is_retracted'}
            # Anonymous calls cannot spend a user's prepaid API credit balance.
            receipt = fetch('https://api.openalex.org/works', params)
            batches.append({'work_ids': [p['work_id'] for p in group], 'receipt': receipt})
            if receipt.get('http_status') != 200:
                stop = 'anonymous_batch_source_unavailable'; break
            try:
                payload = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
                results = payload['results']
                if not isinstance(results, list):
                    raise ValueError('Results are not a list')
            except (KeyError, ValueError, OSError) as error:
                stop = 'invalid_batch_response:' + type(error).__name__; break
            requested = {p['work_id'] for p in group}
            for work in results:
                if work.get('id') in requested:
                    records[work['id']] = {'work': work, 'receipt': receipt['raw_receipt']}
            atomic_json(FOLDER / 'batches.json', batches)
            atomic_json(FOLDER / 'records.json', records)
            atomic_json(FOLDER / 'status.json', {'status': 'collecting', 'fixed_papers': len(scope),
                'batches_processed': len(batches), 'records_returned': len(records), 'qualification_changes': 0})
        outcomes = []; counts = Counter()
        for paper in scope:
            record = records.get(paper['work_id'])
            outcome = {**paper, 'qualification_unchanged': True, 'final_task_closure': False}
            if record is None:
                outcome.update(status='record_not_returned_or_unattempted', locations=[])
            elif not matched(paper, record['work']):
                outcome.update(status='current_metadata_identity_conflict', locations=[], receipt=record['receipt'])
            else:
                work = record['work']
                locations = [v for v in work.get('locations', []) if v.get('is_oa')
                             and (v.get('source') or {}).get('type') == 'repository'
                             and (v.get('pdf_url') or v.get('landing_page_url'))]
                outcome.update(status='public_repository_leads_found' if locations else 'no_public_repository_lead_in_response',
                    locations=locations, receipt=record['receipt'], current_retraction_flag=work.get('is_retracted'))
            counts[outcome['status']] += 1; outcomes.append(outcome)
        atomic_json(FOLDER / 'batches.json', batches)
        atomic_json(FOLDER / 'records.json', records)
        atomic_json(FOLDER / 'outcomes.json', outcomes)
        atomic_json(FOLDER / 'status.json', {'status': 'source_stopped' if stop else 'fixed_location_discovery_processed',
            'stop_reason': stop, 'fixed_papers': len(scope), 'cached_full_records_reused': 19,
            'batches_processed': len(batches), 'outcomes': dict(counts), 'qualification_changes': 0,
            'final_task_closure': False, 'limitation': 'Repository labels and OA flags are discovery leads, not publication-type or first-publication evidence.'})


if __name__ == '__main__':
    run()
