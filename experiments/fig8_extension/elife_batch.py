"""Cache eLife official article-list batches against the fixed existing DOI scope."""
from __future__ import annotations

import fcntl
import gzip
import json
import math
from concurrent.futures import ThreadPoolExecutor

from . import ojs_metadata_harvest as transport
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/elife_official'
URL = 'https://api.elifesciences.org/articles'


def fetch_page(page: int, url: str = URL) -> tuple[int, dict]:
    return page, transport.fetch(url, {'per-page': 100, 'page': page})


def run(endpoint: str = 'articles') -> None:
    if endpoint not in {'articles', 'reviewed-preprints'}:
        raise ValueError('Unsupported official eLife collection')
    folder = FOLDER if endpoint == 'articles' else FOLDER / 'reviewed_preprints'
    folder.mkdir(exist_ok=True)
    transport.FOLDER = folder
    transport.RAW = folder / 'raw'
    transport.RAW.mkdir(exist_ok=True)
    url = 'https://api.elifesciences.org/' + endpoint
    targets = {p['doi'].lower(): p for p in json.loads((FOLDER / 'scope.json').read_text())['papers']}
    first_path = folder / 'articles.json'
    if not first_path.exists():
        atomic_json(first_path, fetch_page(1, url)[1])
    first = json.loads(first_path.read_text())
    if first.get('http_status') != 200:
        atomic_json(folder / 'batch_progress.json', {'status': 'initial_request_failed', 'receipt': first})
        return
    first_data = json.loads(gzip.decompress((ROOT / first['raw_receipt']).read_bytes()))
    total = first_data['total']
    plan_path = folder / 'batch_plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {
        'pages': math.ceil(total / 100), 'initial_api_total': total,
        'target_dois': len(targets), 'scope_expansion': False,
        'limitation': 'Live pagination may shift; duplicates and missing targets remain explicit',
    }
    atomic_json(plan_path, plan)
    checkpoint = folder / 'batch_progress.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        'completed_pages': [], 'matched': {}, 'status': 'running', 'duplicate_records': 0}
    remaining = [p for p in range(1, plan['pages'] + 1) if p not in state['completed_pages']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(remaining), 4):
            for page, receipt in pool.map(lambda page: fetch_page(page, url), remaining[offset:offset + 4]):
                if receipt.get('http_status') != 200:
                    state.update(status='source_stopped_unattempted_retained', failure=receipt)
                    atomic_json(checkpoint, state)
                    return
                payload = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
                for item in payload.get('items', []):
                    doi = item.get('doi', '').lower()
                    if doi not in targets:
                        continue
                    row = {'candidate': targets[doi], 'native': item,
                           'raw_receipt': receipt['raw_receipt'], 'page': page,
                           'status': 'cached_identity_type_date_review_required'}
                    previous = state['matched'].setdefault(doi, [])
                    if any(p['native'] == item for p in previous):
                        state['duplicate_records'] += 1
                    else:
                        previous.append(row)
                state['completed_pages'].append(page)
                atomic_json(checkpoint, state)
            print(json.dumps({'pages': len(state['completed_pages']), 'matched': len(state['matched'])}), flush=True)
    state.update(status='fixed_batch_complete_review_required',
                 unmatched_dois=sorted(set(targets) - set(state['matched'])))
    atomic_json(checkpoint, state)


if __name__ == '__main__':
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
