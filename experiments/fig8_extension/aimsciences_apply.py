"""Apply the complete fixed page harvest, retaining unspecified research types."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter

from .aimsciences_pages import FOLDER
from .aimsciences_parse import native_metadata
from .cached_date_apply import run as apply_sources
from .supplement_apply import apply as apply_metadata
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def prepare() -> tuple[dict, list[dict], list[dict]]:
    allowed = {r['doi'] for r in json.loads((FOLDER / 'page_scope.json').read_text())['records']}
    sources = {}; native = []; reviews = []
    for path in sorted((FOLDER / 'pages').glob('*.json')):
        record = json.loads(path.read_text()); doi = record['doi']
        if doi not in allowed or not record.get('raw_receipt'):
            continue
        raw = gzip.decompress((ROOT / record['raw_receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != record['raw_sha256']:
            raise ValueError('Native page hash changed: ' + str(path))
        parsed = native_metadata(raw)
        if parsed['doi'] != [doi]:
            reviews.append({'doi': doi, 'status': 'source_doi_mismatch', 'receipt': str(path)})
            continue
        parsed.update(evidence_receipt=record['raw_receipt'], source_url=record['url'])
        native.append({'doi': doi, 'payload': parsed, 'raw_hash': record['raw_sha256']})
        sources[doi] = [{'family': 'publisher_aimsciences', 'doi': doi,
            'title': (parsed['title'] or [''])[0], 'issns': parsed['issns'],
            'journal_titles': parsed['journal_titles'], 'kind': kind, 'categories': [],
            'online_date': parsed['online_date'], 'date_precision': parsed['date_precision'],
            'online_history_labels': parsed['online_history_labels'],
            'receipt': '../../' + record['raw_receipt'], 'url': record['url'],
            'raw_sha256': record['raw_sha256']} for kind in parsed['article_types'] or ['unspecified']]
        reviews.append({'doi': doi, 'status': 'native_evidence_for_identity_matching',
            'explicit_online_date': parsed['online_date'], 'literal_types': parsed['article_types'],
            'abstract_present': bool(parsed['abstract']), 'authors': len(parsed['authors']),
            'unresolved_author_links': sum(bool(a['unresolved_affiliation_refs']) for a in parsed['authors']),
            'receipt': record['raw_receipt']})
    return sources, native, reviews


def run() -> dict:
    state = json.loads((FOLDER / 'page_status.json').read_text())
    if state['status'] not in {'complete', 'source_hold'}:
        raise RuntimeError('Fixed page collection still running')
    sources, native, reviews = prepare()
    folder = FOLDER / 'page_application'; folder.mkdir(exist_ok=True)
    atomic_json(folder / 'sources.json', sources)
    atomic_json(folder / 'review.json', reviews)
    qualification = apply_sources(set(sources), folder, explicit_sources=sources)
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = connect(); qualified = set()
        for item in native:
            payload = item['payload']; doi = item['doi']
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                (payload['evidence_receipt'] + '#' + doi, doi, 'jats', item['raw_hash'],
                 json.dumps(payload, ensure_ascii=False)))
            counts['native_metadata_indexed'] += 1
            if db.execute("SELECT 1 FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchone():
                qualified.add(doi)
        db.commit(); db.close()
        metadata = apply_metadata(ROOT, scope_dois=qualified, index_sources=False) if qualified else {}
    result = {'qualification': qualification, **counts, 'qualified_scope': len(qualified),
              'metadata': metadata, 'network_requests': 0, 'final_task_closure': False}
    atomic_json(folder / 'application.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
