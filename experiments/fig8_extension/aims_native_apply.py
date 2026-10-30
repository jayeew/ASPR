"""Apply fixed AIMS page evidence and DOI-linked cached online-date deposits."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from .aims_native_collect import FOLDER, parse
from .cached_date_apply import deposited_title, run as apply_sources
from .fig8b_qualification_build import plain
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def sources_from_pages() -> tuple[dict, list[dict]]:
    sources = defaultdict(list); reviews = []
    allowed = {r['doi'] for r in json.loads((FOLDER / 'scope.json').read_text())['records']}
    for path in sorted((FOLDER / 'pages').glob('*.json')):
        record = json.loads(path.read_text()); doi = record['doi']
        if doi not in allowed or record['status'] != 'cached_for_identity_review':
            continue
        raw = gzip.decompress((ROOT / record['raw_receipt']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record['raw_sha256']
        parsed = parse(raw, doi)
        if not parsed['doi_matched']:
            reviews.append({'doi': doi, 'status': 'native_doi_mismatch', 'receipt': record['raw_receipt']})
            continue
        # Issue-navigation labels share the same span class as article type.
        kinds = [v for v in parsed['literal_types'] if v not in {'Topical Sections'}]
        for kind in kinds:
            sources[doi].append({
                'family': 'publisher_aims', 'doi': doi,
                'title': parsed['title'][0] if parsed['title'] else '',
                'issns': parsed['issns'], 'journal_titles': parsed['journal_titles'],
                'kind': kind, 'categories': [], 'online_date': parsed['online_date'],
                'date_precision': parsed['date_precision'],
                'receipt': '../../' + record['raw_receipt'], 'url': record['url'],
                'raw_sha256': record['raw_sha256'],
                'published_labels_not_online_evidence': parsed['published_labels_not_online_evidence']})
        reviews.append({'doi': doi, 'status': 'native_fields_for_identity_review',
                        'literal_types': kinds, 'receipt': record['raw_receipt']})
    return dict(sources), reviews


def add_cached_dates(sources: dict) -> list[dict]:
    db = connect(); paths = defaultdict(set)
    for doi in sources:
        for receipt, in db.execute("SELECT receipt FROM source_metadata WHERE doi=? AND kind='deposit'", (doi,)):
            paths[receipt.split('#', 1)[0]].add(doi)
    db.close(); reviews = []
    for receipt, dois in paths.items():
        with gzip.open(ROOT / receipt, 'rt') as stream:
            record = json.load(stream)
        items = {v['DOI'].lower(): v for v in record.get('response', {}).get('message', {}).get('items', [])}
        for doi in dois:
            item = items.get(doi, {}); found = False
            for parts in item.get('published-online', {}).get('date-parts', []):
                if not parts or len(parts) > 3 or not all(isinstance(v, int) for v in parts):
                    continue
                try:
                    date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
                except ValueError:
                    continue
                sources[doi].append({'family': 'crossref_publisher_deposit',
                    'title': deposited_title((item.get('title') or [''])[0]),
                    'issns': item.get('ISSN', []), 'journal_titles': item.get('container-title', []),
                    'kind': item.get('type', 'unspecified'), 'categories': [],
                    'online_date': '-'.join(str(v).zfill(4 if i == 0 else 2) for i, v in enumerate(parts)),
                    'date_precision': ['year', 'month', 'day'][len(parts)-1],
                    'receipt': '../../' + receipt, 'url': record['url']})
                found = True
            reviews.append({'doi': doi, 'receipt': receipt, 'explicit_online_date_found': found})
    return reviews


def run() -> dict:
    state = json.loads((FOLDER / 'issue_status.json').read_text())
    if state['status'] not in {'complete', 'source_hold'}:
        raise RuntimeError('Fixed bulk collection is still running; coalesce writes at its end')
    sources, reviews = sources_from_pages()
    allowed = {r['doi'] for r in json.loads((FOLDER / 'scope.json').read_text())['records']}
    reuse_path = FOLDER / 'additional_existing_candidate_reuse.json'
    reused_candidates = json.loads(reuse_path.read_text())['records'] if reuse_path.exists() else []
    allowed.update(r['doi'] for r in reused_candidates)
    db = connect()
    for path in sorted((FOLDER / 'issues').glob('*.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200 or not receipt.get('raw_receipt'):
            continue
        raw_path = Path(receipt['raw_receipt'])
        data = json.loads(gzip.decompress(raw_path.read_bytes()))
        payload = data.get('data')
        if not isinstance(payload, dict):
            continue
        for article in payload.get('articles', []):
            doi = (article.get('doi') or '').casefold()
            if doi not in allowed:
                candidates = db.execute('SELECT work_id,status FROM papers WHERE doi=?', (doi,)).fetchall()
                if len(candidates) != 1 or not candidates[0][1].startswith('pending'):
                    continue
                allowed.add(doi)
                reused_candidates.append({'doi': doi, 'work_id': candidates[0][0],
                                          'prior_status': candidates[0][1], 'receipt': str(raw_path.relative_to(ROOT))})
            journal = article.get('journalMeta') or {}
            online = (article.get('onlineDate') or '')[:10]
            if online:
                try:
                    date.fromisoformat(online)
                except ValueError:
                    online = ''
            sources.setdefault(doi, []).append({
                'family': 'publisher_aims', 'doi': doi,
                'title': plain(article.get('titleEn') or ''),
                'issns': [journal[k] for k in ['issnEpub', 'issnPpub'] if journal.get(k)],
                'journal_titles': [journal['titleEn']] if journal.get('titleEn') else [],
                'kind': article.get('classifyinfoEn') or '', 'categories': [],
                'online_date': online, 'date_precision': 'day' if online else 'missing',
                'receipt': '../../' + str(raw_path.relative_to(ROOT)), 'url': receipt['url'],
                'date_field': 'onlineDate', 'article_id': article.get('id'),
                'publishers': [journal['publisherNameEn']] if journal.get('publisherNameEn') else []})
    db.close()
    atomic_json(FOLDER / 'additional_existing_candidate_reuse.json', {
        'records': reused_candidates, 'network_requests': 0, 'new_candidate_rows': 0,
        'scope': 'Already in current three-field roster; incidental bulk DOI matches require normal identity and qualification checks'})
    cached_dates = add_cached_dates(sources)
    atomic_json(FOLDER / 'source_review.json', {'pages': reviews, 'deposits': cached_dates,
                                              'network_requests': 0, 'final_task_closure': False})
    atomic_json(FOLDER / 'qualification_sources.json', sources)
    result = apply_sources(set(sources), FOLDER, explicit_sources=sources)
    atomic_json(FOLDER / 'qualification_application.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
