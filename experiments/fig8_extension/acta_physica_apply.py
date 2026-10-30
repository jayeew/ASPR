"""Interpret literal native types without promoting print/preprint dates."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict

from .acta_physica_bulk import FOLDER
from .cached_date_apply import run as apply_sources
from .fig8b_qualification_build import key, plain
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

TYPE_LABELS = {'研究论文': 'Research article', '综述': 'Review'}


def journal_identity() -> dict:
    raw = gzip.decompress((FOLDER / 'issue_probe.html.gz').read_bytes()).decode()
    start = raw.index('var web_common_data=') + len('var web_common_data=')
    journal = json.JSONDecoder().raw_decode(raw[start:])[0]['journal']
    if journal['titleEn'] != 'Acta Physica Sinica' or journal['issnEpub'] != '1000-3290':
        raise ValueError('Official journal identity no longer matches this fixed adapter')
    return journal


def native_source(article: dict, paper: dict, receipt: dict, journal: dict) -> dict:
    titles = [plain(article.get(k) or '') for k in ['titleEn', 'titleCn']]
    title = next((t for t in titles if key(t) == key(paper['title'])), titles[0])
    literal = article.get('type') or ''
    return {'family': 'publisher_acta_physica', 'doi': paper['doi'], 'title': title,
        'journal_titles': [journal['titleEn'], journal['titleCn']],
        'issns': [journal['issnEpub']], 'kind': TYPE_LABELS.get(literal, literal),
        'native_kind': literal, 'categories': [], 'online_date': '', 'date_precision': 'missing',
        'date_limit': 'Issue print date, accepted preprint and release timestamps are not final online-date evidence',
        'receipt': '../../' + receipt['raw_receipt'], 'url': receipt['url'],
        'journal_identity_receipt': 'supplement/acta_physica_native/issue_probe.html.gz',
        'type_translation_rule': 'Literal 研究论文 = Research article; 综述 = Review; other labels unchanged',
        'article_id': article.get('id')}


def prepare() -> tuple[dict, dict]:
    scope = {r['doi']: r for r in json.loads((FOLDER / 'scope.json').read_text())['records']}
    journal = journal_identity(); sources = defaultdict(list); types = Counter(); reviews = []
    for path in sorted((FOLDER / 'issues').glob('*.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200 or not receipt.get('raw_receipt'):
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        if receipt.get('sha256') and hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('Cached native source hash mismatch')
        response = json.loads(raw); data = response.get('data') or {}; catalog = data.get('catalog') or {}
        expected = receipt['data']
        if (response.get('result') != 'success' or catalog.get('journalId') != journal['id']
                or str(catalog.get('year')) != expected['year'] or str(catalog.get('issue')) != expected['issue']):
            reviews.append({'receipt': receipt['raw_receipt'], 'status': 'issue_identity_mismatch'}); continue
        for article in (data.get('pagerFilter') or {}).get('records') or []:
            doi = (article.get('doi') or '').casefold()
            if doi not in scope:
                continue
            sources[doi].append(native_source(article, scope[doi], receipt, journal))
            types[article.get('type') or ''] += 1
    return dict(sources), {'candidate_dois': len(scope), 'cached_dois': len(sources),
        'unmatched_dois': sorted(set(scope) - set(sources)), 'native_types': dict(types),
        'issue_reviews': reviews, 'network_requests': 0, 'final_task_closure': False}


def run() -> dict:
    status = json.loads((FOLDER / 'status.json').read_text())
    if status['status'] not in {'complete', 'source_hold'}:
        raise RuntimeError('Fixed issue collector is still running; defer coalesced writes')
    sources, review = prepare()
    atomic_json(FOLDER / 'qualification_sources.json', sources)
    atomic_json(FOLDER / 'source_review.json', review)
    result = apply_sources(set(sources), FOLDER, explicit_sources=sources)
    atomic_json(FOLDER / 'qualification_application.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
