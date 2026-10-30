"""Apply official PLOS types; keep undifferentiated publication dates explicit."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from datetime import date

from .fig8b_qualification_build import plain
from .publisher_delta_apply import apply_available
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def metadata(doc: dict) -> dict:
    abstract = doc.get('abstract_primary_display') or []
    # The general abstract field can concatenate author summaries; never fall back.
    text = plain(abstract[0]) if isinstance(abstract, list) and len(abstract) == 1 else ''
    published = doc.get('publication_date', '')[:10]
    try:
        date.fromisoformat(published)
    except ValueError:
        published = ''
    return {'doi': [doc['id']], 'title': [plain(doc.get('title_display', ''))],
            'issns': [doc.get('eissn', '')], 'journal_titles': [doc.get('journal', '')],
            'article_types': [doc.get('article_type', 'unspecified')],
            'online_date': '', 'date_precision': 'missing_explicit_electronic_date',
            'publication_dates': [published] if published else [],
            'publisher': [doc.get('publisher', '')], 'abstract': text,
            'date_basis': 'Official API publication_date; not automatically classified as electronic',
            'authors': [], 'byline_has_etal': True,
            'author_link_limitation': 'API author and affiliate fields do not supply trustworthy author-unit links'}


def run() -> dict:
    folder = ROOT / 'supplement/plos_batch'
    records = folder / 'records'
    records.mkdir(exist_ok=True)
    allowed = {p['doi'] for p in json.loads((folder / 'scope.json').read_text())['papers']}
    jobs = []
    outcomes = []
    for path in sorted(folder.glob('batch_[0-9][0-9][0-9][0-9].json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('Batch hash mismatch')
        response = json.loads(raw)['response']
        if response['numFound'] != len(response['docs']):
            raise ValueError('Unprocessed pagination')
        found = set()
        for doc in response['docs']:
            doi = doc.get('id')
            if doi not in allowed or doi not in receipt['dois'] or doi in found:
                raise ValueError('Off-scope or duplicate DOI response')
            found.add(doi)
            record = {'doi': doi, 'source': 'publisher', 'route': 'plos',
                      'url': receipt['url'], 'batch_receipt': str(path.relative_to(ROOT)),
                      'raw_receipt': os.path.relpath(ROOT / receipt['raw_receipt'], ROOT / 'fields/qualification'),
                      'raw_sha256': receipt['sha256'], 'parsed': metadata(doc),
                      'source_locator': 'response.docs[id=' + doi + ']'}
            token = hashlib.sha256((doi + receipt['sha256']).encode()).hexdigest()
            target = records / (token + '.json')
            if not target.exists():
                atomic_json(target, record)
            jobs.append((doi, str(target.relative_to(ROOT))))
        outcomes.extend({'doi': doi, 'batch_receipt': str(path.relative_to(ROOT)),
                         'status': 'returned_requires_identity_and_qualification_merge' if doi in found else 'not_returned_by_official_search'}
                        for doi in receipt['dois'])
    atomic_json(folder / 'batch_doi_outcomes.json', outcomes)
    result = apply_available(supplied_jobs=jobs)
    atomic_json(folder / 'apply_status.json', result)
    return result


if __name__ == '__main__':
    from .derived_refresh import refresh
    run()
    refresh()
