"""Adapt cached eLife native author-affiliation arrays to the existing mapper."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3

from .elife_batch import FOLDER
from .elife_text import plain
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def metadata(article: dict) -> dict:
    authors = []
    incomplete = False
    for author in article.get('authors', []):
        name = author.get('name', {})
        name = name.get('preferred', '') if isinstance(name, dict) else name
        if author.get('type') not in {'person', 'group'} or not name:
            incomplete = True
            continue
        affiliations = []
        for affiliation in author.get('affiliations', []):
            address = affiliation.get('address', {})
            country = address.get('components', {}).get('country', '')
            parts = affiliation.get('name', []) + address.get('formatted', [])
            affiliations.append({'text': ', '.join(plain(v) for v in parts),
                                 'countries': [{'code': '', 'text': country}] if country else [],
                                 'names': affiliation.get('name', []), 'identifiers': []})
        authors.append({'name': plain(name), 'nonpersonal': author['type'] != 'person',
                        'orcid': author.get('orcid', ''), 'affiliations': affiliations,
                        'unresolved_affiliation_refs': []})
    abstract = '\n\n'.join(plain(block['text']) for block in article.get('abstract', {}).get('content', [])
                            if block.get('type') == 'paragraph' and block.get('text'))
    return {'abstract': abstract, 'doi': [article['doi'].lower()], 'title': [plain(article['title'])],
            'journal_titles': ['eLife'], 'issns': [], 'authors': authors,
            'byline_has_etal': incomplete or not authors,
            'byline_check': 'Publisher native complete authors array with per-author affiliations'}


def run() -> dict:
    folder = FOLDER / 'author_adapters'
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    jobs = []
    for row in json.loads((FOLDER / 'detail_receipts.json').read_text()):
        response = row['response']
        if response.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / response['raw_receipt']).read_bytes())
        article = json.loads(raw)
        if article.get('doi', '').lower() != row['doi']:
            continue
        path = folder / (row['doi'].rsplit('.', 1)[1] + '.json')
        atomic_json(path, {'source': 'publisher_api', 'route': 'elife_api',
                           'raw_receipt': '../../' + response['raw_receipt'],
                           'raw_sha256': hashlib.sha256(raw).hexdigest(),
                           'parsed': metadata(article), 'url': response['url']})
        paper = db.execute('SELECT shard,status FROM papers WHERE work_id=?', (row['work_id'],)).fetchone()
        if paper and paper[1] == 'included_verified_research':
            jobs.append((row['work_id'], paper[0], str(path.relative_to(ROOT))))
    db.close()
    result = apply_authors(selected_jobs=jobs)
    atomic_json(folder / 'status.json', result)
    return result


if __name__ == '__main__':
    run()
