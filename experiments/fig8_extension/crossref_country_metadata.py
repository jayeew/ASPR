"""Read deposited author-address links without claiming a complete native byline."""
from __future__ import annotations

import gzip
import json

from .supplement_queue import ROOT


def parse(record: dict, doi: str) -> dict:
    if record.get('http_status') != 200:
        return {}
    message = record.get('response', {}).get('message', {})
    items = message.get('items', [message])
    matches = [p for p in items if p.get('DOI', '').casefold() == doi.casefold()]
    if len(matches) != 1:
        return {}
    item = matches[0]
    authors = []
    for author in item.get('author', []):
        name = ' '.join(str(author.get(k, '')).strip() for k in ('given', 'family')).strip()
        if not author.get('given') or not author.get('family'):
            continue
        authors.append({'name': name, 'affiliations': [
            {'text': a['name'], 'countries': []} for a in author.get('affiliation', [])
            if a.get('name')]})
    return {'doi': [item['DOI'].casefold()], 'title': item.get('title', []),
            'issns': item.get('ISSN', []), 'journal_titles': item.get('container-title', []),
            'authors': authors}


def load(receipt: str, doi: str) -> dict:
    if not receipt.startswith('fields/crossref_raw/') or not receipt.endswith('.json.gz'):
        return {}
    return parse(json.loads(gzip.decompress((ROOT / receipt).read_bytes())), doi)
