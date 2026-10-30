"""Normalize already acquired publisher-directory records without date inference."""
from __future__ import annotations

import gzip
import json
from functools import lru_cache

from .fig8b_qualification_build import plain
from .supplement_queue import ROOT


@lru_cache(maxsize=32)
def receipt_records(reference: str) -> dict:
    path = ROOT / reference
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            document = json.load(stream)
        if document.get('http_status') != 200:
            return {}
        return {r.get('identifiers', {}).get('doi', '').lower(): {
            'doi': r.get('identifiers', {}).get('doi', '').lower(),
            'title': plain(r.get('title', {}).get('value', '')),
            'publisher_type': r.get('articleType', ''), 'publisher_date': r.get('date', ''),
            'catalogue_url': document['url'], 'family': 'aps',
            'journal_titles': [r.get('journal', {}).get('name', '')],
            'aps_journal': r.get('journal', {}).get('id', ''),
            'categories': [(r.get('tocSection') or {}).get('label', '')]}
            for r in document.get('response', {}).get('data', [])}
    document = json.loads(path.read_text())
    return {r['doi']: {**r, 'family': 'nature_catalogue'} for r in document.get('rows', [])}
