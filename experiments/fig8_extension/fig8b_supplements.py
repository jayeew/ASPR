"""Exact DOI/title secondary metadata matching and publisher evidence precedence."""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from typing import Any

from lxml import html

from .fig8b_collect import OUT


def title_key(value: str | None) -> str:
    return ''.join(c for c in (value or '').casefold() if c.isalnum())


def html_text(value: str) -> str:
    if '<' not in value:
        return value
    tree = html.fragment_fromstring(value, create_parent='div')
    for node in tree.xpath('.//sup|.//sub'):
        text = node.text_content()
        node.text = ('^' if node.tag == 'sup' else '_') + '{' + text + '}'
        for child in list(node):
            node.remove(child)
    return ' '.join(tree.text_content().split())


def load() -> tuple[dict, dict, dict]:
    epmc: dict[str, list] = defaultdict(list)
    for path in sorted((OUT / 'epmc_raw').glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            receipt = json.load(stream)
        if receipt['status'] != 'complete':
            continue
        requested = set(receipt['dois'])
        for record in receipt['response'].get('resultList', {}).get('result', []):
            doi = (record.get('doi') or '').lower()
            if doi in requested and record.get('source') in ['MED', 'PMC']:
                epmc[doi].append({'record': record, 'receipt': str(path.relative_to(OUT))})
    publisher = {r['doi']: r for r in
                 (json.loads(p.read_text()) for p in (OUT / 'publisher_parsed').glob('*.json'))}
    crossref: dict[str, list] = defaultdict(list)
    for path in sorted((OUT / 'crossref_raw').glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            receipt = json.load(stream)
        if receipt.get('status') != 'success':
            continue
        for record in receipt['response']['message']['items']:
            if record.get('DOI') and record.get('abstract'):
                crossref[record['DOI'].lower()].append({'record': {
                    'title': (record.get('title') or [''])[0], 'abstractText': record['abstract'],
                    'issns': record.get('ISSN') or []}, 'receipt': str(path.relative_to(OUT))})
    return dict(epmc), publisher, dict(crossref)


def select_epmc(record: dict, candidates: list[dict]) -> dict[str, Any] | None:
    if not title_key(record.get('title')):
        return None
    matching = [c for c in candidates if title_key(html_text(c['record'].get('title') or '')) == title_key(record.get('title'))]
    # Multiple records are accepted only when their nonempty abstracts agree exactly.
    abstracts = {html_text(c['record']['abstractText']) for c in matching if c['record'].get('abstractText')}
    if len(abstracts) > 1:
        return None
    if matching:
        return next((c for c in matching if c['record'].get('abstractText')), matching[0])
    return None


def supplementary_addresses(record: dict | None) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = defaultdict(list)
    if record:
        for author in record.get('authorList', {}).get('author', []):
            name = ' '.join([author.get('firstName', ''), author.get('lastName', '')]).strip()
            key = re.sub(r'\s+', ' ', name).casefold()
            addresses = [a['affiliation'] for a in author.get('authorAffiliationDetailsList', {}).get('authorAffiliation', []) if a.get('affiliation')]
            if key and addresses:
                result[key].append(addresses)
    return dict(result)
