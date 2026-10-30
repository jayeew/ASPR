"""Use cached native APS bylines and declared affiliation identifiers, without retrieval."""
from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from html import unescape
from pathlib import Path
import sqlite3

from lxml import html

from .fig8b_qualification_build import plain
from .supplement_sources import atomic_json


@lru_cache(maxsize=16)
def records(path: str) -> list[dict]:
    document = json.loads(gzip.decompress(Path(path).read_bytes()))
    if document.get('http_status') != 200:
        return []
    data = document.get('response', {}).get('data', [])
    return [data] if isinstance(data, dict) else data


def parse_authors(article: dict) -> dict:
    affiliations = {}
    for row in article.get('affiliations', []):
        raw = row.get('name', '')
        tree = html.fragment_fromstring(raw, create_parent='div')
        identifiers = [{'type': 'ror', 'value': href.rstrip('/')} for href in tree.xpath('.//a/@href')
                       if href.startswith('https://ror.org/')]
        affiliations.setdefault(row.get('id'), []).append({
            'text': plain(raw), 'countries': [], 'identifiers': identifiers, 'names': []})
    authors = []
    incomplete = False
    for author in article.get('authors', []):
        kind = author.get('type')
        if kind not in {'Person', 'Organization', 'Collaboration'}:
            incomplete = True
            continue
        refs = author.get('affiliationIds', [])
        missing = [ref for ref in refs if len(affiliations.get(ref, [])) != 1]
        name = plain(author.get('name', ''))
        incomplete |= bool(missing) or not name or name.casefold() in {'et al.', 'et al'}
        authors.append({'name': name, 'nonpersonal': kind != 'Person',
                        'orcid': author.get('orcid', ''),
                        'affiliations': [affiliations[ref][0] for ref in refs if ref not in missing],
                        'unresolved_affiliation_refs': missing})
    return {'authors': authors, 'byline_has_etal': incomplete or not authors,
            'byline_check': 'Native APS authors array with explicit affiliationIds; no inferred links'}


def byline_digest(article: dict) -> str:
    payload = {k: article.get(k) for k in ['identifiers', 'title', 'journal', 'authors', 'affiliations']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def jobs(root: Path, db: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    completed = {r[0] for r in db.execute(
        "SELECT work_id FROM springer_author_processed WHERE digest LIKE 'aps-native-byline-1:%' AND status<>'identity_unmatched'")}
    targets = {doi: (wid, shard) for wid, doi, shard in db.execute(
        "SELECT work_id,doi,shard FROM papers WHERE status='included_verified_research' AND doi LIKE '10.1103/%'") if wid not in completed}
    selected = {}; conflicts = set()
    for path in sorted((root / 'fields/qualification/aps_all_raw').glob('*.gz')):
        for article in records(str(path)):
            doi = article.get('identifiers', {}).get('doi', '').lower()
            if doi not in targets:
                continue
            digest = byline_digest(article)
            if doi in selected and selected[doi][1] != digest:
                conflicts.add(doi)
            selected[doi] = (str(path.relative_to(root)), digest)
    grouped = {}
    for doi, (receipt, _) in selected.items():
        if doi not in conflicts:
            wid, shard = targets[doi]
            grouped.setdefault(shard, []).append((wid, receipt))
    atomic_json(root / 'supplement/springer_author_cache/aps_scope.json', {
        'qualified_dois': len(targets), 'cached_dois': len(selected),
        'conflicting_cached_bylines': sorted(conflicts),
        'no_native_cache': sorted(set(targets) - selected.keys()), 'network_requests': 0})
    return grouped


def journal_issns(article: dict) -> list[str]:
    """Reuse APS native journal IDs already validated by qualification matching."""
    return {'PRD': ['2470-0010', '2470-0029'], 'PRB': ['2469-9950', '2469-9969'],
            'PRL': ['0031-9007', '1079-7114']}.get(article.get('journal', {}).get('id', ''), [])


def source_record(root: Path, receipt: str, doi: str) -> tuple[dict, bytes]:
    matches = [r for r in records(str(root / receipt)) if r.get('identifiers', {}).get('doi', '').lower() == doi]
    if len(matches) != 1:
        return {'parsed': {}, 'raw_receipt': receipt.removeprefix('fields/qualification/'), 'route': 'aps'}, b''
    article = matches[0]
    # Same native journal ID / ISSN rule already used by qualification_build.match.
    issns = journal_issns(article)
    parsed = {'issns': issns, 'doi': [doi], 'title': [unescape(plain(article.get('title', {}).get('value', '')))],
              'journal_titles': [article.get('journal', {}).get('name', '')], **parse_authors(article)}
    title_fix = ':aps-title-entity-decoding-1' if parsed['title'][0] != plain(article.get('title', {}).get('value', '')) else ''
    return {'parsed': parsed, 'raw_receipt': receipt.removeprefix('fields/qualification/'), 'route': 'aps'}, (byline_digest(article)+(':qualified-journal-issn-rule-1' if issns else '')+title_fix).encode()
