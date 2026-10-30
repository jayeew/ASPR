"""Read article-bound bylines from public Frontiers hydration metadata."""
from __future__ import annotations

import json
import unicodedata

from lxml import html

from .fig8b_qualification_build import key


def decode(values: list, index: int, depth: int = 0) -> object:
    if depth > 30 or not isinstance(index, int) or index >= len(values):
        raise ValueError('Invalid metadata reference')
    if index < 0:
        return None
    value = values[index]
    if isinstance(value, dict):
        return {k: decode(values, v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(values, v, depth + 1) for v in value]
    return value


def citation_sequence_matches(authors: list[dict], citation_names: list[str]) -> bool:
    """Confirm ordered names, allowing only explicit suffix and mononym formatting."""
    if not authors or len(authors) != len(citation_names):
        return False
    def normalized(value: str) -> str:
        return key(''.join(c for c in unicodedata.normalize('NFKD', value)
                           if not unicodedata.combining(c)))
    full_names = [' '.join(str(a.get(k) or '').strip() for k in ['givenNames', 'surname', 'suffix']).strip()
                  for a in authors]
    if len({normalized(n) for n in full_names}) != len(full_names):
        return False
    displayed_names = [' '.join(reversed(n.split(',', 1))) if ',' in n else n for n in citation_names]
    if len({normalized(n) for n in displayed_names}) != len(displayed_names):
        return False
    for author, full, citation in zip(authors, full_names, citation_names):
        given = str(author.get('givenNames') or '').strip()
        surname = str(author.get('surname') or '').strip()
        variants = {normalized(full)}
        if author.get('suffix'):
            variants.add(normalized(given + ' ' + surname))
        mononym = given or surname
        if bool(given) != bool(surname) and len(mononym.split()) == 1:
            variants.update({normalized('null ' + mononym), normalized(mononym + ' null'),
                             normalized(mononym + ' ' + mononym)})
        displayed = ' '.join(reversed(citation.split(',', 1))) if ',' in citation else citation
        if normalized(displayed) not in variants:
            return False
    return True


def parse_authors(raw: bytes, expected_doi: str) -> dict:
    tree = html.fromstring(raw)
    payloads = tree.xpath('//script[@id="__NUXT_DATA__"]/text()')
    empty = {'authors': [], 'byline_has_etal': True}
    if len(payloads) != 1:
        return empty
    try:
        values = json.loads(payloads[0])
        candidates = [v for v in values if isinstance(v, dict)
                      and {'doi', 'authors', 'affiliations', 'publicationDate', 'title'} <= set(v)
                      and decode(values, v['doi']) == expected_doi]
        if len(candidates) > 1:
            return empty
        if candidates:
            article = candidates[0]
        else:
            from .frontiers_content_cache import bound_article
            article, _, _ = bound_article(tree, values, expected_doi)
        authors = decode(values, article['authors'])
        affiliations = decode(values, article['affiliations'])
    except (ValueError, TypeError, KeyError, IndexError, RecursionError):
        return empty
    if not isinstance(authors, list) or not isinstance(affiliations, list):
        return empty
    addresses = {}
    for entry in affiliations:
        if not isinstance(entry, dict) or entry.get('index') in addresses:
            return empty
        addresses[entry.get('index')] = {'text':entry.get('textContent', ''),
                                        'countries':[{'text':entry['country'],'code':''}] if entry.get('country') else [],
                                        'identifiers':[], 'names':[entry['institution']] if entry.get('institution') else []}
    result = []
    for author in authors:
        if not isinstance(author, dict):
            return empty
        name = ' '.join(str(author.get(k) or '').strip() for k in ['givenNames', 'surname', 'suffix']).strip()
        refs = author.get('affiliationsIndexes') or []
        if not name or not isinstance(refs, list) or author.get('onBehalfOf'):
            return empty
        result.append({'name': name, 'orcid': '', 'nonpersonal': False,
                       'affiliations': [addresses[r] for r in refs if addresses.get(r,{}).get('text')],
                       'unresolved_affiliation_refs': [str(r) for r in refs if not addresses.get(r,{}).get('text')]})
    # Cross-check the complete publisher byline; metadata also contains editor profiles.
    meta = tree.xpath('//meta[@name="citation_author"]/@content')
    if not result or not citation_sequence_matches(authors, meta):
        return empty
    return {'authors': result, 'byline_has_etal': any(a['unresolved_affiliation_refs'] or not a['affiliations'] for a in result),
            'byline_basis': 'DOI-bound article payload and complete ordered citation_author sequence; explicit suffix omission, mononym metadata placeholders and accent typography allowed without changing source names; explicit affiliation indices'}
