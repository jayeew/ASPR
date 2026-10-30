"""Reuse explicit OJS bylines; no network or inferred address countries."""
from __future__ import annotations

from lxml import html

from .fig8b_qualification_build import key


def parse_authors(raw: bytes, doi: str) -> dict:
    tree = html.fromstring(raw)
    identifiers = tree.xpath('//meta[@name="citation_doi"]/@content')
    if [v.lower().strip() for v in identifiers] != [doi]:
        return {'authors': [], 'byline_has_etal': True}
    lists = tree.xpath('//ul[contains(concat(" ",normalize-space(@class)," ")," authors ")]')
    if len(lists) != 1:
        return {'authors': [], 'byline_has_etal': True}
    authors = []
    for node in lists[0].xpath('./li'):
        names = node.xpath('./span[@class="name"]')
        if len(names) != 1:
            return {'authors': [], 'byline_has_etal': True}
        name = ' '.join(names[0].text_content().split())
        affiliations = [' '.join(v.text_content().split()) for v in node.xpath('./span[@class="affiliation"]')]
        authors.append({'name': name, 'nonpersonal': False,
                        'orcid': next(iter(node.xpath('.//span[@class="orcid"]/a/@href')), ''),
                        'affiliations': [{'text': a, 'countries': [], 'identifiers': [], 'names': []}
                                         for a in affiliations if a],
                        'unresolved_affiliation_refs': []})
    metadata = tree.xpath('//meta[@name="citation_author"]/@content')
    complete = bool(authors) and [key(a['name']) for a in authors] == [key(v) for v in metadata]
    return {'authors': authors, 'byline_has_etal': not complete,
            'byline_check': 'Explicit OJS author-affiliation child links; completeness checked against citation_author order'}
