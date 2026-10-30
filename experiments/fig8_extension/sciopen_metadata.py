"""Extract SciOpen's visible article fields without inferring online dates."""
from __future__ import annotations

from datetime import datetime
from html import unescape
import re

from lxml import html

from .fig8b_qualification_build import key


def parse_authors(raw: bytes | str) -> dict:
    tree = html.fromstring(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
    affiliations = {}
    address_nodes = tree.xpath('//div[@class="art-authors-unit-item"]')
    author_nodes = tree.xpath('//span[@class="art-authors"]')
    shared_address = (len(address_nodes) == 1 and not address_nodes[0].xpath('./label')
                      and not any(node.xpath('./sup') for node in author_nodes))
    if shared_address:
        address = ' '.join(address_nodes[0].text_content().split())
        if address:
            affiliations['shared_unlabelled_address'] = {address}
    for node in address_nodes:
        labels = node.xpath('./label')
        if len(labels) != 1:
            continue
        label = labels[0].text_content().strip()
        text = ' '.join(node.text_content().split())
        text = text[len(label):].strip() if text.startswith(label) else ''
        if label and text:
            affiliations.setdefault(label, set()).add(text)
    authors = []
    for node in author_nodes:
        names = node.xpath('./span[1]')
        if len(names) != 1:
            continue
        name = ' '.join(names[0].text_content().split())
        refs = list(dict.fromkeys(re.findall(r'\d+', ' '.join(node.xpath('./sup//text()')))))
        if shared_address:
            refs = ['shared_unlabelled_address']
        linked = [r for r in refs if len(affiliations.get(r, set())) == 1]
        authors.append({'name': name, 'nonpersonal': False,
            'orcid': next(iter(node.xpath('./a[contains(@href,"orcid.org")]/@href')), ''),
            'affiliations': [{'text': next(iter(affiliations[r])), 'countries': [],
                              'identifiers': [], 'names': []} for r in linked],
            'unresolved_affiliation_refs': [r for r in refs if r not in linked]})
    meta = tree.xpath('//meta[@name="citation_author"]/@content')
    complete = bool(authors) and [key(a['name']) for a in authors] == [key(v) for v in meta]
    return {'authors': authors, 'byline_has_etal': not complete,
            'byline_check': 'SciOpen visible author order compared with citation_author metadata'}


def enrich_metadata(raw: bytes | str, parsed: dict) -> dict:
    tree = html.fromstring(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
    online_values = tree.xpath('//meta[@name="citation_online_date"]/@content')
    online_dates = set()
    invalid_online = False
    for value in online_values:
        try:
            online_dates.add(datetime.strptime(value.strip().replace('-', '/'), '%Y/%m/%d').date().isoformat())
        except ValueError:
            invalid_online = True
    online = parsed.get('online_date', '')
    precision = parsed.get('date_precision', 'missing_explicit_electronic_date')
    if online_values:
        if len(online_dates) == 1 and not invalid_online:
            online, precision = next(iter(online_dates)), 'day'
        else:
            online, precision = '', 'conflicting_or_invalid_explicit_electronic_date'
    info = tree.xpath('//div[contains(concat(" ",normalize-space(@class)," ")," v4-art-info ")]')
    kinds = {" ".join(n.text_content().split()) for node in info
             for n in node.xpath('./span[1]')}
    kinds.discard('')
    kinds.difference_update({'Open Access', 'Free Access', 'Subscription'})
    abstract_nodes = tree.xpath('//div[@id="title_-2"][h2[normalize-space(.)="Abstract"]]/div//p')
    paragraphs = list(dict.fromkeys(' '.join(n.text_content().split()) for n in abstract_nodes))
    history = tree.xpath('//span[normalize-space(.)="Published:"]/following-sibling::span[1]/text()')
    dates = set()
    for value in history:
        try:
            dates.add(datetime.strptime(value.strip(), '%d %B %Y').date().isoformat())
        except ValueError:
            continue
    return {**parsed,
            'title': [unescape(value) for value in parsed.get('title', [])],
            'online_date': online, 'date_precision': precision,
            'article_types': sorted(kinds),
            'article_type_review': {'basis': 'SciOpen visible v4-art-info first span', 'labels': sorted(kinds)},
            'abstract': '\n\n'.join(p for p in paragraphs if p) or parsed.get('abstract', ''),
            'date_field_review': {**parsed.get('date_field_review', {}),
                'sciopen_visible_published': sorted(dates),
                'citation_online_date': online_values,
                'basis': 'Explicit citation_online_date parsed with variable-width month/day; visible Published not substituted'}}
