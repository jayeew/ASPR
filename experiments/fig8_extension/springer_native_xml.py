"""Read publisher-native Springer XML; do not infer dates or article types."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from lxml import etree


def text(node: etree._Element | None) -> str:
    if node is None:return ''
    copy = deepcopy(node)
    for equation in copy.xpath('.//InlineEquation | .//Equation'):
        alternatives = equation.findall('./EquationSource')
        if len(alternatives) < 2:continue
        selected = next((x for x in alternatives if x.get('Format', '').upper() == 'TEX'), alternatives[0])
        for alternative in alternatives:
            if alternative is not selected:equation.remove(alternative)
    return ' '.join(''.join(copy.itertext()).split())


def parse(raw: bytes) -> dict:
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    info = tree.findall('.//ArticleInfo')
    if tree.tag != 'Publisher' or len(info) != 1:
        return {}
    info = info[0]; online = ''; date_error = ''
    nodes = info.findall('./ArticleHistory/OnlineDate')
    if len(nodes) == 1:
        try:
            online = date(*(int(nodes[0].findtext(k, '')) for k in ['Year', 'Month', 'Day'])).isoformat()
        except ValueError:
            date_error = 'invalid_or_incomplete_explicit_online_date'
    groups = tree.findall('.//AuthorGroup'); authors = []
    for group in groups:
        affiliations = {}
        for node in group.findall('./Affiliation'):
            parts = [text(x) for x in node if x.tag not in {'OrgID', 'AffiliationNumber'}]
            affiliations[node.get('ID')] = {'text': ', '.join(p for p in parts if p),
                'countries': [{'text': text(c), 'code': c.get('Code', '')} for c in node.findall('.//Country')],
                'identifiers': [{'type': i.get('Type', ''), 'value': text(i)} for i in node.findall('./OrgID')],
                'names': [text(n) for n in node.findall('./OrgName')]}
        for node in group:
            if node.tag not in {'Author', 'InstitutionalAuthor'}:
                continue
            name = node.find('./AuthorName')
            personal = node.tag == 'Author'
            if personal and name is not None:
                tokens = [text(x) for x in name if x.tag in {'GivenName', 'Particle', 'FamilyName', 'Suffix'}]
                label = ' '.join(tokens)
            else:
                label = text(node.find('./InstitutionalAuthorName'))
            refs = node.get('AffiliationIDS', '').split()
            authors.append({'name': label, 'nonpersonal': not personal, 'orcid': node.get('ORCID', ''),
                'affiliations': [affiliations[r] for r in refs if r in affiliations],
                'unresolved_affiliation_refs': [r for r in refs if r not in affiliations]})
    categories = [text(n) for n in info.findall('./ArticleCategory')]
    kind = info.get('ArticleType', '')
    return {'doi': [text(info.find('./ArticleDOI')).lower()], 'title': [text(info.find('./ArticleTitle'))],
        'issns': [text(n) for n in tree.xpath('.//JournalInfo/JournalElectronicISSN | .//JournalInfo/JournalPrintISSN')],
        'journal_titles': [text(n) for n in tree.findall('.//JournalInfo/JournalTitle')],
        'publisher': [text(n) for n in tree.findall('.//PublisherInfo/PublisherName')],
        'article_types': [v for v in [kind, *categories] if v], 'native_article_type': kind,
        'online_date': online, 'date_precision': 'day' if online else '', 'date_error': date_error,
        'abstract': '\n\n'.join(text(n) for n in tree.findall('.//Abstract/Para') if text(n)),
        'authors': authors, 'byline_has_etal': len(groups) != 1 or not authors or any(not a['name'] for a in authors)}
