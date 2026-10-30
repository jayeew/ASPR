"""Parse Elsevier article front matter without inventing publication dates."""
from __future__ import annotations

from .native_xml_document import document
from .springer_native_xml import text

NS = {'ja': 'http://www.elsevier.com/xml/ja/dtd',
      'ce': 'http://www.elsevier.com/xml/common/dtd',
      'sa': 'http://www.elsevier.com/xml/common/struct-aff/dtd'}


def parse(raw: bytes) -> dict:
    tree, repairs = document(raw)
    roots = {'{'+NS['ja']+'}article': 'ja:head',
             '{'+NS['ja']+'}simple-article': 'ja:simple-head'}
    if tree.tag not in roots:return {}
    head = tree.find(roots[tree.tag], NS)
    if head is None:return {}
    affiliations = {}
    for node in head.findall('.//ce:affiliation', NS):
        country = node.findall('./sa:affiliation/sa:country', NS)
        affiliations[node.get('id')] = {
            'text': text(node.find('./ce:textfn', NS)) or text(node.find('./ce:source-text', NS)),
            'countries': [{'text': text(c), 'code': c.get('iso-3166-alpha-2', '')} for c in country],
            'identifiers': [], 'names': [text(n) for n in node.findall('./sa:affiliation/sa:organization', NS)]}
    authors = []
    for group in head.findall('./ce:author-group', NS):
        for node in group.findall('./ce:author', NS):
            refs = [r for n in node.findall('./ce:cross-ref', NS) for r in n.get('refid', '').split()]
            linked = [r for r in refs if r in affiliations]
            missing = [r for r in refs if r.startswith('aff') and r not in affiliations]
            authors.append({'name': ' '.join(filter(None, [text(node.find('./ce:given-name', NS)), text(node.find('./ce:surname', NS))])),
                'nonpersonal': False, 'orcid': node.get('orcid', ''),
                'affiliations': [affiliations[r] for r in linked], 'unresolved_affiliation_refs': missing})
    kind = tree.get('docsubtype', '')
    mapped = {'fla': 'Full-length article', 'rev': 'Review article', 'sco': 'Short communication'}
    declared = [text(n) for n in head.findall('./ce:dochead/ce:textfn', NS) if text(n)]
    kinds = declared if tree.tag.endswith('}simple-article') and declared else ([mapped.get(kind, kind)] if kind else [])
    return {'xml_normalization': repairs, 'doi': [text(tree.find('./ja:item-info/ce:doi', NS)).lower()],
        'title': [text(head.find('./ce:title', NS))], 'issns': [], 'journal_titles': [],
        'article_types': kinds, 'native_article_type': kind,
        'online_date': '', 'date_precision': '', 'date_limitation': 'No explicit online date in validated native article front schema; received/revised/accepted dates are not publication dates',
        'abstract': '\n\n'.join(text(n) for n in head.findall('./ce:abstract/ce:abstract-sec/ce:simple-para', NS)),
        'authors': authors,
        'byline_has_etal': not authors or len(head.findall('.//ce:author', NS)) != len(authors) or bool(head.findall('.//ce:collaboration', NS)) or any(not a['name'] for a in authors)}
