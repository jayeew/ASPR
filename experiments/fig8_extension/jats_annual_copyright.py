"""Reuse explicit dated publisher statements from identity-matched cached JATS."""
from __future__ import annotations

import gzip
import hashlib
import json
import re

from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import electronic_date
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import jats_metadata
from .three_journal_apply import read_csv


def named_publisher(tree: etree._Element, year: int) -> tuple[str, str] | None:
    permissions = tree.find('./front/article-meta/permissions')
    if permissions is None:
        return None
    text = ' '.join(''.join(permissions.itertext()).split())
    statements = ' '.join(permissions.xpath('./copyright-statement//text()'))
    copyright_years = set(re.findall(r'(?:19|20)\d{2}', ' '.join(permissions.xpath('./copyright-year/text()')) + ' ' + statements))
    if copyright_years != {str(year)}:
        return None
    publisher = ' '.join(tree.xpath('./front/journal-meta/publisher/publisher-name//text()'))
    # Society/imprint headers can differ from the explicitly named publisher.
    # These exact combinations were reviewed in remaining_annual_native/inventory.json.
    titles = set(tree.xpath('./front/journal-meta/journal-title-group/journal-title/text()'))
    scoped = [
        ('Engineering in Life Sciences', 'Maximum Academic Press', 'Wiley', r'published by Wiley[‐-]VCH GmbH'),
        ('Polish Journal of Microbiology', 'The Polish Society of Microbiologists', 'Sciendo', r'published by Sciendo'),
        ('Genes to Cells', '', 'Wiley', r'published by Molecular Biology Society of Japan and John Wiley & Sons Australia'),
        ('European Journal of Breast Health', 'Turkish Federation of Breast Diseases Societies', 'Galenos Publishing House', r'published by Galenos Publishing House'),
        ('Circulation Research', '', 'Wolters Kluwer', r'published on behalf of the American Heart Association, Inc\., by Wolters Kluwer Health, Inc\.'),
        ('Journal of Medical Biochemistry', 'Society of Medical Biochemists of Serbia', 'CEON/CEES', r'published by CEON/CEES'),
        ('Journal of Peptide Science', '', 'Wiley', r'published by European Peptide Society and John Wiley & Sons Ltd'),
        ('Pediatric Diabetes', 'Maximum Academic Press', 'Wiley', r'Pediatric Diabetes published by John Wiley & Sons Ltd'),
        ('Molecular Therapy. Methods & Clinical Development', 'American Society of Gene & Cell Therapy', 'Elsevier', r'Published by Elsevier Inc\. on behalf of The American Society of Gene and Cell Therapy'),
        ('Journal of Clinical Research in Pediatric Endocrinology', 'Galenos Yayinevi', 'Galenos Publishing House', r'published by Galenos Publishing House'),
        ('Journal of Gastroenterology and Hepatology', '', 'Wiley', r'published by Journal of Gastroenterology and Hepatology Foundation and John Wiley & Sons Australia'),
        ('Circulation', '', 'Wolters Kluwer', r'published on behalf of the American Heart Association, Inc\., by Wolters Kluwer Health, Inc\.'),
        ('Medeniyet Medical Journal', 'Istanbul Medeniyet University', 'Galenos Publishing House', r'published by Galenos Publishing House'),
        ('Engineering Microbiology', 'Shandong University', 'Elsevier', r'Published by Elsevier B\.V\. on behalf of Shandong University'),
        ('Journal of Sport and Health Science', 'Shanghai University of Sport', 'Elsevier', r'Published by Elsevier B\.V\. on behalf of Shanghai University of Sport'),
        ('Translational Oncology', 'Neoplasia Press', 'Elsevier', r'Published by Elsevier Inc\.'),
        ('Mycobiology', 'Korean Society of Mycology', 'Taylor & Francis',
         r'Published by Informa UK Limited, trading as Taylor & Francis Group on behalf of the Korean Society of Mycology'),
        ('Blood Advances', 'The American Society of Hematology', 'Elsevier', r'Published by Elsevier Inc\.'),
        ('Journal of the Endocrine Society', 'The Endocrine Society', 'Oxford University Press',
         r'Published by Oxford University Press on behalf of the Endocrine Society'),
        ('Function', 'American Physiological Society', 'Oxford University Press',
         r'Published by Oxford University Press on behalf of American Physiological Society'),
        ('The Plant Journal', '', 'Wiley',
         r'published by Society for Experimental Biology and John Wiley & Sons Ltd'),
        ('Asian Journal of Pharmaceutical Sciences', 'Shenyang Pharmaceutical University',
         'Elsevier', r'Published by Elsevier B\.V\.'),
        ('In Vivo', 'International Institute of Anticancer Research',
         'International Institute of Anticancer Research', r'Published by the International Institute of Anticancer Research\b'),
        ('Cancer Genomics & Proteomics', 'International Institute of Anticancer Research',
         'International Institute of Anticancer Research', r'Published by the International Institute of Anticancer Research\b'),
        ('CytoJournal', 'Cytopathology Foundation, Inc.', 'Scientific Scholar', r'Published by Scientific Scholar\b'),
        ('The Indian Journal of Medical Research', 'Scientific Scholar', 'Scientific Scholar',
         r'published by Scientific Scholar for Director-General, Indian Council of Medical Research'),
        ('Forestry Research', 'Maximum Academic Press', 'Maximum Academic Press', r'Published by Maximum Academic Press, Fayetteville'),
        ('South Asian Journal of Cancer', 'Thieme Medical Publishers', 'Thieme', r'article published by Thieme under the terms'),
        ('Physics in Medicine and Biology', '', 'IOP Publishing',
         r'Published on behalf of Institute of Physics and Engineering in Medicine by IOP Publishing Ltd'),
    ]
    for title, header, label, expression in scoped:
        if title in titles and publisher.strip() == header and re.search(expression, text, re.I):
            return label, text
    if (publisher.strip() == 'Korean Society for Microbiology and Biotechnology'
            and set(tree.xpath('./front/journal-meta/issn/text()')) & {'1017-7825', '1738-8872'}
            and re.search(r'\bLicensee KMB\b', statements)):
        return publisher.strip(), text
    if 'MDPI' in publisher and 'Licensee MDPI, Basel, Switzerland' in text:
        return 'MDPI', text
    if (not publisher.strip() or 'wiley' in publisher.casefold()) and re.search(r'published\s+by\s+(?:John\s+)?Wiley\b', statements, re.I):
        return 'Wiley', text
    for name in ('Elsevier', 'Oxford University Press', 'American Association for Cancer Research',
                 'Wolters Kluwer', 'S. Karger', 'BMJ'):
        if (not publisher.strip() or name.casefold() in publisher.casefold()) and re.search(
                r'published\s+by\s+(?:the\s+)?' + re.escape(name) + r'\b', statements, re.I):
            return name, text
    if re.search(r'published(?:\s+\d{4})?\s+by American Chemical Society\b', statements, re.I):
        return 'American Chemical Society', text
    # These literal statements occur in the retained native permission sections.
    # Header aliases alone never establish publisher attribution.
    additional = [
        ('Cold Spring Harbor Laboratory Press', {'Cold Spring Harbor Laboratory Press'},
         r'Cold Spring Harbor Laboratory Press'),
        ('Portland Press', {'', 'Portland Press Ltd', 'Portland Press Limited'}, r'Portland Press Limited'),
        ('S. Karger', {'', 'Karger Publishers', 'S. Karger AG'}, r'S\. Karger AG'),
        ('Baishideng Publishing Group', {'', 'Baishideng Publishing Group Inc'}, r'Baishideng Publishing Group Inc'),
        ('IOS Press', {'', 'IOS Press'}, r'IOS Press'),
        ('IMR Press', {'', 'IMR Press'}, r'IMR Press'),
        ('Bentham Science', {'', 'Bentham Science Publishers'}, r'Bentham Science Publishers?'),
        ('American Physiological Society', {'', 'American Physiological Society'}, r'(?:the\s+)?American Physiological Society'),
        ('The Company of Biologists', {'', 'The Company of Biologists'}, r'(?:The\s+)?Company of Biologists'),
        ('Thieme', {'', 'Thieme Medical Publishers'}, r'Thieme'),
        ('Taylor & Francis', {'Taylor & Francis'}, r'Informa UK Limited, trading as Taylor & Francis Group'),
        ('Scientific Scholar', {'Scientific Scholar'}, r'Scientific Scholar'),
        ('International Institute of Anticancer Research', {'International Institute of Anticancer Research'},
         r'(?:the\s+)?International Institute of Anticancer Research'),
        ('Maximum Academic Press', {'Maximum Academic Press'}, r'Maximum Academic Press'),
    ]
    for label, headers, literal in additional:
        if publisher.strip() in headers and re.search(r'published\s+by\s+' + literal + r'\b', text, re.I):
            return label, text
    return None


def run() -> dict:
    folder = ROOT / 'supplement/jats_annual_copyright'
    folder.mkdir(exist_ok=True)
    rule_path = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(rule_path.read_text())} if rule_path.exists() else {}
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    db = connect()
    rows = list(db.execute("""SELECT p.doi,p.title,p.journal,p.source_id,s.payload FROM papers p JOIN source_metadata s USING(doi)
            WHERE p.status='included_verified_research' AND s.kind='jats'"""))
    # Directly applied PMC captures include cloud, OAI and named-format batches.
    # Restrict the join before loading cached files; no new article requests.
    unknown_sources = {sid for (sid, _), item in annual.items()
                       if item['nature_portfolio_year'] not in {'True', 'False'}}
    db.execute('CREATE TEMP TABLE annual_unknown_sources(source_id TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO annual_unknown_sources VALUES(?)', [(sid,) for sid in unknown_sources])
    cloud_rows = db.execute("""SELECT p.doi,p.title,p.journal,p.source_id,s.receipt
        FROM papers p JOIN supplied_publisher_processed s USING(doi)
        JOIN annual_unknown_sources u USING(source_id)
        WHERE p.status='included_verified_research'
        """).fetchall()
    for doi, title, journal, sid, receipt in cloud_rows:
        if sid not in unknown_sources:
            continue
        record = json.loads((ROOT / receipt).read_text())
        if record.get('status') != 'parsed' or record.get('source') != 'pmc':
            continue
        payload = {**record['parsed'],
                   'evidence_receipt': 'fields/qualification/' + record['raw_receipt'],
                   'source_url': record['url']}
        rows.append((doi, title, journal, sid, json.dumps(payload)))
    db.close()
    rejected = []
    for doi, title, journal, sid, payload in rows:
        cached = json.loads(payload)
        online, precision = electronic_date(cached.get('dates', []))
        if not online or int(online[:4]) not in {2023, 2024, 2025}:
            continue
        year = int(online[:4])
        if (sid, year) not in annual or annual[(sid, year)]['nature_portfolio_year'] in {'True', 'False'}:
            continue
        receipt = cached['evidence_receipt']
        raw = gzip.decompress((ROOT / receipt).read_bytes())
        document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
        tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
        if etree.QName(tree).localname != 'article':
            articles = tree.xpath('//*[local-name()="article"]')
            if len(articles) != 1:
                rejected.append({'doi': doi, 'receipt': receipt, 'status': 'ambiguous_native_article_envelope'})
                continue
            tree = articles[0]
            document = etree.tostring(tree)
        named = named_publisher(tree, year)
        if named is None:
            continue
        native = jats_metadata(document)
        native['issns'] = native.get('issn', [])
        verified_date, _ = electronic_date(native['dates'])
        if (verified_date != online or doi not in {v.lower() for v in native.get('doi', [])}
                or not signature_match({'title': title, 'journal_name': journal, 'source_id': sid}, native, registry[sid])):
            rejected.append({'doi': doi, 'receipt': receipt, 'status': 'raw_identity_or_date_not_confirmed'})
            continue
        publisher, statement = named
        proof = {'doi': doi, 'publication_date': online, 'statement': statement,
                 'raw_receipt': receipt, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                 'source_url': cached['source_url'], 'locator': 'front/article-meta/permissions',
                 'journal_titles': native.get('journal_titles'), 'issns': native['issns']}
        if (sid, year) not in rules:
            rules[(sid, year)] = {'source_id': sid, 'year': year, 'label': 'False', 'journal': journal,
                'publisher': publisher, 'source_url': proof['source_url'], 'raw_receipt': receipt,
                'raw_sha256': proof['raw_sha256'], 'first_dated_proof': proof, 'last_dated_proof': proof,
                'scope': 'Working annual rule: exact paper DOI/title/journal; explicit electronic year agrees with copyright year and explicit publisher/licensee statement from the reviewed named-publisher rules; no extrapolation; reopen on transfer evidence'}
        rule = rules[(sid, year)]
        if online < rule['first_dated_proof']['publication_date']:
            rule['first_dated_proof'] = proof
        if online > rule['last_dated_proof']['publication_date']:
            rule['last_dated_proof'] = proof
    return apply_proposals(list(rules.values()), folder, rejected,
                           rule_status='dated_native_jats_publication_statement',
                           reason='Native dated copyright and named publisher/licensee, electronic year and exact identity match',
                           route='native_jats_copyright',
                           apply_pairs={pair for pair in rules if annual.get(pair, {}).get('nature_portfolio_year') not in {'True','False'}})


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
