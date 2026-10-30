"""Review exact identities and explicit electronic dates without changing decisions."""
from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

from lxml import etree

from .supplement_apply import signature_match
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run(folder: Path | None = None) -> dict:
    folder = folder if folder is not None else ROOT / 'supplement/pubmed_date_followup'
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'collected_identity_and_date_review_required':
        raise RuntimeError('Fixed collection must finish before review')
    scope = json.loads((folder / 'scope.json').read_text())['candidates']
    requested = {r['pmids'][0]: r for r in scope if r['status'] == 'ready'}
    with (ROOT / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in csv.DictReader(stream)}
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    results = []
    for path in sorted(folder.glob('*.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        tree = etree.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()), etree.XMLParser(resolve_entities=False, no_network=True))
        for article in tree.findall('PubmedArticle'):
            pmid = article.findtext('./MedlineCitation/PMID')
            if pmid not in requested:
                continue
            doi = requested[pmid]['doi']
            dois = {x.strip().lower() for x in article.xpath('./PubmedData/ArticleIdList/ArticleId[@IdType="doi"]/text()')}
            payload = {'title': [''.join(article.find('./MedlineCitation/Article/ArticleTitle').itertext())],
                       'issns': article.xpath('./MedlineCitation/Article/Journal/ISSN/text()'),
                       'journal_titles': article.xpath('./MedlineCitation/Article/Journal/Title/text()')}
            matches = [wid for wid, title, journal, sid in db.execute('SELECT work_id,title,journal,source_id FROM papers WHERE doi=?', (doi,))
                       if doi in dois and (not payload['issns'] or journals[sid] & set(payload['issns']))
                       and signature_match({'title': title, 'journal_name': journal, 'source_id': sid}, payload, journals[sid])]
            dates = []
            for element in article.findall('./MedlineCitation/Article/ArticleDate[@DateType="Electronic"]'):
                try:
                    dates.append(date(int(element.findtext('Year')), int(element.findtext('Month')), int(element.findtext('Day'))).isoformat())
                except (TypeError, ValueError):
                    dates.append('invalid_or_incomplete')
            state = ('identity_unresolved' if len(matches) != 1 else 'no_explicit_electronic_date' if not dates else
                     'conflicting_or_incomplete_dates' if len(set(dates)) != 1 or 'invalid_or_incomplete' in dates else 'matched_electronic_date_requires_primary_evidence_merge')
            results.append({'doi': doi, 'pmid': pmid, 'work_ids': matches, 'status': state, 'electronic_dates': dates,
                            'publication_types': article.xpath('./MedlineCitation/Article/PublicationTypeList/PublicationType/text()'),
                            'receipt': receipt['raw_receipt'], 'identity': payload,
                            'source_url': receipt['url'], 'eligibility_unchanged': True})
    db.close()
    returned = {r['pmid'] for r in results}
    for pmid, row in requested.items():
        if pmid not in returned:
            results.append({'doi': row['doi'], 'pmid': pmid, 'status': 'requested_pmid_not_returned'})
    summary = dict(Counter(r['status'] for r in results))
    atomic_json(folder / 'date_review.json', {'counts': summary, 'records': results, 'eligibility_unchanged': True})
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    run()
