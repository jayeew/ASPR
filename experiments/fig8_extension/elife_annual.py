"""Apply eLife's own dated annual publishing/account reports to matching years."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import html

from .cell_press_annual_cache import apply_proposals
from .elife_batch import FOLDER
from .supplement_queue import ROOT
from .three_journal_apply import read_csv


def run() -> dict:
    journal = next(r for r in read_csv(ROOT / 'fields/journal_registry.csv')
                   if r['source_id'] == 'https://openalex.org/S1336409049')
    if journal['journal_name'] != 'eLife' or '2050-084X' not in json.loads(journal['issn']):
        raise ValueError('eLife journal identity changed')
    rules = []
    for year in [2023, 2024, 2025]:
        receipt = json.loads((FOLDER / f'annual_{year}.json').read_text())
        if receipt.get('http_status') != 200:
            raise ValueError('Annual source unavailable')
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        tree = html.fromstring(raw)
        text = ' '.join(tree.text_content().split())
        documents = [json.loads(node.text) for node in tree.xpath('//script[@type="application/ld+json"]')]
        expected = f'Annual Report: {year} in review'
        matching = [d for d in documents if d.get('headline') == expected
                    and d.get('publisher', {}).get('name') == 'eLife Sciences Publications, Ltd'
                    and d.get('isPartOf', {}).get('issn') == '2050-084X']
        if len(matching) != 1 or f'accounts for eLife Sciences Publications, Ltd for {year}' not in text:
            raise ValueError('Annual publishing organisation/year not explicit')
        rules.append({'source_id': journal['source_id'], 'year': year, 'label': 'False',
                      'journal': 'eLife', 'issn': '2050-084X', 'publisher': 'eLife Sciences Publications, Ltd',
                      'source_url': receipt['url'], 'raw_receipt': receipt['raw_receipt'],
                      'raw_sha256': hashlib.sha256(raw).hexdigest(),
                      'locator': f'{expected}; publishing operations and year-specific audited accounts; JSON-LD publisher/periodical ISSN',
                      'scope': f'Annual report explicitly covers publishing operations in {year}; non-Nature Portfolio; no extension to other years or journals'})
    folder = FOLDER / 'annual_application'
    folder.mkdir(exist_ok=True)
    return apply_proposals(rules, folder, [], 'dated_elife_annual_publishing_report',
                           'Official year-specific eLife publishing/account report; exact journal and ISSN',
                           'elife_annual_report')


if __name__ == '__main__':
    run()
