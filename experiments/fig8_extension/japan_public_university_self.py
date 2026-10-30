"""Corroborate exact university-self signatures from the NIAD-QE directory."""
from __future__ import annotations

import gzip
import json

from lxml import html

from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .uk_devolved_university_self import run as apply_names

URL = 'https://jpcup.niad.ac.jp/what-jpcup/lists.html'
FOLDER = ROOT / 'supplement/official_entities/japan_public_self'


def run() -> None:
    receipt = json.loads((FOLDER / 'directory_receipt.json').read_text())
    if receipt.get('http_status') != 200:
        raise ValueError('Official directory unavailable')
    tree = html.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()).decode('utf-8'))
    rows = []
    for section in tree.xpath('//div[contains(@class,"sortBySection")]'):
        heading = section.find('h2')
        if heading is None or heading.text_content().strip() not in {'National Universities in Japan', 'Public Universities in Japan'}:
            continue
        for name in section.xpath('.//div[@class="sortListInner"]/p'):
            rows.append({'name': ' '.join(name.text_content().split()),
                         'section': section.get('id'), 'heading': heading.text_content().strip()})
    if not rows:
        raise ValueError('No reviewed university sections')
    # These reorganizations require date-specific identities; do not infer predecessors.
    excluded = {'Institute of Science Tokyo', 'Osaka Metropolitan University'}
    names = sorted({r['name'] for r in rows if r['name'] not in excluded})
    atomic_json(FOLDER / 'directory_extraction.json', {'rows': rows, 'excluded_reorganization_names': sorted(excluded),
                'raw_receipt': receipt['raw_receipt'], 'reviewed_names': len(names),
                'not_included': 'Junior universities, company-established section, private universities and all inferred aliases'})
    apply_names({'niad_directory': (URL, '|'.join(names))}, 'japan_public_self', {'niad_directory': 'JP'},
                'Exact university self names only. No affiliated hospitals, departments, laboratories, joint centers, '
                'predecessor/successor mapping, junior colleges, private institutions or country replacement.',
                evidence_access='Retained official directory HTML; supplement/official_entities/japan_public_self/directory_receipt.json',
                evidence_locator='National Universities in Japan and Public Universities in Japan sections; individual section IDs in directory_extraction.json')
    proof = FOLDER / 'niad_directory_review.json'
    review = json.loads(proof.read_text())
    review.update(raw_http_retained=True, raw_receipt=receipt['raw_receipt'])
    atomic_json(proof, review)


if __name__ == '__main__':
    run()
