"""Extract cached publisher catalogue cards without inventing bylines or dates."""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from datetime import datetime
from urllib.parse import unquote, urlparse

from lxml import html

from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def parse_cards(raw: bytes) -> list[dict]:
    tree = html.fromstring(raw.decode('utf-8'))
    titles = tree.xpath('//title/text()')
    match = re.fullmatch(r'Articles \| (.+) \| Springer Nature Link', titles[0].strip()) if titles else None
    if not match:
        raise ValueError('Unrecognized catalogue journal heading')
    journal = match.group(1)
    records = []
    for index, card in enumerate(tree.xpath('//article[contains(@class,"app-card-open")]')):
        links = card.xpath('.//h2/a')
        if len(links) != 1:
            raise ValueError('Catalogue card lacks a unique title link')
        link = links[0]
        url = urlparse(link.get('href', ''))
        if url.netloc not in {'', 'link.springer.com', 'link.springernature.com'}:
            raise ValueError('Unexpected article host')
        doi = unquote(url.path.removeprefix('/article/')).lower()
        if not url.path.startswith('/article/') or not re.fullmatch(r'10\.\d{4,9}/\S+', doi):
            raise ValueError('Invalid article DOI')
        types = card.xpath('.//span[contains(@class,"c-meta__type")]')
        literal_types = [' '.join(t.text_content().split()) for t in types]
        dates = []
        for node in card.xpath('.//div[@data-test="meta"]//span'):
            value = ' '.join(node.text_content().split())
            if re.fullmatch(r'\d{1,2} [A-Za-z]+ \d{4}', value):
                dates.append(datetime.strptime(value, '%d %B %Y').date().isoformat())
        records.append({'doi': doi, 'title': [' '.join(link.text_content().split())],
                        'journal_titles': [journal], 'literal_types': literal_types,
                        'display_dates': sorted(set(dates)), 'online_date': '',
                        'date_status': 'catalogue_display_date_semantics_not_confirmed',
                        'author_status': 'catalogue_byline_not_used_may_be_truncated',
                        'locator': f'article.app-card-open[{index + 1}]'})
    return records


def run() -> dict:
    folder = ROOT / 'supplement/springer_catalogue'
    db = connect()
    records = []
    counts = Counter()
    for path in sorted(folder.glob('*receipt.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        for item in parse_cards(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())):
            matches = []
            for wid, title, journal, state in db.execute(
                'SELECT work_id,title,journal,status FROM papers WHERE doi=?', (item['doi'],)
            ):
                matched = signature_match({'title': title, 'journal_name': journal}, item, set())
                matches.append({'work_id': wid, 'identity_matched': matched, 'eligibility': state})
                counts['matched_current_candidates' if matched else 'identity_conflicts'] += 1
            records.append({**item, 'matches': matches, 'source_url': receipt['url'],
                            'source_params': receipt.get('params', {}),
                            'raw_receipt': receipt['raw_receipt'], 'qualification_applied': False})
    db.close()
    atomic_json(folder / 'cached_cards.json', records)
    result = {'cards': len(records), 'unique_dois': len({r['doi'] for r in records}),
              'network_requests': 0, 'qualification_applied': False, **counts}
    atomic_json(folder / 'cached_cards_status.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()))
