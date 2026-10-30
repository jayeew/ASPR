"""Audit fixed-scope JOSS publisher deposits without promoting untyped dates."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import date

from lxml import etree

from .fig8b_qualification_decide import key
from .joss_deposit_collect import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def value(node: etree._Element, expression: str) -> str:
    return ' '.join(' '.join(node.xpath(expression)).split())


def run() -> dict:
    scope = json.loads((FOLDER / 'source_index.json').read_text())['records']
    db, results = connect(), []
    try:
        for row in scope:
            receipt = FOLDER / 'publisher_deposits' / (row['doi'].split('/')[-1] + '.json')
            if not receipt.exists():
                results.append({'doi': row['doi'], 'status': 'not_requested'})
                continue
            fetched = json.loads(receipt.read_text())
            if fetched['status'] != 200:
                results.append({'doi': row['doi'], 'status': 'retained_source_failure', 'receipt': str(receipt.relative_to(ROOT))})
                continue
            raw = gzip.decompress((ROOT / fetched['raw_receipt']).read_bytes())
            if hashlib.sha256(raw).hexdigest() != fetched['sha256'] or not fetched['git_blob_verified']:
                raise ValueError('Publisher deposit hash mismatch')
            root = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            articles = root.xpath('//*[local-name()="journal_article"]')
            if len(articles) != 1:
                raise ValueError('Expected one deposited article')
            article = articles[0]
            doi = value(article, './*[local-name()="doi_data"]/*[local-name()="doi"]/text()').lower()
            title = value(article, './*[local-name()="titles"]/*[local-name()="title"]//text()')
            issns = root.xpath('//*[local-name()="journal_metadata"]/*[local-name()="issn"]/text()')
            current = db.execute('SELECT title FROM papers WHERE work_id=? AND doi=?', (row['work_id'], doi)).fetchone()
            matched = doi == row['doi'] and current is not None and key(title) == key(current[0]) and '2475-9066' in issns
            dates = []
            for node in article.xpath('./*[local-name()="publication_date"]'):
                parts = [value(node, './*[local-name()="' + field + '"]/text()') for field in ('year', 'month', 'day')]
                if all(parts):
                    day = date(*map(int, parts)).isoformat()
                    dates.append({'date': day, 'media_type': node.get('media_type', 'unspecified')})
            results.append({'doi': doi, 'work_id': row['work_id'], 'status': 'matched_publisher_deposit' if matched else 'identity_unresolved',
                'title': title, 'issns': issns, 'publication_dates': dates, 'receipt': fetched['raw_receipt'],
                'sha256': fetched['sha256'], 'source_url': fetched['url'], 'online_date_promoted': False,
                'remaining': 'Software article eligibility and explicit online-date semantics require separate evidence; publication_type=full_text is not original research.'})
    finally:
        db.close()
    summary = {'scope': len(scope), 'statuses': dict(Counter(r['status'] for r in results)),
        'matched_with_untyped_publication_date': sum(r['status'] == 'matched_publisher_deposit' and any(d['media_type'] == 'unspecified' for d in r['publication_dates']) for r in results),
        'qualification_changes': 0, 'network_requests': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'publisher_deposit_review.json', {'summary': summary, 'records': results})
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    run()
