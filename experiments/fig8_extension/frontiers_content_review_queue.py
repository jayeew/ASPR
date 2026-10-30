"""Unify the fixed Perspective content-review queue without deciding study design."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from lxml import etree

from .frontiers_content_cache import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, text


def run() -> None:
    inventory = json.loads((FOLDER / 'content_availability.json').read_text())['records']
    bodies: dict[str, list[str]] = {}
    for name in ['publisher_body_cache', 'missing_body_publisher']:
        status = FOLDER / name / 'status.json'
        if not status.exists():
            continue
        for row in json.loads(status.read_text())['records']:
            receipt = row.get('content_receipt') or row.get('body_receipt')
            if receipt:
                bodies.setdefault(row['work_id'], []).append(receipt)
    native_folder = FOLDER / 'native_body_cache'
    native_folder.mkdir(exist_ok=True)
    adjudication_path = FOLDER / 'content_adjudications/review.json'
    adjudications = ({r['work_id']: r for r in json.loads(adjudication_path.read_text())['records']}
                    if adjudication_path.exists() else {})
    db = connect()
    reviews = []
    for paper in inventory:
        for source in paper['sources']:
            if source['state'] != 'matched_native_body_available':
                continue
            raw = gzip.decompress((ROOT / 'fields/qualification' / source['receipt']).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == source['raw_sha256']
            article = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            dois = {text(n).lower() for n in article.xpath('./front/article-meta/article-id[@pub-id-type="doi"]')}
            assert paper['doi'] in dois
            body = article.find('./body')
            assert body is not None
            body_text = '\n\n'.join(text(node) for node in body)
            title = article.find('./front/article-meta/title-group/article-title')
            path = native_folder / (hashlib.sha256(source['receipt'].encode()).hexdigest() + '.json')
            atomic_json(path, {'work_id': paper['work_id'], 'doi': paper['doi'],
                'title': text(title) if title is not None else '',
                'abstract': '\n\n'.join(text(n) for n in article.findall('./front/article-meta/abstract')),
                'body': body_text, 'body_sha256': hashlib.sha256(body_text.encode()).hexdigest(),
                'source_receipt': source['receipt'], 'raw_sha256': source['raw_sha256'],
                'locator': 'article/body; excludes article/back references',
                'qualification_adjudicated': False})
            bodies.setdefault(paper['work_id'], []).append(str(path.relative_to(ROOT)))
        paths = sorted(set(bodies.get(paper['work_id'], [])))
        for path in paths:
            content = json.loads((ROOT / path).read_text())
            assert content['doi'] == paper['doi'] and content['work_id'] == paper['work_id']
            assert hashlib.sha256(content['body'].encode()).hexdigest() == content['body_sha256']
        current = db.execute('SELECT status FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()[0]
        review_status = 'content_adjudication_required' if paths else 'content_acquisition_unresolved'
        adjudication = adjudications.get(paper['work_id'])
        if adjudication and adjudication['content_receipt'] in paths:
            content = json.loads((ROOT / adjudication['content_receipt']).read_text())
            if content['body_sha256'] == adjudication['content_sha256']:
                review_status = ('content_review_applied_nonresearch'
                    if adjudication['classification'] == 'nonprimary' and current == 'excluded_primary_nonresearch'
                    else 'content_review_unresolved')
        reviews.append({'work_id': paper['work_id'], 'doi': paper['doi'],
            'current_qualification': current, 'content_receipts': paths,
            'review_status': review_status,
            'adjudication_receipt': str(adjudication_path.relative_to(ROOT)) if adjudication else '',
            'required_decision': 'Original empirical/theoretical/computational result versus synthesis, opinion or proposal; retain ambiguous cases',
            'final_task_closure': False})
    db.close()
    assert len(reviews) == len({r['work_id'] for r in reviews})
    result = {'records': reviews, 'papers': len(reviews),
        'outcomes': dict(Counter(r['review_status'] for r in reviews)),
        'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'content_review_queue.json', result)
    print(json.dumps(result['outcomes']), flush=True)


if __name__ == '__main__':
    run()
