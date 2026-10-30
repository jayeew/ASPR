"""Read DOI-bound Frontiers article text from retained page hydration data."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from lxml import html

from .fig8b_qualification_decide import key
from .frontiers_authors import decode
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/frontiers_perspective_policy'


def content_text(values: list, index: int, active: frozenset[int] = frozenset()) -> str:
    """Read XML text nodes in order; omit attributes and preserve block boundaries."""
    if not isinstance(index, int) or index < 0 or index >= len(values) or index in active:
        raise ValueError('Invalid or cyclic content reference')
    value = values[index]; active = active | {index}
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ''.join(content_text(values, child, active) for child in value)
    if isinstance(value, dict):
        parts = []
        for tag, child in value.items():
            if tag.startswith(('@', ':')):
                continue
            parts.append(content_text(values, child, active))
            if tag in {'p', 'title', 'sec', 'list-item', 'disp-formula'}:
                parts.append('\n')
        return ''.join(parts)
    return ''


def bound_article(tree: html.HtmlElement, values: list, doi: str) -> tuple[dict, str, dict]:
    """Resolve the explicit current-article link, not unrelated reference objects."""
    if tree.xpath('//meta[@name="citation_doi"]/@content') != [doi]:
        raise ValueError('Page DOI unconfirmed')
    stores = [v for v in values if isinstance(v, dict)
              and {'currentArticle', 'xmlParsedArticleContent'} <= set(v)]
    if len(stores) != 1:
        raise ValueError('Article store is ambiguous')
    store = stores[0]; article = values[store['currentArticle']]
    body = values[store['xmlParsedArticleContent']]
    if not isinstance(article, dict) or not isinstance(body, dict):
        raise ValueError('Article content is not available')
    title = decode(values, article['title'])
    journal = decode(values, article['journal'])
    body_journal = decode(values, body['journal'])
    body_doi = str(decode(values, body['doi']))
    if (decode(values, article['doi']) != doi or body_doi not in {doi, doi.rsplit('.', 1)[-1]}
            or key(content_text(values, body['title'])) != key(title)
            or tree.xpath('//meta[@name="citation_title"]/@content') != [title]
            or journal['electronicISSN'] != body_journal['issn']
            or journal['name'] != body_journal['title']):
        raise ValueError('Article/body DOI, title or journal disagreement')
    return body, title, journal


def extract(raw: bytes, doi: str) -> dict:
    tree = html.fromstring(raw)
    payloads = tree.xpath('//script[@id="__NUXT_DATA__"]/text()')
    if len(payloads) != 1:
        raise ValueError('Page hydration payload unconfirmed')
    values = json.loads(payloads[0])
    body, title, journal = bound_article(tree, values, doi)
    text = content_text(values, body['bodyContent']).strip()
    if not text:
        raise ValueError('Article body is empty')
    return {'doi': doi, 'title': title, 'journal': journal['name'],
            'abstract': content_text(values, body['abstractContent']).strip(),
            'body': text, 'body_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'locator': '__NUXT_DATA__: article store currentArticle + xmlParsedArticleContent.bodyContent'}


def run() -> None:
    inventory = json.loads((FOLDER / 'content_availability.json').read_text())
    destination = FOLDER / 'publisher_body_cache'; destination.mkdir(exist_ok=True)
    reviews = []
    for paper in inventory['records']:
        if paper['next_action'] != 'extract_cached_publisher_article_body':
            continue
        for source in paper['sources']:
            if source['state'] != 'publisher_html_cached_content_extraction_required':
                continue
            record = {'work_id': paper['work_id'], 'doi': paper['doi'], 'receipt': source['receipt']}
            try:
                raw = gzip.decompress((ROOT / 'fields/qualification' / source['receipt']).read_bytes())
                content = extract(raw, paper['doi'])
                path = destination / (hashlib.sha256(source['receipt'].encode()).hexdigest() + '.json')
                atomic_json(path, {**record, **content, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                  'qualification_adjudicated': False})
                record.update(status='matched_cached_article_body_extracted',
                              content_receipt=str(path.relative_to(ROOT)), body_characters=len(content['body']))
            except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
                record.update(status='cached_body_unresolved', reason=str(error))
            reviews.append(record)
    atomic_json(destination / 'status.json', {'records': reviews,
        'outcomes': dict(Counter(r['status'] for r in reviews)),
        'unique_papers_with_body': len({r['work_id'] for r in reviews if r['status']=='matched_cached_article_body_extracted'}),
        'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False})
    print(json.dumps({'outcomes': dict(Counter(r['status'] for r in reviews))}), flush=True)


if __name__ == '__main__':
    run()
