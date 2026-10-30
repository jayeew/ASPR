"""Recover explicit publisher redirects from cached Linking Hub locator pages."""
from __future__ import annotations

import gzip
import hashlib
import json
from urllib.parse import unquote, urlparse

from lxml import html

from .supplement_queue import ROOT
from .supplement_sources import atomic_json, publisher_url


def target(raw: bytes, doi: str) -> str:
    tree = html.fromstring(raw)
    identities = tree.xpath('//input[@id="id"]/@value')
    urls = tree.xpath('//input[@id="redirectURL"]/@value')
    if [value.lower() for value in identities] != [doi.lower()] or len(urls) != 1:
        return ''
    url = unquote(urls[0])
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'www.sciencedirect.com' or not parsed.path.startswith('/science/article/pii/'):
        return ''
    return url


def run() -> dict:
    folder = ROOT / 'supplement/elsevier_redirects'
    folder.mkdir(exist_ok=True)
    results = []
    for paper in json.loads((ROOT / 'supplement/pilot.json').read_text()):
        if paper['arm'] != 'elsevier':
            continue
        token = hashlib.sha256(publisher_url(paper['doi'], 'elsevier').encode()).hexdigest()
        path = ROOT / 'fields/qualification/supplement_parsed' / (token + '.json')
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / 'fields/qualification' / receipt['raw_receipt']).read_bytes())
        url = target(raw, paper['doi'])
        results.append({'candidate': paper, 'redirect_url': url,
                        'receipt': str(path.relative_to(ROOT)), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                        'status': 'publisher_redirect_target_not_yet_reviewed' if url else 'no_confirmed_target',
                        'qualification_unchanged': True})
    result = {'records': results, 'network_requests': 0,
              'confirmed_redirects': sum(bool(r['redirect_url']) for r in results)}
    atomic_json(folder / 'cache_review.json', result)
    print(json.dumps({'records': len(results), 'confirmed_redirects': result['confirmed_redirects']}), flush=True)
    return result


if __name__ == '__main__':
    run()
