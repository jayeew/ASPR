"""Capture official journal evidence without treating HTTP success as verification."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from lxml import html

from .fig8b_collect import OUT
from .stage0 import write_json


def capture(url: str) -> dict:
    token = hashlib.sha256(url.encode()).hexdigest()
    base = OUT / 'official' / token
    if base.with_suffix('.json').exists():
        return json.loads(base.with_suffix('.json').read_text())
    result = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=(15, 40))
        result.update(http_status=response.status_code, final_url=response.url)
        base.with_suffix('.html').write_text(response.text)
        result['raw_file'] = str(base.with_suffix('.html').relative_to(OUT))
        if response.ok:
            tree = html.fromstring(response.text)
            result['title'] = ' '.join(tree.xpath('//title/text()'))
            result['text'] = ' '.join(tree.text_content().split())
            result['links'] = [{'text': a.text_content().strip(), 'href': a.get('href')}
                               for a in tree.xpath('//a[@href]')]
            result['status'] = 'captured_requires_content_verification'
        else:
            result['status'] = 'http_error'
    except requests.RequestException as exc:
        result.update(status='network_error', error_type=type(exc).__name__)
    write_json(base.with_suffix('.json'), result)
    return result


def main() -> None:
    with (OUT / 'publisher_review_queue.csv').open(encoding='utf-8-sig') as stream:
        urls = {r['official_journal_url'] for r in csv.DictReader(stream) if r['official_journal_url'].startswith('https://www.nature.com/')}
    urls.update('https://www.nature.com/nature-portfolio/about-journals/' + s for s in
        ['nature','research-journals','nature-reviews','nature-communications',
         'communications-journals','npj-series','scientific-series'])
    urls.update(['https://www.nature.com/hortres/','https://www.nature.com/psp/'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(capture, sorted(urls)))
    write_json(OUT / 'official_capture_index.json', results)
    print('Captured', len(results), flush=True)


if __name__ == '__main__':
    main()
