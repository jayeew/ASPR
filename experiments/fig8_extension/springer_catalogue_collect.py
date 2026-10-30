"""Finite journal-volume catalogue acquisition; apply literal types, not date clues."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter
from urllib.parse import parse_qsl, urlparse

from lxml import html

from .derived_refresh import refresh
from .ojs_metadata_harvest import fetch
from .publisher_delta_apply import apply_available
from .springer_catalogue_cache import parse_cards
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/springer_catalogue/molecular_volumes'
URL = 'https://link.springer.com/journal/11033/articles'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = FOLDER / 'status.json'
        if marker.exists() and json.loads(marker.read_text()).get('stop_reason'):
            return
        db = connect()
        scope_path = FOLDER / 'scope.json'
        if not scope_path.exists():
            papers = [dict(zip(['doi', 'work_id', 'title', 'journal_name'], row)) for row in db.execute(
                "SELECT doi,work_id,title,journal FROM papers WHERE source_id='https://openalex.org/S197962172' AND status LIKE 'pending%' AND doi<>''")]
            atomic_json(scope_path, {'papers': papers, 'volumes': ['50', '51', '52'],
                        'scope': 'Existing pending journal candidates only; volume year is not online year'})
        db.close()
        scope = json.loads(scope_path.read_text())
        papers = {p['doi']: p for p in scope['papers']}
        records = FOLDER / 'parsed'; records.mkdir(exist_ok=True)
        counts = Counter(); stop = ''; jobs = {}
        pages = []
        for volume in scope['volumes']:
            params = {'filter-by-volume': volume}
            seen = set()
            while True:
                token = json.dumps(params, sort_keys=True)
                if token in seen:
                    stop = 'pagination_cycle'; break
                seen.add(token)
                receipt = fetch(URL, params)
                pages.append(receipt)
                if receipt.get('http_status') != 200:
                    stop = 'source_unavailable_no_automatic_retry'; break
                raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                cards = parse_cards(raw)
                if not cards or any(c['journal_titles'] != ['Molecular Biology Reports'] for c in cards):
                    stop = 'catalogue_identity_or_empty_page_review'; break
                counts['pages'] += 1; counts['cards'] += len(cards)
                for item in cards:
                    paper = papers.get(item['doi'])
                    if not paper:
                        continue
                    if not signature_match(paper, item, set()):
                        counts['identity_conflicts'] += 1; continue
                    record = {'source': 'publisher', 'route': 'springer', 'status': 'parsed',
                              'source_mode': 'journal_catalogue_literal_type_only',
                              'url': URL, 'params': params, 'locator': item['locator'],
                              'raw_receipt': '../../' + receipt['raw_receipt'],
                              'parsed': {'doi': [item['doi']], 'title': item['title'],
                                         'issns': [], 'journal_titles': item['journal_titles'],
                                         'article_types': item['literal_types'], 'online_date': '',
                                         'date_precision': 'not_used_catalogue_date',
                                         'abstract': '', 'publisher': []},
                              'display_dates_not_applied': item['display_dates'],
                              'byline_not_applied': item['author_status']}
                    path = records / (hashlib.sha256(item['doi'].encode()).hexdigest() + '.json')
                    if path.exists() and json.loads(path.read_text()) != record:
                        counts['duplicate_catalogue_receipt_held'] += 1; continue
                    atomic_json(path, record)
                    jobs[item['doi']] = str(path.relative_to(ROOT))
                atomic_json(marker, {'status': 'collecting_fixed_volumes', **counts})
                tree = html.fromstring(raw.decode('utf-8'))
                links = tree.xpath('//a[@rel="next"]/@href')
                if not links:
                    break
                if len(links) != 1:
                    stop = 'ambiguous_pagination'; break
                link = urlparse(links[0]); following = dict(parse_qsl(link.query))
                if link.netloc or link.path not in {'', '/journal/11033/articles'} or following.get('filter-by-volume') != volume:
                    stop = 'pagination_outside_fixed_volume'; break
                params = following
            if stop:
                break
        atomic_json(FOLDER / 'page_receipts.json', pages)
        result = apply_available(supplied_jobs=sorted(jobs.items())) if jobs else {}
        atomic_json(marker, {'status': 'catalogue_type_path_applied', 'stop_reason': stop,
                            'fixed_candidates': len(papers), 'matched_dois': len(jobs),
                            'application': result, 'final_task_closure': False, **counts})
        refresh()
        print(json.dumps(json.loads(marker.read_text())), flush=True)


if __name__ == '__main__':
    run()
