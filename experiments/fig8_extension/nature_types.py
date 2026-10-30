"""Verify Nature article types against the publisher's complete annual catalogues."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from lxml import html

from .stage0 import ROOT, write_csv, write_json

OUT = ROOT / 'outputs/fig8_data/nature_type_verification'


def clean_text(element: Any) -> str:
    return ' '.join(' '.join(element.itertext()).split())


def parse_catalogue(text: str, url: str, year: int) -> dict[str, Any]:
    root = html.fromstring(text)
    selected = [clean_text(x) for x in root.xpath('//span[contains(@class,"c-facet__selected")]')]
    count = next((int(match.group(1).replace(',', '')) for value in selected
                  if (match := re.fullmatch(rf'{year} \(([\d,]+)\)', value))), None)
    if count is None:
        raise ValueError('Missing publisher annual count; not a catalogue page')
    rows = []
    for card in root.xpath('//article[.//a[contains(@class,"c-card__link")]]'):
        link = card.xpath('.//a[contains(@class,"c-card__link")]')[0]
        kind = card.xpath('.//span[contains(@class,"c-meta__type")]')
        dates = card.xpath('.//time/@datetime')
        article_url = urljoin('https://www.nature.com', link.get('href'))
        if len(kind) != 1 or len(dates) != 1:
            raise ValueError(f'Unclear catalogue type/date for {article_url}')
        article_id = urlparse(article_url).path.rsplit('/', 1)[-1]
        rows.append({'article_id': article_id, 'doi': '10.1038/' + article_id,
            'title': clean_text(link), 'publisher_type': clean_text(kind[0]),
            'publisher_date': dates[0], 'publisher_year': int(dates[0][:4]),
            'article_url': article_url, 'catalogue_url': url, 'catalogue_year': year})
    pages = [int(parse_qs(urlparse(href).query).get('page', ['1'])[0])
             for href in root.xpath('//ul[contains(@class,"c-pagination")]//a/@href')]
    facets = [{'label': clean_text(a), 'url': urljoin(url, a.get('href'))}
              for a in root.xpath('//nav[@id="Article-Type-target"]//a')]
    return {'expected_count': count, 'last_page': max(pages, default=1), 'rows': rows, 'facets': facets}


class Publisher:
    def __init__(self, out: Path) -> None:
        self.out, self.local = out, threading.local()

    def get(self, url: str) -> str:
        token = hashlib.sha256(url.encode()).hexdigest()
        path = self.out / 'publisher_raw' / f'{token}.html.gz'
        if path.exists():
            return gzip.decompress(path.read_bytes()).decode()
        session = getattr(self.local, 'session', None)
        if session is None:
            session = self.local.session = requests.Session()
        status: str | int = 'network'
        for attempt in range(3):
            try:
                response = session.get(url, timeout=45)
                status = response.status_code
                if response.ok and '<html' in response.text.lower():
                    text = response.text
                    # An authorization/challenge page is not evidence of article content.
                    if 'idp.nature.com/authorize' in response.url:
                        raise ValueError('publisher authorization page')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(gzip.compress(text.encode()))
                    write_json(path.with_suffix('.json'), {'requested_url': url, 'final_url': response.url,
                        'retrieved_at': datetime.now(timezone.utc).isoformat(), 'status': status,
                        'sha256': hashlib.sha256(text.encode()).hexdigest()})
                    return text
                if status not in (429, 500, 502, 503, 504):
                    break
            except (requests.RequestException, ValueError):
                status = 'network_or_interstitial'
            time.sleep(2 ** attempt)
        raise RuntimeError(f'Publisher fetch failed: {status}: {url}')


def fetch_page(publisher: Publisher, year: int, page: int) -> dict[str, Any]:
    url = f'https://www.nature.com/nature/articles?year={year}&page={page}'
    return parse_catalogue(publisher.get(url), url, year)


def catalogue(workers: int) -> None:
    publisher = Publisher(OUT)
    first = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_page, publisher, year, 1): year for year in [2023, 2024, 2025]}
        for future in as_completed(futures):
            year = futures[future]
            first[year] = future.result()
            print(f'{year}: {first[year]["expected_count"]} entries, {first[year]["last_page"]} pages', flush=True)
    rows = [row for data in first.values() for row in data['rows']]
    jobs = [(year, page) for year in sorted(first) for page in range(2, first[year]['last_page'] + 1)]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_page, publisher, year, page): (year, page) for year, page in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            year, page = futures[future]
            try:
                data = future.result()
                if data['expected_count'] != first[year]['expected_count']:
                    raise ValueError('Publisher catalogue count changed during pagination')
                rows.extend(data['rows'])
            except (RuntimeError, ValueError) as error:
                failures.append({'year': year, 'page': page, 'error': str(error)})
            if i % 25 == 0 or i == len(jobs):
                print(f'Catalogue pages {i}/{len(jobs)}; records {len(rows)}; failed {len(failures)}', flush=True)
    write_json(OUT / 'catalogue_failures.json', failures)
    write_json(OUT / 'catalogue_first_pages.json', first)
    write_csv(OUT / 'publisher_catalogue_partial.csv', rows)
    if failures:
        raise RuntimeError(f'{len(failures)} catalogue pages unresolved; resume cached fetches')
    by_year = Counter(row['catalogue_year'] for row in rows)
    for year, data in first.items():
        selected = [r for r in rows if r['catalogue_year'] == year]
        if by_year[year] != data['expected_count'] or len({r['article_id'] for r in selected}) != by_year[year]:
            raise ValueError(f'Catalogue incomplete or duplicated in {year}')
    rows.sort(key=lambda r: (r['catalogue_year'], r['article_id']))
    pq.write_table(pa.Table.from_pylist(rows), OUT / 'publisher_catalogue.parquet')
    write_csv(OUT / 'publisher_catalogue.csv', rows)
    write_json(OUT / 'catalogue_audit.json', {'rows': len(rows), 'counts_by_year': dict(by_year),
        'type_counts': dict(Counter(r['publisher_type'] for r in rows)),
        'date_year_mismatches': sum(r['publisher_year'] != r['catalogue_year'] for r in rows),
        'cross_year_duplicate_ids': len(rows) - len({r['article_id'] for r in rows})})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue(args.workers)


if __name__ == '__main__':
    main()
