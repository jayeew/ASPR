"""Publisher-first abstract repair queue for officially identified Nature Portfolio titles."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from lxml import html

from .fig8b_collect import OUT
from .nature_backfill_fetch import DirectPublisher
from .stage0 import write_json

PUBLISHER = DirectPublisher(OUT)


@lru_cache(maxsize=1)
def nature_catalogue() -> dict[str, dict[str, str]]:
    path = OUT.parent / 'nature_type_verification/publisher_catalogue.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig') as stream:
        return {r['doi']: r for r in csv.DictReader(stream)}


def parse(text: str, doi: str, issns: list[str]) -> dict[str, Any]:
    tree = html.fromstring(text)
    meta: dict[str, list[str]] = {}
    for element in tree.xpath('//meta[@name and @content]'):
        meta.setdefault(element.get('name'), []).append(element.get('content'))
    if [d.lower() for d in meta.get('citation_doi', [])] != [doi.lower()]:
        raise ValueError('DOI identity mismatch')
    if not set(meta.get('citation_issn', [])) & set(issns):
        raise ValueError('Journal ISSN identity mismatch')
    paragraphs = tree.xpath('//section[@data-title="Abstract"]//div[contains(@class,"c-article-section__content")]/p')
    if not paragraphs:
        paragraphs = tree.xpath('//div[@id="Abs1-content"]/p')
    for paragraph in paragraphs:
        for anchor in paragraph.xpath('.//sup[a[@data-test="citation-ref"]]'):
            anchor.drop_tree()
        for element in paragraph.xpath('.//sup|.//sub'):
            content = element.text_content()
            marker = '^' if element.tag == 'sup' else '_'
            for child in list(element):
                element.remove(child)
            element.text = marker + '{' + content + '}'
    return {'doi': doi, 'abstract': '\n\n'.join(' '.join(p.text_content().split()) for p in paragraphs),
        'publisher_type': meta.get('citation_article_type', []),
        'journal': meta.get('citation_journal_title', []), 'issn': meta.get('citation_issn', []),
        'publication_date': meta.get('citation_publication_date', []),
        'online_date': meta.get('citation_online_date', []) or tree.xpath('//time[@itemprop="datePublished"]/@datetime'),
        'status': 'publisher_doi_issn_verified'}


def fetch(item: tuple[str, str]) -> dict[str, Any]:
    doi, issns_json = item
    token = hashlib.sha256(doi.encode()).hexdigest()
    path = OUT / 'publisher_parsed' / f'{token}.json'
    if path.exists():
        return json.loads(path.read_text())
    catalogue = nature_catalogue()
    if doi in catalogue and set(json.loads(issns_json)) & {'0028-0836', '1476-4687'}:
        result = {'doi': doi, 'status': 'existing_nature_catalogue_reused',
                  'catalogue_url': catalogue[doi]['catalogue_url']}
        write_json(path, result)
        return result
    url = 'https://www.nature.com/articles/' + doi.split('/', 1)[1]
    try:
        result = parse(PUBLISHER.get(url), doi, json.loads(issns_json))
    except (RuntimeError, ValueError) as exc:
        result = {'doi': doi, 'status': 'unresolved', 'error': str(exc)}
    result['url'] = url
    result['raw_receipt'] = 'publisher_raw/' + hashlib.sha256(url.encode()).hexdigest() + '.html.gz'
    write_json(path, result)
    return result


def main() -> None:
    with (OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        sources = {r['source_id']: r for r in csv.DictReader(stream) if r['nature_portfolio_official_current'] == 'True'}
    with (OUT / 'journal_publisher_by_year.csv').open(encoding='utf-8-sig') as stream:
        annual_sources = {r['source_id'] for r in csv.DictReader(stream) if r['year'] in ['2023','2024','2025'] and r['nature_portfolio_year'] == 'True'}
    for source_id in annual_sources:
        sources.setdefault(source_id, {})
    connection = sqlite3.connect(OUT / 'publisher_queue.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS queue (doi TEXT PRIMARY KEY, issns TEXT, status TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS seen_annual (path TEXT PRIMARY KEY)')
    while True:
        seen = {r[0] for r in connection.execute('SELECT path FROM seen_annual')}
        files = sorted(p for p in (OUT / 'census_raw').glob('*/*.json.gz') if str(p) not in seen)
        for path in files:
            with gzip.open(path, 'rt') as stream:
                receipt = json.load(stream)
            for record in receipt['response']['results']:
                source = (record.get('primary_location') or {}).get('source') or {}
                doi = (record.get('doi') or '').removeprefix('https://doi.org/').lower()
                if source.get('id') in sources and not record.get('abstract_inverted_index') and doi.startswith('10.1038/'):
                    connection.execute('INSERT OR IGNORE INTO queue VALUES (?,?,?)',
                        (doi, json.dumps(source.get('issn') or [source.get('issn_l')]), 'pending'))
            connection.execute('INSERT INTO seen_annual VALUES (?)', (str(path),))
        connection.commit()
        pending = list(connection.execute("SELECT doi,issns FROM queue WHERE status='pending' ORDER BY doi"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(fetch, pending):
                connection.execute('UPDATE queue SET status=? WHERE doi=?', (result['status'], result['doi']))
                connection.commit()
        write_json(OUT / 'publisher_status.json', dict(connection.execute('SELECT status,count(*) FROM queue GROUP BY status')))
        print('publisher queue', dict(connection.execute('SELECT status,count(*) FROM queue GROUP BY status')), flush=True)
        states = [json.loads(p.read_text()) for p in (OUT / 'shards').glob('*.json')]
        if (len(states) == 108 and all(s['status'] in ['complete','failed','reconciliation_failed'] for s in states)
                and not [p for p in (OUT / 'census_raw').glob('*/*.json.gz') if str(p) not in seen and p not in files]):
            break
        time.sleep(5)
    connection.close()


if __name__ == '__main__':
    main()
