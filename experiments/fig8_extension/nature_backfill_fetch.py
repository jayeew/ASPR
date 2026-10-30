"""Fetch missing Nature metadata with immutable raw pages and DOI checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
import requests
from lxml import html

from .nature_types import OUT as BASE
from .nature_types import Publisher, clean_text
from .stage0 import write_json

OUT = BASE.parent / 'nature_metadata_backfill'


class DirectPublisher(Publisher):
    def get(self, url: str) -> str:
        if getattr(self.local, 'session', None) is None:
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        return super().get(url)


def parse_page(text: str, doi: str) -> dict[str, Any]:
    root = html.fromstring(text)
    meta: dict[str, list[str]] = {}
    for element in root.xpath('//meta[@name and @content]'):
        meta.setdefault(element.get('name'), []).append(element.get('content'))
    if meta.get('citation_doi') != [doi] or meta.get('citation_journal_title') != ['Nature']:
        raise ValueError('Publisher DOI/journal identity check failed')
    if meta.get('citation_article_type') != ['Article']:
        raise ValueError('Publisher article type changed or was not Article')
    paragraphs = root.xpath('//section[@data-title="Abstract"]//div[contains(@class,"c-article-section__content")]/p')
    if not paragraphs:
        paragraphs = root.xpath('//div[@id="Abs1-content"]/p')
    # Remove citation anchors only, preserving scientific superscripts/subscripts.
    for paragraph in paragraphs:
        for anchor in paragraph.xpath('.//sup[a[@data-test="citation-ref"]]'):
            anchor.drop_tree()
        for element in paragraph.xpath('.//sup|.//sub'):
            content = element.text_content()
            marker = '^' if element.tag == 'sup' else '_'
            for child in list(element):
                element.remove(child)
            element.text = marker + '{' + content + '}'
    abstract = '\n\n'.join(' '.join(p.text_content().split()) for p in paragraphs)
    affiliations = []
    for item in root.xpath('//ol[contains(@class,"c-article-author-affiliation__list")]/li'):
        address = item.xpath('./p[contains(@class,"c-article-author-affiliation__address")]')
        affiliations.append({'affiliation_id': item.get('id'),
                             'address': clean_text(address[0]) if len(address) == 1 else '',
                             'authors_text': ' '.join(clean_text(x) for x in item.xpath('./p[contains(@class,"authors-list")]'))})
    authors = []
    primary_ids = {a['affiliation_id'] for a in affiliations}
    present_ids = {item.get('id')[1:] for item in root.xpath('//*[starts-with(@id,"nAff")]')}
    for item in root.xpath('//ol[contains(@class,"c-article-authors-search")]/li[starts-with(@id,"auth-")]'):
        names = item.xpath('.//span[contains(@class,"c-article-authors-search__title")]')
        all_refs = re.findall(r'(?<=-)Aff\d+', item.get('id', ''))
        authors.append({'name': clean_text(names[0]) if len(names) == 1 else '',
                       'publisher_author_id': item.get('id'),
                       'affiliation_ids': [ref for ref in all_refs if ref not in present_ids],
                       'excluded_present_address_ids': [ref for ref in all_refs if ref in present_ids],
                       'author_source': 'publisher_author_node'})
    if not authors and affiliations:
        by_name: dict[str, list[str]] = {}
        for aff in affiliations:
            for name in re.split(r',\s*|\s*&\s*', aff['authors_text']):
                if name.strip():
                    by_name.setdefault(name.strip(), []).append(aff['affiliation_id'])
        authors = [{'name': name, 'publisher_author_id': None, 'affiliation_ids': refs,
                    'author_source': 'publisher_affiliation_author_list'} for name, refs in by_name.items()]
    known = primary_ids
    unresolved_refs = [a for a in authors if not a['name'] or not a['affiliation_ids'] or not set(a['affiliation_ids']).issubset(known)]
    return {'doi': doi, 'metadata': meta, 'abstract_text': abstract,
            'abstract_status': 'publisher_abstract_verified' if len(abstract) >= 100 else 'missing_or_short_publisher_abstract',
            'affiliations': affiliations, 'authors': authors,
            'author_affiliation_parse_complete': bool(authors) and not unresolved_refs,
            'unresolved_author_refs': unresolved_refs}


def fetch(row: dict[str, Any], old: Publisher, new: Publisher) -> dict[str, Any]:
    doi, url = row['doi'], row['article_url']
    token = hashlib.sha256(url.encode()).hexdigest()
    publisher = old if (BASE / 'publisher_raw' / (token + '.html.gz')).exists() else new
    text = publisher.get(url)
    receipt = publisher.out / 'publisher_raw' / (token + '.html.json')
    result = parse_page(text, doi)
    result.update({'source_url': url, 'source_receipt': str(receipt),
                   'source': json.loads(receipt.read_text()), 'processed_at': datetime.now(timezone.utc).isoformat()})
    path = OUT / 'publisher_records' / (doi.split('/')[1] + '.json')
    write_json(path, result)
    return {'doi': doi, 'status': 'ok', 'abstract_status': result['abstract_status'],
            'author_n': len(result['authors']), 'affiliation_n': len(result['affiliations']),
            'author_affiliation_parse_complete': result['author_affiliation_parse_complete']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['abstracts', 'affiliations'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    rows = pq.read_table(BASE / 'research_roster_locked.parquet').to_pylist()
    selected = [r for r in rows if (not r['has_abstract'] if args.phase == 'abstracts' else
                                   not r['all_observed_authors_have_institution'])]
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = DirectPublisher(BASE), DirectPublisher(OUT)
    results = []
    with (OUT / 'fetch_attempts.jsonl').open('a') as attempts, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, row, old, new): row for row in selected}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except (RuntimeError, ValueError, OSError) as error:
                result = {'doi': row['doi'], 'status': 'failed', 'error': str(error)}
            result['attempted_at'] = datetime.now(timezone.utc).isoformat()
            results.append(result)
            attempts.write(json.dumps(result, ensure_ascii=False) + '\n')
            attempts.flush()
            if index % 25 == 0 or index == len(selected):
                print(f'{args.phase} {index}/{len(selected)}; failures={sum(r["status"] != "ok" for r in results)}', flush=True)
    write_json(OUT / (args.phase + '_fetch_summary.json'), sorted(results, key=lambda r: r['doi']))


if __name__ == '__main__':
    main()
