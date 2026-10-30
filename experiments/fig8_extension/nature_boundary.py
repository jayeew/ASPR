"""Save publisher evidence for Analysis and retraction boundary decisions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pyarrow.parquet as pq
from lxml import html

from .nature_types import OUT, Publisher, clean_text, parse_catalogue
from .stage0 import write_csv, write_json


def page_evidence(publisher: Publisher, row: dict[str, Any]) -> dict[str, Any]:
    root = html.fromstring(publisher.get(row['article_url']))
    meta: dict[str, list[str]] = {}
    for element in root.xpath('//meta[@name and @content]'):
        meta.setdefault(element.get('name'), []).append(element.get('content'))
    abstracts = root.xpath('//section[@data-title="Abstract"]')
    relations = []
    for paragraph in root.xpath('//div[contains(@class,"c-article-body")]//p'):
        text = clean_text(paragraph)
        if text.startswith(('Retraction to:', 'Retraction of:', 'Editorial Expression of Concern to:', 'Addendum to:')):
            relations.append({'text': text, 'target_dois': [link.removeprefix('https://doi.org/')
                for link in paragraph.xpath('.//a/@href') if link.startswith('https://doi.org/')]})
    return {**row, 'metadata': meta, 'publisher_abstract': '\n'.join(clean_text(x) for x in abstracts),
            'html_lang': root.get('lang'),
            'publisher_affiliations': [clean_text(x) for x in root.xpath('//ol[contains(@class,"c-article-author-affiliation__list")]/li')],
            'notice_relations': relations}


def main() -> None:
    publisher = Publisher(OUT)
    catalogue = pq.read_table(OUT / 'publisher_catalogue.parquet').to_pylist()
    url = 'https://www.nature.com/nature/articles?type=retraction&year=2026'
    current = parse_catalogue(publisher.get(url), url, 2026)
    current_rows = current['rows']
    for page in range(2, current['last_page'] + 1):
        current_rows.extend(parse_catalogue(publisher.get(url + f'&page={page}'), url + f'&page={page}', 2026)['rows'])
    if len(current_rows) != current['expected_count'] or any(r['publisher_type'] != 'Retraction' for r in current_rows):
        raise ValueError('Incomplete or wrongly filtered current retraction catalogue')
    write_json(OUT / 'retraction_catalogue_2026.json', current)
    selected = [r for r in catalogue if r['publisher_type'] in ['Analysis', 'Retraction', 'Editorial Expression of Concern']]
    selected.extend(current_rows)
    evidence = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(page_evidence, publisher, row) for row in selected]):
            evidence.append(future.result())
    evidence.sort(key=lambda r: r['doi'])
    write_json(OUT / 'boundary_article_evidence.json', evidence)
    notices = []
    for row in evidence:
        if row['publisher_type'] == 'Analysis':
            continue
        if not row['notice_relations']:
            raise ValueError(f'Missing explicit notice relation: {row["doi"]}')
        for relation in row['notice_relations']:
            for target in relation['target_dois']:
                notices.append({'notice_doi': row['doi'], 'notice_type': row['publisher_type'],
                    'notice_date': row['publisher_date'], 'notice_url': row['article_url'],
                    'target_doi': target, 'relation_text': relation['text']})
    write_csv(OUT / 'publisher_notices.csv', notices)
    write_json(OUT / 'publisher_notices.json', notices)
    print(f'Saved {len(evidence)} boundary pages and {len(notices)} explicit notice links')


if __name__ == '__main__':
    main()
