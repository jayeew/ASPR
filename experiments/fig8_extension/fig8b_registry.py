"""Evidence-bound Nature Portfolio labels; group ownership is a separate variable."""
from __future__ import annotations

import csv
import json
import re
from urllib.parse import urljoin

from lxml import html

from .fig8b_collect import OUT
from .stage0 import DEFAULT_OUT, PORTFOLIO, write_csv, write_json


def normalized(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def main() -> None:
    tree = html.fromstring((OUT / 'official/siteindex.html').read_text())
    links = [a for a in tree.xpath('//div[starts-with(@id,"journals-")]//a[@href]')
             if re.fullmatch(r'(?:https://www.nature.com)?/[a-z0-9-]+/', a.get('href'))]
    titles = {normalized(a.text_content()): {'official_title': a.text_content().strip(),
        'journal_url': urljoin('https://www.nature.com', a.get('href'))} for a in links}
    families = {}
    captures = json.loads((OUT / 'official_capture_index.json').read_text())
    for receipt in captures:
        if '/nature-portfolio/about-journals/' in receipt['url']:
            for link in receipt.get('links', []):
                url = urljoin('https://www.nature.com', link['href']).rstrip('/')
                if re.fullmatch(r'https://www.nature.com/[a-z0-9-]+', url):
                    families[url] = receipt['url']
    entities = {r['id']: r for r in json.loads((DEFAULT_OUT / 'source_entities.json').read_text())}
    with (DEFAULT_OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['cohort'] != 'nature']
    for row in rows:
        entity = entities.get(row['source_id'], {})
        lineage = entity.get('host_organization_lineage') or []
        official = titles.get(normalized(row['journal_name']))
        family_url = families.get((official or {}).get('journal_url', '').rstrip('/'))
        row.update(springer_nature_group_current='https://openalex.org/P4310319965' in lineage,
            nature_portfolio_openalex_current=PORTFOLIO in lineage,
            nature_portfolio_official_current=bool(family_url or (official and PORTFOLIO in lineage)),
            official_siteindex_title_match=bool(official),
            official_family_evidence_url=family_url or '',
            publisher_label_status='official_portfolio_family_list' if family_url else 'official_directory_and_registry_agree' if official and PORTFOLIO in lineage
                else 'registry_only_historical_membership_unverified' if PORTFOLIO in lineage
                else 'official_hosted_title_registry_disagreement' if official else 'other_or_unverified',
            official_journal_url=(official or {}).get('journal_url'),
            official_directory_url='https://www.nature.com/siteindex' if official else '',
            portfolio_2023_2025_status='unverified_requires_dated_evidence',
            scope='OpenAlex primary-field journal census; current publisher labels; candidate work types')
    for row in rows:
        url = row['official_journal_url'] or ''
        if url in ('https://www.nature.com/ismej/', 'https://www.nature.com/ismecomms/'):
            row.update(nature_portfolio_official_current=False, publisher_label_status='official_transfer_2024_01_01',
                       portfolio_2023_2025_status='nature_until_2023_12_31_oup_from_2024_01_01')
        if url == 'https://www.nature.com/mi/':
            row.update(nature_portfolio_official_current=False, publisher_label_status='official_transfer_2023_01_01',
                       portfolio_2023_2025_status='elsevier_since_2023_01_01')
        if row['journal_name'] == 'Horticulture Research':
            row.update(nature_portfolio_official_current=False, publisher_label_status='official_no_longer_springer_nature',
                       official_journal_url='https://www.nature.com/hortres/')
    write_csv(OUT / 'journal_registry.csv', rows)
    write_json(OUT / 'official_directory_titles.json', list(titles.values()))
    write_csv(OUT / 'publisher_review_queue.csv', [r for r in rows if r['nature_portfolio_openalex_current'] or r['official_siteindex_title_match']])
    unique = {r['source_id']: r for r in rows}
    write_json(OUT / 'journal_audit.json', {'field_journal_rows': len(rows), 'unique_journals': len(unique),
        'nature_registry_journals': sum(r['nature_portfolio_openalex_current'] for r in unique.values()),
        'nature_official_current_journals': sum(r['nature_portfolio_official_current'] for r in unique.values()),
        'official_only_journals': sum(r['official_siteindex_title_match'] and not r['nature_portfolio_openalex_current'] for r in unique.values()),
        'note': 'Directory hosting alone is not ownership; current membership is not historical membership.'})


if __name__ == '__main__':
    main()
