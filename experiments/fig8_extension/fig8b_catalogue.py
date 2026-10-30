"""Match annual publisher imprint lists by ISSN, preserving historical and ambiguous states."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import openpyxl

from .fig8b_collect import OUT
from .stage0 import write_csv, write_json


def main() -> None:
    with (OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: r for r in csv.DictReader(stream)}
    annual = []
    for year in [2023, 2024, 2025, 2026]:
        path = OUT / 'official' / f'journals_catalog_{year}.xlsx'
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.worksheets[0]
        header = None
        index: dict[str, list] = defaultdict(list)
        for row_number, values in enumerate(sheet.values, 1):
            if values[0] == 'Title':
                header = values
            elif header and values[0]:
                record = {k: v for k, v in zip(header, values) if k}
                record['row_number'] = row_number
                for key in ['ISSN print','ISSN electronic']:
                    if record.get(key):
                        index[str(record[key]).strip()].append(record)
        for source_id, journal in journals.items():
            issns = set(json.loads(journal['issn'] or '[]') or []) | {journal['issn_l']}
            matched = {r['row_number']: r for issn in issns if issn for r in index.get(issn, [])}
            imprints = {r['Imprint'] for r in matched.values()}
            imprint = next(iter(imprints)) if len(imprints) == 1 else None
            status = 'dated_catalogue_exact_issn' if len(matched) == 1 else 'ambiguous_multiple_catalogue_rows' if matched else 'not_in_february_catalogue_unresolved'
            portfolio = imprint == 'Nature Portfolio' if status == 'dated_catalogue_exact_issn' else None
            if journal['official_journal_url'] in ['https://www.nature.com/ismej/','https://www.nature.com/ismecomms/'] and year >= 2024:
                imprint, portfolio, status = 'Oxford University Press', False, 'official_transfer_2024_01_01'
            if journal['official_journal_url'] == 'https://www.nature.com/mi/' and year >= 2023:
                imprint, portfolio, status = 'Elsevier', False, 'official_transfer_2023_01_01'
            registry_other = (journal['nature_portfolio_openalex_current'] != 'True' and journal['nature_portfolio_official_current'] != 'True'
                              and bool(journal['host_organization']) and journal['springer_nature_group_current'] != 'True')
            comparison = ('nature_portfolio_annual_confirmed' if portfolio is True else 'other_imprint_annual_confirmed' if portfolio is False
                          else 'other_publisher_registry_supported_history_unverified' if registry_other else 'unresolved_nature_or_group_candidate')
            annual.append({'source_id': source_id, 'journal_name': journal['journal_name'], 'year': year,
                'canonical_journal_key': 'issn-l:' + journal['issn_l'] if journal['issn_l'] else source_id,
                'imprint': imprint, 'nature_portfolio_year': portfolio, 'status': status, 'comparison_group': comparison,
                'catalogue_file': str(path.relative_to(OUT)), 'catalogue_version': f'February {year}',
                'catalogue_rows': list(matched), 'catalogue_titles': [r['Title'] for r in matched.values()],
                'catalogue_comments': [r.get('Comments') for r in matched.values()],
                'transfer_evidence_url': journal['official_journal_url'] if status.startswith('official_transfer') else '',
                'membership_scope': 'dated annual imprint; midyear changes/new starts still require specific evidence'})
        workbook.close()
    write_csv(OUT / 'journal_publisher_by_year.csv', annual)
    write_csv(OUT / 'nature_portfolio_journals_2023_2025.csv', [r for r in annual if r['year'] <= 2025 and r['nature_portfolio_year'] is True])
    write_csv(OUT / 'publisher_label_corrections.csv', [r for r in annual if r['year'] <= 2025 and r['nature_portfolio_year'] is True
        and journals[r['source_id']]['nature_portfolio_openalex_current'] != 'True'])
    write_json(OUT / 'annual_publisher_summary.json', {str(year): dict(Counter(
        'nature_portfolio_confirmed' if r['nature_portfolio_year'] is True else 'non_nature_imprint_confirmed' if r['nature_portfolio_year'] is False else 'unresolved'
        for r in annual if r['year'] == year)) for year in [2023,2024,2025,2026]})
    groups: dict[str, list] = defaultdict(list)
    for journal in journals.values():
        groups[journal['issn_l'] or journal['source_id']].append(journal['source_id'])
    write_csv(OUT / 'journal_source_aliases.csv', [{'issn_l_or_source_id': k, 'source_ids': v, 'source_count': len(v)} for k, v in groups.items() if len(v) > 1])
    print('Annual memberships', len(annual), flush=True)


if __name__ == '__main__':
    main()
