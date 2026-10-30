"""Audit fixed publisher-window omissions against cached filtered batch responses."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict

from .fig8b_qualification_build import key, plain, match
from .boundary_publisher_record import receipt_records
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/boundary_review'
    scope = json.loads((folder / 'full_directory_scope.json').read_text())['dois']
    with (ROOT / 'supplement/boundary_candidates.csv').open(encoding='utf-8-sig') as stream:
        directory = {r['doi']: r for r in csv.DictReader(stream)}
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    registry = {(r['cohort'], r['source_id']) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    policy_path=folder/'aps_language_review.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    found = defaultdict(dict)
    queried = {}
    for path in sorted((folder / 'full_directory_raw').glob('*.gz')):
        with gzip.open(path, 'rt') as stream:
            receipt = json.load(stream)
        reference = str(path.relative_to(ROOT))
        for doi in receipt['requested_dois']:
            queried[doi] = (reference, receipt.get('status'))
        if receipt.get('status') != 'success':
            continue
        for item in receipt['response'].get('results', []):
            doi = (item.get('doi') or '').lower().removeprefix('https://doi.org/')
            if doi in receipt['requested_dois']:
                found[doi][item['id']] = (item, reference)
    rows = []
    counts = Counter()
    for doi in scope:
        row = {'doi': doi, 'publisher_receipt': directory[doi]['receipt'],
               'current_receipt': queried.get(doi, ('', ''))[0]}
        matches = list(found[doi].values())
        if not matches:
            status = ('batch_not_yet_processed' if doi not in queried else
                      'no_in_scope_record_returned_not_proof_of_absence' if queried[doi][1] == 'success' else
                      'batch_retrieval_unresolved')
        elif len(matches) != 1:
            status = 'multiple_current_work_identities'
        else:
            item, reference = matches[0]
            source = (item.get('primary_location') or {}).get('source') or {}
            field = ((item.get('primary_topic') or {}).get('field') or {}).get('id', '')
            cohort = {'13': 'biochemistry', '25': 'materials', '31': 'physics'}.get(field.rsplit('/', 1)[-1])
            row.update(current_work_id=item['id'], current_source_id=source.get('id'),
                       current_field_id=field, current_receipt=reference)
            official_language = receipt_records(directory[doi]['receipt']).get(doi, {}) if doi.startswith('10.1103/') else {}
            policy_language = (policy.get('status')=='reviewed_indexed_official_policy_with_native_publisher_identity'
                               and official_language.get('family')=='aps'
                               and any(n.startswith('Physical Review') for n in official_language.get('journal_titles', [])))
            row.update(api_language=item.get('language'), language_basis='publisher_policy_pending_identity_match' if policy_language else 'current_metadata',
                       language_evidence=str(policy_path.relative_to(ROOT)) if policy_language else reference)
            if key(plain(item.get('title') or '')) != key(plain(directory[doi]['title'])):
                status = 'current_title_requires_review'
            elif directory[doi]['source_id'] and source.get('id') != directory[doi]['source_id']:
                status = 'current_journal_requires_review'
            elif (cohort, source.get('id')) not in registry:
                status = 'current_field_journal_outside_fixed_registry'
            elif item.get('language') != 'en' and not policy_language:
                status = 'current_language_requires_review'
            elif item.get('is_retracted') or item.get('is_paratext'):
                status = 'current_integrity_or_paratext_requires_review'
            else:
                official = receipt_records(directory[doi]['receipt']).get(doi)
                if not official or source.get('id') not in journals:
                    status = 'publisher_journal_mapping_required'
                else:
                    primary = {'family': official['family'], 'title': official['title'],
                               'kind': official['publisher_type'], 'online_date': official['publisher_date'],
                               'date_precision': 'day', 'receipt': '../../'+directory[doi]['receipt'],
                               'url': official['catalogue_url'], 'journal_titles': official.get('journal_titles', []),
                               'aps_journal': official.get('aps_journal', ''), 'source_id': directory[doi]['source_id']}
                    matched = match({'work_id': item['id'], 'title': item['title'], 'source_id': source['id'], 'journal_name': source['display_name']}, primary, journals[source['id']])
                    status = ('current_in_scope_append_candidate_requires_primary_requalification'
                              if matched['identity'] == 'matched' else 'publisher_journal_mapping_requires_review')
                    row['publisher_family'] = official['family']
                    if matched['identity']=='matched' and policy_language:row['language_basis']='publisher_policy_and_exact_native_journal_identity'
        row['current_assessment'] = status
        rows.append(row)
        counts[status] += 1
    atomic_json(folder / 'full_directory_assessment.json', rows)
    result = {'fixed_targets': len(scope), 'processed_dois': len(queried),
              'appended_by_this_audit': 0, **counts}
    atomic_json(folder / 'full_directory_assessment_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
