"""Journal-level candidate accounting; nonempty abstracts do not imply validated text."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from .supplement_queue import ROOT

PRIORITY = ['Science', 'Cell', 'Proceedings of the National Academy of Sciences',
            'Science Advances', 'Nature', 'Nature Communications', 'Physical Review Letters',
            'Advanced Materials', 'Advanced Functional Materials', 'ACS Nano', 'Nano Letters',
            'Molecular Cell', 'The EMBO Journal', 'eLife', 'PLoS Biology',
            'Nature Materials', 'Nature Physics', 'Nature Genetics', 'Nature Biotechnology']


def abstract_screen(value: str | None, doi: str = "", reviewed: dict | None = None,
                    provenance_review: dict | None = None) -> str:
    """Only flags obvious quality risks; a pass is not a semantic validation."""
    value = (value or '').strip()
    if not value:
        return 'missing'
    digest = hashlib.sha256(' '.join(value.split()).encode()).hexdigest()
    if reviewed and reviewed.get(doi) == digest:
        return 'source_text_confirmed'
    if provenance_review and provenance_review.get(doi) == digest:
        return 'quality_review'
    if len(value.split()) < 40 or value.startswith(('or when ', 'and when ', '...')):
        return 'quality_review'
    return 'screen_pass_unvalidated'


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_comparison_entries(out: Path, rows: list[dict]) -> None:
    """Derive the comparison export from this run's priority counts, never old files."""
    entries = []
    for row in rows:
        total, pending = int(row['candidate_n']), int(row['pending_n'])
        entries.append({**row,
            'qualification_resolved_rate': (total - pending) / total if total else None,
            'next_use': 'verified_subset_preview_only' if pending else 'metadata_missingness_review_required',
            'formal_ready': row['formal_comparison_status'] == 'ready'})
    write_csv(out / 'comparison_entry_assessment.csv', entries)
    (out / 'comparison_entry_provenance.json').write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source': 'priority_journals.csv',
        'source_sha256': hashlib.sha256((out / 'priority_journals.csv').read_bytes()).hexdigest(),
        'rows': len(entries), 'qualification_and_coverage_copied_from_current_source': True,
        'preview_label_is_not_formal_comparison_approval': True}, indent=2))


def run(root: Path = ROOT, lock_held: bool = False) -> dict:
    out = root / 'analysis/journal_readiness'
    out.mkdir(parents=True, exist_ok=True)
    reviewed = {}
    provenance_review = {}
    availability_path = root / 'supplement/pbj_native_formats/abstract_availability.json'
    if availability_path.exists():
        for item in json.loads(availability_path.read_text())['records']:
            if (item['status'] == 'existing_text_without_native_abstract_review'
                    and item.get('current_abstract_sha256')):
                provenance_review[item['doi']] = item['current_abstract_sha256']
    review_path = root / 'supplement/pnas_abstract_review/current.json'
    if review_path.exists():
        for item in json.loads(review_path.read_text())['papers']:
            if item['status'] in {'native_text_confirmed', 'repaired_notice_as_abstract'} and (root / item['receipt']).is_file():
                reviewed[item['doi']] = item['abstract_sha256']
    accepted_abstract = {'screen_pass_unvalidated', 'source_text_confirmed'}
    counts = defaultdict(lambda: defaultdict(int))
    names = {}
    unresolved = []
    abstracts_by_source = defaultdict(lambda: defaultdict(int))
    priority_years = defaultdict(lambda: defaultdict(int))
    priority_dois = {}
    with (root / 'supplement/checkpoint.lock').open('a') as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_SH)
        for path in sorted((root / 'fields/papers').glob('*.parquet')):
            decisions = {r['work_id']: r for r in pq.read_table(root / 'fields/qualification/decisions' / path.name).to_pylist()}
            for batch in pq.ParquetFile(path).iter_batches(columns=['work_id', 'doi', 'source_id', 'journal_name', 'field', 'abstract', 'abstract_source'], batch_size=2000):
                for p in batch.to_pylist():
                    d = decisions[p['work_id']]
                    name = p['journal_name']
                    names[p['source_id']] = name
                    if name in PRIORITY:
                        y = priority_years[(p['source_id'], p['field'], d['online_year'] or d['year_openalex'], 'verified_online_year' if d['online_year'] else 'candidate_year_only')]
                        y['candidate_n'] += 1
                        y['verified_n'] += d['eligibility_status'] == 'included_verified_research'
                        y['pending_n'] += d['eligibility_status'].startswith('pending')
                        y['excluded_n'] += d['eligibility_status'].startswith('excluded')
                        if d['eligibility_status'] == 'included_verified_research':
                            y['abstract_nonempty_n'] += bool((p['abstract'] or '').strip())
                            abstracts_by_source[(p['source_id'], p['abstract_source'] or 'unspecified')]['verified_n'] += 1
                            abstracts_by_source[(p['source_id'], p['abstract_source'] or 'unspecified')]['nonempty_n'] += bool((p['abstract'] or '').strip())
                    for field in [p['field'], 'all_three_fields']:
                        c = counts[(p['source_id'], field)]
                        c['candidate_n'] += 1
                        status = d['eligibility_status']
                        if status.startswith('pending'):
                            c['pending_n'] += 1
                        elif status.startswith('excluded'):
                            c['excluded_n'] += 1
                        elif status == 'included_verified_research':
                            c['verified_n'] += 1
                            quality = abstract_screen(p['abstract'], p['doi'], reviewed, provenance_review)
                            c['abstract_nonempty_n'] += quality != 'missing'
                            c['abstract_screen_pass_n'] += quality in accepted_abstract
                            c['abstract_source_confirmed_n'] += quality == 'source_text_confirmed'
                            c['abstract_quality_review_n'] += quality == 'quality_review'
                            c['brand_unknown_n'] += d['nature_portfolio_at_verified_online_year'] not in ['True', 'False']
                        else:
                            raise ValueError(f'Unexpected eligibility: {status}')
                    if name in PRIORITY and (d['eligibility_status'].startswith('pending') or
                            (d['eligibility_status'] == 'included_verified_research' and (abstract_screen(p['abstract'], p['doi'], reviewed, provenance_review) not in accepted_abstract or d['nature_portfolio_at_verified_online_year'] not in ['True', 'False']))):
                        priority_dois[p['doi']] = PRIORITY.index(name)
                        unresolved.append({'work_id': p['work_id'], 'doi': p['doi'], 'journal': name,
                            'field': p['field'], 'qualification_status': d['eligibility_status'],
                            'qualification_reason': d['reason'], 'abstract_screen_status': abstract_screen(p['abstract'], p['doi'], reviewed, provenance_review),
                            'annual_brand': d['nature_portfolio_at_verified_online_year'],
                            'pmc_ids': d['secondary_pmcids'], 'evidence_ids': d['evidence_ids'],
                            'decision_receipt': str((root / 'fields/qualification/decisions' / path.name).relative_to(root)) + '#' + p['work_id'],
                            'closure': 'unresolved_retained_no_unbounded_requests'})
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_UN)
    rows = []
    for (source, field), c in counts.items():
        n = c['verified_n']
        assert c['candidate_n'] == n + c['pending_n'] + c['excluded_n']
        rows.append({'source_id': source, 'journal': names[source], 'field': field,
            **{k: c[k] for k in ['candidate_n', 'verified_n', 'excluded_n', 'pending_n', 'abstract_nonempty_n', 'abstract_screen_pass_n', 'abstract_source_confirmed_n', 'abstract_quality_review_n', 'brand_unknown_n']},
            'abstract_nonempty_rate': c['abstract_nonempty_n'] / n if n else None,
            'abstract_screen_pass_rate': c['abstract_screen_pass_n'] / n if n else None,
            'abstract_usable_validated_n': None,
            'formal_comparison_status': 'blocked_pending_eligibility' if c['pending_n'] else 'blocked_missingness_and_metric_review',
            'formal_blockers': '|'.join(['common_historical_graph_and_uniform_claims_missing', 'abstract_semantic_quality_not_certified'] +
                (['pending_eligibility'] if c['pending_n'] else []) + (['annual_brand_unknown'] if c['brand_unknown_n'] else []) +
                (['abstract_missing_or_flagged'] if c['abstract_screen_pass_n'] < n else []) + (['fewer_than_30_verified_papers'] if n < 30 else [])),
            'point_display_minimum_met_metadata_only': n >= 30,
            'priority_rank': PRIORITY.index(names[source]) if names[source] in PRIORITY else 999})
    rows.sort(key=lambda r: (r['priority_rank'], r['journal'], r['field']))
    write_csv(out / 'all_journals.csv', rows)
    selected = [r for r in rows if r['priority_rank'] < 999 and r['field'] == 'all_three_fields']
    write_csv(out / 'priority_journals.csv', selected)
    refresh_comparison_entries(out, selected)
    db = sqlite3.connect(f'file:{root / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    for item in unresolved:
        latest = db.execute("SELECT status,reason,receipt FROM tasks WHERE kind='qualification' AND entity_key=?", (item['doi'],)).fetchone()
        item['source_task_status'] = latest[0] if latest else 'not_recorded_in_task_ledger'
        item['source_task_reason'] = latest[1] if latest else ''
        item['source_task_receipt'] = latest[2] if latest else ''
        if item['qualification_status'] == 'included_verified_research':
            item['remaining_action'] = 'abstract_quality_or_dated_brand_evidence_review'
        elif not json.loads(item['pmc_ids'] or '[]'):
            item['remaining_action'] = 'primary_evidence_needed_no_pmc_identifier'
        elif latest and latest[0] == 'evidence_ready':
            item['remaining_action'] = 'source_obtained_but_combined_qualification_unresolved'
        elif latest and latest[0] in ['http_error', 'access_restricted', 'retry_exhausted', 'metadata_unavailable']:
            item['remaining_action'] = 'attempted_source_unavailable_or_unusable_other_paths_not_exhausted'
        else:
            item['remaining_action'] = 'not_completed_within_bounded_run'
    db.close()
    write_csv(out / 'priority_unresolved.csv', unresolved)
    write_csv(out / 'priority_year_coverage.csv', [
        {'source_id': source, 'journal': names[source], 'field': field, 'year': year, 'year_basis': basis,
         **{k: c[k] for k in ['candidate_n', 'verified_n', 'pending_n', 'excluded_n', 'abstract_nonempty_n']}}
        for (source, field, year, basis), c in sorted(priority_years.items(), key=lambda item: str(item[0]))])
    write_csv(out / 'priority_abstract_sources.csv', [
        {'source_id': source, 'journal': names[source], 'abstract_source': origin,
         'verified_n': c['verified_n'], 'nonempty_n': c['nonempty_n'], 'usable_validation': 'not_certified'}
        for (source, origin), c in sorted(abstracts_by_source.items())])
    (root / 'supplement/priority_dois.json').write_text(json.dumps(priority_dois))
    summary = {'at': datetime.now(timezone.utc).isoformat(), 'priority_journals': len(selected),
               'priority_gap_dois': len(priority_dois), 'candidate_n': sum(r['candidate_n'] for r in rows if r['field'] == 'all_three_fields'),
               'notes': 'Abstract coverage denominator is verified research. Screen pass is NOT validated usability. Pending candidates prevent unassessed formal comparisons.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-held', action='store_true', help='Internal: caller holds the shared dataset writer lock')
    args = parser.parse_args()
    run(lock_held=args.lock_held)
