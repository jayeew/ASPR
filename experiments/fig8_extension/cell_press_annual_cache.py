"""Reuse dated Cell Press title lists across all matching current journals."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader

from .derived_refresh import refresh
from .fig8b_qualification_build import key, write_decisions
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv, write_csv


def title_rows(text: str, year: int) -> list[tuple[str, str]]:
    pattern = (r'^(\d{4}-\d{3}[\dX])\s+(.+)$' if year == 2024 else
               r'^\d+\s*(.+?)\s+(\d{4}-\d{3}[\dX])\s+HYBRID\b')
    rows = []
    for line in text.splitlines():
        match = re.search(pattern, line.strip())
        if match:
            rows.append(match.groups() if year == 2024 else tuple(reversed(match.groups())))
    return rows


def run() -> None:
    folder = ROOT / 'supplement/cell_press_annual_cache'
    folder.mkdir(exist_ok=True)
    base = ROOT / 'supplement/cell_science_closure'
    rules = json.loads((base / 'annual_rules.json').read_text())
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    proposals = []
    unmatched = []
    # The 2024 Annex A continues onto PDF page 16 (three society titles).
    for year, filename, page in [(2024, '002.body.gz', 14),
                                 (2024, '002.body.gz', 15),
                                 (2025, '008.body.gz', 0)]:
        raw_path = base / 'requests' / filename
        raw = gzip.decompress(raw_path.read_bytes())
        document = PdfReader(io.BytesIO(raw))
        text = document.pages[page].extract_text()
        proof = next(r for r in rules if r['year'] == year)['proofs'][0]
        for issn, title in title_rows(text, year):
            found = False
            for sid, journal in journals.items():
                if issn not in json.loads(journal['issn'] or '[]'):
                    continue
                if key(title.replace('&', 'and')) != key(journal['journal_name'].replace('&', 'and')):
                    unmatched.append({'year': year, 'issn': issn, 'title': title,
                                      'source_id': sid, 'journal_name': journal['journal_name'],
                                      'status': 'title_difference_requires_review'})
                    continue
                found = True
                proposals.append({'source_id': sid, 'year': year, 'label': 'False',
                                  'journal': title, 'issn': issn, 'publisher': 'Elsevier / Cell Press',
                                  'source_url': proof['source_url'], 'raw_receipt': str(raw_path.relative_to(ROOT)),
                                  'raw_sha256': hashlib.sha256(raw).hexdigest(), 'pdf_page': page + 1,
                                  'scope': 'Dated title-list annual working rule; exact ISSN and title; reopen on transfer/conflict; no extrapolation to other years'})
            if not found:
                unmatched.append({'year': year, 'issn': issn, 'title': title,
                                  'status': 'no_exact_current_registry_title_and_issn'})
    annual = {(r['source_id'], int(r['year'])):r for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
    pairs = {(r['source_id'],r['year']) for r in proposals if annual.get((r['source_id'],r['year']),{}).get('nature_portfolio_year') not in {'True','False'}}
    atomic_json(folder/'application_scope.json', {'pairs':sorted(pairs), 'title_normalization':'Only & and and equivalence; matching ISSN remains mandatory', 'network_requests':0})
    apply_proposals(proposals, folder, unmatched, apply_pairs=pairs)



ANNOTATION_COLUMNS = ['nature_portfolio_at_candidate_year', 'annual_imprint',
                      'annual_publisher_evidence', 'historical_portfolio_status']


def replace_annotation_columns(table: pa.Table, papers: list[dict]) -> pa.Table:
    """Keep all unrelated Arrow buffers and field metadata unchanged."""
    for name in ANNOTATION_COLUMNS:
        index = table.schema.get_field_index(name)
        field = table.schema.field(index)
        table = table.set_column(index, field, pa.array([p[name] for p in papers], type=field.type))
    return table


def apply_proposals(proposals: list[dict], folder: Path, unmatched: list[dict],
                    rule_status: str = 'dated_cell_press_title_list_exact_identity',
                    reason: str = 'Dated Cell Press title list; exact journal title and ISSN',
                    route: str = 'dated_cell_press_list',
                    apply_pairs: set[tuple[str, int]] | None = None,
                    refresh_derived: bool = True) -> dict:
    """Apply scoped annual membership rules; retain conflicts and shared provenance."""
    if any(rule['year'] not in {2023, 2024, 2025} for rule in proposals):
        raise ValueError('Fig8 annual attribution is restricted to 2023–2025; boundary-year evidence cannot be applied')
    if any(rule.get('label') not in {'True', 'False'} for rule in proposals):
        raise ValueError('Annual rules require an explicit membership label')
    evidence = str((folder / 'rules.json').relative_to(ROOT))
    atomic_json(folder / 'rules.json', proposals)
    atomic_json(folder / 'unmatched.json', unmatched)
    by_pair = {(r['source_id'], r['year']): r for r in proposals
               if apply_pairs is None or (r['source_id'], r['year']) in apply_pairs}
    counts = Counter()
    conflicts = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        annual = read_csv(ROOT / 'fields/journal_publisher_by_year.csv')
        for row in annual:
            rule = by_pair.get((row['source_id'], int(row['year'])))
            if not rule:
                continue
            if row['nature_portfolio_year'] in {'True','False'} and row['nature_portfolio_year'] != rule['label']:
                conflicts.append({'source_id': row['source_id'], 'year': row['year'], 'before': row})
                by_pair.pop((row['source_id'], int(row['year'])))
                continue
            if row['nature_portfolio_year'] == rule['label']:
                continue
            before = dict(row)
            row.update(nature_portfolio_year=rule['label'], imprint=rule['publisher'],
                       status=rule_status, comparison_group='nature_portfolio_annual_confirmed' if rule['label']=='True' else 'non_nature_portfolio',
                       transfer_evidence_url=evidence, membership_scope=rule['scope'])
            with (folder / 'annual_changes.jsonl').open('a') as out:
                out.write(json.dumps({'before': before, 'after': row, 'receipt': evidence}) + '\n')
            counts['annual_rows_filled'] += 1
        write_csv(ROOT / 'fields/journal_publisher_by_year.csv', annual)
        sids = sorted({sid for sid, _ in by_pair})
        marks = ','.join('?' for _ in sids)
        shards = [row[0] for row in db.execute(
            f'SELECT DISTINCT shard FROM papers WHERE source_id IN ({marks}) ORDER BY shard', sids)]
        for shard in shards:
            path = ROOT / 'fields/papers' / shard
            from .metadata_task_sync import advance_annual_only_sync
            from .publisher_delta_apply import stamp
            qpath = ROOT / 'fields/qualification/decisions' / shard
            sync_before = 'ownership-queue-1:' + stamp(path) + ':' + stamp(qpath)
            projected = pq.read_table(path, columns=['work_id', 'source_id', 'year_openalex', *ANNOTATION_COLUMNS])
            papers = projected.to_pylist()
            qpath = ROOT / 'fields/qualification/decisions' / shard
            decisions = pq.read_table(qpath).to_pylist()
            dirty = False
            qdirty = False
            for row in decisions:
                rule = by_pair.get((row['source_id'], row['online_year']))
                if rule and row['nature_portfolio_at_verified_online_year'] not in ['True', 'False']:
                    row['nature_portfolio_at_verified_online_year'] = rule['label']
                    row['portfolio_evidence_ids'] = json.dumps([evidence + '#' + row['source_id'] + ':' + str(row['online_year'])])
                    qdirty = True
                    counts['verified_year_labels_filled'] += 1
                    counts['included_labels_filled'] += row['eligibility_status'] == 'included_verified_research'
                rule = by_pair.get((row['source_id'], row['year_openalex'])) if 'year_openalex' in row else None
                if rule and row['nature_portfolio_at_candidate_year'] not in ['True', 'False']:
                    row['nature_portfolio_at_candidate_year'] = rule['label']
                    qdirty = True
            for paper in papers:
                rule = by_pair.get((paper['source_id'], paper['year_openalex']))
                if not rule or paper['nature_portfolio_at_candidate_year'] in ['True', 'False']:
                    continue
                values = {'nature_portfolio_at_candidate_year': rule['label'], 'annual_imprint': rule['publisher'],
                          'annual_publisher_evidence': evidence, 'historical_portfolio_status': rule_status}
                for field, value in values.items():
                    if paper[field] == value:
                        continue
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (paper['work_id'], field, json.dumps(paper[field]), json.dumps(value), evidence,
                                reason + '; candidate-year annotation only'))
                    paper[field] = value
                dirty = True
                counts['candidate_papers_filled'] += 1
            if qdirty:
                write_decisions(qpath, decisions, ROOT / 'fields/qualification/decision_changes.jsonl.gz')
            if dirty:
                tmp = path.with_suffix('.parquet.tmp')
                table = replace_annotation_columns(pq.read_table(path), papers)
                pq.write_table(table, tmp, compression='zstd')
                tmp.replace(path)
            if dirty or qdirty:
                sync_after = 'ownership-queue-1:' + stamp(path) + ':' + stamp(qpath)
                if advance_annual_only_sync(db, shard, sync_before, sync_after, decisions):
                    counts['author_task_relink_shards_avoided'] += 1
            db.commit()
        for (sid, year), rule in by_pair.items():
            tid = task(db, 'portfolio', sid + ':' + str(year), route, rule, status='resolved')
            db.execute('UPDATE tasks SET receipt=?,reason=? WHERE task_id=?',
                       (evidence, reason, tid))
        db.commit()
        db.close()
    atomic_json(folder / 'conflicts.json', conflicts)
    result = {**counts, 'matched_journal_year_rules': len(proposals), 'applied_scope_pairs': len(by_pair), 'network_requests': 0,
              'conflicts': len(conflicts), 'status': 'applied_derived_refresh_pending'}
    atomic_json(folder / 'status.json', result)
    if refresh_derived:
        refresh()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
