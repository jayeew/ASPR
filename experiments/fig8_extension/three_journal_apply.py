"""Apply and audit only the three-journal evidence delta, preserving other rows."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_decide import decide, portfolio_label
from .fig8b_qualification_recompute import metrics
from .journal_readiness import PRIORITY, abstract_screen
from .three_journal_round import AUDIT, OUT, Q, ROOT, SIDS, load, save


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def replace_scope(path: Path, rows: list[dict]) -> None:
    previous = read_csv(path)
    kept = [r for r in previous if r.get('source_id') not in SIDS]
    write_csv(path, kept + rows, list(previous[0]))


def apply() -> None:
    if (AUDIT / 'applied.json').exists():
        print('Already applied; use verify only. No duplicate writes.')
        return
    delta = json.loads((AUDIT / 'decision_delta.json').read_text())
    evidence_delta = json.loads((AUDIT / 'evidence_delta.json').read_text())
    by_id = {r['work_id']: r for r in delta}
    annual = json.loads((AUDIT / 'annual_rules.json').read_text())
    before_counts = read_csv(OUT / 'priority_journals.csv')
    save(AUDIT / 'baseline_counts.json', [r for r in before_counts if r['source_id'] in SIDS])
    protected = []
    # Shared lock excludes existing collectors/writers; only decision shards containing deltas are rewritten.
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = sqlite3.connect(ROOT / 'supplement/tasks.sqlite', timeout=30)
        for path in sorted((Q / 'decisions').glob('*.parquet')):
            table = pq.read_table(path)
            rows = table.to_pylist()
            selected = [r for r in rows if r['work_id'] in by_id]
            if not selected:
                continue
            for old in selected:
                assert old == by_id[old['work_id']]['before'], 'Concurrent decision change'
            unchanged_before = hashlib.sha256(json.dumps([r for r in rows if r['work_id'] not in by_id], sort_keys=True).encode()).hexdigest()
            updated = [by_id[r['work_id']]['after'] if r['work_id'] in by_id else r for r in rows]
            temp = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(updated, schema=table.schema), temp)
            check = pq.read_table(temp).to_pylist()
            unchanged_after = hashlib.sha256(json.dumps([r for r in check if r['work_id'] not in by_id], sort_keys=True).encode()).hexdigest()
            assert unchanged_before == unchanged_after
            temp.replace(path)
            protected.append({'shard': path.name, 'unchanged_rows_equal': True, 'changed': len(selected)})
        # Gzip concatenated members are append-only and supported by the existing offline reader.
        with gzip.open(Q / 'decision_evidence.jsonl.gz', 'at') as stream, (Q / 'curated_evidence.jsonl').open('a') as curated:
            for e in evidence_delta:
                line = json.dumps(e, ensure_ascii=False) + '\n'
                stream.write(line)
                curated.write(line)
        registry = read_csv(ROOT / 'fields/journal_publisher_by_year.csv')
        for row in registry:
            rule = next((r for r in annual if r['source_id'] == row['source_id'] and str(r['year']) == row['year'] and r['label'] == 'False'), None)
            if rule:
                assert row['nature_portfolio_year'] != 'True', 'Annual label conflict'
                row.update(nature_portfolio_year='False', imprint=rule.get('publisher', 'National Academy of Sciences'),
                           status=rule['status'], comparison_group='non_nature_portfolio',
                           transfer_evidence_url=str((AUDIT / 'annual_rules.json').relative_to(ROOT)),
                           membership_scope=rule['scope'])
        write_csv(ROOT / 'fields/journal_publisher_by_year.csv', registry)
        for r in delta:
            for field, value in r['after'].items():
                if value != r['before'][field]:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (r['work_id'], 'qualification.' + field, json.dumps(r['before'][field]), json.dumps(value),
                                str((AUDIT / 'decision_delta.json').relative_to(ROOT)), 'Bounded cached journal identity / explicit deposited online date / dated annual imprint'))
            status = r['after']['eligibility_status']
            db.execute('UPDATE papers SET status=? WHERE work_id=?', (status, r['work_id']))
            db.execute("UPDATE tasks SET status=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE kind='qualification' AND entity_key=?",
                       ('resolved' if status.startswith(('included', 'excluded')) else 'pending', r['after']['reason'], r['after']['doi']))
        for path in sorted((AUDIT / 'requests').glob('*.json')):
            if not path.stem.isdigit():
                continue
            record = json.loads(path.read_text())
            for doi in record['dois']:
                task_id = hashlib.sha256(('qualification:' + doi).encode()).hexdigest()[:32]
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, record.get('source', 'crossref_public_deposit_bounded_three_journals'), 'evidence_ready' if record.get('http_status') == 200 else 'request_failed',
                            str(path.relative_to(ROOT)), json.dumps({'http_status': record.get('http_status'), 'response_success_not_qualification': True})))
                db.execute("UPDATE tasks SET attempt_n=attempt_n+1,receipt=?,updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                           (str(path.relative_to(ROOT)), task_id))
        db.commit()
        db.close()
        save(AUDIT / 'applied.json', {'at': datetime.now(timezone.utc).isoformat(), 'decision_rows_changed': len(delta),
                                    'evidence_appended': len(evidence_delta), 'shards': protected, 'paper_master_unchanged': True})
    print('Applied', len(delta), 'decision changes')


def target_papers() -> dict:
    result = {}
    for path in sorted((ROOT / 'fields/papers').glob('*.parquet')):
        for r in pq.read_table(path, filters=[('source_id', 'in', SIDS)]).to_pylist():
            result[r['work_id']] = r
    return result


def refresh() -> None:
    decisions, evidence, secondary, sources = load()
    papers = target_papers()
    names = {p['source_id']: p['journal_name'] for p in papers.values()}
    counts, years, abstracts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    unresolved = []
    annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    errors = []
    for wid, d in decisions.items():
        p = papers[wid]
        assert (p['doi'], p['title'], p['source_id']) == (d['doi'], d['title'], d['source_id'])
        rerun = decide(evidence[wid], secondary[wid], d['duplicate_of'])
        label, _ = portfolio_label(annual.get((d['source_id'], d['online_year']), ''), evidence[wid], d['online_year'])
        if (rerun['status'], rerun['online_year'], label) != (d['eligibility_status'], d['online_year'], d['nature_portfolio_at_verified_online_year']):
            errors.append(wid)
        quality = abstract_screen(p['abstract'])
        for field in [d['field'], 'all_three_fields']:
            c = counts[(d['source_id'], field)]
            c['candidate_n'] += 1
            status = d['eligibility_status']
            if status == 'included_verified_research':
                c.update(verified_n=1, abstract_nonempty_n=quality != 'missing', abstract_screen_pass_n=quality == 'screen_pass_unvalidated',
                         abstract_quality_review_n=quality == 'quality_review', brand_unknown_n=label not in ['True', 'False'])
            else:
                c['pending_n' if status.startswith('pending') else 'excluded_n'] += 1
        y = years[(d['source_id'], d['field'], d['online_year'] or d['year_openalex'], 'verified_online_year' if d['online_year'] else 'candidate_year_only')]
        y.update(candidate_n=1, verified_n=status == 'included_verified_research', pending_n=status.startswith('pending'), excluded_n=status.startswith('excluded'), abstract_nonempty_n=status == 'included_verified_research' and quality != 'missing')
        if status == 'included_verified_research':
            abstracts[(d['source_id'], p['abstract_source'] or 'unspecified')].update(verified_n=1, nonempty_n=quality != 'missing')
        if status.startswith('pending') or (status == 'included_verified_research' and (quality != 'screen_pass_unvalidated' or label not in ['True', 'False'])):
            unresolved.append({'work_id': wid, 'doi': d['doi'], 'journal': names[d['source_id']], 'field': d['field'],
                               'qualification_status': status, 'qualification_reason': d['reason'], 'abstract_screen_status': quality, 'annual_brand': label,
                               'pmc_ids': d['secondary_pmcids'], 'evidence_ids': d['evidence_ids'],
                               'decision_receipt': 'fields/qualification/decisions/' + Path(d['base_shard']).name + '#' + wid,
                               'closure': 'bounded_round_closed_unresolved_retained', 'source_task_status': '', 'source_task_reason': '', 'source_task_receipt': '',
                               'remaining_action': 'abstract_quality_or_dated_brand_evidence_review' if status.startswith('included') else 'primary_online_date_needed' if status == 'pending_primary_online_date' else 'qualified_primary_evidence_or_conflict_review_needed'})
    assert not errors, errors[:10]
    included = [d for d in decisions.values() if d['eligibility_status'] == 'included_verified_research']
    assert len({r['doi'] for r in included}) == len(included)
    rows = []
    for (sid, field), c in counts.items():
        n = c['verified_n']
        rows.append({'source_id': sid, 'journal': names[sid], 'field': field,
                     **{k: c[k] for k in ['candidate_n', 'verified_n', 'excluded_n', 'pending_n', 'abstract_nonempty_n', 'abstract_screen_pass_n', 'abstract_quality_review_n', 'brand_unknown_n']},
                     'abstract_nonempty_rate': c['abstract_nonempty_n']/n if n else '', 'abstract_screen_pass_rate': c['abstract_screen_pass_n']/n if n else '',
                     'abstract_usable_validated_n': '', 'formal_comparison_status': 'blocked_pending_eligibility',
                     'formal_blockers': '|'.join(['common_historical_graph_and_uniform_claims_missing', 'abstract_semantic_quality_not_certified', 'pending_eligibility'] + (['annual_brand_unknown'] if c['brand_unknown_n'] else []) + (['abstract_missing_or_flagged'] if c['abstract_screen_pass_n'] < n else []) + (['fewer_than_30_verified_papers'] if n < 30 else [])),
                     'point_display_minimum_met_metadata_only': n >= 30, 'priority_rank': PRIORITY.index(names[sid])})
    replace_scope(OUT / 'all_journals.csv', rows)
    replace_scope(OUT / 'priority_journals.csv', [r for r in rows if r['field'] == 'all_three_fields'])
    replace_scope(OUT / 'priority_year_coverage.csv', [{'source_id': s, 'journal': names[s], 'field': f, 'year': y, 'year_basis': b,
                  **{k: c[k] for k in ['candidate_n', 'verified_n', 'pending_n', 'excluded_n', 'abstract_nonempty_n']}} for (s,f,y,b), c in years.items()])
    replace_scope(OUT / 'priority_abstract_sources.csv', [{'source_id': s, 'journal': names[s], 'abstract_source': a, 'verified_n': c['verified_n'], 'nonempty_n': c['nonempty_n'], 'usable_validation': 'not_certified'} for (s,a),c in abstracts.items()])
    db = sqlite3.connect(ROOT / 'supplement/tasks.sqlite')
    for r in unresolved:
        old = db.execute("SELECT status,reason,receipt FROM tasks WHERE kind='qualification' AND entity_key=?", (r['doi'],)).fetchone()
        if old:
            r['source_task_status'], r['source_task_reason'], r['source_task_receipt'] = old
    db.close()
    old = read_csv(OUT / 'priority_unresolved.csv')
    write_csv(OUT / 'priority_unresolved.csv', [r for r in old if r['journal'] not in names.values()] + unresolved, list(old[0]))
    write_csv(OUT / 'three_journal_unresolved.csv', unresolved)
    save(AUDIT / 'verification.json', {'at': datetime.now(timezone.utc).isoformat(), 'scope_candidate_n': len(decisions),
                                     'scope_verified_n': len(included), 'unique_verified_dois': len(included), 'offline_decision_errors': errors,
                                     'counts_conserved': all(r['candidate_n'] == r['verified_n'] + r['pending_n'] + r['excluded_n'] for r in rows),
                                     'global_recompute_run': False, 'current_counts': [r for r in rows if r['field'] == 'all_three_fields']})
    print(json.dumps([r for r in rows if r['field'] == 'all_three_fields']))


if __name__ == '__main__':
    apply()
    refresh()
