"""Refresh derived tables by audited deltas; never recompute the full paper corpus."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .fig8b_qualification_recompute import metrics
from .supplement_queue import connect, task
from .three_journal_apply import read_csv, replace_scope, target_papers, write_csv
from .three_journal_round import AUDIT, OUT, Q, ROOT, SIDS, load, save


def group(d: dict) -> str:
    return 'verified_research' if d['eligibility_status'] == 'included_verified_research' else 'excluded' if d['eligibility_status'].startswith('excluded') else 'pending'


def run() -> None:
    if (AUDIT / 'derived_applied.json').exists():
        print('Derived delta already applied; no repeated increments.')
        return
    decisions, _, _, _ = load()
    papers = target_papers()
    delta = json.loads((AUDIT / 'decision_delta.json').read_text())
    by_id = {r['work_id']: r for r in delta}
    coverage_rows = read_csv(Q / 'coverage.csv')
    coverage = {(r['field'], r['population'], int(r['year'])): Counter({k: int(v) for k,v in r.items() if k not in ['field','population','year']}) for r in coverage_rows}
    publisher_rows = read_csv(Q / 'research_publisher_coverage.csv')
    publisher = {(r['field'], r['nature_portfolio_at_online_year']): Counter({k: int(v) for k,v in r.items() if k not in ['field','nature_portfolio_at_online_year']}) for r in publisher_rows}
    status_delta = Counter()
    for change in delta:
        p = papers[change['work_id']]
        values = metrics(p)
        for side, sign in [('before', -1), ('after', 1)]:
            d = change[side]
            status_delta[d['eligibility_status']] += sign
            year = d['online_year'] if group(d) == 'verified_research' else p['year_openalex']
            for y in [0,year]:
                c = coverage.setdefault((p['field'], group(d), y), Counter())
                for k,v in values.items():
                    c[k] += sign*v
            if group(d) == 'verified_research':
                c = publisher.setdefault((p['field'], d['nature_portfolio_at_verified_online_year']), Counter())
                for k,v in values.items():
                    c[k] += sign*v
    assert all(v >= 0 for c in coverage.values() for v in c.values())
    assert all(v >= 0 for c in publisher.values() for v in c.values())
    write_csv(Q/'coverage.csv', [{'field': f, 'population': p, 'year': y, **dict(c)} for (f,p,y),c in sorted(coverage.items())], list(coverage_rows[0]))
    write_csv(Q/'research_publisher_coverage.csv', [{'field': f, 'nature_portfolio_at_online_year': l, **dict(c)} for (f,l),c in sorted(publisher.items())], list(publisher_rows[0]))
    journals = defaultdict(Counter)
    roster = []
    for wid,d in decisions.items():
        if group(d) != 'verified_research':
            continue
        p = papers[wid]
        journals[(p['field'],p['source_id'],p['journal_name'])].update(metrics(p))
        roster.append({**{k:p[k] for k in ['work_id','doi','field','source_id','journal_name','title']},
                       **{k:d[k] for k in ['online_date','online_year','publisher_types','evidence_ids','nature_portfolio_at_verified_online_year','portfolio_evidence_ids']}})
    replace_scope(Q/'research_roster.csv', roster)
    replace_scope(Q/'research_journal_coverage.csv', [{'field': f,'source_id':s,'journal_name':n,**dict(c)} for (f,s,n),c in journals.items()])
    # Patch the existing unresolved export; no evidence lookup or author recomputation outside the delta.
    path = Q/'unresolved.csv.gz'
    temp = path.with_suffix('.gz.tmp')
    removed = updated = 0
    with gzip.open(path,'rt',encoding='utf-8') as source, gzip.open(temp,'wt',encoding='utf-8',newline='') as dest:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            change = by_id.get(row['work_id'])
            if change:
                if not change['after']['eligibility_status'].startswith('pending'):
                    removed += 1
                    continue
                row.update(change['after'])
                updated += 1
            writer.writerow(row)
        for change in delta:
            if not change['before']['eligibility_status'].startswith('pending') and change['after']['eligibility_status'].startswith('pending'):
                current = {**change['after'], **metrics(papers[change['work_id']])}
                writer.writerow(current)
    temp.replace(path)
    # Aggregate-only delta updates do not claim a fresh whole-corpus independent validation.
    result = json.loads((Q/'recomputed_results.json').read_text())
    for status, n in status_delta.items():
        result['statuses'][status] = result['statuses'].get(status,0) + n
    included_delta = status_delta['included_verified_research']
    pending_delta = sum(n for status,n in status_delta.items() if status.startswith('pending'))
    result.update(verified_research_count=result['verified_research_count']+included_delta,
                  unique_research_dois=result['unique_research_dois']+included_delta, unresolved_count=result['unresolved_count']+pending_delta,
                  latest_update='three_journal_incremental_delta', last_validation_scope=SIDS,
                  current_global_independent_recompute=False)
    for change in delta:
        p = papers[change['work_id']]
        for side, sign in [('before',-1),('after',1)]:
            d = change[side]
            if group(d) == 'verified_research':
                result['verified_author_cap_remaining'] += sign*int(p['author_list_may_be_truncated'])
                result['verified_online_year_shifts'] += sign*int(d['online_year'] != p['year_openalex'])
    save(Q/'recomputed_results.json', result)
    summary = json.loads((Q/'summary.json').read_text())
    for status,n in status_delta.items():
        summary['statuses'][status] = summary['statuses'].get(status,0)+n
    summary['evidence_rows'] += len(json.loads((AUDIT/'evidence_delta.json').read_text()))
    summary['latest_update'] = 'three_journal_delta; aggregate status refreshed; legacy evidence/type/date audits predate this delta'
    save(Q/'summary.json', summary)
    db = connect(ROOT)
    new_metadata = 0
    for change in delta:
        if group(change['before']) == 'verified_research' or group(change['after']) != 'verified_research':
            continue
        p = papers[change['work_id']]
        m = metrics(p)
        gaps = [name for name,k in [('abstract','abstract'),('institutions','every_author_institution'),('countries','every_author_country')] if not m[k]]
        if gaps or p['author_list_may_be_truncated']:
            task(db,'metadata',p['work_id'],'pmc_or_publisher',{'doi':p['doi'],'missing_fields':gaps,'qualification_receipt':str((AUDIT / 'decision_delta.json').relative_to(ROOT))},p['work_id'])
            new_metadata += 1
    db.commit()
    db.close()
    save(AUDIT/'derived_applied.json', {'at':datetime.now(timezone.utc).isoformat(),'method':'patch existing exports and subtract/add target contributions; no global qualification or collection',
                                     'unresolved_export_removed':removed,'unresolved_export_updated':updated,'new_metadata_gap_tasks':new_metadata,
                                     'global_included_from_delta':result['verified_research_count'],'global_pending_from_delta':result['unresolved_count']})
    print(json.dumps({'removed':removed,'updated':updated,'new_metadata_tasks':new_metadata}))


if __name__ == '__main__':
    run()
