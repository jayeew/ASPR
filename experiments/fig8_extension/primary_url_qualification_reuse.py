"""Reuse current qualified decisions without claiming unvisited URLs were checked."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

import pyarrow.parquet as pq

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cached_primary_url_leads'
    db = sqlite3.connect(folder / 'leads.sqlite', timeout=60)
    links = defaultdict(dict)
    for url, wid, doi in db.execute(
        "SELECT l.url,l.work_id,t.doi FROM leads l JOIN targets t USING(work_id) "
        "JOIN url_dispatch d USING(url) WHERE d.status='source_path_review_pending'"
    ):
        links[url][wid] = doi
    wanted = {wid for items in links.values() for wid in items}
    decisions = {}
    columns = ['work_id', 'doi', 'online_year', 'eligibility_status', 'evidence_ids']
    for path in sorted((ROOT / 'fields/qualification/decisions').glob('*.parquet')):
        rows = pq.read_table(path, columns=columns).to_pylist()
        for row in rows:
            included = row['eligibility_status'] == 'included_verified_research'
            excluded = row['eligibility_status'].startswith('excluded_')
            if (row['work_id'] not in wanted or not (included or excluded)
                    or (included and row['online_year'] not in (2023, 2024, 2025))):
                continue
            ids = json.loads(row['evidence_ids'] or '[]')
            if not row['doi'] or not ids:
                continue
            decisions[row['work_id']] = dict(row, evidence_ids=ids,
                decision_receipt=str(path.relative_to(ROOT)))
    changed = 0
    now = datetime.now(timezone.utc).isoformat()
    for url, items in links.items():
        if not all(wid in decisions and decisions[wid]['doi'] == doi
                   for wid, doi in items.items()):
            continue
        old = db.execute('SELECT status,reason,evidence FROM url_dispatch WHERE url=?',
                         (url,)).fetchone()
        if old[0] != 'source_path_review_pending':
            continue
        all_excluded = all(decisions[wid]['eligibility_status'].startswith('excluded_')
                           for wid in items)
        evidence = {'decisions': [decisions[wid] for wid in sorted(items)],
                    'content_fetched': False, 'metadata_applicability_pending': not all_excluded,
                    'qualification_changed': False, 'final_task_closure': False}
        new = ('qualification_excluded_no_main_sample_fetch_needed' if all_excluded
               else 'qualification_satisfied_metadata_path_review_pending',
               'All associated work IDs and DOIs match source-backed current terminal decisions; '
               + ('no main-sample metadata acquisition needed; extension scope remains separate'
                  if all_excluded else 'retain metadata review for included papers'),
               json.dumps(evidence))
        db.execute('INSERT INTO url_dispatch_changes VALUES(?,?,?,?,?)',
                   (url, json.dumps(old), json.dumps(new), 'current_qualification_reuse', now))
        db.execute('UPDATE url_dispatch SET status=?,reason=?,evidence=? WHERE url=?',
                   (*new, url))
        changed += 1
    db.commit()
    summary_path = folder / 'dispatch_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['url_status_counts'] = dict(db.execute(
        'SELECT status,count(*) FROM url_dispatch GROUP BY status'))
    summary['top_unreviewed_hosts'] = db.execute(
        'SELECT host,count(DISTINCT work_id) n FROM leads JOIN url_dispatch USING(url) '
        "WHERE status='source_path_review_pending' GROUP BY host ORDER BY n DESC LIMIT 30"
    ).fetchall()
    summary['updated_at'] = now
    assert sum(summary['url_status_counts'].values()) == summary['distinct_urls']
    atomic_json(summary_path, summary)
    result = {'status': 'qualification_reused_metadata_review_pending',
              'urls_changed_this_run': changed, 'network_requests': 0,
              'qualification_changes': 0, 'final_task_closure': False,
              'changed_at': now, 'ledger': 'leads.sqlite:url_dispatch_changes'}
    atomic_json(folder / 'qualification_reuse.json', result)
    db.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
