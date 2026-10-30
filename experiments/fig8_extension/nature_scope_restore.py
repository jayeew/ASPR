"""Preserve reviewed Nature scope when only a title meta-analysis heuristic reopened it."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .publisher_delta_apply import evidence_index, updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    reviewed = {r['work_id']: r for r in map(json.loads, (ROOT / 'nature/papers.jsonl').open())}
    stats = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(ROOT)
        q = ROOT / 'fields/qualification'
        evidence_index(db, q)
        selected = defaultdict(dict)
        for wid, doi, title, shard in db.execute("SELECT work_id,doi,title,shard FROM papers WHERE status='pending_special_subtype_review'"):
            old = reviewed.get(wid)
            if old and old['doi'] == doi and old['title'] == title:
                selected[shard][wid] = doi
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard, targets in selected.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            changed = False
            for row in rows:
                wid = row['work_id']
                if wid not in targets:
                    continue
                primary = []; secondary = []
                for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (wid,)):
                    value = json.loads(payload)
                    if kind == 'primary':
                        primary.append(value)
                    else:
                        secondary = value
                new = updated_decision(row, primary, secondary, annual)
                if new == row:
                    stats['unchanged_pending'] += 1
                    continue
                stats[row['eligibility_status'] + ' -> ' + new['eligibility_status']] += 1
                row.update(new); changed = True
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], wid))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",
                           ('resolved' if row['eligibility_status'].startswith(('included','excluded')) else 'pending', row['reason'], targets[wid]))
                if row['eligibility_status'] == 'included_verified_research':
                    task(db, 'metadata', wid, 'cache_only', {'doi': targets[wid], 'reason': 'Reuse completed Nature metadata; prior reviewed scope retained'}, wid)
            if changed:
                write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            db.commit()
        db.close()
    result = {'status':'applied_derived_refresh_pending', **stats, 'network_requests':0,
              'scope_evidence':'nature/papers.jsonl', 'rule':'Prior matched nature_v1 Article with title-only meta-analysis flag; actual primary/secondary conflicts remain holds'}
    atomic_json(ROOT / 'supplement/nature_scope_restore.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
