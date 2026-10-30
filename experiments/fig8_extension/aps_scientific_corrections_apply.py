"""Reassess the fixed APS scientific-correction cohort using reviewed content."""
from __future__ import annotations

import fcntl
import json
import shutil
from collections import Counter

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .publisher_delta_apply import updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/aps_scientific_corrections'
    review = json.loads((folder / 'review.json').read_text())
    selected = {r['work_id']: r for r in review['papers']}
    q = ROOT / 'fields/qualification'
    counts: Counter = Counter()
    audit = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
                  for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard in sorted({r['shard'] for r in selected.values()}):
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                if row['work_id'] not in selected:
                    continue
                primary, secondary = [], []
                for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (row['work_id'],)):
                    if kind == 'primary':
                        primary.append(json.loads(payload))
                    else:
                        secondary = json.loads(payload)
                new = updated_decision(row, primary, secondary, annual)
                audit.append({'before': dict(row), 'after': new,
                              'rule': 'supplement/aps_scientific_corrections/review.json'})
                counts[row['eligibility_status'] + ' -> ' + new['eligibility_status']] += 1
                row.update(new)
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",
                           ('resolved' if row['eligibility_status'].startswith(('included', 'excluded')) else 'pending', row['reason'], row['doi']))
                if row['eligibility_status'] == 'included_verified_research':
                    task(db, 'metadata', row['work_id'], 'pmc_or_publisher', {'doi': row['doi'], 'reason': 'Newly qualified by reviewed APS article content; reuse cached authors and abstract'}, row['work_id'])
            write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
        db.commit()
        db.close()
        shutil.copy2(__file__.replace('aps_scientific_corrections_apply.py', 'fig8b_qualification_decide.py'), q / 'decide.py')
        with (folder / 'application.jsonl').open('a') as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        result = {'assessed': len(audit), 'transitions': dict(counts), 'network_requests': 0}
        atomic_json(folder / 'status.json', result)
    from .derived_refresh import refresh
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
