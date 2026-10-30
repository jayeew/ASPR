"""Apply reviewed journal-specific PLOS types to the retained fixed review queue."""
from __future__ import annotations

import fcntl
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .fig8b_qualification_decide import plos_format_class
from .publisher_delta_apply import evidence_index, updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/plos_batch/type_policies'
    policy = 'supplement/plos_batch/type_policies/reviewed_rules.json'
    assert json.loads((ROOT / policy).read_text())['status'] == 'reviewed'
    # A new rule can also change the rationale of an already included record.
    scope = json.loads((folder.parent / 'scope.json').read_text())['papers']
    q = ROOT / 'fields/qualification'
    transitions = Counter()
    audit = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        evidence_index(db, q)
        grouped = defaultdict(dict)
        for item in scope:
            wid = item['work_id']
            row = db.execute('SELECT shard FROM papers WHERE work_id=?', (wid,)).fetchone()
            primary, secondary = [], []
            for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (wid,)):
                if kind == 'primary':
                    primary.append(json.loads(payload))
                else:
                    secondary = json.loads(payload)
            if any(plos_format_class(e) or e.get('family') == 'reviewed_study_design' for e in primary):
                grouped[row[0]][wid] = (primary, secondary)
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
                  for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard, selected in grouped.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            changed = False
            for old in rows:
                if old['work_id'] not in selected:
                    continue
                primary, secondary = selected[old['work_id']]
                new = updated_decision(old, primary, secondary, annual)
                if old == new:
                    continue
                applied_rule = next((e.get('review_rule_receipt', 'supplement/plos_batch/study_design_review.json')
                                     for e in primary if e.get('family') == 'reviewed_study_design'), policy)
                audit.append({'work_id': old['work_id'], 'before': dict(old), 'after': new, 'rule': applied_rule})
                transitions[old['eligibility_status'] + ' -> ' + new['eligibility_status']] += 1
                old.update(new)
                changed = True
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (old['eligibility_status'], old['work_id']))
                resolved = old['eligibility_status'].startswith(('included', 'excluded'))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",
                           ('resolved' if resolved else 'pending', old['reason'], old['doi']))
                if old['eligibility_status'] == 'included_verified_research':
                    task(db, 'metadata', old['work_id'], 'pmc_or_publisher',
                         {'doi': old['doi'], 'source_id': old['source_id'],
                          'reason': 'Newly qualified reviewed PLOS format; assess current metadata gaps'}, old['work_id'])
            if changed:
                write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            db.commit()
        db.close()
        shutil.copy2(Path(__file__).with_name('fig8b_qualification_decide.py'), q / 'decide.py')
        path = folder / 'application_audit.json'
        previous = json.loads(path.read_text()) if path.exists() else []
        atomic_json(path, previous + audit)
    result = {'reviewed_fixed_scope': len(scope), 'changed_decisions': len(audit),
              'cumulative_change_events': len(previous) + len(audit),
              'transitions': dict(transitions), 'network_requests': 0, 'derived_refresh_required': True}
    atomic_json(folder / 'application_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    from .derived_refresh import refresh
    run()
    refresh()
