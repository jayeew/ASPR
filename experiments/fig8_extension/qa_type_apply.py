"""Apply a reviewed literal nonresearch alias cohort from existing source evidence."""
from __future__ import annotations

import fcntl
import json
import shutil
from collections import Counter
from contextlib import nullcontext

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .fig8b_qualification_decide import literal_nonresearch_alias
from .publisher_delta_apply import stamp, updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(folder_name: str = 'qa_type_normalization', lock_held: bool = False) -> dict:
    folder = ROOT / 'supplement' / folder_name
    review = json.loads((folder / 'review.json').read_text())
    selected = {r['work_id']: r for r in review['papers']}
    q = ROOT / 'fields/qualification'
    counts: Counter = Counter()
    audit = []
    context = nullcontext() if lock_held else (ROOT / 'supplement/checkpoint.lock').open('a')
    with context as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
                  for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard in sorted({r['shard'] for r in selected.values()}):
            path = q / 'decisions' / shard
            before_sync = 'ownership-queue-1:' + stamp(ROOT / 'fields/papers' / shard) + ':' + stamp(path)
            eligible_changed = False
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
                for item in primary:
                    if literal_nonresearch_alias(item):
                        corrected = {**item, 'class': 'nonprimary'}
                        db.execute('UPDATE current_evidence_index SET payload=? WHERE work_id=? AND evidence_key=?',
                                   (json.dumps(corrected), row['work_id'], item['id']))
                        with (folder / 'derived_class_changes.jsonl').open('a') as stream:
                            stream.write(json.dumps({'before': item, 'after': corrected,
                                'reason': 'Reviewed literal nonresearch alias; original source evidence unchanged',
                                'rule': str((folder / 'review.json').relative_to(ROOT))})+'\n')
                new = updated_decision(row, primary, secondary, annual)
                eligible_changed |= (new != row and 'included_verified_research' in {row['eligibility_status'], new['eligibility_status']})
                audit.append({'before': dict(row), 'after': new,
                              'rule': str((folder / 'review.json').relative_to(ROOT))})
                counts[row['eligibility_status'] + ' -> ' + new['eligibility_status']] += 1
                row.update(new)
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",
                           ('resolved' if row['eligibility_status'].startswith(('included', 'excluded')) else 'pending', row['reason'], row['doi']))
                if row['eligibility_status'] == 'included_verified_research':
                    task(db, 'metadata', row['work_id'], 'pmc_or_publisher', {'doi': row['doi'], 'reason': 'Qualification reassessed with all existing evidence; reuse cached authors and abstract'}, row['work_id'])
            write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            from .metadata_task_sync import advance_nonqualified_only_sync
            after_sync = 'ownership-queue-1:' + stamp(ROOT / 'fields/papers' / shard) + ':' + stamp(path)
            advance_nonqualified_only_sync(db, shard, before_sync, after_sync, eligible_changed)
        db.commit()
        db.close()
        shutil.copy2(__file__.replace('qa_type_apply.py', 'fig8b_qualification_decide.py'), q / 'decide.py')
        with (folder / 'application.jsonl').open('a') as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        result = {'assessed': len(audit), 'transitions': dict(counts), 'network_requests': 0}
        atomic_json(folder / 'status.json', result)
    from .derived_refresh import refresh
    refresh()
    return result


def reconcile_current_aliases(lock_held: bool = False) -> dict:
    """Bring old-worker evidence into line before final offline recomputation."""
    db = connect()
    selected = {}
    query = """SELECT p.work_id,p.doi,p.shard,e.payload FROM current_evidence_index e
        JOIN papers p USING(work_id) WHERE e.kind='primary'
        AND json_extract(e.payload,'$.class')='unresolved_type'
        AND json_extract(e.payload,'$.kind') IN ('EditorialNotes','Invited Commentary','Q&A','Q & A','PrimeView','abstract','Abstract','ASO Visual Abstract','Aso Visual Abstract',
            'introduction','discussion','in-brief','In Brief','reviewer-report','editor-report','community-comment','ReviewPaper','Focus Review','opinion','Opinion','Where I Work','Career Q&A','News Q&A','Product News','Tools of the Trade','ASO Author Reflections')"""
    for work, doi, shard, payload in db.execute(query):
        record = json.loads(payload)
        if literal_nonresearch_alias(record):
            row = selected.setdefault(work, {'work_id': work, 'doi': doi, 'shard': shard, 'evidence': []})
            row['evidence'].append(record)
    db.close()
    if not selected:
        return {'assessed': 0, 'network_requests': 0}
    folder = ROOT / 'supplement/literal_alias_reconciliation'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'review.json', {'papers': list(selected.values()),
                'rules': ['supplement/qa_type_normalization/review.json',
                          'supplement/springer_nonresearch_aliases/review.json',
                          'supplement/primeview_type_policy/review.json',
                          'supplement/native_abstract_polymer_types/review.json',
                          'supplement/aso_visual_abstract/review.json',
                          'supplement/native_nonresearch_formats/review.json',
                          'supplement/native_introduction_review/validation.json',
                          'supplement/publisher_opinion_policy/review.json',
                          'supplement/nature_where_i_work/review.json',
                          'supplement/nature_literal_formats/review.json',
                          'supplement/nature_tools_trade_policy/review.json',
                          'supplement/laban_in_brief/review.json',
                          'supplement/aso_author_reflections/review.json'],
                'scope': 'All statuses, including previously included records; preserve conflicts',
                'network_requests': 0})
    return run(folder.name, lock_held=lock_held)


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
