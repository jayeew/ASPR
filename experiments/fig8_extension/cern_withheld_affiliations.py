"""Record public affiliation withholding without inventing institute or worksite."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

RULE = 'supplement/official_entities/cern_withheld_affiliation_policy.json'


def run() -> dict:
    policy = json.loads((ROOT / RULE).read_text())
    names = {' '.join(s.casefold().split()) for s in policy['exact_signatures']}
    stats = Counter()
    report = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        selected = defaultdict(set)
        task_ids = set()
        for name in names:
            for tid, wid, shard in db.execute(
                "SELECT t.task_id,p.work_id,p.shard FROM tasks t JOIN task_papers l USING(task_id) "
                "JOIN papers p ON p.work_id=l.work_id WHERE t.kind='institution' AND t.entity_key=? "
                "AND p.status='included_verified_research'", (name,)):
                task_ids.add(tid)
                selected[shard].add(wid)
        for shard, ids in selected.items():
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            changed = False
            for paper in rows:
                if paper['work_id'] not in ids:
                    continue
                before = paper['authors_json']
                authors = json.loads(before)
                slots = 0
                for author in authors:
                    if author.get('nature_v1_slots') is not None or not author.get('publisher_affiliation_source'):
                        continue
                    for slot in author.get('publisher_affiliation_slots') or []:
                        if ' '.join(slot.get('text', '').casefold().split()) not in names:
                            continue
                        slots += 1
                        slot['public_disclosure_limitation'] = {
                            'status': 'institution_name_withheld_in_public_byline', 'policy_receipt': RULE,
                            'signature_receipt': author['publisher_affiliation_source'],
                            'worksite_country_not_identified_by_this_signature': True,
                            'final_task_closure': False}
                if slots:
                    report.append({'work_id': paper['work_id'], 'doi': paper['doi'], 'slots': slots})
                revised = json.dumps(authors, ensure_ascii=False)
                if json.loads(before) == authors:
                    continue
                # This operation only adds the reviewed disclosure annotation.
                restored = json.loads(revised)
                for old, new in zip(json.loads(before), restored, strict=True):
                    for a, b in zip(old.get('publisher_affiliation_slots') or [], new.get('publisher_affiliation_slots') or [], strict=True):
                        if 'public_disclosure_limitation' not in a:
                            b.pop('public_disclosure_limitation', None)
                assert restored == json.loads(before)
                paper['authors_json'] = revised
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'authors_json', before, revised, RULE,
                            'Exact published cooperation signature: retain documented public withholding, no allocation changed'))
                stats['papers_annotated'] += 1
                stats['slots_annotated'] += slots
                changed = True
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
            db.commit()
        for tid in task_ids:
            source = 'official_public_affiliation_disclosure_policy'
            if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (tid, source, RULE)).fetchone():
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, source, 'public_identity_withheld_unresolved', RULE,
                            json.dumps({'final_task_closure': False, 'allocations_unchanged': True})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
        db.commit()
        db.close()
    result = {**stats, 'network_requests': 0, 'allocations_unchanged': True, 'final_task_closure': False}
    atomic_json(ROOT / 'supplement/official_entities/cern_withheld_affiliation_apply.json',
                {'result': result, 'affected_papers': report})
    if stats['papers_annotated']:
        refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
