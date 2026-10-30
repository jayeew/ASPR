"""Correct the reviewed KIP namesake conflict without discarding source identifiers."""
from __future__ import annotations

import fcntl
import json

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .fig8b_qualification_recompute import metrics
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

OLD = 'https://openalex.org/I85384741'
NEW = 'https://openalex.org/I223822909'
SIGNATURE = 'Kirchhoff-Institute for Physics, Heidelberg University, Im Neuenheimer Feld 227, 69120 Heidelberg, Germany'
RECEIPT = 'supplement/official_entities/heidelberg_kip/identifier_correction.json'
WORK = 'https://openalex.org/W4412083429'


def repair(author: dict, signature: str = SIGNATURE, receipt: str = RECEIPT,
           confirm_ownership: bool = True) -> tuple[dict, int]:
    if author.get('nature_v1_slots') is not None:
        return author, 0
    revised = json.loads(json.dumps(author))
    changed = 0
    for slot in revised.get('publisher_affiliation_slots') or []:
        if slot.get('text') != signature or set(slot.get('institution_ids', [])) != {OLD, NEW}:
            continue
        if slot.get('country_codes') != ['DE']:
            continue
        slot.update(institution_ids=[NEW], identity_status='official_review_corrected_source_identifier',
                    university_ownership_status=('officially_confirmed_university_self' if confirm_ownership
                                                 else 'not_inferred_from_registry'),
                    entity_type='education', official_identity_evidence=receipt,
                    identifier_conflict_resolution={'rejected_entity_id': OLD, 'receipt': receipt,
                                                    'original_identifiers_preserved': True})
        changed += 1
    if changed:
        remaining = {i for s in revised.get('publisher_affiliation_slots') or [] for i in s.get('institution_ids', [])}
        if OLD not in remaining:
            revised['institution_ids_or_verified_local_entities'] = [
                i for i in revised['institution_ids_or_verified_local_entities'] if i != OLD]
        if NEW not in revised['institution_ids_or_verified_local_entities']:
            revised['institution_ids_or_verified_local_entities'].append(NEW)
    return revised, changed


def run(work: str = WORK, signature: str = SIGNATURE, receipt: str = RECEIPT,
        confirm_ownership: bool = True, status_name: str = "correction_status.json") -> None:
    count = 0
    with (ROOT/'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        shard, = db.execute('SELECT shard FROM papers WHERE work_id=?', (work,)).fetchone()
        path = ROOT/'fields/papers'/shard
        table = pq.read_table(path)
        rows = table.to_pylist()
        for paper in rows:
            if paper['work_id'] != work:
                continue
            before = paper['authors_json']
            old = json.loads(before)
            resolved = [repair(a, signature, receipt, confirm_ownership) for a in old]
            count = sum(n for _, n in resolved)
            if not count:
                continue
            new = [a for a, _ in resolved]
            assert all(repair(a, signature, receipt, confirm_ownership)[1] == 0 for a in new)
            for a, b in zip(old, new):
                assert a.get('countries') == b.get('countries')
                for x, y in zip(a.get('publisher_affiliation_slots') or [], b.get('publisher_affiliation_slots') or []):
                    assert x.get('identifiers') == y.get('identifiers')
                    assert x.get('country_codes') == y.get('country_codes')
            paper['authors_json'] = json.dumps(new, ensure_ascii=False)
            db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                       (work, 'authors_json', before, paper['authors_json'], receipt,
                        'Exact reviewed German signature and official source supersede incorrect US namesake ROR; raw identifier retained'))
            values = metrics(paper)
            for metric, value in values.items():
                field = 'after_'+metric
                if field in paper and paper[field] != value:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (work, field, json.dumps(paper[field]), json.dumps(value), receipt, 'Recompute after reviewed identity correction'))
                    paper[field] = value
            db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                       (not values['every_author_institution'], not values['every_author_country'], work))
        if count:
            tmp = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tmp, compression='zstd')
            tmp.replace(path)
        db.commit()
        db.close()
    result = {'corrected_papers': int(bool(count)), 'corrected_slots': count,
              'original_identifiers_and_countries_preserved': True, 'repeat_safe': True,
              'unreviewed_other_signatures_unchanged': True, 'network_requests': 0}
    atomic_json(ROOT/'supplement/official_entities/heidelberg_kip'/status_name, result)
    if count:
        refresh()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
