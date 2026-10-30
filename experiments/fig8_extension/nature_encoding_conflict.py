"""Resolve only source-identical official-slot conflicts proven to be decoding changes."""
from __future__ import annotations

import fcntl
import gzip
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_recompute import metrics
from .springer_author_cache import parse_authors
from .supplement_apply import preserve_official_slots
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/journal_type_policies/encoding_review'


def repair(author: dict, raw: bytes, receipt: str) -> dict:
    """Require exact author, ordered raw addresses, same receipt and compatible IDs."""
    conflict = author.get('conflicting_publisher_slots') or {}
    if (not author.get('official_identity_source_conflict') or author.get('nature_v1_slots')
            or author.get('publisher_affiliation_source') != receipt or conflict.get('receipt') != receipt):
        return author
    old = [a for a in parse_authors(raw)['authors'] if a['name'] == author['name']]
    new = [a for a in parse_authors(raw.decode('utf-8'))['authors'] if a['name'] == author['name']]
    if len(old) != 1 or len(new) != 1:
        return author
    previous = author.get('publisher_affiliation_slots') or []
    incoming = conflict.get('slots') or []
    if not any(s.get('official_identity_evidence') for s in previous):
        return author
    if ([a['text'] for a in old[0]['affiliations']] != [s['text'] for s in previous]
            or [a['text'] for a in new[0]['affiliations']] != [s['text'] for s in incoming]
            or len(previous) != len(incoming)):
        return author
    corrected = deepcopy(previous)
    for old_slot, new_slot in zip(corrected, incoming):
        old_slot['text'] = new_slot['text']
    retained, mismatch = preserve_official_slots(corrected, incoming)
    if mismatch:
        return author
    result = deepcopy(author)
    result['publisher_affiliation_slots'] = retained
    result['official_identity_source_conflict'] = False
    result.pop('conflicting_publisher_slots', None)
    result['official_identity_conflict_resolution'] = {
        'reason': 'Same raw page and ordered exact author addresses; UTF-8 decoding only; official entity IDs preserved',
        'receipt': str((FOLDER / 'official_conflict_review.json').relative_to(ROOT))}
    return result


def run() -> dict:
    cases = json.loads((FOLDER / 'official_conflict_review.json').read_text())['records']
    events = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for case in cases:
            wid = case['work_id']
            shard, = db.execute('SELECT shard FROM papers WHERE work_id=?', (wid,)).fetchone()
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            paper = next(r for r in rows if r['work_id'] == wid)
            before = deepcopy(paper)
            raw = gzip.decompress((ROOT / case['receipt']).read_bytes())
            authors = json.loads(paper['authors_json'])
            revised = [repair(a, raw, case['receipt']) if a['position'] == case['after']['position'] else a for a in authors]
            if revised == authors:
                events.append({'work_id': wid, 'status': 'unchanged_or_previously_resolved'})
                continue
            paper['authors_json'] = json.dumps(revised, ensure_ascii=False)
            values = metrics(paper)
            for metric, value in values.items():
                if 'after_' + metric in paper:
                    paper['after_' + metric] = value
            for field in paper:
                if before[field] != paper[field]:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (wid, field, json.dumps(before[field]), json.dumps(paper[field]),
                                str((FOLDER / 'official_conflict_review.json').relative_to(ROOT)),
                                'Source-identical UTF-8 correction; exact author/ordered addresses; official identity preserved'))
            db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                       (not values['every_author_institution'], not values['every_author_country'], wid))
            temporary = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
            temporary.replace(path)
            db.commit()
            events.append({'work_id': wid, 'status': 'encoding_only_conflict_resolved', 'country_unchanged': [a['countries'] for a in revised] == [a['countries'] for a in authors]})
        db.close()
    atomic_json(FOLDER / 'official_conflict_application.json', {'events': events, 'network_requests': 0})
    return {'events': events, 'network_requests': 0}


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
    from .derived_refresh import refresh
    refresh()
