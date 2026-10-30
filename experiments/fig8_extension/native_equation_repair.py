"""Replay affected native XML after selecting a single equation representation."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .publisher_delta_apply import apply_available
from .springer_native_xml import parse
from .scoap3_native_collect import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder = FOLDER / 'equation_repair'
    staged = folder / 'staged'
    staged.mkdir(parents=True, exist_ok=True)
    jobs = []
    affected = {}
    for old_path in (FOLDER / 'staged').glob('*.json'):
        old = json.loads(old_path.read_text())
        raw = (ROOT / 'fields/qualification' / old['raw_receipt']).resolve()
        parsed = parse(gzip.decompress(raw.read_bytes()))
        if parsed == old['parsed']:
            continue
        record = {**old, 'parsed': parsed, 'supersedes_parsing_receipt': str(old_path.relative_to(ROOT)),
                  'parse_correction': 'Select one EquationSource; do not concatenate alternative MathML and TeX'}
        target = staged / old_path.name
        atomic_json(target, record)
        jobs.append((old['doi'], str(target.relative_to(ROOT))))
        affected[old['doi']] = (old, record, str(target.relative_to(ROOT)))
    applied = apply_available(supplied_jobs=jobs) if jobs else {}
    corrected = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        groups = defaultdict(dict)
        for doi, source in affected.items():
            for wid, shard in db.execute('SELECT work_id,shard FROM papers WHERE doi=?', (doi,)):
                groups[shard][wid] = source
        for shard, selected in groups.items():
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            changed = False
            for paper in rows:
                if paper['work_id'] not in selected:
                    continue
                old, new, receipt = selected[paper['work_id']]
                before = old['parsed']['abstract']
                after = new['parsed']['abstract']
                source = 'matched_publisher:fields/qualification/' + old['raw_receipt']
                if (before == after or not after or paper['abstract'] != before
                        or paper.get('abstract_source') != source):
                    continue
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'abstract', json.dumps(before), json.dumps(after), receipt,
                            'Correct duplicated alternative equation representations from same publisher XML; exact old value and source match'))
                paper['abstract'] = after
                corrected.append(paper['work_id'])
                changed = True
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
            db.commit()
        db.close()
    result = {'affected_source_records': len(affected), 'application': applied,
              'existing_abstracts_corrected': corrected, 'network_requests': 0}
    atomic_json(folder / 'application.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()))
