"""Correct only byte-identical-source abstract decoding errors in the fixed cohort."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .supplement_apply import abstract_ok
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/native_abstract_polymer_types/encoding_review'


def run() -> None:
    review = json.loads((FOLDER / 'audit.json').read_text())['records']
    groups = defaultdict(dict)
    for r in review:
        if r['status'] == 'matched_cache_decoding_repair':
            wid, shard, receipt = r['job']
            groups[shard][wid] = (r['receipt'], receipt)
    counts, audit = Counter(), []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        for shard, selected in groups.items():
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path); rows = table.to_pylist(); changed = False
            for p in rows:
                wid = p['work_id']
                if wid not in selected:
                    continue
                old_receipt, new_receipt = selected[wid]
                old = json.loads((ROOT / old_receipt).read_text())
                new = json.loads((ROOT / new_receipt).read_text())
                assert new['encoding_review']['utf8_roundtrip_verified']
                assert new['raw_receipt'] == old['raw_receipt']
                before, after = old['parsed'].get('abstract', ''), new['parsed'].get('abstract', '')
                state = 'unchanged_or_different_source_preserved'
                if (before != after and p['abstract'] == before and before and abstract_ok(after)
                        and p['abstract_source'] == 'matched_publisher:fields/qualification/' + old['raw_receipt']):
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                        (wid, 'abstract', json.dumps(before), json.dumps(after), new_receipt,
                         'Same raw publisher page, explicit round-trip UTF-8 and exact DOI/title/journal; current abstract equals prior byte-parser output; decoding correction only'))
                    p['abstract'] = after; changed = True; state = 'same_source_abstract_encoding_corrected'
                counts[state] += 1
                audit.append({'work_id': wid, 'status': state, 'receipt': new_receipt})
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd'); temporary.replace(path)
            db.commit()
        db.close()
    atomic_json(FOLDER / 'abstract_corrections.json', {'counts': dict(counts), 'records': audit, 'network_requests': 0})
    refresh()
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
