"""Reconcile newly qualified PBJ bylines under the shared metadata writer lock."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from . import pnas_byline_repair as repair
from .springer_author_cache import prepare_source
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pbj_native_rosters'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'application.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = ROOT / 'supplement/pbj_native_formats'
        with (parent / 'application.lock').open('a') as dependency:
            fcntl.flock(dependency, fcntl.LOCK_EX)
            if json.loads((parent / 'status.json').read_text())['status'] != 'fixed_named_format_scope_applied':
                raise RuntimeError('PBJ qualification/metadata application did not complete')
        receipts = {json.loads(p.read_text())['doi']: str(p.relative_to(ROOT))
                    for p in (parent / 'staged').glob('*.json')}
        db = connect()
        groups = defaultdict(dict)
        for wid, doi, shard in db.execute("SELECT work_id,doi,shard FROM papers WHERE "
                "source_id=? AND status='included_verified_research'", ('https://openalex.org/S36510773',)):
            if doi in receipts:
                groups[shard][wid] = receipts[doi]
        db.close()
        records, reviews = [], []
        for shard, selected in groups.items():
            rows = pq.read_table(ROOT / 'fields/papers' / shard,
                filters=[('work_id', 'in', list(selected))]).to_pylist()
            for paper in rows:
                wid = paper['work_id']
                record, _, native = prepare_source(ROOT, paper, selected[wid], {'1467-7644', '1467-7652'})
                source = {**native, 'evidence_receipt': 'fields/qualification/' + record['raw_receipt']}
                current = json.loads(paper['authors_json'])
                authors, status = repair.reconcile(current, source)
                changed = authors != current or paper['author_list_may_be_truncated']
                if status == 'publisher_roster_reconciled' and changed:
                    records.append({'work_id': wid, 'doi': paper['doi'], 'shard': shard,
                                    'receipt': selected[wid]})
                elif status == 'publisher_roster_reconciled':
                    status = 'current_roster_already_agrees'
                reviews.append({'work_id': wid, 'doi': paper['doi'], 'receipt': selected[wid], 'status': status})
        atomic_json(FOLDER / 'duplicate_byline_evidence.json', {'records': records})
        atomic_json(FOLDER / 'review.json', {'records': reviews,
                    'outcomes': dict(Counter(r['status'] for r in reviews)), 'network_requests': 0})
        repair.FOLDER = FOLDER
        repair.run()


if __name__ == '__main__':
    run()
