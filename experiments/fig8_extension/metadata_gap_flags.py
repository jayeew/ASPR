"""Synchronize ledger gap caches with the just-recomputed qualified-paper gaps."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
from contextlib import nullcontext
from pathlib import Path

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(root: Path = ROOT, lock_held: bool = False) -> dict:
    context = nullcontext() if lock_held else (root/'supplement/checkpoint.lock').open('a')
    with context as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_EX)
        folder=root/'fields/qualification'
        expected=json.loads((folder/'research_metadata_gaps_summary.json').read_text())
        db=connect(root)
        try:
            count=db.execute("SELECT count(*) FROM papers WHERE status='included_verified_research'").fetchone()[0]
            if count!=expected['qualified_denominator']:
                raise ValueError('Qualification changed since gap recomputation')
            db.execute('CREATE TEMP TABLE current_gap_flags(work_id TEXT PRIMARY KEY,a INTEGER,i INTEGER,c INTEGER)')
            with gzip.open(folder/'research_metadata_gaps.csv.gz','rt',encoding='utf-8-sig') as stream:
                rows=[(r['work_id'],int(r['missing_abstract']=='True'),int(r['missing_institution']=='True'),int(r['missing_country']=='True')) for r in csv.DictReader(stream)]
            db.executemany('INSERT INTO current_gap_flags VALUES(?,?,?,?)',rows)
            totals=db.execute('SELECT count(*),coalesce(sum(a),0),coalesce(sum(i),0),coalesce(sum(c),0) FROM current_gap_flags').fetchone()
            wanted=tuple(expected[k] for k in ['any_metadata_gap','missing_abstract','missing_institution','missing_country'])
            if totals!=wanted or db.execute("SELECT count(*) FROM current_gap_flags g LEFT JOIN papers p USING(work_id) WHERE p.work_id IS NULL OR p.status!='included_verified_research'").fetchone()[0]:
                raise ValueError('Gap inventory does not agree with summary or current roster')
            before=db.execute("SELECT sum(abstract_gap),sum(institution_gap),sum(country_gap) FROM papers WHERE status='included_verified_research'").fetchone()
            db.execute("""UPDATE papers SET abstract_gap=coalesce((SELECT a FROM current_gap_flags g WHERE g.work_id=papers.work_id),0),
                institution_gap=coalesce((SELECT i FROM current_gap_flags g WHERE g.work_id=papers.work_id),0),
                country_gap=coalesce((SELECT c FROM current_gap_flags g WHERE g.work_id=papers.work_id),0)
                WHERE status='included_verified_research'""")
            after=db.execute("SELECT sum(abstract_gap),sum(institution_gap),sum(country_gap) FROM papers WHERE status='included_verified_research'").fetchone()
            assert after==wanted[1:]
            db.commit()
        finally:
            db.close()
    result={'qualified_papers':count,'before':dict(zip(['abstract','institution','country'],before)),
            'after':dict(zip(['abstract','institution','country'],after)),
            'source':'fields/qualification/research_metadata_gaps.csv.gz','paper_metadata_changes':0,
            'note':'Derived ledger gap cache only; task evidence, ownership and source outcomes preserved.'}
    atomic_json(root/'supplement/metadata_gap_flags_status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-held',action='store_true')
    print(json.dumps(run(lock_held=parser.parse_args().lock_held)),flush=True)
