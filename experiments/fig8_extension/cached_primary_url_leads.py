"""Index cached publication URLs for the fixed current unresolved work scope."""
from __future__ import annotations

import fcntl
import gzip
import json
import sqlite3
from collections import Counter
from urllib.parse import urlparse

import pyarrow.parquet as pq

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cached_primary_url_leads'
    folder.mkdir(exist_ok=True)
    with (folder / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(folder / 'leads.sqlite')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS targets(work_id TEXT PRIMARY KEY,doi TEXT);
        CREATE TABLE IF NOT EXISTS reviewed(work_id TEXT PRIMARY KEY,status TEXT,receipt TEXT);
        CREATE TABLE IF NOT EXISTS leads(work_id TEXT,url TEXT,host TEXT,kind TEXT,receipt TEXT,position INTEGER,
            PRIMARY KEY(work_id,url));
        CREATE TABLE IF NOT EXISTS completed_shards(shard TEXT PRIMARY KEY);
        ''')
        if not (folder / 'scope.json').exists():
            current = connect()
            db.executemany('INSERT OR IGNORE INTO targets VALUES(?,?)', current.execute(
                "SELECT work_id,doi FROM papers WHERE status LIKE 'pending%'"))
            current.close()
            db.commit()
            atomic_json(folder / 'scope.json', {'fixed_work_ids': db.execute('SELECT count(*) FROM targets').fetchone()[0],
                'scope': 'Current unresolved fixed candidates only; cached primary_location URLs; no network or qualification change'})
        targets = dict(db.execute('SELECT work_id,doi FROM targets'))
        done = {r[0] for r in db.execute('SELECT shard FROM completed_shards')}
        counts = Counter()
        for shard in sorted((ROOT / 'fields/papers').glob('*.parquet')):
            if shard.name in done:
                continue
            rows = pq.read_table(shard, columns=['work_id','doi','raw_receipt','raw_position']).to_pylist()
            by_receipt: dict[str, list[dict]] = {}
            for row in rows:
                if row['work_id'] in targets:
                    by_receipt.setdefault(row['raw_receipt'] or '', []).append(row)
            for receipt, items in by_receipt.items():
                path = ROOT / 'fields' / receipt
                if not receipt or not path.is_file():
                    for row in items:
                        db.execute('INSERT OR REPLACE INTO reviewed VALUES(?,?,?)', (row['work_id'],'raw_receipt_unavailable',receipt))
                    continue
                raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
                document = json.loads(raw)
                records = document.get('response', document).get('results', [])
                indexed = {r.get('id'): r for r in records}
                for row in items:
                    wid = row['work_id']; record = indexed.get(wid)
                    status = 'cached_work_identity_missing'
                    if record is not None:
                        doi = (record.get('doi') or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/')
                        if not targets[wid]:
                            status = 'current_doi_missing_source_identity_review_required'
                        elif doi != targets[wid]:
                            status = 'cached_doi_identity_conflict'
                        else:
                            status = 'no_primary_publication_url_in_selected_cache'
                            location = record.get('primary_location') or {}
                            for kind in ['landing_page_url','pdf_url']:
                                url = location.get(kind) or ''; parsed = urlparse(url)
                                if parsed.scheme not in {'http','https'} or not parsed.hostname:
                                    continue
                                db.execute('INSERT OR IGNORE INTO leads VALUES(?,?,?,?,?,?)',
                                    (wid,url,parsed.hostname,kind,receipt,row['raw_position']))
                                status = 'primary_publication_url_identified_unreviewed'
                    db.execute('INSERT OR REPLACE INTO reviewed VALUES(?,?,?)', (wid,status,receipt))
                    counts[status] += 1
            db.execute('INSERT INTO completed_shards VALUES(?)',(shard.name,));db.commit()
            atomic_json(folder / 'status.json', {'status':'indexing_cache','shard':shard.name,'this_run':dict(counts),'network_requests':0})
        summary = {'status':'fixed_cache_scope_indexed_not_source_closure','network_requests':0,
            'targets':len(targets),'reviewed':db.execute('SELECT count(*) FROM reviewed').fetchone()[0],
            'outcomes':dict(db.execute('SELECT status,count(*) FROM reviewed GROUP BY status')),
            'top_hosts':list(db.execute('SELECT host,count(DISTINCT work_id) n FROM leads GROUP BY host ORDER BY n DESC LIMIT 40')),
            'limitation':'A cached link is a discovery lead, not access or identity verification at the destination; locations outside the original select were not collected.'}
        atomic_json(folder / 'status.json', summary);db.close();print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    run()
