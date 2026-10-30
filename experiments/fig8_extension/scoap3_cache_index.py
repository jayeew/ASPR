"""Index cached public SCOAP3 records and match existing candidates without fetching."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from lxml import html

from .supplement_apply import signature_match
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def journal_variants(titles: list[str]) -> list[str]:
    """Only the optional English leading article; retain all other journal tokens."""
    result=set(titles)
    for title in titles:
        base=re.sub(r'^the\s+', '', title.strip(), flags=re.IGNORECASE)
        result.update([base,'The '+base])
    return sorted(result)


def rematch(root: Path = ROOT) -> None:
    folder=root/'supplement/scoap3'
    db=sqlite3.connect(folder/'cache_index.sqlite')
    candidates=sqlite3.connect(f'file:{root / "supplement/tasks.sqlite"}?mode=ro',uri=True)
    changes=0
    for ident,digest,doi,payload in db.execute('SELECT id,payload_hash,doi,payload FROM records').fetchall():
        row=json.loads(payload)
        source={'title':[row.get('title','')], 'journal_titles':journal_variants([
            p.get('journal_title','') for p in row.get('publication_info',[])])}
        for wid,title,journal in candidates.execute('SELECT work_id,title,journal FROM papers WHERE doi=?',(doi,)):
            if not signature_match({'title':title,'journal_name':journal},source,set()):continue
            changes+=db.execute("UPDATE matches SET status='identity_matched' WHERE id=? AND payload_hash=? AND work_id=? AND status!='identity_matched'",(ident,digest,wid)).rowcount
    db.commit();candidates.close();db.close()
    atomic_json(folder/'journal_name_normalization.json',{'new_identity_matches':changes,
        'rule':'Exact DOI/title; journal name differs only by optional leading English The; raw names retained'})
    index(root)


def index(root: Path = ROOT) -> None:
    folder = root / 'supplement/scoap3'
    db = sqlite3.connect(folder / 'cache_index.sqlite')
    candidates = sqlite3.connect(f'file:{root / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS pages(receipt TEXT PRIMARY KEY, hash TEXT, status TEXT);
        CREATE TABLE IF NOT EXISTS records(id INTEGER, doi TEXT, payload_hash TEXT, payload TEXT,
            receipt TEXT, PRIMARY KEY(id,payload_hash));
        CREATE INDEX IF NOT EXISTS records_doi ON records(doi);
        CREATE TABLE IF NOT EXISTS matches(id INTEGER,payload_hash TEXT,work_id TEXT,
            status TEXT, PRIMARY KEY(id,payload_hash,work_id));
    ''')
    for receipt in sorted(folder.glob('*.json')):
        record = json.loads(receipt.read_text())
        if record.get('url') != 'https://repo.scoap3.org/search' or record.get('http_status') != 200:
            continue
        rel = str(receipt.relative_to(root))
        if db.execute('SELECT 1 FROM pages WHERE receipt=? AND hash=?', (rel, record['sha256'])).fetchone():
            continue
        tree = html.fromstring(gzip.decompress((root / record['raw_receipt']).read_bytes()),
                               parser=html.HTMLParser(huge_tree=True))
        scripts = tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
        if not scripts:
            db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?)', (rel,record['sha256'],'payload_missing'))
            db.commit()
            continue
        rows = json.loads(scripts[0])['props']['pageProps'].get('results', [])
        for row in rows:
            doi = (row.get('doi') or '').lower().removeprefix('https://doi.org/').strip()
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(payload.encode()).hexdigest()
            db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?)',
                       (row['id'],doi,digest,payload,rel))
            source = {'title':[row.get('title','')], 'journal_titles':journal_variants([
                p.get('journal_title','') for p in row.get('publication_info',[])])}
            for wid,title,journal in candidates.execute('SELECT work_id,title,journal FROM papers WHERE doi=?',(doi,)):
                matched = signature_match({'title':title,'journal_name':journal},source,set())
                db.execute('INSERT OR IGNORE INTO matches VALUES(?,?,?,?)',
                    (row['id'],digest,wid,'identity_matched' if matched else 'title_or_journal_mismatch'))
        db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?)',(rel,record['sha256'],'indexed'))
        db.commit()
    result = {'records':db.execute('SELECT count(DISTINCT id) FROM records').fetchone()[0],
        'record_variants':db.execute('SELECT count(*) FROM records').fetchone()[0],
        'matched_candidates':db.execute("SELECT count(DISTINCT work_id) FROM matches WHERE status='identity_matched'").fetchone()[0],
        'match_statuses':dict(db.execute('SELECT status,count(*) FROM matches GROUP BY status')),
        'pages':dict(db.execute('SELECT status,count(*) FROM pages GROUP BY status')),
        'qualification_applied':False,
        'limitations':'Generic material and publication_date are not original type or online-year proof; author mapping and source conflicts require separate review; pagination incompleteness retained'}
    atomic_json(folder/'cache_index_status.json',result)
    print(json.dumps(result),flush=True)
    candidates.close()
    db.close()


if __name__ == '__main__':
    index()
