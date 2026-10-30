"""Index alternative publication links from existing exact-matched secondary cache."""
from __future__ import annotations

import fcntl
import gzip
import json
import sqlite3
from collections import defaultdict
from urllib.parse import urlparse

from .secondary_markup_recovery import title_key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT/'supplement/cached_secondary_publication_links'
    folder.mkdir(exist_ok=True)
    with (folder/'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX|fcntl.LOCK_NB)
        target = sqlite3.connect(folder/'leads.sqlite')
        target.executescript('''CREATE TABLE IF NOT EXISTS reviewed(work_id TEXT PRIMARY KEY,doi TEXT,status TEXT,receipt TEXT,position INTEGER);
            CREATE TABLE IF NOT EXISTS leads(work_id TEXT,url TEXT,host TEXT,availability TEXT,style TEXT,receipt TEXT,position INTEGER,PRIMARY KEY(work_id,url));
            CREATE TABLE IF NOT EXISTS completed_files(path TEXT PRIMARY KEY);
        ''')
        scope = folder/'scope.json'
        if not scope.exists():
            source = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
            source.execute('ATTACH DATABASE ? AS bulk',(str(ROOT/'supplement/bulk_cache.sqlite'),))
            groups: dict[str, list[dict]] = defaultdict(list)
            for wid,doi,title,path,pos in source.execute('''SELECT p.work_id,p.doi,p.title,r.path,r.position
                FROM papers p JOIN bulk.records r USING(doi) WHERE p.status LIKE 'pending%' '''):
                groups[path].append({'work_id':wid,'doi':doi,'title':title,'position':pos})
            source.close()
            atomic_json(scope,{'files':dict(groups),'scope':'Current pending candidates with cached secondary records; no new candidates'})
        groups = json.loads(scope.read_text())['files']
        done = {r[0] for r in target.execute('SELECT path FROM completed_files')}
        for path, papers in groups.items():
            if path in done:
                continue
            records = json.loads(gzip.decompress((ROOT/path).read_bytes()))['response']['resultList']['result']
            for paper in papers:
                record = records[paper['position']]
                matched = ((record.get('doi') or '').lower() == paper['doi'] and bool(title_key(paper['title'] or ''))
                           and title_key(record.get('title') or '') == title_key(paper['title'] or ''))
                links = []
                if matched:
                    for link in record.get('fullTextUrlList',{}).get('fullTextUrl',[]):
                        url = link.get('url') or ''; parsed = urlparse(url)
                        if parsed.scheme in {'http','https'} and parsed.hostname not in {None,'doi.org','dx.doi.org'}:
                            links.append((paper['work_id'],url,parsed.hostname,link.get('availability',''),
                                          link.get('documentStyle',''),path,paper['position']))
                status = 'identity_conflict' if not matched else 'alternative_urls_found' if links else 'no_nonresolver_url'
                target.execute('INSERT OR REPLACE INTO reviewed VALUES(?,?,?,?,?)',
                               (paper['work_id'],paper['doi'],status,path,paper['position']))
                target.executemany('INSERT OR IGNORE INTO leads VALUES(?,?,?,?,?,?,?)',links)
            target.execute('INSERT INTO completed_files VALUES(?)',(path,));target.commit()
        summary = {'status':'fixed_cached_links_indexed','network_requests':0,'qualification_changes':0,
            'outcomes':dict(target.execute('SELECT status,count(*) FROM reviewed GROUP BY status')),
            'top_hosts':list(target.execute('SELECT host,count(DISTINCT work_id) n FROM leads GROUP BY host ORDER BY n DESC LIMIT 40')),
            'final_task_closure':False,'limitation':'Links require source applicability review; manuscript versions cannot establish final publisher type or online date.'}
        atomic_json(folder/'status.json',summary);print(json.dumps(summary),flush=True);target.close()


if __name__ == '__main__':
    run()
