"""Index candidate identity and cached evidence, then capture publisher JATS front matter."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from lxml import etree

BASE = Path('outputs/fig8_data/fields')
OUT = BASE / 'qualification'
LOCAL = threading.local()


def key(value: str | None) -> str:
    return ''.join(c for c in (value or '').casefold() if c.isalnum())


def index() -> None:
    db = sqlite3.connect(OUT/'evidence.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS candidates(work_id TEXT PRIMARY KEY,doi TEXT,title TEXT,source_id TEXT,issn TEXT,field TEXT,np INTEGER); CREATE INDEX IF NOT EXISTS doi_idx ON candidates(doi); CREATE TABLE IF NOT EXISTS secondary(doi TEXT,pmcid TEXT,types TEXT,online_date TEXT,title TEXT,receipt TEXT,oa TEXT); CREATE TABLE IF NOT EXISTS scanned(path TEXT PRIMARY KEY);')
    # Refresh identities from the current table when papers are appended or merged.
    db.execute("DELETE FROM candidates")
    for path in sorted((BASE/'papers').glob('*.parquet')):
        rows=pq.read_table(path,columns=['work_id','doi','title','source_id','issn_l','field','nature_portfolio_official_current']).to_pylist()
        db.executemany('INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?)',[(r['work_id'],r['doi'],r['title'],r['source_id'],r['issn_l'],r['field'],r['nature_portfolio_official_current']) for r in rows])
        db.commit()
    seen={r[0] for r in db.execute('SELECT path FROM scanned')}
    paths=sorted((BASE/'epmc_raw').glob('*.gz'))
    for i,path in enumerate(paths,1):
        if path.name in seen:continue
        with gzip.open(path,'rt') as stream:receipt=json.load(stream)
        if receipt.get('status')=='complete':
            requested=set(receipt['dois'])
            rows=[]
            for r in receipt['response'].get('resultList',{}).get('result',[]):
                doi=(r.get('doi') or '').lower()
                if doi in requested and r.get('source') in ['MED','PMC']:
                    rows.append((doi,r.get('pmcid',''),json.dumps(r.get('pubTypeList',{}).get('pubType',[])),r.get('electronicPublicationDate',''),r.get('title',''),'../epmc_raw/'+path.name,r.get('isOpenAccess','')))
            db.executemany('INSERT INTO secondary VALUES (?,?,?,?,?,?,?)',rows)
        db.execute('INSERT INTO scanned VALUES (?)',(path.name,))
        if i%200==0:db.commit();print('indexed',i,len(paths),flush=True)
    db.execute('CREATE INDEX IF NOT EXISTS secondary_doi ON secondary(doi)');db.commit()
    targets=[]
    for doi,pmc,title in db.execute("SELECT DISTINCT s.doi,s.pmcid,s.title FROM secondary s JOIN candidates c USING(doi) WHERE s.oa='Y' AND s.pmcid<>'' AND c.np=0"):
        titles={key(r[0]) for r in db.execute('SELECT title FROM candidates WHERE doi=?',(doi,))}
        if key(title) in titles: targets.append({'doi':doi,'pmcid':pmc})
    (OUT/'jats_targets.json').write_text(json.dumps(targets))
    print('JATS targets',len(targets),flush=True)
    db.close()


def parse_front(data: bytes) -> dict:
    parser=etree.XMLParser(resolve_entities=False,no_network=True)
    root=etree.fromstring(data,parser)
    meta=root.find('./front/article-meta')
    if meta is None: raise ValueError('No article-meta')
    text=lambda nodes:[''.join(n.itertext()).strip() for n in nodes]
    dates=[]
    for date in meta.findall('./pub-date'):
        dates.append({'attributes':dict(date.attrib),'year':date.findtext('year'),'month':date.findtext('month'),'day':date.findtext('day')})
    return {'article_type':root.get('article-type',''),'doi':text(meta.xpath('./article-id[@pub-id-type="doi"]')),
            'title':text(meta.xpath('./title-group/article-title')),'categories':text(meta.xpath('./article-categories//subject')),
            'issn':text(root.xpath('./front/journal-meta/issn')),
            'journal_titles':text(root.xpath('./front/journal-meta/journal-title-group/journal-title|./front/journal-meta/journal-title')),
            'publisher':text(root.xpath('./front/journal-meta/publisher/publisher-name')),'dates':dates,
            'related_articles':[dict(n.attrib) for n in meta.findall('./related-article')]}


def fetch(item: dict) -> dict:
    pmc=item['pmcid']; dest=OUT/'jats_parsed'/f'{pmc}.json'
    if dest.exists():return json.loads(dest.read_text())
    url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML'
    receipt={**item,'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
    session=getattr(LOCAL,'session',None)
    if session is None:
        session=LOCAL.session=requests.Session();session.trust_env=False
    for attempt in range(2):
        try:
            with session.get(url,timeout=(10,35),stream=True) as response:
                receipt['http_status']=response.status_code
                if not response.ok:
                    receipt['status']='http_error'
                    if response.status_code not in [429,500,502,503,504]:break
                else:
                    data=b'';prefix=None
                    for chunk in response.iter_content(131072):
                        if prefix is not None:continue
                        data+=chunk
                        end=data.find(b'</front>')
                        if end>=0:
                            prefix=data[:end+len(b'</front>')];data=b''
                        elif len(data)>8_000_000:raise ValueError('Front exceeds limit')
                    if prefix is None:raise ValueError('No complete front matter')
                    # Save the exact returned prefix; parse a synthetic closing root separately.
                    path=OUT/'jats_raw'/f'{pmc}.front.xml.gz'
                    path.write_bytes(gzip.compress(prefix,mtime=0))
                    parsed=parse_front(prefix+b'</article>')
                    receipt.update(parsed=parsed,status='parsed',raw_receipt=str(path.relative_to(OUT)),raw_sha256=hashlib.sha256(prefix).hexdigest(),capture_scope='Exact XML prefix through </front> retained; remaining HTTP body consumed without retention for connection reuse')
                    break
        except (requests.RequestException,ValueError,etree.XMLSyntaxError) as error:
            receipt.update(status='unresolved',error=str(error)[:300])
        time.sleep(2**attempt)
    dest.write_text(json.dumps(receipt,ensure_ascii=False))
    return receipt


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['index','fetch']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.action=='index':index();return
    for name in ['jats_raw','jats_parsed']:(OUT/name).mkdir(exist_ok=True)
    items=json.loads((OUT/'jats_targets.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fetch,r) for r in {r['pmcid']:r for r in items}.values()]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%500==0:print('JATS',i,len(futures),flush=True)
    print('JATS complete',len(futures),flush=True)


if __name__=='__main__':main()
