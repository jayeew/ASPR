"""Finish fixed Copernicus candidate metadata using cached XML then public article XML."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import etree

from .copernicus_bulk import NS, parse_article
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def front_article(raw: bytes) -> etree._Element:
    if raw.rstrip().endswith(b'</front>'):raw+=b'</article>'
    return etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True,huge_tree=True))


def fetch(url: str, folder: Path) -> dict:
    token=hashlib.sha256(url.encode()).hexdigest();path=folder/(token+'.json');raw_path=folder/(token+'.xml.gz')
    if path.exists():return json.loads(path.read_text())
    s=requests.Session();s.trust_env=False
    r={'url':url,'attempts':[],'retrieved_at':datetime.now(timezone.utc).isoformat(),'raw_receipt':str(raw_path)}
    for attempt in range(3):
        try:
            with s.get(url,stream=True,timeout=(10,45)) as response:
                r['http_status']=response.status_code;r['attempts'].append({'http_status':response.status_code});data=b''
                for chunk in response.iter_content(16384):
                    data+=chunk
                    if b'</front>' in data:data=data.split(b'</front>',1)[0]+b'</front>';break
                    if len(data)>8_000_000:r['metadata_size_limit']=True;break
                atomic_bytes(raw_path,gzip.compress(data));r['sha256']=hashlib.sha256(data).hexdigest()
                if response.status_code not in {500,502,503,504}:break
        except requests.RequestException as error:r['attempts'].append({'error':type(error).__name__})
        if attempt<2:time.sleep(2**attempt)
    atomic_json(path,r);time.sleep(1.05);return r


def run() -> dict:
    out=ROOT/'supplement/copernicus';folder=out/'direct';folder.mkdir(exist_ok=True)
    scope=set(json.loads((out/'scope.json').read_text())['candidate_dois']);db=sqlite3.connect(out/'metadata.sqlite');stats=Counter();audit=[]
    # Reuse native receipts even if the old collector stopped before its page commit.
    for path in (ROOT/'supplement/official_bulk').glob('*.json'):
        r=json.loads(path.read_text())
        if r.get('params',{}).get('verb')!='GetRecord' or r.get('http_status')!=200:continue
        try:
            tree=front_article(gzip.decompress(Path(r['raw_receipt']).read_bytes()));node=tree.find('./o:GetRecord/o:record/o:metadata',NS)
            if node is None or not len(node):continue
            payload=parse_article(node[0]);ids=set(payload['doi'])&scope
            if len(ids)!=1:continue
            doi=ids.pop()
            if not db.execute('SELECT 1 FROM records WHERE doi=?',(doi,)).fetchone():
                db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)',(r['params']['identifier'],doi,json.dumps(payload),r['raw_receipt']));stats['native_cache_recovered']+=1
        except (OSError,etree.XMLSyntaxError,ValueError):stats['invalid_cached_native']+=1
    db.commit();blocked_hosts=set()
    for doi in sorted(scope):
        if db.execute('SELECT 1 FROM records WHERE doi=?',(doi,)).fetchone():continue
        match=re.fullmatch(r'10.5194/([a-z]+)-(\d+)-(\d+)-(202[3-5])',doi)
        if not match:audit.append({'doi':doi,'status':'no_validated_article_path_pattern'});continue
        journal,volume,page,year=match.groups();url=f'https://{journal}.copernicus.org/articles/{volume}/{page}/{year}/{doi.split("/")[1]}.xml'
        if journal in blocked_hosts:audit.append({'doi':doi,'status':'source_access_restricted_no_repeat','url':url});continue
        if doi=='10.5194/gi-12-239-2023':r=json.loads((out/'gi_direct_xml_probe.json').read_text())
        else:r=fetch(url,folder)
        event={'doi':doi,'url':url,'receipt':r,'status':'source_unresolved'}
        if r.get('http_status') in {401,403,429}:blocked_hosts.add(journal)
        if r.get('http_status')==200 and not r.get('metadata_size_limit'):
            try:
                payload=parse_article(front_article(gzip.decompress(Path(r['raw_receipt']).read_bytes())))
                if payload['doi']==[doi]:
                    db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)',(url,doi,json.dumps(payload),r['raw_receipt']));event['status']='doi_matched_pending_title_journal_review';stats['direct_metadata']+=1
                else:event['status']='doi_identity_mismatch'
            except (OSError,etree.XMLSyntaxError,ValueError) as error:event.update(status='xml_parse_unresolved',error=str(error))
        audit.append(event);db.commit();atomic_json(out/'direct_status.json',{'status':'running',**stats,'processed_remaining':len(audit)})
    db.close();atomic_json(out/'direct_audit.json',audit)
    result={'status':'fixed_remaining_paths_processed','scoped_candidates':len(scope),**stats,'processed_remaining':len(audit),'unresolved_direct':sum(e['status']!='doi_matched_pending_title_journal_review' for e in audit)}
    atomic_json(out/'direct_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
