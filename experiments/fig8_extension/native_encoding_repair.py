"""Reparse fixed cached native identity failures, without downloading or fuzzy matching."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, native_html_text, parse_publisher
from .three_journal_apply import read_csv

F=ROOT/'supplement/target_500k/springer_unrouted'


def run() -> None:
    folder=F/'encoding_repair';folder.mkdir(exist_ok=True)
    selected=set(json.loads((F/'scope.json').read_text())['dois'])
    selected.update(r['doi'] for r in json.loads((F/'other_series_inventory.json').read_text())['records'] if r['journal']!='Scientific American')
    registry={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    results=[]
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect()
        rows=db.execute("select doi,route,receipt from publisher_remaining_jobs where route in ('springer','nature') and status='publication_identity_mismatch'").fetchall()
        for doi,route,receipt in rows:
            if doi not in selected:continue
            dest=folder/(hashlib.sha256(receipt.encode()).hexdigest()+'.json')
            if dest.exists():continue
            old=json.loads((ROOT/receipt).read_text())
            raw=gzip.decompress((ROOT/'fields/qualification'/old['raw_receipt']).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==old['raw_sha256']
            page=native_html_text(raw,route)
            parsed=parse_publisher(page)
            candidates=db.execute('select title,journal,source_id from papers where doi=?',(doi,)).fetchall()
            matched=doi in parsed['doi'] and len(candidates)==1 and signature_match({'title':candidates[0][0],'journal_name':candidates[0][1],'source_id':candidates[0][2]},parsed,registry[candidates[0][2]])
            revised={**old,'parsed':parsed,'raw_text_encoding':'utf-8' if isinstance(page,str) else old.get('raw_text_encoding',''), 'prior_receipt':receipt,'network_requests':0,'repair_reason':'Decode original UTF-8 HTML bytes before native metadata parsing; exact identity rule unchanged','identity_confirmed_after_reparse':bool(matched)}
            atomic_json(dest,revised)
            relative=str(dest.relative_to(ROOT))
            if matched:
                db.execute("update publisher_remaining_jobs set status='parsed',receipt=? where doi=? and receipt=?",(relative,doi,receipt))
                ident=hashlib.sha256(('qualification:'+doi).encode()).hexdigest()[:32]
                detail={'previous_receipt':receipt,'before_title':old['parsed']['title'],'after_title':parsed['title'],'raw_sha256':old['raw_sha256'],'qualification_not_yet_applied':True}
                db.execute('insert into attempts(task_id,source,status,receipt,detail) values(?,?,?,?,?)',(ident,'native_cached_utf8_reparse','identity_matched',relative,json.dumps(detail)))
            results.append({'doi':doi,'prior_receipt':receipt,'reparsed_receipt':relative,'identity_confirmed':bool(matched)})
        db.commit();db.close()
    result={'newly_reviewed':len(results),'recovered_exact_identities':sum(r['identity_confirmed'] for r in results),'network_requests':0,'records':results,'qualification_changes':0}
    with (folder/'runs.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':run()
