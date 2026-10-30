"""Reuse native Elsevier front matter with deposited journal identity; dates stay unknown."""
from __future__ import annotations
import gzip
import json
import os
import sqlite3
from .elsevier_native_xml import parse
from .publisher_delta_apply import apply_available
from .scoap3_native_collect import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder=FOLDER/'elsevier';folder.mkdir(exist_ok=True);staged=folder/'staged';staged.mkdir(exist_ok=True)
    db=sqlite3.connect(f'file:{ROOT}/supplement/scoap3/cache_index.sqlite?mode=ro',uri=True)
    ledger=connect()
    processed={r[0] for r in ledger.execute("SELECT receipt FROM supplied_publisher_processed WHERE receipt LIKE 'supplement/scoap3_native/elsevier/staged/%'")}
    ledger.close()
    jobs=[];held=[];seen=set()
    for path in sorted((FOLDER/'receipts').glob('*.json')):
        if str((staged/path.name).relative_to(ROOT)) in processed:continue
        r=json.loads(path.read_text())
        if r.get('root_tag') not in {'{http://www.elsevier.com/xml/ja/dtd}article', '{http://www.elsevier.com/xml/ja/dtd}simple-article'}:continue
        doi=r['doi']
        if doi in seen:continue
        parsed=parse(gzip.decompress((ROOT/r['raw_receipt']).read_bytes()))
        rows=db.execute('SELECT payload,receipt FROM records WHERE doi=?',(doi,)).fetchall()
        journals={x['journal_title'] for payload,_ in rows for x in json.loads(payload).get('publication_info',[]) if x.get('journal_title')}
        if parsed.get('doi')!=[doi] or len(journals)!=1:
            held.append({'receipt':str(path.relative_to(ROOT)),'reason':'native_doi_or_deposited_journal_not_unique'});continue
        parsed['journal_titles']=sorted(journals)
        record={'source':'publisher_native_xml','route':'elsevier','status':'parsed','doi':doi,'url':r['url'],
                'raw_sha256':r['sha256'],'raw_receipt':os.path.relpath(ROOT/r['raw_receipt'],ROOT/'fields/qualification'),
                'parsed':parsed,'journal_identity_receipts':sorted({receipt for _,receipt in rows}),
                'type_rule':'supplement/scoap3_native/elsevier/code_rule.json',
                'limitation':'Full-length article retained as native type, not promoted to original research without journal rule; no inferred online date'}
        target=staged/path.name;atomic_json(target,record);jobs.append((doi,str(target.relative_to(ROOT))));seen.add(doi)
    db.close()
    atomic_json(folder/'scope.json',{'jobs':jobs,'held':held,'source':'cached publisher XML plus DOI-matched publisher deposit journal identity'})
    result=apply_available(supplied_jobs=jobs)
    ledger=connect();ledger.execute('PRAGMA busy_timeout=900000')
    for doi,receipt in jobs:
        for tid, in ledger.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(doi,)).fetchall():
            if not ledger.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,'scoap3_elsevier_native_xml',receipt)).fetchone():
                ledger.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(tid,'scoap3_elsevier_native_xml','native_type_and_identity_evidence_reviewed',receipt,json.dumps({'final_task_closure':False,'online_date_not_inferred':True})))
                ledger.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,))
    ledger.commit();ledger.close()
    atomic_json(folder/'application.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':run()
