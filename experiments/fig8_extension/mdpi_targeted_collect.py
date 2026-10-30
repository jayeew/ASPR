"""Consume fixed known MDPI identifiers after unproductive bulk pagination ends."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from lxml import etree

from . import ojs_metadata_harvest as transport
from .mdpi_cache_apply import evidence
from .mdpi_oai_harvest import FOLDER, NS, URL, extract
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

TARGET = FOLDER / 'targeted'


def run(workers: int = 16) -> None:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    with (TARGET / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = TARGET / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('stop_reason'):
            return
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        out = TARGET / 'results';out.mkdir(exist_ok=True)
        jobs = json.loads((TARGET / 'scope.json').read_text())['papers']
        pending = [p for p in jobs if not (out / (hashlib.sha256(p['doi'].encode()).hexdigest()+'.json')).exists()]
        applied = json.loads((FOLDER/'type_application/applied_fingerprints.json').read_text())
        known = {k.split(':native:',1)[0] for k in applied}
        for paper in list(pending):
            if paper['doi'] in known:
                token=hashlib.sha256(paper['doi'].encode()).hexdigest()
                atomic_json(out/(token+'.json'),{'doi':paper['doi'],'work_id':paper['work_id'],
                    'status':'existing_native_type_reused','receipt':'supplement/mdpi_oai/type_application/applied_fingerprints.json',
                    'network_requests':0})
        pending = [p for p in pending if p['doi'] not in known]
        db=connect();db.execute('PRAGMA busy_timeout=900000')
        counts = Counter()
        for p in out.glob('*.json'):
            counts[json.loads(p.read_text())['status']] += 1
        registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        recent = deque(maxlen=100);stop='';iterator=iter(pending)
        def status(state: str) -> None:
            atomic_json(status_path, {'status':state,'stop_reason':stop,'fixed_dois':len(jobs),
                'completed':sum(counts.values()),'matched_records':counts['matched_native_type'],
                'outcomes':dict(counts),'workers':workers,'network_route':'direct_trust_env_false'})
        def submit(pool: ThreadPoolExecutor, active: dict) -> None:
            paper = next(iterator,None)
            if paper:
                active[pool.submit(transport.fetch,URL,{'verb':'GetRecord','metadataPrefix':'oai_dc',
                    'identifier':paper['oai_identifier']})] = paper
        status('running')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active={}
            for _ in range(workers):submit(pool,active)
            while active:
                completed,_=wait(active,return_when=FIRST_COMPLETED)
                for future in completed:
                    paper=active.pop(future);receipt=future.result();result='source_unavailable';proof=None
                    if receipt.get('http_status')==200:
                        raw=gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes())
                        try:
                            tree=etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True))
                            nodes=tree.findall('o:GetRecord/o:record',NS)
                            result='identifier_absent_or_nonunique'
                            if len(nodes)==1:
                                item=extract(nodes[0])
                                if item['oai_identifier']==paper['oai_identifier']:
                                    proof=evidence(item,paper,registry[paper['source_id']],receipt,hashlib.sha256(raw).hexdigest())
                                result='matched_native_type' if proof else 'identity_or_type_unconfirmed'
                        except etree.XMLSyntaxError:
                            result='invalid_xml'
                    token=hashlib.sha256(paper['doi'].encode()).hexdigest()
                    if proof:
                        atomic_json(FOLDER/'matched_pages'/('targeted_'+token+'.json'),
                            {'receipt':receipt,'matches':[{'doi':paper['doi'],'oai_identifier':paper['oai_identifier']}],
                             'discovery_receipt':paper['discovery_receipt']})
                    atomic_json(out/(token+'.json'),{'doi':paper['doi'],'work_id':paper['work_id'],
                        'status':result,'receipt':receipt,'discovery_receipt':paper['discovery_receipt'],
                        'qualification_not_implied':True})
                    outcome=str((out/(token+'.json')).relative_to(ROOT))
                    for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(paper['doi'],)).fetchall():
                        if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                                          (tid,'mdpi_targeted_official_oai',outcome)).fetchone():
                            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                                (tid,'mdpi_targeted_official_oai',result,outcome,json.dumps({'final_task_closure':False,
                                    'qualification_not_implied':True,'http_attempts':len(receipt.get('attempts',[]))})))
                            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,))
                    db.commit()
                    counts[result]+=1;recent.append(bool(proof))
                    if receipt.get('http_status') in {401,403,429}:stop='source_permission_or_rate_restriction'
                    elif result=='source_unavailable':stop='transport_attempts_exhausted_or_source_failure'
                    elif len(recent)==100 and not any(recent):stop='100_consecutive_unmatched_records_review_required'
                    status('draining_inflight' if stop else 'running')
                    if not stop:submit(pool,active)
        status('stopped_unattempted_retained' if stop else 'fixed_queue_processed')
        db.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    run(workers=parser.parse_args().workers)
