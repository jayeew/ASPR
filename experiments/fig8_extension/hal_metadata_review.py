"""Record exact repository identity and remaining publisher-evidence limits."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter, defaultdict

from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> None:
    folder = ROOT/'supplement/hal_metadata'
    state = json.loads((folder/'status.json').read_text())
    if state['status'] != 'fixed_metadata_collected_identity_review_required':
        raise RuntimeError('HAL fixed acquisition not finished')
    scope = json.loads((folder/'scope.json').read_text())['identifiers']
    records = defaultdict(list)
    for path in (folder/'batches').glob('*.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        raw = ROOT/receipt['raw_receipt']
        for position,item in enumerate(json.loads(gzip.decompress(raw.read_bytes()))['response']['docs']):
            records[item['halId_s']].append((item, str(raw.relative_to(ROOT)),position))
    journals = {r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    authority_path = folder / 'journal_identity/rules.json'
    authority = json.loads(authority_path.read_text())['rules'] if authority_path.exists() else {}
    results = [];db = connect()
    for identifier,links in scope.items():
        for wid in sorted({link['work_id'] for link in links}):
            row = db.execute('SELECT doi,title,journal,source_id,status FROM papers WHERE work_id=?',(wid,)).fetchone()
            if row is None:
                raise ValueError('Fixed Work identity disappeared')
            doi,title,journal,sid,status = row
            reviewed = []
            for record,receipt,pos in records[identifier]:
                titles = record.get('title_s') or []
                if isinstance(titles,str):titles = [titles]
                authority_proofs = [proof for proof in authority.get(record.get('journalTitle_s') or '', [])
                                    if proof['source_id'] == sid]
                authority_issns = sorted({issn for proof in authority_proofs for issn in proof['issns']})
                parsed = {'title':titles,'journal_titles':[record.get('journalTitle_s') or ''],
                          'issns':authority_issns}
                matched = (doi and (record.get('doiId_s') or '').lower() == doi and
                           signature_match({'title':title or '', 'journal_name':journal,
                                            'source_id':sid}, parsed,journals[sid]))
                reviewed.append({'receipt':receipt,'position':pos,'identity_matched':bool(matched),
                    'journal_authority_proofs':authority_proofs,
                    'repository_type':record.get('docType_s'),'repository_date':record.get('publicationDate_s'),
                    'repository_file_urls':record.get('files_s',[]), 'file_main':record.get('fileMain_s','')})
            matched = [r for r in reviewed if r['identity_matched']]
            results.append({'work_id':wid,'doi':doi,'hal_identifier':identifier,'current_qualification':status,
                'status':'matched_repository_metadata_insufficient_primary_type_date' if matched else 'repository_identity_unresolved' if reviewed else 'identifier_not_returned',
                'record_reviews':reviewed,'qualification_changed':False,'final_task_closure':False})
    db.close()
    receipt = folder/'identity_review.jsonl'
    with receipt.open('w') as stream:
        for row in results:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    inserted = 0
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db = connect()
        for row in results:
            detail = json.dumps(row,ensure_ascii=False,sort_keys=True)
            for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(row['doi'],)).fetchall():
                if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND detail=?',
                    (task_id,'hal_repository_batch',str(receipt.relative_to(ROOT)),detail)).fetchone():
                    db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                        (task_id,'hal_repository_batch',row['status'],str(receipt.relative_to(ROOT)),detail));inserted += 1
        db.commit();db.close()
    atomic_json(folder/'identity_status.json',{'fixed_work_identifier_pairs':len(results),
        'outcomes':dict(Counter(row['status'] for row in results)),'new_ledger_attempts':inserted,
        'qualification_changes':0,'network_requests':0,'final_task_closure':False})


if __name__ == '__main__':
    run()
