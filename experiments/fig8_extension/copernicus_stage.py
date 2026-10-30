"""Stage exact-matched cached publisher XML for the shared incremental qualifier."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .fig8b_qualification_decide import electronic_date
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def adapted(payload: dict) -> dict:
    date,precision=electronic_date(payload['dates'])
    return {**payload,'issns':payload['issn'],
        'article_types':list(dict.fromkeys([payload['article_type'],*payload['categories']])),
        'online_date':date,'date_precision':precision,
        'normalization_note':'Official NLM namespace and contrib@rid parsed; explicit electronic date only; native categories retained'}


def stage(root: Path = ROOT) -> dict:
    source=sqlite3.connect(root/'supplement/copernicus/metadata.sqlite')
    grouped=defaultdict(list)
    for ident,doi,payload,receipt in source.execute('SELECT identifier,doi,payload,receipt FROM records'):
        grouped[doi].append((ident,json.loads(payload),receipt))
    source.close();db=connect(root);stats=Counter();audit=[]
    journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(root/'fields/journal_registry.csv')}
    for doi,records in grouped.items():
        variants={json.dumps(row[1],sort_keys=True) for row in records}
        if len(variants)!=1:
            stats['source_metadata_conflicts']+=1;audit.append({'doi':doi,'status':'conflicting_cached_metadata','receipts':[v[2] for v in records]});continue
        ident,payload,raw=records[0];p=adapted(payload)
        matches=[]
        for wid,title,journal,sid in db.execute('SELECT work_id,title,journal,source_id FROM papers WHERE doi=?',(doi,)):
            if signature_match({'title':title,'journal_name':journal},p,journals[sid]):matches.append(wid)
        if len(matches)!=1:
            stats['identity_unmatched_or_ambiguous']+=1;audit.append({'doi':doi,'status':'identity_unmatched_or_ambiguous','matched_work_ids':matches,'receipt':raw});continue
        url=ident if ident.startswith('https://') else 'https://oai-pmh.copernicus.org/oai.php?verb=GetRecord&metadataPrefix=pmc&identifier='+ident
        token=hashlib.sha256((url+raw).encode()).hexdigest()
        destination=root/'fields/qualification/supplement_parsed'/('copernicus_'+token+'.json')
        record={'source':'publisher','route':'copernicus','status':'parsed','doi':doi,'url':url,
            'raw_receipt':os.path.relpath(Path(raw),root/'fields/qualification'),
            'parsed':p,'retrieval_mode':'Reused official ListRecords response; GetRecord URL identifies record, not an additional request',
            'native_evidence':raw,'matched_work_id':matches[0]}
        if not destination.exists():atomic_json(destination,record)
        stats['matched_sources_staged']+=1
        stats['new_incremental_jobs']+=db.execute("INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status,receipt,applied) VALUES(?,?,?,?,0)",
            (doi,'copernicus','parsed',str(destination.relative_to(root)))).rowcount
        stats['explicit_online_date_present']+=bool(p['online_date'])
        audit.append({'doi':doi,'status':'staged_primary_evidence','receipt':str(destination.relative_to(root)),'explicit_online_date_present':bool(p['online_date'])})
    db.commit();db.close()
    scope=json.loads((root/'supplement/copernicus/scope.json').read_text())
    audit.extend({'doi':doi,'status':'not_found_in_harvested_update_date_scope','qualification_not_resolved':True} for doi in scope['candidate_dois'] if doi not in grouped)
    atomic_json(root/'supplement/copernicus/stage_records.json',audit)
    atomic_json(root/'supplement/copernicus/stage_status.json',dict(stats))
    print(json.dumps(dict(stats)),flush=True)
    return dict(stats)


if __name__=='__main__':stage()
