"""Explain every current unmatched country slot in the fixed ISO-postal scope."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .iso_postal_country_cache import FOLDER, ISO
from .nature_backfill_resolve import clean_address
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv


def run() -> None:
    summary = json.loads((FOLDER/'combined_application_summary.json').read_text())
    selected = {r['work_id'] for r in summary['remaining']}
    targets = defaultdict(dict)
    for relative in ['scope.json','cleaned_native/scope.json']:
        for paper in json.loads((FOLDER/relative).read_text())['papers']:
            if paper['work_id'] in selected:
                for slot in paper['slots']:
                    targets[paper['work_id']][(slot['receipt'],clean_address(slot['address']))] = slot['code']
    iso = json.loads(ISO.read_text())['3166-1']
    codes = {e[k]:e['alpha_2'] for e in iso for k in ['alpha_2','alpha_3']}
    journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    db=connect();groups=defaultdict(list);paper_states=[]
    for wid in sorted(selected):
        row=db.execute('SELECT shard,status FROM papers WHERE work_id=?',(wid,)).fetchone()
        if row and row[1]=='included_verified_research':groups[row[0]].append(wid)
        else:paper_states.append({'work_id':wid,'status':'qualification_changed','current_status':row[1] if row else 'missing'})
    db.close();reviews=[]
    for shard, ids in groups.items():
        rows=pq.read_table(ROOT/'fields/papers'/shard,columns=['work_id','doi','title','journal_name','source_id','authors_json'],filters=[('work_id','in',ids)]).to_pylist()
        for paper in rows:
            wid=paper['work_id'];authors=json.loads(paper['authors_json']);old_counts=Counter(key(a['name']) for a in authors);native_cache={};count=0
            for author in authors:
                receipt=author.get('publisher_affiliation_source','')
                for slot in author.get('publisher_affiliation_slots') or []:
                    address=clean_address(slot.get('text',''));expected=targets[wid].get((receipt,address))
                    if not expected or slot.get('country_codes'):continue
                    count+=1
                    if receipt not in native_cache:native_cache[receipt]=native_metadata(receipt,paper['doi'])
                    native=native_cache[receipt];names=Counter(key(a['name']) for a in native.get('authors',[]));name=key(author['name'])
                    affiliations=[s for a in native.get('authors',[]) if key(a['name'])==name for s in a['affiliations'] if clean_address(s['text'])==address]
                    signed={c.get('code','').upper() for s in affiliations for c in s.get('countries',[])}-{''}
                    if author.get('nature_v1_slots') is not None:status='nature_review_preserved'
                    elif not native.get('authors'):status='native_adapter_or_author_links_missing'
                    elif paper['doi'] not in {d.lower() for d in native.get('doi',[])} or not signature_match(paper,native,journals[paper['source_id']]):status='native_identity_unconfirmed'
                    elif old_counts[name]!=1 or names[name]>1:status='duplicate_author_name_requires_disambiguation'
                    elif names[name]==0:status='native_author_name_not_matched'
                    elif not affiliations:status='native_author_address_not_matched'
                    elif signed and signed!={expected}:
                        status='explicit_ISO_code_format_normalization_available' if {codes.get(c,c) for c in signed}=={expected} else 'native_country_code_conflicts_with_literal_country'
                    else:status='matching_native_link_requires_application_review'
                    reviews.append({'work_id':wid,'doi':paper['doi'],'author_name':author['name'],'position':author['position'],
                        'address':address,'expected_literal_code':expected,'source_receipt':receipt,'native_country_codes':sorted(signed),
                        'current_name_count':old_counts[name],'native_name_count':names[name],'status':status,'final_task_closure':False})
            paper_states.append({'work_id':wid,'current_unfilled_target_slots':count,'status':'remaining_target_slots_reviewed' if count else 'no_current_unfilled_target_slots_found'})
    result={'records':reviews,'paper_states':paper_states,'current_unfilled_target_slots':len(reviews),
            'outcomes':dict(Counter(r['status'] for r in reviews)),'network_requests':0,'final_task_closure':False}
    atomic_json(FOLDER/'remaining_native_review.json',result);print(json.dumps({k:v for k,v in result.items() if k not in {'records','paper_states'}}),flush=True)


if __name__=='__main__':
    run()
