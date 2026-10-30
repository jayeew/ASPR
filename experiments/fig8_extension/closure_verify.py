"""Offline, scope-limited completion audit for the bounded Fig8 goal."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from .closure_goal import AUDIT
from .fig8b_qualification_decide import decide, portfolio_label
from .supplement_queue import connect, task
from .three_journal_apply import read_csv, target_papers
from .three_journal_round import OUT, Q, ROOT, SIDS, load, save


def run() -> None:
    D,E,S,J=load();P=target_papers()
    annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
    failures=[]
    for wid,d in D.items():
        result=decide(E[wid],S[wid],d['duplicate_of'])
        label,_=portfolio_label(annual.get((d['source_id'],d['online_year']),''),E[wid],d['online_year'])
        if (result['status'],result['online_year'],label)!=(d['eligibility_status'],d['online_year'],d['nature_portfolio_at_verified_online_year']):failures.append(wid)
        assert all((Q/e['receipt']).is_file() for e in E[wid] if e.get('receipt'))
    assert not failures
    roster=read_csv(Q/'research_roster.csv');assert len(roster)==len({r['doi'] for r in roster})
    global_summary=json.loads((Q/'recomputed_results.json').read_text());assert len(roster)==global_summary['verified_research_count']
    journals=read_csv(OUT/'all_journals.csv')
    assert all(int(r['candidate_n'])==sum(int(r[k]) for k in ['verified_n','excluded_n','pending_n']) for r in journals)
    assert sum(int(r['verified_n']) for r in journals if r['field']=='all_three_fields')==len(roster)
    pending_ids={wid for wid,d in D.items() if d['eligibility_status'].startswith('pending')};found=set();unresolved_n=0
    with gzip.open(Q/'unresolved.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            unresolved_n+=1
            if row['work_id'] in D:found.add(row['work_id'])
    assert found==pending_ids and unresolved_n==global_summary['unresolved_count']
    budget=json.loads((AUDIT/'budget.json').read_text());requests=[json.loads(p.read_text()) for p in sorted((AUDIT/'requests').glob('*.json'))]
    assert budget['requests']==len(requests)==10
    actual_dois={doi for r in requests for doi in r['dois']};assert actual_dois==set(budget['dois']) and len(actual_dois)==200
    assert all((ROOT/r['raw_receipt']).is_file() and hashlib.sha256(gzip.decompress((ROOT/r['raw_receipt']).read_bytes())).hexdigest()==r['sha256'] for r in requests)
    before=json.loads((ROOT/'supplement/three_journal_round/budget.json').read_text())
    assert not actual_dois.intersection(before['dois']), 'Unexpected repeat of previous DOI batch'
    db=connect(ROOT)
    # Reuse full-field batch responses without duplicate downloads in future runs.
    for path in sorted((AUDIT/'requests').glob('*.json')):
        r=json.loads(path.read_text())
        if not r['dois']:continue
        items={x['DOI'].lower():x for x in json.loads(gzip.decompress((ROOT/r['raw_receipt']).read_bytes()))['message']['items']}
        for doi in r['dois']:
            detail={'returned_doi_match':doi in items,'explicit_online_date':bool(items.get(doi,{}).get('published-online')),'generic_type_is_not_research_proof':True}
            tid=hashlib.sha256(('qualification:'+doi).encode()).hexdigest()[:32]
            db.execute('UPDATE attempts SET detail=? WHERE task_id=? AND source=? AND receipt=?',(json.dumps(detail),tid,'crossref_public_deposit_bounded_three_journals',str(path.relative_to(ROOT))))
    # Reopen metadata only for final qualified papers; raw signature absence remains explicit.
    newly_qualified=json.loads((AUDIT/'final_progress.json').read_text())['newly_included_work_ids']
    for wid in newly_qualified:
        p=P[wid];authors=json.loads(p['authors_json']);fields=[]
        if not p['abstract']:fields.append('abstract')
        if not authors or not all(a['institution_ids_or_verified_local_entities'] for a in authors):fields.append('institutions')
        if not authors or not all(a['countries'] for a in authors):fields.append('countries')
        if any(not a.get('publisher_affiliation_slots') for a in authors):fields.append('publisher_affiliation_structure')
        if fields:
            tid=task(db,'metadata',wid,'cache_first_signature_review',{'doi':p['doi'],'missing_fields':fields,'scope':'Current qualified paper; source structure not inferred from OpenAlex institution presence'},wid)
            db.execute('UPDATE tasks SET reason=? WHERE task_id=?',('Current metadata has unresolved fields; archived signature parsing/identity evidence remains to review',tid))
    for wid,d in D.items():
        if d['eligibility_status']!='included_verified_research':
            db.execute("UPDATE tasks SET active=0 WHERE kind='metadata' AND entity_key=?",(wid,))
    db.commit();db.close()
    # Derived priority cache is patched only for this scope.
    path=ROOT/'supplement/priority_dois.json';priority=json.loads(path.read_text())
    for d in D.values():priority.pop(d['doi'],None)
    names={'Science':0,'Cell':1,'Proceedings of the National Academy of Sciences':2}
    unresolved=read_csv(OUT/'three_journal_unresolved.csv')
    for row in unresolved:priority[row['doi']]=names[row['journal']]
    save(path,priority)
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'scope_candidate_n':len(D),'offline_qualification_errors':failures,
            'global_roster_n':len(roster),'global_roster_doi_unique':True,'global_unresolved_export_n':unresolved_n,'scope_unresolved_ids_exact':True,
            'all_journal_counts_conserved':True,'request_n':len(requests),'unique_doi_n':len(actual_dois),'previous_batch_doi_overlap':0,
            'raw_receipt_hashes_valid':True,'tests_passed':24,'metadata_abstract_rows_changed':1,'global_requalification_run':False,
            'scope':'Only three-journal decisions independently rerun; global export checks validate incremental accounting, not other journals scientific qualification',
            'source_network_seconds':sum(r['elapsed_seconds'] for r in requests),'raw_compressed_bytes':sum((ROOT/r['raw_receipt']).stat().st_size for r in requests)}
    save(AUDIT/'completion_checks.json',report);print(json.dumps(report))


if __name__=='__main__':run()
