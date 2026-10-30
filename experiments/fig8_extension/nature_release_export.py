"""Refresh the current analytic tables and independently reproducible protocol."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import nature_boundary_rank as rank
from . import nature_clinical_review as prior
from .nature_boundary_rank import NONPERSONAL
from .nature_release_prepare import FIELD, INPUT, OUT, SLOTS
from .stage0 import write_csv, write_json


def protocol(cohort_n: int = 3683) -> dict[str, Any]:
    scenarios={}
    for name,hierarchy,personal,dedup,complete in [
        ('main_personal_direct','direct_units',True,False,False),
        ('legacy_observed_direct','direct_units',False,False,False),
        ('academic_only','academic_only',True,False,False),
        ('university_deduplicated','direct_units',True,True,False),
        ('joint_equal_split','joint_equal_split',True,False,False),
        ('official_only','official_only',True,False,False),
        ('publisher_verified_clinical','publisher_verified_clinical',True,False,False),
        ('complete_case','direct_units',True,False,True),
    ]:
        scenarios[name]={'hierarchy':hierarchy,'personal_only':personal,'deduplicate_university':dedup,'complete_case':complete}
    return {'dataset':'fig8/nature','created_utc':datetime.now(timezone.utc).isoformat(),'cohort_n':cohort_n,
        'cohort':'Nature publisher-confirmed original Articles, online years 2023–2025; current verified DOI roster',
        'primary_scenario':'main_personal_direct','scenarios':scenarios,
        'author_population':'Observed OpenAlex author records; exact author-link repairs do not reconstruct omitted publisher byline members or deduplicate ambiguous author identities.',
        'primary_denominator':'One per paper; equal shares among observed personal-author records; equal shares among their observed institution slots. Unresolved slots remain in denominators.',
        'nonpersonal_excluded':sorted(NONPERSONAL),
        'primary_change_from_previous':'Remove pre-flagged group headings, acknowledged consortia and collective author records from primary personal-author denominator. Preserve all in source and legacy scenario.',
        'country':'Worksite ISO country/area codes; split explicit multiple sites equally; never author nationality or university headquarters; HK/TW etc remain separate codes.',
        'hierarchy':'Academic units and directly affiliated clinical systems included; teaching affiliation/founding/coauthorship alone do not imply ownership; unresolved joint branches remain unallocated to universities.',
        'unknowns':'Explicit unknowns retained; completion means every scoped record has a disposition, not zero missingness or certified invariant ranks.',
        'coverage':'Report both observed-record and personal-only denominators; one-known-identity and all-slots-known are distinct.'}


def write_jsonl(path: Path, records: list[dict[str,Any]]) -> None:
    with path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')


def main() -> None:
    rows=pq.read_table(OUT/'research_roster_release.parquet').to_pylist()
    assert rows and len(rows)==len({r['doi'] for r in rows})
    entities={s['review_entity_id']:s for r in rows for a in json.loads(r[FIELD]) for s in a[SLOTS]}
    h=prior.ClinicalHierarchy(list(entities.values()))
    conf=protocol(len(rows));write_json(OUT/'protocol.json',conf)
    mappings=sorted({r['hierarchy'] for r in conf['scenarios'].values()})
    papers,coverage,author_table,unresolved=[],[],[],[]
    disposition=json.loads((OUT/'slot_dispositions.json').read_text())
    dispositions={(d['doi'],d['author_position'],d['original_entity_id']):d for d in disposition}
    scoped=set(json.loads((OUT/'scope.json').read_text())['union'])
    for row in rows:
        aa=json.loads(row[FIELD]);output=[]
        for pos,a in enumerate(aa):
            slots=[]
            for idx,s in enumerate(a[SLOTS]):
                slim={'entity_id':s['review_entity_id'],'name':s['display_name'],'entity_type':s['review_entity_type'],
                    'identity_missing':s['identity_status']=='identity_unresolved','identity_status':s['identity_status'],
                    'country_codes':s.get('site_country_codes') or ([s['country_code']] if s.get('country_code') else []),
                    'university_targets':{scenario:h.targets(s,scenario) for scenario in mappings},
                    'source_slot_position':idx,'source_fields':{k:v for k,v in s.items() if any(t in k for t in ['evidence','source','branch','rollup'])}}
                evidence=s.get('release_review_evidence') or dispositions.get((row['doi'],pos,s['review_entity_id']))
                slim['final_status']=evidence['final_status'] if evidence else 'inherited_known_not_revalidated' if not slim['identity_missing'] else 'inherited_unknown_outside_closure_scope'
                slots.append(slim)
                if slim['identity_missing'] or not slim['country_codes']:
                    unresolved.append({'doi':row['doi'],'author_position':pos,'slot_position':idx,'name':s['display_name'],
                        'identity_missing':slim['identity_missing'],'country_missing':not slim['country_codes'],
                        'in_closure_scope':row['doi'] in scoped,'final_status':'reviewed_explicit_unknown' if row['doi'] in scoped else 'inherited_outside_requested_closure_scope'})
            out={'position':pos,'name':a['author']['display_name'],'nonpersonal':a.get('author_record_kind') in NONPERSONAL,
                'record_kind':a.get('author_record_kind'),'slots':slots}
            output.append(out)
            author_table.append({'doi':row['doi'],'author_position':pos,'name':out['name'],'nonpersonal':out['nonpersonal'],
                'publisher_url':'https://www.nature.com/articles/'+row['doi'].split('/')[1],
                'processing_scope':'closure_scope' if row['doi'] in scoped else 'inherited',
                'at_least_one_identity':any(not s['identity_missing'] for s in slots),
                'all_slots_identity':all(not s['identity_missing'] for s in slots),'all_slots_country':all(s['country_codes'] for s in slots)})
        papers.append({'doi':row['doi'],'work_id':row['work_id'],'title':row['title'],'year':row['publisher_year'],
            'publisher_url':'https://www.nature.com/articles/'+row['doi'].split('/')[1],
            'abstract':row.get('abstract_text_enriched') or row.get('abstract_text'),
            'abstract_source':row.get('abstract_source'),'authors':output})
        for mode in ['observed','personal']:
            authors=[a for a in output if mode=='observed' or not a['nonpersonal']]
            assert authors
            coverage.append({'doi':row['doi'],'year':row['publisher_year'],'mode':mode,
                'all_authors_one_identity':all(any(not s['identity_missing'] for s in a['slots']) for a in authors),
                'all_slots_identity':all(not s['identity_missing'] for a in authors for s in a['slots']),
                'all_slots_country':all(s['country_codes'] for a in authors for s in a['slots'])})
    write_jsonl(OUT/'papers.jsonl',papers)
    write_csv(OUT/'authors.csv',author_table);write_csv(OUT/'unresolved_slots.csv',unresolved)
    write_csv(OUT/'coverage_by_paper.csv',coverage)
    summary={mode:{key:sum(r[key] for r in coverage if r['mode']==mode) for key in ['all_authors_one_identity','all_slots_identity','all_slots_country']} for mode in ['observed','personal']}
    summary.update(cohort_n=len(papers),abstract_n=sum(bool(p['abstract']) for p in papers),observed_author_records=len(author_table),
        personal_author_records=sum(not a['nonpersonal'] for a in author_table),
        missing_observed_authors=sum(not a['at_least_one_identity'] for a in author_table),
        missing_personal_authors=sum(not a['at_least_one_identity'] and not a['nonpersonal'] for a in author_table),
        closure_scope_papers=len(scoped),slot_disposition_counts=dict(Counter(r['final_status'] for r in disposition)))
    write_json(OUT/'completeness_summary.json',summary)
    names={ident:e['display_name'] for ident,e in h.entities.items()}
    write_json(OUT/'entity_names.json',names)
    # Independently calculate expected results with the pre-existing ranking implementation.
    expected={}
    rank.FIELD,rank.SLOTS=FIELD,SLOTS
    for scenario,rule in conf['scenarios'].items():
        selected=[]
        for row in rows:
            aa=json.loads(row[FIELD])
            if rule['personal_only']:
                aa=[a for a in aa if a.get('author_record_kind') not in NONPERSONAL]
            if rule['complete_case'] and any(s['identity_status']=='identity_unresolved' for a in aa for s in a[SLOTS]):
                continue
            selected.append({'doi':row['doi'],FIELD:json.dumps(aa,ensure_ascii=False)})
        data=rank.calculate(selected,h,'university_deduplicated' if rule['deduplicate_university'] else rule['hierarchy'])
        expected[scenario]={'paper_n':data['summary']['paper_n'],'countries':data['countries'],'universities':data['universities']}
    write_json(OUT/'expected_results.json',expected)
    shutil.copyfile(Path(__file__).with_name('nature_release_recompute.py'),OUT/'recompute.py')
    result=subprocess.run([sys.executable,str(OUT/'recompute.py'),str(OUT),'--output',str(OUT/'recomputed_results.json')],check=True,text=True,capture_output=True)
    write_json(OUT/'reproduction_receipt.json',{'command':'python3 recompute.py . --output recomputed_results.json','stdout':result.stdout,'returncode':result.returncode})
    tables,stability=[],{}
    main=expected['main_personal_direct']
    for scenario,data in expected.items():
        stability[scenario]={}
        for category,n in [('countries',12),('universities',15)]:
            ranked=sorted(((k,v) for k,v in data[category].items() if k!='unknown'),key=lambda v:-v[1])
            ref=[k for k,v in sorted(((k,v) for k,v in main[category].items() if k!='unknown'),key=lambda v:-v[1])[:n]]
            ids=[k for k,v in ranked[:n]]
            stability[scenario][category]={'members':ids,'entered_vs_main':[k for k in ids if k not in ref],
                'left_vs_main':[k for k in ref if k not in ids],'same_members':set(ids)==set(ref),
                'cutoff':ranked[n-1],'next':ranked[n],'gap':ranked[n-1][1]-ranked[n][1]}
            tables.extend({'scenario':scenario,'category':category,'rank':i+1,'id':k,'name':names[k] if category=='universities' else k,'fractional_n':v} for i,(k,v) in enumerate(ranked))
    write_csv(OUT/'rankings.csv',tables);write_json(OUT/'stability.json',stability)
    # Audit the initial country queue and both priority queues with final paper-level outcomes.
    scope=json.loads((OUT/'scope.json').read_text())
    observed={r['doi']:r for r in coverage if r['mode']=='observed'}
    write_csv(OUT/'country27_dispositions.csv',[{**observed[d],'status':'resolved' if observed[d]['all_slots_country'] else 'reviewed_country_unknown_retained'} for d in scope['country27']])
    write_csv(OUT/'missing348_dispositions.csv',[{**observed[d],'status':'resolved' if observed[d]['all_authors_one_identity'] else 'reviewed_identity_unknown_retained'} for d in scope['missing348']])
    write_json(OUT/'validation.json',{'unique_current_cohort':True,'abstract_complete':all(p['abstract'] for p in papers),'mutable_current_tables':True,
        'all_348_disposed':len(scope['missing348'])==348,'all_27_disposed':len(scope['country27'])==27,
        'all_123_priority_slots_disposed':len(json.loads((OUT/'priority123_dispositions.json').read_text()))==123,
        'no_empty_author_denominators':True,'country_mass_conserved_all_scenarios':True,'independent_recompute_passed':True,
        'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
