"""Recompute coverage and ranking sensitivities from the cutoff review layer."""
from __future__ import annotations

import hashlib
import json

import pyarrow.parquet as pq

from . import nature_boundary_rank as rank
from . import nature_joint_branch_review as prior
from .nature_cutoff_review import FIELD, INPUT, OUT, PREV, SLOTS, CutoffHierarchy
from .nature_joint_branch_rank import SensitivityHierarchy
from .stage0 import write_csv, write_json


def main() -> None:
    rows=pq.read_table(OUT/'research_roster_cutoff_review.parquet').to_pylist()
    entities={s['review_entity_id']:s for r in rows for a in json.loads(r[FIELD]) for s in a[SLOTS]}
    for row in rows:
        for author in json.loads(row[prior.FIELD]):
            for slot in author[prior.SLOTS]:
                entities.setdefault(slot['review_entity_id'], slot)
    h=CutoffHierarchy(list(entities.values()))
    rank.FIELD,rank.SLOTS=prior.FIELD,prior.SLOTS
    results={'previous_main':rank.calculate(rows,SensitivityHierarchy(list(entities.values())),'direct_units')}
    rank.FIELD,rank.SLOTS=FIELD,SLOTS
    scenarios=['direct_units','academic_only','joint_equal_split','joint_two_partner_sensitivity',
               'university_deduplicated','exclude_nonpersonal','complete_case','official_only']
    for scenario in scenarios:
        results[scenario]=rank.calculate(rows,h,scenario)
    stability={s:rank.bounds(d,h,s) for s,d in results.items()}
    for record in stability.values():
        for category in ['country','university']:
            names,ref=record[category]['members'],stability['direct_units'][category]['members']
            record[category].update(same_members_as_direct=set(names)==set(ref),same_order_as_direct=names==ref,
                entered_vs_direct=[n for n in names if n not in ref],left_vs_direct=[n for n in ref if n not in names])
    write_json(OUT/'topk_stability.json',{'scenarios':stability,'limitations':
        'Missing-only bounds condition on fixed mappings and denominators; exclude hierarchy uncertainty, publisher conflicts and overseas campus exceptions. Not confidence intervals.'})
    write_json(OUT/'fractional_summary.json',{s:d['summary'] for s,d in results.items()})
    for category in ['countries','universities']:
        table=[]
        for scenario,result in results.items():
            for i,(key,value) in enumerate(sorted(((k,v) for k,v in result[category].items() if k!='unknown'),key=lambda x:-x[1])):
                table.append({'scenario':scenario,'rank':i+1,'id':key,'name':h.entities[key]['display_name'] if category=='universities' else key,'fractional_n':value})
        write_csv(OUT/(category+'_rank_scenarios.csv'),table)
    coverage,remaining,authors=[],[],[]
    for row in rows:
        aa,bb=json.loads(row[FIELD]),json.loads(row[prior.FIELD])
        assert all(a[k]==v for a,b in zip(aa,bb,strict=True) for k,v in b.items())
        records=[]
        for pos,(a,b) in enumerate(zip(aa,bb,strict=True)):
            slots=a[SLOTS]
            record={'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                'before_known':any(s['identity_status']!='identity_unresolved' for s in b[prior.SLOTS]),
                'after_known':any(s['identity_status']!='identity_unresolved' for s in slots),
                'standard_after':any(s['identity_status']=='standard_registry' for s in slots),
                'all_slots_known':all(s['identity_status']!='identity_unresolved' for s in slots),
                'all_slots_country':all(s.get('site_country_codes') or s.get('country_code') for s in slots),
                'remaining_slots':[s for s in slots if s['identity_status']=='identity_unresolved']}
            records.append(record)
            authors.append(record)
        values={k:all(a[k] for a in records) for k in ['after_known','standard_after','all_slots_known','all_slots_country']}
        coverage.append({'doi':row['doi'],'year':row['publisher_year'],**values})
        if not values['after_known']:
            remaining.append({'doi':row['doi'],'title':row['title'],'year':row['publisher_year'],
                              'unresolved_author_n':sum(not a['after_known'] for a in records)})
    changes=json.loads((OUT/'identity_changes.json').read_text())
    branches=json.loads((OUT/'branch_audit.json').read_text())
    oldbranch=[b for b in branches if b['from_previous91']]
    assert len(oldbranch)==91
    summary={'cohort_n':len(rows),'before_unresolved_papers':357,'before_unresolved_authors':1051,
        'remaining_identity_papers':len(remaining),'remaining_identity_authors':sum(not a['after_known'] for a in authors),
        'newly_resolved_authors':sum(not a['before_known'] and a['after_known'] for a in authors),
        'changed_identity_slots':len(changes),'changed_identity_papers':len({c['doi'] for c in changes}),
        'coverage':{k:sum(c[k] for c in coverage) for k in ['after_known','standard_after','all_slots_known','all_slots_country']},
        'previous91_confirmed_nonuniversity':sum(b['status']=='verified_nonuniversity_branch' for b in oldbranch),
        'previous91_still_unresolved':sum(b['status'] in ['unresolved_branch','publisher_assignment_conflict'] for b in oldbranch),
        'previous91_publisher_conflicts':sum(b['status']=='publisher_assignment_conflict' for b in oldbranch),
        'new_scope_confirmed_university_branches':sum(b['status']=='confirmed_branch' for b in branches if not b['from_previous91']),
        'new_scope_confirmed_nonuniversity_branches':sum(b['status']=='verified_nonuniversity_branch' for b in branches if not b['from_previous91']),
        'new_scope_unresolved_branches':sum(b['status']=='unresolved_branch' for b in branches if not b['from_previous91'])}
    write_json(OUT/'review_summary.json',summary)
    write_csv(OUT/'remaining_papers.csv',remaining)
    write_csv(OUT/'remaining_identity_authors.csv',[a for a in authors if not a['after_known']])
    write_json(OUT/'author_coverage_audit.json',authors)
    write_csv(OUT/'coverage_by_year.csv',[{'year':year,'paper_n':sum(c['year']==year for c in coverage),
        **{k:sum(c[k] for c in coverage if c['year']==year) for k in summary['coverage']}} for year in [2023,2024,2025]])
    write_csv(OUT/'unresolved_branches.csv',[b for b in branches if b['status'] in ['unresolved_branch','publisher_assignment_conflict']])
    write_csv(OUT/'publisher_assignment_conflicts.csv',[b for b in branches if b['status']=='publisher_assignment_conflict'])
    units=[]
    for u in sorted(results['direct_units']['units'].values(),key=lambda u:-u['weight']):
        units.append({**u,'dois':sorted(u['dois']),'reasons':sorted(u['reasons']),'paper_n':len(u['dois'])})
    write_json(OUT/'remaining_weighted_units.json',units)
    write_csv(OUT/'remaining_weighted_units.csv',units)
    decisions=[h.decision(i) for i in h.entities]
    write_json(OUT/'hierarchy_decisions.json',decisions)
    write_csv(OUT/'remaining_registry_hierarchy.csv',[d for d in decisions if d['evidence_level']=='registry_structural_screen' and d['target_ids']])
    for scenario in scenarios:
        if scenario not in ['complete_case','exclude_nonpersonal']:
            assert results[scenario]['countries']==results['previous_main']['countries']
    old=pq.read_table(INPUT).to_pylist()
    assert all(all(new[k]==v for k,v in before.items()) for new,before in zip(rows,old,strict=True))
    assert len({r['doi'] for r in rows})==3683
    oldmanifest=json.loads((PREV/'artifact_manifest.json').read_text())
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==oldmanifest['files'][INPUT.name]
    write_json(OUT/'validation.json',{'cohort_unique':3683,'country_allocations_unchanged':True,
        'prior_fields_and_roster_hash_unchanged':True,'mass_conserved':True})
    print(json.dumps(summary,indent=2))
    for s,v in stability.items():
        print(s,v['university']['cutoff'],v['university']['next'],v['university']['entered_vs_direct'])


if __name__=='__main__':
    main()
