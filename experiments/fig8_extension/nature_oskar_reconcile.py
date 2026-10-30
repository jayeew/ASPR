"""Reconcile one reopened Nature identity from new dated official evidence."""
from __future__ import annotations

import fcntl
import json

import pyarrow as pa
import pyarrow.parquet as pq

from . import nature_release_export, nature_release_report
from .derived_refresh import refresh
from .fig8b_qualification_recompute import metrics
from .stage0 import write_csv
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

DOI = '10.1038/s41586-024-07824-z'
OLD = 'unresolved:aa107d42d1fc5a252321'


def run() -> None:
    folder = ROOT/'supplement/official_entities/oskar_klein_identity'
    nature = ROOT/'nature'; receipt = str((folder/'nature_reconciliation.json').relative_to(ROOT))
    rule = json.loads((folder/'rules.json').read_text())['rules'][0]
    final = 'unresolved_university_allocation_identity_confirmed'
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        baseline = json.loads((nature/'expected_results.json').read_text())
        table = pq.read_table(nature/'research_roster_release.parquet'); rows=table.to_pylist()
        row=next(r for r in rows if r['doi']==DOI); authors=json.loads(row['authorships_release_json'])
        assert authors[85]['author']['display_name']=='S. Y. Andrean'
        slot=authors[85]['institution_release_entities'][1];before=json.loads(json.dumps(slot))
        assert slot['review_entity_id'] in {OLD,rule['entity_id']}
        dispositions=json.loads((nature/'slot_dispositions.json').read_text())
        disposition=next(r for r in dispositions if r['doi']==DOI and r['author_position']==85 and r['slot_position']==1)
        old_disposition=json.loads(json.dumps(disposition))
        assert 'Oskar Klein Centre, Stockholm, Sweden' in disposition['publisher_addresses']
        disposition.update(final_status=final,final_identity_missing=False,
            new_official_identity_source=str((folder/'official_review.json').relative_to(ROOT)),
            university_allocation_status='joint_centre_partner_fraction_unresolved')
        slot.update(review_entity_id=rule['entity_id'],review_entity_type='research_organization',
            identity_status='official_exact_declared_name',
            release_source_evidence={'official_url':rule['official_url'],'receipt':rule['source'],
                                     'publisher_url':disposition['publisher_url']},
            release_review_evidence=disposition)
        if not (folder/'nature_reconciliation.json').exists():
            atomic_json(folder/'nature_reconciliation.json',{'doi':DOI,'author_position':85,'slot_position':1,
                'old_slot':before,'new_slot':slot,'old_disposition':old_disposition,'new_disposition':disposition,
                'reason':'Dated university evidence corroborates exact publisher centre signature; targets remain empty.'})
        row['authorships_release_json']=json.dumps(authors,ensure_ascii=False)
        path=nature/'research_roster_release.parquet';tmp=path.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),tmp,compression='zstd');tmp.replace(path)
        atomic_json(nature/'slot_dispositions.json',dispositions);write_csv(nature/'slot_dispositions.csv',dispositions)
        ad=json.loads((nature/'author_dispositions.json').read_text())
        for a in ad:
            if a['doi']==DOI and a['author_position']==85:
                a.update(all_slots_known=True,slot_statuses=[final],status='new_official_identity_evidence_applied')
        atomic_json(nature/'author_dispositions.json',ad);write_csv(nature/'author_dispositions.csv',ad)
        changes=json.loads((nature/'changes.json').read_text())
        if not any(c.get('reconciliation_receipt')==receipt for c in changes):
            changes.append({**disposition,'old_slot':before,'new_slot':slot,'reconciliation_receipt':receipt})
        atomic_json(nature/'changes.json',changes);write_csv(nature/'changes.csv',changes)
        nature_release_export.main()
        after=json.loads((nature/'expected_results.json').read_text())
        assert after==baseline, 'Unexpected allocation change requires review'
        paper=next(json.loads(line) for line in (nature/'papers.jsonl').open() if DOI in line)
        new_slot=paper['authors'][85]['slots'][1]
        assert not new_slot['identity_missing'] and not any(new_slot['university_targets'].values())
        db=connect();wid,shard=db.execute('SELECT work_id,shard FROM papers WHERE doi=?',(DOI,)).fetchone()
        master=ROOT/'fields/papers'/shard;mt=pq.read_table(master);mr=mt.to_pylist();updates=0
        for r in mr:
            if r['work_id']!=wid:continue
            old=r['authors_json'];aa=json.loads(old)
            matching=[a for a in aa if a['name']=='S. Y. Andrean' and a.get('nature_v1_slots') is not None]
            assert len(matching)==1
            a=matching[0];ss=[s for s in a['nature_v1_slots'] if s['entity_id'] in {OLD,rule['entity_id']}]
            assert len(ss)==1 and ss[0]['country_codes']==new_slot['country_codes']
            ss[0].clear();ss[0].update(new_slot)
            if rule['entity_id'] not in a['institution_ids_or_verified_local_entities']:
                a['institution_ids_or_verified_local_entities'].append(rule['entity_id'])
            r['authors_json']=json.dumps(aa,ensure_ascii=False)
            if old!=r['authors_json']:
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                    (wid,'authors_json',old,r['authors_json'],receipt,'New official evidence: synchronize reopened Nature centre identity; no country or university target change'))
                updates+=1
            values=metrics(r)
            for k,v in values.items():
                if 'after_'+k in r:r['after_'+k]=v
            db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                (not values['every_author_institution'],not values['every_author_country'],wid))
        tmp=master.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(mr,schema=mt.schema),tmp,compression='zstd');tmp.replace(master)
        db.commit();db.close()
        nature_release_report.main()
        queue=json.loads((folder/'nature_reopen_queue.json').read_text())
        for r in queue['records']:r['status']='identity_reconciled_university_allocation_still_unresolved'
        queue['protected_records_modified']=True;queue['modification_basis']='New official source and exact publisher signature, scoped reconciliation'
        atomic_json(folder/'nature_reopen_queue.json',queue)
        atomic_json(folder/'nature_reconciliation_validation.json',{'nature_slots_updated':1,'unified_papers_updated':updates,
            'all_eight_scenario_allocations_unchanged':True,'country_and_university_targets_unchanged':True,
            'nature_export_and_offline_recompute_passed':True,'remaining_paper_identity_slots':sum(s['identity_missing'] for a in paper['authors'] for s in a['slots']),
            'university_allocation_still_unknown':True})
    refresh()


if __name__=='__main__':run()
