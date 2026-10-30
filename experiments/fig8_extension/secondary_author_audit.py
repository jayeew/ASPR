"""Audit indexed secondary statements without promoting them to publisher verification."""
from __future__ import annotations

import csv
import fcntl
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_recompute import metrics
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json


def audit(root: Path = ROOT) -> dict:
    folder=root/'supplement';stats=Counter();groups={}
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db=connect(root)
        destination=folder/'secondary_author_evidence.csv';temporary=destination.with_suffix('.csv.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            columns=['work_id','doi','author_position','source_receipt','source_url','slot_count','known_country_slots','resolved_identity_slots','country_conflict','authority']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for path in sorted((root/'fields/papers').glob('*.parquet')):
                qualified={r['work_id'] for r in pq.read_table(root/'fields/qualification/decisions'/path.name,columns=['work_id','eligibility_status'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist()}
                for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
                    for paper in batch.to_pylist():
                        if paper['work_id'] not in qualified:continue
                        authors=json.loads(paper['authors_json']);selected=[a for a in authors if a.get('secondary_affiliation_slots')]
                        if not selected:continue
                        stats['papers_with_secondary_statements']+=1
                        group=groups.setdefault((paper['field'],paper['year_openalex'],paper['source_id']),Counter());group['papers']+=1
                        values=metrics(paper)
                        db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',
                            (not values['abstract'],not values['every_author_institution'],not values['every_author_country'],paper['work_id']))
                        for author in selected:
                            slots=author['secondary_affiliation_slots'];evidence=author['secondary_affiliation_evidence']
                            conflict=bool(author.get('secondary_country_conflict'));stats['authors_with_secondary_statements']+=1;group['authors']+=1
                            stats['secondary_country_conflicts']+=conflict;group['country_conflicts']+=conflict
                            stats['secondary_slots']+=len(slots);stats['secondary_known_country_slots']+=sum(bool(s['country_codes']) for s in slots)
                            writer.writerow({'work_id':paper['work_id'],'doi':paper['doi'],'author_position':author['position'],
                                'source_receipt':evidence['evidence_receipt'],'source_url':evidence['source_url'],'slot_count':len(slots),
                                'known_country_slots':sum(bool(s['country_codes']) for s in slots),
                                'resolved_identity_slots':sum(bool(s['institution_ids']) for s in slots),'country_conflict':conflict,'authority':evidence['authority']})
                            if conflict:
                                tid=task(db,'country_conflict',paper['work_id']+':'+str(author['position'])+':secondary','publisher_signature_then_official',
                                    {'countries':author['countries'],'secondary_countries':sorted({c for s in slots for c in s['country_codes']}),'evidence':evidence,
                                     'reason':'Exact secondary author statement differs from existing country assignment; no overwrite'},paper['work_id'],author['position'])
                                db.execute('UPDATE tasks SET receipt=?,reason=? WHERE task_id=?',(evidence['evidence_receipt'],'Secondary statement country conflict; publisher/official resolution required',tid))
                db.commit();print('Audited secondary statements',path.name,flush=True)
        temporary.replace(destination);db.close()
    result={'counts':dict(stats),'scope':'Current qualified papers; secondary affiliation statements are not publisher-verified complete bylines',
        'allocation_policy':'Current main country/university allocations do not use secondary-only slots; evidence retained for corroboration and explicit uncertainty',
        'groups':[{'field':k[0],'openalex_year':k[1],'source_id':k[2],**v} for k,v in sorted(groups.items())]}
    atomic_json(folder/'secondary_author_audit.json',result)
    return result


if __name__=='__main__':audit()
