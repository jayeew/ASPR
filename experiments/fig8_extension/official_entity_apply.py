"""Apply reviewed exact unit identities without assigning headquarters countries."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

CERN='https://openalex.org/I67311998'
RECEIPT='supplement/official_entities/cern_rule.json'


def resolve(author: dict, rule: dict) -> tuple[dict,int]:
    result=json.loads(json.dumps(author));count=0
    if author.get('nature_v1_slots'):return result,count
    aliases={key(v) for v in rule['exact_names']}
    for slot in result.get('publisher_affiliation_slots',[]) or []:
        if key(slot.get('text','')) not in aliases:continue
        if set(slot.get('institution_ids',[]))-{rule['entity_id']}:continue
        before=json.dumps(slot,sort_keys=True)
        slot.update(institution_ids=[rule['entity_id']],identity_status='official_exact_declared_name',
                    entity_type='research_organization',university_ownership_status='officially_confirmed_nonuniversity',
                    official_identity_evidence=RECEIPT)
        # Keep paper-signature country values as-is. CERN spans multiple countries.
        if before!=json.dumps(slot,sort_keys=True):count+=1
    if count and CERN not in result['institution_ids_or_verified_local_entities']:
        result['institution_ids_or_verified_local_entities'].append(CERN)
        result['identity_source']='official_exact_publisher_unit_with_retained_existing_entities'
    return result,count


def run() -> None:
    rule=json.loads((ROOT/RECEIPT).read_text());stats=Counter()
    if rule['status']!='reviewed':raise ValueError('Unreviewed rule')
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(ROOT)
        db.execute('INSERT INTO identities(entity_id,name,entity_type,countries,evidence,ownership_status,university_targets) VALUES(?,?,?,?,?,?,?) ON CONFLICT(entity_id) DO UPDATE SET entity_type=excluded.entity_type,evidence=excluded.evidence,ownership_status=excluded.ownership_status,university_targets=excluded.university_targets',
            (CERN,'European Organization for Nuclear Research','research_organization','[]',json.dumps({'receipt':RECEIPT,'official_url':rule['official_source']}),'officially_confirmed_nonuniversity','{}'))
        affected=defaultdict(set)
        for name in rule['exact_names']:
            for wid,shard in db.execute('SELECT DISTINCT p.work_id,p.shard FROM tasks t JOIN task_papers l USING(task_id) JOIN papers p ON p.work_id=l.work_id WHERE t.kind=? AND t.entity_key=?',('institution',' '.join(name.casefold().split()))):
                affected[shard].add(wid)
        for shard,ids in affected.items():
            path=ROOT/'fields/papers'/shard;table=pq.read_table(path);rows=table.to_pylist();changed=False
            for paper in rows:
                if paper['work_id'] not in ids:continue
                before=paper['authors_json'];authors=[];delta=0
                for author in json.loads(before):
                    new,n=resolve(author,rule);authors.append(new);delta+=n
                if not delta:continue
                after=json.dumps(authors,ensure_ascii=False);paper['authors_json']=after;changed=True
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                    (paper['work_id'],'authors_json',before,after,RECEIPT,'Official exact institution name; non-university research organization; no country or headquarters inference'))
                values=metrics(paper)
                for metric,value in values.items():
                    field='after_'+metric
                    if field not in paper or paper[field]==value:continue
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                        (paper['work_id'],field,json.dumps(paper[field]),json.dumps(value),RECEIPT,'Recompute observed metadata after official identity repair'))
                    paper[field]=value
                db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                    (not values['every_author_institution'],not values['every_author_country'],paper['work_id']))
                stats['changed_papers']+=1;stats['resolved_primary_slots']+=delta
            if changed:
                temporary=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),temporary,compression='zstd');temporary.replace(path)
            db.commit()
        # Country/site gaps may remain: do not close the institution acquisition task globally.
        for name in rule['exact_names']:
            db.execute("UPDATE tasks SET reason=?,receipt=? WHERE kind='institution' AND entity_key=?",('Exact identity and non-university type confirmed; signature worksite country still unresolved where absent',RECEIPT,' '.join(name.casefold().split())))
        db.commit();db.close()
    atomic_json(ROOT/'supplement/official_entities/cern_apply_status.json',dict(stats))
    print(json.dumps(dict(stats)),flush=True)


if __name__=='__main__':run()
