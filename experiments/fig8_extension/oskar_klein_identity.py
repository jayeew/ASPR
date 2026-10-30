"""Confirm the joint Oskar Klein Centre without inferring a university allocation."""
from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

NAMES = ['oskar klein centre', 'oskar klein centre, stockholm sweden',
         'oskar klein centre, stockholm, sweden',
         'the oskar klein centre for cosmoparticle physics, albanova university centre, se-10691, stockholm, sweden',
         'the oskar klein centre, albanova university center, roslagstullsbacken 21, 106 91 stockholm, sweden']


def run() -> None:
    folder = ROOT/'supplement/official_entities/oskar_klein_identity';folder.mkdir(exist_ok=True)
    source = str((folder/'official_review.json').relative_to(ROOT));receipt = str((folder/'rules.json').relative_to(ROOT))
    url = 'https://www.su.se/english/divisions/oskar-klein-centre/about-the-okc'
    entity = 'reviewed-local:'+hashlib.sha256(url.encode()).hexdigest()[:20]
    proof = {'official_url':url,'dated_source':'https://www.su.se/english/news/articles/2023-01-25-creative-research-environment-attracts-physicists-and-astronomers-to-stockholm',
        'reviewed_at':datetime.now(timezone.utc).isoformat(),'access':'Web tool official university text; raw HTTP not retained',
        'findings':'Centre established jointly by Stockholm University and KTH in 2008; 2023 university account confirms shared research environment at AlbaNova. Current organization page places staff appointments within participating departments.',
        'limitation':'Bare centre signature does not identify the author employer. No sole university target or equal ownership fraction inferred.',
        'nature_placeholder':'unresolved:aa107d42d1fc5a252321',
        'nature_reopening':'New evidence available; protected Nature authors unchanged by this application and still require their existing reconciliation workflow.'}
    atomic_json(folder/'official_review.json',proof)
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect()
        old=db.execute('SELECT * FROM identities WHERE entity_id=?',(entity,)).fetchone()
        if old is None:
            evidence={'source':source,'official_url':url,'local_id_basis':'sha256 of reviewed official centre URL'}
            db.execute('INSERT INTO identities VALUES(?,?,?,?,?,?,?)',(entity,'Oskar Klein Centre','research_organization','[]',json.dumps(evidence),'unverified','{}'))
            atomic_json(folder/'entity_creation.json',{'entity_id':entity,'before':None,'after':{'name':'Oskar Klein Centre','entity_type':'research_organization','countries':[],'evidence':evidence,'ownership_status':'unverified','university_targets':{}},'reason':'No matching existing entity; official shared-centre identity confirmed. Country not inferred.'})
            db.commit()
        db.close()
    rule={'entity_id':entity,'name':'Oskar Klein Centre','exact_names':NAMES,'entity_type':'research_organization',
        'ownership_status':'unverified','university_targets':{},'source':source,'official_url':url,
        'locator':'About OKC, Organization; dated 2023 university news introduction',
        'confirmed_partners':['Stockholm University','KTH Royal Institute of Technology'],
        'scope':'Exact centre-only signatures. Explicit department/university compound signatures excluded; original country preserved; allocation unresolved.'}
    atomic_json(folder/'rules.json',{'status':'reviewed','rules':[rule]})
    db=connect();ids=set().union(*affected_papers(db,{n:rule for n in NAMES}).values())
    low=db.execute('SELECT coalesce(max(id),0) FROM changes').fetchone()[0];db.close()
    atomic_json(folder/'scope.json',{'selected_work_ids':sorted(ids),'exact_names':NAMES})
    result=apply_rules(rule_paths=[receipt],selected_work_ids=ids,status_name='oskar_klein_identity_status.json')
    checked=verify_changes(low,ids,[receipt]);assert checked==result.get('papers_changed',0)
    with (folder.parent/'rule_reuse.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=folder.parent/'current_rule_reuse_scope.json';state=json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:state['next_rule_paths'].append(receipt)
        atomic_json(path,state)
    db=connect();added=0
    for name in NAMES:
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='institution' AND entity_key=?",(name,)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,'oskar_klein_official_identity',receipt)).fetchone():continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(tid,'oskar_klein_official_identity','joint_centre_identity_confirmed_allocation_unknown',receipt,json.dumps({'exact_name':name,'final_task_closure':False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,));added+=1
    db.commit();db.close()
    atomic_json(folder/'validation.json',{'result':result,'changed_papers_reproduced':checked,'repeat_safe':True,
        'countries_and_protected_nature_preserved':True,'ledger_attempts_added':added,'university_allocation_unknown':True})
    if checked:refresh()


if __name__ == '__main__':run()
