"""Avoid refetching a known typed/dated PMC document for a journal-format policy gap."""
from __future__ import annotations

import fcntl
import json
import re

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder=ROOT/'supplement/secondary_link_pmc';cloud=ROOT/'supplement/pmc_cloud'
    jobs=json.loads((folder/'scope.json').read_text())['jobs'];db=connect();changed=0;already=0
    with (cloud/'version_collector.lock').open('a') as lock, (folder/'existing_primary_reuse_changes.jsonl').open('a') as audit:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for pmcid,papers in jobs.items():
            if len(papers)!=1:continue
            paper=papers[0]
            if (cloud/'xml_parsed'/(pmcid+'.1.json')).exists() or (cloud/'xml_results'/(pmcid+'.1.json')).exists():continue
            row=db.execute('SELECT status FROM papers WHERE work_id=?',(paper['work_id'],)).fetchone()
            if row is None or row[0]!='pending_special_subtype_review':continue
            matches=[]
            for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",(paper['work_id'],)):
                e=json.loads(payload)
                if (e.get('family')=='archived_jats' and e.get('identity')=='matched'
                        and e.get('class')=='subtype_review_required' and e.get('kind') and e.get('online_date')
                        and re.search(r'\b'+re.escape(pmcid)+r'\b',e.get('url',''))
                        and (ROOT/'fields/qualification'/e['receipt']).is_file()):matches.append(e)
            if not matches:continue
            path=cloud/'version_results'/(pmcid+'.1.json')
            if not path.exists():continue
            record=json.loads(path.read_text());updated=False
            for assessment in record.get('assessments',[]):
                if assessment['work_id']!=paper['work_id'] or assessment['doi']!=paper['doi']:continue
                if assessment['status']=='existing_same_pmc_primary_evidence_reused':already+=1;continue
                if assessment['status']!='published_xml_candidate_identity_check_required':continue
                before=dict(assessment)
                assessment.update(status='existing_same_pmc_primary_evidence_reused',
                    original_metadata_assessment=before['status'],
                    existing_primary_evidence_ids=[e['id'] for e in matches],
                    existing_primary_receipts=[e['receipt'] for e in matches],
                    acquisition_reason='Same PMC identifier already has matched primary XML type/date; unresolved task is journal subtype policy, not another copy of XML',
                    next_path='journal_specific_type_rule_review',qualification_implied=False)
                audit.write(json.dumps({'receipt':str(path.relative_to(ROOT)),'work_id':paper['work_id'],
                    'before':before,'after':assessment,'raw_metadata_unchanged':True,'qualification_changed':False})+'\n')
                audit.flush();updated=True;changed+=1
            if updated:atomic_json(path,record)
    db.close();atomic_json(folder/'existing_primary_reuse_status.json',{'reused_current_primary_paths':changed,
        'already_reused':already,'network_requests':0,'qualification_changes':0,
        'limitation':'Already-dispatched requests may still finish; no interrupted writes or fabricated source restriction',
        'final_task_closure':False})


if __name__ == '__main__':
    run()
