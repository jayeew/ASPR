"""Reuse prior archive and host outcomes for cached alternate links, without requests."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def archive_pmcid(url: str, host: str) -> str:
    """Recognize PMC identifiers only within known archive URL routes."""
    if host not in {'europepmc.org', 'www.ncbi.nlm.nih.gov', 'pmc.ncbi.nlm.nih.gov'}:
        return ''
    match = re.search(r'\bPMC(\d+)\b', url, re.I)
    if match:
        return 'PMC' + match[1]
    match = re.search(r'/pmc/articles/(\d+)(?:[/?#]|$)', url) if host == 'www.ncbi.nlm.nih.gov' else None
    return 'PMC' + match[1] if match else ''


def run() -> None:
    folder=ROOT/'supplement/cached_secondary_publication_links'
    db=sqlite3.connect(folder/'leads.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS dispatch(work_id TEXT,url TEXT,status TEXT,evidence TEXT,PRIMARY KEY(work_id,url))')
    current=sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro',uri=True)
    holds=json.loads((ROOT/'supplement/cached_primary_url_leads/host_hold_application.json').read_text())['hosts']
    additional_holds = ROOT / 'supplement/remaining_publisher_link_access/host_holds.json'
    if additional_holds.exists():
        holds.update(json.loads(additional_holds.read_text())['hosts'])
    hal_holds=ROOT/'supplement/hal_metadata/pdf_fronts/host_holds.json'
    if hal_holds.exists():
        for host in json.loads(hal_holds.read_text())['hosts']:
            holds[host]={'receipt':str(hal_holds.relative_to(ROOT)),'basis':'Exact host HTTP restriction retained; no alternate URL retry'}
    cache={};new_jobs={};stats=Counter()
    links=db.execute('SELECT l.work_id,l.url,l.host,r.doi FROM leads l JOIN reviewed r USING(work_id)').fetchall()
    for wid,url,host,doi in links:
        previous = db.execute('SELECT status,evidence FROM dispatch WHERE work_id=? AND url=?', (wid,url)).fetchone()
        if previous:
            # Dedicated reconciliation/review passes own existing dispositions.
            # Discovery must not reopen policy deferrals or erase source receipts.
            status, stored = previous
            evidence = json.loads(stored)
            stats[status] += 1
            if status == 'existing_parsed_archive_application_review' and evidence.get('parsed_receipt'):
                new_jobs[doi] = evidence['parsed_receipt']
            continue
        evidence={'qualification_changed':False,'final_task_closure':False}
        status='source_path_review_required'
        pmc=archive_pmcid(url,host)
        if pmc:
            identifier=pmc
            if identifier not in cache:
                version=ROOT/'supplement/pmc_cloud/version_results'/f'{identifier}.1.json'
                xml=ROOT/'supplement/pmc_cloud/xml_results'/f'{identifier}.1.json'
                cache[identifier]=(version, json.loads(version.read_text()) if version.exists() else {},
                                   xml,json.loads(xml.read_text()) if xml.exists() else {})
            version,v,xml,x=cache[identifier]
            associated=any(p.get('work_id')==wid and p.get('doi')==doi for p in v.get('paper_links',[]))
            if associated:
                evidence.update(version_result=str(version.relative_to(ROOT)),version_status=v.get('status'))
                status='existing_archive_metadata_outcome_reused'
                if x:
                    evidence.update(xml_result=str(xml.relative_to(ROOT)),xml_status=x.get('status'))
                    status='existing_archive_xml_outcome_reused'
                    parsed=x.get('parsed_receipt')
                    if parsed:
                        applied=current.execute('SELECT 1 FROM supplied_publisher_processed WHERE doi=? AND receipt=?',(doi,parsed)).fetchone()
                        evidence['parsed_receipt']=parsed;evidence['already_applied']=bool(applied)
                        if not applied:
                            status='existing_parsed_archive_application_review';new_jobs[doi]=parsed
            else:
                status='archive_association_review_required' if v else 'archive_metadata_not_yet_requested'
                evidence['pmcid']=identifier
        elif host in holds:
            status='existing_host_restriction_alternatives_pending';evidence.update(holds[host])
        db.execute('INSERT INTO dispatch VALUES(?,?,?,?)',(wid,url,status,json.dumps(evidence)))
        stats[status]+=1
    db.commit();db.close();current.close()
    atomic_json(folder/'archive_application_candidates.json',{'jobs':sorted(new_jobs.items()),'review_required':True})
    atomic_json(folder/'dispatch_status.json',{'fixed_links':len(links),'outcomes':dict(stats),
        'unapplied_cached_archive_candidates':len(new_jobs),'network_requests':0,'qualification_changes':0,'final_task_closure':False})


if __name__ == '__main__':
    run()
