"""Reuse numeric PMC URL outcomes and dispatch only genuinely missing metadata."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .cached_secondary_dispatch import archive_pmcid
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/numeric_pmc_links'; folder.mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'supplement/cached_secondary_publication_links/leads.sqlite')
    current = connect(); counts = Counter(); jobs = {}; records = []
    rows = db.execute("SELECT work_id,url,status,evidence FROM dispatch WHERE status='source_path_review_required' AND url LIKE 'https://www.ncbi.nlm.nih.gov/pmc/articles/%'").fetchall()
    for wid, url, before_status, before_evidence in rows:
        pmc = archive_pmcid(url, urlsplit(url).hostname or '')
        if not pmc: continue
        paper = current.execute('SELECT doi,title,journal,status FROM papers WHERE work_id=?', (wid,)).fetchone()
        if not paper: continue
        doi, title, journal, qualification = paper
        vp = ROOT / 'supplement/pmc_cloud/version_results' / (pmc + '.1.json')
        xp = ROOT / 'supplement/pmc_cloud/xml_results' / (pmc + '.1.json')
        v = json.loads(vp.read_text()) if vp.exists() else {}
        status = 'archive_association_review_required'
        proof = {'pmcid': pmc, 'url': url, 'qualification_changed': False, 'final_task_closure': False}
        if any(p['work_id'] == wid and p['doi'] == doi for p in v.get('paper_links', [])):
            status = 'existing_archive_metadata_outcome_reused'
            proof.update(version_result=str(vp.relative_to(ROOT)), version_status=v['status'])
            if xp.exists():
                x = json.loads(xp.read_text()); status = 'existing_archive_xml_outcome_reused'
                proof.update(xml_result=str(xp.relative_to(ROOT)), xml_status=x['status'])
                if x.get('parsed_receipt'):
                    applied = current.execute('SELECT 1 FROM supplied_publisher_processed WHERE doi=? AND receipt=?', (doi,x['parsed_receipt'])).fetchone()
                    proof.update(parsed_receipt=x['parsed_receipt'], already_applied=bool(applied))
                    if not applied: status = 'existing_parsed_archive_application_review'
        elif not qualification.startswith('pending'):
            status = 'qualification_decided_no_new_acquisition'
            proof['current_qualification'] = qualification
        elif not v:
            status = 'archive_metadata_not_yet_requested'
            jobs.setdefault(pmc, []).append({'work_id':wid,'doi':doi,'title':title,'journal':journal,'pmcid':pmc,'cached_link':url})
        db.execute('INSERT INTO dispatch_changes VALUES(?,?,?,?,?,?)', (wid,url,
                   json.dumps({'status':before_status,'evidence':json.loads(before_evidence)}),
                   json.dumps({'status':status,'evidence':proof}),
                   'Numeric PMC path normalized; exact cached Work/DOI association checked', datetime.now(timezone.utc).isoformat()))
        db.execute('UPDATE dispatch SET status=?,evidence=? WHERE work_id=? AND url=?',(status,json.dumps(proof),wid,url))
        records.append({'work_id':wid, 'status':status, **proof}); counts[status] += 1
    scope = folder / 'scope.json'
    if not scope.exists(): atomic_json(scope, {'jobs':jobs,'scope':'Only fixed current pending Works with numeric PMC URL and no cached version result'})
    atomic_json(folder / 'dispatch_review.json', {'records':records,'outcomes':dict(counts),'network_requests':0})
    db.commit(); db.close(); current.close(); print(json.dumps(dict(counts)),flush=True)


if __name__ == '__main__':
    run()
