"""Consume fixed new PMC identifiers found in cached alternative full-text URLs."""
from __future__ import annotations

import fcntl
import json
import re
import sqlite3
from collections import Counter, defaultdict
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cached_secondary_dispatch import archive_pmcid
from .pmc_cloud_inventory import FOLDER as CLOUD
from .pmc_cloud_versions import STOP
from .pmc_cloud_xml import apply_batch
from .priority_cloud_leads import acquire
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER=ROOT/'supplement/secondary_link_pmc'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER/'pipeline.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope=FOLDER/'scope.json'
        if not scope.exists():
            links=sqlite3.connect(ROOT/'supplement/cached_secondary_publication_links/leads.sqlite');db=connect()
            groups=defaultdict(dict)
            for wid,url in links.execute("SELECT work_id,url FROM dispatch WHERE status='archive_metadata_not_yet_requested'"):
                row=db.execute('SELECT doi,title,journal,status FROM papers WHERE work_id=?',(wid,)).fetchone()
                if row is None or not row[3].startswith('pending'):continue
                pmcid=archive_pmcid(url,urlsplit(url).hostname or '')
                if not pmcid:continue
                groups[pmcid][wid]={'work_id':wid,'doi':row[0],'title':row[1],'journal':row[2],
                    'pmcid':pmcid,'cached_link':url,'discovery_receipt':'supplement/cached_secondary_publication_links/leads.sqlite'}
            db.close();links.close()
            atomic_json(scope,{'jobs':{pmc:list(papers.values()) for pmc,papers in groups.items()},
                'scope':'Existing pending Work identities only; cached exact DOI/title secondary links; identifier is a lead, not qualification'})
        jobs=sorted(json.loads(scope.read_text())['jobs'].items(),key=lambda item:min(
            {'Science':0,'Cell':1,'Proceedings of the National Academy of Sciences':2}.get(p['journal'],3) for p in item[1]))
        counts=Counter();handled=[]
        with (CLOUD/'version_collector.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with ThreadPoolExecutor(max_workers=64) as pool:
                futures={pool.submit(acquire,item):item[0] for item in jobs}
                for future in as_completed(futures):
                    result=future.result();counts[result['status']]+=1
                    path=CLOUD/'version_results'/(futures[future]+'.1.json')
                    if path.exists():handled.append(path)
                    if sum(counts.values())%100==0:
                        atomic_json(FOLDER/'status.json',{'status':'collecting_fixed_metadata','fixed_identifiers':len(jobs),
                            'processed':sum(counts.values()),'outcomes':dict(counts),'workers':64,'final_task_closure':False})
        atomic_json(FOLDER/'status.json',{'status':'applying_fixed_xml_evidence','fixed_identifiers':len(jobs),
            'metadata_processed':len(handled),'metadata_outcomes':dict(counts),'final_task_closure':False})
        with (CLOUD/'xml_collector.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            done_path=CLOUD/'xml_consumed.json'
            done=set(json.loads(done_path.read_text())) if done_path.exists() else set()
            pending=[p for p in handled if str(p.relative_to(ROOT)) not in done]
            for start in range(0,len(pending),2000):
                batch=pending[start:start+2000];done.update(apply_batch(batch));atomic_json(done_path,sorted(done))
                atomic_json(FOLDER/'status.json',{'status':'applying_fixed_xml_evidence','fixed_identifiers':len(jobs),
                    'metadata_processed':len(handled),'xml_version_paths_consumed_this_run':min(start+2000,len(pending)),
                    'metadata_outcomes':dict(counts),'final_task_closure':False})
                if STOP.is_set():break
        atomic_json(FOLDER/'status.json',{'status':'source_stopped_unresolved_retained' if STOP.is_set() else 'fixed_metadata_and_available_xml_processed',
            'fixed_identifiers':len(jobs),'metadata_processed':len(handled),'metadata_outcomes':dict(counts),
            'final_task_closure':False,'full_refresh':'scheduled_only'})


if __name__ == '__main__':
    run()
