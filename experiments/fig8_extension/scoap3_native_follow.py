"""Consume the fixed native XML queue in batches without repeating evidence work."""
from __future__ import annotations
import fcntl
import json
import time
from .scoap3_native_collect import FOLDER
from .native_xml_cache_repair import run as repair_cached_xml
from .scoap3_native_apply import run as apply_qualification
from .scoap3_elsevier_apply import run as apply_elsevier
from .springer_author_cache import run as apply_authors
from .batch_memory import release_batch_memory
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER/'consumer.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (FOLDER/'author_application.json').exists():
            release_batch_memory()
            time.sleep(30)
        while True:
            repair_cached_xml()
            done=json.loads((FOLDER/'applied_receipts.json').read_text())
            db=connect()
            other_done={r[0].rsplit('/',1)[-1] for r in db.execute("SELECT receipt FROM supplied_publisher_processed WHERE receipt LIKE 'supplement/scoap3_native/elsevier/staged/%'")}
            db.close()
            pending=[]
            for path in (FOLDER/'receipts').glob('*.json'):
                r=json.loads(path.read_text())
                if ((r.get('root_tag')=='Publisher' and str(path.relative_to(ROOT)) not in done) or (r.get('root_tag')=='{http://www.elsevier.com/xml/ja/dtd}article' and path.name not in other_done)):
                    pending.append(path)
            state=json.loads((FOLDER/'status.json').read_text())['status']
            terminal=state not in {'running','draining_inflight'}
            if len(pending)>=250 or (terminal and pending):
                qualification=apply_qualification();other=apply_elsevier();db=connect();jobs=[]
                repaired = {p.name: p for p in (FOLDER/'equation_repair/staged').glob('*.json')}
                originals = [repaired.get(p.name, p) for p in (FOLDER/'staged').glob('*.json')]
                for path in [*originals, *(FOLDER/'elsevier/staged').glob('*.json')]:
                    r=json.loads(path.read_text())
                    for wid,shard in db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'",(r['doi'],)):
                        jobs.append((wid,shard,str(path.relative_to(ROOT))))
                db.close();authors=apply_authors(selected_jobs=jobs,workers=4,fill_abstract=True)
                atomic_json(FOLDER/'consumer_status.json',{'status':'batch_processed','qualification':qualification,'authors':authors})
                if not qualification.get('selected_receipts') and not other.get('applied_source_records'):
                    atomic_json(FOLDER/'consumer_status.json',{'status':'remaining_native_evidence_requires_review','pending':len(pending)})
                    return
                continue
            if terminal:
                atomic_json(FOLDER/'consumer_status.json',{'status':'available_supported_xml_processed_other_schemas_and_failures_retained','source_status':state})
                return
            release_batch_memory()
            time.sleep(30)


if __name__=='__main__':run()
