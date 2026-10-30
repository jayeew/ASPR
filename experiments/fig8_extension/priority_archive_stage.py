"""Stage DOI-validated priority archive receipts for the shared incremental qualifier."""
from __future__ import annotations

import json
from collections import Counter

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def stage() -> dict:
    db=connect(ROOT);stats=Counter();q=ROOT/'fields/qualification'
    for pmc,doi,status,receipt in db.execute('SELECT pmcid,doi,status,receipt FROM priority_archive_jobs'):
        if status!='parsed':stats['unresolved_retained']+=1;continue
        record=json.loads((ROOT/receipt).read_text());p=record['parsed']
        if doi not in p['doi']:stats['doi_mismatch_retained']+=1;continue
        record={**record,'route':'pmc_archive_priority','parsed':{**p,'issns':p['issn']}}
        path=q/'supplement_parsed'/('priority_archive_'+pmc+'.json')
        if not path.exists():atomic_json(path,record)
        stats['new_incremental_jobs']+=db.execute('INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status,receipt,applied) VALUES(?,?,?,?,0)',(doi,'pmc_archive_priority','parsed',str(path.relative_to(ROOT)))).rowcount
        stats['cached_primary_records_staged']+=1
    db.commit();db.close();result=dict(stats);atomic_json(ROOT/'supplement/priority_archive/stage_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':stage()
