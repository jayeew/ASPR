"""Combine incremental country repairs without double-counting papers or closing gaps."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(folder: Path) -> dict:
    scope=json.loads((folder/'scope.json').read_text())
    expected=Counter((p['work_id'],s['receipt']) for p in scope['papers'] for s in p['slots'])
    filled=Counter();outcomes=defaultdict(list);papers=set();sources=[]
    db=connect()
    for path in sorted(folder.rglob('application.json')):
        record=json.loads(path.read_text())
        if record.get('status')!='cache_application_complete':continue
        sources.append(str(path.relative_to(ROOT)))
        for row in record['audit']:
            pair=(row['work_id'],row['receipt']);filled[pair]+=row.get('slots_filled',0)
            outcomes[pair].append({**row,'application_receipt':str(path.relative_to(ROOT))})
        rule=str(path.with_name('scope.json').relative_to(ROOT))
        papers.update(w for w, in db.execute("SELECT DISTINCT work_id FROM changes WHERE field='authors_json' AND receipt=?",(rule,)))
    db.close()
    assert all(filled[pair]<=count for pair,count in expected.items())
    assert not set(filled)-set(expected)
    remaining=[{'work_id':w,'source_receipt':r,'scope_slots':n,'slots_filled_by_this_rule':filled[(w,r)],
        'not_filled_by_this_rule':n-filled[(w,r)],'attempts':outcomes[(w,r)],'requires_current_slot_review':True}
        for (w,r),n in expected.items() if n>filled[(w,r)]]
    result={'scope_papers':len(scope['papers']),'scope_country_slots':sum(expected.values()),
        'unique_papers_changed':len(papers),'country_slots_filled':sum(filled.values()),
        'remaining_pairs_for_current_slot_review':len(remaining),
        'remaining_scope_slots_not_filled_by_this_rule':sum(r['not_filled_by_this_rule'] for r in remaining),
        'remaining':remaining,'application_receipts':sources,'network_requests':0,'final_task_closure':False}
    atomic_json(folder/'combined_application_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'remaining','application_receipts'}}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();run(args.folder)
