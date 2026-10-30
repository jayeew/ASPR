"""Defer redundant qualification fetches when explicit cached subtype/date need policy."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from .fig8b_qualification_decide import SPECIAL, key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cached_secondary_publication_links'
    links = sqlite3.connect(folder/'leads.sqlite'); db = connect()
    rows = links.execute("SELECT work_id,url,status,evidence FROM dispatch WHERE status='source_path_review_required'").fetchall()
    cache = {}; reviews = []; counts = Counter()
    for wid,url,old_status,old_evidence in rows:
        if wid not in cache:
            paper = db.execute('SELECT doi,status FROM papers WHERE work_id=?',(wid,)).fetchone()
            items = [json.loads(v[0]) for v in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",(wid,))] if paper and paper[1]=='pending_special_subtype_review' else []
            matched = [v for v in items if v.get('identity')=='matched']
            explicit = [v for v in matched if key(v.get('kind')) in SPECIAL and v.get('online_date') and v.get('receipt') and (ROOT/'fields/qualification'/v['receipt']).is_file()]
            valid = (explicit and len({key(v['kind']) for v in explicit})==1
                     and len({v['online_date'][:4] for v in matched if v.get('online_date')})==1
                     and not any(v.get('scope_review_required') or v.get('title_scope_flag') or v.get('class') in {'research','nonprimary','integrity_hold'} for v in matched))
            cache[wid] = (paper,explicit if valid else [])
        paper,explicit = cache[wid]
        if not explicit:continue
        status='deferred_cached_subtype_requires_policy'
        detail={'work_id':wid,'doi':paper[0],'url':url,'status':status,
                'evidence':[{'id':v['id'],'kind':v['kind'],'online_date':v['online_date'],'receipt':v['receipt']} for v in explicit],
                'reason':'Existing explicit subtype and consistent online year; settle journal/type inclusion policy before fetching another copy for the same fields',
                'next_action':'Journal-specific policy adjudication; reopen acquisition for new missing fields or substantive conflicts',
                'pdf_visited':False,'qualification_changed':False,'final_task_closure':False}
        links.execute('INSERT INTO dispatch_changes VALUES(?,?,?,?,?,?)',(wid,url,json.dumps({'status':old_status,'evidence':json.loads(old_evidence)}),json.dumps({'status':status,'evidence':detail}),detail['reason'],datetime.now(timezone.utc).isoformat()))
        links.execute('UPDATE dispatch SET status=?,evidence=? WHERE work_id=? AND url=?',(status,json.dumps(detail),wid,url))
        reviews.append(detail);counts[key(explicit[0]['kind'])]+=1
    atomic_json(folder/'known_subtype_download_deferrals.json',{'records':reviews,'deferred_links':len(reviews),'affected_papers':len({r['work_id'] for r in reviews}),'subtypes':dict(counts),'network_requests':0,'final_task_closure':False})
    links.commit();links.close();db.close();print(json.dumps({'deferred_links':len(reviews),'subtypes':dict(counts)}),flush=True)


if __name__ == '__main__':
    run()
