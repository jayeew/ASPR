"""Fixed eScholarship alternate-link metadata via the public GraphQL interface."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/escholarship_metadata'


def fetch(query: str) -> dict:
    FOLDER.mkdir(exist_ok=True)
    folder = FOLDER / 'requests'; folder.mkdir(exist_ok=True)
    path = folder / (hashlib.sha256(query.encode()).hexdigest() + '.json')
    if path.exists(): return json.loads(path.read_text())
    if (FOLDER / 'source_stop.json').exists():
        return {'status': 'not_requested_after_source_stop'}
    result = {'url': 'https://escholarship.org/graphql', 'query': query, 'attempts': [], 'proxy_mode': 'direct'}
    session = requests.Session(); session.trust_env = False
    for attempt in range(3):
        try:
            response = session.post(result['url'], json={'query': query}, timeout=(10,35))
            result['attempts'].append({'http_status': response.status_code})
            raw = path.with_suffix('.response.gz'); raw.write_bytes(gzip.compress(response.content))
            result.update(http_status=response.status_code, raw_receipt=str(raw.relative_to(ROOT)))
            if response.status_code in {401,403,429}:
                atomic_json(FOLDER / 'source_stop.json', result); break
            if response.status_code not in {500,502,503,504}: break
        except requests.RequestException as error:
            result['attempts'].append({'error': type(error).__name__})
        if attempt < 2: time.sleep(2 ** attempt)
    session.close(); atomic_json(path, result)
    return result



def collect() -> None:
    import csv
    import re
    import sqlite3
    from collections import Counter

    from .cached_date_apply import deposited_title
    from .fig8b_qualification_decide import key
    from .supplement_queue import connect

    scope = FOLDER / 'scope.json'
    if not scope.exists():
        links = sqlite3.connect(ROOT / 'supplement/cached_secondary_publication_links/leads.sqlite')
        db = connect(); jobs = []
        for wid, url in links.execute("SELECT work_id,url FROM dispatch WHERE status='source_path_review_required' AND url LIKE 'https://escholarship.org/uc/item/%'"):
            row = db.execute('SELECT doi,title,journal,status FROM papers WHERE work_id=?',(wid,)).fetchone()
            match = re.search(r'/uc/item/([a-z0-9]{8})(?:[/?#]|$)',url)
            if row and match and row[3].startswith('pending'):
                jobs.append({'work_id':wid,'doi':row[0],'title':row[1],'journal':row[2],'url':url,'ark':'ark:/13030/qt'+match[1]})
        db.close(); links.close()
        atomic_json(scope, {'jobs': jobs, 'scope':'Existing pending candidate links only; repository dates/types do not establish publisher qualification'})
    jobs = json.loads(scope.read_text())['jobs']; reviews = []; counts = Counter()
    db = connect()
    source_ids = {p['work_id']: db.execute('SELECT source_id FROM papers WHERE work_id=?',(p['work_id'],)).fetchone()[0] for p in jobs}
    db.close()
    with (ROOT/'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in csv.DictReader(stream)}
    fields = 'id title journal issn published added source type ucpmsPubType publisher abstract contentLink contentType rights isPeerReviewed permalink'
    for start in range(0,len(jobs),25):
        batch = jobs[start:start+25]
        query = '{' + ' '.join('p'+str(i)+':item(id:'+json.dumps(p['doi'])+',scheme:DOI){'+fields+'}' for i,p in enumerate(batch)) + '}'
        receipt = fetch(query); response = {}
        if receipt.get('http_status') == 200:
            try: response = json.loads(gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes()))
            except (ValueError, OSError): response = {'errors':'Unparseable response'}
        for i,paper in enumerate(batch):
            data = (response.get('data') or {}).get('p'+str(i))
            status = 'repository_metadata_unavailable'
            if response.get('errors'): status = 'repository_query_error'
            elif data:
                matched = (data['id'] == paper['ark']
                           and key(deposited_title(data['title'])) == key(deposited_title(paper['title']))
                           and (key(data.get('journal')) == key(paper['journal']) or data.get('issn') in journals[source_ids[paper['work_id']]]))
                status = 'matched_repository_metadata_not_publisher_qualification' if matched else 'repository_identity_conflict'
            record = {**paper,'status':status,'receipt':receipt,'metadata':data,'qualification_changed':False,'final_task_closure':False,
                      'identity_rule':'Exact DOI lookup and ARK; decoded title; exact journal name or registry ISSN',
                      'registry_source_id':source_ids[paper['work_id']]}
            reviews.append(record); counts[status] += 1
        atomic_json(FOLDER/'status.json',{'fixed_papers':len(jobs),'processed':len(reviews),'outcomes':dict(counts),'final_task_closure':False})
        if (FOLDER/'source_stop.json').exists():break
        time.sleep(0.2)
    atomic_json(FOLDER/'review.json',{'records':reviews,'unattempted':jobs[len(reviews):],'outcomes':dict(counts),'qualification_changes':0})
    print(json.dumps(dict(counts)),flush=True)


def record_outcomes() -> None:
    import sqlite3
    from collections import Counter
    from datetime import datetime, timezone

    from .supplement_queue import connect

    review = json.loads((FOLDER / 'review.json').read_text())
    db = connect()
    links = sqlite3.connect(ROOT / 'supplement/cached_secondary_publication_links/leads.sqlite')
    counts = Counter()
    receipt = str((FOLDER / 'review.json').relative_to(ROOT))
    for record in review['records']:
        task_row = db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(record['doi'],)).fetchone()
        detail = {**record, 'next_action':'Inspect native published document if available; repository published date/type alone insufficient'}
        if task_row and not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND status=?',
                                      (task_row[0],'escholarship_graphql',receipt,record['status'])).fetchone():
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (task_row[0],'escholarship_graphql',record['status'],receipt,json.dumps(detail)))
        old = links.execute('SELECT status,evidence FROM dispatch WHERE work_id=? AND url=?',(record['work_id'],record['url'])).fetchone()
        if old and (old[0] == 'source_path_review_required' or (old[0] == 'repository_identity_conflict' and record['status'] == 'matched_repository_metadata_not_publisher_qualification')):
            status = 'repository_metadata_reviewed_native_version_required' if record['status']=='matched_repository_metadata_not_publisher_qualification' else record['status']
            proof = {'receipt':receipt,'work_id':record['work_id'],'doi':record['doi'],
                     'status':record['status'],'qualification_changed':False,'final_task_closure':False}
            links.execute('INSERT INTO dispatch_changes VALUES(?,?,?,?,?,?)',(record['work_id'],record['url'],
                          json.dumps({'status':old[0],'evidence':json.loads(old[1])}),json.dumps({'status':status,'evidence':proof}),
                          'Fixed repository batch outcomes retained; no primary type/date inference',datetime.now(timezone.utc).isoformat()))
            links.execute('UPDATE dispatch SET status=?,evidence=? WHERE work_id=? AND url=?',
                          (status,json.dumps(proof),record['work_id'],record['url']))
        counts[record['status']]+=1
    db.commit();links.commit();db.close();links.close()
    atomic_json(FOLDER/'ledger_status.json',{'outcomes':dict(counts),'unattempted':len(review['unattempted']),
                'qualification_changes':0,'final_task_closure':False})


if __name__ == '__main__':
    collect()
    record_outcomes()
