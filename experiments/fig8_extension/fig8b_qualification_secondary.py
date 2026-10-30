"""Screen all remaining candidate DOIs for secondary publication types and online dates."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode

from . import fig8b_epmc
from .fig8b_qualification_evidence import BASE, OUT


def main() -> None:
    fig8b_epmc.OUT=OUT
    original=sqlite3.connect(BASE/'epmc_queue.sqlite')
    previously={r[0] for r in original.execute("SELECT doi FROM queue WHERE status='complete'")};original.close()
    candidates=sqlite3.connect(OUT/'evidence.sqlite')
    dois=[r[0] for r in candidates.execute("SELECT DISTINCT doi FROM candidates WHERE doi<>'' ORDER BY doi") if r[0] not in previously and '"' not in r[0]]
    candidates.close()
    db=sqlite3.connect(OUT/'secondary_attempts.sqlite');db.execute('CREATE TABLE IF NOT EXISTS attempts(doi TEXT PRIMARY KEY,status TEXT,receipt TEXT)')
    db.executemany('INSERT OR IGNORE INTO attempts VALUES (?,?,?)',[(d,'pending','') for d in dois]);db.commit()
    for path in (OUT/'epmc_raw').glob('*.gz'):
        with gzip.open(path,'rt') as stream:receipt=json.load(stream)
        db.executemany('UPDATE attempts SET status=?,receipt=? WHERE doi=?',[(receipt['status'],str(path.relative_to(OUT)),d) for d in receipt['dois']])
    db.commit()
    pending=[r[0] for r in db.execute("SELECT doi FROM attempts WHERE status='pending' ORDER BY doi")]
    batches=[];current=[]
    for doi in pending:
        proposed=current+[doi]
        query=' OR '.join(f'DOI:"{d}"' for d in proposed)
        if current and (len(urlencode({'query':query}))>6500 or len(proposed)>140):
            batches.append(current);current=[doi]
        else:current=proposed
    if current:batches.append(current)
    (OUT/'secondary_scope.json').write_text(json.dumps({'rule':'All nonempty quote-safe candidate DOIs not previously completed in initial metadata-repair Europe PMC queue','prior_completed_dois':len(previously),'additional_dois':len(dois),'batches':len(batches),'batch_rule':'up to 140 DOIs; URL-encoded query <=6500 bytes; earlier 60-DOI receipts retained','scope':'Secondary screening only; generic Journal Article never confirms research eligibility'}))
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures=[pool.submit(fig8b_epmc.fetch,b) for b in batches]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();token=hashlib.sha256(json.dumps(r['params'],sort_keys=True).encode()).hexdigest()
            db.executemany('UPDATE attempts SET status=?,receipt=? WHERE doi=?',[(r['status'],f'epmc_raw/{token}.json.gz',d) for d in r['dois']]);db.commit()
            if i%500==0:print('Secondary batches',i,len(batches),flush=True)
    result=dict(db.execute('SELECT status,count(*) FROM attempts GROUP BY status'));db.close()
    (OUT/'secondary_status.json').write_text(json.dumps(result));print(result,flush=True)


if __name__=='__main__':main()
