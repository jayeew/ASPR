"""Retry transient XML failures once, retaining each prior attempt receipt."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .fig8b_qualification_evidence import OUT, fetch


def main() -> None:
    history=OUT/'jats_attempt_history';history.mkdir(exist_ok=True)
    targets=[]
    for path in sorted((OUT/'jats_parsed').glob('*.json')):
        r=json.loads(path.read_text())
        if r['status']=='parsed' or r.get('http_status') in [401,403,404]:continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        saved=history/(path.stem+'.'+digest+'.json')
        if not saved.exists():shutil.copy2(path,saved)
        path.unlink()
        targets.append({'doi':r['doi'],'pmcid':r['pmcid']})
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes=list(pool.map(fetch,targets))
    summary={'targets':len(targets),'statuses':dict(Counter(r['status'] for r in outcomes)),
             'scope':'One additional retry pass after initial fetch; prior attempt receipts retained'}
    (OUT/'jats_retry_summary.json').write_text(json.dumps(summary,indent=2));print(summary)


if __name__=='__main__':main()
