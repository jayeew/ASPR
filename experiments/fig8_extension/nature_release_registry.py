"""Query every scoped unresolved unit once, reusing previous exact query receipts."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow.parquet as pq

from . import nature_clinical_review as prior
from . import nature_institution_review as registry
from .nature_release_prepare import INPUT, OUT
from .stage0 import write_json


def main() -> None:
    rows=pq.read_table(INPUT).to_pylist()
    scope=set(json.loads((OUT/'scope.json').read_text())['union'])
    terms=sorted({t for r in rows if r['doi'] in scope for a in json.loads(r[prior.FIELD]) for s in a[prior.SLOTS]
        if s['identity_status']=='identity_unresolved' and s['display_name']!='No affiliation evidence'
        for t in registry.name_queries(s['display_name'])})
    old=registry.OUT
    registry.OUT=OUT
    client=registry.ReviewRegistry()

    def query(term: str) -> dict[str, Any]:
        cached=old/'ror_queries'/(hashlib.sha256(term.encode()).hexdigest()+'.json')
        if cached.exists():
            result=json.loads(cached.read_text())
            result.update(receipt=str(cached.resolve()),reused=True)
        else:
            result=client.query(term)
        return {'query':term,'receipt':result.get('receipt'),'error':result.get('error'),
            'reused':result.get('reused',False),'result_n':len(result.get('response',{}).get('items',[]))}

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(query,terms))
    finally:
        registry.OUT=old
    write_json(OUT/'registry_query_log.json',results)


if __name__=='__main__':
    main()
