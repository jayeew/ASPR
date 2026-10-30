"""Apply journal-specific named format rules using existing JATS, then reuse author metadata."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .fig8b_qualification_decide import native_named_research_format
from .official_rule_reuse import run as reuse_rules
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata


def run(source_ids: list[str] | None = None, output_folder: Path | None = None) -> None:
    folder=output_folder or ROOT/'supplement/named_native_formats';folder.mkdir(exist_ok=True)
    staged=folder/'staged';staged.mkdir(exist_ok=True)
    policies=json.loads((ROOT/'supplement/journal_type_policies/named_native_formats.json').read_text())['rules']
    ids=source_ids or [p['source_id'] for p in policies];db=connect();rows=[]
    placeholders = ','.join('?' for _ in ids)
    rows = db.execute(f'''SELECT p.work_id,p.doi,e.payload FROM papers p JOIN current_evidence_index e USING(work_id)
        WHERE p.source_id IN ({placeholders}) AND p.status LIKE 'pending%' AND e.kind='primary' ''', ids).fetchall()
    db.close();jobs={};reviews=[]
    for wid,doi,payload in rows:
        evidence=json.loads(payload)
        if doi in jobs or not native_named_research_format(evidence):continue
        raw=gzip.decompress((ROOT/'fields/qualification'/evidence['receipt']).read_bytes())
        if raw.rstrip().endswith(b'</front>'):
            raw += b'</article>'
        parsed=jats_metadata(raw);parsed['doi']=[v.lower() for v in parsed['doi']];parsed['issns']=parsed['issn']
        if doi not in parsed['doi']:
            reviews.append({'work_id':wid,'doi':doi,'status':'raw_doi_conflict','receipt':evidence['receipt']});continue
        path=staged/(hashlib.sha256(doi.encode()).hexdigest()+'.json')
        atomic_json(path,{'doi':doi,'source':'pmc','route':'pmc','url':evidence['url'],
            'raw_receipt':evidence['receipt'],'status':'parsed','parsed':parsed,
            'reviewed_type_rules':str((folder/'policy.json').relative_to(ROOT)) if (folder/'policy.json').exists() else 'supplement/journal_type_policies/named_native_formats.json'})
        jobs[doi]=str(path.relative_to(ROOT));reviews.append({'work_id':wid,'doi':doi,'status':'cached_native_heading_selected','receipt':evidence['receipt']})
    atomic_json(folder/'scope_review.json',{'records':reviews,'network_requests':0,
        'scope':'All currently pending candidates in the selected reviewed journals; not limited to the recent PMC batch'})
    result=apply_available(supplied_jobs=sorted(jobs.items())) if jobs else {}
    db=connect();selected=[(wid,shard,receipt) for doi,receipt in jobs.items() for wid,shard in db.execute(
        "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'",(doi,))];db.close()
    authors=apply_authors(selected_jobs=selected,workers=8,fill_abstract=True) if selected else {}
    if authors.get('papers_changed'):reuse_rules()
    atomic_json(folder/'status.json',{'status':'fixed_named_format_scope_applied','qualification':result,'authors':authors,
        'network_requests':0,'scope_outcomes':dict(Counter(r['status'] for r in reviews)),
        'full_refresh':'scheduled_only','final_task_closure':False})
    print(json.dumps({'qualification':result,'authors':authors}),flush=True)


if __name__ == '__main__':
    run()
