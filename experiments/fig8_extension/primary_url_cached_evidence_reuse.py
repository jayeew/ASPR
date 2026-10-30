"""Route unvisited URL leads to matched existing evidence without claiming access."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def usable_evidence(item: dict, doi: str, root: Path) -> bool:
    """Require current DOI identity and a concrete retained source, not just a URL."""
    receipt = item.get('receipt')
    return bool(doi and item.get('doi') == doi and item.get('identity') == 'matched'
                and item.get('id') and isinstance(receipt, str) and receipt
                and (root / 'fields/qualification' / receipt).is_file())


def run() -> None:
    folder = ROOT / 'supplement/cached_primary_url_leads'
    links = sqlite3.connect(folder / 'leads.sqlite', timeout=60)
    current = sqlite3.connect(f'file:{ROOT / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    current.execute('BEGIN')
    selected: dict[str, set[str]] = defaultdict(set)
    for url, wid in links.execute("SELECT l.url,l.work_id FROM leads l JOIN url_dispatch d USING(url) WHERE d.status='source_path_review_pending'"):
        selected[url].add(wid)
    wanted = {wid for ids in selected.values() for wid in ids}
    papers = {wid: (doi, status) for wid, doi, status in current.execute('SELECT work_id,doi,status FROM papers') if wid in wanted}
    evidence: dict[str, list[dict]] = defaultdict(list)
    for wid, raw in current.execute("SELECT work_id,payload FROM current_evidence_index WHERE kind='primary'"):
        if wid not in papers or not papers[wid][1].startswith('pending'):continue
        item = json.loads(raw)
        if usable_evidence(item, papers[wid][0], ROOT):
            evidence[wid].append({k: item.get(k) for k in ['id','doi','family','receipt','url','kind','online_date','class']})
    current.close()
    changed = 0; associated = set(); statuses = Counter(); now = datetime.now(timezone.utc).isoformat()
    links.execute('BEGIN IMMEDIATE')
    for url, ids in selected.items():
        if not all(wid in evidence for wid in ids):continue
        old = links.execute('SELECT status,reason,evidence FROM url_dispatch WHERE url=?',(url,)).fetchone()
        if old[0] != 'source_path_review_pending':continue
        proof = {'records': [{'work_id': wid,'doi': papers[wid][0], 'qualification_status': papers[wid][1], 'primary_evidence': evidence[wid]} for wid in sorted(ids)],
                 'destination_url_access_confirmed': False,'qualification_changed': False,
                 'final_task_closure': False,'next_action':'Interpret matched cached fields; acquire only fields absent or conflicting in those sources. Existing evidence may be from an alternative URL.'}
        new = ('cached_primary_evidence_available_field_review_pending',
               'Same current work/DOI has matched primary evidence with retained source file; URL remains unvisited, missing fields and conflicts remain pending.',json.dumps(proof))
        links.execute('INSERT INTO url_dispatch_changes VALUES(?,?,?,?,?)',(url,json.dumps(old),json.dumps(new),'matched_cached_evidence_reuse',now))
        links.execute('UPDATE url_dispatch SET status=?,reason=?,evidence=? WHERE url=?',(*new,url))
        changed += 1;associated.update(ids)
    links.commit()
    for wid in associated:statuses[papers[wid][1]] += 1
    result = dict(status='cached_evidence_routed_not_source_closure',updated_at=now,urls_changed=changed,
                  distinct_papers=len(associated),paper_statuses=dict(statuses),network_requests=0,
                  qualification_changes=0,final_task_closure=False)
    summary_path=folder/'dispatch_summary.json'
    summary=json.loads(summary_path.read_text());summary['url_status_counts']=dict(links.execute('SELECT status,count(*) FROM url_dispatch GROUP BY status'));summary['updated_at']=now
    summary['top_unreviewed_hosts']=links.execute("SELECT host,count(DISTINCT work_id) FROM leads JOIN url_dispatch USING(url) WHERE status='source_path_review_pending' GROUP BY host ORDER BY count(DISTINCT work_id) DESC LIMIT 30").fetchall()
    assert sum(summary['url_status_counts'].values())==summary['distinct_urls']
    atomic_json(summary_path,summary);atomic_json(folder/'cached_evidence_reuse.json',result)
    links.close();print(json.dumps(result),flush=True)


if __name__=='__main__':run()
