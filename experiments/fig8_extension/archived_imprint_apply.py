"""Join dated archival imprint evidence to independently verified full online dates."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

from .fig8b_qualification_build import match, write_decisions
from .fig8b_qualification_decide import imprint_dates, portfolio_label
from .publisher_delta_apply import evidence_index, stamp
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def dates_from_receipt(path: Path) -> list[str]:
    raw=gzip.decompress(path.read_bytes())
    if raw.rstrip().endswith(b'</front>'):raw+=b'</article>'
    tree=etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True,huge_tree=True))
    rows=[]
    for node in tree.xpath('//*[local-name()="article-meta"]/*[local-name()="pub-date"]'):
        rows.append({'attributes':dict(node.attrib),**{k:node.xpath(f'string(./*[local-name()="{k}"])') for k in ['year','month','day']}})
    return imprint_dates(rows)


def run(root: Path = ROOT, scope_dois: set[str] | None = None,
        output_folder: Path | None = None) -> dict:
    q=root/'fields/qualification';folder=root/'supplement';stats=Counter()
    destination = output_folder or folder
    destination.mkdir(parents=True, exist_ok=True)
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root);evidence_index(db,q)
        journals={r['source_id']:r for r in read_csv(root/'fields/journal_registry.csv')}
        annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(root/'fields/journal_publisher_by_year.csv')}
        cache={};additions=[];changes=defaultdict(dict);audit=[]
        shards = None if scope_dois is None else {
            row[0] for doi in scope_dois
            for row in db.execute('SELECT shard FROM papers WHERE doi=?', (doi,))}
        for path in sorted((q/'decisions').glob('*.parquet')):
            if shards is not None and path.name not in shards:continue
            for row in pq.read_table(path).to_pylist():
                if scope_dois is not None and row['doi'] not in scope_dois:continue
                if row['eligibility_status']!='included_verified_research' or row['nature_portfolio_at_verified_online_year'] in ['True','False']:continue
                wid=row['work_id'];items=[json.loads(p) for p, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",(wid,))]
                paper=db.execute('SELECT title,journal FROM papers WHERE work_id=?',(wid,)).fetchone()
                new=[]
                for item in items:
                    if item.get('family')!='archived_jats' or item.get('identity')!='matched' or not item.get('publishers') or item.get('online_date'):continue
                    raw=q/item['receipt']
                    if str(raw) not in cache:
                        try:cache[str(raw)]=dates_from_receipt(raw)
                        except (OSError,etree.XMLSyntaxError):cache[str(raw)]=[];stats['raw_unavailable_or_invalid']+=1
                    dates=cache[str(raw)]
                    if not dates:continue
                    source={k:v for k,v in item.items() if k not in ['id','work_id','doi','identity','class']}
                    source.update(publication_dates=dates,imprint_date_rule='Archival full publication date must exactly agree with matched explicit online date; not a new online-date qualification rule')
                    evidence=match({'work_id':wid,'doi':row['doi'],'source_id':row['source_id'],'title':paper[0],'journal_name':paper[1]},source,journals[row['source_id']])
                    if evidence['id'] not in {v['id'] for v in items}:new.append({'work_id':wid,'doi':row['doi'],**evidence})
                if not new:continue
                label,ids=portfolio_label(annual.get((row['source_id'],row['online_year']),''),items+new,row['online_year'])
                if label!='False':stats['unresolved_after_date_check']+=1;continue
                updates={'nature_portfolio_at_verified_online_year':label,'portfolio_evidence_ids':json.dumps(ids),
                         'evidence_ids':json.dumps(list(dict.fromkeys(json.loads(row['evidence_ids'])+[v['id'] for v in new])))}
                changes[path.name][wid]=updates;additions.extend(new)
                audit.append({'work_id':wid,'doi':row['doi'],'before':row['nature_portfolio_at_verified_online_year'],'after':label,'online_date':row['online_date'],'evidence_ids':ids,'status':'archival_imprint_and_full_date_agreement'})
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as stream,(q/'curated_evidence.jsonl').open('a') as curated:
            for value in additions:
                line=json.dumps(value,ensure_ascii=False)+'\n';stream.write(line);curated.write(line)
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',(value['work_id'],value['id'],'primary',json.dumps(value,ensure_ascii=False)))
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',('primary',stamp(q/'decision_evidence.jsonl.gz')))
        for shard,selected in changes.items():
            rows=pq.read_table(q/'decisions'/shard).to_pylist()
            for row in rows:
                if row['work_id'] in selected:row.update(selected[row['work_id']])
            write_decisions(q/'decisions'/shard,rows,q/'decision_changes.jsonl.gz')
        db.commit();db.close()
    result={**stats,'paper_brand_labels_filled':len(audit),'evidence_added':len(additions),'raw_receipts_reused':len(cache),'network_requests':0}
    atomic_json(destination/'archived_imprint_apply_status.json',result);atomic_json(destination/'archived_imprint_changes.json',audit)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
