"""Apply exact candidate-bound public DC evidence without inferring online dates or research."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

from lxml import etree
import pyarrow.parquet as pq

from .fig8b_qualification_build import key, match, plain, write_decisions
from .ojs_metadata_harvest import FOLDER, NS
from .publisher_delta_apply import evidence_index, stamp, updated_decision
from .supplement_queue import ROOT, connect, attempt
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


@lru_cache(maxsize=16)
def entries(receipt: str) -> dict:
    tree=etree.fromstring(gzip.decompress((ROOT/receipt).read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
    return {e.findtext('o:header/o:identifier',namespaces=NS):e for e in
            tree.findall('o:ListRecords/o:record',NS)+tree.findall('o:GetRecord/o:record',NS)}


def dc_evidence(info: dict, url: str, title: str) -> list[dict]:
    entry=entries(info['raw_receipt']).get(info['identifier'])
    if entry is None:return []
    titles=entry.xpath('.//dc:title/text()',namespaces=NS)
    if key(title) not in {key(plain(v)) for v in titles}:return []
    sources=entry.xpath('.//dc:source/text()',namespaces=NS)
    issns=[v.strip() for v in sources if re.fullmatch(r'\d{4}-\d{3}[\dXx]',v.strip())]
    journal_names=[v.split(';')[0].strip() for v in sources if ';' in v]
    types=[v.strip() for v in entry.xpath('.//dc:type/text()',namespaces=NS) if not v.startswith(('http:','https:','info:'))]
    return [{'family':'publisher_oai_dc','title':title,'issns':issns,'journal_titles':journal_names,
             'kind':kind,'categories':types,'online_date':'','date_precision':'missing',
             'receipt':'../../'+info['raw_receipt'],'url':url,'native_identifier':info['identifier'],
             'declared_dc_dates_not_verified_online':entry.xpath('.//dc:date/text()',namespaces=NS),
             'scope_review_required':key(kind) in {'researcharticle','originalresearch','originalresearcharticle'},
             'scope_review_note':'OJS export requires journal-specific original-research rule; generic article label and DC date are insufficient'}
            for kind in types or ['unspecified']]


def run() -> dict:
    cached=[]
    for path in (FOLDER/'harvested').glob('*.json'):
        source=json.loads(path.read_text())
        cached.extend((wid,info,source['url']) for wid,info in source.get('matched',{}).items() if info['title_identity']=='exact')
    stats=Counter();q=ROOT/'fields/qualification'
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(ROOT);evidence_index(db,q)
        journals={r['source_id']:r for r in read_csv(ROOT/'fields/journal_registry.csv')}
        annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
        selected=defaultdict(dict);additions=[]
        for wid,info,url in cached:
            current=db.execute('SELECT doi,title,journal,source_id,shard FROM papers WHERE work_id=?',(wid,)).fetchone()
            if not current:continue
            doi,title,journal,sid,shard=current
            paper={'work_id':wid,'title':title,'journal_name':journal,'source_id':sid}
            prior={};secondary=[]
            for kind,payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?',(wid,)):
                value=json.loads(payload)
                if kind=='primary':prior[value['id']]=value
                else:secondary=value
            new=[]
            for item in dc_evidence(info,url,title):
                e={'work_id':wid,'doi':doi,**match(paper,item,journals[sid])}
                if e['id'] in prior:continue
                prior[e['id']]=e;new.append(e);stats['identity_'+e['identity']]+=1
            if not new:continue
            additions.extend(new);selected[shard][wid]=(list(prior.values()),secondary)
            task_id=hashlib.sha256(('qualification:'+(doi or wid)).encode()).hexdigest()[:32]
            attempt(db,task_id,'publisher_oai_dc','evidence_ready_type_date_unresolved',info['raw_receipt'],
                    {'landing_link_and_title_matched':True,'journal_identity':[e['identity'] for e in new],'native_identifier':info['identifier'],'qualification_not_confirmed':True})
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as stream,(q/'curated_evidence.jsonl').open('a') as curated:
            for e in additions:
                line=json.dumps(e,ensure_ascii=False)+'\n';stream.write(line);curated.write(line)
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',(e['work_id'],e['id'],'primary',json.dumps(e,ensure_ascii=False)))
        db.commit()
        for shard,targets in selected.items():
            path=q/'decisions'/shard;rows=pq.read_table(path).to_pylist()
            for row in rows:
                if row['work_id'] not in targets:continue
                primary,secondary=targets[row['work_id']];new=updated_decision(row,primary,secondary,annual)
                stats[row['eligibility_status']+' -> '+new['eligibility_status']]+=1;row.update(new)
                db.execute('UPDATE papers SET status=? WHERE work_id=?',(row['eligibility_status'],row['work_id']))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",('resolved' if row['eligibility_status'].startswith(('included','excluded')) else 'pending',row['reason'],row['doi'] or row['work_id']))
            write_decisions(path,rows,q/'decision_changes.jsonl.gz');db.commit()
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',('primary',stamp(q/'decision_evidence.jsonl.gz')));db.commit();db.close()
    result={'status':'cached_publisher_evidence_applied_not_research_confirmation','matched_cache_candidates':len(cached),'new_evidence_records':len(additions),**stats}
    atomic_json(FOLDER/'evidence_apply_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
