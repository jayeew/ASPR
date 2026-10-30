"""Apply explicit Springer publisher type aliases to cached evidence only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_build import key, match, write_decisions
from .fig8b_qualification_decide import decide, portfolio_label
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(apply: bool = False, root: Path = ROOT) -> None:
    q=root/'fields/qualification';folder=root/'supplement/springer_type_aliases';folder.mkdir(exist_ok=True)
    if (folder/'applied.json').exists():
        print('Already applied; no repeated evidence or decisions.');return
    additions=defaultdict(list);types=defaultdict(set)
    with (root/'fields/journal_registry.csv').open(encoding='utf-8-sig') as f:sources={r['source_id']:r for r in csv.DictReader(f)}
    rows={}
    for path in (q/'decisions').glob('*.parquet'):
        for r in pq.read_table(path).to_pylist():
            if any(key(t.strip()) in {'originalpaper','originalarticle','researchpaper','reviewpaper'} for t in r['publisher_types'].split('|')):rows[r['work_id']]=r
    existing=defaultdict(list)
    with gzip.open(q/'decision_evidence.jsonl.gz','rt') as f:
        for line in f:
            e=json.loads(line)
            if e['work_id'] not in rows:continue
            existing[e['work_id']].append(e)
            if e['family']=='publisher_springer':types[e['work_id']].add(e['kind'])
    for wid,items in existing.items():
        for e in items:
            if e['family']!='publisher_springer' or key(e['kind']) not in {'originalpaper','originalarticle','researchpaper','reviewpaper'}:continue
            if not e.get('url','').startswith('https://link.springer.com/article/'):continue
            item={k:v for k,v in e.items() if k not in {'id','class','identity','work_id','doi'}}
            item['categories']=sorted(types[wid]);item['type_rule_receipt']='../../supplement/springer_type_aliases/rule.json'
            row={**rows[wid],'journal_name':sources[rows[wid]['source_id']]['journal_name']}
            updated=match(row,item,sources[row['source_id']])
            if updated['class']!=e['class']:additions[wid].append({'work_id':wid,'doi':rows[wid]['doi'],**updated})
    secondary={}
    with gzip.open(q/'secondary_evidence.jsonl.gz','rt') as f:
        for line in f:
            e=json.loads(line)
            if e['work_id'] in additions:secondary[e['work_id']]=e['records']
    with (root/'fields/journal_publisher_by_year.csv').open(encoding='utf-8-sig') as f:annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in csv.DictReader(f)}
    delta={};transitions=Counter()
    for wid,extra in additions.items():
        old=rows[wid];evidence=existing[wid]+extra;d=decide(evidence,secondary.get(wid,[]),old['duplicate_of'])
        label,ids=portfolio_label(annual.get((old['source_id'],d['online_year']),''),evidence,d['online_year'])
        new={**old,'online_year':d['online_year'],'online_date':d['online_date'],'publisher_types':d['type'],
            'eligibility_status':d['status'],'reason':d['reason'],'evidence_ids':json.dumps(d['evidence_ids']),
            'rejected_evidence_ids':json.dumps(d.get('rejected_evidence_ids',[])),
            'nature_portfolio_at_verified_online_year':label,'portfolio_evidence_ids':json.dumps(ids)}
        delta[wid]={'before':old,'after':new};transitions[old['eligibility_status']+' -> '+new['eligibility_status']]+=1
    atomic_json(folder/'rule.json',{'authority':'Springer Nature explicit article-type definitions and API genre vocabulary',
        'urls':['https://www.springernature.com/gp/open-science/oa-agreements/usa/nist','https://dev.springernature.com/docs/advanced-querying/example-advanced-queries/'],
        'scope':'Only original publisher_springer evidence from link.springer.com/article; exact paper identity and online year still required',
        'research_aliases':['OriginalPaper','Original Paper','Original Article','Research Paper'],'excluded_alias':'ReviewPaper',
        'guards':'Additional categories, title subtypes, secondary review/integrity flags and all primary conflicts remain holds; funding eligibility itself is not research evidence'})
    atomic_json(folder/'decision_delta.json',list(delta.values()))
    atomic_json(folder/'evidence_additions.json',[e for v in additions.values() for e in v])
    summary={'affected_records':len(delta),'evidence_additions':sum(map(len,additions.values())),'transitions':dict(transitions),'status':'proposed'}
    atomic_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)
    if not apply:return
    with (root/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Check all rows before writing any shard, so concurrent work cannot be silently overwritten.
        for path in (q/'decisions').glob('*.parquet'):
            for r in pq.read_table(path).to_pylist():
                if r['work_id'] in delta:assert r==delta[r['work_id']]['before'],'Decision changed while proposals were prepared'
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as f,(q/'curated_evidence.jsonl').open('a') as curated:
            for items in additions.values():
                for e in items:
                    line=json.dumps(e,ensure_ascii=False)+'\n';f.write(line);curated.write(line)
        for path in (q/'decisions').glob('*.parquet'):
            current=pq.read_table(path).to_pylist()
            if any(r['work_id'] in delta for r in current):
                write_decisions(path,[delta[r['work_id']]['after'] if r['work_id'] in delta else r for r in current],q/'decision_changes.jsonl.gz')
        db=connect(root)
        for wid,change in delta.items():
            value=change['after'];resolved=value['eligibility_status'].startswith(('included','excluded'))
            db.execute('UPDATE papers SET status=? WHERE work_id=?',(value['eligibility_status'],wid))
            db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",('resolved' if resolved else 'pending',value['reason'],value['doi']))
        db.commit();db.close()
        shutil.copy2(Path(__file__).with_name('fig8b_qualification_decide.py'),q/'decide.py')
        summary['status']='applied_derived_refresh_required';atomic_json(folder/'applied.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');run(parser.parse_args().apply)
