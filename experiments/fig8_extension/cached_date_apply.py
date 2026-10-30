"""Apply matched explicit online dates to the fixed Copernicus primary-type queue."""
from __future__ import annotations

import fcntl
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import date
from html import unescape
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_build import match, plain, write_decisions
from .publisher_delta_apply import evidence_index, stamp, updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def deposited_title(value: str) -> str:
    """Decode entity/inline-formatting layers without deleting scientific comparisons."""
    for _ in range(3):
        decoded = unescape(value)
        decoded = re.sub(r'</?(?:i|b|em|strong|sub|sup|italic|jats:italic|jats:bold|jats:sub|jats:sup)(?:\s[^>]*)?>', '', decoded)
        if decoded == value:
            break
        value = decoded
    return value


def run(scope_dois: set[str] | None = None, output_folder: Path | None = None, nonblocking: bool = False,
        explicit_sources: dict[str, list[dict]] | None = None) -> dict:
    folder = output_folder if output_folder is not None else ROOT / 'supplement/copernicus'
    q = ROOT / 'fields/qualification'
    scope = set(scope_dois) if scope_dois is not None else set(json.loads((folder / 'date_deposit_scope.json').read_text())['dois'])
    sources = defaultdict(list, explicit_sources or {})
    if not set(sources).issubset(scope):
        raise ValueError('Date sources outside fixed DOI scope')
    stats = Counter()
    for path in ([] if explicit_sources is not None else (ROOT/'fields/crossref_raw').glob('*.gz')):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        if r.get('status')!='success':continue
        for item in r['response']['message']['items']:
            doi=item['DOI'].lower()
            if doi not in scope:continue
            for parts in item.get('published-online',{}).get('date-parts',[]):
                if not parts or len(parts)>3 or not all(isinstance(v,int) for v in parts):continue
                try:date(parts[0],parts[1] if len(parts)>1 else 1,parts[2] if len(parts)>2 else 1)
                except ValueError:continue
                sources[doi].append({'family':'crossref_publisher_deposit','title':deposited_title(plain((item.get('title') or [''])[0])),'issns':item.get('ISSN',[]),'journal_titles':item.get('container-title',[]),'kind':item.get('type','unspecified'),'online_date':'-'.join(str(v).zfill(4 if i==0 else 2) for i,v in enumerate(parts)),'date_precision':['year','month','day'][len(parts)-1],'receipt':'../'+str(path.relative_to(ROOT/'fields')),'url':r['url'],'categories':[]})
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0));db=connect(ROOT);evidence_index(db,q)
        journals={r['source_id']:r for r in read_csv(ROOT/'fields/journal_registry.csv')};annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
        grouped=defaultdict(dict);new_evidence=[]
        for doi,items in sources.items():
            candidates=list(db.execute('SELECT work_id,title,journal,source_id,shard FROM papers WHERE doi=?',(doi,)))
            if len(candidates)!=1:stats['ambiguous_work_identity']+=1;continue
            wid,title,journal,sid,shard=candidates[0];paper={'work_id':wid,'doi':doi,'title':title,'journal_name':journal,'source_id':sid}
            prior={};secondary=[]
            for kind,payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?',(wid,)):
                v=json.loads(payload)
                if kind=='primary':prior[v['id']]=v
                else:secondary=v
            added=[]
            for item in items:
                e=match(paper,item,journals[sid])
                if e['id'] not in prior:
                    e={'work_id':wid,'doi':doi,**e};prior[e['id']]=e;added.append(e)
            if not added:stats['already_applied']+=1;continue
            new_evidence.extend(added);grouped[shard][wid]=(list(prior.values()),secondary)
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as stream,(q/'curated_evidence.jsonl').open('a') as curated:
            for e in new_evidence:
                line=json.dumps(e,ensure_ascii=False)+'\n';stream.write(line);curated.write(line);db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',(e['work_id'],e['id'],'primary',json.dumps(e,ensure_ascii=False)))
        for shard,selected in grouped.items():
            path=q/'decisions'/shard;rows=pq.read_table(path).to_pylist()
            before_sync = 'ownership-queue-1:' + stamp(ROOT / 'fields/papers' / shard) + ':' + stamp(path)
            eligible_changed = False
            for row in rows:
                if row['work_id'] not in selected:continue
                items,secondary=selected[row['work_id']];new=updated_decision(row,items,secondary,annual)
                eligible_changed |= (new != row and 'included_verified_research' in {row['eligibility_status'], new['eligibility_status']})
                if new!=row:stats[row['eligibility_status']+' -> '+new['eligibility_status']]+=1
                row.update(new);wid=row['work_id'];status=row['eligibility_status'];db.execute('UPDATE papers SET status=? WHERE work_id=?',(status,wid))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",('resolved' if status.startswith(('included','excluded')) else 'pending',row['reason'],row['doi']))
                if status=='included_verified_research':task(db,'metadata',wid,'pmc_or_publisher',{'doi':row['doi'],'reason':'Current matched qualification evidence; reuse cached article metadata'},wid)
            write_decisions(path,rows,q/'decision_changes.jsonl.gz')
            from .metadata_task_sync import advance_nonqualified_only_sync
            after_sync = 'ownership-queue-1:' + stamp(ROOT / 'fields/papers' / shard) + ':' + stamp(path)
            advance_nonqualified_only_sync(db, shard, before_sync, after_sync, eligible_changed)
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',('primary',stamp(q/'decision_evidence.jsonl.gz')));db.commit();db.close()
    dated = sum(any(item.get('online_date') for item in items) for items in sources.values())
    deposited = sum(any(item.get('online_date') and item.get('family') == 'crossref_publisher_deposit'
                        for item in items) for items in sources.values())
    result = {'fixed_dois': len(scope), 'source_dois': len(sources),
              'with_explicit_online_date': dated, 'with_explicit_deposited_date': deposited,
              'new_evidence': len(new_evidence), **stats}
    atomic_json(folder / 'date_apply_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def consume_pending() -> dict:
    """Consume explicit, already-published DOI shards without waiting on another writer."""
    queue = ROOT / 'supplement/crossref_delta_pending'
    if not queue.exists():
        return {'applied_shards': 0}
    count = 0
    with (queue / 'consumer.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'applied_shards': 0, 'deferred': 'another_crossref_delta_consumer'}
        for path in sorted(queue.glob('*.json')):
            job = json.loads(path.read_text())
            if job['status'] == 'applied':
                continue
            folder = (ROOT / job['output_folder']).resolve()
            if not folder.is_relative_to((ROOT / 'supplement').resolve()):
                raise ValueError('Scoped Crossref audit folder outside dataset supplement')
            if not all((ROOT / p).is_file() for p in job['published_receipts']):
                raise ValueError('Scoped Crossref receipt missing; cannot mark applied')
            try:
                result = run(set(job['dois']), folder, nonblocking=True)
            except BlockingIOError:
                return {'applied_shards': count, 'deferred': 'existing_checkpoint_writer'}
            atomic_json(folder / 'integration_first.json', result)
            atomic_json(path, {**job, 'status': 'applied', 'result': result})
            count += 1
    return {'applied_shards': count}


if __name__=='__main__':run()
