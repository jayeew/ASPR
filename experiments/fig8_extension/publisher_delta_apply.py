"""Incrementally qualify newly acquired publisher metadata against indexed prior evidence."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .arrow_row_patch import patch_rows, selected_rows
from .derived_refresh import refresh
from .fig8b_qualification_build import match, plain, write_decisions
from .fig8b_qualification_decide import decide, electronic_date, imprint_dates, key, portfolio_label
from .fig8b_qualification_recompute import metrics
from .native_abstract_scope import native_abstract_scope
from .supplement_apply import abstract_ok
from .batch_memory import release_batch_memory
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def stamp(path: Path) -> str:
    value=path.stat()
    return f'{value.st_size}:{value.st_mtime_ns}'


def evidence_index(db: sqlite3.Connection, q: Path) -> None:
    db.executescript('''CREATE TABLE IF NOT EXISTS current_evidence_index(work_id TEXT,evidence_key TEXT,kind TEXT,payload TEXT,PRIMARY KEY(work_id,evidence_key));
        CREATE INDEX IF NOT EXISTS current_evidence_kind ON current_evidence_index(kind);
        CREATE INDEX IF NOT EXISTS current_evidence_work_kind ON current_evidence_index(work_id,kind);
        CREATE TABLE IF NOT EXISTS current_evidence_files(kind TEXT PRIMARY KEY,stamp TEXT);''')
    for kind,name in [('primary','decision_evidence.jsonl.gz'),('secondary','secondary_evidence.jsonl.gz')]:
        path=q/name;before=stamp(path)
        old=db.execute('SELECT stamp FROM current_evidence_files WHERE kind=?',(kind,)).fetchone()
        if old and old[0]==before:continue
        db.execute('DELETE FROM current_evidence_index WHERE kind=?',(kind,));db.commit();batch=[];count=0
        with gzip.open(path,'rt') as stream:
            for line in stream:
                item=json.loads(line);payload=item if kind=='primary' else item['records']
                batch.append((item['work_id'],item['id'] if kind=='primary' else 'secondary',kind,json.dumps(payload,ensure_ascii=False)))
                if len(batch)>=1000:
                    db.executemany('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',batch);db.commit();count+=len(batch);batch=[]
            if batch:db.executemany('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',batch)
        if stamp(path)!=before:raise RuntimeError('Evidence changed during indexing')
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',(kind,before));db.commit()
        print('Indexed',kind,count+len(batch),flush=True)


def publisher_items(record: dict) -> list[dict]:
    p=record['parsed']
    if record.get('source')=='pmc' or record.get('route') in {'ecovector_oai_nlm', 'rcsi_oai_nlm'}:
        native_dois = {value.strip().casefold() for value in p.get('doi', []) if value.strip()}
        candidate_doi = (record.get('doi') or '').casefold()
        native_doi = (candidate_doi if candidate_doi in native_dois else
                      next(iter(native_dois)) if len(native_dois) == 1 else '')
        online,precision=electronic_date(p['dates'])
        kind = p['article_type']
        section_evidence = record.get('native_section_type_evidence')
        categories = p.get('categories', [])
        research_heading = next((heading for heading in categories if key(heading) in {
            'researcharticle', 'originalarticle', 'originalresearch', 'originalresearcharticle',
            'researcharticles', 'originalarticles', 'originalpapers', 'originalstudyarticles',
            'clinicalbasicscienceresearcharticle', 'experimentalsectionresearcharticle'}), '')
        if key(kind) in {'', 'other'} and research_heading:
            kind = 'research-article'
            section_evidence = {
                'rule_receipt': 'supplement/native_explicit_sections/rule.json',
                'native_article_type': p['article_type'], 'native_section': research_heading,
                'locator': 'front/article-meta/article-categories/subject',
                'normalized_kind': kind}
        if record.get('route') == 'rcsi_oai_nlm' and not kind and len(categories) == 1:
            heading = categories[0]
            experimental = heading == 'Experimental articles' and bool(
                set(p.get('issn', [])) & {'0026-3656', '0015-3303', '0869-8139'})
            if heading == 'Reviews' or experimental:
                kind = 'research-article' if experimental else 'review'
                section_evidence = {
                    'rule_receipt': 'supplement/rcsi_source_review/explicit_sections/policy.json',
                    'native_article_type': p['article_type'], 'native_section': heading,
                    'locator': 'front/article-meta/article-categories/subject', 'normalized_kind': kind}
        return [{'family':'archived_jats','doi':native_doi,'title':plain(' '.join(p['title'])),
            'issns':p['issn'],'journal_titles':p['journal_titles'],'kind':kind,
            'online_date':online,'date_precision':precision,'publishers':p.get('publisher',[]),
            'categories':p['categories'],'publication_dates':imprint_dates(p['dates']),
            **native_abstract_scope(p, kind),
            **({'native_section_type_evidence':section_evidence} if section_evidence else {}),
            'receipt':record['raw_receipt'],'url':record['url']}]
    family = 'nature_page' if record['route'] == 'nature' else 'publisher_' + record['route']
    return [{'family':family,'title':plain((p['title'] or [''])[0]),
        'issns':p['issns'],'journal_titles':p['journal_titles'],'kind':kind,'online_date':p['online_date'],
        'date_precision':p['date_precision'],'publishers':p.get('publisher',[]),'categories':p.get('evidence_categories', p['article_types']),
        **({'publication_dates':p['publication_dates']} if p.get('publication_dates') else {}),
        'receipt':record.get('raw_receipt',''),'url':record['url']} for kind in p['article_types'] or ['unspecified']]


def retained_annual_evidence(old: dict, year: int, label: str) -> list[str]:
    """Keep still-applicable annual rule citations when eligibility is replayed."""
    if old.get('online_year') != year or old.get('nature_portfolio_at_verified_online_year') != label:
        return []
    retained = []
    suffix = '#' + old['source_id'] + ':' + str(year)
    for reference in json.loads(old.get('portfolio_evidence_ids') or '[]'):
        if not reference.endswith(suffix):
            continue
        path = ROOT / reference[:-len(suffix)]
        if not path.is_file() or path.suffix != '.json':
            continue
        rules = json.loads(path.read_text())
        if isinstance(rules, list) and any(
                r.get('source_id') == old['source_id'] and r.get('year') == year
                and r.get('label') == label for r in rules):
            retained.append(reference)
    return retained


def updated_decision(old: dict, items: list[dict], secondary: list[dict], annual: dict) -> dict:
    value=decide(items,secondary,old['duplicate_of'])
    label,ids=portfolio_label(annual.get((old['source_id'],value['online_year']),''),items,value['online_year'])
    if annual.get((old['source_id'], value['online_year'])) == label and label in {'True', 'False'}:
        ids = list(dict.fromkeys(ids + retained_annual_evidence(old, value['online_year'], label)))
    if (label == 'False' and old.get('online_year') == value['online_year']
            and old.get('nature_portfolio_at_verified_online_year') == label):
        # Annual agreement must not discard still-valid dated primary citations.
        source_label, source_ids = portfolio_label('', items, value['online_year'])
        if source_label == label:
            retained = [i for i in json.loads(old.get('portfolio_evidence_ids') or '[]')
                        if i in source_ids]
            ids = list(dict.fromkeys(ids + retained))
    return {**old,'online_date':value['online_date'],'online_year':value['online_year'],'publisher_types':value['type'],
        'eligibility_status':value['status'],'reason':value['reason'],'evidence_ids':json.dumps(value['evidence_ids']),
        'rejected_evidence_ids':json.dumps(value.get('rejected_evidence_ids',[])),
        'nature_portfolio_at_verified_online_year':label,'portfolio_evidence_ids':json.dumps(ids)}


def apply_available(root: Path = ROOT, supplied_jobs: list[tuple[str, str]] | None = None,
                    author_jobs_out: list[tuple[str, str, str]] | None = None) -> dict:
    if root == ROOT and supplied_jobs is None:
        from .cached_date_apply import consume_pending
        consume_pending()
    folder=root/'supplement';q=root/'fields/qualification';stats=Counter();transitions=Counter()
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root)
        if supplied_jobs is None:
            if 'applied' not in {r[1] for r in db.execute('PRAGMA table_info(publisher_remaining_jobs)')}:
                db.execute('ALTER TABLE publisher_remaining_jobs ADD COLUMN applied INTEGER DEFAULT 0');db.commit()
            jobs=list(db.execute("SELECT doi,receipt FROM publisher_remaining_jobs WHERE applied=0 AND status='parsed' ORDER BY doi"))
        else:
            db.execute('CREATE TABLE IF NOT EXISTS supplied_publisher_processed(doi TEXT,receipt TEXT,PRIMARY KEY(doi,receipt))')
            jobs=[(doi, receipt) for doi, receipt in supplied_jobs if not db.execute(
                'SELECT 1 FROM supplied_publisher_processed WHERE doi=? AND receipt=?', (doi, receipt)).fetchone()]
            if len({doi for doi, _ in jobs}) != len(jobs):
                raise ValueError('One supplied source receipt per DOI per phase is required')
        if not jobs:db.close();return {'applied_source_records':0}
        evidence_index(db,q)
        sources={r['source_id']:r for r in read_csv(root/'fields/journal_registry.csv')}
        annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(root/'fields/journal_publisher_by_year.csv')}
        grouped=defaultdict(dict);records={};additions=[]
        for doi,receipt in jobs:
            record=json.loads((root/receipt).read_text());records[doi]=(record,receipt)
            candidates=[]
            for wid,title,sid,journal,shard in db.execute('SELECT work_id,title,source_id,journal,shard FROM papers WHERE doi=?',(doi,)):
                paper={'work_id':wid,'doi':doi,'title':title,'source_id':sid,'journal_name':journal}
                items=[match(paper,item,sources[sid]) for item in publisher_items(record)]
                candidates.append((shard,paper,items))
            ambiguous=sum(any(e['identity']=='matched' for e in items) for _,_,items in candidates)>1
            for shard,paper,items in candidates:
                if ambiguous:
                    items=[match(paper,{**item,'scope_review_required':True,'scope_review_note':'Multiple current Work identities match the same DOI/title/journal; canonical identity requires review'},sources[paper['source_id']]) for item in publisher_items(record)]
                grouped[shard][paper['work_id']]=(paper,items)
        changes={};new_qualified=set();abstracts={}
        for shard,selected in grouped.items():
            rows=pq.read_table(q/'decisions'/shard).to_pylist();changed=False
            for old in rows:
                if old['work_id'] not in selected:continue
                paper,new=selected[old['work_id']];wid=old['work_id']
                prior={};secondary=[]
                for kind,payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?',(wid,)):
                    if kind=='primary':
                        value=json.loads(payload);prior[value['id']]=value
                    else:secondary=json.loads(payload)
                for item in new:
                    if item['id'] not in prior:
                        item={'work_id':wid,'doi':paper['doi'],**item};additions.append(item);prior[item['id']]=item
                updated=updated_decision(old,list(prior.values()),secondary,annual)
                if updated!=old:
                    changes[wid]={'before':dict(old),'after':updated};transitions[old['eligibility_status']+' -> '+updated['eligibility_status']]+=1
                    old.update(updated);changed=True
                if old['eligibility_status']=='included_verified_research':
                    new_qualified.add(wid);record,receipt=records[paper['doi']]
                    if abstract_ok(record['parsed'].get('abstract','')):abstracts[wid]=(record['parsed']['abstract'],receipt,record['raw_receipt'])
            if changed:grouped[shard]['_decision_rows']=rows
        # Append evidence before any decision refers to it. Raw publisher receipts remain authoritative.
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as stream, (q/'curated_evidence.jsonl').open('a') as curated:
            for item in additions:
                stream.write(json.dumps(item,ensure_ascii=False)+'\n')
                if item.get('scope_review_required'):curated.write(json.dumps(item,ensure_ascii=False)+'\n')
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',(item['work_id'],item['id'],'primary',json.dumps(item,ensure_ascii=False)))
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',('primary',stamp(q/'decision_evidence.jsonl.gz')));db.commit()
        for shard,selected in grouped.items():
            if '_decision_rows' in selected:write_decisions(q/'decisions'/shard,selected.pop('_decision_rows'),q/'decision_changes.jsonl.gz')
            selected_ids=set(selected)&new_qualified
            if not selected_ids:continue
            path=root/'fields/papers'/shard;table=pq.read_table(path);rows=selected_rows(table,selected_ids);patches={}
            for paper in rows:
                if paper['work_id'] not in selected_ids:continue
                if not paper['abstract'] and paper['work_id'] in abstracts:
                    value,receipt,raw=abstracts[paper['work_id']]
                    updates={'abstract':value,'abstract_source':'matched_publisher:fields/qualification/'+raw,'after_abstract':1}
                    for field,new in updates.items():
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(paper['work_id'],field,json.dumps(paper[field]),json.dumps(new),receipt,'Exact publisher DOI/title/journal match; fill empty abstract only'))
                        paper[field]=new
                    patches[paper['work_id']]=updates;stats['abstracts_filled']+=1
                values=metrics(paper)
                db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',(not values['abstract'],not values['every_author_institution'],not values['every_author_country'],paper['work_id']))
                author_rows=json.loads(paper['authors_json'])
                if not values['abstract'] or not values['every_author_institution'] or not values['every_author_country'] or paper['author_list_may_be_truncated'] or any(not a.get('nature_v1_slots') and not a.get('publisher_affiliation_slots') for a in author_rows):
                    task(db,'metadata',paper['work_id'],'pmc_or_publisher',{'doi':paper['doi'],'reason':'New publisher evidence qualified paper; authorship/abstract checks remain','source_id':paper['source_id']},paper['work_id'])
            if patches:
                temporary=path.with_suffix('.parquet.tmp');pq.write_table(patch_rows(table,patches),temporary,compression='zstd');temporary.replace(path)
            db.commit()  # Release acquisition bookkeeping writers between completed shards.
        for wid,change in changes.items():
            row=change['after'];resolved=row['eligibility_status'].startswith(('included','excluded'))
            db.execute('UPDATE papers SET status=? WHERE work_id=?',(row['eligibility_status'],wid))
            db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",('resolved' if resolved else 'pending',row['reason'],row['doi']))
        if supplied_jobs is None:
            db.executemany('UPDATE publisher_remaining_jobs SET applied=1 WHERE doi=?',[(doi,) for doi,_ in jobs])
        else:
            db.executemany('INSERT INTO supplied_publisher_processed VALUES(?,?)', jobs)
        if author_jobs_out is not None:
            for wid in sorted(new_qualified):
                doi, shard = db.execute('SELECT doi,shard FROM papers WHERE work_id=?', (wid,)).fetchone()
                record, receipt = records[doi]
                if record.get('route') in {'springer', 'copernicus', 'pmc_archive_priority', 'frontiers'}:
                    author_jobs_out.append((wid, shard, receipt))
        db.commit();db.close()
    result={'applied_source_records':len(jobs),'changed_decisions':len(changes),'appended_evidence_records':len(additions),'transitions':dict(transitions),'derived_refresh_required':True,**stats}
    with (folder/'publisher_delta_events.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    atomic_json(folder/'publisher_delta_status.json',result);print(json.dumps(result),flush=True);return result


def acquisition_chain_live() -> bool:
    """Keep the consumer alive while the fixed extension dispatcher owns its lock."""
    for relative in ('springer_remaining/collector.lock', 'pleiades_route/dispatcher.lock'):
        path = ROOT / 'supplement' / relative
        if not path.exists():
            continue
        with path.open('a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def unprocessed_author_jobs(db: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Catch later date/type qualifications using the receipt index, not file rereads."""
    return db.execute("SELECT p.work_id,p.shard,j.receipt FROM papers p "
        "JOIN publisher_remaining_jobs j USING(doi) WHERE p.status='included_verified_research' "
        "AND j.applied=1 AND j.status='parsed' "
        "AND j.route IN ('springer','copernicus','pmc_archive_priority','frontiers') "
        "AND NOT EXISTS (SELECT 1 FROM springer_author_processed a "
        "WHERE a.work_id=p.work_id AND a.receipt=j.receipt)").fetchall()


def follow(recover_cache: bool = True) -> None:
    """Process growing evidence in phases; never repeat previously applied source jobs."""
    from .springer_author_cache import run as apply_author_cache
    # One recovery pass catches interrupted qualification-before-byline applications.
    # Subsequent batches only inspect their own qualified receipt list.
    recovered = apply_author_cache(workers=8, fill_abstract=True) if recover_cache else {}
    if recovered.get('papers_changed'):
        from .official_rule_reuse import run as reuse_entities
        reuse_entities()
        refresh()
    while True:
        db=connect(ROOT)
        columns={r[1] for r in db.execute('PRAGMA table_info(publisher_remaining_jobs)')}
        condition=' AND applied=0' if 'applied' in columns else ''
        waiting=db.execute("SELECT count(*) FROM publisher_remaining_jobs WHERE status='parsed'"+condition).fetchone()[0];db.close()
        state=json.loads((ROOT/'supplement/springer_remaining/status.json').read_text())
        terminal=(state['status'] not in ['running','draining_inflight']
                  and not acquisition_chain_live())
        if waiting>=5000 or (terminal and waiting):
            author_jobs = []
            apply_available(author_jobs_out=author_jobs)
            db = connect(ROOT)
            author_jobs = list(dict.fromkeys(author_jobs + unprocessed_author_jobs(db)))
            db.close()
            authors = (apply_author_cache(selected_jobs=author_jobs, workers=8, fill_abstract=True)
                       if author_jobs else {})
            if authors.get('papers_changed'):
                from .official_rule_reuse import run as reuse_entities
                reuse_entities()
            refresh()
        if terminal:return
        release_batch_memory()
        time.sleep(60)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true');args=parser.parse_args()
    if args.follow:
        apply_available()
        follow()
    else:apply_available()
