"""Refresh current metadata/slot task links from changed qualified shards only."""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from . import metadata_task_delta
from .fig8b_qualification_recompute import metrics
from .publisher_delta_apply import stamp
from .supplement_allocation import known_universities, official_relationships, slots_for
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json

KINDS=('metadata','institution','country_conflict','portfolio','university_ownership')


def advance_annual_only_sync(db: sqlite3.Connection, shard: str, before: str,
                             after: str, decisions: list[dict]) -> bool:
    """Skip author relinking only when a fully synced shard changed annual labels alone."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata_task_shards'").fetchone():
        return False
    current = db.execute('SELECT stamp FROM metadata_task_shards WHERE shard=?', (shard,)).fetchone()
    if current != (before,):
        return False  # Prior metadata/eligibility changes must still receive the full sync.
    resolved = [(r['work_id'],) for r in decisions
                if r['nature_portfolio_at_verified_online_year'] in {'True', 'False'}]
    db.executemany("DELETE FROM task_papers WHERE work_id=? AND EXISTS "
                   "(SELECT 1 FROM tasks WHERE tasks.task_id=task_papers.task_id "
                   "AND tasks.kind='portfolio')", resolved)
    db.execute('UPDATE metadata_task_shards SET stamp=? WHERE shard=?', (after, shard))
    return True



def advance_nonqualified_only_sync(db: sqlite3.Connection, shard: str, before: str,
                                   after: str, eligible_changed: bool) -> bool:
    """Keep synchronized links when only nonqualified decisions have changed."""
    if eligible_changed:
        return False
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata_task_shards'").fetchone():
        return False
    cursor = db.execute('UPDATE metadata_task_shards SET stamp=? WHERE shard=? AND stamp=?',
                        (after, shard, before))
    return cursor.rowcount == 1

def requests_for(paper: dict, decision: dict, known: dict | None = None,
                 relationships: dict | None = None) -> list[tuple]:
    wid=paper['work_id'];authors=json.loads(paper['authors_json']);requests=[];values=metrics(paper)
    missing=[k for k in ['abstract','every_author_institution','every_author_country'] if not values[k]]
    inherited=bool(authors) and all(a.get('nature_v1_slots') is not None or a.get('nonpersonal_verified') for a in authors)
    if paper['author_list_may_be_truncated']:missing.append('complete_byline')
    for author in authors:
        if author.get('nonpersonal_verified'):continue
        position=author['position'];entity=wid+':'+str(position)
        for flag,suffix in [('country_conflict_retained',''),('secondary_country_conflict',':secondary')]:
            if author.get(flag) and not inherited:
                requests.append(('country_conflict',entity+suffix,'publisher_signature_then_official',{'countries':author['countries'],'source':author.get('publisher_affiliation_source'),'reason':flag},position))
        if author.get('nature_v1_slots') is not None:continue  # Existing completed Nature adjudication is not reopened.
        slots=author.get('publisher_affiliation_slots') or []
        if slots:
            allocated = slots_for(author, 'main_personal_direct',
                                  known if known is not None else {'main_personal_direct': set()}, relationships)
            for slot, allocation in zip(slots, allocated):
                if not allocation['unknown_university']:continue
                for ident in sorted(set(slot.get('institution_ids') or [])):
                    requests.append(('university_ownership',ident,'publisher_signature_and_official_relationship',
                        {'entity_id':ident,'name':slot.get('text',''),'source':author.get('publisher_affiliation_source'),
                         'online_year':decision['online_year'],'reason':'Institution identity observed but main-scenario university allocation remains unknown',
                         'basis':'individual_publisher_affiliation_slot','no_parent_inference':True},position))
            for slot in slots:
                identity_missing=not slot.get('institution_ids')
                country_missing=not slot.get('country_codes')
                if not identity_missing and not country_missing:continue
                name=slot.get('text') or '[missing publisher affiliation]'
                requests.append(('institution',' '.join(name.casefold().split()) if name!='[missing publisher affiliation]' else entity+':missing_affiliation','publisher_signature_then_official',{'name':name,'missing_identity':identity_missing,'missing_country':country_missing,'basis':'individual_publisher_affiliation_slot','source':author.get('publisher_affiliation_source')},position))
                missing.append('unresolved_publisher_affiliation_slot')
        else:
            missing.append('publisher_affiliation_structure')
            if author['institution_ids_or_verified_local_entities'] and author['countries']:continue
            names=author.get('raw_affiliation_strings') or author.get('supplementary_addresses') or ['[missing publisher affiliation]']
            if isinstance(names,str):names=[names]
            for name in names:
                entity_key=' '.join(name.casefold().split()) if name!='[missing publisher affiliation]' else wid+':missing_affiliation:'+str(position)
                requests.append(('institution',entity_key,'publisher_signature_then_official',{'name':name,'missing_identity':not author['institution_ids_or_verified_local_entities'],'missing_country':not author['countries'],'basis':'unverified_address_requires_primary_signature'},position))
    if missing:
        requests.append(('metadata',wid,'reuse_nature_final_review' if inherited else 'pmc_or_publisher',{'doi':paper['doi'],'source_id':paper['source_id'],'missing_fields':sorted(set(missing)),'inherited_nature_review_preserved':inherited},-1))
    if decision['nature_portfolio_at_verified_online_year'] not in {'True','False'}:
        requests.append(('portfolio',paper['source_id']+':'+str(decision['online_year']),'dated_publisher',{'source_id':paper['source_id'],'year':decision['online_year'],'journal':paper['journal_name']},-1))
    return requests


def run(root: Path = ROOT, lock_held: bool = False) -> dict:
    folder=root/'supplement';stats=Counter()
    with (folder/'checkpoint.lock').open('a') as lock:
        if not lock_held:fcntl.flock(lock,fcntl.LOCK_EX)
        db=connect(root)
        known=known_universities(root);relationships=official_relationships(root)
        db.execute('CREATE TABLE IF NOT EXISTS metadata_task_shards(shard TEXT PRIMARY KEY,stamp TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS task_link_requirements(task_id TEXT,work_id TEXT,position INTEGER,missing_identity INTEGER,missing_country INTEGER,receipt TEXT,PRIMARY KEY(task_id,work_id,position))')
        db.execute('CREATE INDEX IF NOT EXISTS task_requirements_work ON task_link_requirements(work_id)')
        db.execute('CREATE INDEX IF NOT EXISTS paper_shard_work ON papers(shard,work_id)');db.commit()
        seen=dict(db.execute('SELECT shard,stamp FROM metadata_task_shards'))
        metadata_task_delta.ensure_table(db)
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            decisions=root/'fields/qualification/decisions'/path.name;signature='ownership-queue-1:'+stamp(path)+':'+stamp(decisions)
            if seen.get(path.name)==signature:stats['unchanged_shards_skipped']+=1;continue
            selected = metadata_task_delta.selected_ids(db, path.name, seen.get(path.name), signature)
            included={r['work_id']:r for r in pq.read_table(decisions,filters=[('eligibility_status','=','included_verified_research')]).to_pylist()}
            if selected is None:
                db.execute('DELETE FROM task_link_requirements WHERE work_id IN (SELECT work_id FROM papers WHERE shard=?)',(path.name,))
                db.execute("DELETE FROM task_papers WHERE work_id IN (SELECT work_id FROM papers WHERE shard=?) AND EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=task_papers.task_id AND t.kind IN ('metadata','institution','country_conflict','portfolio','university_ownership'))",(path.name,))
                batches = pq.ParquetFile(path).iter_batches(batch_size=500)
            else:
                db.executemany('DELETE FROM task_link_requirements WHERE work_id=?', [(w,) for w in selected])
                db.executemany("DELETE FROM task_papers WHERE work_id=? AND EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=task_papers.task_id AND t.kind IN ('metadata','institution','country_conflict','portfolio','university_ownership'))", [(w,) for w in selected])
                batches = pq.read_table(path, filters=[('work_id', 'in', sorted(selected))]).to_batches(max_chunksize=500)
                stats['exact_delta_shards_processed'] += 1
                stats['exact_delta_papers_selected'] += len(selected)
            db.commit()
            for batch in batches:
                pending_writes=0
                for paper in batch.to_pylist():
                    wid=paper['work_id']
                    if wid not in included:continue
                    stats['qualified_papers_assessed']+=1
                    for kind,entity,route,payload,position in requests_for(paper,included[wid],known,relationships):
                        ident=task(db,kind,entity,route,payload,wid,position);stats[kind+'_task_candidates']+=1
                        if kind=='institution':
                            db.execute('INSERT INTO task_link_requirements VALUES(?,?,?,?,?,?) ON CONFLICT(task_id,work_id,position) DO UPDATE SET missing_identity=max(missing_identity,excluded.missing_identity),missing_country=max(missing_country,excluded.missing_country),receipt=excluded.receipt',
                                (ident,wid,position,payload['missing_identity'],payload['missing_country'],payload.get('source') or ('fields/'+paper['raw_receipt'] if paper.get('raw_receipt') else '')))
                        pending_writes+=1
                        if pending_writes>=1000:db.commit();pending_writes=0
                db.commit()

            db.execute('INSERT OR REPLACE INTO metadata_task_shards VALUES(?,?)',(path.name,signature));db.commit();stats['changed_shards_processed']+=1
            db.execute('DELETE FROM metadata_task_deltas WHERE shard=?', (path.name,));db.commit()
            atomic_json(folder/'metadata_task_sync_status.json',{'status':'running','shard':path.name,**stats})
        db.execute("UPDATE tasks SET active=EXISTS(SELECT 1 FROM task_papers l WHERE l.task_id=tasks.task_id) WHERE kind IN ('metadata','institution','country_conflict','portfolio','university_ownership')")
        db.execute("UPDATE tasks SET status='pending',reason='Entity evidence retained; main-scenario university assignment remains unknown' WHERE kind='university_ownership' AND active=1 AND status='resolved'")
        db.execute("UPDATE tasks SET payload=json_set(payload,'$.missing_identity',coalesce((SELECT max(r.missing_identity) FROM task_link_requirements r WHERE r.task_id=tasks.task_id),0),'$.missing_country',coalesce((SELECT max(r.missing_country) FROM task_link_requirements r WHERE r.task_id=tasks.task_id),0)) WHERE kind='institution' AND active=1")
        db.execute("UPDATE tasks SET status='pending',reason='Entity evidence retained; current paper slots still require assignment' WHERE kind='institution' AND active=1 AND status='resolved' AND EXISTS(SELECT 1 FROM task_link_requirements r WHERE r.task_id=tasks.task_id AND (r.missing_identity=1 OR r.missing_country=1))")
        db.commit();db.close()
    result={'status':'current_links_synchronized_review_outcomes_preserved',**stats};atomic_json(folder/'metadata_task_sync_status.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-held',action='store_true');args=parser.parse_args();run(lock_held=args.lock_held)
