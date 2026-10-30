"""Reconcile erroneous inherited PNAS rosters using complete cached publisher bylines."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from . import metadata_task_delta
from .derived_refresh import refresh
from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .metadata_task_sync import KINDS, requests_for
from .springer_author_cache import prepare_source
from .supplement_apply import complete_byline, enrich_authors
from .nature_backfill_resolve import country_names
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/pnas_cached_byline_gaps'


def reconcile(current: list[dict], source: dict) -> tuple[list[dict], str]:
    native = source.get('authors', [])
    counts = Counter(key(a['name']) for a in current)
    declared = Counter(key(a['name']) for a in native)
    if not native or source.get('byline_has_etal') or '' in declared or any(v != 1 for v in declared.values()):
        return current, 'native_byline_incomplete_or_ambiguous'
    if any(a.get('nature_v1_slots') is not None for a in current):
        return current, 'nature_preserved'
    if any(a.get('nonpersonal') for a in native) and any(key(a['name']) not in declared for a in current):
        return current, 'group_membership_requires_full_byline_evidence'
    unique = {key(a['name']): a for a in current if counts[key(a['name'])] == 1}
    for author in current:
        if author.get('publisher_affiliation_slots') and (
                counts[key(author['name'])] != 1 or key(author['name']) not in declared
                or author.get('publisher_affiliation_source') != source['evidence_receipt']):
            return current, 'existing_primary_identity_conflict'
    fresh, complete = complete_byline([], source)
    assert complete
    output = []
    for position, author in enumerate(fresh):
        preserved = unique.get(key(author['name']))
        record = json.loads(json.dumps(preserved if preserved else author))
        record['position'] = position
        output.append(record)
    return output, 'publisher_roster_reconciled'


def run(defer_task_rebuild: bool = False) -> None:
    scope = json.loads((FOLDER / 'duplicate_byline_evidence.json').read_text())['records']
    jobs = defaultdict(dict)
    for r in scope:
        jobs[r['shard']][r['work_id']] = r
    journal = {r['source_id']: set(json.loads(r['issn'] or '[]'))
               for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    registry, by_name = defaultdict(list), defaultdict(list)
    for r in json.loads((ROOT / 'fields/institution_registry.json').read_text()):
        if r.get('ror'):
            registry[r['ror'].rstrip('/')].append(r['id'])
        by_name[key(r['display_name'])].append(r)
    names = country_names()
    for value in ['korea', 'georgia', 'congo']:
        names.pop(value, None)
    counts, results = Counter(), []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for shard, selected in jobs.items():
            path = ROOT / 'fields/papers' / shard
            decision_path = ROOT / 'fields/qualification/decisions' / shard
            before_signature = metadata_task_delta.signature(path, decision_path)
            changed_ids = set()
            table = pq.read_table(path)
            rows = table.to_pylist()
            decisions = {r['work_id']: r for r in pq.read_table(ROOT / 'fields/qualification/decisions' / shard).to_pylist()}
            changed = False
            for paper in rows:
                wid = paper['work_id']
                if wid not in selected or decisions[wid]['eligibility_status'] != 'included_verified_research':
                    continue
                r = selected[wid]
                receipt, _, native = prepare_source(ROOT, paper, r['receipt'], journal[paper['source_id']],
                    aps_only=r['receipt'].startswith('fields/qualification/aps_all_raw/'))
                source = {**native, 'evidence_receipt': 'fields/qualification/' + receipt['raw_receipt']}
                current = json.loads(paper['authors_json'])
                authors, status = reconcile(current, source)
                if status != 'publisher_roster_reconciled':
                    counts[status] += 1
                    results.append({'work_id': wid, 'status': status, 'receipt': r['receipt']})
                    continue
                authors, delta = enrich_authors(authors, source, registry, by_name, names)
                before = dict(paper)
                paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
                paper['author_list_may_be_truncated'] = False
                paper['authorships_source_receipt'] = source['evidence_receipt']
                paper['full_authorships_status'] = 'publisher_complete_roster_reconciled_unmatched_person_ids_unknown'
                values = metrics(paper)
                for metric, value in values.items():
                    if 'after_' + metric in paper:
                        paper['after_' + metric] = value
                fields = [f for f in paper if paper[f] != before[f]]
                if not fields:
                    counts['unchanged'] += 1
                    continue
                for field in fields:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (wid, field, json.dumps(before[field]), json.dumps(paper[field]), source['evidence_receipt'],
                                'Complete matched native byline replaces differing inherited roster; unmatched person IDs remain unknown; unique existing primary links preserved'))
                # Position-specific review outcomes must never transfer to different people.
                db.execute("UPDATE tasks SET status='pending',reason='Author roster reconciled; positional review invalidated' WHERE kind='country_conflict' AND task_id IN (SELECT task_id FROM task_papers WHERE work_id=?)", (wid,))
                db.execute('DELETE FROM task_link_requirements WHERE work_id=?', (wid,))
                placeholders = ','.join('?' for _ in KINDS)
                db.execute(f'DELETE FROM task_papers WHERE work_id=? AND EXISTS (SELECT 1 FROM tasks t WHERE t.task_id=task_papers.task_id AND t.kind IN ({placeholders}))', (wid, *KINDS))
                if defer_task_rebuild:
                    # The phase finalizer rebuilds all changed shards exactly once.
                    # Old position links were invalidated above and must not be reused.
                    counts['task_rebuild_deferred_papers'] += 1
                else:
                    for kind, entity, route, payload, position in requests_for(paper, decisions[wid]):
                        ident = task(db, kind, entity, route, payload, wid, position)
                        if kind == 'institution':
                            db.execute('INSERT OR REPLACE INTO task_link_requirements VALUES(?,?,?,?,?,?)',
                                       (ident, wid, position, payload['missing_identity'], payload['missing_country'], payload.get('source', '')))
                db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',
                           (not values['abstract'], not values['every_author_institution'], not values['every_author_country'], wid))
                changed_ids.add(wid)
                counts['papers_repaired'] += 1
                counts.update(delta)
                results.append({'work_id': wid, 'status': status, 'receipt': source['evidence_receipt'], 'old_count': len(current), 'new_count': len(authors)})
                changed = True
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
                metadata_task_delta.record(db, shard, before_signature,
                    metadata_task_delta.signature(path, decision_path), changed_ids, True)
                db.commit()
            atomic_json(FOLDER / 'roster_progress.json', {
                'status': 'applying_fixed_scope', 'last_completed_shard': shard,
                'counts': dict(counts), 'network_requests': 0, 'final_task_closure': False})
        db.close()
    atomic_json(FOLDER / 'roster_repair.json', {'counts': dict(counts), 'records': results, 'network_requests': 0})
    atomic_json(FOLDER / 'roster_progress.json', {
        'status': 'fixed_scope_applied', 'counts': dict(counts),
        'network_requests': 0, 'final_task_closure': False})
    refresh()
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
