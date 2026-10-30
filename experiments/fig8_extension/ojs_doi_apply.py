"""Fill unique publisher-declared DOIs and migrate qualification task keys."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .fig8b_qualification_build import match, write_decisions
from .ojs_cached_evidence import dc_evidence
from .ojs_native_dois import native_dois
from .supplement_queue import ROOT, attempt, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(proposals: list[dict] | None = None) -> dict:
    folder = ROOT / 'supplement/missing_doi'
    if proposals is None:
        proposals = json.loads((folder / 'native_doi_proposals.json').read_text())['records']
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    stats = Counter()
    grouped = defaultdict(dict)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        for proposal in proposals:
            if proposal['status'] != 'native_doi_identity_matched_pending_apply':
                stats[proposal['status']] += 1
                continue
            wid, doi = proposal['work_id'], proposal['doi']
            current = db.execute('SELECT doi,title,journal,source_id,shard,route FROM papers WHERE work_id=?', (wid,)).fetchone()
            if not current or current[0] not in (None, '', doi):
                stats['current_identity_conflict'] += 1
                continue
            if db.execute('SELECT 1 FROM papers WHERE doi=? AND work_id<>?', (doi, wid)).fetchone():
                stats['new_doi_collision_held'] += 1
                continue
            info = {'raw_receipt': proposal['receipt'], 'identifier': proposal['native_identifier']}
            paper = {'work_id': wid, 'title': current[1], 'journal_name': current[2], 'source_id': current[3]}
            if native_dois(info) != {doi} or not any(match(paper, e, journals[current[3]])['identity'] == 'matched'
                                                    for e in dc_evidence(info, proposal['source_url'], current[1])):
                stats['native_identity_not_confirmed'] += 1
                continue
            grouped[current[4]][wid] = {**proposal, 'route': current[5], 'previous_doi': current[0]}
        for shard, selected in grouped.items():
            main_path = ROOT / 'fields/papers' / shard
            decision_path = ROOT / 'fields/qualification/decisions' / shard
            table = pq.read_table(main_path)
            papers = table.to_pylist()
            decisions = pq.read_table(decision_path).to_pylist()
            for rows in (papers, decisions):
                for row in rows:
                    if row['work_id'] in selected and row['doi'] not in (None, '', selected[row['work_id']]['doi']):
                        raise ValueError('Current shard DOI conflicts with validated proposal')
            with (folder / 'doi_changes.jsonl').open('a') as audit:
                for row in papers:
                    if row['work_id'] not in selected:
                        continue
                    value = selected[row['work_id']]
                    if row['doi'] != value['doi']:
                        audit.write(json.dumps({**value, 'before': row['doi'], 'after': value['doi'],
                            'reason': 'Unique native publisher DOI identifier; exact article title and journal; qualification unchanged'}) + '\n')
                    row['doi'] = value['doi']
            for row in decisions:
                if row['work_id'] in selected:
                    row['doi'] = selected[row['work_id']]['doi']
            write_decisions(decision_path, decisions, ROOT / 'fields/qualification/decision_changes.jsonl.gz')
            temporary = main_path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(papers, schema=table.schema), temporary, compression='zstd')
            temporary.replace(main_path)
            for wid, value in selected.items():
                doi = value['doi']
                old_id = hashlib.sha256(('qualification:' + wid).encode()).hexdigest()[:32]
                old = db.execute('SELECT payload,status FROM tasks WHERE task_id=?', (old_id,)).fetchone()
                payload = json.loads(old[0]) if old else {}
                payload.update(doi=doi, work_id=wid, doi_evidence=value['receipt'])
                new_id = task(db, 'qualification', doi, value['route'], payload, wid)
                db.execute('UPDATE papers SET doi=? WHERE work_id=?', (doi, wid))
                if old:
                    retained = {**json.loads(old[0]), 'work_id': wid, 'superseded_by': new_id}
                    db.execute("UPDATE tasks SET active=0,payload=?,reason=? WHERE task_id=?",
                               (json.dumps(retained), 'Qualification key migrated from Work ID to recovered DOI; attempts retained', old_id))
                    db.execute('DELETE FROM task_papers WHERE task_id=? AND work_id=?', (old_id, wid))
                if not db.execute("SELECT 1 FROM attempts WHERE task_id=? AND source='native_ojs_doi' AND receipt=?", (new_id, value['receipt'])).fetchone():
                    attempt(db, new_id, 'native_ojs_doi', 'pending', value['receipt'],
                            {'reason': 'DOI recovered; original research type and online date still require evidence',
                             'work_id': wid, 'native_identifier': value['native_identifier'], 'previous_task_id': old_id})
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (wid, 'doi', json.dumps(value['previous_doi']), json.dumps(doi), value['receipt'], 'Exact native publisher identifier; unique within current candidates'))
                    stats['dois_filled'] += 1
                else:
                    stats['already_applied'] += 1
            db.commit()
        db.close()
    result = {'status': 'identifiers_updated_qualification_still_requires_evidence', **stats}
    atomic_json(folder / 'doi_apply_status.json', result)
    refresh()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
