"""Exclude proven native multilingual aliases while retaining original candidate rows."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .derived_refresh import refresh
from .fig8b_qualification_build import key, plain, write_decisions
from .ojs_cached_evidence import NS, entries
from .ojs_native_dois import native_dois
from .publisher_delta_apply import updated_decision
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/missing_doi'
    reviews = json.loads((folder / 'doi_conflict_review.json').read_text())['records']
    q = ROOT / 'fields/qualification'
    rule_path = q / 'native_duplicate_aliases.json'
    stats = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        rules = json.loads(rule_path.read_text())['records'] if rule_path.exists() else {}
        previous_rules = dict(rules)
        selected = defaultdict(dict)
        for review in reviews:
            if review['review_status'] != 'same_native_paper_candidate_alias':
                continue
            wid = review['work_id']
            aliases = db.execute('SELECT title,source_id,shard,doi FROM papers WHERE work_id=?', (wid,)).fetchall()
            canonical = db.execute('SELECT work_id,title,source_id,status FROM papers WHERE doi=?', (review['doi'],)).fetchall()
            if len(aliases) != 1 or len(canonical) != 1:
                stats['identity_changed_held'] += 1
                continue
            title, sid, shard, doi = aliases[0]
            retained, other_title, other_sid, status = canonical[0]
            info = {'raw_receipt': review['receipt'], 'identifier': review['native_identifier']}
            entry = entries(review['receipt']).get(review['native_identifier'])
            titles = {key(plain(t)) for t in entry.xpath('.//dc:title/text()', namespaces=NS)} if entry is not None else set()
            if sid != other_sid or doi or retained == wid or status == 'excluded_duplicate_wrong_journal' or native_dois(info) != {review['doi']} or not {key(title), key(other_title)} <= titles:
                stats['identity_changed_held'] += 1
                continue
            proof = {'work_id': wid, 'retained_work_id': retained, 'doi': review['doi'], 'source_id': sid,
                     'receipt': '../../' + review['receipt'], 'url': review['source_url'],
                     'native_identifier': review['native_identifier'], 'native_titles': review['native_titles'],
                     'reason': 'One native publisher record declares DOI and both exact titles in the same journal; original DOI-bearing Work retained; no machine translation or title fuzziness',
                     'original_candidate_doi_retained': True}
            if wid in rules and rules[wid] != proof:
                raise ValueError('Conflicting canonical identity rule')
            rules[wid] = proof
            selected[shard][wid] = proof
        if rules != previous_rules or not rule_path.exists():
            atomic_json(rule_path, {'status': 'reviewed', 'records': rules})
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard, aliases in selected.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            changed = False
            for row in rows:
                wid = row['work_id']
                if wid not in aliases:
                    continue
                proof = aliases[wid]
                if row['duplicate_of'] == proof['retained_work_id'] and row['eligibility_status'] == 'excluded_duplicate_wrong_journal':
                    stats['already_applied'] += 1
                    continue
                if row['duplicate_of'] and row['duplicate_of'] != proof['retained_work_id']:
                    raise ValueError('Conflicting persisted duplicate mapping')
                row['duplicate_of'] = proof['retained_work_id']
                evidence = [json.loads(p) for p, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,))]
                row.update(updated_decision(row, evidence, [], annual))
                row['reason'] = proof['reason']
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], wid))
                for tid, in db.execute("SELECT t.task_id FROM tasks t JOIN task_papers l USING(task_id) WHERE t.kind='qualification' AND l.work_id=?", (wid,)).fetchall():
                    attempt(db, tid, 'native_publisher_duplicate_identity', 'resolved', 'fields/qualification/native_duplicate_aliases.json', proof)
                stats['duplicates_excluded'] += 1
                changed = True
            if changed:
                write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            db.commit()
        db.close()
    result = {'status': 'native_aliases_excluded_original_rows_preserved', **stats}
    atomic_json(folder / 'duplicate_apply_status.json', result)
    if stats['duplicates_excluded']:
        refresh()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
