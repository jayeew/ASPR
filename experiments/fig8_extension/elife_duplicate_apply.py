"""Persist publisher-declared canonical/version DOI aliases without deleting rows."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .derived_refresh import refresh
from .elife_version_details import OUT
from .elife_text import plain
from .fig8b_qualification_build import key, write_decisions
from .publisher_delta_apply import updated_decision
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    proposals = json.loads((OUT / 'application/review.json').read_text())['aliases']
    q = ROOT / 'fields/qualification'
    rule_path = q / 'native_duplicate_aliases.json'
    stats: Counter = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        rules = json.loads(rule_path.read_text())['records'] if rule_path.exists() else {}
        selected: dict[str, dict] = defaultdict(dict)
        for proposal in proposals:
            rows = db.execute('SELECT work_id,doi,title,source_id,shard,status FROM papers WHERE work_id IN (?,?)',
                              (proposal['work_id'], proposal['retained_work_id'])).fetchall()
            papers = {r[0]: dict(zip(['work_id', 'doi', 'title', 'source_id', 'shard', 'status'], r)) for r in rows}
            if len(papers) != 2:
                stats['candidate_missing'] += 1
                continue
            alias, retained = papers[proposal['work_id']], papers[proposal['retained_work_id']]
            raw = gzip.decompress((ROOT / proposal['raw_receipt']).read_bytes())
            native = json.loads(raw)
            exact = (alias['doi'] == native.get('doiVersion', '').lower()
                     and retained['doi'] == native.get('doi', '').lower()
                     and alias['source_id'] == retained['source_id'] == 'https://openalex.org/S1336409049'
                     and key(alias['title']) == key(retained['title']) == key(plain(native['title']))
                     and native.get('status') == 'vor' and native.get('stage') == 'published'
                     and not retained['status'].startswith('excluded_duplicate')
                     and retained['work_id'] not in rules)
            if not exact:
                stats['identity_or_canonical_conflict'] += 1
                continue
            proof = {'work_id': alias['work_id'], 'retained_work_id': retained['work_id'],
                     'doi': retained['doi'], 'version_doi': alias['doi'], 'source_id': alias['source_id'],
                     'receipt': '../../' + proposal['raw_receipt'],
                     'raw_sha256': hashlib.sha256(raw).hexdigest(),
                     'url': 'https://api.elifesciences.org/articles/' + native['id'],
                     'reason': 'Publisher v8 explicitly links canonical DOI and doiVersion for the same title and journal; canonical candidate retained',
                     'original_candidate_doi_retained': True}
            existing = rules.get(alias['work_id'])
            if existing and existing != proof:
                stats['existing_alias_rule_conflict'] += 1
                continue
            rules[alias['work_id']] = proof
            selected[alias['shard']][alias['work_id']] = proof
        atomic_json(rule_path, {'status': 'reviewed', 'records': rules})
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
                  for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        for shard, aliases in selected.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            changed = False
            for row in rows:
                proof = aliases.get(row['work_id'])
                if not proof:
                    continue
                if row['duplicate_of'] and row['duplicate_of'] != proof['retained_work_id']:
                    raise ValueError('Conflicting persisted eLife duplicate identity')
                if row['duplicate_of'] == proof['retained_work_id']:
                    stats['already_applied'] += 1
                    continue
                row['duplicate_of'] = proof['retained_work_id']
                row.update(updated_decision(row, [], [], annual))
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status='resolved',reason=?,receipt=? WHERE kind='qualification' AND entity_key=?",
                           (proof['reason'], 'fields/qualification/native_duplicate_aliases.json', row['doi']))
                stats['version_aliases_excluded'] += 1
                changed = True
            if changed:
                write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            db.commit()
        db.close()
    result = dict(stats)
    atomic_json(OUT / 'application/duplicate_status.json', result)
    with (OUT / 'application/duplicate_events.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    if stats['version_aliases_excluded']:
        refresh()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
