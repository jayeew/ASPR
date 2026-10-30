"""Apply evidence-reviewed repository variants without changing their stored DOIs."""
from __future__ import annotations

import fcntl
import gzip
import json
import shutil
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .derived_refresh import refresh
from .fig8b_qualification_build import match, write_decisions
from .publisher_delta_apply import stamp, updated_decision
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/aps_repository_identity'
    q = ROOT / 'fields/qualification'
    proposed = json.loads((folder / 'dedup_proposals.json').read_text())['pairs']
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    counts, events = Counter(), []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        aliases_path = q / 'native_duplicate_aliases.json'
        aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {'status': 'reviewed', 'records': {}}
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        grouped = defaultdict(dict)
        for pair in proposed:
            old = pair['repository']
            target = pair['target']
            row = db.execute('SELECT title,source_id,shard FROM papers WHERE work_id=?', (old['work_id'],)).fetchone()
            kept = db.execute('SELECT source_id FROM papers WHERE work_id=? AND doi=?', (target['target_work_id'], target['target_doi'])).fetchone()
            if not kept or kept[0] != row[1]:
                counts['journal_conflict_held'] += 1
                continue
            if old['work_id'] in aliases['records']:
                counts['existing_alias_unchanged'] += 1
                continue
            paper = {'work_id': old['work_id'], 'doi': old['doi'], 'title': row[0], 'source_id': row[1], 'journal_name': journals[row[1]]['journal_name']}
            item = {'family': 'repository_version_review', 'kind': 'repository_version', 'title': row[0],
                    'source_id': row[1], 'online_date': '', 'date_precision': 'not_used',
                    'retained_work_id': target['target_work_id'], 'related_doi': target['target_doi'],
                    'relation_type': 'IsVariantFormOf', 'content_identity_confirmed': True,
                    'receipt': '../../' + old['raw_receipt'], 'url': 'https://api.datacite.org/dois/' + old['doi'],
                    'review_receipt': 'supplement/aps_repository_identity/dedup_proposals.json',
                    'target_publisher_evidence_ids': [e['id'] for e in pair['target_publisher_evidence']]}
            evidence = {'work_id': old['work_id'], 'doi': old['doi'], **match(paper, item, journals[row[1]])}
            if evidence['identity'] != 'matched':
                raise ValueError('Repository identity review failed')
            aliases['records'][old['work_id']] = {'doi': old['doi'], 'source_id': row[1],
                'retained_work_id': target['target_work_id'], 'related_doi': target['target_doi'],
                'receipt': item['receipt'], 'url': item['url'], 'review_receipt': item['review_receipt']}
            for path in [q / 'decision_evidence.jsonl.gz', q / 'curated_evidence.jsonl']:
                handle = gzip.open(path, 'at') if path.suffix == '.gz' else path.open('a')
                with handle as stream:
                    stream.write(json.dumps(evidence, ensure_ascii=False) + '\n')
            db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)', (old['work_id'], evidence['id'], 'primary', json.dumps(evidence)))
            grouped[row[2]][old['work_id']] = target['target_work_id']
        for shard, selected in grouped.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                if row['work_id'] not in selected:
                    continue
                before = dict(row)
                primary, secondary = [], []
                for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (row['work_id'],)):
                    if kind == 'primary':
                        primary.append(json.loads(payload))
                    else:
                        secondary = json.loads(payload)
                row['duplicate_of'] = selected[row['work_id']]
                row.update(updated_decision(row, primary, secondary, annual))
                if row['eligibility_status'] != 'excluded_duplicate_repository_version':
                    raise ValueError('Expected explicit repository duplicate disposition')
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status='resolved',reason=? WHERE kind='qualification' AND entity_key=?", (row['reason'], row['doi']))
                events.append({'before': before, 'after': dict(row), 'review': 'supplement/aps_repository_identity/dedup_proposals.json'})
                counts['repository_variants_excluded'] += 1
            write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
        atomic_json(aliases_path, aliases)
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)', ('primary', stamp(q / 'decision_evidence.jsonl.gz')))
        db.commit()
        db.close()
        shutil.copy2(__file__.replace('aps_repository_dedup_apply.py', 'fig8b_qualification_decide.py'), q / 'decide.py')
        with (folder / 'dedup_changes.jsonl').open('a') as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    result = {**counts, 'network_requests': 0}
    atomic_json(folder / 'dedup_status.json', result)
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
