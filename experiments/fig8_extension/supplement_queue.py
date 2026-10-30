"""Current-data supplementation ledger, stratified pilot and append-only attempts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .fig8b_qualification_recompute import metrics

ROOT = Path('outputs/fig8_data')


def route(name: str, doi: str = '') -> str:
    # DOI routing only: does not establish historical publisher/Portfolio membership.
    # Reviewed journal host: supplement/btpr_route/rule.json. The source registry
    # retains an ACS name for this journal; current BTPR papers are hosted by Wiley.
    if doi.strip().casefold().startswith('10.1002/btpr.'):
        return 'wiley'
    if (doi.strip().casefold().startswith('10.1063/')
            or name.strip().casefold() in {'american institute of physics', 'aip publishing'}):
        return 'aip'
    if doi.strip().casefold().startswith('10.1152/') or 'american physiological society' in name.casefold():
        return 'physiology'
    value = name.casefold()
    for label, terms in [('elsevier', ['elsevier']), ('wiley', ['wiley']),
                         ('springer', ['springer', 'biomed central']), ('mdpi', ['multidisciplinary digital', 'mdpi']),
                         ('acs', ['american chemical society']), ('iop', ['iop publishing', 'institute of physics']),
                         ('aps', ['american physical society']), ('frontiers', ['frontiers']),
                         ('oxford', ['oxford university']), ('rsc', ['royal society of chemistry'])]:
        if any(term in value for term in terms):
            return label
    return 'other'


def connect(root: Path = ROOT) -> sqlite3.Connection:
    folder = root / 'supplement'
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(folder / 'tasks.sqlite', timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS papers(work_id TEXT PRIMARY KEY,doi TEXT,field TEXT,year INTEGER,source_id TEXT,
      journal TEXT,title TEXT,publisher TEXT,route TEXT,shard TEXT,status TEXT,pmcids TEXT,pmc_unread INTEGER,
      abstract_gap INTEGER,institution_gap INTEGER,country_gap INTEGER,author_cap INTEGER,priority TEXT);
    CREATE INDEX IF NOT EXISTS paper_doi ON papers(doi);
    CREATE INDEX IF NOT EXISTS paper_route ON papers(route,status);
    CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,kind TEXT,entity_key TEXT,route TEXT,status TEXT,
      reason TEXT,payload TEXT,attempt_n INTEGER DEFAULT 0,receipt TEXT DEFAULT '',updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE INDEX IF NOT EXISTS task_status ON tasks(kind,status,route);
    CREATE INDEX IF NOT EXISTS task_entity ON tasks(kind,entity_key);
    CREATE TABLE IF NOT EXISTS task_papers(task_id TEXT,work_id TEXT,position INTEGER DEFAULT -1,
      PRIMARY KEY(task_id,work_id,position));
    CREATE INDEX IF NOT EXISTS task_papers_work ON task_papers(work_id,task_id);
    CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,source TEXT,
      status TEXT,receipt TEXT,detail TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE INDEX IF NOT EXISTS attempts_task_source_receipt ON attempts(task_id,source,receipt);
    CREATE TABLE IF NOT EXISTS changes(id INTEGER PRIMARY KEY AUTOINCREMENT,work_id TEXT,field TEXT,
      before_value TEXT,after_value TEXT,receipt TEXT,reason TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE INDEX IF NOT EXISTS changes_receipt_field ON changes(receipt,field);
    CREATE TABLE IF NOT EXISTS identities(entity_id TEXT PRIMARY KEY,name TEXT,entity_type TEXT,countries TEXT,
      evidence TEXT,ownership_status TEXT,university_targets TEXT);
    CREATE TABLE IF NOT EXISTS ownership_rules(entity_id TEXT,university_id TEXT,valid_from TEXT,valid_to TEXT,
      fraction REAL,status TEXT,evidence TEXT,PRIMARY KEY(entity_id,university_id,valid_from));
    CREATE TABLE IF NOT EXISTS pilot(doi TEXT PRIMARY KEY,work_id TEXT,arm TEXT,pmcid TEXT,route TEXT,field TEXT,
      year INTEGER,source_id TEXT,selected_at TEXT DEFAULT CURRENT_TIMESTAMP);
    ''')
    if 'active' not in {r[1] for r in db.execute('PRAGMA table_info(tasks)')}:
        db.execute('ALTER TABLE tasks ADD COLUMN active INTEGER DEFAULT 1');db.commit()
    return db


def task(db: sqlite3.Connection, kind: str, entity: str, source: str, payload: dict,
         work_id: str = '', position: int = -1, status: str = 'pending') -> str:
    ident = hashlib.sha256(f'{kind}:{entity}'.encode()).hexdigest()[:32]
    db.execute('INSERT INTO tasks(task_id,kind,entity_key,route,status,reason,payload) VALUES(?,?,?,?,?,?,?) '
               "ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload,route=excluded.route,active=1,status=CASE WHEN excluded.status='resolved' THEN excluded.status ELSE tasks.status END",
               (ident, kind, entity, source, status, '', json.dumps(payload, ensure_ascii=False)))
    if work_id:
        db.execute('INSERT OR IGNORE INTO task_papers VALUES(?,?,?)', (ident, work_id, position))
    return ident


def discover(root: Path = ROOT) -> dict:
    db = connect(root)
    base = root / 'fields'
    with (base / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {(r['cohort'], r['source_id']): r for r in csv.DictReader(stream)}
    known = {p.stem for p in (base / 'qualification/jats_parsed').glob('*.json') if json.loads(p.read_text()).get('status')=='parsed'}
    names: dict[str, dict] = {}
    db.execute('DELETE FROM papers')
    db.execute('UPDATE tasks SET active=0')
    for path in sorted((base / 'papers').glob('*.parquet')):
        decisions = {r['work_id']: r for r in pq.read_table(base / 'qualification/decisions' / path.name).to_pylist()}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
            for row in batch.to_pylist():
                decision = decisions[row['work_id']]
                j = journals[(row['field'], row['source_id'])]
                publisher = j.get('host_name') or ''
                publisher_route = route(publisher, row['doi'])
                pmcs = json.loads(decision['secondary_pmcids'] or '[]')
                values = metrics(row)
                gaps = [not values['abstract'], not values['every_author_institution'], not values['every_author_country']]
                priority = hashlib.sha256(('20261021:' + (row['doi'] or row['work_id'])).encode()).hexdigest()
                db.execute('INSERT INTO papers VALUES(' + ','.join('?' * 18) + ')',
                           (row['work_id'], row['doi'], row['field'], row['year_openalex'], row['source_id'],
                            row['journal_name'], row['title'], publisher, publisher_route, path.name,
                            decision['eligibility_status'], json.dumps(pmcs), bool(set(pmcs) - known),
                            *gaps, row['author_list_may_be_truncated'], priority))
                payload = {'doi': row['doi'], 'missing_fields': [], 'pmcids': pmcs,
                           'source_id': row['source_id'], 'issns': json.loads(j['issn'] or '[]'),'qualification_evidence_ids':json.loads(decision['evidence_ids'] or '[]'),'qualification_reason':decision['reason']}
                confirmed = decision['eligibility_status'].startswith(('included_', 'excluded_'))
                task(db, 'qualification', row['doi'] or row['work_id'], publisher_route, payload,
                     row['work_id'], status='resolved' if confirmed else 'pending')
                if decision['eligibility_status'] != 'included_verified_research':
                    continue
                author_rows=json.loads(row['authors_json'])
                source_structure_missing=any(not a.get('publisher_affiliation_slots') and not a.get('nature_v1_slots') for a in author_rows)
                payload['missing_fields'] = [k for k, missing in zip(['abstract', 'institutions', 'countries'], gaps, strict=True) if missing]
                if source_structure_missing:payload['missing_fields'].append('publisher_affiliation_structure')
                if any(gaps) or row['author_list_may_be_truncated'] or source_structure_missing:
                    task(db, 'metadata', row['work_id'], 'pmc_or_publisher', payload, row['work_id'])
                for author in author_rows:
                    if author.get('country_conflict_retained'):
                        task(db,'country_conflict',row['work_id']+':'+str(author['position']),'publisher_signature_then_official',
                             {'countries':author['countries'],'source':author.get('publisher_affiliation_source'),'reason':author.get('country_conflict_disposition')},row['work_id'],author['position'])
                    unresolved_slots=[s['text'] for s in author.get('publisher_affiliation_slots',[]) if s['identity_status']=='unresolved']
                    if author['institution_ids_or_verified_local_entities'] and author['countries'] and not unresolved_slots:
                        continue
                    addresses = unresolved_slots or author.get('raw_affiliation_strings') or author.get('supplementary_addresses') or []
                    if isinstance(addresses, str):
                        addresses = [addresses]
                    for address in addresses or ['[missing publisher affiliation]']:
                        entity = ' '.join(address.casefold().split()) if address!='[missing publisher affiliation]' else row['work_id']+':missing_affiliation:'+str(author['position'])
                        task(db, 'institution', entity, 'publisher_signature_then_official',
                             {'name': address, 'missing_identity': not author['institution_ids_or_verified_local_entities'],
                              'missing_country': not author['countries']}, row['work_id'], author['position'])
                label = decision['nature_portfolio_at_verified_online_year']
                if label not in ['True', 'False']:
                    task(db, 'portfolio', row['source_id'] + ':' + str(decision['online_year']), 'dated_publisher',
                         {'source_id': row['source_id'], 'year': decision['online_year'], 'journal': row['journal_name']}, row['work_id'])
        db.commit()
        print('queue indexed', path.name, flush=True)
    for record in json.loads((base / 'institution_registry.json').read_text()):
        names[record['id']] = record
    db.executemany('INSERT OR IGNORE INTO identities VALUES(?,?,?,?,?,?,?)',
                   [(ident, r['display_name'], r.get('type', 'unknown'), json.dumps([r['country_code']] if r.get('country_code') else []),
                     json.dumps({'source': 'inherited_openalex_registry_not_ownership_proof', 'ror': r.get('ror')}),
                     'unverified', '{}') for ident, r in names.items()])
    db.execute("UPDATE tasks SET status='pending',reason='Canonical or another same-DOI Work still requires qualification' WHERE kind='qualification' AND active=1 AND status='resolved' AND EXISTS(SELECT 1 FROM papers p WHERE (p.doi=tasks.entity_key OR p.work_id=tasks.entity_key) AND p.status LIKE 'pending%')")
    db.commit()
    summary = {'candidate_count': db.execute('SELECT count(*) FROM papers').fetchone()[0],
               'tasks': list(db.execute('SELECT kind,status,count(*) FROM tasks WHERE active=1 GROUP BY kind,status')),
               'pending_publishers': list(db.execute("SELECT publisher,count(*) n FROM papers WHERE status LIKE 'pending%' GROUP BY publisher ORDER BY n DESC")),
               'unfetched_pmc_papers': db.execute("SELECT count(*) FROM papers WHERE status LIKE 'pending%' AND pmc_unread=1").fetchone()[0]}
    (root / 'supplement/queue_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    db.close()
    return summary


def select_pilot(root: Path = ROOT) -> list[dict]:
    db = connect(root)
    sampling_path=root/'supplement/pilot_sampling.json'
    sampling_rule='equal_field_year_round_robin_journals_then_seeded_doi'
    if db.execute('SELECT count(*) FROM pilot').fetchone()[0] and (not sampling_path.exists() or json.loads(sampling_path.read_text()).get('rule')!=sampling_rule):
        prior=list(db.execute('SELECT arm,field,year,count(*) FROM pilot GROUP BY arm,field,year'))
        (root/'supplement/pilot_sampling_correction.json').write_text(json.dumps({'reason':'Initial SQL ordered field/year labels before rotating strata; replaced before interpretation. All acquired receipts remain general supplementation evidence.','previous_distribution':prior},indent=2))
        db.execute('DELETE FROM pilot');db.commit()
    if db.execute('SELECT count(*) FROM pilot').fetchone()[0]:
        db.row_factory = sqlite3.Row
        result = [dict(r) for r in db.execute('SELECT doi,work_id,arm,pmcid,route,field,year,source_id FROM pilot ORDER BY arm,doi')]
        db.close()
        return result
    selected: list[dict] = []
    seen = set()
    for arm, n in [('pmc', 1000), *[(x, 500) for x in ['elsevier', 'wiley', 'springer', 'mdpi', 'acs', 'iop']]]:
        condition = 'pmc_unread=1' if arm == 'pmc' else 'route=?'
        query = f'''WITH journals_first AS (SELECT *,row_number() OVER(PARTITION BY field,year,source_id ORDER BY priority) journal_rank
          FROM papers WHERE status LIKE 'pending%' AND doi<>'' AND {condition}),
          strata AS (SELECT *,row_number() OVER(PARTITION BY field,year ORDER BY journal_rank,priority) strata_rank FROM journals_first)
          SELECT doi,work_id,pmcids,route,field,year,source_id FROM strata ORDER BY strata_rank,field,year,priority'''
        for doi, wid, pmcs, publisher, field, year, sid in db.execute(query, () if arm == 'pmc' else (arm,)):
            if doi in seen:
                continue
            pmcid = next((p for p in json.loads(pmcs) if not (root / 'fields/qualification/jats_parsed' / (p + '.json')).exists()), '')
            row = {'doi': doi, 'work_id': wid, 'arm': arm, 'pmcid': pmcid, 'route': publisher,
                   'field': field, 'year': year, 'source_id': sid}
            selected.append(row)
            seen.add(doi)
            db.execute('INSERT INTO pilot(doi,work_id,arm,pmcid,route,field,year,source_id) VALUES(?,?,?,?,?,?,?,?)', tuple(row.values()))
            n -= 1
            if n == 0:
                break
    db.commit()
    db.close()
    sampling_path.write_text(json.dumps({'rule':sampling_rule,'seed':20261021,'maximum_unique_dois':4000},indent=2))
    selected.sort(key=lambda r:(r['arm'],r['doi']))
    (root / 'supplement/pilot.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    return selected


def attempt(db: sqlite3.Connection, ident: str, source: str, status: str, receipt: str, detail: dict) -> None:
    db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
               (ident, source, status, receipt, json.dumps(detail, ensure_ascii=False)))
    db.execute("UPDATE tasks SET attempt_n=attempt_n+1,status=CASE WHEN status='resolved' THEN status ELSE ? END,reason=CASE WHEN status='resolved' THEN reason ELSE ? END,receipt=CASE WHEN status='resolved' THEN receipt ELSE ? END,updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
               (status, detail.get('reason', ''), receipt, ident))
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['discover', 'pilot'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result: Any = discover(args.root) if args.action == 'discover' else select_pilot(args.root)
    print(json.dumps({k:v for k,v in result.items() if k!='pending_publishers'} if isinstance(result,dict) else dict(Counter(r['arm'] for r in result)),ensure_ascii=False))


if __name__ == '__main__':
    main()
