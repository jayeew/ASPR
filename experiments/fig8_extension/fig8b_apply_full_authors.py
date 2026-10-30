"""Add individual-work author evidence while retaining the original list-response layer."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_build import NATURE, country_names, doi_key, enrich_authors, metrics
from .fig8b_collect import OUT
from .fig8b_epmc import fetch as fetch_secondary
from .fig8b_supplements import load, select_epmc, title_key
from .stage0 import write_csv, write_json


def main() -> None:
    with sqlite3.connect(OUT/'epmc_queue.sqlite') as connection:
        known = {r[0] for r in connection.execute('SELECT doi FROM queue')}
        with (OUT/'full_author_targets.csv').open(encoding='utf-8-sig') as stream:
            supplemental = sorted({r['doi'] for r in csv.DictReader(stream) if r['doi'] and r['doi'] not in known and '"' not in r['doi']})
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(fetch_secondary,[supplemental[i:i+60] for i in range(0,len(supplemental),60)]):
                token = hashlib.sha256(json.dumps(result['params'],sort_keys=True).encode()).hexdigest()
                connection.executemany('INSERT OR REPLACE INTO queue VALUES (?,?,?)',
                    [(d,result['status'],f'epmc_raw/{token}.json.gz') for d in result['dois']])
        connection.commit()
        secondary_status = dict(connection.execute('SELECT doi,status FROM queue'))
        write_json(OUT/'full_author_secondary_scope.json',{'additional_dois_queried':len(supplemental)})
        write_json(OUT/'epmc_status.json',{'doi_status':dict(connection.execute('SELECT status,count(*) FROM queue GROUP BY status'))})
    donors = {r['doi']:r for r in map(json.loads,(NATURE/'papers.jsonl').open())}
    epmc, _, _crossref = load()
    names = country_names()
    for ambiguous in ['korea','georgia','congo']:
        names.pop(ambiguous,None)
    old_schema = pq.read_schema(next((OUT/'papers').glob('*.parquet')))
    extra = [('listed_author_records',pa.int64()),('full_endpoint_authors_added',pa.int64()),
             ('full_authorships_status',pa.string()),('authorships_source_receipt',pa.string())]
    schema = pa.schema(list(old_schema) + [pa.field(k,t) for k,t in extra if k not in old_schema.names])
    totals: dict[tuple,Counter] = defaultdict(Counter)
    journal_totals: dict[tuple,Counter] = defaultdict(Counter)
    changes = []
    institution_map = {r['id']:r for r in json.loads((OUT/'institution_registry.json').read_text())}
    status_fields = ['work_id','doi','field','year_openalex','year_publisher_verified','source_id','journal_name',
        'type_status','publisher_type','date_status','abstract_source','after_abstract',
        'after_every_author_institution','after_every_author_country','metadata_final_status',
        'publisher_comparison_group','historical_portfolio_status','epmc_attempt_status','publisher_attempt_status',
        'listed_author_records','full_endpoint_authors_added','full_authorships_status']
    with gzip.open(OUT/'paper_status.csv.gz','wt',newline='',compresslevel=1) as status_stream:
        status_writer = csv.DictWriter(status_stream,fieldnames=status_fields)
        status_writer.writeheader()
        for path in sorted((OUT/'papers').glob('*.parquet')):
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                listed = row.get('listed_author_records',row['before_author_records'])
                row.update(listed_author_records=listed,full_endpoint_authors_added=0,
                    full_authorships_status='not_needed_below_list_limit',authorships_source_receipt=row['raw_receipt'])
                if listed >= 100:
                    receipt_path = OUT/'full_authors_raw'/(row['work_id'].rsplit('/',1)[-1]+'.json.gz')
                    with gzip.open(receipt_path,'rt') as stream:
                        receipt = json.load(stream)
                    raw = receipt.get('response') or {}
                    raw_authors = raw.get('authorships') or []
                    same = (raw.get('id') == row['work_id'] and title_key(raw.get('title')) == title_key(row['title'])
                        and (not row['doi'] or doi_key(raw.get('doi')) == row['doi']))
                    if receipt['status'] == 'retrieved' and same and len(raw_authors) >= listed:
                        donor = donors.get(row['doi']) if row['type_status'] == 'nature_v1_original_article_verified' else None
                        secondary = select_epmc(row,epmc.get(row['doi'],[]))
                        row['epmc_attempt_status'] = secondary_status.get(row['doi'],'no_doi' if not row['doi'] else 'not_queued')
                        row['epmc_match_status'] = 'exact_doi_title_accepted' if secondary else 'conflict_or_title_mismatch' if epmc.get(row['doi']) else 'no_accepted_record'
                        authors, _updates = enrich_authors(raw_authors,donor,names,secondary)
                        original = [{**a,'institution_ids_or_verified_local_entities':a['original_institution_ids'],
                                     'countries':a['original_countries']} for a in authors]
                        before = metrics(original,row['abstract'] if row['abstract_source']=='openalex_inverted_index' else None)
                        after = metrics(authors,row['abstract'])
                        for phase,values in [('before',before),('after',after)]:
                            row.update({phase+'_'+k:v for k,v in values.items()})
                        row.update(authors_json=json.dumps(authors,ensure_ascii=False),
                            full_endpoint_authors_added=len(raw_authors)-listed,
                            full_authorships_status='individual_endpoint_identity_verified',
                            authorships_source_receipt=str(receipt_path.relative_to(OUT)),
                            author_list_may_be_truncated=len(raw_authors)>=5000)
                        row['metadata_final_status'] = 'gaps_remain' if not after['abstract'] or not after['every_author_institution'] or not after['every_author_country'] else 'observed_fields_present_not_publisher_verified'
                        for author in raw_authors:
                            for institution in author.get('institutions') or []:
                                if institution.get('id'):
                                    institution_map[institution['id']] = institution
                    else:
                        row['full_authorships_status'] = 'unresolved_endpoint_identity_count_or_request'
                    changes.append({'work_id':row['work_id'],'doi':row['doi'],'field':row['field'],
                        'listed_authors':listed,'final_author_records':row['after_author_records'],
                        'status':row['full_authorships_status'],'raw_receipt':str(receipt_path.relative_to(OUT)),
                        'cap_risk_remaining':row['author_list_may_be_truncated']})
                for phase in ['before','after']:
                    values = {k:row[phase+'_'+k] for k in metrics([],None)}
                    for year in [0,row['year_openalex']]:
                        totals[(row['field'],year,phase)].update(values)
                    journal_totals[(row['field'],row['source_id'],phase)].update(values)
                status_writer.writerow({k:row[k] for k in status_fields})
            temporary = path.with_suffix('.tmp')
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),temporary,compression='zstd')
            temporary.replace(path)
            print('full-author layer',path.stem,flush=True)
    write_json(OUT/'institution_registry.json',list(institution_map.values()))
    write_csv(OUT/'coverage.csv',[{'field':k[0],'year':k[1],'phase':k[2],**v} for k,v in sorted(totals.items())])
    write_csv(OUT/'journal_coverage.csv',[{'field':k[0],'source_id':k[1],'phase':k[2],**v} for k,v in sorted(journal_totals.items())])
    write_csv(OUT/'full_author_changes.csv',changes)
    write_json(OUT/'full_author_summary.json',{'target_papers':len(changes),
        'statuses':dict(Counter(r['status'] for r in changes)),
        'additional_author_records':sum(r['final_author_records']-r['listed_authors'] for r in changes),
        'cap_risk_remaining':sum(r['cap_risk_remaining'] for r in changes),
        'scope':'All available single-endpoint authorships; not a publisher-byline census. Original list-response rows and metrics retained separately.'})


if __name__ == '__main__':
    main()
