"""One-pass journal, country and all-institution figure inputs from current metadata."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .current_extract import JOURNALS, write_json
from .supplement_queue import ROOT

OUT = ROOT/'analysis/tables'
COUNTRY_SCHEMA = pa.schema([('work_id',pa.string()),('country',pa.string()),
    ('field',pa.string()),('online_year',pa.int64()),('fractional_count',pa.float64()),
    ('international_collaboration',pa.bool_())])
INSTITUTION_SCHEMA = pa.schema([('work_id',pa.string()),('institution_id',pa.string()),
    ('field',pa.string()),('online_year',pa.int64()),('fractional_count',pa.float64()),
    ('observed_multi_institution',pa.bool_()),('international_collaboration',pa.bool_())])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


def add_entity(groups: dict, entity: str, field: str, year: int, fraction: float,
               international: bool, multi: bool) -> None:
    for f,y in [(field,year),(field,0),('all',year),('all',0)]:
        v=groups[(entity,f,y)]
        v[0]+=1;v[1]+=fraction;v[2]+=international;v[3]+=multi


def entity_rows(groups: dict, registry: dict, denominators: Counter, kind: str) -> list[dict]:
    output=[]
    for (entity,field,year),v in groups.items():
        meta=registry.get(entity,{})
        name=meta.get('display_name',entity)
        output.append({'entity_id':entity,'name':name,'entity_type':meta.get('type','unknown'),
            'field':field,'online_year':year or 'all','full_count':int(v[0]),
            'fractional_count':v[1],'sample_paper_share':v[0]/denominators[(field,year)],
            'international_collaboration_rate':v[2]/v[0],
            'observed_multi_institution_rate':v[3]/v[0],
            'denominator':denominators[(field,year)],'counting_scope':kind})
    rank_groups=defaultdict(list)
    for row in output:
        rank_groups[(row['field'],row['online_year'])].append(row)
    for group in rank_groups.values():
        for metric,key in [('full_count','full_rank'),('fractional_count','fractional_rank')]:
            for rank,row in enumerate(sorted(group,key=lambda r:(-r[metric],r['entity_id'])),1):
                row[key]=rank
        for row in group:
            row['fractional_rank_change']=row['full_rank']-row['fractional_rank']
    return sorted(output,key=lambda r:(r['field'],str(r['online_year']),r['full_rank']))


def run() -> dict[str, Any]:
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    selection=pq.read_table(ROOT/'analysis/cohort/selection.parquet',filters=[('included','=',True)]).to_pylist()
    selected={r['work_id']:r for r in selection}
    shards=defaultdict(set)
    for r in selection:
        shards[r['paper_shard']].add(r['work_id'])
    registry={r['id']:r for r in json.loads((ROOT/'fields/institution_registry.json').read_text())}
    country_groups=defaultdict(lambda:[0,0.,0,0]);institution_groups=defaultdict(lambda:[0,0.,0,0])
    country_den=Counter();institution_den=Counter();journals={};journal_years={}
    country_n=institution_n=unknown_country_n=0
    country_path=OUT/'paper_countries.parquet';institution_path=OUT/'paper_institutions.parquet'
    with pq.ParquetWriter(country_path,COUNTRY_SCHEMA,compression='zstd') as cw, \
         pq.ParquetWriter(institution_path,INSTITUTION_SCHEMA,compression='zstd') as iw:
        for shard,ids in sorted(shards.items()):
            paper_country=[];paper_institution=[]
            for batch in pq.ParquetFile(ROOT/'fields/papers'/shard).iter_batches(
                    columns=['work_id','topic_id','authors_json'],batch_size=1000):
                for p in batch.to_pylist():
                    if p['work_id'] not in ids:
                        continue
                    s=selected[p['work_id']];field=s['field'];year=s['online_year']
                    authors=json.loads(p['authors_json'])
                    countries=sorted({c for a in authors for c in a.get('countries',[]) if c})
                    entities=sorted({e for a in authors for e in a.get('institution_ids_or_verified_local_entities',[]) if e})
                    international=len(countries)>1;multi=len(entities)>1
                    # Countries and institution entities have independent paper denominators.
                    country_n+=1;institution_n+=bool(entities);unknown_country_n+=not bool(countries)
                    for f,y in [(field,year),(field,0),('all',year),('all',0)]:
                        country_den[(f,y)]+=1;institution_den[(f,y)]+=bool(entities)
                    for c in countries:
                        add_entity(country_groups,c,field,year,1/len(countries),international,multi)
                        paper_country.append({'work_id':p['work_id'],'country':c,'field':field,
                            'online_year':year,'fractional_count':1/len(countries),
                            'international_collaboration':international})
                    for e in entities:
                        add_entity(institution_groups,e,field,year,1/len(entities),international,multi)
                        paper_institution.append({'work_id':p['work_id'],'institution_id':e,'field':field,
                            'online_year':year,'fractional_count':1/len(entities),
                            'observed_multi_institution':multi,'international_collaboration':international})
                    key=(s['source_id'],field)
                    if key not in journals:
                        journals[key]={'journal':s['journal'],'n':0,'international':0,
                            'topics':Counter(),'years':Counter(),'portfolio':Counter()}
                    group=journals[key];group['n']+=1;group['international']+=international
                    if p['topic_id']:
                        group['topics'][p['topic_id']]+=1
                    group['years'][year]+=1;group['portfolio'][str(s['portfolio'])]+=1
                    ykey=(*key,year)
                    yg=journal_years.setdefault(ykey,{'n':0,'international':0,'topics':Counter(),'portfolio':Counter()})
                    yg['n']+=1;yg['international']+=international
                    if p['topic_id']:
                        yg['topics'][p['topic_id']]+=1
                    yg['portfolio'][str(s['portfolio'])]+=1
            cw.write_table(pa.Table.from_pylist(paper_country,schema=COUNTRY_SCHEMA))
            iw.write_table(pa.Table.from_pylist(paper_institution,schema=INSTITUTION_SCHEMA))
            print(json.dumps({'shard':shard,'country_papers':country_n,'institution_papers':institution_n}),flush=True)
    country_registry={c:{'display_name':c,'type':'country'} for c,_,_ in country_groups}
    try:
        import pycountry
    except ImportError:
        pass
    else:
        for c,meta in country_registry.items():
            found=pycountry.countries.get(alpha_2=c)
            if found:
                meta['display_name']=found.name
    countries=entity_rows(country_groups,country_registry,country_den,'known_countries')
    institutions=entity_rows(institution_groups,registry,institution_den,'identified_institutions_only')
    write_csv(OUT/'panel_c_countries.csv',countries)
    write_csv(OUT/'panel_c_institutions.csv',institutions)
    write_csv(OUT/'panel_c_country_top12.csv',[r for r in countries if r['field']=='all' and r['online_year']=='all' and r['full_rank']<=12])
    write_csv(OUT/'panel_c_institution_top15.csv',[r for r in institutions if r['field']=='all' and r['online_year']=='all' and r['full_rank']<=15])
    used_entities={e for e,f,y in institution_groups if f=='all' and y==0}
    write_csv(OUT/'institution_entities.csv',[{'institution_id':e,
        'name':registry.get(e,{}).get('display_name',e),'type':registry.get(e,{}).get('type','unknown'),
        'registry_country':registry.get(e,{}).get('country_code'),
        'identity_source':'existing_registry' if e in registry else 'existing_entity_id_without_registry_label',
        'name_missing':e not in registry} for e in sorted(used_entities)])
    journal_rows=[]
    for (source,field),g in sorted(journals.items()):
        journal_rows.append(journal_row(source,field,'all',g))
    write_csv(OUT/'panel_b_journals.csv',journal_rows)
    write_csv(OUT/'panel_b_journal_years.csv',[journal_row(source,field,year,{**g,'journal':journals[(source,field)]['journal']})
        for (source,field,year),g in sorted(journal_years.items())])
    summary={'country_papers':country_n,'institution_papers':institution_n,
        'no_identifiable_institution':country_n-institution_n,'countries_missing':unknown_country_n,
        'journal_field_groups':len(journals),'display_journal_field_points':sum(r['display_main'] for r in journal_rows),
        'unique_journals':len({s for s,f in journals}),'country_entities':len(country_registry),
        'institution_entities':len(used_entities),'institution_names_missing':sum(e not in registry for e in used_entities),
        'fields':{f:{'country_papers':country_den[(f,0)],'institution_papers':institution_den[(f,0)]}
                  for f in ['biochemistry','materials','physics']},
        'llm_calls':0,'elapsed_seconds':round(time.monotonic()-started,1),
        'institution_country_note':'Registry country is descriptive only; paper countries come from authorship work locations.',
        'institution_ownership_merge':False,'new_acquisition':False}
    write_json(OUT/'metadata_summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def journal_row(source: str,field: str,year: int|str,g: dict) -> dict[str,Any]:
    n=g['n'];known=sum(g['topics'].values());brand=g['portfolio']
    status='Nature Portfolio' if brand['True']==n else 'non Nature Portfolio' if brand['False']==n else 'unconfirmed_or_mixed'
    return {'source_id':source,'journal':g['journal'],'field':field,'online_year':year,
        'n':n,'topic_known_n':known,'topic_diversity':1-sum(v*(v-1) for v in g['topics'].values())/(known*(known-1)) if known>=2 else None,
        'topic_coverage':known/n,'international_collaboration_rate':g['international']/n,
        'nature_portfolio_n':brand['True'],'non_nature_portfolio_n':brand['False'],
        'publisher_unconfirmed_n':n-brand['True']-brand['False'],'publisher_group':status,
        'is_nature_main':source=='https://openalex.org/S137773608',
        'highlight':JOURNALS.get(source,''),'display_main':n>=30 and known/n>=.95,
        'n_2023':g.get('years',{}).get(2023,0),'n_2024':g.get('years',{}).get(2024,0),
        'n_2025':g.get('years',{}).get(2025,0)}


if __name__=='__main__':
    run()
