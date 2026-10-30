"""Append identity-verified publisher-window omissions to the single current tables."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .boundary_scope_assessment import run as assess
from .boundary_publisher_record import receipt_records
from .fig8b_build import enrich_authors
from .fig8b_qualification_build import match, write_decisions
from .fig8b_qualification_decide import decide, portfolio_label
from .fig8b_qualification_recompute import metrics
from .nature_backfill_resolve import country_names
from .publisher_delta_apply import stamp
from .stage0_roster import abstract_text
from .supplement_queue import ROOT, connect, route, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(full_directory: bool = False) -> dict:
    folder=ROOT/'supplement/boundary_review';q=ROOT/'fields/qualification';stats=Counter()
    if full_directory:
        from .boundary_full_assessment import run as assess_full
        status_file=folder/'full_directory_status.json'
        assessment_file=folder/'full_directory_assessment.json'
        assessment=assess_full
    else:
        status_file=folder/'current_status.json'
        assessment_file=folder/'current_assessment.json'
        assessment=assess
    if json.loads(status_file.read_text())['status']!='fixed_batches_processed_identity_scope_review_required':
        raise RuntimeError('Current metadata batches are not complete')
    assessment()
    selected=[r for r in json.loads(assessment_file.read_text()) if r['current_assessment']=='current_in_scope_append_candidate_requires_primary_requalification']
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(ROOT)
        schema=pq.read_schema(next((ROOT/'fields/papers').glob('*.parquet')))
        journals={r['source_id']:r for r in read_csv(ROOT/'fields/journal_registry.csv')}
        annual={(r['source_id'],int(r['year'])):r for r in read_csv(ROOT/'fields/journal_publisher_by_year.csv')}
        paper_groups=defaultdict(list);decision_groups=defaultdict(list);evidence=[];audit=[]
        for item in selected:
            wid=item['current_work_id'];doi=item['doi']
            if db.execute('SELECT 1 FROM papers WHERE work_id=? OR doi=?',(wid,doi)).fetchone():stats['already_present']+=1;continue
            with gzip.open(ROOT/item['current_receipt'],'rt') as stream:receipt=json.load(stream)
            position,raw=next((i,v) for i,v in enumerate(receipt['response']['results']) if v['id']==wid)
            official=receipt_records(item['publisher_receipt'])[doi]
            sid=item['current_source_id'];year=int(official['publisher_date'][:4]);field={'13':'biochemistry','25':'materials','31':'physics'}[item['current_field_id'].rsplit('/',1)[-1]]
            if year not in {2023,2024,2025} or sid not in journals:stats['scope_not_confirmed']+=1;continue
            source=raw['primary_location']['source'];membership=annual.get((sid,year),{})
            row={f.name:False if pa.types.is_boolean(f.type) else 0 if pa.types.is_integer(f.type) else '' for f in schema}
            authors,_=enrich_authors(raw.get('authorships') or [],None,country_names());abstract=abstract_text(raw.get('abstract_inverted_index')) or ''
            row.update(work_id=wid,doi=doi,field=field,field_id=item['current_field_id'],topic_id=(raw.get('primary_topic') or {}).get('id',''),title=raw['title'],publication_date_openalex=raw['publication_date'],year_openalex=raw['publication_year'],source_id=sid,journal_name=source['display_name'],issn_l=source.get('issn_l') or '',abstract=abstract,abstract_source='openalex_inverted_index' if abstract else 'unresolved',authors_json=json.dumps(authors,ensure_ascii=False),raw_receipt='../'+item['current_receipt'],raw_position=position,openalex_updated_date=raw.get('updated_date') or '',author_list_may_be_truncated=len(authors)>=100,listed_author_records=len(authors),full_authorships_status='publisher_byline_not_yet_verified',authorships_source_receipt=item['current_receipt'],type_status='unverified_candidate_not_final_research',date_status='aps_metadata_online_verified' if official['family']=='aps' else 'publisher_catalogue_online_verified',publisher_type=official['publisher_type'],publisher_online_date=official['publisher_date'],year_publisher_verified=year,publisher_evidence='../'+item['publisher_receipt'],publisher_attempt_status='official_catalogue_reused',epmc_attempt_status='not_yet_queried',epmc_match_status='not_yet_queried',nature_portfolio_official_current=journals[sid].get('nature_portfolio_official_current')=='True',nature_portfolio_at_candidate_year=membership.get('nature_portfolio_year') or 'unknown',historical_portfolio_status=membership.get('status','unverified'),publisher_comparison_group=membership.get('comparison_group','unresolved'),canonical_journal_key=membership.get('canonical_journal_key',sid),annual_publisher_evidence=membership.get('catalogue_file') or '',annual_imprint=membership.get('imprint') or '')
            values=metrics(row)
            for phase in ['before','after']:
                for k,v in values.items():
                    if phase+'_'+k in row:row[phase+'_'+k]=v
            row['metadata_final_status']='gaps_remain' if not all(values[k] for k in ['abstract','every_author_institution','every_author_country']) else 'observed_fields_present_not_publisher_verified'
            primary=match(row,{'family':official['family'],'aps_journal':official.get('aps_journal',''),'categories':official.get('categories',[]),'title':official['title'],'source_id':sid,'kind':official['publisher_type'],'online_date':official['publisher_date'],'date_precision':'day','receipt':'../../'+item['publisher_receipt'],'url':official['catalogue_url']},journals[sid])
            if primary['identity']!='matched':stats['primary_identity_unresolved']+=1;continue
            value=decide([primary],[]);label,ids=portfolio_label(membership.get('nature_portfolio_year',''),[primary],value['online_year']);shard=f'{field}_{year}_boundary.parquet'
            decision={k:row[k] for k in ['work_id','doi','field','source_id','title','year_openalex','nature_portfolio_at_candidate_year']}
            decision.update(online_year=value['online_year'],online_date=value['online_date'],publisher_types=value['type'],eligibility_status=value['status'],reason=value['reason'],rejected_evidence_ids=json.dumps(value.get('rejected_evidence_ids',[])),duplicate_of='',evidence_ids=json.dumps(value['evidence_ids']),secondary_evidence_present=False,secondary_pmcids='[]',secondary_online_dates='[]',secondary_pubtypes='[]',base_shard='../papers/'+shard,nature_portfolio_at_verified_online_year=label,portfolio_evidence_ids=json.dumps(ids))
            paper_groups[shard].append(row);decision_groups[shard].append(decision);evidence.append({'work_id':wid,'doi':doi,**primary})
            publisher=journals[sid].get('host_name','');rte=route(publisher)
            db.execute('INSERT INTO papers VALUES('+','.join('?'*18)+')',(wid,doi,field,raw['publication_year'],sid,row['journal_name'],row['title'],publisher,rte,shard,value['status'],'[]',False,not values['abstract'],not values['every_author_institution'],not values['every_author_country'],row['author_list_may_be_truncated'],hashlib.sha256(('20261021:'+doi).encode()).hexdigest()))
            task(db,'qualification',doi,rte,{'doi':doi,'source_id':sid,'reason':'Publisher-window boundary omission; current exact identity and field confirmed','evidence_ids':value['evidence_ids']},wid,status='resolved' if value['status'].startswith(('included','excluded')) else 'pending')
            if value['status']=='included_verified_research':task(db,'metadata',wid,'pmc_or_publisher',{'doi':doi,'reason':'New publisher-window paper; metadata/author-signature verification required'},wid)
            db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(wid,'paper_added','null',json.dumps(row,ensure_ascii=False),json.dumps([item['current_receipt'],item['publisher_receipt'],item.get('language_evidence',item['current_receipt'])]),'Exact DOI/title/journal and current three-field scope; publisher online year controls window instead of erroneous API year; research status separately decided'))
            audit.append({'work_id':wid,'doi':doi,'status':value['status'],'api_year':raw['publication_year'],'publisher_year':year,'current_metadata_receipt':item['current_receipt'],'publisher_receipt':item['publisher_receipt'],'api_language':item.get('api_language'),'language_basis':item.get('language_basis','current_metadata'),'language_evidence':item.get('language_evidence',item['current_receipt'])});stats[value['status']]+=1
        with gzip.open(q/'decision_evidence.jsonl.gz','at') as stream,(q/'curated_evidence.jsonl').open('a') as curated:
            for item in evidence:
                line=json.dumps(item,ensure_ascii=False)+'\n';stream.write(line);curated.write(line)
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)',(item['work_id'],item['id'],'primary',json.dumps(item,ensure_ascii=False)))
        for shard,rows in paper_groups.items():
            path=ROOT/'fields/papers'/shard;existing=pq.read_table(path).to_pylist() if path.exists() else []
            tmp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(existing+rows,schema=schema),tmp,compression='zstd');tmp.replace(path)
            path=q/'decisions'/shard;existing=pq.read_table(path).to_pylist() if path.exists() else []
            write_decisions(path,existing+decision_groups[shard],q/'decision_changes.jsonl.gz')
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)',('primary',stamp(q/'decision_evidence.jsonl.gz')));db.commit();db.close()
    with (folder/'append_events.jsonl').open('a') as stream:
        for item in audit:stream.write(json.dumps(item)+'\n')
    result={'appended':len(audit),**stats};atomic_json(folder/('full_directory_append_status.json' if full_directory else 'append_status.json'),result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
