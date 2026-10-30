"""Update evidence-linked qualification of the current Fig8b paper table."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date as calendar_date
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import html

from .fig8b_qualification_decide import (
    decide,
    electronic_date,
    imprint_dates,
    key,
    portfolio_label,
    type_class,
)
from .fig8b_qualification_evidence import parse_front
from .fig8b_qualification_recompute import gzip_text
from .fig8b_qualification_recompute import metrics as paper_metrics
from .supplement_oai import parse_oai

BASE=Path('outputs/fig8_data/fields')
OUT=BASE / 'qualification'


def plain(value: str) -> str:
    # A literal comparison such as 0.0<x<0.5 is not HTML markup.
    has_tag = re.search(r'</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*?)?\s*/?>', value)
    return html.fragment_fromstring(value,create_parent='div').text_content() if has_tag else value


def load_evidence() -> tuple[dict,dict,list]:
    evidence: dict[str,list]=defaultdict(list)
    audits=[]
    catalogues=defaultdict(list)
    for path in sorted((OUT/'catalogue_parsed').glob('*.json')):
        r=json.loads(path.read_text());catalogues[(r['source_id'],r['year'])].append(r)
        if r['status']!='parsed':continue
        for row in r['rows']:
            evidence[row['doi']].append({'family':'nature_catalogue','title':row['title'],'source_id':r['source_id'],
                'kind':row['publisher_type'],'online_date':row['publisher_date'],'date_precision':'day',
                'receipt':r['raw_receipt'],'url':r['url'],'categories':[]})
    for (source,year),pages in sorted(catalogues.items()):
        good=[p for p in pages if p['status']=='parsed']
        expected={p['expected_count'] for p in good}
        rows=[r for p in good for r in p['rows']]
        complete=len(expected)==1 and len(rows)==next(iter(expected)) and len({r['doi'] for r in rows})==len(rows)
        audits.append({'source_id':source,'year':year,'pages':len(pages),'parsed_pages':len(good),
                       'expected_counts':json.dumps(sorted(expected)),'records':len(rows),'unique_dois':len({r['doi'] for r in rows}),'complete':complete})
    for path in sorted([*(OUT/'jats_parsed').glob('*.json'),*(OUT/'oai_parsed').glob('*.json')]):
        r=json.loads(path.read_text())
        if r['status']!='parsed':continue
        raw=gzip.decompress((OUT/r['raw_receipt']).read_bytes())
        p=parse_oai(raw) if r.get('source')=='pmc_front_oai' else parse_front(raw+b'</article>')
        date,precision=electronic_date(p['dates'])
        for doi in p['doi']:
            evidence[doi.lower()].append({'family':'archived_jats','title':' '.join(p['title']),'issns':p['issn'],
                'journal_titles':p['journal_titles'],'kind':p['article_type'],'online_date':date,'date_precision':precision,
                'categories':p['categories'],'publishers':p['publisher'],'publication_dates':imprint_dates(p['dates']),'receipt':r['raw_receipt'],'url':r.get('source_url',r['url'])})
    for path in sorted([*(OUT/'aps_all_raw').glob('*.gz'),*(OUT/'aps_single_raw').glob('*.gz')]):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        if r['http_status']!=200:continue
        records=r['response']['data']
        for p in records if isinstance(records,list) else [records]:
            doi=p.get('identifiers',{}).get('doi','').lower()
            if doi:evidence[doi].append({'family':'aps','title':plain(p.get('title',{}).get('value','')),
                'journal_titles':[p.get('journal',{}).get('name','')],'aps_journal':p.get('journal',{}).get('id',''),
                'kind':p.get('articleType',''),'online_date':p.get('date',''),'date_precision':'day',
                'categories':[(p.get('tocSection') or {}).get('label','')],'receipt':str(path.relative_to(OUT)),'url':r['url']})
    for path in sorted((OUT/'nature_single/parsed').glob('*.json')):
        r=json.loads(path.read_text())
        if r.get('status')!='publisher_doi_issn_verified':continue
        for online in r.get('online_date') or ['']:
            evidence[r['doi']].append({'family':'nature_page','title':plain(r.get('publisher_title','')),
                'source_id':r['source_id'],'kind':r['publisher_type'][0] if len(r['publisher_type'])==1 else '',
                'online_date':online.replace('/','-'),'raw_online_date':online,'date_precision':'day' if online else 'missing','categories':[],
                'receipt':r['raw_receipt'],'url':r['url']})
    crossref=[]
    for path in sorted((BASE/'crossref_raw').glob('*.gz')):
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        if r.get('status')=='success':
            crossref.extend((item,'../'+str(path.relative_to(BASE)),r['url']) for item in r['response']['message']['items'])
    for path in sorted((OUT/'duplicate_evidence').glob('*.json')):
        if '.aps.' in path.name:continue
        r=json.loads(path.read_text())
        if r.get('http_status')==200:crossref.append((r['response']['message'],str(path.relative_to(OUT)),r['url']))
    for item,receipt,url in crossref:
        for parts in item.get('published-online',{}).get('date-parts',[]):
            if not parts or not all(isinstance(x,int) for x in parts):continue
            try:
                calendar_date(parts[0],parts[1] if len(parts)>1 else 1,parts[2] if len(parts)>2 else 1)
            except (ValueError,TypeError):
                continue
            online='-'.join(f'{x:04d}' if i==0 else f'{x:02d}' for i,x in enumerate(parts))
            evidence[item['DOI'].lower()].append({'family':'crossref_publisher_deposit','title':plain((item.get('title') or [''])[0]),
                'issns':item.get('ISSN',[]),'journal_titles':item.get('container-title',[]),
                'kind':item.get('type','unspecified'),'online_date':online,'date_precision':['year','month','day'][min(len(parts),3)-1],
                'receipt':receipt,'url':url,'categories':[]})
    for path in sorted((OUT/'supplement_parsed').glob('*.json')):
        r=json.loads(path.read_text())
        if r.get('status')!='parsed' or r.get('source')=='pmc':continue
        p=r['parsed']
        for doi in p['doi']:
            for kind in p['article_types'] or ['unspecified']:
                evidence[doi.lower()].append({'family':'publisher_'+r['route'],'title':plain((p['title'] or [''])[0]),
                    'issns':p['issns'],'journal_titles':p['journal_titles'],'kind':kind,
                    'online_date':p['online_date'],'date_precision':p['date_precision'],
                    'publishers':p.get('publisher',[]),'categories':p['article_types'],
                    'receipt':r.get('raw_receipt',''),'url':r['url']})
    db=sqlite3.connect(OUT/'evidence.sqlite');secondary=defaultdict(list)
    for doi,pmc,types,date,title,receipt,oa in db.execute('SELECT * FROM secondary'):
        secondary[doi].append({'types':json.loads(types),'online_date':date,'title':title,'receipt':receipt,'pmcid':pmc})
    db.close()
    for path in sorted((OUT/'epmc_raw').glob('*.gz')):
        with gzip.open(path,'rt') as stream:receipt=json.load(stream)
        if receipt.get('status')!='complete':continue
        requested=set(receipt['dois'])
        for r in receipt['response'].get('resultList',{}).get('result',[]):
            doi=(r.get('doi') or '').lower()
            if doi in requested and r.get('source') in ['MED','PMC']:
                secondary[doi].append({'types':r.get('pubTypeList',{}).get('pubType',[]),
                    'online_date':r.get('electronicPublicationDate',''),'title':r.get('title',''),
                    'receipt':str(path.relative_to(OUT)),'pmcid':r.get('pmcid','')})
    # Evidence-only scoped repairs survive future rebuilds without copying paper subsets.
    curated = OUT / 'curated_evidence.jsonl'
    if curated.exists():
        with curated.open() as stream:
            for line in stream:
                item = json.loads(line)
                evidence[item['doi']].append({k: v for k, v in item.items()
                    if k not in {'work_id', 'identity', 'class', 'id'}})
    return dict(evidence),dict(secondary),audits


def duplicate_map(sources: dict) -> tuple[dict,list]:
    db=sqlite3.connect(OUT/'evidence.sqlite');mapping={};audits=[]
    for path in sorted((OUT/'duplicate_evidence').glob('*.json')):
        if '.aps.' in path.name:continue
        receipt=json.loads(path.read_text());r=receipt['response']['message'];doi=r['DOI'].lower()
        rows=list(db.execute('SELECT work_id,title,source_id FROM candidates WHERE doi=?',(doi,)))
        matches=[wid for wid,title,sid in rows if key(title)==key(plain((r.get('title') or [''])[0])) and set(json.loads(sources[sid]['issn']))&set(r.get('ISSN',[]))]
        for wid,title,sid in rows:
            retained=matches[0] if len(matches)==1 else ''
            if retained and wid!=retained:mapping[wid]=retained
            audits.append({'doi':doi,'work_id':wid,'retained_work_id':retained,'status':'canonical_retained' if wid==retained else 'duplicate_wrong_journal_excluded' if retained else 'unresolved',
                'candidate_source':sid,'publisher_journal':' | '.join(r.get('container-title',[])),
                'publisher_issns':json.dumps(r.get('ISSN',[])),'receipt':str(path.relative_to(OUT)),'url':receipt['url']})
    native_path=OUT/'native_duplicate_aliases.json'
    if native_path.exists():
        native=json.loads(native_path.read_text())
        if native['status']!='reviewed':raise ValueError('Unreviewed native duplicate rules')
        for wid,proof in native['records'].items():
            retained=proof['retained_work_id']
            if wid in mapping and mapping[wid]!=retained:raise ValueError('Conflicting canonical Work identity')
            mapping[wid]=retained
            audits.append({'doi':proof['doi'],'work_id':wid,'retained_work_id':retained,'status':'native_alias_excluded',
                'candidate_source':proof['source_id'],'publisher_journal':'','publisher_issns':'[]','receipt':proof['receipt'],'url':proof['url']})
    db.close();return mapping,audits


def pnas_journal_alias(paper_name: str, native_names: list[str], expected_issns: set[str]) -> bool:
    """PNAS's exact full title and short title, bound to its existing ISSN registry."""
    names = {key('Proceedings of the National Academy of Sciences'),
             key('Proceedings of the National Academy of Sciences of the United States of America')}
    return bool(expected_issns & {'0027-8424','1091-6490'}) and key(paper_name) in names and any(key(v) in names for v in native_names)


def spelling_title_key(value: str) -> str:
    """Normalize only reviewed whole-word British/American spelling pairs."""
    aliases = {'stabilisation': 'stabilization', 'analogue': 'analog',
               'analogues': 'analogs', 'modelling': 'modeling',
               'signalling': 'signaling', 'randomisation': 'randomization'}
    return key(re.sub(r'\b\w+\b', lambda m: aliases.get(m[0].casefold(), m[0].casefold()), value))


def match(row: dict, item: dict, source: dict) -> dict:
    e=dict(item)
    title_ok=key(row['title'])==key(e['title']) and bool(key(row['title']))
    if (not title_ok and row.get('doi') and e.get('doi')
            and row['doi'].casefold() == e['doi'].casefold()
            and spelling_title_key(row['title']) == spelling_title_key(e['title'])
            and spelling_title_key(row['title'])):
        title_ok = True
        e['title_identity_rule'] = 'DOI-confirmed finite spelling aliases; supplement/target_500k/title_spelling/rule.json'
    if e.get('source_id'):journal_ok=e['source_id']==row['source_id']
    elif e.get('issns'):journal_ok=bool(set(e['issns']) & set(json.loads(source['issn'])))
    else:
        journal_ok=key(row['journal_name']) in {key(t) for t in e.get('journal_titles',[])}
        if e.get('aps_journal')=='PRD':journal_ok=bool(set(json.loads(source['issn'])) & {'2470-0010','2470-0029'})
        if e.get('aps_journal')=='PRB':journal_ok=bool(set(json.loads(source['issn'])) & {'2469-9950','2469-9969'})
        if e.get('aps_journal')=='PRL':journal_ok=bool(set(json.loads(source['issn'])) & {'0031-9007','1079-7114'})
    if not e.get('issns') and pnas_journal_alias(row['journal_name'],e.get('journal_titles',[]),set(json.loads(source['issn'] or '[]'))):
        journal_ok=True
        e['journal_identity_rule']='PNAS exact full/short title pair with registry ISSN; supplement/priority_archive/journal_alias_rule.json'
    bridge=e.get('native_title_identity_bridge') or {}
    if (not title_ok and journal_ok and e.get('family')=='archived_jats'
            and row.get('doi','').casefold()==e.get('doi','').casefold()
            and bridge.get('rule')in{'exact_doi_issn_complete_original_byline','exact_doi_issn_original_abstract_and_endpoint_authors','exact_doi_issn_complete_original_abstract_shared_author','exact_doi_issn_complete_byline_initialisation'}
            and bridge.get('candidate_title')==row['title']
            and bridge.get('native_title')==e['title']
            and ((len(bridge.get('native_author_keys',[]))>=3
                 and (bridge.get('native_author_keys')==bridge.get('candidate_author_keys')
                 or (bridge.get('rule')=='exact_doi_issn_original_abstract_and_endpoint_authors'
                     and bridge.get('abstract_key_length',0)>=300
                     and bridge.get('original_abstract_key_sha256')==bridge.get('native_abstract_key_sha256')
                     and bridge.get('native_author_keys')[0]==bridge.get('candidate_author_keys',[None])[0]
                     and bridge.get('native_author_keys')[-1]==bridge.get('candidate_author_keys',[None])[-1])))
                 or (bridge.get('rule')=='exact_doi_issn_complete_original_abstract_shared_author'
                     and bridge.get('abstract_key_length',0)>=600
                     and bridge.get('original_abstract_key_sha256')==bridge.get('native_abstract_key_sha256')
                     and bool(set(bridge.get('native_author_keys',[]))&set(bridge.get('candidate_author_keys',[]))))
                 or (bridge.get('rule')=='exact_doi_issn_complete_byline_initialisation'
                     and len(set(bridge.get('native_bibliographic_author_keys',[])))>=3
                     and all(bridge.get('native_bibliographic_author_keys',[]))
                     and bridge.get('native_bibliographic_author_keys')==bridge.get('candidate_bibliographic_author_keys')
                     and bool(set(bridge.get('native_author_keys',[]))&set(bridge.get('candidate_author_keys',[])))))
            and bridge.get('candidate_source_receipt','').startswith('census_raw/')
            and bridge.get('candidate_raw_sha256') and bridge.get('native_raw_sha256')):
        title_ok=True
        e['title_identity_rule']='Exact DOI/ISSN and original census identity bridge: complete author bylines or entire abstract plus first/last authors; retain titles and hashes'
    e['identity']='matched' if title_ok and journal_ok else 'title_or_journal_mismatch'
    e['class']=type_class(e['kind'],e['family'],e.get('categories'))
    if e['family']=='publisher_springer' and key(e['kind']) in {'originalpaper','originalarticle','researchpaper','reviewpaper'}:
        e['policy_url']='https://www.springernature.com/gp/open-science/oa-agreements/usa/nist'
        if e['class']=='research' and re.search(r'\b(systematic review|scoping review|narrative review|meta.analysis|case report|study protocol|bibliometric analysis)\b',row['title'],re.I):
            e['class']='subtype_review_required'
            e['scope_review_note']='OriginalPaper label conflicts with a potentially nonprimary title subtype; retained for adjudication'
    if e.get('aps_journal') in {'RMP','PHYSICS'} and e['class']=='research':
        e['class']='nonprimary'
        e['policy_url']='https://journals.aps.org/rmp/about' if e['aps_journal']=='RMP' else 'https://physics.aps.org/faq'
    if e['class']=='research' and e['family'].startswith('nature') and any(x in row['journal_name'].lower() for x in ['nature india','nature africa','nature reviews','nature protocols','bdj team']):
        e['class']='subtype_review_required'
    if e['class']=='research' and key(row['title']).startswith(('correction','retraction','erratum','expressionofconcern')):
        e['class']='subtype_review_required'
    if e['class']=='research' and re.search(r'\b(systematic review|meta.analysis|study protocol|case report|case series|scoping review|narrative review|umbrella review)\b',row['title'],re.IGNORECASE):
        e['class']='subtype_review_required'
        e['title_scope_flag']='Explicit title phrase requires subtype adjudication; not automatically excluded'
    if e['class']=='research' and row['source_id']=='https://openalex.org/S125754415' and row['title'].casefold().startswith('profile of '):
        e['class']='subtype_review_required'
        e['title_scope_flag']='PNAS person Profile conflicts with generic research-article; retain for subtype review'
    if e['class']=='research' and row['source_id']=='https://openalex.org/S125754415' and {key(v) for v in e.get('categories',[])} & {'profile','opinion','innerworkings'}:
        e['class']='subtype_review_required'
        e['title_scope_flag']='PNAS editorial subsection requires adjudication beyond generic JATS type'
    if e.get('scope_review_required'):
        e['class']='subtype_review_required'
    e['id']=hashlib.sha256(json.dumps([row['work_id'],e],sort_keys=True).encode()).hexdigest()[:24]
    return e


def base_evidence(row: dict) -> list[dict]:
    if not row['publisher_type']:return []
    family={'nature_v1_original_article_verified':'nature_v1','nature_catalogue_type_verified':'nature_catalogue'}.get(row['type_status'],'nature_page')
    if row.get('date_status')=='publisher_catalogue_online_verified':family='nature_catalogue'
    if row.get('date_status')=='aps_metadata_online_verified':
        from .boundary_publisher_record import receipt_records
        official=receipt_records(row['publisher_evidence'].removeprefix('../')).get(row['doi'])
        if not official:return []
        return [{'family':'aps','title':official['title'],'source_id':row['source_id'],
                 'kind':official['publisher_type'],'aps_journal':official.get('aps_journal',''),
                 'online_date':official['publisher_date'],'date_precision':'day',
                 'receipt':'../'+row['publisher_evidence'],'url':official['catalogue_url'],
                 'categories':official.get('categories',[])}]
    title=row['title']
    if row['type_status']=='publisher_type_verified':
        raw=BASE/(row['publisher_evidence'] or '')
        if raw.is_file() and raw.suffix=='.gz':
            tree=html.fromstring(gzip.decompress(raw.read_bytes()))
            titles=tree.xpath('//meta[@name="citation_title"]/@content')
            title=plain(titles[0]) if len(titles)==1 else ''
        else:title=''
    return [{'family':family,'title':title,'source_id':row['source_id'],'kind':row['publisher_type'],
        'online_date':row['publisher_online_date'] or '', 'date_precision':'day' if row['publisher_online_date'] else 'missing',
        'receipt':'../'+(row['publisher_evidence'] or ''),'url':'https://doi.org/'+row['doi'],'categories':[]}]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:path.write_text('');return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_decisions(path: Path, rows: list[dict], audit: Path) -> None:
    """Replace one shard atomically and preserve field-level changes, not snapshots."""
    previous = {r['work_id']: r for r in pq.read_table(path).to_pylist()} if path.exists() else {}
    changes = []
    for row in rows:
        old = previous.get(row['work_id'], {})
        fields = {k: {'before': old.get(k), 'after': v} for k,v in row.items() if k not in old or old[k] != v}
        if fields:
            changes.append({'work_id':row['work_id'], 'doi':row['doi'], 'shard':path.name,
                            'changed_fields':fields, 'evidence_ids':row['evidence_ids'],
                            'reason':'Recomputed from current evidence and dated annual rules'})
    temporary = path.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows),temporary)
    temporary.replace(path)
    at = datetime.now(timezone.utc).isoformat()
    with gzip.open(audit,'at',encoding='utf-8') as stream:
        for change in changes:
            stream.write(json.dumps({'applied_at':at, **change},ensure_ascii=False)+'\n')


def main() -> None:
    with (BASE/'journal_registry.csv').open(encoding='utf-8-sig') as stream:sources={r['source_id']:r for r in csv.DictReader(stream)}
    with (BASE/'journal_publisher_by_year.csv').open(encoding='utf-8-sig') as stream:
        annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in csv.DictReader(stream)}
    ev,secondary,catalogue_audit=load_evidence()
    dup,dup_audit=duplicate_map(sources)
    write_csv(OUT/'catalogue_completeness.csv',catalogue_audit);write_csv(OUT/'duplicate_resolution.csv',dup_audit)
    for folder in ['decisions']:(OUT/folder).mkdir(exist_ok=True)
    counts=Counter();cover=defaultdict(Counter);identity=Counter();types=Counter();dates=Counter();evidence_count=0
    with gzip_text(OUT/'decision_evidence.jsonl.gz') as trace, gzip_text(OUT/'secondary_evidence.jsonl.gz') as secondary_trace:
        for path in sorted((BASE/'papers').glob('*.parquet')):
            output=[];verified=0
            table=pq.read_table(path)
            for row in table.to_pylist():
                items=[match(row,e,sources[row['source_id']]) for e in ev.get(row['doi'],[])+base_evidence(row)]
                # Collapse identical evidence repeated in catalogue pages/year boundaries.
                items=list({e['id']:e for e in items}.values())
                secs=[s for s in secondary.get(row['doi'],[]) if key(plain(s['title']))==key(row['title'])]
                decision=decide(items,secs,dup.get(row['work_id'],''))
                status=decision['status'];counts[status]+=1
                for e in items:
                    identity[e['identity']]+=1
                    trace.write(json.dumps({'work_id':row['work_id'],'doi':row['doi'],**e},ensure_ascii=False)+'\n');evidence_count+=1
                if secs:secondary_trace.write(json.dumps({'work_id':row['work_id'],'doi':row['doi'],'records':secs},ensure_ascii=False)+'\n')
                found=[e for e in items if e['identity']=='matched']
                types['primary_evidence_present']+=bool(found)
                types['specific_primary_type_present']+=any(e['class']!='unresolved_type' for e in found)
                types['secondary_type_present']+=bool(secs)
                dates['primary_online_year_present']+=bool(decision['online_year'])
                dates['primary_online_year_differs_openalex']+=bool(decision['online_year'] and decision['online_year']!=row['year_openalex'])
                group='verified_research' if status=='included_verified_research' else 'excluded' if status.startswith('excluded') else 'pending'
                metrics=paper_metrics(row)
                for year in [0,decision['online_year'] if group=='verified_research' else row['year_openalex']]:
                    cover[(row['field'],group,year)].update(metrics)
                portfolio,portfolio_ids=portfolio_label(annual.get((row['source_id'],decision['online_year']),''),items,decision['online_year'])
                output.append({'work_id':row['work_id'],'doi':row['doi'],'field':row['field'],'source_id':row['source_id'],
                    'title':row['title'],'year_openalex':row['year_openalex'],'online_year':decision['online_year'],
                    'online_date':decision['online_date'],'publisher_types':decision['type'],'eligibility_status':status,
                    'reason':decision['reason'],'rejected_evidence_ids':json.dumps(decision.get('rejected_evidence_ids',[])),'duplicate_of':dup.get(row['work_id'],''),'evidence_ids':json.dumps(decision['evidence_ids']),
                    'secondary_evidence_present':bool(secs),
                    'secondary_pmcids':json.dumps(sorted({v['pmcid'] for v in secs if v.get('pmcid')})),
                    'secondary_online_dates':json.dumps(sorted({v['online_date'] for v in secs if v['online_date']})),
                    'secondary_pubtypes':json.dumps(sorted({t for v in secs for t in v['types']})),
                    'base_shard':'../papers/'+path.name,
                    'nature_portfolio_at_candidate_year':row['nature_portfolio_at_candidate_year'],
                    'nature_portfolio_at_verified_online_year':portfolio,'portfolio_evidence_ids':json.dumps(portfolio_ids),
                    })
                if group=='verified_research':verified+=1
            write_decisions(OUT/'decisions'/path.name, output, OUT/'decision_changes.jsonl.gz')
            print('built',path.name,'verified',verified,flush=True)
    current_shards={p.name for p in (BASE/'papers').glob('*.parquet')}
    for stale in (OUT/'decisions').glob('*.parquet'):
        if stale.name not in current_shards:stale.unlink()
    coverage=[{'field':k[0],'population':k[1],'year':k[2],**dict(v)} for k,v in sorted(cover.items())]
    write_csv(OUT/'coverage.csv',coverage)
    summary={'candidate_count':sum(counts.values()),'statuses':dict(counts),'evidence_identity':dict(identity),
        'type_audit':dict(types),'date_audit':dict(dates),'evidence_rows':evidence_count,'duplicate_aliases_excluded':len(dup)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
