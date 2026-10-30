"""Bounded membership review of unknown journals in the active Fig8 cohort."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .additional_dated_imprints import ENTITIES
from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import electronic_date, key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv, write_csv

FOLDER = ROOT / 'supplement/current_cohort_portfolio'



def native_article(raw: bytes, doi: str) -> etree._Element:
    """Select one DOI-matched article and normalize only its JATS namespace."""
    document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
    tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
    articles = [tree] if etree.QName(tree).localname == 'article' else tree.xpath('//*[local-name()="article"]')
    matches = [a for a in articles if doi.lower() in {
        str(v).lower() for v in a.xpath('.//*[local-name()="article-id" and @pub-id-type="doi"]/text()')}]
    if len(matches) != 1:
        raise ValueError('No unique DOI-matched native article')
    article = deepcopy(matches[0])
    namespace = etree.QName(article).namespace
    if namespace:
        if not namespace.startswith(('https://jats.nlm.nih.gov/', 'http://jats.nlm.nih.gov/')):
            raise ValueError('Unrecognized native article namespace')
        for element in article.iter():
            if isinstance(element.tag, str) and etree.QName(element).namespace == namespace:
                element.tag = etree.QName(element).localname
        etree.cleanup_namespaces(article)
    return article


def catalogue_indexes() -> dict:
    result = {}
    for year in [2023,2024,2025]:
        path = ROOT / f'fields/official/journals_catalog_{year}.xlsx'
        book = openpyxl.load_workbook(path,read_only=True,data_only=True)
        rows = list(book.active.values)
        header = next(i for i,r in enumerate(rows) if r[0]=='Title')
        index = defaultdict(list)
        for n,values in enumerate(rows[header+1:],header+2):
            row = dict(zip(rows[header],values))
            row['row_number'] = n
            for col in ['ISSN print','ISSN electronic']:
                if row.get(col): index[row[col]].append(row)
        result[year] = {'index':index,'receipt':str(path.relative_to(ROOT)),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'url':json.loads(path.with_suffix('.json').read_text())['url']}
        book.close()
    return result


def catalogue_rule(sid: str, year: int, registry: dict, cats: dict) -> dict | None:
    title = registry['journal_name']
    for cy in [year,year+1]:
        if cy not in cats: continue
        cat = cats[cy]
        matches = {r['row_number']:r for issn in json.loads(registry['issn']) for r in cat['index'].get(issn,[])
                   if key(r.get('Title'))==key(title)}
        if len(matches)!=1: continue
        row = next(iter(matches.values()))
        # A later catalogue is only retrospective evidence if it explicitly gives the start year.
        if cy!=year and f'New Start in {year}' not in str(row.get('Comments','')): continue
        imprint = row.get('Imprint') or ''
        if imprint not in {'Nature Portfolio','Springer','BioMed Central','Palgrave Macmillan'}: continue
        return {'source_id':sid,'year':year,'journal':title,'label':str(imprint=='Nature Portfolio'),
                'publisher':imprint,'source_url':cat['url'],'raw_receipt':cat['receipt'],'raw_sha256':cat['sha256'],
                'catalogue_year':cy,'catalogue_row':row,
                'scope':'Exact ISSN and title in official annual catalogue; later catalogue used only for its explicit New Start year statement. No inference from catalogue absence.'}
    return None


def native_rule(pair: tuple, papers: list[dict], registry: dict, db: sqlite3.Connection) -> dict | None:
    proofs = []
    for p in papers:
        for (raw,) in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",(p['work_id'],)):
            e = json.loads(raw)
            publishers = set(e.get('publishers',[]))
            if (e.get('family')!='archived_jats' or e.get('identity')!='matched' or not publishers
                    or not publishers<=ENTITIES or str(e.get('online_date',''))[:4]!=str(pair[1]) or not e.get('receipt')):
                continue
            path = ROOT / 'fields/qualification' / e['receipt']
            if not path.exists(): continue
            data = path.read_bytes()
            if path.suffix=='.gz': data=gzip.decompress(data)
            try:
                native=jats_metadata(data+b'</article>' if data.rstrip().endswith(b'</front>') else data)
            except (ValueError,TypeError):
                continue
            date,_=electronic_date(native.get('dates',[]))
            if (date!=e['online_date'] or p['doi'] not in {x.lower() for x in native.get('doi',[])}
                    or set(native.get('publisher',[]))!=publishers
                    or not set(native.get('issn',[])) & set(json.loads(registry['issn']))):
                continue
            proofs.append({'doi':p['doi'],'work_id':p['work_id'],'evidence_id':e['id'],
                           'online_date':date,'publishers':sorted(publishers),'source_url':e['url'],
                           'raw_receipt':str(path.relative_to(ROOT)),
                           'raw_sha256':hashlib.sha256(data).hexdigest()})
            break
        if len(proofs)>=2: break
    if not proofs: return None
    return {'source_id':pair[0],'year':pair[1],'journal':registry['journal_name'],'label':'False',
            'publisher':' | '.join(sorted({v for p in proofs for v in p['publishers']})),
            'source_url':proofs[0]['source_url'],'raw_receipt':proofs[0]['raw_receipt'],
            'raw_sha256':proofs[0]['raw_sha256'],'proofs':proofs,
            'scope':'Same journal/year dated native publisher statements, matched DOI and ISSN; only publishers in existing accepted non-Portfolio policy. Annual working rule; reopen on transfer/conflicting evidence.'}


def prepare() -> None:
    FOLDER.mkdir(exist_ok=True)
    selected=pq.read_table(ROOT/'analysis/cohort/selection.parquet',filters=[('included','=',True)]).to_pylist()
    unknown=[r for r in selected if r['portfolio'] not in {'True','False'}]
    registry={r['source_id']:r for r in read_csv(ROOT/'fields/journal_registry.csv')}
    groups=defaultdict(list)
    for r in unknown:groups[(r['source_id'],r['online_year'])].append(r)
    atomic_json(FOLDER/'scope.json',{'paper_n':len(unknown),'journal_n':len({r['source_id'] for r in unknown}),
                'journal_year_n':len(groups),'pairs':[[s,y,len(rs)] for (s,y),rs in groups.items()],
                'source':'analysis/cohort/selection.parquet','scope':'Current included papers with unknown annual membership only'})
    cats=catalogue_indexes()
    db=sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro',uri=True)
    rules,unresolved=[],[]
    for pair,papers in groups.items():
        rule=catalogue_rule(*pair,registry[pair[0]],cats)
        if rule is None:
            rule=native_rule(pair,papers,registry[pair[0]],db)
        if rule:rules.append(rule)
        else:unresolved.append({'source_id':pair[0],'year':pair[1],'journal':registry[pair[0]]['journal_name'],
                                'paper_n':len(papers),'current_registry_publisher':registry[pair[0]]['host_name'],
                                'status':'No accepted dated membership proof in reviewed cache; current publisher and list absence alone not treated as annual proof'})
    db.close()
    atomic_json(FOLDER/'rules.json',rules)
    atomic_json(FOLDER/'unmatched.json',unresolved)
    print(json.dumps({'rules':len(rules),'positive_rules':[r for r in rules if r['label']=='True'],
                      'resolved_current_papers':sum(len(groups[(r['source_id'],r['year'])]) for r in rules)},ensure_ascii=False),flush=True)


def refresh_active(rules: list[dict]) -> None:
    by_pair={(r['source_id'],r['year']):r for r in rules}
    path=ROOT/'analysis/cohort/selection.parquet'
    table=pq.read_table(path)
    rows=table.to_pylist()
    changes=[]
    for r in rows:
        rule=by_pair.get((r['source_id'],r['online_year']))
        if rule and r['portfolio'] not in {'True','False'}:
            changes.append({'doi':r['doi'],'work_id':r['work_id'],'journal':r['journal'],'field':r['field'],
                'source_id':r['source_id'],'year':r['online_year'],'in_current_analysis':r['included'],
                'before':r['portfolio'],'after':rule['label'],'evidence':str((FOLDER/'rules.json').relative_to(ROOT))})
            r['portfolio']=rule['label']
    table=table.set_column(table.schema.get_field_index('portfolio'),table.schema.field('portfolio'),pa.array([r['portfolio'] for r in rows]))
    temp=path.with_suffix('.tmp.parquet');pq.write_table(table,temp,compression='zstd');temp.replace(path)
    old_changes = read_csv(FOLDER/'paper_label_changes.csv') if (FOLDER/'paper_label_changes.csv').exists() else []
    write_csv(FOLDER/'paper_label_changes.csv',old_changes + changes)
    db=sqlite3.connect(ROOT/'analysis/contributions/contributions.sqlite')
    db.executemany('UPDATE papers SET portfolio=? WHERE doi=?',[(r['after'],r['doi']) for r in changes]);db.commit();db.close()
    current=[r for r in rows if r['included']]
    counts=Counter(r['portfolio'] if r['portfolio'] in {'True','False'} else 'unknown' for r in current)
    summary_path=ROOT/'analysis/cohort/summary.json'
    summary=json.loads(summary_path.read_text());summary['counts']['retained_brand_unknown']=counts['unknown']
    summary['portfolio_counts']=dict(counts);summary['portfolio_update_receipt']=str((FOLDER/'result.json').relative_to(ROOT))
    atomic_json(summary_path,summary)
    # Update existing coverage flags without rereading abstracts or author arrays.
    delta=Counter()
    for r in changes:
        delta[(r['source_id'],r['field'])]+=1;delta[(r['source_id'],'all_three_fields')]+=1
    for filename in ['all_journals.csv','priority_journals.csv','comparison_entry_assessment.csv']:
        p=ROOT/'analysis/journal_readiness'/filename
        data=read_csv(p)
        for row in data:
            row['brand_unknown_n']=str(max(0,int(row['brand_unknown_n'])-delta[(row['source_id'],row['field'])]))
            if row['brand_unknown_n']=='0':row['formal_blockers']='|'.join(x for x in row['formal_blockers'].split('|') if x!='annual_brand_unknown')
        write_csv(p,data)
    active_changes=[r for r in changes if r['in_current_analysis']]
    result={'analysis_papers':len(current),'before_unknown':json.loads((FOLDER/'scope.json').read_text())['paper_n'],
            'new_nature_portfolio':sum(r['after']=='True' for r in active_changes),
            'new_non_nature_portfolio':sum(r['after']=='False' for r in active_changes),
            'remaining_unknown':counts['unknown'],'current_portfolio_counts':dict(counts),
            'positive_papers':[r for r in active_changes if r['after']=='True'],
            'paper_or_eligibility_changes':0,'contribution_extraction_calls':0,
            'source_use':'Official annual catalogues, dated licensing title lists, existing NLM imprint intervals and matched native publication statements; no article downloads',
            'limitation':'Unresolved journals were checked but lack accepted annual proof. Current names/publisher labels are retained as clues; absence from a list alone is not negative proof.'}
    atomic_json(FOLDER/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def apply() -> None:
    rules=json.loads((FOLDER/'rules.json').read_text())
    unmatched=json.loads((FOLDER/'unmatched.json').read_text())
    annual = read_csv(ROOT/'fields/journal_publisher_by_year.csv')
    pending_pairs = {(r['source_id'], int(r['year'])) for r in annual
                     if r['nature_portfolio_year'] not in {'True', 'False'}}
    apply_proposals(rules,FOLDER,unmatched,apply_pairs=pending_pairs,rule_status='current_cohort_dated_membership',
                    reason='Dated publisher catalogue/licensing title list, explicit NLM publisher interval, or matched dated native publisher proof',
                    route='current_cohort_journal_membership',refresh_derived=False)
    conflicts=json.loads((FOLDER/'conflicts.json').read_text())
    blocked={(r['source_id'],int(r['year'])) for r in conflicts}
    refresh_active([r for r in rules if (r['source_id'],r['year']) not in blocked])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','apply'])
    args=parser.parse_args()
    prepare() if args.action=='prepare' else apply()
