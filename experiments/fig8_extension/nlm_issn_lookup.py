"""Find unresolved journal catalogue identities by batched exact ISSNs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pyarrow.parquet as pq
import requests
from lxml import etree

from .annual_nlm import collect, propose
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json
from .three_journal_apply import write_csv


def run(all_journals: bool = False) -> None:
    folder=ROOT/'supplement/annual_nlm';lookup=folder/('full_lookup' if all_journals else 'lookup');lookup.mkdir(exist_ok=True)
    with (ROOT/'fields/journal_registry.csv').open(encoding='utf-8-sig') as f:journals={r['source_id']:r for r in csv.DictReader(f)}
    reviewed=json.loads((folder/'reviewed_journals.json').read_text())
    done={r['source_id'] for r in reviewed if r['review_status']!='nlm_id_not_returned_requires_identifier_lookup'}
    counts=Counter()
    for path in (ROOT/'fields/qualification/decisions').glob('*.parquet'):
        for r in pq.read_table(path,columns=['source_id','eligibility_status','nature_portfolio_at_verified_online_year'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist():
            if r['source_id'] not in done and r['nature_portfolio_at_verified_online_year'] not in ['True','False']:counts[r['source_id']]+=1
    if all_journals:
        # Previous exact-ISSN searches, including no-match results, are not retried.
        for previous in [folder/'lookup/identity_results.json',lookup/'identity_results.json']:
            if previous.exists():done.update(r['source_id'] for r in json.loads(previous.read_text()))
        with (ROOT/'fields/journal_publisher_by_year.csv').open(encoding='utf-8-sig') as f:
            unresolved={r['source_id'] for r in csv.DictReader(f) if r['nature_portfolio_year'] not in ['True','False']}
        counts=Counter({sid:counts.get(sid,0) for sid in unresolved-done})
    issn_sources=defaultdict(set)
    for sid in counts:
        for issn in json.loads(journals[sid]['issn'] or '[]'):issn_sources[issn].add(sid)
    issns=sorted(issn_sources)
    atomic_json(lookup/'scope.json',{'journals':len(counts),'included_unknown_papers':sum(counts.values()),'issns':len(issns),'scope':'Previously unmapped or not-returned journal identities only; no repeat of catalogues lacking dates'})
    session=requests.Session();session.trust_env=False
    session.headers['User-Agent']='Fig8MetadataResearch/1.0 (batch journal identity lookup)'
    uids=set()
    for offset in range(0,len(issns),40):
        group=issns[offset:offset+40]
        params={'db':'nlmcatalog','term':' OR '.join('"'+i+'"[ISSN]' for i in group),'retmode':'json','retmax':500}
        token=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest();path=lookup/(token+'.json')
        if path.exists():r=json.loads(path.read_text())
        else:
            r={'url':'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi','params':params,'issns':group,'attempts':[],'retrieved_at':datetime.now(timezone.utc).isoformat()}
            for n in range(3):
                try:
                    response=session.get(r['url'],params=params,timeout=(10,45),allow_redirects=False)
                    r['attempts'].append({'http_status':response.status_code});r['http_status']=response.status_code
                    raw=lookup/(token+'.body.gz');atomic_bytes(raw,gzip.compress(response.content,mtime=0));r['raw_receipt']=str(raw.relative_to(ROOT))
                    if response.status_code==200:r['response']=response.json()
                    if response.status_code not in {500,502,503,504}:break
                except (requests.RequestException,ValueError) as error:r['attempts'].append({'error':type(error).__name__})
                if n<2:time.sleep(2**n)
            atomic_json(path,r);time.sleep(1.05)
        if r.get('http_status')!=200:
            atomic_json(lookup/'status.json',{'status':'source_stopped','receipt':str(path.relative_to(ROOT))});return
        result=r.get('response',{}).get('esearchresult',{})
        if int(result.get('count',0))>len(result.get('idlist',[])):
            raise RuntimeError('Truncated catalogue identity search; split group before continuing')
        uids.update(result.get('idlist',[]))
        print('ISSN lookup',min(offset+40,len(issns)),'/',len(issns),flush=True)
    records={}
    for path in (folder/'raw').glob('*.xml.gz'):
        tree=etree.fromstring(gzip.decompress(path.read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
        for r in tree.findall('NLMCatalogRecord'):records[r.findtext('NlmUniqueID')]=(r,str(path.relative_to(ROOT)))
    known={str(int(i)) for i in records if i.isdigit()}
    collect(request_ids=sorted(uids-known))
    for path in (folder/'raw').glob('*.xml.gz'):
        tree=etree.fromstring(gzip.decompress(path.read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
        for r in tree.findall('NLMCatalogRecord'):records[r.findtext('NlmUniqueID')]=(r,str(path.relative_to(ROOT)))
    matches=defaultdict(list)
    for nlmid,(record,receipt) in records.items():
        ids={n.text for n in record.findall('ISSN') if n.get('ValidYN')=='Y'}
        for sid in set().union(*(issn_sources.get(i,set()) for i in ids)):
            matches[sid].append({'nlm_id':nlmid,'mapping_receipt':receipt,'matched_issns':sorted(ids & set(json.loads(journals[sid]['issn'] or '[]')))})
    with (folder/'queue.csv').open(encoding='utf-8-sig') as f:queue=list(csv.DictReader(f))
    by_sid={r['source_id']:r for r in queue};out=[]
    for sid,n in counts.items():
        candidates=matches[sid]
        status='no_issn_cannot_lookup' if not json.loads(journals[sid]['issn'] or '[]') else 'exact_issn_unique_catalog_match' if len(candidates)==1 else 'ambiguous_catalog_matches' if candidates else 'no_catalog_match'
        out.append({'source_id':sid,'journal':journals[sid]['journal_name'],'unknown_included_papers':n,'status':status,'matches':candidates})
        if len(candidates)==1:by_sid[sid]={'source_id':sid,'journal':journals[sid]['journal_name'],'unknown_included_papers':n,'nlm_id':candidates[0]['nlm_id'],'mapping_receipt':candidates[0]['mapping_receipt']}
    write_csv(folder/'queue.csv',list(by_sid.values()))
    atomic_json(lookup/'identity_results.json',out)
    atomic_json(lookup/'status.json',{'status':'identity_lookup_complete_interval_review_follows','journals':len(out),'unique_matches':sum(r['status']=='exact_issn_unique_catalog_match' for r in out),'returned_uids':len(uids)})
    propose(proposal_name='full_scope_proposals.json' if all_journals else 'lookup_proposals.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-journals',action='store_true')
    run(parser.parse_args().all_journals)
