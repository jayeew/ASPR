"""Batch official serial histories; confirm only explicit dated publisher intervals."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json

URL='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'


def collect(root: Path = ROOT, request_ids: list[str] | None = None) -> None:
    folder=root/'supplement/annual_nlm';raw=folder/'raw';raw.mkdir(exist_ok=True)
    with (folder/'queue.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    ids=sorted(set(request_ids)) if request_ids is not None else sorted({r['nlm_id'] for r in rows if r['nlm_id'].isdigit()})
    session=requests.Session();session.trust_env=False
    session.headers['User-Agent']='Fig8MetadataResearch/1.0 (public serial history; serial batches)'
    status={'target_nlm_ids':len(ids),'batches_total':(len(ids)+49)//50,'completed':0,'status':'running'}
    for offset in range(0,len(ids),50):
        group=ids[offset:offset+50];params={'db':'nlmcatalog','id':','.join(group),'retmode':'xml'}
        token=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest();path=raw/(token+'.json')
        if path.exists():r=json.loads(path.read_text())
        else:
            r={'url':URL,'params':params,'nlm_ids':group,'attempts':[],'retrieved_at':datetime.now(timezone.utc).isoformat()}
            for n in range(3):
                try:
                    response=session.get(URL,params=params,timeout=(10,45),allow_redirects=False)
                    r['attempts'].append({'http_status':response.status_code})
                    body=raw/(token+'.xml.gz');atomic_bytes(body,gzip.compress(response.content,mtime=0))
                    r.update(http_status=response.status_code,raw_receipt=str(body.relative_to(root)),sha256=hashlib.sha256(response.content).hexdigest(),retry_after=response.headers.get('Retry-After',''))
                    if response.status_code not in {500,502,503,504}:break
                except requests.RequestException as error:r['attempts'].append({'error':type(error).__name__})
                if n<2:time.sleep(2**n)
            atomic_json(path,r);time.sleep(1.05)
        status['completed']+=1
        if r.get('http_status')!=200:
            status.update(status='source_stopped_with_unattempted_remaining',last_receipt=str(path.relative_to(root)))
            atomic_json(folder/'status.json',status)
            return
        atomic_json(folder/'status.json',status)
        print('Serial-history batch',status['completed'],'/',status['batches_total'],flush=True)
    status['status']='collected_identity_and_interval_review_required';atomic_json(folder/'status.json',status)


if __name__=='__main__':
    collect()


def imprint_interval(text: str) -> tuple[int, int] | None:
    """Accept an explicit year interval; never substitute a journal founding year."""
    import re
    match=re.match(r'^\s*<?(?:[A-Za-z]+\.?\s+\d{1,2},?\s*)?\[?((?:18|19|20)\d{2})\]?\s*[-–]\s*\[?((?:18|19|20)\d{2})?\]?\s*(?:>|:|$)',text)
    if not match:return None
    return int(match.group(1)),int(match.group(2)) if match.group(2) else 9999


def propose(root: Path = ROOT, proposal_name: str = 'annual_proposals.json') -> dict:
    from collections import Counter
    from lxml import etree

    folder=root/'supplement/annual_nlm'
    marker='applied.json' if proposal_name=='annual_proposals.json' else proposal_name.replace('_proposals.json','_applied.json')
    if (folder/marker).exists():
        raise RuntimeError('Applied source rules are immutable evidence; use an additional proposal receipt')
    with (folder/'queue.csv').open(encoding='utf-8-sig') as f:queue=list(csv.DictReader(f))
    with (root/'fields/journal_registry.csv').open(encoding='utf-8-sig') as f:journals={r['source_id']:r for r in csv.DictReader(f)}
    with (root/'fields/journal_publisher_by_year.csv').open(encoding='utf-8-sig') as f:annual={(r['source_id'],int(r['year'])):r for r in csv.DictReader(f)}
    records={}
    for path in (folder/'raw').glob('*.json'):
        receipt=json.loads(path.read_text())
        if receipt.get('http_status')!=200:continue
        tree=etree.fromstring(gzip.decompress((root/receipt['raw_receipt']).read_bytes()),etree.XMLParser(resolve_entities=False,no_network=True))
        for r in tree.findall('NLMCatalogRecord'):records[r.findtext('NlmUniqueID')]=(r,receipt)
    publishers=['elsevier','wiley','mdpi','multidisciplinary digital','sage','wolters kluwer','karger','oxford university press','cambridge university press','taylor & francis','frontiers','american chemical society','american physical society','iop publishing','royal society of chemistry','international union of crystallography','public library of science','ieee','optica','american institute of physics']
    proposals=[];reviews=[];reasons=Counter();conflicts=[]
    for row in queue:
        sid=row['source_id'];pair=records.get(row['nlm_id']);reason='nlm_id_not_returned_requires_identifier_lookup'
        accepted=[]
        if pair:
            record,receipt=pair
            issns={i.text for i in record.findall('ISSN') if i.get('ValidYN')=='Y'}
            if not issns & set(json.loads(journals[sid]['issn'])):reason='catalog_issn_identity_not_confirmed'
            else:
                imprints=record.findall('./PublicationInfo/Imprint[@ImprintType="Current"]')
                if not imprints:
                    imprints=record.findall('./PublicationInfo/Imprint')
                    if len(imprints)!=1:imprints=[]
                reason='no_explicit_applicable_publisher_interval'
                for imprint in imprints:
                    entity=imprint.findtext('Entity') or ''
                    interval=imprint_interval(imprint.findtext('DateIssued') or '') or imprint_interval(imprint.findtext('ImprintFull') or '')
                    if not interval or not any(t in entity.casefold() for t in publishers):continue
                    for year in [2023,2024,2025]:
                        if not interval[0]<year<interval[1]:continue
                        old=annual.get((sid,year))
                        if not old:continue
                        if old['nature_portfolio_year']=='True':
                            conflicts.append({'source_id':sid,'year':year,'before':old,'nlm_id':row['nlm_id'],
                                              'publisher':entity,'imprint_full':imprint.findtext('ImprintFull'),
                                              'receipt':receipt['raw_receipt'],'status':'conflicting_annual_sources_requires_review'})
                            continue
                        if old['nature_portfolio_year']=='False':continue
                        proposal={'source_id':sid,'journal':row['journal'],'year':year,'label':'False','publisher':entity,
                                  'status':'official_nlm_explicit_imprint_interval','nlm_id':row['nlm_id'],'issns':sorted(issns),
                                  'source_url':URL,'receipt':receipt['raw_receipt'],'sha256':receipt['sha256'],
                                  'locator':f'NLMCatalogRecord[NlmUniqueID={row["nlm_id"]}]/PublicationInfo/Imprint[@ImprintType="{imprint.get("ImprintType")}"]',
                                  'imprint_full':imprint.findtext('ImprintFull'),'date_issued':imprint.findtext('DateIssued'),'interval':interval,
                                  'scope':'Catalog publisher imprint interval; boundary years excluded; not inferred from current entity or journal founding year alone',
                                  'before':old}
                        proposals.append(proposal);accepted.append(year)
                if accepted:reason='dated_annual_rule_proposed'
        reasons[reason]+=1
        reviews.append({**row,'review_status':reason,'years_proposed':accepted})
    atomic_json(folder/proposal_name,proposals);atomic_json(folder/'reviewed_journals.json',reviews)
    atomic_json(folder/'annual_conflicts.json',conflicts)
    summary={'queued_journals':len(queue),'returned_catalog_records':len(records),'annual_proposals':len(proposals),'reason_counts':dict(reasons),'status':'proposed_not_applied'}
    atomic_json(folder/'review_summary.json',summary);print(json.dumps(summary),flush=True)
    return summary
