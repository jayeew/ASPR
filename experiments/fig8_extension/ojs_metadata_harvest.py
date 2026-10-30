"""Finite cached OAI metadata acquisition for exact existing DOI-less landing pages."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests
from lxml import etree

from .fig8b_qualification_build import key, plain
from .missing_doi_oai import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

NS={'o':'http://www.openarchives.org/OAI/2.0/','dc':'http://purl.org/dc/elements/1.1/'}
RAW=FOLDER/'records_raw'


@lru_cache(maxsize=4)
def official_receipts(root: str) -> dict[str, dict]:
    """Index already retained direct GET evidence; never retry a known access denial."""
    base = Path(root); indexed = {}
    for relative in ['supplement/official_entities', 'supplement/journal_type_policies']:
        for path in sorted((base / relative).glob('*.json')):
            try:
                value = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict) or value.get('params') or not value.get('url'):
                continue
            if value.get('http_status') not in {200, 401, 403, 429} or not value.get('raw_receipt'):
                continue
            raw = base / value['raw_receipt']
            if not raw.is_file():
                continue
            candidate = {**value, 'params': {}, 'reused_receipt': True,
                         'reused_from': str(path.relative_to(base))}
            prior = indexed.get(value['url'])
            if prior is None or (value['http_status'] == 200 and prior['http_status'] != 200):
                indexed[value['url']] = candidate
    return indexed


def fetch(url: str, params: dict) -> dict:
    token=hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest()
    path=RAW/(token+'.json')
    if path.exists():return json.loads(path.read_text())
    if not params:
        previous = official_receipts(str(ROOT)).get(url)
        if previous is not None:
            atomic_json(path, previous)
            return previous
    # Reuse the two successful original protocol-control pages, without a second request.
    for name in ['dc_control','jats_control_ajr']:
        control_path=FOLDER/(name+'.json')
        if not control_path.exists():continue
        control=json.loads(control_path.read_text())
        if control['url']==url and control['params']==params:
            result={**control,'raw_receipt':str(Path(control['raw_receipt']).relative_to(ROOT)),
                    'attempts':[{'http_status':control['http_status']}], 'reused_control':True}
            atomic_json(path,result);return result
    session=requests.Session();session.trust_env=False
    result={'url':url,'params':params,'attempts':[], 'retrieved_at':datetime.now(timezone.utc).isoformat()}
    for n in range(3):
        paced(url)
        try:
            response=session.get(url,params=params,timeout=(10,35))
            result['attempts'].append({'http_status':response.status_code})
            result['http_status']=response.status_code
            raw=path.with_suffix('.xml.gz');atomic_bytes(raw,gzip.compress(response.content))
            result['raw_receipt']=str(raw.relative_to(ROOT))
            if response.status_code not in {500,502,503,504}:break
            retry=response.headers.get('Retry-After','')
            if retry:
                result['retry_after']=retry;break
        except requests.RequestException as error:
            result['attempts'].append({'error':type(error).__name__})
        if n<2:time.sleep(2**n)
    session.close();atomic_json(path,result);return result


def record_info(entry: etree._Element) -> dict:
    identifier=entry.findtext('o:header/o:identifier',namespaces=NS) or ''
    node=entry.find('o:metadata',NS)
    if node is None:return {'identifier':identifier,'status':'deleted_or_no_metadata'}
    titles=node.xpath('.//*[local-name()="article-title" or (local-name()="title" and namespace-uri()="http://purl.org/dc/elements/1.1/")]')
    links=node.xpath('.//dc:identifier/text()',namespaces=NS)
    links+=node.xpath('.//*[local-name()="self-uri"]/@*[local-name()="href"]')
    dois=node.xpath('.//*[local-name()="article-id" and @pub-id-type="doi"]/text()')
    dois += [v.removeprefix('https://doi.org/').removeprefix('http://dx.doi.org/') for v in links if 'doi.org/' in v]
    return {'identifier':identifier,'titles':[plain(''.join(v.itertext())) for v in titles],
            'landing_links':[v.rstrip('/') for v in links if v.startswith(('http://','https://'))],
            'declared_dois':sorted(set(dois)),
            'types':node.xpath('.//dc:type/text() | .//*[local-name()="article"]/@article-type',namespaces=NS),
            'sections':node.xpath('.//*[local-name()="article-categories"]//*[local-name()="subject"]/text()'),
            'status':'metadata_found_type_requires_review'}


def harvest(source: dict) -> dict:
    url=source['url'];token=hashlib.sha256(url.encode()).hexdigest()
    output=FOLDER/'harvested'/(token+'.json')
    if output.exists():return json.loads(output.read_text())
    available={v['prefix'] for v in source['formats']}
    # DC is lighter; use the successfully reviewed JATS adapter for this native source.
    prefix='oai_openaire_jats' if 'asean-journal-radiology.org/' in url else 'oai_dc'
    if prefix not in available:
        result={'url':url,'status':'no_validated_available_format','candidates':source['candidates']}
        atomic_json(output,result);return result
    pending={r['work_id']:r for r in source['candidates']};matched={};pages=[];seen=set();seen_pages=set();identifier_prefix=''
    params={'verb':'ListRecords','metadataPrefix':prefix};individual=[];reason=''
    while params:
        receipt=fetch(url,params);pages.append(receipt)
        if receipt.get('http_status')!=200:
            reason='source_request_failed';break
        try:
            tree=etree.fromstring(gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes()),
                                 etree.XMLParser(resolve_entities=False,no_network=True))
        except (etree.XMLSyntaxError,EOFError,gzip.BadGzipFile):
            reason='invalid_oai_payload';break
        errors=tree.findall('o:error',NS)
        if errors:reason='oai_error:'+','.join(e.get('code','') for e in errors);break
        entries=tree.findall('o:ListRecords/o:record',NS)+tree.findall('o:GetRecord/o:record',NS)
        page_ids=tuple(e.findtext('o:header/o:identifier',namespaces=NS) for e in entries)
        if params['verb']=='ListRecords' and page_ids in seen_pages:
            reason='repeated_page_identifiers';break
        seen_pages.add(page_ids)
        for entry in entries:
            info=record_info(entry)
            native=re.fullmatch(r'(.+:article/)(\d+)',info['identifier'])
            if native and not identifier_prefix:identifier_prefix=native[1]
            for wid,candidate in list(pending.items()):
                if candidate['landing_page_url'].rstrip('/') not in info.get('landing_links',[]):continue
                title_ok=key(candidate['title']) in {key(t) for t in info['titles']}
                matched[wid]={**info,'raw_receipt':receipt['raw_receipt'],'title_identity':'exact' if title_ok else 'mismatch',
                              'qualification_unchanged':True}
                del pending[wid]
        if not pending:reason='all_target_links_returned';break
        if params['verb']=='GetRecord':
            params=individual.pop(0) if individual else None
            if not params:reason='fixed_identifiers_processed'
            continue
        next_token=tree.find('o:ListRecords/o:resumptionToken',NS)
        if next_token is None or not (next_token.text or '').strip():reason='publisher_listing_exhausted';break
        value=next_token.text.strip()
        if value in seen:reason='repeated_pagination_token';break
        seen.add(value)
        total=next_token.get('completeListSize','');cursor=next_token.get('cursor','0')
        pages_left=math.ceil((int(total)-int(cursor)-len(entries))/max(1,len(entries))) if total.isdigit() and cursor.isdigit() else None
        if identifier_prefix and pages_left is not None and len(pending)<pages_left:
            identifiers=[re.search(r'/article/view/(\d+)/?$',c['landing_page_url']) for c in pending.values()]
            if all(identifiers):
                individual=[{'verb':'GetRecord','metadataPrefix':prefix,'identifier':identifier_prefix+m[1]} for m in identifiers]
                params=individual.pop(0);continue
        params={'verb':'ListRecords','resumptionToken':value}
    result={'url':url,'status':reason,'matched':matched,'unresolved':list(pending.values()),
            'page_receipts':[p.get('raw_receipt','') for p in pages], 'qualification_unchanged':True,
            'limitation':'OAI datestamp is not publication date; generic research-article export is not a validated journal type rule.'}
    atomic_json(output,result);return result


def run() -> None:
    RAW.mkdir(exist_ok=True);(FOLDER/'harvested').mkdir(exist_ok=True)
    titles={r['work_id']:r['title'] for r in map(json.loads,(FOLDER/'source_leads.jsonl').open())}
    done=set();summary={'sources_processed':0,'exact_title_matches':0,'link_matches_title_mismatch':0}
    while True:
        sources=[]
        for line in (FOLDER/'oai_results.jsonl').read_text().splitlines():
            source=json.loads(line)
            if source['url'] in done or source.get('status')!='public_formats_available':continue
            for candidate in source['candidates']:candidate['title']=titles[candidate['work_id']]
            sources.append(source);done.add(source['url'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(harvest,sources):
                summary['sources_processed']+=1
                for v in result.get('matched',{}).values():
                    summary['exact_title_matches' if v['title_identity']=='exact' else 'link_matches_title_mismatch']+=1
                atomic_json(FOLDER/'harvest_status.json',{'status':'running','qualification_unchanged':True,**summary})
        if json.loads((FOLDER/'oai_status.json').read_text())['status']!='running':break
        time.sleep(30)
    atomic_json(FOLDER/'harvest_status.json',{'status':'fixed_discovered_sources_processed_identity_type_review_required',**summary})


if __name__=='__main__':run()
