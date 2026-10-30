"""Discover public journal batch formats for the fixed DOI-less OJS landing-page scope."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from lxml import etree

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

FOLDER = ROOT / 'supplement/missing_doi'
NS = {'oai':'http://www.openarchives.org/OAI/2.0/'}


def formats(raw: bytes) -> list[dict]:
    tree = etree.fromstring(raw, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    if tree.tag != '{http://www.openarchives.org/OAI/2.0/}OAI-PMH':
        return []
    return [{'prefix':node.findtext('oai:metadataPrefix',namespaces=NS),
             'schema':node.findtext('oai:schema',namespaces=NS)}
            for node in tree.findall('oai:ListMetadataFormats/oai:metadataFormat',NS)]


def fetch(url: str) -> dict:
    token = hashlib.sha256(url.encode()).hexdigest()
    path = FOLDER / 'oai_raw' / (token+'.json')
    if path.exists():
        return {**json.loads(path.read_text()),'reused':True}
    session = requests.Session(); session.trust_env = False
    result = {'url':url,'params':{'verb':'ListMetadataFormats'},'attempts':[],
              'retrieved_at':datetime.now(timezone.utc).isoformat(),'formats':[]}
    for attempt in range(3):
        paced(url)
        try:
            response = session.get(url,params=result['params'],timeout=(10,25))
            result['attempts'].append({'http_status':response.status_code})
            result.update(http_status=response.status_code,final_url=response.url)
            raw_path = path.with_suffix('.xml.gz')
            atomic_bytes(raw_path,gzip.compress(response.content))
            result['raw_receipt'] = str(raw_path.relative_to(ROOT))
            if response.status_code == 200:
                try:result['formats'] = formats(response.content)
                except etree.XMLSyntaxError:result['parse_error'] = 'not_valid_oai_xml'
                result['status'] = 'public_formats_available' if result['formats'] else 'no_oai_formats_in_response'
                break
            result['status'] = 'access_restricted' if response.status_code in {401,403,429} else 'endpoint_unavailable'
            if response.status_code not in {500,502,503,504}:
                break
        except requests.RequestException as error:
            result['attempts'].append({'error':type(error).__name__})
            result['status'] = 'transport_unresolved'
        if attempt<2:time.sleep(2**attempt)
    session.close();atomic_json(path,result)
    return result


def run() -> dict:
    (FOLDER / 'oai_raw').mkdir(exist_ok=True)
    queue = deque(json.loads((FOLDER / 'oai_scope.json').read_text())['endpoints'])
    total = len(queue);stats=Counter();blocked={}
    output = (FOLDER / 'oai_results.jsonl').open('w')
    with output, ThreadPoolExecutor(max_workers=8) as pool:
        jobs={}
        def submit() -> None:
            while queue:
                entry=queue.popleft();host=urlsplit(entry['url']).netloc
                if host in blocked:
                    output.write(json.dumps({**entry,'status':'host_restriction_not_requested','restriction_receipt':blocked[host]})+'\n')
                    stats['host_restriction_not_requested']+=1
                    continue
                jobs[pool.submit(fetch,entry['url'])]=entry
                break
        for _ in range(8):submit()
        while jobs:
            done,_=wait(jobs,return_when=FIRST_COMPLETED)
            for future in done:
                entry=jobs.pop(future);result=future.result();stats[result['status']]+=1
                stats['candidate_links_processed']+=len(entry['candidates'])
                stats['requests']+=0 if result.get('reused') else len(result['attempts'])
                if result.get('http_status') in {401,403,429}:
                    blocked[urlsplit(entry['url']).netloc]=result.get('raw_receipt','')
                output.write(json.dumps({**entry,**result},ensure_ascii=False)+'\n');output.flush()
                atomic_json(FOLDER/'oai_status.json',{'status':'running','fixed_endpoints':total,**stats})
                submit()
    result={'status':'fixed_endpoint_discovery_complete_not_candidate_qualification','fixed_endpoints':total,**stats}
    atomic_json(FOLDER/'oai_status.json',result);print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':run()
