"""Finite official OAI journal harvest; retain raw XML and parse front matter only."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import etree

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, jats_metadata

URL='https://oai-pmh.copernicus.org/oai.php'
NS={'o':'http://www.openarchives.org/OAI/2.0/'}


def parse_article(article: etree._Element) -> dict:
    node=copy.deepcopy(article)
    namespace=etree.QName(node).namespace
    for child in node.iter():
        if isinstance(child.tag,str) and etree.QName(child).namespace==namespace:
            child.tag=etree.QName(child).localname
    # This publisher's NLM 2.3 records use contrib@rid for affiliation links.
    for author in node.findall('./front/article-meta/contrib-group/contrib'):
        if author.get('rid') and not author.xpath('./xref[@ref-type="aff"]'):
            etree.SubElement(author,'xref',{'ref-type':'aff','rid':author.get('rid')})
    etree.cleanup_namespaces(node)
    return jats_metadata(etree.tostring(node))


def fetch(params: dict, folder: Path, page: int = 1) -> dict:
    identity=URL+json.dumps(params,sort_keys=True)
    if 'resumptionToken' in params and page>2:identity+=f':page:{page}'
    token=hashlib.sha256(identity.encode()).hexdigest()
    receipt=folder/(token+'.json');raw=folder/(token+'.xml.gz')
    if receipt.exists():return json.loads(receipt.read_text())
    session=requests.Session();session.trust_env=False
    record={'url':URL,'params':params,'retrieved_at':datetime.now(timezone.utc).isoformat(),'attempts':[],'chain_page':page}
    for attempt in range(3):
        try:
            response=session.get(URL,params=params,timeout=(10,60))
            record.update(http_status=response.status_code,raw_receipt=str(raw),sha256=hashlib.sha256(response.content).hexdigest())
            atomic_bytes(raw,gzip.compress(response.content));record['attempts'].append({'http_status':response.status_code})
            if response.status_code not in {500,502,503,504}:break
        except requests.RequestException as error:
            record['attempts'].append({'error':type(error).__name__})
        if attempt<2:time.sleep(2**attempt)
    atomic_json(receipt,record);time.sleep(1.05)
    return record


def read_xml(record: dict, out: Path, context: dict) -> etree._Element | None:
    """Preserve malformed source evidence without guessing or repeating retrieval."""
    try:
        raw=gzip.decompress(Path(record['raw_receipt']).read_bytes())
        return etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True,huge_tree=True))
    except (OSError,etree.XMLSyntaxError) as error:
        # One official DC feed emits a literal ampersand in creator text.
        # Normalize only that text node; DC creators never supply author evidence.
        if isinstance(error,etree.XMLSyntaxError) and 'xmlParseEntityRef: no name' in str(error):
            def creator(match: re.Match[bytes]) -> bytes:
                value=re.sub(rb'&(?!#(?:[0-9]+|x[0-9A-Fa-f]+);|[A-Za-z_][A-Za-z0-9_.:-]*;)',b'&amp;',match.group(1))
                return b'<dc:creator>'+value+b'</dc:creator>'
            fixed=re.sub(rb'<dc:creator>([^<]*)</dc:creator>',creator,raw)
            if fixed!=raw:
                try:
                    tree=etree.fromstring(fixed,etree.XMLParser(resolve_entities=False,no_network=True,huge_tree=True))
                    target=out/('normalized_dc_'+hashlib.sha256(raw).hexdigest()+'.xml.gz')
                    atomic_bytes(target,gzip.compress(fixed))
                    with (out/'normalizations.jsonl').open('a') as stream:
                        stream.write(json.dumps({**context,'original_receipt':record,'normalized_receipt':str(target),
                            'rule':'Escape bare ampersands in dc:creator text only; discovery feed only, no author metadata inferred'})+'\n')
                    return tree
                except etree.XMLSyntaxError:
                    pass
        event={**context,'status':'cached_source_xml_invalid','receipt':record,
               'error':str(error),'qualification_not_resolved':True}
        with (out/'parse_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        return None


def run() -> None:
    folder=ROOT/'supplement/official_bulk';folder.mkdir(exist_ok=True)
    out=ROOT/'supplement/copernicus';out.mkdir(exist_ok=True)
    source=sqlite3.connect(ROOT/'supplement/tasks.sqlite')
    dois={r[0] for r in source.execute("SELECT doi FROM papers WHERE publisher='Copernicus Publications' AND doi LIKE '10.5194/%'")};source.close()
    scope_path=out/'scope.json'
    if scope_path.exists():scope=json.loads(scope_path.read_text())
    else:
        sets=sorted({re.match(r'10.5194/([a-z]+)-',d).group(1) for d in dois if re.match(r'10.5194/([a-z]+)-',d)})
        scope={'sets':sets,'candidate_dois':sorted(dois),'source_policy':'https://publications.copernicus.org/services/xml_harvesting_and_oai-pmh.html',
               'from':'2023-01-01','until':'2026-10-23','date_semantics':'OAI update date, not publication eligibility; current candidates only'}
        atomic_json(scope_path,scope)
    db=sqlite3.connect(out/'metadata.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records(identifier TEXT,doi TEXT,payload TEXT,receipt TEXT,PRIMARY KEY(identifier,receipt))')
    previous=out/'status.json'
    state=json.loads(previous.read_text()) if previous.exists() else {'sets':{}}
    state['status']='running'
    state['strategy']='Dublin Core bulk discovery then native XML only for matched current candidate DOI; completed sets reused'
    for name in scope['sets']:
        if state['sets'].get(name,{}).get('status') in {'collected','oai_error','cached_source_xml_invalid','pagination_repeated_records_unresolved'}:continue
        params={'verb':'ListRecords','metadataPrefix':'oai_dc','set':name,'from':scope['from'],'until':scope['until']}
        responses=set();pages=0;matched=0
        while True:
            record=fetch(params,folder,pages+1);pages+=1
            if record.get('http_status')!=200:
                state['sets'][name]={'status':'source_transport_or_access_unresolved','pages':pages,'receipt':record};state['status']='source_stopped'
                atomic_json(out/'status.json',state);return
            tree=read_xml(record,out,{'set':name,'page':pages})
            if tree is None:
                state['sets'][name]={'status':'cached_source_xml_invalid','pages':pages,'receipt':record}
                atomic_json(out/'status.json',state);break
            entries=tree.findall('./o:ListRecords/o:record',NS)
            identity_page=tuple(e.findtext('./o:header/o:identifier',namespaces=NS) for e in entries)
            if identity_page and identity_page in responses:
                state['sets'][name]={'status':'pagination_repeated_records_unresolved','pages':pages,'matched_records':matched}
                atomic_json(out/'status.json',state);break
            responses.add(identity_page)
            for entry in entries:
                identifiers=entry.xpath('.//*[local-name()="identifier" and namespace-uri()="http://purl.org/dc/elements/1.1/"]/text()')
                candidate_dois={v.removeprefix('https://doi.org/').removeprefix('http://doi.org/').lower() for v in identifiers} & set(scope['candidate_dois'])
                if len(candidate_dois)!=1:continue
                expected=next(iter(candidate_dois))
                if db.execute('SELECT 1 FROM records WHERE doi=?',(expected,)).fetchone():continue
                native_id=entry.findtext('./o:header/o:identifier',namespaces=NS)
                native=fetch({'verb':'GetRecord','metadataPrefix':'pmc','identifier':native_id},folder)
                if native.get('http_status')!=200:
                    atomic_json(out/'last_native_error.json',native)
                    if native.get('http_status') in {401,403,429}:
                        state['status']='native_source_access_restricted';atomic_json(out/'status.json',state);return
                    continue
                native_tree=read_xml(native,out,{'set':name,'doi':expected,'identifier':native_id})
                if native_tree is None:continue
                native_entry=native_tree.find('./o:GetRecord/o:record',NS)
                if native_entry is None:continue
                article=native_entry.find('./o:metadata',NS)
                if article is None or len(article)==0:continue
                payload=parse_article(article[0]);identities=set(payload.get('doi',[])) & set(scope['candidate_dois'])
                if identities!={expected}:continue
                doi=identities.pop()
                identifier=native_id
                db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)',(identifier,doi,json.dumps(payload),native['raw_receipt']));matched+=1
            db.commit()
            errors=[{'code':e.get('code'),'text':e.text} for e in tree.findall('./o:error',NS)]
            token=tree.findtext('./o:ListRecords/o:resumptionToken',namespaces=NS)
            state['sets'][name]={'status':'oai_error' if errors else 'collected' if not token else 'running','pages':pages,'matched_records':matched,'errors':errors}
            atomic_json(out/'status.json',state)
            if errors or not token:break
            # OAI opaque tokens may remain constant while the server advances its cursor.
            # Page ordinal distinguishes cached responses; repeated record pages stop the source.
            params={'verb':'ListRecords','resumptionToken':token}
        print(name,state['sets'][name],flush=True)
    state['status']='fixed_scope_processed_with_source_gaps' if any(v['status']!='collected' for v in state['sets'].values()) else 'fixed_scope_collected_identity_review_required';atomic_json(out/'status.json',state);db.close()


if __name__=='__main__':run()
