"""Resolve transferred-journal metadata from publisher landing pages and PDF fronts."""
from __future__ import annotations

import fcntl
import gzip
import io
import json
import os
import re
from datetime import datetime
from urllib.parse import urlparse

from lxml import html
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from . import ojs_metadata_harvest as transport
from .rrpharm_bulk import FOLDER
from .publisher_delta_apply import apply_available
from .supplement_queue import ROOT
from .supplement_sources import atomic_json, atomic_bytes

OUT = FOLDER / 'pdf_metadata'


def extract(first: str, doi: str) -> dict:
    header = re.search(r'\bDOI\s+([^\n]+)', '\n'.join(first.splitlines()[:8]), re.I)
    if not header or re.sub(r'\s+', '', header[1]).lower() != doi:
        return {'status':'pdf_header_doi_unconfirmed'}
    kinds = re.findall(r'^\s*(Research[ \t]+Article|Review[ \t]+Article|Review|Editorial|Short[ \t]+Communication)\s*$',first,re.M)
    kinds = [' '.join(k.split()) for k in kinds]
    dates = re.findall(r'\bPublished\s+(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})',first)
    if len(set(kinds))!=1 or len(set(dates))!=1:
        return {'status':'pdf_type_or_published_date_unconfirmed'}
    day, month, year = dates[0]
    try:
        date=datetime.strptime(f'{day} {month} {year}','%d %B %Y').date().isoformat()
    except ValueError:
        return {'status':'invalid_pdf_publication_date'}
    return {'status':'pdf_front_explicit_type_date','article_type':kinds[0],'online_date':date}


def process(path) -> dict:
    target=OUT/'results'/path.name
    if target.exists():return json.loads(target.read_text())
    source=json.loads(path.read_text());doi=source['doi']
    result={'doi':doi,'oai_receipt':str(path.relative_to(ROOT)),'status':'no_unique_landing'}
    urls=[x for x in source['identifiers'] if x.startswith('https://rrpharmacology.ru/')]
    if len(urls)!=1:atomic_json(target,result);return result
    landing=transport.fetch(urls[0],{});result.update(landing_receipt=landing,status='landing_unavailable')
    if landing.get('http_status')!=200:atomic_json(target,result);return result
    tree=html.fromstring(gzip.decompress((ROOT/landing['raw_receipt']).read_bytes()))
    meta={n.get('name'):n.get('content','') for n in tree.xpath('//meta[@name]')}
    if meta.get('citation_doi',meta.get('DC.Identifier.DOI','')).lower()!=doi:
        result['status']='landing_doi_mismatch';atomic_json(target,result);return result
    pdf=meta.get('citation_pdf_url','')
    if urlparse(pdf).hostname!='rrpharmacology.ru' or not pdf.startswith('https://'):
        result['status']='publisher_pdf_link_unconfirmed';atomic_json(target,result);return result
    receipt=transport.fetch(pdf,{});result.update(pdf_receipt=receipt,status='pdf_unavailable')
    if receipt.get('http_status')!=200:atomic_json(target,result);return result
    try:
        first=PdfReader(io.BytesIO(gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes()))).pages[0].extract_text(extraction_mode='layout')
    except (PdfReadError,ValueError,IndexError) as exc:
        result.update(status='pdf_parse_failed',error=str(exc)[:250]);atomic_json(target,result);return result
    textpath=OUT/'firstpages'/(path.stem+'.txt');atomic_bytes(textpath,first.encode());proof=extract(first,doi)
    result.update(proof,first_page_receipt=str(textpath.relative_to(ROOT)))
    if proof['status']=='pdf_front_explicit_type_date':
        page_date=meta.get('citation_date',meta.get('citation_publication_date','')).replace('/','-')
        if page_date!=proof['online_date']:
            result.update(status='publisher_landing_pdf_date_conflict',landing_date=page_date)
        else:
            parsed={'doi':[doi],'title':[meta.get('citation_title','')], 'issns':[meta.get('citation_issn','')],
                    'journal_titles':[meta.get('citation_journal_title','')], 'article_types':[proof['article_type']],
                    'online_date':proof['online_date'],'date_precision':'day','abstract':'','authors':[]}
            staged=OUT/'staged'/path.name
            atomic_json(staged,{'doi':doi,'source':'publisher_pdf_front','route':'rrpharmacology','url':pdf,'parsed':parsed,
                        'raw_receipt':os.path.relpath(ROOT/receipt['raw_receipt'],ROOT/'fields/qualification'),
                        'landing_identity_receipt':landing,'date_type_proof':str(textpath.relative_to(ROOT)),
                        'discovery_receipt':str(path.relative_to(ROOT))})
            result['staged_receipt']=str(staged.relative_to(ROOT))
    atomic_json(target,result);return result


def run() -> None:
    for p in [OUT,OUT/'results',OUT/'staged',OUT/'firstpages']:p.mkdir(exist_ok=True)
    with (OUT/'collector.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        transport.FOLDER,transport.RAW=FOLDER,FOLDER/'raw'
        results=[];blocked=False
        for path in sorted((FOLDER/'rrpharm_matches').glob('*.json')):
            result=process(path);results.append(result)
            if any(result.get(k,{}).get('http_status') in {401,403,429} for k in ['landing_receipt','pdf_receipt']):
                blocked=True;break
        jobs=[(r['doi'],r['staged_receipt']) for r in results if r.get('staged_receipt')]
        applied=apply_available(supplied_jobs=jobs) if jobs else {}
        atomic_json(OUT/'status.json',{'status':'source_restriction_retained' if blocked else 'fixed_pdf_metadata_queue_processed',
                    'reviewed':len(results),'date_type_proofs':len(jobs),'application':applied,'authors_not_inferred_from_pdf':True})


if __name__=='__main__':run()
