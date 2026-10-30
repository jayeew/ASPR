"""Review the three observed publisher front conflicts using immutable cached sources."""
from __future__ import annotations

import gzip
import json
import os

from lxml import html

from .fig8b_qualification_decide import key
from .rrpharm_pdf_metadata import OUT, extract
from .publisher_delta_apply import apply_available
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

TARGETS = {'10.18413/rrpharmacology.10.498', '10.18413/rrpharmacology.11.587', '10.18413/rrpharmacology.9.10028'}


def run() -> dict:
    folder=OUT/'cached_review';folder.mkdir(exist_ok=True)
    jobs=[];reviews=[];db=connect()
    for path in (OUT/'results').glob('*.json'):
        source=json.loads(path.read_text());doi=source['doi']
        if doi not in TARGETS:continue
        first=(ROOT/source['first_page_receipt']).read_text();proof=extract(first,doi)
        assert proof['status']=='pdf_front_explicit_type_date'
        page=html.fromstring(gzip.decompress((ROOT/source['landing_receipt']['raw_receipt']).read_bytes()))
        meta={n.get('name'):n.get('content','') for n in page.xpath('//meta[@name]')}
        assert meta.get('citation_doi',meta.get('DC.Identifier.DOI','')).lower()==doi
        title=meta['citation_title'];date=proof['online_date'];precision='day';reason='PDF review label whitespace normalized'
        detail={'pdf_date':date,'landing_date':meta['citation_date'].replace('/','-')}
        if doi=='10.18413/rrpharmacology.10.498':
            title=' '.join(first.split('Research Article',1)[1].strip().split('\n\n')[0].split())
            candidate=db.execute('SELECT title FROM papers WHERE doi=?',(doi,)).fetchone()[0]
            assert key(title)==key(candidate)
            detail.update(pdf_title=title,landing_title=meta['citation_title'],pdf_title_span='First title block after Research Article label')
            reason='PDF native title matches candidate exactly; landing page adds The; raw discrepant title retained'
        elif doi=='10.18413/rrpharmacology.9.10028':
            assert detail['pdf_date']!=detail['landing_date'] and detail['pdf_date'][:4]==detail['landing_date'][:4]=='2023'
            date='2023';precision='year'
            reason='Both publisher sources explicitly establish 2023; month/day conflict retained and no precise date selected'
            detail['exact_date_final_status']='unresolved_publisher_sources_disagree'
        else:
            assert detail['pdf_date']==detail['landing_date']
        parsed={'doi':[doi],'title':[title],'issns':[meta['citation_issn']], 'journal_titles':[meta['citation_journal_title']],
                'article_types':[proof['article_type']],'online_date':date,'date_precision':precision,'abstract':'','authors':[]}
        target=folder/path.name
        atomic_json(target,{'source':'publisher_pdf_front','route':'rrpharmacology','doi':doi,'url':source['pdf_receipt']['url'],
                    'parsed':parsed,'raw_receipt':os.path.relpath(ROOT/source['pdf_receipt']['raw_receipt'],ROOT/'fields/qualification'),
                    'review_source_receipt':str(path.relative_to(ROOT)),'review_reason':reason,'review_evidence':detail})
        jobs.append((doi,str(target.relative_to(ROOT))));reviews.append({'doi':doi,'reason':reason,**detail})
    db.close();result=apply_available(supplied_jobs=jobs)
    atomic_json(folder/'application.json',{'application':result,'reviews':reviews,'network_requests':0})
    return result


if __name__=='__main__':print(json.dumps(run()))
