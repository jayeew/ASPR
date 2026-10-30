"""Apply literal native Cureus PDF types and explicitly labelled published dates."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/cureus_native_pdf'
    policy = {'url':'https://www.cureus.com/author_guide','access':'Official indexed author guide',
              'retrieved_at':'2026-10-25','locator':'Introduction',
              'interpretation':'Online-only journal with continuous online publication; use the native Published label, never received/review dates.',
              'scope':'Cureus native PDF with matching DOI, title and journal. No blanket original-research admission for case reports or generic Original Article.'}
    atomic_json(folder/'publication_date_policy.json', policy)
    scope = json.loads((folder/'scope.json').read_text()); sources = {}; reviews = []
    for job in scope['jobs']:
        receipt = folder/'receipts'/(hashlib.sha256(job['url'].encode()).hexdigest()+'.json')
        result = json.loads(receipt.read_text())
        if not result.get('front_receipt'): continue
        text = (ROOT/result['front_receipt']).read_text(); flat = ' '.join(text.split())
        for paper in job['papers']:
            dates = re.findall(r'\bPublished\s+(\d{2}/\d{2}/\d{4})\b',text)
            types = re.findall(r'Open Access\s+(Case\s+Report|Original\s+Article|Review\s+Article|Review|Editorial|Technical\s+Report)\b',text)
            valid = (re.search(r'\bDOI\s*:?\s*'+re.escape(paper['doi'])+r'(?!\d)',text,re.I)
                     and key(paper['title']) in key(flat) and 'Cureus' in text and len(set(dates))==1 and len(set(types))==1)
            record = {**paper,'receipt':str(receipt.relative_to(ROOT)),'status':'native_identity_or_type_date_unresolved'}
            if valid:
                date = datetime.strptime(dates[0],'%m/%d/%Y').date().isoformat()
                kind = ' '.join(types[0].split())
                source = {'family':'archived_publisher_pdf','title':paper['title'],'journal_titles':['Cureus'],
                          'issns':[],'kind':kind,'online_date':date,'date_precision':'day','categories':[],
                          'receipt':os.path.relpath(ROOT/result['raw_receipt'],ROOT/'fields/qualification'),
                          'url':job['url'],'native_front_text_receipt':result['front_receipt'],
                          'publication_date_policy':'supplement/cureus_native_pdf/publication_date_policy.json',
                          'type_locator':'Open Access native publication-type label','date_locator':'Published MM/DD/YYYY'}
                sources[paper['doi']] = [source]
                record.update(status='native_type_and_date_confirmed',kind=kind,online_date=date)
            reviews.append(record)
    atomic_json(folder/'metadata_review.json',{'records':reviews,'network_requests':0})
    result = apply_evidence(set(sources),folder,explicit_sources=sources) if sources else {}
    atomic_json(folder/'application.json',result)
    from .derived_refresh import refresh
    refresh()


if __name__ == '__main__':
    run()
