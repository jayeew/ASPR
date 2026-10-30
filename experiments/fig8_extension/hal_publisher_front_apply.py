"""Reuse explicit Elsevier PDF front dates/types archived by HAL, without repository-date inference."""
from __future__ import annotations

import fcntl
import gzip
import io
import json
import logging
import os
import re
from collections import Counter

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .fig8b_qualification_decide import key
from .publisher_delta_apply import apply_available
from .publisher_pdf_online_date import extract
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT/'supplement/hal_metadata/pdf_fronts'
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    with (folder/'apply.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'collector.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs = {}; reviews = [];db = connect();staged = folder/'staged';staged.mkdir(exist_ok=True)
            for path in (folder/'receipts').glob('*.json'):
                record = json.loads(path.read_text())
                if record.get('review_status') != 'front_metadata_extracted_not_qualification':continue
                try:
                    reader = PdfReader(io.BytesIO(gzip.decompress((ROOT/record['raw_receipt']).read_bytes())))
                    pages = [reader.pages[i].extract_text() or '' for i in range(min(3,len(reader.pages)))]
                except (PdfReadError,ValueError,IndexError,TypeError) as error:
                    reviews.append({'receipt':str(path.relative_to(ROOT)),'status':'parse_unresolved','error':str(error)});continue
                for paper in record['fixed_papers']:
                    row = db.execute('SELECT title,journal FROM papers WHERE work_id=?',(paper['work_id'],)).fetchone()
                    title,journal = row;proofs=[]
                    for page_number,text in enumerate(pages,1):
                        if 'Contents lists available at ScienceDirect' not in text:continue
                        proof = extract(text,paper['doi'])
                        if proof['status'] != 'explicit_online_date_identified':continue
                        if not title or key(title) not in key(text) or key(journal) not in key(text):continue
                        header = text.split('ABSTRACT',1)[0].split('ARTICLE INFO',1)[0]
                        types = sorted(set(' '.join(value.split()) for value in re.findall(
                            r'^\s*(Original Article|Research Article|Review Article|Review|Editorial)\s*$',header,re.M|re.I)))
                        if len(types)>1:continue
                        publishers = ['Elsevier'] if re.search(r'Published by Elsevier\b',text) else []
                        proofs.append({**proof,'page':page_number,'types':types,'publishers':publishers})
                    review = {'doi':paper['doi'],'work_id':paper['work_id'],'receipt':str(path.relative_to(ROOT)),
                              'proofs':proofs,'status':'native_front_confirmed' if len(proofs)==1 else 'no_unique_explicit_publisher_front'}
                    reviews.append(review)
                    if len(proofs)!=1:continue
                    proof = proofs[0];output=staged/(path.stem+'.json')
                    atomic_json(output,{'doi':paper['doi'],'source':'archived_publisher_pdf','route':'hal_publisher_front',
                        'url':record['url'],'raw_receipt':os.path.relpath(ROOT/record['raw_receipt'],ROOT/'fields/qualification'),
                        'native_front_proof':proof,'repository_discovery':paper['repository_record'],
                        'parsed':{'title':[title],'issns':[],'journal_titles':[journal],
                                  'article_types':proof['types'],'online_date':proof['online_date'],
                                  'date_precision':'day','publisher':proof['publishers'],'abstract':''}})
                    jobs[paper['doi']]=str(output.relative_to(ROOT))
            db.close();atomic_json(folder/'native_front_review.json',{'records':reviews,'network_requests':0})
            result=apply_available(supplied_jobs=sorted(jobs.items())) if jobs else {}
            atomic_json(folder/'native_front_application.json',{'application':result,'network_requests':0,
                'outcomes':dict(Counter(r['status'] for r in reviews)),
                'rule':'Exact native standalone DOI, title and journal in same PDF page; ScienceDirect front and explicit Available online date; only explicit header types; repository cover/date never used',
                'final_task_closure':False})
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    run()
