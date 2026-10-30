"""Use explicit nonresearch publication headings when native JATS type is 'other'."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import NONPRIMARY, key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder=ROOT/'supplement/jats_nonresearch_heading';folder.mkdir(exist_ok=True)
    db=connect();rows=list(db.execute('''SELECT p.work_id,p.doi,e.payload FROM current_evidence_index e
        JOIN papers p USING(work_id) WHERE e.kind='primary' AND p.status LIKE 'pending%'
        AND e.payload LIKE '%"kind": "other"%' '''));db.close()
    reviews=[];sources=defaultdict(list);seen=set()
    for wid,doi,payload in rows:
        e=json.loads(payload)
        if (e.get('family')!='archived_jats' or e.get('identity')!='matched' or e.get('scope_review_required')
                or key(e.get('kind'))!='other' or (wid,e['receipt']) in seen):continue
        if not ({key(v) for v in e.get('categories',[])} & NONPRIMARY):continue
        seen.add((wid,e['receipt']));review={'work_id':wid,'doi':doi,'receipt':e['receipt'],'status':'heading_not_confirmed'}
        try:
            raw=gzip.decompress((ROOT/'fields/qualification'/e['receipt']).read_bytes())
            if raw.rstrip().endswith(b'</front>'):raw+=b'</article>'
            t=etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True))
            native_dois={v.strip().lower() for v in t.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}
            title=''.join(t.xpath('./front/article-meta/title-group/article-title//text()'))
            headings=[' '.join(''.join(n.itertext()).split()) for n in t.xpath('./front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject')]
            valid=(native_dois=={doi} and key(t.get('article-type'))=='other' and key(title)==key(e['title'])
                   and headings and all(key(v) in NONPRIMARY for v in headings))
            if valid:
                extra={k:v for k,v in e.items() if k not in {'id','class','identity','work_id','doi'}}
                extra.update(kind=' | '.join(sorted(set(headings))) if len(set(headings))>1 else headings[0],
                    original_article_type='other',native_nonresearch_headings=headings,
                    classification_rule='Explicit native publication heading; all headings nonresearch; generic other type is not positive research evidence',
                    type_locator='front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject',
                    heading_review_receipt=str((folder/'review.json').relative_to(ROOT)))
                # Emit each independently declared heading; do not invent a composite type.
                for heading in sorted(set(headings)):sources[doi].append({**extra,'kind':heading})
                review.update(status='native_nonresearch_heading_confirmed',headings=headings)
        except (OSError,etree.XMLSyntaxError,ValueError) as error:review['error']=str(error)
        reviews.append(review)
    atomic_json(folder/'review.json',{'records':reviews,'outcomes':dict(Counter(r['status'] for r in reviews)),
        'network_requests':0,'scope':'Existing pending native other-type records with explicit nonresearch category; research-article conflicts are untouched'})
    result=apply_evidence(set(sources),folder,explicit_sources=dict(sources)) if sources else {}
    atomic_json(folder/'application.json',result)


if __name__ == '__main__':
    run()
