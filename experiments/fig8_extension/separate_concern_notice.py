"""Resolve separate concern notices using their native JATS target relationship."""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import expression_notice, key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder=ROOT/'supplement/separate_concern_notices';folder.mkdir(exist_ok=True)
    db=connect();rows=list(db.execute('''SELECT p.work_id,p.doi,e.payload FROM papers p
        JOIN current_evidence_index e USING(work_id) WHERE p.status='pending_integrity_review'
        AND e.kind='primary' AND e.payload LIKE '%expression%' '''));db.close()
    previous=json.loads((folder/'review.json').read_text()).get('records',[]) if (folder/'review.json').exists() else []
    reviews=[];sources=defaultdict(list)
    for wid,doi,payload in rows:
        evidence=json.loads(payload)
        if evidence.get('family')!='archived_jats' or evidence.get('identity')!='matched' or not expression_notice(evidence):continue
        review={'work_id':wid,'doi':doi,'receipt':evidence['receipt'],'status':'native_notice_relation_unconfirmed'}
        try:
            raw=gzip.decompress((ROOT/'fields/qualification'/evidence['receipt']).read_bytes())
            if raw.rstrip().endswith(b'</front>'):raw+=b'</article>'
            tree=etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True))
            native_dois={v.strip().lower() for v in tree.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}
            title=''.join(tree.xpath('./front/article-meta/title-group/article-title//text()'))
            targets={v.strip().lower() for v in tree.xpath('.//related-article[@related-article-type="object-of-concern"]//pub-id[@pub-id-type="doi"]/text()')}
            targets={value for value in targets if value.startswith('10.') and '/' in value}
            target_pmcs={m.group(0) for node in tree.xpath('.//related-article[@related-article-type="object-of-concern"]')
                         for value in node.xpath('.//@*[local-name()="href"] | .//pub-id[@pub-id-type="pmcid"]/text()')
                         for m in re.finditer(r'\bPMC\d+\b',value)}
            own_pmcs=set(re.findall(r'\bPMC\d+\b',evidence.get('url','')))
            separate_pmc=bool(target_pmcs and own_pmcs and not target_pmcs & own_pmcs)
            valid=(native_dois=={doi} and key(tree.get('article-type'))=='expressionofconcern'
                   and key(title)==key(evidence['title']) and key(title).startswith('expressionofconcern')
                   and (bool(targets) or separate_pmc) and doi not in targets
                   and not target_pmcs & own_pmcs)
            if valid:
                extra={k:v for k,v in evidence.items() if k not in {'id','identity','class','work_id','doi'}}
                extra.update(separate_notice_rule='native_related_article_object_of_concern',
                             notice_target_dois=sorted(targets),
                             notice_target_pmcids=sorted(target_pmcs) if separate_pmc else [],
                             notice_rule_receipt=str((folder/'review.json').relative_to(ROOT)),
                             notice_locator='article@article-type; front/article-meta/article-id; related-article@related-article-type=object-of-concern/pub-id@pub-id-type=doi')
                sources[doi].append(extra);review.update(status='separate_notice_confirmed',target_dois=sorted(targets),target_pmcids=sorted(target_pmcs))
        except (OSError,etree.XMLSyntaxError,ValueError) as error:
            review['error']=str(error)
        reviews.append(review)
    merged={(r['work_id'],r['receipt']):r for r in previous}
    merged.update({(r['work_id'],r['receipt']):r for r in reviews})
    atomic_json(folder/'review.json',{'records':list(merged.values()),'network_requests':0,
        'rule':'Matched native expression-of-concern type and title; explicit object-of-concern identifies a different DOI or PMC article; excludes only the notice itself',
        'outcomes':dict(Counter(r['status'] for r in merged.values()))})
    result=apply_evidence(set(sources),folder,explicit_sources=dict(sources)) if sources else {}
    atomic_json(folder/'application.json',result)


if __name__ == '__main__':
    run()
