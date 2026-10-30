"""Recover explicit Research Highlight headings from retained STTT publication XML."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/sigtrans_highlight_cache'
JOURNAL = 'Signal Transduction and Targeted Therapy'


def native_heading(raw: bytes, doi: str, title: str) -> bool:
    """Require exact publication identity and a sole article-level highlight heading."""
    if raw.rstrip().endswith(b'</front>'):
        raw += b'</article>'
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    meta = './front/article-meta/'
    dois = {v.strip().lower() for v in tree.xpath(meta + 'article-id[@pub-id-type="doi"]/text()')}
    titles = ''.join(tree.xpath(meta + 'title-group/article-title//text()'))
    journals = tree.xpath('./front/journal-meta/journal-title-group/journal-title/text()')
    issns = set(tree.xpath('./front/journal-meta/issn/text()'))
    headings = [''.join(n.itertext()) for n in tree.xpath(
        meta + 'article-categories/subj-group[@subj-group-type="heading"]/subject')]
    return (dois == {doi} and key(titles) == key(title) and journals == [JOURNAL]
            and bool(issns & {'2095-9907', '2059-3635'})
            and key(tree.get('article-type')) == 'briefreport'
            and {key(v) for v in headings} == {'researchhighlight'})


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'application.json').exists():
        return
    db = connect()
    rows = list(db.execute("""SELECT p.work_id,p.doi,e.payload FROM papers p
        JOIN current_evidence_index e USING(work_id)
        WHERE p.journal=? AND p.status LIKE 'pending%' AND e.kind='primary'""", (JOURNAL,)))
    db.close()
    checks = []; sources = defaultdict(list)
    for wid, doi, payload in rows:
        item = json.loads(payload)
        if (item.get('family') != 'archived_jats' or item.get('identity') != 'matched'
                or item.get('scope_review_required') or item.get('class') == 'integrity_hold'
                or key(item.get('kind')) != 'briefreport'
                or {key(v) for v in item.get('categories', [])} != {'researchhighlight'}):
            continue
        check = {'work_id': wid, 'doi': doi, 'receipt': item['receipt'], 'confirmed': False}
        try:
            raw = gzip.decompress((ROOT / 'fields/qualification' / item['receipt']).read_bytes())
            check.update(confirmed=native_heading(raw, doi, item['title']),
                         raw_sha256=hashlib.sha256(raw).hexdigest())
        except (OSError, ValueError, etree.XMLSyntaxError) as error:
            check['error'] = str(error)
        checks.append(check)
        if not check['confirmed']:
            continue
        extra = {k: v for k, v in item.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
        extra.update(kind='Research Highlight', original_article_type='brief-report',
                     type_locator='front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject',
                     heading_review_receipt=str((FOLDER / 'review.json').relative_to(ROOT)),
                     raw_sha256=check['raw_sha256'])
        sources[doi].append(extra)
    atomic_json(FOLDER / 'review.json', {'checks': checks, 'unique_selected_dois': len(sources),
        'outcomes': dict(Counter(str(c['confirmed']) for c in checks)), 'network_requests': 0,
        'rule': 'Retain the explicit article-level Research Highlight label as primary type evidence. Generic brief-report is retained in original evidence. Existing nonresearch classification applies; conflicting research and integrity evidence remain pending.',
        'final_task_closure': False})
    result = apply_evidence(set(sources), FOLDER, explicit_sources=dict(sources)) if sources else {}
    atomic_json(FOLDER / 'application.json', result)


if __name__ == '__main__':
    run()
