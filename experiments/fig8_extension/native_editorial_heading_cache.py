"""Recover literal editorial publication headings without interpreting short formats."""
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

FOLDER = ROOT / 'supplement/native_editorial_heading_cache'
LABELS = {'comment', 'commentary', 'editorial', 'newsviews', 'newsandviews',
          'researchhighlight', 'researchhighlights', 'review', 'reviewarticle',
          'erratum', 'correction', 'authorcorrection', 'publishercorrection',
          'bookreview', 'obituary', 'news', 'perspective', 'perspectives'}


def headings(raw: bytes, item: dict) -> list[str]:
    """Require DOI, title, journal and literal top-level publication headings."""
    if key(item.get('kind')) not in {'other', 'briefreport', 'letter'}:
        return []
    if raw.rstrip().endswith(b'</front>'):
        raw += b'</article>'
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    meta = './front/article-meta/'
    dois = {v.strip().lower() for v in tree.xpath(meta + 'article-id[@pub-id-type="doi"]/text()')}
    title = ''.join(tree.xpath(meta + 'title-group/article-title//text()'))
    issns = set(tree.xpath('./front/journal-meta/issn/text()'))
    journals = tree.xpath('./front/journal-meta/journal-title-group/journal-title/text()')
    groups = tree.xpath(meta + 'article-categories/subj-group[@subj-group-type="heading"]')
    values = [' '.join(''.join(n.itertext()).split()) for group in groups for n in group.findall('subject')]
    # Nested subject taxonomies can mix discipline and format: retain them for separate review.
    if (dois != {item['doi']} or key(title) != key(item['title'])
            or not (issns & set(item.get('issns', [])))
            or not (set(journals) & set(item.get('journal_titles', [])))
            or key(tree.get('article-type')) != key(item['kind'])
            or any(group.findall('subj-group') for group in groups)
            or not values or not {key(v) for v in values} <= LABELS):
        return []
    return sorted(set(values))


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'application.json').exists():
        return
    db = connect()
    rows = list(db.execute("""SELECT e.payload FROM current_evidence_index e JOIN papers p USING(work_id)
        WHERE p.status LIKE 'pending%' AND e.kind='primary'
        AND json_extract(e.payload,'$.family')='archived_jats'
        AND json_extract(e.payload,'$.kind') IN ('other','brief-report','letter')"""))
    db.close()
    sources = defaultdict(list); checks = []
    for payload, in rows:
        item = json.loads(payload)
        if (item.get('identity') != 'matched' or item.get('scope_review_required')
                or item.get('class') == 'integrity_hold'
                or not ({key(v) for v in item.get('categories', [])} & LABELS)):
            continue
        check = {'work_id': item['work_id'], 'doi': item['doi'], 'receipt': item['receipt'],
                 'status': 'native_publication_heading_unconfirmed'}
        try:
            raw = gzip.decompress((ROOT / 'fields/qualification' / item['receipt']).read_bytes())
            values = headings(raw, item)
            check.update(raw_sha256=hashlib.sha256(raw).hexdigest(), headings=values)
            if values:
                check['status'] = 'native_editorial_heading_confirmed'
                extra = {k: v for k, v in item.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
                extra.update(original_article_type=item['kind'], raw_sha256=check['raw_sha256'],
                             heading_review_receipt=str((FOLDER / 'review.json').relative_to(ROOT)),
                             type_locator='front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject')
                for value in values:
                    sources[item['doi']].append({**extra, 'kind': value})
        except (OSError, ValueError, etree.XMLSyntaxError) as error:
            check['error'] = str(error)
        checks.append(check)
    atomic_json(FOLDER / 'review.json', {'checks': checks, 'unique_selected_dois': len(sources),
        'outcomes': dict(Counter(r['status'] for r in checks)), 'network_requests': 0,
        'rule': 'Literal article-level editorial headings already in the nonresearch vocabulary. Generic other/brief-report/letter preserved. No research-article reclassification, no nested taxonomy inference, no inference from article title. Conflicting primary research evidence and integrity holds remain unresolved.',
        'final_task_closure': False})
    result = apply_evidence(set(sources), FOLDER, explicit_sources=dict(sources)) if sources else {}
    atomic_json(FOLDER / 'application.json', result)


if __name__ == '__main__':
    run()
