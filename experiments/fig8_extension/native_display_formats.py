"""Identify separate cover-art records and scoped author interviews from native caches."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata

COVERS = {'coverpicture', 'coverimage', 'frontcover', 'backcover',
          'insidefrontcover', 'insidebackcover', 'frontispiece'}
INTERVIEW_JOURNALS = {'Biology Open', 'Disease Models & Mechanisms', 'Journal of Cell Science'}


def classified_heading(item: dict, native: dict) -> str:
    """A pure publication heading is required; a title/body mention is insufficient."""
    headings = {key(v) for v in native.get('categories', [])}
    if (item.get('family') != 'archived_jats' or item.get('identity') != 'matched'
            or key(item.get('kind')) not in {'other', ''} or item.get('scope_review_required')
            or item.get('class') == 'integrity_hold' or key(native.get('article_type')) != key(item.get('kind'))):
        return ''
    if len(headings) == 1 and headings <= COVERS:
        return native['categories'][0]
    if headings == {'firstperson'} and set(native.get('journal_titles', [])) & INTERVIEW_JOURNALS:
        return 'Interview'
    return ''


def run() -> None:
    folder = ROOT / 'supplement/native_display_formats'; folder.mkdir(exist_ok=True)
    if (folder / 'application.json').exists():
        return
    db = connect(); rows = db.execute("""SELECT e.payload FROM papers p JOIN current_evidence_index e USING(work_id)
        WHERE p.status LIKE 'pending%' AND e.kind='primary'
        AND json_extract(e.payload,'$.family')='archived_jats'
        AND json_extract(e.payload,'$.kind') IN ('other','')""").fetchall(); db.close()
    sources, reviews = defaultdict(list), []
    for payload, in rows:
        item = json.loads(payload)
        headings = {key(v) for v in item.get('categories', [])}
        if not headings or not headings <= COVERS | {'firstperson'}:
            continue
        receipt = 'fields/qualification/' + item['receipt']
        check = {'work_id': item['work_id'], 'doi': item['doi'], 'receipt': receipt,
                 'status': 'native_format_unresolved'}
        try:
            native = native_metadata(receipt, item['doi']); label = classified_heading(item, native)
            titles = native.get('title') or []; titles = [titles] if isinstance(titles, str) else titles
            if (label and item['doi'] in {v.lower() for v in native.get('doi', [])}
                    and key(item['title']) in {key(v) for v in titles}
                    and set(item.get('journal_titles', [])) & set(native.get('journal_titles', []))
                    and set(item.get('issns', [])) & set(native.get('issns', []))):
                digest = hashlib.sha256(gzip.decompress((ROOT / receipt).read_bytes())).hexdigest()
                check.update(status='native_heading_confirmed', native_heading=native['categories'],
                             derived_kind=label, raw_sha256=digest)
                extra = {k: v for k, v in item.items() if k not in {'id','class','identity','work_id','doi'}}
                extra.update(kind=label, original_article_type=item['kind'], original_publication_heading=native['categories'],
                             raw_sha256=digest, heading_review_receipt=str((folder/'review.json').relative_to(ROOT)))
                sources[item['doi']].append(extra)
        except (OSError, ValueError, etree.XMLSyntaxError) as error:
            check['error'] = str(error)
        reviews.append(check)
    atomic_json(folder/'review.json', {'records': reviews, 'selected_dois': len(sources), 'network_requests': 0,
        'policy_sources': ['https://authors.aws.wiley.com/author-resources/Journal-Authors/Promotion/journal-cover-image.html',
                           'https://journals.biologists.com/jcs/pages/first-person',
                           'https://journals.biologists.com/dmm/article/18/4/dmm052388/367839/First-person-Mikiko-Oka',
                           'https://pmc.ncbi.nlm.nih.gov/articles/PMC11391815/'],
        'scope': 'Separate publication-level cover artwork; First Person is a scoped companion author interview. Native DOI/title/journal/ISSN required; no title-only or mixed-category inference. All other primary evidence retained.',
        'final_task_closure': False})
    print(json.dumps({'selected': len(sources), 'native_checks': len(reviews)}), flush=True)
    result = apply_evidence(set(sources), folder, explicit_sources=dict(sources)) if sources else {}
    atomic_json(folder/'application.json', result)
    from .derived_refresh import refresh
    refresh()


if __name__ == '__main__':
    run()
