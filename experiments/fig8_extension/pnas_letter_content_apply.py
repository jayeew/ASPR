"""Apply source-bound individual PNAS Letter content exclusions from existing cache."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from lxml import etree

from .cached_date_apply import run as apply_sources
from .mdpi_brief_content_apply import native_article_xml
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json, text

FOLDER = ROOT / 'supplement/pnas_remaining_native_scope'


def prepare() -> dict:
    reviews = json.loads((FOLDER / 'individual_content_reviews.json').read_text())['records']
    db, sources = connect(), {}
    try:
        for review in reviews:
            if review['classification'] != 'nonprimary':
                continue
            raw = gzip.decompress((ROOT / 'fields/qualification' / review['native_receipt']).read_bytes())
            article = etree.fromstring(native_article_xml(raw))
            paragraphs = [text(n) for n in article.xpath('./body//p[not(ancestor::table-wrap) and not(ancestor::fig)]')]
            if (hashlib.sha256(raw).hexdigest() != review['raw_sha256']
                    or hashlib.sha256('\n'.join(paragraphs).encode()).hexdigest() != review['content_sha256']
                    or review['doi'] not in [text(n).lower() for n in article.xpath('./front/article-meta/article-id[@pub-id-type="doi"]')]):
                raise ValueError('Reviewed body identity/content changed: ' + review['doi'])
            prior = {e['id']: e for p, in db.execute(
                "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (review['work_id'],))
                for e in [json.loads(p)]}
            backing = prior[review['backing_evidence_id']]
            targets = [e['id'] for e in prior.values() if e.get('identity') == 'matched'
                       and e.get('doi') == review['doi'] and e.get('family') == 'archived_jats'
                       and e.get('kind') == 'letter']
            if backing['id'] not in targets:
                raise ValueError('Missing matched native Letter backing')
            sources[review['doi']] = [{
                'family': 'reviewed_pnas_letter_content', 'doi': review['doi'],
                'title': backing['title'], 'issns': ['0027-8424', '1091-6490'],
                'kind': 'Individually reviewed pure commentary', 'categories': [],
                'online_date': '', 'date_precision': 'not_a_date_source',
                'study_design_class': 'nonprimary', 'review_protocol': 'pnas_letter_full_body_individual_1',
                'reviewed_primary_evidence_ids': targets, 'source_evidence_ids': [backing['id']],
                'review_rule_receipt': 'supplement/pnas_remaining_native_scope/individual_content_reviews.json',
                'receipt': '../../supplement/pnas_remaining_native_scope/individual_content_reviews.json',
                'url': 'https://doi.org/' + review['doi'], 'content_review_reason': review['reason'],
                **{k: review[k] for k in ('raw_sha256', 'content_sha256', 'content_locator')},
            }]
    finally:
        db.close()
    return sources


def run() -> dict:
    sources = prepare()
    folder = FOLDER / 'content_application'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'sources.json', sources)
    result = apply_sources(set(sources), folder, explicit_sources=sources)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_bytes(ROOT / 'fields/qualification/decide.py', Path(__file__).with_name('fig8b_qualification_decide.py').read_bytes())
    atomic_json(folder / 'application.json', {'qualification': result, 'network_requests': 0,
                                             'final_task_closure': False})
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
