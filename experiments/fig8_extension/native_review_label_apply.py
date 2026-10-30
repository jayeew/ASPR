"""Apply individual Review-category adjudications using retained author abstracts."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import etree

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .fig8b_qualification_decide import key, reviewed_study_design
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/native_review_label_conflicts'


def prepare() -> dict[str, list[dict]]:
    reviews = json.loads((FOLDER / 'reviews.json').read_text())['records']
    db, sources = connect(), {}
    try:
        for review in reviews:
            if review['classification'] != 'nonprimary':
                continue
            raw = gzip.decompress((ROOT / 'fields/qualification' / review['native_receipt']).read_bytes())
            document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
            tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
            abstract = ' '.join(' '.join(tree.xpath('//article-meta/abstract//text()')).split())
            dois = tree.xpath('//article-meta/article-id[@pub-id-type="doi"]/text()')
            categories = tree.xpath('//article-categories//subject/text()')
            if (hashlib.sha256(raw).hexdigest() != review['raw_sha256']
                    or hashlib.sha256(abstract.encode()).hexdigest() != review['abstract_sha256']
                    or review['doi'] not in [d.lower() for d in dois]
                    or 'Review' not in categories
                    or not review['review_quotes']
                    or not all(q in abstract for q in review['review_quotes'])):
                raise ValueError('Reviewed content or identity changed: ' + review['doi'])
            primary = [json.loads(p) for p, in db.execute(
                "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'",
                (review['work_id'],))]
            backing = next(p for p in primary if p['id'] == review['backing_evidence_id'])
            title = ' '.join(' '.join(tree.xpath('//article-meta/title-group/article-title//text()')).split())
            if key(title) != key(backing['title']) or backing.get('identity') != 'matched':
                raise ValueError('Reviewed title/identity changed: ' + review['doi'])
            targets = [p['id'] for p in primary if p.get('identity') == 'matched'
                       and p.get('doi') == review['doi'] and p.get('family') == 'archived_jats'
                       and p.get('kind') == 'research-article'
                       and p.get('categories', []) in ([], ['Review'])
                       and not p.get('scope_review_required')]
            if backing['id'] not in targets:
                raise ValueError('Reviewed source no longer eligible for adjudication')
            source = {
                'family': 'reviewed_study_design', 'doi': review['doi'], 'title': backing['title'],
                'issns': backing.get('issns', []), 'journal_titles': backing.get('journal_titles', []),
                'kind': 'Individually corroborated narrative review', 'categories': [],
                'online_date': '', 'date_precision': 'not_a_date_source',
                'study_design_class': 'nonprimary', 'review_protocol': 'native_individual_design_1',
                'reviewed_primary_evidence_ids': targets, 'source_evidence_ids': [backing['id']],
                'review_rule_receipt': 'supplement/native_review_label_conflicts/reviews.json',
                'receipt': '../../supplement/native_review_label_conflicts/reviews.json',
                'url': backing.get('url', ''), 'study_design_reason': review['reason'],
                'review_quotes': review['review_quotes'],
                **{k: review[k] for k in ('raw_sha256', 'abstract_sha256', 'content_locator')},
            }
            resolved = reviewed_study_design(primary + [{**source, 'id': 'preflight-review'}])
            if not all(p['class'] == 'nonprimary' for p in resolved if p['id'] in targets):
                raise ValueError('Existing adjudication rule did not accept reviewed source')
            sources[review['doi']] = [source]
    finally:
        db.close()
    return sources


def run() -> dict:
    sources = prepare()
    atomic_json(FOLDER / 'sources.json', sources)
    result = apply_sources(set(sources), FOLDER, explicit_sources=sources)
    atomic_json(FOLDER / 'application.json', {'qualification': result, 'network_requests': 0,
                                             'final_task_closure': False})
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
