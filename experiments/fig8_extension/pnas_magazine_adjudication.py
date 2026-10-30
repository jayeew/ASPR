"""Resolve explicit PNAS magazine headings using editorial definitions and cached JATS."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import key, pnas_frontmatter
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(folder_name: str = 'pnas_magazine_adjudication') -> None:
    folder = ROOT / 'supplement' / folder_name
    folder.mkdir(exist_ok=True)
    policy = {'source_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3574923/',
              'doi': '10.1073/pnas.1300186110', 'published': '2013-01-30',
              'retrieved_at': '2026-10-25', 'access': 'PMC indexed full editorial text via web search',
              'categories': ['News Features', 'Core Concepts', 'Inner Workings', 'Science & Culture', 'Opinions', 'Profile', 'QnAs'],
              'interpretation': 'Editor-in-Chief and Senior Recruiting Editor define magazine reporting, explanatory writing and opinion, distinct from original research.',
              'scope': 'Exact native magazine heading only; not generic PNAS research-article or all letters/perspectives.'}
    atomic_json(folder / 'policy.json', policy)
    db = connect(); groups = defaultdict(list)
    for wid, doi, payload in db.execute("SELECT p.work_id,p.doi,e.payload FROM papers p JOIN current_evidence_index e USING(work_id) WHERE p.source_id='https://openalex.org/S125754415' AND p.status LIKE 'pending%' AND e.kind='primary'"):
        groups[(wid, doi)].append(json.loads(payload))
    db.close(); sources = defaultdict(list); reviews = []
    allowed = {key(v) for v in policy['categories']} | {'scienceandculture', 'scienceculture'}
    for (wid, doi), items in groups.items():
        seen = set()
        for e in items:
            if not pnas_frontmatter(e) or e['receipt'] in seen:
                continue
            seen.add(e['receipt'])
            raw = gzip.decompress((ROOT / 'fields/qualification' / e['receipt']).read_bytes())
            if raw.rstrip().endswith(b'</front>'): raw += b'</article>'
            xml = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            dois = {v.strip().lower() for v in xml.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}
            title = ''.join(xml.xpath('./front/article-meta/title-group/article-title//text()'))
            headings = [' '.join(''.join(n.itertext()).split()) for n in xml.xpath('./front/article-meta/article-categories//subject')]
            native = sorted({v for v in headings if key(v) in allowed})
            record = {'work_id': wid, 'doi': doi, 'receipt': e['receipt'], 'native_headings': native}
            if dois != {doi} or key(title) != key(e['title']) or not native:
                reviews.append({**record, 'status': 'native_identity_or_heading_unconfirmed'}); continue
            targets = [v['id'] for v in items if v.get('family') == 'archived_jats'
                       and v.get('identity') == 'matched' and key(v.get('title')) == key(title)
                       and (pnas_frontmatter(v) or (not v.get('categories') and key(v.get('kind')) == key(e['kind'])))
                       and v.get('class') != 'integrity_hold']
            extra = {k: v for k, v in e.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
            extra.update(native_magazine_rule='pnas_editorial_1300186110',
                         native_heading_sha256=hashlib.sha256(raw).hexdigest(),
                         reviewed_primary_evidence_ids=targets,
                         magazine_policy_receipt=str((folder / 'policy.json').relative_to(ROOT)))
            sources[doi].append(extra)
            reviews.append({**record, 'status': 'native_magazine_confirmed', 'reviewed_primary_evidence_ids': targets})
    atomic_json(folder / 'review.json', {'records': reviews, 'network_article_requests': 0,
                'outcomes': dict(Counter(r['status'] for r in reviews))})
    result = apply_evidence(set(sources), folder, explicit_sources=dict(sources)) if sources else {}
    atomic_json(folder / 'application.json', result)
    from .derived_refresh import refresh
    refresh()


if __name__ == '__main__':
    run()
