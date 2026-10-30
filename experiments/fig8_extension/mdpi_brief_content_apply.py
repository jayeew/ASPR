"""Apply individual native-abstract decisions without blanket Brief Report inclusion."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json

from lxml import etree

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .supplement_apply import apply as apply_metadata
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata, text

FOLDER = ROOT / 'supplement/mdpi_brief_report_policy'


def native_article_xml(raw: bytes) -> bytes:
    """Normalize one archived JATS article, including a PMC OAI envelope."""
    xml = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
    root = etree.fromstring(xml, etree.XMLParser(resolve_entities=False, no_network=True))
    if etree.QName(root).localname != 'article':
        articles = root.xpath('//*[local-name()="metadata"]/*[local-name()="article"]')
        if len(articles) != 1:
            raise ValueError('Expected one archived JATS article in OAI metadata')
        root = etree.fromstring(etree.tostring(articles[0]),
                               etree.XMLParser(resolve_entities=False, no_network=True))
    for node in root.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    etree.cleanup_namespaces(root)
    return etree.tostring(root)


def explicit_nested_abstract(xml: bytes) -> str | None:
    """Some native records place Simple Summary and Abstract in sibling sections."""
    tree = etree.fromstring(xml, etree.XMLParser(resolve_entities=False, no_network=True))
    sections = tree.xpath('./front/article-meta/abstract/sec[normalize-space(title)="Abstract"]')
    if len(sections) != 1:
        return None
    return '\n\n'.join(text(n) for n in sections[0].xpath('./p|./sec') if text(n))


def prepare() -> tuple[dict, dict]:
    scope = {r['doi']: r for r in json.loads((FOLDER / 'content_scope.json').read_text())['records']}
    reviews = json.loads((FOLDER / 'content_reviews.json').read_text())['records']
    db = connect(); sources = {}; metadata = {}
    for review in reviews:
        if review['classification'] not in {'research', 'nonprimary'}:
            continue
        doi = review['doi']; row = scope[doi]
        raw_path = ROOT / 'fields/qualification' / row['native_receipt']
        raw = gzip.decompress(raw_path.read_bytes())
        if (hashlib.sha256(raw).hexdigest() != review['raw_sha256']
                or hashlib.sha256(row['abstract'].encode()).hexdigest() != review['abstract_sha256']
                or row['abstract'] != review['full_native_abstract']):
            raise ValueError('Reviewed native content changed: ' + doi)
        prior = {json.loads(p)['id']: json.loads(p) for p, in db.execute(
            "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))}
        backing = prior[row['backing_evidence_id']]
        targets = [e['id'] for e in prior.values() if e.get('identity') == 'matched'
            and e.get('doi') == doi and e.get('family') in {'archived_jats', 'publisher_mdpi_oai'}
            and e.get('kind') in {'brief-report', 'research-article'}]
        sources[doi] = [{'family': 'reviewed_study_design', 'doi': doi, 'title': backing['title'],
            'issns': backing.get('issns', []), 'journal_titles': backing.get('journal_titles', []),
            'kind': ('Individual completed original study' if review['classification'] == 'research'
                     else 'Individual native-content nonresearch review'), 'categories': [], 'online_date': '',
            'date_precision': 'not_a_date_source', 'study_design_class': review['classification'],
            'review_protocol': 'native_mdpi_brief_report_design_1',
            'reviewed_primary_evidence_ids': targets, 'source_evidence_ids': [backing['id']],
            'review_rule_receipt': 'supplement/mdpi_brief_report_policy/content_reviews.json',
            'format_policy_receipt': 'supplement/mdpi_brief_report_policy/policy.json',
            'study_design_reason': review['reason'], 'raw_sha256': review['raw_sha256'],
            'abstract_sha256': review['abstract_sha256'], 'content_locator': review['content_locator'],
            'receipt': '../../supplement/mdpi_brief_report_policy/content_reviews.json',
            'url': 'https://doi.org/' + doi}]
        if review['classification'] == 'nonprimary':
            continue
        if row.get('native_family') == 'publisher_mdpi_oai':
            if backing.get('family') != 'publisher_mdpi_oai' or backing.get('identity') != 'matched':
                raise ValueError('Publisher OAI review lacks matched primary backing')
            parsed = {'doi': [doi], 'title': [backing['title']], 'issn': backing.get('issns', []),
                'journal_titles': backing.get('journal_titles', []), 'abstract': row['abstract'],
                'authors': [], 'byline_has_etal': True, 'native_metadata_kind': 'publisher',
                'abstract_locator': 'OAI record/metadata/dc:description',
                'author_link_limitation': 'This OAI abstract source does not establish author-affiliation links'}
        else:
            xml = native_article_xml(raw)
            parsed = jats_metadata(xml)
            nested_abstract = explicit_nested_abstract(xml)
            if nested_abstract:
                parsed['abstract'] = nested_abstract
                parsed['abstract_locator'] = 'front/article-meta/abstract/sec[title="Abstract"]'
        if doi not in parsed['doi']:
            raise ValueError('Native metadata DOI differs from review')
        metadata[doi] = {**parsed, 'issns': parsed.get('issn', []),
            'evidence_receipt': str(raw_path.resolve().relative_to(ROOT.resolve())),
            'source_url': 'https://doi.org/' + doi, 'raw_sha256': review['raw_sha256']}
    db.close()
    return sources, metadata


def run() -> dict:
    sources, metadata = prepare()
    folder = FOLDER / 'content_application'; folder.mkdir(exist_ok=True)
    atomic_json(folder / 'sources.json', sources)
    qualification = apply_sources(set(sources), folder, explicit_sources=sources)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = connect(); selected = set()
        for doi, payload in metadata.items():
            if not db.execute("SELECT 1 FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchone():
                continue
            selected.add(doi)
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                (payload['evidence_receipt'] + '#' + doi, doi, payload.get('native_metadata_kind', 'jats'),
                 payload['raw_sha256'], json.dumps(payload)))
        db.commit(); db.close()
        merged = apply_metadata(ROOT, scope_dois=selected, index_sources=False) if selected else {}
    result = {'qualification': qualification, 'qualified_scope': len(selected),
              'metadata': merged, 'network_requests': 0, 'final_task_closure': False}
    with (folder / 'applications.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    atomic_json(folder / 'application.json', result)
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
