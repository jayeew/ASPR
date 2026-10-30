"""Confirm literal publisher Opinion records using retained source documents."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import etree

from .fig8b_qualification_decide import key, literal_nonresearch_alias
from .mdpi_cache_apply import evidence as mdpi_evidence
from .mdpi_oai_harvest import NS, extract
from .qa_type_apply import run as apply_review
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/publisher_opinion_policy'


def confirmed(paper: dict, evidence: dict, issns: set[str]) -> bool:
    receipt = 'fields/qualification/' + evidence['receipt']
    if evidence['family'] == 'publisher_frontiers':
        native = native_metadata(receipt, paper['doi'])
        return (paper['doi'] in {v.lower() for v in native.get('doi', [])}
                and signature_match({**paper, 'journal_name': paper['journal']}, native, issns)
                and 'opinion' in {key(v) for v in native.get('article_types', [])})
    raw = gzip.decompress((ROOT / receipt).read_bytes())
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    records = [extract(n) for n in tree.findall('.//o:record', NS)]
    records = [r for r in records if paper['doi'] in r['dois']]
    if len(records) != 1:
        return False
    proof = mdpi_evidence(records[0], paper, issns,
                          {'raw_receipt': receipt}, hashlib.sha256(raw).hexdigest())
    return bool(proof and key(proof['kind']) == 'opinion')


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'status.json').exists():
        return
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect(); papers, checks, unresolved = {}, [], []
    query = """SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,p.shard,p.status,e.payload
        FROM current_evidence_index e JOIN papers p USING(work_id)
        WHERE e.kind='primary' AND json_extract(e.payload,'$.kind') IN ('opinion','Opinion')"""
    for wid, doi, title, journal, sid, shard, status, payload in db.execute(query):
        e = json.loads(payload)
        if not literal_nonresearch_alias(e):
            continue
        paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal': journal,
                 'source_id': sid, 'shard': shard, 'before_status': status}
        if not confirmed(paper, e, journals[sid]):
            unresolved.append({'work_id': wid, 'receipt': e['receipt'], 'status': 'native_identity_or_type_unconfirmed'})
            continue
        papers.setdefault(wid, {**paper, 'evidence': []})['evidence'].append(e)
        checks.append({'work_id': wid, 'receipt': e['receipt'], 'kind': e['kind']})
    db.close()
    atomic_json(FOLDER / 'review.json', {'papers': list(papers.values()), 'native_checks': checks,
        'unresolved': unresolved, 'policy_sources': [
            {'url': 'https://www.mdpi.com/about/article_types', 'section': 'Opinion',
             'method': 'Official page returned by web search 2026-10-25; direct 403 retained, not retried',
             'interpretation': 'Short viewpoints with review-like structure, rather than original research. Hypothesis and other formats remain separate.'},
            {'url': 'https://www.frontiersin.org/for-authors/where-to-publish/article-types',
             'receipt': 'supplement/frontiers_type_policy/receipt.json', 'section': 'Opinion',
             'interpretation': 'Viewpoints on findings, methods or hypotheses; original unpublished data not permitted.'}],
        'paper_network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'selected_papers': len(papers), 'native_checks': len(checks), 'unresolved': len(unresolved)}), flush=True)
    print(json.dumps(apply_review(FOLDER.name)), flush=True)


if __name__ == '__main__':
    run()
