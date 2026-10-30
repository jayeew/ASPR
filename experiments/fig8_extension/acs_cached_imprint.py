"""Recover explicit ACS publisher statements from a fixed, qualified archival scope."""
from __future__ import annotations

import gzip
import json
import re

from lxml import etree

from .cached_date_apply import run as apply_evidence
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def publisher_statement(raw: bytes) -> str:
    if raw.rstrip().endswith(b'</front>'):
        raw += b'</article>'
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    nodes = tree.xpath('./front/article-meta/permissions/copyright-statement')
    for node in nodes:
        value = ' '.join(''.join(node.itertext()).split())
        if re.search(r'\bPublished(?:\s+\d{4})?\s+by\s+American Chemical Society\b', value, re.I):
            return value
    return ''


def run() -> None:
    folder = ROOT / 'supplement/journal_type_policies/acs_imprint'
    folder.mkdir(exist_ok=True)
    scope = json.loads((folder.parent / 'acs_letter_application.json').read_text())
    sources = {}
    audit = []
    db = connect()
    for change in scope['changes']:
        row = change['after']
        for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],)):
            item = json.loads(payload)
            if (item.get('family') != 'archived_jats' or item.get('identity') != 'matched'
                    or not item.get('online_date') or item.get('publisher_statement_rule')):
                continue
            raw = gzip.decompress((ROOT / 'fields/qualification' / item['receipt']).read_bytes())
            statement = publisher_statement(raw)
            audit.append({'work_id': row['work_id'], 'doi': row['doi'], 'receipt': item['receipt'],
                          'statement': statement, 'status': 'explicit_publisher' if statement else 'no_explicit_publisher'})
            if not statement or set(item.get('publishers', [])) - {'American Chemical Society'}:
                continue
            extra = {k: v for k, v in item.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
            extra.update(publishers=['American Chemical Society'], publisher_statement=statement,
                         publisher_statement_rule='Explicit Published [year] by phrase in primary article permissions; copyright holder alone never used',
                         publisher_statement_locator='front/article-meta/permissions/copyright-statement')
            sources.setdefault(row['doi'], []).append(extra)
    db.close()
    atomic_json(folder / 'source_review.json', {'records': audit, 'network_requests': 0})
    result = apply_evidence(set(sources), folder, explicit_sources=sources)
    atomic_json(folder / 'application.json', result)


if __name__ == '__main__':
    run()
