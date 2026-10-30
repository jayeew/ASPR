"""Apply identity-checked group-wide JATS addresses from the fixed cache review."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from lxml import etree

from .primary_byline_comparison import apply_missing_links, compare
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/group_affiliation_cache'
    candidates = [json.loads(line) for line in (folder / 'review.jsonl').open()
                  if json.loads(line)['status'] == 'group_affiliation_candidate']
    if len({row['work_id'] for row in candidates}) != len(candidates):
        raise ValueError('Multiple source candidates require explicit comparison')
    grouped = defaultdict(list)
    db = connect()
    audit = []
    for candidate in candidates:
        row = db.execute('SELECT shard,status FROM papers WHERE work_id=?',
                         (candidate['work_id'],)).fetchone()
        if row and row[1] == 'included_verified_research':
            grouped[row[0]].append(candidate)
        else:
            audit.append({**candidate, 'status': 'no_longer_qualified'})
    db.close()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    for shard, jobs in grouped.items():
        papers = {p['work_id']: p for p in pq.read_table(
            ROOT / 'fields/papers' / shard,
            columns=['work_id', 'doi', 'title', 'journal_name', 'source_id', 'authors_json'],
            filters=[('work_id', 'in', [j['work_id'] for j in jobs])]).to_pylist()}
        for candidate in jobs:
            paper = papers[candidate['work_id']]
            item = {'work_id': paper['work_id'], 'doi': paper['doi'],
                    'incoming_receipt': candidate['receipt'],
                    'retained_authors_sha256': hashlib.sha256(paper['authors_json'].encode()).hexdigest()}
            try:
                record = json.loads((ROOT / candidate['receipt']).read_text())
                raw = gzip.decompress((ROOT / 'fields/qualification' / record['raw_receipt']).read_bytes())
                if hashlib.sha256(raw).hexdigest() != record['raw_sha256']:
                    item['status'] = 'source_hash_mismatch'
                else:
                    source = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
                    if paper['doi'] not in source.get('doi', []) or not signature_match(
                            paper, source, journals[paper['source_id']]):
                        item['status'] = 'incoming_identity_not_confirmed'
                    else:
                        item['status'] = compare(json.loads(paper['authors_json']), source)
                        item['rule'] = 'single_unreferenced_affiliation_for_contributor_group'
                        item['rule_source'] = 'https://pmc.ncbi.nlm.nih.gov/tagging-guidelines/article/tags/'
            except (OSError, EOFError, ValueError, etree.XMLSyntaxError) as error:
                item.update(status='source_read_or_parse_failure', error=type(error).__name__)
            audit.append(item)
    atomic_json(folder / 'audit.json', audit)
    atomic_json(folder / 'audit_status.json', {
        'reviewed': len(audit), 'statuses': dict(Counter(row['status'] for row in audit)),
        'network_requests': 0, 'scope': 'All shared-group candidates in completed fixed cache review'})
    return apply_missing_links(folder)


if __name__ == '__main__':
    run()
