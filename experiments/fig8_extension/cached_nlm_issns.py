"""Enrich current serial identifiers from cached exact-title NLM catalog records."""
from __future__ import annotations

import csv
import fcntl
import gzip
import json
from collections import defaultdict

from lxml import etree

from .fig8b_qualification_decide import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/annual_nlm'
    records = []
    for path in sorted((folder / 'raw').glob('*.xml.gz')):
        try:
            tree = etree.fromstring(gzip.decompress(path.read_bytes()), etree.XMLParser(resolve_entities=False, no_network=True))
        except (OSError, etree.XMLSyntaxError):
            continue
        for node in tree.findall('NLMCatalogRecord'):
            issns = {n.text for n in node.findall('./ISSN[@ValidYN="Y"]') if n.text}
            title = node.findtext('./TitleMain/Title') or ''
            if title and issns:
                records.append((title, issns, node.findtext('NlmUniqueID'), str(path.relative_to(ROOT))))
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = ROOT / 'fields/journal_registry.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = list(reader)
        owners = defaultdict(set)
        by_title = defaultdict(list)
        for row in rows:
            by_title[key(row['journal_name'])].append(row)
            for issn in json.loads(row['issn'] or '[]'):
                owners[issn].add(row['source_id'])
        changes = []
        for title, issns, nlm_id, receipt in records:
            matching = by_title[key(title)]
            if len({r['source_id'] for r in matching}) != 1:
                continue
            sid = matching[0]['source_id']
            old_ids = set().union(*(set(json.loads(r['issn'] or '[]')) for r in matching))
            if not old_ids & issns or any(owners[i] - {sid} for i in issns):
                continue
            for row in matching:
                before = json.loads(row['issn'] or '[]')
                after = sorted(set(before) | issns)
                if set(before) == set(after):
                    continue
                event = {'source_id': sid, 'cohort': row['cohort'], 'before': before, 'after': after,
                         'nlm_id': nlm_id, 'nlm_title': title, 'receipt': receipt,
                         'reason': 'Exact current title and shared valid ISSN; additional direct valid ISSNs from same serial; related-title ISSNs excluded; no conflicting registry owner'}
                changes.append(event)
                row['issn'] = json.dumps(after)
            for issn in issns:
                owners[issn].add(sid)
        with (folder / 'issn_changes.jsonl').open('a') as stream:
            for event in changes:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        if changes:
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
    result = {'status': 'registry_updated_affected_evidence_requires_recheck', 'changed_rows': len(changes),
              'changed_sources': sorted({r['source_id'] for r in changes}), 'network_requests': 0}
    atomic_json(folder / 'issn_apply_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
