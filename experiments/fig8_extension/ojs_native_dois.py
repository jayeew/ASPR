"""Recover candidate DOI leads from cached native OAI identifiers, not references."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter

from .fig8b_qualification_build import match
from .ojs_cached_evidence import NS, dc_evidence, entries
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def native_dois(info: dict) -> set[str]:
    entry = entries(info['raw_receipt']).get(info['identifier'])
    if entry is None:
        return set()
    result = set()
    for value in entry.xpath('.//dc:identifier/text()', namespaces=NS):
        normalized = re.sub(r'^https?://(?:dx\.)?doi\.org/|^doi:\s*', '', value.strip(), flags=re.I).lower()
        if re.fullmatch(r'10\.\d{4,9}/\S+', normalized):
            result.add(normalized)
    return result


def run() -> dict:
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    proposals = []
    seen = set()
    stats = Counter()
    folder = ROOT / 'supplement/missing_doi'
    for source_folder in [folder, folder / 'additional_ojs']:
        for path in sorted((source_folder / 'harvested').glob('*.json')):
            source = json.loads(path.read_text())
            for wid, info in source.get('matched', {}).items():
                if wid in seen or info['title_identity'] != 'exact':
                    continue
                seen.add(wid)
                row = db.execute("SELECT doi,title,journal,source_id FROM papers WHERE work_id=? AND status<>'excluded_duplicate_wrong_journal'", (wid,)).fetchone()
                if not row or row[0]:
                    continue
                _, title, journal, sid = row
                paper = {'work_id': wid, 'title': title, 'journal_name': journal, 'source_id': sid}
                evidence = dc_evidence(info, source['url'], title)
                if not any(match(paper, item, journals[sid])['identity'] == 'matched' for item in evidence):
                    stats['journal_identity_unresolved'] += 1
                    continue
                dois = native_dois(info)
                if len(dois) != 1:
                    stats['no_single_native_doi'] += 1
                    continue
                doi = next(iter(dois))
                existing = [r[0] for r in db.execute('SELECT work_id FROM papers WHERE doi=?', (doi,))]
                status = 'existing_doi_requires_duplicate_review' if existing else 'native_doi_identity_matched_pending_apply'
                if doi.startswith(('10.5281/', '10.6084/', '10.5061/', '10.48550/')):
                    status = 'repository_doi_requires_version_identity_review'
                proposals.append({'work_id': wid, 'doi': doi, 'title': title, 'source_id': sid,
                                  'status': status, 'existing_work_ids': existing, 'receipt': info['raw_receipt'],
                                  'native_identifier': info['identifier'], 'source_url': source['url']})
    db.close()
    counts = Counter(r['doi'] for r in proposals)
    for row in proposals:
        if counts[row['doi']] > 1:
            row['status'] = 'multiple_candidates_share_native_doi'
    stats.update(r['status'] for r in proposals)
    result = {'status': 'cached_doi_proposals_no_main_table_update', 'counts': dict(stats),
              'records': proposals, 'network_requests': 0, 'scope': 'Current harvested fixed candidate endpoints only'}
    atomic_json(folder / 'native_doi_proposals.json', result)
    print(json.dumps(dict(stats)), flush=True)
    return result


if __name__ == '__main__':
    run()
