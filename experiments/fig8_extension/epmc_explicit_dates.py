"""Recover explicit electronic dates from DOI-linked existing Europe PMC core caches."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import date

from .cached_date_apply import run as apply_dates
from .fig8b_qualification_build import plain
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/epmc_explicit_electronic_dates'


def date_source(record: dict, doi: str, receipt: str) -> dict | None:
    if record.get('http_status') != 200:
        return None
    matches = [r for r in record.get('response', {}).get('resultList', {}).get('result', [])
               if r.get('doi', '').lower() == doi]
    if len(matches) != 1 or matches[0].get('source') != 'MED':
        return None
    item = matches[0]; online = item.get('electronicPublicationDate', '')
    try:
        if len(online) != 10 or date.fromisoformat(online).isoformat() != online:
            return None
    except (ValueError, TypeError):
        return None
    journal = item.get('journalInfo', {}).get('journal', {})
    if not item.get('title') or not journal.get('title') or not (journal.get('issn') or journal.get('essn')):
        return None
    return {'family': 'europepmc_explicit_electronic_date', 'title': plain(item['title']),
            'native_title': item['title'], 'title_normalization': 'Decode literal inline publication-title markup; preserve text and scientific symbols',
            'issns': [journal[k] for k in ['issn', 'essn'] if journal.get(k)],
            'journal_titles': [journal['title']], 'kind': 'unspecified', 'categories': [],
            'online_date': online, 'date_precision': 'day', 'receipt': receipt,
            'url': record['url'], 'pmid': item.get('id'),
            'native_date_locator': 'resultList.result[doi].electronicPublicationDate',
            'date_rule_receipt': str((FOLDER / 'policy.json').relative_to(ROOT)),
            'source_authority': 'Europe PMC named bibliographic index; date only, not original-research classification'}


def run() -> None:
    folder = FOLDER / 'remaining_cached_dates'; folder.mkdir(exist_ok=True)
    if (folder / 'application.json').exists():
        return
    db = connect(); paths = defaultdict(set)
    papers = db.execute("SELECT work_id,doi FROM papers WHERE status='pending_primary_online_date'").fetchall()
    for wid, doi in papers:
        for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='secondary'", (wid,)):
            for item in json.loads(payload):
                if item.get('receipt', '').startswith('epmc_raw/'):
                    paths[item['receipt']].add(doi)
    db.close()
    sources = defaultdict(list); checks = []
    prior = json.loads((FOLDER / 'cell_review.json').read_text())['records']
    done = {(r['doi'], r['receipt']) for r in prior}
    for receipt, dois in paths.items():
        dois = {doi for doi in dois if (doi, receipt) not in done}
        if not dois:
            continue
        record = json.loads(gzip.decompress((ROOT / 'fields/qualification' / receipt).read_bytes()))
        for doi in sorted(dois):
            source = date_source(record, doi, receipt)
            if source:
                sources[doi].append(source)
            checks.append({'doi': doi, 'receipt': receipt,
                           'status': 'explicit_date_for_identity_merge' if source else 'no_unique_valid_MED_electronic_date',
                           'online_date': source['online_date'] if source else '', 'final_task_closure': False})
    atomic_json(folder / 'review.json', {'records': checks, 'scoped_papers': len(papers),
        'doi_linked_cache_pairs': sum(map(len, paths.values())),
        'outcomes': dict(Counter(r['status'] for r in checks)), 'network_requests': 0,
        'scope': 'Existing pending-online-date candidates only; unlinked cache/source paths not closed',
        'final_task_closure': False})
    result = apply_dates(set(sources), folder, explicit_sources=dict(sources)) if sources else {}
    atomic_json(folder / 'application.json', result)


if __name__ == '__main__':
    run()
