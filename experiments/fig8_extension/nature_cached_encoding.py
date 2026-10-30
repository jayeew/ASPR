"""Review existing Nature byline caches for UTF-8 decoding differences, without HTTP."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .springer_author_cache import already_processed, parse_authors, run as apply_authors
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/journal_type_policies/encoding_review'


def inspect_receipt(job: tuple, journals: dict[str, set[str]]) -> dict:
    wid, receipt, shard, doi, title, journal, sid = job
    record = json.loads((ROOT / receipt).read_text())
    result = {'work_id': wid, 'receipt': receipt, 'doi': doi}
    if record.get('route') not in {'nature_cached_page', 'nature'}:
        return {**result, 'status': 'other_adapter_not_in_scope'}
    if record.get('raw_text_encoding') == 'utf-8':
        return {**result, 'status': 'explicit_utf8_already_present'}
    path = ROOT / 'fields/qualification' / record['raw_receipt']
    try:
        raw = gzip.decompress(path.read_bytes())
        text = raw.decode('utf-8')
    except (OSError, EOFError, UnicodeDecodeError) as error:
        return {**result, 'status': 'unresolved_cache_encoding', 'reason': str(error)}
    parsed = parse_publisher(text)
    authors = parse_authors(text)
    differences = [name for name, old, new in (
        ('publisher_metadata', parse_publisher(raw), parsed),
        ('byline', parse_authors(raw), authors)) if old != new]
    if not differences:
        return {**result, 'status': 'no_decoding_difference'}
    paper = {'doi': doi, 'title': title, 'journal_name': journal, 'source_id': sid}
    if doi not in parsed['doi'] or not signature_match(paper, parsed, journals[sid]):
        return {**result, 'status': 'corrected_identity_unconfirmed', 'differences': differences}
    if not authors['authors']:
        return {**result, 'status': 'corrected_page_without_byline'}
    target = FOLDER / (hashlib.sha256((wid + receipt).encode()).hexdigest() + '.json')
    updated = {**record, 'parsed': parsed, 'raw_text_encoding': 'utf-8',
               'raw_sha256': hashlib.sha256(raw).hexdigest(), 'network_requests': 0,
               'encoding_review': {'prior_receipt': receipt, 'differences': differences,
                                   'utf8_roundtrip_verified': text.encode('utf-8') == raw}}
    atomic_json(target, updated)
    return {**result, 'status': 'matched_cache_decoding_repair', 'differences': differences,
            'job': [wid, shard, str(target.relative_to(ROOT))]}


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    db = connect()
    jobs = db.execute("""SELECT DISTINCT s.work_id,s.receipt,p.shard,p.doi,p.title,p.journal,p.source_id
        FROM springer_author_processed s JOIN papers p ON p.work_id=s.work_id
        WHERE s.receipt LIKE 'supplement/journal_type_policies/%'
        AND s.receipt NOT LIKE 'supplement/journal_type_policies/encoding_review/%'
        AND p.status='included_verified_research'""").fetchall()
    db.close()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    with ThreadPoolExecutor(max_workers=8) as pool:
        audit = list(pool.map(lambda job: inspect_receipt(job, journals), jobs))
    # Multiple retained receipts for one work remain in the audit; use one matching source.
    selected = {row['work_id']: row['job'] for row in audit if 'job' in row}
    db = connect()
    selected = {wid: job for wid, job in selected.items()
                if not already_processed(db, ROOT, wid, job[2])}
    db.close()
    counts = dict(Counter(row['status'] for row in audit))
    atomic_json(FOLDER / 'audit.json', {'counts': counts, 'papers': audit, 'network_requests': 0})
    print(json.dumps({'review': counts, 'selected_papers': len(selected)}), flush=True)
    result = apply_authors(selected_jobs=list(selected.values()), workers=8,
                           fill_abstract=True, reprocess_selected=True) if selected else {}
    report = {'review': counts, 'selected_papers': len(selected), 'application': result,
              'network_requests': 0, 'all_metadata_complete_not_implied': True}
    atomic_json(FOLDER / 'status.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
    from .derived_refresh import refresh
    refresh()
