"""Fill empty countries from reviewed exact Cureus signatures and ISO code evidence."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .fig8b_qualification_build import key
from .nature_backfill_resolve import clean_address
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/cureus_country_codes'
RULE = 'supplement/cureus_country_codes/candidate_scope.json'


def enrich(authors: list[dict], source: dict, receipt: str, rules: dict[str, str],
           rule_receipt: str = RULE, basis: str = 'ISO_alpha3_suffix') -> tuple[list[dict], int, int]:
    """Require unique names after punctuation/space normalization and exact addresses."""
    result = json.loads(json.dumps(authors))
    native = source['authors']
    native_names = Counter(key(a['name']) for a in native)
    current_names = Counter(key(a['name']) for a in authors)
    links = {(key(a['name']), clean_address(f['text'])) for a in native for f in a['affiliations']}
    slots = 0
    people = 0
    for author in result:
        if author.get('nature_v1_slots') is not None or author.get('publisher_affiliation_source') != receipt:
            continue
        if native_names[key(author['name'])] != 1 or current_names[key(author['name'])] != 1:
            continue
        changed = False
        for slot in author.get('publisher_affiliation_slots') or []:
            address = clean_address(slot.get('text', ''))
            if slot.get('country_codes') or address not in rules or (key(author['name']), address) not in links:
                continue
            slot['country_codes'] = [rules[address]]
            slot['country_evidence'] = {'rule': rule_receipt, 'raw_signature': receipt,
                                        'basis': 'unique_normalized_native_author_and_exact_address_with_' + basis}
            slots += 1
            changed = True
        assigned = author.get('publisher_affiliation_slots') or []
        if changed and not author.get('countries') and assigned and all(s.get('country_codes') for s in assigned):
            author['countries'] = sorted({c for s in assigned for c in s['country_codes']})
            author['country_source'] = ('all_publisher_slots_explicit_country_with_ISO_suffix_rule'
                                        if basis == 'ISO_alpha3_suffix' else 'all_publisher_slots_explicit_terminal_country')
            people += 1
        elif changed and assigned and all(s.get('country_codes') for s in assigned):
            explicit = {c for s in assigned for c in s['country_codes']}
            if set(author.get('countries', [])) != explicit:
                author['country_conflict_retained'] = True
                prefix = 'ISO' if basis == 'ISO_alpha3_suffix' else basis
                author['country_conflict_disposition'] = prefix + '_signed_countries_disagree_with_existing_author_country; review_required'
    return result, slots, people


def run() -> dict:
    specification = json.loads((ROOT / RULE).read_text())
    iso = json.loads((ROOT / specification['iso_receipt']).read_text())['3166-1']
    codes = {r['alpha_3']: r['alpha_2'] for r in iso}
    assert all(codes[r['signed_code']] == r['iso_alpha2']
               and r['exact_address'].rsplit(',', 1)[-1].strip() == r['signed_code']
               for r in specification['entries'])
    rules = {r['exact_address']: r['iso_alpha2'] for r in specification['entries']}
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    counts = Counter()
    audit = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        groups = defaultdict(set)
        for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE journal='Cureus' AND status='included_verified_research'"):
            groups[shard].add(wid)
        for shard, ids in groups.items():
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            changed_shard = False
            for paper in rows:
                if paper['work_id'] not in ids:
                    continue
                before = paper['authors_json']
                authors = json.loads(before)
                sources = {a.get('publisher_affiliation_source', '') for a in authors
                           if any(not s.get('country_codes') and clean_address(s.get('text', '')) in rules
                                  for s in a.get('publisher_affiliation_slots') or [])}
                for receipt in sorted(sources):
                    if not receipt.startswith('fields/qualification/jats_raw/'):
                        audit.append({'work_id': paper['work_id'], 'receipt': receipt, 'status': 'unsupported_original_source'})
                        continue
                    raw = gzip.decompress((ROOT / receipt).read_bytes())
                    native = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
                    native['issns'] = native['issn']
                    if (paper['doi'] not in {d.lower() for d in native['doi']}
                            or native['journal_titles'] != ['Cureus']
                            or not signature_match(paper, native, journals[paper['source_id']])):
                        audit.append({'work_id': paper['work_id'], 'receipt': receipt, 'status': 'identity_unconfirmed'})
                        continue
                    authors, slots, people = enrich(authors, native, receipt, rules)
                    assert enrich(authors, native, receipt, rules)[0] == authors
                    counts['empty_country_slots_filled'] += slots
                    counts['empty_author_countries_filled'] += people
                    audit.append({'work_id': paper['work_id'], 'receipt': receipt, 'slots_filled': slots,
                                  'status': ('matched_native_identity_and_author_links' if slots else
                                             'native_identity_confirmed_no_unique_empty_author_address_match'),
                                  'repeat_safe': True})
                if authors == json.loads(before):
                    continue
                after = json.dumps(authors, ensure_ascii=False)
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'authors_json', before, after, RULE,
                            'Exact Cureus native author-address link and reviewed ISO alpha3 suffix; empty country fields only'))
                paper['authors_json'] = after
                counts['papers_changed'] += 1
                changed_shard = True
            if changed_shard:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
                db.commit()
        db.close()
    result = {**counts, 'audit': audit, 'network_requests': 0, 'qualification_unchanged': True}
    with (FOLDER / 'application_events.jsonl').open('a') as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    atomic_json(FOLDER / 'application.json', result)
    refresh()
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != 'audit'}))
