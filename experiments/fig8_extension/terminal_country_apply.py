"""Fill empty country slots using exact native author/address evidence, cache only."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from lxml import etree

from . import metadata_task_delta
from .arrow_row_patch import patch_rows, selected_rows
from .cureus_country_apply import enrich
from .aps_authors import source_record as aps_source_record
from .derived_refresh import refresh
from .nature_backfill_resolve import clean_address
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata, parse_publisher
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/explicit_terminal_country'
RULE = 'supplement/explicit_terminal_country/scope.json'


def native_metadata(receipt: str, doi: str) -> dict:
    """Reuse existing adapters; do not infer links from citation-author lists."""
    if receipt.startswith('fields/qualification/aps_all_raw/'):
        return aps_source_record(ROOT, receipt, doi)[0]['parsed']
    cloud_xml = '/supplement/pmc_cloud/xml_raw/' in str((ROOT / receipt).resolve())
    if not receipt.endswith(('.xml.gz', '.html.gz')) and not cloud_xml:
        return {}
    raw = gzip.decompress((ROOT / receipt).read_bytes())
    if receipt.endswith('.html.gz'):
        from .springer_author_cache import parse_authors
        from .frontiers_authors import parse_authors as frontiers_authors
        native = parse_publisher(raw)
        authors = frontiers_authors(raw, doi) if doi.startswith('10.3389/') else parse_authors(raw)
        if doi.startswith('10.26599/'):
            from .sciopen_metadata import parse_authors as sciopen_authors
            authors = sciopen_authors(raw)
        return {**native, **authors}
    document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
    tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
    if etree.QName(tree).localname == 'Publisher':
        from .springer_native_xml import parse as parse_springer
        return parse_springer(raw)
    articles = [tree] if etree.QName(tree).localname == 'article' else tree.xpath('//*[local-name()="article"]')
    if len(articles) > 1:
        articles = [article for article in articles if doi.casefold() in {
            value.strip().casefold().removeprefix('https://doi.org/') for value in article.xpath(
                './*[local-name()="front"]/*[local-name()="article-meta"]/*[local-name()="article-id"][@pub-id-type="doi"]/text()')}]
    if len(articles) != 1:
        return {}
    article = articles[0]
    if article.getparent() is not None:
        article.getparent().remove(article)
    for node in article.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    etree.cleanup_namespaces(article)
    native = jats_metadata(etree.tostring(article))
    native['issns'] = native['issn']
    return native


def run(remaining_formats_only: bool = False,
        selected_pairs: set[tuple[str, str]] | None = None,
        allow_unique_accent_names: bool = False,
        deposited_country_metadata_only: bool = False,
        verified_identity_only: bool = False) -> dict:
    match_authors = enrich
    if allow_unique_accent_names:
        from .country_accent_names import enrich as accent_enrich
        match_authors = accent_enrich
    if verified_identity_only:
        from .country_verified_identity import enrich as identity_enrich
        match_authors = identity_enrich
    load_metadata = native_metadata
    if deposited_country_metadata_only:
        from .crossref_country_metadata import load
        load_metadata = load
    scope = json.loads((ROOT / RULE).read_text())
    remaining = selected_pairs
    if remaining_formats_only:
        previous = json.loads((FOLDER / 'application.json').read_text())
        remaining = {(v['work_id'], v['receipt']) for v in previous['audit']
                     if v['status'] == 'native_adapter_required'}
        atomic_json(FOLDER / 'xml_application.json', previous)
    groups = defaultdict(dict)
    for paper in scope['papers']:
        if remaining is not None and not any((paper['work_id'], s['receipt']) in remaining for s in paper['slots']):
            continue
        for slot in paper['slots']:
            suffix = slot['address'].rsplit(',', 1)[-1].strip().rstrip('.').casefold()
            assert scope['mapping'].get(suffix) == slot['code']
        groups[paper['shard']][paper['work_id']] = paper
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    counts = Counter()
    audit = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for shard, targets in groups.items():
            path = ROOT / 'fields/papers' / shard
            decision_path = ROOT / 'fields/qualification/decisions' / shard
            before_signature = metadata_task_delta.signature(path, decision_path)
            table = pq.read_table(path)
            rows = selected_rows(table, set(targets))
            updates = {}
            for paper in rows:
                if paper['work_id'] not in targets:
                    continue
                current = db.execute('SELECT status FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()
                if not current or current[0] != 'included_verified_research':
                    for receipt in {s['receipt'] for s in targets[paper['work_id']]['slots']}:
                        audit.append({'work_id': paper['work_id'], 'receipt': receipt,
                                      'status': 'current_qualification_not_included'})
                    continue
                target = targets[paper['work_id']]
                before = paper['authors_json']
                authors = json.loads(before)
                receipts = {s['receipt'] for s in target['slots']}
                for receipt in receipts:
                    if remaining is not None and (paper['work_id'], receipt) not in remaining:
                        continue
                    detail = {'work_id': paper['work_id'], 'receipt': receipt}
                    if not (ROOT / receipt).is_file():
                        audit.append({**detail, 'status': 'native_adapter_required'})
                        continue
                    try:
                        native = load_metadata(receipt, paper['doi'])
                    except (ValueError, etree.XMLSyntaxError) as error:
                        audit.append({**detail, 'status': 'native_parse_unresolved', 'reason': str(error)})
                        continue
                    if not native or not native.get('authors'):
                        audit.append({**detail, 'status': 'native_adapter_or_author_links_missing'})
                        continue
                    if paper['doi'] not in {v.lower() for v in native['doi']} or not signature_match(paper, native, journals[paper['source_id']]):
                        audit.append({**detail, 'status': 'identity_unconfirmed'})
                        continue
                    rules = {clean_address(s['address']): s['code'] for s in target['slots']
                             if s['receipt'] == receipt and ';' not in s['address']}
                    for author in native['authors']:
                        for affiliation in author['affiliations']:
                            address = clean_address(affiliation['text'])
                            codes = {c.get('code', '').upper() for c in affiliation.get('countries', [])} - {''}
                            if address in rules and codes and codes != {rules[address]}:
                                rules.pop(address)
                    updated, slots, people = match_authors(authors, native, receipt, rules, RULE, 'explicit_terminal_country_name')
                    if deposited_country_metadata_only:
                        for old, new in zip(authors, updated):
                            for a, b in zip(old.get('publisher_affiliation_slots') or [], new.get('publisher_affiliation_slots') or []):
                                if not a.get('country_codes') and b.get('country_codes'):
                                    b['country_evidence'].update(source_authority='Crossref publisher-deposited author-affiliation metadata',
                                        native_full_byline_confirmed=False,
                                        basis='unique_deposited_author_and_exact_address_with_explicit_terminal_country_name')
                    assert match_authors(updated, native, receipt, rules, RULE, 'explicit_terminal_country_name')[0] == updated
                    authors = updated
                    counts['country_slots_filled'] += slots
                    counts['author_countries_filled'] += people
                    audit.append({**detail, 'status': 'native_links_checked', 'slots_filled': slots})
                if authors == json.loads(before):
                    continue
                after = json.dumps(authors, ensure_ascii=False)
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'authors_json', before, after, RULE,
                            'Exact unambiguous terminal country; matched DOI, journal, title and unique author-address link; existing countries and Nature protected; source authority: ' +
                            ('Crossref deposited metadata, no full-byline confirmation' if deposited_country_metadata_only else 'native publication')))
                updates[paper['work_id']] = {'authors_json': after}
                counts['papers_changed'] += 1
            if updates:
                temp = path.with_suffix('.parquet.tmp')
                pq.write_table(patch_rows(table, updates), temp, compression='zstd')
                temp.replace(path)
                metadata_task_delta.record(db, shard, before_signature,
                    metadata_task_delta.signature(path, decision_path), set(updates), True)
            db.commit()
        db.close()
    result = {'status': 'cache_application_complete', **counts, 'audit': audit,
              'network_requests': 0, 'final_task_closure': False,
              'unique_latin_accent_name_matching': allow_unique_accent_names,
              'deposited_country_metadata_only': deposited_country_metadata_only,
              'verified_identity_only': verified_identity_only}
    atomic_json(FOLDER / 'application.json', result)
    refresh()
    print(json.dumps({k: v for k, v in result.items() if k != 'audit'}), flush=True)
    return result


if __name__ == '__main__':
    run()
