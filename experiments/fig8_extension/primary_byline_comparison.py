"""Compare cached JATS alternatives with retained primary links without overwriting."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .supplement_apply import signature_match, enrich_authors
from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized as country_key
from .nature_backfill_resolve import country_names
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv


def load_primary(receipt: str, doi: str) -> tuple[dict, bytes]:
    """Use the existing APS adapter on retained batches; otherwise read native receipt."""
    if receipt.startswith('fields/qualification/aps_all_raw/') and receipt.endswith('.json.gz'):
        from .aps_authors import source_record
        record, identity = source_record(ROOT, receipt, doi)
        if not identity:
            raise ValueError('No unique DOI in APS batch')
        raw = gzip.decompress((ROOT/receipt).read_bytes())
        return {**record, 'source':'publisher_api', 'raw_sha256':hashlib.sha256(raw).hexdigest()}, raw
    record = json.loads((ROOT/receipt).read_text())
    raw = gzip.decompress((ROOT/'fields/qualification'/record['raw_receipt']).read_bytes())
    if record.get('route') == 'pmc_cloud' and record.get('source_extent') in {'complete_front_only', 'full_object'}:
        acquisition = record.get('acquisition_receipt', {})
        if acquisition.get('retained_sha256'):
            record = {**record, 'raw_sha256': acquisition['retained_sha256']}
    if record.get('source') == 'publisher' and record.get('route') == 'copernicus':
        record = {**record, 'raw_sha256':hashlib.sha256(raw).hexdigest()}
    return record, raw


def incoming_metadata(record: dict, raw: bytes) -> dict:
    """Parse verified raw bytes using their declared publisher/archival format."""
    if record.get('route') == 'frontiers':
        from .frontiers_authors import parse_authors
        from .supplement_sources import parse_publisher
        return {**parse_publisher(raw), **parse_authors(raw, record['doi'])}
    if record.get('source') == 'publisher' and record.get('route') == 'copernicus':
        from .copernicus_bulk import parse_article
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        matches = []
        for article in tree.xpath('//*[local-name()="metadata"]/*[local-name()="article"]'):
            parsed = parse_article(article)
            if record['doi'].lower() in [d.lower() for d in parsed.get('doi', [])]:
                matches.append(parsed)
        if len(matches) != 1:
            raise ValueError('No unique Copernicus DOI in raw batch')
        parsed = matches[0]
        parsed['doi'] = [d.lower() for d in parsed['doi']]
        parsed['issns'] = parsed.get('issn', [])
        return parsed
    if record.get('source') == 'publisher_api' and record.get('route') == 'aps':
        return record['parsed']
    if record.get('source') == 'publisher_api' and record.get('route') == 'elife_api':
        from .elife_authors import metadata
        article = json.loads(raw)
        parsed = metadata(article)
        if article.get('doiVersion'):
            parsed['doi'].append(article['doiVersion'].lower())
        return parsed
    if record.get('source') == 'pmc' and record.get('route') == 'pmc_journal_oai':
        from .supplement_oai import parse_oai
        parsed = parse_oai(raw)
    else:
        parsed = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
    parsed['doi'] = [value.strip().lower() for value in parsed.get('doi', [])]
    parsed['issns'] = parsed.get('issn', [])
    return parsed


def normalized(value: str) -> str:
    return ' '.join(value.casefold().split())


def compare(current: list[dict], incoming: dict) -> str:
    """Require complete, unique, ordered names and exact complete address strings."""
    authors = incoming.get('authors') or []
    old_names = [normalized(a['name']) for a in current]
    new_names = [normalized(a['name']) for a in authors]
    if not old_names or old_names != new_names:
        return 'author_sequence_differs'
    if len(set(old_names)) != len(old_names):
        return 'duplicate_author_names_require_review'
    if incoming.get('byline_has_etal'):
        return 'incoming_byline_completeness_unconfirmed'
    for old, new in zip(current, authors):
        if old.get('nature_v1_slots'):
            return 'nature_review_preserved'
        if bool(old.get('publisher_author_nonpersonal', old.get('nonpersonal_verified'))) != bool(new.get('nonpersonal')):
            return 'personal_collective_type_differs'
        slots = old.get('publisher_affiliation_slots') or []
        addresses = new.get('affiliations') or []
        if not slots or not addresses or new.get('unresolved_affiliation_refs'):
            return 'incomplete_affiliation_links'
        before = Counter(normalized(s.get('text', '')) for s in slots)
        after = Counter(normalized(s.get('text', '')) for s in addresses)
        if '' in before or '' in after:
            return 'incomplete_affiliation_links'
        if before != after:
            return 'affiliation_statements_differ'
    return 'equivalent_complete_declared_links_entity_mapping_not_asserted'


def fill_missing_links(current: list[dict], source: dict, registry: dict,
                       by_name: dict, names: dict) -> tuple[list[dict], int]:
    """Reuse unique exact author matches without replacing any primary signature."""
    if all(a.get("nature_v1_slots") is not None or a.get("publisher_affiliation_slots") for a in current):
        return current, 0
    revised, _ = enrich_authors(current, source, registry, by_name, names)
    output = []
    changed = 0
    source_names = defaultdict(list)
    for author in source.get('authors', []):
        source_names[key(author['name'])].append(author)
    for old, new in zip(current, revised):
        matching = source_names[key(old['name'])]
        if old.get('nature_v1_slots') is not None or old.get('publisher_affiliation_slots'):
            output.append(old)
        elif len(matching) != 1 or bool(old.get('nonpersonal_verified')) != bool(matching[0].get('nonpersonal')):
            output.append(old)
        elif new.get('publisher_affiliation_slots'):
            output.append(new)
            changed += new != old
        else:
            output.append(old)
    return output, changed


def apply_missing_links(audit_folder: Path | None = None) -> dict:
    folder = audit_folder or ROOT / 'supplement/primary_byline_comparison'
    audit = json.loads((folder / 'audit.json').read_text())
    eligible = {'author_sequence_differs', 'duplicate_author_names_require_review',
                'incoming_byline_completeness_unconfirmed', 'incomplete_affiliation_links',
                'affiliation_statements_differ', 'personal_collective_type_differs',
                'equivalent_complete_declared_links_entity_mapping_not_asserted'}
    registry, by_name = defaultdict(list), defaultdict(list)
    for entity in json.loads((ROOT / 'fields/institution_registry.json').read_text()):
        if entity.get('ror'):
            registry[entity['ror'].rstrip('/')].append(entity['id'])
        by_name[key(entity['display_name'])].append(entity)
    names = country_names()
    for value in ['korea', 'georgia', 'congo']:
        names.pop(country_key(value), None)
    stats = Counter()
    attempts = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        grouped = defaultdict(dict)
        for item in audit:
            if item['status'] not in eligible:
                continue
            row = db.execute('SELECT shard,status FROM papers WHERE work_id=?', (item['work_id'],)).fetchone()
            if row and row[1] == 'included_verified_research':
                grouped[row[0]][item['work_id']] = item
        for shard, selected in grouped.items():
            path = ROOT / 'fields/papers' / shard
            candidates = pq.read_table(path, columns=['work_id', 'authors_json'],
                filters=[('work_id', 'in', list(selected))]).to_pylist()
            needed = {}
            for candidate in candidates:
                item = selected[candidate['work_id']]
                if hashlib.sha256(candidate['authors_json'].encode()).hexdigest() != item['retained_authors_sha256']:
                    attempts.append({'work_id': candidate['work_id'], 'status': 'retained_state_changed_requires_recomparison'})
                elif all(a.get('nature_v1_slots') is not None or a.get('publisher_affiliation_slots')
                         for a in json.loads(candidate['authors_json'])):
                    attempts.append({'work_id': candidate['work_id'], 'incoming_receipt': item['incoming_receipt'],
                                     'authors_filled': 0, 'status': 'no_additional_unique_exact_links'})
                else:
                    needed[candidate['work_id']] = item
            selected = needed
            if not selected:
                continue
            table = pq.read_table(path)
            rows = table.to_pylist()
            changed = False
            for paper in rows:
                item = selected.get(paper['work_id'])
                if not item:
                    continue
                # The comparison is tied to the retained state; changed state needs fresh review.
                if hashlib.sha256(paper['authors_json'].encode()).hexdigest() != item['retained_authors_sha256']:
                    attempts.append({'work_id': paper['work_id'], 'status': 'retained_state_changed_requires_recomparison'})
                    continue
                record, raw = load_primary(item['incoming_receipt'], paper['doi'])
                if hashlib.sha256(raw).hexdigest() != (item.get('incoming_raw_sha256') or record.get('raw_sha256')):
                    raise ValueError('Reviewed source changed')
                source = incoming_metadata(record, raw)
                source['evidence_receipt'] = 'fields/qualification/' + record['raw_receipt']
                old = json.loads(paper['authors_json'])
                authors, count = fill_missing_links(old, source, registry, by_name, names)
                attempts.append({'work_id': paper['work_id'], 'incoming_receipt': item['incoming_receipt'],
                                 'authors_filled': count, 'status': 'missing_links_filled' if count else 'no_additional_unique_exact_links'})
                if not count:
                    continue
                before = dict(paper)
                paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
                values = metrics(paper)
                for metric, value in values.items():
                    if 'after_' + metric in paper:
                        paper['after_' + metric] = value
                for field, value in paper.items():
                    if value != before[field]:
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                                   (paper['work_id'], field, json.dumps(before[field]), json.dumps(value), item['incoming_receipt'],
                                    'Reviewed alternate primary source; unique exact author name; fill missing slots only; existing primary links and author roster retained'))
                db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                           (not values['every_author_institution'], not values['every_author_country'], paper['work_id']))
                stats['papers_changed'] += 1
                stats['authors_primary_links_filled'] += count
                changed = True
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
            db.commit()
        db.close()
    atomic_json(folder / 'missing_link_attempts.json', attempts)
    result = {**stats, 'reviewed_papers': len(attempts), 'network_requests': 0,
              'existing_primary_values_overwritten': 0, 'byline_completeness_not_reclassified': True}
    atomic_json(folder / 'missing_link_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def run(audit_folder: Path | None = None, selected_jobs: list[tuple] | None = None) -> dict:
    folder = audit_folder or ROOT / 'supplement/primary_byline_comparison'
    folder.mkdir(exist_ok=True)
    db = connect()
    selected = selected_jobs if selected_jobs is not None else list(db.execute("""SELECT s.work_id,p.shard,s.receipt,s.digest
        FROM springer_author_processed s JOIN papers p USING(work_id)
        WHERE s.status='existing_primary_requires_source_comparison'
        AND s.receipt LIKE 'supplement/journal_type_policies/author_cache/%'
        AND p.status='included_verified_research'"""))
    db.close()
    grouped = defaultdict(list)
    for wid, shard, receipt, digest in selected:
        grouped[shard].append((wid, receipt, digest))
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    audit = []
    for shard, jobs in grouped.items():
        papers = {p['work_id']: p for p in pq.read_table(
            ROOT / 'fields/papers' / shard,
            columns=['work_id', 'doi', 'title', 'journal_name', 'source_id', 'authors_json'],
            filters=[('work_id', 'in', [j[0] for j in jobs])]).to_pylist()}
        for wid, receipt, digest in jobs:
            paper = papers[wid]
            current = json.loads(paper['authors_json'])
            item = {'work_id': wid, 'doi': paper['doi'], 'incoming_receipt': receipt,
                    'incoming_processed_digest': digest,
                    'retained_authors_sha256': hashlib.sha256(paper['authors_json'].encode()).hexdigest(),
                    'retained_receipts': sorted({a['publisher_affiliation_source'] for a in current
                                                if a.get('publisher_affiliation_source')}),
                    'status': 'unreviewed'}
            try:
                record, raw = load_primary(receipt, paper['doi'])
                raw_hash = hashlib.sha256(raw).hexdigest()
                native_pensoft = record.get('source') == 'publisher_jats' and record.get('route') == 'pensoft'
                item['incoming_raw_sha256'] = raw_hash
                if record.get('raw_sha256') and raw_hash != record['raw_sha256']:
                    item['status'] = 'source_hash_mismatch'
                elif not record.get('raw_sha256') and not native_pensoft:
                    item['status'] = 'source_hash_missing'
                elif record.get('source') != 'pmc' and not native_pensoft and not (record.get('source') == 'publisher_api' and record.get('route') in {'elife_api', 'aps'}) and not (record.get('source') == 'publisher' and record.get('route') in {'copernicus', 'frontiers'}):
                    item['status'] = 'different_source_parser_required'
                else:
                    parsed = incoming_metadata(record, raw)
                    if paper['doi'] not in parsed.get('doi', []) or not signature_match(paper, parsed, journals[paper['source_id']]):
                        item['status'] = 'incoming_identity_not_confirmed'
                    else:
                        item['status'] = compare(current, parsed)
                        old_keys = [key(a['name']) for a in current]
                        new_keys = [key(a['name']) for a in parsed.get('authors', [])]
                        old_counts, new_counts = Counter(old_keys), Counter(new_keys)
                        item['author_comparison'] = {
                            'retained_count': len(old_keys), 'incoming_count': len(new_keys),
                            'same_normalized_name_sequence': old_keys == new_keys,
                            'unique_exact_name_overlap': sum(old_counts[k] == new_counts[k] == 1 for k in old_counts),
                            'incoming_byline_has_etal': parsed.get('byline_has_etal'),
                            'different_names_do_not_prove_truncation': True}
            except (OSError, EOFError, ValueError, etree.XMLSyntaxError) as error:
                item['status'] = 'source_read_or_parse_failure'
                item['error'] = type(error).__name__
            audit.append(item)
    atomic_json(folder / 'audit.json', audit)
    result = {'reviewed_pairs': len(audit), 'status_counts': dict(Counter(a['status'] for a in audit)),
              'network_requests': 0, 'master_values_overwritten': 0,
              'scope': 'Existing cached primary-source alternatives held for comparison; author and address equivalence only, not entity ownership or country validation',
              'receipt': str((folder / 'audit.json').relative_to(ROOT))}
    atomic_json(folder / 'status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
