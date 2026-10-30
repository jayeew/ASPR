"""Repair structural JATS labels from existing exact paper/affiliation evidence."""
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

from .supplement_queue import ROOT, connect
from .supplement_sources import affiliation_text, atomic_json, text


def replacements(raw: bytes, doi: str) -> dict[str, str]:
    if raw.rstrip().endswith(b'</front>'):
        raw += b'</article>'
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    for node in tree.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    if tree.tag != 'article':
        articles = [a for a in tree.xpath('.//article') if doi.casefold() in {
            v.strip().casefold().removeprefix('https://doi.org/') for v in
            a.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}]
        if len(articles) != 1:
            return {}
        tree = articles[0]
    identifiers = tree.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')
    if doi.casefold() not in {v.strip().casefold().removeprefix('https://doi.org/') for v in identifiers}:
        return {}
    variants = defaultdict(set)
    for node in tree.xpath('./front/article-meta//aff'):
        old, new = text(node), affiliation_text(node)
        if old != new and new:
            variants[old].add(new)
    return {old: next(iter(values)) for old, values in variants.items() if len(values) == 1}


def clean_slot(slot: dict, changes: dict[str, str], receipt: str) -> bool:
    """A cleaned value must never be mistaken for another raw affiliation."""
    old = slot.get('text', '')
    if old not in changes or slot.get('text_cleanup_source') or slot.get('official_identity_evidence'):
        return False
    slot['text'] = changes[old]
    slot['text_cleanup_source'] = receipt
    return True


def run(root: Path = ROOT, selected_work_ids: set[str] | None = None,
        folder_name: str = 'affiliation_text_repair') -> dict:
    folder = root / 'supplement' / folder_name
    folder.mkdir(exist_ok=True)
    stats = Counter()
    with (root / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(root)
        # Query current affected task links, not the entire candidate library.
        selected = db.execute("""SELECT DISTINCT p.work_id,p.shard FROM tasks t
            JOIN task_papers l USING(task_id) JOIN papers p ON p.work_id=l.work_id
            WHERE p.status='included_verified_research' AND t.active=1 AND
            ((t.kind='institution' AND (t.entity_key LIKE '%ror.org%' OR
              t.entity_key LIKE '%grid.%' OR t.entity_key GLOB '[0-9]*')) OR
             (t.kind='university_ownership' AND t.entity_key IN
              ('https://openalex.org/I67311998','https://openalex.org/I148283060',
               'https://openalex.org/I200870766','https://openalex.org/I1314696892')))""").fetchall() if selected_work_ids is None else []
        if selected_work_ids is not None:
            selected = [row for wid in sorted(selected_work_ids)
                        for row in db.execute("SELECT work_id,shard FROM papers WHERE work_id=? AND status='included_verified_research'", (wid,))]
        scope_path = folder / 'scope.json'
        if not scope_path.exists():
            atomic_json(scope_path, {'work_ids': sorted({w for w, _ in selected}),
                        'basis': ('Caller-selected current qualified papers with native signature mismatch'
                                  if selected_work_ids is not None else
                                  'Current qualified task links with contaminated signatures, plus four reviewed laboratory ownership queues'),
                        'network_requests': 0})
        scope = set(json.loads(scope_path.read_text())['work_ids'])
        grouped = defaultdict(set)
        for wid, shard in selected:
            if wid in scope:
                grouped[shard].add(wid)
        for shard, ids in grouped.items():
            path = root / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            changed = False
            for paper in rows:
                if paper['work_id'] not in ids:
                    continue
                authors = json.loads(paper['authors_json'])
                before = paper['authors_json']
                cache = {}
                for author in authors:
                    if author.get('nature_v1_slots') is not None:
                        continue
                    source = author.get('publisher_affiliation_source')
                    if not source or not author.get('publisher_affiliation_slots'):
                        continue
                    if source not in cache:
                        raw_path = root / source
                        try:
                            raw = gzip.decompress(raw_path.read_bytes())
                            changes = replacements(raw, paper['doi'])
                        except (OSError, EOFError, etree.XMLSyntaxError) as error:
                            cache[source] = ({}, '')
                            stats['source_' + type(error).__name__] += 1
                            continue
                        token = hashlib.sha256((source + hashlib.sha256(raw).hexdigest()).encode()).hexdigest()
                        receipt = folder / (token + '.json')
                        if changes:
                            atomic_json(receipt, {'source': source, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                        'doi': paper['doi'], 'method': 'Remove only XML label and institution-id elements, preserve tails',
                                        'replacements': changes})
                        cache[source] = (changes, str(receipt.relative_to(root)))
                    changes, receipt = cache[source]
                    for slot in author['publisher_affiliation_slots']:
                        old = slot.get('text', '')
                        if old not in changes:
                            continue
                        if slot.get('official_identity_evidence'):
                            stats['existing_official_signature_preserved'] += 1
                            continue
                        if clean_slot(slot, changes, receipt):
                            stats['affiliation_slots_cleaned'] += 1
                after = json.dumps(authors, ensure_ascii=False)
                if json.loads(before) == authors:
                    continue
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'authors_json', before, after,
                            str(scope_path.relative_to(root)),
                            'Exact raw JATS DOI and original affiliation text match; structural labels/IDs removed; per-slot raw provenance retained; no entity/country/ownership changes'))
                paper['authors_json'] = after
                stats['papers_changed'] += 1
                changed = True
            if changed:
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
            db.commit()
            atomic_json(folder / 'status.json', {'status': 'running', 'shard': shard, **stats})
        db.close()
    result = {'status': 'cache_text_repair_complete_allocation_unchanged', 'fixed_scope_papers': len(scope),
              'network_requests': 0, **stats}
    atomic_json(folder / 'status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    result = run()
    if result.get('papers_changed'):
        from .derived_refresh import refresh
        refresh()
