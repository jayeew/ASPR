"""Recover exact publisher raw-name links from retained OpenAlex authorship identities."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

from .fig8b_qualification_build import key
from .springer_author_cache import prepare_source
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/raw_author_name_recovery'


@lru_cache(maxsize=8)
def original_works(receipt: str) -> tuple[list[dict], str]:
    raw = gzip.decompress((ROOT / 'fields' / receipt).read_bytes())
    response = json.loads(raw).get('response') or {}
    return response.get('results', [response]), hashlib.sha256(raw).hexdigest()


def reconcile(current: list[dict], native: list[dict], original: dict, proof: dict) -> tuple[list[dict], list[dict]]:
    authors = json.loads(json.dumps(native))
    native_counts = Counter(key(a['name']) for a in authors)
    current_counts = Counter(key(a['name']) for a in current)
    raw_authors = original.get('authorships') or []
    raw_counts = Counter(key(a.get('raw_author_name', '')) for a in raw_authors)
    links = []
    for old in current:
        if old.get('publisher_affiliation_slots') or key(old['name']) in native_counts:
            continue
        matches = [a for a in raw_authors if a.get('author', {}).get('id') == old.get('author_id')
                   and key(a['author'].get('display_name', '')) == key(old['name'])]
        if not old.get('author_id') or len(matches) != 1 or current_counts[key(old['name'])] != 1:
            continue
        name = matches[0].get('raw_author_name', '')
        if not name or raw_counts[key(name)] != 1 or native_counts[key(name)] != 1:
            continue
        target = next(a for a in authors if key(a['name']) == key(name))
        if target.get('nonpersonal') or not target.get('affiliations') or target.get('unresolved_affiliation_refs'):
            continue
        evidence = dict(proof, author_id=old['author_id'], canonical_name=old['name'],
                        raw_author_name=name, native_name=target['name'])
        target.update(native_name=target['name'], name=old['name'], raw_name_identity_evidence=evidence)
        links.append(evidence)
    if len({key(a['name']) for a in authors}) != len(authors):
        return native, []
    return authors, links


def prepare(paper: dict, receipt: str, issns: set[str],
            allow_different_primary: bool = False) -> tuple[dict | None, str]:
    current = json.loads(paper['authors_json'])
    if any(a.get('nature_v1_slots') for a in current):
        return None, 'nature_review_preserved'
    if all(a.get('publisher_affiliation_slots') for a in current):
        return None, 'current_links_already_present'
    aps = receipt.startswith('fields/qualification/aps_all_raw/') and receipt.endswith('.json.gz')
    record, _, native = prepare_source(ROOT, paper, receipt, issns, aps_only=aps)
    if not native.get('authors'):
        return None, 'native_identity_or_author_data_unavailable'
    source = 'fields/qualification/' + record['raw_receipt']
    different_primary = any(a.get('publisher_affiliation_slots') and a.get('publisher_affiliation_source') != source for a in current)
    if different_primary and not allow_different_primary:
        return None, 'different_primary_source_requires_comparison'
    works, digest = original_works(paper['raw_receipt'])
    original = [w for w in works if w.get('id') == paper['work_id']
                and (w.get('doi') or '').lower().removeprefix('https://doi.org/') == paper['doi']
                and key(w.get('title', '')) == key(paper['title'])]
    if len(original) != 1:
        return None, 'original_identity_not_unique'
    proof = dict(work_id=paper['work_id'], doi=paper['doi'],
                 openalex_receipt='fields/' + paper['raw_receipt'], openalex_raw_sha256=digest,
                 publisher_raw_receipt=record['raw_receipt'], original_parsed_receipt=receipt,
                 rule='Unique original author ID and canonical name; unique original raw name equals native publisher name; exact paper DOI/title/work; no fuzzy matching')
    authors, links = reconcile(current, native['authors'], original[0], proof)
    if not links:
        return None, 'no_unique_original_raw_name_link'
    return dict(record, source='publisher_native_xml', route='raw_name_reconciled_cache',
                parsed={**record['parsed'], **native, 'authors': authors}, raw_name_links=links,
                missing_links_only=different_primary), 'prepared'


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    (FOLDER / 'staged').mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{ROOT / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    groups: dict[str, dict] = defaultdict(dict)
    query = """SELECT p.work_id,p.shard FROM tasks t JOIN papers p ON p.work_id=t.entity_key
        WHERE t.kind='metadata' AND t.active=1 AND p.status='included_verified_research'
        AND instr(t.payload,'publisher_affiliation_structure')>0
        AND EXISTS(SELECT 1 FROM springer_author_processed s WHERE s.work_id=p.work_id AND s.status='cache_processed')"""
    for wid, shard in db.execute(query):
        receipt = db.execute("SELECT receipt FROM springer_author_processed WHERE work_id=? AND status='cache_processed' ORDER BY rowid DESC LIMIT 1", (wid,)).fetchone()[0]
        groups[shard][wid] = receipt
    db.close()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    stats: Counter = Counter()
    jobs = []
    with (FOLDER / 'preparation_outcomes.jsonl').open('w') as out:
        for shard, selected in groups.items():
            cols = ['work_id', 'doi', 'title', 'journal_name', 'source_id', 'authors_json', 'raw_receipt']
            for paper in pq.read_table(ROOT / 'fields/papers' / shard, columns=cols).to_pylist():
                wid = paper['work_id']
                if wid not in selected:
                    continue
                try:
                    record, status = prepare(paper, selected[wid], journals[paper['source_id']])
                except (OSError, ValueError, KeyError, etree.XMLSyntaxError) as exc:
                    record, status = None, 'retained_source_parse_error:' + type(exc).__name__
                stats[status] += 1
                if record:
                    path = FOLDER / 'staged' / (hashlib.sha256(wid.encode()).hexdigest() + '.json')
                    atomic_json(path, record)
                    jobs.append((wid, shard, str(path.relative_to(ROOT))))
                    stats['confirmed_name_links'] += len(record['raw_name_links'])
                out.write(json.dumps(dict(work_id=wid, receipt=selected[wid], status=status)) + '\n')
            out.flush()
            atomic_json(FOLDER / 'status.json', dict(status='preparing_cache_only', counts=dict(stats), prepared_jobs=len(jobs)))
    result = dict(status='prepared', selected_papers=sum(map(len, groups.values())), counts=dict(stats), jobs=jobs,
                  network_requests=0, final_task_closure=False)
    atomic_json(FOLDER / 'scope.json', result)
    atomic_json(FOLDER / 'status.json', {k: v for k, v in result.items() if k != 'jobs'})
    return {k: v for k, v in result.items() if k != 'jobs'}


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
