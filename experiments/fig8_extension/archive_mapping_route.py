"""Keep complete native bylines in entity review rather than reacquiring their text."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq

from .supplement_apply import abstract_ok
from .supplement_queue import ROOT


def mapping_only(paper: dict) -> bool:
    if (paper.get('full_authorships_status') != 'publisher_byline_reconstructed_unique_exact_links'
            or paper.get('author_list_may_be_truncated') or not abstract_ok(paper.get('abstract') or '')
            or not paper.get('authorships_source_receipt')):
        return False
    authors = json.loads(paper['authors_json'])
    if not authors or len({a['name'].strip().casefold() for a in authors}) != len(authors):
        return False
    for author in authors:
        if author.get('nature_v1_slots') is not None:
            return False
        if author.get('country_conflict_retained') or author.get('secondary_country_conflict'):
            return False
        if author.get('publisher_author_nonpersonal'):
            continue
        slots = author.get('publisher_affiliation_slots') or []
        if (not slots or not author.get('publisher_affiliation_source')
                or author.get('publisher_author_identity_match') != 'unique_exact_name_within_matched_paper'):
            return False
        if any(not s.get('text') or not s.get('country_codes') for s in slots):
            return False
        countries = {c for s in slots for c in s['country_codes']}
        if countries != set(author.get('countries') or []):
            return False
    return True


def assess(jobs: list[tuple]) -> dict[str, dict]:
    grouped = defaultdict(set)
    for _, links in jobs:
        for paper in links:
            if paper['eligibility_status'] == 'included_verified_research':
                grouped[paper['shard']].add(paper['work_id'])
    proofs = {}
    columns = ['work_id', 'doi', 'authors_json', 'full_authorships_status',
               'author_list_may_be_truncated', 'authorships_source_receipt', 'abstract']
    for shard, ids in grouped.items():
        rows = pq.read_table(ROOT / 'fields/papers' / shard, columns=columns,
                            filters=[('work_id', 'in', sorted(ids))]).to_pylist()
        for paper in rows:
            if not mapping_only(paper):
                continue
            authors = json.loads(paper['authors_json'])
            receipts = {paper['authorships_source_receipt']} | {
                a['publisher_affiliation_source'] for a in authors if a.get('publisher_affiliation_source')}
            if not all((ROOT / receipt).is_file() for receipt in receipts):
                continue
            proofs[paper['work_id']] = {'doi': paper['doi'], 'shard': shard,
                'authors_sha256': hashlib.sha256(paper['authors_json'].encode()).hexdigest(),
                'existing_native_receipts': sorted(receipts),
                'reason': 'Native complete unique byline and explicit affiliation slots retained; abstract and slot countries present without country conflict. Remaining entity/university adjudication stays open.',
                'final_task_closure': False}
    return {pmcid: {p['work_id']: proofs[p['work_id']] for p in links}
            for pmcid, links in jobs if links and all(
                p['eligibility_status'] == 'included_verified_research'
                and p['work_id'] in proofs and p['doi'] == proofs[p['work_id']]['doi'] for p in links)}
