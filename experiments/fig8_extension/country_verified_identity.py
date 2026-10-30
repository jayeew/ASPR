"""Reuse source-bound raw-name and ORCID proofs for exact country-slot fills."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from functools import lru_cache

from .cureus_country_apply import enrich as exact_enrich
from .fig8b_qualification_build import key
from .orcid_cached_links import canonical_orcid
from .supplement_queue import ROOT


@lru_cache(maxsize=8)
def raw_authorships(receipt: str, digest: str, work_id: str, doi: str) -> list[dict]:
    raw = gzip.decompress((ROOT / receipt).read_bytes())
    if hashlib.sha256(raw).hexdigest() != digest:
        return []
    response = json.loads(raw).get('response', {})
    records = response.get('results', [response])
    matches = [r for r in records if r.get('id') == work_id and
               (r.get('doi') or '').removeprefix('https://doi.org/').lower() == doi]
    return matches[0].get('authorships', []) if len(matches) == 1 else []


def match(author: dict, authors: list[dict], source: dict, receipt: str) -> dict | None:
    native = source.get('authors', [])
    if Counter(key(a['name']) for a in authors)[key(author['name'])] != 1:
        return None
    kind = author.get('publisher_author_identity_match')
    if kind == 'unique_checksum_valid_orcid_in_both_paper_bylines':
        identifier = canonical_orcid(author.get('corroborated_orcid', ''))
        if not identifier or sum(canonical_orcid(a.get('corroborated_orcid', '')) == identifier for a in authors) != 1:
            return None
        matches = [a for a in native if canonical_orcid(a.get('orcid', '')) == identifier]
    elif kind == 'unique_publisher_raw_name_and_original_author_id':
        proof = author.get('publisher_author_identity_evidence') or {}
        if proof.get('author_id') != author.get('author_id') or proof.get('canonical_name') != author['name']:
            return None
        if (ROOT / 'fields/qualification' / proof['publisher_raw_receipt']).resolve() != (ROOT / receipt).resolve():
            return None
        if proof['doi'] not in source.get('doi', []):
            return None
        original = raw_authorships(proof['openalex_receipt'], proof['openalex_raw_sha256'], proof['work_id'], proof['doi'])
        linked = [a for a in original if a.get('author', {}).get('id') == author.get('author_id')
                  and key(a['author'].get('display_name', '')) == key(author['name'])]
        if len(linked) != 1 or key(linked[0].get('raw_author_name', '')) != key(proof['raw_author_name']):
            return None
        if sum(key(a.get('raw_author_name', '')) == key(proof['raw_author_name']) for a in original) != 1:
            return None
        matches = [a for a in native if key(a['name']) == key(proof['raw_author_name'])]
    else:
        return None
    if len(matches) != 1 or matches[0].get('unresolved_affiliation_refs') or matches[0].get('nonpersonal'):
        return None
    return matches[0]


def enrich(authors: list[dict], source: dict, receipt: str, rules: dict[str, str],
           rule_receipt: str, basis: str) -> tuple[list[dict], int, int]:
    result = list(authors)
    slots = people = 0
    for i, author in enumerate(authors):
        if author.get('nature_v1_slots') is not None or author.get('publisher_affiliation_source') != receipt:
            continue
        native = match(author, authors, source, receipt)
        if native is None:
            continue
        adapted = {**source, 'authors': [{**native, 'name': author['name']}]}
        values, filled, persons = exact_enrich([author], adapted, receipt, rules, rule_receipt, basis)
        if filled:
            for old, new in zip(author['publisher_affiliation_slots'], values[0]['publisher_affiliation_slots']):
                if not old.get('country_codes') and new.get('country_codes'):
                    new['country_evidence'].update(native_author_name=native['name'],
                        identity_method=author['publisher_author_identity_match'])
        result[i] = values[0]
        slots += filled
        people += persons
    return result, slots, people
