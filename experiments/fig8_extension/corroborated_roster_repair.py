"""Reconcile complete bylines while retaining uniquely corroborated primary authors."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from .fig8b_qualification_build import key
from .orcid_cached_links import canonical_orcid
from .supplement_apply import complete_byline


def reconcile(current: list[dict], source: dict) -> tuple[list[dict], str]:
    native = source.get('authors', [])
    declared = Counter(key(a['name']) for a in native)
    if not native or source.get('byline_has_etal') or '' in declared or any(n != 1 for n in declared.values()):
        return current, 'native_byline_incomplete_or_ambiguous'
    if any(a.get('nature_v1_slots') is not None for a in current):
        return current, 'nature_preserved'
    if any(a.get('nonpersonal') for a in native):
        return current, 'collective_roster_requires_separate_review'
    mapping = defaultdict(list)
    current_orcids = Counter(canonical_orcid(a.get('corroborated_orcid', '')) for a in current)
    for author in current:
        if not author.get('publisher_affiliation_slots'):
            continue
        if author.get('publisher_affiliation_source') != source['evidence_receipt']:
            return current, 'existing_primary_source_conflict'
        identifier = canonical_orcid(author.get('corroborated_orcid', ''))
        if identifier:
            matches = [a for a in native if canonical_orcid(a.get('orcid', '')) == identifier]
            if current_orcids[identifier] != 1:
                return current, 'ambiguous_current_orcid'
        else:
            matches = [a for a in native if key(a['name']) == key(author['name'])]
        if len(matches) != 1:
            return current, 'existing_primary_identity_conflict'
        mapping[key(matches[0]['name'])].append(author)
    if any(len(records) != 1 for records in mapping.values()):
        return current, 'multiple_primary_records_for_native_author'
    fresh, complete = complete_byline([], source)
    if not complete:
        return current, 'native_byline_incomplete_or_ambiguous'
    counts = Counter(key(a['name']) for a in current)
    inherited = {key(a['name']): a for a in current if counts[key(a['name'])] == 1
                 and not a.get('publisher_affiliation_slots')}
    output = []
    for position, author in enumerate(fresh):
        candidates = mapping.get(key(author['name']))
        retained = candidates[0] if candidates else inherited.get(key(author['name']), author)
        record = json.loads(json.dumps(retained))
        record['position'] = position
        output.append(record)
    return output, 'publisher_roster_reconciled'
