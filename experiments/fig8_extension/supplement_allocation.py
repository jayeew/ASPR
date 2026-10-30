"""Conservative Nature-compatible fractional allocations with explicit unknown mass."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SCENARIOS = ['main_personal_direct', 'legacy_observed_direct', 'academic_only', 'university_deduplicated',
             'joint_equal_split', 'official_only', 'publisher_verified_clinical', 'complete_case']


def known_universities(root: Path) -> dict[str, set[str]]:
    known = {name: set() for name in SCENARIOS}
    for line in (root / 'nature/papers.jsonl').open():
        for author in json.loads(line)['authors']:
            for slot in author['slots']:
                for scenario in SCENARIOS:
                    rule = scenario if scenario in ['academic_only', 'joint_equal_split', 'official_only', 'publisher_verified_clinical'] else 'direct_units'
                    if slot['university_targets'].get(rule) == {slot['entity_id']: 1.0}:
                        known[scenario].add(slot['entity_id'])
    return known


def has_official_reference(value: object) -> bool:
    if isinstance(value,dict):
        return any((k=='official_url' and isinstance(v,str) and v.startswith('https://')) or has_official_reference(v) for k,v in value.items())
    if isinstance(value,list):return any(has_official_reference(v) for v in value)
    return False


def official_relationships(root: Path) -> dict:
    """Only consistent officially supported Nature relationships, with branch/site guards."""
    names=json.loads((root/'nature/entity_names.json').read_text())
    rules={};conflicts=set()
    for line in (root/'nature/papers.jsonl').open():
        for author in json.loads(line)['authors']:
            for slot in author['slots']:
                main=slot['university_targets'].get('direct_units',{})
                if not main or main=={slot['entity_id']:1.0} or slot['university_targets'].get('official_only')!=main:continue
                if not has_official_reference(slot.get('source_fields',{})) or not slot.get('country_codes'):continue
                ident=slot['entity_id'];signature=json.dumps(slot['university_targets'],sort_keys=True)
                if ident in rules and rules[ident]['signature']!=signature:conflicts.add(ident);continue
                rules[ident]={'signature':signature,'targets':slot['university_targets'],'countries':slot['country_codes'],
                    'parent_names':{k:names.get(k,'') for k in main},'source_fields':slot['source_fields']}
    return {k:v for k,v in rules.items() if k not in conflicts}


def declared_relationship(slot: dict, scenario: str, relationships: dict) -> dict:
    rule=scenario if scenario in ['academic_only','joint_equal_split','official_only','publisher_verified_clinical'] else 'direct_units'
    candidates=[]
    for ident in slot.get('institution_ids',[]):
        relation=relationships.get(ident)
        if not relation or not slot.get('country_codes') or not set(slot['country_codes'])<=set(relation['countries']):continue
        normalized=''.join(c for c in slot.get('text','').casefold() if c.isalnum())
        if not all(parent in slot.get('institution_ids',[]) or (name and ''.join(c for c in name.casefold() if c.isalnum()) in normalized) for parent,name in relation['parent_names'].items()):continue
        candidates.append(relation['targets'].get(rule,{}))
    variants={json.dumps(v,sort_keys=True) for v in candidates if v}
    return json.loads(next(iter(variants))) if len(variants)==1 else {}


def slots_for(author: dict, scenario: str, known: dict[str, set[str]], relationships: dict | None = None) -> list[dict]:
    original = author.get('nature_v1_slots')
    if original:
        rule = scenario if scenario in ['academic_only', 'joint_equal_split', 'official_only', 'publisher_verified_clinical'] else 'direct_units'
        return [{**s, 'targets': s['university_targets'].get(rule, {}),
                 'unknown_university': s.get('identity_missing', True) or s.get('final_status', '').startswith('unresolved'),
                 'country_codes': [] if author.get('country_conflict_retained') else s.get('country_codes', [])} for s in original]
    publisher = author.get('publisher_affiliation_slots')
    if not publisher:
        return [{'entity_id': '', 'targets': {}, 'identity_missing': True, 'unknown_university': True, 'country_codes': [], 'structure_missing': True}]
    slots = []
    for slot in publisher:
        ids = slot.get('institution_ids', [])
        own = [i for i in ids if i in known[scenario]]
        targets = {own[0]: 1.0} if len(own) == len(ids) == 1 else {}
        if not targets and len(ids)==1 and slot.get('university_ownership_status')=='officially_confirmed_university_self' and slot.get('official_identity_evidence'):
            targets={ids[0]:1.0}
        if (not targets and len(ids)==1
                and slot.get('university_ownership_status')=='officially_confirmed_university_unit'
                and slot.get('official_identity_evidence')):
            parent_targets=slot.get('official_university_targets', {})
            if len(parent_targets)==1 and list(parent_targets.values())==[1.0]:
                targets=dict(parent_targets)
        if not targets and relationships:
            declared=declared_relationship(slot,scenario,relationships)
            if declared and (not own or set(declared)==set(own)):targets=declared
        slots.append({'entity_id': ids[0] if len(ids) == 1 else '', 'targets': targets,
                      'identity_missing': slot.get('identity_status') == 'unresolved',
                      'unknown_university': bool(author.get('official_identity_source_conflict')) or (not bool(targets) and not (len(ids)==1 and slot.get('university_ownership_status')=='officially_confirmed_nonuniversity' and bool(slot.get('official_identity_evidence')))),
                      'country_codes': [] if author.get('country_conflict_retained') else slot.get('country_codes', [])})
    return slots


def allocate(paper: dict, scenario: str, known: dict[str, set[str]], relationships: dict | None = None,
             parsed_authors: list[dict] | None = None) -> tuple[Counter, Counter, Counter]:
    authors = json.loads(paper['authors_json']) if parsed_authors is None else parsed_authors
    if scenario != 'legacy_observed_direct':
        authors = [a for a in authors if not a.get('nonpersonal_verified')]
    countries, universities, stats = Counter(), Counter(), Counter()
    if not authors:
        stats['no_personal_author_denominator_papers'] += 1
        if scenario=='complete_case':stats['complete_case_excluded_papers']+=1
        else:
            countries['unknown']=1;universities['unknown']=1;stats['allocated_papers']+=1
        return countries, universities, stats
    prepared = [slots_for(a, scenario, known,relationships) for a in authors]
    if scenario == 'complete_case' and any(s['identity_missing'] for slots in prepared for s in slots):
        stats['complete_case_excluded_papers'] += 1
        return countries, universities, stats
    stats['allocated_papers'] += 1
    for slots in prepared:
        stats['affiliation_structure_unverified_author_records'] += any(s.get('structure_missing') for s in slots)
        stats['publisher_or_nature_linked_author_records'] += not any(s.get('structure_missing') for s in slots)
        for slot in slots:
            stats['observed_affiliation_slots'] += not slot.get('structure_missing', False)
            stats['identity_unresolved_slots'] += slot['identity_missing']
            stats['country_unresolved_slots'] += not slot['country_codes']
            codes = slot['country_codes'] or ['unknown']
            for code in codes:
                countries[code] += 1 / len(authors) / len(slots) / len(codes)
        maps = slots
        if scenario == 'university_deduplicated':
            groups = {}
            for index, slot in enumerate(slots):
                target = next(iter(slot['targets'])) if len(slot['targets']) == 1 else slot['entity_id'] or f'unknown-slot-{index}'
                groups[target] = slot
            maps = list(groups.values())
        for slot in maps:
            for ident, fraction in slot['targets'].items():
                universities[ident] += fraction / len(authors) / len(maps)
            if slot['unknown_university']:
                universities['unknown'] += max(0, 1 - sum(slot['targets'].values())) / len(authors) / len(maps)
    assert abs(sum(countries.values()) - 1) < 1e-8
    assert sum(universities.values()) <= 1 + 1e-8
    return countries, universities, stats
