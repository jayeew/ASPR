"""Compare identity repairs, university definitions and missing-affiliation bounds."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import pyarrow.parquet as pq

from .nature_boundary_hierarchy import EDUCATION, Hierarchy
from .nature_boundary_sources import FIELD, OLD_FIELD, OLD_SLOTS, OUT, SLOTS
from .stage0 import write_csv, write_json

SCENARIOS = ['before', 'identity_review', 'academic_only', 'direct_units', 'joint_equal_split',
             'university_deduplicated', 'exclude_nonpersonal', 'complete_case']
NONPERSONAL = {'group_subdivision_heading', 'collective_author', 'acknowledged_consortium_not_byline'}


def university_shares(slots: list[dict[str, Any]], hierarchy: Hierarchy, scenario: str) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    if scenario == 'university_deduplicated':
        units = {}
        for slot in slots:
            targets = hierarchy.targets(slot, 'direct_units')
            key = next(iter(targets)) if len(targets) == 1 else slot['review_entity_id']
            units[key] = targets
        for targets in units.values():
            for key, fraction in targets.items():
                result[key] += fraction / len(units)
        return result
    for slot in slots:
        targets = ({slot['review_entity_id']: 1.0} if slot['review_entity_type'] in EDUCATION else {}) if scenario in [
            'before', 'identity_review'] else hierarchy.targets(slot, scenario)
        for key, fraction in targets.items():
            result[key] += fraction / len(slots)
    assert sum(result.values()) <= 1 + 1e-9
    return result


def calculate(rows: list[dict[str, Any]], hierarchy: Hierarchy, scenario: str) -> dict[str, Any]:
    cc, uu, unknown_by_country, actions = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    totals: Counter[str] = Counter()
    units: dict[str, Any] = {}
    for row in rows:
        field, key = (OLD_FIELD, OLD_SLOTS) if scenario == 'before' else (FIELD, SLOTS)
        authors = json.loads(row[field])
        if scenario == 'complete_case' and any(s['identity_status'] == 'identity_unresolved' for a in authors for s in a[key]):
            continue
        if scenario == 'exclude_nonpersonal':
            authors = [a for a in authors if a['author_record_kind'] not in NONPERSONAL]
        if not authors:
            raise ValueError('Empty authorship denominator')
        totals['paper_n'] += 1
        for a in authors:
            slots = a[key]
            shares = university_shares(slots, hierarchy, scenario)
            for ident, fraction in shares.items():
                uu[ident] += fraction / len(authors)
            for s in slots:
                weight = 1 / len(authors) / len(slots)
                codes = s.get('site_country_codes') or [s.get('country_code') or 'unknown']
                for code in codes:
                    cc[code] += weight / len(codes)
                if scenario != 'before':
                    d = hierarchy.decision(s['review_entity_id'])
                    actions[d['action']] += weight
                    if d['evidence_level'] == 'registry_structural_screen' and d['action'] in ['academic', 'health_system']:
                        totals['registry_only_rollup_mass'] += weight
                if s['identity_status'] == 'identity_unresolved':
                    totals['identity_unknown'] += weight
                    if s['review_entity_type'] == 'unknown':
                        totals['possible_university_unknown'] += weight
                        for code in codes:
                            unknown_by_country[code] += weight / len(codes)
                    u = units.setdefault(s['display_name'], {'address': s['display_name'], 'weight': 0.0, 'dois': set(),
                        'author_n': 0, 'entity_type': s['review_entity_type'], 'reasons': set()})
                    u['weight'] += weight
                    u['dois'].add(row['doi'])
                    u['author_n'] += 1
                    u['reasons'].add(s.get('unresolved_reason', 'identity_or_hierarchy_not_confirmed'))
    assert abs(sum(cc.values()) - totals['paper_n']) < 1e-7
    return {'countries': dict(cc), 'universities': dict(uu), 'summary': dict(totals,
        country_mass=sum(cc.values()), country_unknown=cc.get('unknown', 0), education_mass=sum(uu.values()),
        university_unknown_by_country=dict(unknown_by_country), hierarchy_action_mass=dict(actions)), 'units': units}


def bounds(data: dict[str, Any], hierarchy: Hierarchy, scenario: str) -> dict[str, Any]:
    c = sorted([(k, v) for k, v in data['countries'].items() if k != 'unknown'], key=lambda x: -x[1])
    u = sorted(data['universities'].items(), key=lambda x: -x[1])
    unknown = data['summary']['country_unknown']
    cg = c[11][1] - c[12][1]
    masses = data['summary']['university_unknown_by_country']
    upper = {k: v + masses.get(hierarchy.entities[k].get('country_code'), 0) + masses.get('unknown', 0) for k, v in u}
    unseen = [v + masses.get('unknown', 0) for k, v in masses.items() if k != 'unknown']
    max_outside = max([upper[k] for k, _ in u[15:]] + unseen + [masses.get('unknown', 0)])
    return {'scenario': scenario, 'country': {'members': [k for k, _ in c[:12]], 'cutoff': c[11], 'next': c[12],
        'gap': cg, 'unknown_mass': unknown, 'fixed_slot_missing_only_top12_guaranteed': cg > unknown,
        'guaranteed_current_members': [k for k, v in c[:12] if v > c[12][1] + unknown]},
        'university': {'members': [hierarchy.entities[k]['display_name'] for k, _ in u[:15]],
            'cutoff': [hierarchy.entities[u[14][0]]['display_name'], u[14][1]],
            'next': [hierarchy.entities[u[15][0]]['display_name'], u[15][1]], 'gap': u[14][1] - u[15][1],
            'identity_unknown_mass': data['summary'].get('possible_university_unknown', 0),
            'fixed_mapping_country_constrained_guaranteed_current_members': [hierarchy.entities[k]['display_name']
                for k, v in u[:15] if v > max_outside],
            'bound_excludes_hierarchy_uncertainty': True}}


def main() -> None:
    rows = pq.read_table(OUT / 'research_roster_boundary_review.parquet').to_pylist()
    entities = {s['review_entity_id']: s for r in rows for a in json.loads(r[FIELD]) for s in a[SLOTS]}
    # Keep previous identities available for the unmodified baseline scenario.
    for row in rows:
        for a in json.loads(row[OLD_FIELD]):
            for s in a[OLD_SLOTS]:
                entities.setdefault(s['review_entity_id'], s)
    hierarchy = Hierarchy(list(entities.values()))
    hierarchy.save()
    results = {s: calculate(rows, hierarchy, s) for s in SCENARIOS}
    countries, universities, stability = [], [], {}
    for scenario, data in results.items():
        countries.extend({'scenario': scenario, 'rank': i + 1, 'country': k, 'fractional_n': v} for i, (k, v) in
            enumerate(sorted(((k, v) for k, v in data['countries'].items() if k != 'unknown'), key=lambda x: -x[1])))
        universities.extend({'scenario': scenario, 'rank': i + 1, 'institution_id': k,
            'name': hierarchy.entities[k]['display_name'], 'fractional_n': v} for i, (k, v) in
            enumerate(sorted(data['universities'].items(), key=lambda x: -x[1])))
        stability[scenario] = bounds(data, hierarchy, scenario)
    base = stability['direct_units']
    for scenario, record in stability.items():
        for category in ['country', 'university']:
            names, ref = record[category]['members'], base[category]['members']
            record[category].update(same_members_as_direct=set(names) == set(ref), same_order_as_direct=names == ref,
                entered_vs_direct=[n for n in names if n not in ref], left_vs_direct=[n for n in ref if n not in names])
    write_json(OUT / 'topk_stability.json', {'scenarios': stability,
        'scope': 'Observed scenario comparisons, not confidence intervals. Bounds condition on fixed existing mappings and '
                 'author/entity denominators, and university country matching worksite country. Parent uncertainty, overseas '
                 'campuses and future deduplication are excluded; no unconditional membership guarantee.'})
    write_csv(OUT / 'country_rank_scenarios.csv', countries)
    write_csv(OUT / 'university_rank_scenarios.csv', universities)
    write_json(OUT / 'fractional_summary.json', {s: d['summary'] for s, d in results.items()})
    units = []
    for unit in sorted(results['direct_units']['units'].values(), key=lambda x: -x['weight']):
        units.append({**unit, 'dois': sorted(unit['dois']), 'paper_n': len(unit['dois']), 'reasons': sorted(unit['reasons'])})
    write_json(OUT / 'remaining_weighted_units.json', units)
    write_csv(OUT / 'remaining_weighted_units.csv', units)
    # University grouping must never change a publication's country allocation.
    reference = results['identity_review']['countries']
    for scenario in ['academic_only', 'direct_units', 'joint_equal_split', 'university_deduplicated']:
        assert results[scenario]['countries'] == reference
    write_json(OUT / 'ranking_validation.json', {'passed': True, 'all_scenarios_country_mass_conserved': True,
        'country_allocation_invariant_to_university_grouping': True, 'university_share_per_author_at_most_one': True})
    for s in SCENARIOS:
        print(s, stability[s]['university']['cutoff'], stability[s]['university']['next'],
              stability[s]['university']['entered_vs_direct'], stability[s]['university']['left_vs_direct'])


if __name__ == '__main__':
    main()
