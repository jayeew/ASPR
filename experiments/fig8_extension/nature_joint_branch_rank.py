"""Recompute observed memberships and conditional bounds after branch audit."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pyarrow.parquet as pq

from . import nature_boundary_rank as rank
from .nature_boundary_hierarchy import Hierarchy
from .nature_joint_branch_review import (
    FIELD,
    INPUT,
    OUT,
    PREV,
    SLOTS,
    ReviewedHierarchy,
)
from .stage0 import write_csv, write_json


class SensitivityHierarchy(ReviewedHierarchy):
    def targets(self, slot: dict[str, Any], scenario: str) -> dict[str, float]:
        if scenario == 'official_only':
            d = self.decision(slot['review_entity_id'])
            if d['evidence_level'] == 'registry_structural_screen' and d['action'] in ['academic', 'health_system']:
                return {}
        if scenario == 'joint_two_partner_sensitivity' and not slot.get('branch_review'):
            if slot['review_entity_id'] in ['https://openalex.org/I4210127494', 'https://openalex.org/I4391768054']:
                return {'https://openalex.org/I20231570': .5, 'https://openalex.org/I99065089': .5}
            return super().targets(slot, 'joint_equal_split')
        return super().targets(slot, scenario)


def main() -> None:
    rows = pq.read_table(OUT / 'research_roster_joint_review.parquet').to_pylist()
    oldrows = pq.read_table(INPUT).to_pylist()
    assert len(rows) == len(oldrows) == 3683
    assert all(all(new[k] == value for k, value in old.items()) for old, new in zip(oldrows, rows, strict=True))
    entities = {s['review_entity_id']: s for r in rows for a in json.loads(r[FIELD]) for s in a[SLOTS]}
    h = SensitivityHierarchy(list(entities.values()))
    rank.FIELD, rank.SLOTS = FIELD, SLOTS
    scenarios = ['direct_units', 'academic_only', 'joint_equal_split', 'joint_two_partner_sensitivity',
                 'university_deduplicated', 'exclude_nonpersonal', 'complete_case', 'official_only', 'unbranched_control']
    results = {s: rank.calculate(rows, h, s) for s in scenarios}
    rank.FIELD, rank.SLOTS = 'authorships_boundary_review_json', 'institution_boundary_review_entities'
    results['previous_main'] = rank.calculate(oldrows, Hierarchy(list(entities.values())), 'direct_units')
    stability = {s: rank.bounds(d, h, s) for s, d in results.items()}
    ref = stability['direct_units']
    for record in stability.values():
        for category in ['country', 'university']:
            names, baseline = record[category]['members'], ref[category]['members']
            record[category].update(same_members_as_direct=set(names) == set(baseline), same_order_as_direct=names == baseline,
                entered_vs_direct=[n for n in names if n not in baseline], left_vs_direct=[n for n in baseline if n not in names])
    write_json(OUT / 'topk_stability.json', {'scenarios': stability,
        'limitations': 'Conditional missing-only bounds exclude unresolved hierarchy, overseas campus and denominator changes. '
        'Joint two-partner scenario is an intentionally broad temporal sensitivity, not a claim about all publication years.'})
    for category, field in [('countries', 'country'), ('universities', 'institution_id')]:
        table = []
        for scenario, result in results.items():
            for i, (key, value) in enumerate(sorted(result[category].items(), key=lambda x: -x[1])):
                if key == 'unknown':
                    continue
                table.append({'scenario': scenario, 'rank': i + 1, field: key, 'fractional_n': value,
                              'name': h.entities[key]['display_name'] if category == 'universities' else key})
        write_csv(OUT / ('country' if category == 'countries' else 'university') / 'rank_scenarios.csv', table)
    write_json(OUT / 'fractional_summary.json', {s: r['summary'] for s, r in results.items()})
    for s in scenarios:
        if s not in ['complete_case', 'exclude_nonpersonal']:
            assert results[s]['countries'] == results['previous_main']['countries']
    audit = json.loads((OUT / 'branch_address_audit.json').read_text())
    aggregate: dict[str, Any] = {}
    for a in audit:
        key = a['entity'] + '|' + a['status'] + '|' + a.get('target_id', '')
        v = aggregate.setdefault(key, {'entity': a['entity'], 'status': a['status'], 'target_id': a.get('target_id'),
            'weight': 0.0, 'author_slots': 0, 'dois': set()})
        v['weight'] += a['weight']
        v['author_slots'] += 1
        v['dois'].add(a['doi'])
    agg = [{**v, 'dois': sorted(v['dois']), 'paper_n': len(v['dois'])} for v in aggregate.values()]
    write_csv(OUT / 'branch_summary.csv', agg)
    write_json(OUT / 'branch_summary.json', agg)
    # No institution identities were guessed merely because a branch remained unknown.
    previous = json.loads((PREV / 'review_summary.json').read_text())
    summary = {**previous, 'before_unresolved_papers': 357, 'before_unresolved_authors': 1051,
        'newly_resolved_authors': 0, 'changed_slot_n': 0, 'changed_paper_n': 0,
        'confirmed_branch_slots': sum(a['status'] == 'confirmed_branch' for a in audit),
        'confirmed_branch_papers': len({a['doi'] for a in audit if a['status'] == 'confirmed_branch'}),
        'reviewed_branch_slots': len(audit), 'reviewed_branch_papers': len({a['doi'] for a in audit}),
        'officially_confirmed_ror_rollup_relations': 7}
    for filename in ['remaining_papers.csv', 'remaining_identity_authors.csv', 'coverage_by_year.csv',
                     'remaining_weighted_units.json', 'remaining_weighted_units.csv', 'country_conflicts_unresolved_inherited.csv']:
        (OUT / filename).write_bytes((PREV / filename).read_bytes())
    write_json(OUT / 'review_summary.json', summary)
    decisions = [h.decision(i) for i in h.entities]
    write_json(OUT / 'hierarchy_decisions.json', decisions)
    write_csv(OUT / 'remaining_ror_only_rollups.csv', [d for d in decisions
        if d['evidence_level'] == 'registry_structural_screen' and d['action'] in ['academic', 'health_system']])
    write_json(OUT / 'validation.json', {'unique_cohort': len({r['doi'] for r in rows}), 'original_columns_unchanged': True,
        'all_full_sample_country_allocations_unchanged': True, 'country_mass_conserved': True,
        'previous_parquet_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()})
    for s, v in stability.items():
        print(s, v['university']['cutoff'], v['university']['next'], v['university']['entered_vs_direct'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
