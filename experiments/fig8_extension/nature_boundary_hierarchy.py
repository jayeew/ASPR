"""Evidence-ranked university rollups; a parent edge is not inferred from coauthorship."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .nature_boundary_sources import OUT, root_university_name
from .stage0 import write_csv, write_json

EDUCATION = {'university', 'education_proxy', 'education_other_or_unreviewed'}
HOSPITALS = ['Peking University First Hospital', "Peking University People's Hospital",
             'Peking University Third Hospital', 'Peking University Cancer Hospital',
             'Peking University School and Hospital of Stomatology', 'Peking University Sixth Hospital']


class Hierarchy:
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.entities = {s['review_entity_id']: s for s in entities}
        self.by_ror = {s['ror']: s for s in entities if s.get('ror')}
        self.registry = json.loads((OUT / 'ror_registry_index.json').read_text())
        self.manual = self.manual_rules()
        self._decisions: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def by_name(self, name: str) -> dict[str, Any]:
        found = [e for e in self.entities.values() if e['display_name'] == name]
        if len(found) != 1:
            raise ValueError(f'Ambiguous/missing entity {name}: {len(found)}')
        return found[0]

    def manual_rules(self) -> dict[str, Any]:
        rules = {}

        def add(name: str, kind: str, targets: list[str], url: str, note: str, divisor: int = 1) -> None:
            matching = [e for e in self.entities.values() if e['display_name'] == name]
            if not matching:
                return
            rules[matching[0]['review_entity_id']] = {'action': kind,
                'target_ids': [self.by_name(n)['review_entity_id'] for n in targets],
                'divisor': divisor, 'source_url': url, 'note': note,
                'evidence_level': 'official_primary_override'}

        cornell = ['Cornell University']
        add('Weill Cornell Medical College in Qatar', 'academic', cornell,
            'https://weill.cornell.edu/our-story/history', 'Cornell medical-school campus; retain QA worksite country.')
        add('New York State College of Veterinary Medicine', 'academic', cornell,
            'https://www.vet.cornell.edu/about-us/cvm-strategic-plans/strategic-plan-2023-2027/about-college',
            'Cornell contract college; SUNY umbrella is not a second campus assignment.')
        for name in HOSPITALS:
            add(name, 'health_system', ['Peking University'], 'https://www.bjmu.edu.cn/ylfw/zsfsyy/index.htm',
                'Official directly affiliated clinical school/hospital; not an ownership assertion.')
        add('Stanford Medicine', 'health_system', ['Stanford University'], 'https://med.stanford.edu/about/principles.html',
            'Integrated academic health system; mixed medical school and hospital umbrella.')
        add('Weill Cornell Medicine', 'academic', cornell, 'https://weill.cornell.edu/our-story/history',
            'Medical college and graduate school of Cornell; distinct from NYP hospital.')
        add('Cornell High Energy Synchrotron Source', 'academic', cornell,
            'https://research-and-innovation.cornell.edu/centers/', 'Listed Cornell research facility; grant funders are not university parents.')
        add('Cornell Feline Health Center', 'academic', cornell,
            'https://news.cornell.edu/stories/2024/03/cornell-feline-health-center-launches-playful-catgpt',
            'University news explicitly places the center in its College of Veterinary Medicine.')
        add('Koch Institute for Integrative Cancer Research At MIT', 'academic', ['Massachusetts Institute of Technology'],
            'https://ki.mit.edu/about', 'MIT interdisciplinary research institute, not a clinical hospital despite registry healthcare type.')
        for name in ['Weill Cornell Medical Center', 'NewYork-Presbyterian Hospital']:
            add(name, 'independent', [], 'https://www.nyp.org/locations/manhattan/nyp-weill-cornell-medical-center',
                'NYP hospital/teaching affiliation; not Cornell medical school.')
        add("Brigham and Women's Hospital", 'independent', [], 'https://identityguide.hms.harvard.edu/page-5',
            'Independent hospital; Harvard teaching affiliation and suspect ROR Tufts parent do not support rollup.')
        add('Whitehead Institute for Biomedical Research', 'independent', [],
            'https://wi.mit.edu/sites/default/files/2023-05/About_Whitehead_5-19-23.pdf',
            'Independent governance, finance and research despite teaching affiliation with MIT.')
        add('Munich Center for Quantum Science and Technology', 'joint',
            ['Ludwig-Maximilians-Universität München', 'Technical University of Munich'],
            'https://www.mcqst.de/about/ecosystem/', 'Five core partners; ROR single LMU parent incomplete.', 5)
        add('The University of Texas at San Antonio Health Science Center', 'temporal_unresolved', [],
            'https://uthscsa.edu/news/utsa-and-ut-health-san-antonio-complete-merger-become-university-texas-san-antonio',
            '2025 merger; retain publication identity for entire cohort pending record-level effective-date alignment.')
        # Academic units of federations remain separate; the rule applies to every entity below.
        return rules

    def decision(self, ident: str, trail: tuple[str, ...] = ()) -> dict[str, Any]:
        key = (ident, trail)
        if key not in self._decisions:
            self._decisions[key] = self._decision(ident, trail)
        return self._decisions[key]

    def _decision(self, ident: str, trail: tuple[str, ...]) -> dict[str, Any]:
        e = self.entities[ident]
        base = {'entity_id': ident, 'name': e['display_name'], 'ror': e.get('ror'), 'target_ids': [],
                'action': 'nonuniversity_or_parent_unresolved', 'evidence_level': 'inherited_identity', 'divisor': 1}
        if ident in self.manual:
            return {**base, **self.manual[ident]}
        if e['identity_status'] == 'identity_unresolved':
            return {**base, 'action': 'identity_unresolved'}
        if e.get('joint_university_targets'):
            return {**base, 'action': 'joint', 'target_ids': e['joint_university_targets'],
                    'divisor': e['joint_partner_n'], 'evidence_level': 'official_primary_override'}
        entry = self.registry.get(e.get('ror'), {})
        org = entry.get('organization', {})
        parents = [p for p in org.get('relationships', []) if p['type'].lower() == 'parent']
        base.update(parent_edges=parents, registry_receipt=entry.get('receipt'),
                    registry_status=org.get('status'), registry_evidence_available=bool(org))
        if re.search(r'^(Jet Propulsion Laboratory|SLAC National|Argonne National|Lawrence Berkeley National|Princeton Plasma Physics|Fermi National|Los Alamos National|Brookhaven National)', e['display_name']):
            return {**base, 'action': 'national_facility_parent_role_unresolved',
                    'note': 'Government/university operation or sponsorship is not evidence of a university subunit; excluded from joint split too.'}
        if ident in trail:
            return {**base, 'action': 'parent_cycle_unresolved'}
        # An autonomous university or an education institution in a federation is a counting root.
        federation = any(re.search(r'University of London$|University.*System|University System|Université de Lyon$',
                                  p['label'], re.IGNORECASE) for p in parents)
        if e['review_entity_type'] in EDUCATION and (root_university_name(e['display_name']) or federation):
            return {**base, 'action': 'university_root', 'target_ids': [ident]}
        if e['review_entity_type'] == 'company':
            return {**base, 'action': 'independent_company'}
        targets, inherited_health = [], False
        for parent in parents:
            pe = self.by_ror.get(parent['id'])
            if not pe:
                continue
            d = self.decision(pe['review_entity_id'], trail + (ident,))
            if d['action'] in ['university_root', 'academic', 'health_system']:
                targets.extend(d['target_ids'])
                inherited_health |= d['action'] == 'health_system'
        if targets:
            if len(parents) > 1:
                return {**base, 'action': 'joint', 'target_ids': sorted(set(targets)), 'divisor': len(parents),
                        'target_multiplicity': dict(Counter(targets)),
                        'evidence_level': 'registry_structural_screen'}
            health = e['review_entity_type'] == 'healthcare' or inherited_health
            return {**base, 'action': 'health_system' if health else 'academic',
                    'target_ids': sorted(set(targets)), 'evidence_level': 'registry_structural_screen'}
        if e['review_entity_type'] in EDUCATION:
            return {**base, 'action': 'standalone_education_proxy', 'target_ids': [ident],
                    'note': 'Education identity retained; degree-granting/root status not independently audited.'}
        return base

    def targets(self, slot: dict[str, Any], scenario: str) -> dict[str, float]:
        d = self.decision(slot['review_entity_id'])
        action = d['action']
        if action in ['university_root', 'standalone_education_proxy']:
            return {i: 1.0 for i in d['target_ids']}
        if action == 'temporal_unresolved':
            return {slot['review_entity_id']: 1.0} if slot['review_entity_type'] in EDUCATION else {}
        if action == 'academic' or (action == 'health_system' and scenario != 'academic_only'):
            return {i: 1.0 for i in d['target_ids']}
        if action == 'joint' and scenario == 'joint_equal_split':
            return {i: d.get('target_multiplicity', {}).get(i, 1) / d['divisor'] for i in d['target_ids']}
        return {}

    def save(self) -> None:
        decisions = [self.decision(i) for i in self.entities]
        write_json(OUT / 'hierarchy_decisions.json', decisions)
        write_csv(OUT / 'hierarchy_decisions.csv', decisions)
        write_json(OUT / 'hierarchy_manual_overrides.json', list(self.manual.values()))

