"""Scientific accounting regressions for university-family affiliation comparisons."""
from __future__ import annotations

import json
from typing import Any

import pytest

import experiments.fig8_extension.nature_boundary_review as review
from experiments.fig8_extension.nature_boundary_hierarchy import Hierarchy
from experiments.fig8_extension.nature_boundary_rank import calculate, university_shares
from experiments.fig8_extension.nature_boundary_sources import FIELD, SLOTS


def entity(ident: str, kind: str = 'education_proxy', country: str = 'US') -> dict[str, Any]:
    return {'review_entity_id': ident, 'display_name': ident, 'ror': ident,
            'review_entity_type': kind, 'country_code': country, 'identity_status': 'standard_registry'}


def hierarchy() -> Hierarchy:
    h = object.__new__(Hierarchy)
    es = [entity('Cornell University'), entity('Medical school', country='QA'),
          entity('Teaching hospital', 'healthcare'), entity('Company', 'company'), entity('Joint centre', 'research_institute')]
    h.entities = {e['review_entity_id']: e for e in es}
    h.by_ror = dict(h.entities)
    h.registry = {}
    h._decisions = {}
    h.manual = {
        'Medical school': {'action': 'academic', 'target_ids': ['Cornell University'], 'divisor': 1},
        'Teaching hospital': {'action': 'independent', 'target_ids': [], 'divisor': 1},
        'Joint centre': {'action': 'joint', 'target_ids': ['Cornell University'], 'divisor': 3}}
    return h


def test_medical_school_is_not_teaching_hospital() -> None:
    h = hierarchy()
    assert h.targets(h.entities['Medical school'], 'direct_units') == {'Cornell University': 1}
    assert not h.targets(h.entities['Teaching hospital'], 'direct_units')


def test_joint_centre_does_not_assign_every_partner_share_to_university() -> None:
    h = hierarchy()
    slot = h.entities['Joint centre']
    assert h.targets(slot, 'direct_units') == {}
    assert h.targets(slot, 'joint_equal_split') == {'Cornell University': pytest.approx(1 / 3)}


def test_overseas_school_keeps_publication_country_after_rollup() -> None:
    h = hierarchy()
    row = {FIELD: json.dumps([{'author_record_kind': 'person', SLOTS: [h.entities['Medical school']]}])}
    result = calculate([row], h, 'direct_units')
    assert result['countries'] == {'QA': 1}
    assert result['universities'] == {'Cornell University': 1}


def test_unknown_mass_is_not_redistributed_to_known_university() -> None:
    h = hierarchy()
    missing = entity('Unknown', 'unknown')
    missing['identity_status'] = 'identity_unresolved'
    h.entities['Unknown'] = missing
    slots = [h.entities['Cornell University'], missing]
    assert university_shares(slots, h, 'direct_units') == {'Cornell University': .5}


def test_rollup_and_deduplicated_denominators_are_distinct_explicit_scenarios() -> None:
    h = hierarchy()
    slots = [h.entities[k] for k in ['Cornell University', 'Medical school', 'Company']]
    assert university_shares(slots, h, 'direct_units')['Cornell University'] == pytest.approx(2 / 3)
    assert university_shares(slots, h, 'university_deduplicated')['Cornell University'] == .5


def test_autonomous_university_is_not_merged_into_federation() -> None:
    h = hierarchy()
    ucl, london = entity('University College London'), entity('University of London')
    for e in [ucl, london]:
        h.entities[e['review_entity_id']] = e
        h.by_ror[e['ror']] = e
    h.registry[ucl['ror']] = {'organization': {'relationships': [
        {'type': 'parent', 'id': london['ror'], 'label': london['display_name']}]}}
    assert h.targets(ucl, 'direct_units') == {'University College London': 1}


def test_company_does_not_inherit_university_from_parent_edge() -> None:
    h = hierarchy()
    h.registry['Company'] = {'organization': {'relationships': [
        {'type': 'parent', 'id': 'Cornell University', 'label': 'Cornell University'}]}}
    assert not h.targets(h.entities['Company'], 'direct_units')
    national = entity('Jet Propulsion Laboratory', 'research_institute')
    h.entities[national['review_entity_id']] = national
    h.registry[national['ror']] = h.registry['Company']
    assert not h.targets(national, 'joint_equal_split')


def test_email_domain_does_not_link_different_publisher_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review, 'author_addresses', lambda a, p: (['School of Physics, Peking University, Beijing, China'],
                                                                'publisher_exact_author_name'))
    matches, _ = review.matching_addresses({}, {}, 'Tsientang Institute, Hangzhou, China. pgao@pku.edu.cn',
        {'pattern': None, 'entity': {'country_code': 'CN'}})
    assert matches == []
