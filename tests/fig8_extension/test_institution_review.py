"""Regression tests for conservative identity matching and unknown weight retention."""
from typing import Any

from experiments.fig8_extension.nature_institution_apply import (
    author_slots,
    type_from_address,
)
from experiments.fig8_extension.nature_institution_review import exact_candidate


def organization(name: str, city: str, country: str = 'US') -> dict[str, Any]:
    return {'id': 'https://ror.org/example', 'names': [{'value': name, 'types': ['ror_display']}],
            'status': 'active', 'types': ['education'],
            'locations': [{'geonames_details': {'country_code': country, 'name': city}}]}


def test_city_name_cannot_be_an_institution_match() -> None:
    assert not exact_candidate('Allchemy, Highland, IN, USA', organization('Highland', 'Highland'))[0]


def test_prefix_name_collision_is_not_exact_identity() -> None:
    org = organization('Cyprus Institute', 'Nicosia', 'CY')
    assert not exact_candidate('Cyprus Institute of Neurology & Genetics, Nicosia, Cyprus', org)[0]


def test_branch_requires_evidence_but_exact_name_city_can_pass() -> None:
    org = organization('EpiCypher (United States)', 'The Woodlands')
    assert exact_candidate('EpiCypher, Durham, NC, USA', org)[1] == 'city_requires_review'
    assert exact_candidate('EpiCypher, The Woodlands, TX, USA', org)[0]


def test_partial_raw_affiliation_does_not_disappear() -> None:
    author = {'institutions': [{'id': 'I1', 'ror': None, 'display_name': 'University A', 'type': 'education', 'country_code': 'US'}],
              'affiliations': [{'institution_ids': [], 'raw_affiliation_string': 'Unknown Lab, Paris, France'}]}
    slots = author_slots(author, {}, True)
    assert len(slots) == 2
    assert sum(i['identity_status'] == 'identity_unresolved' for i in slots) == 1
    assert {i['country_code'] for i in slots} == {'US', 'FR'}


def test_local_identity_is_not_standard_registry() -> None:
    address = 'Example Company, Durham, NC, USA'
    from experiments.fig8_extension.nature_backfill_registry import normalized
    entity = {'review_entity_id': 'reviewed-local:example', 'identity_status': 'verified_local_no_registry_id',
              'review_entity_type': 'company', 'country_code': 'US'}
    author = {'institutions': [], 'backfill_affiliations': [{'address': address, 'resolved_institution_ids': []}]}
    result = author_slots(author, {normalized(address): {'entities': [entity]}}, True)
    assert result == [entity]
    assert result[0]['identity_status'] != 'standard_registry'


def test_type_only_never_claims_university_identity() -> None:
    assert type_from_address('Acme Inc., Boston, MA, USA')[0] == 'company'
    assert type_from_address('Independent researcher, Cambridge, UK')[0] == 'independent_researcher'
    assert type_from_address('University Hospital, City, UK')[0] == 'unknown'


def test_country_sensitivity_preserves_region_code_granularity() -> None:
    from experiments.fig8_extension.nature_institution_apply import country_sensitivity
    author = {'affiliations': [{'institution_ids': ['I1'], 'raw_affiliation_string': 'University, Hong Kong, China'}]}
    slots = [{'review_entity_id': 'I1', 'country_code': 'HK'}]
    assert country_sensitivity(author, slots) == slots


def test_country_sensitivity_is_separate_from_stored_registry_data() -> None:
    from experiments.fig8_extension.nature_institution_apply import country_sensitivity
    author = {'affiliations': [{'institution_ids': ['I1'], 'raw_affiliation_string': 'Company, Paris, France'}]}
    slots = [{'review_entity_id': 'I1', 'country_code': 'US'}]
    assert country_sensitivity(author, slots)[0]['country_code'] == 'FR'
    assert slots[0]['country_code'] == 'US'
