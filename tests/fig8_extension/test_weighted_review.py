"""Checks for the material identity/country accounting decisions in round two."""
from __future__ import annotations

import pytest

import experiments.fig8_extension.nature_weighted_review as review
from experiments.fig8_extension.nature_weighted_review import apply_country, deduplicate, site_country


def test_country_site_preserves_registry_headquarters() -> None:
    slot = {'review_entity_id': 'NUS', 'country_code': 'SG'}
    result = apply_country(slot, {'action': 'site_country_verified', 'sites': ['CN', 'SG']})
    assert result['country_code'] == 'SG'
    assert result['site_country_codes'] == ['CN', 'SG']
    assert 'site_country_codes' not in slot


def test_wrong_identity_is_quarantined_not_country_only_relabelled() -> None:
    result = apply_country({'review_entity_id': 'UNESCO', 'country_code': 'FR'}, {
        'action': 'quarantine_wrong_identity_country_verified', 'replacement_name': 'Local laboratory',
        'replacement_type': 'unknown', 'sites': ['DE']})
    assert result['identity_status'] == 'identity_unresolved'
    assert result['id'] is None
    assert result['country_code'] == 'DE'
    assert result['rejected_original_entity_id'] == 'UNESCO'


def test_present_address_only_slot_is_removed_from_analysis_layer() -> None:
    assert apply_country({'review_entity_id': 'Hitachi'}, {'action': 'remove_present_address_only_slot'}) is None


def test_duplicate_parent_keeps_both_source_receipts_and_both_sites() -> None:
    a = {'review_entity_id': 'university', 'site_country_codes': ['SG'], 'weighted_review_evidence': [{'receipt': 'a'}]}
    b = {'review_entity_id': 'university', 'site_country_codes': ['CN'], 'weighted_review_evidence': [{'receipt': 'b'}]}
    result = deduplicate([a, b])
    assert len(result) == 1
    assert result[0]['site_country_codes'] == ['CN', 'SG']
    assert len(result[0]['weighted_review_evidence']) == 2
    assert a['site_country_codes'] == ['SG']


def test_country_area_convention_is_preserved() -> None:
    assert site_country('Lab, Hong Kong, China') == 'HK'
    assert site_country('Lab, Macau, China') == 'MO'
    assert site_country('Lab, Beijing, China') == 'CN'


def test_other_affiliation_country_not_transferred_to_unrelated_raw_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review, 'publisher', lambda doi: {'source_receipt': 'receipt'})
    monkeypatch.setattr(review, 'author_addresses', lambda a, p: (['National Biobank, Klong Luang, Thailand'], 'publisher_exact_author_name'))
    slot = {'display_name': 'Administration Bureau of Naban River Watershed National Nature Reserve', 'country_code': None}
    result = review.fill_missing_country(slot, {}, 'example-doi')
    assert not result.get('site_country_codes')
    assert result['country_review_unresolved_reason'] == 'raw_fragment_not_confirmed_in_primary_publisher_affiliations'
