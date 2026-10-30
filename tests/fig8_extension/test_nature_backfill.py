from __future__ import annotations

import pytest

from experiments.fig8_extension.nature_backfill_fetch import parse_page
from experiments.fig8_extension.nature_backfill_resolve import (
    accepted_ror,
    address_country,
    author_addresses,
)


def test_publisher_parser_checks_doi_and_preserves_scientific_superscripts() -> None:
    text = '''<html><head><meta name="citation_doi" content="10.test/a">
    <meta name="citation_journal_title" content="Nature"><meta name="citation_article_type" content="Article"></head>
    <section data-title="Abstract"><div class="c-article-section__content"><p>
    We measure 10<sup>−3</sup> units<sup><a data-test="citation-ref">1</a></sup> in repeated experiments.
    The reference marker is not part of the scientific text and must be removed safely.</p></div></section>
    <ol class="c-article-author-affiliation__list"><li id="Aff1"><p class="c-article-author-affiliation__address">Lab, USA</p></li></ol>
    <ol class="c-article-authors-search"><li id="auth-Jane_Doe-Aff1"><span class="c-article-authors-search__title">Jane Doe</span></li></ol></html>'''
    result = parse_page(text, '10.test/a')
    assert '10^{−3}' in result['abstract_text']
    assert 'units1' not in result['abstract_text']
    assert result['author_affiliation_parse_complete']
    with pytest.raises(ValueError, match='identity'):
        parse_page(text, '10.test/b')


def test_author_name_collision_does_not_assign_arbitrary_affiliation() -> None:
    author = {'author': {'display_name': 'Jane Doe'}, 'raw_affiliation_strings': ['Original Lab, USA']}
    pub = {'authors': [{'name': 'Jane Doe', 'affiliation_ids': ['Aff1']},
                       {'name': 'Jane Doe', 'affiliation_ids': ['Aff2']}],
           'affiliations': [{'affiliation_id': 'Aff1', 'address': 'A, USA'}, {'affiliation_id': 'Aff2', 'address': 'B, UK'}]}
    assert author_addresses(author, pub) == (['Original Lab, USA'], 'openalex_raw_affiliation')


def test_missing_author_popup_uses_explicit_affiliation_author_list() -> None:
    author = {'author': {'display_name': 'Jane Doe'}, 'raw_affiliation_strings': []}
    pub = {'authors': [], 'affiliations': [{'affiliation_id': 'Aff1', 'address': 'Lab, USA',
                                         'authors_text': 'Jane Doe & John Smith'}]}
    assert author_addresses(author, pub) == (['Lab, USA'], 'publisher_exact_affiliation_author_list')


def test_country_is_explicit_final_token_not_name_keyword() -> None:
    names = {'united states': 'US', 'usa': 'US'}
    assert address_country('United States University, Unknown city', names) is None
    assert address_country('Lab, Boston, USA. a@example.org', names) == 'US'


def test_ror_chosen_needs_exact_alias_and_consistent_country() -> None:
    result = {'chosen': [{'organization': {'names': [{'value': 'Example University'}],
                      'locations': [{'geonames_details': {'country_code': 'US'}}]}}]}
    assert accepted_ror('Example University, USA', result, {'usa': 'US'})[0]
    assert accepted_ror('Other University, USA', result, {'usa': 'US'})[0] is None
    assert accepted_ror('Example University, UK', result, {'uk': 'GB'})[0] is None


def test_ror_high_score_without_chosen_is_not_accepted() -> None:
    assert accepted_ror('A Lab, USA', {'chosen': []}, {'usa': 'US'})[0] is None


def test_ror_same_country_different_city_is_reviewed() -> None:
    result = {'chosen': [{'organization': {'names': [{'value': 'Second Hospital'}],
        'locations': [{'geonames_details': {'country_code': 'CN', 'name': 'Xianyang'}}]}}]}
    organization, status = accepted_ror('Second Hospital, Liaocheng, China', result, {'china': 'CN'})
    assert organization is None and status == 'ror_location_requires_review'


def test_geographic_region_alias_does_not_count_as_organization_name() -> None:
    result = {'chosen': [{'organization': {'names': [{'value': 'Australian Capital Territory'}],
        'locations': [{'geonames_details': {'country_code': 'AU', 'name': 'Canberra'}}]}}]}
    assert accepted_ror('Example Institute, Canberra, Australian Capital Territory, Australia',
                        result, {'australia': 'AU'})[0] is None
