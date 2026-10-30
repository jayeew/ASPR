"""Guard against unsupported identity, country, and abstract inheritance."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.fig8_extension.fig8b_build import (
    enrich_authors,
    explicit_countries,
    metrics,
)
from experiments.fig8_extension.fig8b_publisher import parse
from experiments.fig8_extension.fig8b_recompute import deterministic_gzip_text
from experiments.fig8_extension.fig8b_supplements import select_epmc


def test_empty_authors_are_not_complete() -> None:
    assert metrics([], 'abstract')['every_author_institution'] == 0
    assert metrics([], 'abstract')['every_author_country'] == 0


def test_partial_raw_country_not_inferred() -> None:
    names = {'usa': 'US', 'china': 'CN'}
    assert explicit_countries({'raw_affiliation_strings': ['Boston, USA', 'Unknown laboratory']}, names) == []
    assert explicit_countries({'raw_affiliation_strings': ['Hong Kong, China']}, names) == []
    assert explicit_countries({'raw_affiliation_strings': ['Boston, USA; Beijing, China']}, names) == ['CN', 'US']
    assert explicit_countries({'raw_affiliation_strings': ['Current address: Boston, USA']}, names) == []


def test_unique_author_match_required() -> None:
    authors = [{'author': {'display_name': 'Same Name'}, 'institutions': []}] * 2
    donor = {'authors': [{'name': 'Same Name', 'nonpersonal': False, 'slots': [
        {'identity_missing': False, 'entity_id': 'I1', 'country_codes': ['US']}]}]}
    result, updates = enrich_authors(authors, donor, {})
    assert not updates['nature_v1_author_reused']
    assert all(not a['institution_ids_or_verified_local_entities'] for a in result)


def test_secondary_title_and_abstract_conflicts() -> None:
    paper = {'title': 'Target title'}
    assert select_epmc(paper, [{'record': {'title': 'Other title', 'abstractText': 'A'}}]) is None
    candidates = [{'record': {'title': 'Target title.', 'abstractText': a}} for a in ['A', 'B']]
    assert select_epmc(paper, candidates) is None


def test_publisher_requires_both_doi_and_issn() -> None:
    text = '<html><meta name="citation_doi" content="10.1038/test"><meta name="citation_issn" content="1234-5678"></html>'
    assert parse(text, '10.1038/test', ['1234-5678'])['status'] == 'publisher_doi_issn_verified'
    with pytest.raises(ValueError, match='ISSN'):
        parse(text, '10.1038/test', ['wrong'])
    with pytest.raises(ValueError, match='DOI'):
        parse(text, '10.1038/other', ['1234-5678'])


def test_recomputed_compressed_outputs_have_stable_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / 'first.gz', tmp_path / 'second.gz'
    for path in [first, second]:
        with deterministic_gzip_text(path) as stream:
            stream.write('same evidence\n')
    assert first.read_bytes() == second.read_bytes()


def test_known_exclusions_override_inherited_eligibility() -> None:
    from experiments.fig8_extension.fig8b_eligibility import disposition

    row = {'doi':'10.1038/test','source_id':'https://openalex.org/S137773608',
           'year_publisher_verified':2023,'publisher_type':'Article',
           'type_status':'nature_v1_original_article_verified'}
    assert disposition(row, {'10.1038/test'}, set()) == 'excluded_known_publisher_retraction'
    assert disposition(row, set(), {'10.1038/test'}) == 'pending_doi_version_resolution'
    assert disposition({**row, 'publisher_type':'Review Article'}, set(), set()) == 'excluded_publisher_nonprimary_type'
