from __future__ import annotations

import pytest

from experiments.fig8_extension.nature_research_roster import enrich, research_rows
from experiments.fig8_extension.nature_types import parse_catalogue


def test_catalogue_uses_visible_publisher_type_and_online_date() -> None:
    text = '''<html><span class="c-facet__selected">2023 (1)</span>
    <article><a class="c-card__link" href="/articles/s41586-022-12345-6">Title</a>
    <span class="c-meta__type">Article</span><time datetime="2023-01-03"></time></article></html>'''
    result = parse_catalogue(text, 'https://www.nature.com/nature/articles?year=2023', 2023)
    assert result['rows'][0]['publisher_year'] == 2023
    assert result['rows'][0]['publisher_type'] == 'Article'
    assert result['expected_count'] == 1


def test_interstitial_does_not_become_empty_catalogue() -> None:
    with pytest.raises(ValueError, match='Missing publisher'):
        parse_catalogue('<html>Please sign in</html>', 'https://www.nature.com', 2023)


def test_research_does_not_include_commentary_or_wrong_year() -> None:
    def row(kind: str, year: int) -> dict:
        return {'doi': f'{kind}{year}', 'publisher_type': kind, 'publisher_year': year,
                'publisher_date': f'{year}-01-02', 'title': kind}
    result = research_rows([row('Article', 2023), row('Article', 2022),
                            row('Correspondence', 2023), row('Research Briefing', 2023)])
    assert [r['doi'] for r in result] == ['Article2023']


def test_ambiguous_doi_does_not_silently_merge_metadata() -> None:
    source = {'id': 'https://openalex.org/S137773608'}
    matches = [{'id': f'W{i}', 'primary_location': {'source': source}} for i in [1, 2]]
    result = enrich({'doi': '10.test/1'}, matches)
    assert result['metadata_match_status'] == 'ambiguous_doi_match'
    assert result['work_id'] is None
    assert result['include_primary'] is True
    assert result['has_institution'] is False


def test_partial_country_and_truncated_authors_are_not_complete() -> None:
    raw = {'id': 'W1', 'primary_location': {'source': {'id': 'https://openalex.org/S137773608'}},
           'authors_count': 2, 'authorships': [{'institutions': [{'country_code': 'US'}, {'country_code': None}]}],
           'is_retracted': False}
    result = enrich({'doi': '10.test/1'}, [raw])
    assert result['has_institution'] is True
    assert result['authors_truncated'] is True
    assert result['complete_institution_country'] is False
