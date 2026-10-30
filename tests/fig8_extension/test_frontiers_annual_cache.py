from experiments.fig8_extension.frontiers_annual_cache import matches


def test_frontiers_annual_requires_native_identity_and_explicit_published_date() -> None:
    paper = {'doi': '10.3389/example', 'title': 'A study', 'online_date': '2023-01-04'}
    journal = {'journal_name': 'Frontiers in Physics', 'issn': '["2296-424X"]'}
    parsed = {'doi': [paper['doi']], 'title': ['A study'], 'journal_titles': ['Frontiers in Physics'],
              'issns': ['2296-424X'], 'publisher': ['Frontiers'], 'online_date': '2023-01-04',
              'date_field_review': {'explicit_article_history': {'Published': ['2023-01-04'], 'Accepted': ['2022-12-15']}}}
    assert matches(parsed, paper, journal)
    assert not matches({**parsed, 'publisher': ['Springer Nature']}, paper, journal)
    assert not matches({**parsed, 'issns': ['0000-0000']}, paper, journal)
    assert not matches({**parsed, 'date_field_review': {}}, paper, journal)
    assert not matches(parsed, {**paper, 'online_date': '2022-12-15'}, journal)
    assert not matches(parsed, {**paper, 'title': 'Other study'}, journal)
