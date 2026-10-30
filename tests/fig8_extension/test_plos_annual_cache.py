from experiments.fig8_extension.plos_annual_cache import matched_record


def test_native_date_requires_matching_independent_electronic_day() -> None:
    paper = {'doi': '10.1371/test', 'title': 'A study', 'online_date': '2023-01-02', 'online_year': 2023, 'independent_dates': ['2023-01-02']}
    doc = {'id': paper['doi'], 'title_display': 'A <i>study</i>', 'publisher': 'Public Library of Science',
           'eissn': '1932-6203', 'publication_date': '2023-01-02T00:00:00Z'}
    assert matched_record(doc, paper, {'1932-6203'})
    assert not matched_record(doc, {**paper, 'online_date': '2023-01-03'}, {'1932-6203'})
    assert not matched_record(doc, paper, {'other'})
    assert not matched_record({**doc, 'publisher': 'Other'}, paper, {'1932-6203'})

    assert not matched_record(doc, {**paper, 'independent_dates': []}, {'1932-6203'})
