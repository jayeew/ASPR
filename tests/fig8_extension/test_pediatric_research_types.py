from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def test_named_research_types_and_holds() -> None:
    e = {'id': 'test', 'identity': 'matched', 'family': 'nature_page', 'class': 'unresolved_type',
         'kind': 'OriginalPaper', 'categories': ['Clinical Research Article', 'OriginalPaper'],
         'journal_titles': ['Pediatric Research'], 'issns': ['1530-0447'],
         'online_date': '2024-02-01', 'date_precision': 'day'}
    assert journal_research_format(e)
    assert decide([e], [])['status'] == 'included_verified_research'
    for change in [{'journal_titles': ['Other']}, {'issns': []},
                   {'categories': ['Review']}, {'categories': ['Study Protocol Article']},
                   {'categories': ['OriginalPaper']}, {'scope_review_required': True},
                   {'class': 'integrity_hold'}]:
        assert not journal_research_format({**e, **change})
