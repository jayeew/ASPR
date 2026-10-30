from experiments.fig8_extension.fig8b_qualification_decide import aso_visual_abstract, decide


def test_companion_visual_abstract_is_not_an_original_article() -> None:
    evidence = {'id': 'test', 'identity': 'matched', 'family': 'publisher_springer',
                'class': 'unresolved_type', 'kind': 'Abstract',
                'categories': ['ASO Visual Abstract', 'Abstract'],
                'journal_titles': ['Annals of Surgical Oncology'], 'issns': ['1534-4681'],
                'title': 'ASO Visual Abstract: A Clinical Study',
                'online_date': '2024-01-01', 'date_precision': 'day'}
    assert aso_visual_abstract(evidence)
    assert decide([evidence], [])['status'] == 'excluded_primary_nonresearch'
    for change in [{'title': 'A Clinical Study'}, {'journal_titles': ['Other']},
                   {'issns': []}, {'identity': 'title_or_journal_mismatch'},
                   {'categories': ['Original Article'], 'kind': 'OriginalPaper'},
                   {'scope_review_required': True}, {'class': 'integrity_hold'}]:
        assert not aso_visual_abstract({**evidence, **change})
