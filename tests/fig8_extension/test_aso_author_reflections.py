from experiments.fig8_extension.fig8b_qualification_decide import aso_author_reflections, decide


def test_companion_commentary_does_not_exclude_generic_research() -> None:
    item = {'id': 'aso', 'identity': 'matched', 'family': 'publisher_springer',
            'class': 'unresolved_type', 'kind': 'ASO Author Reflections',
            'title': 'ASO Author Reflections: Treatment outcomes', 'online_date': '2024-01-01', 'date_precision': 'day',
            'categories': ['ASO Author Reflections', 'BriefCommunication'],
            'journal_titles': ['Annals of Surgical Oncology'], 'issns': ['1534-4681']}
    generic = {**item, 'id': 'generic', 'kind': 'BriefCommunication',
               'class': 'subtype_review_required'}
    assert decide([item, generic], [])['status'] == 'excluded_primary_nonresearch'
    for change in [{'kind': 'BriefCommunication'}, {'title': 'Treatment outcomes'},
                   {'journal_titles': ['Other']}, {'issns': []}, {'identity': 'mismatch'},
                   {'categories': ['Original Article']}, {'scope_review_required': True},
                   {'class': 'integrity_hold'}]:
        assert not aso_author_reflections({**item, **change})
