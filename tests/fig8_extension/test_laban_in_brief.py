from experiments.fig8_extension.fig8b_qualification_decide import decide, laban_in_brief


def test_news_summary_is_distinct_from_primary_brief_communication() -> None:
    summary = {'id': 'summary', 'identity': 'matched', 'family': 'nature_page',
               'class': 'unresolved_type', 'kind': 'In Brief',
               'categories': ['In Brief', 'BriefCommunication'],
               'journal_titles': ['Lab Animal'], 'issns': ['1548-4475'],
               'online_date': '2024-01-01', 'date_precision': 'day'}
    generic = {**summary, 'id': 'generic', 'kind': 'BriefCommunication',
               'class': 'subtype_review_required'}
    assert decide([summary, generic], [])['status'] == 'excluded_primary_nonresearch'
    for change in [{'kind': 'Brief Communication'}, {'journal_titles': ['Other']},
                   {'issns': []}, {'identity': 'mismatch'}, {'scope_review_required': True},
                   {'categories': ['Research Article']}, {'class': 'integrity_hold'}]:
        assert not laban_in_brief({**summary, **change})
