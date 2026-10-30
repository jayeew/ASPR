from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def observation() -> dict:
    return {'id': 'native', 'identity': 'matched', 'family': 'archived_jats',
            'class': 'subtype_review_required', 'issns': ['2165-0497'],
            'journal_titles': ['Microbiology Spectrum'],
            'doi': '10.1128/spectrum.00001-23', 'kind': 'brief-report',
            'categories': ['Observation'], 'title': 'Experimental microbial findings',
            'online_date': '2023-06-01', 'date_precision': 'day'}


def test_exact_journal_format_and_scope_guards() -> None:
    source = observation()
    assert journal_research_format(source)
    assert decide([source], [])['status'] == 'included_verified_research'
    for patch in ({'issns': []}, {'journal_titles': ['Other Journal']},
                  {'doi': '10.1128/mra.00001-23'}, {'categories': []},
                  {'categories': ['Observation', 'Case Report']},
                  {'kind': 'letter'}, {'identity': 'mismatch'},
                  {'scope_review_required': True}, {'title_scope_flag': True},
                  {'title': 'A systematic review of microbial findings'},
                  {'class': 'integrity_hold'}):
        assert not journal_research_format({**source, **patch})


def test_conflicts_and_missing_dates_are_not_erased() -> None:
    source = observation()
    assert decide([{**source, 'online_date': ''}], [])['status'] == 'pending_primary_online_date'
    assert decide([source], [], duplicate_of='other')['status'] == 'excluded_duplicate_wrong_journal'
    conflict = {**source, 'id': 'editorial', 'kind': 'editorial', 'categories': [], 'class': 'nonprimary'}
    assert decide([source, conflict], [])['status'] == 'pending_conflicting_primary_types'


def test_category_free_legacy_front_needs_exact_identity() -> None:
    source = observation()
    legacy = {**source, 'id': 'legacy', 'categories': [], 'issns': []}
    assert decide([source, legacy], [])['status'] == 'included_verified_research'
    for patch in ({'doi': '10.1128/spectrum.other'}, {'title': 'Another manuscript'},
                  {'journal_titles': ['Another Journal']}, {'scope_review_required': True}):
        assert decide([source, {**legacy, **patch}], [])['status'] != 'included_verified_research'
