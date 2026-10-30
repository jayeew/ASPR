from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def test_brief_report_is_journal_and_native_source_bound():
    e = {'id': 'native', 'identity': 'matched', 'family': 'publisher_springer',
         'source_id': 'https://openalex.org/S197962172', 'kind': 'Brief Report',
         'class': 'subtype_review_required', 'title': 'An experimental investigation',
         'online_date': '2024-02-03', 'categories': ['Brief Report']}
    assert journal_research_format(e)
    assert decide([e], [], '')['status'] == 'included_verified_research'
    for delta in [{'identity': 'mismatch'}, {'source_id': 'https://openalex.org/S2248630'},
                  {'kind': 'Short Communication'}, {'family': 'archived_jats'},
                  {'scope_review_required': True}, {'categories': ['Review']},
                  {'title': 'Systematic review of the literature'}]:
        assert not journal_research_format({**e, **delta})
    assert decide([{**e, 'online_date': '2022-01-01'}], [], '')['status'] == 'excluded_primary_online_year_outside_window'
    assert decide([{**e, 'online_date': ''}], [], '')['status'] != 'included_verified_research'


def test_native_journal_issn_fallback_and_metallurgy_scope() -> None:
    cases = [('Molecular Biology Reports', '1573-4978', 'Brief Report'),
             ('Metallurgical and Materials Transactions A', '1543-1940', 'Brief Communication')]
    for journal, issn, kind in cases:
        e = {'id': 'native', 'identity': 'matched', 'family': 'publisher_springer',
             'journal_titles': [journal], 'issns': [issn], 'kind': kind,
             'class': 'subtype_review_required', 'title': 'An experimental investigation',
             'online_date': '2024-02-03', 'categories': [kind]}
        assert decide([e], [], '')['status'] == 'included_verified_research'
        for delta in [{'issns': []}, {'journal_titles': ['Different journal']},
                      {'source_id': 'https://openalex.org/S0'}, {'kind': 'Letter to the Editor'},
                      {'family': 'archived_jats'}, {'categories': ['Review']},
                      {'scope_review_required': True}, {'identity': 'mismatch'}]:
            assert not journal_research_format({**e, **delta})
        assert decide([{**e, 'online_date': ''}], [], '')['status'] != 'included_verified_research'


def test_microbiology_internal_type_requires_literal_short_category() -> None:
    e = {'identity': 'matched', 'family': 'publisher_springer',
         'journal_titles': ['Current Microbiology'], 'issns': ['1432-0991'],
         'kind': 'Short Communication', 'categories': ['Short Communication']}
    assert journal_research_format(e)
    assert journal_research_format({**e, 'kind': 'BriefCommunication'})
    for delta in [{'kind': 'BriefCommunication', 'categories': []},
                  {'family': 'archived_jats'}, {'issns': []},
                  {'kind': 'Letter to the Editor'}, {'scope_review_required': True}]:
        assert not journal_research_format({**e, **delta})
