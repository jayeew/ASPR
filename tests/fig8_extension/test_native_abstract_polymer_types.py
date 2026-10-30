from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format, type_class


def test_document_abstract_is_not_an_embedded_abstract() -> None:
    assert type_class('abstract', 'archived_jats') == 'nonprimary'
    assert type_class('research-article', 'archived_jats', ['Abstract']) == 'research'
    evidence = {'id': 'e', 'identity': 'matched', 'family': 'archived_jats', 'kind': 'abstract',
                'class': 'unresolved_type', 'online_date': '2024-01-01', 'date_precision': 'day'}
    assert decide([evidence], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**evidence, 'scope_review_required': True}], [])['status'].startswith('pending')


def test_polymer_original_is_journal_scoped_and_keeps_holds() -> None:
    e = {'identity': 'matched', 'family': 'nature_page', 'kind': 'OriginalPaper',
         'title': 'Polymer synthesis', 'issns': ['1349-0540'], 'journal_titles': ['Polymer Journal'],
         'categories': ['Original Article', 'OriginalPaper']}
    assert journal_research_format(e)
    for patch in [{'journal_titles': ['Another Journal']}, {'issns': ['0000-0000']},
                  {'kind': 'Short Communication'}, {'scope_review_required': True},
                  {'title': 'A systematic review of polymers'}, {'categories': ['Review Article']}]:
        assert not journal_research_format({**e, **patch})
