from experiments.fig8_extension.fig8b_qualification_decide import decide, literal_nonresearch_alias, native_nonresearch_format, type_class


def test_only_publication_type_introduction_is_nonresearch() -> None:
    assert type_class('introduction', 'archived_jats', []) == 'nonprimary'
    assert type_class('research-article', 'archived_jats', ['Introduction']) == 'research'
    assert type_class('introduction', 'archived_jats', ['Original Research']) == 'unresolved_type'
    assert type_class('introduction', 'crossref_publisher_deposit', []) != 'nonprimary'
    e = {'id': 'intro', 'family': 'archived_jats', 'kind': 'introduction',
         'identity': 'matched', 'class': 'unresolved_type', 'categories': [],
         'online_date': '2024-01-01'}
    assert native_nonresearch_format(e)
    assert literal_nonresearch_alias(e)
    assert decide([e], [])['status'] == 'excluded_primary_nonresearch'
    research = {**e, 'id': 'research', 'kind': 'research-article', 'class': 'research'}
    assert decide([e, research], [])['status'] == 'pending_conflicting_primary_types'
    assert not native_nonresearch_format({**e, 'identity': 'title_or_journal_mismatch'})
    assert not native_nonresearch_format({**e, 'categories': ['Research Article']})


def test_introduction_reconciliation_keeps_nonresearch_precedence_outside_window() -> None:
    intro = {'id': 'intro', 'family': 'archived_jats', 'kind': 'introduction',
             'identity': 'matched', 'class': 'unresolved_type', 'categories': [],
             'online_date': ''}
    deposited = {'id': 'deposit', 'family': 'crossref_publisher_deposit',
                 'kind': 'journal-article', 'identity': 'matched',
                 'class': 'unresolved_type', 'online_date': '2026-01-08'}
    assert decide([intro, deposited], [])['status'] == 'excluded_primary_nonresearch'
