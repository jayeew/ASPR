from experiments.fig8_extension.fig8b_qualification_decide import literal_nonresearch_alias


def test_tools_trade_requires_verified_journal_and_preserves_holds() -> None:
    evidence = {
        'family': 'nature_page', 'kind': 'Tools of the Trade',
        'identity': 'matched', 'class': 'unresolved_type',
        'issns': ['3005-0685'], 'journal_titles': ['Nature Reviews Clean Technology'],
    }
    assert literal_nonresearch_alias(evidence)
    for replacement in [
        {'issns': ['0000-0000']}, {'journal_titles': ['Nature']},
        {'source_id': 'wrong'}, {'identity': 'title_or_journal_mismatch'},
        {'class': 'subtype_review_required'}, {'scope_review_required': True},
        {'categories': ['Research Article']}, {'family': 'crossref_publisher_deposit'},
    ]:
        assert not literal_nonresearch_alias({**evidence, **replacement})


def test_tools_trade_does_not_exclude_other_types_or_other_journals() -> None:
    evidence = {
        'family': 'nature_catalogue', 'kind': 'Tools of the Trade',
        'identity': 'matched', 'class': 'unresolved_type',
        'source_id': 'https://openalex.org/S4210234670',
    }
    assert literal_nonresearch_alias(evidence)
    assert not literal_nonresearch_alias({**evidence, 'kind': 'Research Article'})
    assert not literal_nonresearch_alias({**evidence, 'source_id': 'https://openalex.org/S137773608'})
