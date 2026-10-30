from experiments.fig8_extension.fig8b_qualification_decide import native_nonresearch_format, decide, type_class


def test_jats_document_type_not_body_section_or_unresolved_special_format() -> None:
    base = {'id': 'test', 'identity': 'matched', 'class': 'unresolved_type',
            'family': 'archived_jats', 'online_date': '2024-02-01', 'date_precision': 'day'}
    for kind in ['discussion', 'in-brief', 'reviewer-report', 'editor-report', 'community-comment']:
        item = {**base, 'kind': kind}
        assert native_nonresearch_format(item)
        assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
    assert type_class('research-article', 'archived_jats') == 'research'
    assert not native_nonresearch_format({**base, 'kind': 'brief-report'})
    assert not native_nonresearch_format({**base, 'kind': 'discussion', 'scope_review_required': True})


def test_polymer_review_scope() -> None:
    item = {'identity': 'matched', 'class': 'unresolved_type', 'family': 'nature_page',
            'kind': 'Focus Review', 'journal_titles': ['Polymer Journal'], 'issns': ['0032-3896']}
    assert native_nonresearch_format(item)
    for changed in [{'journal_titles': ['Other']}, {'issns': []}, {'kind': 'OriginalPaper'},
                    {'identity': 'unmatched'}, {'class': 'integrity_hold'}]:
        assert not native_nonresearch_format({**item, **changed})
