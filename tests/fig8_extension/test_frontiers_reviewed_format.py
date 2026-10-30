from experiments.fig8_extension.fig8b_qualification_decide import frontiers_reviewed_format


def test_frontiers_literal_format_requires_identity_and_preserves_holds() -> None:
    item = {'family': 'publisher_frontiers', 'identity': 'matched', 'doi': '10.3389/a.2024.1',
            'kind': 'Brief Research Report', 'class': 'unresolved_type', 'title': 'Experimental measurements'}
    assert frontiers_reviewed_format(item) == 'research'
    assert frontiers_reviewed_format({**item, 'kind': 'Mini Review'}) == 'nonprimary'
    for changes in [{'identity': 'unmatched'}, {'family': 'archived_jats'}, {'scope_review_required': True},
                    {'title_scope_flag': True}, {'class': 'integrity_hold'}, {'doi': '10.1/test'},
                    {'categories': ['Review']}, {'title': 'A case report'}]:
        assert frontiers_reviewed_format({**item, **changes}) == ''
