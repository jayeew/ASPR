from experiments.fig8_extension.fig8b_qualification_decide import decide, reviewed_aps_tutorial


def test_official_tutorial_type_is_not_original_research() -> None:
    for journal in ['PRXQUANTUM', 'PRXENERGY', 'PRXLIFE']:
        e = {'id': 'native', 'identity': 'matched', 'family': 'aps', 'aps_journal': journal,
             'kind': 'tutorial', 'class': 'unresolved_type', 'online_date': '2024-01-01'}
        assert reviewed_aps_tutorial(e)
        assert decide([e], [], '')['status'] == 'excluded_primary_nonresearch'
        for change in [{'family': 'crossref_publisher_deposit'}, {'aps_journal': 'PRPER'},
                       {'kind': 'article'}, {'identity': 'mismatch'}, {'class': 'integrity_hold'}]:
            assert not reviewed_aps_tutorial({**e, **change})
        research = {**e, 'id': 'conflicting', 'kind': 'article', 'class': 'research'}
        assert decide([e, research], [], '')['status'] == 'pending_conflicting_primary_types'
