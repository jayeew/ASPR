from experiments.fig8_extension.fig8b_qualification_decide import mdpi_reviewed_communication, decide


def test_native_mapping_does_not_infer_date_or_apply_to_other_publishers() -> None:
    e = {'id': 'test', 'family': 'publisher_mdpi_oai', 'identity': 'matched', 'doi': '10.3390/a1',
         'kind': 'rapid-communication', 'class': 'unresolved_type', 'title': 'Experimental findings',
         'online_date': '', 'date_precision': ''}
    assert mdpi_reviewed_communication(e)
    assert decide([e], [])['status'] == 'pending_primary_online_date'
    assert decide([{**e, 'online_date': '2024-03-04'}], [])['status'] == 'included_verified_research'
    for changes in [{'family': 'archived_jats'}, {'doi': '10.1/a'}, {'identity': 'unmatched'},
                    {'kind': 'short-communication'}, {'categories': ['Review']},
                    {'scope_review_required': True}, {'title': 'A systematic review'}]:
        assert not mdpi_reviewed_communication({**e, **changes})
