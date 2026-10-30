from experiments.fig8_extension.fig8b_qualification_decide import decide, literal_nonresearch_alias, type_class


def test_opinion_only_for_documented_publisher_formats() -> None:
    for family in ['publisher_mdpi_oai', 'publisher_frontiers']:
        e = {'id': 'test', 'identity': 'matched', 'family': family, 'class': 'unresolved_type', 'kind': 'Opinion', 'online_date': ''}
        assert literal_nonresearch_alias(e)
        assert decide([e], [])['status'] == 'excluded_primary_nonresearch'
        assert not literal_nonresearch_alias({**e, 'scope_review_required': True})
        assert type_class('Hypothesis', family) != 'nonprimary'
    assert not literal_nonresearch_alias({**e, 'family': 'other'})
