from experiments.fig8_extension.fig8b_qualification_decide import acs_reviewed_rapid_format, decide


def test_exact_acs_native_formats_and_conflicts() -> None:
    for journal, issn, category in [
        ('Journal of the American Chemical Society', '0002-7863', 'Communication'),
        ('ACS Macro Letters', '2161-1653', 'Letter'),
        ('Organic Letters', '1523-7060', 'Letter'),
    ]:
        item = {'id': 'native', 'family': 'archived_jats', 'identity': 'matched',
                'class': 'unresolved_type', 'kind': 'rapid-communication',
                'doi': '10.1021/example', 'issns': [issn], 'journal_titles': [journal],
                'categories': [category], 'title': 'An experimental study', 'online_date': '2024-01-01'}
        assert acs_reviewed_rapid_format(item)
        assert decide([item], [])['status'] == 'included_verified_research'
        assert decide([{**item, 'online_date': ''}], [])['status'] == 'pending_primary_online_date'
        assert decide([{**item, 'scope_review_required': True}], [])['status'].startswith('pending')
        assert decide([item], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
        other = {**item, 'id': 'review', 'kind': 'Review', 'class': 'nonprimary'}
        assert decide([item, other], [])['status'] == 'pending_conflicting_primary_types'
        for change in [{'issns': ['other']}, {'journal_titles': ['other']}, {'doi': None},
                       {'categories': [category, 'Review']}, {'kind': 'letter'}]:
            assert not acs_reviewed_rapid_format({**item, **change})


def test_plural_heading_is_not_a_generic_or_featured_format() -> None:
    item = {'id': 'native', 'family': 'archived_jats', 'identity': 'matched',
            'class': 'unresolved_type', 'kind': 'rapid-communication',
            'doi': '10.1021/example', 'issns': ['1554-8929'],
            'journal_titles': ['ACS Chemical Biology'], 'categories': ['Letters'],
            'title': 'An experimental study', 'online_date': '2024-01-01'}
    assert decide([item], [])['status'] == 'included_verified_research'
    for labels in [[], ['Article'], ['Featured Letter'], ['Rapid Report'], ['Letter', 'Review']]:
        assert not acs_reviewed_rapid_format({**item, 'categories': labels})
