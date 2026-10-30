"""Native FLA evidence qualifies research without erasing independent holds."""
from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def native() -> dict:
    return {'id': 'native-fla', 'identity': 'matched', 'family': 'publisher_elsevier',
            'kind': 'Full-length article', 'journal_titles': ['Nuclear Physics B'],
            'doi': '10.1016/j.nuclphysb.2023.116105',
            'categories': ['Full-length article'], 'class': 'unresolved_type',
            'title': 'Elastic scattering', 'online_date': '', 'date_precision': ''}


def test_native_type_requires_independent_online_date() -> None:
    evidence = native()
    assert journal_research_format(evidence)
    assert decide([evidence], [])['status'] == 'pending_primary_online_date'
    evidence['online_date'] = '2023-02-01'
    assert decide([evidence], [])['status'] == 'included_verified_research'


def test_native_type_preserves_identity_scope_and_conflicts() -> None:
    evidence = native()
    for delta in [{'identity': 'mismatched'}, {'scope_review_required': True},
                  {'family': 'crossref_publisher_deposit'}, {'kind': 'journal-article'},
                  {'journal_titles': ['Other journal']}, {'class': 'integrity_hold'},
                  {'categories': ['Full-length article', 'Review']}]:
        assert not journal_research_format({**evidence, **delta})
    review = {**evidence, 'id': 'review', 'kind': 'Review article', 'class': 'nonprimary'}
    assert decide([evidence, review], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([evidence], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
