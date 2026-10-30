import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


@pytest.mark.parametrize('kind', ['Letter', 'Analysis', 'Technical Report', 'Brief Communication'])
def test_native_genetics_research_formats_keep_evidence_holds(kind: str) -> None:
    evidence = {'id': 'native', 'identity': 'matched', 'family': 'nature_page',
                'source_id': 'https://openalex.org/S137905309', 'kind': kind,
                'class': 'subtype_review_required', 'title': 'A new genomic measurement',
                'online_date': '2024-02-03', 'date_precision': 'day', 'categories': []}
    assert journal_research_format(evidence)
    assert decide([evidence], [], '')['status'] == 'included_verified_research'
    for change in [{'identity': 'mismatch'}, {'family': 'archived_jats'},
                   {'source_id': 'another_journal'}, {'kind': 'Correspondence'},
                   {'scope_review_required': True}, {'title_scope_flag': 'review'},
                   {'categories': ['Review']}, {'title': 'Systematic review of genomic data'}]:
        assert not journal_research_format({**evidence, **change})
    assert decide([{**evidence, 'online_date': ''}], [], '')['status'] == 'pending_primary_online_date'
    assert decide([{**evidence, 'online_date': '2022-02-03'}], [], '')['status'] == 'excluded_primary_online_year_outside_window'
    conflicting = {**evidence, 'id': 'other', 'kind': 'Review', 'class': 'nonprimary'}
    assert decide([evidence, conflicting], [], '')['status'] == 'pending_conflicting_primary_types'
