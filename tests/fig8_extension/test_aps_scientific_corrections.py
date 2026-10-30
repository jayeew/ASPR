from experiments.fig8_extension.fig8b_qualification_decide import (
    APS_REVIEWED_SCIENTIFIC_CORRECTIONS, decide, reviewed_aps_scientific_correction,
)


def source() -> dict:
    return {'id': sorted(APS_REVIEWED_SCIENTIFIC_CORRECTIONS)[0], 'identity': 'matched',
            'family': 'aps', 'kind': 'article', 'class': 'subtype_review_required',
            'online_date': '2024-01-01', 'title': 'Corrections to a physical model'}


def test_review_applies_only_to_exact_research_evidence() -> None:
    e = source()
    assert reviewed_aps_scientific_correction(e)
    assert decide([e], [], '')['status'] == 'included_verified_research'
    for update in [{'id': 'unreviewed'}, {'identity': 'mismatch'}, {'family': 'archived_jats'},
                   {'kind': 'correction'}, {'scope_review_required': True}, {'class': 'integrity_hold'}]:
        assert not reviewed_aps_scientific_correction({**e, **update})


def test_other_primary_conflicts_and_integrity_still_hold() -> None:
    e = source()
    correction = {**e, 'id': 'other', 'kind': 'erratum', 'class': 'nonprimary'}
    assert decide([e, correction], [], '')['status'] == 'pending_conflicting_primary_types'
    concern = {**correction, 'class': 'integrity_hold', 'kind': 'expressionofconcern'}
    assert decide([e, concern], [], '')['status'] == 'pending_integrity_review'
