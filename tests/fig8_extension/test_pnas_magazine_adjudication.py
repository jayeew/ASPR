from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence(**values: object) -> dict:
    return {'id': 'native', 'identity': 'matched', 'family': 'archived_jats',
            'issns': ['0027-8424'], 'kind': 'research-article', 'class': 'subtype_review_required',
            'categories': ['Inner Workings'], 'title': 'Scientists explain their work',
            'online_date': '2024-01-01', **values}


def test_native_proof_required_and_conflicts_retained() -> None:
    native = evidence()
    assert decide([native], [])['status'] == 'pending_special_subtype_review'
    proof = evidence(id='proof', native_magazine_rule='pnas_editorial_1300186110',
                     native_heading_sha256='hash', magazine_policy_receipt='policy.json',
                     reviewed_primary_evidence_ids=['native'])
    assert decide([native, proof], [])['status'] == 'excluded_primary_nonresearch'
    conflict = evidence(id='other', family='publisher_page', categories=[], **{'class': 'research'})
    assert decide([native, proof, conflict], [])['status'] == 'pending_conflicting_primary_types'
    integrity = evidence(id='integrity', **{'class': 'integrity_hold'})
    assert decide([native, proof, integrity], [])['status'] == 'pending_integrity_review'
    assert decide([evidence(issns=['9999-9999'])], [])['status'] == 'pending_special_subtype_review'


def test_profile_and_qnas_require_native_heading_proof_not_title() -> None:
    for heading in ['Profile', 'QnAs']:
        native = evidence(categories=[heading])
        proof = evidence(id='proof', categories=[heading],
                         native_magazine_rule='pnas_editorial_1300186110',
                         native_heading_sha256='hash', magazine_policy_receipt='policy.json',
                         reviewed_primary_evidence_ids=['native'])
        assert decide([native], [])['status'] == 'pending_special_subtype_review'
        assert decide([native, proof], [])['status'] == 'excluded_primary_nonresearch'
        unrelated = {**proof, 'categories': ['Research Article'], 'title': 'Profile of a protein'}
        assert decide([native, unrelated], [])['status'] == 'pending_special_subtype_review'
