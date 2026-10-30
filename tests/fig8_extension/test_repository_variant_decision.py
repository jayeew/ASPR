from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_repository_variant_needs_retained_identity_and_content_review() -> None:
    e = {'id': 'review', 'identity': 'matched', 'family': 'repository_version_review',
         'retained_work_id': 'Wtarget', 'relation_type': 'IsVariantFormOf',
         'content_identity_confirmed': True, 'class': 'unresolved_type',
         'kind': 'repository_version', 'online_date': ''}
    result = decide([e], [], 'Wtarget')
    assert result['status'] == 'excluded_duplicate_repository_version'
    assert result['evidence_ids'] == ['review']
    assert decide([e], [], '')['status'].startswith('pending')
    for changes in [{'retained_work_id': 'another'}, {'content_identity_confirmed': False},
                    {'identity': 'mismatch'}, {'relation_type': 'References'}]:
        assert decide([{**e, **changes}], [], 'Wtarget')['status'] != 'excluded_duplicate_repository_version'
