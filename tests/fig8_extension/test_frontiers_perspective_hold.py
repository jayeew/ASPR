from experiments.fig8_extension.fig8b_qualification_decide import decide, frontiers_perspective_hold


def test_perspective_label_does_not_prove_no_original_data() -> None:
    item = {'id': 'p', 'identity': 'matched', 'class': 'nonprimary', 'kind': 'Perspective',
            'family': 'publisher_frontiers', 'doi': '10.3389/fimmu.test',
            'online_date': '2024-01-01'}
    assert frontiers_perspective_hold(item)
    assert decide([item], [])['status'] == 'pending_special_subtype_review'
    for change in [{'kind': 'Review'}, {'doi': '10.1234/test'}, {'doi': None}, {'class': 'integrity_hold'}]:
        assert not frontiers_perspective_hold({**item, **change})
    review = {**item, 'id': 'review', 'kind': 'Review'}
    assert decide([item, review], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**item, 'class': 'integrity_hold'}], [])['status'] == 'pending_integrity_review'


def test_archived_perspective_requires_frontiers_identity() -> None:
    item = {'identity': 'matched', 'kind': 'Perspective', 'family': 'archived_jats',
            'doi': '10.3389/fimmu.test', 'journal_titles': ['Frontiers in Immunology'],
            'publishers': ['Frontiers Media SA']}
    assert frontiers_perspective_hold(item)
    assert not frontiers_perspective_hold({**item, 'publishers': []})
    assert not frontiers_perspective_hold({**item, 'journal_titles': ['Other Journal']})


def test_content_review_is_bound_to_named_primary_records() -> None:
    item = {'id': 'p', 'identity': 'matched', 'class': 'nonprimary', 'kind': 'Perspective',
            'family': 'publisher_frontiers', 'doi': '10.3389/fimmu.test',
            'online_date': '2024-01-01'}
    review = {**item, 'id': 'r', 'family': 'reviewed_frontiers_content',
              'kind': 'unspecified', 'class': 'unresolved_type', 'study_design_class': 'nonprimary',
              'reviewed_primary_evidence_ids': ['p'], 'content_sha256': 'hash',
              'content_locator': 'retained-body', 'review_quotes': ['author statement'],
              'content_review_reason': 'Narrative synthesis of previous studies'}
    assert decide([item, review], [])['status'] == 'excluded_primary_nonresearch'
    for change in [{'reviewed_primary_evidence_ids': ['missing']}, {'doi': '10.3389/other'},
                   {'content_sha256': ''}, {'review_quotes': []}, {'study_design_class': 'research'}]:
        assert decide([item, {**review, **change}], [])['status'] == 'pending_special_subtype_review'
    conflicting = {**item, 'id': 'research', 'kind': 'Original Research', 'class': 'research'}
    assert decide([item, review, conflicting], [])['status'] == 'pending_conflicting_primary_types'
    # Ambiguous evidence IDs cannot authorize replacing either original source.
    collision = {**conflicting, 'id': item['id']}
    assert decide([item, review, collision], [])['status'] == 'pending_special_subtype_review'
