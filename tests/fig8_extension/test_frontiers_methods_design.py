from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_methods_require_bound_individual_review_and_preserve_holds() -> None:
    native = dict(id='native', doi='10.3389/fmicb.test', identity='matched',
                  family='archived_jats', kind='methods-article', online_date='2024-01-01',
                  journal_titles=['Frontiers in Microbiology'], categories=[], **{'class': 'unresolved_type'})
    review = dict(id='review', doi=native['doi'], identity='matched', family='reviewed_study_design',
                  kind='Individual completed original study', online_date='', study_design_class='research',
                  reviewed_primary_evidence_ids=['native'], source_evidence_ids=['native'],
                  review_protocol='native_frontiers_methods_design_1', raw_sha256='raw', abstract_sha256='abstract',
                  content_locator='article/front/article-meta/abstract', study_design_reason='Measured validation results',
                  review_rule_receipt='supplement/frontiers_remaining_formats/methods_content_reviews.json',
                  format_policy_receipt='supplement/frontiers_remaining_formats/methods_policy_review.json')
    assert decide([native], [])['status'] == 'pending_primary_type'
    assert decide([native, review], [])['status'] == 'included_verified_research'
    publisher = {**native, 'family': 'publisher_frontiers', 'kind': 'Methods'}
    assert decide([publisher, review], [])['status'] == 'included_verified_research'
    assert decide([{**publisher, 'family': 'publisher_other'}, review], [])['status'] == 'pending_primary_type'
    for field in ['raw_sha256', 'abstract_sha256', 'format_policy_receipt', 'study_design_reason']:
        assert decide([native, {k: v for k, v in review.items() if k != field}], [])['status'] == 'pending_primary_type'
    assert decide([native, {**review, 'doi': '10.3389/other'}], [])['status'] == 'pending_primary_type'
    assert decide([{**native, 'kind': 'study-protocol'}, review], [])['status'] != 'included_verified_research'
    assert decide([{**native, 'online_date': ''}, review], [])['status'] == 'pending_primary_online_date'
    assert decide([native, review], [{'types': ['Study Protocol'], 'receipt': 'secondary'}])['status'] == 'pending_primary_secondary_type_conflict'
    contrary = {**native, 'id': 'review-type', 'kind': 'review', 'class': 'nonprimary'}
    assert decide([native, review, contrary], [])['status'] == 'pending_conflicting_primary_types'
