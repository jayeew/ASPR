from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence() -> list[dict]:
    common = dict(doi='10.1038/test', identity='matched', title='A meta-analysis',
                  online_date='2024-01-01', categories=[], class_='subtype_review_required')
    common['class'] = common.pop('class_')
    return [{**common, 'id': 'native', 'family': 'archived_jats', 'kind': 'research-article'},
            {**common, 'id': 'publisher', 'family': 'nature_catalogue', 'kind': 'Article'}]


def review(classification: str) -> dict:
    return dict(id='review', doi='10.1038/test', identity='matched',
                family='reviewed_study_design', kind='Individual study design',
                online_date='', study_design_class=classification,
                reviewed_primary_evidence_ids=['native', 'publisher'],
                source_evidence_ids=['native', 'publisher'],
                review_protocol='native_nature_individual_design_1',
                raw_sha256='raw', abstract_sha256='abstract', content_locator='article-meta/abstract',
                review_rule_receipt='review.json')


def test_individual_design_resolves_only_supported_primary_records() -> None:
    assert decide(evidence() + [review('research')], [])['status'] == 'included_verified_research'
    assert decide(evidence() + [review('nonprimary')], [])['status'] == 'excluded_primary_nonresearch'
    incomplete = review('research')
    incomplete.pop('raw_sha256')
    assert decide(evidence() + [incomplete], [])['status'] == 'pending_special_subtype_review'
    wrong = review('research')
    wrong['doi'] = '10.1038/another'
    assert decide(evidence() + [wrong], [])['status'] == 'pending_special_subtype_review'


def test_individual_design_preserves_other_conflicts_and_integrity() -> None:
    contrary = {**evidence()[0], 'id': 'contrary', 'class': 'nonprimary', 'kind': 'Review'}
    assert decide(evidence() + [review('research'), contrary], [])['status'] == 'pending_conflicting_primary_types'
    integrity = {**evidence()[0], 'id': 'integrity', 'class': 'integrity_hold'}
    assert decide(evidence() + [review('research'), integrity], [])['status'] == 'pending_integrity_review'
    secondary = [{'types': ['Systematic Review'], 'receipt': 'secondary'}]
    assert decide(evidence() + [review('research')], secondary)['status'] == 'pending_primary_secondary_type_conflict'


def test_native_only_review_requires_its_bound_archive_record() -> None:
    source = evidence()[:1]
    r = review('research')
    r.update(review_protocol='native_individual_design_1',
             source_evidence_ids=['native'], reviewed_primary_evidence_ids=['native'])
    assert decide(source + [r], [])['status'] == 'included_verified_research'
    r['source_evidence_ids'] = ['missing']
    assert decide(source + [r], [])['status'] == 'pending_special_subtype_review'
