from experiments.fig8_extension.native_abstract_scope import native_abstract_scope


def test_self_described_review_is_held_not_excluded() -> None:
    result = native_abstract_scope({'abstract': 'In this review, we discuss current mechanisms.', 'categories': ['Article']}, 'research-article')
    assert result['scope_review_required'] is True
    assert result['review_quote'] == 'In this review'
    assert result['review_abstract'].startswith('In this review')


def test_reviewing_data_is_not_a_review_label() -> None:
    assert native_abstract_scope({'abstract': 'We reviewed patient records and tested a hypothesis.'}, 'research-article') == {}


def test_native_review_remains_explicitly_nonprimary() -> None:
    assert native_abstract_scope({'abstract': 'In this review, we summarize results.'}, 'review-article') == {}


def test_missing_abstract_does_not_change_qualification() -> None:
    assert native_abstract_scope({}, 'research-article') == {}
