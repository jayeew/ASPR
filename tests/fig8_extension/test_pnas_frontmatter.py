from experiments.fig8_extension.fig8b_qualification_decide import decide


def record(categories: list[str], issn: str = '0027-8424') -> dict:
    return {'id': 'native', 'identity': 'matched', 'family': 'archived_jats',
            'kind': 'research-article', 'class': 'research', 'categories': categories,
            'issns': [issn], 'online_date': '2023-11-01'}


def test_generic_research_label_cannot_admit_pnas_frontmatter() -> None:
    assert decide([record(['Core Concepts', 'Astronomy'])], [])['status'] == 'pending_special_subtype_review'
    assert decide([record(['Core Concepts']), {**record([]), 'id': 'other'}], [])['status'] == 'pending_special_subtype_review'


def test_scope_preserves_research_and_other_journals() -> None:
    assert decide([record(['Astronomy'])], [])['status'] == 'included_verified_research'
    assert decide([record(['Core Concepts'], '1234-5678')], [])['status'] == 'included_verified_research'
    assert decide([{**record(['Core Concepts']), 'identity': 'mismatch'}], [])['status'] != 'included_verified_research'
