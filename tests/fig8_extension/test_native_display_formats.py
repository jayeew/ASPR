from experiments.fig8_extension.native_display_formats import classified_heading
from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_display_labels_require_native_scope_and_unmixed_headings() -> None:
    item = {'family': 'archived_jats', 'identity': 'matched', 'kind': 'other'}
    native = {'article_type': 'other', 'categories': ['Frontispiece', 'Frontispiece']}
    assert classified_heading(item, native) == 'Frontispiece'
    assert not classified_heading(item, {**native, 'categories': ['Frontispiece', 'Research Article']})
    assert not classified_heading({**item, 'identity': 'mismatch'}, native)
    assert not classified_heading({**item, 'kind': 'research-article'}, {**native, 'article_type': 'research-article'})
    interview = {'article_type': 'other', 'categories': ['First Person'], 'journal_titles': ['Biology Open']}
    assert classified_heading(item, interview) == 'Interview'
    assert not classified_heading(item, {**interview, 'journal_titles': ['Different Journal']})


def test_cover_proof_does_not_override_conflicting_research() -> None:
    cover = {'id': 'c', 'family': 'archived_jats', 'identity': 'matched', 'kind': 'Frontispiece',
             'class': 'nonprimary', 'categories': ['Frontispiece'], 'online_date': '2024-01-01'}
    research = {**cover, 'id': 'r', 'kind': 'research-article', 'class': 'research', 'categories': []}
    assert decide([cover], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([cover, research], [])['status'] == 'pending_conflicting_primary_types'
