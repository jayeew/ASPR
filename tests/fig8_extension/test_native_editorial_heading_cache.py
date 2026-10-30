from experiments.fig8_extension.native_editorial_heading_cache import headings
from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_literal_heading_not_body_topic_or_original_research() -> None:
    raw = b'''<article article-type="brief-report"><front><journal-meta>
    <journal-title-group><journal-title>Journal</journal-title></journal-title-group>
    <issn>1234-5678</issn></journal-meta><article-meta>
    <article-id pub-id-type="doi">10.1/test</article-id>
    <title-group><article-title>Title</article-title></title-group>
    <article-categories><subj-group subj-group-type="heading"><subject>Commentary</subject>
    </subj-group></article-categories></article-meta></front></article>'''
    item = {'doi': '10.1/test', 'title': 'Title', 'kind': 'brief-report',
            'journal_titles': ['Journal'], 'issns': ['1234-5678']}
    assert headings(raw, item) == ['Commentary']
    for changed in [{'doi': '10.1/other'}, {'title': 'Other'}, {'issns': ['9999-0000']},
                    {'journal_titles': ['Other']}, {'kind': 'research-article'}]:
        assert headings(raw, {**item, **changed}) == []
    for changed in [raw.replace(b'heading', b'topic'),
                    raw.replace(b'Commentary', b'Original Research'),
                    raw.replace(b'</subject>', b'</subject><subj-group><subject>Other</subject></subj-group>')]:
        assert headings(changed, item) == []


def test_independent_research_evidence_keeps_conflict() -> None:
    editorial = {'id': 'a', 'identity': 'matched', 'class': 'nonprimary',
                 'family': 'archived_jats', 'kind': 'Commentary', 'online_date': '2024-01-01'}
    research = {**editorial, 'id': 'b', 'class': 'research', 'kind': 'research-article'}
    assert decide([editorial], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([editorial, research], [])['status'] == 'pending_conflicting_primary_types'
