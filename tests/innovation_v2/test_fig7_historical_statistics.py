"""Regression checks for unbiased historical selection and literal evidence binding."""
from figure_pipeline.fig7_reference.historical_statistics import matching_mentions, rank_rows


def test_no_punctuation_or_numeric_prefix_bridging() -> None:
    vocabulary = {'t cell activation','lithium metal battery'}
    assert not matching_mentions('T cell. Activation; 15-T cell activation.',vocabulary)
    matches = matching_mentions('Lithium metal batteries and lithium metal battery.',vocabulary)
    assert set(matches)=={'lithium metal battery'}
    assert matches['lithium metal battery']=='lithium metal battery'


def test_global_frequency_not_category_or_whitelist_selects_topics() -> None:
    names = [f'topic number {i:02d}' for i in range(70)]
    # This high-frequency topic never appears in the former hand-selected vocabulary.
    names[0] = 'lithium metal battery'
    papers = {p:{f'paper-{j}' for j in range(75-i)} for i,p in enumerate(names)}
    rows = rank_rows(papers,{p:999 for p in names},{p:p for p in names})
    assert rows[0]['phrase']=='lithium metal battery'
    assert rows[0]['frequency']==75
    assert [r['phrase'] for r in rows if r['displayed']]==names[:68]
    assert len({r['category'] for r in rows})==1  # No forced category quotas.


def test_frequency_ties_have_stable_order_and_equal_font_size() -> None:
    papers = {'zeta test topic':set('abcde'),'alpha test topic':set('vwxyz')}
    rows = rank_rows(papers,{p:5 for p in papers},{p:p for p in papers})
    assert [r['phrase'] for r in rows]==['alpha test topic','zeta test topic']
    assert rows[0]['font_size_pt']==rows[1]['font_size_pt']


def test_contained_fragment_is_suppressed_without_changing_counts() -> None:
    papers={'oxygen species production':set('abcdefghij'),
            'reactive oxygen species production':set('abcdefgh')}
    rows=rank_rows(papers,{p:len(s) for p,s in papers.items()},{p:p for p in papers})
    assert rows[0]['frequency']==10 and not rows[0]['displayed']
    assert rows[0]['suppressed_by_fuller_phrase']==['reactive oxygen species production']
    assert rows[1]['frequency']==8 and rows[1]['displayed']
