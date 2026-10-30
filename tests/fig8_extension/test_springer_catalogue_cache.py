import pytest

from experiments.fig8_extension.springer_catalogue_cache import parse_cards


def test_catalogue_does_not_invent_online_date_or_complete_authors() -> None:
    raw = b'''<title>Articles | Example Journal | Springer Nature Link</title>
    <article class="app-card-open"><h2><a href="/article/10.1007/example">A <i>title</i></a></h2>
    <div data-test="meta"><span class="c-meta__type">Regular Article</span>
    <span>4 January 2025</span></div><ul class="truncated"><li>A</li></ul></article>'''
    card = parse_cards(raw)[0]
    assert card['display_dates'] == ['2025-01-04']
    assert card['online_date'] == ''
    assert 'authors' not in card
    assert card['literal_types'] == ['Regular Article']
    assert card['title'] == ['A title']


def test_rejects_unrecognized_page() -> None:
    with pytest.raises(ValueError, match='journal heading'):
        parse_cards(b'<title>Access denied</title>')
