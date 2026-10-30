from lxml import etree

from experiments.fig8_extension.edp_oai_review import dc_fields


def test_dc_preserves_nested_text_and_tails() -> None:
    node = etree.fromstring(b'<record xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:description>Fe <i>II</i> and H<sub>2</sub>O.</dc:description><dc:title>A <b>star</b></dc:title></record>')
    result = dc_fields(node)
    assert result['description'] == ['Fe II and H2O.']
    assert result['title'] == ['A star']
    assert result['creator'] == []


def test_consumer_batches_and_drains_terminal_tail() -> None:
    from experiments.fig8_extension.edp_cache_follow import ready

    assert ready(12, 0, False)
    assert not ready(13, 12, False)
    assert ready(1012, 12, False)
    assert ready(13, 12, True)
    assert not ready(12, 12, True)
