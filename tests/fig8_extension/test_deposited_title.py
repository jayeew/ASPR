from experiments.fig8_extension.cached_date_apply import deposited_title


def test_deposited_entity_and_inline_markup() -> None:
    assert deposited_title('CUT&amp;RUN GSK-3&amp;beta;') == 'CUT&RUN GSK-3β'
    assert deposited_title('&lt;i&gt;Solanum&lt;/i&gt;') == 'Solanum'
    assert deposited_title('c.3624C&gt;T') == 'c.3624C>T'
    assert deposited_title('x &lt; y and z &gt; 0') == 'x < y and z > 0'
    assert deposited_title('different substantive title') == 'different substantive title'


def test_double_encoded_literal_italic_tag():
    assert deposited_title("&amp;lt;italic&amp;gt;Lycium barbarum&amp;lt;/italic&amp;gt; polysaccharide") == "Lycium barbarum polysaccharide"
