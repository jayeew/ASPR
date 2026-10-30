import pytest
from lxml import etree
from experiments.fig8_extension.native_xml_document import document


def test_missing_mathml_binding_preserves_text_and_other_namespaces():
    raw=b'<article xmlns="http://www.elsevier.com/xml/ja/dtd"><mml:math><mml:mi>x</mml:mi></mml:math></article>'
    tree, repairs = document(raw)
    assert tree[0].tag == '{http://www.w3.org/1998/Math/MathML}math'
    assert ''.join(tree.itertext()) == 'x'
    assert len(repairs) == 1
    assert b'xmlns:mml' not in raw
    assert document(b'<article/>')[1] == []


@pytest.mark.parametrize('raw', [
    b'<article xmlns="http://www.elsevier.com/xml/ja/dtd"><unknown:math/></article>',
    b'<article xmlns="http://www.elsevier.com/xml/ja/dtd"><mml:math></article>',
    b'<article><mml:math/></article>',
])
def test_unrelated_errors_remain_failures(raw):
    with pytest.raises(etree.XMLSyntaxError):
        document(raw)
