from lxml import etree

from experiments.fig8_extension.jats_annual_copyright import named_publisher


def tree(statement: str) -> etree._Element:
    return etree.fromstring(('<article><front><article-meta><permissions><copyright-statement>'
                             + statement + '</copyright-statement></permissions></article-meta></front></article>').encode())


def test_explicit_acs_published_by_requires_matching_year() -> None:
    for statement in ['© 2023 The Authors. Published by American Chemical Society',
                      'Not subject to U.S. Copyright. Published 2023 by American Chemical Society']:
        assert named_publisher(tree(statement), 2023)[0] == 'American Chemical Society'
        assert named_publisher(tree(statement), 2024) is None
    assert named_publisher(tree('© 2023 American Chemical Society'), 2023) is None
    assert named_publisher(tree('© 2023 The Authors'), 2023) is None


def test_wiley_dated_statement_with_missing_imprint_preserves_conflicts() -> None:
    statement = '© 2023 The Authors. Small published by Wiley-VCH GmbH'
    assert named_publisher(tree(statement), 2023)[0] == 'Wiley'
    assert named_publisher(tree(statement), 2024) is None
    assert named_publisher(tree('© 2023 Wiley'), 2023) is None
    assert named_publisher(tree('© 2023 Published by WileySomething'), 2023) is None
    conflicting = tree(statement)
    journal = etree.Element('journal-meta')
    publisher = etree.SubElement(journal, 'publisher')
    etree.SubElement(publisher, 'publisher-name').text = 'Oxford University Press'
    conflicting.find('front').insert(0, journal)
    assert named_publisher(conflicting, 2023) is None
