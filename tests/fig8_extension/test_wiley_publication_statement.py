from lxml import etree
from experiments.fig8_extension.jats_annual_copyright import named_publisher


def article(statement: str, publisher: str = '') -> etree._Element:
    node = etree.fromstring(b'<article><front><journal-meta><publisher><publisher-name/></publisher></journal-meta><article-meta><permissions><copyright-statement/></permissions></article-meta></front></article>')
    node.find('.//publisher-name').text = publisher
    node.find('.//copyright-statement').text = statement
    return node


def test_john_wiley_explicit_publication_statement() -> None:
    result = named_publisher(article('© 2023 The Authors. Molecular Ecology published by John Wiley & Sons Ltd.'), 2023)
    assert result and result[0] == 'Wiley'


def test_year_and_role_and_conflicting_imprint_remain_required() -> None:
    assert named_publisher(article('© 2023 John Wiley & Sons Ltd.'), 2023) is None
    assert named_publisher(article('© 2023 The Authors. Published by John Wiley & Sons Ltd.'), 2024) is None
    assert named_publisher(article('© 2023 The Authors. Published by John Wiley & Sons Ltd.', 'Another publisher'), 2023) is None


def test_explicit_other_publisher_and_conflict() -> None:
    for name in ('Elsevier', 'Oxford University Press', 'American Association for Cancer Research', 'Wolters Kluwer', 'S. Karger', 'BMJ'):
        result = named_publisher(article(f'© 2024 The Authors. Published by {name}', name), 2024)
        assert result and result[0] == name
        assert named_publisher(article(f'© 2024 {name}'), 2024) is None
        assert named_publisher(article(f'© 2024 Published by {name}', 'Nature Publishing Group'), 2024) is None
        assert named_publisher(article(f'© 2024 Published by {name}'), 2025) is None


def test_single_paper_statement_needs_matching_date_and_identity() -> None:
    from experiments.fig8_extension.fig8b_qualification_decide import portfolio_label
    source = {'id': 'native', 'identity': 'matched', 'family': 'archived_jats',
              'publishers': ['American Association for Cancer Research'],
              'online_date': '2024-03-01', 'publisher_statement': '©2024 Published by the American Association for Cancer Research',
              'publisher_statement_rule': 'Explicit dated named published-by statement; copyright holder alone insufficient'}
    assert portfolio_label('', [source], 2024)[0] == 'False'
    assert portfolio_label('', [source], 2023)[0] == 'unresolved'
    assert portfolio_label('', [{**source, 'identity': 'mismatch'}], 2024)[0] == 'unresolved'
    assert portfolio_label('', [{**source, 'publisher_statement': ''}], 2024)[0] == 'unresolved'
    assert portfolio_label('True', [source], 2024)[0] == 'unresolved'
