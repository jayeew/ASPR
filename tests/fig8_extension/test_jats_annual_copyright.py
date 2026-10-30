from lxml import etree
from experiments.fig8_extension.jats_annual_copyright import named_publisher


def test_kmb_licensee_requires_named_publisher_journal_and_year() -> None:
    source = ('<article><front><journal-meta><issn>1738-8872</issn><publisher>'
              '<publisher-name>Korean Society for Microbiology and Biotechnology</publisher-name>'
              '</publisher></journal-meta><article-meta><permissions><copyright-year>2025</copyright-year>'
              '<copyright-statement>Copyright © 2025 by the authors. Licensee KMB</copyright-statement>'
              '</permissions></article-meta></front></article>')
    assert named_publisher(etree.fromstring(source.encode()), 2025)[0] == 'Korean Society for Microbiology and Biotechnology'
    assert named_publisher(etree.fromstring(source.encode()), 2024) is None
    for before, after in [('1738-8872', '0000-0000'), ('Licensee KMB', 'The Authors'),
                          ('Korean Society for Microbiology and Biotechnology', 'Other Publisher')]:
        assert named_publisher(etree.fromstring(source.replace(before, after).encode()), 2025) is None


def test_named_publication_statement_is_required_not_authors_copyright_alone() -> None:
    def document(extra: str) -> etree._Element:
        return etree.fromstring(('<article><front><journal-meta><publisher><publisher-name>MDPI</publisher-name></publisher></journal-meta><article-meta><permissions><copyright-statement>Copyright 2023 The authors.</copyright-statement><copyright-year>2023</copyright-year>' + extra + '</permissions></article-meta></front></article>').encode())
    assert named_publisher(document(''), 2023) is None
    assert named_publisher(document('<license>Licensee MDPI, Basel, Switzerland.</license>'), 2023)[0] == 'MDPI'
    assert named_publisher(document('<license>Licensee MDPI, Basel, Switzerland.</license>'), 2024) is None


def test_other_reviewed_publishers_require_explicit_consistent_dated_statement() -> None:
    def document(publisher: str, statement: str) -> etree._Element:
        return etree.fromstring((
            '<article><front><journal-meta><publisher><publisher-name>' + publisher +
            '</publisher-name></publisher></journal-meta><article-meta><permissions>'
            '<copyright-year>2024</copyright-year><copyright-statement>' + statement +
            '</copyright-statement></permissions></article-meta></front></article>').encode())
    assert named_publisher(document('Elsevier', '2024 Published by Elsevier.'), 2024)[0] == 'Elsevier'
    assert named_publisher(document('Oxford University Press', '2024 Published by Oxford University Press.'), 2024)[0] == 'Oxford University Press'
    assert named_publisher(document('BMJ', '2024 Published by BMJ.'), 2024)[0] == 'BMJ'
    assert named_publisher(document('Elsevier', '2024 The Authors.'), 2024) is None
    assert named_publisher(document('Elsevier', '2024 Published by BMJ.'), 2024) is None
    assert named_publisher(document('Elsevier', '2024 Published by Elsevier.'), 2023) is None


def test_explicit_license_statement_requires_consistent_header_and_year() -> None:
    def document(publisher: str, statement: str) -> etree._Element:
        return etree.fromstring((
            '<article><front><journal-meta><publisher><publisher-name>' + publisher +
            '</publisher-name></publisher></journal-meta><article-meta><permissions>'
            '<copyright-year>2024</copyright-year><copyright-statement>2024 The Authors.</copyright-statement>'
            '<license>' + statement + '</license></permissions></article-meta></front></article>').encode())
    assert named_publisher(document('Portland Press Ltd', 'Published by Portland Press Limited on behalf of the Biochemical Society.'), 2024)[0] == 'Portland Press'
    assert named_publisher(document('Karger Publishers', 'Published by S. Karger AG, Basel.'), 2024)[0] == 'S. Karger'
    assert named_publisher(document('Bentham Science Publishers', 'Published by Bentham Science Publisher.'), 2024)[0] == 'Bentham Science'
    assert named_publisher(document('Elsevier', 'Published by Portland Press Limited.'), 2024) is None
    assert named_publisher(document('Portland Press Ltd', 'Copyright The Authors.'), 2024) is None
    assert named_publisher(document('Portland Press Ltd', 'Published by Portland Press Limited.'), 2025) is None


def test_society_imprint_header_exception_is_journal_and_year_scoped() -> None:
    source = ('<article><front><journal-meta><journal-title-group><journal-title>'
              'Translational Oncology</journal-title></journal-title-group><publisher>'
              '<publisher-name>Neoplasia Press</publisher-name></publisher></journal-meta>'
              '<article-meta><permissions><copyright-year>2024</copyright-year>'
              '<copyright-statement>2024 The Authors. Published by Elsevier Inc.</copyright-statement>'
              '</permissions></article-meta></front></article>')
    assert named_publisher(etree.fromstring(source.encode()), 2024)[0] == 'Elsevier'
    assert named_publisher(etree.fromstring(source.encode()), 2023) is None
    for before, after in [('Translational Oncology', 'Different Journal'),
                          ('Neoplasia Press', 'Nature Publishing Group'),
                          ('Published by Elsevier Inc.', 'All rights reserved.')]:
        assert named_publisher(etree.fromstring(source.replace(before, after).encode()), 2024) is None


def test_new_named_publishers_do_not_infer_from_header_or_wrong_year() -> None:
    for header, literal, label in [
        ('Thieme Medical Publishers', 'Thieme', 'Thieme'),
        ('Taylor &amp; Francis', 'Informa UK Limited, trading as Taylor &amp; Francis Group', 'Taylor & Francis'),
        ('Scientific Scholar', 'Scientific Scholar', 'Scientific Scholar'),
        ('International Institute of Anticancer Research', 'the International Institute of Anticancer Research', 'International Institute of Anticancer Research'),
        ('Maximum Academic Press', 'Maximum Academic Press', 'Maximum Academic Press'),
    ]:
        source = ('<article><front><journal-meta><publisher><publisher-name>' + header +
                  '</publisher-name></publisher></journal-meta><article-meta><permissions>'
                  '<copyright-year>2025</copyright-year><license>Published by ' + literal +
                  '.</license></permissions></article-meta></front></article>')
        assert named_publisher(etree.fromstring(source.encode()), 2025)[0] == label
        assert named_publisher(etree.fromstring(source.encode()), 2024) is None
        assert named_publisher(etree.fromstring(source.replace('Published by ', 'Copyright ').encode()), 2025) is None


def test_cshl_requires_publication_statement_and_agreeing_year() -> None:
    source = '<article><front><journal-meta><publisher><publisher-name>Cold Spring Harbor Laboratory Press</publisher-name></publisher></journal-meta><article-meta><permissions><copyright-statement>2025 Authors; Published by Cold Spring Harbor Laboratory Press for the RNA Society</copyright-statement></permissions></article-meta></front></article>'
    assert named_publisher(etree.fromstring(source.encode()), 2025)[0] == 'Cold Spring Harbor Laboratory Press'
    assert named_publisher(etree.fromstring(source.encode()), 2024) is None
    assert named_publisher(etree.fromstring(source.replace('Published by', 'Copyright owner').encode()), 2025) is None


def test_society_publisher_statement_is_bound_to_title_and_header() -> None:
    source = '<article><front><journal-meta><journal-title-group><journal-title>Genes to Cells</journal-title></journal-title-group></journal-meta><article-meta><permissions><copyright-statement>2024 Authors. Genes to Cells published by Molecular Biology Society of Japan and John Wiley &amp; Sons Australia, Ltd.</copyright-statement></permissions></article-meta></front></article>'
    assert named_publisher(etree.fromstring(source.encode()), 2024)[0] == 'Wiley'
    assert named_publisher(etree.fromstring(source.replace('<journal-title>Genes to Cells', '<journal-title>Other Journal').encode()), 2024) is None
    assert named_publisher(etree.fromstring(source.encode()), 2025) is None
