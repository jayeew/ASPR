from experiments.fig8_extension.acs_cached_imprint import publisher_statement


def test_publisher_statement_is_not_copyright_ownership() -> None:
    def xml(text: str) -> bytes:
        return ('<article><front><article-meta><permissions><copyright-statement>' + text +
                '</copyright-statement></permissions></article-meta></front>').encode()
    assert publisher_statement(xml('© 2023 The Authors. Published by American Chemical Society'))
    assert publisher_statement(xml('Published 2023 by American Chemical Society'))
    assert not publisher_statement(xml('© 2023 American Chemical Society'))
    assert not publisher_statement(xml('Published by another publisher'))
