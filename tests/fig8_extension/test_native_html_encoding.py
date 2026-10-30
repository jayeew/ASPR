from experiments.fig8_extension.supplement_sources import native_html_text, parse_publisher


def test_native_utf8_title_without_html_charset_preserves_identity() -> None:
    title = 'Alzheimer’s disease and β-cells'
    raw = f'<html><head><meta name="citation_title" content="{title}"></head></html>'.encode()
    for route in ('nature', 'springer'):
        assert parse_publisher(native_html_text(raw, route))['title'] == [title]


def test_non_utf8_and_other_routes_retain_existing_parser_input() -> None:
    raw = b'<meta charset="iso-8859-1"><meta name="citation_title" content="caf\xe9">'
    assert native_html_text(raw, 'springer') == raw
    assert parse_publisher(native_html_text(raw, 'springer'))['title'] == ['café']
    utf8 = 'β'.encode()
    assert native_html_text(utf8, 'other') == utf8
