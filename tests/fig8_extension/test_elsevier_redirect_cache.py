from experiments.fig8_extension.elsevier_redirect_cache import target


def test_only_explicit_doi_bound_publisher_target() -> None:
    raw = b'<html><input id="id" value="10.1016/x"><input id="redirectURL" value="https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fpii%2FS123"></html>'
    assert target(raw, '10.1016/x') == 'https://www.sciencedirect.com/science/article/pii/S123'
    assert not target(raw, '10.1016/y')
    assert not target(raw.replace(b'www.sciencedirect.com', b'example.com'), '10.1016/x')
