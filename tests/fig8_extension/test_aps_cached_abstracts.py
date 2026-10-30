from experiments.fig8_extension.aps_cached_abstracts import abstract_text


def test_native_abstract_only_not_body_or_unknown_format():
    assert abstract_text({'abstract': {'format': 'html', 'value': '<p>Original <i>text</i>.</p>'}}) == 'Original text.'
    assert abstract_text({'body': 'Not an abstract'}) == ''
    assert abstract_text({'abstract': {'format': 'unsupported', 'value': 'not parsed'}}) == ''
