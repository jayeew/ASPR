from experiments.fig8_extension.cached_native_subtype_holds import flagged


def test_explicit_format_conflict_is_not_inferred_from_research_topic() -> None:
    assert flagged('An unusual clinical finding', ['Innovations: Images'])
    assert flagged('Letter to the editor: a comment', [])
    assert flagged('Comments on “A study”', [])
    assert not flagged('Image processing for microscopy', ['Research Article'])
    assert not flagged('A new perspective on gene regulation', ['Research Article'])
    assert not flagged('Original experiments', ['Research Letter'])
