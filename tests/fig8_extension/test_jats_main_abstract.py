from experiments.fig8_extension.supplement_sources import jats_metadata


def test_significance_and_translated_abstract_do_not_contaminate_main() -> None:
    xml = b'''<article xml:lang="en"><front><article-meta>
      <abstract abstract-type="executive-summary"><title>Significance</title><p>Auxiliary</p></abstract>
      <abstract><p>Main with H<sub>2</sub>O.</p></abstract>
      <abstract xml:lang="fr"><p>French</p></abstract>
    </article-meta></front></article>'''
    parsed = jats_metadata(xml)
    assert parsed['abstract'] == 'Main with H2O.'
    assert parsed['auxiliary_abstracts'] == [{'type': 'executive-summary', 'text': 'SignificanceAuxiliary'}]


def test_auxiliary_alone_does_not_substitute_for_missing_abstract() -> None:
    xml = b'''<article><front><article-meta><abstract abstract-type="executive-summary">
    <p>Significance statement alone.</p></abstract></article-meta></front></article>'''
    assert jats_metadata(xml)['abstract'] == ''


def test_short_review_requires_exact_doi_and_content() -> None:
    import hashlib
    from experiments.fig8_extension.journal_readiness import abstract_screen
    value = 'A short but complete native abstract.'
    digest = hashlib.sha256(value.encode()).hexdigest()
    assert abstract_screen(value, '10.1/a', {'10.1/a': digest}) == 'source_text_confirmed'
    assert abstract_screen(value + ' Changed.', '10.1/a', {'10.1/a': digest}) == 'quality_review'
    assert abstract_screen(value, '10.1/b', {'10.1/a': digest}) == 'quality_review'


def test_uppercase_english_language_retains_only_english_abstract() -> None:
    xml = b'''<article xml:lang="EN"><front><article-meta>
      <abstract xml:lang="EN"><p>Original English abstract.</p></abstract>
      <abstract xml:lang="RU"><p>Translated abstract.</p></abstract>
    </article-meta></front></article>'''
    parsed = jats_metadata(xml)
    assert parsed['abstract'] == 'Original English abstract.'
    assert parsed['abstract_language'] == 'en'
