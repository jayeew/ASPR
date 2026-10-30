from experiments.fig8_extension.sciopen_metadata import enrich_metadata
from experiments.fig8_extension.sciopen_metadata import parse_authors


def test_native_fields_do_not_infer_online_date_or_use_graphical_abstract():
    raw = '''<div class="v4-art-info"><span>Research Article</span><span>Open Access</span></div>
    <div id="title_-2"><h2>Abstract</h2><div><p>Original scientific abstract.</p></div></div>
    <div id="title_-2-1"><h2>Graphical Abstract</h2><div><p>Image caption.</p></div></div>
    <div><span>Published: </span><span>29 May 2025</span></div>'''
    result = enrich_metadata(raw, {'online_date': '', 'date_precision': 'missing_explicit_electronic_date'})
    assert result['article_types'] == ['Research Article']
    assert result['abstract'] == 'Original scientific abstract.'
    assert result['online_date'] == ''
    assert result['date_field_review']['sciopen_visible_published'] == ['2025-05-29']


def test_byline_links_keep_missing_and_ambiguous_addresses_unresolved():
    raw = '''<meta name="citation_author" content="Alice Smith">
    <span class="art-authors"><span>Alice Smith</span><sup>1,2,3</sup></span>
    <div class="art-authors-unit-item"><label>1</label>University A, France</div>
    <div class="art-authors-unit-item"><label>2</label>University B, UK</div>
    <div class="art-authors-unit-item"><label>2</label>University C, USA</div>'''
    result = parse_authors(raw)
    assert result['byline_has_etal'] is False
    assert result['authors'][0]['affiliations'][0]['text'] == 'University A, France'
    assert result['authors'][0]['unresolved_affiliation_refs'] == ['2', '3']
    assert parse_authors(raw.replace('content="Alice Smith"', 'content="Someone Else"'))['byline_has_etal'] is True


def test_single_unlabelled_address_requires_all_authors_without_markers():
    raw = '''<meta name="citation_author" content="Alice Smith">
    <span class="art-authors"><span>Alice Smith</span></span>
    <div class="art-authors-unit-item">University A, France</div>'''
    assert parse_authors(raw)['authors'][0]['affiliations'][0]['text'] == 'University A, France'
    ambiguous = raw + '<div class="art-authors-unit-item">University B, UK</div>'
    assert parse_authors(ambiguous)['authors'][0]['affiliations'] == []
    marked = raw.replace('Smith</span>', 'Smith</span><sup>2</sup>')
    assert parse_authors(marked)['authors'][0]['affiliations'] == []


def test_explicit_online_date_supports_unpadded_fields_and_retains_conflicts():
    raw = '<meta name="citation_online_date" content="2025/5/29">'
    assert enrich_metadata(raw, {})['online_date'] == '2025-05-29'
    assert enrich_metadata(raw + '<meta name="citation_online_date" content="2024/5/29">', {})['online_date'] == ''
    assert enrich_metadata('<meta name="citation_online_date" content="2025/2/30">', {})['online_date'] == ''
    assert enrich_metadata('<html/>', {'title': ['Pores &lt; 1 nm']})['title'] == ['Pores < 1 nm']
