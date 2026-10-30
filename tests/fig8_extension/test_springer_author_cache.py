from experiments.fig8_extension.springer_author_cache import parse_authors


def test_explicit_links_not_affiliation_order():
    raw=b'''<html><meta name="citation_author" content="Doe, Jane"><ul data-test="authors-list"><li><a data-test="author-name">Jane Doe</a><sup><a href="#Aff2">2</a><a href="#Aff3">3</a></sup></li></ul><li id="Aff1"><p class="c-article-author-affiliation__address">Wrong unit</p></li><li id="Aff2"><p class="c-article-author-affiliation__address">Correct unit, Germany</p></li></html>'''
    result=parse_authors(raw)
    assert not result['byline_has_etal']
    assert result['authors'][0]['affiliations'][0]['text']=='Correct unit, Germany'
    assert result['authors'][0]['unresolved_affiliation_refs']==['Aff3']
    assert parse_authors(raw.replace(b'Doe, Jane',b'Doe, John'))['byline_has_etal']
    assert parse_authors(b'<html/>')['authors']==[]


def test_explicit_utf8_cache_adapter_preserves_names_and_addresses(tmp_path):
    import gzip
    import json
    from experiments.fig8_extension.springer_author_cache import prepare_source, source_digest
    raw = '''<html><meta name="citation_author" content="Müller, Léa"><ul data-test="authors-list"><li><a data-test="author-name">Léa Müller</a><sup><a href="#Aff1">1</a></sup></li></ul><li id="Aff1"><p class="c-article-author-affiliation__address">Université de Genève, Switzerland</p></li></html>'''.encode('utf-8')
    q = tmp_path / 'fields/qualification'
    q.mkdir(parents=True)
    (q / 'raw.gz').write_bytes(gzip.compress(raw))
    record = {'route': 'nature_cached_page', 'source': 'publisher', 'raw_text_encoding': 'utf-8', 'raw_receipt': 'raw.gz', 'parsed': {'doi': ['10.1038/test'], 'title': ['Valid title'], 'issns': ['1234-5678'], 'journal_titles': ['Nature']}}
    record['parsed']['abstract'] = 'Explicit abstract from the same retained publisher page.'
    (tmp_path / 'receipt.json').write_text(json.dumps(record))
    paper = {'doi': '10.1038/test', 'title': 'Valid title', 'journal_name': 'Nature'}
    _, digest, authors = prepare_source(tmp_path, paper, 'receipt.json', {'1234-5678'})
    assert authors['authors'][0]['name'] == 'Léa Müller'
    assert authors['authors'][0]['affiliations'][0]['text'] == 'Université de Genève, Switzerland'
    assert not authors['byline_has_etal']
    assert authors['abstract'] == record['parsed']['abstract']
    assert digest.endswith(':html-encoding-utf-8')
    assert source_digest(record, 'hash') != source_digest({k: v for k, v in record.items() if k != 'raw_text_encoding'}, 'hash')
