import pytest

from experiments.fig8_extension.primary_byline_comparison import compare


def example():
    old = [{'name': 'Alice Smith', 'publisher_affiliation_slots': [{'text': 'University A, City'}]}]
    new = {'authors': [{'name': 'Alice Smith', 'affiliations': [{'text': 'University A, City'}]}]}
    return old, new


def test_exact_links_equivalent_but_different_addresses_not_merged():
    old, new = example()
    assert compare(old, new).startswith('equivalent_complete_declared_links')
    new['authors'][0]['affiliations'][0]['text'] = 'University A and Hospital B, City'
    assert compare(old, new) == 'affiliation_statements_differ'


def test_missing_links_and_collective_authors_are_not_equivalent():
    old, new = example()
    new['authors'][0]['unresolved_affiliation_refs'] = ['aff2']
    assert compare(old, new) == 'incomplete_affiliation_links'
    new['authors'][0]['unresolved_affiliation_refs'] = []
    new['authors'][0]['nonpersonal'] = True
    assert compare(old, new) == 'personal_collective_type_differs'


def test_duplicate_names_truncated_byline_and_nature_are_held():
    old, new = example()
    assert compare(old * 2, {'authors': new['authors'] * 2}) == 'duplicate_author_names_require_review'
    new['byline_has_etal'] = True
    assert compare(old, new) == 'incoming_byline_completeness_unconfirmed'
    new['byline_has_etal'] = False
    old[0]['nature_v1_slots'] = [{'entity_id': 'existing'}]
    assert compare(old, new) == 'nature_review_preserved'


def test_partial_fill_keeps_existing_primary_and_never_guesses_initials():
    from collections import defaultdict
    from experiments.fig8_extension.primary_byline_comparison import fill_missing_links
    def author(name):
        return {'name': name, 'countries': [], 'institution_ids_or_verified_local_entities': []}
    old = [author('Alice Smith'), author('Bob Jones'), author('C. Doe')]
    old[0]['publisher_affiliation_slots'] = [{'text': 'Retained institute', 'country_codes': ['DE']}]
    source = {'evidence_receipt': 'cached.xml', 'authors': [
        {'name': name, 'nonpersonal': False, 'unresolved_affiliation_refs': [],
         'affiliations': [{'text': 'Institute, France', 'countries': [], 'identifiers': [], 'names': []}]}
        for name in ['Alice Smith', 'Bob Jones', 'Carol Doe']]}
    revised, count = fill_missing_links(old, source, defaultdict(list), defaultdict(list), {})
    assert count == 1
    assert revised[0] == old[0]
    assert revised[2] == old[2]
    assert revised[1]['publisher_affiliation_source'] == 'cached.xml'
    assert fill_missing_links(revised, source, defaultdict(list), defaultdict(list), {}) == (revised, 0)


def test_native_jats_doi_case_is_normalized_without_changing_identity():
    from experiments.fig8_extension.primary_byline_comparison import incoming_metadata
    raw = b'<article><front><journal-meta><journal-title-group><journal-title>Journal</journal-title></journal-title-group><issn>1314-2836</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.3897/BDJ.12.e126553</article-id><title-group><article-title>Exact title</article-title></title-group></article-meta></front></article>'
    parsed = incoming_metadata({'source': 'publisher_jats', 'route': 'pensoft'}, raw)
    assert parsed['doi'] == ['10.3897/bdj.12.e126553']
    assert '10.3897/bdj.12.e126554' not in parsed['doi']
    assert parsed['issns'] == ['1314-2836']
    assert parsed['title'] == ['Exact title']


def test_pmc_oai_envelope_reuses_unique_native_record_parser():
    import pytest
    from experiments.fig8_extension.primary_byline_comparison import incoming_metadata
    article = b'<article xmlns="https://jats.nlm.nih.gov"><front><journal-meta><issn>1234-5678</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.1234/TEST</article-id><title-group><article-title>Matched title</article-title></title-group></article-meta></front></article>'
    raw = b'<OAI-PMH><GetRecord><record><metadata>'+article+b'</metadata></record></GetRecord></OAI-PMH>'
    record = {'source':'pmc','route':'pmc_journal_oai'}
    parsed = incoming_metadata(record, raw)
    assert parsed['doi'] == ['10.1234/test']
    assert parsed['title'] == ['Matched title']
    with pytest.raises(ValueError):
        incoming_metadata(record, raw.replace(b'</metadata>',article+b'</metadata>'))


def test_aps_batch_selection_requires_unique_exact_doi(tmp_path, monkeypatch):
    import gzip
    import json
    import pytest
    from experiments.fig8_extension import primary_byline_comparison as mod
    from experiments.fig8_extension.aps_authors import records
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    rel = 'fields/qualification/aps_all_raw/test.json.gz'
    path = tmp_path/rel
    path.parent.mkdir(parents=True)
    article = {'identifiers':{'doi':'10.1103/test'}, 'title':{'value':'Exact APS title'},
               'journal':{'id':'PRL','name':'Physical Review Letters'},
               'authors':[{'type':'Person','name':'Alice Smith','affiliationIds':['a1']}],
               'affiliations':[{'id':'a1','name':'Institute, Germany'}]}
    path.write_bytes(gzip.compress(json.dumps({'http_status':200,'response':{'data':[article]}}).encode()))
    record, raw = mod.load_primary(rel, '10.1103/test')
    parsed = mod.incoming_metadata(record, raw)
    assert parsed['doi'] == ['10.1103/test'] and parsed['title'] == ['Exact APS title']
    assert parsed['authors'][0]['affiliations'][0]['text'] == 'Institute, Germany'
    with pytest.raises(ValueError):
        mod.load_primary(rel, '10.1103/other')
    records.cache_clear()
    path.write_bytes(gzip.compress(json.dumps({'http_status':200,'response':{'data':[article,article]}}).encode()))
    with pytest.raises(ValueError):
        mod.load_primary(rel, '10.1103/test')
    records.cache_clear()


def test_fully_linked_primary_bypasses_reenrichment(monkeypatch):
    from experiments.fig8_extension import primary_byline_comparison as mod
    def unexpected(*args, **kwargs):
        raise AssertionError('No missing links; expensive reenrichment must not run')
    monkeypatch.setattr(mod, 'enrich_authors', unexpected)
    current = [{'name':'Alice','publisher_affiliation_slots':[{'text':'Primary address'}]},
               {'name':'Bob','nature_v1_slots':[]}]
    assert mod.fill_missing_links(current, {}, {}, {}, {}) == (current, 0)


def test_copernicus_batch_keeps_native_rid_links_and_rejects_duplicate():
    import pytest
    from experiments.fig8_extension.primary_byline_comparison import incoming_metadata
    article = b'<article xmlns="urn:nlm"><front><journal-meta><issn>1234-5678</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.5194/test</article-id><title-group><article-title>Title</article-title></title-group><contrib-group><contrib contrib-type="author" rid="aff1"><name><surname>Smith</surname><given-names>Alice</given-names></name></contrib></contrib-group><aff id="aff1">Institute, Germany</aff></article-meta></front></article>'
    raw = b'<OAI-PMH><ListRecords><record><metadata>'+article+b'</metadata></record></ListRecords></OAI-PMH>'
    record = {'source':'publisher','route':'copernicus','doi':'10.5194/test'}
    parsed = incoming_metadata(record, raw)
    assert parsed['authors'][0]['affiliations'][0]['text'] == 'Institute, Germany'
    with pytest.raises(ValueError):
        incoming_metadata(record,raw.replace(b'</metadata>', article+b'</metadata>'))


@pytest.mark.parametrize("extent", ["complete_front_only", "full_object"])
def test_cloud_front_uses_recorded_hash_not_recomputed_hash(tmp_path, monkeypatch, extent):
    import gzip
    import json
    from experiments.fig8_extension import primary_byline_comparison as module

    monkeypatch.setattr(module, 'ROOT', tmp_path)
    folder = tmp_path / 'fields/qualification'
    folder.mkdir(parents=True)
    (folder / 'front.gz').write_bytes(gzip.compress(b'<article><front/></article>'))
    (tmp_path / 'parsed.json').write_text(json.dumps({
        'source': 'pmc', 'route': 'pmc_cloud', 'source_extent': extent,
        'raw_receipt': 'front.gz', 'acquisition_receipt': {'retained_sha256': 'recorded_expected_hash'}}))
    record, raw = module.load_primary('parsed.json', '10.test/test')
    assert record['raw_sha256'] == 'recorded_expected_hash'
    assert raw == b'<article><front/></article>'
