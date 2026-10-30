import json

from experiments.fig8_extension.frontiers_authors import parse_authors


def fixture(missing: bool = False, duplicate: bool = False) -> bytes:
    values = []
    def encode(value):
        index = len(values)
        values.append(None)
        if isinstance(value, dict):
            values[index] = {k: encode(v) for k, v in value.items()}
        elif isinstance(value, list):
            values[index] = [encode(v) for v in value]
        else:
            values[index] = value
        return index
    article = {'doi': '10.3389/example', 'title': 'Example', 'publicationDate': '2023-01-06',
               'authors': [{'givenNames': 'Jane', 'surname': 'Doe', 'affiliationsIndexes': [2 if missing else 1]}],
               'affiliations': [{'index': 1, 'textContent': 'Declared department, Example University, France'}]}
    encode(article)
    if duplicate:
        encode(article)
    return ('<html><meta name="citation_author" content="Doe, Jane"><script id="__NUXT_DATA__" type="application/json">' + json.dumps(values) + '</script></html>').encode()


def test_exact_paper_bound_signature_and_unresolved_reference():
    result = parse_authors(fixture(), '10.3389/example')
    assert result['authors'][0]['name'] == 'Jane Doe'
    assert not result['byline_has_etal']
    assert result['authors'][0]['affiliations'][0]['text'].endswith('France')
    from experiments.fig8_extension.supplement_apply import complete_byline, enrich_authors
    source={**result,'evidence_receipt':'fixture'}
    authors,complete=complete_byline([],source)
    enriched,_=enrich_authors(authors,source,{}, {}, {'france':'FR'})
    assert complete and enriched[0]['countries']==['FR']
    assert parse_authors(fixture(), '10.3389/other')['authors'] == []
    assert parse_authors(fixture(duplicate=True), '10.3389/example')['authors'] == []
    missing = parse_authors(fixture(missing=True), '10.3389/example')
    assert missing['byline_has_etal']
    assert missing['authors'][0]['unresolved_affiliation_refs'] == ['2']
    assert parse_authors(fixture().replace(b'Doe, Jane', b'Editor, Other'), '10.3389/example')['authors'] == []
